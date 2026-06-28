"""
审批路由 — HITL 审批的 REST API + 内置 Web UI

端点：
  GET  /api/v1/approval/list              — 待审批列表
  GET  /api/v1/approval/{request_id}      — 审批详情
  POST /api/v1/approval/{request_id}/approve — 批准
  POST /api/v1/approval/{request_id}/reject  — 拒绝
  GET  /api/v1/approval/ui                — 审批管理 Web 页面

使用方式（main.py 中注册）：
  from app.human_in_the_loop.approval_routes import router
  app.include_router(router, prefix="/api/v1/approval", tags=["审批"])
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.human_in_the_loop.approval_schema import ApprovalStatus, ApprovalType
from app.human_in_the_loop.approval_service import approval_service
from app.graph.supervisor_graph import resume_from_checkpoint, get_graph

logger = logging.getLogger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
#  Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class ApproveRequest(BaseModel):
    reviewer: str = "admin"
    comment: str = ""


class RejectRequest(BaseModel):
    reviewer: str = "admin"
    comment: str = ""


class ApprovalListItem(BaseModel):
    id: str
    type: str
    status: str
    from_agent: str
    priority: str
    reason: str
    context: dict
    time_remaining: float
    created_at: float


class ApprovalDetail(BaseModel):
    id: str
    type: str
    status: str
    from_agent: str
    thread_id: str
    priority: str
    reason: str
    context: dict
    created_at: float
    expires_at: float
    time_remaining: float
    reviewed_at: float | None
    reviewer: str
    comment: str


class ApprovalActionResponse(BaseModel):
    success: bool
    message: str
    request_id: str


# ═══════════════════════════════════════════════════════════════
#  REST API
# ═══════════════════════════════════════════════════════════════

@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def approval_ui():
    """审批管理 Web 页面"""
    return HTMLResponse(content=APPROVAL_UI_HTML)


@router.get("/list", response_model=list[ApprovalListItem])
async def list_pending():
    """获取所有待审批的请求列表"""
    pending = approval_service.list_pending()
    return [
        ApprovalListItem(
            id=r.id,
            type=r.type.value,
            status=r.status.value,
            from_agent=r.from_agent,
            priority=r.priority,
            reason=r.reason,
            context=r.context,
            time_remaining=r.time_remaining,
            created_at=r.created_at,
        )
        for r in pending
    ]


@router.get("/{request_id}", response_model=ApprovalDetail)
async def get_approval(request_id: str):
    """获取单个审批请求的详情"""
    req = approval_service.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="审批请求不存在")
    return ApprovalDetail(
        id=req.id,
        type=req.type.value,
        status=req.status.value,
        from_agent=req.from_agent,
        thread_id=req.thread_id,
        priority=req.priority,
        reason=req.reason,
        context=req.context,
        created_at=req.created_at,
        expires_at=req.expires_at,
        time_remaining=req.time_remaining,
        reviewed_at=req.reviewed_at,
        reviewer=req.reviewer,
        comment=req.comment,
    )


@router.post("/{request_id}/approve", response_model=ApprovalActionResponse)
async def approve_request(request_id: str, body: ApproveRequest):
    """
    批准一个审批请求，并自动恢复 LangGraph 图执行。

    审批通过后：
      1. 更新 ApprovalRequest 状态为 approved
      2. 更新 LangGraph 状态中的 approval_pending
      3. 调用 resume_from_checkpoint() 恢复图执行
    """
    req = approval_service.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="审批请求不存在")
    if req.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"审批请求状态为 {req.status.value}，无法操作")

    # 更新审批状态
    result = approval_service.approve(request_id, reviewer=body.reviewer, comment=body.comment)
    if result is None:
        raise HTTPException(status_code=500, detail="审批操作失败")

    # 恢复 LangGraph 图执行
    # 注意：需要先获取当前状态中的 approval_pending，合并审批结果后再更新
    # （LangGraph 默认对 dict 字段是覆盖模式，直接 update 会丢失 from_agent 等字段）
    if req.thread_id:
        try:
            graph = get_graph()
            run_config = {"configurable": {"thread_id": req.thread_id}}

            # 获取当前状态中的 approval_pending
            current_state = graph.get_state(run_config)
            current_values = current_state.values if current_state else {}
            current_approval = dict(current_values.get("approval_pending", {}))

            # 合并审批结果（保留原有字段，只更新状态）
            current_approval.update({
                "status": "approved",
                "reviewer": body.reviewer,
                "comment": body.comment,
                "resolved_at": time.time(),
            })

            # 更新状态 + 恢复执行
            graph.update_state(run_config, {"approval_pending": current_approval})
            graph_result = graph.invoke(Command(resume="approved"), run_config)
            logger.info("图已恢复执行: thread=%s, result=%s", req.thread_id,
                        graph_result.get("final_response", "")[:50] if graph_result else "")

            # 升级工单批准后 → 将人工回复写入消息队列，供 Chat UI 轮询拉取
            if req.type == ApprovalType.ESCALATION and graph_result:
                try:
                    from app.api.agent_routes import store_human_reply
                    hr = graph_result.get("final_response", "")
                    if hr:
                        store_human_reply(req.thread_id, hr)
                        logger.info("人工回复已写入轮询队列: thread=%s", req.thread_id)
                except Exception as e:
                    logger.warning("写入人工回复队列失败: %s", e)
        except Exception as e:
            logger.error("恢复图执行失败: %s", e)

    return ApprovalActionResponse(success=True, message="审批已通过，Agent 继续执行", request_id=request_id)


@router.post("/{request_id}/reject", response_model=ApprovalActionResponse)
async def reject_request(request_id: str, body: RejectRequest):
    """
    拒绝一个审批请求，并自动恢复 LangGraph 图执行。

    审批拒绝后：
      1. 更新 ApprovalRequest 状态为 rejected
      2. 更新 LangGraph 状态中的 approval_pending
      3. 调用 resume_from_checkpoint() 恢复图执行
    """
    req = approval_service.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="审批请求不存在")
    if req.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"审批请求状态为 {req.status.value}，无法操作")

    result = approval_service.reject(request_id, reviewer=body.reviewer, comment=body.comment)
    if result is None:
        raise HTTPException(status_code=500, detail="审批操作失败")

    if req.thread_id:
        try:
            graph = get_graph()
            run_config = {"configurable": {"thread_id": req.thread_id}}
            current_state = graph.get_state(run_config)
            current_values = current_state.values if current_state else {}
            current_approval = dict(current_values.get("approval_pending", {}))
            current_approval.update({
                "status": "rejected",
                "reviewer": body.reviewer,
                "comment": body.comment,
                "resolved_at": time.time(),
            })
            graph.update_state(run_config, {"approval_pending": current_approval})
            graph_result = graph.invoke(Command(resume="rejected"), run_config)
            logger.info("图已恢复执行: thread=%s, result=%s", req.thread_id,
                        graph_result.get("final_response", "")[:50] if graph_result else "")
        except Exception as e:
            logger.error("恢复图执行失败: %s", e)

    return ApprovalActionResponse(success=True, message="审批已拒绝，Agent 继续执行", request_id=request_id)


# ═══════════════════════════════════════════════════════════════
#  内置 Web UI
# ═══════════════════════════════════════════════════════════════

APPROVAL_UI_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>人工客服控制台</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", system-ui, sans-serif;
         background:#f5f6fa; color:#333; height:100vh; display:flex; flex-direction:column; }

  /* 顶部 */
  .header { background:#fff; border-bottom:1px solid #e0e0e0; padding:12px 24px;
            display:flex; justify-content:space-between; align-items:center; }
  .header h1 { font-size:18px; }
  .header .badge { padding:3px 10px; border-radius:12px; font-size:11px; }
  .badge-live { background:#e6f4ea; color:#1e8e3e; }

  /* 主体：左侧工单列表 + 右侧对话窗口 */
  .main { flex:1; display:grid; grid-template-columns:380px 1fr; overflow:hidden; }
  @media (max-width:768px) { .main { grid-template-columns:1fr; } }

  /* 左侧工单列表 */
  .ticket-list { border-right:1px solid #e0e0e0; display:flex; flex-direction:column; overflow:hidden; }
  .ticket-list-header { padding:16px; border-bottom:1px solid #e0e0e0;
                        font-size:14px; font-weight:600; display:flex; justify-content:space-between; }
  .tickets { flex:1; overflow-y:auto; }
  .ticket {
    padding:14px 16px; border-bottom:1px solid #f0f0f0; cursor:pointer;
    transition: background 0.15s; border-left:3px solid transparent;
  }
  .ticket:hover { background:#f9f9f9; }
  .ticket.active { background:#e8f0fe; border-left-color:#4A90D9; }
  .ticket .t-type { font-size:12px; color:#888; }
  .ticket .t-reason { font-size:14px; margin:4px 0; }
  .ticket .t-meta { font-size:11px; color:#aaa; }
  .ticket-row { display:flex; justify-content:space-between; align-items:start; }
  .priority-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; margin-top:6px; }
  .pri-urgent { background:#d93025; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
  .pri-high { background:#e67e22; }
  .pri-normal { background:#1a73e8; }

  /* 右侧对话窗口 */
  .chat-panel { display:flex; flex-direction:column; overflow:hidden; background:#fff; }
  .chat-placeholder { flex:1; display:flex; align-items:center; justify-content:center;
                      color:#bbb; font-size:14px; }
  .chat-area { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  .chat-header { padding:14px 20px; border-bottom:1px solid #e0e0e0;
                 font-size:14px; font-weight:600; display:flex; justify-content:space-between; }
  .chat-messages { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .chat-msg { display:flex; gap:8px; max-width:85%; }
  .chat-msg.customer { align-self:flex-start; }
  .chat-msg.agent { align-self:flex-end; flex-direction:row-reverse; }
  .chat-msg .avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center;
                      justify-content:center; font-size:13px; flex-shrink:0; }
  .chat-msg.customer .avatar { background:#e8f0fe; color:#4A90D9; }
  .chat-msg.agent .avatar { background:#4A90D9; color:#fff; }
  .chat-msg .bubble { padding:8px 12px; border-radius:10px; font-size:13px; line-height:1.6;
                      max-width:100%; word-break:break-word; }
  .chat-msg.customer .bubble { background:#f0f0f0; border-bottom-left-radius:3px; }
  .chat-msg.agent .bubble { background:#4A90D9; color:#fff; border-bottom-right-radius:3px; }
  .chat-msg.system .bubble { background:#fff3e0; color:#333; font-size:12px; font-style:italic; }
  .chat-input-bar { border-top:1px solid #e0e0e0; padding:14px 20px; display:flex; gap:10px; }
  .chat-input-bar textarea { flex:1; padding:10px; border:1px solid #e0e0e0; border-radius:8px;
                              font-size:14px; resize:none; height:60px; outline:none;
                              font-family:inherit; }
  .chat-input-bar textarea:focus { border-color:#4A90D9; }
  .chat-input-bar button { padding:10px 20px; background:#4A90D9; color:#fff; border:none;
                            border-radius:8px; font-size:14px; cursor:pointer; }
  .chat-input-bar button:hover { opacity:0.85; }
  .chat-input-bar button:disabled { opacity:0.5; cursor:not-allowed; }

  .toast { position:fixed; top:20px; right:20px; padding:12px 24px; border-radius:6px;
           color:#fff; font-size:14px; z-index:999; animation:slideIn 0.3s ease; }
  .toast-success { background:#1e8e3e; }
  .toast-error { background:#d93025; }
  @keyframes slideIn { from { transform:translateX(100%);opacity:0; } to { transform:translateX(0);opacity:1; } }
</style>
</head>
<body>

<div class="header">
  <h1>人工客服控制台</h1>
  <span class="badge badge-live" id="statusDot">在线</span>
</div>

<div class="main">
  <!-- 左侧：工单列表 -->
  <div class="ticket-list">
    <div class="ticket-list-header">
      待处理工单 <span id="counter" style="color:#4A90D9">--</span>
    </div>
    <div class="tickets" id="ticketList">
      <div style="text-align:center;color:#bbb;padding:40px">加载中...</div>
    </div>
  </div>

  <!-- 右侧：对话窗口 -->
  <div class="chat-panel" id="chatPanel">
    <div class="chat-placeholder" id="chatPlaceholder">
      选择一个升级工单开始对话
    </div>
    <div class="chat-area" id="chatArea" style="display:none">
      <div class="chat-header">
        <span id="chatTitle">工单详情</span>
        <span style="font-size:12px;color:#888" id="chatTicketId"></span>
      </div>
      <div class="chat-messages" id="chatMessages"></div>
      <div class="chat-input-bar">
        <textarea id="chatInput" placeholder="输入回复内容..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendReply();}"></textarea>
        <button id="sendBtn" onclick="sendReply()">发送回复</button>
      </div>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
const API = '/api/v1/approval';
let activeTicket = null;

function showToast(msg, type) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function formatTime(seconds) {
  if (seconds <= 0) return '已超时';
  const m = Math.floor(seconds / 60);
  return m > 0 ? `${m}分${Math.floor(seconds % 60)}秒` : `${Math.floor(seconds)}秒`;
}

// 解析对话摘要，拆分为用户消息列表
function parseConversation(summary) {
  if (!summary) return [{role:'system', text:'(无对话记录)'}];
  const msgs = [];
  const lines = summary.split('\n');
  for (const line of lines) {
    const m = line.match(/\[(用户|客服|AI)\]:\s*(.*)/);
    if (m) {
      const role = m[1] === '用户' ? 'customer' : 'agent';
      msgs.push({role, text: m[2]});
    }
  }
  if (!msgs.length && summary) {
    msgs.push({role:'system', text: summary.slice(0, 300)});
  }
  return msgs;
}

function selectTicket(ticket) {
  activeTicket = ticket;

  // 高亮
  document.querySelectorAll('.ticket').forEach(t => t.classList.remove('active'));
  document.getElementById(`ticket-${ticket.id}`).classList.add('active');

  // 显示对话窗口
  document.getElementById('chatPlaceholder').style.display = 'none';
  document.getElementById('chatArea').style.display = 'flex';
  document.getElementById('chatTitle').textContent =
    ticket.type === 'escalation' ? '升级转人工' : '审批工单';
  document.getElementById('chatTicketId').textContent = ticket.id;

  // 渲染对话历史
  const ctx = ticket.context || {};
  const summary = ctx.summary || '用户消息: ' + (ctx.user_message || '(无)');
  const msgs = parseConversation(summary);
  let html = '<div class="chat-msg system"><div class="bubble">工单理由: ' + (ticket.reason || '—') + '</div></div>';
  for (const m of msgs) {
    const avatar = m.role === 'customer' ? '客' : 'AI';
    html += `<div class="chat-msg ${m.role}">
      <div class="avatar">${avatar}</div>
      <div class="bubble">${escapeHtml(m.text)}</div>
    </div>`;
  }
  document.getElementById('chatMessages').innerHTML = html;
  document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
  document.getElementById('chatInput').focus();
}

async function sendReply() {
  const text = document.getElementById('chatInput').value.trim();
  if (!text || !activeTicket) return;
  document.getElementById('chatInput').value = '';
  document.getElementById('sendBtn').disabled = true;

  // 在对话区添加人工消息
  const msgsEl = document.getElementById('chatMessages');
  msgsEl.innerHTML += `<div class="chat-msg agent">
    <div class="avatar">我</div>
    <div class="bubble">${escapeHtml(text)}</div>
  </div>`;
  msgsEl.scrollTop = msgsEl.scrollHeight;

  // 调用批准 API（comment = 人工回复内容）
  try {
    const res = await fetch(`${API}/${activeTicket.id}/approve`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer: '人工客服', comment: text}),
    });
    const result = await res.json();
    if (result.success) {
      showToast('回复已发送', 'success');
      activeTicket = null;
      loadTickets();
      // 恢复占位
      document.getElementById('chatArea').style.display = 'none';
      document.getElementById('chatPlaceholder').style.display = 'flex';
    } else {
      showToast(result.message || '发送失败', 'error');
    }
  } catch(e) {
    showToast('网络错误', 'error');
  }
  document.getElementById('sendBtn').disabled = false;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderTicket(r) {
  // 优先级映射：urgent → 红色闪烁, high → 橙色, normal → 蓝色
  let urgency;
  if (r.priority === 'urgent') urgency = 'pri-urgent';
  else if (r.priority === 'high' || r.time_remaining < 300) urgency = 'pri-high';
  else urgency = 'pri-normal';
  const typeCn = r.type === 'escalation' ? '升级转人工' : r.type === 'refund' ? '退款审批' : r.type;
  return `<div class="ticket" id="ticket-${r.id}" onclick="selectTicket(${JSON.stringify(r).replace(/"/g, '&quot;')})">
    <div class="ticket-row">
      <div>
        <div class="t-type">${typeCn} · ${r.from_agent || ''}</div>
        <div class="t-reason">${r.reason || '—'}</div>
        <div class="t-meta">ID: ${r.id} · ${formatTime(r.time_remaining)}</div>
      </div>
      <div class="priority-dot ${urgency}"></div>
    </div>
  </div>`;
}

async function loadTickets() {
  try {
    const res = await fetch(`${API}/list`);
    const data = await res.json();
    document.getElementById('counter').textContent = `${data.length} 个`;
    const el = document.getElementById('ticketList');
    if (!data.length) {
      el.innerHTML = '<div style="text-align:center;color:#bbb;padding:40px">暂无待处理工单</div>';
      return;
    }
    el.innerHTML = data.map(renderTicket).join('');
  } catch(e) {
    document.getElementById('ticketList').innerHTML =
      '<div style="text-align:center;color:#d93025;padding:40px">加载失败</div>';
  }
}

loadTickets();
setInterval(loadTickets, 15000);
</script>
</body>
</html>"""
