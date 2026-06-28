"""
Agent 工作流 API — 聊天端点 + SSE 流式 + 聊天 UI

端点：
  POST /api/v1/agent/chat          — 同步调用 Agent 工作流
  POST /api/v1/agent/chat/stream   — SSE 流式输出，逐节点推送
  GET  /api/v1/agent/chat/ui       — 聊天界面
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.graph.supervisor_graph import run_workflow, astream_workflow
from app.security import (
    get_input_guard, get_output_audit, get_audit_log,
    GuardAction, OutputAction, AuditAction,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
#  Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    user_id: str = ""
    session_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    intent: str = ""
    sentiment: str = "neutral"
    agent_path: list[str] = []
    resolved: bool = False
    quality_score: float = 1.0
    thread_id: str = ""
    security: list = []
    csat_ready: bool = False
    ui_component: dict = {}


class FeedbackRequest(BaseModel):
    session_id: str
    score: int  # 1-5
    feedback: str = ""


class FeedbackResponse(BaseModel):
    success: bool
    message: str = ""


# ═══════════════════════════════════════════════════════════════
#  REST API
# ═══════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def agent_chat(request: ChatRequest):
    """同步调用 Agent 工作流，返回最终结果（含五层纵深防御）。"""
    session_id = request.session_id or uuid.uuid4().hex[:12]
    # 如果前端没传 user_id，自动生成匿名用户 ID（保证长期记忆链路不断）
    user_id = request.user_id or f"ANON_{session_id}"
    trace_id = f"trace_{session_id}"
    audit = get_audit_log()
    security_events: list[dict] = []

    # ══ 第一层: 输入安全网关 ═══════════════════════════════
    input_guard = get_input_guard()
    guard_result = input_guard.check(request.message)
    audit.record(
        trace_id=trace_id, session_id=session_id, actor="input_guard",
        action=AuditAction.INPUT_GUARD,
        input_data={"message": request.message[:200]},
        output_data={"action": guard_result.action.value, "risk": guard_result.risk_level.value},
        risk_level=guard_result.risk_level.value,
    )
    if guard_result.blocked:
        security_events.append({"layer": "input", "action": "block", "reason": guard_result.reason})
        return ChatResponse(
            reply=f"输入安全检测未通过: {guard_result.reason}",
            intent="blocked", thread_id=session_id,
            security=security_events,
        )
    # 使用脱敏后的内容（如有）
    safe_message = guard_result.sanitized_content or request.message

    try:
        result = run_workflow(
            user_message=safe_message,
            user_id=user_id,
            session_id=session_id,
        )

        final_reply = result.get("final_response", "")

        # ══ 第四层: 输出安全审核 ═══════════════════════════
        output_audit = get_output_audit()
        audit_result = output_audit.audit(
            final_reply,
            context={"intent": result.get("intent", ""), "session_id": session_id},
        )
        audit.record(
            trace_id=trace_id, session_id=session_id, actor="output_audit",
            action=AuditAction.OUTPUT_AUDIT,
            input_data={"output": final_reply[:200]},
            output_data={"action": audit_result.action.value, "risk_score": audit_result.risk_score},
            parent_id="",
        )

        if audit_result.blocked:
            security_events.append({"layer": "output", "action": "block", "reason": audit_result.reason})
            final_reply = audit_result.block_replacement
        elif audit_result.action == OutputAction.REDACT:
            security_events.append({"layer": "output", "action": "redact", "pii_count": len(audit_result.pii_matches)})
            final_reply = audit_result.redacted_text

        # ══ 第五层: 审计日志 ═══════════════════════════════
        audit.record_from_state(
            trace_id=trace_id, session_id=session_id, state=result,
            action=AuditAction.OUTPUT, actor="system",
        )

        # ── 记录业务指标 ──────────────────────────────────
        metadata = result.get("metadata", {})
        record_session_metrics(session_id, {
            "intent": result.get("intent", "unknown"),
            "sentiment": result.get("sentiment", "neutral"),
            "resolved": result.get("resolved", False),
            "quality_score": result.get("quality_score", 1.0),
            "escalation": result.get("escalation_flag", False),
            "iteration_count": result.get("iteration_count", 0),
            "agent_path": result.get("agents_sequence", []),
            "duration_ms": metadata.get("observability", {}).get("duration_ms", 0),
        })

        return ChatResponse(
            reply=final_reply,
            intent=result.get("intent", "unknown"),
            sentiment=result.get("sentiment", "neutral"),
            agent_path=result.get("agents_sequence", []),
            resolved=result.get("resolved", False),
            quality_score=result.get("quality_score", 1.0),
            thread_id=session_id,
            security=security_events,
            csat_ready=result.get("metadata", {}).get("csat_ready", False),
            ui_component=result.get("ui_component", {}),
        )
    except Exception as e:
        logger.error("Agent 工作流执行失败: %s", e)
        audit.record(
            trace_id=trace_id, session_id=session_id, actor="system",
            action=AuditAction.ERROR,
            output_data={"error": str(e)},
            risk_level="high",
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", tags=["Agent"])
async def agent_chat_stream(request: ChatRequest):
    """SSE 流式调用 Agent 工作流，结构化事件推送给前端可视化。"""
    session_id = request.session_id or uuid.uuid4().hex[:12]
    user_id = request.user_id or f"ANON_{session_id}"

    async def event_generator():
        previous_node = None
        csat_ready = False
        ui_component = {}
        try:
            async for event in astream_workflow(
                user_message=request.message,
                user_id=user_id,
                session_id=session_id,
            ):
                for node_name, state_update in event.items():
                    if node_name == "__start__":
                        continue

                    node_data = _extract_node_data(node_name, state_update)

                    # ── 检测 CSAT 就绪 + UI 组件 ──
                    if node_data.get("csat_ready"):
                        csat_ready = True
                    if node_data.get("ui_component"):
                        ui_component = node_data["ui_component"]

                    # ── 结束上一个节点 ──
                    if previous_node and previous_node != node_name:
                        prev_meta = _NODE_META.get(previous_node, {})
                        yield _sse("node_done", {
                            "node": previous_node,
                            "label": prev_meta.get("label", previous_node),
                            "icon": prev_meta.get("icon", "gear"),
                        })

                    # ── 新节点开始 ──
                    if previous_node != node_name:
                        node_meta = _NODE_META.get(node_name, {})
                        yield _sse("node_start", {
                            "node": node_name,
                            "label": node_meta.get("label", node_name),
                            "icon": node_meta.get("icon", "gear"),
                        })

                    # ── 节点进度（含工具调用详情）──
                    yield _sse("node_progress", {
                        "node": node_name,
                        "data": node_data,
                    })

                    # ── 流式回复内容（前端打字机动画渲染）──
                    reply = node_data.get("reply", "")
                    if reply:
                        yield _sse("reply_chunk", {"text": reply})

                    previous_node = node_name

            # ── 结束最后一个节点 ──
            if previous_node:
                prev_meta = _NODE_META.get(previous_node, {})
                yield _sse("node_done", {
                    "node": previous_node,
                    "label": prev_meta.get("label", previous_node),
                    "icon": prev_meta.get("icon", "gear"),
                })

            yield _sse("done", {"thread_id": session_id, "csat_ready": csat_ready, "ui_component": ui_component})

        except Exception as e:
            logger.error("SSE 流式执行失败: %s", e)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: dict) -> str:
    """构建 SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ═══════════════════════════════════════════════════════════════
#  人工回复轮询（升级工单批准后，Chat UI 自动拉取人工回复）
# ═══════════════════════════════════════════════════════════════

# 简易消息队列：session_id → 人工回复文本
_human_replies: dict[str, str] = {}


def store_human_reply(session_id: str, reply: str):
    """存储人工回复，供前端轮询拉取。"""
    if session_id:
        _human_replies[session_id] = reply


# ═══════════════════════════════════════════════════════════════
#  CSAT 满意度收集
# ═══════════════════════════════════════════════════════════════

# 简易内存存储：session_id → CSAT 评分记录
_csat_records: dict[str, dict] = {}

# 会话级业务指标（Dashboard 数据源）
_session_metrics: dict[str, dict] = {}


def record_session_metrics(session_id: str, metrics: dict):
    """记录一次会话的业务指标，供 Dashboard 聚合。"""
    if not session_id:
        return
    existing = _session_metrics.get(session_id, {})
    existing.update(metrics)
    existing.setdefault("created_at", __import__("time").time())
    _session_metrics[session_id] = existing


@router.post("/feedback", response_model=FeedbackResponse, tags=["Agent"])
async def submit_feedback(req: FeedbackRequest):
    """收集用户满意度评分（CSAT），并推送到 LangFuse 可观测平台。"""
    # 校验评分范围
    if req.score < 1 or req.score > 5:
        return FeedbackResponse(success=False, message="评分必须在 1-5 之间")

    record = {
        "session_id": req.session_id,
        "score": req.score,
        "feedback": req.feedback,
        "timestamp": __import__("time").time(),
    }
    _csat_records[req.session_id] = record

    # ── 推送到 LangFuse（如果已配置）──
    try:
        from app.observability.tracing import get_langfuse_client
        client = get_langfuse_client()
        if client is not None:
            # 在 LangFuse 中创建一条 Score 记录，关联到对应的 Trace
            trace_id = client.get_current_trace_id()
            score_labels = {1: "非常不满意", 2: "不满意", 3: "一般", 4: "满意", 5: "非常满意"}
            client.create_score(
                trace_id=trace_id,
                name="csat",
                value=float(req.score),
                comment=req.feedback or score_labels.get(req.score, ""),
            )
            logger.info("CSAT 评分已记录: session=%s score=%d", req.session_id, req.score)
    except Exception as e:
        logger.debug("CSAT 推送到 LangFuse 失败（非关键）: %s", e)

    return FeedbackResponse(success=True, message="感谢您的反馈！")


@router.get("/feedback/stats", tags=["Agent"])
async def get_feedback_stats():
    """获取 CSAT 统计数据（Demo 用）"""
    if not _csat_records:
        return {"total": 0, "average": 0.0, "distribution": {}, "recent": []}

    scores = [r["score"] for r in _csat_records.values()]
    avg = sum(scores) / len(scores)
    distribution: dict[str, int] = {}
    for s in range(1, 6):
        distribution[str(s)] = scores.count(s)

    recent = sorted(
        _csat_records.values(),
        key=lambda r: r.get("timestamp", 0),
        reverse=True,
    )[:10]

    return {
        "total": len(scores),
        "average": round(avg, 2),
        "distribution": distribution,
        "recent": recent,
    }


# ═══════════════════════════════════════════════════════════════
#  Dashboard 业务指标
# ═══════════════════════════════════════════════════════════════

@router.get("/dashboard", tags=["Agent"])
async def get_dashboard():
    """聚合业务指标：会话概览、情绪/意图分布、CSAT趋势、质检分布、成本。"""
    import time as _time
    now = _time.time()
    today_start = now - (now % 86400)  # 今天 0 点

    metrics = list(_session_metrics.values())

    # ── 今日筛选 ──────────────────────────────────────────
    today = [m for m in metrics if m.get("created_at", 0) >= today_start]

    # ── 摘要卡片 ──────────────────────────────────────────
    total = len(metrics)
    today_total = len(today)

    # CSAT 统计：从 _csat_records 取（因反馈可能延迟/跳过）
    csat_scores = [r["score"] for r in _csat_records.values()]
    avg_csat = round(sum(csat_scores) / len(csat_scores), 2) if csat_scores else 0.0

    today_csat = [
        r["score"] for r in _csat_records.values()
        if r.get("timestamp", 0) >= today_start
    ]
    today_avg_csat = round(sum(today_csat) / len(today_csat), 2) if today_csat else 0.0

    # 质检
    quality_scores = [m.get("quality_score", 0) for m in metrics if m.get("quality_score")]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0

    # 升级率
    escalated = sum(1 for m in metrics if m.get("escalation"))
    escalation_rate = round(escalated / total * 100, 1) if total > 0 else 0.0

    # 解决率
    resolved_count = sum(1 for m in metrics if m.get("resolved"))
    resolution_rate = round(resolved_count / total * 100, 1) if total > 0 else 0.0

    # ── 情绪分布 ──────────────────────────────────────────
    sentiment_map = {"positive": 0, "neutral": 0, "anxious": 0, "angry": 0, "frustrated": 0}
    for m in metrics:
        s = m.get("sentiment", "neutral")
        if s in sentiment_map:
            sentiment_map[s] += 1

    # ── 意图分布 ──────────────────────────────────────────
    intent_labels = {
        "tech_support": "技术支持", "finance": "财务",
        "after_sale": "售后", "unknown": "其他",
    }
    intent_map: dict[str, int] = {}
    for m in metrics:
        i = m.get("intent", "unknown")
        intent_map[i] = intent_map.get(i, 0) + 1

    # ── CSAT 趋势（最近 7 天） ─────────────────────────────
    csat_trend: list[dict] = []
    for day_offset in range(6, -1, -1):
        day_start = today_start - day_offset * 86400
        day_end = day_start + 86400
        day_scores = [
            r["score"] for r in _csat_records.values()
            if day_start <= r.get("timestamp", 0) < day_end
        ]
        import datetime
        date_str = datetime.date.fromtimestamp(day_start).isoformat()
        csat_trend.append({
            "date": date_str,
            "avg": round(sum(day_scores) / len(day_scores), 2) if day_scores else 0,
            "count": len(day_scores),
        })

    # ── 质检分布 ──────────────────────────────────────────
    quality_buckets = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    for q in quality_scores:
        if q >= 0.9:
            quality_buckets["excellent"] += 1
        elif q >= 0.7:
            quality_buckets["good"] += 1
        elif q >= 0.5:
            quality_buckets["fair"] += 1
        else:
            quality_buckets["poor"] += 1

    # ── 成本 ──────────────────────────────────────────────
    cost_data = {"total_tokens": 0, "total_cost_cny": 0.0}
    try:
        from app.observability import get_cost_tracker
        tracker = get_cost_tracker()
        summary = tracker.get_summary()
        cost_data["total_tokens"] = summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0)
        cost_data["total_cost_cny"] = round(summary.get("total_cost_cny", 0), 4)
    except Exception:
        pass

    # ── 今日会话时间线 ────────────────────────────────────
    today_timeline = sorted(
        [m for m in today],
        key=lambda m: m.get("created_at", 0),
        reverse=True,
    )[:20]

    return {
        "summary": {
            "total_sessions": total,
            "today_sessions": today_total,
            "avg_csat": avg_csat,
            "today_avg_csat": today_avg_csat,
            "avg_quality": avg_quality,
            "escalation_rate": escalation_rate,
            "resolution_rate": resolution_rate,
            "csat_count": len(csat_scores),
        },
        "sentiment_distribution": sentiment_map,
        "intent_distribution": {
            intent_labels.get(k, k): v for k, v in intent_map.items()
        },
        "csat_trend": csat_trend,
        "quality_distribution": quality_buckets,
        "cost": cost_data,
        "today_timeline": today_timeline,
    }


@router.get("/pending", tags=["Agent"])
async def poll_pending_reply(session_id: str = ""):
    """轮询是否有新的人工回复（升级工单被批准后）。"""
    reply = _human_replies.pop(session_id, None)
    return {"has_reply": reply is not None, "reply": reply or ""}


# ── 节点元数据（标签、图标）──────────────────────────────
_NODE_META: dict[str, dict[str, str]] = {
    "classify_intent":       {"label": "分析意图",   "icon": "brain"},
    "extract_context":       {"label": "提取上下文",  "icon": "search"},
    "tech_support_process":  {"label": "技术支持 Agent", "icon": "wrench"},
    "finance_process":       {"label": "财务 Agent",    "icon": "dollar"},
    "after_sale_process":    {"label": "售后 Agent",    "icon": "package"},
    "quality_check":         {"label": "质检评估",      "icon": "shield"},
    "escalation_process":    {"label": "升级处理",      "icon": "warning"},
    "human_approval_node":   {"label": "人工审批",      "icon": "user"},
    "compile_result":        {"label": "汇总回复",      "icon": "message"},
}


def _extract_node_data(node_name: str, state: dict) -> dict:
    """从状态更新中提取前端关心的详细字段。"""
    data: dict = {}
    meta = _NODE_META.get(node_name, {})
    data["label"] = meta.get("label", node_name)
    data["icon"] = meta.get("icon", "gear")

    if node_name == "classify_intent":
        data["intent"] = state.get("intent", "")
        data["confidence"] = state.get("confidence", 0.0)
        data["sentiment"] = state.get("sentiment", "neutral")
    elif node_name == "extract_context":
        slots = state.get("slot_state", {})
        data["slots"] = slots.get("filled_slots", []) if isinstance(slots, dict) else []
    elif node_name in ("tech_support_process", "finance_process", "after_sale_process"):
        data["agent"] = state.get("current_agent", node_name)
        data["resolved"] = state.get("resolved", False)
        tools = state.get("tool_results", [])
        if tools:
            data["tool_calls"] = [
                {"name": t.get("name", "?"), "args": t.get("args", {}), "result": t.get("result")}
                for t in tools if isinstance(t, dict)
            ]
    elif node_name == "quality_check":
        data["quality_score"] = state.get("quality_score", 1.0)
        data["passed"] = data["quality_score"] >= 0.8
        data["escalation_reason"] = state.get("escalation_reason", "")
    elif node_name == "escalation_process":
        data["escalation"] = True
        data["reason"] = state.get("escalation_reason", "")
        data["summary"] = state.get("escalation_summary", "")
    elif node_name == "compile_result":
        data["reply"] = state.get("final_response", "")
        data["resolved"] = state.get("resolved", False)
        data["intent"] = state.get("intent", "")
        meta = state.get("metadata", {})
        if isinstance(meta, dict) and meta.get("csat_ready"):
            data["csat_ready"] = True
        ui = state.get("ui_component", {})
        if ui:
            data["ui_component"] = ui
    elif node_name == "human_approval_node":
        approval = state.get("approval_pending", {})
        data["approval"] = approval if isinstance(approval, dict) else {}
    return data


# ═══════════════════════════════════════════════════════════════
#  Chat UI
# ═══════════════════════════════════════════════════════════════

@router.get("/chat/ui", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui():
    """Agent 聊天界面 - 现代两栏布局，实时 Agent Trace 可视化"""
    import os
    ui_path = os.path.join(os.path.dirname(__file__), "..", "static", "chat_ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    # 兜底：旧版嵌入 UI（兼容）
    return HTMLResponse(content="<h1>Chat UI not found. Please check app/static/chat_ui.html</h1>")

