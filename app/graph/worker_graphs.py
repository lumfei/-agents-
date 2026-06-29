"""
Worker Agent 工作流节点

重大简化（2026-06）：
  - ReAct 循环 → LangGraph create_react_agent() 替代手写 while
  - 工具执行 + ToolMessage → LangGraph 内置处理，不再手写
  - 工具定义 → @tool 函数直接传入 create_react_agent
  - 结构化输出 → with_structured_output(method="function_calling")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.agents.base_agent import BaseAgent, SystemPromptBuilder, IntentClassification
from app.utils.text_utils import strip_markdown_formatting
from app.dependencies import get_llm
from app.graph.dialogue_state import DialogueStateManager, DialogueStage
from app.prompts.registry import get_prompt_registry

# 工具直接导入（@tool 函数就是 BaseTool 对象，可直接传给 create_react_agent）
from app.tools.system_tools import check_service_status, query_user_info, get_system_announcements
from app.tools.knowledge_base import search_knowledge_base
from app.tools.order_tools import query_order, list_user_orders
from app.tools.refund_tools import create_refund, query_refund_status
from app.tools.logistics_tools import track_logistics, query_logistics_by_order

# HITL 审批
from app.human_in_the_loop.approval_schema import ApprovalType
from app.human_in_the_loop.approval_service import approval_service

# 安全模块（仅保留输入校验 + 审计日志，RBAC/护栏在 demo 中移除）
from app.security import (
    get_audit_log, AuditAction,
)
# 推理追踪
from app.observability.tracing import get_reasoning_capture


# 全局实例（懒初始化，避免 memory → graph → worker → memory 循环导入）
_memory_manager = None
_context_engine = None


def _get_memory():
    global _memory_manager
    if _memory_manager is None:
        from app.memory import MemoryManager
        _memory_manager = MemoryManager()
    return _memory_manager


def _get_context():
    global _context_engine
    if _context_engine is None:
        from app.graph.context_engine import ContextEngine
        _context_engine = ContextEngine()
    return _context_engine


# ── 辅助函数 ───────────────────────────────────────────────

def _extract_user_message(state: dict) -> str:
    """从状态中提取用户的最新输入消息（兼容 BaseMessage 和 dict 两种格式）"""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, BaseMessage):
            return msg.content or ""
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _extract_response_text(last_msg: BaseMessage | None) -> str:
    """从最后一条消息中提取文本内容（兼容 DeepSeek 列表格式）"""
    if not last_msg:
        return ""
    content = last_msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts) if parts else str(content)
    return str(content)


def _extract_tool_results(react_messages: list) -> list[dict]:
    """从 ReAct 消息历史中提取所有工具调用 + 返回结果。

    遍历 AIMessage.tool_calls（工具请求）和 ToolMessage（工具结果），
    按 tool_call_id 配对，生成统一的工具调用记录列表。
    """
    results = []
    for msg in react_messages:
        # AIMessage 包含 tool_calls（工具请求）
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                entry = {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                }
                # 记录 tool_call_id 用于后续匹配 ToolMessage
                tc_id = tc.get("id", "")
                if tc_id:
                    entry["_tc_id"] = tc_id
                results.append(entry)

        # ToolMessage 包含工具返回值
        if hasattr(msg, "type") and msg.type == "tool":
            tc_id = getattr(msg, "tool_call_id", "")
            content = msg.content
            # 尝试解析 JSON 字符串
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass
            # 反向查找匹配的 tool_call 并附加 result
            for r in reversed(results):
                if tc_id and r.get("_tc_id") == tc_id:
                    r["result"] = content
                    break
                # 没有 tool_call_id 时，附加到最近的未匹配条目
                if not tc_id and "result" not in r:
                    r["result"] = content
                    break

    # 清理内部 _tc_id 字段
    for r in results:
        r.pop("_tc_id", None)

    return results


# ═══════════════════════════════════════════════════════════════
#  节点 1：意图分类（Supervisor）
# ═══════════════════════════════════════════════════════════════

def classify_intent(state: dict) -> dict:
    """意图分类 — 使用 with_structured_output() 获取结构化的分类结果"""
    user_message = _extract_user_message(state)
    if not user_message:
        return {"intent": "unknown", "confidence": 0.0, "current_agent": "supervisor"}

    llm = get_llm()
    agent = BaseAgent(name="supervisor", llm=llm,
                      system_prompt=SystemPromptBuilder.supervisor_prompt())

    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    audit = get_audit_log()

    # ── 构建上下文（含对话历史，用于理解指代和追问） ──
    context: dict = {"state": "initial_classification"}
    if session_id:
        mem_ctx = _get_memory().retrieve_context(
            session_id=session_id,
            user_id=user_id,
            current_message=user_message,
        )
        short_term = mem_ctx.get("short_term", [])
        if short_term:
            recent = short_term[-6:]  # 最近 3 轮对话
            context["conversation_history"] = "\n".join(
                f"[{m['role']}]: {m['content'][:300]}" for m in recent
            )

    try:
        result = agent.process_structured(
            message=user_message,
            output_schema=IntentClassification,
            context=context,
        )

        # ── 估算分类 LLM 调用的 Token 消耗 ──────────────
        # 系统提示词约 500 tokens + 用户消息约 len/4 tokens + 输出约 100 tokens
        sys_prompt_estimate = 500
        input_estimate = sys_prompt_estimate + max(len(user_message) // 4, 10)
        output_estimate = 100
        classify_tokens = {"input_tokens": input_estimate, "output_tokens": output_estimate}

        # 审计: 意图分类
        audit.record(
            trace_id=f"trace_{session_id}", session_id=session_id,
            actor="supervisor", action=AuditAction.CLASSIFY,
            input_data={"message": user_message[:200]},
            output_data={"intent": result.intent, "confidence": result.confidence},
        )
        # ── 提取情绪（兼容旧版模型可能不返回 sentiment 字段） ──
        sentiment = getattr(result, "sentiment", "neutral") or "neutral"
        # 确保 sentiment 值合法
        valid_sentiments = {"positive", "neutral", "anxious", "angry", "frustrated"}
        if sentiment not in valid_sentiments:
            sentiment = "neutral"

        # ── 检测对话结束意图 ──────────────────────────────────
        # 只有用户表达了结束/满意/告别，才触发 CSAT 满意度收集
        ending_keywords = [
            "谢谢", "感谢", "多谢", "太好了", "解决了", "完美",
            "没问题了", "没有了", "就这样", "可以了", "行了",
            "再见", "拜拜", "bye", "没事了", "不用了",
            "知道了", "收到", "好嘞", "明白了",
        ]
        user_lower = user_message.lower().strip()
        # 精确匹配：用户消息主要由结束语构成（避免误触发）
        conversation_ending = any(kw in user_lower for kw in ending_keywords)

        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "sentiment": sentiment,
            "extracted_entities": result.extracted_entities,
            "current_agent": "supervisor",
            "agents_sequence": ["supervisor"],
            "classify_token_usage": classify_tokens,
            "conversation_ending": conversation_ending,
        }
    except Exception as e:
        audit.record(
            trace_id=f"trace_{session_id}", session_id=session_id,
            actor="supervisor", action=AuditAction.ERROR,
            output_data={"error": str(e)},
            risk_level="medium",
        )
        return {
            "intent": "unknown", "confidence": 0.0, "current_agent": "supervisor",
            "metadata": {**state.get("metadata", {}), "classification_error": str(e)},
        }


# ═══════════════════════════════════════════════════════════════
#  节点 2：上下文提取
# ═══════════════════════════════════════════════════════════════

def extract_context(state: dict) -> dict:
    """提取上下文 + 初始化槽位追踪（纯逻辑，不调 LLM）"""
    intent = state.get("intent", "unknown")
    entities = state.get("extracted_entities", {})
    session_id = state.get("session_id", "")
    dsm = DialogueStateManager(intent_type=intent)
    dsm.set_stage(DialogueStage.COLLECTING_INFO)
    dsm.batch_fill(entities)
    for msg in state.get("messages", []):
        if isinstance(msg, BaseMessage) and msg.type == "human":
            dsm._extract_entities(msg.content or "")
        elif isinstance(msg, dict) and msg.get("role") == "user":
            dsm._extract_entities(msg.get("content", ""))

    updates = {"slot_state": dsm.to_dict()}
    if session_id:
        _get_memory().get_or_create_session(session_id)
        updates["metadata"] = {**state.get("metadata", {}), "session_initialized": True}
    return updates


# ═══════════════════════════════════════════════════════════════
#  3 个 Worker 节点
# ═══════════════════════════════════════════════════════════════

def tech_support_process(state: dict) -> dict:
    registry = get_prompt_registry()
    role_info = registry.get_worker_role("tech_support")
    return _worker_process(
        state=state, agent_name="tech_support",
        agent_role=role_info.get("agent_role", ""),
        responsibilities=role_info.get("responsibilities", ""),
        tools=[check_service_status, get_system_announcements, search_knowledge_base, query_user_info],
    )


def finance_process(state: dict) -> dict:
    registry = get_prompt_registry()
    role_info = registry.get_worker_role("finance")
    return _worker_process(
        state=state, agent_name="finance",
        agent_role=role_info.get("agent_role", ""),
        responsibilities=role_info.get("responsibilities", ""),
        tools=[query_order, list_user_orders, create_refund, query_refund_status],
    )


def after_sale_process(state: dict) -> dict:
    registry = get_prompt_registry()
    role_info = registry.get_worker_role("after_sale")
    return _worker_process(
        state=state, agent_name="after_sale",
        agent_role=role_info.get("agent_role", ""),
        responsibilities=role_info.get("responsibilities", ""),
        tools=[track_logistics, query_logistics_by_order, query_order, search_knowledge_base],
    )


# ═══════════════════════════════════════════════════════════════
#  Worker 通用处理逻辑
# ═══════════════════════════════════════════════════════════════

def _worker_process(state: dict, agent_name: str, agent_role: str,
                    responsibilities: str = "", tools: list = None) -> dict:
    """
    Worker 通用逻辑，使用 LangGraph create_react_agent。

    与旧版的区别：
      旧版：agent.react_process() → 手写 while 循环 + 手动 ToolMessage
      ↓
      新版：create_react_agent() → LangGraph 自带的 ReAct 循环

    create_react_agent 会自动处理：
      1. LLM 思考（收到用户消息，决定调什么工具）
      2. 工具执行（调用 @tool 函数，获取结果）
      3. 结果回传（ToolMessage 自动格式化）
      4. 循环直到 LLM 不再调用工具
    """
    user_message = _extract_user_message(state)
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    iteration = state.get("iteration_count", 0) + 1
    slot_state = state.get("slot_state", {})

    # ── 1. 构建系统提示词 ──────────────────────────────────
    system_prompt = SystemPromptBuilder.worker_base_prompt(
        agent_name=agent_role, responsibilities=responsibilities or agent_role,
        tools_desc=f"可用工具: {', '.join(t.name if hasattr(t, 'name') else str(t) for t in (tools or []))}",
    )
    # 如果会话已绑定 user_id，告诉 Agent 避免反复追问
    if user_id:
        system_prompt += (
            f"\n\n【当前用户信息】\n"
            f"用户ID: {user_id}\n"
            f"在调用需要 user_id 参数的工具时，直接使用 {user_id}，不要再向用户追问。"
        )
    extra_context = ""
    if slot_state and slot_state.get("missing_slots"):
        extra_context += f"\n【待收集信息】{', '.join(slot_state['missing_slots'])}"

    # ── 2. 构建完整系统提示词（含上下文） ─────────────────
    if extra_context:
        system_prompt += f"\n\n{extra_context}"

    # ── 3. 使用 LangGraph 内置 ReAct Agent ────────────────
    llm = get_llm()
    react = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    try:
        # ── 从记忆系统取对话历史，注入 LLM 上下文 ────────
        memory = _get_memory()
        history_messages: list = []
        if session_id:
            mem_ctx = memory.retrieve_context(
                session_id=session_id,
                user_id=user_id,
                current_message=user_message,
            )
            # 短期记忆：转换为 LangChain message 对象
            for m in mem_ctx.get("short_term", []):
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "user":
                    history_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    history_messages.append(AIMessage(content=content))
            # 长期记忆/用户画像：拼入系统提示词
            if mem_ctx.get("long_term"):
                system_prompt += f"\n\n{mem_ctx['long_term']}"

        # 追加当前用户消息
        history_messages.append(HumanMessage(content=user_message))

        # create_react_agent.invoke 内部处理整个 ReAct 循环
        result = react.invoke({"messages": history_messages})

        all_messages = result.get("messages", [])
        last_msg = all_messages[-1] if all_messages else None
        response_text = _extract_response_text(last_msg)
        tool_calls = _extract_tool_results(all_messages)

        # ── 捕获推理过程（DeepSeek Thinking Token） ──────────
        reasoning = get_reasoning_capture()
        for i, msg in enumerate(all_messages):
            if hasattr(msg, "additional_kwargs"):
                reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
            elif isinstance(msg, dict):
                reasoning_content = msg.get("additional_kwargs", {}).get("reasoning_content", "")
            else:
                reasoning_content = ""
            if reasoning_content:
                reasoning.record_reasoning(
                    trace_id=f"trace_{session_id}",
                    agent=agent_name,
                    reasoning_content=reasoning_content,
                    reasoning_tokens=len(reasoning_content) // 4,  # 粗略估计
                    step=f"react_step_{i}",
                )

        # ── 4. 工具调用审计日志 ──
        audit = get_audit_log()
        trace_id = f"trace_{session_id}"

        for tc in tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {})

            logger.debug("[TOOL CALL] tool=%s, args=%s", tc_name, tc_args)
            audit.record(
                trace_id=trace_id, session_id=session_id, actor=agent_name,
                action=AuditAction.TOOL_CALL,
                input_data={"tool": tc_name, "args": {k: str(v)[:100] for k, v in tc_args.items()}},
                risk_level="low",
            )

            # ── RBAC/ABAC + 策略护栏已移除 ──
            # 原因：Demo 环境无用户认证，默认 role=user 权限不足导致所有工具被拦截。
            # 完整实现保留在 app/security/permissions.py 和 policy_engine.py 中。

        # ── 5. 检查 HITL 审批 ─────────────────────────────
        # 当工具调用触发审批条件时：
        #   1. 通过 approval_service 创建审批请求（自动发通知）
        #   2. 将审批信息写入 AgentState.approval_pending
        #   3. 图的 interrupt_before 会在 human_approval_node 前暂停
        approval_pending = {}
        resolved = True
        for tc in tool_calls:
            if tc.get("name") == "create_refund":
                amount = tc.get("args", {}).get("amount", 0)
                if amount > 1000:
                    # 创建审批请求 → 自动触发通知
                    req = approval_service.create_request(
                        approval_type=ApprovalType.REFUND,
                        from_agent=agent_name,
                        thread_id=session_id,  # session_id 即 LangGraph thread_id
                        context={
                            "amount": amount,
                            "order_id": tc["args"].get("order_id", ""),
                            "reason": tc["args"].get("reason", ""),
                        },
                        reason=f"退款金额 ¥{amount} 超过阈值，需要人工审批",
                        priority="high" if amount > 5000 else "normal",
                    )
                    approval_pending = {
                        "request_id": req.id,
                        "type": "refund",
                        "amount": amount,
                        "order_id": tc["args"].get("order_id", ""),
                        "reason": tc["args"].get("reason", ""),
                        "status": "pending",
                        "created_at": time.time(),
                        "from_agent": agent_name,
                    }
                    resolved = False

        # ── 5. 提取 Token 使用量（从 ReAct Agent 内部消息中汇总） ──
        worker_input_tokens = 0
        worker_output_tokens = 0
        for msg in all_messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                meta = msg.usage_metadata
                worker_input_tokens += meta.get("input_tokens", 0) or 0
                worker_output_tokens += meta.get("output_tokens", 0) or 0

        # ── 6. 记忆存储 ───────────────────────────────────
        if session_id:
            _get_memory().store_interaction(
                session_id=session_id, user_message=user_message,
                assistant_message=response_text, user_id=user_id,
                tool_calls=tool_calls,
            )

        return {
            "current_agent": agent_name,
            "agents_sequence": [agent_name],  # Reducer 自动追加
            "iteration_count": iteration,
            "final_response": strip_markdown_formatting(response_text),
            "resolved": resolved,
            "tool_results": tool_calls,
            "approval_pending": approval_pending,
            "worker_token_usage": {
                "input_tokens": worker_input_tokens,
                "output_tokens": worker_output_tokens,
            },
        }

    except Exception as e:
        return {
            "current_agent": agent_name,
            "iteration_count": iteration,
            "final_response": f"处理出错: {str(e)}",
            "resolved": False,
            "escalation_flag": True,
            "escalation_reason": f"{agent_name} 处理异常: {str(e)}",
        }


# ═══════════════════════════════════════════════════════════════
#  节点 6：质检
# ═══════════════════════════════════════════════════════════════

def quality_check(state: dict) -> dict:
    """质检节点——规则评分 + 情感维度（后续可升级为 LLM-as-Judge）"""
    resolved = state.get("resolved", False)
    final_response = state.get("final_response", "")
    tool_results = state.get("tool_results", [])
    escalation = state.get("escalation_flag", False)
    sentiment = state.get("sentiment", "neutral")
    session_id = state.get("session_id", "")
    audit = get_audit_log()

    score = 0.0
    report = []
    if final_response and len(final_response) > 5:
        score += 0.3
        report.append("有回复内容")
    if resolved:
        score += 0.3
        report.append("问题标记为已解决")
    if tool_results:
        score += 0.2
        report.append(f"调用了 {len(tool_results)} 个工具")
    if escalation:
        score -= 0.3
        report.append("已触发升级标记")
    for kw in ["密码", "银行卡", "身份证", "验证码"]:
        if kw in final_response:
            score -= 0.5
            report.append(f"包含敏感关键词: {kw}")
            break

    # ── 情感维度评分调整 ───────────────────────────────────
    # 愤怒/沮丧用户的回复需要更高质量标准
    sentiment_penalty = 0.0
    if sentiment == "angry":
        # 愤怒用户：回复是否包含道歉/安抚措辞
        apology_keywords = ["抱歉", "对不起", "理解您", "非常抱", "给您带来", "尽快", "优先"]
        has_apology = any(kw in final_response for kw in apology_keywords)
        if not has_apology:
            sentiment_penalty -= 0.15
            report.append("愤怒用户缺少安抚措辞")
        # 愤怒用户没解决 → 加重扣分
        if not resolved:
            sentiment_penalty -= 0.1
            report.append("愤怒用户问题未解决")
    elif sentiment == "anxious":
        # 焦虑用户：回复是否告知进度/预计时间
        progress_keywords = ["预计", "进度", "尽快", "正在", "已经", "当前状态", "会尽快"]
        has_progress = any(kw in final_response for kw in progress_keywords)
        if not has_progress:
            sentiment_penalty -= 0.1
            report.append("焦虑用户缺少进度告知")
    elif sentiment == "frustrated":
        # 沮丧用户：回复是否包含共情
        empathy_keywords = ["理解", "明白", "感受", "帮您", "一起"]
        has_empathy = any(kw in final_response for kw in empathy_keywords)
        if not has_empathy:
            sentiment_penalty -= 0.1
            report.append("沮丧用户缺少共情回应")

    score += sentiment_penalty

    final_score = round(max(0.0, min(1.0, score)), 2)
    quality_report = "; ".join(report) if report else "质检通过"

    # 审计: 质检评分
    audit.record(
        trace_id=f"trace_{session_id}", session_id=session_id,
        actor="quality_check", action=AuditAction.QUALITY_CHECK,
        input_data={"resolved": resolved, "tool_count": len(tool_results), "sentiment": sentiment},
        output_data={"score": final_score, "report": quality_report},
        risk_level="high" if final_score < 0.5 else ("medium" if final_score < 0.8 else "low"),
    )

    return {"quality_score": final_score, "quality_report": quality_report}


# ═══════════════════════════════════════════════════════════════
#  节点 7：升级处理
# ═══════════════════════════════════════════════════════════════

def escalation_process(state: dict) -> dict:
    """升级处理——生成上下文摘要，创建人工客服工单（含情感优先级）"""
    user_msg = _extract_user_message(state)
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    sentiment = state.get("sentiment", "neutral")

    # ── 检索完整对话历史，生成有意义的摘要 ──
    conversation_text = ""
    if session_id:
        mem_ctx = _get_memory().retrieve_context(
            session_id=session_id, user_id=user_id, current_message=user_msg,
        )
        short_term = mem_ctx.get("short_term", [])
        if short_term:
            lines = []
            for m in short_term[-12:]:  # 最近 6 轮对话
                role_label = "用户" if m["role"] == "user" else "客服"
                lines.append(f"[{role_label}]: {m['content'][:200]}")
            conversation_text = "\n".join(lines)

    parts = [
        "【升级转人工 — 上下文摘要】",
        f"意图分类: {state.get('intent', 'unknown')}",
        f"用户情绪: {sentiment}",
        f"处理轮次: {state.get('iteration_count', 0)}",
        f"经手Agent: {' → '.join(state.get('agents_sequence', []))}",
    ]
    if conversation_text:
        parts.append(f"\n--- 对话历史 ---\n{conversation_text}")

    escalation_summary = "\n".join(parts)
    intent = state.get("intent", "unknown")

    # ── 判断升级原因 ──
    is_escalation_request = any(kw in user_msg for kw in [
        "转人工", "人工客服", "找人工", "总结", "汇总", "整理",
    ])

    # ── 无法识别意图（如"你好"）→ 友好引导，不建工单，不转人工 ──
    if intent == "unknown" and not is_escalation_request:
        return {
            "escalation_flag": False,
            "final_response": (
                "你好！我是智能客服助手，可以帮您处理以下问题：\n"
                "🔧 【技术支持】：系统故障、软件问题、蓝屏报错等\n"
                "💰 【财务查询】：订单查询、退款申请、发票等\n"
                "📦 【物流追踪】：快递查询、退换货等\n\n"
                "请描述您遇到的问题，我会尽力帮您解决！"
            ),
            "resolved": True,
        }

    # ── 情感驱动的优先级判定 ──────────────────────────────
    quality_score = state.get("quality_score", 0)
    if is_escalation_request:
        reason = "用户主动请求转人工"
        # 用户主动转人工 + 愤怒情绪 = urgent
        if sentiment == "angry":
            priority = "urgent"
            reason += "（用户情绪愤怒，需紧急处理）"
        elif sentiment in ("anxious", "frustrated"):
            priority = "high"
        else:
            priority = "normal"
    else:
        reason = state.get("escalation_reason", "质检未通过") or "质检未通过"
        # 情绪驱动的优先级
        if sentiment == "angry":
            priority = "urgent"
            reason += "（愤怒用户自动升级）"
        elif sentiment == "anxious":
            priority = "high"
        elif quality_score < 0.5:
            priority = "high"
        else:
            priority = "normal"

    # ── 情感安抚前缀 ─────────────────────────────────────
    sentiment_prefix = ""
    if sentiment == "angry":
        sentiment_prefix = "非常抱歉给您带来这么糟糕的体验，我立刻为您优先处理。"
    elif sentiment == "anxious":
        sentiment_prefix = "理解您很着急，我马上帮您加急处理。"
    elif sentiment == "frustrated":
        sentiment_prefix = "非常理解您的感受，我来帮您解决这个问题。"

    approval_id = ""
    try:
        from app.human_in_the_loop.approval_service import approval_service as _approval_svc
        from app.human_in_the_loop.approval_schema import ApprovalType
        req = _approval_svc.create_request(
            approval_type=ApprovalType.ESCALATION,
            from_agent=intent,
            thread_id=session_id,
            context={
                "user_message": user_msg[:200],
                "summary": escalation_summary,
                "intent": intent,
                "sentiment": sentiment,
                "quality_score": quality_score,
                "agent_path": state.get("agents_sequence", []),
            },
            reason=reason,
            priority=priority,
        )
        approval_id = req.id
        logger.info("工单已创建: %s, sentiment=%s, priority=%s, reason=%s", approval_id, sentiment, priority, reason)
    except Exception as e:
        logger.exception("工单创建失败: %s", e)

    # ── 转人工 → interrupt_before 暂停，等待人工处理 ──
    return {
        "escalation_flag": True,
        "escalation_reason": reason,
        "escalation_summary": escalation_summary,
        "approval_pending": {"request_id": approval_id, "type": "escalation", "status": "pending", "priority": priority},
        "final_response": (
            (sentiment_prefix + "\n\n" if sentiment_prefix else "")
            + f"好的，已为您创建工单（编号: {approval_id}），人工客服将尽快跟进处理。"
            + (f"\n\n以下是问题摘要：\n{escalation_summary}" if conversation_text else "")
        ),
        "resolved": False,
    }


# ═══════════════════════════════════════════════════════════════
#  节点 8：HITL 审批
# ═══════════════════════════════════════════════════════════════

def human_approval_node(state: dict) -> dict:
    """
    HITL 审批节点（interrupt_before 模式）。

    由于 graph.compile(interrupt_before=["human_approval_node"])，
    此节点在 interrupt 解除后才执行。

    退款审批：approval_pending 已由外部 API 更新状态 → Worker 继续
    升级审批：approval_pending 已由外部 API 更新状态 → 人工回复直接输出
    """
    approval = state.get("approval_pending", {})
    status = approval.get("status", "pending")
    is_escalation = approval.get("type") == "escalation"

    if status == "approved":
        reviewer = approval.get("reviewer", "审批人")
        comment = approval.get("comment", "")

        if is_escalation:
            # 升级工单批准 → 人工的 comment 作为直接回复
            if comment:
                reply = f"【人工客服回复】\n{comment}\n\n— {reviewer}"
            else:
                reply = f"人工客服 {reviewer} 已接入，请问还需要什么帮助？"
            return {
                "final_response": reply,
                "resolved": True,
            }
        else:
            # 退款审批 → 回到 Worker 继续
            msg = f"审批已通过（审批人: {reviewer}）"
            if comment:
                msg += f"，意见: {comment}"
            return {
                "final_response": msg,
                "resolved": False,
            }

    elif status == "rejected":
        comment = approval.get("comment", "未提供原因")
        if is_escalation:
            reply = f"抱歉，人工客服暂时无法处理您的请求。原因: {comment}"
        else:
            reply = f"很抱歉，您的申请未通过审批。原因: {comment}"
        return {
            "final_response": reply,
            "resolved": True,
        }
    else:
        return {
            "escalation_flag": True,
            "escalation_reason": "审批超时或状态异常",
            "final_response": "审批处理异常，正在转人工处理...",
            "resolved": False,
        }


# ═══════════════════════════════════════════════════════════════
#  节点 9：结果汇总
# ═══════════════════════════════════════════════════════════════

def _build_ui_component(tool_results: list) -> dict | None:
    """从工具调用结果中构建动态 UI 组件。

    当前支持的组件类型：
      - logistics_tracking_card：物流轨迹追踪卡片
    """
    if not tool_results:
        return None

    for tc in tool_results:
        result = tc.get("result", {})
        # ToolMessage 可能返回未解析的 JSON 字符串
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(result, dict):
            continue

        # 物流追踪：tracking_events 是其特征字段
        if "tracking_events" in result:
            return {
                "type": "logistics_tracking_card",
                "data": {
                    "tracking_no": result.get("tracking_no", ""),
                    "company": result.get("company", ""),
                    "current_status": result.get("current_status", ""),
                    "current_location": result.get("current_location", ""),
                    "estimated_delivery": result.get("estimated_delivery", ""),
                    "events": result.get("tracking_events", []),
                },
            }

    return None


def compile_result(state: dict) -> dict:
    """结果汇总——收尾，归档会话。仅在用户表达结束意图时标记可收集 CSAT。"""
    final_response = state.get("final_response", "")
    resolved = state.get("resolved", False)
    conversation_ending = state.get("conversation_ending", False)

    meta_update = {"completed_at": time.time()}
    # 只有用户说了"谢谢/好的/解决了/再见"等结束语，才弹出 CSAT 评分
    if conversation_ending:
        meta_update["csat_ready"] = True

    updates = {
        "metadata": {**state.get("metadata", {}), **meta_update},
        "final_response": final_response,
        "resolved": resolved,
    }

    # ── Generative UI：检测物流等动态组件 ──
    tool_results = state.get("tool_results", [])
    ui_component = _build_ui_component(tool_results)
    if ui_component:
        updates["ui_component"] = ui_component

    if final_response and resolved:
        msgs = list(state.get("messages", []))
        msgs.append({"role": "assistant", "content": final_response})
        updates["messages"] = msgs

    # 注意：不在此处 archive_session，多轮对话需要短期记忆持续存在。
    # 会话真正的归档由外部触发（会话超时 / 显式结束 / 定时清理）。
    # if state.get("session_id"):
    #     _get_memory().archive_session(state["session_id"], state.get("user_id", ""))
    return updates


# ═══════════════════════════════════════════════════════════════
#  节点映射表
# ═══════════════════════════════════════════════════════════════

NODE_FUNCTIONS = {
    "classify_intent": classify_intent,
    "extract_context": extract_context,
    "tech_support_process": tech_support_process,
    "finance_process": finance_process,
    "after_sale_process": after_sale_process,
    "quality_check": quality_check,
    "escalation_process": escalation_process,
    "human_approval_node": human_approval_node,
    "compile_result": compile_result,
}
