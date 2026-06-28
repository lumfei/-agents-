"""
条件路由逻辑 — 状态图中的条件边判断

这个文件定义了状态图中"岔路口"的判断逻辑。
根据当前状态决定下一步走哪条路。

路由决策的依据：
  1. 意图分类结果（tech_support / finance / after_sale / unknown）
  2. 迭代次数（>3 触发升级）
  3. 质检评分（<0.8 强制升级）
  4. 用户主动要求转人工
  5. HITL 审批状态

这些函数会被 LangGraph 的条件边（conditional_edges）调用：
  graph.add_conditional_edges("classify_intent", supervisor_routing, {...})
"""

from __future__ import annotations

from typing import Literal


# ═══════════════════════════════════════════════════════════════
#  路由标签（用于条件边的输出）
# ═══════════════════════════════════════════════════════════════

# Worker 路由目标
ROUTE_TECH = "tech_support"
ROUTE_FINANCE = "finance"
ROUTE_AFTER_SALE = "after_sale"
ROUTE_ESCALATE = "escalation"
ROUTE_END = "end"
ROUTE_QUALITY = "quality_check"
ROUTE_HUMAN = "human_approval"


# ═══════════════════════════════════════════════════════════════
#  路由函数
#
#  每个路由函数接收 state（AgentState 字典），返回字符串标签。
#  LangGraph 根据返回的标签匹配对应的节点或边。
# ═══════════════════════════════════════════════════════════════

def supervisor_routing(state: dict) -> str:
    """
    Supervisor 完成意图分类后的路由决策。

    决策逻辑：
      - intent=tech_support  → 路由到技术支持 Agent
      - intent=finance       → 路由到财务 Agent
      - intent=after_sale    → 路由到售后 Agent
      - intent=unknown       → 路由到升级 Agent（无法分类）
    """
    intent = state.get("intent", "unknown")

    if intent == "tech_support":
        return ROUTE_TECH
    elif intent == "finance":
        return ROUTE_FINANCE
    elif intent == "after_sale":
        return ROUTE_AFTER_SALE
    else:
        return ROUTE_ESCALATE


def worker_completion_routing(state: dict) -> str:
    """
    Worker Agent 处理完成后的路由决策（含情感感知）。

    判断：
      - 问题已解决（resolved=True）→ 质检
      - 迭代次数超限              → 升级
      - 需要审批（有 approval_pending）→ 暂停等审批
      - 还需要更多信息             → 继续当前 Worker（但限制迭代）

    防死循环保护：
      iteration_count >= max_iterations → 强制升级

    情感加速：
      angry/anxious 用户更早进入升级流程
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 10)
    approval = state.get("approval_pending", {})
    resolved = state.get("resolved", False)
    sentiment = state.get("sentiment", "neutral")

    # 有 HITL 审批待处理 → 暂停等审批
    if approval:
        return ROUTE_HUMAN

    # ── 情感驱动的加速升级 ──────────────────────────────
    # 愤怒用户：2 次未解决即升级
    if sentiment == "angry" and iteration >= 2 and not resolved:
        return ROUTE_ESCALATE
    # 焦虑用户：3 次未解决即升级
    if sentiment == "anxious" and iteration >= 3 and not resolved:
        return ROUTE_ESCALATE

    # 超过最大迭代 → 强制升级
    if iteration >= max_iter:
        return ROUTE_ESCALATE

    # 问题已解决 → 质检
    if resolved:
        return ROUTE_QUALITY

    # 未解决但已给出有意义的回复（如反问用户、等待用户输入）
    # → 去质检，不循环回 Worker（避免死循环导致强制升级）
    final_response = state.get("final_response", "")
    if final_response and len(final_response) > 10:
        return ROUTE_QUALITY

    # 还在处理中 → 继续当前 Worker
    current = state.get("current_agent", "")
    if current == "tech_support":
        return ROUTE_TECH
    elif current == "finance":
        return ROUTE_FINANCE
    elif current == "after_sale":
        return ROUTE_AFTER_SALE
    else:
        return ROUTE_QUALITY


def quality_routing(state: dict) -> str:
    """
    质检完成后的路由决策（含情感感知）。

    判断：
      - 质检合格（quality_score >= 阈值）→ 结束
      - 质检不合格（< 阈值）→ 触发升级
      - 多次质检不合格且之前已经升级过 → 强制结束

    情感阈值调整：
      - angry 用户阈值提升到 0.85（需要更高质量的回复）
      - anxious 用户阈值不变（0.8）
      - neutral/positive 用户阈值不变（0.8）
    """
    score = state.get("quality_score", 1.0)
    escalation = state.get("escalation_flag", False)
    resolved = state.get("resolved", True)
    final_response = state.get("final_response", "")
    sentiment = state.get("sentiment", "neutral")

    # ── 情感驱动的质检阈值 ──────────────────────────────
    if sentiment == "angry":
        threshold = 0.85  # 愤怒用户需要更高质量回复
    else:
        threshold = 0.8

    if not resolved:
        # 如果 Agent 给出了有意义的回复（如反问用户补充信息），
        # 说明 Agent 完成了本轮的职责，不应强制升级
        if final_response and len(final_response) > 10:
            return ROUTE_END
        return ROUTE_ESCALATE

    if score >= threshold:
        return ROUTE_END

    # 评分低于阈值但有有意义的回复（已解决或反问中）
    # → 直接结束，不升级（Agent 已尽力，升级不会更好）
    if final_response and len(final_response) > 10:
        return ROUTE_END

    # 评分低于阈值且无有效回复 → 升级
    if not escalation:
        return ROUTE_ESCALATE

    # 已经升级过了还不行 → 强制结束降级处理
    return ROUTE_END


def escalation_routing(state: dict) -> str:
    """
    升级流程结束后，判断后续：
      - 升级到人工客服 → 进入 HITL 审批节点（interrupt_before 暂停）
      - 降级回 Worker 重试 → 继续处理
    """
    escalation_summary = state.get("escalation_summary", "")
    if escalation_summary:
        # 已生成升级摘要 → 进入人工审批节点（interrupt_before 暂停，等待人工处理）
        return ROUTE_HUMAN
    else:
        # 尝试降级回 Worker 重试
        intent = state.get("intent", "unknown")
        if intent == "tech_support":
            return ROUTE_TECH
        elif intent == "finance":
            return ROUTE_FINANCE
        elif intent == "after_sale":
            return ROUTE_AFTER_SALE
        return ROUTE_END


def approval_routing(state: dict) -> str:
    """
    HITL 审批完成后的路由决策。

    退款审批 → 回到 Worker 继续
    升级审批 → 汇总回复结束
    """
    approval = state.get("approval_pending", {})

    # 升级工单审批 → 直接汇总回复
    if approval.get("type") == "escalation":
        return ROUTE_END

    status = approval.get("status", "")
    if status == "approved":
        prev_agent = approval.get("from_agent", "")
        if prev_agent == "finance":
            return ROUTE_FINANCE
        return ROUTE_QUALITY
    elif status == "rejected":
        return ROUTE_END
    else:
        return ROUTE_ESCALATE


# ═══════════════════════════════════════════════════════════════
#  路由映射表（给 add_conditional_edges 用）
# ═══════════════════════════════════════════════════════════════

# Supervisor 分流路由映射
SUPERVISOR_ROUTE_MAP = {
    ROUTE_TECH: "tech_support_process",
    ROUTE_FINANCE: "finance_process",
    ROUTE_AFTER_SALE: "after_sale_process",
    ROUTE_ESCALATE: "escalation_process",
    ROUTE_QUALITY: "quality_check",
}

# Worker 完成路由映射
WORKER_ROUTE_MAP = {
    ROUTE_TECH: "tech_support_process",
    ROUTE_FINANCE: "finance_process",
    ROUTE_AFTER_SALE: "after_sale_process",
    ROUTE_ESCALATE: "escalation_process",
    ROUTE_QUALITY: "quality_check",
    ROUTE_HUMAN: "human_approval_node",
}

# 质检路由映射
QUALITY_ROUTE_MAP = {
    ROUTE_END: "compile_result",
    ROUTE_ESCALATE: "escalation_process",
}

# 审批路由映射
APPROVAL_ROUTE_MAP = {
    ROUTE_FINANCE: "finance_process",
    ROUTE_END: "__end__",
    ROUTE_QUALITY: "quality_check",
    ROUTE_ESCALATE: "escalation_process",
}

# 升级路由映射
ESCALATION_ROUTE_MAP = {
    ROUTE_END: "compile_result",
    ROUTE_TECH: "tech_support_process",
    ROUTE_FINANCE: "finance_process",
    ROUTE_AFTER_SALE: "after_sale_process",
    ROUTE_HUMAN: "human_approval_node",  # 升级转人工 → HITL 暂停
}
