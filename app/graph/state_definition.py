"""
LangGraph 状态定义

这个文件定义了 Agent 工作流的"共享状态"——所有节点（处理步骤）
都能读写的数据结构。状态是 Agent 间传递信息的核心。

状态的更新规则：
  LangGraph 的 StateGraph 用"归约器"（Reducer）来控制状态的更新。
  默认行为是"覆盖"——后写入的覆盖先写入的。
  列表字段用"追加"归约器——后写入的追加到列表末尾。

Checkpointing（检查点持久化）：
  LangGraph 的 MemorySaver 会在每步执行后自动保存完整状态快照。
  这支持：
    - 暂停/恢复：应用崩溃后从最近 checkpoint 恢复
    - Time-travel 调试：回退到任意历史状态重新执行
    - 人工审批：HITL 节点暂停，等待人类输入后继续

AgentState 字段说明：
  messages:         对话消息历史（来自 MessagesState，用 AddMessages reducer）
  intent:           意图分类结果（tech_support / finance / after_sale / unknown）
  confidence:       意图分类置信度
  current_agent:    当前正在处理的 Agent 名称
  slot_state:       对话状态管理（Slot-Filling 的当前进度）
  tool_results:     工具调用记录和返回结果
  iteration_count:  当前处理轮次（超过上限触发升级）
  task_id:          任务唯一标识
  session_id:       会话标识
  user_id:          用户标识
  approval_pending: 是否有待审批的请求
  quality_score:    质检评分（0-1，<0.8 触发升级）
  escalation_flag:  是否需要升级到人工
  metadata:         元数据（时间戳、Token消耗等）
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


# ── List Reducer（追加模式，用于 agents_sequence 等列表字段） ─
# LangGraph 默认不合并列表，后写入的覆盖先写入的。
# 用 Annotated[list[str], add_string_list] 告诉 LangGraph：
# "这个列表要追加，不要覆盖"

def add_string_list(left: list[str], right: list[str]) -> list[str]:
    """追加模式 Reducer"""
    return left + right


# ═══════════════════════════════════════════════════════════════
#  工具调用记录
# ═══════════════════════════════════════════════════════════════

class ToolCallRecord(TypedDict, total=False):
    """一次工具调用的完整记录"""
    tool_name: str
    arguments: dict
    result: Any
    success: bool
    duration_ms: float


# ═══════════════════════════════════════════════════════════════
#  主状态定义
# ═══════════════════════════════════════════════════════════════

class AgentState(MessagesState):
    """
    多 Agent 客服分流系统的共享状态。

    继承自 MessagesState（LangGraph 内置），自带 messages 字段，
    默认使用 AddMessages reducer（追加模式）。

    所有 Agent 字段使用"覆盖"模式（后写入的覆盖先写入的）。
    列表字段需要在节点函数中手动合并（不是自动追加）。
    """

    # ── 意图分类 ──────────────────────────────────────────────
    intent: str = ""
    """意图分类结果：tech_support / finance / after_sale / unknown"""

    confidence: float = 0.0
    """意图分类的置信度（0.0-1.0）"""

    sentiment: str = "neutral"
    """用户情绪：positive / neutral / anxious / angry / frustrated"""

    extracted_entities: dict[str, Any] = {}
    """从用户输入中提取的关键信息（订单号、用户ID等）"""

    # ── Agent 追踪 ────────────────────────────────────────────
    current_agent: str = ""
    """当前正在处理的 Agent 名称"""

    agents_sequence: Annotated[list[str], add_string_list] = []
    """已经过哪些 Agent 的处理记录"""

    # ── 对话状态管理 ─────────────────────────────────────────
    slot_state: dict[str, Any] = {}
    """Slot-Filling 状态（DialogueStateManager.to_dict()）"""

    # ── 工具调用 ──────────────────────────────────────────────
    tool_results: list[dict] = []
    """本轮所有工具调用的结果列表"""

    next_tool_call_id: int = 0
    """工具调用计数器，用于生成唯一 ID"""

    # ── 任务追踪 ──────────────────────────────────────────────
    task_id: str = ""
    """任务唯一标识"""

    session_id: str = ""
    """会话标识"""

    user_id: str = ""
    """用户标识"""

    # ── 迭代控制 ──────────────────────────────────────────────
    iteration_count: int = 0
    """当前 Agent 的迭代轮次（用于判断是否升级）"""

    max_iterations: int = 10
    """最大允许的迭代轮次"""

    # ── HITL 审批 ─────────────────────────────────────────────
    approval_pending: dict[str, Any] = {}
    """待审批的请求信息（如果有的话）"""

    approval_history: list[dict] = []
    """审批记录列表"""

    # ── 质检评分 ──────────────────────────────────────────────
    quality_score: float = 1.0
    """质检评分（0-1，<0.8 触发升级）"""

    quality_report: str = ""
    """质检报告"""

    # ── 升级 ─────────────────────────────────────────────────
    escalation_flag: bool = False
    """是否需要升级到人工"""

    escalation_reason: str = ""
    """升级原因"""

    escalation_summary: str = ""
    """升级上下文摘要（转人工时附带的）"""

    # ── 元数据 ────────────────────────────────────────────────
    metadata: dict[str, Any] = {}
    """元数据：时间戳、Token消耗等"""

    # ── Token 追踪 ──────────────────────────────────────────────
    worker_token_usage: dict[str, int] = {}
    """Worker Agent 汇总的 Token 用量（input_tokens / output_tokens）"""

    classify_token_usage: dict[str, int] = {}
    """意图分类 LLM 调用的 Token 估算"""

    # ── CSAT 满意度收集 ──────────────────────────────────────
    conversation_ending: bool = False
    """用户是否表达了对话结束意图（谢谢/再见/解决了等）"""

    csat_score: int = 0
    """用户满意度评分（1-5，0=未评分）"""

    csat_feedback: str = ""
    """用户满意度文字反馈"""

    # ── 最终输出 ──────────────────────────────────────────────
    final_response: str = ""
    """最终回复内容"""

    resolved: bool = False
    """问题是否已解决"""

    # ── Generative UI ──────────────────────────────────────────
    ui_component: dict = {}
    """动态 UI 组件（物流卡片、退款进度等），前端按 type 渲染"""


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def create_initial_state(
    user_message: str,
    user_id: str = "",
    session_id: str = "",
    task_id: str = "",
) -> dict:
    """
    创建初始状态（用用户消息初始化一个空状态）。

    这是每个新工单的起点。
    接收用户消息后，将消息放入 messages 字段，
    其他字段保持默认值。

    参数：
      user_message: 用户的原始输入
      user_id:      用户 ID（可选）
      session_id:   会话 ID（可选）
      task_id:      任务 ID（可选，自动生成）

    返回：
      符合 AgentState 字段要求的字典
    """
    import uuid
    return {
        "messages": [{"role": "user", "content": user_message}],
        "intent": "",
        "confidence": 0.0,
        "sentiment": "neutral",
        "extracted_entities": {},
        "current_agent": "",
        "agents_sequence": [],
        "slot_state": {},
        "tool_results": [],
        "next_tool_call_id": 0,
        "task_id": task_id or f"task_{uuid.uuid4().hex[:12]}",
        "session_id": session_id,
        "user_id": user_id,
        "iteration_count": 0,
        "max_iterations": 10,
        "approval_pending": {},
        "approval_history": [],
        "quality_score": 1.0,
        "quality_report": "",
        "escalation_flag": False,
        "escalation_reason": "",
        "escalation_summary": "",
        "metadata": {
            "created_at": time.time(),
            "started_at": time.time(),
        },
        "worker_token_usage": {},
        "classify_token_usage": {},
        "conversation_ending": False,
        "csat_score": 0,
        "csat_feedback": "",
        "final_response": "",
        "resolved": False,
        "ui_component": {},
    }
