"""
Agent 模块

本项目的 Agent 架构基于 LangGraph StateGraph 实现（非传统 class 继承模式）。

架构说明：
  - BaseAgent (base_agent.py)        — Agent 基类 + 系统提示词构建器 + 结构化输出
  - worker_graphs.py (graph/)        — 所有 Worker Agent 的节点函数（真实逻辑在这里）
  - supervisor_graph.py (graph/)     — Supervisor 编排图（组装所有节点）

Worker Agent 的实际处理逻辑在 app/graph/worker_graphs.py 中：
  classify_intent()     → Supervisor 意图分类
  tech_support_process() → 技术支持 Worker
  finance_process()      → 财务 Worker
  after_sale_process()   → 售后 Worker
  quality_check()        → 质检
  escalation_process()   → 升级转人工
  human_approval_node()  → HITL 审批
  compile_result()       → 结果汇总

这种设计选择 LangGraph 的函数式节点模式而非类继承模式，
因为 LangGraph 的 StateGraph + create_react_agent 已内置 ReAct 循环，
不需要在每个 Agent 类中重复实现。
"""

from app.agents.base_agent import (
    BaseAgent,
    SystemPromptBuilder,
    IntentClassification,
    AgentResponse,
    ToolChoice,
)

__all__ = [
    "BaseAgent",
    "SystemPromptBuilder",
    "IntentClassification",
    "AgentResponse",
    "ToolChoice",
]
