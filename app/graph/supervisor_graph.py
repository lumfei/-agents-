"""
Supervisor 工作流图 — LangGraph StateGraph 完整编排

这个文件把之前定义的所有"零件"（状态定义、路由逻辑、节点函数）
组装成一个可执行的 LangGraph StateGraph。

图结构：

  START → classify_intent → extract_context
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              tech_support   finance  after_sale
                    │           │           │
                    └───────────┼───────────┘
                                │
                          quality_check
                           │        │
                      score≥0.8   score<0.8
                           │        │
                       compile ← escalation
                           │        │
                          END    human_approval
                                      │
                                 approved/rejected
                                      │
                                   compile → END

支持的特性：
  - Checkpointing（PostgresSaver，每步自动保存到 PostgreSQL）
  - HITL 审批暂停点（使用 LangGraph interrupt_before）
  - 条件路由（意图分类、质检评分、迭代次数）
  - 重试机制
  - 可序列化/恢复（服务重启不丢状态）

interrupt_before 模式说明：
  当图执行到 human_approval_node 时，LangGraph 在节点执行前自动暂停。
  外部审批完成后，通过 resume_from_checkpoint() 调用
  graph.invoke(Command(resume=...), config) 恢复执行。
  与旧版"条件路由"模式的区别：
    旧版：Worker → 条件边 → human_approval_node 执行后自然停止
    新版：Worker → 条件边 → interrupt_before 拦截，节点不执行就暂停
    新版优势：暂停点由框架管理，状态更干净，支持 Command(resume) 传值

使用方式：
  from app.graph.supervisor_graph import build_supervisor_graph

  graph = build_supervisor_graph()
  # 同步执行
  result = graph.invoke(initial_state)
  # 流式执行（逐节点查看中间状态）
  for event in graph.stream(initial_state):
      print(event)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.graph.checkpoint import get_checkpointer

from app.graph.state_definition import AgentState, create_initial_state
from app.graph.routing_logic import (
    supervisor_routing,
    worker_completion_routing,
    quality_routing,
    escalation_routing,
    approval_routing,
    SUPERVISOR_ROUTE_MAP,
    WORKER_ROUTE_MAP,
    QUALITY_ROUTE_MAP,
    ESCALATION_ROUTE_MAP,
    APPROVAL_ROUTE_MAP,
)
from app.graph.worker_graphs import NODE_FUNCTIONS
from app.prompts.registry import get_prompt_registry
from app.observability import (
    TracingContext,
    get_tracing_handler,
    flush_traces,
    get_cost_tracker,
    get_alert_manager,
    WorkflowEvent,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  图构建
# ═══════════════════════════════════════════════════════════════

def build_supervisor_graph() -> StateGraph:
    """
    构建完整的 Supervisor 工作流图。

    这张图包含：
      - 9 个节点（每个节点对应一个处理步骤）
      - 条件边（根据意图、评分等动态路由）
      - 普通边（固定路径）

    步骤：
      1. 创建 StateGraph（指定状态类型为 AgentState）
      2. 注册所有节点
      3. 连接边（普通边 + 条件边）
      4. 编译图（返回可执行实例）

    返回：
      编译后的 StateGraph 实例
    """

    # ── 1. 创建图 ───────────────────────────────────────────
    # StateGraph(AgentState) 表示这个图操作的是 AgentState 类型的状态
    graph = StateGraph(AgentState)

    # ── 2. 注册节点 ─────────────────────────────────────────
    # add_node(name, function) 把处理函数注册为图的节点
    # name: 节点名称（路由时用字符串引用）
    # function: 接收 state 返回 dict 的函数

    graph.add_node("classify_intent", NODE_FUNCTIONS["classify_intent"])
    graph.add_node("extract_context", NODE_FUNCTIONS["extract_context"])
    graph.add_node("tech_support_process", NODE_FUNCTIONS["tech_support_process"])
    graph.add_node("finance_process", NODE_FUNCTIONS["finance_process"])
    graph.add_node("after_sale_process", NODE_FUNCTIONS["after_sale_process"])
    graph.add_node("quality_check", NODE_FUNCTIONS["quality_check"])
    graph.add_node("escalation_process", NODE_FUNCTIONS["escalation_process"])
    graph.add_node("human_approval_node", NODE_FUNCTIONS["human_approval_node"])
    graph.add_node("compile_result", NODE_FUNCTIONS["compile_result"])

    # ── 3. 连接边 ───────────────────────────────────────────

    # START → classify_intent
    # 图执行的起点，START 是 LangGraph 内置的起始节点
    graph.add_edge(START, "classify_intent")

    # classify_intent → extract_context
    # 固定边：意图分类后必然进入上下文提取
    graph.add_edge("classify_intent", "extract_context")

    # extract_context → 条件路由（根据意图分流到不同 Worker）
    # 条件边：根据 supervisor_routing 函数的返回值决定下一步
    graph.add_conditional_edges(
        "extract_context",
        supervisor_routing,
        SUPERVISOR_ROUTE_MAP,
    )

    # Worker → 条件路由（判断是否完成 / 需要审批 / 需要升级）
    for worker in ["tech_support_process", "finance_process", "after_sale_process"]:
        graph.add_conditional_edges(
            worker,
            worker_completion_routing,
            WORKER_ROUTE_MAP,
        )

    # quality_check → 条件路由（根据评分判断结束或升级）
    graph.add_conditional_edges(
        "quality_check",
        quality_routing,
        QUALITY_ROUTE_MAP,
    )

    # escalation_process → 条件路由
    graph.add_conditional_edges(
        "escalation_process",
        escalation_routing,
        ESCALATION_ROUTE_MAP,
    )

    # human_approval_node → 条件路由（根据审批结果判断后续）
    graph.add_conditional_edges(
        "human_approval_node",
        approval_routing,
        APPROVAL_ROUTE_MAP,
    )

    # compile_result → END
    # 结果汇总后结束
    graph.add_edge("compile_result", END)

    # ── 4. 编译图 ───────────────────────────────────────────
    # compile() 把图的定义转换为可执行对象
    # checkpointer: PostgresSaver（生产）或 MemorySaver（开发/测试回退）
    # 这带来：
    #   - 每步执行后自动保存状态快照到 PostgreSQL
    #   - 服务重启后对话状态不丢失
    #   - 支持暂停/恢复（HITL 审批）
    #   - 支持 time-travel 调试
    #   - 多实例共享同一 PG，任意实例可接管对话

    checkpointer = get_checkpointer()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval_node"],
    )

    return app


# ═══════════════════════════════════════════════════════════════
#  便捷 API
# ═══════════════════════════════════════════════════════════════

# 编译好的图实例（全局唯一）
_app: StateGraph | None = None


def get_graph() -> StateGraph:
    """
    获取编译后的图实例（单例模式）。

    第一次调用时构建，之后返回缓存的实例。
    """
    global _app
    if _app is None:
        _app = build_supervisor_graph()
    return _app


def run_workflow(
    user_message: str,
    user_id: str = "",
    session_id: str = "",
    config: dict | None = None,
) -> dict:
    """
    运行完整的工作流。

    这是最上层 API——传入用户消息，返回处理结果。

    参数：
      user_message: 用户的输入消息
      user_id:      用户 ID（可选）
      session_id:   会话 ID（可选）
      config:       LangGraph 运行时配置（可选）
                    - recursion_limit: 最大递归深度（防死循环）
                    - thread_id: 线程 ID（用于 checkpoint 恢复）

    返回：
      最终状态字典（包含 final_response、resolved 等字段）
    """
    graph = get_graph()

    # 创建初始状态
    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )

    # 运行时配置
    # LangGraph 格式：
    #   recursion_limit → 顶层（控制最大递归深度，防死循环）
    #   thread_id      → configurable 内（Checkpointing 用）
    run_config = {
        "recursion_limit": 25,
        "configurable": {
            "thread_id": session_id or f"thread_{hash(user_message) % 100000}",
        },
    }
    if config:
        if "recursion_limit" in config:
            run_config["recursion_limit"] = config["recursion_limit"]
        if "thread_id" in config or "configurable" in config:
            run_config.setdefault("configurable", {}).update(
                config.get("configurable", {})
            )
            if "thread_id" in config:
                run_config["configurable"]["thread_id"] = config["thread_id"]

    # ── 执行图（带可观测性追踪） ──────────────────────────
    start_time = time.time()

    with TracingContext(
        user_id=user_id,
        session_id=session_id,
        input_data={"message": user_message},
    ) as trace_ctx:
        handler = trace_ctx["handler"]
        if handler:
            run_config["callbacks"] = [handler]

        try:
            result = graph.invoke(initial_state, run_config)
        except Exception:
            # 即使 graph 执行失败，也要记录 observability
            duration_ms = (time.time() - start_time) * 1000
            _record_observability(
                result={}, user_id=user_id, session_id=session_id,
                duration_ms=duration_ms,
            )
            raise

        duration_ms = (time.time() - start_time) * 1000

        # 更新 Tracing 元数据
        trace_ctx["output"] = {
            "reply": result.get("final_response", ""),
            "intent": result.get("intent", "unknown"),
        }
        trace_ctx["metadata"] = {
            "intent": result.get("intent", "unknown"),
            "quality_score": result.get("quality_score", 1.0),
            "resolved": result.get("resolved", False),
            "iteration_count": result.get("iteration_count", 0),
            "agent_path": result.get("agents_sequence", []),
            "prompt_versions": get_prompt_registry().get_active_versions(),
        }

    # ── 记录成本 & 告警 ──────────────────────────────────
    observability_data = _record_observability(
        result, user_id=user_id, session_id=session_id,
        duration_ms=duration_ms,
    )
    result.setdefault("metadata", {})["observability"] = observability_data

    return result


def stream_workflow(
    user_message: str,
    user_id: str = "",
    session_id: str = "",
) -> list[dict]:
    """
    流式执行工作流（逐节点查看中间状态）。

    用于调试和前端展示——可以看到每个节点的处理结果。
    """
    graph = get_graph()

    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )

    run_config = {
        "recursion_limit": 25,
        "configurable": {
            "thread_id": session_id or f"thread_{hash(user_message) % 100000}",
        },
    }

    start_time = time.time()
    events = []

    with TracingContext(
        user_id=user_id,
        session_id=session_id,
        input_data={"message": user_message},
    ) as trace_ctx:
        handler = trace_ctx["handler"]
        if handler:
            run_config["callbacks"] = [handler]

        for event in graph.stream(initial_state, run_config):
            events.append(event)

        # 从事件中提取最终状态
        final_state: dict = {}
        for event in events:
            for node_name, state_update in event.items():
                final_state.update(state_update)

        duration_ms = (time.time() - start_time) * 1000
        trace_ctx["output"] = {
            "reply": final_state.get("final_response", ""),
            "intent": final_state.get("intent", "unknown"),
        }
        trace_ctx["metadata"] = {
            "intent": final_state.get("intent", "unknown"),
            "quality_score": final_state.get("quality_score", 1.0),
            "resolved": final_state.get("resolved", False),
            "iteration_count": final_state.get("iteration_count", 0),
        }

    # ── 记录成本 & 告警 ──────────────────────────────────
    _record_observability(
        final_state, user_id=user_id, session_id=session_id,
        duration_ms=duration_ms,
    )

    return events


# ═══════════════════════════════════════════════════════════════
#  异步流式执行
# ═══════════════════════════════════════════════════════════════

async def astream_workflow(
    user_message: str,
    user_id: str = "",
    session_id: str = "",
):
    """
    异步流式执行工作流（逐节点推送事件）。

    使用 LangGraph 的 .astream() 方法，每经过一个节点就 yield 一次，
    供 SSE 端点逐事件推给前端。

    用法：
      async for event in astream_workflow("我的电脑蓝屏了"):
          # event 格式: {node_name: state_update_dict}
          yield f"data: {json.dumps(event)}\\n\\n"
    """
    graph = get_graph()
    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )
    run_config = {
        "recursion_limit": 25,
        "configurable": {
            "thread_id": session_id or f"thread_{hash(user_message) % 100000}",
        },
    }

    start_time = time.time()
    final_state: dict = {}

    with TracingContext(
        user_id=user_id,
        session_id=session_id,
        input_data={"message": user_message},
    ) as trace_ctx:
        handler = trace_ctx["handler"]
        if handler:
            run_config["callbacks"] = [handler]

        async for event in graph.astream(initial_state, run_config):
            for _node_name, state_update in event.items():
                final_state.update(state_update)
            yield event

        duration_ms = (time.time() - start_time) * 1000
        trace_ctx["output"] = {
            "reply": final_state.get("final_response", ""),
            "intent": final_state.get("intent", "unknown"),
        }
        trace_ctx["metadata"] = {
            "intent": final_state.get("intent", "unknown"),
            "quality_score": final_state.get("quality_score", 1.0),
            "resolved": final_state.get("resolved", False),
            "iteration_count": final_state.get("iteration_count", 0),
        }

    # ── 记录成本 & 告警 ──────────────────────────────────
    _record_observability(
        final_state, user_id=user_id, session_id=session_id,
        duration_ms=duration_ms,
    )


# ═══════════════════════════════════════════════════════════════
#  检查点管理
# ═══════════════════════════════════════════════════════════════

def get_state(thread_id: str) -> dict | None:
    """
    获取指定线程的最新状态（从 PostgreSQL checkpoint 恢复）。

    用于：
      - 查看某个会话当前的执行状态
      - HITL 审批后恢复执行
      - 服务重启后继续未完成的对话
    """
    graph = get_graph()
    try:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        return state
    except Exception:
        return None


def resume_from_checkpoint(
    thread_id: str,
    updates: dict | None = None,
) -> dict:
    """
    从检查点恢复执行（使用 LangGraph interrupt_before 模式）。

    用于 HITL 审批场景：
      1. 图在 human_approval_node 前被 interrupt_before 拦截暂停
      2. 人工审批后，通过此函数注入审批结果
      3. 图从暂停处继续执行 human_approval_node

    参数：
      thread_id: 线程 ID（对应的会话）
      updates:   要注入的状态更新（如审批结果）

    返回：
      最终状态
    """
    graph = get_graph()
    run_config = {
        "recursion_limit": 25,
        "configurable": {
            "thread_id": thread_id,
        },
    }

    if updates:
        # 先更新状态（注入审批结果），再通过 Command(resume=...) 恢复
        graph.update_state(run_config, updates)
    # Command(resume=...) 告诉 LangGraph 从 interrupt 点继续
    result = graph.invoke(Command(resume="approved"), run_config)

    return result


# ═══════════════════════════════════════════════════════════════
#  可观测性辅助函数
# ═══════════════════════════════════════════════════════════════

def _extract_token_usage(result: dict) -> tuple[int, int, str]:
    """
    从 LangGraph 执行结果中提取 Token 使用量和 Agent 名称。

    数据来源有两个（按优先级）：
      1. worker_token_usage — Worker Agent 从 ReAct 内部消息汇总的精确数据
      2. result["messages"] — 遍历 AIMessage 的 usage_metadata（兜底）

    返回：
      (total_input_tokens, total_output_tokens, primary_agent)
    """
    total_input = 0
    total_output = 0
    primary_agent = "unknown"

    # 提取 Agent 名称
    agents = result.get("agents_sequence", [])
    if agents:
        # 取最后一个 Worker Agent（排除 supervisor）
        for a in reversed(agents):
            if a != "supervisor":
                primary_agent = a
                break

    # ── 汇总所有来源的 Token 数据 ─────────────────────────
    worker_usage = result.get("worker_token_usage", {})
    classify_usage = result.get("classify_token_usage", {})

    total_input = worker_usage.get("input_tokens", 0) + classify_usage.get("input_tokens", 0)
    total_output = worker_usage.get("output_tokens", 0) + classify_usage.get("output_tokens", 0)

    if total_input > 0 or total_output > 0:
        return total_input, total_output, primary_agent

    # ── 兜底：遍历 messages 中的 usage_metadata ─────────────
    for msg in result.get("messages", []):
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            meta = msg.usage_metadata
            total_input += meta.get("input_tokens", 0) or 0
            total_output += meta.get("output_tokens", 0) or 0

    return total_input, total_output, primary_agent


def _record_observability(
    result: dict,
    user_id: str,
    session_id: str,
    duration_ms: float,
) -> dict:
    """
    记录可观测性数据：Token 成本 + 告警检查。

    返回：
      可观测性元数据字典，会被合并到 result["metadata"] 中。
    """
    observability: dict = {"duration_ms": duration_ms}

    try:
        # ── Token 成本追踪 ──────────────────────────────────
        input_tokens, output_tokens, agent = _extract_token_usage(result)
        total_tokens = input_tokens + output_tokens

        cost_tracker = get_cost_tracker()
        cost_usd, cost_cny = cost_tracker.record_usage(
            session_id=session_id,
            user_id=user_id,
            agent=agent,
            model=settings.LLM_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        observability["token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        observability["cost_usd"] = cost_usd
        observability["cost_cny"] = cost_cny

        # ── 告警检查 ───────────────────────────────────────
        alert_mgr = get_alert_manager()
        # success = 请求被正常处理（包括升级转人工，那是正常流程）
        # 只有真正的异常才算失败：Worker 报异常、安全拦截、质检不过等
        escalation_reason = result.get("escalation_reason", "")
        is_real_error = any(kw in escalation_reason for kw in [
            "异常", "error", "Error", "校验未通过", "安全校验",
        ])
        success = not is_real_error

        event = WorkflowEvent(
            session_id=session_id,
            user_id=user_id,
            intent=result.get("intent", "unknown"),
            success=success,
            duration_ms=duration_ms,
            token_count=total_tokens,
            quality_score=result.get("quality_score", 1.0),
            escalation=result.get("escalation_flag", False),
            error="" if success else escalation_reason,
        )
        triggered = alert_mgr.check_all(event)
        if triggered:
            observability["alerts"] = [a.to_dict() for a in triggered]

    except Exception:
        # 可观测性记录失败不应影响主流程
        pass

    return observability
