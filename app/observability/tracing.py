"""
LangFuse 全链路追踪模块

职责：
  - 创建和管理 LangFuse Trace / Span
  - 集成 LangChain CallbackHandler，自动捕获所有 LLM 调用和工具调用
  - 在 LangFuse 密钥未配置时自动降级为 no-op（不影响主流程）

Span 层次结构（LangFuse UI 中可见）：
  agent-workflow（根 Span / Trace）
    ├── classify_intent（LLM Generation，自动捕获）
    ├── extract_context（无 LLM，不产生 Span）
    ├── finance_process（React Agent）
    │   ├── LLM 调用（Generation，自动捕获）
    │   ├── query_order（Tool Span，自动捕获）
    │   └── LLM 调用（Generation，自动捕获）
    ├── quality_check（无 LLM，不产生 Span）
    └── compile_result（无 LLM，不产生 Span）

使用方式：
  from app.observability.tracing import TracingContext, get_tracing_handler

  # 方式 1：上下文管理器（推荐）
  with TracingContext(user_id="USR-001", session_id="sess-123") as ctx:
      result = graph.invoke(state, {"callbacks": [ctx.handler]})

  # 方式 2：手动获取 handler
  handler = get_tracing_handler("USR-001", "sess-123")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── 模块级缓存 ────────────────────────────────────────────────
_langfuse_client: Any = None  # Langfuse | None
_client_initialized: bool = False


def get_langfuse_client():
    """
    获取 LangFuse 客户端实例（全局单例，懒初始化）。

    密钥为空时返回 None，上层代码据此判断是否启用追踪。
    """
    global _langfuse_client, _client_initialized

    if _client_initialized:
        return _langfuse_client

    _client_initialized = True

    # ── 检查密钥是否配置 ──────────────────────────────────────
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.debug("LangFuse 密钥未配置，追踪已禁用（no-op 模式）")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info(
            "LangFuse 客户端已初始化: host=%s",
            settings.LANGFUSE_HOST,
        )
        return _langfuse_client
    except Exception as e:
        logger.warning("LangFuse 客户端初始化失败（追踪将不可用）: %s", e)
        _langfuse_client = None
        return None


def _create_callback_handler(trace_id: str | None = None):
    """
    创建 LangChain CallbackHandler。

    CallbackHandler 自动捕获所有 LangChain/LangGraph 的：
      - on_llm_start / on_llm_end → 创建 Generation（含 Token 用量）
      - on_tool_start / on_tool_end  → 创建 Tool Span
      - on_chain_start / on_chain_end → 创建 Chain Span

    参数：
      trace_id: 已有的 Trace ID（用于把 CallbackHandler 产生的 Span
                挂到我们手动创建的根 Span 下面）。None 则自动创建新 Trace。
    """
    try:
        from langfuse.langchain import CallbackHandler as LangchainCallbackHandler

        kwargs = {}
        if trace_id:
            kwargs["trace_context"] = {"trace_id": trace_id}

        # 只在密钥已配置时显式传入 public_key，否则让 CallbackHandler
        # 自己从环境变量读取（避免无密钥时的 stderr 认证警告）
        if settings.LANGFUSE_PUBLIC_KEY:
            kwargs["public_key"] = settings.LANGFUSE_PUBLIC_KEY

        return LangchainCallbackHandler(**kwargs)
    except ImportError:
        logger.warning(
            "langfuse.langchain.CallbackHandler 不可用，"
            "请确认 langfuse>=2.55.0 已安装"
        )
        return None
    except Exception as e:
        logger.warning("CallbackHandler 创建失败: %s", e)
        return None


def get_tracing_handler(
    user_id: str = "",
    session_id: str = "",
) -> Any | None:
    """
    便捷函数：获取 CallbackHandler。

    密钥未配置时返回 None，调用方检查 if handler 即可。

    使用示例：
      handler = get_tracing_handler("USR-001", "sess-123")
      if handler:
          run_config["callbacks"] = [handler]
      result = graph.invoke(state, run_config)
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        # ── 创建根 Span 作为 Trace ─────────────────────────────
        root_span = client.start_as_current_observation(
            name="agent-workflow",
            as_type="span",
            metadata={
                "user_id": user_id or "anonymous",
                "session_id": session_id or "unknown",
                "app_name": settings.APP_NAME,
                "environment": settings.APP_ENV,
            },
        )
        # 进入上下文，拿到根 Span 对象
        root = root_span.__enter__()
        trace_id = client.get_current_trace_id()

        # ── 创建 CallbackHandler，挂到根 Span 下 ────────────────
        handler = _create_callback_handler(trace_id=trace_id)

        # 把清理信息附加到 handler 上（供 TracingContext 用）
        if handler:
            handler._langfuse_root_span = root
            handler._langfuse_root_ctx = root_span

        return handler
    except Exception as e:
        logger.warning("创建 Tracing Handler 失败: %s", e)
        return None


def finalize_tracing(
    handler: Any,
    output: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """
    完成追踪：更新根 Span 的输出并关闭上下文。

    必须在 workflow 执行完成后调用。

    参数：
      handler:    get_tracing_handler() 返回的 handler
      output:     最终输出（如 {"reply": "...", "intent": "..."}）
      metadata:   额外元数据（如 {"quality_score": 0.9, "iteration_count": 2}）
    """
    if handler is None:
        return

    try:
        root_span = getattr(handler, "_langfuse_root_span", None)
        root_ctx = getattr(handler, "_langfuse_root_ctx", None)

        if root_span is not None:
            if output is not None or metadata is not None:
                update_kwargs = {}
                if output is not None:
                    update_kwargs["output"] = output
                if metadata is not None:
                    current_meta = getattr(root_span, "metadata", {}) or {}
                    current_meta.update(metadata)
                    update_kwargs["metadata"] = current_meta
                if update_kwargs:
                    root_span.update(**update_kwargs)

        if root_ctx is not None:
            root_ctx.__exit__(None, None, None)
    except Exception as e:
        logger.debug("finalize_tracing 出错（非关键）: %s", e)


def flush_traces() -> None:
    """
    将待发送的追踪数据刷新到 LangFuse 服务器。

    应在以下时机调用：
      - 每次 workflow 执行完成后（确保数据及时上报）
      - 应用关闭前（发送最后一批数据）
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception as e:
        logger.debug("LangFuse flush 出错（非关键）: %s", e)


# ═══════════════════════════════════════════════════════════════
#  TracingContext — 推荐的上下文管理器
# ═══════════════════════════════════════════════════════════════

@contextmanager
def TracingContext(
    user_id: str = "",
    session_id: str = "",
    input_data: dict | None = None,
):
    """
    Tracing 上下文管理器——创建 Trace + CallbackHandler + 自动清理。

    使用示例：
      with TracingContext(user_id="USR-001", session_id="sess-123",
                          input_data={"message": "查订单"}) as ctx:
          handler = ctx["handler"]
          result = graph.invoke(state, {"callbacks": [handler]})
          ctx["output"] = {"reply": result["final_response"]}
          ctx["metadata"] = {"intent": result["intent"]}

    上下文退出时自动：
      - 用 output/metadata 更新根 Span
      - 关闭 Span 上下文
      - flush 到 LangFuse
    """
    ctx: dict[str, Any] = {"handler": None, "output": None, "metadata": None}

    client = get_langfuse_client()
    if client is None:
        yield ctx
        return

    root_span = None
    root_ctx_manager = None

    try:
        # ── 创建根 Span ─────────────────────────────────────────
        root_kwargs: dict = {
            "name": "agent-workflow",
            "as_type": "span",
            "metadata": {
                "user_id": user_id or "anonymous",
                "session_id": session_id or "unknown",
                "app_name": settings.APP_NAME,
                "environment": settings.APP_ENV,
            },
        }
        if input_data:
            root_kwargs["input"] = safe_trace_input(input_data)

        root_ctx_manager = client.start_as_current_observation(**root_kwargs)
        root_span = root_ctx_manager.__enter__()
        trace_id = client.get_current_trace_id()

        # ── 创建 CallbackHandler ────────────────────────────────
        handler = _create_callback_handler(trace_id=trace_id)
        ctx["handler"] = handler

        yield ctx

        # ── 退出时更新根 Span ──────────────────────────────────
        if root_span is not None:
            update_kwargs = {}
            output = ctx.get("output")
            meta = ctx.get("metadata")
            if output is not None:
                update_kwargs["output"] = safe_trace_output(output)
            if meta is not None:
                current_meta = getattr(root_span, "metadata", {}) or {}
                current_meta.update(meta)
                update_kwargs["metadata"] = current_meta
            if update_kwargs:
                root_span.update(**update_kwargs)

    except Exception as e:
        logger.warning("Tracing 上下文出错（非关键）: %s", e)
    finally:
        # ── 关闭根 Span ────────────────────────────────────
        if root_ctx_manager is not None:
            try:
                root_ctx_manager.__exit__(None, None, None)
            except Exception as e:
                logger.debug("关闭根 Span 时出错: %s", e)
        # ── Flush ──────────────────────────────────────────
        if client is not None:
            try:
                client.flush()
            except Exception as e:
                logger.debug("flush 出错: %s", e)


# ═══════════════════════════════════════════════════════════════
#  Reasoning Trace 捕获（DeepSeek Thinking Token 追踪）
# ═══════════════════════════════════════════════════════════════

class ReasoningTraceCapture:
    """
    捕获 LLM 推理过程（Reasoning Tokens / Thinking Tokens）。

    DeepSeek V3/V4、Claude、GPT-4 等模型支持"思考模式"，
    在最终输出前产生内部推理 Token。这些 Token：
      - 不计入 output_tokens（通常单独计费或免费）
      - 可帮助理解 Agent 的决策过程
      - 是调试幻觉和错误路由的关键数据

    使用方式：
      capture = ReasoningTraceCapture()
      capture.record_reasoning(
          trace_id="trace_123",
          agent="finance_agent",
          reasoning_content="用户要求退款...检查订单状态...",
          reasoning_tokens=150,
      )
    """

    def __init__(self, max_entries: int = 1000):
        self._entries: list[dict] = []
        self._max_entries = max_entries
        self._by_trace: dict[str, list[dict]] = {}

    def record_reasoning(
        self,
        trace_id: str,
        agent: str,
        reasoning_content: str,
        reasoning_tokens: int = 0,
        step: str = "",
    ) -> None:
        """记录一条推理过程"""
        entry = {
            "trace_id": trace_id,
            "agent": agent,
            "step": step,
            "reasoning_content": reasoning_content[:2000],  # 截断过长的推理
            "reasoning_tokens": reasoning_tokens,
            "timestamp": __import__("time").time(),
        }

        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries // 2:]

        if trace_id not in self._by_trace:
            self._by_trace[trace_id] = []
        self._by_trace[trace_id].append(entry)

    def get_reasoning_chain(self, trace_id: str) -> list[dict]:
        """获取某个 trace 的完整推理链"""
        return self._by_trace.get(trace_id, [])

    def get_stats(self) -> dict:
        """推理 Token 统计"""
        total_tokens = sum(e["reasoning_tokens"] for e in self._entries)
        by_agent: dict[str, int] = {}
        for e in self._entries:
            agent = e["agent"]
            by_agent[agent] = by_agent.get(agent, 0) + e["reasoning_tokens"]
        return {
            "total_reasoning_entries": len(self._entries),
            "total_reasoning_tokens": total_tokens,
            "by_agent": by_agent,
        }


_reasoning_capture: ReasoningTraceCapture | None = None


def get_reasoning_capture() -> ReasoningTraceCapture:
    global _reasoning_capture
    if _reasoning_capture is None:
        _reasoning_capture = ReasoningTraceCapture()
    return _reasoning_capture


# ═══════════════════════════════════════════════════════════════
#  Tracing 隐私脱敏（PII Scrubbing before LangFuse）
# ═══════════════════════════════════════════════════════════════

import re as _re

# 需要在发送到 LangFuse 前脱敏的字段
TRACE_SENSITIVE_FIELDS = ["input", "output", "metadata.user_message"]


def scrub_trace_data(data: dict | str | list, max_value_length: int = 500) -> Any:
    """
    递归脱敏追踪数据中的 PII。

    脱敏规则：
      - 手机号 → 138****5678
      - 身份证 → 110***********1234
      - 邮箱 → us***@domain.com
      - API Key → sk-xxx***
      - IP 地址 → 192.***.***.1
      - 银行卡 → 6222 **** **** 1234

    同时限制字符串值最大长度（防止超长内容撑爆 LangFuse）。
    """
    if isinstance(data, str):
        # PII 脱敏
        data = _re.sub(r"1[3-9]\d{9}", lambda m: m.group()[:3] + "****" + m.group()[-4:], data)
        data = _re.sub(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
                       lambda m: m.group()[:3] + "***********" + m.group()[-4:], data)
        data = _re.sub(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                       lambda m: (m.group(1)[:2] if len(m.group(1)) > 2 else m.group(1)[0]) + "***@" + m.group(2), data)
        data = _re.sub(r"(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9]+", r"\1***", data)
        data = _re.sub(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b",
                       r"\1.***.***.\4", data)
        data = _re.sub(r"\b(\d{4})[\s-]?\d{4}[\s-]?\d{4}[\s-]?(\d{4,7})\b",
                       r"\1 **** **** \2", data)

        # 长度限制
        if len(data) > max_value_length:
            data = data[:max_value_length] + "...[truncated]"
        return data

    elif isinstance(data, dict):
        return {k: scrub_trace_data(v, max_value_length) for k, v in data.items()}

    elif isinstance(data, list):
        return [scrub_trace_data(item, max_value_length) for item in data]

    return data


def safe_trace_input(data: dict) -> dict:
    """对传入 LangFuse 的 trace input 进行安全脱敏"""
    return scrub_trace_data(data, max_value_length=1000)


def safe_trace_output(data: dict) -> dict:
    """对传入 LangFuse 的 trace output 进行安全脱敏"""
    return scrub_trace_data(data, max_value_length=2000)


# ═══════════════════════════════════════════════════════════════
#  成本优化建议引擎
# ═══════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dc_dataclass, field as _dc_field


@_dc_dataclass
class CostOptimization:
    """一条成本优化建议"""
    category: str           # token / cache / model / routing
    severity: str           # high / medium / low
    title: str
    description: str
    estimated_savings_pct: float  # 预估节省百分比
    action: str             # 具体行动建议


class CostOptimizer:
    """
    成本优化分析器。

    基于 CostTracker 的数据分析，生成优化建议：
      - Token 浪费检测（过长 system prompt、重复上下文）
      - 缓存建议（重复查询可缓存）
      - 模型选择建议（简单任务用小模型）
      - 路由优化（减少不必要的 Agent 调用）
    """

    def analyze(self, cost_tracker) -> list[CostOptimization]:
        """分析成本数据并生成优化建议"""
        suggestions: list[CostOptimization] = []
        summary = cost_tracker.get_summary()

        total_requests = summary.get("total_requests", 0)
        if total_requests < 10:
            return suggestions  # 数据不足

        # ── 1. 缓存建议 ──────────────────────────────────────
        # 如果输入/输出 Token 比例极高，可能存在重复上下文
        total_input = summary.get("total_input_tokens", 0)
        total_output = summary.get("total_output_tokens", 0)
        if total_input > 0 and total_output > 0:
            ratio = total_input / total_output
            if ratio > 20:
                suggestions.append(CostOptimization(
                    category="token",
                    severity="high",
                    title="输入输出 Token 比过高",
                    description=f"输入/输出比 = {ratio:.1f}:1，可能存在重复的 System Prompt 或过长上下文。每次请求平均消耗 {total_input // max(total_requests, 1):,} 输入 Token。",
                    estimated_savings_pct=0.30,
                    action="缩短 System Prompt、使用 Conversation Summary 压缩历史消息、启用 Prompt Caching（DeepSeek 支持）",
                ))

        # ── 2. 模型选择建议 ──────────────────────────────────
        avg_tokens_per_request = (total_input + total_output) / max(total_requests, 1)
        if avg_tokens_per_request < 500:
            suggestions.append(CostOptimization(
                category="model",
                severity="medium",
                title="简单任务可降级模型",
                description=f"平均每次请求仅 {avg_tokens_per_request:.0f} Token，可能不需要使用大模型。",
                estimated_savings_pct=0.40,
                action="对简单分类/路由任务使用 deepseek/deepseek-v4-flash 替代 v4-pro",
            ))

        # ── 3. 路由优化 ──────────────────────────────────────
        by_agent = summary.get("by_agent", {})
        if by_agent:
            # 某个 Agent 消耗特别高
            total_cost = sum(a.get("cost", 0) for a in by_agent.values())
            for agent_name, agent_data in by_agent.items():
                agent_cost = agent_data.get("cost", 0)
                if total_cost > 0 and agent_cost / total_cost > 0.5:
                    suggestions.append(CostOptimization(
                        category="routing",
                        severity="medium",
                        title=f"Agent '{agent_name}' 成本占比过高",
                        description=f"该 Agent 消耗了 {agent_cost/total_cost:.0%} 的总成本。",
                        estimated_savings_pct=0.15,
                        action=f"检查 '{agent_name}' 的 System Prompt 长度和工具调用次数，考虑增加前置过滤/缓存。",
                    ))

        # ── 4. Token 浪费检测 ─────────────────────────────────
        recent = cost_tracker.get_recent_usages(50)
        if recent:
            zero_output = sum(1 for u in recent if u.output_tokens == 0)
            if zero_output > len(recent) * 0.1:
                suggestions.append(CostOptimization(
                    category="token",
                    severity="low",
                    title="存在无效调用",
                    description=f"最近 {len(recent)} 次请求中有 {zero_output} 次输出为 0 Token（可能是工具调用失败或空回复）。",
                    estimated_savings_pct=0.05,
                    action="检查工具调用失败原因，添加前置参数校验减少无效 LLM 调用。",
                ))

        return suggestions


_cost_optimizer: CostOptimizer | None = None


def get_cost_optimizer() -> CostOptimizer:
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
    return _cost_optimizer
