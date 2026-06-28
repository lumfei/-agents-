"""
可观测性包 — Tracing + 成本追踪 + 告警 + 推理追踪 + 隐私脱敏 + 成本优化

导出：
  TracingContext       — 推荐的上下文管理器（自动创建 Trace + CallbackHandler + flush）
  get_tracing_handler   — 便捷函数，返回 CallbackHandler（需手动管理生命周期）
  flush_traces         — 刷新待发送的追踪数据
  get_cost_tracker     — Token 成本追踪器单例
  get_alert_manager    — 告警管理器单例
  WorkflowEvent        — 告警用的事件结构
  get_reasoning_capture — 推理过程追踪
  scrub_trace_data     — 追踪数据隐私脱敏
  get_cost_optimizer   — 成本优化建议引擎
"""

from app.observability.tracing import (
    TracingContext,
    get_tracing_handler,
    flush_traces,
    get_reasoning_capture,
    ReasoningTraceCapture,
    scrub_trace_data,
    safe_trace_input,
    safe_trace_output,
    get_cost_optimizer,
    CostOptimizer,
    CostOptimization,
)
from app.observability.cost_tracker import (
    CostTracker,
    get_cost_tracker,
)
from app.observability.alerts import (
    AlertManager,
    Alert,
    AlertSeverity,
    WorkflowEvent,
    get_alert_manager,
)

__all__ = [
    # Tracing
    "TracingContext",
    "get_tracing_handler",
    "flush_traces",
    # Reasoning Trace
    "get_reasoning_capture",
    "ReasoningTraceCapture",
    # Privacy
    "scrub_trace_data",
    "safe_trace_input",
    "safe_trace_output",
    # Cost
    "CostTracker",
    "get_cost_tracker",
    # Cost Optimization
    "get_cost_optimizer",
    "CostOptimizer",
    "CostOptimization",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "WorkflowEvent",
    "get_alert_manager",
]
