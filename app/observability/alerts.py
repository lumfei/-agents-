"""
告警规则模块

职责：
  - 定义和检测系统告警条件
  - 支持多级告警（INFO / WARNING / CRITICAL）
  - 告警抑制：同类告警 5 分钟内不重复触发

告警规则：
  1. 高延迟告警：单次处理时间 > 30 秒
  2. Token 异常：单次请求 Token > 10000
  3. 质检评分低：连续 3 次评分 < 0.8
  4. 升级率过高：最近 20 次请求中升级比例 > 30%
  5. 级联失败：最近 10 次请求中 3+ 次失败
  6. 错误率过高：最近 20 次请求中错误率 > 20%

使用方式：
  from app.observability.alerts import get_alert_manager, WorkflowEvent

  mgr = get_alert_manager()
  event = WorkflowEvent(
      session_id="sess-123", duration_ms=2500, success=True,
      quality_score=0.9, token_count=3000, escalation=True,
  )
  alerts = mgr.check_all(event)
  for alert in alerts:
      print(f"[{alert.severity}] {alert.message}")
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ═══════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════

class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """一条告警"""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }


@dataclass
class WorkflowEvent:
    """
    一次 workflow 执行完成后的摘要事件。

    由 supervisor_graph 在执行完成后构造，传入 AlertManager。
    """
    session_id: str = ""
    user_id: str = ""
    intent: str = "unknown"
    success: bool = True          # workflow 是否正常完成（未抛异常）
    duration_ms: float = 0.0      # 执行耗时（毫秒）
    token_count: int = 0          # 本次消耗的总 Token（input + output）
    quality_score: float = 1.0    # 质检评分（0.0-1.0）
    escalation: bool = False      # 是否触发了升级转人工
    error: str = ""               # 错误信息（success=False 时）
    metadata: dict = field(default_factory=dict)


# ── 告警规则检查器类型 ─────────────────────────────────────

AlertChecker = Callable[[list[WorkflowEvent], WorkflowEvent], Alert | None]
"""告警检查函数签名：(recent_events, current_event) -> Alert | None"""


# ═══════════════════════════════════════════════════════════════
#  AlertManager
# ═══════════════════════════════════════════════════════════════

class AlertManager:
    """
    告警管理器（全内存，线程安全）。

    维护一个滑动窗口（最近 100 个事件），对每个事件运行所有规则。
    同类告警在 5 分钟内不重复触发（抑制期）。
    """

    # 默认抑制期（秒）
    DEFAULT_SUPPRESSION_SECONDS = 300

    def __init__(self):
        self._lock = threading.Lock()
        self._events: deque[WorkflowEvent] = deque(maxlen=100)
        self._active_alerts: list[Alert] = []
        self._alert_history: list[Alert] = []  # 所有已关闭的告警
        self._suppression: dict[str, float] = {}  # rule_name → 上次触发时间
        self._rules: list[AlertChecker] = []
        self._setup_default_rules()

    # ── 事件与检查 ──────────────────────────────────────────

    def record_event(self, event: WorkflowEvent) -> None:
        """记录一个 workflow 事件到滑动窗口"""
        with self._lock:
            self._events.append(event)

    def check_all(self, event: WorkflowEvent) -> list[Alert]:
        """
        先记录事件，然后运行所有告警规则。

        返回：
          本次触发的告警列表（含抑制期内被跳过的）
        """
        self.record_event(event)

        triggered: list[Alert] = []
        with self._lock:
            events = list(self._events)

        for rule in self._rules:
            try:
                alert = rule(events, event)
                if alert is not None:
                    with self._lock:
                        # 检查抑制期
                        rule_name = alert.rule_name
                        last_trigger = self._suppression.get(rule_name, 0)
                        now = time.time()
                        if now - last_trigger < self.DEFAULT_SUPPRESSION_SECONDS:
                            continue  # 抑制期内，跳过
                        self._suppression[rule_name] = now
                        self._active_alerts.append(alert)
                    triggered.append(alert)
            except Exception:
                # 单个规则失败不影响其他规则
                continue

        return triggered

    # ── 告警生命周期 ───────────────────────────────────────

    def resolve_alert(self, alert_id: str) -> Alert | None:
        """标记告警为已解决"""
        with self._lock:
            for alert in self._active_alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    self._active_alerts.remove(alert)
                    self._alert_history.append(alert)
                    return alert
        return None

    def get_active_alerts(self) -> list[Alert]:
        """获取当前所有未解决的告警"""
        with self._lock:
            return list(self._active_alerts)

    def get_alert_history(self, count: int = 50) -> list[Alert]:
        """获取最近的历史告警"""
        with self._lock:
            return self._alert_history[-count:]

    def reset(self) -> None:
        """清空所有状态（用于测试）"""
        with self._lock:
            self._events.clear()
            self._active_alerts.clear()
            self._alert_history.clear()
            self._suppression.clear()

    # ═══════════════════════════════════════════════════════════
    #  默认告警规则
    # ═══════════════════════════════════════════════════════════

    def _setup_default_rules(self) -> None:
        """注册 6 条默认告警规则"""
        self._rules = [
            self._check_high_latency,
            self._check_token_anomaly,
            self._check_low_quality,
            self._check_high_escalation,
            self._check_cascade_failure,
            self._check_high_error_rate,
        ]

    # ── 规则 1：高延迟 ──────────────────────────────────────

    def _check_high_latency(
        self, events: list[WorkflowEvent], current: WorkflowEvent
    ) -> Alert | None:
        """单次处理时间 > 30 秒"""
        if current.duration_ms > 30_000:
            return Alert(
                id=f"latency-{uuid.uuid4().hex[:8]}",
                rule_name="high_latency",
                severity=AlertSeverity.WARNING,
                message=f"高延迟告警：单次请求耗时 {current.duration_ms / 1000:.1f} 秒",
                details={
                    "duration_ms": current.duration_ms,
                    "session_id": current.session_id,
                    "intent": current.intent,
                },
            )
        return None

    # ── 规则 2：Token 异常 ─────────────────────────────────

    def _check_token_anomaly(
        self, events: list[WorkflowEvent], current: WorkflowEvent
    ) -> Alert | None:
        """单次请求 Token > 10000"""
        if current.token_count > 10_000:
            return Alert(
                id=f"token-{uuid.uuid4().hex[:8]}",
                rule_name="token_anomaly",
                severity=AlertSeverity.WARNING,
                message=f"Token 消耗异常：单次请求 {current.token_count} Token",
                details={
                    "token_count": current.token_count,
                    "session_id": current.session_id,
                },
            )
        return None

    # ── 规则 3：低质检分 ───────────────────────────────────

    def _check_low_quality(
        self, events: list[WorkflowEvent], current: WorkflowEvent
    ) -> Alert | None:
        """连续 3 次评分 < 0.8"""
        recent = events[-3:]
        if len(recent) >= 3 and all(
            e.quality_score < 0.8 for e in recent
        ):
            scores = [e.quality_score for e in recent]
            return Alert(
                id=f"quality-{uuid.uuid4().hex[:8]}",
                rule_name="low_quality_score",
                severity=AlertSeverity.CRITICAL,
                message=f"质检评分连续过低：最近 3 次评分 {scores}",
                details={"recent_scores": scores},
            )
        return None

    # ── 规则 4：升级率过高 ─────────────────────────────────

    def _check_high_escalation(
        self, events: list[WorkflowEvent], current: WorkflowEvent
    ) -> Alert | None:
        """最近 20 次中升级率 > 30%"""
        recent = events[-20:]
        if len(recent) >= 10:
            escalation_count = sum(1 for e in recent if e.escalation)
            rate = escalation_count / len(recent)
            if rate > 0.3:
                return Alert(
                    id=f"escalation-{uuid.uuid4().hex[:8]}",
                    rule_name="high_escalation_rate",
                    severity=AlertSeverity.WARNING,
                    message=f"升级率过高：{rate:.0%}（{escalation_count}/{len(recent)}）",
                    details={
                        "escalation_rate": rate,
                        "escalation_count": escalation_count,
                        "window_size": len(recent),
                    },
                )
        return None

    # ── 规则 5：级联失败 ───────────────────────────────────

    def _check_cascade_failure(
        self, events: list[WorkflowEvent], current: WorkflowEvent
    ) -> Alert | None:
        """最近 10 次中 3+ 次失败"""
        recent = events[-10:]
        if len(recent) >= 5:
            failure_count = sum(1 for e in recent if not e.success)
            if failure_count >= 3:
                return Alert(
                    id=f"cascade-{uuid.uuid4().hex[:8]}",
                    rule_name="cascade_failure",
                    severity=AlertSeverity.CRITICAL,
                    message=f"级联失败告警：最近 {len(recent)} 次中 {failure_count} 次失败",
                    details={
                        "failure_count": failure_count,
                        "window_size": len(recent),
                        "recent_errors": [
                            e.error for e in recent if e.error
                        ],
                    },
                )
        return None

    # ── 规则 6：错误率过高 ─────────────────────────────────

    def _check_high_error_rate(
        self, events: list[WorkflowEvent], current: WorkflowEvent
    ) -> Alert | None:
        """最近 20 次中错误率 > 20%"""
        recent = events[-20:]
        if len(recent) >= 10:
            failure_count = sum(1 for e in recent if not e.success)
            rate = failure_count / len(recent)
            if rate > 0.2:
                return Alert(
                    id=f"error-rate-{uuid.uuid4().hex[:8]}",
                    rule_name="high_error_rate",
                    severity=AlertSeverity.WARNING,
                    message=f"错误率过高：{rate:.0%}（{failure_count}/{len(recent)}）",
                    details={
                        "error_rate": rate,
                        "failure_count": failure_count,
                        "window_size": len(recent),
                    },
                )
        return None


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """获取 AlertManager 全局单例"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
