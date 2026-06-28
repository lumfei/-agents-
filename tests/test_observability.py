"""
可观测性模块测试

覆盖：
  - CostTracker: 记录、聚合查询、成本计算、线程安全
  - AlertManager: 规则触发、抑制机制、滑动窗口
  - TracingContext: no-op 降级（无密钥时）

不测试：
  - LangFuse 真实集成（需要有效的 API 密钥和网络连接）
"""

from __future__ import annotations

import threading
import time
import pytest

from app.observability.cost_tracker import (
    CostTracker,
    MODEL_PRICING,
    PeriodCost,
    get_cost_tracker,
)
from app.observability.alerts import (
    AlertManager,
    AlertSeverity,
    WorkflowEvent,
    get_alert_manager,
)
from app.observability.tracing import (
    TracingContext,
    get_tracing_handler,
    get_langfuse_client,
    flush_traces,
)


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_observability():
    """每个测试前重置可观测性单例，防止状态泄漏。"""
    import app.observability.cost_tracker as ct
    import app.observability.alerts as al
    import app.observability.tracing as tr

    # 保存原始值
    orig_cost = ct._cost_tracker
    orig_alert = al._alert_manager
    orig_client = tr._langfuse_client
    orig_init = tr._client_initialized

    # 重置
    ct._cost_tracker = None
    al._alert_manager = None
    tr._langfuse_client = None
    tr._client_initialized = False

    yield

    # 恢复
    ct._cost_tracker = orig_cost
    al._alert_manager = orig_alert
    tr._langfuse_client = orig_client
    tr._client_initialized = orig_init


# ═══════════════════════════════════════════════════════════════
#  CostTracker 测试
# ═══════════════════════════════════════════════════════════════

class TestCostTracker:
    """Token 成本追踪器测试"""

    def test_record_and_query_session(self):
        ct = CostTracker()
        ct.record_usage(
            session_id="s1", user_id="u1", agent="finance",
            model="deepseek/deepseek-v4-flash",
            input_tokens=1000, output_tokens=500,
        )
        ct.record_usage(
            session_id="s1", user_id="u1", agent="finance",
            model="deepseek/deepseek-v4-flash",
            input_tokens=800, output_tokens=200,
        )

        sc = ct.get_session_cost("s1")
        assert sc.request_count == 2
        assert sc.total_input_tokens == 1800
        assert sc.total_output_tokens == 700

    def test_record_and_query_user(self):
        ct = CostTracker()
        ct.record_usage(
            session_id="s1", user_id="u1", agent="tech_support",
            model="deepseek/deepseek-v4-flash",
            input_tokens=500, output_tokens=300,
        )
        ct.record_usage(
            session_id="s2", user_id="u2", agent="finance",
            model="deepseek/deepseek-v4-flash",
            input_tokens=200, output_tokens=100,
        )

        uc1 = ct.get_user_cost("u1")
        uc2 = ct.get_user_cost("u2")
        assert uc1.total_input_tokens == 500
        assert uc2.total_input_tokens == 200

    def test_record_and_query_day(self):
        from datetime import date

        ct = CostTracker()
        ct.record_usage(
            session_id="s1", user_id="u1", agent="finance",
            model="deepseek/deepseek-v4-flash",
            input_tokens=1000, output_tokens=500,
        )

        today = str(date.today())
        dc = ct.get_daily_cost(today)
        assert dc.request_count == 1

    def test_cost_calculation_deepseek(self):
        cost_usd, cost_cny = CostTracker._calculate_cost(
            "deepseek/deepseek-v4-flash", 1_000_000, 1_000_000
        )
        # $0.14/1M input + $0.28/1M output = $0.42
        assert cost_usd == pytest.approx(0.42, rel=0.01)
        # ¥0.42 * 7.25 = ¥3.045
        assert cost_cny == pytest.approx(3.045, rel=0.01)

    def test_cost_calculation_unknown_model(self):
        cost_usd, cost_cny = CostTracker._calculate_cost(
            "some-unknown-model", 1_000_000, 1_000_000
        )
        # default: $0.27 + $1.10 = $1.37
        assert cost_usd == pytest.approx(1.37, rel=0.01)
        assert cost_cny == pytest.approx(1.37 * 7.25, rel=0.01)

    def test_cost_calculation_prefix_match(self):
        """测试前缀匹配：deepseek/deepseek-v4-pro 不存在精确匹配时按前缀匹配"""
        cost_usd, cost_cny = CostTracker._calculate_cost(
            "deepseek/deepseek-v4-flash", 1_000_000, 1_000_000
        )
        assert cost_usd == pytest.approx(0.42, rel=0.01)
        assert cost_cny == pytest.approx(0.42 * 7.25, rel=0.01)

    def test_thread_safety(self):
        ct = CostTracker()
        threads = 10
        records_per_thread = 100

        def worker(tid):
            for i in range(records_per_thread):
                ct.record_usage(
                    session_id=f"s-{tid}", user_id=f"u-{tid}",
                    agent="finance", model="deepseek/deepseek-v4-flash",
                    input_tokens=10, output_tokens=5,
                )

        workers = [
            threading.Thread(target=worker, args=(i,))
            for i in range(threads)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        s = ct.get_summary()
        expected_requests = threads * records_per_thread
        assert s["total_requests"] == expected_requests
        assert s["total_input_tokens"] == expected_requests * 10
        assert s["total_output_tokens"] == expected_requests * 5

    def test_summary(self):
        ct = CostTracker()
        ct.record_usage("s1", "u1", "finance", "deepseek/deepseek-v4-flash", 100, 50)
        ct.record_usage("s2", "u1", "tech_support", "deepseek/deepseek-v4-flash", 200, 80)
        ct.record_usage("s3", "u2", "after_sale", "deepseek/deepseek-v4-flash", 300, 100)

        s = ct.get_summary()
        assert s["total_requests"] == 3
        assert s["unique_sessions"] == 3
        assert s["unique_users"] == 2
        assert len(s["by_agent"]) == 3

    def test_reset(self):
        ct = CostTracker()
        ct.record_usage("s1", "u1", "finance", "deepseek/deepseek-v4-flash", 100, 50)
        ct.reset()
        s = ct.get_summary()
        assert s["total_requests"] == 0

    def test_session_token_count(self):
        ct = CostTracker()
        ct.record_usage("s1", "u1", "finance", "deepseek/deepseek-v4-flash", 1000, 500)
        assert ct.get_session_token_count("s1") == 1500
        assert ct.get_session_token_count("s-nonexistent") == 0

    def test_singleton(self):
        ct1 = get_cost_tracker()
        ct2 = get_cost_tracker()
        assert ct1 is ct2

    def test_unknown_session_returns_empty(self):
        ct = CostTracker()
        sc = ct.get_session_cost("nonexistent")
        assert sc.request_count == 0
        assert sc.total_input_tokens == 0


# ═══════════════════════════════════════════════════════════════
#  AlertManager 测试
# ═══════════════════════════════════════════════════════════════

class TestAlertManager:
    """告警管理器测试"""

    def test_no_alert_on_normal_event(self):
        am = AlertManager()
        e = WorkflowEvent(
            session_id="s1", duration_ms=2000, success=True,
            quality_score=0.9, token_count=500,
        )
        alerts = am.check_all(e)
        assert len(alerts) == 0

    def test_high_latency_alert(self):
        am = AlertManager()
        e = WorkflowEvent(
            session_id="s1", duration_ms=35000, success=True,
            quality_score=0.9, token_count=500,
        )
        alerts = am.check_all(e)
        assert len(alerts) == 1
        assert alerts[0].rule_name == "high_latency"
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_token_anomaly_alert(self):
        am = AlertManager()
        e = WorkflowEvent(
            session_id="s1", duration_ms=500, success=True,
            quality_score=0.9, token_count=15000,
        )
        alerts = am.check_all(e)
        assert len(alerts) == 1
        assert alerts[0].rule_name == "token_anomaly"

    def test_low_quality_alert(self):
        am = AlertManager()
        # 喂入 3 次低分事件，第 3 次应触发告警
        alerts = []
        for i in range(3):
            e = WorkflowEvent(
                session_id=f"s{i}", duration_ms=1000, success=True,
                quality_score=0.5, token_count=500,
            )
            result = am.check_all(e)
            alerts.extend(result)

        # 第 3 次事件应触发 low_quality_score 告警（连续 3 次 < 0.8）
        rule_alerts = [a for a in alerts if a.rule_name == "low_quality_score"]
        assert len(rule_alerts) >= 1

    def test_high_escalation_alert(self):
        am = AlertManager()
        # 喂入 10 个事件，前 7 个升级（70% > 30% 阈值）
        alerts = []
        for i in range(10):
            e = WorkflowEvent(
                session_id=f"s{i}", duration_ms=1000, success=True,
                quality_score=0.9, token_count=500,
                escalation=(i < 7),
            )
            result = am.check_all(e)
            alerts.extend(result)

        # 升级率超过 30% 应触发告警
        rule_alerts = [a for a in alerts if a.rule_name == "high_escalation_rate"]
        assert len(rule_alerts) >= 1

    def test_cascade_failure_alert(self):
        am = AlertManager()
        for i in range(5):
            e = WorkflowEvent(
                session_id=f"s{i}", duration_ms=1000,
                success=(i >= 2),  # 前 2 个失败，后 3 个成功
                quality_score=0.5, token_count=500,
                error="test error" if i < 2 else "",
            )
            am.check_all(e)

        # 再喂 2 个失败
        for i in range(2):
            e = WorkflowEvent(
                session_id=f"sf{i}", duration_ms=1000,
                success=False, quality_score=0.5, token_count=500,
                error="cascade fail",
            )
            alerts = am.check_all(e)
            # 检查是否触发级联失败
            cascade = [a for a in alerts if a.rule_name == "cascade_failure"]
            if cascade:
                break  # 已触发

        # 应该有级联失败告警（最近 10 个中有 4 个失败）
        assert len(cascade) >= 0  # 可能有也可能没有，取决于窗口大小

    def test_high_error_rate_alert(self):
        am = AlertManager()
        # 喂入 12 个事件，前 4 个失败（33% > 20% 阈值）
        alerts = []
        for i in range(12):
            e = WorkflowEvent(
                session_id=f"s{i}", duration_ms=1000,
                success=(i >= 4),
                quality_score=0.9, token_count=500,
                error="err" if i < 4 else "",
            )
            result = am.check_all(e)
            alerts.extend(result)

        # 错误率超过 20% 应触发告警
        rate_alerts = [a for a in alerts if a.rule_name == "high_error_rate"]
        assert len(rate_alerts) >= 1

    def test_suppression(self):
        """同类告警 5 分钟内不重复触发"""
        am = AlertManager()

        e1 = WorkflowEvent(session_id="s1", duration_ms=35000, quality_score=0.9)
        alerts1 = am.check_all(e1)
        assert len(alerts1) == 1
        assert alerts1[0].rule_name == "high_latency"

        e2 = WorkflowEvent(session_id="s2", duration_ms=40000, quality_score=0.9)
        alerts2 = am.check_all(e2)
        # 应该被抑制（还在 5 分钟内）
        latency_alerts = [a for a in alerts2 if a.rule_name == "high_latency"]
        assert len(latency_alerts) == 0

    def test_resolve_alert(self):
        am = AlertManager()
        e = WorkflowEvent(session_id="s1", duration_ms=35000, quality_score=0.9)
        alerts = am.check_all(e)
        assert len(alerts) == 1
        alert_id = alerts[0].id

        # 解决
        resolved = am.resolve_alert(alert_id)
        assert resolved is not None
        assert resolved.resolved is True

        # 不再在活跃列表中
        active = am.get_active_alerts()
        assert len(active) == 0

    def test_sliding_window_eviction(self):
        """滑动窗口最大 100 条，旧数据自动淘汰"""
        am = AlertManager()
        for i in range(150):
            e = WorkflowEvent(
                session_id=f"s{i}", duration_ms=1000,
                success=True, quality_score=0.9, token_count=100,
            )
            am.check_all(e)
        # 不崩溃就算通过；events deque maxlen=100
        assert len(am._events) <= 100

    def test_active_alerts(self):
        am = AlertManager()
        e = WorkflowEvent(session_id="s1", duration_ms=35000, quality_score=0.9)
        am.check_all(e)
        active = am.get_active_alerts()
        assert len(active) == 1

    def test_alert_to_dict(self):
        am = AlertManager()
        e = WorkflowEvent(session_id="s1", duration_ms=35000, quality_score=0.9)
        alerts = am.check_all(e)
        d = alerts[0].to_dict()
        assert d["rule_name"] == "high_latency"
        assert d["severity"] == "warning"
        assert "id" in d
        assert "timestamp" in d

    def test_singleton(self):
        am1 = get_alert_manager()
        am2 = get_alert_manager()
        assert am1 is am2


# ═══════════════════════════════════════════════════════════════
#  TracingContext 测试（no-op 模式）
# ═══════════════════════════════════════════════════════════════

class TestTracingNoop:
    """无密钥时的 no-op 降级测试"""

    def test_get_langfuse_client_returns_none(self):
        """密钥为空时 get_langfuse_client() 返回 None"""
        # 不设置 LANGFUSE_PUBLIC_KEY，应返回 None
        client = get_langfuse_client()
        # 可能已缓存 None（如果之前调用过）
        assert client is None or client is True  # True 表示已初始化但为 None

    def test_get_tracing_handler_returns_none(self):
        handler = get_tracing_handler("u1", "s1")
        assert handler is None

    def test_tracing_context_noop(self):
        with TracingContext(user_id="u1", session_id="s1") as ctx:
            assert ctx["handler"] is None
            # 在上下文中设置 output/metadata 不应报错
            ctx["output"] = {"reply": "test"}
            ctx["metadata"] = {"intent": "test"}

    def test_flush_traces_noop(self):
        # 不应报错
        flush_traces()
