"""
Token 成本追踪模块

职责：
  - 追踪每次请求的 Token 消耗
  - 按 Agent、会话、用户、日期聚合
  - 计算美元成本（基于模型定价）

追踪维度：
  - 按 Agent 汇总：Supervisor / Tech / Finance / AfterSale
  - 按会话汇总：每个 session 消耗的总 Token
  - 按用户汇总：每个用户的总消耗
  - 按日期汇总：每天的成本

成本计算：
  - 输入 Token × 输入单价 + 输出 Token × 输出单价
  - 支持不同模型的不同计价
  - 未知模型使用 "default" 定价

使用方式：
  from app.observability.cost_tracker import get_cost_tracker

  tracker = get_cost_tracker()
  tracker.record_usage(
      session_id="sess-123", user_id="USR-001",
      agent="finance", model="deepseek/deepseek-v4-flash",
      input_tokens=1500, output_tokens=300,
  )
  print(tracker.get_session_cost("sess-123"))
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from app.config import settings

# ═══════════════════════════════════════════════════════════════
#  模型定价表（$/1M Token，2026 年参考价）
# ═══════════════════════════════════════════════════════════════

# 美元兑人民币汇率（2026 年参考）
USD_TO_CNY = 7.25

MODEL_PRICING: dict[str, dict[str, float]] = {
    "deepseek/deepseek-v4-flash": {
        "input": 0.14,   # $0.14 / 1M input tokens
        "output": 0.28,  # $0.28 / 1M output tokens
    },
    "deepseek/deepseek-v4-pro": {
        "input": 0.55,
        "output": 2.19,
    },
    "deepseek/deepseek-chat": {
        "input": 0.14,
        "output": 0.28,
    },
    "default": {
        "input": 0.27,
        "output": 1.10,
    },
}


# ═══════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class TokenUsage:
    """单次使用的 Token 记录"""
    session_id: str
    user_id: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_cny: float
    timestamp: float  # time.time()


@dataclass
class PeriodCost:
    """某个时间段的成本汇总"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_cost_cny: float = 0.0
    request_count: int = 0
    by_agent: dict = field(default_factory=dict)

    def merge(self, usage: TokenUsage) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost_usd += usage.cost_usd
        self.total_cost_cny += usage.cost_cny
        self.request_count += 1
        agent_key = usage.agent or "unknown"
        if agent_key not in self.by_agent:
            self.by_agent[agent_key] = {"input": 0, "output": 0, "cost": 0.0, "cost_cny": 0.0, "requests": 0}
        self.by_agent[agent_key]["input"] += usage.input_tokens
        self.by_agent[agent_key]["output"] += usage.output_tokens
        self.by_agent[agent_key]["cost"] += usage.cost_usd
        self.by_agent[agent_key]["cost_cny"] += usage.cost_cny
        self.by_agent[agent_key]["requests"] += 1


# ═══════════════════════════════════════════════════════════════
#  CostTracker
# ═══════════════════════════════════════════════════════════════

class CostTracker:
    """
    Token 成本追踪器（全内存，线程安全）。

    所有聚合数据存在内存中，应用重启后丢失。
    Phase 4 可扩展为 PostgreSQL 持久化。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._usages: list[TokenUsage] = []  # 完整记录（用于审计）
        self._by_session: dict[str, PeriodCost] = defaultdict(PeriodCost)
        self._by_user: dict[str, PeriodCost] = defaultdict(PeriodCost)
        self._by_day: dict[str, PeriodCost] = defaultdict(PeriodCost)
        self._total: PeriodCost = PeriodCost()

    # ── 记录 ────────────────────────────────────────────────

    def record_usage(
        self,
        session_id: str,
        user_id: str,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        记录一次 Token 使用。

        参数：
          session_id:    会话 ID
          user_id:       用户 ID
          agent:         调用的 Agent 名称（tech_support / finance / after_sale）
          model:         LLM 模型名（LiteLLM 格式）
          input_tokens:  输入 Token 数
          output_tokens: 输出 Token 数

        返回：
          本次使用的成本（美元）
        """
        import time

        cost_usd, cost_cny = self._calculate_cost(model, input_tokens, output_tokens)
        usage = TokenUsage(
            session_id=session_id or "unknown",
            user_id=user_id or "anonymous",
            agent=agent or "unknown",
            model=model or settings.LLM_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cost_cny=cost_cny,
            timestamp=time.time(),
        )

        today = str(date.today())

        with self._lock:
            self._usages.append(usage)
            self._by_session[session_id].merge(usage)
            self._by_user[user_id].merge(usage)
            self._by_day[today].merge(usage)
            self._total.merge(usage)

        return cost_usd, cost_cny

    @staticmethod
    def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        """
        计算 Token 成本（美元 + 人民币）。

        返回值：(cost_usd, cost_cny)

        定价逻辑：
          - 精确匹配 MODEL_PRICING 中的模型名
          - 前缀匹配（如 "deepseek/deepseek-v4-flash" 匹配前缀 "deepseek"）
          - 都不匹配时使用 "default" 定价
        """
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            # 尝试前缀匹配
            for prefix, p in MODEL_PRICING.items():
                if prefix != "default" and model.startswith(prefix):
                    pricing = p
                    break
            if pricing is None:
                pricing = MODEL_PRICING["default"]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cost_usd = round(input_cost + output_cost, 8)
        cost_cny = round(cost_usd * USD_TO_CNY, 8)
        return cost_usd, cost_cny

    # ── 查询 ────────────────────────────────────────────────

    def get_session_cost(self, session_id: str) -> PeriodCost:
        """获取指定会话的成本汇总"""
        with self._lock:
            return self._by_session.get(session_id, PeriodCost())

    def get_user_cost(self, user_id: str) -> PeriodCost:
        """获取指定用户的成本汇总"""
        with self._lock:
            return self._by_user.get(user_id, PeriodCost())

    def get_daily_cost(self, day: str | None = None) -> PeriodCost:
        """
        获取指定日期的成本汇总。

        参数：
          day: 日期字符串（"2026-06-24"），None 表示今天
        """
        if day is None:
            day = str(date.today())
        with self._lock:
            return self._by_day.get(day, PeriodCost())

    def get_session_token_count(self, session_id: str) -> int:
        """便捷方法：获取指定会话的总 Token 数"""
        cost = self.get_session_cost(session_id)
        return cost.total_input_tokens + cost.total_output_tokens

    def get_summary(self) -> dict:
        """
        获取全局统计摘要。

        返回示例：
          {
            "total_requests": 150,
            "total_input_tokens": 120000,
            "total_output_tokens": 45000,
            "total_cost_usd": 0.0425,
            "unique_sessions": 42,
            "unique_users": 18,
            "by_agent": {
              "finance": {"input": 50000, "output": 20000, "cost": 0.018, "requests": 60}
            }
          }
        """
        with self._lock:
            return {
                "total_requests": self._total.request_count,
                "total_input_tokens": self._total.total_input_tokens,
                "total_output_tokens": self._total.total_output_tokens,
                "total_cost_usd": round(self._total.total_cost_usd, 6),
                "total_cost_cny": round(self._total.total_cost_cny, 6),
                "unique_sessions": len(self._by_session),
                "unique_users": len(self._by_user),
                "by_agent": dict(self._total.by_agent),
            }

    # ── 管理 ────────────────────────────────────────────────

    def reset(self) -> None:
        """清空所有数据（用于测试）"""
        with self._lock:
            self._usages.clear()
            self._by_session.clear()
            self._by_user.clear()
            self._by_day.clear()
            self._total = PeriodCost()

    def get_recent_usages(self, count: int = 20) -> list[TokenUsage]:
        """获取最近 N 条使用记录"""
        with self._lock:
            return self._usages[-count:]


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """获取 CostTracker 全局单例"""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
