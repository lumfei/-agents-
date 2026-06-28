"""
策略引擎 (Policy Engine) + 护栏系统 (Guardrails)

职责：
  - 统一规则管理中心：所有安全规则在一处定义和管理
  - 硬护栏（Hard Guardrail）：触发即阻断，不可绕过
  - 软护栏（Soft Guardrail）：触发即警告+记录，但放行
  - Hooks 模式：PreToolUse / PostToolUse / PreOutput / PostInput
  - Guardrails 并行执行，快速失败（任一阻断即停止）

架构：
  Policy Engine（规则中心）
  ├── Guardrails（护栏执行器）
  │   ├── PreToolUseGuard   → 工具调用前检查
  │   ├── PostToolUseGuard  → 工具调用后检查
  │   ├── PreOutputGuard    → 输出返回前检查
  │   └── PostInputGuard    → 用户输入后检查
  └── Policy Rules（规则定义）
      ├── HardRule: refund_amount > 1000 → 必须人审
      ├── HardRule: tool=create_refund AND role=user AND amount > 1000 → BLOCK
      ├── SoftRule: output_contains_uncertain_terms → WARN
      └── ...

与 InputGuard/OutputAudit 的区别：
  - InputGuard/OutputAudit: 内容安全检测（注入、PII等）
  - Policy Engine: 业务规则管控（退款金额、审批流程、权限边界）

设计原则：
  - 声明式规则：规则用数据定义，不用代码
  - 独立执行：每个 Guardrail 独立运行，互不依赖
  - 快速失败：任一 Hard Guardrail 触发即停止
  - 可观测：所有触发事件记录到审计日志
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

class GuardrailType(str, Enum):
    """护栏类型"""
    HARD = "hard"  # 硬护栏：触发即阻断
    SOFT = "soft"  # 软护栏：警告但放行


class GuardrailHook(str, Enum):
    """护栏挂载点"""
    POST_INPUT = "post_input"        # 用户输入后
    PRE_TOOL_USE = "pre_tool_use"    # 工具调用前
    POST_TOOL_USE = "post_tool_use"  # 工具调用后
    PRE_OUTPUT = "pre_output"        # 输出返回前
    PRE_ROUTE = "pre_route"          # 路由决策前
    POST_CLASSIFY = "post_classify"  # 意图分类后


@dataclass
class GuardrailResult:
    """护栏执行结果"""
    passed: bool
    guardrail_name: str
    guardrail_type: GuardrailType
    hook: GuardrailHook
    reason: str = ""
    risk_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.passed and self.guardrail_type == GuardrailType.HARD


@dataclass
class PolicyRule:
    """
    一条策略规则。

    规则示例：
      # 硬规则：退款金额 > 1000 必须人工审批
      PolicyRule(
          name="refund_approval_threshold",
          hook=GuardrailHook.PRE_TOOL_USE,
          type_=GuardrailType.HARD,
          condition=lambda ctx: ctx.get("tool_name") == "create_refund"
                                and ctx.get("args", {}).get("amount", 0) > 1000,
          action="require_approval",
          message="退款金额超过 ¥1000，需要人工审批",
          priority=10,
      )
    """
    name: str
    hook: GuardrailHook
    type_: GuardrailType
    condition: Callable[[dict], bool]  # 触发条件函数
    action: str = "block"              # block / warn / require_approval / flag
    message: str = ""
    priority: int = 0                  # 优先级（数字越大越先执行）


# ═══════════════════════════════════════════════════════════════
#  Guardrail 基类
# ═══════════════════════════════════════════════════════════════

class BaseGuardrail(ABC):
    """护栏基类"""

    def __init__(self, name: str, hook: GuardrailHook, type_: GuardrailType):
        self.name = name
        self.hook = hook
        self.type_ = type_
        self._rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule) -> None:
        """注册规则"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_name: str) -> bool:
        """移除规则"""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < before

    @abstractmethod
    async def execute(self, context: dict) -> list[GuardrailResult]:
        """执行护栏检查，返回所有触发的结果"""
        ...


# ═══════════════════════════════════════════════════════════════
#  具体护栏实现
# ═══════════════════════════════════════════════════════════════

class RuleBasedGuardrail(BaseGuardrail):
    """
    基于规则的护栏（串行执行规则，Hard 快速失败）。
    """

    async def execute(self, context: dict) -> list[GuardrailResult]:
        results: list[GuardrailResult] = []

        for rule in self._rules:
            try:
                triggered = rule.condition(context)
            except Exception as e:
                logger.warning("护栏规则 '%s' 执行异常: %s", rule.name, e)
                continue

            if triggered:
                result = GuardrailResult(
                    passed=False,
                    guardrail_name=rule.name,
                    guardrail_type=rule.type_,
                    hook=self.hook,
                    reason=rule.message,
                    metadata={
                        "action": rule.action,
                        "priority": rule.priority,
                        "context_snapshot": {
                            k: str(v)[:100] for k, v in context.items()
                        },
                    },
                )
                results.append(result)

                # Hard Guardrail 触发 → 立即停止
                if rule.type_ == GuardrailType.HARD:
                    break

        return results


# ═══════════════════════════════════════════════════════════════
#  Policy Engine — 统一规则管理中心
# ═══════════════════════════════════════════════════════════════

class PolicyEngine:
    """
    策略引擎——统一管理所有护栏和规则。

    使用方式：
      engine = PolicyEngine()
      engine.register_default_rules()

      # 工具调用前检查
      results = await engine.check(
          hook=GuardrailHook.PRE_TOOL_USE,
          context={"tool_name": "create_refund", "args": {"amount": 5000}},
      )
      if any(r.blocked for r in results):
          return error_response(results[0].reason)

    并行 vs 串行：
      - 同一 Hook 的多个 Guardrail 并行执行
      - 单个 Guardrail 内的规则串行执行（Hard 快速失败）
    """

    def __init__(self):
        # hook → list of guardrails
        self._guardrails: dict[GuardrailHook, list[BaseGuardrail]] = {
            hook: [] for hook in GuardrailHook
        }

    def register_guardrail(self, guardrail: BaseGuardrail) -> None:
        """注册护栏"""
        self._guardrails[guardrail.hook].append(guardrail)

    def register_rule(self, rule: PolicyRule) -> None:
        """注册单条规则（自动创建对应护栏）"""
        # 查找是否有匹配的 guardrail
        for gr in self._guardrails.get(rule.hook, []):
            if gr.name == f"default_{rule.hook.value}":
                gr.add_rule(rule)
                return

        # 创建新的 guardrail
        gr = RuleBasedGuardrail(
            name=f"default_{rule.hook.value}",
            hook=rule.hook,
            type_=rule.type_,
        )
        gr.add_rule(rule)
        self.register_guardrail(gr)

    async def check(
        self,
        hook: GuardrailHook,
        context: dict,
    ) -> list[GuardrailResult]:
        """
        执行指定 Hook 的所有护栏检查。

        护栏并行执行，Hard 护栏快速失败。
        """
        guardrails = self._guardrails.get(hook, [])
        if not guardrails:
            return []

        # 并行执行所有 guardrails
        import asyncio
        all_results: list[list[GuardrailResult]] = await asyncio.gather(
            *[gr.execute(context) for gr in guardrails],
            return_exceptions=True,
        )

        # 展开结果
        results: list[GuardrailResult] = []
        for item in all_results:
            if isinstance(item, Exception):
                logger.error("护栏执行异常: %s", item)
            else:
                results.extend(item)

        return results

    def check_sync(
        self,
        hook: GuardrailHook,
        context: dict,
    ) -> list[GuardrailResult]:
        """
        同步版本的护栏检查（适用于 LangGraph 同步节点）。
        """
        guardrails = self._guardrails.get(hook, [])
        if not guardrails:
            return []

        results: list[GuardrailResult] = []
        for gr in guardrails:
            try:
                # 使用 asyncio.run 在同步上下文中执行
                import asyncio
                gr_results = asyncio.run(gr.execute(context))
                results.extend(gr_results)
            except Exception as e:
                logger.error("护栏 '%s' 执行异常: %s", gr.name, e)

        return results

    def get_hard_blocks(self, results: list[GuardrailResult]) -> list[GuardrailResult]:
        """获取所有硬护栏阻断结果"""
        return [r for r in results if r.blocked]

    def get_soft_warnings(self, results: list[GuardrailResult]) -> list[GuardrailResult]:
        """获取所有软护栏警告结果"""
        return [r for r in results if not r.passed and r.guardrail_type == GuardrailType.SOFT]

    # ── 默认规则 ────────────────────────────────────────────

    def register_default_rules(self) -> None:
        """
        注册默认的业务规则。

        这些规则定义了：
          - 退款金额阈值（>1000 需要审批）
          - 敏感操作频率限制
          - 越权操作拦截
          - 输出质量检查
        """
        # ── PreToolUse 规则 ──────────────────────────────────

        # 硬规则: 退款 > 1000 需要审批
        self.register_rule(PolicyRule(
            name="refund_approval_threshold",
            hook=GuardrailHook.PRE_TOOL_USE,
            type_=GuardrailType.HARD,
            condition=lambda ctx: (
                ctx.get("tool_name") == "create_refund"
                and ctx.get("args", {}).get("amount", 0) > 1000
            ),
            action="require_approval",
            message="退款金额超过 ¥1000，需要人工审批",
            priority=10,
        ))

        # 硬规则: 禁止修改其他用户的订单
        self.register_rule(PolicyRule(
            name="cross_user_order_access",
            hook=GuardrailHook.PRE_TOOL_USE,
            type_=GuardrailType.HARD,
            condition=lambda ctx: (
                ctx.get("tool_name") in ("query_order", "list_user_orders")
                and ctx.get("args", {}).get("user_id", "")
                and ctx.get("user_id", "")
                and ctx["args"]["user_id"] != ctx["user_id"]
                and ctx.get("role", "user") not in ("admin", "operator")
            ),
            action="block",
            message="无权查询其他用户的订单信息",
            priority=20,
        ))

        # 软规则: 频繁退款警告
        self.register_rule(PolicyRule(
            name="frequent_refund_warning",
            hook=GuardrailHook.PRE_TOOL_USE,
            type_=GuardrailType.SOFT,
            condition=lambda ctx: (
                ctx.get("tool_name") == "create_refund"
                and ctx.get("recent_refund_count", 0) >= 3
            ),
            action="warn",
            message="用户近期已多次申请退款，请注意风险",
            priority=5,
        ))

        # 硬规则: 空订单号
        self.register_rule(PolicyRule(
            name="empty_order_id",
            hook=GuardrailHook.PRE_TOOL_USE,
            type_=GuardrailType.HARD,
            condition=lambda ctx: (
                ctx.get("tool_name") == "query_order"
                and not ctx.get("args", {}).get("order_id", "").strip()
            ),
            action="block",
            message="订单号不能为空",
            priority=100,
        ))

        # 硬规则: 退款金额为负
        self.register_rule(PolicyRule(
            name="negative_refund_amount",
            hook=GuardrailHook.PRE_TOOL_USE,
            type_=GuardrailType.HARD,
            condition=lambda ctx: (
                ctx.get("tool_name") == "create_refund"
                and ctx.get("args", {}).get("amount", 0) <= 0
            ),
            action="block",
            message="退款金额必须大于 0",
            priority=100,
        ))

        # ── PreOutput 规则 ──────────────────────────────────

        # 软规则: 输出包含不确定表述
        self.register_rule(PolicyRule(
            name="uncertain_output_warning",
            hook=GuardrailHook.PRE_OUTPUT,
            type_=GuardrailType.SOFT,
            condition=lambda ctx: _check_uncertain_output(ctx.get("output", "")),
            action="warn",
            message="Agent 输出包含不确定表述，建议人工复核",
            priority=5,
        ))

        # 硬规则: 输出泄露系统提示词
        self.register_rule(PolicyRule(
            name="prompt_leak_block",
            hook=GuardrailHook.PRE_OUTPUT,
            type_=GuardrailType.HARD,
            condition=lambda ctx: _check_prompt_leak(ctx.get("output", "")),
            action="block",
            message="检测到系统提示词泄露，输出已拦截",
            priority=100,
        ))

        # ── PostInput 规则 ──────────────────────────────────

        # 硬规则: 空输入拦截
        self.register_rule(PolicyRule(
            name="empty_input_block",
            hook=GuardrailHook.POST_INPUT,
            type_=GuardrailType.HARD,
            condition=lambda ctx: not ctx.get("message", "").strip(),
            action="block",
            message="输入不能为空",
            priority=100,
        ))

        # ── PreRoute 规则 ──────────────────────────────────

        # 软规则: 低置信度分类
        self.register_rule(PolicyRule(
            name="low_confidence_route",
            hook=GuardrailHook.PRE_ROUTE,
            type_=GuardrailType.SOFT,
            condition=lambda ctx: ctx.get("confidence", 1.0) < 0.5,
            action="warn",
            message="意图分类置信度较低，可能路由不准确",
            priority=5,
        ))

        logger.info("默认策略规则已注册: %d 条", 10)


# ═══════════════════════════════════════════════════════════════
#  辅助检查函数
# ═══════════════════════════════════════════════════════════════

def _check_uncertain_output(output: str) -> bool:
    """检查输出是否包含不确定表述"""
    uncertain_terms = [
        "我不确定", "我推测", "可能", "应该", "大概",
        "或许", "据我所知", "根据我的了解",
    ]
    return any(term in output for term in uncertain_terms)


def _check_prompt_leak(output: str) -> bool:
    """检查输出是否泄露系统提示词"""
    leak_markers = [
        "# 角色（Role）", "# 任务（Task）", "# 边界（Boundary）",
        "系统提示词", "system prompt", "supervisor",
        "可用工具:", "你只能使用", "不要编造",
    ]
    return sum(1 for m in leak_markers if m in output) >= 2


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_policy_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
        _policy_engine.register_default_rules()
    return _policy_engine
