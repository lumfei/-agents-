"""
权限模型 RBAC/ABAC（Role-Based + Attribute-Based Access Control）

核心原则：Agent 的操作权限与用户身份绑定，Agent 不能做用户自己没权限做的事。

RBAC（基于角色）：
  - 定义角色层级：admin > operator > agent > user > viewer
  - 每个角色有对应的工具权限和数据权限
  - Agent 继承用户的角色权限（Agent 是用户的代理）

ABAC（基于属性）：
  - 用户只能访问自己的数据（user_id 绑定）
  - 订单金额限制（普通用户退款 <1000，VIP <5000）
  - 时间窗口限制（退款仅在 30 天内）
  - 操作频率限制（每小时最多 N 次退款）
  - 设备/地域限制（异常登录检测）

权限检查流程：
  用户请求 → 角色检查 → 属性检查 → 频率检查 → 允许/拒绝

关键安全规则：
  - Agent 不能越权操作（用户的权限 ≤ Agent 的权限）
  - "被授权" 不等于 "安全"——每个操作都需要显式授权
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  角色和权限定义
# ═══════════════════════════════════════════════════════════════

class Role(str, Enum):
    """用户/Agent 角色"""
    ADMIN = "admin"         # 管理员：所有权限
    OPERATOR = "operator"   # 运营人员：查看+操作+审批
    AGENT = "agent"         # AI Agent：受限操作（继承用户权限）
    USER_VIP = "user_vip"   # VIP 用户：更高的操作限额
    USER = "user"           # 普通用户：基本操作
    VIEWER = "viewer"       # 只读用户：仅查看


# 角色层级（数字越大权限越高）
ROLE_HIERARCHY: dict[Role, int] = {
    Role.ADMIN: 100,
    Role.OPERATOR: 80,
    Role.AGENT: 60,
    Role.USER_VIP: 50,
    Role.USER: 30,
    Role.VIEWER: 10,
}


class Permission(str, Enum):
    """细粒度权限"""
    # 订单
    ORDER_READ = "order:read"
    ORDER_READ_OWN = "order:read_own"
    ORDER_LIST = "order:list"
    ORDER_UPDATE = "order:update"

    # 退款
    REFUND_CREATE = "refund:create"
    REFUND_READ = "refund:read"
    REFUND_APPROVE = "refund:approve"

    # 物流
    LOGISTICS_READ = "logistics:read"
    LOGISTICS_READ_OWN = "logistics:read_own"

    # 知识库
    KB_SEARCH = "kb:search"
    KB_MANAGE = "kb:manage"

    # 系统
    SYSTEM_STATUS = "system:status"
    SYSTEM_CONFIG = "system:config"
    USER_INFO_READ = "user_info:read"
    USER_INFO_MANAGE = "user_info:manage"

    # 审批
    APPROVAL_VIEW = "approval:view"
    APPROVAL_DECIDE = "approval:decide"

    # 审计
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


# 角色 → 权限映射
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),  # 所有权限

    Role.OPERATOR: {
        Permission.ORDER_READ, Permission.ORDER_LIST,
        Permission.REFUND_READ, Permission.REFUND_APPROVE,
        Permission.LOGISTICS_READ,
        Permission.KB_SEARCH, Permission.KB_MANAGE,
        Permission.SYSTEM_STATUS,
        Permission.USER_INFO_READ,
        Permission.APPROVAL_VIEW, Permission.APPROVAL_DECIDE,
        Permission.AUDIT_READ,
    },

    Role.AGENT: {
        # Agent 权限与用户绑定，这些是基础权限
        Permission.ORDER_READ_OWN, Permission.ORDER_LIST,
        Permission.REFUND_CREATE, Permission.REFUND_READ,
        Permission.LOGISTICS_READ_OWN,
        Permission.KB_SEARCH,
        Permission.SYSTEM_STATUS,
        Permission.USER_INFO_READ,
    },

    Role.USER_VIP: {
        Permission.ORDER_READ_OWN, Permission.ORDER_LIST,
        Permission.REFUND_CREATE, Permission.REFUND_READ,
        Permission.LOGISTICS_READ_OWN,
        Permission.KB_SEARCH,
        Permission.SYSTEM_STATUS,
    },

    Role.USER: {
        Permission.ORDER_READ_OWN, Permission.ORDER_LIST,
        Permission.REFUND_CREATE, Permission.REFUND_READ,
        Permission.LOGISTICS_READ_OWN,
    },

    Role.VIEWER: {
        Permission.ORDER_READ_OWN,
        Permission.LOGISTICS_READ_OWN,
    },
}

# 工具 → 所需权限映射
TOOL_PERMISSION_MAP: dict[str, Permission] = {
    "query_order": Permission.ORDER_READ_OWN,       # 用户可查自己的订单
    "list_user_orders": Permission.ORDER_LIST,
    "create_refund": Permission.REFUND_CREATE,
    "query_refund_status": Permission.REFUND_READ,
    "track_logistics": Permission.LOGISTICS_READ_OWN,
    "query_logistics_by_order": Permission.LOGISTICS_READ_OWN,
    "search_knowledge_base": Permission.KB_SEARCH,
    "check_service_status": Permission.SYSTEM_STATUS,
    "query_user_info": Permission.USER_INFO_READ,
    "get_system_announcements": Permission.SYSTEM_STATUS,
}


# ═══════════════════════════════════════════════════════════════
#  ABAC 属性策略
# ═══════════════════════════════════════════════════════════════

@dataclass
class ABACContext:
    """ABAC 评估上下文"""
    user_id: str = ""
    role: Role = Role.USER
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PermissionResult:
    """权限检查结果"""
    allowed: bool
    reason: str = ""
    required_permission: Permission | None = None
    user_role: Role = Role.USER
    details: dict = field(default_factory=dict)


# ABAC 策略函数类型
ABACPolicy = Callable[[ABACContext], PermissionResult]


# ═══════════════════════════════════════════════════════════════
#  ABAC 策略集
# ═══════════════════════════════════════════════════════════════

class ABACPolicies:
    """ABAC 属性策略集合"""

    @staticmethod
    def own_data_only(ctx: ABACContext) -> PermissionResult:
        """
        用户只能操作自己的数据。

        检查 tool_args 中的 user_id 是否与 ctx.user_id 匹配。
        """
        args_user = ctx.tool_args.get("user_id", "")
        if args_user and args_user != ctx.user_id:
            # 除非是 OPERATOR/ADMIN（可以查看他人数据）
            if ROLE_HIERARCHY[ctx.role] < ROLE_HIERARCHY[Role.OPERATOR]:
                return PermissionResult(
                    allowed=False,
                    reason=f"无权操作其他用户的数据（请求: {args_user}, 实际: {ctx.user_id}）",
                    user_role=ctx.role,
                    details={"requested_user": args_user, "actual_user": ctx.user_id},
                )
        return PermissionResult(allowed=True, user_role=ctx.role)

    @staticmethod
    def refund_amount_limit(ctx: ABACContext) -> PermissionResult:
        """
        退款金额限制（按角色）。

        - USER:  最大 1000 元/次
        - VIP:   最大 5000 元/次
        - AGENT: 继承用户角色限制
        - 超过阈值需要 OPERATOR/ADMIN 审批
        """
        if ctx.tool_name != "create_refund":
            return PermissionResult(allowed=True, user_role=ctx.role)

        amount = ctx.tool_args.get("amount", 0)
        limits = {
            Role.USER: 1000.0,
            Role.USER_VIP: 5000.0,
            Role.AGENT: 1000.0,  # Agent 默认按普通用户限制
            Role.OPERATOR: 50000.0,
            Role.ADMIN: float("inf"),
        }
        limit = limits.get(ctx.role, 1000.0)

        if amount > limit:
            return PermissionResult(
                allowed=False,
                reason=f"退款金额 ¥{amount} 超过角色限额 ¥{limit}",
                user_role=ctx.role,
                details={"amount": amount, "limit": limit},
            )
        return PermissionResult(allowed=True, user_role=ctx.role)

    @staticmethod
    def order_ownership(ctx: ABACContext) -> PermissionResult:
        """
        订单归属检查。

        用户只能查询自己的订单。
        订单号中的用户段需要匹配。
        """
        if ctx.tool_name not in ("query_order", "list_user_orders", "query_logistics_by_order"):
            return PermissionResult(allowed=True, user_role=ctx.role)

        # Admin/Operator 可查所有
        if ROLE_HIERARCHY[ctx.role] >= ROLE_HIERARCHY[Role.OPERATOR]:
            return PermissionResult(allowed=True, user_role=ctx.role)

        # 检查 order_id 是否属于该用户
        order_id = ctx.tool_args.get("order_id", "")
        # 简单的用户关联检查（生产环境应查数据库）
        # 这里用 user_id 直接参数做检查
        args_user = ctx.tool_args.get("user_id", "")
        if args_user and args_user != ctx.user_id:
            return PermissionResult(
                allowed=False,
                reason=f"无权查询其他用户的订单",
                user_role=ctx.role,
            )
        return PermissionResult(allowed=True, user_role=ctx.role)

    @staticmethod
    def tool_access_allowed(ctx: ABACContext) -> PermissionResult:
        """
        检查用户角色是否有权限调用该工具。
        """
        required = TOOL_PERMISSION_MAP.get(ctx.tool_name)
        if required is None:
            # 未注册的工具 → 默认拒绝（白名单策略）
            return PermissionResult(
                allowed=False,
                reason=f"工具 '{ctx.tool_name}' 未在权限系统中注册",
                user_role=ctx.role,
            )

        user_perms = ROLE_PERMISSIONS.get(ctx.role, set())
        if required not in user_perms:
            return PermissionResult(
                allowed=False,
                reason=f"角色 {ctx.role.value} 无权执行 '{ctx.tool_name}'（需要权限 {required.value}）",
                required_permission=required,
                user_role=ctx.role,
            )
        return PermissionResult(allowed=True, required_permission=required, user_role=ctx.role)


# ═══════════════════════════════════════════════════════════════
#  PermissionEngine — 权限引擎
# ═══════════════════════════════════════════════════════════════

class PermissionEngine:
    """
    权限引擎——统一 RBAC + ABAC 检查。

    使用方式：
      engine = PermissionEngine()
      ctx = ABACContext(user_id="USR-001", role=Role.USER,
                        tool_name="create_refund", tool_args={"amount": 500})
      result = engine.check(ctx)
      if not result.allowed:
          raise PermissionDenied(result.reason)
    """

    def __init__(self):
        self._abac_policies: list[ABACPolicy] = [
            ABACPolicies.tool_access_allowed,   # 1. RBAC: 角色权限
            ABACPolicies.own_data_only,          # 2. ABAC: 数据归属
            ABACPolicies.refund_amount_limit,    # 3. ABAC: 金额限制
            ABACPolicies.order_ownership,        # 4. ABAC: 订单归属
        ]

        # 频率限制状态（内存实现，生产用 Redis）
        self._rate_limits: dict[str, list[float]] = {}

    def check(self, ctx: ABACContext) -> PermissionResult:
        """
        执行完整的权限检查链。

        检查顺序（短路求值，第一个拒绝即返回）：
          1. RBAC 角色权限
          2. ABAC 数据归属
          3. ABAC 金额限制
          4. ABAC 订单归属
          5. 频率限制

        返回第一个拒绝的 PermissionResult，或最终允许。
        """
        for policy in self._abac_policies:
            result = policy(ctx)
            if not result.allowed:
                logger.warning(
                    "权限拒绝: user=%s role=%s tool=%s reason=%s",
                    ctx.user_id, ctx.role.value, ctx.tool_name, result.reason,
                )
                return result

        # 频率限制
        rate_result = self._check_rate_limit(ctx)
        if not rate_result.allowed:
            return rate_result

        logger.debug(
            "权限通过: user=%s role=%s tool=%s",
            ctx.user_id, ctx.role.value, ctx.tool_name,
        )
        return PermissionResult(allowed=True, user_role=ctx.role)

    def check_permission(self, role: Role, permission: Permission) -> bool:
        """简单 RBAC 权限检查（不涉及 ABAC 上下文）"""
        return permission in ROLE_PERMISSIONS.get(role, set())

    def get_user_permissions(self, role: Role) -> set[Permission]:
        """获取某角色的所有权限"""
        return ROLE_PERMISSIONS.get(role, set())

    def get_role_level(self, role: Role) -> int:
        """获取角色层级数值"""
        return ROLE_HIERARCHY.get(role, 0)

    # ── 频率限制 ────────────────────────────────────────────

    def _check_rate_limit(self, ctx: ABACContext) -> PermissionResult:
        """
        频率限制检查（滑动窗口）。

        规则：
          - 退款操作：每用户每小时最多 5 次
          - 查询操作：每用户每分钟最多 30 次
        """
        now = time.time()
        key = f"{ctx.user_id}:{ctx.tool_name}"

        # 获取该用户+工具的历史调用时间
        history = self._rate_limits.get(key, [])

        # 根据工具类型设置限制
        if ctx.tool_name == "create_refund":
            window = 3600  # 1 小时
            max_calls = 5
        else:
            window = 60    # 1 分钟
            max_calls = 30

        # 清理窗口外的记录
        history = [t for t in history if now - t < window]

        if len(history) >= max_calls:
            return PermissionResult(
                allowed=False,
                reason=f"频率限制: '{ctx.tool_name}' 每 {window}s 最多 {max_calls} 次",
                user_role=ctx.role,
                details={"current_count": len(history), "max": max_calls, "window": window},
            )

        # 记录本次调用
        history.append(now)
        self._rate_limits[key] = history

        return PermissionResult(allowed=True, user_role=ctx.role)


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_permission_engine: PermissionEngine | None = None


def get_permission_engine() -> PermissionEngine:
    global _permission_engine
    if _permission_engine is None:
        _permission_engine = PermissionEngine()
    return _permission_engine
