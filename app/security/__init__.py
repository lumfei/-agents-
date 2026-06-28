"""
安全防护包 — 五层纵深防御

导出:
  - InputGuard / get_input_guard          — 第一层: 输入安全网关
  - ToolSandbox / get_tool_sandbox        — 第二层: 工具沙箱
  - ToolSecurityValidator / get_tool_validator — 第三层: 工具调用安全校验
  - OutputAudit / get_output_audit         — 第四层: 输出安全审核
  - AuditLogService / get_audit_log        — 第五层: 审计日志
  - PermissionEngine / get_permission_engine — 权限模型 (RBAC/ABAC)
  - PolicyEngine / get_policy_engine       — 策略引擎 & 护栏
  - GuardrailHook / GuardrailType          — 护栏挂载点和类型
  - SecurityContext                        — 安全上下文（聚合所有检查）
"""

from app.security.input_guard import (
    InputGuard, GuardResult, GuardAction, RiskLevel, MatchDetail,
    get_input_guard,
)
from app.security.tool_sandbox import (
    ToolSandbox, ToolSecurityValidator, ToolResult, ToolSecurityResult,
    SandboxMode, SecurityVerdict, ParamRule, ParamCheckResult,
    get_tool_sandbox, get_tool_validator,
)
from app.security.output_audit import (
    OutputAudit, OutputAuditResult, OutputAction, PIIMatch,
    get_output_audit,
)
from app.security.audit_log import (
    AuditLogService, AuditEntry, AuditAction, AuditRiskLevel,
    get_audit_log,
)
from app.security.permissions import (
    PermissionEngine, PermissionResult, Role, Permission,
    ABACContext, ROLE_PERMISSIONS, TOOL_PERMISSION_MAP,
    get_permission_engine,
)
from app.security.policy_engine import (
    PolicyEngine, PolicyRule, GuardrailResult,
    GuardrailHook, GuardrailType,
    get_policy_engine,
)

__all__ = [
    # Input Guard (Layer 1)
    "InputGuard", "GuardResult", "GuardAction", "RiskLevel", "MatchDetail",
    "get_input_guard",
    # Tool Sandbox (Layer 2-3)
    "ToolSandbox", "ToolSecurityValidator", "ToolResult", "ToolSecurityResult",
    "SandboxMode", "SecurityVerdict", "ParamRule", "ParamCheckResult",
    "get_tool_sandbox", "get_tool_validator",
    # Output Audit (Layer 4)
    "OutputAudit", "OutputAuditResult", "OutputAction", "PIIMatch",
    "get_output_audit",
    # Audit Log (Layer 5)
    "AuditLogService", "AuditEntry", "AuditAction", "AuditRiskLevel",
    "get_audit_log",
    # Permissions (RBAC/ABAC)
    "PermissionEngine", "PermissionResult", "Role", "Permission",
    "ABACContext", "ROLE_PERMISSIONS", "TOOL_PERMISSION_MAP",
    "get_permission_engine",
    # Policy Engine & Guardrails
    "PolicyEngine", "PolicyRule", "GuardrailResult",
    "GuardrailHook", "GuardrailType",
    "get_policy_engine",
]
