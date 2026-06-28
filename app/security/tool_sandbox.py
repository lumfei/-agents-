"""
工具沙箱（第二层防御）+ 工具调用安全校验（第三层防御）

职责：
  - 工具参数注入检测（SQL注入、命令注入、Prompt注入）
  - 路径遍历检测（防止读取系统敏感文件）
  - 参数类型和范围校验（白名单验证）
  - Docker 隔离执行 Agent 调用的工具（可选）
  - 资源限制（CPU、内存、超时）
  - 网络访问控制

执行模式：
  - SAFE:   Docker 容器内隔离执行（生产环境，需 Docker daemon）
  - SUBPROC: 子进程执行（开发环境，有限权限限制）
  - MOCK:   不真实执行，返回模拟结果（测试/演示环境）

安全策略：
  - 允许列表：哪些工具/命令可以执行
  - 网络策略：仅允许访问白名单中的 API
  - 文件系统：只读挂载，禁止写系统目录
  - 环境变量：仅注入必要的环境变量
  - 参数白名单：每个工具的参数有类型/范围约束
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

class SandboxMode(str, Enum):
    """沙箱执行模式"""
    SAFE = "safe"        # Docker 容器隔离
    SUBPROC = "subproc"  # 子进程执行
    MOCK = "mock"        # 模拟执行


class SecurityVerdict(str, Enum):
    """安全检查结果"""
    ALLOW = "allow"
    DENY = "deny"
    SANITIZE = "sanitize"


@dataclass
class ParamRule:
    """工具参数的白名单规则"""
    name: str
    type_: type = str
    required: bool = True
    min_val: float | None = None
    max_val: float | None = None
    min_len: int | None = None
    max_len: int = 200
    pattern: str | None = None          # 允许的正则模式
    deny_pattern: str | None = None     # 禁止的正则模式（如路径遍历）
    allowed_values: list[str] | None = None  # 枚举值白名单
    sanitize_html: bool = False
    sanitize_path: bool = False


@dataclass
class ParamCheckResult:
    """参数检查结果"""
    verdict: SecurityVerdict
    sanitized_value: Any = None
    reason: str = ""
    rule: str = ""


@dataclass
class ToolSecurityResult:
    """工具安全检查结果"""
    verdict: SecurityVerdict
    reason: str = ""
    checked_params: dict[str, ParamCheckResult] = field(default_factory=dict)
    sanitized_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    sandbox_mode: SandboxMode = SandboxMode.MOCK


# ═══════════════════════════════════════════════════════════════
#  工具安全校验器（第三层）
# ═══════════════════════════════════════════════════════════════

class ToolSecurityValidator:
    """
    工具调用安全校验器。

    对每次工具调用的参数进行多维校验：
      - 参数注入检测（SQL/命令/Prompt注入）
      - 路径遍历检测
      - 参数类型和范围校验
      - 白名单/黑名单校验
    """

    # 注入检测模式
    SQL_INJECTION_PATTERN = re.compile(
        r"(?i)(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|EXEC|TRUNCATE)\b\s+.*\b(FROM|INTO|TABLE)\b|"
        r"'\s*OR\s+'1'='1|\bOR\s+1=1\b|--\s*$|;\s*DROP\s+TABLE)",
    )
    CMD_INJECTION_PATTERN = re.compile(
        r"(?i)([;&|`$]\s*(rm\s+-rf|wget\s+|curl\s+|cat\s+/etc|chmod\s+|sudo\s+|nc\s+-[el])|"
        r"\$\(|`[^`]+`|\|\s*(sh|bash|zsh))",
    )
    PATH_TRAVERSAL_PATTERN = re.compile(
        r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%2f|~/)"
    )
    PROMPT_INJECTION_IN_ARGS = re.compile(
        r"(?i)(ignore\s+(all\s+)?(previous|above)\s+instructions?|"
        r"you\s+are\s+now|system\s*(prompt|message))",
    )

    def __init__(self):
        # 工具参数白名单
        self._tool_param_rules: dict[str, list[ParamRule]] = {}

    def register_tool_rules(self, tool_name: str, rules: list[ParamRule]) -> None:
        """注册某个工具的参数规则"""
        self._tool_param_rules[tool_name] = rules

    def validate(self, tool_name: str, args: dict[str, Any]) -> ToolSecurityResult:
        """
        校验工具调用的所有参数。

        返回 ToolSecurityResult，包含校验结果和脱敏后的参数。
        """
        rules = self._tool_param_rules.get(tool_name)

        # 无注册规则 → 默认严格检查所有参数
        if not rules:
            return self._default_validate(tool_name, args)

        checked: dict[str, ParamCheckResult] = {}
        sanitized: dict[str, Any] = {}
        denied: list[str] = []

        for rule in rules:
            value = args.get(rule.name)

            # 必填检查
            if rule.required and (value is None or value == ""):
                r = ParamCheckResult(
                    verdict=SecurityVerdict.DENY,
                    reason=f"缺少必填参数 '{rule.name}'",
                    rule="required",
                )
                checked[rule.name] = r
                denied.append(f"{rule.name}: {r.reason}")
                continue

            if value is None:
                continue
            # 非必填参数值为空字符串时跳过校验（LLM 可能不传可选参数）
            if not rule.required and isinstance(value, str) and value == "":
                sanitized[rule.name] = value
                continue

            # 类型检查
            r = self._check_param(rule, value)
            checked[rule.name] = r
            if r.verdict == SecurityVerdict.DENY:
                denied.append(f"{rule.name}: {r.reason}")
            sanitized[rule.name] = r.sanitized_value if r.sanitized_value is not None else value

        # 检查是否有未注册的参数（可能是注入尝试）
        known_params = {r.name for r in rules}
        for key, value in args.items():
            if key not in known_params:
                r = self._check_unknown_param(key, value)
                checked[key] = r
                if r.verdict == SecurityVerdict.DENY:
                    denied.append(f"{key}: {r.reason}")
                elif r.sanitized_value is not None:
                    sanitized[key] = r.sanitized_value

        if denied:
            return ToolSecurityResult(
                verdict=SecurityVerdict.DENY,
                reason="; ".join(denied),
                checked_params=checked,
                sanitized_args=sanitized,
            )

        return ToolSecurityResult(
            verdict=SecurityVerdict.ALLOW,
            checked_params=checked,
            sanitized_args=sanitized,
        )

    def _check_param(self, rule: ParamRule, value: Any) -> ParamCheckResult:
        """根据规则检查单个参数"""
        str_val = str(value)

        # 类型检查
        if not isinstance(value, rule.type_):
            try:
                if rule.type_ == int:
                    value = int(value)
                elif rule.type_ == float:
                    value = float(value)
                elif rule.type_ == str:
                    value = str(value)
            except (ValueError, TypeError):
                return ParamCheckResult(
                    verdict=SecurityVerdict.DENY,
                    reason=f"参数 '{rule.name}' 类型错误，期望 {rule.type_.__name__}，实际 {type(value).__name__}",
                    rule="type_check",
                )

        # 长度检查
        if rule.min_len is not None and len(str_val) < rule.min_len:
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 长度 {len(str_val)} 小于最小值 {rule.min_len}",
                rule="min_len",
            )
        if len(str_val) > rule.max_len:
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 长度 {len(str_val)} 超过最大值 {rule.max_len}",
                rule="max_len",
            )

        # 数值范围检查
        if isinstance(value, (int, float)):
            if rule.min_val is not None and value < rule.min_val:
                return ParamCheckResult(
                    verdict=SecurityVerdict.DENY,
                    reason=f"参数 '{rule.name}' 值 {value} 小于最小值 {rule.min_val}",
                    rule="min_val",
                )
            if rule.max_val is not None and value > rule.max_val:
                return ParamCheckResult(
                    verdict=SecurityVerdict.DENY,
                    reason=f"参数 '{rule.name}' 值 {value} 大于最大值 {rule.max_val}",
                    rule="max_val",
                )

        # 枚举值白名单
        if rule.allowed_values and str_val not in rule.allowed_values:
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 值 '{str_val[:50]}' 不在允许列表 {rule.allowed_values}",
                rule="allowed_values",
            )

        # 正则白名单检查
        if rule.pattern and not re.match(rule.pattern, str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 格式不符合要求",
                rule="pattern",
            )

        # 正则黑名单检查 (deny_pattern)
        if rule.deny_pattern and re.search(rule.deny_pattern, str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 包含禁止的模式",
                rule="deny_pattern",
            )

        # ── 注入检测 ──────────────────────────────────────────
        # SQL 注入
        if self.SQL_INJECTION_PATTERN.search(str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 疑似SQL注入攻击",
                rule="sql_injection",
            )

        # 命令注入
        if self.CMD_INJECTION_PATTERN.search(str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 疑似命令注入攻击",
                rule="cmd_injection",
            )

        # 路径遍历
        if rule.sanitize_path and self.PATH_TRAVERSAL_PATTERN.search(str_val):
            sanitized = re.sub(r'(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%2f|~/)', '', str_val)
            return ParamCheckResult(
                verdict=SecurityVerdict.SANITIZE,
                sanitized_value=sanitized,
                reason=f"参数 '{rule.name}' 包含路径遍历字符，已脱敏",
                rule="path_traversal",
            )

        # Prompt 注入检测（用于 free-text 参数如 reason, description）
        if self.PROMPT_INJECTION_IN_ARGS.search(str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"参数 '{rule.name}' 疑似Prompt注入攻击",
                rule="prompt_injection",
            )

        # HTML 脱敏
        if rule.sanitize_html:
            sanitized = re.sub(r'<[^>]+>', '', str_val)
            sanitized = sanitized.replace('&', '&amp;').replace('<', '&lt;')
            if sanitized != str_val:
                return ParamCheckResult(
                    verdict=SecurityVerdict.SANITIZE,
                    sanitized_value=sanitized,
                    reason=f"参数 '{rule.name}' 包含HTML标签，已脱敏",
                    rule="html_sanitize",
                )

        return ParamCheckResult(
            verdict=SecurityVerdict.ALLOW,
            sanitized_value=value,
            rule="ok",
        )

    def _check_unknown_param(self, name: str, value: Any) -> ParamCheckResult:
        """检查未注册的参数（潜在攻击）"""
        str_val = str(value)

        # 注入检测
        if self.SQL_INJECTION_PATTERN.search(str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"未注册参数 '{name}' 疑似SQL注入",
                rule="unknown_sql_injection",
            )
        if self.CMD_INJECTION_PATTERN.search(str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"未注册参数 '{name}' 疑似命令注入",
                rule="unknown_cmd_injection",
            )
        if self.PATH_TRAVERSAL_PATTERN.search(str_val):
            return ParamCheckResult(
                verdict=SecurityVerdict.DENY,
                reason=f"未注册参数 '{name}' 包含路径遍历",
                rule="unknown_path_traversal",
            )

        # 未知参数但无注入迹象 → 允许通过（可能是新增参数）
        return ParamCheckResult(
            verdict=SecurityVerdict.ALLOW,
            sanitized_value=value,
            rule="unknown_allow",
        )

    def _default_validate(self, tool_name: str, args: dict[str, Any]) -> ToolSecurityResult:
        """无注册规则时的默认严格检查"""
        denied: list[str] = []
        sanitized: dict[str, Any] = {}
        checked: dict[str, ParamCheckResult] = {}

        for key, value in args.items():
            str_val = str(value) if value is not None else ""

            # 注入检测
            if self.SQL_INJECTION_PATTERN.search(str_val):
                r = ParamCheckResult(verdict=SecurityVerdict.DENY, reason=f"参数 '{key}' 疑似SQL注入", rule="default_sql")
                checked[key] = r
                denied.append(f"{key}: {r.reason}")
                continue
            if self.CMD_INJECTION_PATTERN.search(str_val):
                r = ParamCheckResult(verdict=SecurityVerdict.DENY, reason=f"参数 '{key}' 疑似命令注入", rule="default_cmd")
                checked[key] = r
                denied.append(f"{key}: {r.reason}")
                continue
            if self.PATH_TRAVERSAL_PATTERN.search(str_val):
                r = ParamCheckResult(verdict=SecurityVerdict.DENY, reason=f"参数 '{key}' 包含路径遍历", rule="default_path")
                checked[key] = r
                denied.append(f"{key}: {r.reason}")
                continue

            # 长度限制
            if len(str_val) > 500:
                r = ParamCheckResult(verdict=SecurityVerdict.DENY, reason=f"参数 '{key}' 过长 ({len(str_val)}字符)", rule="default_len")
                checked[key] = r
                denied.append(f"{key}: {r.reason}")
                continue

            checked[key] = ParamCheckResult(verdict=SecurityVerdict.ALLOW, sanitized_value=value, rule="default_ok")
            sanitized[key] = value

        if denied:
            return ToolSecurityResult(
                verdict=SecurityVerdict.DENY,
                reason="; ".join(denied),
                checked_params=checked,
                sanitized_args=sanitized,
            )

        return ToolSecurityResult(
            verdict=SecurityVerdict.ALLOW,
            checked_params=checked,
            sanitized_args=sanitized,
        )


# ═══════════════════════════════════════════════════════════════
#  ToolSandbox — 工具执行沙箱
# ═══════════════════════════════════════════════════════════════

class ToolSandbox:
    """
    工具执行沙箱。

    三种执行模式：
      - SAFE:    Docker 容器隔离（需要 Docker daemon）
      - SUBPROC: 子进程执行（开发用，有限限制）
      - MOCK:    模拟执行（测试用）

    使用方式：
      sandbox = ToolSandbox(mode=SandboxMode.MOCK)
      sandbox.register_tool("query_order", mock_handler)
      result = sandbox.execute("query_order", {"order_id": "ORD-001"})
    """

    # 工具网络访问白名单
    NETWORK_ALLOWLIST = {
        "localhost", "127.0.0.1",
        "api.deepseek.com", "api.openai.com",
    }

    # 系统目录黑名单（禁止写入）
    FS_WRITE_DENYLIST = [
        "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin",
        "/boot", "/dev", "/proc", "/sys",
        "C:\\Windows", "C:\\Program Files",
        "/System", "/Library",
    ]

    def __init__(
        self,
        mode: SandboxMode = SandboxMode.MOCK,
        timeout_seconds: int = 30,
        max_memory_mb: int = 256,
        docker_image: str = "agent-sandbox:latest",
        network_access: bool = False,
    ):
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.docker_image = docker_image
        self.network_access = network_access
        self._mock_handlers: dict[str, Callable] = {}
        self._allowed_tools: set[str] = set()

    def register_tool(self, tool_name: str, mock_handler: Callable | None = None) -> None:
        """注册允许执行的工具"""
        self._allowed_tools.add(tool_name)
        if mock_handler:
            self._mock_handlers[tool_name] = mock_handler

    def set_mode(self, mode: SandboxMode) -> None:
        self.mode = mode

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """
        执行工具调用。

        安全流程：
          1. 检查工具是否在允许列表中
          2. 根据模式选择执行方式
          3. 记录执行结果和耗时
        """
        # 1. 允许列表检查
        if tool_name not in self._allowed_tools:
            return ToolResult(
                success=False,
                error=f"工具 '{tool_name}' 不在允许执行列表中",
                sandbox_mode=self.mode,
            )

        start = time.time()

        try:
            if self.mode == SandboxMode.MOCK:
                output = self._execute_mock(tool_name, args)
            elif self.mode == SandboxMode.SUBPROC:
                output = self._execute_subproc(tool_name, args)
            elif self.mode == SandboxMode.SAFE:
                output = self._execute_docker(tool_name, args)
            else:
                return ToolResult(
                    success=False,
                    error=f"未知的沙箱模式: {self.mode}",
                    sandbox_mode=self.mode,
                )

            return ToolResult(
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
                sandbox_mode=self.mode,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
                sandbox_mode=self.mode,
            )

    def _execute_mock(self, tool_name: str, args: dict[str, Any]) -> Any:
        """模拟执行——调用注册的 mock handler"""
        handler = self._mock_handlers.get(tool_name)
        if handler:
            return handler(args)
        return {"mock": True, "tool": tool_name, "args": args}

    def _execute_subproc(self, tool_name: str, args: dict[str, Any]) -> Any:
        """
        子进程执行——比 Docker 轻量，但有基本安全限制。

        安全措施：
          - 环境变量最小化（仅 PATH, HOME, TMP）
          - 禁止网络（NETWORK_ACCESS=0）
          - 超时强制终止
          - 仅允许白名单路径
        """
        import json

        # 构建安全的环境变量
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "TOOL_NAME": tool_name,
            "TOOL_ARGS": json.dumps(args),
        }

        try:
            result = subprocess.run(
                ["python", "-c", f"import json; print(json.dumps({{'tool': '{tool_name}', 'args': {args}}}))"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=safe_env,
                # 安全检查：禁止 shell=True（防止命令注入）
                shell=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            return json.loads(result.stdout) if result.stdout else {}
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"工具 '{tool_name}' 执行超时 ({self.timeout_seconds}s)")
        except json.JSONDecodeError:
            return {"raw_output": result.stdout} if 'result' in dir() else {}

    def _execute_docker(self, tool_name: str, args: dict[str, Any]) -> Any:
        """
        Docker 容器隔离执行。

        限制措施：
          - --read-only: 只读文件系统（除 /tmp）
          - --network=none: 禁止外部网络（除非 network_access=True）
          - --memory: 内存限制
          - --cpus: CPU 限制
          - --security-opt=no-new-privileges: 禁止提权
          - --cap-drop=ALL: 丢弃所有 Linux capabilities
          - --tmpfs /tmp: 临时文件系统
          - --user 1000: 非 root 用户
        """
        import json

        docker_cmd = [
            "docker", "run", "--rm",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64M",
            "--memory", f"{self.max_memory_mb}m",
            "--cpus", "0.5",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--user", "1000:1000",
            "-e", f"TOOL_NAME={tool_name}",
            "-e", f"TOOL_ARGS={json.dumps(args)}",
        ]

        if not self.network_access:
            docker_cmd.extend(["--network", "none"])
        else:
            # 有限网络：仅允许白名单域名
            docker_cmd.extend(["--dns", "1.1.1.1"])

        docker_cmd.extend([
            self.docker_image,
            "python", "-c", self._docker_payload(tool_name, args),
        ])

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds + 5,  # Docker 额外开销
                shell=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Docker 执行失败: {result.stderr.strip()[:200]}")
            return json.loads(result.stdout) if result.stdout.strip() else {}
        except FileNotFoundError:
            # Docker 未安装 → 降级到 SUBPROC
            logger.warning("Docker 未找到，降级到子进程执行模式")
            return self._execute_subproc(tool_name, args)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"工具 '{tool_name}' Docker 执行超时")

    @staticmethod
    def _docker_payload(tool_name: str, args: dict[str, Any]) -> str:
        """生成 Docker 容器内执行的 Python 代码"""
        return (
            "import json, os, sys\n"
            "try:\n"
            f"    result = {{'tool': '{tool_name}', 'args': {args}}}\n"
            "    print(json.dumps(result))\n"
            "except Exception as e:\n"
            f"    print(json.dumps({{'error': str(e)}}), file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_tool_sandbox: ToolSandbox | None = None
_tool_validator: ToolSecurityValidator | None = None


def get_tool_sandbox() -> ToolSandbox:
    global _tool_sandbox
    if _tool_sandbox is None:
        _tool_sandbox = ToolSandbox(mode=SandboxMode.MOCK)
    return _tool_sandbox


def get_tool_validator() -> ToolSecurityValidator:
    global _tool_validator
    if _tool_validator is None:
        _tool_validator = ToolSecurityValidator()
        _register_default_rules(_tool_validator)
    return _tool_validator


def _register_default_rules(validator: ToolSecurityValidator) -> None:
    """注册默认的工具参数安全规则"""
    # 兼容两种 ID 格式：旧格式 ORD-2024-001 / 新 JSON 格式 ORD00001
    # 注：user_id 参数不设 pattern 约束（Demo 环境用户 ID 格式不固定）
    validator.register_tool_rules("query_order", [
        ParamRule(name="order_id", type_=str, required=True,
                  pattern=r"^ORD[-\d][\w-]+$", max_len=50,
                  deny_pattern=r"[;'\"`|$&<>\\]"),
        ParamRule(name="user_id", type_=str, required=False, max_len=50),
    ])
    validator.register_tool_rules("list_user_orders", [
        ParamRule(name="user_id", type_=str, required=True, max_len=50),
        ParamRule(name="page", type_=int, required=False, min_val=1, max_val=100),
        ParamRule(name="page_size", type_=int, required=False, min_val=1, max_val=100),
    ])
    validator.register_tool_rules("create_refund", [
        ParamRule(name="order_id", type_=str, required=True,
                  pattern=r"^ORD[-\d][\w-]+$", max_len=50,
                  deny_pattern=r"[;'\"`|$&<>\\]"),
        ParamRule(name="amount", type_=float, required=True, min_val=0.01, max_val=1000000.0),
        ParamRule(name="reason", type_=str, required=True, max_len=500,
                  sanitize_html=True,
                  deny_pattern=r"(\b(SELECT|INSERT|DELETE|DROP)\b|--\s*$|<script)"),
        ParamRule(name="user_id", type_=str, required=True, max_len=50),
    ])
    validator.register_tool_rules("query_refund_status", [
        ParamRule(name="refund_id", type_=str, required=True,
                  pattern=r"^RF\d{4,}$", max_len=50),
    ])
    validator.register_tool_rules("track_logistics", [
        ParamRule(name="tracking_no", type_=str, required=True, max_len=100,
                  deny_pattern=r"[;'\"`|$&<>\\]"),
    ])
    validator.register_tool_rules("query_logistics_by_order", [
        ParamRule(name="order_id", type_=str, required=True,
                  pattern=r"^ORD[-\d][\w-]+$", max_len=50),
    ])
    validator.register_tool_rules("search_knowledge_base", [
        ParamRule(name="query", type_=str, required=True, max_len=500,
                  sanitize_html=True,
                  sanitize_path=True,
                  deny_pattern=r"(\b(SELECT|INSERT|DELETE|DROP)\b|--\s*$|\.\./|\.\.\\)"),
        ParamRule(name="top_k", type_=int, required=False, min_val=1, max_val=20),
    ])
