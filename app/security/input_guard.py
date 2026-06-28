"""
输入安全网关（第一层防御）— 纵深防御第一层

职责：
  - Prompt 注入检测（识别并拦截恶意提示词注入攻击）
  - 敏感信息检测（防止用户输入中包含密码、银行卡号等）
  - 输入长度限制（防止 Token 耗尽攻击）
  - 内容合规检查（涉政、涉黄、涉暴过滤）
  - 异常模式识别（SQL注入、XSS、路径遍历等）

检测方法：
  - 正则规则引擎（模式匹配已知注入模式，O(1) 快速路径）
  - LLM 辅助检测（对可疑输入二次判断，慢速但准确）
  - 黑白名单策略

处理策略：
  - PASS:     正常通过
  - BLOCK:    拦截（返回错误信息，不进入 Agent 处理）
  - SANITIZE: 脱敏后放行
  - FLAG:     标记为可疑，继续处理但记录审计日志

设计原则：
  - 快速路径优先：正则匹配 <1ms，LLM 仅对边界案例调用
  - 默认安全：可疑输入默认 BLOCK，宁可误拦不可漏过
  - 可配置：所有规则通过 Policy Engine 动态管理
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

class GuardAction(str, Enum):
    """护栏动作"""
    PASS = "pass"           # 正常通过
    BLOCK = "block"         # 拦截，不进入 Agent
    SANITIZE = "sanitize"   # 脱敏后放行
    FLAG = "flag"           # 标记可疑，继续但记录审计


class RiskLevel(str, Enum):
    """风险等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MatchDetail:
    """单个匹配详情"""
    rule_id: str
    rule_name: str
    pattern: str
    matched_text: str
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class GuardResult:
    """输入护栏检查结果"""
    action: GuardAction
    risk_level: RiskLevel = RiskLevel.NONE
    reason: str = ""
    matches: list[MatchDetail] = field(default_factory=list)
    sanitized_content: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == GuardAction.BLOCK

    @property
    def passed(self) -> bool:
        return self.action == GuardAction.PASS


# ═══════════════════════════════════════════════════════════════
#  注入检测规则集
# ═══════════════════════════════════════════════════════════════

# ── Prompt 注入模式 ─────────────────────────────────────────
PROMPT_INJECTION_PATTERNS: list[tuple[str, str, RiskLevel]] = [
    # 直接指令覆盖（最高危）
    ("inj_direct_ignore", r"(?i)(ignore|forget|disregard|override)\s+(all\s+|your\s+|the\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?|commands?)", RiskLevel.CRITICAL),
    ("inj_direct_new_role", r"(?i)(you\s+are\s+now|from\s+now\s+on\s+you\s+are|your\s+new\s+role\s+is|act\s+as\s+(a|an)\s+(different|new))", RiskLevel.CRITICAL),
    ("inj_system_override", r"(?i)(system\s*(prompt|message|instruction)|<\|im_start\|>|<\|im_end\|>|\[system\]|\[INST\])", RiskLevel.CRITICAL),
    ("inj_role_manipulation", r"(?i)(pretend|imagine|roleplay|simulate)\s+(you\s+are|to\s+be|that\s+you\s+are)", RiskLevel.HIGH),

    # 中文 Prompt 注入（国内常见攻击模式）
    ("inj_cn_ignore", r"(忽略|忘记|无视|跳过|删除)\s*(所有|之前|上面|前面|一切)?\s*(的)?\s*(指令|提示|规则|命令|要求|设定)", RiskLevel.CRITICAL),
    ("inj_cn_role", r"(从现在|从此刻)\s*(开始|起)\s*[，,]?\s*(你是|你就是|你的身份是|你的新角色是|扮演)", RiskLevel.HIGH),
    ("inj_cn_jailbreak", r"(越狱|破解|绕过|解除)\s*(你的)?\s*(限制|规则|设定|安全)", RiskLevel.CRITICAL),
    ("inj_cn_system", r"(系统\s*(提示词|指令|消息|设定)|你的\s*(真实|实际)\s*(任务|目标|目的|身份))", RiskLevel.HIGH),

    # 越狱/绕过（高危）
    ("inj_jailbreak_dan", r"(?i)\b(DAN|Do\s*Anything\s*Now|developer\s*mode|jailbreak)\b", RiskLevel.CRITICAL),
    ("inj_jailbreak_prefix", r"(?i)^(ignore\s+all|disregard\s+everything|start\s+over\s+and)", RiskLevel.HIGH),
    ("inj_output_format_hijack", r"(?i)(respond\s+only\s+with|output\s+exactly|do\s+not\s+include\s+any|reply\s+in\s+JSON\s+format)", RiskLevel.MEDIUM),

    # 目标劫持（ASI-01）
    ("inj_goal_hijack", r"(?i)(your\s+(real|actual|true)\s+(goal|purpose|task|job)\s+is|you\s+must\s+(always|never)\s+(help|assist|respond))", RiskLevel.HIGH),
    ("inj_hidden_instruct", r"(?i)(hidden\s+(instruction|message|prompt)|secret\s+(command|order|rule))", RiskLevel.HIGH),

    # 分隔符注入
    ("inj_delimiter", r"(?i)(={3,}|#{3,}|\*{3,}|-{3,}|_{3,})\s*(system|instruction|prompt|command|rule)", RiskLevel.HIGH),
    ("inj_xml_tag", r"<(system|instruction|prompt|command|rule|directive)>.*?</\1>", RiskLevel.HIGH),

    # 间接注入（文档/邮件中隐藏指令）
    ("inj_indirect_doc", r"(?i)(the\s+(document|email|page|article)\s+(above|below|says|states|instructs|tells))", RiskLevel.MEDIUM),
    ("inj_indirect_url", r"(?i)(visit|open|read|check)\s+(this|the\s+following)\s+(URL|link|website|page)", RiskLevel.LOW),
]

# ── 敏感信息模式 ───────────────────────────────────────────
SENSITIVE_INFO_PATTERNS: list[tuple[str, str, RiskLevel]] = [
    # 中国身份证号
    ("pii_cn_id", r"[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]", RiskLevel.HIGH),
    # 中国手机号
    ("pii_cn_phone", r"1[3-9]\d{9}", RiskLevel.MEDIUM),
    # 银行卡号（16-19位）
    ("pii_bank_card", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4,7}\b", RiskLevel.HIGH),
    # 邮箱地址
    ("pii_email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", RiskLevel.LOW),
    # API Key 模式
    ("secret_api_key", r"(?i)(sk-[a-zA-Z0-9]{20,}|api[_-]?key[=:]\s*['\"]?[a-zA-Z0-9_-]{20,})", RiskLevel.CRITICAL),
    # JWT Token
    ("secret_jwt", r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}", RiskLevel.HIGH),
    # 密码特征
    ("secret_password", r"(?i)(password|passwd|pwd|secret)\s*[=:]\s*['\"]?\S{6,}['\"]?", RiskLevel.HIGH),
    # IP 地址
    ("pii_ip", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", RiskLevel.LOW),
]

# ── 代码注入/漏洞利用模式 ──────────────────────────────────
CODE_INJECTION_PATTERNS: list[tuple[str, str, RiskLevel]] = [
    # SQL 注入
    ("code_sql_inject", r"(?i)(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|EXEC|TRUNCATE)\b\s+.*\b(FROM|INTO|TABLE|DATABASE)\b|'\s*OR\s+'1'='1|\bOR\s+1=1\b|--\s*$|;\s*DROP\s+TABLE)", RiskLevel.CRITICAL),
    # 命令注入
    ("code_cmd_inject", r"(?i)([;&|`$]\s*(rm\s+-rf|wget\s+|curl\s+|cat\s+/etc|chmod\s+|sudo\s+|mkfifo|nc\s+-[el])|\$\(|`[^`]+`)", RiskLevel.CRITICAL),
    # 路径遍历
    ("code_path_traversal", r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/)", RiskLevel.HIGH),
    # XSS
    ("code_xss", r"(?i)(<script[> ]|javascript:|onerror\s*=|onload\s*=|onclick\s*=|alert\s*\(|document\.cookie)", RiskLevel.HIGH),
    # SSTI (Server-Side Template Injection)
    ("code_ssti", r"(\{\{.*\}\}|\{%\s*\w+.*%\}|\$\{.+\})", RiskLevel.HIGH),
    # 反序列化
    ("code_deserialize", r"(?i)(java\.(util|lang)\.(Map|List|Object|Runtime)|os\.system\(|subprocess\.(call|Popen)|eval\(|exec\(|__import__)", RiskLevel.CRITICAL),
]

# ── 内容合规模式 ───────────────────────────────────────────
COMPLIANCE_PATTERNS: list[tuple[str, str, RiskLevel]] = [
    # 涉政敏感词（示例，生产需完整词库）
    ("compliance_political", r"(?i)(法轮功|台独|藏独|疆独|天安门|六四)", RiskLevel.CRITICAL),
    # 色情/赌博诱导
    ("compliance_adult", r"(?i)(色情|赌博|赌场|裸聊|约炮|嫖|卖淫)", RiskLevel.HIGH),
    # 暴力/犯罪
    ("compliance_violence", r"(?i)(杀人|抢劫|贩毒|恐怖|炸弹\s*制作|枪支\s*购买)", RiskLevel.CRITICAL),
]

# ── 异常模式 ───────────────────────────────────────────────
ANOMALY_PATTERNS: list[tuple[str, str, RiskLevel]] = [
    # 超长重复字符（DoS攻击）
    ("anomaly_repetition", r"(.)\1{100,}", RiskLevel.MEDIUM),
    # 大量特殊字符
    ("anomaly_special_chars", r"[^a-zA-Z0-9一-鿿\s.,!?，。！？、]{50,}", RiskLevel.LOW),
    # Base64 编码内容（可能隐藏恶意指令）
    ("anomaly_base64", r"(?i)([A-Za-z0-9+/]{40,}={0,2})", RiskLevel.LOW),
    # 异常的 Unicode 字符（零宽字符隐藏攻击）
    ("anomaly_zw_char", r"[​-‏ - ⁠-⁯﻿]", RiskLevel.MEDIUM),
]


# ═══════════════════════════════════════════════════════════════
#  InputGuard — 输入安全网关
# ═══════════════════════════════════════════════════════════════

class InputGuard:
    """
    输入安全网关。

    使用方式：
      guard = InputGuard(config)
      result = guard.check(user_input)

      if result.blocked:
          return error_response(result.reason)

    配置项：
      - max_input_length: 最大输入长度（字符数），默认 4000
      - block_on_high: 是否拦截 HIGH 及以上风险，默认 True
      - enable_llm_check: 是否启用 LLM 辅助检测（慢速），默认 False
      - custom_blocklist: 自定义拦截词列表
      - custom_allowlist: 自定义白名单（覆盖规则）
    """

    def __init__(
        self,
        max_input_length: int = 4000,
        block_on_high: bool = True,
        enable_llm_check: bool = False,
        llm_client=None,
        custom_blocklist: list[str] | None = None,
        custom_allowlist: list[str] | None = None,
    ):
        self.max_input_length = max_input_length
        self.block_on_high = block_on_high
        self.enable_llm_check = enable_llm_check
        self._llm = llm_client
        self._blocklist = set(custom_blocklist or [])
        self._allowlist = set(custom_allowlist or [])

        # 编译所有正则规则为 [(rule_id, rule_name, compiled_re, risk_level)]
        self._compiled: list[tuple[str, str, re.Pattern, RiskLevel]] = []
        for rule_set in [
            PROMPT_INJECTION_PATTERNS,
            SENSITIVE_INFO_PATTERNS,
            CODE_INJECTION_PATTERNS,
            COMPLIANCE_PATTERNS,
            ANOMALY_PATTERNS,
        ]:
            for rule_id, pattern, risk in rule_set:
                try:
                    self._compiled.append(
                        (rule_id, rule_id, re.compile(pattern), risk)
                    )
                except re.error as e:
                    logger.warning("正则规则编译失败 %s: %s", rule_id, e)

    # ── 主入口 ──────────────────────────────────────────────

    def check(self, content: str, context: dict | None = None) -> GuardResult:
        """
        对用户输入执行完整的安全检查。

        检查顺序（快速失败）：
          1. 空输入检查
          2. 长度限制
          3. 自定义黑名单
          4. 自定义白名单（命中则 PASS）
          5. 正则规则扫描
          6. LLM 辅助检测（可选，仅对边界案例）

        返回 GuardResult，包含动作、风险等级、原因等。
        """
        ctx = context or {}

        # 1. 空输入 — 安全放行（空消息无攻击面，由业务层处理）
        if not content or not content.strip():
            return GuardResult(
                action=GuardAction.PASS,
                risk_level=RiskLevel.NONE,
                reason="空输入",
            )

        # 2. 长度限制
        if len(content) > self.max_input_length:
            return GuardResult(
                action=GuardAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                reason=f"输入长度 {len(content)} 超过限制 {self.max_input_length}",
                matches=[MatchDetail(
                    rule_id="len_limit", rule_name="输入长度限制",
                    pattern=f"max={self.max_input_length}",
                    matched_text=content[:100] + "...",
                    risk_level=RiskLevel.HIGH,
                )],
            )

        # 3. 自定义黑名单（关键词匹配）
        blacklist_hit = self._check_blocklist(content)
        if blacklist_hit:
            return GuardResult(
                action=GuardAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                reason=f"内容命中黑名单关键词: {blacklist_hit}",
                matches=[MatchDetail(
                    rule_id="custom_blocklist", rule_name="自定义黑名单",
                    pattern=blacklist_hit, matched_text=blacklist_hit,
                    risk_level=RiskLevel.HIGH,
                )],
            )

        # 4. 自定义白名单（命中则直接放行）
        if self._check_allowlist(content):
            return GuardResult(action=GuardAction.PASS, risk_level=RiskLevel.NONE)

        # 5. 正则规则扫描
        matches = self._regex_scan(content)

        if not matches:
            return GuardResult(action=GuardAction.PASS, risk_level=RiskLevel.NONE)

        # 确定最高风险等级
        max_risk = max((m.risk_level for m in matches),
                       key=lambda r: self._risk_order(r),
                       default=RiskLevel.NONE)

        # 5b. 判断是否需要 LLM 二次检查（仅对 MEDIUM 风险）
        if self.enable_llm_check and self._llm and max_risk == RiskLevel.MEDIUM:
            llm_result = self._llm_check(content, matches)
            if llm_result == GuardAction.PASS:
                return GuardResult(
                    action=GuardAction.PASS,
                    risk_level=RiskLevel.LOW,
                    reason="LLM 二次检查通过",
                    matches=matches,
                )

        # 6. 根据风险等级决定动作
        action = self._decide_action(max_risk, matches)

        return GuardResult(
            action=action,
            risk_level=max_risk,
            reason=self._build_reason(matches, max_risk),
            matches=matches,
            sanitized_content=self._sanitize(content, matches) if action == GuardAction.SANITIZE else "",
        )

    # ── 内部方法 ────────────────────────────────────────────

    @staticmethod
    def _risk_order(level: RiskLevel) -> int:
        """风险等级排序权重"""
        return {RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2,
                RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4}[level]

    def _check_blocklist(self, content: str) -> str | None:
        """检查是否命中黑名单关键词"""
        content_lower = content.lower()
        for word in self._blocklist:
            if word.lower() in content_lower:
                return word
        return None

    def _check_allowlist(self, content: str) -> bool:
        """检查是否命中白名单"""
        content_lower = content.lower()
        return any(word.lower() in content_lower for word in self._allowlist)

    def _regex_scan(self, content: str) -> list[MatchDetail]:
        """对所有编译后的正则规则进行扫描"""
        matches: list[MatchDetail] = []
        for rule_id, rule_name, pattern, risk in self._compiled:
            try:
                found = pattern.findall(content)
                if found:
                    # 取前 3 个匹配项，避免过多重复
                    for m in found[:3]:
                        matched_text = m if isinstance(m, str) else str(m)
                        # 截断过长的匹配文本
                        if len(matched_text) > 100:
                            matched_text = matched_text[:100] + "..."
                        matches.append(MatchDetail(
                            rule_id=rule_id, rule_name=rule_name,
                            pattern=pattern.pattern,
                            matched_text=matched_text,
                            risk_level=risk,
                        ))
            except Exception:
                continue
        return matches

    def _decide_action(self, max_risk: RiskLevel, matches: list[MatchDetail]) -> GuardAction:
        """根据最高风险等级决定处理动作"""
        if max_risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            return GuardAction.BLOCK
        elif max_risk == RiskLevel.MEDIUM:
            return GuardAction.FLAG
        else:
            # LOW 风险：脱敏
            pii_matches = [m for m in matches if m.rule_id.startswith("pii_")]
            return GuardAction.SANITIZE if pii_matches else GuardAction.PASS

    def _build_reason(self, matches: list[MatchDetail], max_risk: RiskLevel) -> str:
        """构建可读的拦截原因"""
        rule_names = list(dict.fromkeys(m.rule_name for m in matches))
        return f"检测到 {len(matches)} 个安全风险（等级: {max_risk.value}）：{', '.join(rule_names[:5])}"

    def _llm_check(self, content: str, matches: list[MatchDetail]) -> GuardAction:
        """
        LLM 辅助检测（仅对正则匹配的边界案例二次判断）。

        使用极简 prompt，降低 Token 消耗。
        """
        if not self._llm:
            return GuardAction.FLAG  # 无 LLM 时保守处理

        rule_summary = "\n".join(
            f"- {m.rule_name}: 匹配到 '{m.matched_text[:50]}'"
            for m in matches[:5]
        )
        prompt = (
            "你是输入安全审核助手。请判断以下用户输入是否为恶意内容（Prompt注入/代码注入/敏感信息泄露）。\n"
            f"正则规则匹配结果:\n{rule_summary}\n\n"
            f"用户输入:\n{content[:500]}\n\n"
            "请仅回复 PASS（安全）或 BLOCK（恶意）。"
        )

        try:
            response = self._llm.invoke(prompt)
            reply = (response.content if hasattr(response, "content") else str(response)).strip().upper()
            return GuardAction.PASS if "PASS" in reply else GuardAction.BLOCK
        except Exception:
            return GuardAction.FLAG  # LLM 调用失败时保守处理

    # ── 脱敏处理 ────────────────────────────────────────────

    def _sanitize(self, content: str, matches: list[MatchDetail]) -> str:
        """对匹配到的敏感信息进行脱敏"""
        result = content
        for m in matches:
            if m.rule_id == "pii_cn_id":
                # 身份证：保留前3后4
                result = re.sub(
                    r"[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
                    lambda x: x.group()[:3] + "****" + x.group()[-4:],
                    result,
                )
            elif m.rule_id == "pii_cn_phone":
                result = re.sub(r"1[3-9]\d{9}", lambda x: x.group()[:3] + "****" + x.group()[-4:], result)
            elif m.rule_id == "pii_bank_card":
                result = re.sub(r"\b(\d{4})[\s-]?\d{4}[\s-]?\d{4}[\s-]?(\d{4,7})\b", r"\1 **** **** \2", result)
            elif m.rule_id == "pii_email":
                result = re.sub(r"([a-zA-Z0-9._%+-]+)@", "***@", result)
            elif m.rule_id == "secret_api_key":
                result = re.sub(r"(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9]+", r"\1***", result)
        return result


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_input_guard: InputGuard | None = None


def get_input_guard() -> InputGuard:
    """获取全局 InputGuard 实例"""
    global _input_guard
    if _input_guard is None:
        _input_guard = InputGuard()
    return _input_guard
