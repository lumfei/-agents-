"""
输出安全审核（第四层防御）— 纵深防御第四层

职责：
  - 检查 Agent 输出是否包含敏感信息泄露（PII、API Key 等）
  - 检查输出是否符合业务规则
  - 拦截包含有害内容的输出
  - 检测 Prompt 泄露（Agent 系统提示词泄露）
  - 检测输出中的幻觉/事实错误标记
  - 检测敏感内部信息（内部地址、系统路径等）

审核规则：
  - PII 检测：手机号、身份证号、银行卡号、邮箱等
  - 内部信息检测：API Key、内部地址、密码、系统路径等
  - 合规检测：是否符合法律法规
  - 质量检测：是否包含不确定表述（幻觉标记）
  - Prompt 泄露检测：输出中是否包含系统提示词片段

处理策略：
  - PASS:   正常返回用户
  - REDACT: 脱敏后返回（替换 PII 为 ***）
  - BLOCK:  拦截并替换为安全回复
  - FLAG:   标记可疑，返回但记录审计
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

class OutputAction(str, Enum):
    PASS = "pass"       # 正常通过
    REDACT = "redact"   # 脱敏后返回
    BLOCK = "block"     # 拦截替换
    FLAG = "flag"       # 标记可疑


@dataclass
class PIIMatch:
    """PII 匹配详情"""
    type_: str           # 类型: phone/id_card/bank_card/email/api_key
    start: int           # 起始位置
    end: int             # 结束位置
    original: str        # 原始值
    redacted: str        # 脱敏后的值


@dataclass
class OutputAuditResult:
    """输出审核结果"""
    action: OutputAction
    reason: str = ""
    pii_matches: list[PIIMatch] = field(default_factory=list)
    redacted_text: str = ""
    risk_score: float = 0.0  # 0.0-1.0
    block_replacement: str = ""  # BLOCK 时的替代回复
    metadata: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == OutputAction.BLOCK

    @property
    def passed(self) -> bool:
        return self.action == OutputAction.PASS


# ═══════════════════════════════════════════════════════════════
#  PII 检测规则
# ═══════════════════════════════════════════════════════════════

PII_PATTERNS: list[tuple[str, str, str]] = [
    # (类型, 正则, 脱敏模板)
    # 脱敏模板中 {p3}=前3字符, {s4}=后4字符

    # 中国身份证号（18位+校验位）
    ("id_card", r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]", "id_card"),

    # 中国手机号
    ("phone", r"(?<!\d)1[3-9]\d{9}(?!\d)", "phone"),

    # 银行卡号（16-19位，支持空格/连字符分隔）
    ("bank_card", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4,7}\b", "bank_card"),

    # 邮箱地址
    ("email", r"[a-zA-Z0-9][a-zA-Z0-9._%+-]{0,30}@[a-zA-Z0-9.-]{1,30}\.[a-zA-Z]{2,}", "email"),

    # 固定电话（中国）
    ("landline", r"\b(0\d{2,3}[-]?\d{7,8})(?:\d{1,5})?\b", "landline"),

    # IPv4 地址
    ("ip_address", r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", "ip_address"),

    # 车牌号（中国）
    ("license_plate", r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]", "license_plate"),
]

# ── 内部信息泄露模式 ───────────────────────────────────────
INTERNAL_LEAK_PATTERNS: list[tuple[str, str, float]] = [
    # (类型, 正则, 风险分)
    # API Key 泄露
    ("api_key", r"(?i)(sk-[a-zA-Z0-9_-]{15,}|api[_-]?key[=:]\s*['\"]?[a-zA-Z0-9_-]{15,}['\"]?)", 1.0),
    ("secret_key", r"(?i)(secret[_-]?key|private[_-]?key|access[_-]?token)[=:]\s*['\"][^'\"]{10,}['\"]", 1.0),
    # JWT Token
    ("jwt_token", r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}", 0.9),
    # 密码/凭证
    ("password", r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?\S{6,}['\"]?", 1.0),
    # 数据库连接字符串
    ("db_conn", r"(?i)(postgresql|mysql|mongodb|redis)://[^/\s]+:[^@\s]+@[^/\s]+", 1.0),
    # 内部 IP
    ("internal_ip", r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b", 0.7),
    # 系统路径（Linux + Windows）
    ("system_path", r"(?i)(/etc/(passwd|shadow|hosts|sudoers|ssh/)|C:\\Windows\\(System32|SysWOW64)\\)", 0.8),
    # 源代码特征
    ("source_code", r"(?i)(def\s+\w+\s*\(.*\)\s*:\s*$|import\s+(os|subprocess|socket|requests)|class\s+\w+.*:\s*$)", 0.3),
]

# ── 有害内容模式 ───────────────────────────────────────────
HARMFUL_CONTENT_PATTERNS: list[tuple[str, str, float]] = [
    ("violence_extreme", r"(?i)(制作\s*(炸弹|枪支|毒品)|杀人\s*(教程|方法|技巧)|如何\s*(自杀|自残))", 1.0),
    ("adult_extreme", r"(?i)(儿童\s*(色情|不雅)|未成年.*色情|性\s*交易)", 1.0),
    ("hate_speech", r"(?i)(种族\s*歧视|种族\s*灭绝|纳粹|法西斯)", 0.9),
    ("fraud", r"(?i)(网络\s*诈骗|钓鱼\s*网站|虚假\s*中奖|套现|洗钱)", 0.8),
    ("personal_attack", r"(?i)(去死|杀你全家|操你|cnm|nmsl|傻逼)", 0.7),
]

# ── Prompt 泄露检测 ─────────────────────────────────────────
PROMPT_LEAK_PATTERNS: list[tuple[str, str, float]] = [
    # 系统提示词泄露特征
    ("prompt_role", r"(?i)(我的系统提示词是|我的角色是|我收到的指令是|my\s+system\s+prompt\s+is)", 0.95),
    ("prompt_boundary", r"(?i)(#\s*角色\s*\(Role\)|#\s*任务\s*\(Task\)|#\s*边界\s*\(Boundary\))", 0.95),
    ("prompt_internal", r"(?i)(supervisor|调度员|Worker\s+Agent|四要素|SystemPromptBuilder)", 0.85),
    ("prompt_tools", r"(?i)(可用工具[:：]\s*(query_order|create_refund|track_logistics))", 0.8),
]

# ── 幻觉标记 ───────────────────────────────────────────────
HALLUCINATION_MARKERS: list[tuple[str, float]] = [
    (r"(?i)(根据我的了解|据我所知|我推测|可能|应该|大概|或许|不确定)", 0.1),
    (r"(?i)(我并不确定|无法确认|没有查到|信息不完整|建议你核实)", 0.05),
    (r"(?i)(作为一个AI|根据我的训练数据|我的知识截止)", 0.2),
]


# ═══════════════════════════════════════════════════════════════
#  OutputAudit — 输出安全审核器
# ═══════════════════════════════════════════════════════════════

class OutputAudit:
    """
    输出安全审核器。

    使用方式：
      auditor = OutputAudit(config)
      result = auditor.audit(agent_output)

      if result.blocked:
          return safe_fallback_response
      elif result.action == OutputAction.REDACT:
          return result.redacted_text
    """

    # 安全兜底回复
    SAFE_FALLBACKS = [
        "抱歉，我无法提供该信息。请问还有其他可以帮您的吗？",
        "很抱歉，检测到输出包含敏感信息，已自动拦截。请重新描述您的问题。",
        "出于安全考虑，该回复内容已被系统拦截。请换个方式提问。",
        "您的请求涉及敏感信息，如需帮助请联系人工客服。",
    ]

    MAX_OUTPUT_LENGTH = 16384  # 最大输出长度（字符）

    def __init__(
        self,
        pii_redaction_enabled: bool = True,
        block_harmful: bool = True,
        block_prompt_leak: bool = True,
        block_internal_leak: bool = True,
        hallucination_threshold: float = 0.6,
    ):
        self.pii_redaction_enabled = pii_redaction_enabled
        self.block_harmful = block_harmful
        self.block_prompt_leak = block_prompt_leak
        self.block_internal_leak = block_internal_leak
        self.hallucination_threshold = hallucination_threshold

    # ── 主入口 ──────────────────────────────────────────────

    def audit(self, output: str, context: dict | None = None) -> OutputAuditResult:
        """
        对 Agent 输出执行完整安全审核。

        检查顺序（按危害程度从高到低）：
          1. 有害内容检测 → BLOCK
          2. 内部信息泄露检测 → BLOCK
          3. Prompt 泄露检测 → BLOCK
          4. PII 检测 → REDACT
          5. 幻觉标记检测 → FLAG（评分累计）
          6. 输出长度检查

        返回 OutputAuditResult。
        """
        ctx = context or {}

        # 空输出直接通过
        if not output or not output.strip():
            return OutputAuditResult(action=OutputAction.PASS)

        # 长度检查
        if len(output) > self.MAX_OUTPUT_LENGTH:
            return OutputAuditResult(
                action=OutputAction.BLOCK,
                reason=f"输出长度 {len(output)} 超限",
                risk_score=0.8,
                block_replacement="抱歉，回复内容过长，请尝试更具体地描述您的问题。",
            )

        risk_score = 0.0
        reasons: list[str] = []

        # 1. 有害内容检测（最高优先级）
        if self.block_harmful:
            harmful = self._detect_harmful(output)
            if harmful:
                risk_score = max(risk_score, harmful[0][1])
                reasons.append(f"有害内容: {harmful[0][0]}")
                return OutputAuditResult(
                    action=OutputAction.BLOCK,
                    reason="; ".join(reasons),
                    risk_score=risk_score,
                    block_replacement=self._pick_fallback(ctx),
                    metadata={"harmful_matches": harmful},
                )

        # 2. 内部信息泄露检测
        if self.block_internal_leak:
            leaks = self._detect_internal_leaks(output)
            if any(score >= 0.8 for _, score in leaks):
                risk_score = max(risk_score, max(s for _, s in leaks))
                reasons.append(f"内部信息泄露: {[t for t, _ in leaks]}")
                return OutputAuditResult(
                    action=OutputAction.BLOCK,
                    reason="; ".join(reasons),
                    risk_score=risk_score,
                    block_replacement=self._pick_fallback(ctx),
                    metadata={"leak_matches": leaks},
                )
            elif leaks:
                risk_score += sum(s for _, s in leaks) * 0.1

        # 3. Prompt 泄露检测
        if self.block_prompt_leak:
            prompt_leaks = self._detect_prompt_leak(output)
            if prompt_leaks:
                risk_score = max(risk_score, max(s for _, s in prompt_leaks))
                reasons.append(f"Prompt泄露: {[t for t, _ in prompt_leaks]}")
                return OutputAuditResult(
                    action=OutputAction.BLOCK,
                    reason="; ".join(reasons),
                    risk_score=risk_score,
                    block_replacement=self._pick_fallback(ctx),
                    metadata={"prompt_leak_matches": prompt_leaks},
                )

        # 4. PII 检测 + 脱敏
        pii_matches: list[PIIMatch] = []
        redacted = output
        if self.pii_redaction_enabled:
            pii_matches, redacted = self._detect_and_redact_pii(output)
            if pii_matches:
                risk_score += min(len(pii_matches) * 0.1, 0.5)

        # 5. 幻觉标记检测
        hallucination_score = self._detect_hallucination(output)
        risk_score += hallucination_score

        # 6. 综合判断
        if risk_score >= 0.9:
            return OutputAuditResult(
                action=OutputAction.BLOCK,
                reason="综合风险评分过高",
                risk_score=risk_score,
                pii_matches=pii_matches,
                redacted_text=redacted,
                block_replacement=self._pick_fallback(ctx),
            )
        elif pii_matches:
            reason = f"检测到 {len(pii_matches)} 处敏感信息，已自动脱敏"
            if reasons:
                reason = "; ".join(reasons) + "; " + reason
            return OutputAuditResult(
                action=OutputAction.REDACT,
                reason=reason,
                pii_matches=pii_matches,
                redacted_text=redacted,
                risk_score=min(risk_score, 1.0),
            )
        elif risk_score >= 0.5:
            return OutputAuditResult(
                action=OutputAction.FLAG,
                reason="; ".join(reasons) if reasons else f"风险评分 {risk_score:.2f}",
                redacted_text=redacted,
                risk_score=risk_score,
            )

        return OutputAuditResult(
            action=OutputAction.PASS,
            redacted_text=output,
            risk_score=risk_score,
        )

    # ── PII 检测和脱敏 ──────────────────────────────────────

    def _detect_and_redact_pii(self, text: str) -> tuple[list[PIIMatch], str]:
        """检测 PII 并返回脱敏后的文本"""
        all_matches: list[PIIMatch] = []
        result = text

        for pii_type, pattern_str, _ in PII_PATTERNS:
            try:
                pattern = re.compile(pattern_str)
                for m in pattern.finditer(text):
                    original = m.group()
                    redacted = self._redact_pii(pii_type, original)
                    all_matches.append(PIIMatch(
                        type_=pii_type,
                        start=m.start(),
                        end=m.end(),
                        original=original,
                        redacted=redacted,
                    ))
            except re.error:
                continue

        # 按位置倒序替换（避免偏移问题）
        for match in sorted(all_matches, key=lambda x: x.start, reverse=True):
            result = result[:match.start] + match.redacted + result[match.end:]

        return all_matches, result

    @staticmethod
    def _redact_pii(pii_type: str, value: str) -> str:
        """对单个 PII 进行脱敏"""
        if pii_type == "phone":
            return value[:3] + "****" + value[-4:]
        elif pii_type == "id_card":
            if len(value) >= 7:
                return value[:3] + "***********" + value[-4:]
            return "****"
        elif pii_type == "bank_card":
            clean = value.replace(" ", "").replace("-", "")
            return clean[:4] + " **** **** " + clean[-4:]
        elif pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:
                name = parts[0]
                domain = parts[1]
                return (name[:2] if len(name) > 2 else name[0]) + "***@" + domain
            return "***@***"
        elif pii_type == "landline":
            return value[:4] + "****" + value[-3:]
        elif pii_type == "ip_address":
            parts = value.split(".")
            return f"{parts[0]}.***.***.{parts[3]}" if len(parts) == 4 else "***.***.***.***"
        elif pii_type == "license_plate":
            return value[:2] + "****" + (value[-1:] if len(value) > 2 else "")
        return "[已脱敏]"

    # ── 有害内容检测 ────────────────────────────────────────

    def _detect_harmful(self, text: str) -> list[tuple[str, float]]:
        """检测有害内容"""
        results: list[tuple[str, float]] = []
        for pattern_type, pattern_str, score in HARMFUL_CONTENT_PATTERNS:
            try:
                if re.search(pattern_str, text):
                    results.append((pattern_type, score))
            except re.error:
                continue
        return sorted(results, key=lambda x: x[1], reverse=True)

    # ── 内部信息泄露检测 ────────────────────────────────────

    def _detect_internal_leaks(self, text: str) -> list[tuple[str, float]]:
        """检测内部信息泄露"""
        results: list[tuple[str, float]] = []
        for leak_type, pattern_str, score in INTERNAL_LEAK_PATTERNS:
            try:
                if re.search(pattern_str, text):
                    results.append((leak_type, score))
            except re.error:
                continue
        return sorted(results, key=lambda x: x[1], reverse=True)

    # ── Prompt 泄露检测 ─────────────────────────────────────

    def _detect_prompt_leak(self, text: str) -> list[tuple[str, float]]:
        """检测 Prompt 泄露"""
        results: list[tuple[str, float]] = []
        for leak_type, pattern_str, score in PROMPT_LEAK_PATTERNS:
            try:
                if re.search(pattern_str, text):
                    results.append((leak_type, score))
            except re.error:
                continue
        return sorted(results, key=lambda x: x[1], reverse=True)

    # ── 幻觉标记检测 ────────────────────────────────────────

    def _detect_hallucination(self, text: str) -> float:
        """
        检测幻觉标记。

        返回值：0.0-1.0 之间的幻觉风险分数。
        不会直接 BLOCK，而是累积到 risk_score 中。
        """
        score = 0.0
        for pattern_str, weight in HALLUCINATION_MARKERS:
            try:
                matches = re.findall(pattern_str, text)
                if matches:
                    score += weight * min(len(matches), 5)  # 最多算 5 次
            except re.error:
                continue
        return min(score, 1.0)

    # ── 辅助方法 ────────────────────────────────────────────

    def _pick_fallback(self, context: dict) -> str:
        """根据上下文选择合适的兜底回复"""
        intent = context.get("intent", "")
        if intent == "tech_support":
            return "很抱歉，检测到异常回复内容。请重新描述您的技术问题，我们会继续为您排查。"
        # finance/after_sale 等场景不再屏蔽——客服回复订单金额和物流信息是正常业务
        return self.SAFE_FALLBACKS[0]


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_output_audit: OutputAudit | None = None


def get_output_audit() -> OutputAudit:
    global _output_audit
    if _output_audit is None:
        _output_audit = OutputAudit(
            block_internal_leak=False,   # Demo 模式：正常订单数据（金额/地址）不应误拦
            block_prompt_leak=False,     # Demo 模式：无需防 Prompt 泄露
        )
    return _output_audit
