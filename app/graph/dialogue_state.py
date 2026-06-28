"""
对话状态管理 — Slot-Filling 状态追踪 + 上下文压缩 + 指代消解

这个文件解决三个问题：

1. Slot-Filling 状态追踪（核心）
   对话历史是"发生了什么"的日志，对话状态是"当前处于什么阶段、还需要什么信息"的结构化表示。
   每个工单类型（技术支持/财务/售后）需要收集不同的信息槽位：
     - 技术支持：问题类型、设备型号、错误代码、已尝试步骤
     - 财务：订单号、退款金额、退款原因
     - 售后：运单号、退换货原因
   显式的 Slot-Filling 比靠 LLM 从对话历史中推断更省 Token、更可靠。

2. 上下文压缩
   用 LLM 将长对话压缩为精炼摘要。不同于 ConversationSummarizer 的规则摘要，
   这里用 LLM 做更高质量的语义摘要，保留决策关键路径，丢弃过程细节。

3. 指代消解（轻量级）
   处理"它"、"那个"、"这个"指代——提取上一轮提到的实体作为当前轮的上下文。

使用方式：
  dsm = DialogueStateManager()
  dsm.set_intent("tech_support")
  dsm.fill_slot("issue_type", "电脑蓝屏")
  dsm.fill_slot("error_code", "0x0000001a")
  state_dict = dsm.get_state()  # 当前状态摘要
  missing = dsm.get_missing_slots()  # 还缺什么信息
  entity_context = dsm.resolve_reference("它还是不行")  # "它"→"电脑"
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
#  槽位定义
# ═══════════════════════════════════════════════════════════════

class IntentType(str, Enum):
    """意图类型——即工单分类"""
    TECH_SUPPORT = "tech_support"
    FINANCE = "finance"
    AFTER_SALE = "after_sale"
    UNKNOWN = "unknown"


class SlotDef:
    """
    槽位定义——描述一个"需要收集的信息"。

    字段说明：
      name:        槽位名称（如 "order_id"）
      label:       显示名（如 "订单号"）
      description: 给 LLM 看的描述（如 "用户的订单编号，格式 ORD-xxx"）
      required:    是否必填
      examples:    示例值（用于引导 LLM 识别）
    """

    def __init__(
        self,
        name: str,
        label: str,
        description: str = "",
        required: bool = True,
        examples: Optional[list[str]] = None,
    ):
        self.name = name
        self.label = label
        self.description = description or f"{label}"
        self.required = required
        self.examples = examples or []

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "required": self.required,
        }


# ═══════════════════════════════════════════════════════════════
#  各意图类型的槽位模板
# ═══════════════════════════════════════════════════════════════

INTENT_SLOTS: dict[str, list[SlotDef]] = {
    IntentType.TECH_SUPPORT: [
        SlotDef("issue_type", "问题类型", "用户遇到的具体问题描述，如蓝屏、死机、无法联网"),
        SlotDef("device_model", "设备型号", "用户的设备型号，如 ThinkPad X1 Carbon"),
        SlotDef("error_code", "错误代码", "系统提示的错误代码，如 0x0000001a"),
        SlotDef("steps_tried", "已尝试步骤", "用户已经尝试过的解决方法"),
    ],
    IntentType.FINANCE: [
        SlotDef("order_id", "订单号", "用户的订单编号，格式 ORD-xxx", required=True),
        SlotDef("amount", "金额", "退款或涉及金额，单位元"),
        SlotDef("reason", "原因", "退款或财务操作的原因"),
        SlotDef("payment_method", "支付方式", "用户使用的支付方式，支付宝/微信/银行卡"),
    ],
    IntentType.AFTER_SALE: [
        SlotDef("order_id", "订单号", "用户的订单编号，格式 ORD-xxx", required=True),
        SlotDef("tracking_no", "运单号", "快递运单号"),
        SlotDef("return_reason", "退换货原因", "用户退换货的原因"),
        SlotDef("complaint_type", "投诉类型", "投诉的类型，如物流慢/商品破损/服务差"),
    ],
    IntentType.UNKNOWN: [],
}


# ═══════════════════════════════════════════════════════════════
#  槽位状态
# ═══════════════════════════════════════════════════════════════

class SlotValue:
    """
    一个槽位的值状态。

    状态流转：
      empty → filled（用户提供了）→ confirmed（用户确认了）
    """

    def __init__(self, slot_def: SlotDef):
        self.defn = slot_def
        self.value: Any = None
        self.filled: bool = False
        self.confirmed: bool = False
        self.source: str = ""  # "user_explicit", "user_implicit", "inferred", "confirmed"

    def fill(self, value: Any, source: str = "user_explicit"):
        """填入值"""
        self.value = value
        self.filled = True
        self.source = source

    def confirm(self):
        """确认值"""
        self.confirmed = True

    @property
    def is_missing(self) -> bool:
        """是否还需要收集"""
        return self.defn.required and not self.filled

    @property
    def summary(self) -> str:
        """槽位摘要"""
        if not self.filled:
            return f"{self.defn.label}:（待收集）"
        status = "" if self.confirmed else "（待确认）"
        return f"{self.defn.label}: {self.value}{status}"

    def __repr__(self) -> str:
        return f"Slot({self.defn.name}, filled={self.filled}, value={self.value})"


# ═══════════════════════════════════════════════════════════════
#  对话阶段
# ═══════════════════════════════════════════════════════════════

class DialogueStage(str, Enum):
    """对话阶段"""
    INITIAL = "initial"                    # 刚建立连接
    INTENT_CLASSIFYING = "intent_classifying"  # 正在判断意图
    COLLECTING_INFO = "collecting_info"    # 正在收集信息
    PROCESSING = "processing"              # 正在处理
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # 等待用户确认
    COMPLETED = "completed"               # 已解决
    ESCALATED = "escalated"               # 已升级


# ═══════════════════════════════════════════════════════════════
#  对话状态管理器
# ═══════════════════════════════════════════════════════════════

class DialogueStateManager:
    """
    对话状态管理器。

    职责：
      1. 跟踪当前对话阶段和已收集的槽位
      2. 判断还缺什么信息
      3. 生成状态摘要（给 LLM 注入用）
      4. 轻量级指代消解
      5. 上下文压缩（用 LLM 生成高质量摘要）

    使用方式：
      dsm = DialogueStateManager()
      dsm.set_intent("tech_support")
      dsm.fill_slot("issue_type", "电脑蓝屏了")
      dsm.fill_slot("error_code", "0x0000001a")
      print(dsm.get_missing_slots())  # ['device_model', 'steps_tried']
      print(dsm.get_state_summary())
    """

    def __init__(self, intent_type: str = "unknown"):
        # 当前意图
        self.intent_type = intent_type

        # 当前阶段
        self.stage = DialogueStage.INITIAL

        # 槽位列表：按当前意图的模板初始化
        self.slots: dict[str, SlotValue] = {}
        self._init_slots()

        # 实体缓存（用于指代消解）：{实体名: 值}
        # 比如 {"电脑": "联想 ThinkPad", "订单": "ORD-001"}
        self._entity_cache: dict[str, str] = {}

        # 决策记录（用于压缩）：[{"decision": "用户同意退款", "time": ...}]
        self._decisions: list[dict] = []

        # 话题切换标记
        self._topic_switches: int = 0

        # 本轮最新消息（用于指代消解）
        self._last_user_message: str = ""
        self._last_assistant_message: str = ""

    def _init_slots(self):
        """按当前意图初始化槽位"""
        self.slots.clear()
        slot_defs = INTENT_SLOTS.get(self.intent_type, [])
        for sd in slot_defs:
            self.slots[sd.name] = SlotValue(sd)

    # ── 意图和阶段 ──────────────────────────────────────────

    def set_intent(self, intent_type: str):
        """
        设置意图类型，重置槽位。
        调用时机：Supervisor Agent 完成意图分类后。
        """
        if intent_type != self.intent_type:
            self.intent_type = intent_type
            self._init_slots()
            self.stage = DialogueStage.COLLECTING_INFO

    def set_stage(self, stage: DialogueStage):
        """设置当前阶段"""
        self.stage = stage

    # ── 槽位填充 ────────────────────────────────────────────

    def fill_slot(
        self,
        slot_name: str,
        value: Any,
        source: str = "user_explicit",
    ):
        """
        填充一个槽位。

        参数：
          slot_name: 槽位名称，如 "order_id"
          value:     槽位的值
          source:    来源（user_explicit=用户明确说 / inferred=推断 / confirmed=确认）
        """
        sv = self.slots.get(slot_name)
        if sv is None:
            return
        sv.fill(value, source)
        # 同步更新实体缓存
        self._entity_cache[slot_name] = str(value)[:60]

    def batch_fill(self, entities: dict[str, Any]):
        """
        批量填充槽位（来自意图分类的 extracted_entities）。

        参数：
          entities: {"order_id": "ORD-001", "amount": 299}
        """
        for key, value in entities.items():
            self.fill_slot(key, value, "inferred")

    def confirm_slot(self, slot_name: str):
        """用户确认了某个槽位的值"""
        sv = self.slots.get(slot_name)
        if sv and sv.filled:
            sv.confirm()

    def get_missing_slots(self) -> list[SlotDef]:
        """获取所有尚未填写的必填槽位"""
        return [sv.defn for sv in self.slots.values() if sv.is_missing]

    @property
    def all_required_filled(self) -> bool:
        """所有必填槽位是否都已填写"""
        return len(self.get_missing_slots()) == 0

    @property
    def fill_rate(self) -> float:
        """已填写比例"""
        required = [sv for sv in self.slots.values() if sv.defn.required]
        if not required:
            return 1.0
        filled = sum(1 for sv in required if sv.filled)
        return filled / len(required)

    # ── 状态摘要（给 LLM 注入用） ───────────────────────────

    def get_state_summary(self) -> str:
        """
        生成结构化的对话状态摘要。

        这个文本会被注入 System Prompt 的 Memory 层。
        格式：XML 标签包裹，方便 LLM 快速定位。

        返回示例：
          <dialogue_state>
          <intent>tech_support</intent>
          <stage>collecting_info</stage>
          <slots>
            <slot name="issue_type" filled="true">电脑蓝屏</slot>
            <slot name="device_model" filled="false"/>
            <slot name="error_code" filled="false"/>
          </slots>
          <missing>device_model,error_code</missing>
          </dialogue_state>
        """
        parts = ["<dialogue_state>"]

        # 意图和阶段
        parts.append(f"  <intent>{self.intent_type}</intent>")
        parts.append(f"  <stage>{self.stage.value}</stage>")

        # 槽位
        parts.append("  <slots>")
        for sv in self.slots.values():
            if sv.filled:
                value_escaped = str(sv.value).replace("<", "&lt;").replace(">", "&gt;")
                parts.append(f'    <slot name="{sv.defn.name}" filled="true">{value_escaped}</slot>')
            else:
                parts.append(f'    <slot name="{sv.defn.name}" filled="false"/>')
        parts.append("  </slots>")

        # 缺失槽位
        missing = self.get_missing_slots()
        if missing:
            names = ",".join(s.name for s in missing)
            parts.append(f"  <missing>{names}</missing>")

        # 决策记录
        if self._decisions:
            parts.append("  <decisions>")
            for d in self._decisions[-5:]:  # 保留最近 5 条决策
                parts.append(f"    <decision>{d.get('text', '')}</decision>")
            parts.append("  </decisions>")

        parts.append("</dialogue_state>")
        return "\n".join(parts)

    def get_compact_summary(self) -> str:
        """
        紧凑型摘要（给 Token 紧张时用）。

        返回示例：
          [状态] tech_support | 收集信息中 | issue_type=电脑蓝屏 | 待收集: device_model,error_code
        """
        slots_text = " | ".join(
            sv.summary for sv in self.slots.values()
        )
        return (
            f"[状态] {self.intent_type} | {self.stage.value} | {slots_text}"
        )

    # ── 决策记录 ───────────────────────────────────────────

    def record_decision(self, text: str):
        """记录一个决策"""
        self._decisions.append({
            "text": text,
            "time": time.time(),
        })

    def get_decisions_summary(self) -> str:
        """获取决策摘要"""
        if not self._decisions:
            return ""
        parts = ["【历史决策】"]
        for d in self._decisions[-5:]:
            parts.append(f"- {d['text']}")
        return "\n".join(parts)

    # ── 轻量级指代消解 ────────────────────────────────────

    def update_messages(self, user_message: str, assistant_message: str):
        """
        更新最新消息（用于指代消解）。

        每次对话后调用，缓存最近一轮的用户和 AI 消息。
        """
        self._last_user_message = user_message
        self._last_assistant_message = assistant_message

        # 从用户消息中提取实体（简单的关键词抽取）
        self._extract_entities(user_message)

    def _extract_entities(self, text: str):
        """从文本中提取实体关键词，加入实体缓存"""
        # 常见的客服实体类型
        entity_patterns = {
            "订单": ["订单", "ORD-", "order"],
            "电脑": ["电脑", "笔记本", "台式机", "服务器"],
            "手机": ["手机", "iPhone", "Android", "华为", "小米"],
            "快递": ["快递", "包裹", "运单号", "SF", "顺丰"],
            "退款": ["退款", "退钱", "返还"],
        }

        for entity_type, keywords in entity_patterns.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    # 提取包含该关键词的上下文
                    idx = text.lower().find(kw.lower())
                    start = max(0, idx - 5)
                    end = min(len(text), idx + len(kw) + 20)
                    context = text[start:end].strip()
                    self._entity_cache[entity_type] = self._entity_cache.get(
                        entity_type, context
                    )
                    break

    def resolve_reference(self, user_message: str) -> str:
        """
        解析指代关系。

        如果用户消息中包含"它"、"这个"、"那个"、"这里"等代词，
        尝试从实体缓存中找到最可能的指代对象。

        参数：
          user_message: 用户当前消息

        返回：
          指代解析后的增强提示文本
        """
        reference_words = ["它", "这个", "那个", "这里", "那里", "他", "她", "它们", "这些", "那些"]
        has_reference = any(rw in user_message for rw in reference_words)

        if not has_reference or not self._entity_cache:
            return ""

        # 没有指代词或没有实体缓存，返回空
        if not has_reference:
            return ""

        # 构建指代上下文
        entities_text = "、".join(
            f"{k}({v})" for k, v in self._entity_cache.items()
        )
        return f"【指代上下文】当前已知实体: {entities_text} | 用户消息中的代词可能指代上述实体。"

    def get_entity_context(self) -> str:
        """获取实体缓存摘要"""
        if not self._entity_cache:
            return ""
        entities = "、".join(
            f"{k}={v[:30]}" for k, v in self._entity_cache.items()
        )
        return f"【已知实体】{entities}"

    # ── LLM 上下文压缩 ─────────────────────────────────────

    def generate_compression_prompt(self, conversation_log: str) -> str:
        """
        生成"用 LLM 压缩对话"的提示词。

        这个方法不是直接用 LLM 压缩，而是返回一个提示词模板，
        调用方可以用 LLM 生成高质量摘要。

        参数：
          conversation_log: 原始对话日志文本

        返回：
          发给 LLM 的压缩提示词
        """
        from app.prompts.registry import get_prompt_registry
        template = get_prompt_registry().get_compression_prompt("dialogue_compression")
        return template.format(conversation_log=conversation_log)

    def compress_with_llm(self, conversation_log: str, llm) -> str:
        """
        用 LLM 压缩对话。

        参数：
          conversation_log: 原始对话日志文本
          llm:              LLM 实例（如 ChatLiteLLM）

        返回：
          LLM 生成的压缩摘要
        """
        prompt = self.generate_compression_prompt(conversation_log)
        response = llm.invoke(prompt)
        return response.content.strip()

    # ── 序列化 ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "intent_type": self.intent_type,
            "stage": self.stage.value,
            "slots": {
                name: {
                    "value": sv.value,
                    "filled": sv.filled,
                    "confirmed": sv.confirmed,
                }
                for name, sv in self.slots.items()
            },
            "missing_slots": [s.name for s in self.get_missing_slots()],
            "fill_rate": self.fill_rate,
            "decisions": self._decisions,
        }

    def __repr__(self) -> str:
        return (
            f"DialogueState(intent={self.intent_type}, "
            f"stage={self.stage.value}, "
            f"fill_rate={self.fill_rate:.0%})"
        )
