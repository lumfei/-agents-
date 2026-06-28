"""
短期记忆模块 — 当前会话的对话历史

核心职责：
  存储当前会话的 N 轮对话历史，作为每次 LLM 调用时传入的上下文。

关键概念：
  - 会话（Session）：一次用户从"你好"到"再见"的完整对话链
  - 轮次（Round）：用户发一条 + AI 回一条 = 一轮
  - 滑动窗口：只保留最近 N 轮，超出部分丢弃或摘要

什么时候用短期记忆？
  - 用户说"刚才那个问题我还没说完"——短期记忆知道"刚才"指什么
  - AI 说"根据您之前提到的..."——短期记忆提供了前文

什么时候不用？
  - 跨天的用户偏好（那是长期记忆的事）
  - 当前任务做到哪一步了（那是工作记忆的事）

存储方式：
  当前版本用内存中的字典（进程级，重启丢失）。
  后续版本可以换成 Redis（带 TTL 自动过期，跨进程共享）。

技术要点：
  - 滑动窗口：保留最近 WINDOW_SIZE 轮对话
  - 对话摘要：当历史超过窗口时，用 LLM 把旧历史压缩成摘要
  - 摘要保留关键信息：用户意图、已确认的决策、未解决的问题
  - 摘要丢弃噪声：寒暄、重复提问、已解决的子问题
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
#  配置常量
# ═══════════════════════════════════════════════════════════════

# 默认滑动窗口大小：保留最近 10 轮对话
DEFAULT_WINDOW_SIZE = 10

# 默认会话超时时间（秒）：30 分钟无操作自动过期
DEFAULT_SESSION_TTL = 1800

# 触发摘要的阈值：当历史超过此轮数时触发摘要压缩
SUMMARY_TRIGGER_THRESHOLD = 15

# 摘要保留的最大轮数（摘要本身占 1 轮，保留最近 N 轮）
SUMMARY_RECENT_KEEP = 8


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

class MessageEntry:
    """
    单条消息的存储格式。

    字段说明：
      role:    谁发的（user=用户 / assistant=AI / system=系统 / tool=工具）
      content: 消息内容
      time:    发送时间戳
      metadata: 附加信息（token用量、工具调用等）
    """

    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ):
        self.role = role
        self.content = content
        self.time = time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "time": self.time,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"MessageEntry(role={self.role}, len={len(self.content)}, time={self.time})"


class ConversationRound:
    """
    一轮对话：用户消息 + AI 回复（+ 可选的工具调用和结果）。

    为什么把一轮对话打包成一个对象？
      方便滑动窗口的截断——截断时一次去掉一整轮，
      而不是去掉用户的提问留下 AI 的回答（那样上下文就断了）。
    """

    def __init__(self, user_message: str):
        self.user_message = user_message  # 用户的问题
        self.assistant_message: Optional[str] = None  # AI 的回答
        self.tool_calls: list[dict] = []  # 本轮调用的工具
        self.tool_results: list[dict] = []  # 工具返回的结果
        self.round_time = time.time()  # 本轮开始时间

    def is_complete(self) -> bool:
        """一轮对话是否完成（有用户消息也有 AI 回复）"""
        return self.assistant_message is not None

    def to_dict(self) -> dict:
        return {
            "user": self.user_message,
            "assistant": self.assistant_message,
            "tool_calls": self.tool_calls,
            "round_time": self.round_time,
        }

    def __repr__(self) -> str:
        return (
            f"Round(user={self.user_message[:30] if self.user_message else ''}..., "
            f"assistant={'yes' if self.assistant_message else 'no'})"
        )


# ═══════════════════════════════════════════════════════════════
#  对话摘要器
# ═══════════════════════════════════════════════════════════════

class ConversationSummarizer:
    """
    对话摘要器——把多轮对话压缩成一段摘要。

    什么时候触发？
      当历史对话超过 SUMMARY_TRIGGER_THRESHOLD 轮时，
      对最旧的轮次进行摘要，保留最近的轮次不动。

    摘要保留什么（关键信息）？
      - 用户的真实意图（"他想退款"）
      - 已确认的决策（"同意退款 199 元"）
      - 未解决的问题（"发票还没开"）
      - 关键信息（订单号、用户 ID 等）

    摘要丢弃什么（噪声）？
      - 寒暄（"你好"、"在吗"）
      - 重复提问
      - 已解决的子问题
      - 工具调用的技术细节

    实现方式：
      当前用简单的轮次摘要策略（取最早轮次的用户消息提炼）。
      后续可以用 LLM 生成更高质量的语义摘要。
    """

    @staticmethod
    def summarize_rounds(rounds: list[ConversationRound]) -> str:
        """
        对多轮对话生成摘要。

        策略：
          - 提取所有轮次的用户意图（看用户问了什么）
          - 提取 AI 给出的关键结论
          - 提取未解决的问题
          - 压缩成一段话

        这个方法不用 LLM，用规则提取关键信息。
        好处：零成本、速度快、稳定。
        代价：不如 LLM 摘要"智能"，但够用。
        """
        user_intents: list[str] = []
        key_decisions: list[str] = []
        open_issues: list[str] = []
        key_entities: dict[str, str] = {}

        for rnd in rounds:
            user_msg = rnd.user_message or ""
            assistant_msg = rnd.assistant_message or ""

            # 提取用户意图（取前 50 个字作为意图描述）
            if user_msg and len(user_msg) > 2:
                intent = user_msg[:80]
                if rnd.tool_calls:
                    intent += f" [调用了工具: {', '.join(t.get('name', '?') for t in rnd.tool_calls[:3])}]"
                user_intents.append(intent)

            # 提取 AI 的关键结论（包含"已"、"完成"、"决定"等关键词的句子）
            if assistant_msg:
                # 简单启发式：包含决策关键词的句子
                decision_keywords = ["已", "完成", "决定", "同意", "驳回", "确认", "改为", "设置"]
                for kw in decision_keywords:
                    if kw in assistant_msg:
                        # 取包含关键词的那句
                        for sentence in assistant_msg.split("。"):
                            if kw in sentence and len(sentence) > 5:
                                key_decisions.append(sentence.strip())
                                break

        # ── 构建摘要 ──────────────────────────────────────────
        if not key_decisions:
            summary = f"用户共提出了 {len(rounds)} 个问题，涉及: "
            summary += "; ".join(u[:40] for u in user_intents[:5])
            if len(user_intents) > 5:
                summary += f" 等共 {len(user_intents)} 个问题"
            return summary

        # 有关键决策时，构建结构化摘要
        summary_parts = ["【历史摘要】"]
        if user_intents:
            summary_parts.append(f"用户意图: {'; '.join(u[:50] for u in user_intents[:4])}")
            if len(user_intents) > 4:
                summary_parts.append(f"以及另外 {len(user_intents) - 4} 个问题")
        summary_parts.append(f"关键决策: {'; '.join(key_decisions[:5])}")
        if open_issues:
            summary_parts.append(f"未解决: {'; '.join(open_issues[:3])}")
        return " | ".join(summary_parts)

    @staticmethod
    def llm_summarize(conversation_log: str, llm=None) -> str:
        """
        用 LLM 生成高质量对话摘要。

        和 summarize_rounds() 的区别：
          - summarize_rounds() 用规则提取，零成本但质量一般
          - llm_summarize() 用 LLM 生成，质量高但有 Token 成本

        什么时候用 LLM 版本？
          - 会话结束时归档到长期记忆（一次调用，值得）
          - 超长对话需要压缩时（几百分辨率对话压缩为一段话）
          - 用户主动要求"总结一下刚才说的"

        参数：
          conversation_log: 原始对话文本
          llm:              LLM 实例（如 ChatLiteLLM）。如果为 None，使用规则摘要。

        返回：
          高质量的压缩摘要
        """
        if llm is None:
            # 没有 LLM，降级为规则摘要
            # 创建一个临时 ShortTermMemory 来复用规则摘要
            temp_stm = ShortTermMemory()
            # 直接把文本作为摘要返回（调用方应先用规则摘要兜底）
            return ConversationSummarizer.summarize_rounds_raw(conversation_log)

        from app.prompts.registry import get_prompt_registry
        template = get_prompt_registry().get_compression_prompt("conversation_compression")
        prompt = template.format(conversation_log=conversation_log)

        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"[摘要生成失败，降级为规则摘要] {ConversationSummarizer.summarize_rounds_raw(conversation_log)}"

    @staticmethod
    def summarize_rounds_raw(text: str) -> str:
        """对纯文本做规则摘要（llm_summarize 的降级方案）"""
        if len(text) <= 200:
            return text
        return text[:200] + "..."


# ═══════════════════════════════════════════════════════════════
#  短期记忆
# ═══════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    短期记忆——当前会话的对话历史。

    管理方式：
      以"轮（Round）"为单位存储对话。
      每轮包含：用户说了什么、AI 回了什么、调了什么工具。
      用滑动窗口控制大小，超出时摘要旧轮次。

    使用方式：
      stm = ShortTermMemory(window_size=10)
      stm.add_round("我的电脑蓝屏了")
      # ... 得到 AI 回复后 ...
      stm.complete_round("这是解决方法...")
      context = stm.get_window_context()  # 获取当前上下文
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        session_ttl: int = DEFAULT_SESSION_TTL,
    ):
        """
        初始化短期记忆。

        参数：
          window_size: 滑动窗口大小，保留最近多少轮
          session_ttl: 会话超时秒数（备用，当前版本不强制过期）
        """
        # 对话历史：按轮次存储
        self._rounds: list[ConversationRound] = []

        # 摘要缓存：如果历史被摘要了，存这里
        self._summary: Optional[str] = None

        # 自增计数器：自创建以来总共产生过多少轮对话
        # 这个计数不受窗口截断影响，用于触发摘要判定
        self._total_created: int = 0

        # 配置
        self.window_size = window_size
        self.session_ttl = session_ttl

        # 会话开始时间
        self.created_at = time.time()
        self.last_activity = time.time()

    # ── 核心操作 ──────────────────────────────────────────────

    def add_round(self, user_message: str) -> ConversationRound:
        """
        开始一轮新对话。

        调用时机：用户发了新消息时。
        之后需要调用 complete_round() 把 AI 的回复补上。

        参数：
          user_message: 用户的输入

        返回：
          新创建的 ConversationRound 对象
        """
        self.last_activity = time.time()
        self._total_created += 1
        new_round = ConversationRound(user_message)
        self._rounds.append(new_round)

        # 检查是否触发滑动窗口截断
        self._apply_window()

        return new_round

    def complete_round(
        self,
        assistant_message: str,
        tool_calls: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
    ) -> bool:
        """
        完成当前轮次（补上 AI 的回复）。

        调用时机：LLM 返回了回复时。

        参数：
          assistant_message: AI 的回复内容
          tool_calls: 本轮调用的工具列表（可选）
          tool_results: 工具返回的结果（可选）

        返回：
          True=成功补全, False=当前没有未完成的轮次
        """
        self.last_activity = time.time()

        # 找到最后一轮未完成的
        for rnd in reversed(self._rounds):
            if not rnd.is_complete():
                rnd.assistant_message = assistant_message
                rnd.tool_calls = tool_calls or []
                rnd.tool_results = tool_results or []
                return True

        return False

    # ── 滑动窗口与摘要 ────────────────────────────────────────

    def _apply_window(self):
        """
        检查窗口大小并决定是否需要截断/摘要。

        策略：
          1. 如果历史总轮数（_total_created）> 触发摘要阈值（15 轮）
             且当前有轮次可以摘要
             → 把最早的部分轮次摘要，保留最近的轮次
             使用 _total_created（累计总量）而非 len(self._rounds）（当前存量）
             因为窗口截断会减少当前轮次数，但摘要应该基于总对话量触发
          2. 如果当前轮次数 > 窗口大小但还没到摘要阈值
             → 直接丢弃最早的几轮
        """
        total = len(self._rounds)
        total_ever = self._total_created

        if total_ever > SUMMARY_TRIGGER_THRESHOLD and total >= 2:
            # 超出摘要阈值 → 把旧轮次压缩为摘要
            # 保留最近 SUMMARY_RECENT_KEEP 轮，其余摘要
            keep = min(SUMMARY_RECENT_KEEP, total - 1)
            rounds_to_summarize = total - keep
            old_rounds = self._rounds[:rounds_to_summarize]
            remaining = self._rounds[rounds_to_summarize:]

            # 生成摘要
            summary_text = ConversationSummarizer.summarize_rounds(old_rounds)

            # 把摘要合并到现有的摘要中
            if self._summary:
                self._summary = f"{self._summary} | {summary_text}"
            else:
                self._summary = summary_text

            self._rounds = remaining

        elif total > self.window_size:
            # 超出窗口但还没到摘要阈值 → 直接丢弃最早的
            excess = total - self.window_size
            self._rounds = self._rounds[excess:]

    def get_window_context(self) -> list[dict]:
        """
        获取当前上下文的"消息列表"格式。

        这是最重要的方法——它的输出会直接拼到 LLM 的对话历史中。

        返回格式（符合 LangChain 的 messages 格式）：
          [
            {"role": "system", "content": "你是客服..."},  # 摘要放这里
            {"role": "user", "content": "我的电脑蓝屏了"},
            {"role": "assistant", "content": "好的，我来查一下"},
            {"role": "user", "content": "怎么办"},
            ...
          ]

        注意：
          - 摘要以 system role 放在最前面
          - 最近的对话轮次按实际顺序排列
          - 工具调用不会出现在这里（属于工作记忆的范畴）
        """
        context: list[dict] = []

        # 如果有摘要，放在最前面
        if self._summary:
            context.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{self._summary}",
            })

        # 添加最近的对话轮次
        for rnd in self._rounds:
            context.append({"role": "user", "content": rnd.user_message})
            if rnd.assistant_message:
                context.append({"role": "assistant", "content": rnd.assistant_message})

        return context

    # ── 查询与信息提取 ────────────────────────────────────────

    def get_recent_rounds(self, n: Optional[int] = None) -> list[ConversationRound]:
        """获取最近 N 轮对话"""
        if n is None:
            return list(self._rounds)
        return self._rounds[-n:]

    def get_recent_messages(self, n: int = 5) -> list[dict]:
        """
        获取最近 N 条消息（不是轮次，是单条消息）。

        用于快速查看最近说了什么。
        """
        messages = []
        for rnd in self._rounds[-n:]:
            messages.append({"role": "user", "content": rnd.user_message})
            if rnd.assistant_message:
                messages.append({"role": "assistant", "content": rnd.assistant_message})
        return messages

    def get_all_messages(self) -> list[dict]:
        """获取全部消息（用于持久化或调试）"""
        return self.get_window_context()

    def has_summary(self) -> bool:
        """是否有摘要（历史被压缩过）"""
        return bool(self._summary)

    @property
    def round_count(self) -> int:
        """当前轮次数"""
        return len(self._rounds)

    @property
    def total_rounds_ever(self) -> int:
        """历史总轮次（含被摘要的）"""
        # 简略估算：如果摘要存在，加上摘要对应的轮次
        # 当前无法精确统计，但可以通过额外计数来实现
        return len(self._rounds) + (10 if self._summary else 0)

    # ── 管理操作 ──────────────────────────────────────────────

    def clear(self):
        """清空当前会话的所有记忆"""
        self._rounds.clear()
        self._summary = None
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        """检查会话是否超时"""
        return (time.time() - self.last_activity) > self.session_ttl

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）"""
        return {
            "summary": self._summary,
            "rounds": [r.to_dict() for r in self._rounds],
            "window_size": self.window_size,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(rounds={len(self._rounds)}, "
            f"window={self.window_size}, "
            f"has_summary={self._summary is not None})"
        )
