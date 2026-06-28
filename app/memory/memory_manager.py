"""
记忆管理器 — 三类记忆的统一调度中枢

这个文件是整个记忆系统的"大脑"：
  - 短期记忆：管"这次对话说到哪了"
  - 长期记忆：管"这个用户是谁"
  - 工作记忆：管"当前任务做到哪了"

记忆管理器的职责：
  1. 统一接口：调用方只需要说"存这段对话"或"取相关上下文"，
     具体存到哪、怎么取，管理器内部决定
  2. 记忆检索策略：不是把所有记忆统统塞进上下文，
     而是根据当前对话内容做语义检索，只注入最相关的
  3. 记忆流动协调：
     - 每轮对话 → 存入短期记忆
     - 关键信息（用户偏好、决策）→ 同步到长期记忆
     - 当前任务状态 → 存入工作记忆
     - 会话结束时 → 短期记忆的关键内容归档到长期记忆

记忆检索策略（重要）：
  不是把所有记忆一股脑塞进 context——这既浪费 token 又稀释有效信息。
  策略：
    1. 短期记忆：总是注入最近的 N 轮（滑动窗口）
    2. 长期记忆：根据当前消息做语义检索，只取最相关的 top_k 条
    3. 工作记忆：总是注入当前任务的状态摘要
  三种记忆以不同的"角色"出现在 prompt 中：
    - 短期记忆 → 对话历史 messages
    - 长期记忆 → system message 中的用户画像
    - 工作记忆 → system message 中的任务状态

使用方式：
  mm = MemoryManager()
  session = mm.create_session("USR-001", "SESS-001")

  # 每轮对话后
  mm.store_interaction("SESS-001", "USR-001", "我的电脑蓝屏了", "这是解决方法...")

  # 需要上下文时
  context = mm.retrieve_context("SESS-001", "USR-001",
                                current_message="我的电脑蓝屏了")
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.memory.short_term import ShortTermMemory, ConversationSummarizer
from app.memory.long_term import (
    LongTermMemory,
    MemoryCategory,
    SearchResult,
)
from app.memory.working_memory import WorkingMemoryManager
from app.graph.context_engine import ContextEngine


# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

# 短期记忆滑动窗口大小
DEFAULT_WINDOW_SIZE = 10

# 长期记忆检索返回的最大条数
DEFAULT_MEMORY_TOP_K = 5

# 会话超时秒数
SESSION_TTL = 1800


# ═══════════════════════════════════════════════════════════════
#  记忆管理器
# ═══════════════════════════════════════════════════════════════

class MemoryManager:
    """
    记忆管理器——三类记忆的调度中枢。

    提供高层 API，调用方不需要关心：
      - 数据存在哪（短期 vs 长期 vs 工作）
      - 什么时候摘要（滑动窗口逻辑）
      - 什么时候归档（会话结束时的数据迁移）
      - 怎么检索（关键词匹配 vs 向量检索）

    使用时：
      1. 用户发起对话 → create_session()
      2. 每轮对话结束 → store_interaction()
      3. 需要 LLM 上下文 → retrieve_context()
      4. 需要创建任务 → 通过 working_memory 创建
      5. 会话结束 → archive_session()
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        memory_top_k: int = DEFAULT_MEMORY_TOP_K,
    ):
        # 三类记忆实例
        self.short_term: dict[str, ShortTermMemory] = {}  # session_id → STM
        self.long_term = LongTermMemory()
        self.working_memory = WorkingMemoryManager()

        # 配置
        self.window_size = window_size
        self.memory_top_k = memory_top_k

        # 统计
        self.total_sessions = 0

    # ══════════════════════════════════════════════════════════
    #  会话管理
    # ══════════════════════════════════════════════════════════

    def create_session(self, session_id: str) -> ShortTermMemory:
        """
        创建一个新的会话（初始化短期记忆）。

        参数：
          session_id: 会话唯一标识

        返回：
          新建的 ShortTermMemory 实例
        """
        stm = ShortTermMemory(
            window_size=self.window_size,
            session_ttl=SESSION_TTL,
        )
        self.short_term[session_id] = stm
        self.total_sessions += 1
        return stm

    def get_or_create_session(self, session_id: str) -> ShortTermMemory:
        """
        获取已有会话，不存在则创建。
        """
        session = self.short_term.get(session_id)
        if session is None:
            session = self.create_session(session_id)
        return session

    def get_session(self, session_id: str) -> Optional[ShortTermMemory]:
        """获取会话的短期记忆"""
        return self.short_term.get(session_id)

    # ══════════════════════════════════════════════════════════
    #  交互记录
    # ══════════════════════════════════════════════════════════

    def store_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_id: str = "",
        tool_calls: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
        task_id: Optional[str] = None,
    ):
        """
        存储一轮完整的对话交互。

        这个方法同时做了三件事：
          1. 把对话存入短期记忆（滑动窗口管理）
          2. 提取关键信息存入长期记忆（偏好、决策等）
          3. 更新工作记忆的任务进度

        参数：
          session_id:       会话 ID
          user_message:     用户的消息
          assistant_message: AI 的回复
          user_id:          用户 ID（可选，用于长期记忆）
          tool_calls:        本轮调用的工具列表（可选）
          tool_results:      工具返回的结果（可选）
          task_id:           关联的任务 ID（可选，用于工作记忆）
        """
        # ── 存入短期记忆 ──────────────────────────────────────
        stm = self.get_or_create_session(session_id)
        stm.add_round(user_message)
        stm.complete_round(assistant_message, tool_calls, tool_results)

        # ── 提取关键信息并存入长期记忆 ──────────────────────────
        if user_id:
            self._extract_and_store_facts(
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
                tool_calls=tool_calls,
            )

        # ── 更新工作记忆 ──────────────────────────────────────
        if task_id:
            task = self.working_memory.get_task(task_id)
            if task:
                action_desc = f"对话交互: {user_message[:30]}..."
                task.add_step(
                    agent_name=task.current_agent or "unknown",
                    action=action_desc,
                    result=assistant_message[:50],
                )

    def _extract_and_store_facts(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        tool_calls: Optional[list[dict]] = None,
    ):
        """
        从对话中提取关键信息，存入长期记忆。

        这是"记忆压缩"的核心——从自由对话中提炼出可存储的事实。
        当前用简单的规则提取，后续可以用 LLM 做更精准的信息提取。

        提取策略：
          - 用户消息中的"我..." → 偏好或关键事实
          - 工具调用 → 操作记录（如申请退款）
          - AI 确认的决策 → 决策记录
        """
        # 提取用户偏好（简单的关键词规则）
        pref_keywords = {
            "喜欢": "preference_like",
            "不喜欢": "preference_dislike",
            "习惯": "preference_habit",
            "希望": "preference_wish",
        }
        for kw, key in pref_keywords.items():
            if kw in user_message:
                self.long_term.store(
                    user_id=user_id,
                    category=MemoryCategory.PREFERENCE,
                    content=f"用户表示: {user_message[:60]}",
                    key=key,
                    weight=0.7,  # 偏好权重要适中，可能随时间变化
                )

        # 提取关键事实（涉及订单号、地址、联系方式等）
        fact_indicators = ["订单", "地址", "电话", "邮箱", "账号", "退款"]
        has_fact = any(ind in user_message for ind in fact_indicators)
        if has_fact:
            self.long_term.store(
                user_id=user_id,
                category=MemoryCategory.KEY_FACT,
                content=f"会话关键信息: {user_message[:100]}",
                key="conversation_fact",
                weight=0.8,
            )

        # 提取决策（AI 确认了某件事）
        decision_keywords = ["好的", "已为您", "已经", "正在为您"]
        has_decision = any(kw in assistant_message for kw in decision_keywords)
        if has_decision:
            self.long_term.store(
                user_id=user_id,
                category=MemoryCategory.DECISION,
                content=f"决策: {assistant_message[:80]}",
                key=f"decision_{int(time.time())}",
                weight=0.6,
            )

        # 记录工具调用（操作记录）
        if tool_calls:
            tool_names = [t.get("name", "?") for t in tool_calls]
            self.long_term.store(
                user_id=user_id,
                category=MemoryCategory.KEY_FACT,
                content=f"用户发起了操作: {', '.join(tool_names)}",
                key=f"tool_usage_{int(time.time())}",
                weight=0.5,
            )

    # ══════════════════════════════════════════════════════════
    #  上下文检索（核心方法）
    # ══════════════════════════════════════════════════════════

    def retrieve_context(
        self,
        session_id: str,
        user_id: str = "",
        current_message: str = "",
    ) -> dict[str, Any]:
        """
        检索所有相关记忆，组装成 LLM 可用的上下文。

        这是整个记忆管理器最重要的方法。
        它决定把哪些记忆注入到 LLM 的 prompt 中。

        返回结构：
        {
          "short_term": [  ← 对话历史，作为 messages
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
          ],
          "long_term": "..."  ← 用户画像，作为 system prompt 的一部分
          "working_memory": "..."  ← 任务状态，作为 system prompt 的一部分
        }

        检索策略（不是把所有记忆都塞进去）：
          1. 短期记忆：最近的 N 轮（滑动窗口控制）
          2. 长期记忆：根据 current_message 做语义检索，只取 top_k 条
          3. 工作记忆：当前进行中的任务状态摘要
        """
        context: dict[str, Any] = {
            "short_term": [],
            "long_term": "",
            "working_memory": "",
        }

        # ── 1. 短期记忆：获取对话历史 ─────────────────────────
        stm = self.get_session(session_id)
        if stm:
            context["short_term"] = stm.get_window_context()

        # ── 2. 长期记忆：根据当前消息检索用户画像 ─────────────
        if user_id:
            # 根据当前消息做语义检索
            search_results = self.long_term.search(
                user_id=user_id,
                query=current_message,
                top_k=self.memory_top_k,
                min_weight=0.1,
            )

            if search_results:
                # 把检索到的记忆组装成文字摘要
                memory_lines = ["【用户历史信息】"]
                for sr in search_results:
                    mem = sr.memory
                    memory_lines.append(
                        f"- [{mem.category.value}] {mem.content}"
                    )
                context["long_term"] = "\n".join(memory_lines)

        # ── 3. 工作记忆：当前任务状态 ─────────────────────────
        active_tasks = self.working_memory.get_active_tasks()
        if active_tasks:
            task_summaries = []
            for task in active_tasks:
                task_summaries.append(task.summary())
            context["working_memory"] = "\n".join(task_summaries)

        return context

    # ── 分层上下文组装（上下文工程集成） ───────────────────

    def build_layered_context(
        self,
        session_id: str,
        system_prompt: str,
        user_id: str = "",
        current_message: str = "",
        knowledge_text: str = "",
        recent_tool_calls: Optional[list[dict]] = None,
        context_budget: int = 16384,
    ) -> list[dict]:
        """
        从所有记忆源构建分层的、受预算控制的 LLM 上下文。

        这是"记忆管理器 × 上下文工程引擎"的集成方法。
        流程：
          1. 从短期记忆取对话历史
          2. 从长期记忆检索用户画像
          3. 从工作记忆取任务状态
          4. 全部喂给 ContextEngine 进行分层 + 预算截断
          5. 返回可直接用于 LLM.invoke() 的 messages

        参数：
          session_id:       会话 ID
          system_prompt:    系统提示词（Agent 角色和边界）
          user_id:          用户 ID（可选，用于长期记忆检索）
          current_message:  当前用户消息（用于语义检索）
          knowledge_text:   知识库检索结果（可选）
          recent_tool_calls: 最近工具调用记录（可选）
          context_budget:   上下文预算 Token 数（默认 16384）

        返回：
          [{"role": "...", "content": "..."}, ...] 格式的 messages
          可以直接传给 LLM.invoke()
        """
        # ── 1. 获取原始记忆数据 ─────────────────────────────
        raw = self.retrieve_context(
            session_id=session_id,
            user_id=user_id,
            current_message=current_message,
        )

        # ── 2. 合并额外上下文（工作记忆 + 长期记忆摘要） ──
        extra_parts = []
        if raw.get("working_memory"):
            extra_parts.append(raw["working_memory"])
        if raw.get("long_term"):
            extra_parts.append(raw["long_term"])

        # ── 3. 通过 ContextEngine 分层组装 ─────────────────
        engine = ContextEngine(total_budget=context_budget)

        messages = engine.build_and_assemble(
            system_prompt=system_prompt,
            extra_context="\n".join(extra_parts) if extra_parts else "",
            long_term_context="",  # 已合入 extra_context
            knowledge_text=knowledge_text,
            conversation_messages=raw.get("short_term", []),
            recent_tool_calls=recent_tool_calls,
        )

        return messages

    # ══════════════════════════════════════════════════════════
    #  工作记忆快捷操作
    # ══════════════════════════════════════════════════════════

    def create_task(
        self,
        original_query: str,
        user_id: str = "",
        session_id: str = "",
    ) -> Any:
        """创建工作记忆任务"""
        return self.working_memory.create_task(
            original_query=original_query,
            user_id=user_id,
            session_id=session_id,
        )

    def complete_task(self, task_id: str) -> Optional[dict]:
        """完成任务并返回最终状态"""
        return self.working_memory.complete_task(task_id)

    def set_current_agent(self, task_id: str, agent_name: str):
        """设置当前处理任务的 Agent"""
        self.working_memory.update_task(
            task_id, current_agent=agent_name
        )

    # ══════════════════════════════════════════════════════════
    #  会话结束
    # ══════════════════════════════════════════════════════════

    def archive_session(self, session_id: str, user_id: str = ""):
        """
        会话结束时归档记忆。

        做的事：
          1. 把短期记忆中的重要信息归档到长期记忆
          2. 清理短期记忆
          3. 清理该会话关联的已完成工作记忆

        什么信息值得归档到长期记忆？
          - 用户明确表达的偏好
          - 决策记录（已确认的方案）
          - 关键事实（订单号、地址变更等）
          - 对话摘要（当前短期记忆的摘要状态）
        """
        stm = self.short_term.pop(session_id, None)
        if stm is None:
            return

        if not user_id:
            return

        # 归档短期记忆的摘要（如果有）
        if stm.has_summary() or stm.round_count > 0:
            summary_text = f"会话 {session_id} 摘要"
            if stm.has_summary():
                summary_text += f": {stm._summary}"
            else:
                rounds = stm.get_recent_rounds(3)
                topics = [r.user_message[:30] for r in rounds if r.user_message]
                summary_text += f" 涉及话题: {'; '.join(topics)}"

            self.long_term.store(
                user_id=user_id,
                category=MemoryCategory.CONVERSATION,
                content=summary_text,
                key=f"session_{session_id}",
                weight=0.4,  # 会话摘要的权重适中
            )

        # 清理该用户已完成的工作记忆任务
        for task in self.working_memory.get_active_tasks(user_id=user_id):
            if task.session_id == session_id:
                self.working_memory.complete_task(task.task_id)

    # ══════════════════════════════════════════════════════════
    #  维护
    # ══════════════════════════════════════════════════════════

    def clear_session(self, session_id: str):
        """清理会话数据"""
        self.short_term.pop(session_id, None)

    def clear_all(self):
        """清理所有记忆（调试/测试用）"""
        self.short_term.clear()
        self.long_term = LongTermMemory()
        self.working_memory.clear()

    def stats(self) -> dict:
        """统计各类记忆的数量"""
        return {
            "active_sessions": len(self.short_term),
            "total_sessions_created": self.total_sessions,
            "long_term_items": self.long_term.count(),
            "long_term_users": self.long_term.count_users(),
            "active_tasks": self.working_memory.active_count,
        }

    def __repr__(self) -> str:
        return (
            f"MemoryManager(sessions={len(self.short_term)}, "
            f"long_term={self.long_term.count()}, "
            f"tasks={self.working_memory.active_count})"
        )
