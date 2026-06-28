"""
记忆系统 — 三类记忆统一管理

导入这个包后，可以直接使用：
  from app.memory import MemoryManager

内存管理器会自动创建短期/长期/工作记忆实例，
你也可以单独使用某个记忆类型。

快速开始：
  # 创建管理器
  mm = MemoryManager()

  # 创建会话
  session = mm.create_session("SESS-001")

  # 每轮对话存储
  mm.store_interaction(
      session_id="SESS-001",
      user_message="我的电脑蓝屏了",
      assistant_message="我来帮您查一下...",
      user_id="USR-001",
  )

  # 获取 LLM 上下文
  ctx = mm.retrieve_context(
      session_id="SESS-001",
      user_id="USR-001",
      current_message="我的电脑蓝屏了",
  )
"""

# 短期记忆
from app.memory.short_term import (
    ShortTermMemory,
    ConversationRound,
    MessageEntry,
    ConversationSummarizer,
)

# 长期记忆
from app.memory.long_term import (
    LongTermMemory,
    MemoryItem,
    MemoryCategory,
    SearchResult,
)

# 工作记忆
from app.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryManager,
)

# 记忆管理器
from app.memory.memory_manager import MemoryManager

__all__ = [
    # 管理器
    "MemoryManager",
    # 短期记忆
    "ShortTermMemory",
    "ConversationRound",
    "MessageEntry",
    "ConversationSummarizer",
    # 长期记忆
    "LongTermMemory",
    "MemoryItem",
    "MemoryCategory",
    "SearchResult",
    # 工作记忆
    "WorkingMemory",
    "WorkingMemoryManager",
]
