"""
工作记忆模块 — 当前任务执行过程中的中间状态

核心职责：
  跟踪"当前任务做到哪一步了"。
  短期记忆管"说了什么"，长期记忆管"用户是谁"，
  工作记忆管"活干到哪了"。

三类记忆的边界（防止混淆）：
  短期记忆 = 对话内容（"用户说了xx，AI回复了xx"）
  长期记忆 = 用户画像（"这个用户喜欢微信沟通"）
  工作记忆 = 任务状态（"已查了订单，正在处理退款，还差审批"）

如果把工作记忆的信息存到长期记忆：
  → 临时任务状态变成了永久用户画像（"用户当前在处理退款"这种话
    会一直留在用户画像里，即使退款已经处理完了）

如果把长期记忆的信息放到工作记忆：
  → 用户的上个月的偏好占用了当前任务的思考空间（浪费 Token）

存储方式：
  进程内存（不持久化）。
  任务完成后自动清理（complete_task）。
  如果进程崩溃，正在进行的任务会丢失（这是预期的——未完成的任务
  重启后从最后 checkpoint 恢复，不依赖工作记忆的持久化）。

用途：
  - Handoff 四要素的数据源（original_query, completed_steps, pending_items, excluded_hypotheses）
  - Agent 决策的依据（"已经试过方案 A 了，别试了"）
  - 防死循环的判断依据（"已经试了 5 次了，第 6 次换个方法"）
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
#  工作记忆数据类
# ═══════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    单个任务的工作记忆。

    每个任务实例包含：
      - 原始问题：用户一开始问了什么
      - 当前 Agent：现在谁在处理
      - 已完成步骤：已经做了什么
      - 待解决项：还需要做什么
      - 已排除假设：试过了但不对的方向
      - 中间结果：工具调用的返回值缓存
      - 迭代次数：已经循环了多少轮

    这些字段合起来就是 Handoff 四要素：
      - original_query        → 原始问题
      - completed_steps       → 已完成步骤
      - pending_items         → 待解决项
      - excluded_hypotheses   → 已排除假设
    """

    def __init__(
        self,
        task_id: str,
        original_query: str,
        user_id: str = "",
        session_id: str = "",
    ):
        """
        初始化一个新任务的工作记忆。

        参数：
          task_id:       任务唯一标识
          original_query: 用户的原始问题（第一次说的话）
          user_id:       用户 ID（可选）
          session_id:    会话 ID（可选）
        """
        self.task_id = task_id
        self.original_query = original_query
        self.user_id = user_id
        self.session_id = session_id

        # 当前由哪个 Agent 处理
        self.current_agent: str = ""

        # 已完成步骤（按顺序记录）
        self.completed_steps: list[dict] = []

        # 待解决项
        self.pending_items: list[str] = []

        # 已排除假设（试过但不对的方法）
        self.excluded_hypotheses: list[str] = []

        # 中间结果缓存（工具调用结果等）
        self.intermediate_results: dict[str, Any] = {}

        # 迭代计数器
        self.iteration_count: int = 0

        # 时间戳
        self.created_at = time.time()
        self.updated_at = time.time()

    # ── 状态更新 ──────────────────────────────────────────────

    def add_step(self, agent_name: str, action: str, result: str = ""):
        """
        记录一个已完成步骤。

        参数：
          agent_name: 哪个 Agent 做的
          action:     做了什么（如"调用 query_order 查订单"）
          result:     结果摘要（如"订单已发货，预计1月20日送达"）
        """
        self.completed_steps.append({
            "agent": agent_name,
            "action": action,
            "result": result,
            "time": time.time(),
        })
        self.iteration_count += 1
        self.updated_at = time.time()

    def add_pending(self, item: str):
        """添加一个待解决项"""
        if item not in self.pending_items:
            self.pending_items.append(item)
            self.updated_at = time.time()

    def resolve_pending(self, item: str):
        """标记一个待解决项为已完成"""
        if item in self.pending_items:
            self.pending_items.remove(item)
            self.updated_at = time.time()

    def add_excluded(self, hypothesis: str):
        """添加一个已排除的假设"""
        if hypothesis not in self.excluded_hypotheses:
            self.excluded_hypotheses.append(hypothesis)
            self.updated_at = time.time()

    def cache_result(self, key: str, value: Any):
        """缓存中间结果"""
        self.intermediate_results[key] = value
        self.updated_at = time.time()

    def get_cached(self, key: str, default: Any = None) -> Any:
        """获取缓存的中间结果"""
        return self.intermediate_results.get(key, default)


    # ── 状态查询 ──────────────────────────────────────────────

    def should_escalate(self, max_iterations: int = 10) -> bool:
        """
        判断是否需要升级到人工。

        条件：
          - 迭代次数超过上限
          - 连续多次尝试后仍然有未解决的待办项
        """
        if self.iteration_count >= max_iterations:
            return True
        if self.iteration_count >= 5 and len(self.pending_items) > 0:
            return True
        return False

    def summary(self) -> str:
        """
        生成工作记忆的文字摘要（用于 LLM 上下文注入）。
        """
        parts = [f"【当前任务】{self.original_query}"]
        parts.append(f"处理 Agent: {self.current_agent or '待分配'}")
        parts.append(f"已迭代: {self.iteration_count} 次")

        if self.completed_steps:
            steps = [f"  {i+1}. {s['agent']}: {s['action']}"
                     for i, s in enumerate(self.completed_steps[-5:])]
            parts.append("已完成:\n" + "\n".join(steps))

        if self.pending_items:
            parts.append(f"待解决: {'; '.join(self.pending_items)}")

        if self.excluded_hypotheses:
            parts.append(f"已排除: {'; '.join(self.excluded_hypotheses)}")

        return " | ".join(parts)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "task_id": self.task_id,
            "original_query": self.original_query,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "current_agent": self.current_agent,
            "completed_steps": self.completed_steps,
            "pending_items": self.pending_items,
            "excluded_hypotheses": self.excluded_hypotheses,
            "intermediate_results": self.intermediate_results,
            "iteration_count": self.iteration_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(task={self.task_id[:12]}, "
            f"agent={self.current_agent or '?'}, "
            f"steps={len(self.completed_steps)}, "
            f"pending={len(self.pending_items)})"
        )


# ═══════════════════════════════════════════════════════════════
#  工作记忆管理器
# ═══════════════════════════════════════════════════════════════

class WorkingMemoryManager:
    """
    工作记忆管理器——管理所有进行中的任务。

    职责：
      - 创建任务 → 分配 task_id
      - 按 task_id 查找任务
      - 更新任务状态
      - 任务完成后清理
      - 检查是否需要升级
    """

    def __init__(self):
        # {task_id: WorkingMemory}
        self._tasks: dict[str, WorkingMemory] = {}

    def create_task(
        self,
        original_query: str,
        user_id: str = "",
        session_id: str = "",
        task_id: Optional[str] = None,
    ) -> WorkingMemory:
        """
        创建一个新的工作记忆任务。

        参数：
          original_query: 用户的原始问题
          user_id:        用户 ID（可选）
          session_id:     会话 ID（可选）
          task_id:        自定义任务 ID（可选，自动生成）

        返回：
          WorkingMemory 实例
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:12]}"

        task = WorkingMemory(
            task_id=task_id,
            original_query=original_query,
            user_id=user_id,
            session_id=session_id,
        )
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[WorkingMemory]:
        """获取指定任务的工作记忆"""
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, **updates) -> Optional[WorkingMemory]:
        """
        更新任务的属性。

        参数：
          task_id: 任务 ID
          **updates: 要更新的字段（如 current_agent="tech_support"）
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.updated_at = time.time()
        return task

    def complete_task(self, task_id: str) -> Optional[dict]:
        """
        完成任务并清理。

        返回任务的最后状态（用于归档到长期记忆）。
        """
        task = self._tasks.pop(task_id, None)
        if task is None:
            return None
        return task.to_dict()

    def get_active_tasks(self, user_id: Optional[str] = None) -> list[WorkingMemory]:
        """获取所有进行中的任务"""
        if user_id:
            return [t for t in self._tasks.values() if t.user_id == user_id]
        return list(self._tasks.values())

    @property
    def active_count(self) -> int:
        """当前进行中的任务数"""
        return len(self._tasks)

    def clear(self):
        """清理所有任务"""
        self._tasks.clear()

    def __repr__(self) -> str:
        return f"WorkingMemoryManager(active_tasks={self.active_count})"
