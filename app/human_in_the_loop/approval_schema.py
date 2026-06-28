"""
审批数据结构定义

这个文件定义了 HITL（Human-in-the-Loop）审批所需的全部数据结构。

HITL 工作流程：
  1. Agent 发起审批请求（如退款 1500 元 → 创建 ApprovalRequest）
  2. 状态图中 human_approval_node 检测到 approval_pending → 暂停执行
  3. Checkpointing 保存当前状态（所有上下文不丢失）
  4. 审批人通过 API 查看待审批列表
  5. 审批人 approve/reject
  6. ApprovalService 更新状态
  7. 图从 checkpoint 恢复执行（注入审批结论）
  8. Agent 根据审批结果继续或中止操作

审批触发条件（当前实现）：
  - 退款金额 > 1000 元（来自 create_refund 工具）
  - 质检评分 < 0.5（严重违规）
  - 用户主动要求转人工
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  审批状态枚举
# ═══════════════════════════════════════════════════════════════

class ApprovalStatus(str, Enum):
    """审批状态"""
    PENDING = "pending"            # 待审批
    APPROVED = "approved"           # 已通过
    REJECTED = "rejected"           # 已拒绝
    EXPIRED = "expired"             # 已超时


class ApprovalType(str, Enum):
    """审批类型"""
    REFUND = "refund"               # 退款审批
    ESCALATION = "escalation"       # 升级转人工审批
    SENSITIVE_OPERATION = "sensitive"  # 敏感操作审批


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

class ApprovalRequest(BaseModel):
    """
    审批请求——当 Agent 需要人类确认时创建此记录。

    字段说明：
      id:              审批唯一标识（自动生成）
      type:            审批类型（refund/escalation/sensitive）
      status:          当前状态（pending/approved/rejected/expired）
      from_agent:      发起审批的 Agent
      thread_id:       关联的 LangGraph 线程 ID（用于恢复执行）
      context:         上下文信息（订单号、金额、原因等）
      priority:        优先级（high=紧急 / normal=普通 / low=低）
      reason:          审批理由
      created_at:      创建时间戳
      expires_at:      过期时间戳（超时后自动标记为 expired）
      reviewed_at:     审核时间戳
      reviewer:        审核人
      comment:          审核意见
    """

    id: str = Field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:12]}")
    """审批唯一标识"""
    type: ApprovalType = ApprovalType.REFUND
    """审批类型"""
    status: ApprovalStatus = ApprovalStatus.PENDING
    """当前状态"""
    from_agent: str = ""
    """发起审批的 Agent 名称"""
    thread_id: str = ""
    """关联的 LangGraph 线程 ID"""
    context: dict[str, Any] = Field(default_factory=dict)
    """上下文（订单信息、金额等）"""
    priority: str = "normal"
    """优先级 high/normal/low"""
    reason: str = ""
    """审批理由"""
    created_at: float = Field(default_factory=time.time)
    """创建时间"""
    expires_at: float = Field(default_factory=lambda: time.time() + 3600)
    """过期时间（默认 1 小时）"""
    reviewed_at: float | None = None
    """审核时间"""
    reviewer: str = ""
    """审核人用户名"""
    comment: str = ""
    """审核意见"""

    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        return self.status == ApprovalStatus.PENDING and time.time() > self.expires_at

    @property
    def time_remaining(self) -> float:
        """剩余审批时间（秒）"""
        if self.status != ApprovalStatus.PENDING:
            return 0
        return max(0, self.expires_at - time.time())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "from_agent": self.from_agent,
            "thread_id": self.thread_id,
            "context": self.context,
            "priority": self.priority,
            "reason": self.reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "time_remaining": self.time_remaining,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "comment": self.comment,
        }


class ApprovalPolicy(BaseModel):
    """
    审批策略——定义什么情况下触发 HITL。

    字段说明：
      type:            审批类型
      enabled:         是否启用（可动态关闭）
      threshold:       触发阈值
      timeout_seconds: 超时秒数（超时后自动拒绝/升级）
      auto_action:     超时自动动作（approve/reject/escalate）
      description:     策略描述
    """

    type: ApprovalType = ApprovalType.REFUND
    enabled: bool = True
    threshold: float = 1000.0
    timeout_seconds: int = 3600
    auto_action: str = "escalate"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "timeout_seconds": self.timeout_seconds,
            "auto_action": self.auto_action,
            "description": self.description,
        }


# ═══════════════════════════════════════════════════════════════
#  默认审批策略
# ═══════════════════════════════════════════════════════════════

DEFAULT_POLICIES: dict[ApprovalType, ApprovalPolicy] = {
    ApprovalType.REFUND: ApprovalPolicy(
        type=ApprovalType.REFUND,
        threshold=1000.0,
        timeout_seconds=3600,
        auto_action="escalate",
        description="退款金额超过 1000 元需要人工审批",
    ),
    ApprovalType.SENSITIVE_OPERATION: ApprovalPolicy(
        type=ApprovalType.SENSITIVE_OPERATION,
        threshold=0.0,
        timeout_seconds=300,
        auto_action="reject",
        description="敏感操作需要人工审批",
    ),
}
