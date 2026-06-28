"""
人工参与包 — HITL 审批机制

提供审批数据结构、审批服务、通知和审批管理界面。

快速开始：
  from app.human_in_the_loop import approval_service, ApprovalRequest
  from app.human_in_the_loop.approval_schema import ApprovalType

  # 创建审批请求
  req = approval_service.create_request(
      approval_type=ApprovalType.REFUND,
      from_agent="finance",
      thread_id="thread_xxx",
      context={"amount": 1500, "order_id": "ORD-001"},
  )

  # 审批人批准（会自动恢复 LangGraph 图执行）
  approval_service.approve(req.id, reviewer="admin", comment="同意退款")

审批管理界面：
  启动服务后访问 GET /api/v1/approval/ui
"""

from app.human_in_the_loop.approval_schema import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalType,
    ApprovalPolicy,
    DEFAULT_POLICIES,
)

from app.human_in_the_loop.approval_service import (
    ApprovalService,
    approval_service,
)

from app.human_in_the_loop import notification

__all__ = [
    # Schema
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalType",
    "ApprovalPolicy",
    "DEFAULT_POLICIES",
    # Service
    "ApprovalService",
    "approval_service",
    # Notification
    "notification",
]
