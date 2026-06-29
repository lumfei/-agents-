"""
退款工具

功能：创建退款申请、查询退款进度。
数据来源：初始退款数据来自 data/seed/refunds.json（40 条退款记录），
         运行时新增退款写入内存。

关键设计：
  - 退款 > 1000 元自动标记为"需审批"（HITL 机制）
  - 同一订单不能重复退款（检查是否已有 pending_approval/refunding 状态）
  - 退款金额不能超过订单金额
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Annotated

from langchain_core.tools import tool

from app.data.loader import get_loader


@tool
def create_refund(
    order_id: Annotated[str, "要退款的订单号，格式如 ORD00001（必填）"],
    amount: Annotated[float, "退款金额，单位元，不能超过订单金额（必填）"],
    reason: Annotated[str, "退款原因，如'商品质量问题'、'不想要了'等（必填）"],
    user_id: Annotated[str, "用户 ID，格式如 CUxxxx（必填）"],
) -> dict[str, Any]:
    """创建退款申请。

使用场景：
  - 当用户明确要求退款、退货、退钱时
  - 用户说"我不想要了"、"把钱退给我"、"申请退款"时

返回字段：
  - refund_id: 退款编号，用于后续查询进度
  - status: 退款状态（pending_approval=待审批 / approved=已通过 / refunding=退款中 / completed=已完成 / rejected=已拒绝）
  - needs_approval: 是否需人工审批（退款金额>1000元时自动触发）

安全约束：
  - 同一订单不可重复提交退款申请（已存在进行中的退款时会拒绝）
  - 退款金额 > 1000 元自动进入审批流程（HITL）"""
    loader = get_loader()

    # 检查是否已存在进行中的退款
    existing = loader.get_refund_by_order(order_id)
    if existing is not None:
        return {
            "error": f"订单 {order_id} 已有进行中的退款申请（{existing['refund_id']}，状态: {existing['status']}）",
            "existing_refund_id": existing["refund_id"],
        }

    refund = loader.create_refund(order_id, amount, reason, user_id)
    refund["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    needs_approval = refund.get("hitl_required", False)
    refund["needs_approval"] = needs_approval
    if not needs_approval:
        refund["status"] = "approved"
    return dict(refund)


@tool
def query_refund_status(
    refund_id: Annotated[str, "退款编号（必填），接受多种格式如 RF0001 或 REF-2024-0001 等"],

) -> dict[str, Any]:
    """查询退款申请的处理进度。

使用场景：
  - 用户问"退款退了吗"、"退款到哪一步了"
  - 用户想查看之前申请的退款的处理状态

返回字段：
  - status: 当前状态（pending_approval=待审批 / approved=已通过 / refunding=退款中 / completed=已完成 / rejected=已拒绝）
  - amount: 退款金额
  - reason: 退款原因
  - solution_note: 处理备注"""
    loader = get_loader()
    refund = loader.get_refund(refund_id)
    if refund is None:
        return {"error": f"未找到退款申请 {refund_id}", "refund_id": refund_id}
    return dict(refund)
