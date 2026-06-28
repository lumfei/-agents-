"""
退款处理 MCP Server — 含 HITL 审批
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.refund_tools import create_refund as _create_refund, query_refund_status as _query_refund_status

logger = logging.getLogger(__name__)

refund_server = FastMCP("refund-server")


@refund_server.tool()
async def create_refund(
    order_id: Annotated[str, Field(description="要退款的订单号，格式如 ORD00001", title="订单号")],
    amount: Annotated[float, Field(description="退款金额（元）。超过 1000 元自动触发人工审批", title="退款金额")],
    reason: Annotated[str, Field(description="退款原因，如'商品质量问题'、'不想要了'等", title="退款原因")],
    user_id: Annotated[str, Field(description="用户 ID，格式如 CUxxxx", title="用户ID")],
) -> dict:
    """创建退款申请。超过 1000 元自动进入 HITL 人工审批。同一订单不可重复提交。"""
    return _create_refund.invoke({
        "order_id": order_id, "amount": amount, "reason": reason, "user_id": user_id,
    })


@refund_server.tool()
async def query_refund_status(
    refund_id: Annotated[str, Field(description="退款编号，格式如 RF0001", title="退款编号")],
) -> dict:
    """查询退款申请的处理进度。返回状态: 待审批/已通过/退款中/已完成/已拒绝"""
    return _query_refund_status.invoke({"refund_id": refund_id})


async def main():
    await refund_server.run_stdio_async()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
