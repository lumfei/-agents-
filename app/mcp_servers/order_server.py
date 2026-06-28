"""
订单查询 MCP Server
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.order_tools import query_order as _query_order, list_user_orders as _list_user_orders

logger = logging.getLogger(__name__)

order_server = FastMCP("order-server")


@order_server.tool()
async def query_order(
    order_id: Annotated[str, Field(description="订单号，格式如 ORD00001", title="订单号")],
    user_id: Annotated[str, Field(description="用户 ID（可选，不传则不校验订单归属）", title="用户ID")] = "",
) -> dict:
    """根据订单号查询订单的完整信息。返回订单状态、商品清单、金额、收货地址、快递单号等。"""
    return _query_order.invoke({"order_id": order_id, "user_id": user_id})


@order_server.tool()
async def list_user_orders(
    user_id: Annotated[str, Field(description="用户 ID，格式如 CUxxxx", title="用户ID")],
    page: Annotated[int, Field(description="页码，从 1 开始", title="页码")] = 1,
    page_size: Annotated[int, Field(description="每页返回的订单数量", title="每页条数")] = 10,
) -> dict:
    """查询指定用户的所有订单列表（分页）。返回订单号、状态、金额、下单时间。"""
    return _list_user_orders.invoke({"user_id": user_id, "page": page, "page_size": page_size})


async def main():
    await order_server.run_stdio_async()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
