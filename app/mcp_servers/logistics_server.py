"""
物流追踪 MCP Server
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.logistics_tools import (
    track_logistics as _track_logistics,
    query_logistics_by_order as _query_logistics_by_order,
)

logger = logging.getLogger(__name__)

logistics_server = FastMCP("logistics-server")


@logistics_server.tool()
async def track_logistics(
    tracking_no: Annotated[str, Field(description="运单号/快递单号，由快递公司提供", title="快递单号")],
) -> dict:
    """根据运单号查询物流轨迹。返回快递公司、当前状态、当前位置、轨迹事件列表。"""
    return _track_logistics.invoke({"tracking_no": tracking_no})


@logistics_server.tool()
async def query_logistics_by_order(
    order_id: Annotated[str, Field(description="订单号（无需知道快递单号）", title="订单号")],
) -> dict:
    """根据订单号查询物流信息。订单未发货时返回提示而非错误。"""
    return _query_logistics_by_order.invoke({"order_id": order_id})


async def main():
    await logistics_server.run_stdio_async()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
