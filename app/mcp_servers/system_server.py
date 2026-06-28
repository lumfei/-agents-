"""
系统工具 MCP Server
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.system_tools import (
    check_service_status as _check_service_status,
    query_user_info as _query_user_info,
    get_system_announcements as _get_system_announcements,
)

logger = logging.getLogger(__name__)

system_server = FastMCP("system-server")


@system_server.tool()
async def check_service_status(
    service_name: Annotated[str, Field(description="服务名称，为空则返回所有服务状态", title="服务名称")] = "",
) -> dict:
    """检查系统各服务的运行状态。为空时返回所有服务。"""
    return _check_service_status.invoke({"service_name": service_name})


@system_server.tool()
async def query_user_info(
    user_id: Annotated[str, Field(description="用户 ID，格式如 CUxxxx", title="用户ID")],
) -> dict:
    """根据用户 ID 查询基本信息。返回昵称、会员等级、注册时间、联系方式。"""
    return _query_user_info.invoke({"user_id": user_id})


@system_server.tool()
async def get_system_announcements() -> dict:
    """获取系统公告列表。返回当前活跃公告，含标题、内容、生效时间。"""
    return _get_system_announcements.invoke({})


async def main():
    await system_server.run_stdio_async()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
