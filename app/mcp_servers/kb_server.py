"""
知识库搜索 MCP Server — Qdrant 语义检索
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.knowledge_base import search_knowledge_base as _search_knowledge_base

logger = logging.getLogger(__name__)

kb_server = FastMCP("kb-server")


@kb_server.tool()
async def search_knowledge_base(
    query: Annotated[str, Field(description="搜索查询（自然语言问题），如'电脑蓝屏了怎么办'", title="查询内容")],
    category: Annotated[str, Field(description="限定类别：政策/物流/使用指南/支付/售后/安全/账户/财务/内部，不传则搜全部", title="限定类别")] = "",
    top_k: Annotated[int, Field(description="返回最相关的几条结果（默认 3，最大 10）", title="返回条数")] = 3,
) -> dict:
    """从客服知识库中语义搜索解决方案和文档。使用 Qdrant 向量检索，支持自然语言查询。"""
    return _search_knowledge_base.invoke({
        "query": query, "category": category, "top_k": top_k,
    })


async def main():
    await kb_server.run_stdio_async()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
