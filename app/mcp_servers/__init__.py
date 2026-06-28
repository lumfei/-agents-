"""
MCP Server 层 — 将现有 LangChain @tool 包装为标准化 MCP 工具

每个 MCP Server 以 stdio 模式运行，可被以下客户端接入：
  - Claude Desktop / Claude Code
  - Cursor / Cline
  - 任何支持 MCP 协议的 Agent 框架
  - MCP Inspector 调试 (npx @modelcontextprotocol/inspector)

架构：
  MCP Client (外部) ──stdio──▶ FastMCP Server ──.invoke()──▶ 现有 app/tools 逻辑

用法示例：
  # 启动单个 server（stdio 模式）
  python -m app.mcp_servers.order_server

  # 列出所有 server
  python -m app.mcp_servers.run_all
"""

from app.mcp_servers.order_server import order_server
from app.mcp_servers.refund_server import refund_server
from app.mcp_servers.logistics_server import logistics_server
from app.mcp_servers.kb_server import kb_server
from app.mcp_servers.system_server import system_server

__all__ = [
    "order_server",
    "refund_server",
    "logistics_server",
    "kb_server",
    "system_server",
]
