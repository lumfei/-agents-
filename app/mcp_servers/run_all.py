"""
MCP Server 清单 — 展示所有 MCP 工具服务

用法:
  python -m app.mcp_servers.run_all
"""
from __future__ import annotations

import json

SERVERS = {
    "order-server": {
        "module": "app.mcp_servers.order_server",
        "description": "订单查询服务",
        "tools": ["query_order", "list_user_orders"],
    },
    "refund-server": {
        "module": "app.mcp_servers.refund_server",
        "description": "退款处理服务（含 HITL 审批）",
        "tools": ["create_refund", "query_refund_status"],
    },
    "logistics-server": {
        "module": "app.mcp_servers.logistics_server",
        "description": "物流追踪服务",
        "tools": ["track_logistics", "query_logistics_by_order"],
    },
    "kb-server": {
        "module": "app.mcp_servers.kb_server",
        "description": "知识库搜索服务（Qdrant 语义检索）",
        "tools": ["search_knowledge_base"],
    },
    "system-server": {
        "module": "app.mcp_servers.system_server",
        "description": "系统工具服务",
        "tools": ["check_service_status", "query_user_info", "get_system_announcements"],
    },
}


def print_server_list():
    """打印所有 MCP Server 的启动信息。"""
    total_tools = sum(len(v["tools"]) for v in SERVERS.values())
    print("=" * 60)
    print(f"  MCP Server Tool Layer -- {len(SERVERS)} services, {total_tools} tools")
    print("=" * 60)
    print()
    for name, info in SERVERS.items():
        print(f"  [{name}]")
        print(f"     {info['description']}")
        print(f"     MCP tools: {', '.join(info['tools'])}")
        print(f"     Run: python -m {info['module']}")
        print()
    print("-" * 60)
    print("  MCP Inspector (visual debug):")
    print(f"    npx @modelcontextprotocol/inspector python -m app.mcp_servers.kb_server")
    print()
    print("  Claude Desktop config (claude_desktop_config.json):")
    cfg = {
        "mcpServers": {
            name: {"command": "python", "args": ["-m", info["module"]]}
            for name, info in SERVERS.items()
        }
    }
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    print_server_list()
