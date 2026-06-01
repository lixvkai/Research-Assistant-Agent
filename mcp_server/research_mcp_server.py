"""
科研助手 MCP Server — 用官方 MCP SDK 把内置工具暴露为真正的 MCP 工具。

通过 stdio 传输，可被任何 MCP 客户端（Cursor / Claude Desktop / 本项目的 MCP Client）调用。

直接运行：
    python -m mcp_server.research_mcp_server

在 MCP 客户端中配置（示例）：
    {
      "mcpServers": {
        "research-assistant": {
          "command": "python",
          "args": ["-m", "mcp_server.research_mcp_server"]
        }
      }
    }
"""

import asyncio
import json
import logging

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from core.mcp import MCPServer, create_default_mcp_server

logger = logging.getLogger(__name__)


def build_server(registry: MCPServer | None = None) -> Server:
    """基于工具注册表构建一个 MCP Server。"""
    registry = registry or create_default_mcp_server()
    server: Server = Server("research-assistant")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.parameters,
            )
            for spec in registry.specs
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        # 工具实现是同步的，放到线程里执行避免阻塞事件循环。
        result = await asyncio.to_thread(registry.call_tool, name, arguments or {})
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return [types.TextContent(type="text", text=text)]

    return server


async def _main():
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
