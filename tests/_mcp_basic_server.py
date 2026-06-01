"""测试用轻量 MCP Server：仅暴露 basic_tools，供 client↔server 往返测试。

非测试模块（无 test_ 前缀，pytest 不会收集）。作为子进程被 test_mcp 启动。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.stdio import stdio_server  # noqa: E402
from core.mcp import MCPServer  # noqa: E402
from mcp_server.research_mcp_server import build_server  # noqa: E402

_reg = MCPServer()
_reg.register_from_module("tools.basic_tools", category="基础工具")
_server = build_server(_reg)


async def _main():
    async with stdio_server() as (read, write):
        await _server.run(read, write, _server.create_initialization_options())


asyncio.run(_main())
