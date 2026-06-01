"""
MCP Client 适配器 — 连接外部 MCP（Model Context Protocol）Server，
把远程工具转换成本项目的工具规范，透明注册给 Agent。

MCP SDK 是异步的，而本项目 Agent 是同步的，因此这里用一个后台事件循环线程
做异步→同步桥接：会话在后台线程常驻，工具调用通过 run_coroutine_threadsafe 同步等待结果。

配置示例（config/settings.py 的 MCP_SERVERS）：
    MCP_SERVERS = [
        {"name": "filesystem", "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]},
    ]
"""

import asyncio
import atexit
import logging
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClientManager:
    """管理到一个或多个外部 MCP Server 的常驻连接，并暴露同步调用接口。"""

    def __init__(self, call_timeout: float = 60.0):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_evt: asyncio.Event | None = None
        self._ready = threading.Event()
        self._sessions: dict[str, ClientSession] = {}
        self._specs: list[dict] = []
        self._call_timeout = call_timeout

    # ── 生命周期 ────────────────────────────────────────────────

    def start(self, servers: list[dict]):
        """启动后台事件循环线程并连接所有配置的 MCP Server。"""
        if not servers or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, args=(servers,), daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60)

    def _run(self, servers: list[dict]):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve(servers))
        finally:
            self._loop.close()

    async def _serve(self, servers: list[dict]):
        """单协程内完成 连接→就绪→常驻→关闭，避免跨任务退出 async 上下文。"""
        self._stop_evt = asyncio.Event()
        async with AsyncExitStack() as stack:
            for cfg in servers:
                await self._connect_one(stack, cfg)
            self._ready.set()
            await self._stop_evt.wait()

    async def _connect_one(self, stack: AsyncExitStack, cfg: dict):
        name = cfg.get("name", "mcp")
        try:
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env"),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session

            resp = await session.list_tools()
            category = cfg.get("category", f"MCP:{name}")
            for tool in resp.tools:
                self._specs.append(self._make_spec(name, tool, category))
            logger.info("已连接 MCP Server '%s'，发现 %d 个工具", name, len(resp.tools))
        except Exception as e:
            logger.warning("连接 MCP Server '%s' 失败（跳过）：%s", name, e)

    def stop(self):
        if self._loop and self._stop_evt and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        if self._thread:
            self._thread.join(timeout=10)
        self._thread = None

    # ── 工具规范与调用 ──────────────────────────────────────────

    def _make_spec(self, server_name: str, tool, category: str) -> dict:
        def func(**kwargs):
            return self.call(server_name, tool.name, kwargs)

        return {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            "func": func,
            "category": category,
        }

    def get_tool_specs(self) -> list[dict]:
        return list(self._specs)

    def call(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """同步调用远程 MCP 工具，返回文本结果。"""
        if self._loop is None or self._loop.is_closed():
            return "错误：MCP 客户端未就绪"
        future = asyncio.run_coroutine_threadsafe(
            self._call_async(server_name, tool_name, arguments), self._loop
        )
        try:
            return future.result(timeout=self._call_timeout)
        except Exception as e:
            return f"MCP 工具 '{tool_name}' 调用出错：{e}"

    async def _call_async(self, server_name: str, tool_name: str, arguments: dict) -> str:
        session = self._sessions.get(server_name)
        if session is None:
            return f"错误：未连接 MCP Server '{server_name}'"
        result = await session.call_tool(tool_name, arguments or {})
        parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        return "\n".join(parts) if parts else ""


# ── 模块级单例：供 create_default_mcp_server 懒加载 ──────────────────

_manager: MCPClientManager | None = None
_cached_specs: list[dict] | None = None


def get_external_tool_specs(servers: list[dict]) -> list[dict]:
    """懒启动全局 MCP 客户端并返回外部工具规范（进程内缓存，只启动一次）。"""
    global _manager, _cached_specs
    if _cached_specs is not None:
        return _cached_specs
    _manager = MCPClientManager()
    _manager.start(servers)
    atexit.register(shutdown_external_mcp)  # 进程退出时关闭外部 MCP 子进程
    _cached_specs = _manager.get_tool_specs()
    return _cached_specs


def shutdown_external_mcp():
    """关闭外部 MCP 连接（进程退出时可选调用）。"""
    global _manager, _cached_specs
    if _manager is not None:
        _manager.stop()
        _manager = None
        _cached_specs = None
