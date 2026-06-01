"""MCP 测试：注册表单元测试 + client↔server 真协议往返。"""

import os
import sys

import pytest

from core.mcp import MCPServer
from core.schemas import ToolSpec


# ── MCPServer 注册表单元测试 ───────────────────────────────────────

def test_register_and_call_tool():
    s = MCPServer()
    s.register(ToolSpec(name="calc", description="计算", parameters={"type": "object"},
                        func=lambda x: x * 2, category="基础工具"))
    assert s.tool_count == 1
    assert s.call_tool("calc", {"x": 21}) == 42


def test_call_unknown_tool():
    s = MCPServer()
    assert "未找到工具" in s.call_tool("nope", {})


def test_list_tools_shape():
    s = MCPServer()
    s.register(ToolSpec(name="t", description="d", parameters={"type": "object"},
                        func=lambda: 1, category="基础工具"))
    info = s.list_tools()
    assert info == [{"name": "t", "description": "d", "category": "基础工具", "version": "1.0"}]


def test_register_external():
    s = MCPServer()
    s.register_external("ext", "外部工具", {"type": "object"}, lambda: "ok", category="MCP:x")
    assert s.call_tool("ext", {}) == "ok"
    assert s.list_tools()[0]["category"] == "MCP:x"


# ── client ↔ server 往返（真 MCP stdio 协议） ─────────────────────

def test_mcp_client_server_roundtrip():
    from core.mcp_client import MCPClientManager

    script = os.path.join(os.path.dirname(__file__), "_mcp_basic_server.py")
    m = MCPClientManager()
    m.start([{
        "name": "selftest",
        "command": sys.executable,
        "args": [script],
        "env": dict(os.environ),
    }])
    try:
        specs = m.get_tool_specs()
        names = {s["name"] for s in specs}
        assert "calculator" in names and "get_current_time" in names

        calc = next(s for s in specs if s["name"] == "calculator")
        assert calc["parameters"].get("type") == "object"  # schema 透传
        out = calc["func"](expression="2**10")
        assert "1024" in out
    finally:
        m.stop()
