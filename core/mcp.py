"""
MCP (Model Context Protocol) 工具管理层

提供统一的工具注册、发现和调用协议，
使新工具可以即插即用，无需修改 Agent 代码。
"""

import importlib
import os
from dataclasses import dataclass, field
from core.react_agent import ReActAgent


@dataclass
class MCPToolSpec:
    """MCP 工具规范。"""
    name: str
    description: str
    parameters: dict
    func: callable
    category: str = "general"
    version: str = "1.0"


class MCPServer:
    """MCP 服务端 — 统一管理和暴露工具。"""

    def __init__(self):
        self._tools: dict[str, MCPToolSpec] = {}

    def register(self, spec: MCPToolSpec):
        self._tools[spec.name] = spec

    def register_from_module(self, module_path: str, category: str = "general"):
        """从模块自动加载工具定义。模块需导出 TOOL_DEFINITION 或 TOOL_DEFINITIONS。"""
        module = importlib.import_module(module_path)

        defs = []
        if hasattr(module, "TOOL_DEFINITIONS"):
            defs = module.TOOL_DEFINITIONS
        elif hasattr(module, "TOOL_DEFINITION"):
            defs = [module.TOOL_DEFINITION]

        for defn in defs:
            self.register(MCPToolSpec(
                name=defn["name"],
                description=defn["description"],
                parameters=defn["parameters"],
                func=defn["func"],
                category=category,
            ))

    def list_tools(self) -> list[dict]:
        """列出所有已注册工具的元信息。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "version": t.version,
            }
            for t in self._tools.values()
        ]

    def bind_to_agent(self, agent: ReActAgent):
        """将所有工具绑定到 Agent。"""
        for spec in self._tools.values():
            agent.register_tool(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                func=spec.func,
            )

    @property
    def tool_count(self) -> int:
        return len(self._tools)


def create_default_mcp_server() -> MCPServer:
    """创建默认的 MCP Server，加载所有内置工具。"""
    server = MCPServer()
    server.register_from_module("tools.basic_tools", category="基础工具")
    server.register_from_module("tools.arxiv_tool", category="论文检索")
    server.register_from_module("tools.web_tool", category="网络工具")
    server.register_from_module("tools.summarize_tool", category="文本处理")
    server.register_from_module("tools.rag_tool", category="知识库")
    server.register_from_module("tools.memory_tool", category="记忆系统")
    server.register_from_module("tools.multi_agent_tool", category="多Agent协作")
    server.register_from_module("tools.skill_tool", category="技能系统")
    server.register_from_module("tools.compare_tool", category="论文分析")
    server.register_from_module("tools.trend_tool", category="趋势分析")
    return server
