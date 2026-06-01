"""
MCP (Model Context Protocol) 工具管理层

提供统一的工具注册、发现和调用协议，
使新工具可以即插即用，无需修改 Agent 代码。

本层既管理本地内置工具，也可接入真正的 MCP（Model Context Protocol）外部工具：
- 对外：`mcp_server/research_mcp_server.py` 用官方 SDK 把这些工具暴露成 MCP Server。
- 对内：`core/mcp_client.py` 连接外部 MCP Server，并把其工具注册到这里。
"""

import importlib
import logging

from core.react_agent import ReActAgent
from core.schemas import ToolInfo, ToolSpec

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP 服务端 — 统一管理和暴露工具。"""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def register_external(self, name: str, description: str, parameters: dict,
                          func, category: str = "MCP外部"):
        """注册来自外部 MCP Server 的工具。"""
        self.register(ToolSpec(
            name=name, description=description, parameters=parameters,
            func=func, category=category,
        ))

    def call_tool(self, name: str, arguments: dict):
        """按名称调用已注册工具（供 MCP Server 端暴露使用）。"""
        spec = self._tools.get(name)
        if spec is None:
            return f"错误：未找到工具 '{name}'"
        try:
            return spec.func(**(arguments or {}))
        except Exception as e:
            return f"工具 '{name}' 执行出错：{e}"

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def register_from_module(self, module_path: str, category: str = "general"):
        """从模块自动加载工具定义。模块需导出 TOOL_DEFINITION 或 TOOL_DEFINITIONS。"""
        module = importlib.import_module(module_path)

        defs = []
        if hasattr(module, "TOOL_DEFINITIONS"):
            defs = module.TOOL_DEFINITIONS
        elif hasattr(module, "TOOL_DEFINITION"):
            defs = [module.TOOL_DEFINITION]

        for defn in defs:
            self.register(ToolSpec(
                name=defn["name"],
                description=defn["description"],
                parameters=defn["parameters"],
                func=defn["func"],
                category=category,
            ))

    def list_tools(self) -> list[dict]:
        """列出所有已注册工具的元信息。"""
        return [
            ToolInfo(
                name=t.name,
                description=t.description,
                category=t.category,
                version=t.version,
            ).model_dump()
            for t in self._tools.values()
        ]

    def bind_to_agent(self, agent: ReActAgent, categories: tuple[str, ...] | None = None):
        """将工具绑定到 Agent；若指定 categories，则只绑定匹配类别的工具。"""
        for spec in self._tools.values():
            if categories is not None and spec.category not in categories:
                continue
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
    _load_external_mcp_tools(server)
    return server


def _load_external_mcp_tools(server: "MCPServer"):
    """从配置的外部 MCP Server 加载工具（best-effort，失败不影响启动）。"""
    try:
        from config.settings import MCP_SERVERS
    except Exception:
        MCP_SERVERS = []
    if not MCP_SERVERS:
        return
    try:
        from core.mcp_client import get_external_tool_specs
        for spec in get_external_tool_specs(MCP_SERVERS):
            server.register_external(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
                func=spec["func"],
                category=spec.get("category", "MCP外部"),
            )
            logger.info("已接入外部 MCP 工具：%s", spec["name"])
    except Exception as e:
        logger.warning("加载外部 MCP 工具失败（忽略）：%s", e)
