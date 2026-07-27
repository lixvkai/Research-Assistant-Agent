"""服务层 — 承载业务逻辑，供 Gradio UI 与 FastAPI 路由共同复用。

这一层刻意不持有「当前会话」：会话 id 一律由调用方显式传入，
因此同一进程内可以并发服务多个互不干扰的会话。
"""

import threading

from services.agent_service import (
    AgentService,
    SessionBusyError,
    SessionNotFoundError,
)
from services.kb_service import KnowledgeBaseService

_agent_service: AgentService | None = None
_kb_service: KnowledgeBaseService | None = None
_lock = threading.Lock()


def get_agent_service() -> AgentService:
    """进程级 AgentService 单例（Agent 与工具的加载成本较高，只做一次）。"""
    global _agent_service
    if _agent_service is None:
        with _lock:
            if _agent_service is None:
                _agent_service = AgentService()
    return _agent_service


def get_kb_service() -> KnowledgeBaseService:
    global _kb_service
    if _kb_service is None:
        with _lock:
            if _kb_service is None:
                _kb_service = KnowledgeBaseService()
    return _kb_service


__all__ = [
    "AgentService",
    "KnowledgeBaseService",
    "SessionBusyError",
    "SessionNotFoundError",
    "get_agent_service",
    "get_kb_service",
]
