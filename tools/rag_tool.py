"""RAG 知识库工具 — 让 Agent 可以检索本地论文库。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import PAPERS_DIR
from utils.path_safety import resolve_under

if TYPE_CHECKING:
    from rag.rag_engine import RAGEngine

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rag.rag_engine import RAGEngine

        _engine = RAGEngine()
    return _engine


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在本地知识库中搜索相关内容。"""
    engine = _get_engine()
    stats = engine.get_stats()
    if stats["document_count"] == 0:
        return "知识库为空，请先使用 ingest_paper 导入论文。"
    return engine.retrieve(query, top_k=top_k)


def ingest_paper(file_path: str) -> str:
    """将论文文件导入知识库（仅允许 PAPERS_DIR 内的文件）。"""
    try:
        safe_path = resolve_under(PAPERS_DIR, file_path)
    except ValueError as e:
        return f"导入失败：{e}"

    engine = _get_engine()
    try:
        n_chunks = engine.ingest_file(safe_path)
        stats = engine.get_stats()
        return (
            f"成功导入论文：{safe_path}\n"
            f"生成 {n_chunks} 个文本块\n"
            f"知识库当前共有 {stats['document_count']} 个文档块"
        )
    except Exception as e:
        return f"导入失败：{e}"


def get_knowledge_base_stats() -> str:
    """获取知识库统计信息。"""
    engine = _get_engine()
    stats = engine.get_stats()
    return f"知识库: {stats['collection']}\n文档块数量: {stats['document_count']}"


TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": "在本地知识库中语义搜索相关论文内容。使用此工具来查找已导入的论文中的信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，描述你想查找的内容",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的结果数量，默认 5",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        "func": search_knowledge_base,
    },
    {
        "name": "ingest_paper",
        "description": "将论文文件（PDF/TXT/MD）导入本地知识库，之后就可以用 search_knowledge_base 检索其内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "论文文件的完整路径（须位于 papers 目录内）",
                },
            },
            "required": ["file_path"],
        },
        "func": ingest_paper,
    },
    {
        "name": "get_knowledge_base_stats",
        "description": "查看知识库的统计信息，包括已导入的文档数量。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "func": get_knowledge_base_stats,
    },
]
