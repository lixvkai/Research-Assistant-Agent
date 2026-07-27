"""长期记忆的语义索引 — 复用 RAG 的嵌入模型与 ChromaDB。

由 `MEMORY_SEMANTIC_SEARCH` 开关控制；不可用时 `LongTermMemory` 会自动降级为关键词检索，
因此本模块的任何导入/运行失败都不应影响主流程。
"""

from __future__ import annotations

import logging
import os

from config.settings import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "long_term_memories"


class SemanticMemoryIndex:
    """记忆向量索引（与论文库分开 collection，互不干扰）。"""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        import chromadb

        from rag.embeddings import SharedEmbeddingFunction

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=SharedEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, memory_id: int, content: str, category: str) -> None:
        self._collection.upsert(
            ids=[str(memory_id)],
            documents=[content],
            metadatas=[{"category": category}],
        )

    def query(self, query: str, limit: int = 5) -> list[int]:
        if self._collection.count() == 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=limit)
        ids = (res.get("ids") or [[]])[0]
        out: list[int] = []
        for raw in ids:
            try:
                out.append(int(raw))
            except (TypeError, ValueError):
                continue
        return out

    def delete(self, memory_id: int) -> None:
        self._collection.delete(ids=[str(memory_id)])
