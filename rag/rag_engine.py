"""RAG 引擎 — 文档处理 + 检索 + 生成 的完整管线。

升级版流程: 查询改写 → 多召回 → Cross-Encoder 重排序
"""

import os
import math
import logging

from rag.document_loader import process_document
from rag.vector_store import VectorStore
from core.llm import chat

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    """懒加载 Cross-Encoder 重排序模型（首次调用时下载约 400MB）。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
        logger.info("Reranker 模型加载完成")
    return _reranker


REWRITE_SYSTEM = (
    "你是学术搜索查询优化器。"
    "将用户的问题改写为更适合在学术论文知识库中语义检索的查询。"
    "要求：保留核心学术术语，补充同义词或英文关键词，只输出改写后的查询，不解释。"
)


class RAGEngine:
    """RAG 检索增强生成引擎。"""

    def __init__(self):
        self.vector_store = VectorStore()

    # ── 文档导入 ──────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> int:
        """导入单个文件到知识库，返回生成的块数。"""
        chunks = process_document(file_path)
        if not chunks:
            return 0

        filename = os.path.basename(file_path)
        # 重复导入时先清掉旧块，避免固定 ID 冲突
        self.vector_store.delete_by_source(filename)
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        self.vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> dict:
        """批量导入目录下所有支持的文件。"""
        stats = {"total_files": 0, "total_chunks": 0, "files": []}
        supported = (".pdf", ".txt", ".md", ".tex")

        for filename in os.listdir(dir_path):
            if not any(filename.lower().endswith(ext) for ext in supported):
                continue
            file_path = os.path.join(dir_path, filename)
            try:
                n_chunks = self.ingest_file(file_path)
                stats["total_files"] += 1
                stats["total_chunks"] += n_chunks
                stats["files"].append({"name": filename, "chunks": n_chunks})
            except Exception as e:
                stats["files"].append({"name": filename, "error": str(e)})

        return stats

    # ── 查询改写 ──────────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        """用 LLM 将口语化问题改写为适合语义检索的学术查询。"""
        try:
            response = chat(
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                observation_name="rag-query-rewrite",
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten:
                logger.info(f"查询改写: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"查询改写失败，使用原始查询: {e}")
        return query

    # ── 重排序 ────────────────────────────────────────────────

    def _rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """用 Cross-Encoder 对候选文档精排。"""
        if not docs:
            return docs
        try:
            reranker = _get_reranker()
            pairs = [[query, doc["content"]] for doc in docs]
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                docs[i]["rerank_score"] = 1.0 / (1.0 + math.exp(-float(score)))

            docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"重排序: {len(docs)} 候选 → top {top_k}")
            return docs[:top_k]
        except Exception as e:
            logger.warning(f"重排序失败，保持原始排序: {e}")
            return docs[:top_k]

    # ── 检索（完整流程） ──────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """检索流程: 查询改写 → 多召回 → 重排序 → 格式化输出。"""
        rewritten = self._rewrite_query(query)

        recall_k = min(top_k * 4, 20)
        candidates = self.vector_store.search(rewritten, top_k=recall_k)

        if not candidates:
            return "知识库中未找到相关内容。"

        results = self._rerank(rewritten, candidates, top_k)

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc["metadata"].get("source", "未知")
            if "rerank_score" in doc:
                score = doc["rerank_score"]
            else:
                score = 1 - doc.get("distance", 0)
            context_parts.append(
                f"[来源{i}: {source} | 相关度: {score:.2f}]\n{doc['content']}"
            )

        return "\n\n---\n\n".join(context_parts)

    # ── 删除 / 统计 ──────────────────────────────────────────

    def delete_file(self, filename: str) -> int:
        """从向量库中删除指定文件的所有块。"""
        return self.vector_store.delete_by_source(filename)

    def get_stats(self) -> dict:
        return self.vector_store.get_stats()
