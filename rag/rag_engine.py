"""RAG 引擎 — 文档处理 + 检索 + 生成 的完整管线。

升级版流程: 查询改写 → 多召回 → Cross-Encoder 重排序
"""

import os
import math
import logging
import time
from dataclasses import dataclass

from rag.document_loader import process_document
from rag.vector_store import VectorStore
from core.llm import chat
from core.observability import capture_value, observe_operation

logger = logging.getLogger(__name__)

_reranker = None
_RERANKER_MODEL = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class RewriteOutcome:
    """查询改写结果及其降级状态。"""

    query: str
    changed: bool
    fallback: bool


@dataclass
class RerankOutcome:
    """重排序结果及其降级状态。"""

    results: list[dict]
    applied: bool
    fallback: bool


def _get_reranker():
    """懒加载 Cross-Encoder 重排序模型（首次调用时下载约 400MB）。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(_RERANKER_MODEL)
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

    @staticmethod
    def _document_metrics(doc: dict) -> dict:
        """生成适合观测面板的紧凑检索指标，不重复上传文档正文。"""
        metadata = doc.get("metadata") or {}
        fields = {
            "document_id": doc.get("id"),
            "source": metadata.get("source"),
            "recall_rank": doc.get("recall_rank"),
            "distance": doc.get("distance"),
            "similarity": doc.get("similarity"),
            "rerank_raw_score": doc.get("rerank_raw_score"),
            "rerank_score": doc.get("rerank_score"),
            "rerank_rank": doc.get("rerank_rank"),
        }
        return {key: value for key, value in fields.items() if value is not None}

    def _rewrite_query_with_details(self, query: str) -> RewriteOutcome:
        """改写查询，并返回是否发生改变或降级。"""
        with observe_operation(
            "rag.rewrite-query",
            as_type="chain",
            input={"query": query},
        ) as observation:
            try:
                response = chat(
                    messages=[
                        {"role": "system", "content": REWRITE_SYSTEM},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,
                    observation_name="rag-query-rewrite",
                )
                rewritten = (response.choices[0].message.content or "").strip()
                if rewritten:
                    logger.info("查询改写: '%s' → '%s'", query, rewritten)
                    if observation is not None:
                        observation.update(
                            output=capture_value({"rewritten_query": rewritten}),
                            metadata={"changed": rewritten != query, "fallback": False},
                        )
                    return RewriteOutcome(
                        query=rewritten,
                        changed=rewritten != query,
                        fallback=False,
                    )
                status_message = "empty rewritten query; using original query"
            except Exception as e:
                logger.warning("查询改写失败，使用原始查询: %s", e)
                status_message = f"{type(e).__name__}: using original query"

            if observation is not None:
                observation.update(
                    output=capture_value({"rewritten_query": query}),
                    metadata={"changed": False, "fallback": True},
                    level="WARNING",
                    status_message=status_message,
                )
            return RewriteOutcome(query=query, changed=False, fallback=True)

    def _rewrite_query(self, query: str) -> str:
        """用 LLM 将口语化问题改写为适合语义检索的学术查询。"""
        return self._rewrite_query_with_details(query).query

    # ── 重排序 ────────────────────────────────────────────────

    def _rerank_with_details(
        self,
        query: str,
        docs: list[dict],
        top_k: int,
    ) -> RerankOutcome:
        """重排候选文档，并返回是否应用成功或发生降级。"""
        if not docs:
            return RerankOutcome(results=[], applied=False, fallback=False)
        for index, doc in enumerate(docs, 1):
            doc.setdefault("recall_rank", index)
            if "distance" in doc:
                doc.setdefault("similarity", 1.0 - float(doc["distance"]))

        before = [self._document_metrics(doc) for doc in docs]
        with observe_operation(
            "rag.rerank-results",
            as_type="chain",
            input={
                "query": query,
                "top_k": top_k,
                "candidate_count": len(docs),
                "candidates": before,
            },
            metadata={"reranker_model": _RERANKER_MODEL},
        ) as observation:
            try:
                reranker = _get_reranker()
                pairs = [[query, doc["content"]] for doc in docs]
                scores = reranker.predict(pairs)

                for i, score in enumerate(scores):
                    raw_score = float(score)
                    docs[i]["rerank_raw_score"] = raw_score
                    if raw_score >= 0:
                        normalized = 1.0 / (1.0 + math.exp(-raw_score))
                    else:
                        exp_score = math.exp(raw_score)
                        normalized = exp_score / (1.0 + exp_score)
                    docs[i]["rerank_score"] = normalized

                docs.sort(key=lambda item: item["rerank_score"], reverse=True)
                for rank, doc in enumerate(docs, 1):
                    doc["rerank_rank"] = rank
                results = docs[:top_k]
                logger.info("重排序: %d 候选 → top %d", len(docs), top_k)
                if observation is not None:
                    observation.update(
                        output=capture_value(
                            {
                                "rerank_applied": True,
                                "returned_count": len(results),
                                "candidates": [
                                    self._document_metrics(doc) for doc in results
                                ],
                            }
                        ),
                        metadata={
                            "reranker_model": _RERANKER_MODEL,
                            "candidate_count": len(docs),
                            "returned_count": len(results),
                            "fallback": False,
                        },
                    )
                return RerankOutcome(
                    results=results,
                    applied=True,
                    fallback=False,
                )
            except Exception as e:
                logger.warning("重排序失败，保持原始排序: %s", e)
                results = docs[:top_k]
                for rank, doc in enumerate(results, 1):
                    doc["rerank_rank"] = rank
                if observation is not None:
                    observation.update(
                        output=capture_value(
                            {
                                "rerank_applied": False,
                                "returned_count": len(results),
                                "candidates": [
                                    self._document_metrics(doc) for doc in results
                                ],
                            }
                        ),
                        metadata={
                            "reranker_model": _RERANKER_MODEL,
                            "candidate_count": len(docs),
                            "returned_count": len(results),
                            "fallback": True,
                        },
                        level="WARNING",
                        status_message=f"{type(e).__name__}: kept recall order",
                    )
                return RerankOutcome(
                    results=results,
                    applied=False,
                    fallback=True,
                )

    def _rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """用 Cross-Encoder 对候选文档精排。"""
        return self._rerank_with_details(query, docs, top_k).results

    @staticmethod
    def _copy_documents(docs: list[dict]) -> list[dict]:
        """复制检索结果，避免评测 rerank 改写原始召回排名。"""
        return [
            {
                **doc,
                "metadata": dict(doc.get("metadata") or {}),
            }
            for doc in docs
        ]

    def retrieve_structured(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        final_k: int = 5,
        rewrite: bool = True,
        rerank: bool = True,
        rewrite_override: str | None = None,
    ) -> dict:
        """返回适合本地评测的结构化检索结果。

        该接口不格式化上下文、不生成最终答案，也不改变线上 ``retrieve``
        的返回契约。调用者可以分别关闭查询改写和 rerank 做对照实验。
        """
        original_query = (query or "").strip()
        if not original_query:
            raise ValueError("query 不能为空")
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k 和 final_k 必须为正整数")
        if candidate_k < final_k:
            raise ValueError("candidate_k 不能小于 final_k")

        total_started = time.perf_counter()

        rewrite_started = time.perf_counter()
        if rewrite and rewrite_override is not None:
            cached_query = rewrite_override.strip()
            if not cached_query:
                raise ValueError("rewrite_override 不能为空")
            rewrite_outcome = RewriteOutcome(
                query=cached_query,
                changed=cached_query != original_query,
                fallback=False,
            )
        elif rewrite:
            rewrite_outcome = self._rewrite_query_with_details(original_query)
        else:
            rewrite_outcome = RewriteOutcome(
                query=original_query,
                changed=False,
                fallback=False,
            )
        rewrite_ms = (
            (time.perf_counter() - rewrite_started) * 1000
            if rewrite and rewrite_override is None
            else 0.0
        )

        search_started = time.perf_counter()
        candidates = self.vector_store.search(
            rewrite_outcome.query,
            top_k=candidate_k,
        )
        search_ms = (time.perf_counter() - search_started) * 1000
        candidate_snapshot = self._copy_documents(candidates)

        rerank_started = time.perf_counter()
        if rerank and candidates:
            rerank_outcome = self._rerank_with_details(
                rewrite_outcome.query,
                self._copy_documents(candidates),
                final_k,
            )
        else:
            plain_results = self._copy_documents(candidates[:final_k])
            rerank_outcome = RerankOutcome(
                results=plain_results,
                applied=False,
                fallback=False,
            )
        rerank_ms = (
            (time.perf_counter() - rerank_started) * 1000
            if rerank and candidates
            else 0.0
        )

        results = self._copy_documents(rerank_outcome.results)
        for rank, doc in enumerate(results, 1):
            doc["final_rank"] = rank

        return {
            "query": original_query,
            "retrieval_query": rewrite_outcome.query,
            "rewrite_enabled": rewrite,
            "rewrite_cached": rewrite and rewrite_override is not None,
            "rewrite_changed": rewrite_outcome.changed,
            "rewrite_fallback": rewrite_outcome.fallback,
            "rerank_enabled": rerank,
            "rerank_applied": rerank_outcome.applied,
            "rerank_fallback": rerank_outcome.fallback,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "candidate_count": len(candidate_snapshot),
            "returned_count": len(results),
            "candidates": candidate_snapshot,
            "results": results,
            "latency_ms": {
                "rewrite": round(rewrite_ms, 3),
                "search": round(search_ms, 3),
                "rerank": round(rerank_ms, 3),
                "total": round((time.perf_counter() - total_started) * 1000, 3),
            },
        }

    # ── 检索（完整流程） ──────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """检索流程: 查询改写 → 多召回 → 重排序 → 格式化输出。"""
        with observe_operation(
            "rag.retrieve-context",
            as_type="chain",
            input={"query": query, "top_k": top_k},
        ) as pipeline_observation:
            rewritten = self._rewrite_query(query)

            recall_k = min(top_k * 4, 20)
            with observe_operation(
                "rag.search-vectors",
                as_type="retriever",
                input={"query": rewritten, "top_k": recall_k},
                metadata={
                    "collection": self.vector_store.collection.name,
                    "distance_metric": "cosine",
                },
            ) as search_observation:
                candidates = self.vector_store.search(rewritten, top_k=recall_k)
                if search_observation is not None:
                    search_observation.update(
                        output=capture_value(
                            {
                                "candidate_count": len(candidates),
                                "candidates": [
                                    self._document_metrics(doc) for doc in candidates
                                ],
                            }
                        ),
                        metadata={
                            "collection": self.vector_store.collection.name,
                            "distance_metric": "cosine",
                            "requested_count": recall_k,
                            "candidate_count": len(candidates),
                        },
                    )

            if not candidates:
                result_text = "知识库中未找到相关内容。"
                if pipeline_observation is not None:
                    pipeline_observation.update(
                        output={
                            "rewritten_query": rewritten,
                            "candidate_count": 0,
                            "returned_count": 0,
                        }
                    )
                return result_text

            results = self._rerank(rewritten, candidates, top_k)

            context_parts = []
            for i, doc in enumerate(results, 1):
                source = doc["metadata"].get("source", "未知")
                if "rerank_score" in doc:
                    score = doc["rerank_score"]
                else:
                    score = doc.get("similarity", 1 - doc.get("distance", 0))
                context_parts.append(
                    f"[来源{i}: {source} | 相关度: {score:.2f}]\n{doc['content']}"
                )

            result_text = "\n\n---\n\n".join(context_parts)
            if pipeline_observation is not None:
                pipeline_observation.update(
                    output=capture_value(
                        {
                            "rewritten_query": rewritten,
                            "candidate_count": len(candidates),
                            "returned_count": len(results),
                            "results": [
                                self._document_metrics(doc) for doc in results
                            ],
                        }
                    )
                )
            return result_text

    # ── 删除 / 统计 ──────────────────────────────────────────

    def delete_file(self, filename: str) -> int:
        """从向量库中删除指定文件的所有块。"""
        return self.vector_store.delete_by_source(filename)

    def get_stats(self) -> dict:
        return self.vector_store.get_stats()
