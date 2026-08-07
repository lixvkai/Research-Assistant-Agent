"""Experiment runner for local, retrieval-only RAG evaluation."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from evaluation.rag.dataset import EvaluationQuery
from evaluation.rag.metrics import evaluate_ranking, mean_metrics


@dataclass(frozen=True)
class ExperimentConfig:
    """Switches and retrieval sizes for one controlled experiment."""

    name: str
    rewrite: bool
    rerank: bool
    candidate_k: int = 20
    final_k: int = 5

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("实验名称不能为空")
        if self.candidate_k <= 0 or self.final_k <= 0:
            raise ValueError("candidate_k 和 final_k 必须为正整数")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k 不能小于 final_k")


def default_experiments(
    candidate_k: int = 20,
    final_k: int = 5,
) -> list[ExperimentConfig]:
    """Return the four experiments defined in the evaluation plan."""
    return [
        ExperimentConfig("vector_baseline", False, False, candidate_k, final_k),
        ExperimentConfig("rewrite_vector", True, False, candidate_k, final_k),
        ExperimentConfig("vector_rerank", False, True, candidate_k, final_k),
        ExperimentConfig("full_pipeline", True, True, candidate_k, final_k),
    ]


def _document_id(doc: dict[str, Any]) -> str:
    value = doc.get("id") or doc.get("document_id")
    return str(value) if value is not None else ""


def _compact_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Keep ranking diagnostics while excluding full chunk content."""
    metadata = doc.get("metadata") or {}
    keys = (
        "distance",
        "similarity",
        "recall_rank",
        "rerank_raw_score",
        "rerank_score",
        "rerank_rank",
        "final_rank",
    )
    compact: dict[str, Any] = {
        "document_id": _document_id(doc),
        "source": metadata.get("source"),
    }
    compact.update({key: doc[key] for key in keys if doc.get(key) is not None})
    return compact


def _percentile(values: Iterable[float], percentile: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    position = (len(items) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    return items[lower] + (items[upper] - items[lower]) * (position - lower)


def _query_metrics(
    query: EvaluationQuery,
    retrieval: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, float]:
    relevance = query.relevance_by_id
    candidate_ids = [_document_id(doc) for doc in retrieval["candidates"]]
    result_ids = [_document_id(doc) for doc in retrieval["results"]]
    candidate = evaluate_ranking(candidate_ids, relevance, config.candidate_k)
    final = evaluate_ranking(result_ids, relevance, config.final_k)
    return {
        f"candidate_hit_at_{config.candidate_k}": candidate["hit"],
        f"candidate_recall_at_{config.candidate_k}": candidate["recall"],
        f"hit_at_{config.final_k}": final["hit"],
        f"recall_at_{config.final_k}": final["recall"],
        f"mrr_at_{config.final_k}": final["mrr"],
        f"ndcg_at_{config.final_k}": final["ndcg"],
    }


def _failure_labels(metrics: dict[str, float], retrieval: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if retrieval["candidate_count"] == 0:
        labels.append("empty_retrieval")
    candidate_recall = next(
        value for key, value in metrics.items() if key.startswith("candidate_recall_at_")
    )
    final_hit = next(value for key, value in metrics.items() if key.startswith("hit_at_"))
    if candidate_recall == 0:
        labels.append("missed_by_vector_recall")
    elif final_hit == 0:
        labels.append("relevant_chunk_dropped_from_final_top_k")
    if retrieval.get("rewrite_fallback"):
        labels.append("rewrite_fallback")
    if retrieval.get("rerank_fallback"):
        labels.append("rerank_fallback")
    return labels


def run_experiment(
    engine: Any,
    queries: Iterable[EvaluationQuery],
    config: ExperimentConfig,
    *,
    rewrite_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one experiment and retain partial results when a query fails."""
    cache = rewrite_cache if rewrite_cache is not None else {}
    query_results: list[dict[str, Any]] = []

    for query in queries:
        cached_rewrite = cache.get(query.query_id) if config.rewrite else None
        try:
            retrieval = engine.retrieve_structured(
                query.query,
                candidate_k=config.candidate_k,
                final_k=config.final_k,
                rewrite=config.rewrite,
                rerank=config.rerank,
                rewrite_override=cached_rewrite,
            )
            if (
                config.rewrite
                and not retrieval.get("rewrite_fallback")
                and query.query_id not in cache
            ):
                cache[query.query_id] = retrieval["retrieval_query"]
            metrics = _query_metrics(query, retrieval, config)
            failures = _failure_labels(metrics, retrieval)
            query_results.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "category": query.category,
                    "difficulty": query.difficulty,
                    "gold": [asdict(chunk) for chunk in query.relevant_chunks],
                    "retrieval_query": retrieval["retrieval_query"],
                    "rewrite_changed": retrieval.get("rewrite_changed", False),
                    "rewrite_fallback": retrieval.get("rewrite_fallback", False),
                    "rerank_applied": retrieval.get("rerank_applied", False),
                    "rerank_fallback": retrieval.get("rerank_fallback", False),
                    "candidate_count": retrieval["candidate_count"],
                    "returned_count": retrieval["returned_count"],
                    "candidates": [
                        _compact_document(doc) for doc in retrieval["candidates"]
                    ],
                    "results": [
                        _compact_document(doc) for doc in retrieval["results"]
                    ],
                    "metrics": metrics,
                    "latency_ms": retrieval["latency_ms"],
                    "failure_labels": failures,
                    "error": None,
                }
            )
        except Exception as exc:
            query_results.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "category": query.category,
                    "difficulty": query.difficulty,
                    "gold": [asdict(chunk) for chunk in query.relevant_chunks],
                    "metrics": {},
                    "failure_labels": ["execution_error"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = [row for row in query_results if row["error"] is None]
    failed = [row for row in query_results if row["error"] is not None]
    aggregate = mean_metrics(row["metrics"] for row in successful)
    latencies = [row["latency_ms"]["total"] for row in successful]
    rewrite_rows = [row for row in successful if config.rewrite]
    rerank_rows = [row for row in successful if config.rerank]
    summary = {
        "experiment": config.name,
        "config": asdict(config),
        "query_count": len(query_results),
        "success_count": len(successful),
        "error_count": len(failed),
        "metrics": {key: round(value, 6) for key, value in aggregate.items()},
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "empty_rate": (
            sum(row.get("candidate_count", 0) == 0 for row in successful)
            / len(successful)
            if successful
            else 0.0
        ),
        "rewrite_change_rate": (
            sum(bool(row["rewrite_changed"]) for row in rewrite_rows) / len(rewrite_rows)
            if rewrite_rows
            else 0.0
        ),
        "rewrite_fallback_rate": (
            sum(bool(row["rewrite_fallback"]) for row in rewrite_rows) / len(rewrite_rows)
            if rewrite_rows
            else 0.0
        ),
        "rerank_fallback_rate": (
            sum(bool(row["rerank_fallback"]) for row in rerank_rows) / len(rerank_rows)
            if rerank_rows
            else 0.0
        ),
    }
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary": summary,
        "query_results": query_results,
    }


def run_experiments(
    engine: Any,
    queries: Iterable[EvaluationQuery],
    configs: Iterable[ExperimentConfig],
) -> list[dict[str, Any]]:
    """Run experiments with one shared rewrite cache for reproducible comparisons."""
    query_list = list(queries)
    rewrite_cache: dict[str, str] = {}
    return [
        run_experiment(
            engine,
            query_list,
            config,
            rewrite_cache=rewrite_cache,
        )
        for config in configs
    ]
