"""Pure retrieval metrics with no model, database, or network dependency."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k 必须是正整数")


def _unique_top_k(ranking: Iterable[str], k: int) -> list[str]:
    _validate_k(k)
    unique: list[str] = []
    seen: set[str] = set()
    for document_id in ranking:
        if document_id in seen:
            continue
        seen.add(document_id)
        unique.append(document_id)
        if len(unique) == k:
            break
    return unique


def _positive_relevance(relevance_by_id: Mapping[str, int]) -> dict[str, int]:
    relevant = {
        document_id: int(relevance)
        for document_id, relevance in relevance_by_id.items()
        if int(relevance) > 0
    }
    if not relevant:
        raise ValueError("至少需要一个 relevance > 0 的 Gold 文档")
    return relevant


def hit_at_k(
    ranking: Iterable[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Whether at least one relevant document appears in the top-k."""
    relevant = _positive_relevance(relevance_by_id)
    return float(any(document_id in relevant for document_id in _unique_top_k(ranking, k)))


def recall_at_k(
    ranking: Iterable[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Fraction of all labelled relevant documents found in the top-k."""
    relevant = _positive_relevance(relevance_by_id)
    retrieved = set(_unique_top_k(ranking, k))
    return len(retrieved.intersection(relevant)) / len(relevant)


def reciprocal_rank_at_k(
    ranking: Iterable[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant top-k result."""
    relevant = _positive_relevance(relevance_by_id)
    for rank, document_id in enumerate(_unique_top_k(ranking, k), 1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranking: Iterable[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Normalized discounted cumulative gain using graded relevance."""
    relevant = _positive_relevance(relevance_by_id)
    top_ids = _unique_top_k(ranking, k)
    dcg = sum(
        (2 ** relevant.get(document_id, 0) - 1) / math.log2(rank + 1)
        for rank, document_id in enumerate(top_ids, 1)
    )
    ideal_relevances = sorted(relevant.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2 ** relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_ranking(
    ranking: Iterable[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> dict[str, float]:
    """Compute all first-stage retrieval metrics for one query."""
    ranking_list = list(ranking)
    return {
        "hit": hit_at_k(ranking_list, relevance_by_id, k),
        "recall": recall_at_k(ranking_list, relevance_by_id, k),
        "mrr": reciprocal_rank_at_k(ranking_list, relevance_by_id, k),
        "ndcg": ndcg_at_k(ranking_list, relevance_by_id, k),
    }


def mean_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Average metric dictionaries with identical keys."""
    items = list(rows)
    if not items:
        return {}
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("所有指标行必须具有相同字段")
    return {
        key: sum(float(item[key]) for item in items) / len(items)
        for key in sorted(keys)
    }
