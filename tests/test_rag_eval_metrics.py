"""Pure metric tests for local RAG evaluation."""

import math

import pytest

from evaluation.rag.metrics import (
    evaluate_ranking,
    hit_at_k,
    mean_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_binary_retrieval_metrics_match_hand_calculation():
    ranking = ["noise", "high", "partial"]
    relevance = {"high": 2, "partial": 1}

    assert hit_at_k(ranking, relevance, 3) == 1.0
    assert recall_at_k(ranking, relevance, 3) == 1.0
    assert reciprocal_rank_at_k(ranking, relevance, 3) == 0.5


def test_ndcg_uses_graded_relevance_and_ranking_order():
    relevance = {"high": 2, "partial": 1}
    actual_dcg = 1 / math.log2(2) + 3 / math.log2(3)
    ideal_dcg = 3 / math.log2(2) + 1 / math.log2(3)

    assert ndcg_at_k(["partial", "high"], relevance, 2) == pytest.approx(
        actual_dcg / ideal_dcg
    )
    assert ndcg_at_k(["high", "partial"], relevance, 2) == 1.0


def test_duplicate_results_only_count_once():
    relevance = {"a": 1, "b": 1}
    metrics = evaluate_ranking(["a", "a", "b"], relevance, 2)

    assert metrics == {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "ndcg": 1.0}


def test_no_hits_return_zero_metrics():
    metrics = evaluate_ranking(["x", "y"], {"a": 2}, 2)

    assert metrics == {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}


@pytest.mark.parametrize("k", [0, -1, True])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="正整数"):
        hit_at_k(["a"], {"a": 1}, k)


def test_gold_set_must_have_positive_relevance():
    with pytest.raises(ValueError, match="至少需要一个"):
        recall_at_k(["a"], {"a": 0}, 1)


def test_mean_metrics_requires_consistent_fields():
    assert mean_metrics([{"hit": 1.0}, {"hit": 0.0}]) == {"hit": 0.5}
    with pytest.raises(ValueError, match="相同字段"):
        mean_metrics([{"hit": 1.0}, {"recall": 1.0}])
