"""Offline runner and structured RAG adapter tests."""

import json

import pytest

from evaluation.rag.dataset import EvaluationQuery
from evaluation.rag.report import compare_summaries, write_experiment_outputs, write_run_overview
from evaluation.rag.runner import ExperimentConfig, run_experiments
from rag.rag_engine import RAGEngine, RerankOutcome, RewriteOutcome


def _query() -> EvaluationQuery:
    return EvaluationQuery.from_dict(
        {
            "query_id": "q001",
            "query": "什么是 RAG？",
            "category": "concept",
            "difficulty": "easy",
            "relevant_chunks": [
                {
                    "document_id": "relevant",
                    "source": "paper.pdf",
                    "relevance": 2,
                    "text_anchor": "retrieval augmented generation",
                }
            ],
        }
    )


class FakeEvaluationEngine:
    def __init__(self):
        self.live_rewrite_calls = 0
        self.overrides = []

    def retrieve_structured(self, query, **options):
        override = options["rewrite_override"]
        self.overrides.append(override)
        if options["rewrite"] and override is None:
            self.live_rewrite_calls += 1
        retrieval_query = override or (f"{query} rewritten" if options["rewrite"] else query)
        candidates = [
            {
                "id": "noise",
                "metadata": {"source": "other.pdf"},
                "distance": 0.1,
                "recall_rank": 1,
            },
            {
                "id": "relevant",
                "metadata": {"source": "paper.pdf"},
                "distance": 0.2,
                "recall_rank": 2,
            },
        ]
        results = list(reversed(candidates)) if options["rerank"] else candidates
        return {
            "retrieval_query": retrieval_query,
            "rewrite_changed": options["rewrite"],
            "rewrite_fallback": False,
            "rerank_applied": options["rerank"],
            "rerank_fallback": False,
            "candidate_count": len(candidates),
            "returned_count": len(results),
            "candidates": candidates,
            "results": results,
            "latency_ms": {"rewrite": 1.0, "search": 2.0, "rerank": 3.0, "total": 6.0},
        }


def test_experiments_reuse_rewrite_and_measure_rerank_gain():
    engine = FakeEvaluationEngine()
    configs = [
        ExperimentConfig("rewrite_vector", True, False, 2, 2),
        ExperimentConfig("full_pipeline", True, True, 2, 2),
    ]

    results = run_experiments(engine, [_query()], configs)

    assert engine.live_rewrite_calls == 1
    assert engine.overrides == [None, "什么是 RAG？ rewritten"]
    assert results[0]["summary"]["metrics"]["mrr_at_2"] == 0.5
    assert results[1]["summary"]["metrics"]["mrr_at_2"] == 1.0


def test_runner_keeps_query_errors_in_report():
    class BrokenEngine:
        def retrieve_structured(self, *_args, **_kwargs):
            raise RuntimeError("offline")

    result = run_experiments(
        BrokenEngine(),
        [_query()],
        [ExperimentConfig("broken", False, False, 2, 1)],
    )[0]

    assert result["summary"]["error_count"] == 1
    assert result["query_results"][0]["failure_labels"] == ["execution_error"]


def test_report_writes_compact_artifacts(tmp_path):
    result = run_experiments(
        FakeEvaluationEngine(),
        [_query()],
        [ExperimentConfig("baseline", False, False, 2, 2)],
    )[0]

    paths = write_experiment_outputs(result, tmp_path)

    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["experiment"] == "baseline"
    assert "retrieval augmented generation" in paths["details"].read_text(encoding="utf-8")
    assert paths["failures"].is_file()


def test_compare_summaries_reports_metric_and_latency_deltas():
    comparison = compare_summaries(
        {"experiment": "a", "metrics": {"mrr": 0.5}, "latency_ms": {"p50": 10}},
        {"experiment": "b", "metrics": {"mrr": 0.7}, "latency_ms": {"p50": 12}},
    )

    assert comparison["metric_deltas"] == {"mrr": 0.2}
    assert comparison["latency_ms_deltas"] == {"p50": 2.0}


def test_run_overview_marks_synthetic_dataset(tmp_path):
    path = write_run_overview(
        [
            {
                "experiment": "baseline",
                "metrics": {
                    "candidate_recall_at_20": 0.5,
                    "hit_at_5": 0.4,
                    "mrr_at_5": 0.3,
                    "ndcg_at_5": 0.35,
                },
                "latency_ms": {"p50": 10, "p95": 20},
            }
        ],
        tmp_path,
        dataset_type="synthetic",
    )

    content = path.read_text(encoding="utf-8")
    assert "synthetic" in content
    assert "| baseline | 0.500 | 0.400 |" in content


def test_structured_retrieval_can_disable_rewrite_and_rerank():
    engine = RAGEngine.__new__(RAGEngine)
    engine.vector_store = type(
        "Store",
        (),
        {
            "search": lambda self, query, top_k: [
                {
                    "id": "a",
                    "content": "body",
                    "metadata": {"source": "a.pdf"},
                    "distance": 0.2,
                    "recall_rank": 1,
                }
            ]
        },
    )()

    result = engine.retrieve_structured(
        "query", candidate_k=2, final_k=1, rewrite=False, rerank=False
    )

    assert result["retrieval_query"] == "query"
    assert result["rewrite_enabled"] is False
    assert result["rerank_enabled"] is False
    assert result["results"][0]["final_rank"] == 1


def test_structured_retrieval_preserves_recall_order_snapshot(monkeypatch):
    engine = RAGEngine.__new__(RAGEngine)
    engine.vector_store = type(
        "Store",
        (),
        {
            "search": lambda self, query, top_k: [
                {"id": "a", "content": "a", "metadata": {}, "recall_rank": 1},
                {"id": "b", "content": "b", "metadata": {}, "recall_rank": 2},
            ]
        },
    )()
    monkeypatch.setattr(
        engine,
        "_rewrite_query_with_details",
        lambda query: RewriteOutcome(f"{query} academic", True, False),
    )
    monkeypatch.setattr(
        engine,
        "_rerank_with_details",
        lambda query, docs, top_k: RerankOutcome(list(reversed(docs)), True, False),
    )

    result = engine.retrieve_structured(
        "query", candidate_k=2, final_k=2, rewrite=True, rerank=True
    )

    assert [doc["id"] for doc in result["candidates"]] == ["a", "b"]
    assert [doc["id"] for doc in result["results"]] == ["b", "a"]
    assert result["rewrite_changed"] is True
    assert result["rerank_applied"] is True


@pytest.mark.parametrize(
    "options",
    [
        {"candidate_k": 0, "final_k": 1},
        {"candidate_k": 1, "final_k": 2},
    ],
)
def test_structured_retrieval_rejects_invalid_sizes(options):
    engine = RAGEngine.__new__(RAGEngine)
    with pytest.raises(ValueError):
        engine.retrieve_structured("query", rewrite=False, rerank=False, **options)
