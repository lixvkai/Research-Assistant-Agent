"""RAG 细粒度可观测性测试：不加载真实模型，也不访问 Langfuse。"""

from __future__ import annotations

import contextlib
import json
import types

from rag import rag_engine
from rag.rag_engine import RAGEngine
from rag.vector_store import VectorStore


class FakeObservation:
    def __init__(self, name: str, options: dict):
        self.name = name
        self.options = options
        self.updates: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeVectorStore:
    collection = types.SimpleNamespace(name="research_papers")

    def search(self, query: str, top_k: int):
        assert query == "retrieval augmented generation RAG"
        assert top_k == 12
        return [
            {
                "id": "chunk-a",
                "content": "first chunk body",
                "metadata": {"source": "a.pdf"},
                "distance": 0.1,
                "similarity": 0.9,
                "recall_rank": 1,
            },
            {
                "id": "chunk-b",
                "content": "second chunk body",
                "metadata": {"source": "b.pdf"},
                "distance": 0.2,
                "similarity": 0.8,
                "recall_rank": 2,
            },
            {
                "id": "chunk-c",
                "content": "third chunk body",
                "metadata": {"source": "c.pdf"},
                "distance": 0.3,
                "similarity": 0.7,
                "recall_rank": 3,
            },
        ]


def _fake_response(content: str):
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)]
    )


def test_vector_store_search_exposes_distance_similarity_and_rank():
    store = VectorStore.__new__(VectorStore)
    store.collection = types.SimpleNamespace(
        query=lambda **_kwargs: {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["one", "two"]],
            "metadatas": [[{"source": "a.pdf"}, {"source": "b.pdf"}]],
            "distances": [[0.15, 0.4]],
        }
    )

    docs = store.search("query", top_k=2)

    assert docs[0]["id"] == "chunk-1"
    assert docs[0]["distance"] == 0.15
    assert docs[0]["similarity"] == 0.85
    assert docs[0]["recall_rank"] == 1
    assert docs[1]["recall_rank"] == 2


def test_retrieve_records_rewrite_recall_and_rerank_metrics(monkeypatch):
    observations: list[FakeObservation] = []

    @contextlib.contextmanager
    def fake_observe(name, **options):
        observation = FakeObservation(name, options)
        observations.append(observation)
        yield observation

    monkeypatch.setattr(rag_engine, "observe_operation", fake_observe)
    monkeypatch.setattr(
        rag_engine,
        "chat",
        lambda **_kwargs: _fake_response("retrieval augmented generation RAG"),
    )
    monkeypatch.setattr(
        rag_engine,
        "_get_reranker",
        lambda: types.SimpleNamespace(predict=lambda _pairs: [0.1, 2.0, -1.0]),
    )

    engine = RAGEngine.__new__(RAGEngine)
    engine.vector_store = FakeVectorStore()
    result = engine.retrieve("什么是 RAG", top_k=3)

    by_name = {observation.name: observation for observation in observations}
    assert list(by_name) == [
        "rag.retrieve-context",
        "rag.rewrite-query",
        "rag.search-vectors",
        "rag.rerank-results",
    ]
    assert by_name["rag.search-vectors"].options["as_type"] == "retriever"
    assert by_name["rag.rerank-results"].options["as_type"] == "chain"

    search_output = by_name["rag.search-vectors"].updates[-1]["output"]
    assert search_output["candidate_count"] == 3
    assert search_output["candidates"][0] == {
        "document_id": "chunk-a",
        "source": "a.pdf",
        "recall_rank": 1,
        "distance": 0.1,
        "similarity": 0.9,
    }

    rerank_output = by_name["rag.rerank-results"].updates[-1]["output"]
    assert rerank_output["returned_count"] == 3
    assert rerank_output["candidates"][0]["document_id"] == "chunk-b"
    assert rerank_output["candidates"][0]["recall_rank"] == 2
    assert rerank_output["candidates"][0]["rerank_rank"] == 1
    assert "rerank_raw_score" in rerank_output["candidates"][0]
    assert "rerank_score" in rerank_output["candidates"][0]

    serialized_observations = json.dumps(
        [observation.updates for observation in observations], ensure_ascii=False
    )
    assert "first chunk body" not in serialized_observations
    assert result.startswith("[来源1: b.pdf")


def test_rerank_failure_records_fallback_and_keeps_recall_order(monkeypatch):
    observations: list[FakeObservation] = []

    @contextlib.contextmanager
    def fake_observe(name, **options):
        observation = FakeObservation(name, options)
        observations.append(observation)
        yield observation

    class BrokenReranker:
        def predict(self, _pairs):
            raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(rag_engine, "observe_operation", fake_observe)
    monkeypatch.setattr(rag_engine, "_get_reranker", lambda: BrokenReranker())

    engine = RAGEngine.__new__(RAGEngine)
    docs = FakeVectorStore().search("retrieval augmented generation RAG", 12)
    results = engine._rerank("query", docs, top_k=2)

    assert [doc["id"] for doc in results] == ["chunk-a", "chunk-b"]
    update = observations[-1].updates[-1]
    assert update["level"] == "WARNING"
    assert update["metadata"]["fallback"] is True
    assert update["output"]["rerank_applied"] is False


def test_rewrite_failure_records_fallback(monkeypatch):
    observations: list[FakeObservation] = []

    @contextlib.contextmanager
    def fake_observe(name, **options):
        observation = FakeObservation(name, options)
        observations.append(observation)
        yield observation

    monkeypatch.setattr(rag_engine, "observe_operation", fake_observe)
    monkeypatch.setattr(
        rag_engine,
        "chat",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    engine = RAGEngine.__new__(RAGEngine)
    rewritten = engine._rewrite_query("原始查询")

    assert rewritten == "原始查询"
    update = observations[-1].updates[-1]
    assert update["level"] == "WARNING"
    assert update["metadata"] == {"changed": False, "fallback": True}

