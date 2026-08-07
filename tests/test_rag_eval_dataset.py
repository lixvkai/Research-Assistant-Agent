"""Gold Dataset schema and corpus validation tests."""

import json

import pytest

from evaluation.rag.dataset import (
    DatasetValidationError,
    dataset_summary,
    load_dataset,
    validate_chunk_ids,
)


def _query(query_id="q001", relevance=2):
    return {
        "query_id": query_id,
        "query": "该方法如何处理遮挡？",
        "category": "method",
        "difficulty": "medium",
        "relevant_chunks": [
            {
                "document_id": "paper.pdf_chunk_1",
                "source": "paper.pdf",
                "relevance": relevance,
                "text_anchor": "occlusion-aware",
            }
        ],
        "notes": "方法章节",
    }


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_dataset_and_summary(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [_query()])

    queries = load_dataset(path)

    assert queries[0].query_id == "q001"
    assert queries[0].relevance_by_id == {"paper.pdf_chunk_1": 2}
    assert dataset_summary(queries) == {
        "query_count": 1,
        "relevant_chunk_count": 1,
        "categories": {"method": 1},
        "difficulties": {"medium": 1},
    }


def test_duplicate_query_id_is_rejected(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [_query(), _query()])

    with pytest.raises(DatasetValidationError, match="query_id 重复"):
        load_dataset(path)


@pytest.mark.parametrize("relevance", [0, 3, "2", True])
def test_invalid_relevance_is_rejected(tmp_path, relevance):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [_query(relevance=relevance)])

    with pytest.raises(DatasetValidationError, match="relevance"):
        load_dataset(path)


def test_missing_gold_chunk_is_reported(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [_query()])
    queries = load_dataset(path)

    assert validate_chunk_ids(queries, {"other_chunk"}) == [
        "q001: 找不到 paper.pdf_chunk_1（paper.pdf）"
    ]
    assert validate_chunk_ids(queries, {"paper.pdf_chunk_1"}) == []
