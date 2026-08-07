"""Offline tests for synthetic RAG evaluation dataset generation."""

import json

import pytest

from evaluation.rag.generator import generate_dataset, write_generation


def _documents(count=5):
    return [
        {
            "id": f"paper.pdf_chunk_{index}",
            "content": f"第 {index} 个文档块包含足够长的测试内容，用于验证问题生成和证据锚点。核心方法采用时空注意力机制。",
            "metadata": {"source": "paper.pdf", "chunk_index": index},
        }
        for index in range(count)
    ]


def test_generate_dataset_labels_target_and_selected_neighbor():
    calls = 0

    def fake_generate(messages):
        nonlocal calls
        calls += 1
        target = messages[1]["content"].split("[TARGET]\n", 1)[1].split("\n\n", 1)[0]
        anchor = "核心方法采用时空注意力机制"
        assert anchor in target
        questions = [
            "文档中的核心时空注意力机制是什么？",
            "该方法如何利用注意力处理跨帧信息？",
        ]
        return json.dumps(
            {
                "query": questions[calls - 1],
                "category": "method",
                "difficulty": "easy",
                "evidence_anchor": anchor,
                "adjacent_relevant": ["NEXT"],
            },
            ensure_ascii=False,
        )

    result = generate_dataset(
        _documents(), {"paper.pdf": 2}, fake_generate, model="deepseek-chat"
    )

    assert len(result.rows) == 2
    assert result.manifest["dataset_type"] == "synthetic"
    assert result.manifest["generation_calls"] == 2
    assert result.rows[0]["relevant_chunks"][0]["relevance"] == 2
    assert result.rows[0]["relevant_chunks"][1]["relevance"] == 1


def test_invalid_anchor_is_rejected_and_next_candidate_is_used():
    calls = 0

    def fake_generate(_messages):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "query": f"这是第 {calls} 个有效问题吗？",
                "category": "detail",
                "difficulty": "easy",
                "evidence_anchor": "不存在于目标块中的证据文本",
                "adjacent_relevant": [],
            },
            ensure_ascii=False,
        ) if calls == 1 else json.dumps(
            {
                "query": "该文档采用了什么核心机制？",
                "category": "method",
                "difficulty": "easy",
                "evidence_anchor": "核心方法采用时空注意力机制",
                "adjacent_relevant": [],
            },
            ensure_ascii=False,
        )

    result = generate_dataset(_documents(), {"paper.pdf": 1}, fake_generate, model="test")

    assert calls == 2
    assert result.manifest["generation_calls"] == 2
    assert result.manifest["rejection_reasons"]


def test_write_generation_requires_explicit_overwrite(tmp_path):
    result = generate_dataset(
        _documents(),
        {"paper.pdf": 1},
        lambda _messages: json.dumps(
            {
                "query": "该文档采用了什么核心机制？",
                "category": "method",
                "difficulty": "easy",
                "evidence_anchor": "核心方法采用时空注意力机制",
                "adjacent_relevant": [],
            },
            ensure_ascii=False,
        ),
        model="test",
    )
    path = tmp_path / "gold.jsonl"
    write_generation(result, path)

    with pytest.raises(FileExistsError):
        write_generation(result, path)


def test_generation_resumes_without_regenerating_completed_source():
    initial = {
        "query_id": "q001",
        "query": "该文档采用了什么核心机制？",
        "category": "method",
        "difficulty": "easy",
        "relevant_chunks": [
            {
                "document_id": "paper.pdf_chunk_0",
                "source": "paper.pdf",
                "relevance": 2,
                "text_anchor": "核心方法采用时空注意力机制",
            }
        ],
        "notes": "synthetic:test; target_chunk:paper.pdf_chunk_0",
    }

    result = generate_dataset(
        _documents(),
        {"paper.pdf": 1},
        lambda _messages: pytest.fail("已完成来源不应再次调用模型"),
        model="test",
        initial_rows=[initial],
    )

    assert result.rows == [initial]
    assert result.manifest["resumed_query_count"] == 1
    assert result.manifest["generation_calls"] == 0
