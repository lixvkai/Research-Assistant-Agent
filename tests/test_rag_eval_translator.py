"""Offline tests for evaluation-query translation."""

from evaluation.rag.translator import translate_rows, validate_english_query


def _row(query_id, source, query):
    return {
        "query_id": query_id,
        "query": query,
        "category": "method",
        "difficulty": "easy",
        "relevant_chunks": [
            {
                "document_id": f"{source}_chunk_1",
                "source": source,
                "relevance": 2,
                "text_anchor": "anchor text",
            }
        ],
        "notes": "synthetic:test",
    }


def test_translate_selected_source_preserves_gold_and_other_language():
    rows = [
        _row("q001", "paper.pdf", "该方法有什么优势？"),
        _row("q002", "chinese-paper.pdf", "该模型有什么优势？"),
    ]

    translated, checkpoint = translate_rows(
        rows,
        {"paper.pdf"},
        lambda _messages: "What advantages does the method provide?",
    )

    assert translated[0]["query"] == "What advantages does the method provide?"
    assert translated[0]["relevant_chunks"] == rows[0]["relevant_chunks"]
    assert "query_language:en" in translated[0]["notes"]
    assert translated[1]["query"] == rows[1]["query"]
    assert "query_language:zh" in translated[1]["notes"]
    assert checkpoint == {"q001": "What advantages does the method provide?"}


def test_translation_checkpoint_avoids_model_call():
    translated, _ = translate_rows(
        [_row("q001", "paper.pdf", "该方法有什么优势？")],
        {"paper.pdf"},
        lambda _messages: (_ for _ in ()).throw(AssertionError("should not call")),
        checkpoint={"q001": "What advantages does the method provide?"},
    )
    assert translated[0]["query"].startswith("What advantages")


def test_english_validation_rejects_chinese_output():
    try:
        validate_english_query("This result 仍然包含中文")
    except ValueError as exc:
        assert "中文" in str(exc)
    else:
        raise AssertionError("Chinese output should be rejected")
