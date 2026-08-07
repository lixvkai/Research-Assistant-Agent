"""Gold Dataset schema and validation for local RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset is malformed or stale."""


def _required_text(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{context}.{key} 必须是非空字符串")
    return value.strip()


@dataclass(frozen=True)
class RelevantChunk:
    """A manually labelled relevant document chunk."""

    document_id: str
    source: str
    relevance: int
    text_anchor: str = ""

    @classmethod
    def from_dict(cls, data: Any, context: str) -> "RelevantChunk":
        if not isinstance(data, dict):
            raise DatasetValidationError(f"{context} 必须是对象")
        relevance = data.get("relevance")
        if not isinstance(relevance, int) or isinstance(relevance, bool):
            raise DatasetValidationError(f"{context}.relevance 必须是整数")
        if relevance not in (1, 2):
            raise DatasetValidationError(
                f"{context}.relevance 只能是 1（部分相关）或 2（高度相关）"
            )
        text_anchor = data.get("text_anchor", "")
        if not isinstance(text_anchor, str):
            raise DatasetValidationError(f"{context}.text_anchor 必须是字符串")
        return cls(
            document_id=_required_text(data, "document_id", context),
            source=_required_text(data, "source", context),
            relevance=relevance,
            text_anchor=text_anchor.strip(),
        )


@dataclass(frozen=True)
class EvaluationQuery:
    """One query and its manually labelled relevant chunks."""

    query_id: str
    query: str
    category: str
    difficulty: str
    relevant_chunks: tuple[RelevantChunk, ...]
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Any, context: str = "query") -> "EvaluationQuery":
        if not isinstance(data, dict):
            raise DatasetValidationError(f"{context} 必须是对象")
        raw_chunks = data.get("relevant_chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise DatasetValidationError(f"{context}.relevant_chunks 至少需要一项")
        chunks = tuple(
            RelevantChunk.from_dict(item, f"{context}.relevant_chunks[{index}]")
            for index, item in enumerate(raw_chunks)
        )
        document_ids = [chunk.document_id for chunk in chunks]
        if len(document_ids) != len(set(document_ids)):
            raise DatasetValidationError(f"{context} 包含重复 document_id")

        notes = data.get("notes", "")
        if not isinstance(notes, str):
            raise DatasetValidationError(f"{context}.notes 必须是字符串")
        return cls(
            query_id=_required_text(data, "query_id", context),
            query=_required_text(data, "query", context),
            category=_required_text(data, "category", context),
            difficulty=_required_text(data, "difficulty", context),
            relevant_chunks=chunks,
            notes=notes.strip(),
        )

    @property
    def relevance_by_id(self) -> dict[str, int]:
        return {
            chunk.document_id: chunk.relevance
            for chunk in self.relevant_chunks
        }


def load_dataset(path: str | Path) -> list[EvaluationQuery]:
    """Load and validate a UTF-8 JSONL Gold Dataset."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetValidationError(f"评测数据集不存在：{dataset_path}")

    queries: list[EvaluationQuery] = []
    query_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"第 {line_number} 行不是合法 JSON：{exc.msg}"
                ) from exc
            query = EvaluationQuery.from_dict(payload, f"第 {line_number} 行")
            if query.query_id in query_ids:
                raise DatasetValidationError(f"query_id 重复：{query.query_id}")
            query_ids.add(query.query_id)
            queries.append(query)

    if not queries:
        raise DatasetValidationError("评测数据集不能为空")
    return queries


def validate_chunk_ids(
    queries: Iterable[EvaluationQuery],
    available_document_ids: Iterable[str],
) -> list[str]:
    """Return human-readable errors for Gold chunks absent from the corpus."""
    available = set(available_document_ids)
    errors: list[str] = []
    for query in queries:
        for chunk in query.relevant_chunks:
            if chunk.document_id not in available:
                errors.append(
                    f"{query.query_id}: 找不到 {chunk.document_id}（{chunk.source}）"
                )
    return errors


def dataset_summary(queries: Iterable[EvaluationQuery]) -> dict[str, Any]:
    """Build a compact summary without exposing document contents."""
    items = list(queries)
    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    relevant_chunk_count = 0
    for query in items:
        categories[query.category] = categories.get(query.category, 0) + 1
        difficulties[query.difficulty] = difficulties.get(query.difficulty, 0) + 1
        relevant_chunk_count += len(query.relevant_chunks)
    return {
        "query_count": len(items),
        "relevant_chunk_count": relevant_chunk_count,
        "categories": dict(sorted(categories.items())),
        "difficulties": dict(sorted(difficulties.items())),
    }
