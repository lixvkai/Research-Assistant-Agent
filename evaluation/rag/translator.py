"""Translate synthetic evaluation queries while preserving their gold labels."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable


_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def translation_prompt(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Translate the supplied academic retrieval question into natural English. "
                "Preserve all model names, dataset names, symbols, numbers, metrics and technical meaning. "
                "Do not answer, explain, expand or simplify the question. "
                "The output must contain no Chinese characters under any circumstance. "
                "Return only the translated question."
            ),
        },
        {"role": "user", "content": query},
    ]


def validate_english_query(value: str) -> str:
    query = value.strip().strip("`").strip()
    if not 6 <= len(query) <= 240:
        raise ValueError("英文问题长度不合格")
    if _CJK_RE.search(query):
        raise ValueError("翻译结果仍包含中文字符")
    if len(re.findall(r"[A-Za-z]", query)) < 4:
        raise ValueError("翻译结果缺少英文文本")
    return query


def translate_rows(
    rows: Iterable[dict[str, Any]],
    sources: set[str],
    translate: Callable[[list[dict[str, str]]], str],
    *,
    checkpoint: dict[str, str] | None = None,
    on_accept: Callable[[str, str], None] | None = None,
    max_attempts: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Translate selected sources without changing IDs, labels or metadata."""
    translated_by_id = dict(checkpoint or {})
    output: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    for original in rows:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        relevant = row.get("relevant_chunks") or []
        source = str(relevant[0].get("source") if relevant else "")
        query_id = str(row.get("query_id") or "")
        if source in sources:
            translated = translated_by_id.get(query_id)
            if translated is None:
                last_error: Exception | None = None
                for _attempt in range(max_attempts):
                    try:
                        translated = validate_english_query(
                            translate(translation_prompt(str(row.get("query") or "")))
                        )
                        break
                    except ValueError as exc:
                        last_error = exc
                else:
                    raise ValueError(f"{query_id} 连续 {max_attempts} 次翻译不合格：{last_error}")
                translated_by_id[query_id] = translated
                if on_accept:
                    on_accept(query_id, translated)
            else:
                translated = validate_english_query(translated)
            row["query"] = translated
            row["notes"] = f"{row.get('notes', '')}; query_language:en".strip("; ")
        else:
            row["notes"] = f"{row.get('notes', '')}; query_language:zh".strip("; ")

        normalized = re.sub(r"[\W_]+", "", str(row.get("query") or "")).lower()
        if normalized in seen_queries:
            raise ValueError(f"转换后问题重复：{query_id}")
        seen_queries.add(normalized)
        output.append(row)
    return output, translated_by_id


def write_translated_dataset(
    rows: list[dict[str, Any]],
    dataset_path: Path,
    manifest_path: Path,
    *,
    translated_sources: Iterable[str],
    model: str,
) -> dict[str, Path]:
    """Back up the current local dataset, then atomically replace it."""
    backup_path = dataset_path.with_name(f"{dataset_path.stem}.before_english{dataset_path.suffix}")
    if not backup_path.exists():
        backup_path.write_bytes(dataset_path.read_bytes())

    temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    temporary_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(dataset_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "translated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "translation_model": model,
            "translated_sources": sorted(translated_sources),
            "query_languages": {
                "en": sum("query_language:en" in str(row.get("notes")) for row in rows),
                "zh": sum("query_language:zh" in str(row.get("notes")) for row in rows),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"dataset": dataset_path, "backup": backup_path, "manifest": manifest_path}
