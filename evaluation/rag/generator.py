"""Generate a synthetic retrieval dataset from the local Chroma corpus."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable


CATEGORIES = {"concept", "method", "comparison", "experiment", "contribution", "detail"}
DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class CorpusChunk:
    document_id: str
    source: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class GenerationResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def corpus_chunks(documents: Iterable[dict[str, Any]]) -> list[CorpusChunk]:
    """Normalize Chroma rows and discard empty or malformed chunks."""
    chunks: list[CorpusChunk] = []
    for document in documents:
        metadata = document.get("metadata") or {}
        content = str(document.get("content") or "").strip()
        source = str(metadata.get("source") or "").strip()
        chunk_index = metadata.get("chunk_index")
        document_id = str(document.get("id") or "").strip()
        if not document_id or not source or not content:
            continue
        try:
            index = int(chunk_index)
        except (TypeError, ValueError):
            continue
        chunks.append(CorpusChunk(document_id, source, index, content))
    return sorted(chunks, key=lambda item: (item.source, item.chunk_index))


def _candidate_order(chunks: list[CorpusChunk], count: int, seed: int) -> list[CorpusChunk]:
    """Prefer evenly spread chunks, then retain all remaining chunks as fallbacks."""
    if count <= 0:
        return []
    eligible = [chunk for chunk in chunks if len(chunk.content) >= 60]
    if len(eligible) < count:
        eligible = list(chunks)
    if len(eligible) < count:
        raise ValueError(f"{chunks[0].source if chunks else 'source'} 只有 {len(eligible)} 个可用块，无法生成 {count} 条")

    positions = [round((i + 0.5) * len(eligible) / count - 0.5) for i in range(count)]
    selected_indexes = list(dict.fromkeys(max(0, min(len(eligible) - 1, p)) for p in positions))
    selected = [eligible[index] for index in selected_indexes]
    remaining = [chunk for index, chunk in enumerate(eligible) if index not in set(selected_indexes)]
    random.Random(seed).shuffle(remaining)
    return selected + remaining


def _clean_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("DeepSeek 未返回 JSON 对象")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek 返回值不是 JSON 对象")
    return payload


def _normalized_question(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def _is_duplicate(question: str, accepted: Iterable[str]) -> bool:
    normalized = _normalized_question(question)
    if not normalized:
        return True
    for existing in accepted:
        other = _normalized_question(existing)
        if normalized == other or SequenceMatcher(None, normalized, other).ratio() >= 0.86:
            return True
    return False


def _prompt(target: CorpusChunk, previous: CorpusChunk | None, following: CorpusChunk | None) -> list[dict[str, str]]:
    context = []
    if previous:
        context.append(f"[PREVIOUS]\n{previous.content}")
    context.append(f"[TARGET]\n{target.content}")
    if following:
        context.append(f"[NEXT]\n{following.content}")
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的 RAG 检索评测数据生成器。只能依据提供的文本生成一个自然的中文检索问题。"
                "问题必须可由 TARGET 直接回答，不得提及‘本文片段/上述内容’，不得照抄完整句子，也不得引入外部知识。"
                "不要生成作者、发表年份、期刊会议、引用编号或参考文献条目的问题。"
                "如果 TARGET 主要是参考文献、目录、页眉页脚、残缺公式或缺乏可提问的实质信息，"
                "返回 {\"skip\":true,\"reason\":\"简短原因\"}。"
                "evidence_anchor 必须逐字摘自 TARGET，长度 8 到 80 个字符。"
                "adjacent_relevant 只能从 PREVIOUS、NEXT 中选择真正能辅助回答该问题的块；不相关则返回空数组。"
                "只返回单个 JSON 对象，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "JSON 字段：query、category、difficulty、evidence_anchor、adjacent_relevant。"
                "category 只能是 concept/method/comparison/experiment/contribution/detail；"
                "difficulty 只能是 easy/medium/hard；adjacent_relevant 是字符串数组。\n\n"
                + "\n\n".join(context)
            ),
        },
    ]


def _actual_anchor(content: str, proposed: str) -> str | None:
    """Locate an anchor despite PDF whitespace, punctuation and width variants."""
    compact_content: list[str] = []
    original_positions: list[int] = []
    for index, original_char in enumerate(content):
        for char in unicodedata.normalize("NFKC", original_char):
            if char.isalnum():
                compact_content.append(char.lower())
                original_positions.append(index)
    compact_anchor = "".join(
        char.lower()
        for char in unicodedata.normalize("NFKC", proposed)
        if char.isalnum()
    )
    start = "".join(compact_content).find(compact_anchor)
    if start < 0 or not compact_anchor:
        return None
    end = start + len(compact_anchor) - 1
    return content[original_positions[start] : original_positions[end] + 1].strip()


def _validate_payload(payload: dict[str, Any], target: CorpusChunk) -> tuple[str, str, str, str, set[str]]:
    if payload.get("skip") is True:
        raise ValueError(f"模型跳过低信息块：{str(payload.get('reason') or '未说明')[:80]}")
    query = str(payload.get("query") or "").strip()
    category = str(payload.get("category") or "").strip().lower()
    difficulty = str(payload.get("difficulty") or "").strip().lower()
    anchor = str(payload.get("evidence_anchor") or "").strip()
    adjacent = payload.get("adjacent_relevant")
    if not 6 <= len(query) <= 120:
        raise ValueError("问题长度不合格")
    if category not in CATEGORIES:
        raise ValueError("category 不合格")
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty 不合格")
    actual_anchor = _actual_anchor(target.content, anchor)
    if not 8 <= len(anchor) <= 80 or actual_anchor is None:
        raise ValueError("证据锚点不是 TARGET 中的原文")
    if not isinstance(adjacent, list) or any(item not in ("PREVIOUS", "NEXT") for item in adjacent):
        raise ValueError("adjacent_relevant 不合格")
    return query, category, difficulty, actual_anchor, set(adjacent)


def generate_dataset(
    documents: Iterable[dict[str, Any]],
    source_counts: dict[str, int],
    generate: Callable[[list[dict[str, str]]], str],
    *,
    model: str,
    seed: int = 42,
    max_attempt_multiplier: int = 4,
    on_accept: Callable[[dict[str, Any]], None] | None = None,
    initial_rows: Iterable[dict[str, Any]] = (),
) -> GenerationResult:
    """Generate, locally validate and label synthetic retrieval questions."""
    chunks = corpus_chunks(documents)
    by_source: dict[str, list[CorpusChunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    unknown = sorted(set(source_counts) - set(by_source))
    if unknown:
        raise ValueError(f"知识库中不存在来源：{', '.join(unknown)}")
    if any(count <= 0 for count in source_counts.values()):
        raise ValueError("每个来源的生成数量必须是正整数")

    rows: list[dict[str, Any]] = list(initial_rows)
    resumed_query_count = len(rows)
    accepted_questions: list[str] = [str(row.get("query") or "") for row in rows]
    used_target_ids = {
        str(row.get("relevant_chunks", [{}])[0].get("document_id") or "")
        for row in rows
        if row.get("relevant_chunks")
    }
    rejection_reasons: dict[str, int] = {}
    calls = 0

    for source, requested in source_counts.items():
        source_chunks = by_source[source]
        by_index = {chunk.chunk_index: chunk for chunk in source_chunks}
        candidates = [
            chunk
            for chunk in _candidate_order(source_chunks, requested, seed + len(rows))
            if chunk.document_id not in used_target_ids
        ]
        accepted_for_source = sum(
            bool(row.get("relevant_chunks"))
            and row["relevant_chunks"][0].get("source") == source
            for row in rows
        )
        maximum_attempts = min(len(candidates), max(requested * max_attempt_multiplier, requested + 80))
        for target in candidates[:maximum_attempts]:
            if accepted_for_source >= requested:
                break
            previous = by_index.get(target.chunk_index - 1)
            following = by_index.get(target.chunk_index + 1)
            try:
                calls += 1
                payload = _clean_json(generate(_prompt(target, previous, following)))
                query, category, difficulty, anchor, adjacent = _validate_payload(payload, target)
                if _is_duplicate(query, accepted_questions):
                    raise ValueError("问题与已接受问题重复")
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            relevant = [
                {
                    "document_id": target.document_id,
                    "source": source,
                    "relevance": 2,
                    "text_anchor": anchor,
                }
            ]
            if previous and "PREVIOUS" in adjacent:
                relevant.append({"document_id": previous.document_id, "source": source, "relevance": 1, "text_anchor": ""})
            if following and "NEXT" in adjacent:
                relevant.append({"document_id": following.document_id, "source": source, "relevance": 1, "text_anchor": ""})

            rows.append(
                {
                    "query_id": f"q{len(rows) + 1:03d}",
                    "query": query,
                    "category": category,
                    "difficulty": difficulty,
                    "relevant_chunks": relevant,
                    "notes": f"synthetic:{model}; target_chunk:{target.document_id}",
                }
            )
            accepted_questions.append(query)
            used_target_ids.add(target.document_id)
            accepted_for_source += 1
            if on_accept:
                on_accept(rows[-1])

        if accepted_for_source != requested:
            top_rejections = sorted(rejection_reasons.items(), key=lambda item: item[1], reverse=True)[:5]
            raise RuntimeError(
                f"{source} 仅生成 {accepted_for_source}/{requested} 条合格问题；"
                f"主要淘汰原因：{top_rejections}"
            )

    corpus_digest = hashlib.sha256(
        "\n".join(f"{chunk.document_id}:{hashlib.sha256(chunk.content.encode('utf-8')).hexdigest()}" for chunk in chunks).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_type": "synthetic",
        "generator": "deepseek",
        "model": model,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": seed,
        "source_counts": source_counts,
        "query_count": len(rows),
        "generation_calls": calls,
        "resumed_query_count": resumed_query_count,
        "rejection_reasons": rejection_reasons,
        "corpus": {"chunk_count": len(chunks), "sha256": corpus_digest},
    }
    return GenerationResult(rows, manifest)


def write_generation(result: GenerationResult, dataset_path: Path, *, overwrite: bool = False) -> dict[str, Path]:
    """Write JSONL and its provenance manifest without silently overwriting data."""
    manifest_path = dataset_path.with_name("corpus_manifest.json")
    existing = [path for path in (dataset_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出已存在：{', '.join(str(path) for path in existing)}；如需替换请使用 --overwrite")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result.rows) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(result.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"dataset": dataset_path, "manifest": manifest_path}
