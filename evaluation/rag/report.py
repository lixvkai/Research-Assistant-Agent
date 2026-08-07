"""Write local RAG evaluation results and compare summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_experiment_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write summary, query details, and failure cases for one experiment."""
    directory = Path(output_dir)
    name = result["summary"]["experiment"]
    summary_path = directory / f"{name}_summary.json"
    details_path = directory / f"{name}_query_details.jsonl"
    failures_path = directory / f"{name}_failure_cases.md"

    _write_json(summary_path, result["summary"])
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as handle:
        for row in result["query_results"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failure_rows = [
        row
        for row in result["query_results"]
        if row.get("failure_labels") or row.get("error")
    ]
    lines = [
        f"# {name} 失败案例",
        "",
        f"共 {len(failure_rows)} 条需要检查。",
        "",
    ]
    for row in failure_rows:
        labels = ", ".join(row.get("failure_labels") or []) or "unknown"
        lines.extend(
            [
                f"## {row['query_id']} · {labels}",
                "",
                f"- Query：{row['query']}",
                f"- Error：{row.get('error') or '无'}",
                "",
            ]
        )
    failures_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary": summary_path,
        "details": details_path,
        "failures": failures_path,
    }


def compare_summaries(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Compare aggregate metric and latency fields from two summaries."""
    baseline_metrics = baseline.get("metrics") or {}
    current_metrics = current.get("metrics") or {}
    shared_metrics = sorted(set(baseline_metrics).intersection(current_metrics))
    metric_deltas = {
        key: round(float(current_metrics[key]) - float(baseline_metrics[key]), 6)
        for key in shared_metrics
    }
    baseline_latency = baseline.get("latency_ms") or {}
    current_latency = current.get("latency_ms") or {}
    shared_latency = sorted(set(baseline_latency).intersection(current_latency))
    latency_deltas = {
        key: round(float(current_latency[key]) - float(baseline_latency[key]), 3)
        for key in shared_latency
    }
    return {
        "baseline": baseline.get("experiment"),
        "current": current.get("experiment"),
        "metric_deltas": metric_deltas,
        "latency_ms_deltas": latency_deltas,
    }


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_run_overview(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    dataset_type: str = "unknown",
) -> Path:
    """Write one human-readable comparison table for a complete run."""
    directory = Path(output_dir)
    path = directory / "evaluation_overview.md"
    lines = [
        "# RAG 检索评测汇总",
        "",
        f"- 数据集类型：{dataset_type}",
        "- 评测范围：查询改写、向量召回、Cross-Encoder rerank",
        "- 不包含：最终答案质量、LLM Judge、Langfuse 评测",
        "",
        "| 实验 | Recall@20 | Hit@5 | MRR@5 | nDCG@5 | P50 延迟 | P95 延迟 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        metrics = summary.get("metrics") or {}
        latency = summary.get("latency_ms") or {}
        lines.append(
            "| {name} | {recall:.3f} | {hit:.3f} | {mrr:.3f} | {ndcg:.3f} | {p50:.1f} ms | {p95:.1f} ms |".format(
                name=summary.get("experiment", "unknown"),
                recall=float(metrics.get("candidate_recall_at_20", 0)),
                hit=float(metrics.get("hit_at_5", 0)),
                mrr=float(metrics.get("mrr_at_5", 0)),
                ndcg=float(metrics.get("ndcg_at_5", 0)),
                p50=float(latency.get("p50", 0)),
                p95=float(latency.get("p95", 0)),
            )
        )
    lines.extend(
        [
            "",
            "> 合成数据集结果仅用于本项目内部对照和回归，不应等同于人工标注集上的客观质量结论。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
