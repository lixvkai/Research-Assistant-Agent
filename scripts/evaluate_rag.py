"""Command-line entry point for local, retrieval-only RAG evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local evaluation must never upload traces, even when the developer's .env
# enables Langfuse for the interactive application.
os.environ["LANGFUSE_ENABLED"] = "false"

from evaluation.rag.dataset import (  # noqa: E402
    DatasetValidationError,
    dataset_summary,
    load_dataset,
    validate_chunk_ids,
)
from evaluation.rag.generator import generate_dataset, write_generation  # noqa: E402
from evaluation.rag.report import (  # noqa: E402
    compare_summaries,
    load_summary,
    write_experiment_outputs,
    write_run_overview,
)
from evaluation.rag.runner import (  # noqa: E402
    default_experiments,
    run_experiments,
)
from evaluation.rag.translator import translate_rows, write_translated_dataset  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "data" / "rag_eval" / "gold_queries.jsonl"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "rag_eval"


def _dataset_path(value: str | None) -> Path:
    return Path(value).resolve() if value else DEFAULT_DATASET


def _validate(args: argparse.Namespace) -> int:
    queries = load_dataset(_dataset_path(args.dataset))
    summary = dataset_summary(queries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.schema_only:
        print("数据集 schema 校验通过（未检查本地 Chroma collection）。")
        return 0

    from rag.vector_store import VectorStore

    store = VectorStore()
    errors = validate_chunk_ids(queries, store.list_document_ids())
    if errors:
        print("Gold Dataset 与当前知识库不一致：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print("数据集 schema 与当前知识库文档块校验通过。")
    return 0


def _parse_source_counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        source, separator, raw_count = value.rpartition("=")
        if not separator or not source.strip():
            raise ValueError(f"来源数量格式错误：{value}，应为 文件名=数量")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"来源数量不是整数：{value}") from exc
        if count <= 0:
            raise ValueError(f"来源数量必须为正整数：{value}")
        if source.strip() in result:
            raise ValueError(f"来源重复：{source.strip()}")
        result[source.strip()] = count
    return result


def _generate(args: argparse.Namespace) -> int:
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    from core.llm import chat
    from rag.vector_store import VectorStore

    if not DEEPSEEK_API_KEY:
        raise ValueError("未配置 DEEPSEEK_API_KEY")
    source_counts = _parse_source_counts(args.source_count)
    store = VectorStore()
    dataset_path = _dataset_path(args.dataset)
    partial_path = dataset_path.with_suffix(dataset_path.suffix + ".partial")
    if dataset_path.exists() and not args.overwrite:
        raise FileExistsError(f"数据集已存在：{dataset_path}；如需替换请使用 --overwrite")
    if args.overwrite and partial_path.exists():
        partial_path.unlink()
    initial_rows = []
    if partial_path.exists():
        initial_rows = [
            json.loads(line)
            for line in partial_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"从检查点恢复 {len(initial_rows)} 条已生成问题：{partial_path}", flush=True)

    def call_deepseek(messages):
        response = chat(
            messages,
            model=DEEPSEEK_MODEL,
            temperature=0.2,
            observation_name="rag-eval-dataset-generation",
        )
        return response.choices[0].message.content or ""

    target_count = sum(source_counts.values())
    print(f"将通过 DeepSeek 生成 {target_count} 条合成检索问题；Langfuse 已关闭。", flush=True)
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    def save_accepted(row):
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[{int(row['query_id'][1:]):02d}/{target_count}] {row['query_id']} {row['query']}",
            flush=True,
        )

    result = generate_dataset(
        store.list_documents(),
        source_counts,
        call_deepseek,
        model=DEEPSEEK_MODEL,
        seed=args.seed,
        on_accept=save_accepted,
        initial_rows=initial_rows,
    )
    paths = write_generation(result, dataset_path, overwrite=args.overwrite)
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))
    print(f"数据集：{paths['dataset']}")
    print(f"清单：{paths['manifest']}")
    return 0


def _select_experiments(args: argparse.Namespace):
    configs = default_experiments(args.candidate_k, args.final_k)
    if args.experiment == "all":
        return configs
    return [config for config in configs if config.name == args.experiment]


def _translate(args: argparse.Namespace) -> int:
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    from core.llm import chat

    if not DEEPSEEK_API_KEY:
        raise ValueError("未配置 DEEPSEEK_API_KEY")
    dataset_path = _dataset_path(args.dataset)
    if not dataset_path.is_file():
        raise DatasetValidationError(f"评测数据集不存在：{dataset_path}")
    raw_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = set(args.source)
    checkpoint_path = dataset_path.with_suffix(dataset_path.suffix + ".translation.partial")
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {}
    )
    if checkpoint:
        print(f"从翻译检查点恢复 {len(checkpoint)} 条。", flush=True)

    def call_deepseek(messages):
        response = chat(
            messages,
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            observation_name="rag-eval-query-translation",
        )
        return response.choices[0].message.content or ""

    target_count = sum(
        bool(row.get("relevant_chunks"))
        and row["relevant_chunks"][0].get("source") in sources
        for row in raw_rows
    )

    def save_checkpoint(query_id, translated):
        checkpoint[query_id] = translated
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{len(checkpoint):02d}/{target_count}] {query_id} {translated}", flush=True)

    translated_rows, _ = translate_rows(
        raw_rows,
        sources,
        call_deepseek,
        checkpoint=checkpoint,
        on_accept=save_checkpoint,
    )
    manifest_path = dataset_path.with_name("corpus_manifest.json")
    paths = write_translated_dataset(
        translated_rows,
        dataset_path,
        manifest_path,
        translated_sources=sources,
        model=DEEPSEEK_MODEL,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"已替换数据集：{paths['dataset']}")
    print(f"原数据集备份：{paths['backup']}")
    return 0


def _run(args: argparse.Namespace) -> int:
    queries = load_dataset(_dataset_path(args.dataset))
    if not 30 <= len(queries) <= 50:
        print(
            f"提示：当前数据集有 {len(queries)} 条 Query，正式首版建议 30～50 条。",
            file=sys.stderr,
        )

    from rag.rag_engine import RAGEngine

    engine = RAGEngine()
    configs = _select_experiments(args)
    results = run_experiments(engine, queries, configs)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_REPORT_ROOT / run_id

    print(f"报告目录：{output_dir}")
    had_errors = False
    for result in results:
        paths = write_experiment_outputs(result, output_dir)
        summary = result["summary"]
        had_errors = had_errors or summary["error_count"] > 0
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"明细：{paths['details']}")
    manifest_path = _dataset_path(args.dataset).with_name("corpus_manifest.json")
    dataset_type = "unknown"
    if manifest_path.is_file():
        dataset_type = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "dataset_type", "unknown"
        )
    overview = write_run_overview(
        [result["summary"] for result in results],
        output_dir,
        dataset_type=dataset_type,
    )
    print(f"汇总：{overview}")
    return 2 if had_errors else 0


def _compare(args: argparse.Namespace) -> int:
    comparison = compare_summaries(
        load_summary(args.baseline),
        load_summary(args.current),
    )
    rendered = json.dumps(comparison, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="科研助手 RAG 本地检索评测（不评测最终生成答案）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="校验 Gold Dataset")
    validate_parser.add_argument("--dataset", help="JSONL 数据集路径")
    validate_parser.add_argument(
        "--schema-only",
        action="store_true",
        help="只检查 JSONL schema，不读取本地 Chroma collection",
    )
    validate_parser.set_defaults(handler=_validate)

    generate_parser = subparsers.add_parser("generate", help="使用 DeepSeek 生成合成检索数据集")
    generate_parser.add_argument("--dataset", help="JSONL 数据集路径")
    generate_parser.add_argument(
        "--source-count",
        action="append",
        required=True,
        help="按来源指定数量，可重复使用，例如 --source-count paper.pdf=20",
    )
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.set_defaults(handler=_generate)

    translate_parser = subparsers.add_parser("translate", help="转换指定来源的评测问题语言")
    translate_parser.add_argument("--dataset", help="JSONL 数据集路径")
    translate_parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="需要转换为英文的问题来源文件，可重复使用",
    )
    translate_parser.set_defaults(handler=_translate)

    run_parser = subparsers.add_parser("run", help="运行本地检索评测")
    run_parser.add_argument("--dataset", help="JSONL 数据集路径")
    run_parser.add_argument("--output-dir", help="报告输出目录")
    run_parser.add_argument(
        "--experiment",
        choices=(
            "all",
            "vector_baseline",
            "rewrite_vector",
            "vector_rerank",
            "full_pipeline",
        ),
        default="all",
    )
    run_parser.add_argument("--candidate-k", type=int, default=20)
    run_parser.add_argument("--final-k", type=int, default=5)
    run_parser.set_defaults(handler=_run)

    compare_parser = subparsers.add_parser("compare", help="比较两份汇总结果")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("current")
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(handler=_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except DatasetValidationError as exc:
        print(f"数据集校验失败：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"参数或数据错误：{exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
