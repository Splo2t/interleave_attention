from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET_ID = "dilab-cau/kobest-query-context-stress-v3"
SOURCE_DATASET_ID = "skt/kobest_v1"
DEFAULT_KOBEST_TASKS = ["boolq", "copa", "hellaswag", "sentineg", "wic"]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_from_doc(doc: dict[str, Any], fallback: Any = None) -> int:
    if fallback is not None:
        return int(fallback)
    if "gold" in doc:
        return int(doc["gold"])
    return int(doc["label"])


def _kobest_prompt_choices_gold(task: str, doc: dict[str, Any], *, fallback_gold: Any = None) -> tuple[str, list[str], int]:
    if task == "boolq":
        return (
            f'{doc["paragraph"]} 질문: {doc["question"]} 답변: ',
            ["아니오", "예"],
            _label_from_doc(doc, fallback_gold),
        )
    if task == "copa":
        connector = {"원인": " 왜냐하면", "결과": " 그래서"}[str(doc["question"]).strip()]
        return (
            f'{doc["premise"]} {connector}',
            [str(doc["alternative_1"]), str(doc["alternative_2"])],
            _label_from_doc(doc, fallback_gold),
        )
    if task == "hellaswag":
        if "query" in doc and "choices" in doc and "gold" in doc:
            return str(doc["query"]), [str(choice) for choice in doc["choices"]], _label_from_doc(doc, fallback_gold)
        return (
            f'문장: {doc["context"]}',
            [str(doc[f"ending_{idx}"]) for idx in range(1, 5)],
            _label_from_doc(doc, fallback_gold),
        )
    if task == "sentineg":
        return f'문장: {doc["sentence"]} 긍부정:', ["부정", "긍정"], _label_from_doc(doc, fallback_gold)
    if task == "wic":
        return (
            f'문장1: {doc["context_1"]} 문장2: {doc["context_2"]} 두 문장에서 {doc["word"]}가 같은 뜻으로 쓰였나?',
            ["아니오", "예"],
            _label_from_doc(doc, fallback_gold),
        )
    raise ValueError(f"Unsupported KoBEST task: {task}")


def _build_eval_row(row: dict[str, Any], *, row_field: str, query_kind: str) -> dict[str, Any]:
    task = str(row["source_subset"])
    prompt, choices, gold = _kobest_prompt_choices_gold(task, row[row_field], fallback_gold=row.get("answer_idx"))
    return {
        "prompt": prompt,
        "choices": choices,
        "gold": gold,
        "source_config": task,
        "source_dataset": row.get("source_dataset"),
        "source_index": row.get("row_idx"),
        "source_split": row.get("split"),
        "source_item_id": row.get("source_item_id"),
        "stress_id": row.get("stress_id"),
        "query_kind": query_kind,
        "original_row": row.get("original_row"),
        "stressed_row": row.get("stressed_row"),
        "stressed_field_names": row.get("stressed_field_names"),
        "field_stress_count": row.get("field_stress_count"),
        "field_stress": row.get("field_stress"),
        "label_preservation_proxy": row.get("label_preservation_proxy"),
        "review_status": row.get("review_status"),
        "review_label_preserved": row.get("review_label_preserved"),
        "review_label_natural": row.get("review_label_natural"),
        "review_notes": row.get("review_notes"),
    }


def _build_source_train_rows(task: str, *, cache_dir: str | None, split: str) -> list[dict[str, Any]]:
    ds = load_dataset(SOURCE_DATASET_ID, task, split=split, cache_dir=cache_dir)
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(ds):
        doc = dict(doc)
        prompt, choices, gold = _kobest_prompt_choices_gold(task, doc)
        rows.append(
            {
                "prompt": prompt,
                "choices": choices,
                "gold": gold,
                "source_config": task,
                "source_dataset": SOURCE_DATASET_ID,
                "source_index": idx,
                "source_split": split,
                "source_item_id": f"{task}:{split}:{idx}",
                "query_kind": "fewshot_original",
            }
        )
    return rows


def _materialize_eval_rows(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
    row_field: str,
    query_kind: str,
    approved_only: bool,
) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for row in rows:
        if approved_only and str(row.get("review_status") or "").strip().lower() != "approved":
            skipped += 1
            continue
        task = str(row.get("source_subset") or "").strip()
        if task not in DEFAULT_KOBEST_TASKS or not row.get(row_field):
            skipped += 1
            continue
        grouped[task].append(_build_eval_row(row, row_field=row_field, query_kind=query_kind))

    counts: dict[str, int] = {}
    for task in DEFAULT_KOBEST_TASKS:
        task_rows = grouped.get(task, [])
        _write_jsonl(output_root / "kobest" / task / "test.jsonl", task_rows)
        counts[task] = len(task_rows)

    return counts | {"_skipped": skipped}


def _materialize_train_rows(
    *,
    output_root: Path,
    tasks: list[str],
    cache_dir: str | None,
    train_split: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        rows = _build_source_train_rows(task, cache_dir=cache_dir, split=train_split)
        _write_jsonl(output_root / "kobest" / task / "train.jsonl", rows)
        counts[task] = len(rows)
    return counts


def _write_manifest(
    *,
    output_root: Path,
    query_kind: str,
    row_field: str,
    dataset_id: str,
    eval_counts: dict[str, int],
    train_counts: dict[str, int],
    approved_only: bool,
) -> None:
    task_counts = {task: eval_counts.get(task, 0) for task in DEFAULT_KOBEST_TASKS}
    manifest = {
        "dataset_id": dataset_id,
        "source_dataset_id": SOURCE_DATASET_ID,
        "query_kind": query_kind,
        "row_field": row_field,
        "approved_only": approved_only,
        "num_test_rows": sum(task_counts.values()),
        "num_train_rows": sum(train_counts.values()),
        "num_skipped": eval_counts.get("_skipped", 0),
        "tasks": task_counts,
        "train_tasks": train_counts,
        "kobest_tasks_arg": ",".join(DEFAULT_KOBEST_TASKS),
        "format": "krong_eval_variants.kobest_variant",
        "fewshot_note": (
            "The HF stress dataset contains stressed test examples only. "
            "train.jsonl is materialized from skt/kobest_v1 original train split for identical few-shot demonstrations."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "kobest_tasks.txt").write_text(",".join(DEFAULT_KOBEST_TASKS) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a Hugging Face KoBEST query-context stress dataset into kobest_variant JSONL layout."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-train-split", default="train")
    parser.add_argument(
        "--output-root",
        default="variant_benchmarks/kobest_query_context_stress_v3",
        help="Root for stressed-row variant data.",
    )
    parser.add_argument(
        "--original-output-root",
        default="variant_benchmarks/kobest_query_context_stress_v3_original",
        help="Root for paired original-row variant data. Use empty string to skip.",
    )
    parser.add_argument(
        "--approved-only",
        action="store_true",
        help="Keep only rows with review_status == approved. Default keeps all rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    original_output_root = Path(args.original_output_root).expanduser().resolve() if args.original_output_root else None
    for root in [output_root, original_output_root]:
        if root is not None and root.exists() and not args.overwrite:
            raise FileExistsError(f"Output root already exists: {root} (pass --overwrite)")

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=args.cache_dir)
    rows = [dict(row) for row in dataset]

    stressed_eval_counts = _materialize_eval_rows(
        rows,
        output_root=output_root,
        row_field="stressed_row",
        query_kind="stressed",
        approved_only=args.approved_only,
    )
    stressed_train_counts = _materialize_train_rows(
        output_root=output_root,
        tasks=DEFAULT_KOBEST_TASKS,
        cache_dir=args.cache_dir,
        train_split=args.source_train_split,
    )
    _write_manifest(
        output_root=output_root,
        query_kind="stressed",
        row_field="stressed_row",
        dataset_id=args.dataset_id,
        eval_counts=stressed_eval_counts,
        train_counts=stressed_train_counts,
        approved_only=args.approved_only,
    )
    print(f"[stressed] root={output_root}")
    print(f"[stressed] test_rows={sum(stressed_eval_counts.get(task, 0) for task in DEFAULT_KOBEST_TASKS)}")
    print(f"[stressed] train_rows={sum(stressed_train_counts.values())}")
    print(f"[stressed] tasks={','.join(DEFAULT_KOBEST_TASKS)}")

    if original_output_root is not None:
        original_eval_counts = _materialize_eval_rows(
            rows,
            output_root=original_output_root,
            row_field="original_row",
            query_kind="original",
            approved_only=args.approved_only,
        )
        original_train_counts = _materialize_train_rows(
            output_root=original_output_root,
            tasks=DEFAULT_KOBEST_TASKS,
            cache_dir=args.cache_dir,
            train_split=args.source_train_split,
        )
        _write_manifest(
            output_root=original_output_root,
            query_kind="original",
            row_field="original_row",
            dataset_id=args.dataset_id,
            eval_counts=original_eval_counts,
            train_counts=original_train_counts,
            approved_only=args.approved_only,
        )
        print(f"[original] root={original_output_root}")
        print(f"[original] test_rows={sum(original_eval_counts.get(task, 0) for task in DEFAULT_KOBEST_TASKS)}")
        print(f"[original] train_rows={sum(original_train_counts.values())}")
        print(f"[original] tasks={','.join(DEFAULT_KOBEST_TASKS)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
