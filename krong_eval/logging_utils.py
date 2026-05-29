from __future__ import annotations

import csv
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .benchmarks.constants import DEFAULT_KOBEST_TASKS

TASK_OVERVIEW_ORDER = (
    "mmlu",
    "kmmlu",
    "kobest",
    "csatqa",
    "click",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "korean_rerank",
)


def _slugify(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    text = text.replace(os.sep, "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_.")
    if not text:
        return "item"
    return text[:max_len]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _default_log_root() -> str:
    return str(Path(__file__).resolve().parent.parent / "logs")


def _infer_checkpoint_step(ckpt_name: str) -> str:
    match = re.search(r"(?:checkpoint|ckpt|step)[-_]?(\d+)$", ckpt_name or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _resolve_model_group(log_group: str, model_arch: str, ckpt_path: str, model_label: str) -> str:
    if log_group in {"krong", "kormo", "others"}:
        return log_group
    text = " ".join([model_arch or "", ckpt_path or "", model_label or ""]).lower()
    if "kormo" in text:
        return "kormo"
    if "krong" in text:
        return "krong"
    return "others"


def _effective_k_shot(args) -> int:
    # CSATQA's public converted parquet currently exposes test split only, so
    # the benchmark runs as 0-shot even if the global default k_shot is 5.
    # CLIcK supports few-shot prompts in our evaluator, so keep
    # its logged shot count aligned with args.k_shot.
    if getattr(args, "task", "") == "csatqa":
        return 0
    return int(getattr(args, "k_shot", 0) or 0)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, (list, tuple)):
        return "|".join(str(x) for x in value)
    return str(value)


def _read_csv_rows(path: str) -> tuple[list[dict[str, str]], list[str]]:
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _merge_fieldnames(*field_groups: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for group in field_groups:
        for field in group:
            if field and field not in seen:
                seen.add(field)
                out.append(field)
    return out


def _write_csv_rows(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _append_csv_row(path: str, row: dict[str, Any], preferred_fieldnames: list[str]) -> None:
    rows, existing_fieldnames = _read_csv_rows(path)
    rows.append(row)
    fieldnames = _merge_fieldnames(existing_fieldnames, preferred_fieldnames, list(row.keys()))
    _write_csv_rows(path, rows, fieldnames)


def _upsert_csv_row(
    path: str,
    row: dict[str, Any],
    *,
    key_fields: list[str],
    preferred_fieldnames: list[str],
) -> None:
    rows, existing_fieldnames = _read_csv_rows(path)
    key = tuple(_csv_value(row.get(field, "")) for field in key_fields)
    found = False
    for idx, existing in enumerate(rows):
        existing_key = tuple(existing.get(field, "") for field in key_fields)
        if existing_key == key:
            merged = dict(existing)
            for key_name, value in row.items():
                merged[key_name] = _csv_value(value)
            rows[idx] = merged
            found = True
            break
    if not found:
        rows.append(row)
    fieldnames = _merge_fieldnames(existing_fieldnames, preferred_fieldnames, list(row.keys()))
    _write_csv_rows(path, rows, fieldnames)


def _build_profile_tag(args, selected_items: Optional[list[str]]) -> str:
    parts = [f"k{_effective_k_shot(args)}"]
    add_bos = getattr(args, "add_bos", "auto")
    if add_bos != "auto":
        parts.append(f"addbos-{add_bos}")
    batch_scoring = getattr(args, "batch_scoring", "auto")
    if batch_scoring != "auto":
        parts.append(f"batch-{batch_scoring}")
    continuation_scoring = getattr(args, "continuation_scoring", "dynamic")
    if continuation_scoring != "dynamic":
        parts.append(f"cont-{continuation_scoring}")
    if args.use_chat_template:
        parts.append("chat")
    if args.enable_thinking:
        parts.append("thinking")
    if args.dec_max_len and args.dec_max_len > 0:
        parts.append(f"dec{args.dec_max_len}")
    if args.limit and args.limit > 0:
        parts.append(f"limit{args.limit}")

    if args.task in {"mmlu", "kmmlu", "csatqa", "click"}:
        parts.append(f"subjects{len(selected_items)}" if selected_items else "full")
    elif args.task in {"arc_easy", "arc_challenge", "hellaswag", "openbookqa"}:
        parts.append("full")
    elif args.task == "kobest":
        is_full = not selected_items or selected_items == DEFAULT_KOBEST_TASKS
        parts.append("full" if is_full and args.kobest_split == "test" else f"tasks{len(selected_items or [])}")
        if args.kobest_split != "test":
            parts.append(f"split-{_slugify(args.kobest_split, 24)}")

    if args.experiment_tag:
        parts.append(_slugify(args.experiment_tag, 32))
    return "__".join(parts)


def _build_scope_text(args, selected_items: Optional[list[str]]) -> str:
    bits = [f"k_shot={_effective_k_shot(args)}"]
    if args.task in {"mmlu", "kmmlu", "csatqa", "click"}:
        bits.append("subjects=ALL" if not selected_items else f"subjects={','.join(selected_items)}")
    elif args.task == "kobest":
        bits.append(f"tasks={','.join(selected_items or DEFAULT_KOBEST_TASKS)}")
        bits.append(f"split={args.kobest_split}")
    elif args.task in {"hellaswag", "openbookqa"}:
        benchmark_split = getattr(args, "benchmark_split", "") or "default"
        bits.append(f"split={benchmark_split}")
    if args.limit and args.limit > 0:
        bits.append(f"limit={args.limit}")
    if args.use_chat_template:
        bits.append("chat_template=1")
    if args.enable_thinking:
        bits.append("enable_thinking=1")
    if args.dec_max_len and args.dec_max_len > 0:
        bits.append(f"dec_max_len={args.dec_max_len}")
    add_bos = getattr(args, "add_bos", "auto")
    if add_bos != "auto":
        bits.append(f"add_bos={add_bos}")
    bits.append(f"batch_scoring={getattr(args, 'batch_scoring', 'auto')}")
    bits.append(f"continuation_scoring={getattr(args, 'continuation_scoring', 'dynamic')}")
    return "; ".join(bits)


def _detail_field_order() -> list[str]:
    return [
        "timestamp_utc",
        "run_id",
        "task",
        "metric_name",
        "metric_scope",
        "score",
        "model_group",
        "model_label",
        "profile_tag",
        "ckpt_name",
        "ckpt_step",
        "ckpt_path",
        "model_arch",
        "add_bos",
        "batch_scoring",
        "continuation_scoring",
        "dtype",
        "device_map",
        "k_shot",
        "seed",
        "limit",
        "scope",
        "run_dir",
    ]


def _run_summary_field_order() -> list[str]:
    return [
        "timestamp_utc",
        "run_id",
        "task",
        "overall_macro",
        "overall_micro",
        "overall_acc_norm",
        "overall_f1",
        "acc",
        "acc_norm",
        "f1",
        "num_parts",
        "scope",
        "model_group",
        "model_label",
        "profile_tag",
        "ckpt_name",
        "ckpt_step",
        "ckpt_path",
        "model_arch",
        "add_bos",
        "batch_scoring",
        "continuation_scoring",
        "dtype",
        "device_map",
        "k_shot",
        "seed",
        "limit",
        "subjects",
        "kobest_tasks",
        "kobest_split",
        "benchmark_split",
        "use_chat_template",
        "enable_thinking",
        "dec_max_len",
        "detail_csv",
        "summary_csv",
        "run_dir",
        "command",
    ]


def _overview_field_order() -> list[str]:
    fields = [
        "model_group",
        "model_label",
        "profile_tag",
        "ckpt_name",
        "ckpt_step",
        "ckpt_path",
        "model_arch",
        "add_bos",
        "batch_scoring",
        "continuation_scoring",
        "dtype",
        "device_map",
        "latest_task",
        "latest_run_id",
        "latest_run_at",
    ]
    for task in TASK_OVERVIEW_ORDER:
        fields.extend(
            [
                f"{task}_macro",
                f"{task}_micro",
                f"{task}_acc",
                f"{task}_acc_norm",
                f"{task}_f1",
                f"{task}_num_parts",
                f"{task}_scope",
                f"{task}_run_id",
                f"{task}_updated_at",
                f"{task}_detail_csv",
            ]
        )
    return fields


def _make_log_context(args, selected_items: Optional[list[str]]) -> dict[str, Any]:
    ts = _now_utc()
    ckpt_path = os.path.abspath(args.ckpt_path)
    ckpt_name = os.path.basename(os.path.normpath(ckpt_path)) or ckpt_path
    model_label = (args.model_label or "").strip() or ckpt_name
    model_group = _resolve_model_group(args.log_group, args.model_arch, ckpt_path, model_label)
    profile_tag = _build_profile_tag(args, selected_items)
    run_id = (
        f"{ts.strftime('%Y%m%d_%H%M%S_%f')}"
        f"_{_slugify(model_label, 32)}"
        f"_{args.task}"
        f"_{_slugify(profile_tag, 48)}"
    )
    log_root = os.path.abspath(args.log_root or _default_log_root())
    run_dir = os.path.join(log_root, model_group, "runs", run_id)
    detail_csv = os.path.join(run_dir, "metric_details.csv")
    summary_csv = os.path.join(run_dir, "run_summary.csv")
    scope = _build_scope_text(args, selected_items)

    effective_seed = args.seed

    return {
        "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "task": args.task,
        "model_group": model_group,
        "model_label": model_label,
        "profile_tag": profile_tag,
        "ckpt_name": ckpt_name,
        "ckpt_step": _infer_checkpoint_step(ckpt_name),
        "ckpt_path": ckpt_path,
        "model_arch": args.model_arch or "",
        "add_bos": getattr(args, "add_bos", "auto"),
        "batch_scoring": getattr(args, "batch_scoring", "auto"),
        "continuation_scoring": getattr(args, "continuation_scoring", "dynamic"),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "k_shot": _effective_k_shot(args),
        "seed": effective_seed,
        "limit": args.limit if args.limit and args.limit > 0 else "",
        "subjects": ",".join(selected_items or [])
        if args.task in {"mmlu", "kmmlu", "csatqa", "click"}
        else "",
        "kobest_tasks": ",".join(selected_items or []) if args.task == "kobest" else "",
        "kobest_split": args.kobest_split if args.task == "kobest" else "",
        "benchmark_split": getattr(args, "benchmark_split", "") if args.task in {"hellaswag", "openbookqa"} else "",
        "use_chat_template": args.use_chat_template,
        "enable_thinking": args.enable_thinking,
        "dec_max_len": args.dec_max_len if args.dec_max_len and args.dec_max_len > 0 else "",
        "scope": scope,
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "log_root": log_root,
        "run_dir": run_dir,
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
    }


def _result_metric_scope(task: str, metric_name: str) -> str:
    if metric_name.startswith("overall_"):
        return "overall"
    return "subject" if task in {"mmlu", "kmmlu", "click"} else "benchmark"


def _result_num_parts(results: dict[str, float]) -> int:
    return sum(
        1
        for key in results
        if not key.startswith("overall_")
        and key not in {"acc", "acc_norm", "f1"}
        and not key.endswith("_f1")
        and not key.endswith("_acc_norm")
    )


def _detail_rows_from_results(results: dict[str, float], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, score in results.items():
        rows.append(
            {
                "timestamp_utc": ctx["timestamp_utc"],
                "run_id": ctx["run_id"],
                "task": ctx["task"],
                "metric_name": metric_name,
                "metric_scope": _result_metric_scope(ctx["task"], metric_name),
                "score": score,
                "model_group": ctx["model_group"],
                "model_label": ctx["model_label"],
                "profile_tag": ctx["profile_tag"],
                "ckpt_name": ctx["ckpt_name"],
                "ckpt_step": ctx["ckpt_step"],
                "ckpt_path": ctx["ckpt_path"],
                "model_arch": ctx["model_arch"],
                "add_bos": ctx["add_bos"],
                "batch_scoring": ctx["batch_scoring"],
                "continuation_scoring": ctx["continuation_scoring"],
                "dtype": ctx["dtype"],
                "device_map": ctx["device_map"],
                "k_shot": ctx["k_shot"],
                "seed": ctx["seed"],
                "limit": ctx["limit"],
                "scope": ctx["scope"],
                "run_dir": ctx["run_dir"],
            }
        )
    return rows


def _run_summary_row(results: dict[str, float], ctx: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in ctx.items() if key not in {"log_root"}}
    row["overall_macro"] = results.get("overall_macro", "")
    row["overall_micro"] = results.get("overall_micro", "")
    row["overall_acc_norm"] = results.get("overall_acc_norm", "")
    row["overall_f1"] = results.get("overall_f1", "")
    row["acc"] = results.get("acc", results.get("overall_micro", ""))
    row["acc_norm"] = results.get("acc_norm", results.get("overall_acc_norm", ""))
    row["f1"] = results.get("f1", results.get("overall_f1", ""))
    row["num_parts"] = _result_num_parts(results)
    return row


def _overview_update_row(results: dict[str, float], ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_group": ctx["model_group"],
        "model_label": ctx["model_label"],
        "profile_tag": ctx["profile_tag"],
        "ckpt_name": ctx["ckpt_name"],
        "ckpt_step": ctx["ckpt_step"],
        "ckpt_path": ctx["ckpt_path"],
        "model_arch": ctx["model_arch"],
        "add_bos": ctx["add_bos"],
        "batch_scoring": ctx["batch_scoring"],
        "continuation_scoring": ctx["continuation_scoring"],
        "dtype": ctx["dtype"],
        "device_map": ctx["device_map"],
        "latest_task": ctx["task"],
        "latest_run_id": ctx["run_id"],
        "latest_run_at": ctx["timestamp_utc"],
        f"{ctx['task']}_macro": results.get("overall_macro", ""),
        f"{ctx['task']}_micro": results.get("overall_micro", ""),
        f"{ctx['task']}_acc": results.get("acc", results.get("overall_micro", "")),
        f"{ctx['task']}_acc_norm": results.get("acc_norm", results.get("overall_acc_norm", "")),
        f"{ctx['task']}_f1": results.get("f1", results.get("overall_f1", "")),
        f"{ctx['task']}_num_parts": _result_num_parts(results),
        f"{ctx['task']}_scope": ctx["scope"],
        f"{ctx['task']}_run_id": ctx["run_id"],
        f"{ctx['task']}_updated_at": ctx["timestamp_utc"],
        f"{ctx['task']}_detail_csv": ctx["detail_csv"],
    }


def write_result_logs(args, results: dict[str, float], selected_items: Optional[list[str]]) -> dict[str, str]:
    if args.disable_csv_log:
        return {}

    ctx = _make_log_context(args, selected_items)
    detail_rows = _detail_rows_from_results(results, ctx)
    run_summary = _run_summary_row(results, ctx)
    overview_row = _overview_update_row(results, ctx)

    os.makedirs(ctx["run_dir"], exist_ok=True)
    _write_csv_rows(ctx["detail_csv"], detail_rows, _detail_field_order())
    _write_csv_rows(ctx["summary_csv"], [run_summary], _run_summary_field_order())

    all_dirs = [
        os.path.join(ctx["log_root"], ctx["model_group"]),
        os.path.join(ctx["log_root"], "all"),
    ]
    for base_dir in all_dirs:
        _append_csv_row(
            os.path.join(base_dir, "run_history.csv"),
            run_summary,
            _run_summary_field_order(),
        )
        _upsert_csv_row(
            os.path.join(base_dir, "checkpoint_overview.csv"),
            overview_row,
            key_fields=["model_group", "model_label", "profile_tag", "ckpt_path"],
            preferred_fieldnames=_overview_field_order(),
        )

    print(f"[csv] detail={ctx['detail_csv']}")
    print(f"[csv] summary={ctx['summary_csv']}")
    print(f"[csv] history={os.path.join(ctx['log_root'], ctx['model_group'], 'run_history.csv')}")
    print(f"[csv] overview={os.path.join(ctx['log_root'], ctx['model_group'], 'checkpoint_overview.csv')}")
    return {
        "detail_csv": ctx["detail_csv"],
        "summary_csv": ctx["summary_csv"],
        "group_history_csv": os.path.join(ctx["log_root"], ctx["model_group"], "run_history.csv"),
        "group_overview_csv": os.path.join(ctx["log_root"], ctx["model_group"], "checkpoint_overview.csv"),
        "all_history_csv": os.path.join(ctx["log_root"], "all", "run_history.csv"),
        "all_overview_csv": os.path.join(ctx["log_root"], "all", "checkpoint_overview.csv"),
    }
