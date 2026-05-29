#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import time
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SWEEP_ROOT = SCRIPT_DIR / "sweep_results"
DASHBOARD_HTML_PATH = SCRIPT_DIR / "dashboard_frontend.html"
STEP_PATTERN = re.compile(r"(?:checkpoint|ckpt|step)[-_]?(\d+)$", flags=re.IGNORECASE)
METRIC_KEYS = (
    "overall_micro",
    "overall_macro",
    "overall_acc_norm",
    "acc",
    "acc_norm",
    "f1",
    "overall_f1",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _checkpoint_step(name: str) -> int | None:
    match = STEP_PATTERN.search(name or "")
    return int(match.group(1)) if match else None


def _latest_sweep_dir() -> Path:
    if not DEFAULT_SWEEP_ROOT.exists():
        raise FileNotFoundError(f"No sweep_results directory found: {DEFAULT_SWEEP_ROOT}")
    candidates = [path for path in DEFAULT_SWEEP_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No sweep result directories found under: {DEFAULT_SWEEP_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _manifest_tasks(manifest: dict[str, Any]) -> list[str]:
    tasks = manifest.get("tasks") or []
    if not tasks and manifest.get("task"):
        tasks = [str(manifest["task"])]
    return [str(task) for task in tasks if str(task)]


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> float | None:
    value = _safe_float(value)
    if value is None:
        return None
    return value


def _infer_task_from_json(root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to((root / "json").resolve())
    except ValueError:
        return ""
    return rel.parts[0] if len(rel.parts) > 1 else ""


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "sweep_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return _read_json(manifest_path)
    except Exception:
        return {}


@lru_cache(maxsize=512)
def _infer_arch_from_ckpt_path(path_text: str) -> str:
    if not path_text:
        return "plain"
    path = Path(path_text).expanduser()
    config_path = path / "config.json"
    if not config_path.exists():
        return "plain"
    try:
        config = _read_json(config_path)
    except Exception:
        return "plain"
    blob = json.dumps(
        {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "auto_map": config.get("auto_map"),
        },
        ensure_ascii=False,
    ).lower()
    if "krong" in blob or "kormo" in blob:
        return "krong"
    return "plain"


def _model_arch_from_row(row: dict[str, str]) -> str:
    command = str(row.get("command", "") or "").strip()
    if command:
        try:
            parts = shlex.split(command)
            for idx, token in enumerate(parts[:-1]):
                if token == "--model_arch":
                    value = parts[idx + 1].strip().lower()
                    if value in {"krong", "plain", "hf", "normal"}:
                        return "plain" if value in {"plain", "hf", "normal"} else value
                    if value:
                        return value
        except Exception:
            pass
    return _infer_arch_from_ckpt_path(str(row.get("ckpt_path", "") or ""))


def _metrics_from_result(path_text: str) -> dict[str, float | None]:
    if not path_text:
        return {key: None for key in METRIC_KEYS}
    path = Path(path_text)
    if not path.exists():
        return {key: None for key in METRIC_KEYS}
    try:
        data = _read_json(path)
    except Exception:
        return {key: None for key in METRIC_KEYS}
    return {key: _format_metric(data.get(key)) for key in METRIC_KEYS}


def _local_artifact_path(root: Path, path_text: str, kind: str, task: str, checkpoint_name: str) -> str:
    """
    Old sweep summaries stored absolute paths under the original sweep_results/.
    If a run folder was copied to sweep_results_test/, prefer the copied local files.
    """
    suffix = ".json" if kind == "json" else ".log"
    candidates = []
    if task and checkpoint_name:
        candidates.append(root / kind / task / f"{checkpoint_name}{suffix}")
    if checkpoint_name:
        candidates.append(root / kind / f"{checkpoint_name}{suffix}")
    if path_text:
        basename = Path(path_text).name
        if task:
            candidates.append(root / kind / task / basename)
        candidates.append(root / kind / basename)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if path_text and Path(path_text).exists():
        return path_text
    return path_text or (str(candidates[0]) if candidates else "")


def _result_json_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    json_root = root / "json"
    if not json_root.exists():
        return rows
    for path in sorted(json_root.rglob("*.json")):
        checkpoint_name = path.stem
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "step": str(_checkpoint_step(checkpoint_name) or ""),
                "task": _infer_task_from_json(root, path),
                "status": "ok",
                "returncode": "0",
                "result_json": str(path),
                "stdout_log": "",
                "duration_sec": "",
                "ckpt_path": "",
                "command": "",
            }
        )
    return rows


def _expected_rows_from_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    checkpoints = manifest.get("checkpoints") or []
    tasks = manifest.get("tasks") or []
    if not tasks and manifest.get("task"):
        tasks = [str(manifest["task"])]
    rows: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        name = str(checkpoint.get("name") or "")
        step = str(checkpoint.get("step") or _checkpoint_step(name) or "")
        for task in tasks:
            rows.append(
                {
                    "checkpoint_name": name,
                    "step": step,
                    "task": str(task),
                    "status": "pending",
                    "returncode": "",
                    "result_json": "",
                    "stdout_log": "",
                    "duration_sec": "",
                    "ckpt_path": str(checkpoint.get("path") or ""),
                    "command": "",
                }
            )
    return rows


def load_dashboard_data(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load_manifest(root)
    manifest_tasks = _manifest_tasks(manifest)
    default_task = manifest_tasks[0] if len(manifest_tasks) == 1 else ""
    summary_rows = _read_csv_rows(root / "sweep_summary.csv")
    source_rows = summary_rows or _result_json_rows(root)

    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in _expected_rows_from_manifest(manifest):
        by_key[(row.get("checkpoint_name", ""), row.get("task", ""))] = row
    for row in source_rows:
        task = row.get("task", "") or default_task
        if not task and row.get("result_json"):
            task = _infer_task_from_json(root, Path(row["result_json"]))
            row["task"] = task
        key = (row.get("checkpoint_name", ""), task)
        by_key[key] = {**by_key.get(key, {}), **row}

    rows: list[dict[str, Any]] = []
    for row in by_key.values():
        checkpoint_name = row.get("checkpoint_name", "")
        task = row.get("task", "")
        if task and checkpoint_name:
            row["stdout_log"] = _local_artifact_path(root, row.get("stdout_log", ""), "stdout", task, checkpoint_name)
            row["result_json"] = _local_artifact_path(root, row.get("result_json", ""), "json", task, checkpoint_name)
            if row.get("status") == "pending":
                log_path = Path(row["stdout_log"])
                if log_path.exists() and log_path.stat().st_size > 0:
                    row["status"] = "running"

        metrics = _metrics_from_result(row.get("result_json", ""))
        step_value = _checkpoint_step(row.get("checkpoint_name", "")) if not row.get("step") else None
        step = int(row.get("step") or step_value or 0)
        rows.append(
            {
                "checkpoint_name": row.get("checkpoint_name", ""),
                "step": step,
                "task": row.get("task", ""),
                "status": row.get("status", ""),
                "returncode": row.get("returncode", ""),
                "duration_sec": _safe_float(row.get("duration_sec")),
                "ckpt_path": row.get("ckpt_path", ""),
                "model_arch": _model_arch_from_row(row),
                "result_json": row.get("result_json", ""),
                "stdout_log": row.get("stdout_log", ""),
                "started_at_utc": row.get("started_at_utc", ""),
                "ended_at_utc": row.get("ended_at_utc", ""),
                "metrics": metrics,
            }
        )

    rows.sort(key=lambda item: (item["task"], item["step"], item["checkpoint_name"]))
    ok_runs = sum(1 for row in rows if row["status"] == "ok")
    failed_runs = sum(1 for row in rows if row["status"] == "failed")
    pending_runs = sum(1 for row in rows if row["status"] in {"pending", "running"})

    return {
        "result_root": str(root),
        "updated_at": _utc_now(),
        "manifest": manifest,
        "expected_runs": len(rows),
        "ok_runs": ok_runs,
        "failed_runs": failed_runs,
        "pending_runs": pending_runs,
        "rows": rows,
    }


def _discover_sweep_dirs(parent: Path) -> list[Path]:
    parent = parent.resolve()
    if (parent / "sweep_summary.csv").exists() or (parent / "sweep_manifest.json").exists():
        return [parent]
    if not parent.exists():
        raise FileNotFoundError(f"Compare root not found: {parent}")
    sweeps = [
        path
        for path in parent.iterdir()
        if path.is_dir()
        and (
            (path / "sweep_summary.csv").exists()
            or (path / "sweep_manifest.json").exists()
            or (path / "json").exists()
        )
    ]
    return sorted(sweeps, key=lambda path: path.name)


def _sweep_label(root: Path, manifest: dict[str, Any]) -> str:
    explicit_label = str(manifest.get("series_label") or manifest.get("display_series") or "").strip()
    if explicit_label:
        return explicit_label

    name = root.name
    normalized_name = re.sub(r"^\d{8}_\d{6}_", "", name)

    # Some experiments are intentionally split by task group or fill run, but
    # should render as one training curve in compare mode.
    series_aliases = {
        "llama31_8b_interleave_mlm00_copylow_main7": "llama31_8b_interleave_mlm00_copylow",
        "llama31_8b_interleave_mlm00_copylow_extra4": "llama31_8b_interleave_mlm00_copylow",
        "llama31_8b_base_lmeval_click_click": "llama31_8b_base_main7",
        "llama31_8b_base_ko_rerun": "llama31_8b_base_main7",
        "llama31_8b_interleave_mlm00_copylow_ckpt2000_extra_missing_fill": "llama31_8b_interleave_mlm00_copylow",
        "llama31_8b_interleave_mlm00_copylow_mmlu_fill_3000_6000": "llama31_8b_interleave_mlm00_copylow",
        "llama31_8b_interleave_mlm00_copylow_ckpt13000_main_extra": "llama31_8b_interleave_mlm00_copylow",
        "llama32_1b_interleave_cpt_ckpt1000_main7": "llama32_1b_interleave_cpt",
        "llama32_1b_interleave_cpt_ckpt7000_main_extra": "llama32_1b_interleave_cpt",
        "llama32_1b_interleave_cpt_ckpt2000_6000_main_extra_fill_cuda0": "llama32_1b_interleave_cpt",
        "llama32_1b_interleave_cpt_8000_13000_main_extra": "llama32_1b_interleave_cpt",
        "stage1_checkpoint_extra4_step0_k5": "stage1_checkpoint",
        "stage1_checkpoint_click_fixed_step0_k5": "stage1_checkpoint",
        "stage1_checkpoint_rerun_main_extra": "stage1_checkpoint",
    }
    if normalized_name in series_aliases:
        return series_aliases[normalized_name]

    if re.match(
        r"llama31_8b_interleave_mlm00_copylow_(?:ckpt\d+_(?:ko|en|main_extra_fill)|extra(?:4)?_\d+(?:_\d+)?_fill)$",
        normalized_name,
    ):
        return "llama31_8b_interleave_mlm00_copylow"
    if re.match(r"llama32_1b_interleave_cpt(?:_|$)", normalized_name):
        return "llama32_1b_interleave_cpt"

    checkpoints_root = str(manifest.get("checkpoints_root") or "").strip()
    if checkpoints_root:
        label = Path(checkpoints_root).name
        if label:
            return label

    name = re.sub(r"_(mmlu|kmmlu|kobest|csatqa|click|arc_easy|arc_challenge)_every\d+$", "", normalized_name)
    return name or root.name


def _is_stage1_series(series: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", (series or "").lower()).strip("_")
    return normalized in {"stage1", "stage1_checkpoint"}


def _receives_stage1_start(series: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", (series or "").lower()).strip("_")
    # Stage1 is only the shared baseline for the original 1B stage2 sweeps.
    # Llama-3.x derived experiments have their own HF base model at step 0.
    return normalized.startswith("checkpoints_")


def _attach_stage1_start_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Treat stage1_checkpoint as the shared pre-training baseline.

    In compare mode, a standalone stage1 sweep would otherwise render as its own
    model series. For downstream training sweeps, it is more useful as step 0 of
    each curve, because those checkpoints were continued from stage1.
    """
    stage1_rows = [
        row
        for row in rows
        if _is_stage1_series(str(row.get("series", ""))) and int(row.get("step") or 0) == 0
    ]
    if not stage1_rows:
        return rows

    target_series = sorted(
        {
            str(row.get("series", ""))
            for row in rows
            if row.get("series")
            and not _is_stage1_series(str(row.get("series", "")))
            and _receives_stage1_start(str(row.get("series", "")))
        }
    )
    if not target_series:
        return rows

    arch_by_series: dict[str, str] = {}
    tasks_by_series: dict[str, set[str]] = {}
    for series in target_series:
        arches = sorted(
            {
                str(row.get("model_arch", ""))
                for row in rows
                if row.get("series") == series and row.get("model_arch")
            }
        )
        if arches:
            arch_by_series[series] = "/".join(arches)
        tasks_by_series[series] = {
            str(row.get("task", ""))
            for row in rows
            if row.get("series") == series and row.get("task")
        }

    existing = {
        (str(row.get("series", "")), str(row.get("task", "")), int(row.get("step") or 0))
        for row in rows
    }

    augmented = [
        row
        for row in rows
        if not _is_stage1_series(str(row.get("series", "")))
    ]
    for series in target_series:
        for base in stage1_rows:
            if str(base.get("task", "")) not in tasks_by_series.get(series, set()):
                continue
            key = (series, str(base.get("task", "")), 0)
            if key in existing:
                continue
            cloned = dict(base)
            cloned["series"] = series
            cloned["step"] = 0
            cloned["checkpoint_name"] = "stage1_checkpoint"
            # Keep the source stage1 architecture tag; stage1 is a plain decoder checkpoint.
            cloned["model_arch"] = cloned.get("model_arch", "")
            cloned["is_stage1_start"] = True
            cloned["source_series"] = base.get("series", "stage1_checkpoint")
            cloned["source_sweep_name"] = base.get("sweep_name", "")
            augmented.append(cloned)
            existing.add(key)

    return augmented


def _compare_row_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    status = str(row.get("status", "") or "").lower()
    status_rank = {
        "ok": 4,
        "running": 3,
        "pending": 2,
        "failed": 1,
    }.get(status, 0)
    # Prefer rows with actual metric values over empty/placeholder JSON rows.
    metric_rank = 1 if any(value is not None for value in (row.get("metrics") or {}).values()) else 0
    # Prefer real rows over cloned stage1 rows when both exist for the same
    # series/task/step. This matters for experiments with an explicit step 0.
    source_rank = 0 if row.get("source_series") else 1
    timestamp = str(row.get("ended_at_utc") or row.get("started_at_utc") or row.get("sweep_name") or "")
    return (status_rank, metric_rank, source_rank, timestamp)


def _dedupe_compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compare mode can intentionally stitch together multiple fill sweeps for the
    same series. Keep one row per series/task/step so stale failed/pending rows
    do not render beside later successful fill runs.
    """
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("series", "")),
            str(row.get("task", "")),
            int(row.get("step") or 0),
        )
        current = by_key.get(key)
        if current is None or _compare_row_rank(row) >= _compare_row_rank(current):
            by_key[key] = row
    return list(by_key.values())


def load_compare_data(parent: Path) -> dict[str, Any]:
    parent = parent.resolve()
    sweeps = _discover_sweep_dirs(parent)
    rows: list[dict[str, Any]] = []
    sweep_summaries: list[dict[str, Any]] = []

    for sweep_root in sweeps:
        data = load_dashboard_data(sweep_root)
        manifest = data.get("manifest") or {}
        series_label = _sweep_label(sweep_root, manifest)

        sweep_summaries.append(
            {
                "series": series_label,
                "sweep_name": sweep_root.name,
                "sweep_root": str(sweep_root),
                "tasks": _manifest_tasks(manifest),
                "expected_runs": data.get("expected_runs", 0),
                "ok_runs": data.get("ok_runs", 0),
                "failed_runs": data.get("failed_runs", 0),
                "pending_runs": data.get("pending_runs", 0),
            }
        )

        for row in data.get("rows", []):
            enriched = dict(row)
            enriched["series"] = series_label
            enriched["sweep_name"] = sweep_root.name
            enriched["sweep_root"] = str(sweep_root)
            rows.append(enriched)

    rows = _attach_stage1_start_rows(rows)
    rows = _dedupe_compare_rows(rows)
    rows.sort(key=lambda item: (item.get("task", ""), item.get("step", 0), item.get("series", "")))
    ok_runs = sum(1 for row in rows if row.get("status") == "ok")
    failed_runs = sum(1 for row in rows if row.get("status") == "failed")
    pending_runs = sum(1 for row in rows if row.get("status") in {"pending", "running"})

    return {
        "mode": "compare",
        "compare_root": str(parent),
        "result_root": str(parent),
        "updated_at": _utc_now(),
        "series": sorted({row.get("series", "") for row in rows if row.get("series")}),
        "sweeps": sweep_summaries,
        "expected_runs": len(rows),
        "ok_runs": ok_runs,
        "failed_runs": failed_runs,
        "pending_runs": pending_runs,
        "rows": rows,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KRong Eval Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1714;
      --panel: rgba(246, 239, 220, 0.08);
      --panel-strong: rgba(246, 239, 220, 0.14);
      --ink: #f6efdc;
      --muted: #aeb8aa;
      --line: rgba(246, 239, 220, 0.14);
      --good: #7bd88f;
      --bad: #ff7f6e;
      --wait: #f2c36b;
      --accent: #72d6c9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 10% 5%, rgba(114, 214, 201, 0.24), transparent 32rem),
        radial-gradient(circle at 90% 15%, rgba(242, 195, 107, 0.18), transparent 30rem),
        linear-gradient(135deg, #0f1714 0%, #18231f 55%, #111715 100%);
    }
    main { width: min(1480px, calc(100vw - 40px)); margin: 0 auto; padding: 32px 0 40px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 20px; }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 52px); letter-spacing: -0.05em; }
    .sub { color: var(--muted); margin-top: 8px; font-size: 14px; }
    .pill { border: 1px solid var(--line); border-radius: 999px; padding: 9px 12px; color: var(--muted); background: rgba(0,0,0,0.18); }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 18px; backdrop-filter: blur(14px); }
    .card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }
    .card .value { font-size: 34px; font-weight: 800; margin-top: 8px; letter-spacing: -0.04em; }
    .toolbar { display: flex; gap: 10px; margin: 18px 0; flex-wrap: wrap; }
    .model-strip { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 18px; }
    .model-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(0,0,0,0.2);
      color: var(--ink);
      font-size: 13px;
    }
    .chip-arch {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .trend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
    .trend-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 14px;
      backdrop-filter: blur(14px);
    }
    .trend-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 8px; }
    .trend-title { font-size: 15px; font-weight: 700; letter-spacing: -0.01em; }
    .trend-meta { color: var(--muted); font-size: 12px; }
    .legend-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
    .legend-item { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }
    .legend-swatch { width: 10px; height: 10px; border-radius: 999px; }
    .chart-svg { width: 100%; height: auto; display: block; }
    .chart-label { fill: var(--muted); font-size: 11px; }
    .chart-grid { stroke: rgba(246, 239, 220, 0.10); stroke-width: 1; }
    .chart-axis { stroke: rgba(246, 239, 220, 0.24); stroke-width: 1.2; }
    .chart-point { stroke: #0f1714; stroke-width: 2; }
    input, select {
      color: var(--ink);
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 11px 12px;
      outline: none;
    }
    input { min-width: 280px; flex: 1; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 22px; background: rgba(0,0,0,0.2); }
    th, td { padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.09em; background: var(--panel-strong); position: sticky; top: 0; }
    tr:hover td { background: rgba(255,255,255,0.04); cursor: pointer; }
    h2 { margin: 18px 0 10px; font-size: 18px; letter-spacing: -0.02em; }
    .status { display: inline-flex; align-items: center; gap: 7px; font-weight: 700; }
    .dot { width: 9px; height: 9px; border-radius: 99px; background: var(--muted); }
    .ok .dot { background: var(--good); }
    .failed .dot { background: var(--bad); }
    .pending .dot { background: var(--wait); }
    .running .dot { background: var(--accent); box-shadow: 0 0 16px var(--accent); }
    .metric { font-variant-numeric: tabular-nums; font-weight: 700; }
    .best { color: var(--good); text-shadow: 0 0 18px rgba(123, 216, 143, 0.25); }
    .delta-pos { color: var(--good); }
    .delta-neg { color: var(--bad); }
    .muted { color: var(--muted); }
    .split { display: grid; grid-template-columns: 1fr; gap: 14px; }
    pre {
      display: none;
      margin: 14px 0 0;
      max-height: 340px;
      overflow: auto;
      white-space: pre-wrap;
      background: rgba(0,0,0,0.35);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      color: #e8eadf;
    }
    @media (max-width: 900px) {
      main { width: min(100vw - 22px, 1480px); padding-top: 20px; }
      header { align-items: start; flex-direction: column; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .table-wrap { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>KRong Eval Dashboard</h1>
        <div class="sub" id="root">Loading...</div>
      </div>
      <div class="pill" id="updated">Waiting for data</div>
    </header>
    <section class="cards">
      <div class="card"><div class="label">Expected</div><div class="value" id="expected">0</div></div>
      <div class="card"><div class="label">Finished</div><div class="value" id="ok">0</div></div>
      <div class="card"><div class="label">Pending</div><div class="value" id="pending">0</div></div>
      <div class="card"><div class="label">Failed</div><div class="value" id="failed">0</div></div>
    </section>
    <section class="toolbar">
      <input id="query" placeholder="Filter by model, checkpoint, task, status..." />
      <select id="series"><option value="">All models</option></select>
      <select id="arch"><option value="">All archs</option></select>
      <select id="task"><option value="">All tasks</option></select>
      <select id="status"><option value="">All statuses</option><option>ok</option><option>running</option><option>pending</option><option>failed</option><option>skipped_existing_json</option></select>
      <select id="metric"><option value="overall_micro">Micro</option><option value="overall_macro">Macro</option><option value="overall_acc_norm">Acc Norm</option><option value="acc">Acc</option><option value="f1">F1</option></select>
      <select id="diff-base"><option value="">Diff baseline</option></select>
      <select id="diff-compare"><option value="">Diff compare</option></select>
    </section>
    <section>
      <div class="model-strip" id="model-strip"></div>
    </section>
    <section class="split">
      <div>
        <h2>Trend</h2>
        <div class="sub" id="trend-summary">Score trend by step for the currently visible rows.</div>
        <div class="trend-grid" id="trend-panels"></div>
      </div>
      <div>
        <h2>Score Compare</h2>
        <div class="sub" id="compare-summary">Average diff is computed only on steps where both compared models have data.</div>
        <div class="table-wrap">
          <table>
            <thead id="compare-head"></thead>
            <tbody id="compare-rows"></tbody>
          </table>
        </div>
      </div>
      <div class="table-wrap">
        <h2>Runs</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Arch</th><th>Task</th><th>Step</th><th>Checkpoint</th><th>Status</th>
              <th>Micro</th><th>Macro</th><th>Acc Norm</th><th>Duration</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
      <pre id="log"></pre>
    </section>
  </main>
  <script>
    const refreshMs = Number(new URLSearchParams(location.search).get("refresh") || "3000");
    let data = { rows: [] };
    const chartColors = ["#72d6c9", "#f2c36b", "#7bd88f", "#ff7f6e", "#8ca8ff", "#e9a6ff", "#ffb870", "#9de3a5"];
    const pct = (x) => x === null || x === undefined ? "" : `${(x * 100).toFixed(2)}%`;
    const sec = (x) => x === null || x === undefined || x === "" ? "" : `${Number(x).toFixed(1)}s`;
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;" }[c]));
    const metricValue = (row, key) => {
      const m = row.metrics || {};
      if (key === "overall_acc_norm") return m.overall_acc_norm ?? m.acc_norm ?? null;
      if (key === "acc") return m.acc ?? m.overall_micro ?? null;
      if (key === "f1") return m.f1 ?? m.overall_f1 ?? null;
      return m[key] ?? null;
    };
    const diffText = (x) => x === null || x === undefined || Number.isNaN(x) ? "" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(2)}pp`;
    const seriesKey = (row) => row.series || row.checkpoint_name || "model";

    function seriesArchMap(rows) {
      const out = {};
      for (const row of rows || []) {
        const key = seriesKey(row);
        const arch = row.model_arch || "";
        if (!key || !arch) continue;
        if (!out[key]) out[key] = new Set();
        out[key].add(arch);
      }
      const finalMap = {};
      for (const [key, vals] of Object.entries(out)) {
        finalMap[key] = [...vals].sort().join(", ");
      }
      return finalMap;
    }

    async function load() {
      const res = await fetch("/api/summary", { cache: "no-store" });
      data = await res.json();
      render();
    }

    function renderCards() {
      const label = data.mode === "compare" ? "compare root" : "result root";
      document.getElementById("root").textContent = `${label}: ${data.compare_root || data.result_root || ""}`;
      document.getElementById("updated").textContent = `Updated ${data.updated_at || ""}`;
      document.getElementById("expected").textContent = data.expected_runs ?? 0;
      document.getElementById("ok").textContent = data.ok_runs ?? 0;
      document.getElementById("pending").textContent = data.pending_runs ?? 0;
      document.getElementById("failed").textContent = data.failed_runs ?? 0;
    }

    function renderSeriesOptions() {
      const select = document.getElementById("series");
      const current = select.value;
      const series = [...new Set((data.rows || []).map(r => seriesKey(r)).filter(Boolean))].sort();
      const archMap = seriesArchMap(data.rows || []);
      select.innerHTML = `<option value="">All models</option>` + series.map(s => `<option value="${esc(s)}">${esc(s)}${archMap[s] ? ` [${esc(archMap[s])}]` : ""}</option>`).join("");
      select.value = series.includes(current) ? current : "";
    }

    function renderDiffOptions() {
      const series = [...new Set((data.rows || []).map(r => seriesKey(r)).filter(Boolean))].sort();
      const archMap = seriesArchMap(data.rows || []);
      const base = document.getElementById("diff-base");
      const compare = document.getElementById("diff-compare");
      const previousBase = base.value;
      const previousCompare = compare.value;
      const options = series.map(s => `<option value="${esc(s)}">${esc(s)}${archMap[s] ? ` [${esc(archMap[s])}]` : ""}</option>`).join("");
      base.innerHTML = `<option value="">Diff baseline</option>${options}`;
      compare.innerHTML = `<option value="">Diff compare</option>${options}`;
      base.value = series.includes(previousBase) ? previousBase : (series[0] || "");
      compare.value = series.includes(previousCompare) ? previousCompare : (series.find(s => s !== base.value) || "");
    }

    function renderTaskOptions() {
      const select = document.getElementById("task");
      const current = select.value;
      const tasks = [...new Set((data.rows || []).map(r => r.task).filter(Boolean))].sort();
      select.innerHTML = `<option value="">All tasks</option>` + tasks.map(t => `<option>${esc(t)}</option>`).join("");
      select.value = tasks.includes(current) ? current : "";
    }

    function renderArchOptions() {
      const select = document.getElementById("arch");
      const current = select.value;
      const archs = [...new Set((data.rows || []).map(r => r.model_arch).filter(Boolean))].sort();
      select.innerHTML = `<option value="">All archs</option>` + archs.map(a => `<option>${esc(a)}</option>`).join("");
      select.value = archs.includes(current) ? current : "";
    }

    function renderModelStrip() {
      const strip = document.getElementById("model-strip");
      const rows = filteredRows();
      const archMap = seriesArchMap(rows);
      const series = [...new Set(rows.map(r => seriesKey(r)).filter(Boolean))].sort();
      if (!series.length) {
        strip.innerHTML = `<span class="muted">No visible models.</span>`;
        return;
      }
      strip.innerHTML = series.map(s => `<span class="model-chip"><span>${esc(s)}</span><span class="chip-arch">${esc(archMap[s] || "unknown")}</span></span>`).join("");
    }

    function filteredRows() {
      const q = document.getElementById("query").value.toLowerCase();
      const series = document.getElementById("series").value;
      const arch = document.getElementById("arch").value;
      const task = document.getElementById("task").value;
      const status = document.getElementById("status").value;
      return (data.rows || []).filter(row => {
        const key = seriesKey(row);
        const hay = `${key} ${row.model_arch || ""} ${row.task} ${row.step} ${row.checkpoint_name} ${row.status}`.toLowerCase();
        return (!q || hay.includes(q)) && (!series || key === series) && (!arch || row.model_arch === arch) && (!task || row.task === task) && (!status || row.status === status);
      });
    }

    function renderTrend() {
      const container = document.getElementById("trend-panels");
      const metric = document.getElementById("metric").value;
      const rows = filteredRows().filter(row => row.status === "ok" && metricValue(row, metric) !== null && metricValue(row, metric) !== undefined);
      const archMap = seriesArchMap(rows);
      const byTask = new Map();
      for (const row of rows) {
        const task = row.task || "all";
        const series = seriesKey(row);
        if (!byTask.has(task)) byTask.set(task, new Map());
        const taskMap = byTask.get(task);
        if (!taskMap.has(series)) taskMap.set(series, []);
        taskMap.get(series).push({ step: Number(row.step || 0), value: metricValue(row, metric) });
      }
      const tasks = [...byTask.keys()].sort();
      if (!tasks.length) {
        document.getElementById("trend-summary").textContent = "No completed rows for the selected metric yet.";
        container.innerHTML = "";
        return;
      }

      document.getElementById("trend-summary").textContent = `Showing ${tasks.length} task chart(s) for ${metric}. Y-axis is absolute percent (0-100%).`;
      container.innerHTML = tasks.map((task, taskIdx) => {
        const seriesMap = byTask.get(task);
        const seriesList = [...seriesMap.keys()].sort();
        const width = 680, height = 240, ml = 44, mr = 16, mt = 16, mb = 34;
        const plotW = width - ml - mr, plotH = height - mt - mb;
        const allSteps = [...new Set(seriesList.flatMap(series => seriesMap.get(series).map(p => p.step)))].sort((a, b) => a - b);
        const minStep = allSteps[0] ?? 0;
        const maxStep = allSteps[allSteps.length - 1] ?? minStep;
        const xFor = (step) => {
          if (allSteps.length <= 1 || maxStep === minStep) return ml + plotW / 2;
          return ml + ((step - minStep) / (maxStep - minStep)) * plotW;
        };
        const yFor = (value) => mt + (1 - value) * plotH;
        const yTicks = [0, 0.5, 1.0];
        const grid = yTicks.map(tick => {
          const y = yFor(tick);
          return `<line class="chart-grid" x1="${ml}" y1="${y}" x2="${ml + plotW}" y2="${y}"></line><text class="chart-label" x="${ml - 8}" y="${y + 4}" text-anchor="end">${Math.round(tick * 100)}%</text>`;
        }).join("");
        const xAxis = `<line class="chart-axis" x1="${ml}" y1="${mt + plotH}" x2="${ml + plotW}" y2="${mt + plotH}"></line><line class="chart-axis" x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + plotH}"></line>`;
        const xLabels = allSteps.length ? `<text class="chart-label" x="${ml}" y="${height - 8}" text-anchor="start">${esc(minStep)}</text><text class="chart-label" x="${ml + plotW}" y="${height - 8}" text-anchor="end">${esc(maxStep)}</text>` : "";
        const lines = seriesList.map((series, idx) => {
          const color = chartColors[idx % chartColors.length];
          const pts = [...seriesMap.get(series)].sort((a, b) => a.step - b.step);
          const poly = pts.map(p => `${xFor(p.step)},${yFor(p.value)}`).join(" ");
          const circles = pts.map(p => `<circle class="chart-point" cx="${xFor(p.step)}" cy="${yFor(p.value)}" r="4" fill="${color}"></circle>`).join("");
          const latest = pts[pts.length - 1]?.value;
          return {
            svg: `${poly ? `<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${poly}"></polyline>` : ""}${circles}`,
            legend: `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span><span>${esc(series)}</span><span class="chip-arch">${esc(archMap[series] || "")}</span><span class="metric">${pct(latest)}</span></span>`,
          };
        });
        return `<div class="trend-card">
          <div class="trend-head">
            <div class="trend-title">${esc(task)}</div>
            <div class="trend-meta">${seriesList.length} model series</div>
          </div>
          <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Trend chart for ${esc(task)}">
            ${grid}
            ${xAxis}
            ${xLabels}
            ${lines.map(x => x.svg).join("")}
          </svg>
          <div class="legend-row">${lines.map(x => x.legend).join("")}</div>
        </div>`;
      }).join("");
    }

    function renderCompare() {
      const rows = filteredRows().filter(row => row.status === "ok");
      const metric = document.getElementById("metric").value;
      const series = [...new Set(rows.map(r => seriesKey(r)).filter(Boolean))].sort();
      const archMap = seriesArchMap(rows);
      const selectedBase = document.getElementById("diff-base").value;
      const selectedCompare = document.getElementById("diff-compare").value;
      const baseSeries = selectedBase && series.includes(selectedBase) ? selectedBase : (series[0] || "");
      const compareSeries = selectedCompare && series.includes(selectedCompare) ? selectedCompare : (series.find(s => s !== baseSeries) || "");
      const byKey = new Map();
      for (const row of rows) {
        const key = `${row.task}||${row.step}`;
        if (!byKey.has(key)) byKey.set(key, { task: row.task, step: row.step, values: {} });
        byKey.get(key).values[seriesKey(row)] = metricValue(row, metric);
      }

      const diffHeader = series.length >= 2 ? `<th>Diff<br><span class="muted">${esc(compareSeries)} - ${esc(baseSeries)}</span></th>` : "";
      document.getElementById("compare-head").innerHTML = `<tr><th>Task</th><th>Step</th>${series.map(s => `<th>${esc(s)}<br><span class="muted">${esc(archMap[s] || "")}</span></th>`).join("")}${diffHeader}<th>Best</th></tr>`;
      const items = [...byKey.values()].sort((a, b) => String(a.task).localeCompare(String(b.task)) || Number(a.step) - Number(b.step));
      const diffByTask = new Map();
      let allDiffSum = 0;
      let allDiffN = 0;

      document.getElementById("compare-rows").innerHTML = items.map(item => {
        const vals = series.map(s => item.values[s]);
        const best = vals.filter(v => v !== null && v !== undefined).reduce((a, b) => Math.max(a, b), -Infinity);
        const bestSeries = best === -Infinity ? "" : series.filter(s => item.values[s] === best).join(", ");
        let diffCell = "";
        if (series.length >= 2) {
          const a = item.values[baseSeries];
          const b = item.values[compareSeries];
          const hasBoth = a !== null && a !== undefined && b !== null && b !== undefined;
          const diff = hasBoth ? b - a : null;
          if (hasBoth) {
            const current = diffByTask.get(item.task) || { sum: 0, n: 0 };
            current.sum += diff;
            current.n += 1;
            diffByTask.set(item.task, current);
            allDiffSum += diff;
            allDiffN += 1;
          }
          const cls = diff === null ? "metric muted" : diff >= 0 ? "metric delta-pos" : "metric delta-neg";
          diffCell = `<td class="${cls}">${diffText(diff)}</td>`;
        }
        return `<tr>
          <td>${esc(item.task)}</td>
          <td class="metric">${esc(item.step)}</td>
          ${series.map(s => {
            const value = item.values[s];
            const cls = value !== null && value !== undefined && value === best ? "metric best" : "metric";
            return `<td class="${cls}">${pct(value)}</td>`;
          }).join("")}
          ${diffCell}
          <td>${esc(bestSeries)}</td>
        </tr>`;
      }).join("");

      if (series.length < 2) {
        document.getElementById("compare-summary").textContent = "Average diff needs at least two model series.";
        return;
      }
      if (!baseSeries || !compareSeries || baseSeries === compareSeries) {
        document.getElementById("compare-summary").textContent = "Choose two different models for diff.";
        return;
      }
      const taskParts = [...diffByTask.entries()]
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([task, stat]) => `${task}: ${diffText(stat.sum / stat.n)} (n=${stat.n})`);
      const overall = allDiffN > 0 ? diffText(allDiffSum / allDiffN) : "n/a";
      document.getElementById("compare-summary").textContent =
        `Diff = ${compareSeries} - ${baseSeries}. Average diff: ${overall} (paired n=${allDiffN})`
        + (taskParts.length ? ` | ${taskParts.join(" | ")}` : "");
    }

    async function showLog(path) {
      const pre = document.getElementById("log");
      if (!path) {
        pre.style.display = "block";
        pre.textContent = "No stdout log path for this row yet.";
        return;
      }
      const res = await fetch(`/api/log?path=${encodeURIComponent(path)}`, { cache: "no-store" });
      const text = await res.text();
      pre.style.display = "block";
      pre.textContent = text;
    }

    function renderRows() {
      const tbody = document.getElementById("rows");
      tbody.innerHTML = filteredRows().map(row => {
        const m = row.metrics || {};
        const status = row.status || "";
        return `<tr onclick="showLog('${esc(row.stdout_log || "")}')">
          <td>${esc(seriesKey(row))}</td>
          <td>${esc(row.model_arch || "")}</td>
          <td>${esc(row.task)}</td>
          <td class="metric">${esc(row.step)}</td>
          <td>${esc(row.checkpoint_name)}</td>
          <td><span class="status ${esc(status)}"><span class="dot"></span>${esc(status || "unknown")}</span></td>
          <td class="metric">${pct(m.overall_micro)}</td>
          <td class="metric">${pct(m.overall_macro)}</td>
          <td class="metric">${pct(m.overall_acc_norm ?? m.acc_norm)}</td>
          <td class="muted">${sec(row.duration_sec)}</td>
        </tr>`;
      }).join("");
    }

    function render() {
      renderCards();
      renderSeriesOptions();
      renderDiffOptions();
      renderArchOptions();
      renderTaskOptions();
      renderModelStrip();
      renderTrend();
      renderCompare();
      renderRows();
    }

    document.getElementById("query").addEventListener("input", render);
    document.getElementById("series").addEventListener("change", render);
    document.getElementById("arch").addEventListener("change", render);
    document.getElementById("task").addEventListener("change", render);
    document.getElementById("status").addEventListener("change", render);
    document.getElementById("metric").addEventListener("change", render);
    document.getElementById("diff-base").addEventListener("change", render);
    document.getElementById("diff-compare").addEventListener("change", render);
    load();
    setInterval(load, refreshMs);
  </script>
</body>
</html>
"""


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KRong Eval Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface-alt: #f0f3f8;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee8;
      --line-strong: #b8c2d1;
      --blue: #2563eb;
      --green: #16805d;
      --red: #c2413d;
      --amber: #a16207;
      --violet: #6d5bd0;
      --shadow: 0 10px 28px rgba(21, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: "Aptos", "Segoe UI", "Noto Sans", sans-serif;
      font-size: 14px;
    }
    button, input, select {
      font: inherit;
      color: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--line-strong); background: var(--surface-alt); }
    select, input {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      padding: 8px 10px;
      outline: none;
    }
    select:focus, input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12); }
    .app {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #f9fbfe;
      padding: 18px;
      overflow: auto;
    }
    main {
      min-width: 0;
      padding: 18px 22px 28px;
      overflow: auto;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.25;
      font-weight: 760;
    }
    h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.3;
      font-weight: 760;
    }
    .caption {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .sidebar-head {
      display: grid;
      gap: 6px;
      margin-bottom: 18px;
    }
    .field {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }
    .field label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 720;
      text-transform: uppercase;
    }
    .segmented {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
      padding: 4px;
      background: var(--surface-alt);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .segmented button {
      border: 0;
      background: transparent;
      padding: 7px 8px;
    }
    .segmented button.active {
      color: white;
      background: var(--blue);
    }
    .model-list {
      display: grid;
      gap: 6px;
      max-height: 260px;
      overflow: auto;
      padding-right: 4px;
    }
    .check-row {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
    }
    .check-row input { width: auto; }
    .check-row span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 760;
      background: #e7eefb;
      color: #1d4ed8;
    }
    .badge.plain { background: #eef2f7; color: #475467; }
    .button-row {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .button-row button { flex: 1; min-width: 76px; }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 18px;
      margin-bottom: 14px;
    }
    .pathline {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }
    .status-line {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .stat {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
    }
    .stat .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 760;
    }
    .stat .value {
      margin-top: 6px;
      font-size: 20px;
      font-weight: 780;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      margin-bottom: 12px;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .panel-tools {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .panel-tools select, .panel-tools input { width: auto; min-width: 120px; }
    .chart-wrap {
      position: relative;
      padding: 10px 12px 12px;
    }
    .chart-selection {
      position: absolute;
      display: none;
      top: 10px;
      bottom: 12px;
      border: 1px solid rgba(37, 99, 235, 0.85);
      background: rgba(37, 99, 235, 0.14);
      pointer-events: none;
      border-radius: 4px;
    }
    .chart {
      width: 100%;
      height: 460px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background:
        linear-gradient(#ffffff, #ffffff),
        repeating-linear-gradient(0deg, transparent, transparent 31px, rgba(102,112,133,0.08) 32px);
      touch-action: none;
      cursor: crosshair;
    }
    .chart:active { cursor: crosshair; }
    .axis { stroke: #9aa4b2; stroke-width: 1.2; }
    .grid { stroke: #e4e9f2; stroke-width: 1; }
    .series-line { fill: none; stroke-width: 2.4; }
    .series-point { stroke: white; stroke-width: 2; }
    .label-text { fill: var(--muted); font-size: 11px; }
    .tooltip {
      position: fixed;
      z-index: 30;
      display: none;
      max-width: 280px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111827;
      color: white;
      box-shadow: var(--shadow);
      pointer-events: none;
      font-size: 12px;
      line-height: 1.4;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      padding: 10px 2px 0;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
    }
    .swatch { width: 10px; height: 10px; border-radius: 99px; }
    .chart-panel.expanded {
      position: fixed;
      inset: 14px;
      z-index: 20;
      margin: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .chart-panel.expanded .chart-wrap { min-height: 0; display: grid; grid-template-rows: minmax(0, 1fr) auto; }
    .chart-panel.expanded .chart { height: 100%; min-height: 560px; }
    .matrix-wrap, .runs-wrap {
      overflow: auto;
      max-height: 460px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      color: var(--muted);
      background: #f7f9fc;
      font-size: 11px;
      text-transform: uppercase;
    }
    tr:hover td { background: #f8fbff; }
    .num { font-variant-numeric: tabular-nums; font-weight: 720; }
    .good { color: var(--green); }
    .bad { color: var(--red); }
    .warn { color: var(--amber); }
    .muted { color: var(--muted); }
    .run-status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 720;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 99px;
      background: var(--muted);
    }
    .ok .dot { background: var(--green); }
    .failed .dot { background: var(--red); }
    .pending .dot { background: var(--amber); }
    .running .dot { background: var(--blue); }
    pre {
      display: none;
      max-height: 280px;
      overflow: auto;
      white-space: pre-wrap;
      margin: 0;
      padding: 12px;
      color: #d7e2f1;
      background: #111827;
      border-radius: 0 0 8px 8px;
      font-size: 12px;
    }
    @media (max-width: 1000px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .topbar { flex-direction: column; }
      .status-line { justify-content: flex-start; }
      .chart { height: 360px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="sidebar-head">
        <h1>KRong Eval Dashboard</h1>
        <div class="caption" id="root-label">Loading...</div>
      </div>

      <div class="field">
        <label for="query">Search</label>
        <input id="query" placeholder="model, task, checkpoint, arch" />
      </div>

      <div class="field">
        <label for="metric">Metric</label>
        <select id="metric">
          <option value="overall_micro">Micro</option>
          <option value="overall_macro">Macro</option>
          <option value="overall_acc_norm">Acc Norm</option>
          <option value="acc">Acc</option>
          <option value="f1">F1</option>
        </select>
      </div>

      <div class="field">
        <label for="task-filter">Task</label>
        <select id="task-filter"><option value="">All tasks</option></select>
      </div>

      <div class="field">
        <label for="arch-filter">Architecture</label>
        <select id="arch-filter"><option value="">All archs</option></select>
      </div>

      <div class="field">
        <label for="status-filter">Status</label>
        <select id="status-filter">
          <option value="">All statuses</option>
          <option value="ok">ok</option>
          <option value="running">running</option>
          <option value="pending">pending</option>
          <option value="failed">failed</option>
          <option value="skipped_existing_json">skipped_existing_json</option>
        </select>
      </div>

      <div class="field">
        <label>Y Scale</label>
        <div class="segmented" id="scale-control">
          <button data-scale="auto" class="active">Auto</button>
          <button data-scale="tight">Tight</button>
          <button data-scale="fixed">0-100</button>
        </div>
      </div>

      <div class="field">
        <label>Step Window</label>
        <div class="button-row">
          <input id="step-min" placeholder="min" />
          <input id="step-max" placeholder="max" />
        </div>
        <div class="button-row">
          <button id="fit-steps">Fit</button>
          <button id="zoom-in">Zoom In</button>
          <button id="zoom-out">Zoom Out</button>
        </div>
      </div>

      <div class="field">
        <label for="baseline-select">Baseline</label>
        <select id="baseline-select"><option value="">Baseline</option></select>
      </div>

      <div class="field">
        <label for="compare-select">Compare</label>
        <select id="compare-select"><option value="">Compare</option></select>
      </div>

      <div class="field">
        <label>Models</label>
        <div class="button-row">
          <button id="models-all">All</button>
          <button id="models-none">None</button>
        </div>
        <div class="model-list" id="model-list"></div>
      </div>
    </aside>

    <main>
      <div class="topbar">
        <div>
          <h1>Evaluation Overview</h1>
          <div class="pathline" id="updated-label">Waiting for data</div>
        </div>
        <div class="status-line" id="status-line"></div>
      </div>

      <section class="stat-grid">
        <div class="stat"><div class="label">Visible Runs</div><div class="value" id="stat-visible">0</div></div>
        <div class="stat"><div class="label">Finished</div><div class="value" id="stat-ok">0</div></div>
        <div class="stat"><div class="label">Best Visible</div><div class="value" id="stat-best">n/a</div></div>
        <div class="stat"><div class="label">Best Model</div><div class="value" id="stat-best-model">n/a</div></div>
        <div class="stat"><div class="label">Paired Diff</div><div class="value" id="stat-diff">n/a</div></div>
      </section>

      <section class="panel chart-panel" id="chart-panel">
        <div class="panel-head">
          <div>
            <h2>Score Trend</h2>
            <div class="caption" id="chart-note">Wheel or drag on the chart to zoom and pan step range.</div>
          </div>
          <div class="panel-tools">
            <select id="chart-task"><option value="__aggregate">Aggregate visible tasks</option></select>
            <button id="chart-reset">Reset View</button>
            <button id="chart-back">Back</button>
            <button id="chart-expand">Expand</button>
          </div>
        </div>
        <div class="chart-wrap">
          <div id="chart-selection" class="chart-selection"></div>
          <svg id="trend-chart" class="chart" viewBox="0 0 1000 440" role="img"></svg>
          <div class="legend" id="chart-legend"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Task Matrix</h2>
            <div class="caption">Each cell shows latest score and best score@step for the selected metric.</div>
          </div>
        </div>
        <div class="matrix-wrap">
          <table>
            <thead id="matrix-head"></thead>
            <tbody id="matrix-body"></tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Runs</h2>
            <div class="caption">Click a row to inspect the tail of the stdout log.</div>
          </div>
        </div>
        <div class="runs-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>Arch</th><th>Task</th><th>Step</th><th>Checkpoint</th><th>Status</th>
                <th>Micro</th><th>Macro</th><th>Acc Norm</th><th>F1</th><th>Duration</th>
              </tr>
            </thead>
            <tbody id="run-body"></tbody>
          </table>
        </div>
        <pre id="log-view"></pre>
      </section>
    </main>
  </div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const refreshMs = Number(new URLSearchParams(location.search).get("refresh") || "3000");
    const colors = ["#2563eb", "#16805d", "#c2413d", "#a16207", "#6d5bd0", "#0891b2", "#be185d", "#475467", "#d97706", "#059669"];
    let data = { rows: [] };
    let selectedModels = new Set();
    let modelSelectionInitialized = false;
    let selectedScale = "auto";
    let currentDomain = null;
    let domainHistory = [];
    let selectionDrag = null;

    const pct = (x) => x === null || x === undefined || Number.isNaN(Number(x)) ? "" : `${(Number(x) * 100).toFixed(2)}%`;
    const pp = (x) => x === null || x === undefined || Number.isNaN(Number(x)) ? "n/a" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(2)}pp`;
    const sec = (x) => x === null || x === undefined || x === "" ? "" : `${Number(x).toFixed(1)}s`;
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;" }[c]));
    const seriesKey = (row) => row.series || row.checkpoint_name || "model";
    const metricValue = (row, key) => {
      const m = row.metrics || {};
      if (key === "overall_acc_norm") return m.overall_acc_norm ?? m.acc_norm ?? null;
      if (key === "acc") return m.acc ?? m.overall_micro ?? null;
      if (key === "f1") return m.f1 ?? m.overall_f1 ?? null;
      return m[key] ?? null;
    };

    async function load() {
      const res = await fetch("/api/summary", { cache: "no-store" });
      data = await res.json();
      reconcileModelSelection();
      render();
    }

    function allSeries() {
      return [...new Set((data.rows || []).map(seriesKey).filter(Boolean))].sort();
    }

    function seriesArchMap(rows = data.rows || []) {
      const map = {};
      for (const row of rows) {
        const key = seriesKey(row);
        if (!map[key]) map[key] = new Set();
        if (row.model_arch) map[key].add(row.model_arch);
      }
      const out = {};
      for (const [key, val] of Object.entries(map)) out[key] = [...val].sort().join(", ") || "unknown";
      return out;
    }

    function reconcileModelSelection() {
      const series = allSeries();
      if (!modelSelectionInitialized) {
        selectedModels = new Set(series);
        modelSelectionInitialized = true;
      }
      selectedModels = new Set([...selectedModels].filter(s => series.includes(s)));
    }

    function baseFilteredRows({ includeModels = true } = {}) {
      const q = document.getElementById("query")?.value?.toLowerCase() || "";
      const task = document.getElementById("task-filter")?.value || "";
      const arch = document.getElementById("arch-filter")?.value || "";
      const status = document.getElementById("status-filter")?.value || "";
      return (data.rows || []).filter(row => {
        const key = seriesKey(row);
        const hay = `${key} ${row.model_arch || ""} ${row.task || ""} ${row.step || ""} ${row.checkpoint_name || ""} ${row.status || ""}`.toLowerCase();
        return (!q || hay.includes(q))
          && (!task || row.task === task)
          && (!arch || row.model_arch === arch)
          && (!status || row.status === status)
          && (!includeModels || selectedModels.has(key));
      });
    }

    function renderControls() {
      const rows = data.rows || [];
      const tasks = [...new Set(rows.map(r => r.task).filter(Boolean))].sort();
      const archs = [...new Set(rows.map(r => r.model_arch).filter(Boolean))].sort();
      fillSelect("task-filter", [["", "All tasks"], ...tasks.map(t => [t, t])]);
      fillSelect("arch-filter", [["", "All archs"], ...archs.map(a => [a, a])]);
      fillSelect("chart-task", [["__aggregate", "Aggregate visible tasks"], ...tasks.map(t => [t, t])]);

      const archMap = seriesArchMap(rows);
      const series = allSeries();
      fillSelect("baseline-select", [["", "Baseline"], ...series.map(s => [s, `${s} [${archMap[s] || "unknown"}]`])]);
      fillSelect("compare-select", [["", "Compare"], ...series.map(s => [s, `${s} [${archMap[s] || "unknown"}]`])]);

      document.getElementById("model-list").innerHTML = series.map((s, idx) => {
        const checked = selectedModels.has(s) ? "checked" : "";
        const arch = archMap[s] || "unknown";
        const badgeClass = arch.includes("krong") ? "" : " plain";
        return `<label class="check-row"><input type="checkbox" data-model="${esc(s)}" ${checked}><span title="${esc(s)}">${esc(s)}</span><span class="badge${badgeClass}">${esc(arch)}</span></label>`;
      }).join("");
      document.querySelectorAll("#model-list input").forEach(input => {
        input.addEventListener("change", () => {
          const model = input.dataset.model;
          if (input.checked) selectedModels.add(model);
          else selectedModels.delete(model);
          render();
        });
      });
    }

    function fillSelect(id, options) {
      const el = document.getElementById(id);
      const prev = el.value;
      el.innerHTML = options.map(([value, label]) => `<option value="${esc(value)}">${esc(label)}</option>`).join("");
      if (options.some(([value]) => value === prev)) el.value = prev;
    }

    function renderHeader() {
      const root = data.compare_root || data.result_root || "";
      const mode = data.mode === "compare" ? "compare root" : "result root";
      document.getElementById("root-label").textContent = `${mode}: ${root}`;
      document.getElementById("updated-label").textContent = `Updated ${data.updated_at || ""}`;
      document.getElementById("status-line").innerHTML = [
        `expected ${data.expected_runs ?? 0}`,
        `finished ${data.ok_runs ?? 0}`,
        `pending ${data.pending_runs ?? 0}`,
        `failed ${data.failed_runs ?? 0}`,
      ].map(text => `<span class="badge plain">${esc(text)}</span>`).join("");
    }

    function visibleOkRows() {
      const metric = document.getElementById("metric").value;
      return baseFilteredRows().filter(row => row.status === "ok" && metricValue(row, metric) !== null && metricValue(row, metric) !== undefined);
    }

    function renderStats() {
      const rows = baseFilteredRows();
      const okRows = visibleOkRows();
      const metric = document.getElementById("metric").value;
      const best = okRows.reduce((acc, row) => {
        const value = metricValue(row, metric);
        return value > acc.value ? { value, row } : acc;
      }, { value: -Infinity, row: null });
      const diff = pairedDiff();
      document.getElementById("stat-visible").textContent = rows.length;
      document.getElementById("stat-ok").textContent = rows.filter(r => r.status === "ok").length;
      document.getElementById("stat-best").textContent = best.row ? pct(best.value) : "n/a";
      document.getElementById("stat-best-model").textContent = best.row ? `${seriesKey(best.row)} / ${best.row.task} @ ${best.row.step}` : "n/a";
      document.getElementById("stat-diff").textContent = diff.n ? `${pp(diff.avg)} (${diff.n})` : "n/a";
    }

    function pairedDiff() {
      const metric = document.getElementById("metric").value;
      const base = document.getElementById("baseline-select").value;
      const comp = document.getElementById("compare-select").value;
      if (!base || !comp || base === comp) return { avg: null, n: 0 };
      const map = new Map();
      for (const row of visibleOkRows()) {
        const key = `${row.task}||${row.step}`;
        if (!map.has(key)) map.set(key, {});
        map.get(key)[seriesKey(row)] = metricValue(row, metric);
      }
      let sum = 0, n = 0;
      for (const vals of map.values()) {
        if (vals[base] !== undefined && vals[comp] !== undefined) {
          sum += vals[comp] - vals[base];
          n += 1;
        }
      }
      return { avg: n ? sum / n : null, n };
    }

    function chartRows() {
      const metric = document.getElementById("metric").value;
      const chartTask = document.getElementById("chart-task").value;
      const rows = visibleOkRows();
      if (chartTask !== "__aggregate") {
        return rows.filter(r => r.task === chartTask).map(r => ({ series: seriesKey(r), step: Number(r.step || 0), value: metricValue(r, metric), task: r.task }));
      }
      const grouped = new Map();
      for (const row of rows) {
        const key = `${seriesKey(row)}||${row.step}`;
        if (!grouped.has(key)) grouped.set(key, { series: seriesKey(row), step: Number(row.step || 0), values: [] });
        grouped.get(key).values.push(metricValue(row, metric));
      }
      return [...grouped.values()].map(g => ({ series: g.series, step: g.step, value: g.values.reduce((a, b) => a + b, 0) / g.values.length, task: "aggregate" }));
    }

    function chartPlotRect(svg) {
      const bbox = svg.getBoundingClientRect();
      const width = 1000, height = 440;
      const ml = 58, mr = 24, mt = 20, mb = 46;
      const sx = bbox.width / width;
      const sy = bbox.height / height;
      return {
        bbox,
        ml: ml * sx,
        mr: mr * sx,
        mt: mt * sy,
        mb: mb * sy,
        plotLeft: bbox.left + ml * sx,
        plotRight: bbox.left + (width - mr) * sx,
        plotTop: bbox.top + mt * sy,
        plotBottom: bbox.top + (height - mb) * sy,
      };
    }

    function stepFromClientX(clientX) {
      const svg = document.getElementById("trend-chart");
      const rect = chartPlotRect(svg);
      if (!currentDomain) renderChart();
      const x = Math.min(Math.max(clientX, rect.plotLeft), rect.plotRight);
      const ratio = (x - rect.plotLeft) / Math.max(rect.plotRight - rect.plotLeft, 1);
      return currentDomain.xMin + ratio * (currentDomain.xMax - currentDomain.xMin);
    }

    function setSelectionBox(startX, currentX) {
      const box = document.getElementById("chart-selection");
      const svg = document.getElementById("trend-chart");
      const rect = chartPlotRect(svg);
      const left = Math.min(Math.max(startX, rect.plotLeft), Math.max(Math.min(currentX, rect.plotRight), rect.plotLeft));
      const right = Math.max(Math.min(Math.max(startX, rect.plotLeft), rect.plotRight), Math.min(Math.max(currentX, rect.plotLeft), rect.plotRight));
      box.style.display = "block";
      box.style.left = `${left - rect.bbox.left}px`;
      box.style.width = `${Math.max(right - left, 1)}px`;
      box.style.top = `${rect.plotTop - rect.bbox.top}px`;
      box.style.height = `${rect.plotBottom - rect.plotTop}px`;
    }

    function hideSelectionBox() {
      const box = document.getElementById("chart-selection");
      box.style.display = "none";
      box.style.width = "0px";
    }

    function renderChart() {
      const svg = document.getElementById("trend-chart");
      const legend = document.getElementById("chart-legend");
      const points = chartRows().filter(p => Number.isFinite(p.step) && Number.isFinite(p.value));
      if (!points.length) {
        svg.innerHTML = `<text x="500" y="220" text-anchor="middle" class="label-text">No completed data for this view</text>`;
        legend.innerHTML = "";
        return;
      }
      const width = 1000, height = 440;
      const ml = 58, mr = 24, mt = 20, mb = 46;
      const plotW = width - ml - mr;
      const plotH = height - mt - mb;
      const allSteps = [...new Set(points.map(p => p.step))].sort((a, b) => a - b);
      let xMin = currentDomain?.xMin ?? Number(document.getElementById("step-min").value || allSteps[0]);
      let xMax = currentDomain?.xMax ?? Number(document.getElementById("step-max").value || allSteps[allSteps.length - 1]);
      if (!Number.isFinite(xMin)) xMin = allSteps[0];
      if (!Number.isFinite(xMax)) xMax = allSteps[allSteps.length - 1];
      if (xMin > xMax) [xMin, xMax] = [xMax, xMin];
      if (xMin === xMax) { xMin -= 1; xMax += 1; }
      const allMin = allSteps[0];
      const allMax = allSteps[allSteps.length - 1];
      const span = xMax - xMin;
      if (span >= allMax - allMin) {
        xMin = allMin;
        xMax = allMax;
        if (xMin === xMax) { xMin -= 1; xMax += 1; }
      } else {
        if (xMin < allMin) { xMax += allMin - xMin; xMin = allMin; }
        if (xMax > allMax) { xMin -= xMax - allMax; xMax = allMax; }
      }
      currentDomain = { xMin, xMax };
      document.getElementById("step-min").value = Math.round(xMin);
      document.getElementById("step-max").value = Math.round(xMax);

      const visible = points.filter(p => p.step >= xMin && p.step <= xMax);
      const values = visible.map(p => p.value);
      let yMin = 0, yMax = 1;
      if (selectedScale !== "fixed" && values.length) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const pad = selectedScale === "tight" ? Math.max((max - min) * 0.18, 0.015) : Math.max((max - min) * 0.35, 0.04);
        yMin = Math.max(0, min - pad);
        yMax = Math.min(1, max + pad);
        if (yMax - yMin < 0.08) {
          const mid = (yMax + yMin) / 2;
          yMin = Math.max(0, mid - 0.04);
          yMax = Math.min(1, mid + 0.04);
        }
      }

      const xFor = step => ml + ((step - xMin) / (xMax - xMin)) * plotW;
      const yFor = value => mt + (1 - (value - yMin) / (yMax - yMin)) * plotH;
      const yTicks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i / 4));
      const xTicks = Array.from({ length: 5 }, (_, i) => xMin + ((xMax - xMin) * i / 4));
      const series = [...new Set(visible.map(p => p.series))].sort();
      const colorFor = Object.fromEntries(series.map((s, idx) => [s, colors[idx % colors.length]]));
      const lines = series.map(s => {
        const pts = visible.filter(p => p.series === s).sort((a, b) => a.step - b.step);
        const poly = pts.map(p => `${xFor(p.step)},${yFor(p.value)}`).join(" ");
        const circles = pts.map(p => `<circle class="series-point" r="4.2" cx="${xFor(p.step)}" cy="${yFor(p.value)}" fill="${colorFor[s]}" data-series="${esc(s)}" data-step="${p.step}" data-value="${p.value}"></circle>`).join("");
        return `${poly ? `<polyline class="series-line" stroke="${colorFor[s]}" points="${poly}"></polyline>` : ""}${circles}`;
      }).join("");
      const grid = yTicks.map(t => {
        const y = yFor(t);
        return `<line class="grid" x1="${ml}" y1="${y}" x2="${ml + plotW}" y2="${y}"></line><text class="label-text" x="${ml - 8}" y="${y + 4}" text-anchor="end">${(t * 100).toFixed(1)}%</text>`;
      }).join("");
      const xLabels = xTicks.map(t => `<text class="label-text" x="${xFor(t)}" y="${height - 16}" text-anchor="middle">${Math.round(t)}</text>`).join("");
      svg.innerHTML = `
        ${grid}
        <line class="axis" x1="${ml}" y1="${mt + plotH}" x2="${ml + plotW}" y2="${mt + plotH}"></line>
        <line class="axis" x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + plotH}"></line>
        ${xLabels}
        <text class="label-text" x="${ml + plotW}" y="${height - 4}" text-anchor="end">step</text>
        ${lines}
      `;
      const archMap = seriesArchMap(baseFilteredRows());
      legend.innerHTML = series.map(s => {
        const latest = visible.filter(p => p.series === s).sort((a, b) => b.step - a.step)[0];
        return `<span class="legend-item"><span class="swatch" style="background:${colorFor[s]}"></span><span>${esc(s)}</span><span class="badge ${archMap[s]?.includes("krong") ? "" : "plain"}">${esc(archMap[s] || "unknown")}</span><span class="num">${latest ? pct(latest.value) : ""}</span></span>`;
      }).join("");
      bindChartPoints();
      document.getElementById("chart-note").textContent = `${document.getElementById("chart-task").selectedOptions[0]?.text || ""} · ${document.getElementById("metric").selectedOptions[0]?.text || ""} · y ${pct(yMin)} to ${pct(yMax)} · step ${Math.round(xMin)} to ${Math.round(xMax)}`;
    }

    function bindChartPoints() {
      const tooltip = document.getElementById("tooltip");
      document.querySelectorAll(".series-point").forEach(point => {
        point.addEventListener("mousemove", event => {
          tooltip.style.display = "block";
          tooltip.style.left = `${event.clientX + 12}px`;
          tooltip.style.top = `${event.clientY + 12}px`;
          tooltip.innerHTML = `<strong>${esc(point.dataset.series)}</strong><br>step ${esc(point.dataset.step)} · ${pct(Number(point.dataset.value))}`;
        });
        point.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
      });
    }

    function renderMatrix() {
      const metric = document.getElementById("metric").value;
      const rows = visibleOkRows();
      const series = [...new Set(rows.map(seriesKey))].sort();
      const archMap = seriesArchMap(rows);
      const tasks = [...new Set(rows.map(r => r.task).filter(Boolean))].sort();
      document.getElementById("matrix-head").innerHTML = `<tr><th>Task</th>${series.map(s => `<th>${esc(s)}<br><span class="muted">${esc(archMap[s] || "")}</span></th>`).join("")}</tr>`;
      document.getElementById("matrix-body").innerHTML = tasks.map(task => {
        return `<tr><td>${esc(task)}</td>${series.map(s => {
          const items = rows.filter(r => r.task === task && seriesKey(r) === s).sort((a, b) => Number(a.step) - Number(b.step));
          if (!items.length) return `<td class="muted">-</td>`;
          const latest = items[items.length - 1];
          const best = items.reduce((acc, row) => metricValue(row, metric) > metricValue(acc, metric) ? row : acc, items[0]);
          return `<td><span class="num">${pct(metricValue(latest, metric))}</span><br><span class="muted">best ${pct(metricValue(best, metric))} @ ${esc(best.step)}</span></td>`;
        }).join("")}</tr>`;
      }).join("");
    }

    function renderRuns() {
      const tbody = document.getElementById("run-body");
      tbody.innerHTML = baseFilteredRows().map(row => {
        const m = row.metrics || {};
        const status = row.status || "";
        return `<tr data-log="${esc(row.stdout_log || "")}">
          <td>${esc(seriesKey(row))}</td>
          <td><span class="badge ${row.model_arch === "krong" ? "" : "plain"}">${esc(row.model_arch || "unknown")}</span></td>
          <td>${esc(row.task)}</td>
          <td class="num">${esc(row.step)}</td>
          <td>${esc(row.checkpoint_name)}</td>
          <td><span class="run-status ${esc(status)}"><span class="dot"></span>${esc(status || "unknown")}</span></td>
          <td class="num">${pct(m.overall_micro)}</td>
          <td class="num">${pct(m.overall_macro)}</td>
          <td class="num">${pct(m.overall_acc_norm ?? m.acc_norm)}</td>
          <td class="num">${pct(m.overall_f1 ?? m.f1)}</td>
          <td class="muted">${sec(row.duration_sec)}</td>
        </tr>`;
      }).join("");
      tbody.querySelectorAll("tr").forEach(tr => tr.addEventListener("click", () => showLog(tr.dataset.log || "")));
    }

    async function showLog(path) {
      const pre = document.getElementById("log-view");
      pre.style.display = "block";
      if (!path) {
        pre.textContent = "No stdout log path for this row.";
        return;
      }
      const res = await fetch(`/api/log?path=${encodeURIComponent(path)}`, { cache: "no-store" });
      pre.textContent = await res.text();
    }

    function render() {
      renderControls();
      renderHeader();
      renderStats();
      renderChart();
      renderMatrix();
      renderRuns();
    }

    function resetDomain() {
      domainHistory = [];
      currentDomain = null;
      document.getElementById("step-min").value = "";
      document.getElementById("step-max").value = "";
      renderChart();
    }

    function zoomDomain(factor) {
      const pts = chartRows();
      if (!pts.length) return;
      const steps = pts.map(p => p.step);
      const minAll = Math.min(...steps), maxAll = Math.max(...steps);
      const domain = currentDomain || { xMin: minAll, xMax: maxAll };
      const mid = (domain.xMin + domain.xMax) / 2;
      const half = Math.max(1, (domain.xMax - domain.xMin) * factor / 2);
      if (currentDomain) domainHistory.push({ ...currentDomain });
      currentDomain = { xMin: Math.max(minAll, mid - half), xMax: Math.min(maxAll, mid + half) };
      renderChart();
    }

    function zoomToStepRange(a, b) {
      const pts = chartRows();
      if (!pts.length) return;
      const steps = pts.map(p => p.step);
      const minAll = Math.min(...steps), maxAll = Math.max(...steps);
      let xMin = Math.max(minAll, Math.min(a, b));
      let xMax = Math.min(maxAll, Math.max(a, b));
      if (xMax - xMin < 1) return;
      if (currentDomain) domainHistory.push({ ...currentDomain });
      currentDomain = { xMin, xMax };
      renderChart();
    }

    function backDomain() {
      const previous = domainHistory.pop();
      if (previous) {
        currentDomain = previous;
        renderChart();
      }
    }

    function setupEvents() {
      ["query", "metric", "task-filter", "arch-filter", "status-filter", "baseline-select", "compare-select", "chart-task"].forEach(id => {
        document.getElementById(id).addEventListener("input", () => { if (id !== "chart-task") currentDomain = currentDomain; render(); });
        document.getElementById(id).addEventListener("change", render);
      });
      document.getElementById("models-all").addEventListener("click", () => { selectedModels = new Set(allSeries()); render(); });
      document.getElementById("models-none").addEventListener("click", () => { selectedModels = new Set(); render(); });
      document.getElementById("fit-steps").addEventListener("click", resetDomain);
      document.getElementById("chart-reset").addEventListener("click", resetDomain);
      document.getElementById("chart-back").addEventListener("click", backDomain);
      document.getElementById("zoom-in").addEventListener("click", () => zoomDomain(0.65));
      document.getElementById("zoom-out").addEventListener("click", () => zoomDomain(1.55));
      document.getElementById("step-min").addEventListener("change", () => { currentDomain = { xMin: Number(document.getElementById("step-min").value), xMax: Number(document.getElementById("step-max").value) }; renderChart(); });
      document.getElementById("step-max").addEventListener("change", () => { currentDomain = { xMin: Number(document.getElementById("step-min").value), xMax: Number(document.getElementById("step-max").value) }; renderChart(); });
      document.querySelectorAll("#scale-control button").forEach(button => {
        button.addEventListener("click", () => {
          document.querySelectorAll("#scale-control button").forEach(x => x.classList.remove("active"));
          button.classList.add("active");
          selectedScale = button.dataset.scale;
          renderChart();
        });
      });
      const panel = document.getElementById("chart-panel");
      document.getElementById("chart-expand").addEventListener("click", () => {
        panel.classList.toggle("expanded");
        document.getElementById("chart-expand").textContent = panel.classList.contains("expanded") ? "Close" : "Expand";
      });
      document.addEventListener("keydown", event => {
        if (event.key === "Escape" && panel.classList.contains("expanded")) {
          panel.classList.remove("expanded");
          document.getElementById("chart-expand").textContent = "Expand";
        }
      });
      const svg = document.getElementById("trend-chart");
      svg.addEventListener("wheel", event => {
        event.preventDefault();
        if (!currentDomain) renderChart();
        const focus = stepFromClientX(event.clientX);
        const factor = event.deltaY < 0 ? 0.82 : 1.22;
        const left = focus - currentDomain.xMin;
        const right = currentDomain.xMax - focus;
        if (currentDomain) domainHistory.push({ ...currentDomain });
        currentDomain = { xMin: focus - left * factor, xMax: focus + right * factor };
        renderChart();
      }, { passive: false });
      svg.addEventListener("mousedown", event => {
        if (event.button !== 0) return;
        if (!currentDomain) renderChart();
        selectionDrag = { startX: event.clientX, startStep: stepFromClientX(event.clientX) };
        setSelectionBox(event.clientX, event.clientX);
      });
      window.addEventListener("mousemove", event => {
        if (!selectionDrag) return;
        setSelectionBox(selectionDrag.startX, event.clientX);
      });
      window.addEventListener("mouseup", event => {
        if (!selectionDrag) return;
        const endStep = stepFromClientX(event.clientX);
        const startStep = selectionDrag.startStep;
        const movedPx = Math.abs(event.clientX - selectionDrag.startX);
        selectionDrag = null;
        hideSelectionBox();
        if (movedPx >= 12) zoomToStepRange(startStep, endStep);
      });
      svg.addEventListener("dblclick", resetDomain);
    }

    setupEvents();
    load();
    setInterval(load, refreshMs);
  </script>
</body>
</html>
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KRong Eval Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #111827;
      --muted: #667085;
      --line: #d9e0ea;
      --line-strong: #b8c3d2;
      --blue: #1d5fd6;
      --green: #0f7a5a;
      --red: #c13b34;
      --amber: #a15c00;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", "Noto Sans", sans-serif;
      font-size: 14px;
    }
    button, input, select { font: inherit; color: inherit; }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--line-strong); background: var(--panel-soft); }
    button.primary { background: var(--blue); color: white; border-color: var(--blue); }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 8px 9px;
      outline: none;
    }
    input:focus, select:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(29, 95, 214, .12); }
    .layout { display: grid; grid-template-columns: 320px minmax(0, 1fr); min-height: 100vh; }
    aside {
      padding: 18px;
      background: #fbfcfe;
      border-right: 1px solid var(--line);
      overflow: auto;
    }
    main { padding: 18px 22px 28px; min-width: 0; }
    h1 { margin: 0; font-size: 22px; line-height: 1.25; }
    h2 { margin: 0; font-size: 15px; }
    .caption { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .header { display: flex; align-items: start; justify-content: space-between; gap: 18px; margin-bottom: 14px; }
    .root { margin-top: 5px; color: var(--muted); font-size: 12px; word-break: break-all; }
    .field { display: grid; gap: 6px; margin-bottom: 12px; }
    .field label { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .small-button-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
    .segmented {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }
    .segmented button { border: 0; background: transparent; padding: 7px 8px; }
    .segmented button.active { background: var(--blue); color: white; }
    .model-list { display: grid; gap: 6px; max-height: 260px; overflow: auto; padding-right: 4px; }
    .model-row {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 7px 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .model-row input { width: auto; }
    .model-row .name { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 800;
      color: #1d4ed8;
      background: #e8f0ff;
    }
    .badge.plain { color: #475467; background: #eef2f6; }
    .badge.ok { color: var(--green); background: #e6f5ef; }
    .badge.warn { color: var(--amber); background: #fff3dc; }
    .badge.bad { color: var(--red); background: #ffe8e6; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .card .label { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .card .value { margin-top: 6px; font-weight: 850; font-size: 18px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      margin-bottom: 12px;
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fcfdff;
    }
    .panel-tools { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .panel-tools select { width: auto; min-width: 140px; }
    .chart-area { padding: 12px 14px 14px; }
    .chart-box { position: relative; border: 1px solid var(--line); border-radius: 6px; background: white; }
    svg { display: block; width: 100%; height: 440px; }
    .grid { stroke: #e7ecf3; stroke-width: 1; }
    .axis { stroke: #9aa4b2; stroke-width: 1.2; }
    .axis-label { fill: var(--muted); font-size: 11px; }
    .line { fill: none; stroke-width: 2.5; }
    .point { stroke: white; stroke-width: 2; }
    .legend { display: flex; gap: 10px 14px; flex-wrap: wrap; padding-top: 10px; }
    .legend-item { display: inline-flex; gap: 7px; align-items: center; color: var(--muted); font-size: 12px; }
    .swatch { width: 10px; height: 10px; border-radius: 999px; }
    .tooltip {
      position: fixed;
      display: none;
      z-index: 50;
      max-width: 260px;
      padding: 8px 10px;
      color: white;
      background: #111827;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.4;
      pointer-events: none;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { position: sticky; top: 0; z-index: 1; color: var(--muted); background: #f7f9fc; font-size: 11px; text-transform: uppercase; }
    tr:hover td { background: #f8fbff; }
    .table-wrap { max-height: 430px; overflow: auto; }
    .num { font-variant-numeric: tabular-nums; font-weight: 760; }
    .muted { color: var(--muted); }
    .good { color: var(--green); }
    .bad { color: var(--red); }
    .run-status { display: inline-flex; align-items: center; gap: 6px; font-weight: 760; }
    .dot { width: 8px; height: 8px; border-radius: 99px; background: var(--muted); }
    .ok .dot { background: var(--green); }
    .failed .dot { background: var(--red); }
    .pending .dot { background: var(--amber); }
    .running .dot { background: var(--blue); }
    pre {
      display: none;
      max-height: 280px;
      overflow: auto;
      margin: 0;
      padding: 12px;
      color: #d7e2f1;
      background: #111827;
      white-space: pre-wrap;
      font-size: 12px;
    }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      svg { height: 360px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="field">
        <h1>KRong Eval Dashboard</h1>
        <div class="caption" id="root-label">Loading...</div>
      </div>
      <div class="field">
        <label for="task">Task</label>
        <select id="task"></select>
      </div>
      <div class="field">
        <label for="metric">Metric</label>
        <select id="metric">
          <option value="overall_micro">Micro</option>
          <option value="overall_macro">Macro</option>
          <option value="overall_acc_norm">Acc Norm</option>
          <option value="acc">Acc</option>
          <option value="f1">F1</option>
        </select>
      </div>
      <div class="field">
        <label for="search">Search</label>
        <input id="search" placeholder="model, checkpoint, arch" />
      </div>
      <div class="field">
        <label for="arch">Architecture</label>
        <select id="arch"><option value="">All archs</option></select>
      </div>
      <div class="field">
        <label for="status">Status</label>
        <select id="status">
          <option value="">All statuses</option>
          <option value="ok">ok</option>
          <option value="running">running</option>
          <option value="pending">pending</option>
          <option value="failed">failed</option>
          <option value="skipped_existing_json">skipped_existing_json</option>
        </select>
      </div>
      <div class="field">
        <label>Step Range</label>
        <div class="row">
          <input id="step-min" placeholder="min step" />
          <input id="step-max" placeholder="max step" />
        </div>
        <div class="small-button-row">
          <button id="range-all">All</button>
          <button id="range-last5">Last 5</button>
          <button id="range-last10">Last 10</button>
          <button id="range-data">Fit</button>
        </div>
      </div>
      <div class="field">
        <label>Y Scale</label>
        <div class="segmented" id="scale">
          <button data-scale="auto" class="active">Auto</button>
          <button data-scale="tight">Tight</button>
          <button data-scale="fixed">0-100</button>
        </div>
      </div>
      <div class="field">
        <label for="baseline">Baseline</label>
        <select id="baseline"><option value="">Baseline</option></select>
      </div>
      <div class="field">
        <label for="compare">Compare</label>
        <select id="compare"><option value="">Compare</option></select>
      </div>
      <div class="field">
        <label>Models</label>
        <div class="row">
          <button id="model-all">All</button>
          <button id="model-none">None</button>
        </div>
        <div class="model-list" id="model-list"></div>
      </div>
    </aside>
    <main>
      <div class="header">
        <div>
          <h1 id="title">Evaluation Overview</h1>
          <div class="root" id="updated">Waiting for data</div>
        </div>
        <div class="row" style="max-width: 360px;">
          <span class="badge ok" id="ok-badge">ok 0</span>
          <span class="badge warn" id="pending-badge">pending 0</span>
          <span class="badge bad" id="failed-badge">failed 0</span>
        </div>
      </div>
      <section class="cards">
        <div class="card"><div class="label">Best Score</div><div class="value" id="best-score">n/a</div></div>
        <div class="card"><div class="label">Best Model</div><div class="value" id="best-model">n/a</div></div>
        <div class="card"><div class="label">Best Step</div><div class="value" id="best-step">n/a</div></div>
        <div class="card"><div class="label">Paired Diff</div><div class="value" id="paired-diff">n/a</div></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Step Trend</h2>
            <div class="caption" id="chart-caption">Choose a task and step range on the left.</div>
          </div>
          <div class="panel-tools">
            <button id="export-visible">Export Visible CSV</button>
          </div>
        </div>
        <div class="chart-area">
          <div class="chart-box">
            <svg id="chart" viewBox="0 0 1000 440" role="img"></svg>
          </div>
          <div class="legend" id="legend"></div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Model Summary</h2>
            <div class="caption">Latest, best, and gain are computed inside the selected step range.</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Arch</th><th>Latest</th><th>Latest Step</th><th>Best</th><th>Best Step</th><th>Gain</th><th>Runs</th></tr>
            </thead>
            <tbody id="summary-body"></tbody>
          </table>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Task Matrix</h2>
            <div class="caption">Latest score for each visible model and task. Use this to find task-specific regressions.</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="matrix-head"></thead>
            <tbody id="matrix-body"></tbody>
          </table>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Runs</h2>
            <div class="caption">Click a run to inspect stdout tail.</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Arch</th><th>Task</th><th>Step</th><th>Status</th><th>Micro</th><th>Macro</th><th>Acc Norm</th><th>F1</th><th>Duration</th></tr>
            </thead>
            <tbody id="runs-body"></tbody>
          </table>
        </div>
        <pre id="log"></pre>
      </section>
    </main>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const refreshMs = Number(new URLSearchParams(location.search).get("refresh") || "3000");
    const colors = ["#1d5fd6", "#0f7a5a", "#c13b34", "#a15c00", "#6d5bd0", "#0891b2", "#be185d", "#475467", "#d97706", "#059669"];
    let data = { rows: [] };
    let selectedModels = new Set();
    let initializedModels = false;
    let yScale = "auto";
    const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;" }[c]));
    const pct = v => v === null || v === undefined || Number.isNaN(Number(v)) ? "" : `${(Number(v) * 100).toFixed(2)}%`;
    const pp = v => v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}pp`;
    const seriesKey = row => row.series || row.checkpoint_name || "model";
    const metricValue = (row, key) => {
      const m = row.metrics || {};
      if (key === "overall_acc_norm") return m.overall_acc_norm ?? m.acc_norm ?? null;
      if (key === "acc") return m.acc ?? m.overall_micro ?? null;
      if (key === "f1") return m.f1 ?? m.overall_f1 ?? null;
      return m[key] ?? null;
    };
    async function load() {
      const res = await fetch("/api/summary", { cache: "no-store" });
      data = await res.json();
      reconcileModels();
      render();
    }
    function allSeries(rows = data.rows || []) {
      return [...new Set(rows.map(seriesKey).filter(Boolean))].sort();
    }
    function archMap(rows = data.rows || []) {
      const map = {};
      for (const row of rows) {
        const s = seriesKey(row);
        if (!map[s]) map[s] = new Set();
        if (row.model_arch) map[s].add(row.model_arch);
      }
      return Object.fromEntries(Object.entries(map).map(([k, v]) => [k, [...v].sort().join(", ") || "unknown"]));
    }
    function reconcileModels() {
      const series = allSeries();
      if (!initializedModels) {
        selectedModels = new Set(series);
        initializedModels = true;
      }
      selectedModels = new Set([...selectedModels].filter(s => series.includes(s)));
    }
    function fillSelect(id, items, keep = true) {
      const el = document.getElementById(id);
      const prev = el.value;
      el.innerHTML = items.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
      if (keep && items.some(([v]) => v === prev)) el.value = prev;
    }
    function selectedMetric() { return document.getElementById("metric").value; }
    function selectedTask() { return document.getElementById("task").value; }
    function rawRows() {
      const q = document.getElementById("search").value.toLowerCase();
      const arch = document.getElementById("arch").value;
      const status = document.getElementById("status").value;
      return (data.rows || []).filter(row => {
        const s = seriesKey(row);
        const hay = `${s} ${row.model_arch || ""} ${row.task || ""} ${row.checkpoint_name || ""} ${row.step || ""} ${row.status || ""}`.toLowerCase();
        return (!q || hay.includes(q)) && (!arch || row.model_arch === arch) && (!status || row.status === status) && selectedModels.has(s);
      });
    }
    function visibleRows({ taskOnly = true, okOnly = false, metricOnly = false } = {}) {
      const task = selectedTask();
      const metric = selectedMetric();
      const min = Number(document.getElementById("step-min").value);
      const max = Number(document.getElementById("step-max").value);
      return rawRows().filter(row => {
        const step = Number(row.step || 0);
        return (!taskOnly || !task || row.task === task)
          && (!okOnly || row.status === "ok")
          && (!metricOnly || metricValue(row, metric) !== null && metricValue(row, metric) !== undefined)
          && (!Number.isFinite(min) || step >= min)
          && (!Number.isFinite(max) || step <= max);
      });
    }
    function renderControls() {
      const rows = data.rows || [];
      const tasks = [...new Set(rows.map(r => r.task).filter(Boolean))].sort();
      const archs = [...new Set(rows.map(r => r.model_arch).filter(Boolean))].sort();
      fillSelect("task", tasks.map(t => [t, t]), true);
      if (!document.getElementById("task").value && tasks.length) document.getElementById("task").value = tasks[0];
      fillSelect("arch", [["", "All archs"], ...archs.map(a => [a, a])]);
      const map = archMap(rows);
      const series = allSeries(rows);
      fillSelect("baseline", [["", "Baseline"], ...series.map(s => [s, `${s} [${map[s] || "unknown"}]`])]);
      fillSelect("compare", [["", "Compare"], ...series.map(s => [s, `${s} [${map[s] || "unknown"}]`])]);
      document.getElementById("model-list").innerHTML = series.map(s => {
        const checked = selectedModels.has(s) ? "checked" : "";
        const arch = map[s] || "unknown";
        return `<label class="model-row"><input type="checkbox" data-model="${esc(s)}" ${checked}><span class="name" title="${esc(s)}">${esc(s)}</span><span class="badge ${arch.includes("krong") ? "" : "plain"}">${esc(arch)}</span></label>`;
      }).join("");
      document.querySelectorAll("#model-list input").forEach(input => {
        input.addEventListener("change", () => {
          if (input.checked) selectedModels.add(input.dataset.model);
          else selectedModels.delete(input.dataset.model);
          render();
        });
      });
    }
    function renderHeader() {
      const root = data.compare_root || data.result_root || "";
      document.getElementById("root-label").textContent = `${data.mode === "compare" ? "compare root" : "result root"}: ${root}`;
      document.getElementById("updated").textContent = `Updated ${data.updated_at || ""}`;
      document.getElementById("ok-badge").textContent = `ok ${data.ok_runs ?? 0}`;
      document.getElementById("pending-badge").textContent = `pending ${data.pending_runs ?? 0}`;
      document.getElementById("failed-badge").textContent = `failed ${data.failed_runs ?? 0}`;
      document.getElementById("title").textContent = `${selectedTask() || "Task"} · ${document.getElementById("metric").selectedOptions[0]?.text || ""}`;
    }
    function currentPoints() {
      const metric = selectedMetric();
      return visibleRows({ okOnly: true, metricOnly: true }).map(row => ({
        row,
        series: seriesKey(row),
        arch: row.model_arch || "unknown",
        step: Number(row.step || 0),
        value: metricValue(row, metric),
      })).filter(p => Number.isFinite(p.step) && Number.isFinite(p.value));
    }
    function modelSummaries() {
      const by = new Map();
      for (const p of currentPoints()) {
        if (!by.has(p.series)) by.set(p.series, []);
        by.get(p.series).push(p);
      }
      return [...by.entries()].map(([series, pts]) => {
        pts.sort((a, b) => a.step - b.step);
        const latest = pts[pts.length - 1];
        const first = pts[0];
        const best = pts.reduce((a, b) => b.value > a.value ? b : a, pts[0]);
        return { series, arch: latest.arch, latest, first, best, gain: latest.value - first.value, count: pts.length };
      }).sort((a, b) => b.latest.value - a.latest.value);
    }
    function renderStats() {
      const summaries = modelSummaries();
      const best = summaries.reduce((a, b) => !a || b.best.value > a.best.value ? b : a, null);
      document.getElementById("best-score").textContent = best ? pct(best.best.value) : "n/a";
      document.getElementById("best-model").textContent = best ? best.series : "n/a";
      document.getElementById("best-step").textContent = best ? best.best.step : "n/a";
      const base = document.getElementById("baseline").value;
      const comp = document.getElementById("compare").value;
      document.getElementById("paired-diff").textContent = pairedDiff(base, comp);
    }
    function pairedDiff(base, comp) {
      if (!base || !comp || base === comp) return "n/a";
      const buckets = new Map();
      for (const p of currentPoints()) {
        const key = `${p.row.task}||${p.step}`;
        if (!buckets.has(key)) buckets.set(key, {});
        buckets.get(key)[p.series] = p.value;
      }
      let sum = 0, n = 0;
      for (const vals of buckets.values()) {
        if (vals[base] !== undefined && vals[comp] !== undefined) {
          sum += vals[comp] - vals[base];
          n += 1;
        }
      }
      return n ? `${pp(sum / n)} (${n})` : "n/a";
    }
    function chartScale(points) {
      const steps = points.map(p => p.step);
      const values = points.map(p => p.value);
      let xMin = Math.min(...steps), xMax = Math.max(...steps);
      if (xMin === xMax) { xMin -= 1; xMax += 1; }
      let yMin = 0, yMax = 1;
      if (yScale !== "fixed" && values.length) {
        const min = Math.min(...values), max = Math.max(...values);
        const pad = yScale === "tight" ? Math.max((max - min) * .2, .01) : Math.max((max - min) * .35, .035);
        yMin = Math.max(0, min - pad);
        yMax = Math.min(1, max + pad);
        if (yMax - yMin < .06) {
          const mid = (yMin + yMax) / 2;
          yMin = Math.max(0, mid - .03);
          yMax = Math.min(1, mid + .03);
        }
      }
      return { xMin, xMax, yMin, yMax };
    }
    function renderChart() {
      const svg = document.getElementById("chart");
      const legend = document.getElementById("legend");
      const points = currentPoints();
      if (!points.length) {
        svg.innerHTML = `<text x="500" y="220" text-anchor="middle" class="axis-label">No completed data for this selection</text>`;
        legend.innerHTML = "";
        document.getElementById("chart-caption").textContent = "No completed data in the selected step range.";
        return;
      }
      const width = 1000, height = 440, ml = 58, mr = 22, mt = 22, mb = 48;
      const pw = width - ml - mr, ph = height - mt - mb;
      const s = chartScale(points);
      const x = step => ml + ((step - s.xMin) / (s.xMax - s.xMin)) * pw;
      const y = val => mt + (1 - (val - s.yMin) / (s.yMax - s.yMin)) * ph;
      const series = [...new Set(points.map(p => p.series))].sort();
      const color = Object.fromEntries(series.map((name, i) => [name, colors[i % colors.length]]));
      const yTicks = Array.from({ length: 5 }, (_, i) => s.yMin + (s.yMax - s.yMin) * i / 4);
      const xTicks = Array.from({ length: 6 }, (_, i) => s.xMin + (s.xMax - s.xMin) * i / 5);
      const grid = yTicks.map(t => `<line class="grid" x1="${ml}" y1="${y(t)}" x2="${ml + pw}" y2="${y(t)}"></line><text class="axis-label" x="${ml - 8}" y="${y(t) + 4}" text-anchor="end">${(t * 100).toFixed(1)}%</text>`).join("");
      const labels = xTicks.map(t => `<text class="axis-label" x="${x(t)}" y="${height - 16}" text-anchor="middle">${Math.round(t)}</text>`).join("");
      const lines = series.map(name => {
        const pts = points.filter(p => p.series === name).sort((a, b) => a.step - b.step);
        const poly = pts.map(p => `${x(p.step)},${y(p.value)}`).join(" ");
        const circles = pts.map(p => `<circle class="point" r="4.2" cx="${x(p.step)}" cy="${y(p.value)}" fill="${color[name]}" data-series="${esc(name)}" data-step="${p.step}" data-value="${p.value}"></circle>`).join("");
        return `${poly ? `<polyline class="line" stroke="${color[name]}" points="${poly}"></polyline>` : ""}${circles}`;
      }).join("");
      svg.innerHTML = `${grid}<line class="axis" x1="${ml}" y1="${mt + ph}" x2="${ml + pw}" y2="${mt + ph}"></line><line class="axis" x1="${ml}" y1="${mt}" x2="${ml}" y2="${mt + ph}"></line>${labels}<text class="axis-label" x="${ml + pw}" y="${height - 5}" text-anchor="end">step</text>${lines}`;
      const map = archMap(rawRows());
      legend.innerHTML = series.map(name => {
        const latest = points.filter(p => p.series === name).sort((a, b) => b.step - a.step)[0];
        return `<span class="legend-item"><span class="swatch" style="background:${color[name]}"></span><span>${esc(name)}</span><span class="badge ${map[name]?.includes("krong") ? "" : "plain"}">${esc(map[name] || "unknown")}</span><span class="num">${pct(latest.value)}</span></span>`;
      }).join("");
      document.getElementById("chart-caption").textContent = `step ${Math.round(s.xMin)}-${Math.round(s.xMax)} · y ${pct(s.yMin)}-${pct(s.yMax)} · ${points.length} points`;
      bindTooltip();
    }
    function bindTooltip() {
      const tip = document.getElementById("tooltip");
      document.querySelectorAll(".point").forEach(point => {
        point.addEventListener("mousemove", ev => {
          tip.style.display = "block";
          tip.style.left = `${ev.clientX + 12}px`;
          tip.style.top = `${ev.clientY + 12}px`;
          tip.innerHTML = `<strong>${esc(point.dataset.series)}</strong><br>step ${esc(point.dataset.step)} · ${pct(Number(point.dataset.value))}`;
        });
        point.addEventListener("mouseleave", () => tip.style.display = "none");
      });
    }
    function renderSummary() {
      document.getElementById("summary-body").innerHTML = modelSummaries().map(item => {
        const gainClass = item.gain >= 0 ? "good" : "bad";
        return `<tr><td>${esc(item.series)}</td><td><span class="badge ${item.arch === "krong" ? "" : "plain"}">${esc(item.arch)}</span></td><td class="num">${pct(item.latest.value)}</td><td class="num">${item.latest.step}</td><td class="num">${pct(item.best.value)}</td><td class="num">${item.best.step}</td><td class="num ${gainClass}">${pp(item.gain)}</td><td>${item.count}</td></tr>`;
      }).join("");
    }
    function renderMatrix() {
      const metric = selectedMetric();
      const rows = rawRows().filter(r => r.status === "ok" && metricValue(r, metric) !== null && metricValue(r, metric) !== undefined);
      const tasks = [...new Set(rows.map(r => r.task).filter(Boolean))].sort();
      const series = [...new Set(rows.map(seriesKey))].sort();
      const map = archMap(rows);
      document.getElementById("matrix-head").innerHTML = `<tr><th>Task</th>${series.map(s => `<th>${esc(s)}<br><span class="muted">${esc(map[s] || "")}</span></th>`).join("")}</tr>`;
      document.getElementById("matrix-body").innerHTML = tasks.map(task => `<tr><td>${esc(task)}</td>${series.map(s => {
        const items = rows.filter(r => r.task === task && seriesKey(r) === s).sort((a, b) => Number(a.step) - Number(b.step));
        if (!items.length) return `<td class="muted">-</td>`;
        const latest = items[items.length - 1];
        const best = items.reduce((a, b) => metricValue(b, metric) > metricValue(a, metric) ? b : a, items[0]);
        return `<td><span class="num">${pct(metricValue(latest, metric))}</span><br><span class="muted">best ${pct(metricValue(best, metric))} @ ${best.step}</span></td>`;
      }).join("")}</tr>`).join("");
    }
    function renderRuns() {
      const metric = selectedMetric();
      document.getElementById("runs-body").innerHTML = visibleRows({ taskOnly: false }).map(row => {
        const status = row.status || "";
        const m = row.metrics || {};
        return `<tr data-log="${esc(row.stdout_log || "")}"><td>${esc(seriesKey(row))}</td><td><span class="badge ${row.model_arch === "krong" ? "" : "plain"}">${esc(row.model_arch || "unknown")}</span></td><td>${esc(row.task)}</td><td class="num">${row.step}</td><td><span class="run-status ${esc(status)}"><span class="dot"></span>${esc(status)}</span></td><td class="num">${pct(m.overall_micro)}</td><td class="num">${pct(m.overall_macro)}</td><td class="num">${pct(m.overall_acc_norm ?? m.acc_norm)}</td><td class="num">${pct(m.overall_f1 ?? m.f1)}</td><td>${row.duration_sec ? Number(row.duration_sec).toFixed(1) + "s" : ""}</td></tr>`;
      }).join("");
      document.querySelectorAll("#runs-body tr").forEach(tr => tr.addEventListener("click", () => showLog(tr.dataset.log || "")));
    }
    async function showLog(path) {
      const pre = document.getElementById("log");
      pre.style.display = "block";
      if (!path) { pre.textContent = "No log path for this row."; return; }
      const res = await fetch(`/api/log?path=${encodeURIComponent(path)}`, { cache: "no-store" });
      pre.textContent = await res.text();
    }
    function render() {
      renderControls();
      renderHeader();
      renderStats();
      renderChart();
      renderSummary();
      renderMatrix();
      renderRuns();
    }
    function setRange(mode) {
      const pts = currentPoints();
      const all = rawRows().filter(r => selectedTask() ? r.task === selectedTask() : true).map(r => Number(r.step || 0)).filter(Number.isFinite).sort((a, b) => a - b);
      const steps = [...new Set(all)];
      if (!steps.length) return;
      if (mode === "all" || mode === "data") {
        document.getElementById("step-min").value = steps[0];
        document.getElementById("step-max").value = steps[steps.length - 1];
      } else {
        const n = mode === "last5" ? 5 : 10;
        const slice = steps.slice(-n);
        document.getElementById("step-min").value = slice[0];
        document.getElementById("step-max").value = slice[slice.length - 1];
      }
      render();
    }
    function exportVisible() {
      const rows = visibleRows({ taskOnly: false });
      const headers = ["model","arch","task","step","checkpoint","status","micro","macro","acc_norm","f1","duration_sec"];
      const lines = [headers.join(",")];
      for (const row of rows) {
        const m = row.metrics || {};
        const vals = [seriesKey(row), row.model_arch || "", row.task || "", row.step || "", row.checkpoint_name || "", row.status || "", m.overall_micro ?? "", m.overall_macro ?? "", m.overall_acc_norm ?? m.acc_norm ?? "", m.overall_f1 ?? m.f1 ?? "", row.duration_sec ?? ""];
        lines.push(vals.map(v => `"${String(v).replaceAll('"','""')}"`).join(","));
      }
      const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "dashboard_visible_rows.csv";
      a.click();
      URL.revokeObjectURL(a.href);
    }
    function setupEvents() {
      ["task","metric","search","arch","status","baseline","compare","step-min","step-max"].forEach(id => {
        document.getElementById(id).addEventListener("input", render);
        document.getElementById(id).addEventListener("change", render);
      });
      document.getElementById("model-all").addEventListener("click", () => { selectedModels = new Set(allSeries()); render(); });
      document.getElementById("model-none").addEventListener("click", () => { selectedModels = new Set(); render(); });
      document.getElementById("range-all").addEventListener("click", () => setRange("all"));
      document.getElementById("range-data").addEventListener("click", () => setRange("data"));
      document.getElementById("range-last5").addEventListener("click", () => setRange("last5"));
      document.getElementById("range-last10").addEventListener("click", () => setRange("last10"));
      document.getElementById("export-visible").addEventListener("click", exportVisible);
      document.querySelectorAll("#scale button").forEach(button => {
        button.addEventListener("click", () => {
          document.querySelectorAll("#scale button").forEach(b => b.classList.remove("active"));
          button.classList.add("active");
          yScale = button.dataset.scale;
          renderChart();
        });
      });
    }
    setupEvents();
    load();
    setInterval(load, refreshMs);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "KRongEvalDashboard/1.0"

    def _send(self, body: bytes, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def result_root(self) -> Path:
        return self.server.result_root  # type: ignore[attr-defined]

    @property
    def compare_mode(self) -> bool:
        return bool(getattr(self.server, "compare_mode", False))  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = DASHBOARD_HTML_PATH.read_text(encoding="utf-8") if DASHBOARD_HTML_PATH.exists() else HTML
            self._send(html.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/summary":
            try:
                data = load_compare_data(self.result_root) if self.compare_mode else load_dashboard_data(self.result_root)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._send(body, content_type="application/json; charset=utf-8")
            except Exception as exc:
                body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(body, content_type="application/json; charset=utf-8", status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/log":
            params = parse_qs(parsed.query)
            path_text = (params.get("path") or [""])[0]
            self._send_log(path_text)
            return
        self._send(b"not found", content_type="text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)

    def _send_log(self, path_text: str) -> None:
        root = self.result_root.resolve()
        if not path_text:
            self._send(b"No log path.", content_type="text/plain; charset=utf-8")
            return
        path = Path(path_text).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError:
            self._send(b"Log path is outside the result root.", content_type="text/plain; charset=utf-8", status=HTTPStatus.FORBIDDEN)
            return
        if not path.exists():
            self._send(f"Log not found: {path}".encode("utf-8"), content_type="text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-120:])
        self._send(tail.encode("utf-8"), content_type="text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a live dashboard for run_eval_checkpoint_sweep.py results.")
    parser.add_argument("--result-root", type=str, default="", help="Sweep result directory. Defaults to latest under sweep_results/.")
    parser.add_argument("--compare-root", type=str, default="", help="Parent directory containing multiple sweep result folders to compare.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-ms", type=int, default=3000, help="Browser polling interval hint. You can also use ?refresh=5000.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_root = (
        Path(args.compare_root).expanduser().resolve()
        if args.compare_root
        else Path(args.result_root).expanduser().resolve()
        if args.result_root
        else _latest_sweep_dir()
    )
    if not result_root.exists():
        raise FileNotFoundError(f"Result root not found: {result_root}")

    compare_mode = bool(args.compare_root)
    if not compare_mode and args.result_root:
        is_single_sweep = (
            (result_root / "sweep_summary.csv").exists()
            or (result_root / "sweep_manifest.json").exists()
            or (result_root / "json").exists()
        )
        if not is_single_sweep and _discover_sweep_dirs(result_root):
            compare_mode = True

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.result_root = result_root  # type: ignore[attr-defined]
    server.compare_mode = compare_mode  # type: ignore[attr-defined]
    print(f"[dashboard] {'compare_root' if compare_mode else 'result_root'}={result_root}")
    print(f"[dashboard] http://{args.host}:{args.port}/?refresh={args.refresh_ms}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
