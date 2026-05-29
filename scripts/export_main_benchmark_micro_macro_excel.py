#!/usr/bin/env python3
"""Export main benchmark micro/macro tables in paper order."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/mnt/nas_server_yhw/eval_krong")
OUT_XLSX = ROOT / "paper_drafts" / "main_benchmark_micro_macro_ordered_20260523.xlsx"
OUT_CSV = ROOT / "paper_drafts" / "main_benchmark_micro_macro_ordered_20260523.csv"

sys.path.insert(0, str(ROOT / "scripts"))
from export_main_benchmark_excel import (  # noqa: E402
    EN_TASKS,
    KO_TASKS,
    MODEL_ALIASES,
    SCAN_ROOTS,
    TASK_LABELS,
    TASKS,
    as_float,
    infer_model,
    infer_task,
    parse_timestamp,
    pct,
    write_xlsx,
)


ORDERED_MODELS: list[dict[str, Any] | None] = [
    {"Model_Group": "Ours 1B", "Model": "Stage2 Interleave @18k", "Step": 18000, "Raw_Model_ID": "stage2_mlm00_interleave_ckpt18000"},
    {"Model_Group": "Ours 1B", "Model": "Matched Decoder CPT @18k", "Step": 18000, "Raw_Model_ID": "normal_random_new_ckpt18000"},
    {"Model_Group": "Ours 1B", "Model": "Token-only CPT @19k", "Step": 19000, "Raw_Model_ID": "token_only_1b_cpt_ckpt19000"},
    {"Model_Group": "Ours 1B", "Model": "mBERT Encoder Interleave @18k", "Step": 18000, "Raw_Model_ID": "mbert5_encoder_interleave_ckpt18000"},
    {"Model_Group": "Public Llama3.2-1B", "Model": "Llama3.2-1B Base", "Step": 0, "Raw_Model_ID": "llama32_1b_base"},
    {"Model_Group": "Public Llama3.2-1B", "Model": "Llama3.2-1B Vanilla CPT @18k", "Step": 18000, "Raw_Model_ID": "llama32_1b_cpt_vanilla_ckpt18000"},
    {"Model_Group": "Public Llama3.2-1B", "Model": "Llama3.2-1B Interleave CPT @18k", "Step": 18000, "Raw_Model_ID": "llama32_1b_interleave_cpt_ckpt18000"},
    None,
    {"Model_Group": "External public model", "Model": "Gemma3-1B-PT", "Step": 0, "Raw_Model_ID": "gemma3_1b_pt"},
    {"Model_Group": "External public model", "Model": "OLMo-2-0425-1B", "Step": 0, "Raw_Model_ID": "olmo2_0425_1b"},
    {"Model_Group": "External public model", "Model": "SmolLM2-1.7B", "Step": 0, "Raw_Model_ID": "smollm2_1p7b"},
    {"Model_Group": "External public model", "Model": "Polyglot-Ko-1.3B", "Step": 0, "Raw_Model_ID": "polyglot_ko_1p3b"},
    None,
    {"Model_Group": "Public Llama3.1-8B", "Model": "Llama3.1-8B Base", "Step": 0, "Raw_Model_ID": "llama31_8b_base"},
    {"Model_Group": "Public Llama3.1-8B", "Model": "Llama3.1-8B Interleave CPT @19k", "Step": 19000, "Raw_Model_ID": "llama31_8b_interleave_mlm00_copylow_ckpt19000"},
    {"Model_Group": "External public model", "Model": "KORMo-10B-base", "Step": 0, "Raw_Model_ID": "kormo10b_base"},
    {"Model_Group": "External public model", "Model": "OLMo2-13B", "Step": 0, "Raw_Model_ID": "olmo2_13b"},
    {"Model_Group": "External public model", "Model": "beomi Llama-3-KoEn-8B", "Step": 0, "Raw_Model_ID": "beomi_llama3_koen_8b"},
    {"Model_Group": "External public model", "Model": "beomi Llama-3-Open-Ko-8B", "Step": 0, "Raw_Model_ID": "beomi_llama3_open_ko_8b"},
    {"Model_Group": "External public model", "Model": "Gemma3-12B-PT addBOS", "Step": 0, "Raw_Model_ID": "gemma3_12b_pt_addbos"},
    {"Model_Group": "External public model", "Model": "Kanana-1.5-8B-Base", "Step": 0, "Raw_Model_ID": "kanana15_8b_base"},
]


def normalize_model_id(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def score_from_json(payload: dict[str, Any], task: str, metric: str) -> float | None:
    if metric == "micro":
        if task == "kobest":
            return pct(payload.get("overall_micro"))
        if task in {"click", "csatqa"}:
            return pct(payload.get("overall_micro") or payload.get("acc_norm") or payload.get("overall_acc_norm"))
        return pct(payload.get("overall_micro") or payload.get("acc"))

    if metric == "macro":
        if task in {"click", "csatqa"}:
            return pct(payload.get("overall_macro") or payload.get("overall_macro_acc_norm") or payload.get("overall_micro") or payload.get("acc_norm"))
        return pct(payload.get("overall_macro") or payload.get("overall_micro") or payload.get("acc"))

    raise ValueError(metric)


def source_priority(path: Path) -> tuple[str, str]:
    return (parse_timestamp(path), str(path))


def scan_scores() -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    best: dict[tuple[str, str, str], tuple[tuple[str, str], float, Path]] = {}
    source_rows: list[dict[str, Any]] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name == "sweep_manifest.json":
                continue
            task = infer_task(path)
            if task is None:
                continue
            model = normalize_model_id(infer_model(path))
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            for metric in ("micro", "macro"):
                score = score_from_json(payload, task, metric)
                if score is None:
                    continue
                key = (model, task, metric)
                prio = source_priority(path)
                if key not in best or prio > best[key][0]:
                    best[key] = (prio, score, path)

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for (model, task, metric), (_, score, path) in sorted(best.items()):
        by_model.setdefault(model, {}).setdefault(metric, {})[task] = {
            "score": round(score, 4),
            "source": str(path.relative_to(ROOT)),
        }
        source_rows.append(
            {
                "Raw_Model_ID": model,
                "Task": task,
                "Metric": metric,
                "Score": round(score, 6),
                "Source_File": str(path.relative_to(ROOT)),
            }
        )
    return by_model, source_rows


def avg(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def add_avgs(row: dict[str, Any]) -> None:
    by_task = {task: as_float(row.get(TASK_LABELS[task])) for task in TASKS}
    ko_vals = [by_task[t] for t in KO_TASKS]
    en_vals = [by_task[t] for t in EN_TASKS]
    ko_avg = avg(ko_vals) if all(v is not None for v in ko_vals) else None
    en_avg = avg(en_vals) if all(v is not None for v in en_vals) else None
    total_avg = (ko_avg + en_avg) / 2.0 if ko_avg is not None and en_avg is not None else None
    row["Ko_avg"] = round(ko_avg, 4) if ko_avg is not None else ""
    row["En_avg"] = round(en_avg, 4) if en_avg is not None else ""
    row["AVG"] = round(total_avg, 4) if total_avg is not None else ""
    row["Missing"] = ", ".join(TASK_LABELS[t] for t, v in by_task.items() if v is None)


def metric_row(model_info: dict[str, Any], scores: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    row = {
        "Model_Group": model_info["Model_Group"],
        "Model": model_info["Model"],
        "Step": model_info["Step"],
        "Metric": metric,
        "Raw_Model_ID": model_info["Raw_Model_ID"],
    }
    for task in TASKS:
        if task in scores:
            row[TASK_LABELS[task]] = scores[task]["score"]
            row[f"{TASK_LABELS[task]}_source"] = scores[task]["source"]
        else:
            row[TASK_LABELS[task]] = ""
            row[f"{TASK_LABELS[task]}_source"] = ""
    add_avgs(row)
    return row


def delta_row(model_info: dict[str, Any], micro: dict[str, dict[str, Any]], macro: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = {
        "Model_Group": model_info["Model_Group"],
        "Model": model_info["Model"],
        "Step": model_info["Step"],
        "Metric": "macro_minus_micro_pp",
        "Raw_Model_ID": model_info["Raw_Model_ID"],
    }
    for task in TASKS:
        if task in micro and task in macro:
            row[TASK_LABELS[task]] = round(macro[task]["score"] - micro[task]["score"], 4)
            row[f"{TASK_LABELS[task]}_source"] = macro[task]["source"]
        else:
            row[TASK_LABELS[task]] = ""
            row[f"{TASK_LABELS[task]}_source"] = ""
    add_avgs(row)
    return row


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scores, source_rows = scan_scores()
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    for model_info in ORDERED_MODELS:
        if model_info is None:
            sep = {"Model_Group": "", "Model": "", "Step": "", "Metric": "", "Raw_Model_ID": ""}
            long_rows.append(dict(sep))
            wide_rows.append(dict(sep))
            continue
        raw_id = str(model_info["Raw_Model_ID"])
        micro = scores.get(raw_id, {}).get("micro", {})
        macro = scores.get(raw_id, {}).get("macro", {})
        micro_row = metric_row(model_info, micro, "micro")
        macro_row = metric_row(model_info, macro, "macro")
        long_rows.extend([micro_row, macro_row, delta_row(model_info, micro, macro)])

        wide = {
            "Model_Group": model_info["Model_Group"],
            "Model": model_info["Model"],
            "Step": model_info["Step"],
            "Raw_Model_ID": raw_id,
        }
        for task in TASKS:
            label = TASK_LABELS[task]
            wide[f"{label}_Micro"] = micro.get(task, {}).get("score", "")
            wide[f"{label}_Macro"] = macro.get(task, {}).get("score", "")
        for metric_name, row in (("Micro", micro_row), ("Macro", macro_row)):
            wide[f"Ko_avg_{metric_name}"] = row.get("Ko_avg", "")
            wide[f"En_avg_{metric_name}"] = row.get("En_avg", "")
            wide[f"AVG_{metric_name}"] = row.get("AVG", "")
        wide["Missing_Micro"] = micro_row.get("Missing", "")
        wide["Missing_Macro"] = macro_row.get("Missing", "")
        wide_rows.append(wide)

        missing_micro = micro_row.get("Missing", "")
        missing_macro = macro_row.get("Missing", "")
        coverage.append(
            {
                "Model_Group": model_info["Model_Group"],
                "Model": model_info["Model"],
                "Step": model_info["Step"],
                "Raw_Model_ID": raw_id,
                "Micro_Status": "OK" if not missing_micro else "MISSING",
                "Missing_Micro": missing_micro,
                "Macro_Status": "OK" if not missing_macro else "MISSING",
                "Missing_Macro": missing_macro,
            }
        )

    return long_rows, wide_rows, coverage, source_rows


def main() -> int:
    long_rows, wide_rows, coverage, source_rows = collect()
    long_headers = [
        "Model_Group",
        "Model",
        "Step",
        "Metric",
        "Raw_Model_ID",
        *[TASK_LABELS[t] for t in TASKS],
        "Ko_avg",
        "En_avg",
        "AVG",
        "Missing",
    ]
    wide_headers = [
        "Model_Group",
        "Model",
        "Step",
        "Raw_Model_ID",
        *[f"{TASK_LABELS[t]}_{metric}" for t in TASKS for metric in ("Micro", "Macro")],
        "Ko_avg_Micro",
        "En_avg_Micro",
        "AVG_Micro",
        "Ko_avg_Macro",
        "En_avg_Macro",
        "AVG_Macro",
        "Missing_Micro",
        "Missing_Macro",
    ]
    readme = [
        {"Item": "Metric=micro", "Description": "Dashboard Micro/Acc convention. Uses overall_micro where available."},
        {"Item": "Metric=macro", "Description": "Unweighted subject/task macro. Uses overall_macro where available; single-task benchmarks fall back to micro/acc."},
        {"Item": "Metric=macro_minus_micro_pp", "Description": "Macro score minus micro score in percentage points."},
        {"Item": "Ko_avg", "Description": "mean(KMMLU, KoBEST, CLICk, CSATQA), computed only when all are present."},
        {"Item": "En_avg", "Description": "mean(MMLU, ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA), computed only when all are present."},
    ]
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(
        OUT_XLSX,
        [
            ("README", readme, ["Item", "Description"]),
            ("paper_ordered_long", long_rows, long_headers),
            ("paper_ordered_wide", wide_rows, wide_headers),
            ("coverage_check", coverage, None),
            ("source_json_latest", source_rows, None),
        ],
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(long_rows)
    missing = [r for r in coverage if r["Micro_Status"] != "OK" or r["Macro_Status"] != "OK"]
    print(f"[saved] {OUT_XLSX}")
    print(f"[csv] {OUT_CSV}")
    print(f"long_rows={len(long_rows)} wide_rows={len(wide_rows)} missing_models={len(missing)} sources={len(source_rows)}")
    if missing:
        for row in missing:
            print("[missing]", row["Model"], row["Missing_Micro"], row["Missing_Macro"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
