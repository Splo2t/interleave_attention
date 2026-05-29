#!/usr/bin/env python3
"""Export KoBEST stress paired-flip metrics as micro and macro XLSX sheets.

The paired-flip CSVs already contain item-level micro aggregation in the
``overall`` row.  This exporter also computes task-macro values by taking an
unweighted average over the five KoBEST subtasks:
BoolQ, COPA, HellaSwag, SentiNeg, and WiC.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_XLSX = ROOT / "paper_drafts" / "kobest_stress_micro_macro_20260522.xlsx"

sys.path.insert(0, str(ROOT / "scripts"))
from export_main_benchmark_excel import write_xlsx  # noqa: E402


PERCENT_FIELDS = {
    "original_acc": "Clean_Acc",
    "stress_acc": "Stress_Acc",
    "drop": "Drop_pp",
    "retention": "Retention_%",
    "relative_drop": "Relative_Drop_%",
    "error_increase": "Error_Increase_%",
    "conditional_robust_acc": "CRA_%",
    "correct_to_wrong_rate": "C_to_W_%",
    "wrong_to_correct_rate": "W_to_C_%",
    "net_flip_rate": "Net_Flip_%",
    "prediction_change_rate": "Pred_Change_%",
}

COUNT_FIELDS = [
    "correct_correct",
    "correct_wrong",
    "wrong_correct",
    "wrong_wrong",
]

TASK_ORDER = ["boolq", "copa", "hellaswag", "sentineg", "wic"]
VARIANT_ORDER = {
    "ko_random_p25": 0,
    "ko_random_p50": 1,
    "ko_josa_preserve_compaction_hard": 2,
}
MODEL_ORDER = {
    "Stage2 Interleave @18k": 0,
    "Matched Decoder CPT @18k": 1,
    "Token-only CPT @19k": 2,
    "mBERT Encoder Interleave @18k": 3,
    "Llama3.2-1B Base": 10,
    "Llama3.2-1B Vanilla CPT @18k": 11,
    "Llama3.2-1B Interleave CPT @18k": 12,
    "Llama3.2-1B Vanilla CPT @19k": 13,
    "Llama3.2-1B Interleave CPT @19k": 12,
    "Llama3.1-8B Base": 20,
    "Llama3.1-8B Interleave CPT @19k": 21,
    "Gemma3-1B-PT": 30,
    "OLMo-2-0425-1B": 31,
    "SmolLM2-1.7B": 32,
    "Polyglot-Ko-1.3B": 33,
    "KORMo-10B-base": 40,
    "Gemma3-12B-PT addBOS": 41,
    "Gemma3-12B-PT": 41,
    "Kanana-1.5-8B-Base": 42,
    "OLMo2-13B": 43,
    "beomi Llama-3-KoEn-8B": 44,
    "beomi Llama-3-Open-Ko-8B": 45,
    "Llama3.1-8B Interleave CPT @7k": 99,
}


PAPER_REQUIRED_MODELS = [
    "Stage2 Interleave @18k",
    "Matched Decoder CPT @18k",
    "Token-only CPT @19k",
    "mBERT Encoder Interleave @18k",
    "Llama3.2-1B Base",
    "Llama3.2-1B Vanilla CPT @18k",
    "Llama3.2-1B Interleave CPT @18k",
    "Polyglot-Ko-1.3B",
    "Gemma3-1B-PT",
    "OLMo-2-0425-1B",
    "SmolLM2-1.7B",
    "Llama3.1-8B Base",
    "Llama3.1-8B Interleave CPT @19k",
    "Gemma3-12B-PT addBOS",
    "Kanana-1.5-8B-Base",
    "OLMo2-13B",
    "KORMo-10B-base",
    "beomi Llama-3-KoEn-8B",
    "beomi Llama-3-Open-Ko-8B",
]

PAPER_REQUIRED_VARIANTS = [
    "ko_random_p25",
    "ko_random_p50",
    "ko_josa_preserve_compaction_hard",
]


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(value: Any) -> float | None:
    v = as_float(value)
    if v is None:
        return None
    return round(v * 100.0, 4)


def parse_source(path: Path) -> dict[str, Any]:
    name = path.name
    stem = name.removesuffix("_paired_flips.csv")
    variant_match = re.search(r"(ko_random_p25|ko_random_p50|ko_josa_preserve_compaction_hard)", stem)
    variant = variant_match.group(1) if variant_match else "unknown"

    model_key = stem
    if variant != "unknown":
        model_key = stem[: stem.index("_" + variant)]
    model_key = model_key.removesuffix("_kobest")

    model_map = {
        "stage2_mlm00_interleave_18000": ("Ours 1B", "Stage2 Interleave @18k", 18000),
        "normal_random_new_18000": ("Ours 1B", "Matched Decoder CPT @18k", 18000),
        "token_only_1b_cpt_19000": ("Ours 1B", "Token-only CPT @19k", 19000),
        "mbert5_encoder_interleave_18000": ("Ours 1B", "mBERT Encoder Interleave @18k", 18000),
        "llama32_1b_base": ("Public Llama3.2-1B", "Llama3.2-1B Base", 0),
        "llama32_1b_cpt_vanilla_ckpt18000": (
            "Public Llama3.2-1B",
            "Llama3.2-1B Vanilla CPT @18k",
            18000,
        ),
        "llama32_1b_interleave_cpt_ckpt18000": (
            "Public Llama3.2-1B",
            "Llama3.2-1B Interleave CPT @18k",
            18000,
        ),
        "llama32_1b_cpt_vanilla_ckpt19000": (
            "Public Llama3.2-1B",
            "Llama3.2-1B Vanilla CPT @19k",
            19000,
        ),
        "llama32_1b_interleave_cpt_ckpt19000": (
            "Public Llama3.2-1B",
            "Llama3.2-1B Interleave CPT @19k",
            19000,
        ),
        "llama31_8b_base": ("Public Llama3.1-8B", "Llama3.1-8B Base", 0),
        "llama31_8b_base_kobest": ("Public Llama3.1-8B", "Llama3.1-8B Base", 0),
        "llama31_8b_interleave_mlm00_copylow_ckpt19000": (
            "Public Llama3.1-8B",
            "Llama3.1-8B Interleave CPT @19k",
            19000,
        ),
        "interleave_ckpt7000": (
            "Legacy / not paper-ready",
            "Llama3.1-8B Interleave CPT @7k",
            7000,
        ),
        "gemma3_1b_pt": (
            "External public model",
            "Gemma3-1B-PT",
            0,
        ),
        "olmo2_0425_1b": (
            "External public model",
            "OLMo-2-0425-1B",
            0,
        ),
        "smollm2_1p7b": (
            "External public model",
            "SmolLM2-1.7B",
            0,
        ),
        "polyglot_ko_1p3b": (
            "External public model",
            "Polyglot-Ko-1.3B",
            0,
        ),
        "gemma3_12b_pt_addbos": (
            "External public model",
            "Gemma3-12B-PT addBOS",
            0,
        ),
        "kanana15_8b_base": (
            "External public model",
            "Kanana-1.5-8B-Base",
            0,
        ),
        "olmo2_13b": (
            "External public model",
            "OLMo2-13B",
            0,
        ),
        "beomi_llama3_koen_8b": (
            "External public model",
            "beomi Llama-3-KoEn-8B",
            0,
        ),
        "beomi_llama3_open_ko_8b": (
            "External public model",
            "beomi Llama-3-Open-Ko-8B",
            0,
        ),
        "kormo10b_base": (
            "External public model",
            "KORMo-10B-base",
            0,
        ),
    }

    group, model, step = model_map.get(model_key, ("unknown", model_key, ""))
    return {
        "Model_Group": group,
        "Model": model,
        "Step": step,
        "Variant": variant,
        "Source_File": str(path),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_row(meta: dict[str, Any], rows: list[dict[str, str]], mode: str) -> dict[str, Any] | None:
    if mode == "micro":
        src = next((r for r in rows if r.get("group") == "overall"), None)
        if src is None:
            return None
        out = {
            **meta,
            "Aggregation": "micro",
            "N": int(float(src.get("n") or 0)),
            "Task_Count": len(TASK_ORDER),
        }
        for raw, label in PERCENT_FIELDS.items():
            out[label] = pct(src.get(raw))
        for field in COUNT_FIELDS:
            out[field] = int(float(src[field])) if src.get(field) not in (None, "") else None
        return out

    task_rows = [r for r in rows if r.get("group") in TASK_ORDER]
    if not task_rows:
        return None
    out = {
        **meta,
        "Aggregation": "macro_task_avg",
        "N_Total": sum(int(float(r.get("n") or 0)) for r in task_rows),
        "Task_Count": len(task_rows),
        "Tasks": ",".join(r["group"] for r in task_rows),
    }
    for raw, label in PERCENT_FIELDS.items():
        values = [pct(r.get(raw)) for r in task_rows]
        values = [v for v in values if v is not None]
        out[label] = round(sum(values) / len(values), 4) if values else None
    return out


def task_detail_rows(meta: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for src in rows:
        if src.get("group") == "overall":
            continue
        row = {
            **meta,
            "Task": src.get("group"),
            "N": int(float(src.get("n") or 0)),
        }
        for raw, label in PERCENT_FIELDS.items():
            row[label] = pct(src.get(raw))
        for field in COUNT_FIELDS:
            row[field] = int(float(src[field])) if src.get(field) not in (None, "") else None
        out.append(row)
    return out


def is_paper_ready(row: dict[str, Any]) -> bool:
    return row.get("Model_Group") != "Legacy / not paper-ready"


def sort_key(row: dict[str, Any]) -> tuple:
    return (
        MODEL_ORDER.get(str(row.get("Model")), 1000),
        VARIANT_ORDER.get(str(row.get("Variant")), 99),
        str(row.get("Task", "")),
    )


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [
        *sorted((ROOT / "variant_eval_outputs_paper" / "paired_flip_core").glob("*paired_flips.csv")),
        *sorted((ROOT / "variant_eval_outputs_paper" / "kobest_stress_missing_20260523").glob("*paired_flips.csv")),
        *sorted((ROOT / "variant_eval_outputs_paired").rglob("*kobest*paired_flips.csv")),
        *sorted((ROOT / "variant_eval_outputs_8b" / "kobest_paired").glob("*paired_flips.csv")),
        *sorted((ROOT / "variant_eval_outputs_external" / "kobest_paired").glob("*paired_flips.csv")),
    ]
    seen = set()
    micro, macro, details = [], [], []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        meta = parse_source(path.relative_to(ROOT))
        rows = read_rows(path)
        micro_row = metric_row(meta, rows, "micro")
        macro_row = metric_row(meta, rows, "macro")
        if micro_row:
            micro.append(micro_row)
        if macro_row:
            macro.append(macro_row)
        details.extend(task_detail_rows(meta, rows))
    micro.sort(key=sort_key)
    macro.sort(key=sort_key)
    details.sort(key=sort_key)
    return micro, macro, details


def coverage_rows(micro: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = {(str(r.get("Model")), str(r.get("Variant"))) for r in micro if is_paper_ready(r)}
    rows: list[dict[str, Any]] = []
    for model in PAPER_REQUIRED_MODELS:
        missing = [v for v in PAPER_REQUIRED_VARIANTS if (model, v) not in present]
        rows.append(
            {
                "Model": model,
                "Status": "OK" if not missing else "MISSING",
                "Missing_Variants": ", ".join(missing),
            }
        )
    return rows


def main() -> int:
    micro, macro, details = collect()
    coverage = coverage_rows(micro)
    metric_headers = [
        "Model_Group",
        "Model",
        "Step",
        "Variant",
        "Aggregation",
        "N",
        "N_Total",
        "Task_Count",
        "Tasks",
        "Clean_Acc",
        "Stress_Acc",
        "Drop_pp",
        "Retention_%",
        "Relative_Drop_%",
        "Error_Increase_%",
        "C_to_W_%",
        "W_to_C_%",
        "CRA_%",
        "Net_Flip_%",
        "Pred_Change_%",
        "correct_correct",
        "correct_wrong",
        "wrong_correct",
        "wrong_wrong",
        "Source_File",
    ]
    detail_headers = [
        "Model_Group",
        "Model",
        "Step",
        "Variant",
        "Task",
        "N",
        "Clean_Acc",
        "Stress_Acc",
        "Drop_pp",
        "Retention_%",
        "Relative_Drop_%",
        "Error_Increase_%",
        "C_to_W_%",
        "W_to_C_%",
        "CRA_%",
        "Net_Flip_%",
        "Pred_Change_%",
        "Source_File",
    ]
    readme = [
        {
            "Item": "micro",
            "Description": "KoBEST 전체 item을 모두 합친 weighted overall row입니다. paired_flips.csv의 group=overall 값을 사용합니다.",
        },
        {
            "Item": "macro",
            "Description": "BoolQ, COPA, HellaSwag, SentiNeg, WiC 5개 task의 metric을 unweighted 평균낸 값입니다.",
        },
        {
            "Item": "Drop_pp",
            "Description": "Clean_Acc - Stress_Acc. percentage point 단위입니다.",
        },
        {
            "Item": "C_to_W_%",
            "Description": "Clean에서 맞춘 문항 중 stress에서 틀린 비율입니다. 낮을수록 stress에 덜 깨집니다.",
        },
        {
            "Item": "W_to_C_%",
            "Description": "Clean에서 틀린 문항 중 stress에서 맞은 비율입니다. drop을 상쇄하는 stress-induced correction입니다.",
        },
        {
            "Item": "CRA_%",
            "Description": "Conditional Robust Accuracy = Clean-correct 문항이 stress에서도 계속 맞는 비율입니다.",
        },
        {
            "Item": "paper_ready",
            "Description": "Legacy @7k 결과는 all_sources 시트에만 남기고 paper_ready 시트에서는 제외했습니다.",
        },
    ]
    write_xlsx(
        OUT_XLSX,
        [
            ("README", readme, None),
            ("micro_paper_ready", [r for r in micro if is_paper_ready(r)], metric_headers),
            ("macro_paper_ready", [r for r in macro if is_paper_ready(r)], metric_headers),
            ("coverage_check", coverage, None),
            ("micro_all_sources", micro, metric_headers),
            ("macro_all_sources", macro, metric_headers),
            ("task_detail_source", details, detail_headers),
        ],
    )
    print(OUT_XLSX)
    print(f"micro_rows={len(micro)} macro_rows={len(macro)} detail_rows={len(details)}")
    print(f"paper_ready_micro_rows={sum(1 for r in micro if is_paper_ready(r))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
