#!/usr/bin/env python3
"""Export shared clean-correct stress robustness comparisons to XLSX."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "variant_eval_outputs_paper" / "shared_clean_correct"
OUT_XLSX = ROOT / "paper_drafts" / "shared_clean_correct_kobest_20260523.xlsx"
OUT_CSV = ROOT / "paper_drafts" / "shared_clean_correct_kobest_20260523_overall.csv"

sys.path.insert(0, str(ROOT / "scripts"))
from export_main_benchmark_excel import write_xlsx  # noqa: E402


VARIANT_ORDER = {
    "ko_random_p25": 0,
    "ko_random_p50": 1,
    "ko_josa_preserve_compaction_hard": 2,
}

PAIR_ORDER = {
    "stage2_vs_normal": 0,
    "stage2_vs_tokenonly": 1,
    "stage2_vs_mbert": 2,
    "mbert_vs_normal": 3,
}

VARIANT_LABEL = {
    "ko_random_p25": "Random-P25",
    "ko_random_p50": "Random-P50",
    "ko_josa_preserve_compaction_hard": "JosaHard",
}

PAIR_LABEL = {
    "stage2_vs_normal": "Stage2 Interleave vs Matched Decoder CPT",
    "stage2_vs_tokenonly": "Stage2 Interleave vs Token-only CPT",
    "stage2_vs_mbert": "Stage2 Interleave vs mBERT Encoder Interleave",
    "mbert_vs_normal": "mBERT Encoder Interleave vs Matched Decoder CPT",
}


def pct(value: Any) -> float | None:
    try:
        return round(float(value) * 100.0, 4)
    except (TypeError, ValueError):
        return None


def parse_name(path: Path) -> tuple[str, str]:
    stem = path.name.removesuffix("_shared_clean_correct.csv")
    match = re.search(r"(ko_random_p25|ko_random_p50|ko_josa_preserve_compaction_hard)$", stem)
    if not match:
        return stem, "unknown"
    variant = match.group(1)
    pair = stem[: -(len(variant) + 1)]
    return pair, variant


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def convert_row(path: Path, row: dict[str, str], *, aggregate: str) -> dict[str, Any]:
    pair, variant = parse_name(path)
    base = row.get("base_label", "")
    compare = row.get("compare_label", "")
    return {
        "Pair": PAIR_LABEL.get(pair, pair),
        "Pair_Key": pair,
        "Variant": VARIANT_LABEL.get(variant, variant),
        "Variant_Key": variant,
        "Group": row.get("group", ""),
        "Aggregation": aggregate,
        "Base_Model": base.replace("_", " "),
        "Compare_Model": compare.replace("_", " "),
        "N_Shared_Clean_Correct": int(float(row.get("n_both_clean_correct") or 0)),
        "Base_C_to_W_%": pct(row.get("base_c_to_w")),
        "Compare_C_to_W_%": pct(row.get("compare_c_to_w")),
        "Delta_C_to_W_pp": pct(row.get("c_to_w_delta_compare_minus_base")),
        "Base_CRA_%": pct(row.get("base_cra")),
        "Compare_CRA_%": pct(row.get("compare_cra")),
        "Delta_CRA_pp": pct(row.get("cra_delta_compare_minus_base")),
        "Base_Pred_Change_%": pct(row.get("base_prediction_change")),
        "Compare_Pred_Change_%": pct(row.get("compare_prediction_change")),
        "Delta_Pred_Change_pp": pct(row.get("prediction_change_delta_compare_minus_base")),
        "Source_File": str(path.relative_to(ROOT)),
    }


def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        PAIR_ORDER.get(str(row.get("Pair_Key")), 99),
        VARIANT_ORDER.get(str(row.get("Variant_Key")), 99),
        str(row.get("Group", "")),
    )


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []
    task_detail: list[dict[str, Any]] = []
    for path in sorted(INPUT_DIR.glob("*_shared_clean_correct.csv")):
        for row in read_rows(path):
            converted = convert_row(path, row, aggregate="shared_clean_correct")
            if row.get("group") == "overall":
                overall.append(converted)
            else:
                task_detail.append(converted)
    overall.sort(key=sort_key)
    task_detail.sort(key=sort_key)
    return overall, task_detail


def main() -> int:
    overall, task_detail = collect()
    if not overall:
        raise SystemExit(f"No shared clean-correct CSVs found under {INPUT_DIR}")

    headers = [
        "Pair",
        "Variant",
        "Group",
        "Aggregation",
        "Base_Model",
        "Compare_Model",
        "N_Shared_Clean_Correct",
        "Base_C_to_W_%",
        "Compare_C_to_W_%",
        "Delta_C_to_W_pp",
        "Base_CRA_%",
        "Compare_CRA_%",
        "Delta_CRA_pp",
        "Base_Pred_Change_%",
        "Compare_Pred_Change_%",
        "Delta_Pred_Change_pp",
        "Source_File",
    ]
    readme = [
        {
            "Item": "Scope",
            "Description": (
                "KoBEST stress robustness on the subset that both compared models "
                "answered correctly in the clean setting."
            ),
        },
        {
            "Item": "C_to_W",
            "Description": "Within the shared clean-correct subset, percentage of items flipped to wrong under stress.",
        },
        {
            "Item": "CRA",
            "Description": "Within the shared clean-correct subset, percentage of items remaining correct under stress.",
        },
        {
            "Item": "Delta columns",
            "Description": "Compare minus Base. Negative Delta_C_to_W is better for Compare; positive Delta_CRA is better.",
        },
    ]

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(
        OUT_XLSX,
        [
            ("README", readme, ["Item", "Description"]),
            ("overall", overall, headers),
            ("task_detail", task_detail, headers),
        ],
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{h: row.get(h, "") for h in headers} for row in overall])

    print(f"[saved] {OUT_XLSX}")
    print(f"[csv] {OUT_CSV}")
    print(f"overall_rows={len(overall)} task_detail_rows={len(task_detail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
