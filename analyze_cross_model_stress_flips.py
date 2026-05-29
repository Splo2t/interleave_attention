#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_predictions(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "item_predictions" not in payload:
        raise ValueError(
            f"{path} does not contain item_predictions. "
            "Re-run evaluation with --save_item_predictions."
        )
    return {str(row["item_id"]): row for row in payload["item_predictions"]}


def _group_key(record: dict[str, Any]) -> str:
    if record.get("task"):
        return str(record["task"])
    if record.get("subject"):
        return str(record["subject"])
    return "overall"


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def _summarize(
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
    *,
    correct_field: str,
) -> dict[str, Any]:
    n = len(rows)
    base_cw = compare_cw = 0
    base_cc = compare_cc = 0
    base_pred_change = compare_pred_change = 0
    pred_key = "pred_norm_idx" if correct_field == "correct_norm" else "pred_idx"

    for base_clean, base_stress, compare_clean, compare_stress in rows:
        base_clean_correct = bool(base_clean[correct_field])
        base_stress_correct = bool(base_stress[correct_field])
        compare_clean_correct = bool(compare_clean[correct_field])
        compare_stress_correct = bool(compare_stress[correct_field])

        if base_clean_correct and not base_stress_correct:
            base_cw += 1
        if compare_clean_correct and not compare_stress_correct:
            compare_cw += 1
        if base_clean_correct and base_stress_correct:
            base_cc += 1
        if compare_clean_correct and compare_stress_correct:
            compare_cc += 1
        if base_clean.get(pred_key) != base_stress.get(pred_key):
            base_pred_change += 1
        if compare_clean.get(pred_key) != compare_stress.get(pred_key):
            compare_pred_change += 1

    base_cw_rate = base_cw / n if n else 0.0
    compare_cw_rate = compare_cw / n if n else 0.0
    base_cra = base_cc / n if n else 0.0
    compare_cra = compare_cc / n if n else 0.0
    base_pred_change_rate = base_pred_change / n if n else 0.0
    compare_pred_change_rate = compare_pred_change / n if n else 0.0

    return {
        "n_both_clean_correct": n,
        "base_c_to_w": base_cw_rate,
        "compare_c_to_w": compare_cw_rate,
        "c_to_w_delta_compare_minus_base": compare_cw_rate - base_cw_rate,
        "base_cra": base_cra,
        "compare_cra": compare_cra,
        "cra_delta_compare_minus_base": compare_cra - base_cra,
        "base_prediction_change": base_pred_change_rate,
        "compare_prediction_change": compare_pred_change_rate,
        "prediction_change_delta_compare_minus_base": compare_pred_change_rate - base_pred_change_rate,
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows: list[dict[str, Any]]) -> None:
    headers = [
        "group",
        "n",
        "base_C->W",
        "compare_C->W",
        "delta",
        "base_CRA",
        "compare_CRA",
        "CRA_delta",
        "base_pred_change",
        "compare_pred_change",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["group"]),
                    str(row["n_both_clean_correct"]),
                    _fmt_pct(row["base_c_to_w"]),
                    _fmt_pct(row["compare_c_to_w"]),
                    _fmt_pct(row["c_to_w_delta_compare_minus_base"]),
                    _fmt_pct(row["base_cra"]),
                    _fmt_pct(row["compare_cra"]),
                    _fmt_pct(row["cra_delta_compare_minus_base"]),
                    _fmt_pct(row["base_prediction_change"]),
                    _fmt_pct(row["compare_prediction_change"]),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stress robustness on the subset that both models answer correctly "
            "in the clean setting."
        )
    )
    parser.add_argument("--base-clean-json", required=True)
    parser.add_argument("--base-stress-json", required=True)
    parser.add_argument("--compare-clean-json", required=True)
    parser.add_argument("--compare-stress-json", required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--compare-label", default="compare")
    parser.add_argument("--correct-field", default="correct", choices=["correct", "correct_norm"])
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    base_clean = _load_predictions(args.base_clean_json)
    base_stress = _load_predictions(args.base_stress_json)
    compare_clean = _load_predictions(args.compare_clean_json)
    compare_stress = _load_predictions(args.compare_stress_json)

    shared_ids = sorted(set(base_clean) & set(base_stress) & set(compare_clean) & set(compare_stress))
    rows = []
    paired = []
    for item_id in shared_ids:
        if args.correct_field not in base_clean[item_id] or args.correct_field not in compare_clean[item_id]:
            continue
        if bool(base_clean[item_id][args.correct_field]) and bool(compare_clean[item_id][args.correct_field]):
            row = (base_clean[item_id], base_stress[item_id], compare_clean[item_id], compare_stress[item_id])
            paired.append(row)

    if not paired:
        raise ValueError("No items were correct for both models in the clean setting.")

    rows.append(
        {
            "group": "overall",
            "base_label": args.base_label,
            "compare_label": args.compare_label,
            "correct_field": args.correct_field,
            **_summarize(paired, correct_field=args.correct_field),
        }
    )

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in paired:
        grouped[_group_key(row[0])].append(row)
    for group in sorted(grouped):
        rows.append(
            {
                "group": group,
                "base_label": args.base_label,
                "compare_label": args.compare_label,
                "correct_field": args.correct_field,
                **_summarize(grouped[group], correct_field=args.correct_field),
            }
        )

    _print_rows(rows)
    output = {
        "base_label": args.base_label,
        "compare_label": args.compare_label,
        "correct_field": args.correct_field,
        "num_shared_items": len(shared_ids),
        "num_both_clean_correct": rows[0]["n_both_clean_correct"],
        "groups": rows,
        "inputs": {
            "base_clean_json": args.base_clean_json,
            "base_stress_json": args.base_stress_json,
            "compare_clean_json": args.compare_clean_json,
            "compare_stress_json": args.compare_stress_json,
        },
    }
    if args.out_json:
        Path(args.out_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.out_json}")
    if args.out_csv:
        _write_csv(args.out_csv, rows)
        print(f"[csv] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
