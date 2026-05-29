#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_eval_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "item_predictions" not in payload:
        raise ValueError(
            f"{path} does not contain item_predictions. "
            "Re-run evaluation with --save_item_predictions."
        )
    return payload


def _group_key(record: dict[str, Any]) -> str:
    if record.get("task"):
        return str(record["task"])
    if record.get("subject"):
        return str(record["subject"])
    return "overall"


def _summarize_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    correct_field: str,
) -> dict[str, Any]:
    cc = cw = wc = ww = 0
    pred_changed = 0
    usable = 0
    for orig, stress in pairs:
        if correct_field not in orig or correct_field not in stress:
            continue
        orig_correct = bool(orig[correct_field])
        stress_correct = bool(stress[correct_field])
        usable += 1
        cc += int(orig_correct and stress_correct)
        cw += int(orig_correct and not stress_correct)
        wc += int((not orig_correct) and stress_correct)
        ww += int((not orig_correct) and (not stress_correct))

        pred_key = "pred_norm_idx" if correct_field == "correct_norm" else "pred_idx"
        if orig.get(pred_key) != stress.get(pred_key):
            pred_changed += 1

    orig_correct_total = cc + cw
    orig_wrong_total = wc + ww
    stress_correct_total = cc + wc
    n = usable
    original_acc = orig_correct_total / n if n else 0.0
    stress_acc = stress_correct_total / n if n else 0.0
    drop = original_acc - stress_acc
    retention = stress_acc / original_acc if original_acc else 0.0
    relative_drop = drop / original_acc if original_acc else 0.0
    error_increase = ((1.0 - stress_acc) / (1.0 - original_acc) - 1.0) if original_acc < 1.0 else 0.0
    conditional_robust_acc = cc / orig_correct_total if orig_correct_total else 0.0
    correct_to_wrong_rate = cw / orig_correct_total if orig_correct_total else 0.0
    wrong_to_correct_rate = wc / orig_wrong_total if orig_wrong_total else 0.0
    net_flip_rate = (cw - wc) / n if n else 0.0
    pred_change_rate = pred_changed / n if n else 0.0

    return {
        "n": n,
        "original_acc": original_acc,
        "stress_acc": stress_acc,
        "drop": drop,
        "retention": retention,
        "relative_drop": relative_drop,
        "error_increase": error_increase,
        "correct_correct": cc,
        "correct_wrong": cw,
        "wrong_correct": wc,
        "wrong_wrong": ww,
        "conditional_robust_acc": conditional_robust_acc,
        "correct_to_wrong_rate": correct_to_wrong_rate,
        "wrong_to_correct_rate": wrong_to_correct_rate,
        "net_flip_rate": net_flip_rate,
        "prediction_change_rate": pred_change_rate,
    }


def _format_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{100.0 * value:6.2f}%"


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "group",
        "n",
        "orig",
        "stress",
        "drop",
        "rel_drop",
        "err_inc",
        "retention",
        "C->W",
        "W->C",
        "CRA",
        "net_flip",
        "pred_change",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["group"]),
                    str(row["n"]),
                    _format_pct(row["original_acc"]),
                    _format_pct(row["stress_acc"]),
                    _format_pct(row["drop"]),
                    _format_pct(row["relative_drop"]),
                    _format_pct(row["error_increase"]),
                    _format_pct(row["retention"]),
                    _format_pct(row["correct_to_wrong_rate"]),
                    _format_pct(row["wrong_to_correct_rate"]),
                    _format_pct(row["conditional_robust_acc"]),
                    _format_pct(row["net_flip_rate"]),
                    _format_pct(row["prediction_change_rate"]),
                ]
            )
        )


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute paired flip analysis from item-level eval JSONs.")
    parser.add_argument("--original-json", required=True, help="Original benchmark JSON saved with --save_item_predictions")
    parser.add_argument("--stress-json", required=True, help="Stress benchmark JSON saved with --save_item_predictions")
    parser.add_argument(
        "--correct-field",
        default="correct",
        choices=["correct", "correct_norm"],
        help="Which correctness field to compare. Use correct_norm for normalized-choice analyses.",
    )
    parser.add_argument("--out-json", default="", help="Optional path to save full summary JSON")
    parser.add_argument("--out-csv", default="", help="Optional path to save group summary CSV")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    original = _load_eval_json(args.original_json)
    stress = _load_eval_json(args.stress_json)

    original_predictions = {str(row["item_id"]): row for row in original["item_predictions"]}
    stress_predictions = {str(row["item_id"]): row for row in stress["item_predictions"]}
    shared_ids = sorted(set(original_predictions) & set(stress_predictions))
    if not shared_ids:
        raise ValueError("No paired item_id overlap between original and stress predictions.")

    all_pairs = [(original_predictions[item_id], stress_predictions[item_id]) for item_id in shared_ids]
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in all_pairs:
        grouped[_group_key(pair[0])].append(pair)

    rows: list[dict[str, Any]] = []
    overall = _summarize_pairs(all_pairs, correct_field=args.correct_field)
    rows.append({"group": "overall", "correct_field": args.correct_field, **overall})
    for group in sorted(grouped):
        summary = _summarize_pairs(grouped[group], correct_field=args.correct_field)
        rows.append({"group": group, "correct_field": args.correct_field, **summary})

    _print_table(rows)

    output = {
        "original_json": args.original_json,
        "stress_json": args.stress_json,
        "correct_field": args.correct_field,
        "num_original_predictions": len(original_predictions),
        "num_stress_predictions": len(stress_predictions),
        "num_paired": len(shared_ids),
        "groups": rows,
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
