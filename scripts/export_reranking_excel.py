#!/usr/bin/env python3
"""Export paper-ready reranking tables.

The environment does not always have openpyxl/xlsxwriter, so this script reuses
the small stdlib XLSX writer from export_main_benchmark_excel.py.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/nas_server_yhw/eval_krong")
OUT_XLSX = ROOT / "paper_drafts" / "reranking_results_20260522.xlsx"
OUT_CSV = ROOT / "paper_drafts" / "reranking_results_20260522_paper_ready.csv"

sys.path.insert(0, str(ROOT / "scripts"))
from export_main_benchmark_excel import write_xlsx  # noqa: E402


SCAN_ROOTS = [
    ROOT / "sweep_results_rerank",
]
PUBLIC_ROOT = ROOT / "public_reranker_results"

DISPLAY = {
    "stage2_mlm00_interleave_ckpt18000": "Stage2 Interleave @18k",
    "stage2_mlm00_interleave_ckpt19000": "Stage2 Interleave @19k",
    "normal_random_new_ckpt18000": "Matched Decoder CPT @18k",
    "normal_random_new_ckpt19000": "Matched Decoder CPT @19k",
    "token_only_1b_cpt_ckpt19000": "Token-only CPT @19k",
    "mbert5_encoder_interleave_ckpt18000": "mBERT Encoder Interleave @18k",
    "llama32_1b_base": "Llama3.2-1B Base",
    "llama32_1b_interleave_cpt_ckpt18000": "Llama3.2-1B Interleave CPT @18k",
    "llama32_1b_interleave_cpt_ckpt19000": "Llama3.2-1B Interleave CPT @19k",
    "llama32_1b_interleave_cpt_ckpt19074": "Llama3.2-1B Interleave CPT @19074",
    "llama32_1b_cpt_vanilla_ckpt18000": "Llama3.2-1B Vanilla CPT @18k",
    "llama32_1b_cpt_vanilla_ckpt19000": "Llama3.2-1B Vanilla CPT @19k",
    "llama32_1b_cpt_vanilla_ckpt19074": "Llama3.2-1B Vanilla CPT @19074",
    "llama31_8b_base": "Llama3.1-8B Base",
    "llama31_8b_interleave_mlm00_copylow_ckpt19000": "Llama3.1-8B Interleave CPT @19k",
    "gemma3_1b_pt": "Gemma3-1B-PT",
    "olmo2_0425_1b": "OLMo-2-0425-1B",
    "smollm2_1p7b": "SmolLM2-1.7B",
    "polyglot_ko_1p3b": "Polyglot-Ko-1.3B",
    "kanana15_2p1b_base": "Kanana-1.5-2.1B-Base",
    "kanana_nano_2p1b_base": "Kanana-Nano-2.1B-Base",
    "beomi_llama3_koen_8b": "beomi Llama-3-KoEn-8B",
    "gemma3_12b_pt_addbos": "Gemma3-12B-PT addBOS",
    "olmo2_13b": "OLMo2-13B",
    "smollm3_3b": "SmolLM3-3B",
    "kanana15_8b_base": "Kanana-1.5-8B-Base",
    "kormo10b_base": "KORMo-10B-base",
    "beomi_llama3_open_ko_8b": "beomi Llama-3-Open-Ko-8B",
    "qwen3_1p7b_base": "Qwen3-1.7B-Base",
    "qwen3_8b_base": "Qwen3-8B-Base",
}

GROUP = {
    "stage2_mlm00_interleave_ckpt18000": "Ours 1B",
    "stage2_mlm00_interleave_ckpt19000": "Ours 1B",
    "normal_random_new_ckpt18000": "Ours 1B baseline",
    "normal_random_new_ckpt19000": "Ours 1B baseline",
    "token_only_1b_cpt_ckpt19000": "Ours 1B baseline",
    "mbert5_encoder_interleave_ckpt18000": "Ours 1B ablation",
    "llama32_1b_base": "Public-backbone 1B",
    "llama32_1b_interleave_cpt_ckpt18000": "Public-backbone 1B",
    "llama32_1b_interleave_cpt_ckpt19000": "Public-backbone 1B",
    "llama32_1b_interleave_cpt_ckpt19074": "Public-backbone 1B",
    "llama32_1b_cpt_vanilla_ckpt18000": "Public-backbone 1B baseline",
    "llama32_1b_cpt_vanilla_ckpt19000": "Public-backbone 1B baseline",
    "llama32_1b_cpt_vanilla_ckpt19074": "Public-backbone 1B baseline",
    "llama31_8b_base": "Public-backbone 8B",
    "llama31_8b_interleave_mlm00_copylow_ckpt19000": "Public-backbone 8B",
    "gemma3_1b_pt": "External 1B-ish",
    "olmo2_0425_1b": "External 1B-ish",
    "smollm2_1p7b": "External 1B-ish",
    "polyglot_ko_1p3b": "External 1B-ish",
    "kanana15_2p1b_base": "External 2B-ish",
    "kanana_nano_2p1b_base": "External 2B-ish",
    "beomi_llama3_koen_8b": "External 8B+",
    "gemma3_12b_pt_addbos": "External 8B+",
    "olmo2_13b": "External 8B+",
    "smollm3_3b": "External 3B",
    "kanana15_8b_base": "External 8B+",
    "kormo10b_base": "External 8B+",
    "beomi_llama3_open_ko_8b": "External 8B+",
    "qwen3_1p7b_base": "External 1B-ish",
    "qwen3_8b_base": "External 8B+",
}

CORE_ORDER = [
    "stage2_mlm00_interleave_ckpt18000",
    "normal_random_new_ckpt18000",
    "token_only_1b_cpt_ckpt19000",
    "mbert5_encoder_interleave_ckpt18000",
    "llama32_1b_base",
    "llama32_1b_cpt_vanilla_ckpt18000",
    "llama32_1b_interleave_cpt_ckpt18000",
    "llama32_1b_cpt_vanilla_ckpt19000",
    "llama32_1b_interleave_cpt_ckpt19000",
    "llama31_8b_base",
    "llama31_8b_interleave_mlm00_copylow_ckpt19000",
]

EXTERNAL_ORDER = [
    "gemma3_1b_pt",
    "olmo2_0425_1b",
    "smollm2_1p7b",
    "polyglot_ko_1p3b",
    "kanana15_2p1b_base",
    "kanana_nano_2p1b_base",
    "beomi_llama3_koen_8b",
    "kormo10b_base",
    "beomi_llama3_open_ko_8b",
    "smollm3_3b",
    "kanana15_8b_base",
    "gemma3_12b_pt_addbos",
    "olmo2_13b",
    "qwen3_1p7b_base",
    "qwen3_8b_base",
]

PAPER_REQUIRED_IDS = [
    "stage2_mlm00_interleave_ckpt18000",
    "normal_random_new_ckpt18000",
    "token_only_1b_cpt_ckpt19000",
    "mbert5_encoder_interleave_ckpt18000",
    "llama32_1b_base",
    "llama32_1b_cpt_vanilla_ckpt18000",
    "llama32_1b_interleave_cpt_ckpt18000",
    "polyglot_ko_1p3b",
    "gemma3_1b_pt",
    "olmo2_0425_1b",
    "smollm2_1p7b",
    "llama31_8b_base",
    "llama31_8b_interleave_mlm00_copylow_ckpt19000",
    "gemma3_12b_pt_addbos",
    "kanana15_8b_base",
    "olmo2_13b",
    "kormo10b_base",
    "beomi_llama3_koen_8b",
    "beomi_llama3_open_ko_8b",
]

# Paper table order requested by the user. Each model becomes three rows in
# the long sheet, corresponding to @1, @5, and @10 retrieval cutoffs.
ORDERED_PAPER_MODELS = [
    ("Ours 1B", "stage2_mlm00_interleave_ckpt18000", 18000),
    ("Ours 1B", "normal_random_new_ckpt18000", 18000),
    ("Ours 1B", "token_only_1b_cpt_ckpt19000", 19000),
    ("Ours 1B", "mbert5_encoder_interleave_ckpt18000", 18000),
    ("Public Llama3.2-1B", "llama32_1b_base", 0),
    ("Public Llama3.2-1B", "llama32_1b_cpt_vanilla_ckpt18000", 18000),
    ("Public Llama3.2-1B", "llama32_1b_interleave_cpt_ckpt18000", 18000),
    ("", "__separator_1__", ""),
    ("External public model", "gemma3_1b_pt", 0),
    ("External public model", "olmo2_0425_1b", 0),
    ("External public model", "smollm2_1p7b", 0),
    ("External public model", "polyglot_ko_1p3b", 0),
    ("", "__separator_2__", ""),
    ("Public Llama3.1-8B", "llama31_8b_base", 0),
    ("Public Llama3.1-8B", "llama31_8b_interleave_mlm00_copylow_ckpt19000", 19000),
    ("External public model", "kormo10b_base", 0),
    ("External public model", "olmo2_13b", 0),
    ("External public model", "beomi_llama3_koen_8b", 0),
    ("External public model", "beomi_llama3_open_ko_8b", 0),
    ("External public model", "gemma3_12b_pt_addbos", 0),
    ("External public model", "kanana15_8b_base", 0),
]

ORDERED_LONG_HEADERS = [
    "Model_Group",
    "Model",
    "Step",
    "Cutoff",
    "nDCG",
    "MRR",
    "Recall",
    "N_queries",
    "N_candidates",
    "Fewshot",
    "PositiveMarginRate",
    "AvgMargin",
    "Source",
    "Status",
]

ORDERED_WIDE_HEADERS = [
    "Model_Group",
    "Model",
    "Step",
    "N_queries",
    "N_candidates",
    "Fewshot",
    "nDCG@1",
    "MRR@1",
    "Recall@1",
    "nDCG@5",
    "MRR@5",
    "Recall@5",
    "nDCG@10",
    "MRR@10",
    "Recall@10",
    "PositiveMarginRate",
    "AvgMargin",
    "Source",
    "Status",
]

HEADERS = [
    "Group",
    "Model",
    "Raw_Model_ID",
    "N_queries",
    "N_candidates",
    "Fewshot",
    "nDCG@1",
    "MRR@1",
    "Recall@1",
    "nDCG@5",
    "MRR@5",
    "Recall@5",
    "nDCG@10",
    "MRR@10",
    "Recall@10",
    "PositiveMarginRate",
    "AvgMargin",
    "Margin_x100",
    "CandidatePool",
    "Source",
    "Notes",
]


def parse_timestamp(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^20\d{6}_\d{6}", part):
            return part[:15]
    return ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    return round(float(value) * 100.0, 4)


def num(value: Any, digits: int = 4) -> float | str:
    if value is None or value == "":
        return ""
    return round(float(value), digits)


def infer_model_id(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("checkpoint-"):
        return stem
    step = stem.split("-", 1)[1]
    run_dir = ""
    for part in path.parts:
        if re.match(r"^20\d{6}_\d{6}", part):
            run_dir = re.sub(r"^20\d{6}_\d{6}_", "", part)
            break
    run_dir = re.sub(r"_miracl.*$", "", run_dir)
    run_dir = re.sub(r"_\d{4,5}_\d{4,5}$", "", run_dir)
    return f"{run_dir}_ckpt{step}" if run_dir else f"checkpoint_{step}"


def candidate_pool_from_path(path: Path, payload: dict[str, Any]) -> str:
    text = str(path)
    if "hygiene" in text or "fixed4" in text or payload.get("num_fewshot") == 4:
        return "MIRACL-ko controlled hard-negative, fixed 4-shot hygiene"
    if "hardneg" in text:
        return "MIRACL-ko controlled hard-negative"
    if "fs4" in text:
        return "MIRACL-ko controlled hard-negative, 4-shot"
    if "fs2" in text:
        return "MIRACL-ko controlled hard-negative, 2-shot"
    return "MIRACL-ko reranking"


def row_from_metrics(model_id: str, payload: dict[str, Any], path: Path, public: bool = False) -> dict[str, Any]:
    metrics = payload.get("metrics", payload)
    raw_name = str(payload.get("model") or model_id) if public else model_id
    row = {
        "Group": "Dedicated reranker baseline" if public else GROUP.get(model_id, "External / Other"),
        "Model": str(payload.get("model") or DISPLAY.get(model_id, model_id)) if public else DISPLAY.get(model_id, model_id),
        "Raw_Model_ID": raw_name,
        "N_queries": int(float(metrics.get("num_queries", 0))) if metrics.get("num_queries") is not None else "",
        "N_candidates": int(float(metrics.get("num_candidates", 0))) if metrics.get("num_candidates") is not None else "",
        "Fewshot": int(float(metrics.get("num_fewshot", 0))) if metrics.get("num_fewshot") is not None else (0 if public else ""),
        "nDCG@1": pct(metrics.get("ndcg@1")),
        "MRR@1": pct(metrics.get("mrr@1")),
        "Recall@1": pct(metrics.get("recall@1")),
        "nDCG@5": pct(metrics.get("ndcg@5")),
        "MRR@5": pct(metrics.get("mrr@5")),
        "Recall@5": pct(metrics.get("recall@5")),
        "nDCG@10": pct(metrics.get("ndcg@10")),
        "MRR@10": pct(metrics.get("mrr@10")),
        "Recall@10": pct(metrics.get("recall@10")),
        "PositiveMarginRate": pct(metrics.get("positive_margin_rate")),
        "AvgMargin": num(metrics.get("avg_margin"), 6),
        "Margin_x100": pct(metrics.get("avg_margin")),
        "CandidatePool": "MIRACL-ko controlled hard-negative" if public else candidate_pool_from_path(path, metrics),
        "Source": rel(path),
        "Notes": "public reranker baseline; not same generative scoring interface" if public else "",
    }
    return row


def is_latest_paper_protocol(payload: dict[str, Any]) -> bool:
    """Return True for the current paper-ready reranking protocol.

    We intentionally keep older 2590-candidate runs out of paper-ready tables.
    They remain visible in the source/audit sheets, but the main CSV/XLSX rows
    should use only the strict 2256-candidate fixed 4-shot hygiene setup.
    """
    metrics = payload.get("metrics", payload)
    try:
        return (
            int(float(metrics.get("num_queries", 0) or 0)) == 209
            and int(float(metrics.get("num_candidates", 0) or 0)) == 2256
            and int(float(metrics.get("num_fewshot", 0) or 0)) == 4
        )
    except (TypeError, ValueError):
        return False


def scan_llm_rows() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    best: dict[str, tuple[tuple[int, str, str], Path, dict[str, Any]]] = {}
    latest_protocol: dict[str, tuple[tuple[int, str, str], Path, dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if "json/korean_rerank" not in str(path):
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if "ndcg@10" not in payload:
                continue
            model_id = infer_model_id(path)
            timestamp = parse_timestamp(path)
            num_queries = int(float(payload.get("num_queries", 0) or 0))
            num_candidates = int(float(payload.get("num_candidates", 0) or 0))
            num_fewshot = int(float(payload.get("num_fewshot", 0) or 0))

            # Best-available is retained only for audit/source sheets.
            quality = 0
            if num_fewshot == 4:
                quality += 1
            if num_queries == 209:
                quality += 1
            if num_candidates == 2256:
                quality += 1
            prio = (quality, timestamp, str(path))
            if model_id not in best or prio > best[model_id][0]:
                best[model_id] = (prio, path, payload)

            # Paper-ready rows must use the current strict 2256-candidate
            # fixed 4-shot hygiene protocol. This prevents old 2590-candidate
            # runs from being silently mixed into the main table.
            if is_latest_paper_protocol(payload):
                latest_prio = (num_candidates, timestamp, str(path))
                if model_id not in latest_protocol or latest_prio > latest_protocol[model_id][0]:
                    latest_protocol[model_id] = (latest_prio, path, payload)

            sources.append(
                {
                    "Raw_Model_ID": model_id,
                    "Timestamp": timestamp,
                    "N_queries": payload.get("num_queries", ""),
                    "N_candidates": payload.get("num_candidates", ""),
                    "Fewshot": payload.get("num_fewshot", ""),
                    "Paper_Protocol": "yes" if is_latest_paper_protocol(payload) else "no",
                    "nDCG@10": pct(payload.get("ndcg@10")),
                    "MRR@10": pct(payload.get("mrr@10")),
                    "Recall@10": pct(payload.get("recall@10")),
                    "AvgMargin": num(payload.get("avg_margin"), 6),
                    "Source": rel(path),
                }
            )
    rows = {model_id: row_from_metrics(model_id, payload, path) for model_id, (_, path, payload) in best.items()}
    latest_rows = {model_id: row_from_metrics(model_id, payload, path) for model_id, (_, path, payload) in latest_protocol.items()}
    return rows, latest_rows, sorted(sources, key=lambda r: (str(r["Raw_Model_ID"]), str(r["Timestamp"]), str(r["Source"])))

def scan_public_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PUBLIC_ROOT.exists():
        return rows
    for path in PUBLIC_ROOT.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if "metrics" not in payload:
            continue
        model_id = re.sub(r"[^a-zA-Z0-9]+", "_", str(payload.get("model", path.stem))).strip("_").lower()
        rows.append(row_from_metrics(model_id, payload, path, public=True))
    return sorted(rows, key=lambda r: str(r["Model"]))


def ordered_rows(rows_by_id: dict[str, dict[str, Any]], order: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model_id in order:
        seen.add(model_id)
        out.append(
            rows_by_id.get(
                model_id,
                {
                    "Group": GROUP.get(model_id, ""),
                    "Model": DISPLAY.get(model_id, model_id),
                    "Raw_Model_ID": model_id,
                    "Source": "",
                    "Notes": "MISSING: no latest reranking JSON found",
                },
            )
        )
    return out


def ordered_paper_wide_rows(latest_rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, model_id, step in ORDERED_PAPER_MODELS:
        if str(model_id).startswith("__separator_"):
            rows.append({h: "" for h in ORDERED_WIDE_HEADERS})
            continue
        src = latest_rows_by_id.get(str(model_id))
        if src is None:
            rows.append(
                {
                    "Model_Group": group,
                    "Model": DISPLAY.get(str(model_id), str(model_id)),
                    "Step": step,
                    "Status": "MISSING",
                }
            )
            continue
        rows.append(
            {
                "Model_Group": group,
                "Model": DISPLAY.get(str(model_id), str(model_id)),
                "Step": step,
                "N_queries": src.get("N_queries", ""),
                "N_candidates": src.get("N_candidates", ""),
                "Fewshot": src.get("Fewshot", ""),
                "nDCG@1": src.get("nDCG@1", ""),
                "MRR@1": src.get("MRR@1", ""),
                "Recall@1": src.get("Recall@1", ""),
                "nDCG@5": src.get("nDCG@5", ""),
                "MRR@5": src.get("MRR@5", ""),
                "Recall@5": src.get("Recall@5", ""),
                "nDCG@10": src.get("nDCG@10", ""),
                "MRR@10": src.get("MRR@10", ""),
                "Recall@10": src.get("Recall@10", ""),
                "PositiveMarginRate": src.get("PositiveMarginRate", ""),
                "AvgMargin": src.get("AvgMargin", ""),
                "Source": src.get("Source", ""),
                "Status": "OK",
            }
        )
    return rows


def ordered_paper_long_rows(latest_rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, model_id, step in ORDERED_PAPER_MODELS:
        if str(model_id).startswith("__separator_"):
            rows.append({h: "" for h in ORDERED_LONG_HEADERS})
            continue
        src = latest_rows_by_id.get(str(model_id))
        if src is None:
            for cutoff in (1, 5, 10):
                rows.append(
                    {
                        "Model_Group": group,
                        "Model": DISPLAY.get(str(model_id), str(model_id)),
                        "Step": step,
                        "Cutoff": cutoff,
                        "Status": "MISSING",
                    }
                )
            continue
        for cutoff in (1, 5, 10):
            rows.append(
                {
                    "Model_Group": group,
                    "Model": DISPLAY.get(str(model_id), str(model_id)),
                    "Step": step,
                    "Cutoff": cutoff,
                    "nDCG": src.get(f"nDCG@{cutoff}", ""),
                    "MRR": src.get(f"MRR@{cutoff}", ""),
                    "Recall": src.get(f"Recall@{cutoff}", ""),
                    "N_queries": src.get("N_queries", ""),
                    "N_candidates": src.get("N_candidates", ""),
                    "Fewshot": src.get("Fewshot", ""),
                    "PositiveMarginRate": src.get("PositiveMarginRate", ""),
                    "AvgMargin": src.get("AvgMargin", ""),
                    "Source": src.get("Source", ""),
                    "Status": "OK",
                }
            )
    return rows


def coverage_rows(latest_rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in PAPER_REQUIRED_IDS:
        row = latest_rows_by_id.get(model_id)
        ok = bool(row) and row.get("N_queries") == 209 and row.get("N_candidates") == 2256 and row.get("Fewshot") == 4
        rows.append(
            {
                "Raw_Model_ID": model_id,
                "Model": DISPLAY.get(model_id, model_id),
                "Status": "OK" if ok else "MISSING",
                "N_queries": "" if row is None else row.get("N_queries", ""),
                "N_candidates": "" if row is None else row.get("N_candidates", ""),
                "Fewshot": "" if row is None else row.get("Fewshot", ""),
                "Source": "" if row is None else row.get("Source", ""),
            }
        )
    return rows


def main() -> int:
    rows_by_id, latest_rows_by_id, source_rows = scan_llm_rows()
    public_rows = scan_public_rows()

    core_rows = ordered_rows(latest_rows_by_id, CORE_ORDER)
    external_rows = ordered_rows(latest_rows_by_id, EXTERNAL_ORDER)
    coverage = coverage_rows(latest_rows_by_id)
    ordered_wide_rows = ordered_paper_wide_rows(latest_rows_by_id)
    ordered_long_rows = ordered_paper_long_rows(latest_rows_by_id)

    group_order = {
        "Ours 1B": 0,
        "Ours 1B baseline": 1,
        "Ours 1B ablation": 2,
        "Public-backbone 1B": 3,
        "Public-backbone 1B baseline": 4,
        "Public-backbone 8B": 5,
        "External 1B-ish": 6,
        "External 2B-ish": 7,
        "External 3B": 8,
        "External 8B+": 9,
        "External / Other": 10,
    }
    all_rows = sorted(
        rows_by_id.values(),
        key=lambda r: (group_order.get(str(r.get("Group")), 99), str(r.get("Model"))),
    )

    readme = [
        {
            "item": "Scope",
            "value": "MIRACL-ko controlled hard-negative reranking results found under sweep_results_rerank and public_reranker_results.",
        },
        {
            "item": "Metric units",
            "value": "nDCG/MRR/Recall/PositiveMarginRate are percentages. AvgMargin is raw log-likelihood margin; Margin_x100 is provided only for old-table compatibility.",
        },
        {
            "item": "Few-shot hygiene",
            "value": "Paper-ready LLM rows require N_queries=209, N_candidates=2256, and Fewshot=4. Older 2590-candidate runs are retained only in all/source sheets and are not mixed into the main CSV.",
        },
        {
            "item": "Generated",
            "value": "2026-05-22",
        },
    ]

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(
        OUT_XLSX,
        [
            ("README", readme, ["item", "value"]),
            ("paper_ordered_long", ordered_long_rows, ORDERED_LONG_HEADERS),
            ("paper_ordered_wide", ordered_wide_rows, ORDERED_WIDE_HEADERS),
            ("paper_ready_core", core_rows, HEADERS),
            ("external_models", external_rows, HEADERS),
            ("coverage_check", coverage, None),
            ("all_best_available", all_rows, HEADERS),
            ("public_rerankers", public_rows, HEADERS),
            (
                "source_json_latest",
                source_rows,
                ["Raw_Model_ID", "Timestamp", "N_queries", "N_candidates", "Fewshot", "Paper_Protocol", "nDCG@10", "MRR@10", "Recall@10", "AvgMargin", "Source"],
            ),
        ],
    )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(core_rows + external_rows)

    print(f"[saved] {OUT_XLSX}")
    print(f"[csv] {OUT_CSV}")
    print(f"[rows] core={len(core_rows)} external={len(external_rows)} all={len(all_rows)} public={len(public_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
