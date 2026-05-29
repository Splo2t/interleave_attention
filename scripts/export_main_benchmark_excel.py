#!/usr/bin/env python3
"""Export paper-ready main benchmark tables.

This script intentionally avoids optional XLSX dependencies. It merges the
existing hand-checked summary CSV with newly run public/external model JSONs.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path("/mnt/nas_server_yhw/eval_krong")
OUT_XLSX = ROOT / "paper_drafts" / "main_benchmark_all_models_20260522.xlsx"
OUT_CSV = ROOT / "paper_drafts" / "main_benchmark_all_models_20260522.csv"

TASKS = [
    "mmlu",
    "kmmlu",
    "kobest",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "click",
    "csatqa",
]
TASK_LABELS = {
    "mmlu": "MMLU",
    "kmmlu": "KMMLU",
    "kobest": "KoBEST",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "openbookqa": "OpenBookQA",
    "click": "CLICk",
    "csatqa": "CSATQA",
}
KO_TASKS = ["kmmlu", "kobest", "click", "csatqa"]
EN_TASKS = ["mmlu", "arc_easy", "arc_challenge", "hellaswag", "openbookqa"]

OLD_SUMMARY = ROOT / "paper_drafts" / "all_models_main9_with_ours_20260521.csv"
SCAN_ROOTS = [
    # Public / external model sweeps.
    ROOT / "sweep_results_public",
    ROOT / "sweep_results_external_1b",
    ROOT / "sweep_results_external_main9",
    # Clean external KoBEST baselines used for stress pairing, still valid clean scores.
    ROOT / "sweep_results_stress_external_clean",
    # Internal Stage1/Stage2 and public-backbone CPT sweeps.
    ROOT / "sweep_results",
    ROOT / "sweep_results_8b",
]

MODEL_DISPLAY = {
    # Ours / controlled ablations.
    "stage2_mlm00_interleave_ckpt18000": "Stage2 Interleave @18k",
    "normal_random_new_ckpt18000": "Matched Decoder CPT @18k",
    "token_only_1b_cpt_ckpt19000": "Token-only CPT @19k",
    "mbert5_encoder_interleave_ckpt18000": "mBERT Encoder Interleave @18k",
    # Public-backbone 1B.
    "llama32_1b_base": "Llama3.2-1B Base",
    "llama32_1b_base_public": "Llama3.2-1B Base",
    "llama32_1b_cpt_vanilla_ckpt18000": "Llama3.2-1B Vanilla CPT @18k",
    "llama32_1b_cpt_vanilla_ckpt19000": "Llama3.2-1B Vanilla CPT @19k",
    "llama32_1b_interleave_cpt_ckpt18000": "Llama3.2-1B Interleave CPT @18k",
    "llama32_1b_interleave_cpt_ckpt19000": "Llama3.2-1B Interleave CPT @19k",
    # Public-backbone 8B.
    "llama31_8b_base": "Llama3.1-8B Base",
    "llama31_8b_base_public": "Llama3.1-8B Base",
    "llama31_8b_interleave_mlm00_copylow_ckpt19000": "Llama3.1-8B Interleave CPT @19k",
    # External public baselines.
    "polyglot_ko_1p3b": "Polyglot-Ko-1.3B",
    "gemma3_1b_pt": "Gemma3-1B-PT",
    "olmo2_0425_1b": "OLMo-2-0425-1B",
    "smollm2_1p7b": "SmolLM2-1.7B",
    "kanana15_2p1b_base": "Kanana-1.5-2.1B-Base",
    "kanana_nano_2p1b_base": "Kanana-Nano-2.1B-Base",
    "minwoo_llama32_1b_korean_base": "LLaMA-3.2-1B-Korean-base",
    "beomi_llama3_koen_8b": "beomi Llama-3-KoEn-8B",
    "beomi_llama3_open_ko_8b": "beomi Llama-3-Open-Ko-8B",
    "gemma3_12b_pt": "Gemma3-12B-PT",
    "gemma3_12b_pt_addbos": "Gemma3-12B-PT addBOS",
    "olmo2_13b": "OLMo2-13B",
    "smollm3_3b": "SmolLM3-3B",
    "kanana15_8b_base": "Kanana-1.5-8B-Base",
    "kormo10b_base": "KORMo-10B-base",
}

GROUP_BY_MODEL = {
    "stage2_mlm00_interleave_ckpt18000": "Ours Stage1→Stage2",
    "normal_random_new_ckpt18000": "Ours controlled ablation",
    "token_only_1b_cpt_ckpt19000": "Ours controlled ablation",
    "mbert5_encoder_interleave_ckpt18000": "Ours controlled ablation",
    "llama32_1b_base": "Public-backbone 1B",
    "llama32_1b_base_public": "Public-backbone 1B",
    "llama32_1b_cpt_vanilla_ckpt18000": "Public-backbone 1B",
    "llama32_1b_cpt_vanilla_ckpt19000": "Public-backbone 1B",
    "llama32_1b_interleave_cpt_ckpt18000": "Public-backbone 1B",
    "llama32_1b_interleave_cpt_ckpt19000": "Public-backbone 1B",
    "llama31_8b_base": "Public-backbone 8B",
    "llama31_8b_base_public": "Public-backbone 8B",
    "llama31_8b_interleave_mlm00_copylow_ckpt19000": "Public-backbone 8B",
    "polyglot_ko_1p3b": "External 1B-ish",
    "gemma3_1b_pt": "External 1B-ish",
    "olmo2_0425_1b": "External 1B-ish",
    "smollm2_1p7b": "External 1B-ish",
    "kanana15_2p1b_base": "External 2B-ish",
    "kanana_nano_2p1b_base": "External 2B-ish",
    "minwoo_llama32_1b_korean_base": "External 1B-ish",
    "beomi_llama3_koen_8b": "External 8B+",
    "beomi_llama3_open_ko_8b": "External 8B+",
    "gemma3_12b_pt": "External 8B+",
    "gemma3_12b_pt_addbos": "External 8B+",
    "olmo2_13b": "External 8B+",
    "smollm3_3b": "External 3B",
    "kanana15_8b_base": "External 8B+",
    "kormo10b_base": "External 8B+",
}

PAPER_READY_IDS = [
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

MODEL_ALIASES = {
    "llama32_1b_base_public": "llama32_1b_base",
    "llama31_8b_base_public": "llama31_8b_base",
    "gemma3_12b_pt": "gemma3_12b_pt_addbos",
}


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value: Any) -> float | None:
    value = as_float(value)
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 1.5 else value


def score_from_json(payload: dict[str, Any], task: str) -> float | None:
    if task == "kobest":
        return pct(payload.get("overall_micro"))
    if task == "click":
        return pct(payload.get("overall_micro") or payload.get("acc_norm") or payload.get("overall_acc_norm"))
    if task == "csatqa":
        return pct(payload.get("overall_micro") or payload.get("acc_norm") or payload.get("overall_acc_norm"))
    return pct(payload.get("overall_micro") or payload.get("acc"))


def parse_timestamp(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^20\d{6}_\d{6}", part):
            return part[:15]
    return ""


def infer_task(path: Path) -> str | None:
    parts = path.parts
    for task in TASKS:
        if task in parts:
            return task
    return None


def normalize_model_id(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def infer_model(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("checkpoint-"):
        return normalize_model_id(stem)

    step = stem.split("-", 1)[1]
    text = str(path).lower().replace("-", "_")
    rules = [
        ("llama32_1b_cpt_vanilla", "llama32_1b_cpt_vanilla"),
        ("llama32_1b_interleave_cpt", "llama32_1b_interleave_cpt"),
        ("llama31_8b_interleave_mlm00_copylow", "llama31_8b_interleave_mlm00_copylow"),
        ("checkpoints_interleave_full_enc4096_mlm025_mbert", "mbert5_encoder_interleave"),
        ("token_only_1b_cpt", "token_only_1b_cpt"),
        ("checkpoints_1b_cpt", "token_only_1b_cpt"),
        ("normal_random_new", "normal_random_new"),
        ("checkpoints_normal_random_new", "normal_random_new"),
        ("checkpoints_interleave_full_enc4096_mlm00_copylow", "stage2_mlm00_copylow"),
        ("checkpoints_interleave_full_enc4096_mlm00", "stage2_mlm00_interleave"),
    ]
    for marker, prefix in rules:
        if marker in text:
            return normalize_model_id(f"{prefix}_ckpt{step}")

    run_dir = ""
    for part in path.parts:
        if re.match(r"^20\d{6}_\d{6}", part):
            run_dir = re.sub(r"^20\d{6}_\d{6}_", "", part)
            break
    run_dir = re.sub(r"_json$", "", run_dir)
    run_dir = re.sub(r"_(main|extra|fill|rerun).*$", "", run_dir)
    return normalize_model_id(f"{run_dir}_ckpt{step}" if run_dir else f"checkpoint_{step}")


def source_priority(path: Path) -> tuple[str, str]:
    return (parse_timestamp(path), str(path))


def scan_json_scores() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    best: dict[tuple[str, str], tuple[tuple[str, str], float, Path]] = {}
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
            model = infer_model(path)
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            score = score_from_json(payload, task)
            if score is None:
                continue
            key = (model, task)
            prio = source_priority(path)
            if key not in best or prio > best[key][0]:
                best[key] = (prio, score, path)

    rows_by_model: dict[str, dict[str, Any]] = {}
    for (model, task), (_, score, path) in sorted(best.items()):
        row = rows_by_model.setdefault(
            model,
            {
                "Group": GROUP_BY_MODEL.get(model, "External / Public"),
                "Model": MODEL_DISPLAY.get(model, model),
                "Raw_Model_ID": model,
                "Step": 0,
                "Source": "latest public/external JSON",
            },
        )
        row[TASK_LABELS[task]] = round(score, 4)
        row[f"{TASK_LABELS[task]}_source"] = str(path.relative_to(ROOT))
        source_rows.append(
            {
                "model": MODEL_DISPLAY.get(model, model),
                "raw_model_id": model,
                "task": task,
                "score": round(score, 6),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows_by_model, source_rows


def avg(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def add_averages(row: dict[str, Any]) -> None:
    by_task = {task: as_float(row.get(TASK_LABELS[task])) for task in TASKS}
    ko_vals = [by_task[t] for t in KO_TASKS]
    en_vals = [by_task[t] for t in EN_TASKS]
    ko_avg = avg(ko_vals) if all(v is not None for v in ko_vals) else None
    en_avg = avg(en_vals) if all(v is not None for v in en_vals) else None
    total_avg = (ko_avg + en_avg) / 2.0 if ko_avg is not None and en_avg is not None else None
    row["Ko_avg"] = round(ko_avg, 4) if ko_avg is not None else ""
    row["En_avg"] = round(en_avg, 4) if en_avg is not None else ""
    row["AVG"] = round(total_avg, 4) if total_avg is not None else ""
    missing = [TASK_LABELS[t] for t, v in by_task.items() if v is None]
    row["Missing"] = ", ".join(missing)


def read_old_summary() -> list[dict[str, Any]]:
    if not OLD_SUMMARY.exists():
        return []
    rows: list[dict[str, Any]] = []
    with OLD_SUMMARY.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {
                "Group": raw.get("Group", ""),
                "Model": raw.get("Model", ""),
                "Raw_Model_ID": "",
                "Step": "",
                "Source": raw.get("Source", "old summary csv"),
            }
            for task in TASKS:
                label = TASK_LABELS[task]
                row[label] = round(float(raw[label]), 4) if raw.get(label) not in ("", None) else ""
            row["Ko_avg"] = round(float(raw["Ko_avg"]), 4) if raw.get("Ko_avg") else ""
            row["En_avg"] = round(float(raw["En_avg"]), 4) if raw.get("En_avg") else ""
            row["AVG"] = round(float(raw["AVG"]), 4) if raw.get("AVG") else ""
            row["Missing"] = ""
            rows.append(row)
    return rows


def ensure_required_rows(rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in PAPER_READY_IDS:
        if model_id in rows_by_id:
            row = rows_by_id[model_id]
        else:
            row = {
                "Group": GROUP_BY_MODEL.get(model_id, "MISSING"),
                "Model": MODEL_DISPLAY.get(model_id, model_id),
                "Raw_Model_ID": model_id,
                "Step": step_from_model_id(model_id),
                "Source": "MISSING: no JSON row found by exporter",
            }
        add_averages(row)
        rows.append(row)
    return rows


def step_from_model_id(model_id: str) -> int | str:
    m = re.search(r"_ckpt(\d+)$", model_id)
    if m:
        return int(m.group(1))
    return 0


def combine_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    old_rows = read_old_summary()
    scanned, sources = scan_json_scores()

    rows_by_id: dict[str, dict[str, Any]] = {}
    extra_rows: list[dict[str, Any]] = []

    # Keep old summary as a fallback only when scanned JSON did not provide an
    # explicit paper-ready model id. This prevents stale hand-merged rows from
    # hiding newly completed runs.
    for row in old_rows:
        raw_id = str(row.get("Raw_Model_ID") or "").strip()
        if raw_id:
            rows_by_id.setdefault(normalize_model_id(raw_id), dict(row))
        else:
            extra_rows.append(dict(row))

    for model_id, row in scanned.items():
        model_id = normalize_model_id(model_id)
        row = dict(row)
        row["Raw_Model_ID"] = model_id
        row["Model"] = MODEL_DISPLAY.get(model_id, row.get("Model", model_id))
        row["Group"] = GROUP_BY_MODEL.get(model_id, row.get("Group", "External / Public"))
        row["Step"] = step_from_model_id(model_id)
        add_averages(row)
        rows_by_id[model_id] = row

    paper_ready = ensure_required_rows(rows_by_id)

    all_rows_by_name: dict[tuple[str, str], dict[str, Any]] = {}
    for row in extra_rows:
        add_averages(row)
        all_rows_by_name[(str(row.get("Model")), str(row.get("Step")))] = row
    for row in rows_by_id.values():
        add_averages(row)
        all_rows_by_name[(str(row.get("Model")), str(row.get("Step")))] = row

    order_groups = {
        "Ours Stage1→Stage2": 0,
        "Ours controlled ablation": 1,
        "Public-backbone 1B": 2,
        "External 1B-ish": 3,
        "External 2B-ish": 4,
        "Public-backbone 8B": 5,
        "External 3B": 6,
        "External 8B+": 7,
    }
    rows = list(all_rows_by_name.values())
    rows.sort(key=lambda r: (order_groups.get(str(r.get("Group")), 99), str(r.get("Model")), str(r.get("Step"))))

    coverage = []
    for row in paper_ready:
        missing = str(row.get("Missing", ""))
        coverage.append(
            {
                "Raw_Model_ID": row.get("Raw_Model_ID", ""),
                "Model": row.get("Model", ""),
                "Status": "OK" if not missing and not str(row.get("Source", "")).startswith("MISSING") else "MISSING",
                "Missing": missing,
                "Source": row.get("Source", ""),
            }
        )
    return rows, paper_ready, sources, coverage

def excel_col(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def cell_xml(value: Any, row: int, col: int) -> str:
    ref = f"{excel_col(col)}{row}"
    if value is None:
        value = ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = html.escape(str(value), quote=True)
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def rows_to_sheet_xml(rows: Sequence[dict[str, Any]], headers: Sequence[str] | None = None) -> str:
    if headers is None:
        headers = []
        for row in rows:
            for key in row.keys():
                if key not in headers:
                    headers.append(key)
    if not rows:
        rows = [{"message": "(empty)"}]
        headers = ["message"]
    xml_rows = [
        '<row r="1">' + "".join(cell_xml(h, 1, i + 1) for i, h in enumerate(headers)) + "</row>"
    ]
    for r_idx, row in enumerate(rows, start=2):
        xml_rows.append(
            f'<row r="{r_idx}">'
            + "".join(cell_xml(row.get(h, ""), r_idx, c_idx + 1) for c_idx, h in enumerate(headers))
            + "</row>"
        )
    dim = f"A1:{excel_col(len(headers))}{len(rows) + 1}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dim}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>' + "".join(xml_rows) + "</sheetData></worksheet>"
    )


def write_xlsx(path: Path, sheets: Sequence[tuple[str, Sequence[dict[str, Any]], Sequence[str] | None]]) -> None:
    def safe_name(name: str) -> str:
        return re.sub(r"[\[\]\*:/\\?]", "_", name)[:31] or "Sheet"

    safe_sheets = [(safe_name(name), rows, headers) for name, rows, headers in sheets]
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx, _ in enumerate(safe_sheets, start=1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")
    workbook_sheets = "".join(
        f'<sheet name="{html.escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, (name, _, _) in enumerate(safe_sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    wb_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, _ in enumerate(safe_sheets, start=1):
        wb_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
    wb_rels.append(
        f'<Relationship Id="rId{len(safe_sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    wb_rels.append("</Relationships>")
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        '</styleSheet>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(content_types))
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels))
        z.writestr("xl/styles.xml", styles)
        for idx, (_, rows, headers) in enumerate(safe_sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{idx}.xml", rows_to_sheet_xml(rows, headers))


def main() -> int:
    rows, paper_ready, sources, coverage = combine_rows()
    headers = [
        "Group",
        "Model",
        "Raw_Model_ID",
        "Step",
        *[TASK_LABELS[t] for t in TASKS],
        "Ko_avg",
        "En_avg",
        "AVG",
        "Missing",
        "Source",
    ]
    new_public = [
        r for r in rows
        if r.get("Raw_Model_ID") in {
            "gemma3_1b_pt",
            "kanana15_2p1b_base",
            "kanana_nano_2p1b_base",
            "beomi_llama3_koen_8b",
            "beomi_llama3_open_ko_8b",
            "olmo2_0425_1b",
            "smollm2_1p7b",
            "polyglot_ko_1p3b",
        }
    ]
    readme = [
        {
            "item": "generated_at",
            "value": datetime.now().isoformat(timespec="seconds"),
            "note": "Generated from curated CSV plus latest JSON files under all active main benchmark sweep roots. Required paper models are coverage-checked explicitly.",
        },
        {
            "item": "metric",
            "value": "overall_micro / acc",
            "note": "Matches the dashboard Micro / Acc convention; no acc_norm is used for main benchmark table.",
        },
        {
            "item": "Ko_avg",
            "value": "mean(KMMLU, KoBEST, CLICk, CSATQA)",
            "note": "Computed only when all four Korean-side metrics are present.",
        },
        {
            "item": "En_avg",
            "value": "mean(MMLU, ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA)",
            "note": "Computed only when all five English-side metrics are present.",
        },
        {
            "item": "AVG",
            "value": "mean(Ko_avg, En_avg)",
            "note": "Computed only when both group averages are present.",
        },
    ]
    write_xlsx(
        OUT_XLSX,
        [
            ("README", readme, None),
            ("paper_ready_required", paper_ready, headers),
            ("coverage_check", coverage, None),
            ("all_detected_rows", rows, headers),
            ("new_public_models", new_public, headers),
            ("source_json_latest", sources, None),
        ],
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paper_ready)
    print(OUT_XLSX)
    print(OUT_CSV)
    print(f"paper_ready_rows={len(paper_ready)} all_rows={len(rows)} new_public_rows={len(new_public)} sources={len(sources)} missing={sum(1 for r in coverage if r['Status'] != 'OK')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
