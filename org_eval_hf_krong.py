
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_paper_benchmarks.py

HF(Transformers)로 래핑된 Krong/KORMo 계열 모델을 대상으로
- MMLU (cais/mmlu)
- KMMLU (HAERAE-HUB/KMMLU)
- KoBEST (skt/kobest_v1; 옵션)

을 "log-likelihood(정답 후보 문자열의 로그우도)" 방식으로 평가합니다.

이 스크립트는 사용자 예시 코드의 형태를 그대로 따릅니다:

  model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
  processor = AutoProcessor.from_pretrained(..., trust_remote_code=True)
  tokenizer = processor.tokenizer

그리고 (가능하면) processor.prepare_generate_inputs(model, prompt)로
forward에 필요한 입력(dict)을 구성합니다.

주의:
- processor.prepare_generate_inputs가 생성용(generate) 입력을 만들면서
  forward에 불필요한 키를 섞어 넣을 수 있으므로, model.forward 시그니처를 보고
  허용되는 키만 필터링합니다.
- chat_template 기반 평가도 옵션으로 지원합니다(--use_chat_template).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import re
import shlex
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset, get_dataset_config_names
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast


def _load_tokenizer_with_fallback(
    ckpt_path: str,
    *,
    trust_remote_code: bool = True,
    use_fast: bool = True,
):
    """
    Local checkpoints may store tokenizer_class=TokenizersBackend with only tokenizer.json.
    In that case AutoTokenizer cannot resolve the class name, so fall back to a generic
    PreTrainedTokenizerFast initialized from tokenizer_config.json + tokenizer.json.
    """
    try:
        return AutoTokenizer.from_pretrained(
            ckpt_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
    except ValueError as e:
        cfg_path = os.path.join(ckpt_path, "tokenizer_config.json")
        tok_json_path = os.path.join(ckpt_path, "tokenizer.json")
        if not (os.path.isfile(cfg_path) and os.path.isfile(tok_json_path)):
            raise

        with open(cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)

        if tok_cfg.get("tokenizer_class") != "TokenizersBackend" and tok_cfg.get("backend") != "tokenizers":
            raise

        kwargs = {}
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ):
            if tok_cfg.get(key) is not None:
                kwargs[key] = tok_cfg[key]

        if tok_cfg.get("model_max_length") is not None:
            kwargs["model_max_length"] = tok_cfg["model_max_length"]
        kwargs["clean_up_tokenization_spaces"] = tok_cfg.get("clean_up_tokenization_spaces", True)

        print(
            f"[tokenizer] AutoTokenizer fallback for {ckpt_path}: "
            f"using PreTrainedTokenizerFast from tokenizer.json ({e})"
        )
        return PreTrainedTokenizerFast(tokenizer_file=tok_json_path, **kwargs)


TASK_OVERVIEW_ORDER = ("mmlu", "kmmlu", "kobest")
DEFAULT_KOBEST_TASKS = ["boolq", "copa", "hellaswag", "sentineg", "wic"]


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


def _now_iso_utc() -> str:
    return _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_log_root() -> str:
    return str(Path(__file__).resolve().parent / "logs")


def _infer_checkpoint_step(ckpt_name: str) -> str:
    m = re.search(r"(?:checkpoint|ckpt|step)[-_]?(\d+)$", ckpt_name or "", flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _resolve_model_group(log_group: str, model_arch: str, ckpt_path: str, model_label: str) -> str:
    if log_group in {"krong", "kormo", "others"}:
        return log_group
    text = " ".join([model_arch or "", ckpt_path or "", model_label or ""]).lower()
    if "kormo" in text:
        return "kormo"
    if "krong" in text:
        return "krong"
    return "others"


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
            for k, v in row.items():
                merged[k] = _csv_value(v)
            rows[idx] = merged
            found = True
            break
    if not found:
        rows.append(row)
    fieldnames = _merge_fieldnames(existing_fieldnames, preferred_fieldnames, list(row.keys()))
    _write_csv_rows(path, rows, fieldnames)


def _build_profile_tag(args, selected_items: Optional[List[str]]) -> str:
    parts = [f"k{args.k_shot}"]
    if args.use_chat_template:
        parts.append("chat")
    if args.enable_thinking:
        parts.append("thinking")
    if args.dec_max_len and args.dec_max_len > 0:
        parts.append(f"dec{args.dec_max_len}")
    if args.limit and args.limit > 0:
        parts.append(f"limit{args.limit}")

    if args.task in {"mmlu", "kmmlu"}:
        parts.append(f"subjects{len(selected_items)}" if selected_items else "full")
    elif args.task == "kobest":
        is_full = not selected_items or selected_items == DEFAULT_KOBEST_TASKS
        parts.append("full" if is_full and args.kobest_split == "test" else f"tasks{len(selected_items or [])}")
        if args.kobest_split != "test":
            parts.append(f"split-{_slugify(args.kobest_split, 24)}")

    if args.experiment_tag:
        parts.append(_slugify(args.experiment_tag, 32))
    return "__".join(parts)


def _build_scope_text(args, selected_items: Optional[List[str]]) -> str:
    bits = [f"k_shot={args.k_shot}"]
    if args.task in {"mmlu", "kmmlu"}:
        bits.append("subjects=ALL" if not selected_items else f"subjects={','.join(selected_items)}")
    elif args.task == "kobest":
        bits.append(f"tasks={','.join(selected_items or DEFAULT_KOBEST_TASKS)}")
        bits.append(f"split={args.kobest_split}")
    if args.limit and args.limit > 0:
        bits.append(f"limit={args.limit}")
    if args.use_chat_template:
        bits.append("chat_template=1")
    if args.enable_thinking:
        bits.append("enable_thinking=1")
    if args.dec_max_len and args.dec_max_len > 0:
        bits.append(f"dec_max_len={args.dec_max_len}")
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
        "num_parts",
        "scope",
        "model_group",
        "model_label",
        "profile_tag",
        "ckpt_name",
        "ckpt_step",
        "ckpt_path",
        "model_arch",
        "dtype",
        "device_map",
        "k_shot",
        "seed",
        "limit",
        "subjects",
        "kobest_tasks",
        "kobest_split",
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
                f"{task}_num_parts",
                f"{task}_scope",
                f"{task}_run_id",
                f"{task}_updated_at",
                f"{task}_detail_csv",
            ]
        )
    return fields


def _make_log_context(args, selected_items: Optional[List[str]]) -> dict[str, Any]:
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
        "dtype": args.dtype,
        "device_map": args.device_map,
        "k_shot": args.k_shot,
        "seed": args.seed,
        "limit": args.limit if args.limit and args.limit > 0 else "",
        "subjects": ",".join(selected_items or []) if args.task in {"mmlu", "kmmlu"} else "",
        "kobest_tasks": ",".join(selected_items or []) if args.task == "kobest" else "",
        "kobest_split": args.kobest_split if args.task == "kobest" else "",
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
    return "subject" if task in {"mmlu", "kmmlu"} else "benchmark"


def _result_num_parts(results: dict[str, float]) -> int:
    return sum(1 for key in results if not key.startswith("overall_"))


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
    row = {k: v for k, v in ctx.items() if k not in {"log_root"}}
    row["overall_macro"] = results.get("overall_macro", "")
    row["overall_micro"] = results.get("overall_micro", "")
    row["num_parts"] = _result_num_parts(results)
    return row


def _overview_update_row(results: dict[str, float], ctx: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model_group": ctx["model_group"],
        "model_label": ctx["model_label"],
        "profile_tag": ctx["profile_tag"],
        "ckpt_name": ctx["ckpt_name"],
        "ckpt_step": ctx["ckpt_step"],
        "ckpt_path": ctx["ckpt_path"],
        "model_arch": ctx["model_arch"],
        "dtype": ctx["dtype"],
        "device_map": ctx["device_map"],
        "latest_task": ctx["task"],
        "latest_run_id": ctx["run_id"],
        "latest_run_at": ctx["timestamp_utc"],
        f"{ctx['task']}_macro": results.get("overall_macro", ""),
        f"{ctx['task']}_micro": results.get("overall_micro", ""),
        f"{ctx['task']}_num_parts": _result_num_parts(results),
        f"{ctx['task']}_scope": ctx["scope"],
        f"{ctx['task']}_run_id": ctx["run_id"],
        f"{ctx['task']}_updated_at": ctx["timestamp_utc"],
        f"{ctx['task']}_detail_csv": ctx["detail_csv"],
    }
    return row


def _write_result_logs(args, results: dict[str, float], selected_items: Optional[List[str]]) -> dict[str, str]:
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

def _split_for_tokenize(context: str, continuation: str) -> tuple[str, str]:
    """
    lm-eval 권장 규칙:
      context의 trailing whitespace는 continuation 앞으로 이동.
    """
    context = context or ""
    continuation = continuation or ""
    n_ws = len(context) - len(context.rstrip())
    if n_ws > 0:
        continuation = context[-n_ws:] + continuation
        context = context[:-n_ws]
    return context, continuation


def _longest_common_prefix(a: list[int], b: list[int]) -> int:
    i = 0
    m = min(len(a), len(b))
    while i < m and a[i] == b[i]:
        i += 1
    return i


def _prompt_and_cont_ids_lmeval(tokenizer, context: str, continuation: str) -> tuple[str, list[int], bool]:
    """
    lm-eval 방식:
      cont_ids = tok(context+cont) - tok(context)
    만약 ctx_ids가 full_ids의 prefix가 아니면(LCP fallback),
      prompt_text를 decode(full_ids[:lcp])로 바꿔서 alignment를 보장.
    return: (prompt_text, cont_ids, diverged)
    """
    context, continuation = _split_for_tokenize(context, continuation)

    ctx_ids  = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)

    lcp = _longest_common_prefix(ctx_ids, full_ids)
    diverged = (lcp != len(ctx_ids))
    if diverged:
        prompt_ids = full_ids[:lcp]
        # decode는 lmeval과 동일하게 special token 제거 X
        try:
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        cont_ids = full_ids[lcp:]
        return prompt_text, cont_ids, True

    cont_ids = full_ids[len(ctx_ids):]
    return context, cont_ids, False

# -----------------------------
# Prompt utils
# -----------------------------
def _letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _choices_block_any(choices: List[str]) -> str:
    L = _letters(len(choices))
    return "\n".join(f"{L[i]}. {choices[i]}" for i in range(len(choices)))


def _to_letter_from_mmlu_answer(ans) -> str:
    """
    MMLU answer를 'A'/'B'/'C'/'D'로 통일.
    - int 0..3  -> A..D
    - int 1..4  -> A..D
    - str 'A'..'D' 그대로
    - str '0'..'3' 또는 '1'..'4' 도 처리
    """
    if isinstance(ans, str):
        s = ans.strip()
        if s in {"A", "B", "C", "D"}:
            return s
        if s.isdigit():
            v = int(s)
            if v in (0, 1, 2, 3):
                return "ABCD"[v]
            if v in (1, 2, 3, 4):
                return "ABCD"[v - 1]
    elif isinstance(ans, int):
        if ans in (0, 1, 2, 3):
            return "ABCD"[ans]
        if ans in (1, 2, 3, 4):
            return "ABCD"[ans - 1]
    raise ValueError(f"Unrecognized MMLU answer value: {ans} (type {type(ans)})")


def build_prompt_mmlu(
    subject: str,
    question: str,
    choices: List[str],
    shots: List[Tuple[str, List[str], str]],
    *,
    add_header: bool = True,
    space_after_answer: bool = True,
    shot_blank_line: bool = True,
) -> str:
    """
    lm-eval-harness의 MMLU 프롬프트 형태와 호환되게 구성.
    """
    buf: List[str] = []
    if add_header:
        buf.append(f"The following are multiple choice questions (with answers) about {subject}.")
        buf.append("")

    for qs, chs, gold in shots:
        buf.append(qs.strip())
        buf.append(_choices_block_any(chs))
        buf.append(f"Answer: {gold}")
        if shot_blank_line:
            buf.append("")

    buf.append(question.strip())
    buf.append(_choices_block_any(choices))
    buf.append("Answer:" + (" " if space_after_answer else ""))
    return "\n".join(buf)


def build_prompt_kmmlu(
    category: str,
    question: str,
    choices: List[str],  # [A,B,C,D]
    shots: List[Tuple[str, List[str], str]],
    *,
    use_fullwidth_colon: bool = True,
    space_after_colon: bool = False,
    shot_blank_line: bool = True,
) -> str:
    """
    KMMLU(Korean) 프롬프트.
    기본은 '정답：' (전각 콜론) + 공백 없음.
    """
    colon = "：" if use_fullwidth_colon else ":"
    mark = f"정답{colon}" + (" " if space_after_colon else "")

    buf: List[str] = []
    for qs, chs, gold in shots:
        buf.append(qs.strip())
        buf.append(_choices_block_any(chs))
        buf.append(f"{mark}{gold}")
        if shot_blank_line:
            buf.append("")

    buf.append(question.strip())
    buf.append(_choices_block_any(choices))
    buf.append(mark)
    return "\n".join(buf)


# -----------------------------
# KoBEST helpers (optional)
# -----------------------------
KOBEST_DATASET_ID = "skt/kobest_v1"


def kobest_boolq_doc_to_text(doc: dict) -> str:
    return f"""{doc["paragraph"]} 질문: {doc["question"]} 답변: """


def kobest_boolq_choices() -> List[str]:
    return ["아니오", "예"]


def kobest_boolq_gold_idx(doc: dict) -> int:
    return int(doc["label"])  # 0/1


def copa_doc_to_text(doc: dict) -> str:
    connector = {"원인": " 왜냐하면", "결과": " 그래서"}[doc["question"].strip()]
    return f"""{doc["premise"]} {connector}"""


def copa_doc_to_choice(doc: dict) -> List[str]:
    return [f"""{doc["alternative_1"]}""", f"""{doc["alternative_2"]}"""]


def sentineg_doc_to_text(doc: dict) -> str:
    return f"""문장: {doc["sentence"]} 긍부정:"""


def wic_doc_to_text(doc: dict) -> str:
    return f"""문장1: {doc["context_1"]} 문장2: {doc["context_2"]} 두 문장에서 {doc["word"]}가 같은 뜻으로 쓰였나?"""


def hellaswag_process_doc(ds):
    def preprocessor(example):
        return {
            "query": f"""문장: {example["context"]}""",
            "choices": [
                example["ending_1"],
                example["ending_2"],
                example["ending_3"],
                example["ending_4"],
            ],
            "gold": int(example["label"]),
        }

    return ds.map(preprocessor)


def _kobest_get_dataset(task: str, split: str):
    ds = load_dataset(KOBEST_DATASET_ID, task, split=split)
    if task == "hellaswag":
        ds = hellaswag_process_doc(ds)
    return ds


def _kobest_doc_to_text_and_choices(task: str, doc: dict) -> Tuple[str, List[str], int]:
    if task == "boolq":
        ctx = kobest_boolq_doc_to_text(doc)
        choices = kobest_boolq_choices()
        gold_idx = kobest_boolq_gold_idx(doc)
    elif task == "copa":
        ctx = copa_doc_to_text(doc)
        choices = copa_doc_to_choice(doc)
        gold_idx = int(doc["label"])
    elif task == "sentineg":
        ctx = sentineg_doc_to_text(doc)
        choices = ["부정", "긍정"]
        gold_idx = int(doc["label"])
    elif task == "wic":
        ctx = wic_doc_to_text(doc)
        choices = ["아니오", "예"]
        gold_idx = int(doc["label"])
    elif task == "hellaswag":
        ctx = doc["query"]
        choices = list(doc["choices"])
        gold_idx = int(doc["gold"])
    else:
        raise ValueError(f"Unknown KoBEST task: {task}")
    return ctx, choices, gold_idx


def _build_kobest_shots(task: str, k_shot: int) -> List[dict]:
    if k_shot <= 0:
        return []
    train_ds = _kobest_get_dataset(task, "train")
    rows = list(train_ds)
    return rows[: min(k_shot, len(rows))]


def _build_mc_fewshot_ctx(task: str, doc: dict, shots: List[dict]) -> str:
    parts: List[str] = []
    for sh in shots:
        s_ctx, s_choices, s_gold = _kobest_doc_to_text_and_choices(task, sh)
        parts.append(s_ctx + s_choices[s_gold])
        parts.append("")
    ctx, _, _ = _kobest_doc_to_text_and_choices(task, doc)
    parts.append(ctx)
    return "\n".join(parts)

def _build_mc_fewshot_ctx_lmeval(task: str, doc: dict, shots: list[dict]) -> str:
    parts: list[str] = []
    LMEVAL_TARGET_DELIM = " "     # lm-eval default target_delimiter 
    LMEVAL_FEWSHOT_DELIM = "\n\n" # lm-eval default fewshot_delimiter 
    for sh in shots:
        s_ctx, s_choices, s_gold = _kobest_doc_to_text_and_choices(task, sh)
        parts.append(s_ctx + LMEVAL_TARGET_DELIM + s_choices[s_gold] + LMEVAL_FEWSHOT_DELIM)
    ctx, _, _ = _kobest_doc_to_text_and_choices(task, doc)
    parts.append(ctx)  # eval doc에는 정답을 붙이지 않음
    return "".join(parts)


# -----------------------------
# Model input / scoring helpers
# -----------------------------
def _resolve_dtype(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown dtype: {name}")


def _parse_device_map_arg(s: str):
    """
    CLI에서 받은 device_map 문자열을 HF from_pretrained에 안전하게 전달하기 위한 헬퍼.

    - "auto" / "balanced" / "balanced_low_0" / "sequential" 는 그대로 사용
    - "cuda:0" / "cuda:1" / "cpu" / "mps" 같은 단일 디바이스 지정은 {"": device} 형태로 변환
      (accelerate/transformers가 dict device_map을 이해함)
    - "0" 같은 숫자면 {"": int}로 변환
    """
    if s is None:
        return "auto"
    s = str(s).strip()
    if not s:
        return "auto"

    presets = {"auto", "balanced", "balanced_low_0", "sequential"}
    if s in presets:
        return s

    if s.isdigit():
        return {"": int(s)}

    if s.startswith("cuda:"):
        try:
            idx = int(s.split(":", 1)[1])
            return {"": idx}
        except Exception:
            return {"": s}

    if s in {"cuda", "gpu"}:
        return {"": 0}

    if s in {"cpu", "mps"}:
        return {"": s}

    # 마지막 폴백: 그대로 넘기고, transformers에서 에러 나면 사용자가 수정
    return s


def _get_forward_accepts_kwargs(model) -> Tuple[Optional[set], bool]:
    """
    returns: (allowed_names or None, has_var_kwargs)
    - has_var_kwargs=True이면 forward가 **kwargs를 받으므로 필터링을 최소화해도 됨.
    """
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None, True  # 보수적으로 kwargs 허용한다고 가정

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return None, True
    return set(params.keys()), False


def _filter_forward_kwargs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    allowed, has_var_kw = _get_forward_accepts_kwargs(model)
    if has_var_kw or allowed is None:
        # 그래도 텐서/기본 타입만 남기고 싶다면 여기서 추가 필터 가능
        return inputs
    return {k: v for k, v in inputs.items() if k in allowed}


def _detect_dec_key(inputs: Dict[str, Any]) -> str:
    for k in ("input_ids", "decoder_input_ids"):
        if k in inputs:
            return k
    raise KeyError(f"Cannot find decoder ids key in inputs. keys={list(inputs.keys())}")


def _detect_attn_key(inputs: Dict[str, Any]) -> Optional[str]:
    for k in ("attention_mask", "decoder_attention_mask"):
        if k in inputs:
            return k
    return None


def _truncate_left_for_decoder_aligned_tensors(
    inputs: Dict[str, Any],
    *,
    max_len: Optional[int],
) -> Dict[str, Any]:
    """
    max_len이 설정되면 디코더 시퀀스 길이를 (좌측) 트렁케이션.
    디코더 길이와 동일한 2D 텐서들은 같이 잘라줌 (cross_k_allow_lens 등).
    """
    if max_len is None or max_len <= 0:
        return inputs

    dec_key = _detect_dec_key(inputs)
    dec = inputs[dec_key]
    if not torch.is_tensor(dec) or dec.ndim != 2:
        return inputs

    T = dec.size(1)
    if T <= max_len:
        return inputs

    off = T - max_len
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.ndim == 2 and v.size(0) == dec.size(0) and v.size(1) == T:
            out[k] = v[:, off:]
        else:
            out[k] = v
    return out


def _build_inputs_with_continuation(
    base_inputs: Dict[str, Any],
    cont_ids: List[int],
) -> Dict[str, Any]:
    """
    base_inputs(프롬프트) 뒤에 cont_ids를 이어붙인 입력 dict를 생성.
    - input_ids/attention_mask/cross_k_allow_lens/position_ids 등 decoder-aligned 텐서는 길이를 함께 늘려줌
    - encoder_input_ids 등은 그대로 유지
    """
    if not cont_ids:
        return dict(base_inputs)

    dec_key = _detect_dec_key(base_inputs)
    attn_key = _detect_attn_key(base_inputs)

    base_dec = base_inputs[dec_key]
    if not torch.is_tensor(base_dec) or base_dec.ndim != 2:
        raise ValueError(f"{dec_key} must be a 2D torch tensor, got: {type(base_dec)} / shape={getattr(base_dec,'shape',None)}")

    B, T = base_dec.size()
    device = base_dec.device
    dtype = base_dec.dtype

    cont = torch.tensor([cont_ids], device=device, dtype=dtype)
    new_dec = torch.cat([base_dec, cont], dim=1)

    out: Dict[str, Any] = {}
    out[dec_key] = new_dec

    # attention mask
    if attn_key is not None and attn_key in base_inputs and torch.is_tensor(base_inputs[attn_key]):
        base_attn = base_inputs[attn_key]
        ones = torch.ones((B, len(cont_ids)), device=base_attn.device, dtype=base_attn.dtype)
        out[attn_key] = torch.cat([base_attn, ones], dim=1)

    # 나머지 키들: decoder-aligned 2D 텐서는 적절히 연장
    for k, v in base_inputs.items():
        if k in out:
            continue
        if k == attn_key:
            # 이미 처리
            continue

        if not torch.is_tensor(v):
            out[k] = v
            continue

        # decoder-aligned 여부: (B,T)이고 T가 base_dec의 T와 같으면 연장
        if v.ndim == 2 and v.size(0) == B and v.size(1) == T:
            if k in ("position_ids", "decoder_position_ids"):
                # 마지막 pos에서 +1씩 증가
                last = v[:, -1]  # (B,)
                add = torch.arange(len(cont_ids), device=v.device, dtype=v.dtype).unsqueeze(0)  # (1,M)
                new_pos = (last.unsqueeze(1) + 1) + add  # (B,M)
                out[k] = torch.cat([v, new_pos], dim=1)
            elif k in ("cross_k_allow_lens", "cross_attention_allow_lens", "cross_k_allow_len"):
                last = v[:, -1:].to(v.dtype)
                rep = last.expand(B, len(cont_ids))
                out[k] = torch.cat([v, rep], dim=1)
            else:
                # 기본: 마지막 값을 반복
                last = v[:, -1:].to(v.dtype)
                rep = last.expand(B, len(cont_ids))
                out[k] = torch.cat([v, rep], dim=1)
        else:
            out[k] = v

    return out


@torch.inference_mode()
def _forward_logits(model, inputs: Dict[str, Any]) -> torch.Tensor:
    """
    model forward 실행 후 logits 텐서를 반환.
    """
    # forward 시그니처에 맞춰 필터
    fwd_inputs = _filter_forward_kwargs(model, dict(inputs))
    # logits만 필요: 캐시 끄기
    #out = model(**fwd_inputs, use_cache=False)
    fwd_inputs["use_cache"] = False   # dict 안에서 강제로 덮어쓰기
    out = model(**fwd_inputs)
    if isinstance(out, dict):
        logits = out.get("logits", None)
        if logits is None:
            raise ValueError("Model output dict does not contain 'logits'.")
        return logits
    if hasattr(out, "logits"):
        return out.logits
    raise ValueError("Model output has no logits attribute.")

import torch.nn.functional as F


def build_inputs_with_cont_ws(base_inputs: dict, cont_ids: list[int]) -> dict:
    if not cont_ids:
        return dict(base_inputs)

    inp = base_inputs["input_ids"]           # (1, T)
    att = base_inputs["attention_mask"]      # (1, T)
    L   = base_inputs["cross_k_allow_lens"]  # (1, T)

    dev = inp.device
    cont = torch.tensor([cont_ids], device=dev, dtype=inp.dtype)
    ones = torch.ones((1, len(cont_ids)), device=dev, dtype=att.dtype)

    # ✅ 핵심: 새 토큰마다 L_cur을 append (generate 때랑 동일)
    # L_cur = int(base_inputs["ws_state"]["L_cur"])
    # L_add = torch.full((1, len(cont_ids)), L_cur, device=dev, dtype=L.dtype)
    L_last = int(L[0, -1].item())
    L_add = torch.full((1, len(cont_ids)), L_last, device=dev, dtype=L.dtype)
    out = dict(base_inputs)
    out["input_ids"]          = torch.cat([inp, cont], dim=1)
    out["attention_mask"]     = torch.cat([att, ones], dim=1)
    out["cross_k_allow_lens"] = torch.cat([L,   L_add], dim=1)
    out["use_cache"]          = False
    return out



@torch.inference_mode()
def loglikelihood_continuation_ws(model, base_inputs: dict, cont_ids: list[int]) -> float:
    if not cont_ids:
        return float("-inf")

    M = len(cont_ids)

    full_inputs = build_inputs_with_cont_ws(base_inputs, cont_ids)

    # ✅ 성능: 마지막 (M+1) 위치 logits만 만들면 됨
    full_inputs["logits_to_keep"] = M + 1
    full_inputs["use_cache"] = False

    out = model(**full_inputs)   # 여기서 use_cache 중복 전달 금지!
    logits = out.logits          # (1, M+1, V)

    # logits은 마지막 M+1개 위치(T-1..T+M-1)이고,
    # cont 토큰 확률에 필요한 건 앞의 M개(T-1..T+M-2)
    slice_logits = logits[:, :-1, :]  # (1, M, V)

    logprobs = F.log_softmax(slice_logits.float(), dim=-1)  # (1, M, V)
    ids_t = torch.tensor(cont_ids, device=logprobs.device).view(1, M, 1)
    picked = torch.gather(logprobs, 2, ids_t).squeeze(-1)   # (1, M)
    return float(picked.sum().item())

@torch.inference_mode()
def loglikelihood_continuation_ws_dynamic(model, base_inputs: dict, cont_ids: list[int]) -> float:
    if not cont_ids:
        return float("-inf")

    # base_inputs를 후보마다 오염시키지 않도록 복사
    input_ids = base_inputs["input_ids"].clone()
    attention_mask = base_inputs.get("attention_mask", torch.ones_like(input_ids)).clone()

    if "ws_state" not in base_inputs:
        # AutoModelForCausalLM 모델:
        total_lp = 0.0
        for tid in cont_ids:
            fwd = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                #ws_state=ws_state,
                use_cache=False,
            )
            fwd["use_cache"] = False
            fwd["logits_to_keep"] = 1  # 다음 토큰 확률만 필요
            out = model(**fwd)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0, tid].item()
            total_lp += float(lp)

            # 토큰을 시퀀스에 붙임 (다음 루프에서 prepare_inputs_for_generation이 prev_len 보고 L append)
            step = torch.tensor([[tid]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, step], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(step)], dim=1)
            
    else:
        ws0 = base_inputs["ws_state"]
        # deepcopy는 텐서까지 복제될 수 있어 비추 → 얕은 복사 + list만 복사
        ws_state = dict(ws0)
        ws_state["L_per_token"] = list(ws0["L_per_token"])
        ws_state["prev_len"] = int(ws0["prev_len"])
        ws_state["enc_text"] = str(ws0["enc_text"])
        ws_state["L_cur"] = int(ws0["L_cur"])

        total_lp = 0.0
        for tid in cont_ids:
            # ✅ 여기서 ws_state 갱신 + encoder 갱신 + cross_k_allow_lens 생성까지 처리됨
            fwd = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ws_state=ws_state,
                use_cache=False,
            )
            fwd["use_cache"] = False
            fwd["logits_to_keep"] = 1  # 다음 토큰 확률만 필요

            out = model(**fwd)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0, tid].item()
            total_lp += float(lp)

            # 토큰을 시퀀스에 붙임 (다음 루프에서 prepare_inputs_for_generation이 prev_len 보고 L append)
            step = torch.tensor([[tid]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, step], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(step)], dim=1)

    return total_lp

def continuation_ids_from_concat(tokenizer, prompt_text: str, prompt_ids: list[int], cand_text: str) -> list[int]:
    full = tokenizer(prompt_text + cand_text, add_special_tokens=False)["input_ids"]

    # processor가 BOS를 수동으로 넣는 경우가 있으니 prompt_ids 기준으로 맞추기
    bos = tokenizer.bos_token_id
    if bos is not None and len(prompt_ids) > 0 and prompt_ids[0] == bos:
        if len(full) == 0 or full[0] != bos:
            full = [bos] + full

    # prefix 정합성 체크(깨지면 폴백)
    if len(full) >= len(prompt_ids) and full[:len(prompt_ids)] == prompt_ids:
        return full[len(prompt_ids):]

    # 폴백: 기존 방식
    return tokenizer(cand_text, add_special_tokens=False)["input_ids"]

@dataclass
class EvalScorerConfig:
    use_chat_template: bool = False
    system_prompt: str = ""
    enable_thinking: bool = False
    dec_max_len: Optional[int] = None
    space_variant_mode: str = "auto"  # "auto" | "both" | "none"


class HFEvalScorer:
    """
    - prompt 문자열 -> (옵션) chat_template로 감싸기
    - processor.prepare_generate_inputs(model, prompt) 로 base_inputs 생성
    - 후보 문자열들의 로그우도를 계산해 argmax 선택
    """

    def __init__(self, model, processor, tokenizer, cfg: EvalScorerConfig):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.cfg = cfg

    def format_prompt(self, raw_prompt: str) -> str:
        if not self.cfg.use_chat_template:
            return raw_prompt
        msgs = []
        if self.cfg.system_prompt:
            msgs.append({"role": "system", "content": self.cfg.system_prompt})
        msgs.append({"role": "user", "content": raw_prompt})
        # add_generation_prompt=True: assistant 헤더까지 포함
        try:
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
        except TypeError:
            # enable_thinking 인자를 지원하지 않는 토크나이저/버전 폴백
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

    def prepare_base_inputs(self, prompt: str) -> Dict[str, Any]:
        # 가능한 한 사용자가 제공한 패턴을 그대로 사용
        if self.processor is not None and hasattr(self.processor, "prepare_generate_inputs"):
            inputs = self.processor.prepare_generate_inputs(self.model, prompt)
        else:
            # 폴백: tokenizer로만 구성 (모델이 extra key를 요구하면 실패할 수 있음)
            tok = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            # device 결정: input_ids만이라도 모델 첫 디바이스로
            dev = getattr(self.model, "device", None)
            if dev is None or (hasattr(dev, "type") and dev.type == "meta"):
                try:
                    dev = self.model._execution_device  # accelerate dispatched model
                except Exception:
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(dev) for k, v in tok.items()}

        # forward에 쓸 수 있게 좌측 트렁케이션(옵션)
        inputs = _truncate_left_for_decoder_aligned_tensors(inputs, max_len=self.cfg.dec_max_len)
        return inputs

    def _tokenize_no_special(self, s: str) -> List[int]:
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        return list(ids) if isinstance(ids, (list, tuple)) else []

    def _label_variants(self, label: str, prompt_text: str) -> List[str]:
        """
        label 후보 문자열 변형:
          - "auto": 프롬프트가 공백으로 끝나면 label만, 아니면 ["label", " "+label]
          - "both": 항상 ["label", " "+label]
          - "none": ["label"]만
        """
        # ✅ lm-eval과 동일: 후보 문자열은 있는 그대로 1개만 평가
        return [label or ""]
        label = label or ""
        mode = (self.cfg.space_variant_mode or "auto").lower()
        if mode == "none":
            return [label]
        # ✅ 기존 벤치마크와 동일: 프롬프트가 공백으로 끝나든 말든 항상 둘 다 시도
        if label.startswith(" "):
            return [label]
        return [label, " " + label]


    @torch.inference_mode()
    def score_labels(self, raw_prompt: str, labels: List[str]) -> Dict[str, float]:
        prompt = self.format_prompt(raw_prompt)
        base_inputs = self.prepare_base_inputs(prompt)
        ws_state = base_inputs["ws_state"]
        #prompt = self.format_prompt(raw_prompt)
        #base_inputs = self.prepare_base_inputs(prompt)
        prompt_ids = base_inputs["input_ids"][0].tolist()
        # forward에 직접 들어가야 interleave가 켜짐
        base_inputs["encoder_hidden_states"]  = ws_state["encoder_hidden_states"]
        base_inputs["encoder_attention_mask"] = ws_state["encoder_attention_mask"]

        T = int(base_inputs["input_ids"].size(1))
        L_list = ws_state["L_per_token"][-T:]   # ✅ 좌측 트렁케이션이면 뒤쪽만 남김
        base_inputs["cross_k_allow_lens"] = torch.tensor([L_list], device=base_inputs["input_ids"].device, dtype=torch.long)


        # 중복 전달 방지: dict 안에서만 통일
        base_inputs["use_cache"] = False
        use_dynamic = True
        if self.cfg.dec_max_len is not None:
            use_dynamic = False
        # 미리 토큰화해보고, 1토큰 후보가 하나라도 있으면 base logits 1회로 재활용
        tokenized_variants: Dict[str, List[List[int]]] = {}
        for lab in labels:
            vars_ = self._label_variants(lab, prompt)
            ids_list = []
            seen = set()
            for v in vars_:
                #ids = tuple(self._tokenize_no_special(v))
                ids = tuple(continuation_ids_from_concat(self.tokenizer, prompt, prompt_ids, v))

                if not ids:
                    continue
                if ids in seen:
                    continue
                seen.add(ids)
                ids_list.append(list(ids))
            tokenized_variants[lab] = ids_list

        need_base = any(len(ids) == 1 for ids_list in tokenized_variants.values() for ids in ids_list)
        base_logprobs = None
        if need_base:
            logits = _forward_logits(self.model, base_inputs)
            # next-token 분포는 마지막 위치 logits (CausalLM 규약)
            base_logprobs = F.log_softmax(logits[:, -1, :].float(), dim=-1)[0]  # (V,)

        scores: Dict[str, float] = {}
        for lab in labels:
            best = float("-inf")
            for ids in tokenized_variants.get(lab, []):
                if len(ids) == 1 and base_logprobs is not None:
                    s = float(base_logprobs[ids[0]].item())
                else:
                    #s = loglikelihood_continuation_ws_dynamic(self.model, base_inputs, ids)
                    s = loglikelihood_continuation_ws(self.model, base_inputs, ids) if not use_dynamic \
                        else loglikelihood_continuation_ws_dynamic(self.model, base_inputs, ids)
                if s > best:
                    best = s
            scores[lab] = best
        return scores
    @torch.inference_mode()
    def score_labels_ll_and_len(self, raw_prompt: str, labels: list[str]) -> dict[str, tuple[float, int]]:
        """
        lm-eval 방식에 맞춘 scoring:
          - context/continuation split + ctx+cont - ctx 토크나이즈
          - (필요 시) LCP fallback으로 prompt_text 자체를 바꿈
          - WS 모델이므로 continuation은 dynamic teacher-forcing로 점수
        return: {label: (loglikelihood_sum, cont_token_len)}
        """
        context = self.format_prompt(raw_prompt)

        # label마다 prompt_text가 달라질 수도 있어서(lcp fallback) prompt_text로 그룹핑
        groups = defaultdict(list)  # prompt_text -> [(label, cont_ids)]
        for lab in labels:
            prompt_text, cont_ids, _ = _prompt_and_cont_ids_lmeval(self.tokenizer, context, lab)
            groups[prompt_text].append((lab, cont_ids))

        out: dict[str, tuple[float, int]] = {}

        for prompt_text, items in groups.items():
            base_inputs = self.prepare_base_inputs(prompt_text)

            # ⚠️ lm-eval emulation 실험에서는 트렁케이션 off를 권장
            #    (on이면 ws_state/prev_len/L_per_token 정합까지 맞춰야 함)
            for lab, cont_ids in items:
                ll = loglikelihood_continuation_ws_dynamic(self.model, base_inputs, cont_ids)
                out[lab] = (ll, max(1, len(cont_ids)))

        return out

# -----------------------------
# Evaluation loops
# -----------------------------
def evaluate_mmlu(
    scorer: HFEvalScorer,
    *,
    k_shot: int = 5,
    subjects: Optional[List[str]] = None,
    seed: int = 42,
    limit_per_subject: Optional[int] = None,
) -> Dict[str, float]:
    random.seed(seed)
    results: Dict[str, float] = {}
    all_correct = 0
    all_total = 0

    if subjects is None:
        subjects = [s for s in get_dataset_config_names("cais/mmlu") if s != "all"]
    else:
        subjects = [s.strip() for s in subjects if s and s.strip()]

    for subj in subjects:
        try:
            dev_ds = load_dataset("cais/mmlu", name=subj, split="dev")
            test_ds = load_dataset("cais/mmlu", name=subj, split="test")
        except Exception as e:
            print(f"[MMLU] Skip subject '{subj}' due to load error: {e}")
            continue

        k = min(k_shot, len(dev_ds))
        shots: List[Tuple[str, List[str], str]] = []
        for i in range(k):
            q = dev_ds[i]["question"]
            ch = dev_ds[i]["choices"]
            gold_letter = _to_letter_from_mmlu_answer(dev_ds[i]["answer"])
            shots.append((q, ch, gold_letter))

        correct = 0
        total = 0
        n_eval = len(test_ds) if limit_per_subject is None else min(len(test_ds), int(limit_per_subject))

        bar = tqdm(total=n_eval, desc=f"MMLU  | {subj}", leave=True, dynamic_ncols=True)
        for j in range(n_eval):
            ex = test_ds[j]
            question = ex["question"]
            choices = ex["choices"]
            gold_letter = _to_letter_from_mmlu_answer(ex["answer"])
            gold_idx = "ABCD".index(gold_letter)

            prompt = build_prompt_mmlu(
                subj, question, choices, shots,
                add_header=True,
                space_after_answer=True,
                shot_blank_line=True,
            )

            #scores = scorer.score_labels(prompt, ["A", "B", "C", "D"])
            scores = scorer.score_labels_ll_and_len(prompt, ["A", "B", "C", "D"])
            pred_letter = max(scores.keys(), key=lambda k_: scores[k_])
            pred_idx = "ABCD".index(pred_letter)

            correct += int(pred_idx == gold_idx)
            total += 1
            bar.update(1)
            bar.set_postfix(acc=f"{100.0*correct/max(1,total):5.2f}%")

        bar.close()

        acc = (correct / total) if total else 0.0
        results[subj] = acc
        all_correct += correct
        all_total += total
        tqdm.write(f"[MMLU ] {subj:32s}  acc={acc*100:5.2f}%  (n={total})")

    results["overall_micro"] = (all_correct / all_total) if all_total else 0.0
    subject_accs = [results[s] for s in results.keys() if s != "overall_micro"]
    results["overall_macro"] = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    print(f"[MMLU] OVERALL micro={results['overall_micro']*100:5.2f}% "
          f"macro={results['overall_macro']*100:5.2f}%  (N={all_total})")
    return results


def evaluate_kmmlu(
    scorer: HFEvalScorer,
    *,
    k_shot: int = 5,
    subjects: Optional[List[str]] = None,
    seed: int = 42,
    limit_per_subject: Optional[int] = None,
) -> Dict[str, float]:
    random.seed(seed)
    if subjects is None:
        subjects = get_dataset_config_names("HAERAE-HUB/KMMLU")
    subjects = sorted({s.strip() for s in subjects if s and s.strip()})

    results: Dict[str, float] = {}
    all_correct = 0
    all_total = 0

    def _ans_to_idx(a: int) -> int:
        return int(a) - 1  # 1..4 -> 0..3

    for cat in subjects:
        try:
            ds_test = load_dataset("HAERAE-HUB/KMMLU", name=cat, split="test")
            ds_dev = load_dataset("HAERAE-HUB/KMMLU", name=cat, split="dev")
        except Exception as e:
            print(f"[KMMLU] Skip '{cat}' (load error): {e}")
            continue

        dev_rows = list(ds_dev)
        k = min(k_shot, len(dev_rows))
        shots: List[Tuple[str, List[str], str]] = []
        for i in range(k):
            ex = dev_rows[i]
            q = ex["question"]
            ch = [ex["A"], ex["B"], ex["C"], ex["D"]]
            gold_letter = "ABCD"[_ans_to_idx(ex["answer"])]
            shots.append((q, ch, gold_letter))

        rows = list(ds_test)
        n_eval = len(rows) if limit_per_subject is None else min(len(rows), int(limit_per_subject))
        correct = 0
        total = 0

        bar = tqdm(total=n_eval, desc=f"KMMLU | {cat}", leave=True, dynamic_ncols=True)
        for j in range(n_eval):
            ex = rows[j]
            q = ex["question"]
            ch = [ex["A"], ex["B"], ex["C"], ex["D"]]
            gold_idx = _ans_to_idx(ex["answer"])

            prompt = build_prompt_kmmlu(
                cat, q, ch, shots,
                use_fullwidth_colon=True,
                space_after_colon=False,
                shot_blank_line=True,
            )

            #scores = scorer.score_labels(prompt, ["A", "B", "C", "D"])
            scores = scorer.score_labels_ll_and_len(prompt, ["A", "B", "C", "D"])
            
            pred_letter = max(scores.keys(), key=lambda k_: scores[k_])
            pred_idx = "ABCD".index(pred_letter)

            correct += int(pred_idx == gold_idx)
            total += 1
            bar.update(1)
            bar.set_postfix(acc=f"{100.0*correct/max(1,total):5.2f}%")

        bar.close()

        acc = (correct / total) if total else 0.0
        results[cat] = acc
        all_correct += correct
        all_total += total
        print(f"[KMMLU] {cat:32s}  acc={acc*100:5.2f}%  (n={total})")

    results["overall_micro"] = (all_correct / all_total) if all_total else 0.0
    subject_accs = [results[c] for c in results.keys() if c != "overall_micro"]
    results["overall_macro"] = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    print(f"[KMMLU] OVERALL micro={results['overall_micro']*100:5.2f}% "
          f"macro={results['overall_macro']*100:5.2f}%  (N={all_total})")
    return results


def evaluate_kobest(
    scorer: HFEvalScorer,
    *,
    tasks: Optional[List[str]] = None,
    k_shot: int = 0,
    split: str = "test",
    limit_per_task: Optional[int] = None,
) -> Dict[str, float]:
    if tasks is None:
        tasks = DEFAULT_KOBEST_TASKS
    rng = random.Random(42)  # lm-eval sampler 흉내

    results: Dict[str, float] = {}
    all_correct = 0
    all_total = 0
    #LMEVAL_TARGET_DELIM = " "     # lm-eval default target_delimiter 
    #LMEVAL_FEWSHOT_DELIM = "\n\n" # lm-eval default fewshot_delimiter 
    for task in tasks:
        try:
            ds_eval = _kobest_get_dataset(task, split)
        except Exception as e:
            print(f"[KoBEST][{task}] dataset load error: {e}")
            continue
        train_rows = list(_kobest_get_dataset(task, "train"))
        eval_rows  = list(_kobest_get_dataset(task, split))

        #eval_rows = list(ds_eval)
        n_eval = len(eval_rows) if limit_per_task is None else min(len(eval_rows), int(limit_per_task))
        #shots = _build_kobest_shots(task, k_shot)

        correct = 0
        total = 0
        bar = tqdm(total=n_eval, desc=f"KoBEST | {task}", dynamic_ncols=True)
        for i in range(n_eval):
            ex = eval_rows[i]
            try:
                shots = rng.sample(train_rows, k_shot) if k_shot > 0 else []
                #ctx = _build_mc_fewshot_ctx_lmeval(task, ex, shots)  # lm-eval 방식으로 context/continuation 분리
                ctx = _build_mc_fewshot_ctx(task, ex, shots)
                
                _, choices, gold_idx = _kobest_doc_to_text_and_choices(task, ex)
            except Exception as e:
                bar.write(f"[KoBEST][{task}][SKIP] parse error: {e}")
                bar.update(1)
                continue
            '''
            labels = [LMEVAL_TARGET_DELIM + c for c in choices]
            scores = scorer.score_labels_ll_and_len(ctx, labels)
            
            pred_label  = max(labels, key=lambda k_: scores[k_][0])  # acc
            pred_choice = pred_label[len(LMEVAL_TARGET_DELIM):] if pred_label.startswith(LMEVAL_TARGET_DELIM) else pred_label
            pred_idx    = choices.index(pred_choice)
            '''
            
            scores = scorer.score_labels_ll_and_len(ctx, choices)
            
            pred_choice = max(scores.keys(), key=lambda k_: scores[k_])
            pred_idx = choices.index(pred_choice)
            
            
            correct += int(pred_idx == gold_idx)
            total += 1
            bar.update(1)
            bar.set_postfix(acc=f"{100.0*correct/max(1,total):5.2f}%")

        bar.close()

        acc = (correct / total) if total else 0.0
        results[task] = acc
        all_correct += correct
        all_total += total
        print(f"[KoBEST] {task:12s} acc={acc*100:5.2f}%  (n={total})")

    results["overall_micro"] = (all_correct / all_total) if all_total else 0.0
    task_accs = [results[t] for t in tasks if t in results]
    results["overall_macro"] = (sum(task_accs) / len(task_accs)) if task_accs else 0.0
    print(f"[KoBEST] OVERALL micro={results['overall_micro']*100:5.2f}% "
          f"macro={results['overall_macro']*100:5.2f}%  (N={all_total})")
    return results


#---------------------------------------


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True, help="from_pretrained 경로 또는 repo id")
    ap.add_argument("--task", type=str, choices=["mmlu", "kmmlu", "kobest"], default="mmlu")
    ap.add_argument("--model_label", type=str, default="", help="CSV 로그에 기록할 모델 표시 이름(미지정 시 ckpt basename)")
    ap.add_argument("--log_group", type=str, choices=["auto", "krong", "kormo", "others"], default="auto",
                    help="CSV 로그 분류. auto면 ckpt_path/model_arch 기반으로 kormo/krong/others 자동 분류")
    ap.add_argument("--experiment_tag", type=str, default="",
                    help="실험 태그. run_id/profile_tag/overview row 구분에 사용")
    ap.add_argument("--log_root", type=str, default="",
                    help="CSV 로그 루트 경로(기본: 스크립트 옆 logs/)")
    ap.add_argument("--disable_csv_log", action="store_true",
                    help="CSV 실험 로그 기록 비활성화")

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device_map", type=str, default="auto", help="HF device_map (예: auto, cuda:0, cpu)")
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--subjects", type=str, default="", help="(mmlu/kmmlu) 콤마로 분리된 과목 목록. 미지정 시 전체")
    ap.add_argument("--limit", type=int, default=0, help="과목(또는 task)당 평가 샘플 수 제한(디버그용). 0이면 전체")

    # chat_template 옵션 (사용자 예시 기반)
    ap.add_argument("--use_chat_template", action="store_true", help="prompt를 chat_template로 감싼 뒤 평가")
    ap.add_argument("--system_prompt", type=str, default="", help="chat_template 사용 시 system 메시지")
    ap.add_argument("--enable_thinking", action="store_true", help="chat_template의 enable_thinking=True로 렌더")

    ap.add_argument("--space_variant_mode", type=str, default="both", choices=["auto","both","none"])

    ap.add_argument("--dec_max_len", type=int, default=0, help=">0이면 디코더 입력을 좌측 트렁케이션")
    ap.add_argument("--out_json", type=str, default="", help="결과를 JSON으로 저장할 경로(옵션)")

    # KoBEST 옵션
    ap.add_argument("--kobest_tasks", type=str, default="", help="(kobest) tasks: boolq,copa,hellaswag,sentineg,wic")
    ap.add_argument("--kobest_split", type=str, default="test", help="(kobest) split: train/validation/test")
    ap.add_argument("--model_arch", type=str, default="", help="모델 아키텍처 이름 (예: gpt2, llama2, etc.) - processor/토크나이저 로드에 활용")

    args = ap.parse_args()
    if args.model_arch == "krong":
        # ---- 모델/프로세서 로드: 사용자 예시 그대로 ----
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            trust_remote_code=True,
            device_map=_parse_device_map_arg(args.device_map),
            torch_dtype=_resolve_dtype(args.dtype),
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = _load_tokenizer_with_fallback(args.ckpt_path, trust_remote_code=True, use_fast=True)
    else:
        # ---- 모델/프로세서 로드: HF 표준 방식 ----
        tokenizer = _load_tokenizer_with_fallback(args.ckpt_path, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            trust_remote_code=True,
            device_map=_parse_device_map_arg(args.device_map),
            torch_dtype=_resolve_dtype(args.dtype),
        )
        model.eval()
        processor = None
        
    cfg = EvalScorerConfig(
        use_chat_template=bool(args.use_chat_template),
        system_prompt=args.system_prompt or "",
        enable_thinking=bool(args.enable_thinking),
        dec_max_len=(args.dec_max_len if args.dec_max_len and args.dec_max_len > 0 else None),
        space_variant_mode=args.space_variant_mode,
    )
    scorer = HFEvalScorer(model, processor, tokenizer, cfg)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    selected_items: Optional[List[str]] = None

    if args.task == "mmlu":
        selected_items = subjects
        results = evaluate_mmlu(
            scorer,
            k_shot=args.k_shot,
            subjects=subjects,
            seed=args.seed,
            limit_per_subject=limit,
        )
    elif args.task == "kmmlu":
        selected_items = subjects
        results = evaluate_kmmlu(
            scorer,
            k_shot=args.k_shot,
            subjects=subjects,
            seed=args.seed,
            limit_per_subject=limit,
        )
    else:  # kobest
        tasks = [t.strip() for t in args.kobest_tasks.split(",") if t.strip()] or None
        selected_items = tasks or list(DEFAULT_KOBEST_TASKS)
        results = evaluate_kobest(
            scorer,
            tasks=tasks,
            k_shot=args.k_shot,
            split=args.kobest_split,
            limit_per_task=limit,
        )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.out_json}")

    _write_result_logs(args, results, selected_items)


if __name__ == "__main__":
    main()
