#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from krong_eval.benchmarks.kobest import (
    DEFAULT_KOBEST_TASKS,
    KOBEST_DATASET_ID,
    _kobest_doc_to_text_and_choices,
    _kobest_get_dataset,
)
from krong_eval.cache import DEFAULT_CACHE_ROOT, prepare_cache_paths

PROJECT_ROOT = Path(__file__).resolve().parent
INLINE_WS_PATTERN = r"[^\S\n]+"
JOSA = (
    "으로", "이나", "이나마", "이라", "라고", "에서", "에게", "까지", "부터", "보다", "처럼",
    "은", "는", "이", "가", "을", "를", "과", "와", "로", "에", "의", "나", "만", "도",
)
COMPACTION_LEFT_BOUNDARY_CHARS = set(":：;；,.?!…)]}〉》」』\"'")
COMPACTION_RIGHT_BOUNDARY_CHARS = set("([{<〈《「『\"'")
VARIANT_TO_RANDOM_P = {
    "ko_random_p25": 0.25,
    "ko_random_p50": 0.50,
}
SUPPORTED_VARIANTS = (*VARIANT_TO_RANDOM_P.keys(), "ko_josa_preserve_compaction_hard")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def stable_rng(seed: int, *parts: Any) -> random.Random:
    payload = json.dumps([seed, *parts], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def inline_whitespace_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(INLINE_WS_PATTERN, text)]


def word_left_of_gap(text: str, gap_start: int) -> str:
    start = gap_start
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:gap_start]


def word_right_of_gap(text: str, gap_end: int) -> str:
    end = gap_end
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[gap_end:end]


def has_compactable_surface(text: str) -> bool:
    return bool(re.search(r"[가-힣A-Za-z0-9]", text or ""))


def surface_ends_with_josa(text: str) -> bool:
    return any(text.endswith(josa) and len(text) > len(josa) for josa in JOSA)


def gap_is_after_surface_josa(text: str, gap_start: int) -> bool:
    left_word = word_left_of_gap(text, gap_start)
    if not left_word:
        return True
    return surface_ends_with_josa(left_word)


def is_safe_random_wsdrop_gap(text: str, gap_start: int, gap_end: int) -> bool:
    if gap_start >= gap_end or "\n" in text[gap_start:gap_end] or text[gap_start:gap_end].strip():
        return False
    left_word = word_left_of_gap(text, gap_start)
    right_word = word_right_of_gap(text, gap_end)
    if not left_word or not right_word:
        return False
    if left_word[-1] in COMPACTION_LEFT_BOUNDARY_CHARS:
        return False
    if right_word[0] in COMPACTION_RIGHT_BOUNDARY_CHARS:
        return False
    return has_compactable_surface(left_word) and has_compactable_surface(right_word)


def is_safe_josa_hard_gap(text: str, gap_start: int, gap_end: int) -> bool:
    if gap_start >= gap_end or "\n" in text[gap_start:gap_end] or text[gap_start:gap_end].strip():
        return False
    left_word = word_left_of_gap(text, gap_start)
    right_word = word_right_of_gap(text, gap_end)
    if not left_word or not right_word:
        return False
    if not (has_compactable_surface(left_word) and has_compactable_surface(right_word)):
        return False
    if left_word[-1] in COMPACTION_LEFT_BOUNDARY_CHARS:
        return False
    if right_word[0] in COMPACTION_RIGHT_BOUNDARY_CHARS:
        return False
    return not gap_is_after_surface_josa(text, gap_start)


def delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return str(text)
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def apply_random_wsdrop(text: str, *, p: float, seed: int, field: str) -> tuple[str, list[str]]:
    rng = stable_rng(seed, "ko_random_p", field, text)
    spans = [
        (start, end)
        for start, end in inline_whitespace_spans(text)
        if is_safe_random_wsdrop_gap(text, start, end) and rng.random() < p
    ]
    if not spans:
        return str(text), []
    return delete_spans(str(text), spans), [f"RP_random_whitespace_drop_p{p:.3f}"]


def apply_josa_hard(text: str) -> tuple[str, list[str]]:
    spans = [
        (start, end)
        for start, end in inline_whitespace_spans(text)
        if is_safe_josa_hard_gap(text, start, end)
    ]
    if not spans:
        return str(text), []
    return delete_spans(str(text), spans), ["JPH_josa_preserving_hard_content_compaction"]


def transform_text(text: Any, *, variant: str, seed: int, field: str) -> tuple[str, list[str]]:
    if variant in VARIANT_TO_RANDOM_P:
        return apply_random_wsdrop(str(text), p=VARIANT_TO_RANDOM_P[variant], seed=seed, field=field)
    if variant == "ko_josa_preserve_compaction_hard":
        return apply_josa_hard(str(text))
    raise ValueError(f"Unsupported variant: {variant}")


def choices_preserve_uniqueness(original: list[str], transformed: list[str]) -> bool:
    if len(set(original)) != len(original):
        return True
    return len(set(transformed)) == len(transformed)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def limited_rows(dataset, limit: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in dataset]
    if limit and limit > 0:
        return rows[:limit]
    return rows


def prepare_output_root(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {path} (pass --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_kobest_variant(args: argparse.Namespace, output_root: Path, cache_dir: str | None) -> dict[str, Any]:
    tasks = split_csv(args.kobest_tasks) or list(DEFAULT_KOBEST_TASKS)
    splits = split_csv(args.kobest_splits) or ["train", "validation", "test"]
    counts: dict[str, dict[str, int]] = {}
    changed_counts: dict[str, dict[str, int]] = {}

    for task in tqdm(tasks, desc=f"build {args.variant}", dynamic_ncols=True):
        counts[task] = {}
        changed_counts[task] = {}
        for split in splits:
            ds = _kobest_get_dataset(task, split, cache_dir=cache_dir)
            out_rows: list[dict[str, Any]] = []
            n_changed = 0
            for idx, row in enumerate(limited_rows(ds, args.limit_per_split)):
                prompt, choices, gold_idx = _kobest_doc_to_text_and_choices(task, row)
                source_choices = [str(choice) for choice in choices]
                prompt, prompt_rules = transform_text(prompt, variant=args.variant, seed=args.seed, field="prompt")
                rules = set(prompt_rules)

                transformed_choices: list[str] = []
                choice_rules: set[str] = set()
                for choice_idx, choice in enumerate(source_choices):
                    transformed_choice, choice_rule_list = transform_text(
                        choice,
                        variant=args.variant,
                        seed=args.seed,
                        field=f"choices.{choice_idx}",
                    )
                    transformed_choices.append(transformed_choice)
                    choice_rules.update(choice_rule_list)
                if choices_preserve_uniqueness(source_choices, transformed_choices):
                    choices = transformed_choices
                    rules.update(choice_rules)
                else:
                    choices = source_choices

                row_out = {
                    "prompt": prompt,
                    "choices": list(choices),
                    "gold": int(gold_idx),
                    "variant_changed": bool(rules),
                    "variant_name": args.variant,
                    "variant_source_gold": int(gold_idx),
                    "variant_permutation": list(range(len(source_choices))),
                    "variant_rules": sorted(rules),
                    "source_dataset": KOBEST_DATASET_ID,
                    "source_config": task,
                    "source_split": split,
                    "source_index": idx,
                }
                n_changed += int(row_out["variant_changed"])
                out_rows.append(row_out)
            counts[task][split] = write_jsonl(output_root / "kobest" / task / f"{split}.jsonl", out_rows)
            changed_counts[task][split] = n_changed
    return {"tasks": tasks, "splits": splits, "counts": counts, "changed_counts": changed_counts}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build paper KoBEST surface-form stress JSONL files from public skt/kobest_v1."
    )
    parser.add_argument("--variant", required=True, choices=SUPPORTED_VARIANTS)
    parser.add_argument("--output-root", default="", help="Default: variant_benchmarks/<variant>")
    parser.add_argument("--kobest-tasks", default=",".join(DEFAULT_KOBEST_TASKS))
    parser.add_argument("--kobest-splits", default="train,validation,test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Debug limit. 0 means full split.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_paths = prepare_cache_paths(args.cache_root)
    output_root = Path(args.output_root or PROJECT_ROOT / "variant_benchmarks" / args.variant).expanduser().resolve()
    prepare_output_root(output_root, overwrite=args.overwrite)

    manifest = {
        "variant": args.variant,
        "seed": args.seed,
        "output_root": str(output_root),
        "source_datasets": {"kobest": KOBEST_DATASET_ID},
        "format": "krong_eval_variants.kobest_variant",
        "rules": {
            "RP": "Random-P whitespace drop control: delete each safe inline whitespace gap with probability p using a deterministic seed.",
            "JPH": "Hard josa-preserving compaction: preserve whitespace after surface josa and protected punctuation/template boundaries, compact other inline whitespace.",
        },
        "random_p": VARIANT_TO_RANDOM_P.get(args.variant),
        "benchmarks": {},
    }
    manifest["benchmarks"]["kobest"] = build_kobest_variant(args, output_root, cache_paths.datasets_cache)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "kobest_tasks.txt").write_text(",".join(split_csv(args.kobest_tasks) or DEFAULT_KOBEST_TASKS) + "\n", encoding="utf-8")

    print(f"[done] variant={args.variant}")
    print(f"[done] output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
