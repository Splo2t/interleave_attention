from __future__ import annotations

import random
import re
from typing import Any, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec

HELLASWAG_DATASET_ID = "Rowan/hellaswag"
OPENBOOKQA_DATASET_ID = "allenai/openbookqa"
OPENBOOKQA_CONFIG = "main"

DEFAULT_SPLITS = {
    "hellaswag": "validation",
    "openbookqa": "test",
}


def _clean_hellaswag_text(text: Any) -> str:
    text = str(text or "").strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _hellaswag_text_choices_gold(doc: dict[str, Any]) -> tuple[str, list[str], int]:
    ctx = f"{doc['ctx_a']} {str(doc['ctx_b']).capitalize()}"
    prompt = _clean_hellaswag_text(f"{doc['activity_label']}: {ctx}") + " "
    choices = [_clean_hellaswag_text(choice) for choice in doc["endings"]]
    return prompt, choices, int(doc["label"])


def _openbookqa_text_choices_gold(doc: dict[str, Any]) -> tuple[str, list[str], int]:
    prompt = f"{str(doc['question_stem']).strip()} "
    choices_obj = doc["choices"]
    labels = [str(label) for label in choices_obj["label"]]
    choices = [str(choice) for choice in choices_obj["text"]]
    answer_key = str(doc["answerKey"]).lstrip()
    if answer_key not in labels:
        raise ValueError(f"OpenBookQA answerKey={answer_key!r} not found in labels={labels!r}")
    return prompt, choices, labels.index(answer_key)


def _doc_to_text_choices_gold(task_name: str, doc: dict[str, Any]) -> tuple[str, list[str], int]:
    if task_name == "hellaswag":
        return _hellaswag_text_choices_gold(doc)
    if task_name == "openbookqa":
        return _openbookqa_text_choices_gold(doc)
    raise ValueError(f"Unknown commonsense task: {task_name}")


def _load_task_dataset(
    task_name: str,
    split: str,
    *,
    cache_dir: str | None = None,
):
    if task_name == "hellaswag":
        return load_dataset(HELLASWAG_DATASET_ID, split=split, cache_dir=cache_dir)
    if task_name == "openbookqa":
        return load_dataset(OPENBOOKQA_DATASET_ID, OPENBOOKQA_CONFIG, split=split, cache_dir=cache_dir)
    raise ValueError(f"Unknown commonsense task: {task_name}")


def _build_fewshot_ctx(task_name: str, doc: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for shot in shots:
        shot_ctx, shot_choices, shot_gold_idx = _doc_to_text_choices_gold(task_name, shot)
        parts.append(shot_ctx + shot_choices[shot_gold_idx])
        parts.append("")
    ctx, _, _ = _doc_to_text_choices_gold(task_name, doc)
    parts.append(ctx)
    return "\n".join(parts)


def evaluate_commonsense_mc(
    scorer,
    *,
    task_name: str,
    k_shot: int = 0,
    seed: int = 42,
    split: str | None = None,
    limit: Optional[int] = None,
    cache_dir: str | None = None,
    eval_batch_size: int = 1,
) -> dict[str, float]:
    if task_name not in DEFAULT_SPLITS:
        raise ValueError(f"Unknown commonsense task: {task_name}")

    eval_split = (split or "").strip() or DEFAULT_SPLITS[task_name]
    train_rows: list[dict[str, Any]] = []
    if k_shot > 0:
        train_rows = list(_load_task_dataset(task_name, "train", cache_dir=cache_dir))
    eval_rows = list(_load_task_dataset(task_name, eval_split, cache_dir=cache_dir))

    rng = random.Random(seed)
    k = min(max(0, int(k_shot)), len(train_rows))

    n_eval = len(eval_rows) if limit is None else min(len(eval_rows), int(limit))
    batch_size = max(1, int(eval_batch_size))

    correct = 0
    correct_norm = 0
    total = 0

    bar = tqdm(total=n_eval, desc=f"{task_name:12s}", leave=True, dynamic_ncols=True)
    for start in range(0, n_eval, batch_size):
        batch_examples = eval_rows[start : min(start + batch_size, n_eval)]
        prompts: list[str] = []
        choices_batch: list[list[str]] = []
        gold_indices: list[int] = []

        for ex in batch_examples:
            try:
                _, choices, gold_idx = _doc_to_text_choices_gold(task_name, ex)
                shots = rng.sample(train_rows, k) if k > 0 else []
                prompts.append(_build_fewshot_ctx(task_name, ex, shots))
                choices_batch.append(choices)
                gold_indices.append(gold_idx)
            except Exception as exc:
                bar.write(f"[{task_name}][SKIP] parse error: {exc}")
                bar.update(1)

        if not prompts:
            continue

        score_dicts = scorer.score_labels_ll_and_len_batch(prompts, choices_batch)
        for choices, gold_idx, scores in zip(choices_batch, gold_indices, score_dicts):
            pred_choice = max(scores.keys(), key=lambda key: scores[key][0])
            pred_norm_choice = max(scores.keys(), key=lambda key: scores[key][0] / max(1, scores[key][1]))
            pred_idx = choices.index(pred_choice)
            pred_norm_idx = choices.index(pred_norm_choice)

            correct += int(pred_idx == gold_idx)
            correct_norm += int(pred_norm_idx == gold_idx)
            total += 1
            bar.update(1)
            bar.set_postfix(acc=f"{100.0 * correct / max(1, total):5.2f}%")

    bar.close()

    acc = correct / total if total else 0.0
    acc_norm = correct_norm / total if total else 0.0
    print(
        f"[{task_name}] split={eval_split} acc={acc * 100:5.2f}% "
        f"acc_norm={acc_norm * 100:5.2f}%  (n={total})"
    )

    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "overall_micro": acc,
        "overall_macro": acc,
        "overall_acc_norm": acc_norm,
    }


def _run_commonsense(task_name: str, scorer, args) -> BenchmarkRun:
    limit = args.limit if args.limit and args.limit > 0 else None
    split = getattr(args, "benchmark_split", "") or None
    results = evaluate_commonsense_mc(
        scorer,
        task_name=task_name,
        k_shot=args.k_shot,
        seed=args.seed,
        split=split,
        limit=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
    )
    return BenchmarkRun(results=results, selected_items=[task_name])


def run_hellaswag(scorer, args) -> BenchmarkRun:
    return _run_commonsense("hellaswag", scorer, args)


def run_openbookqa(scorer, args) -> BenchmarkRun:
    return _run_commonsense("openbookqa", scorer, args)


HELLASWAG_BENCHMARK = BenchmarkSpec(name="hellaswag", run=run_hellaswag)
OPENBOOKQA_BENCHMARK = BenchmarkSpec(name="openbookqa", run=run_openbookqa)
