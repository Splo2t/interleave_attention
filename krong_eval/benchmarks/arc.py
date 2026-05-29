from __future__ import annotations

import random
from typing import Any, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec

ARC_DATASET_ID = "allenai/ai2_arc"
ARC_DATASET_NAMES = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
}


def _arc_doc_to_prompt(doc: dict[str, Any]) -> str:
    # lm-eval adds target_delimiter=" " after doc_to_text.
    return f"Question: {str(doc['question']).strip()}\nAnswer: "


def _arc_choices_and_gold(doc: dict[str, Any]) -> tuple[list[str], int]:
    choices_obj = doc["choices"]
    labels = [str(x) for x in choices_obj["label"]]
    choices = [str(x) for x in choices_obj["text"]]
    answer_key = str(doc["answerKey"])

    if answer_key not in labels:
        raise ValueError(f"ARC answerKey={answer_key!r} not found in labels={labels!r}")
    return choices, labels.index(answer_key)


def _build_arc_fewshot_ctx(doc: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for shot in shots:
        shot_choices, shot_gold_idx = _arc_choices_and_gold(shot)
        parts.append(_arc_doc_to_prompt(shot) + shot_choices[shot_gold_idx])
        parts.append("")
    parts.append(_arc_doc_to_prompt(doc))
    return "\n".join(parts)


def evaluate_arc(
    scorer,
    *,
    task_name: str,
    k_shot: int = 0,
    seed: int = 42,
    limit: Optional[int] = None,
    cache_dir: str | None = None,
    eval_batch_size: int = 1,
) -> dict[str, float]:
    if task_name not in ARC_DATASET_NAMES:
        raise ValueError(f"Unknown ARC task: {task_name}")

    dataset_name = ARC_DATASET_NAMES[task_name]
    train_rows: list[dict[str, Any]] = []
    if k_shot > 0:
        train_rows = list(load_dataset(ARC_DATASET_ID, name=dataset_name, split="train", cache_dir=cache_dir))
    eval_rows = list(load_dataset(ARC_DATASET_ID, name=dataset_name, split="test", cache_dir=cache_dir))

    rng = random.Random(seed)
    k = min(max(0, int(k_shot)), len(train_rows))
    shots = rng.sample(train_rows, k) if k > 0 else []

    n_eval = len(eval_rows) if limit is None else min(len(eval_rows), int(limit))
    batch_size = max(1, int(eval_batch_size))

    correct = 0
    correct_norm = 0
    total = 0

    bar = tqdm(total=n_eval, desc=f"ARC   | {task_name}", leave=True, dynamic_ncols=True)
    for start in range(0, n_eval, batch_size):
        batch_examples = eval_rows[start : min(start + batch_size, n_eval)]
        prompts: list[str] = []
        choices_batch: list[list[str]] = []
        gold_indices: list[int] = []

        for ex in batch_examples:
            try:
                choices, gold_idx = _arc_choices_and_gold(ex)
                prompts.append(_build_arc_fewshot_ctx(ex, shots))
                choices_batch.append(choices)
                gold_indices.append(gold_idx)
            except Exception as exc:
                bar.write(f"[ARC][{task_name}][SKIP] parse error: {exc}")
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
    print(f"[ARC] {task_name:14s} acc={acc * 100:5.2f}%  acc_norm={acc_norm * 100:5.2f}%  (n={total})")

    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "overall_micro": acc,
        "overall_macro": acc,
        "overall_acc_norm": acc_norm,
    }


def _run_arc(task_name: str, scorer, args) -> BenchmarkRun:
    limit = args.limit if args.limit and args.limit > 0 else None
    results = evaluate_arc(
        scorer,
        task_name=task_name,
        k_shot=args.k_shot,
        seed=args.seed,
        limit=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
    )
    return BenchmarkRun(results=results, selected_items=[task_name])


def run_arc_easy(scorer, args) -> BenchmarkRun:
    return _run_arc("arc_easy", scorer, args)


def run_arc_challenge(scorer, args) -> BenchmarkRun:
    return _run_arc("arc_challenge", scorer, args)


ARC_EASY_BENCHMARK = BenchmarkSpec(name="arc_easy", run=run_arc_easy)
ARC_CHALLENGE_BENCHMARK = BenchmarkSpec(name="arc_challenge", run=run_arc_challenge)
