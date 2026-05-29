from __future__ import annotations

import random
from typing import Any, Optional

from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec
from .constants import DEFAULT_KOBEST_TASKS
from .kobest import _macro_f1_score
from .variant_io import get_variant_data_root, read_jsonl


def _load_kobest_variant_split(root, task: str, split: str) -> list[dict]:
    return read_jsonl(root / "kobest" / task / f"{split}.jsonl")


def _row_to_text_choices_gold(row: dict) -> tuple[str, list[str], int]:
    prompt = str(row["prompt"])
    choices = [str(choice) for choice in row["choices"]]
    gold_idx = int(row["gold"])
    if not 0 <= gold_idx < len(choices):
        raise ValueError(f"gold index out of range: {gold_idx} for {len(choices)} choices")
    return prompt, choices, gold_idx


def _build_variant_fewshot_ctx(doc: dict, shots: list[dict]) -> str:
    parts: list[str] = []
    for shot in shots:
        s_ctx, s_choices, s_gold = _row_to_text_choices_gold(shot)
        parts.append(s_ctx + s_choices[s_gold])
        parts.append("")
    ctx, _, _ = _row_to_text_choices_gold(doc)
    parts.append(ctx)
    return "\n".join(parts)


def evaluate_kobest_variant(
    scorer,
    *,
    variant_root,
    tasks: Optional[list[str]] = None,
    k_shot: int = 0,
    split: str = "test",
    seed: int = 42,
    limit_per_task: Optional[int] = None,
    eval_batch_size: int = 1,
    prediction_records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, float]:
    if tasks is None:
        tasks = list(DEFAULT_KOBEST_TASKS)

    rng = random.Random(seed)
    results: dict[str, float] = {}
    all_correct = 0
    all_total = 0
    all_f1_weighted_sum = 0.0
    all_f1_weighted_total = 0
    all_norm_correct = 0
    all_norm_total = 0
    print(f"[KoBEST_VARIANT] data_root={variant_root}")

    for task in tasks:
        try:
            eval_rows = _load_kobest_variant_split(variant_root, task, split)
        except Exception as exc:
            print(f"[KoBEST_VARIANT][{task}] dataset load error: {exc}")
            continue
        if k_shot > 0:
            try:
                train_rows = _load_kobest_variant_split(variant_root, task, "train")
            except Exception as exc:
                print(f"[KoBEST_VARIANT][{task}] train split load error for k_shot={k_shot}: {exc}")
                continue
        else:
            train_rows = []

        n_eval = len(eval_rows) if limit_per_task is None else min(len(eval_rows), int(limit_per_task))
        correct = 0
        correct_norm = 0
        total = 0
        batch_size = max(1, int(eval_batch_size))
        golds_seen: list[int] = []
        preds_seen: list[int] = []

        bar = tqdm(total=n_eval, desc=f"KoBEST_VARIANT | {task}", dynamic_ncols=True)
        for start in range(0, n_eval, batch_size):
            batch_examples = eval_rows[start : min(start + batch_size, n_eval)]
            prompts: list[str] = []
            choices_batch: list[list[str]] = []
            gold_indices: list[int] = []
            item_indices: list[int] = []

            for offset, row in enumerate(batch_examples):
                row_index = start + offset
                try:
                    shots = rng.sample(train_rows, min(k_shot, len(train_rows))) if k_shot > 0 else []
                    ctx = _build_variant_fewshot_ctx(row, shots)
                    _, choices, gold_idx = _row_to_text_choices_gold(row)
                except Exception as exc:
                    bar.write(f"[KoBEST_VARIANT][{task}][SKIP] parse error: {exc}")
                    bar.update(1)
                    continue

                prompts.append(ctx)
                choices_batch.append(choices)
                gold_indices.append(gold_idx)
                item_indices.append(int(row.get("source_index", row_index)))

            if not prompts:
                continue

            score_dicts = scorer.score_labels_ll_and_len_batch(prompts, choices_batch)
            for choices, gold_idx, item_index, scores in zip(choices_batch, gold_indices, item_indices, score_dicts):
                pred_choice = max(scores.keys(), key=lambda key: scores[key][0])
                pred_idx = choices.index(pred_choice)
                pred_norm_choice = max(scores.keys(), key=lambda key: scores[key][0] / max(1, scores[key][1]))
                pred_norm_idx = choices.index(pred_norm_choice)

                correct += int(pred_idx == gold_idx)
                if task == "hellaswag":
                    correct_norm += int(pred_norm_idx == gold_idx)
                golds_seen.append(gold_idx)
                preds_seen.append(pred_idx)
                if prediction_records is not None:
                    prediction_records.append(
                        {
                            "benchmark": "kobest",
                            "task": task,
                            "item_index": item_index,
                            "item_id": f"kobest:{task}:{item_index}",
                            "gold_idx": gold_idx,
                            "gold_label": choices[gold_idx],
                            "pred_idx": pred_idx,
                            "pred_label": pred_choice,
                            "pred_norm_idx": pred_norm_idx,
                            "pred_norm_label": pred_norm_choice,
                            "correct": pred_idx == gold_idx,
                            "correct_norm": pred_norm_idx == gold_idx,
                        }
                    )
                total += 1
                bar.update(1)
                bar.set_postfix(acc=f"{100.0 * correct / max(1, total):5.2f}%")

        bar.close()

        acc = (correct / total) if total else 0.0
        f1 = _macro_f1_score(golds_seen, preds_seen)
        results[task] = acc
        results[f"{task}_f1"] = f1
        all_correct += correct
        all_total += total
        all_f1_weighted_sum += f1 * total
        all_f1_weighted_total += total

        line = f"[KoBEST_VARIANT] {task:12s} acc={acc * 100:5.2f}%  f1={f1 * 100:5.2f}%"
        if task == "hellaswag":
            acc_norm = (correct_norm / total) if total else 0.0
            results[f"{task}_acc_norm"] = acc_norm
            all_norm_correct += correct_norm
            all_norm_total += total
            line += f"  acc_norm={acc_norm * 100:5.2f}%"
        print(f"{line}  (n={total})")

    results["overall_micro"] = (all_correct / all_total) if all_total else 0.0
    task_accs = [results[task] for task in tasks if task in results]
    results["overall_macro"] = (sum(task_accs) / len(task_accs)) if task_accs else 0.0
    results["overall_f1"] = (all_f1_weighted_sum / all_f1_weighted_total) if all_f1_weighted_total else 0.0
    if all_norm_total:
        results["overall_acc_norm"] = all_norm_correct / all_norm_total
    print(
        f"[KoBEST_VARIANT] OVERALL micro={results['overall_micro'] * 100:5.2f}% "
        f"macro={results['overall_macro'] * 100:5.2f}% "
        f"f1={results['overall_f1'] * 100:5.2f}%"
        + (f" acc_norm={results['overall_acc_norm'] * 100:5.2f}%" if "overall_acc_norm" in results else "")
        + f"  (N={all_total})"
    )
    return results


def run_kobest_variant(scorer, args) -> BenchmarkRun:
    tasks = [t.strip() for t in args.kobest_tasks.split(",") if t.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    root = get_variant_data_root(args)
    selected_items = tasks or list(DEFAULT_KOBEST_TASKS)
    prediction_records: list[dict[str, Any]] | None = [] if getattr(args, "save_item_predictions", False) else None
    results = evaluate_kobest_variant(
        scorer,
        variant_root=root,
        tasks=tasks,
        k_shot=args.k_shot,
        split=args.kobest_split,
        seed=args.seed,
        limit_per_task=limit,
        eval_batch_size=args.eval_batch_size,
        prediction_records=prediction_records,
    )
    return BenchmarkRun(results=results, selected_items=selected_items, item_predictions=prediction_records)


KOBEST_VARIANT_BENCHMARK = BenchmarkSpec(name="kobest_variant", run=run_kobest_variant)
