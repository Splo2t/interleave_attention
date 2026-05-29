from __future__ import annotations

import random
from typing import Any
from typing import Optional, Tuple

from datasets import load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec
from .constants import DEFAULT_KOBEST_TASKS

KOBEST_DATASET_ID = "skt/kobest_v1"


def kobest_boolq_doc_to_text(doc: dict) -> str:
    return f"""{doc["paragraph"]} 질문: {doc["question"]} 답변: """


def kobest_boolq_choices() -> list[str]:
    return ["아니오", "예"]


def kobest_boolq_gold_idx(doc: dict) -> int:
    return int(doc["label"])


def copa_doc_to_text(doc: dict) -> str:
    connector = {"원인": " 왜냐하면", "결과": " 그래서"}[doc["question"].strip()]
    return f"""{doc["premise"]} {connector}"""


def copa_doc_to_choice(doc: dict) -> list[str]:
    return [f"""{doc["alternative_1"]}""", f"""{doc["alternative_2"]}"""]


def sentineg_doc_to_text(doc: dict) -> str:
    return f"""문장: {doc["sentence"]} 긍부정:"""


def wic_doc_to_text(doc: dict) -> str:
    return (
        f"""문장1: {doc["context_1"]} 문장2: {doc["context_2"]} """
        f"""두 문장에서 {doc["word"]}가 같은 뜻으로 쓰였나?"""
    )


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


def _macro_f1_score(golds: list[int], preds: list[int]) -> float:
    labels = sorted(set(golds) | set(preds))
    if not labels:
        return 0.0

    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def _kobest_get_dataset(task: str, split: str, cache_dir: str | None = None):
    ds = load_dataset(KOBEST_DATASET_ID, task, split=split, cache_dir=cache_dir)
    if task == "hellaswag":
        ds = hellaswag_process_doc(ds)
    return ds


def _kobest_doc_to_text_and_choices(task: str, doc: dict) -> Tuple[str, list[str], int]:
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


def _build_mc_fewshot_ctx(task: str, doc: dict, shots: list[dict]) -> str:
    parts: list[str] = []
    for shot in shots:
        s_ctx, s_choices, s_gold = _kobest_doc_to_text_and_choices(task, shot)
        parts.append(s_ctx + s_choices[s_gold])
        parts.append("")
    ctx, _, _ = _kobest_doc_to_text_and_choices(task, doc)
    parts.append(ctx)
    return "\n".join(parts)


def evaluate_kobest(
    scorer,
    *,
    tasks: Optional[list[str]] = None,
    k_shot: int = 0,
    split: str = "test",
    limit_per_task: Optional[int] = None,
    cache_dir: str | None = None,
    eval_batch_size: int = 1,
    prediction_records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, float]:
    if tasks is None:
        tasks = list(DEFAULT_KOBEST_TASKS)
    rng = random.Random(42)

    results: dict[str, float] = {}
    all_correct = 0
    all_total = 0
    all_f1_weighted_sum = 0.0
    all_f1_weighted_total = 0
    all_norm_correct = 0
    all_norm_total = 0

    for task in tasks:
        try:
            eval_rows = list(_kobest_get_dataset(task, split, cache_dir=cache_dir))
        except Exception as e:
            print(f"[KoBEST][{task}] dataset load error: {e}")
            continue
        if k_shot > 0:
            try:
                train_rows = list(_kobest_get_dataset(task, "train", cache_dir=cache_dir))
            except Exception as e:
                print(f"[KoBEST][{task}] train split load error for k_shot={k_shot}: {e}")
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

        bar = tqdm(total=n_eval, desc=f"KoBEST | {task}", dynamic_ncols=True)
        for start in range(0, n_eval, batch_size):
            batch_examples = eval_rows[start : min(start + batch_size, n_eval)]
            prompts: list[str] = []
            choices_batch: list[list[str]] = []
            gold_indices: list[int] = []
            item_indices: list[int] = []

            for offset, ex in enumerate(batch_examples):
                row_index = start + offset
                try:
                    shots = rng.sample(train_rows, k_shot) if k_shot > 0 else []
                    ctx = _build_mc_fewshot_ctx(task, ex, shots)
                    _, choices, gold_idx = _kobest_doc_to_text_and_choices(task, ex)
                except Exception as e:
                    bar.write(f"[KoBEST][{task}][SKIP] parse error: {e}")
                    bar.update(1)
                    continue

                prompts.append(ctx)
                choices_batch.append(choices)
                gold_indices.append(gold_idx)
                item_indices.append(int(ex.get("source_index", row_index)) if isinstance(ex, dict) else row_index)

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

        line = f"[KoBEST] {task:12s} acc={acc * 100:5.2f}%  f1={f1 * 100:5.2f}%"
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
        f"[KoBEST] OVERALL micro={results['overall_micro'] * 100:5.2f}% "
        f"macro={results['overall_macro'] * 100:5.2f}% "
        f"f1={results['overall_f1'] * 100:5.2f}%"
        + (f" acc_norm={results['overall_acc_norm'] * 100:5.2f}%" if "overall_acc_norm" in results else "")
        + f"  (N={all_total})"
    )
    return results


def run_kobest(scorer, args) -> BenchmarkRun:
    tasks = [t.strip() for t in args.kobest_tasks.split(",") if t.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    selected_items = tasks or list(DEFAULT_KOBEST_TASKS)
    prediction_records: list[dict[str, Any]] | None = [] if getattr(args, "save_item_predictions", False) else None
    results = evaluate_kobest(
        scorer,
        tasks=tasks,
        k_shot=args.k_shot,
        split=args.kobest_split,
        limit_per_task=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
        prediction_records=prediction_records,
    )
    return BenchmarkRun(results=results, selected_items=selected_items, item_predictions=prediction_records)


KOBEST_BENCHMARK = BenchmarkSpec(name="kobest", run=run_kobest)
