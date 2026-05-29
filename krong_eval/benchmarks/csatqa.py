from __future__ import annotations

from typing import Any, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec

CSATQA_DATASET_ID = "HAERAE-HUB/csatqa"
DEFAULT_CSATQA_SUBJECTS = ["WR", "GR", "RCS", "RCSS", "RCH", "LI"]


def _normalize_csatqa_subject(name: str) -> str:
    subject = str(name or "").strip()
    if subject.lower().startswith("csatqa_"):
        subject = subject.split("_", 1)[1]
    return subject.upper()


def _csatqa_doc_to_prompt(doc: dict[str, Any]) -> str:
    return (
        "다음을 읽고 정답으로 알맞은 것을 고르시요.\n"
        f"### Context: {str(doc['context']).strip()}\n"
        f"### Question: {str(doc['question']).strip()}\n"
        "### Options:\n"
        f"(1) {str(doc['option#1']).strip()}\n"
        f"(2) {str(doc['option#2']).strip()}\n"
        f"(3) {str(doc['option#3']).strip()}\n"
        f"(4) {str(doc['option#4']).strip()}\n"
        f"(5) {str(doc['option#5']).strip()}\n"
        "### Answer: 주어진 문제의 정답은"
    )


def _csatqa_choices() -> list[str]:
    # lm-eval's multiple-choice path scores doc_to_text + target_delimiter + target.
    # CSATQA's doc_to_text ends without a trailing space, so include the delimiter
    # in the continuation string.
    return [" (1)", " (2)", " (3)", " (4)", " (5)"]


def _csatqa_gold_idx(doc: dict[str, Any]) -> int:
    gold_idx = int(doc["gold"]) - 1
    if gold_idx < 0 or gold_idx >= len(_csatqa_choices()):
        raise ValueError(f"CSATQA gold index out of range: {doc['gold']!r}")
    return gold_idx


def _load_csatqa_subject(subject: str, *, cache_dir: str | None = None):
    # The main HAERAE-HUB/csatqa repo contains a dataset script. Newer `datasets`
    # versions reject dataset scripts, so read the auto-converted parquet files
    # directly from the Hub instead.
    parquet_url = f"hf://datasets/{CSATQA_DATASET_ID}@refs/convert/parquet/{subject}/test/0000.parquet"
    return load_dataset("parquet", data_files={"test": parquet_url}, split="test", cache_dir=cache_dir)


def evaluate_csatqa(
    scorer,
    *,
    subjects: Optional[list[str]] = None,
    limit_per_subject: Optional[int] = None,
    cache_dir: str | None = None,
    eval_batch_size: int = 1,
) -> dict[str, float]:
    if subjects is None:
        subjects = list(DEFAULT_CSATQA_SUBJECTS)
    else:
        subjects = [_normalize_csatqa_subject(subject) for subject in subjects if subject and subject.strip()]

    results: dict[str, float] = {}
    all_correct = 0
    all_norm_correct = 0
    all_total = 0

    for subject in subjects:
        try:
            rows = list(_load_csatqa_subject(subject, cache_dir=cache_dir))
        except Exception as exc:
            raise RuntimeError(f"Failed to load CSATQA subject '{subject}'") from exc

        n_eval = len(rows) if limit_per_subject is None else min(len(rows), int(limit_per_subject))
        batch_size = max(1, int(eval_batch_size))
        correct = 0
        correct_norm = 0
        total = 0

        bar = tqdm(total=n_eval, desc=f"CSATQA| {subject}", leave=True, dynamic_ncols=True)
        for start in range(0, n_eval, batch_size):
            batch_examples = rows[start : min(start + batch_size, n_eval)]
            prompts: list[str] = []
            choices_batch: list[list[str]] = []
            gold_indices: list[int] = []

            for ex in batch_examples:
                try:
                    prompts.append(_csatqa_doc_to_prompt(ex))
                    choices_batch.append(_csatqa_choices())
                    gold_indices.append(_csatqa_gold_idx(ex))
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse CSATQA row in subject '{subject}'") from exc

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
                bar.set_postfix(
                    acc=f"{100.0 * correct / max(1, total):5.2f}%",
                    acc_norm=f"{100.0 * correct_norm / max(1, total):5.2f}%",
                )

        bar.close()

        key = f"csatqa_{subject.lower()}"
        acc = correct / total if total else 0.0
        acc_norm = correct_norm / total if total else 0.0
        results[key] = acc
        results[f"{key}_acc_norm"] = acc_norm
        all_correct += correct
        all_norm_correct += correct_norm
        all_total += total
        print(f"[CSATQA] {subject:4s} acc={acc * 100:5.2f}%  acc_norm={acc_norm * 100:5.2f}%  (n={total})")

    results["overall_micro"] = all_correct / all_total if all_total else 0.0
    subject_accs = [results[f"csatqa_{subject.lower()}"] for subject in subjects if f"csatqa_{subject.lower()}" in results]
    results["overall_macro"] = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    results["overall_acc_norm"] = all_norm_correct / all_total if all_total else 0.0
    results["acc_norm"] = results["overall_acc_norm"]
    print(
        f"[CSATQA] OVERALL micro={results['overall_micro'] * 100:5.2f}% "
        f"macro={results['overall_macro'] * 100:5.2f}% "
        f"acc_norm={results['overall_acc_norm'] * 100:5.2f}%  (N={all_total})"
    )
    return results


def run_csatqa(scorer, args) -> BenchmarkRun:
    subjects = [_normalize_csatqa_subject(s) for s in args.subjects.split(",") if s.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    selected_items = subjects or list(DEFAULT_CSATQA_SUBJECTS)
    if getattr(args, "k_shot", 0):
        print(
            "[CSATQA] k_shot is ignored because the public converted dataset "
            "only provides test split parquet files. Evaluating 0-shot."
        )
    results = evaluate_csatqa(
        scorer,
        subjects=subjects,
        limit_per_subject=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
    )
    return BenchmarkRun(results=results, selected_items=selected_items)


CSATQA_BENCHMARK = BenchmarkSpec(name="csatqa", run=run_csatqa)
