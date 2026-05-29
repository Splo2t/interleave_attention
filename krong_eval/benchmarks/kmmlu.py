from __future__ import annotations

import random
from typing import List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec
from .common import choices_block_any

KMMLU_DATASET_ID = "HAERAE-HUB/KMMLU"


def build_prompt_kmmlu(
    category: str,
    question: str,
    choices: List[str],
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
    del category

    colon = "：" if use_fullwidth_colon else ":"
    mark = f"정답{colon}" + (" " if space_after_colon else "")

    buf: List[str] = []
    for qs, chs, gold in shots:
        buf.append(qs.strip())
        buf.append(choices_block_any(chs))
        buf.append(f"{mark}{gold}")
        if shot_blank_line:
            buf.append("")

    buf.append(question.strip())
    buf.append(choices_block_any(choices))
    buf.append(mark)
    return "\n".join(buf)


def evaluate_kmmlu(
    scorer,
    *,
    k_shot: int = 5,
    subjects: Optional[List[str]] = None,
    seed: int = 42,
    limit_per_subject: Optional[int] = None,
    cache_dir: str | None = None,
    eval_batch_size: int = 1,
) -> dict[str, float]:
    random.seed(seed)
    if subjects is None:
        subjects = get_dataset_config_names(KMMLU_DATASET_ID)
    subjects = sorted({s.strip() for s in subjects if s and s.strip()})

    results: dict[str, float] = {}
    all_correct = 0
    all_total = 0

    def _ans_to_idx(answer: int) -> int:
        return int(answer) - 1

    for cat in subjects:
        try:
            ds_test = load_dataset(KMMLU_DATASET_ID, name=cat, split="test", cache_dir=cache_dir)
            ds_dev = load_dataset(KMMLU_DATASET_ID, name=cat, split="dev", cache_dir=cache_dir)
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
        batch_size = max(1, int(eval_batch_size))

        bar = tqdm(total=n_eval, desc=f"KMMLU | {cat}", leave=True, dynamic_ncols=True)
        for start in range(0, n_eval, batch_size):
            batch_examples = rows[start : min(start + batch_size, n_eval)]
            prompts: List[str] = []
            gold_indices: List[int] = []

            for ex in batch_examples:
                q = ex["question"]
                ch = [ex["A"], ex["B"], ex["C"], ex["D"]]
                gold_indices.append(_ans_to_idx(ex["answer"]))
                prompts.append(
                    build_prompt_kmmlu(
                        cat,
                        q,
                        ch,
                        shots,
                        use_fullwidth_colon=True,
                        space_after_colon=False,
                        shot_blank_line=True,
                    )
                )

            score_dicts = scorer.score_labels_ll_and_len_batch(prompts, [["A", "B", "C", "D"] for _ in prompts])
            for scores, gold_idx in zip(score_dicts, gold_indices):
                pred_letter = max(scores.keys(), key=lambda key: scores[key][0])
                pred_idx = "ABCD".index(pred_letter)
                correct += int(pred_idx == gold_idx)
                total += 1
                bar.update(1)
                bar.set_postfix(acc=f"{100.0 * correct / max(1, total):5.2f}%")

        bar.close()

        acc = (correct / total) if total else 0.0
        results[cat] = acc
        all_correct += correct
        all_total += total
        print(f"[KMMLU] {cat:32s}  acc={acc * 100:5.2f}%  (n={total})")

    results["overall_micro"] = (all_correct / all_total) if all_total else 0.0
    subject_accs = [results[c] for c in results.keys() if c != "overall_micro"]
    results["overall_macro"] = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    print(
        f"[KMMLU] OVERALL micro={results['overall_micro'] * 100:5.2f}% "
        f"macro={results['overall_macro'] * 100:5.2f}%  (N={all_total})"
    )
    return results


def run_kmmlu(scorer, args) -> BenchmarkRun:
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    results = evaluate_kmmlu(
        scorer,
        k_shot=args.k_shot,
        subjects=subjects,
        seed=args.seed,
        limit_per_subject=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
    )
    return BenchmarkRun(results=results, selected_items=subjects)


KMMLU_BENCHMARK = BenchmarkSpec(name="kmmlu", run=run_kmmlu)
