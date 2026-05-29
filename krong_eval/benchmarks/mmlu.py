from __future__ import annotations

import random
from typing import List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec
from .common import choices_block_any

MMLU_DATASET_ID = "cais/mmlu"


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
        buf.append(choices_block_any(chs))
        buf.append(f"Answer: {gold}")
        if shot_blank_line:
            buf.append("")

    buf.append(question.strip())
    buf.append(choices_block_any(choices))
    buf.append("Answer:" + (" " if space_after_answer else ""))
    return "\n".join(buf)


def evaluate_mmlu(
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
    results: dict[str, float] = {}
    all_correct = 0
    all_total = 0

    if subjects is None:
        subjects = [s for s in get_dataset_config_names(MMLU_DATASET_ID) if s != "all"]
    else:
        subjects = [s.strip() for s in subjects if s and s.strip()]

    for subj in subjects:
        try:
            dev_ds = load_dataset(MMLU_DATASET_ID, name=subj, split="dev", cache_dir=cache_dir)
            test_ds = load_dataset(MMLU_DATASET_ID, name=subj, split="test", cache_dir=cache_dir)
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
        batch_size = max(1, int(eval_batch_size))

        bar = tqdm(total=n_eval, desc=f"MMLU  | {subj}", leave=True, dynamic_ncols=True)
        for start in range(0, n_eval, batch_size):
            batch_examples = [test_ds[idx] for idx in range(start, min(start + batch_size, n_eval))]
            prompts: List[str] = []
            gold_indices: List[int] = []

            for ex in batch_examples:
                question = ex["question"]
                choices = ex["choices"]
                gold_letter = _to_letter_from_mmlu_answer(ex["answer"])
                gold_indices.append("ABCD".index(gold_letter))
                prompts.append(
                    build_prompt_mmlu(
                        subj.replace("_", " "),
                        question,
                        choices,
                        shots,
                        add_header=True,
                        space_after_answer=True,
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
        results[subj] = acc
        all_correct += correct
        all_total += total
        tqdm.write(f"[MMLU ] {subj:32s}  acc={acc * 100:5.2f}%  (n={total})")

    results["overall_micro"] = (all_correct / all_total) if all_total else 0.0
    subject_accs = [results[s] for s in results.keys() if s != "overall_micro"]
    results["overall_macro"] = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    print(
        f"[MMLU] OVERALL micro={results['overall_micro'] * 100:5.2f}% "
        f"macro={results['overall_macro'] * 100:5.2f}%  (N={all_total})"
    )
    return results


def run_mmlu(scorer, args) -> BenchmarkRun:
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    results = evaluate_mmlu(
        scorer,
        k_shot=args.k_shot,
        subjects=subjects,
        seed=args.seed,
        limit_per_subject=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
    )
    return BenchmarkRun(results=results, selected_items=subjects)


MMLU_BENCHMARK = BenchmarkSpec(name="mmlu", run=run_mmlu)
