from __future__ import annotations

import random
from typing import Any, Callable, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec
from .common import letters

CLICK_DATASET_ID = "EunsuKim/CLIcK"
CLICK_SPLIT = "train"
DEFAULT_CLICK_SUBJECTS = [
    "KL_Grammar",
    "KL_Textual",
    "KL_Functional",
    "KC_Law",
    "KC_Popular",
    "KC_Politics",
    "KC_Geography",
    "KC_Economy",
    "KC_History",
    "KC_Society",
    "KC_Tradition",
]
CLICK_LANGUAGE_SUBJECTS = ["KL_Grammar", "KL_Textual", "KL_Functional"]
CLICK_CULTURE_SUBJECTS = [
    "KC_Law",
    "KC_Popular",
    "KC_Politics",
    "KC_Geography",
    "KC_Economy",
    "KC_History",
    "KC_Society",
    "KC_Tradition",
]

CLICK_LMEVAL_TASK_ALIASES = {
    "click_lang_grammar": "KL_Grammar",
    "click_lang_text": "KL_Textual",
    "click_lang_function": "KL_Functional",
    "click_cul_law": "KC_Law",
    "click_cul_kpop": "KC_Popular",
    "click_cul_politics": "KC_Politics",
    "click_cul_geography": "KC_Geography",
    "click_cul_economy": "KC_Economy",
    "click_cul_history": "KC_History",
    "click_cul_society": "KC_Society",
    "click_cul_tradition": "KC_Tradition",
}


def _norm_key(value: str) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _click_subject_aliases() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {
        "all": list(DEFAULT_CLICK_SUBJECTS),
        "language": list(CLICK_LANGUAGE_SUBJECTS),
        "lang": list(CLICK_LANGUAGE_SUBJECTS),
        "clicklang": list(CLICK_LANGUAGE_SUBJECTS),
        "kl": list(CLICK_LANGUAGE_SUBJECTS),
        "culture": list(CLICK_CULTURE_SUBJECTS),
        "cul": list(CLICK_CULTURE_SUBJECTS),
        "clickcul": list(CLICK_CULTURE_SUBJECTS),
        "kc": list(CLICK_CULTURE_SUBJECTS),
    }
    for subject in DEFAULT_CLICK_SUBJECTS:
        aliases[_norm_key(subject)] = [subject]
        aliases[_norm_key(subject.replace("KL_", "").replace("KC_", ""))] = [subject]
    for lm_eval_name, subject in CLICK_LMEVAL_TASK_ALIASES.items():
        aliases[_norm_key(lm_eval_name)] = [subject]
    return aliases


def _normalize_click_subjects(subjects: Optional[list[str]]) -> list[str]:
    if not subjects:
        return list(DEFAULT_CLICK_SUBJECTS)
    aliases = _click_subject_aliases()
    out: list[str] = []
    seen: set[str] = set()
    for raw in subjects:
        key = _norm_key(raw)
        expanded = aliases.get(key, [raw])
        for subject in expanded:
            if subject not in seen:
                seen.add(subject)
                out.append(subject)
    return out


def _click_id(row: dict[str, Any]) -> str:
    return str(row.get("id", ""))


def _click_id_int_part(row: dict[str, Any], index: int) -> int:
    parts = _click_id(row).split("_")
    return int(parts[index])


def _contains_dialogue_cue(row: dict[str, Any]) -> bool:
    question = str(row.get("question", ""))
    return "대화" in question or "발화" in question or "질의" in question


def _is_text(row: dict[str, Any]) -> bool:
    row_id = _click_id(row)
    return (
        "CSAT_korean_22" in row_id
        or ("CSAT_korean_23" in row_id and _click_id_int_part(row, -1) < 35)
        or ("TK" in row_id and _click_id_int_part(row, -1) > 4)
    )


def _is_grammar(row: dict[str, Any]) -> bool:
    row_id = _click_id(row)
    return (
        (
            "CSAT_korean" in row_id
            and _click_id_int_part(row, 2) < 21
            and _click_id_int_part(row, 3) > 10
        )
        or (
            "Kedu_1" in row_id
            and (
                row_id.split("_")[1] != "16"
                or not _contains_dialogue_cue(row)
            )
        )
        or ("TK" in row_id and _click_id_int_part(row, -1) < 5)
    )


def _is_functional(row: dict[str, Any]) -> bool:
    row_id = _click_id(row)
    return (
        (
            "CSAT_korean" in row_id
            and (
                _click_id_int_part(row, -1) > 34
                or (_click_id_int_part(row, 2) < 21 and _click_id_int_part(row, 3) < 11)
            )
        )
        or ("Kedu_16" in row_id and _contains_dialogue_cue(row))
        or "PSE_korean" in row_id
    )


def _id_contains(token: str) -> Callable[[dict[str, Any]], bool]:
    def _predicate(row: dict[str, Any]) -> bool:
        return token in _click_id(row).lower()

    return _predicate


CLICK_SUBJECT_FILTERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "KL_Grammar": _is_grammar,
    "KL_Textual": _is_text,
    "KL_Functional": _is_functional,
    "KC_Law": lambda row: "law" in _click_id(row).lower() or "PSAT" in _click_id(row),
    "KC_Popular": _id_contains("popular"),
    "KC_Politics": _id_contains("politics"),
    "KC_Geography": _id_contains("geography"),
    "KC_Economy": _id_contains("economy"),
    "KC_History": lambda row: "KHB" in _click_id(row) or "history" in _click_id(row).lower(),
    "KC_Society": _id_contains("society"),
    "KC_Tradition": _id_contains("tradition"),
}


def _load_click_all(cache_dir: str | None = None) -> list[dict[str, Any]]:
    return list(load_dataset(CLICK_DATASET_ID, split=CLICK_SPLIT, cache_dir=cache_dir))


def _filter_click_subject(rows: list[dict[str, Any]], subject: str) -> list[dict[str, Any]]:
    if subject not in CLICK_SUBJECT_FILTERS:
        raise ValueError(f"Unknown CLIcK subject: {subject}")
    predicate = CLICK_SUBJECT_FILTERS[subject]
    return [row for row in rows if predicate(row)]


def _load_click_subject(subject: str, cache_dir: str | None = None) -> list[dict[str, Any]]:
    return _filter_click_subject(_load_click_all(cache_dir=cache_dir), subject)


def _click_labels(row: dict[str, Any]) -> list[str]:
    choices = list(row.get("choices", []))
    if "CSAT" in _click_id(row) and len(choices) >= 5:
        return ["A", "B", "C", "D", "E"]
    return letters(min(4, len(choices)))


def _click_prompt(row: dict[str, Any]) -> str:
    paragraph = str(row.get("paragraph", "") or "").strip()
    question = str(row["question"]).strip()
    choices = [str(choice) for choice in row["choices"]]
    if len(choices) < 4:
        raise ValueError(f"CLIcK row has fewer than four choices: {_click_id(row)!r}")

    options = f"A:{choices[0]}, B: {choices[1]}, C: {choices[2]}, D: {choices[3]}"
    if paragraph:
        return (
            "주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 "
            "알파벳 하나로 답하시오.\n\n"
            f"맥락: {paragraph}\n"
            f"질문: {question}\n"
            f"보기:\n{options}\n"
            "정답:"
        )
    return (
        "주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 "
        "알파벳 하나로 답하시오.\n\n"
        f"질문: {question}\n"
        f"보기:\n{options}\n"
        "정답:"
    )


def _click_gold_idx(row: dict[str, Any]) -> int:
    answer = str(row["answer"])
    choices = [str(choice) for choice in row["choices"]]
    if answer not in choices:
        raise ValueError(f"CLIcK answer is not in choices: {answer!r}")
    gold_idx = choices.index(answer)
    labels = _click_labels(row)
    if gold_idx >= len(labels):
        raise ValueError(f"CLIcK gold index has no label: id={_click_id(row)!r} gold_idx={gold_idx}")
    return gold_idx


def _build_click_fewshot_prompt(row: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for shot in shots:
        shot_labels = _click_labels(shot)
        parts.append(_click_prompt(shot) + shot_labels[_click_gold_idx(shot)])
        parts.append("")
    parts.append(_click_prompt(row))
    return "\n".join(parts)


def evaluate_click(
    scorer,
    *,
    subjects: Optional[list[str]] = None,
    k_shot: int = 5,
    seed: int = 42,
    limit_per_subject: Optional[int] = None,
    cache_dir: str | None = None,
    eval_batch_size: int = 1,
) -> dict[str, float]:
    selected_subjects = _normalize_click_subjects(subjects)
    results: dict[str, float] = {}
    subject_correct: dict[str, int] = {}
    subject_correct_norm: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    all_correct = 0
    all_correct_norm = 0
    all_total = 0
    all_rows = _load_click_all(cache_dir=cache_dir)

    for subject in selected_subjects:
        try:
            rows = _filter_click_subject(all_rows, subject)
        except Exception as exc:
            raise RuntimeError(f"Failed to load CLIcK subject '{subject}'") from exc

        n_eval = len(rows) if limit_per_subject is None else min(len(rows), int(limit_per_subject))
        batch_size = max(1, int(eval_batch_size))
        k = max(0, min(int(k_shot), max(0, len(rows) - 1)))
        rng = random.Random(f"{seed}:{subject}")
        correct = 0
        correct_norm = 0
        total = 0

        bar = tqdm(total=n_eval, desc=f"CLIcK | {subject}", leave=True, dynamic_ncols=True)
        for start in range(0, n_eval, batch_size):
            batch_rows = rows[start : min(start + batch_size, n_eval)]
            prompts: list[str] = []
            labels_batch: list[list[str]] = []
            gold_indices: list[int] = []

            for row in batch_rows:
                try:
                    labels = _click_labels(row)
                    row_id = row.get("id")
                    if k > 0:
                        sampled = rng.sample(rows, k + 1)
                        shots = [shot for shot in sampled if shot is not row and shot.get("id") != row_id][:k]
                        if len(shots) < k:
                            shots = [shot for shot in rows if shot.get("id") != row_id][:k]
                    else:
                        shots = []
                    prompts.append(_build_click_fewshot_prompt(row, shots))
                    labels_batch.append(labels)
                    gold_indices.append(_click_gold_idx(row))
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse CLIcK row in subject '{subject}'") from exc

            if not prompts:
                continue

            score_dicts = scorer.score_labels_ll_and_len_batch(prompts, labels_batch)
            for labels, gold_idx, scores in zip(labels_batch, gold_indices, score_dicts):
                pred = max(scores.keys(), key=lambda key: scores[key][0])
                pred_norm = max(scores.keys(), key=lambda key: scores[key][0] / max(1, scores[key][1]))
                pred_idx = labels.index(pred)
                pred_norm_idx = labels.index(pred_norm)
                correct += int(pred_idx == gold_idx)
                correct_norm += int(pred_norm_idx == gold_idx)
                total += 1
                bar.update(1)
                bar.set_postfix(
                    acc=f"{100.0 * correct / max(1, total):5.2f}%",
                    acc_norm=f"{100.0 * correct_norm / max(1, total):5.2f}%",
                )
        bar.close()

        acc = correct / total if total else 0.0
        acc_norm = correct_norm / total if total else 0.0
        results[f"click_{subject}"] = acc
        results[f"click_{subject}_acc_norm"] = acc_norm
        subject_correct[subject] = correct
        subject_correct_norm[subject] = correct_norm
        subject_total[subject] = total
        all_correct += correct
        all_correct_norm += correct_norm
        all_total += total
        print(f"[CLIcK] {subject:16s} acc={acc * 100:5.2f}%  acc_norm={acc_norm * 100:5.2f}%  (n={total})")

    def _aggregate(subjects_to_use: list[str], correct_by_subject: dict[str, int]) -> float:
        total = sum(subject_total.get(subject, 0) for subject in subjects_to_use)
        if total == 0:
            return 0.0
        correct = sum(correct_by_subject.get(subject, 0) for subject in subjects_to_use)
        return correct / total

    results["click_language"] = _aggregate(CLICK_LANGUAGE_SUBJECTS, subject_correct)
    results["click_culture"] = _aggregate(CLICK_CULTURE_SUBJECTS, subject_correct)
    results["click_language_acc_norm"] = _aggregate(CLICK_LANGUAGE_SUBJECTS, subject_correct_norm)
    results["click_culture_acc_norm"] = _aggregate(CLICK_CULTURE_SUBJECTS, subject_correct_norm)
    results["overall_micro"] = all_correct / all_total if all_total else 0.0
    results["overall_acc_norm"] = all_correct_norm / all_total if all_total else 0.0
    results["acc_norm"] = results["overall_acc_norm"]
    subject_accs = [results[f"click_{subject}"] for subject in selected_subjects if f"click_{subject}" in results]
    subject_acc_norms = [
        results[f"click_{subject}_acc_norm"] for subject in selected_subjects if f"click_{subject}_acc_norm" in results
    ]
    results["overall_macro"] = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    results["overall_macro_acc_norm"] = sum(subject_acc_norms) / len(subject_acc_norms) if subject_acc_norms else 0.0
    print(
        f"[CLIcK] OVERALL micro={results['overall_micro'] * 100:5.2f}% "
        f"acc_norm={results['overall_acc_norm'] * 100:5.2f}% "
        f"macro={results['overall_macro'] * 100:5.2f}% "
        f"language={results['click_language'] * 100:5.2f}% "
        f"culture={results['click_culture'] * 100:5.2f}%  (N={all_total})"
    )
    return results


def run_click(scorer, args) -> BenchmarkRun:
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    limit = args.limit if args.limit and args.limit > 0 else None
    selected_subjects = _normalize_click_subjects(subjects)
    results = evaluate_click(
        scorer,
        subjects=subjects,
        k_shot=args.k_shot,
        seed=args.seed,
        limit_per_subject=limit,
        cache_dir=getattr(args, "datasets_cache", None),
        eval_batch_size=args.eval_batch_size,
    )
    return BenchmarkRun(results=results, selected_items=selected_subjects)


CLICK_BENCHMARK = BenchmarkSpec(name="click", run=run_click)
