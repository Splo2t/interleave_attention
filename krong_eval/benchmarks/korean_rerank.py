from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from tqdm.auto import tqdm

from .base import BenchmarkRun, BenchmarkSpec


DEFAULT_RERANK_PROMPT_TEMPLATE = (
    "질문: {query}\n\n"
    "문서: {passage}\n\n"
    "이 문서는 질문에 답하는 데 관련이 있다:"
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _first_present(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _candidate_text(candidate: dict[str, Any]) -> str:
    title = str(_first_present(candidate, ["title", "doc_title"], "") or "").strip()
    text = str(_first_present(candidate, ["text", "passage", "contents", "content", "document"], "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


def _candidate_id(candidate: dict[str, Any], fallback: int) -> str:
    return str(_first_present(candidate, ["doc_id", "document_id", "pid", "id"], f"candidate-{fallback}"))


def _candidate_rel(candidate: dict[str, Any]) -> float:
    return _as_float(_first_present(candidate, ["relevance", "rel", "label", "score_label", "qrel"], 0.0))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_flat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(_first_present(row, ["query_id", "qid", "id"], ""))
        if not query_id:
            raise ValueError("Flat rerank row must contain query_id/qid/id.")
        query = str(_first_present(row, ["query", "question"], ""))
        if query_id not in grouped:
            grouped[query_id] = {"query_id": query_id, "query": query, "candidates": []}
        candidate = {
            "doc_id": _first_present(row, ["doc_id", "document_id", "pid"], f"{query_id}:{len(grouped[query_id]['candidates'])}"),
            "title": _first_present(row, ["title", "doc_title"], ""),
            "text": _first_present(row, ["text", "passage", "contents", "content", "document"], ""),
            "relevance": _first_present(row, ["relevance", "rel", "label", "qrel"], 0),
            "retriever_rank": _first_present(row, ["rank", "retriever_rank", "bm25_rank"], None),
            "retriever_score": _first_present(row, ["retriever_score", "bm25_score", "score"], None),
        }
        grouped[query_id]["candidates"].append(candidate)
    return list(grouped.values())


def _load_rerank_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    if not rows:
        return []
    if "candidates" in rows[0]:
        return rows
    return _group_flat_rows(rows)


def _dcg(rels: list[float], k: int) -> float:
    total = 0.0
    for idx, rel in enumerate(rels[:k], start=1):
        gain = (2.0**rel) - 1.0
        total += gain / math.log2(idx + 1)
    return total


def _ndcg_at(ranked_rels: list[float], ideal_rels: list[float], k: int) -> float:
    ideal = _dcg(sorted(ideal_rels, reverse=True), k)
    if ideal <= 0:
        return 0.0
    return _dcg(ranked_rels, k) / ideal


def _mrr_at(ranked_rels: list[float], k: int) -> float:
    for idx, rel in enumerate(ranked_rels[:k], start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def _recall_at(ranked_rels: list[float], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for rel in ranked_rels[:k] if rel > 0)
    return found / total_relevant


def _format_prompt(template: str, *, query: str, passage: str) -> str:
    return template.format(query=query, passage=passage)


def _iter_fewshot_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        query_id = str(_first_present(row, ["query_id", "qid", "id"], f"query-{row_idx}"))
        query = str(_first_present(row, ["query", "question"], ""))
        if not query:
            continue
        for cand_idx, candidate in enumerate(list(row.get("candidates") or [])):
            passage = _candidate_text(candidate)
            if not passage:
                continue
            examples.append(
                {
                    "query_id": query_id,
                    "doc_id": _candidate_id(candidate, cand_idx),
                    "query": query,
                    "passage": passage,
                    "relevance": _candidate_rel(candidate),
                }
            )
    return examples


def _select_fewshot_examples(
    examples: list[dict[str, Any]],
    *,
    num_examples: int,
    exclude_query_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    if num_examples <= 0:
        return []

    positives = [ex for ex in examples if ex["query_id"] != exclude_query_id and float(ex["relevance"]) > 0]
    negatives = [ex for ex in examples if ex["query_id"] != exclude_query_id and float(ex["relevance"]) <= 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Keep demonstrations label-balanced when possible; odd k gets one extra positive.
    target_pos = (num_examples + 1) // 2
    target_neg = num_examples // 2
    pos = positives[:target_pos]
    neg = negatives[:target_neg]

    if len(pos) < target_pos:
        neg.extend(negatives[target_neg : target_neg + (target_pos - len(pos))])
    if len(neg) < target_neg:
        pos.extend(positives[target_pos : target_pos + (target_neg - len(neg))])

    selected: list[dict[str, Any]] = []
    for idx in range(max(len(pos), len(neg))):
        if idx < len(pos):
            selected.append(pos[idx])
        if idx < len(neg):
            selected.append(neg[idx])
        if len(selected) >= num_examples:
            break
    return selected[:num_examples]


def _format_fewshot_prefix(
    examples: list[dict[str, Any]],
    *,
    yes_label: str,
    no_label: str,
) -> str:
    blocks: list[str] = []
    for idx, example in enumerate(examples, start=1):
        label = yes_label if float(example["relevance"]) > 0 else no_label
        blocks.append(
            f"예시 {idx}:\n"
            f"질문: {example['query']}\n\n"
            f"문서: {example['passage']}\n\n"
            f"이 문서는 질문에 답하는 데 관련이 있다:{label}"
        )
    return "\n\n".join(blocks)


def _score_candidate_batch(
    scorer,
    prompts: list[str],
    *,
    yes_label: str,
    no_label: str,
    score_mode: str,
) -> list[tuple[float, float, float, int, int]]:
    choices_batch = [[yes_label, no_label] for _ in prompts]
    score_dicts = scorer.score_labels_ll_and_len_batch(prompts, choices_batch)
    out: list[tuple[float, float, float, int, int]] = []
    for scores in score_dicts:
        yes_ll, yes_len = scores[yes_label]
        no_ll, no_len = scores[no_label]
        if score_mode == "norm_diff":
            score = (yes_ll / max(1, yes_len)) - (no_ll / max(1, no_len))
        elif score_mode == "yes":
            score = yes_ll
        elif score_mode == "yes_norm":
            score = yes_ll / max(1, yes_len)
        else:
            score = yes_ll - no_ll
        out.append((float(score), float(yes_ll), float(no_ll), int(yes_len), int(no_len)))
    return out


def evaluate_korean_rerank(
    scorer,
    *,
    data_path: str,
    prompt_template: str = DEFAULT_RERANK_PROMPT_TEMPLATE,
    yes_label: str = " 예",
    no_label: str = " 아니오",
    score_mode: str = "diff",
    max_candidates: int = 100,
    limit_queries: Optional[int] = None,
    eval_batch_size: int = 1,
    num_fewshot: int = 0,
    fewshot_data_path: str = "",
    fewshot_seed: int = 42,
) -> dict[str, float]:
    if not data_path:
        raise ValueError("--rerank_data is required for korean_rerank.")
    rows = _load_rerank_rows(data_path)
    if limit_queries is not None:
        rows = rows[:limit_queries]
    fewshot_rows = _load_rerank_rows(fewshot_data_path) if fewshot_data_path else rows
    fewshot_examples = _iter_fewshot_examples(fewshot_rows) if num_fewshot > 0 else []

    batch_size = max(1, int(eval_batch_size))
    scored_queries: list[dict[str, Any]] = []
    total_candidates = 0

    for row in tqdm(rows, desc="KoreanRerank", dynamic_ncols=True):
        query_id = str(_first_present(row, ["query_id", "qid", "id"], len(scored_queries)))
        query = str(_first_present(row, ["query", "question"], ""))
        candidates = list(row.get("candidates") or [])
        if max_candidates > 0:
            candidates = candidates[:max_candidates]
        if not query or not candidates:
            continue

        fewshot_prefix = ""
        if num_fewshot > 0:
            examples = _select_fewshot_examples(
                fewshot_examples,
                num_examples=num_fewshot,
                exclude_query_id=query_id,
                seed=fewshot_seed,
            )
            fewshot_prefix = _format_fewshot_prefix(examples, yes_label=yes_label, no_label=no_label)

        prompts: list[str] = []
        normalized_candidates: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            candidate = dict(candidate)
            passage = _candidate_text(candidate)
            if not passage:
                continue
            prompt = _format_prompt(prompt_template, query=query, passage=passage)
            if fewshot_prefix:
                prompt = f"{fewshot_prefix}\n\n문제:\n{prompt}"
            prompts.append(prompt)
            normalized_candidates.append(
                {
                    "doc_id": _candidate_id(candidate, idx),
                    "relevance": _candidate_rel(candidate),
                    "retriever_rank": candidate.get("retriever_rank", candidate.get("rank", candidate.get("bm25_rank"))),
                    "retriever_score": candidate.get("retriever_score", candidate.get("bm25_score", candidate.get("score"))),
                }
            )

        if not prompts:
            continue

        candidate_scores: list[tuple[float, float, float, int, int]] = []
        for start in range(0, len(prompts), batch_size):
            candidate_scores.extend(
                _score_candidate_batch(
                    scorer,
                    prompts[start : start + batch_size],
                    yes_label=yes_label,
                    no_label=no_label,
                    score_mode=score_mode,
                )
            )

        scored: list[dict[str, Any]] = []
        for candidate, score_info in zip(normalized_candidates, candidate_scores):
            score, yes_ll, no_ll, yes_len, no_len = score_info
            scored.append(
                {
                    **candidate,
                    "rerank_score": score,
                    "yes_ll": yes_ll,
                    "no_ll": no_ll,
                    "yes_len": yes_len,
                    "no_len": no_len,
                }
            )
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        scored_queries.append({"query_id": query_id, "query": query, "candidates": scored})
        total_candidates += len(scored)

    ks = [1, 5, 10, 100]
    metrics_sum: dict[str, float] = defaultdict(float)
    margin_values: list[float] = []
    positive_queries = 0

    for query_result in scored_queries:
        candidates = query_result["candidates"]
        rels = [float(candidate["relevance"]) for candidate in candidates]
        total_rel = sum(1 for rel in rels if rel > 0)
        if total_rel <= 0:
            continue
        positive_queries += 1
        for k in ks:
            metrics_sum[f"ndcg@{k}"] += _ndcg_at(rels, rels, k)
            metrics_sum[f"mrr@{k}"] += _mrr_at(rels, k)
            metrics_sum[f"recall@{k}"] += _recall_at(rels, total_rel, k)

        best_pos = max((candidate["rerank_score"] for candidate in candidates if float(candidate["relevance"]) > 0), default=None)
        best_neg = max((candidate["rerank_score"] for candidate in candidates if float(candidate["relevance"]) <= 0), default=None)
        if best_pos is not None and best_neg is not None:
            margin_values.append(float(best_pos - best_neg))

    denom = max(1, positive_queries)
    results: dict[str, float] = {
        "num_queries": float(len(scored_queries)),
        "num_positive_queries": float(positive_queries),
        "num_candidates": float(total_candidates),
        "avg_candidates_per_query": total_candidates / max(1, len(scored_queries)),
        "num_fewshot": float(max(0, int(num_fewshot))),
        "avg_margin": sum(margin_values) / len(margin_values) if margin_values else 0.0,
        "positive_margin_rate": (
            sum(1 for value in margin_values if value > 0) / len(margin_values) if margin_values else 0.0
        ),
    }
    for key, value in metrics_sum.items():
        results[key] = value / denom
    print(
        "[KoreanRerank] "
        f"queries={int(results['num_queries'])} positive_queries={positive_queries} "
        f"candidates={int(results['num_candidates'])} "
        f"nDCG@10={results.get('ndcg@10', 0.0) * 100:5.2f}% "
        f"MRR@10={results.get('mrr@10', 0.0) * 100:5.2f}% "
        f"Recall@10={results.get('recall@10', 0.0) * 100:5.2f}% "
        f"margin={results['avg_margin']:.4f}"
    )
    return results


def run_korean_rerank(scorer, args) -> BenchmarkRun:
    limit = args.limit if args.limit and args.limit > 0 else None
    results = evaluate_korean_rerank(
        scorer,
        data_path=args.rerank_data,
        prompt_template=args.rerank_prompt_template or DEFAULT_RERANK_PROMPT_TEMPLATE,
        yes_label=args.rerank_yes_label,
        no_label=args.rerank_no_label,
        score_mode=args.rerank_score_mode,
        max_candidates=args.rerank_max_candidates,
        limit_queries=limit,
        eval_batch_size=args.eval_batch_size,
        num_fewshot=args.rerank_num_fewshot,
        fewshot_data_path=args.rerank_fewshot_data,
        fewshot_seed=args.rerank_fewshot_seed,
    )
    return BenchmarkRun(results=results, selected_items=[Path(args.rerank_data).name if args.rerank_data else "rerank"])


KOREAN_RERANK_BENCHMARK = BenchmarkSpec(name="korean_rerank", run=run_korean_rerank)
