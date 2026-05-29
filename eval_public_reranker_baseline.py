from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from krong_eval.benchmarks.korean_rerank import (
    _candidate_id,
    _candidate_rel,
    _candidate_text,
    _load_rerank_rows,
    _mrr_at,
    _ndcg_at,
    _recall_at,
)


def _score_with_transformers(
    model_name: str,
    pairs: list[list[str]],
    *,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    logit_mode: str,
) -> list[float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), batch_size), desc="score", dynamic_ncols=True):
            batch = pairs[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
            logits = model(**inputs, return_dict=True).logits.float().detach().cpu()
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_scores = logits.view(-1)
            elif logit_mode == "diff":
                batch_scores = logits[:, -1] - logits[:, 0]
            elif logit_mode == "first":
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits[:, -1]
            scores.extend(float(x) for x in batch_scores.tolist())
    return scores


def _score_with_flagembedding(model_name: str, pairs: list[list[str]], *, batch_size: int, normalize: bool) -> list[float]:
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(model_name, use_fp16=torch.cuda.is_available())
    scores: list[float] = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="score", dynamic_ncols=True):
        batch = pairs[start : start + batch_size]
        batch_scores = reranker.compute_score(batch, normalize=normalize)
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(float(x) for x in batch_scores)
    return scores


def _score_with_jina(model_name: str, groups: list[tuple[str, list[str], list[dict[str, Any]]]]) -> None:
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype="auto")
    model.to(device)
    model.eval()

    for query, passages, refs in tqdm(groups, desc="score", dynamic_ncols=True):
        results = model.rerank(query, passages, top_n=None)
        for result in results:
            idx = int(result["index"])
            refs[idx]["score"] = float(result["relevance_score"])


def _score_with_sentence_transformers_cross_encoder(
    model_name: str,
    pairs: list[list[str]],
    *,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
) -> list[float]:
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict[str, Any] = {"max_length": max_length, "device": device}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        model = CrossEncoder(model_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        model = CrossEncoder(model_name, **kwargs)
    scores = model.predict(
        [tuple(pair) for pair in pairs],
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return [float(score) for score in list(scores)]


def _score_with_sentence_transformers_biencoder(
    model_name: str,
    groups: list[tuple[str, list[str], list[dict[str, Any]]]],
    *,
    batch_size: int,
    trust_remote_code: bool,
    normalize_embeddings: bool,
    query_prefix: str,
    passage_prefix: str,
) -> None:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict[str, Any] = {"device": device}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        model = SentenceTransformer(model_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        model = SentenceTransformer(model_name, **kwargs)

    for query, passages, refs in tqdm(groups, desc="score", dynamic_ncols=True):
        query_text = f"{query_prefix}{query}"
        passage_texts = [f"{passage_prefix}{passage}" for passage in passages]
        q_emb = model.encode(
            [query_text],
            batch_size=1,
            normalize_embeddings=normalize_embeddings,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        p_emb = model.encode(
            passage_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        scores = (p_emb @ q_emb[0]).detach().float().cpu().tolist()
        for ref, score in zip(refs, scores):
            ref["score"] = float(score)


def _compute_metrics(query_results: list[dict[str, Any]]) -> dict[str, float]:
    ks = [1, 5, 10, 100]
    sums: dict[str, float] = defaultdict(float)
    margin_values: list[float] = []
    positive_queries = 0
    total_candidates = 0

    for result in query_results:
        candidates = sorted(result["candidates"], key=lambda item: item["score"], reverse=True)
        rels = [float(candidate["relevance"]) for candidate in candidates]
        total_rel = sum(1 for rel in rels if rel > 0)
        total_candidates += len(candidates)
        if total_rel <= 0:
            continue
        positive_queries += 1
        for k in ks:
            sums[f"ndcg@{k}"] += _ndcg_at(rels, rels, k)
            sums[f"mrr@{k}"] += _mrr_at(rels, k)
            sums[f"recall@{k}"] += _recall_at(rels, total_rel, k)

        best_pos = max((candidate["score"] for candidate in candidates if float(candidate["relevance"]) > 0), default=None)
        best_neg = max((candidate["score"] for candidate in candidates if float(candidate["relevance"]) <= 0), default=None)
        if best_pos is not None and best_neg is not None:
            margin_values.append(float(best_pos - best_neg))

    denom = max(1, positive_queries)
    out = {
        "num_queries": float(len(query_results)),
        "num_positive_queries": float(positive_queries),
        "num_candidates": float(total_candidates),
        "avg_candidates_per_query": total_candidates / max(1, len(query_results)),
        "avg_margin": sum(margin_values) / len(margin_values) if margin_values else 0.0,
        "positive_margin_rate": sum(1 for x in margin_values if x > 0) / len(margin_values) if margin_values else 0.0,
    }
    for key, value in sums.items():
        out[key] = value / denom
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate public cross-encoder rerankers on korean_rerank JSONL.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend",
        choices=[
            "transformers",
            "flagembedding",
            "jina",
            "sentence-transformers",
            "st-cross-encoder",
            "st-biencoder",
        ],
        default="transformers",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--normalize", action="store_true", help="FlagEmbedding sigmoid-normalized score.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom HF modeling/tokenization code for transformers backend.")
    parser.add_argument(
        "--transformers-logit-mode",
        choices=["last", "first", "diff"],
        default="last",
        help="For transformers sequence-classification models with >1 logits: use last logit, first logit, or last-first.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="For st-biencoder, do not L2-normalize embeddings before dot-product scoring.",
    )
    parser.add_argument("--query-prefix", default="", help="Optional prefix for st-biencoder query text.")
    parser.add_argument("--passage-prefix", default="", help="Optional prefix for st-biencoder passage text.")
    args = parser.parse_args()

    rows = _load_rerank_rows(args.data)
    pairs: list[list[str]] = []
    jina_groups: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    query_results: list[dict[str, Any]] = []
    pair_refs: list[dict[str, Any]] = []

    for row in rows:
        query = str(row["query"])
        candidates = list(row.get("candidates") or [])[: max(1, args.max_candidates)]
        result_candidates: list[dict[str, Any]] = []
        passages: list[str] = []
        for idx, candidate in enumerate(candidates):
            passage = _candidate_text(candidate)
            if not passage:
                continue
            ref = {
                "doc_id": _candidate_id(candidate, idx),
                "relevance": _candidate_rel(candidate),
                "score": 0.0,
            }
            pairs.append([query, passage])
            pair_refs.append(ref)
            result_candidates.append(ref)
            passages.append(passage)
        if result_candidates:
            query_results.append({"query_id": row.get("query_id", row.get("qid", "")), "candidates": result_candidates})
            jina_groups.append((query, passages, result_candidates))

    if args.backend == "flagembedding":
        scores = _score_with_flagembedding(args.model, pairs, batch_size=args.batch_size, normalize=args.normalize)
    elif args.backend == "jina":
        _score_with_jina(args.model, jina_groups)
        scores = []
    elif args.backend in {"sentence-transformers", "st-cross-encoder"}:
        scores = _score_with_sentence_transformers_cross_encoder(
            args.model,
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.backend == "st-biencoder":
        _score_with_sentence_transformers_biencoder(
            args.model,
            jina_groups,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            normalize_embeddings=not args.no_normalize_embeddings,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )
        scores = []
    else:
        scores = _score_with_transformers(
            args.model,
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
            logit_mode=args.transformers_logit_mode,
        )

    if scores:
        for ref, score in zip(pair_refs, scores):
            ref["score"] = float(score)

    metrics = _compute_metrics(query_results)
    payload = {"model": args.model, "backend": args.backend, "metrics": metrics}
    out_path = Path(args.out_json).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")
    print(
        f"[metrics] nDCG@10={metrics.get('ndcg@10', 0.0) * 100:.2f} "
        f"MRR@10={metrics.get('mrr@10', 0.0) * 100:.2f} "
        f"Recall@10={metrics.get('recall@10', 0.0) * 100:.2f} "
        f"margin={metrics.get('avg_margin', 0.0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
