from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _candidate_text(candidate: dict[str, Any]) -> str:
    title = str(_first(candidate, ["title", "doc_title"], "") or "").strip()
    text = str(_first(candidate, ["text", "passage", "contents", "content", "document"], "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


def _candidate_rel(candidate: dict[str, Any]) -> float:
    value = _first(candidate, ["relevance", "rel", "label", "score_label", "qrel"], 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_grouped_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        return []
    if "candidates" in rows[0]:
        return rows

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(_first(row, ["query_id", "qid", "id"], ""))
        query = str(_first(row, ["query", "question"], ""))
        if not query_id or not query:
            continue
        if query_id not in grouped:
            grouped[query_id] = {"query_id": query_id, "query": query, "candidates": []}
        grouped[query_id]["candidates"].append(
            {
                "doc_id": _first(row, ["doc_id", "document_id", "pid"], f"{query_id}:{len(grouped[query_id]['candidates'])}"),
                "title": _first(row, ["title", "doc_title"], ""),
                "text": _first(row, ["text", "passage", "contents", "content", "document"], ""),
                "relevance": _first(row, ["relevance", "rel", "label", "qrel"], 0),
                "retriever_rank": _first(row, ["rank", "retriever_rank", "bm25_rank"], None),
                "retriever_score": _first(row, ["retriever_score", "bm25_score", "score"], None),
                "candidate_source": _first(row, ["candidate_source", "source"], "flat_input"),
            }
        )
    return list(grouped.values())


def _collect_examples(rows: list[dict[str, Any]], *, min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        query_id = str(_first(row, ["query_id", "qid", "id"], f"query-{row_idx}"))
        query = str(_first(row, ["query", "question"], "")).strip()
        if not query:
            continue
        for cand_idx, candidate in enumerate(list(row.get("candidates") or [])):
            passage = _candidate_text(candidate).strip()
            if not passage:
                continue
            if len(passage) < min_chars:
                continue
            if max_chars > 0 and len(passage) > max_chars:
                continue
            rel = _candidate_rel(candidate)
            examples.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "doc_id": str(_first(candidate, ["doc_id", "document_id", "pid", "id"], f"{query_id}:{cand_idx}")),
                    "title": str(_first(candidate, ["title", "doc_title"], "")),
                    "text": str(_first(candidate, ["text", "passage", "contents", "content", "document"], "")),
                    "relevance": rel,
                    "retriever_rank": _first(candidate, ["retriever_rank", "rank", "bm25_rank"], None),
                    "retriever_score": _first(candidate, ["retriever_score", "bm25_score", "score"], None),
                    "candidate_source": _first(candidate, ["candidate_source", "source"], ""),
                }
            )
    return examples


def _sample_unique_queries(
    examples: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    prefer_low_rank: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    if prefer_low_rank:
        shuffled.sort(
            key=lambda ex: (
                10**12 if ex.get("retriever_rank") is None else int(ex["retriever_rank"]),
                rng.random(),
            )
        )

    selected: list[dict[str, Any]] = []
    used_query_ids: set[str] = set()
    for ex in shuffled:
        if ex["query_id"] in used_query_ids:
            continue
        selected.append(ex)
        used_query_ids.add(ex["query_id"])
        if len(selected) >= count:
            break
    return selected


def _to_demo_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(selected, start=1):
        rows.append(
            {
                "query_id": ex["query_id"],
                "query": ex["query"],
                "fewshot_demo_index": idx,
                "candidates": [
                    {
                        "doc_id": ex["doc_id"],
                        "title": ex["title"],
                        "text": ex["text"],
                        "relevance": ex["relevance"],
                        "retriever_rank": ex["retriever_rank"],
                        "retriever_score": ex["retriever_score"],
                        "candidate_source": ex["candidate_source"],
                    }
                ],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed Korean reranking few-shot demo JSONL from a train candidate file. "
            "The output is intended for krong_eval --task korean_rerank --rerank_fewshot_data."
        )
    )
    parser.add_argument("--candidate-jsonl", required=True, help="Train candidate JSONL, preferably MIRACL-ko train-derived.")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--num-positives", type=int, default=2)
    parser.add_argument("--num-negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-passage-chars", type=int, default=20)
    parser.add_argument("--max-passage-chars", type=int, default=1200)
    parser.add_argument(
        "--prefer-low-rank",
        action="store_true",
        help="Prefer lower retriever_rank examples after seed shuffling. Useful for stable hard-negative demos.",
    )
    args = parser.parse_args()

    rows = _load_grouped_rows(args.candidate_jsonl)
    examples = _collect_examples(rows, min_chars=args.min_passage_chars, max_chars=args.max_passage_chars)
    positives = [ex for ex in examples if float(ex["relevance"]) > 0]
    negatives = [ex for ex in examples if float(ex["relevance"]) <= 0]

    selected_pos = _sample_unique_queries(
        positives,
        count=max(0, args.num_positives),
        seed=args.seed,
        prefer_low_rank=args.prefer_low_rank,
    )
    selected_neg = _sample_unique_queries(
        negatives,
        count=max(0, args.num_negatives),
        seed=args.seed + 1009,
        prefer_low_rank=args.prefer_low_rank,
    )
    if len(selected_pos) < args.num_positives:
        raise RuntimeError(f"Not enough positive demos: requested={args.num_positives} found={len(selected_pos)}")
    if len(selected_neg) < args.num_negatives:
        raise RuntimeError(f"Not enough negative demos: requested={args.num_negatives} found={len(selected_neg)}")

    selected: list[dict[str, Any]] = []
    for idx in range(max(len(selected_pos), len(selected_neg))):
        if idx < len(selected_pos):
            selected.append(selected_pos[idx])
        if idx < len(selected_neg):
            selected.append(selected_neg[idx])

    _write_jsonl(args.out_jsonl, _to_demo_rows(selected))
    print(f"[saved] {args.out_jsonl}")
    print(
        "[summary] "
        f"source={args.candidate_jsonl} demos={len(selected)} "
        f"positives={len(selected_pos)} negatives={len(selected_neg)} seed={args.seed}"
    )
    for ex in selected:
        rel = int(float(ex["relevance"]) > 0)
        rank = ex["retriever_rank"] if ex["retriever_rank"] is not None else "NA"
        print(f"[demo] rel={rel} qid={ex['query_id']} doc={ex['doc_id']} rank={rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
