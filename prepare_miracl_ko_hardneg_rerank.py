from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_REPO = "datalama/miracl-hard-negatives"
DEFAULT_LANGUAGE = "ko"


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Korean reranking candidate JSONL from datalama/miracl-hard-negatives. "
            "This avoids requiring local MIRACL query/corpus/qrels/run files."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-jsonl", default="rerank_candidates/miracl_ko_hardneg_dev.jsonl")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--require-positive",
        action="store_true",
        help="Drop queries that do not contain a positive candidate.",
    )
    args = parser.parse_args()

    lang = args.language
    query_ds = load_dataset(args.repo, f"queries-{lang}", split="queries", cache_dir=args.cache_dir)
    corpus_ds = load_dataset(args.repo, f"corpus-{lang}", split="corpus", cache_dir=args.cache_dir)
    pair_ds = load_dataset(args.repo, lang, split=args.split, cache_dir=args.cache_dir)

    queries = {str(row["_id"]): str(row["text"]) for row in query_ds}
    corpus = {
        str(row["_id"]): {
            "title": str(row.get("title") or ""),
            "text": str(row.get("text") or ""),
        }
        for row in corpus_ds
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_queries = 0
    missing_docs = 0
    for pair in pair_ds:
        qid = str(pair["query-id"])
        doc_id = str(pair["corpus-id"])
        query = queries.get(qid)
        doc = corpus.get(doc_id)
        if query is None:
            missing_queries += 1
            continue
        if doc is None:
            missing_docs += 1
            continue
        grouped[qid].append(
            {
                "doc_id": doc_id,
                "title": doc["title"],
                "text": doc["text"],
                "retriever_rank": len(grouped[qid]) + 1,
                "retriever_score": 0.0,
                "relevance": int(pair["score"]),
            }
        )

    rows: list[dict[str, Any]] = []
    max_candidates = max(1, int(args.max_candidates))
    for qid in sorted(grouped, key=lambda item: int(item) if item.isdigit() else item):
        candidates = grouped[qid][:max_candidates]
        if args.require_positive and not any(candidate["relevance"] > 0 for candidate in candidates):
            continue
        rows.append({"query_id": qid, "query": queries[qid], "candidates": candidates})

    _write_jsonl(args.out_jsonl, rows)
    num_candidates = sum(len(row["candidates"]) for row in rows)
    positive_queries = sum(1 for row in rows if any(candidate["relevance"] > 0 for candidate in row["candidates"]))
    positives = sum(candidate["relevance"] > 0 for row in rows for candidate in row["candidates"])
    negatives = num_candidates - positives

    print(f"[saved] {args.out_jsonl}")
    print(
        "[summary] "
        f"queries={len(rows)} positive_queries={positive_queries} "
        f"candidates={num_candidates} positives={positives} negatives={negatives} "
        f"missing_queries={missing_queries} missing_docs={missing_docs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
