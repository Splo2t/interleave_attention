from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def _first(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _load_queries(path: str, *, id_field: str, text_field: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _read_jsonl(path):
        qid = str(_first(row, [id_field, "query_id", "qid", "id"]))
        text = str(_first(row, [text_field, "query", "question", "text"]))
        if qid and text:
            out[qid] = text
    return out


def _load_corpus(path: str, *, id_field: str, text_field: str, title_field: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(path):
        doc_id = str(_first(row, [id_field, "doc_id", "document_id", "pid", "id", "_id"]))
        text = str(_first(row, [text_field, "text", "passage", "contents", "content"]))
        title = str(_first(row, [title_field, "title", "doc_title"], ""))
        if doc_id and (text or title):
            out[doc_id] = {"title": title, "text": text}
    return out


def _load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                qid, _, doc_id, rel = parts[:4]
            elif len(parts) == 3:
                qid, doc_id, rel = parts
            else:
                continue
            qrels[str(qid)][str(doc_id)] = int(float(rel))
    return qrels


def _load_run(path: str) -> dict[str, list[tuple[str, int, float]]]:
    run: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, *_ = parts
            elif len(parts) >= 3:
                qid, doc_id, score = parts[:3]
                rank = str(len(run[str(qid)]) + 1)
            else:
                raise ValueError(f"Unrecognized run format at line {line_no}: {line}")
            run[str(qid)].append((str(doc_id), int(rank), float(score)))
    for qid, items in run.items():
        items.sort(key=lambda item: (item[1], -item[2]))
    return run


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_candidate(
    *,
    doc_id: str,
    doc: dict[str, str],
    relevance: int,
    retriever_rank: int | None,
    retriever_score: float | None,
    source: str,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "title": doc.get("title", ""),
        "text": doc.get("text", ""),
        "retriever_rank": retriever_rank,
        "retriever_score": retriever_score,
        "relevance": relevance,
        "candidate_source": source,
    }


def _build_retrieval_rows(
    *,
    queries: dict[str, str],
    corpus: dict[str, dict[str, str]],
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, int, float]]],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    missing_docs = 0
    for qid, candidates in sorted(run.items()):
        query = queries.get(qid)
        if not query:
            continue
        candidate_rows: list[dict[str, Any]] = []
        for doc_id, rank, score in candidates[: max(1, max_candidates)]:
            doc = corpus.get(doc_id)
            if doc is None:
                missing_docs += 1
                continue
            candidate_rows.append(
                _make_candidate(
                    doc_id=doc_id,
                    doc=doc,
                    retriever_rank=rank,
                    retriever_score=score,
                    relevance=qrels.get(qid, {}).get(doc_id, 0),
                    source="retriever_run",
                )
            )
        if candidate_rows:
            rows.append({"query_id": qid, "query": query, "candidates": candidate_rows})
    return rows, missing_docs


def _build_controlled_hard_negative_rows(
    *,
    queries: dict[str, str],
    corpus: dict[str, dict[str, str]],
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, int, float]]],
    max_candidates: int,
    max_positives: int,
    min_negatives: int,
    max_run_depth: int,
    positive_selection: str,
) -> tuple[list[dict[str, Any]], int, int, int]:
    rows: list[dict[str, Any]] = []
    missing_docs = 0
    skipped_without_positive = 0
    skipped_without_negatives = 0

    for qid, query in sorted(queries.items()):
        qrel_items = qrels.get(qid, {})
        run_rank_by_doc = {doc_id: (rank, score) for doc_id, rank, score in run.get(qid, [])}
        positive_items = [(doc_id, rel) for doc_id, rel in qrel_items.items() if rel > 0]
        if positive_selection == "bm25-rank":
            positive_items = sorted(
                positive_items,
                key=lambda item: (run_rank_by_doc.get(item[0], (10**12, 0.0))[0], -item[1], item[0]),
            )
        else:
            positive_items = sorted(positive_items, key=lambda item: (-item[1], item[0]))
        if max_positives > 0:
            positive_items = positive_items[:max_positives]
        if not positive_items:
            skipped_without_positive += 1
            continue

        candidate_rows: list[dict[str, Any]] = []
        used_doc_ids: set[str] = set()

        for doc_id, rel in positive_items:
            doc = corpus.get(doc_id)
            if doc is None:
                missing_docs += 1
                continue
            rank_score = run_rank_by_doc.get(doc_id)
            rank = rank_score[0] if rank_score else None
            score = rank_score[1] if rank_score else None
            candidate_rows.append(
                _make_candidate(
                    doc_id=doc_id,
                    doc=doc,
                    retriever_rank=rank,
                    retriever_score=score,
                    relevance=rel,
                    source="qrels_positive",
                )
            )
            used_doc_ids.add(doc_id)

        if not candidate_rows:
            skipped_without_positive += 1
            continue

        run_candidates = run.get(qid, [])
        if max_run_depth > 0:
            run_candidates = run_candidates[:max_run_depth]

        max_negatives = max(0, max_candidates - len(candidate_rows))
        negative_count = 0
        for doc_id, rank, score in run_candidates:
            if negative_count >= max_negatives:
                break
            if doc_id in used_doc_ids:
                continue
            if qrel_items.get(doc_id, 0) > 0:
                continue
            doc = corpus.get(doc_id)
            if doc is None:
                missing_docs += 1
                continue
            candidate_rows.append(
                _make_candidate(
                    doc_id=doc_id,
                    doc=doc,
                    retriever_rank=rank,
                    retriever_score=score,
                    relevance=0,
                    source="hard_negative_run",
                )
            )
            used_doc_ids.add(doc_id)
            negative_count += 1

        if negative_count < min_negatives:
            skipped_without_negatives += 1
            continue
        rows.append({"query_id": qid, "query": query, "candidates": candidate_rows[:max_candidates]})

    return rows, missing_docs, skipped_without_positive, skipped_without_negatives


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build candidate JSONL for krong_eval --task korean_rerank from query/corpus JSONL, "
            "a TREC-style run file, and qrels."
        )
    )
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--run-file", required=True, help="TREC run: qid Q0 docid rank score tag, or qid docid score")
    parser.add_argument("--qrels-file", required=True, help="TREC qrels: qid 0 docid rel, or qid docid rel")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--candidate-mode",
        default="retrieval",
        choices=["retrieval", "controlled-hard-negative"],
        help=(
            "retrieval keeps the run list as-is. controlled-hard-negative injects qrels positives "
            "and fills the remaining slots with top-ranked non-relevant run documents."
        ),
    )
    parser.add_argument("--max-positives", type=int, default=1, help="Maximum qrels positives to inject per query in controlled mode. Use 0 for all positives.")
    parser.add_argument(
        "--positive-selection",
        default="best-qrel",
        choices=["best-qrel", "bm25-rank"],
        help="How to choose positives when qrels contain multiple relevant passages.",
    )
    parser.add_argument("--min-negatives", type=int, default=1, help="Minimum hard negatives required per query in controlled mode.")
    parser.add_argument("--max-run-depth", type=int, default=1000, help="Maximum retrieved depth to search for negatives in controlled mode. Use 0 for all.")
    parser.add_argument("--query-id-field", default="query_id")
    parser.add_argument("--query-text-field", default="query")
    parser.add_argument("--doc-id-field", default="doc_id")
    parser.add_argument("--doc-text-field", default="text")
    parser.add_argument("--doc-title-field", default="title")
    args = parser.parse_args()

    queries = _load_queries(args.queries_jsonl, id_field=args.query_id_field, text_field=args.query_text_field)
    corpus = _load_corpus(
        args.corpus_jsonl,
        id_field=args.doc_id_field,
        text_field=args.doc_text_field,
        title_field=args.doc_title_field,
    )
    qrels = _load_qrels(args.qrels_file)
    run = _load_run(args.run_file)

    if args.candidate_mode == "controlled-hard-negative":
        rows, missing_docs, skipped_without_positive, skipped_without_negatives = _build_controlled_hard_negative_rows(
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            run=run,
            max_candidates=max(1, args.max_candidates),
            max_positives=args.max_positives,
            min_negatives=max(0, args.min_negatives),
            max_run_depth=args.max_run_depth,
            positive_selection=args.positive_selection,
        )
    else:
        rows, missing_docs = _build_retrieval_rows(
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            run=run,
            max_candidates=max(1, args.max_candidates),
        )
        skipped_without_positive = 0
        skipped_without_negatives = 0

    _write_jsonl(args.out_jsonl, rows)
    positives = sum(1 for row in rows if any(candidate["relevance"] > 0 for candidate in row["candidates"]))
    avg_candidates = sum(len(row["candidates"]) for row in rows) / len(rows) if rows else 0.0
    print(f"[saved] {args.out_jsonl}")
    print(
        "[summary] "
        f"mode={args.candidate_mode} queries={len(rows)} positive_queries={positives} "
        f"avg_candidates={avg_candidates:.2f} missing_docs={missing_docs} "
        f"skipped_without_positive={skipped_without_positive} "
        f"skipped_without_negatives={skipped_without_negatives}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
