from __future__ import annotations

import argparse
import json
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


def _query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "qid", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter grouped rerank JSONL rows by query IDs.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--exclude-jsonl", default="", help="JSONL whose query IDs should be removed from input.")
    parser.add_argument("--exclude-query-ids", default="", help="Comma-separated query IDs to remove.")
    args = parser.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    exclude_ids: set[str] = set()
    if args.exclude_jsonl:
        exclude_ids.update(_query_id(row) for row in _read_jsonl(args.exclude_jsonl))
    if args.exclude_query_ids:
        exclude_ids.update(part.strip() for part in args.exclude_query_ids.split(",") if part.strip())
    exclude_ids.discard("")

    kept = [row for row in rows if _query_id(row) not in exclude_ids]
    _write_jsonl(args.out_jsonl, kept)
    print(f"[saved] {args.out_jsonl}")
    print(f"[summary] input={len(rows)} excluded={len(rows) - len(kept)} kept={len(kept)}")
    if exclude_ids:
        print("[excluded_query_ids] " + ",".join(sorted(exclude_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
