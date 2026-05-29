from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_VARIANT_NAME = "ko_spacing_stress"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT_DATA_ROOT = PROJECT_ROOT / "variant_benchmarks" / DEFAULT_VARIANT_NAME


def get_variant_data_root(args: Any) -> Path:
    raw = (
        getattr(args, "variant_data_root", "")
        or os.environ.get("KRONG_VARIANT_DATA_ROOT")
        or str(DEFAULT_VARIANT_DATA_ROOT)
    )
    return Path(raw).expanduser().resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Variant dataset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(item).__name__}")
            rows.append(item)
    return rows


def discover_names(root: Path, benchmark: str, split_name: str) -> list[str]:
    benchmark_root = root / benchmark
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Variant benchmark directory not found: {benchmark_root}")

    names: list[str] = []
    for path in sorted(benchmark_root.iterdir()):
        if path.is_dir() and (path / f"{split_name}.jsonl").exists():
            names.append(path.name)
    if not names:
        raise FileNotFoundError(f"No variant {benchmark} items with {split_name}.jsonl under {benchmark_root}")
    return names
