from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BenchmarkRun:
    results: dict[str, float]
    selected_items: Optional[list[str]]


RunBenchmarkFn = Callable[[Any, Any], BenchmarkRun]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    run: RunBenchmarkFn
