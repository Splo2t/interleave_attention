from __future__ import annotations

from .base import BenchmarkRun, BenchmarkSpec
from .constants import BENCHMARK_NAMES


def get_benchmark(name: str) -> BenchmarkSpec:
    if name == "kobest":
        from .kobest import KOBEST_BENCHMARK

        return KOBEST_BENCHMARK
    if name == "kobest_variant":
        from .kobest_variant import KOBEST_VARIANT_BENCHMARK

        return KOBEST_VARIANT_BENCHMARK

    available = ", ".join(BENCHMARK_NAMES)
    raise ValueError(f"Unknown benchmark '{name}'. Available benchmarks: {available}")


__all__ = ["BENCHMARK_NAMES", "BenchmarkRun", "BenchmarkSpec", "get_benchmark"]
