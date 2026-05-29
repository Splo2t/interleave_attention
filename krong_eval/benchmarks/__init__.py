from __future__ import annotations

from .base import BenchmarkRun, BenchmarkSpec
from .constants import BENCHMARK_NAMES


def get_benchmark(name: str) -> BenchmarkSpec:
    if name == "mmlu":
        from .mmlu import MMLU_BENCHMARK

        return MMLU_BENCHMARK
    if name == "kmmlu":
        from .kmmlu import KMMLU_BENCHMARK

        return KMMLU_BENCHMARK
    if name == "kobest":
        from .kobest import KOBEST_BENCHMARK

        return KOBEST_BENCHMARK
    if name == "csatqa":
        from .csatqa import CSATQA_BENCHMARK

        return CSATQA_BENCHMARK
    if name == "click":
        from .click import CLICK_BENCHMARK

        return CLICK_BENCHMARK
    if name == "arc_easy":
        from .arc import ARC_EASY_BENCHMARK

        return ARC_EASY_BENCHMARK
    if name == "arc_challenge":
        from .arc import ARC_CHALLENGE_BENCHMARK

        return ARC_CHALLENGE_BENCHMARK
    if name == "hellaswag":
        from .commonsense import HELLASWAG_BENCHMARK

        return HELLASWAG_BENCHMARK
    if name == "openbookqa":
        from .commonsense import OPENBOOKQA_BENCHMARK

        return OPENBOOKQA_BENCHMARK
    if name == "korean_rerank":
        from .korean_rerank import KOREAN_RERANK_BENCHMARK

        return KOREAN_RERANK_BENCHMARK

    available = ", ".join(BENCHMARK_NAMES)
    raise ValueError(f"Unknown benchmark '{name}'. Available benchmarks: {available}")


__all__ = ["BENCHMARK_NAMES", "BenchmarkRun", "BenchmarkSpec", "get_benchmark"]
