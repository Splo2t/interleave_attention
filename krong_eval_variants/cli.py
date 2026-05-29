from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import get_benchmark
from .benchmarks.constants import BENCHMARK_NAMES
from .benchmarks.variant_io import DEFAULT_VARIANT_DATA_ROOT, DEFAULT_VARIANT_NAME
from .cache import DEFAULT_CACHE_ROOT, prepare_cache_paths
from .logging_utils import write_result_logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="from_pretrained 경로 또는 repo id")
    parser.add_argument("--task", type=str, choices=list(BENCHMARK_NAMES), default="mmlu")
    parser.add_argument(
        "--model_label",
        type=str,
        default="",
        help="CSV 로그에 기록할 모델 표시 이름(미지정 시 ckpt basename)",
    )
    parser.add_argument(
        "--log_group",
        type=str,
        choices=["auto", "krong", "kormo", "others"],
        default="auto",
        help="CSV 로그 분류. auto면 ckpt_path/model_arch 기반으로 kormo/krong/others 자동 분류",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="",
        help="실험 태그. run_id/profile_tag/overview row 구분에 사용",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="",
        help="CSV 로그 루트 경로(기본: 스크립트 옆 logs/)",
    )
    parser.add_argument("--disable_csv_log", action="store_true", help="CSV 실험 로그 기록 비활성화")

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto", help="HF device_map (예: auto, cuda:0, cpu)")
    parser.add_argument(
        "--cache_root",
        type=str,
        default=DEFAULT_CACHE_ROOT,
        help="HF 모델/토크나이저/데이터셋 캐시 루트 디렉터리",
    )
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=1, help="평가 시 prompt batch 크기")

    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help=(
            "(kobest/kobest_variant) KoBEST subtasks: boolq,copa,hellaswag,sentineg,wic"
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="과목(또는 task)당 평가 샘플 수 제한(디버그용). 0이면 전체")

    parser.add_argument("--use_chat_template", action="store_true", help="prompt를 chat_template로 감싼 뒤 평가")
    parser.add_argument("--system_prompt", type=str, default="", help="chat_template 사용 시 system 메시지")
    parser.add_argument("--enable_thinking", action="store_true", help="chat_template의 enable_thinking=True로 렌더")

    parser.add_argument("--space_variant_mode", type=str, default="both", choices=["auto", "both", "none"])
    parser.add_argument(
        "--batch_scoring",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "choice scoring batch 사용 모드. auto면 batch processor가 있는 모델은 batch, "
            "없는 KRong/KORMo processor는 org_eval처럼 serial fallback"
        ),
    )
    parser.add_argument(
        "--continuation_scoring",
        type=str,
        default="dynamic",
        choices=["dynamic", "oneshot"],
        help=(
            "multi-token choice 로그우도 계산 방식. dynamic은 토큰을 하나씩 붙여 generation 경로로 계산하고, "
            "oneshot은 lm-eval wrapper처럼 prompt+choice를 한 번에 forward해서 계산"
        ),
    )
    parser.add_argument("--dec_max_len", type=int, default=4096, help=">0이면 디코더 입력을 좌측 트렁케이션")
    parser.add_argument(
        "--add_bos",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="KRong processor add_bos override. auto면 processor_config.json 값을 사용",
    )
    parser.add_argument("--out_json", type=str, default="", help="결과를 JSON으로 저장할 경로(옵션)")
    parser.add_argument(
        "--save_item_predictions",
        action="store_true",
        help=(
            "out_json에 문항별 gold/pred/correct를 함께 저장한다. "
            "original vs stress paired flip analysis에 사용한다."
        ),
    )

    parser.add_argument(
        "--kobest_tasks",
        type=str,
        default="",
        help="(kobest/kobest_variant) tasks: boolq,copa,hellaswag,sentineg,wic",
    )
    parser.add_argument("--kobest_split", type=str, default="test", help="(kobest) split: train/validation/test")
    parser.add_argument(
        "--variant_data_root",
        type=str,
        default=str(DEFAULT_VARIANT_DATA_ROOT),
        help="(variant tasks) 변형 JSONL 루트. 기본: variant_benchmarks/choice_shuffle",
    )
    parser.add_argument(
        "--variant_name",
        type=str,
        default=DEFAULT_VARIANT_NAME,
        help="(variant tasks) 로그/추적용 변형 이름",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="",
        help="모델 아키텍처 이름 (예: gpt2, llama2, etc.) - processor/토크나이저 로드에 활용",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_paths = prepare_cache_paths(args.cache_root)
    args.cache_root = cache_paths.cache_root
    args.hf_home = cache_paths.hf_home
    args.transformers_cache = cache_paths.transformers_cache
    args.datasets_cache = cache_paths.datasets_cache
    print(f"[cache] root={args.cache_root}")
    print(f"[cache] transformers={args.transformers_cache}")
    print(f"[cache] datasets={args.datasets_cache}")

    from .scoring import build_scorer_from_args

    scorer = build_scorer_from_args(args)
    benchmark_run = get_benchmark(args.task).run(scorer, args)

    if args.out_json:
        Path(args.out_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
        output_payload = benchmark_run.results
        if args.save_item_predictions:
            output_payload = {
                "metrics": benchmark_run.results,
                "item_predictions": benchmark_run.item_predictions or [],
            }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.out_json}")

    write_result_logs(args, benchmark_run.results, benchmark_run.selected_items)
