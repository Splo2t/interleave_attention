#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from krong_eval.cache import DEFAULT_CACHE_ROOT, prepare_cache_paths
from krong_eval.scoring import build_scorer_from_args


DEFAULT_PROMPTS: list[str] = [
    "대한민국의 수도는",
    "비가 그친 뒤 공원에는",
    "인공지능이 의료 분야에 도움이 되는 이유는",
    "The president said that the new policy",
    "Given the sequence 2, 4, 8, 16, the next number is",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_path: str
    model_arch: str
    add_bos: str = "auto"


DEFAULT_MODELS_10000: list[ModelSpec] = [
    ModelSpec("final", "final", ""),
    ModelSpec(
        "interleave-10000-static",
        "converted_checkpoints_for_experiments/checkpoints-interleave-random-enc4096-mlm05/checkpoint-10000",
        "krong",
    ),
    ModelSpec(
        "interleave-10000-append",
        "converted_checkpoints_for_experiments/checkpoints-interleave-random-enc4096-mlm05/checkpoint-10000-append",
        "krong",
    ),
    ModelSpec(
        "normal-copylayer-10000",
        "converted_checkpoints_for_experiments/checkpoints-normal-copylayer/checkpoint-10000",
        "",
    ),
    ModelSpec(
        "normal-random-10000",
        "converted_checkpoints_for_experiments/checkpoints-normal-random/checkpoint-10000",
        "",
    ),
    ModelSpec(
        "mse001-9000-static",
        "converted_checkpoints_for_experiments/checkpoints-interleave-random-enc4096-mlm05_mse001/checkpoint-9000",
        "krong",
    ),
]

DEFAULT_MODELS_19000: list[ModelSpec] = [
    ModelSpec(
        "interleave-19000-static",
        "converted_checkpoints_for_experiments/checkpoints-interleave-random-enc4096-mlm05/checkpoint-19000",
        "krong",
    ),
    ModelSpec(
        "normal-copylayer-19000",
        "converted_checkpoints_for_experiments/checkpoints-normal-copylayer/checkpoint-19000",
        "",
    ),
]

MODEL_PROFILES: dict[str, list[ModelSpec]] = {
    "10000": DEFAULT_MODELS_10000,
    "19000": DEFAULT_MODELS_19000,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a fixed set of checkpoints and save comparison CSV files.")
    parser.add_argument("--output-csv", required=True, help="Path to the wide CSV output.")
    parser.add_argument("--output-long-csv", default="", help="Optional path to the long-form CSV output.")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES.keys()), default="10000")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--dec_max_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--skip_special_tokens", action="store_true")
    parser.add_argument("--prompt-file", default="", help="Optional text file with one prompt per line.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    return parser


def _load_prompts(prompt_file: str) -> list[str]:
    if not prompt_file:
        return list(DEFAULT_PROMPTS)
    lines = [line.strip("\n") for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line.strip()]


def _make_scorer_args(model: ModelSpec, args: argparse.Namespace, cache_paths) -> argparse.Namespace:
    return argparse.Namespace(
        ckpt_path=model.ckpt_path,
        model_arch=model.model_arch,
        dtype=args.dtype,
        device_map=args.device_map,
        transformers_cache=cache_paths.transformers_cache,
        dec_max_len=args.dec_max_len,
        add_bos=model.add_bos,
        use_chat_template=False,
        system_prompt="",
        enable_thinking=False,
        space_variant_mode="both",
        batch_scoring="off",
        continuation_scoring="oneshot",
    )


def _generate_outputs(models: Sequence[ModelSpec], prompts: Sequence[str], args: argparse.Namespace) -> list[dict[str, str]]:
    cache_paths = prepare_cache_paths(args.cache_root)
    rows: list[dict[str, str]] = []
    for model in models:
        print(f"[load] {model.name} -> {model.ckpt_path}")
        scorer = build_scorer_from_args(_make_scorer_args(model, args, cache_paths))
        try:
            for prompt_idx, prompt in enumerate(prompts, start=1):
                print(f"[generate] {model.name} prompt#{prompt_idx}")
                text = scorer.generate_text(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=bool(args.do_sample),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    skip_special_tokens=bool(args.skip_special_tokens),
                )
                rows.append(
                    {
                        "prompt_id": str(prompt_idx),
                        "prompt": prompt,
                        "model_name": model.name,
                        "ckpt_path": model.ckpt_path,
                        "model_arch": model.model_arch or "plain",
                        "output": text,
                    }
                )
        finally:
            del scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def _write_wide_csv(rows: Sequence[dict[str, str]], output_csv: Path, model_names: Sequence[str], prompts: Sequence[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    by_prompt: dict[str, dict[str, str]] = {}
    for row in rows:
        prompt = row["prompt"]
        bucket = by_prompt.setdefault(prompt, {"prompt_id": row["prompt_id"], "prompt": prompt})
        bucket[row["model_name"]] = row["output"]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "prompt", *model_names])
        writer.writeheader()
        for prompt_idx, prompt in enumerate(prompts, start=1):
            bucket = by_prompt.get(prompt, {"prompt_id": str(prompt_idx), "prompt": prompt})
            writer.writerow(bucket)


def _write_long_csv(rows: Sequence[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "prompt", "model_name", "ckpt_path", "model_arch", "output"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    prompts = _load_prompts(args.prompt_file)
    models = MODEL_PROFILES[args.profile]
    rows = _generate_outputs(models, prompts, args)
    _write_wide_csv(rows, Path(args.output_csv), [m.name for m in models], prompts)
    if args.output_long_csv:
        _write_long_csv(rows, Path(args.output_long_csv))

    metadata = {
        "output_csv": str(Path(args.output_csv).resolve()),
        "output_long_csv": str(Path(args.output_long_csv).resolve()) if args.output_long_csv else "",
        "device_map": args.device_map,
        "dtype": args.dtype,
        "dec_max_len": args.dec_max_len,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "skip_special_tokens": bool(args.skip_special_tokens),
        "profile": args.profile,
        "prompts": prompts,
        "models": [model.__dict__ for model in models],
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
