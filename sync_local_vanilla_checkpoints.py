#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_BASE = SCRIPT_DIR / "converted_checkpoints_for_experiments"

TRAINING_ARTIFACT_PATTERNS = (
    "global_state*",
    "trainer_state.json",
    "training_args.bin",
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scheduler.bin",
    "rng_state*.pth",
    "latest",
    "latest_checkpointed_iteration.txt",
    "global_step*",
    "zero_to_fp32.py",
)


def log(message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize vanilla/normal HF checkpoint copies for experiments. "
            "Files are symlinked from local source checkpoints; config.json is "
            "normalized for evaluation defaults such as tie_word_embeddings=false."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help=(
            "Root containing checkpoint-* directories. Can be repeated. "
            "If omitted, existing checkpoints/checkpoints-normal-* roots are used."
        ),
    )
    parser.add_argument(
        "--experiments-base",
        default=str(DEFAULT_EXPERIMENTS_BASE),
        help="Base directory under which outputs are written using each source root name.",
    )
    parser.add_argument(
        "--experiments-root",
        default="",
        help="Exact output root. Only valid when one --source-root is provided.",
    )
    parser.add_argument("--checkpoint-pattern", default="checkpoint-*")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Value enforced for attn_implementation/_attn_implementation/self_attn_backend.",
    )
    parser.add_argument(
        "--tie-word-embeddings",
        default="false",
        choices=("true", "false"),
        help="Value enforced in config.json. Defaults to false for experiment checkpoints.",
    )
    parser.add_argument(
        "--normalize-tokenizer-config",
        action="store_true",
        help=(
            "Write tokenizer_config.json with tokenizer_class=PreTrainedTokenizerFast. "
            "Not required by this repo's evaluator, which already has a fallback."
        ),
    )
    parser.add_argument(
        "--exclude-training-artifacts",
        action="store_true",
        help="Do not symlink trainer/rng/optimizer style files into the experiment copy.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing output checkpoint directories.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N checkpoints per source root. 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def iter_checkpoints(root: Path, pattern: str) -> list[Path]:
    items = [path for path in root.glob(pattern) if path.is_dir()]
    items.sort(key=checkpoint_step)
    return items


def is_ready_checkpoint(path: Path) -> bool:
    required = ("config.json", "model.safetensors", "tokenizer.json")
    return all((path / name).exists() for name in required)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ensure_removed(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def should_exclude_training_artifact(name: str) -> bool:
    return any(Path(name).match(pattern) for pattern in TRAINING_ARTIFACT_PATTERNS)


def symlink_checkpoint_files(source_ckpt: Path, output_ckpt: Path, *, exclude_training_artifacts: bool) -> None:
    managed = {"config.json", ".vanilla_hf_normalized.json"}
    if exclude_training_artifacts:
        managed.update(name for name in os.listdir(source_ckpt) if should_exclude_training_artifact(name))

    for child in sorted(source_ckpt.iterdir()):
        if child.name in managed:
            continue
        target = output_ckpt / child.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(child.resolve(), target)


def patch_config(source_ckpt: Path, *, attn_implementation: str, tie_word_embeddings: bool) -> dict[str, Any]:
    cfg = read_json(source_ckpt / "config.json")
    cfg["tie_word_embeddings"] = bool(tie_word_embeddings)
    cfg["attn_implementation"] = attn_implementation
    cfg["_attn_implementation"] = attn_implementation
    cfg["self_attn_backend"] = attn_implementation
    return cfg


def normalize_tokenizer_config(source_ckpt: Path) -> dict[str, Any] | None:
    path = source_ckpt / "tokenizer_config.json"
    if not path.exists():
        return None
    cfg = read_json(path)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.setdefault("clean_up_tokenization_spaces", True)
    return cfg


def marker_payload(source_ckpt: Path, output_ckpt: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "style": "vanilla",
        "source_dir": str(source_ckpt.resolve()),
        "output_dir": str(output_ckpt.resolve()),
        "normalized_for_experiments": True,
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "tie_word_embeddings": args.tie_word_embeddings == "true",
        "attn_implementation": args.attn_implementation,
        "normalize_tokenizer_config": bool(args.normalize_tokenizer_config),
        "exclude_training_artifacts": bool(args.exclude_training_artifacts),
    }


def prepare_checkpoint(source_ckpt: Path, output_ckpt: Path, args: argparse.Namespace) -> None:
    if output_ckpt.exists() and args.overwrite:
        if args.dry_run:
            log(f"[dry-run] remove {output_ckpt}")
        else:
            shutil.rmtree(output_ckpt)

    if args.dry_run:
        log(f"[dry-run] prepare {output_ckpt} from {source_ckpt}")
        log(
            "[dry-run] patch config -> "
            f"tie_word_embeddings={args.tie_word_embeddings}, attn={args.attn_implementation}"
        )
        if args.normalize_tokenizer_config:
            log("[dry-run] patch tokenizer_config -> tokenizer_class=PreTrainedTokenizerFast")
        return

    output_ckpt.mkdir(parents=True, exist_ok=True)
    symlink_checkpoint_files(
        source_ckpt,
        output_ckpt,
        exclude_training_artifacts=bool(args.exclude_training_artifacts),
    )

    for managed_name in ("config.json", ".vanilla_hf_normalized.json"):
        managed_path = output_ckpt / managed_name
        if managed_path.exists() or managed_path.is_symlink():
            ensure_removed(managed_path)

    cfg = patch_config(
        source_ckpt,
        attn_implementation=args.attn_implementation,
        tie_word_embeddings=(args.tie_word_embeddings == "true"),
    )
    write_json(output_ckpt / "config.json", cfg)

    if args.normalize_tokenizer_config:
        tok_cfg = normalize_tokenizer_config(source_ckpt)
        tok_path = output_ckpt / "tokenizer_config.json"
        if tok_cfg is not None:
            if tok_path.exists() or tok_path.is_symlink():
                ensure_removed(tok_path)
            write_json(tok_path, tok_cfg)

    write_json(output_ckpt / ".vanilla_hf_normalized.json", marker_payload(source_ckpt, output_ckpt, args))


def resolve_source_roots(args: argparse.Namespace) -> list[Path]:
    if args.source_root:
        roots = [Path(value).expanduser().resolve() for value in args.source_root]
    else:
        roots = [path.resolve() for path in sorted((SCRIPT_DIR / "checkpoints").glob("checkpoints-normal-*")) if path.is_dir()]
    if not roots:
        raise SystemExit("No source roots found. Pass --source-root explicitly.")
    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        raise SystemExit(f"source root does not exist: {', '.join(missing)}")
    return roots


def output_root_for(source_root: Path, roots: list[Path], args: argparse.Namespace) -> Path:
    if args.experiments_root:
        if len(roots) != 1:
            raise SystemExit("--experiments-root can only be used with a single --source-root")
        return Path(args.experiments_root).expanduser().resolve()
    return Path(args.experiments_base).expanduser().resolve() / source_root.name


def main() -> int:
    args = parse_args()
    roots = resolve_source_roots(args)

    log(f"source_roots={len(roots)}")
    log(f"attn_implementation={args.attn_implementation}")
    log(f"tie_word_embeddings={args.tie_word_embeddings}")
    log(f"normalize_tokenizer_config={args.normalize_tokenizer_config}")

    total = 0
    for source_root in roots:
        output_root = output_root_for(source_root, roots, args)
        checkpoints = [ckpt for ckpt in iter_checkpoints(source_root, args.checkpoint_pattern) if is_ready_checkpoint(ckpt)]
        if args.limit > 0:
            checkpoints = checkpoints[: args.limit]

        log(f"[root] source={source_root}")
        log(f"[root] output={output_root}")
        log(f"[root] checkpoints={len(checkpoints)}")

        for source_ckpt in checkpoints:
            output_ckpt = output_root / source_ckpt.name
            log(f"[checkpoint] {source_ckpt.name}{' (update existing)' if output_ckpt.exists() and not args.overwrite else ''}")
            prepare_checkpoint(source_ckpt, output_ckpt, args)
            total += 1

    log(f"[done] prepared={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
