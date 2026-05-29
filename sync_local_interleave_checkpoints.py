#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SOURCE_ROOT = SCRIPT_DIR / "checkpoints" / "checkpoints-interleave-random-enc4096-mlm05_mse001"
DEFAULT_MIRROR_ROOT = SCRIPT_DIR / "checkpoints-interleave-random-enc4096-mlm05_mse001"
DEFAULT_EXPERIMENTS_ROOT = (
    SCRIPT_DIR / "converted_checkpoints_for_experiments" / "checkpoints-interleave-random-enc4096-mlm05_mse001"
)
DEFAULT_CONVERTER = SCRIPT_DIR / "convert_checkpoints_to_checkpoint19000_hf.py"

MANAGED_EXPERIMENT_FILES = {
    "config.json",
    "_modeling_krong.py",
    "_processing_krong.py",
    ".krong_hf_converted.json",
}

WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "model*.bin",
    "pytorch_model*.bin",
)

TRAINING_ARTIFACT_PATTERNS = (
    "trainer_state.json",
    "training_args.bin",
    "rng_state*.pth",
    "latest",
    "latest_checkpointed_iteration.txt",
    "global_state*",
    "global_step*",
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scheduler.bin",
    "zero_to_fp32.py",
)


def log(message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync locally accumulated interleave checkpoints into a KRong HF mirror "
            "directory, then materialize normalized experiment copies "
            "(static helper + tie_word_embeddings=false + flash_attention_2)."
        )
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--mirror-root", default=str(DEFAULT_MIRROR_ROOT))
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument("--converter-script", default=str(DEFAULT_CONVERTER))
    parser.add_argument("--checkpoint-pattern", default="checkpoint-*")
    parser.add_argument(
        "--mirror-attn-implementation",
        default="flash_attention_2",
        help="attn implementation passed to the mirror conversion step.",
    )
    parser.add_argument(
        "--experiment-attn-implementation",
        default="flash_attention_2",
        help="attn implementation enforced in converted_checkpoints_for_experiments.",
    )
    parser.add_argument(
        "--static-modeling-template",
        default="",
        help="Optional path to a canonical static _modeling_krong.py template.",
    )
    parser.add_argument(
        "--static-processing-template",
        default="",
        help="Optional path to a canonical static _processing_krong.py template.",
    )
    parser.add_argument(
        "--overwrite-mirror",
        action="store_true",
        help="Re-convert checkpoints into the mirror root even if they already exist.",
    )
    parser.add_argument(
        "--thin-mirror",
        action="store_true",
        help=(
            "Build mirror checkpoints without copying large weight tensors. Weight files "
            "are symlinked from the source checkpoint, while metadata/config files are "
            "copied and converted in place."
        ),
    )
    parser.add_argument(
        "--overwrite-experiments",
        action="store_true",
        help="Rebuild experiment checkpoint directories from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many checkpoints after sorting by step. 0 means no limit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def iter_checkpoints(root: Path, pattern: str) -> list[Path]:
    items = [p for p in root.glob(pattern) if p.is_dir()]
    items.sort(key=checkpoint_step)
    return items


def is_ready_checkpoint(path: Path) -> bool:
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    return all((path / name).exists() for name in required)


def run(cmd: list[str], *, dry_run: bool, verbose: bool) -> None:
    if dry_run or verbose:
        log(f"run: {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def matches_any_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def ensure_removed(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def build_thin_checkpoint_tree(source_ckpt: Path, mirror_ckpt: Path, *, dry_run: bool) -> None:
    if dry_run:
        log(f"[dry-run] build thin mirror tree {source_ckpt} -> {mirror_ckpt}")
        return

    mirror_ckpt.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_ckpt.iterdir()):
        if matches_any_pattern(child.name, TRAINING_ARTIFACT_PATTERNS):
            continue

        target = mirror_ckpt / child.name
        if target.exists() or target.is_symlink():
            continue

        if child.is_file() and matches_any_pattern(child.name, WEIGHT_FILE_PATTERNS):
            os.symlink(child.resolve(), target)
        elif child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            shutil.copy2(child, target)


def convert_to_mirror(source_ckpt: Path, mirror_root: Path, args: argparse.Namespace) -> Path:
    mirror_ckpt = mirror_root / source_ckpt.name
    if mirror_ckpt.exists() and not args.overwrite_mirror:
        if is_ready_checkpoint(mirror_ckpt):
            log(f"[mirror] skip existing {mirror_ckpt}")
            return mirror_ckpt
        log(f"[mirror] incomplete existing checkpoint; rebuilding {mirror_ckpt}")
        if not args.dry_run:
            ensure_removed(mirror_ckpt)

    if args.thin_mirror:
        if args.overwrite_mirror and mirror_ckpt.exists():
            if args.dry_run:
                log(f"[dry-run] remove {mirror_ckpt}")
            else:
                ensure_removed(mirror_ckpt)
        build_thin_checkpoint_tree(source_ckpt, mirror_ckpt, dry_run=args.dry_run)
        cmd = [
            sys.executable,
            str(Path(args.converter_script).resolve()),
            "--checkpoint-dir",
            str(mirror_ckpt.resolve()),
            "--inplace",
            "--attn-implementation",
            args.mirror_attn_implementation,
        ]
        run(cmd, dry_run=args.dry_run, verbose=args.verbose)
        return mirror_ckpt

    cmd = [
        sys.executable,
        str(Path(args.converter_script).resolve()),
        "--checkpoint-dir",
        str(source_ckpt.resolve()),
        "--output-root",
        str(mirror_root.resolve()),
        "--exclude-training-artifacts",
        "--attn-implementation",
        args.mirror_attn_implementation,
    ]
    if args.overwrite_mirror:
        cmd.append("--overwrite-output")
    run(cmd, dry_run=args.dry_run, verbose=args.verbose)
    return mirror_ckpt


def is_static_helper(modeling_path: Path) -> bool:
    if not modeling_path.exists():
        return False
    text = modeling_path.read_text(encoding="utf-8")
    return "build_prompt_ids_and_L_list(" in text and "list(rebuilt_l)" in text


def default_static_modeling_candidates() -> Iterable[Path]:
    yield SCRIPT_DIR / "converted_checkpoints_for_experiments" / "checkpoints-interleave-random-enc4096-mlm05" / "checkpoint-19000" / "_modeling_krong.py"
    yield SCRIPT_DIR / "converted_compare_19000_checkpoints-interleave-random-enc4096-mlm05" / "checkpoint-19000" / "_modeling_krong.py"
    yield SCRIPT_DIR / "checkpoint-19000" / "_modeling_krong.py"


def default_static_processing_candidates() -> Iterable[Path]:
    yield SCRIPT_DIR / "converted_checkpoints_for_experiments" / "checkpoints-interleave-random-enc4096-mlm05" / "checkpoint-19000" / "_processing_krong.py"
    yield SCRIPT_DIR / "checkpoints-interleave-random-enc4096-mlm05" / "checkpoint-19000" / "_processing_krong.py"
    yield SCRIPT_DIR / "checkpoint-19000" / "_processing_krong.py"


def resolve_static_template(user_path: str, candidates: Iterable[Path], *, label: str) -> Path:
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} template not found: {path}")
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find a default {label} template")


def symlink_non_managed_files(mirror_ckpt: Path, experiment_ckpt: Path) -> None:
    for child in sorted(mirror_ckpt.iterdir()):
        if child.name in MANAGED_EXPERIMENT_FILES:
            continue
        target = experiment_ckpt / child.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(child.resolve(), target)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def prepare_experiment_checkpoint(
    mirror_ckpt: Path,
    experiment_ckpt: Path,
    *,
    static_modeling: Path,
    static_processing: Path,
    experiment_attn: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if overwrite and experiment_ckpt.exists():
        if dry_run:
            log(f"[dry-run] remove {experiment_ckpt}")
        else:
            shutil.rmtree(experiment_ckpt)

    if dry_run:
        log(f"[dry-run] ensure experiment dir {experiment_ckpt}")
    else:
        experiment_ckpt.mkdir(parents=True, exist_ok=True)
        symlink_non_managed_files(mirror_ckpt, experiment_ckpt)

    cfg = json.load(open(mirror_ckpt / "config.json"))
    cfg["tie_word_embeddings"] = False
    cfg["attn_implementation"] = experiment_attn
    cfg["_attn_implementation"] = experiment_attn
    cfg["self_attn_backend"] = experiment_attn

    marker = {}
    marker_path = mirror_ckpt / ".krong_hf_converted.json"
    if marker_path.exists():
        try:
            marker = json.load(open(marker_path))
        except Exception:
            marker = {}
    marker["generation_mode"] = "static"
    marker["style"] = "checkpoint19000"
    marker["source_dir"] = str(mirror_ckpt.resolve())
    marker["normalized_for_experiments"] = True
    marker["normalized_at_utc"] = datetime.now(timezone.utc).isoformat()

    if dry_run:
        log(f"[dry-run] patch {experiment_ckpt / 'config.json'} -> tie=false, attn={experiment_attn}")
        log(f"[dry-run] copy static helpers into {experiment_ckpt}")
        return

    for managed_name in ("config.json", "_modeling_krong.py", "_processing_krong.py", ".krong_hf_converted.json"):
        managed_path = experiment_ckpt / managed_name
        if managed_path.exists() or managed_path.is_symlink():
            ensure_removed(managed_path)

    write_json(experiment_ckpt / "config.json", cfg)
    shutil.copyfile(static_modeling, experiment_ckpt / "_modeling_krong.py")
    shutil.copyfile(static_processing, experiment_ckpt / "_processing_krong.py")
    write_json(experiment_ckpt / ".krong_hf_converted.json", marker)


def main() -> int:
    args = parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    mirror_root = Path(args.mirror_root).expanduser().resolve()
    experiments_root = Path(args.experiments_root).expanduser().resolve()

    if not source_root.exists():
        raise SystemExit(f"source root does not exist: {source_root}")
    converter_script = Path(args.converter_script).expanduser().resolve()
    if not converter_script.exists():
        raise SystemExit(f"converter script does not exist: {converter_script}")

    static_modeling = resolve_static_template(
        args.static_modeling_template,
        default_static_modeling_candidates(),
        label="static modeling",
    )
    static_processing = resolve_static_template(
        args.static_processing_template,
        default_static_processing_candidates(),
        label="static processing",
    )

    checkpoints = [ckpt for ckpt in iter_checkpoints(source_root, args.checkpoint_pattern) if is_ready_checkpoint(ckpt)]
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]

    log(f"source_root={source_root}")
    log(f"mirror_root={mirror_root}")
    log(f"experiments_root={experiments_root}")
    log(f"converter={converter_script}")
    log(f"static_modeling_template={static_modeling}")
    log(f"static_processing_template={static_processing}")
    log(f"checkpoints={len(checkpoints)}")

    for source_ckpt in checkpoints:
        log(f"[checkpoint] {source_ckpt.name}")
        mirror_ckpt = convert_to_mirror(source_ckpt, mirror_root, args)

        if not args.dry_run and not mirror_ckpt.exists():
            raise RuntimeError(f"mirror checkpoint missing after conversion: {mirror_ckpt}")

        experiment_ckpt = experiments_root / source_ckpt.name
        if args.dry_run and not mirror_ckpt.exists():
            log(f"[dry-run] would materialize experiment checkpoint {experiment_ckpt} from {mirror_ckpt}")
            log(f"[dry-run] patch config -> tie=false, attn={args.experiment_attn_implementation}, helper=static")
            continue

        prepare_experiment_checkpoint(
            mirror_ckpt,
            experiment_ckpt,
            static_modeling=static_modeling,
            static_processing=static_processing,
            experiment_attn=args.experiment_attn_implementation,
            overwrite=args.overwrite_experiments,
            dry_run=args.dry_run,
        )

    log("[done] sync complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
