#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


KOBEST_TASKS = ("boolq", "copa", "hellaswag", "sentineg", "wic")
SPLITS = ("train", "validation", "test")


def _copytree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst} (pass --overwrite)")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", ".cache", "*.lock"))


def _validate_materialized_layout(root: Path) -> None:
    kobest_root = root / "kobest"
    if not kobest_root.is_dir():
        raise FileNotFoundError(f"Expected materialized KoBEST layout at {kobest_root}")

    missing: list[str] = []
    for task in KOBEST_TASKS:
        task_root = kobest_root / task
        if not task_root.is_dir():
            missing.append(str(task_root))
            continue
        if not any((task_root / f"{split}.jsonl").exists() for split in SPLITS):
            missing.append(f"{task_root}/{{train,validation,test}}.jsonl")
    if missing:
        raise FileNotFoundError("Missing expected KoBEST JSONL files:\n" + "\n".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a materialized KoBEST variant dataset from Hugging Face Hub. "
            "The dataset repo should contain manifest.json and kobest/<task>/<split>.jsonl files."
        )
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. Splo2t/ko-random-p25")
    parser.add_argument("--revision", default=None, help="Optional HF revision, branch, tag, or commit SHA")
    parser.add_argument("--variant-name", required=True, help="Local folder name under --output-root")
    parser.add_argument("--output-root", default="variant_benchmarks", help="Local output parent directory")
    parser.add_argument(
        "--source-prefix",
        default=".",
        help=(
            "Subdirectory inside the HF repo that contains manifest.json and kobest/. "
            "Use '.' when the repo root is the variant root."
        ),
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local variant directory")
    args = parser.parse_args()

    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=args.cache_dir,
            allow_patterns=[
                "manifest.json",
                "kobest_tasks.txt",
                "README.md",
                "kobest/**",
                "*/manifest.json",
                "*/kobest_tasks.txt",
                "*/README.md",
                "*/kobest/**",
            ],
        )
    )
    source_root = (snapshot_dir / args.source_prefix).resolve()
    _validate_materialized_layout(source_root)

    output_root = Path(args.output_root).expanduser().resolve()
    target = output_root / args.variant_name
    output_root.mkdir(parents=True, exist_ok=True)
    _copytree(source_root, target, overwrite=args.overwrite)

    print(f"[downloaded] repo={args.repo_id}")
    print(f"[downloaded] source={source_root}")
    print(f"[downloaded] target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
