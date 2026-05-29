#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDES = [
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
]


@dataclass(frozen=True)
class RemoteCheckpoint:
    name: str
    path: str
    mtime: float


def log(message: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Poll a remote server for newly created checkpoint directories and "
            "sync them locally with rsync, excluding resume-state files by default."
        )
    )
    parser.add_argument("--remote", required=True, help="SSH target, for example user@train-server")
    parser.add_argument("--remote-dir", required=True, help="Remote directory that contains checkpoint-* folders")
    parser.add_argument("--local-dir", required=True, help="Local directory where checkpoints will be stored")
    parser.add_argument(
        "--checkpoint-pattern",
        default="checkpoint-*",
        help="Glob pattern for remote checkpoint directories",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Polling interval in seconds when --once is not set",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=int,
        default=120,
        help="Only sync checkpoints older than this many seconds to reduce partial-copy risk",
    )
    parser.add_argument(
        "--ready-file",
        default="",
        help="Optional file that must exist inside a checkpoint directory before sync starts",
    )
    parser.add_argument("--once", action="store_true", help="Run one scan/sync pass and exit")
    parser.add_argument(
        "--state-file",
        default="",
        help="Local JSON file used to remember what was already synced",
    )
    parser.add_argument("--ssh-bin", default="ssh", help="SSH executable")
    parser.add_argument("--rsync-bin", default="rsync", help="rsync executable")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default="", help="SSH private key path")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional rsync exclude pattern. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Disable the built-in resume-state exclude patterns",
    )
    parser.add_argument(
        "--rsync-arg",
        action="append",
        default=[],
        help="Additional raw argument forwarded to rsync. Can be repeated.",
    )
    parser.add_argument(
        "--archive-owner-group",
        action="store_true",
        help=(
            "Use rsync -a and preserve owner/group. Disabled by default because "
            "NAS mounts often reject chown/chgrp and make rsync exit with code 23."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full ssh and rsync commands before executing them",
    )
    return parser


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"checkpoints": {}}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"checkpoints": {}}
    data.setdefault("checkpoints", {})
    return data


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_ssh_cmd(args) -> list[str]:
    cmd = [args.ssh_bin, "-p", str(args.ssh_port), "-o", "BatchMode=yes"]
    if args.identity_file:
        cmd.extend(["-i", args.identity_file])
    cmd.append(args.remote)
    return cmd


def build_rsync_transport(args) -> str:
    parts = [args.ssh_bin, "-p", str(args.ssh_port), "-o", "BatchMode=yes"]
    if args.identity_file:
        parts.extend(["-i", args.identity_file])
    return " ".join(shlex.quote(part) for part in parts)


def run_subprocess(cmd: list[str], *, verbose: bool) -> subprocess.CompletedProcess[str]:
    if verbose:
        log(f"run: {' '.join(shlex.quote(part) for part in cmd)}")
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def list_remote_checkpoints(args) -> list[RemoteCheckpoint]:
    remote_dir = shlex.quote(args.remote_dir)
    pattern = shlex.quote(args.checkpoint_pattern)
    remote_cmd = (
        f"find {remote_dir} -mindepth 1 -maxdepth 1 -type d -name {pattern} "
        "-printf '%f\\t%T@\\t%p\\n' | sort"
    )
    cmd = build_ssh_cmd(args) + [remote_cmd]
    proc = run_subprocess(cmd, verbose=args.verbose)

    checkpoints: list[RemoteCheckpoint] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, mtime_raw, path = line.split("\t", 2)
        checkpoints.append(RemoteCheckpoint(name=name, mtime=float(mtime_raw), path=path))
    return checkpoints


def remote_file_exists(args, remote_path: str) -> bool:
    quoted = shlex.quote(remote_path)
    cmd = build_ssh_cmd(args) + [f"test -e {quoted}"]
    try:
        run_subprocess(cmd, verbose=args.verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def should_sync(args, checkpoint: RemoteCheckpoint, state: dict[str, Any], local_dir: Path) -> bool:
    age = time.time() - checkpoint.mtime
    if age < args.min_age_seconds:
        return False

    if args.ready_file:
        if not remote_file_exists(args, f"{checkpoint.path.rstrip('/')}/{args.ready_file}"):
            return False

    local_ckpt_dir = local_dir / checkpoint.name
    prev = state["checkpoints"].get(checkpoint.name)
    if prev is None:
        return True
    if not local_ckpt_dir.exists():
        return True
    if float(prev.get("remote_mtime", 0.0)) < checkpoint.mtime:
        return True
    return False


def sync_checkpoint(args, checkpoint: RemoteCheckpoint, local_dir: Path) -> None:
    local_ckpt_dir = local_dir / checkpoint.name
    local_ckpt_dir.mkdir(parents=True, exist_ok=True)

    excludes = [] if args.no_default_excludes else list(DEFAULT_EXCLUDES)
    excludes.extend(args.exclude)

    source = f"{args.remote}:{shlex.quote(checkpoint.path.rstrip('/'))}/"
    rsync_mode = "-a" if args.archive_owner_group else "-rltD"
    cmd = [
        args.rsync_bin,
        rsync_mode,
        "--partial",
        "--mkpath",
        "--no-owner",
        "--no-group",
        "-e",
        build_rsync_transport(args),
    ]
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    cmd.extend(args.rsync_arg)
    cmd.extend([source, str(local_ckpt_dir)])

    if args.verbose:
        log(f"syncing checkpoint '{checkpoint.name}'")
        if excludes:
            log(f"exclude patterns: {', '.join(excludes)}")
    run_subprocess(cmd, verbose=args.verbose)


def update_state_for_checkpoint(state: dict[str, Any], checkpoint: RemoteCheckpoint, local_dir: Path) -> None:
    state["checkpoints"][checkpoint.name] = {
        "remote_path": checkpoint.path,
        "remote_mtime": checkpoint.mtime,
        "local_path": str((local_dir / checkpoint.name).resolve()),
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def resolve_state_file(args, local_dir: Path) -> Path:
    if args.state_file:
        return Path(args.state_file).expanduser().resolve()
    return local_dir / ".checkpoint_sync_state.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    state_path = resolve_state_file(args, local_dir)
    state = load_state(state_path)

    log(f"remote={args.remote}:{args.remote_dir}")
    log(f"local_dir={local_dir}")
    log(f"state_file={state_path}")
    if not args.no_default_excludes:
        log(f"default resume-state excludes enabled: {', '.join(DEFAULT_EXCLUDES)}")

    while True:
        try:
            checkpoints = list_remote_checkpoints(args)
            if args.verbose:
                log(f"found {len(checkpoints)} remote checkpoint directories")

            synced_any = False
            for checkpoint in checkpoints:
                if not should_sync(args, checkpoint, state, local_dir):
                    continue
                log(f"start sync: {checkpoint.name}")
                sync_checkpoint(args, checkpoint, local_dir)
                update_state_for_checkpoint(state, checkpoint, local_dir)
                save_state(state_path, state)
                synced_any = True
                log(f"finished sync: {checkpoint.name}")

            if not synced_any:
                log("no eligible checkpoint to sync")

            if args.once:
                return 0

            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            log("stopped by user")
            return 130
        except subprocess.CalledProcessError as exc:
            log(f"command failed with exit code {exc.returncode}")
            if exc.stdout:
                log(f"stdout: {exc.stdout.strip()}")
            if exc.stderr:
                log(f"stderr: {exc.stderr.strip()}")
            if args.once:
                return exc.returncode or 1
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
