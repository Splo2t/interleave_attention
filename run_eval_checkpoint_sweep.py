#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_SCRIPT = SCRIPT_DIR / "eval_paper_benchmarks.py"
DEFAULT_VARIANT_EVAL_SCRIPT = SCRIPT_DIR / "eval_variant_hf_krong.py"
STEP_PATTERN = re.compile(r"(?:checkpoint|ckpt|step)[-_]?(\d+)$", flags=re.IGNORECASE)
VARIANT_TASKS = ("kobest_variant",)
VALID_TASKS = (
    "mmlu",
    "kmmlu",
    "kobest",
    "kobest_variant",
    "csatqa",
    "click",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "korean_rerank",
)


@dataclass(frozen=True)
class CheckpointInfo:
    step: int
    path: str
    display_name: str = ""

    @property
    def name(self) -> str:
        if self.display_name:
            return self.display_name
        path_text = str(self.path).rstrip("/")
        return Path(path_text).name or _slugify(path_text)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _slugify(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_.")
    return (text or "item")[:max_len]


def _checkpoint_step(path: Path) -> int | None:
    match = STEP_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _discover_checkpoints(
    root: Path,
    pattern: str,
    *,
    step_interval: int,
    start_step: int | None,
    end_step: int | None,
    reverse: bool,
    max_checkpoints: int,
) -> list[CheckpointInfo]:
    items: list[CheckpointInfo] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_dir():
            continue
        step = _checkpoint_step(path)
        if step is None:
            continue
        if step_interval > 0 and step % step_interval != 0:
            continue
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        items.append(CheckpointInfo(step=step, path=str(path.resolve())))

    items.sort(key=lambda item: item.step, reverse=reverse)
    if max_checkpoints > 0:
        items = items[:max_checkpoints]
    return items


def _merge_fieldnames(*field_groups: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for group in field_groups:
        for field in group:
            if field and field not in seen:
                seen.add(field)
                out.append(field)
    return out


def _append_summary_row(path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "checkpoint_name",
        "step",
        "task",
        "status",
        "returncode",
        "started_at_utc",
        "ended_at_utc",
        "duration_sec",
        "ckpt_path",
        "result_json",
        "stdout_log",
        "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    rows.append(row)
    final_fieldnames = _merge_fieldnames(existing_fieldnames, fieldnames, row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow({field: item.get(field, "") for field in final_fieldnames})


def _write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _infer_model_arch_from_checkpoint(path_text: str | Path) -> str:
    path = Path(str(path_text)).expanduser()
    config_path = path / "config.json"
    if not config_path.exists():
        return ""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return ""
    blob = json.dumps(
        {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "auto_map": config.get("auto_map"),
        },
        ensure_ascii=False,
    ).lower()
    if "krong" in blob or "kormo" in blob:
        return "krong"
    return ""


def _build_eval_command(
    args: argparse.Namespace,
    checkpoint: CheckpointInfo,
    task: str,
    result_json: Path,
    child_log_root: Path,
) -> list[str]:
    cmd = [
        args.python_bin,
        str(args.eval_script),
        "--ckpt_path",
        str(checkpoint.path),
        "--task",
        task,
        "--dtype",
        args.dtype,
        "--device_map",
        args.device_map,
        "--k_shot",
        str(args.k_shot),
        "--seed",
        str(args.seed),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--space_variant_mode",
        args.space_variant_mode,
        "--batch_scoring",
        args.batch_scoring,
        "--continuation_scoring",
        args.continuation_scoring,
        "--dec_max_len",
        str(args.dec_max_len),
        "--limit",
        str(args.limit),
        "--out_json",
        str(result_json),
        "--log_root",
        str(child_log_root),
    ]

    model_arch = args.model_arch or _infer_model_arch_from_checkpoint(checkpoint.path)
    if model_arch:
        cmd.extend(["--model_arch", model_arch])
    if args.cache_root:
        cmd.extend(["--cache_root", args.cache_root])
    if args.subjects:
        cmd.extend(["--subjects", args.subjects])
    if args.use_chat_template:
        cmd.append("--use_chat_template")
    if args.system_prompt:
        cmd.extend(["--system_prompt", args.system_prompt])
    if args.enable_thinking:
        cmd.append("--enable_thinking")
    if args.add_bos != "auto":
        cmd.extend(["--add_bos", args.add_bos])
    if task == "kobest":
        if args.kobest_tasks:
            cmd.extend(["--kobest_tasks", args.kobest_tasks])
        if args.kobest_split:
            cmd.extend(["--kobest_split", args.kobest_split])
    if task == "kobest_variant":
        if args.kobest_tasks:
            cmd.extend(["--kobest_tasks", args.kobest_tasks])
        if args.kobest_split:
            cmd.extend(["--kobest_split", args.kobest_split])
    if task in {"hellaswag", "openbookqa"} and args.benchmark_split:
        cmd.extend(["--benchmark_split", args.benchmark_split])
    if task == "korean_rerank":
        if args.rerank_data:
            cmd.extend(["--rerank_data", args.rerank_data])
        cmd.extend(["--rerank_max_candidates", str(args.rerank_max_candidates)])
        cmd.extend(["--rerank_yes_label", args.rerank_yes_label])
        cmd.extend(["--rerank_no_label", args.rerank_no_label])
        cmd.extend(["--rerank_score_mode", args.rerank_score_mode])
        if args.rerank_prompt_template:
            cmd.extend(["--rerank_prompt_template", args.rerank_prompt_template])
        if args.rerank_num_fewshot > 0:
            cmd.extend(["--rerank_num_fewshot", str(args.rerank_num_fewshot)])
        if args.rerank_fewshot_data:
            cmd.extend(["--rerank_fewshot_data", args.rerank_fewshot_data])
        cmd.extend(["--rerank_fewshot_seed", str(args.rerank_fewshot_seed)])
    if task in VARIANT_TASKS:
        if args.variant_data_root:
            cmd.extend(["--variant_data_root", args.variant_data_root])
        if args.variant_name:
            cmd.extend(["--variant_name", args.variant_name])
    if args.model_label:
        cmd.extend(["--model_label", args.model_label])
    if args.log_group:
        cmd.extend(["--log_group", args.log_group])
    if args.experiment_tag:
        cmd.extend(["--experiment_tag", args.experiment_tag])
    if args.disable_csv_log:
        cmd.append("--disable_csv_log")

    cmd.extend(args.eval_args)
    return cmd


def _run_with_tee(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def _task_slug(tasks: Sequence[str]) -> str:
    return tasks[0] if len(tasks) == 1 else "multi-" + "-".join(_slugify(task, 16) for task in tasks)


def _default_result_root(args: argparse.Namespace, checkpoints_root_name: str) -> Path:
    tasks = getattr(args, "tasks_list", None) or ([args.task] if args.task else ["multi"])
    run_name = args.run_name or f"{_timestamp()}_{_slugify(checkpoints_root_name)}_{_task_slug(tasks)}_every{args.step_interval}"
    return SCRIPT_DIR / "sweep_results" / run_name


def _parse_task_list(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    raw = (args.tasks or args.task or "").strip()
    if not raw:
        parser.error("one of --task or --tasks is required")

    if raw.lower() == "all":
        return list(VALID_TASKS)

    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [task for task in tasks if task not in VALID_TASKS]
    if invalid:
        parser.error(f"unknown task(s): {', '.join(invalid)}. valid: {', '.join(VALID_TASKS)}")

    deduped: list[str] = []
    seen = set()
    for task in tasks:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval_paper_benchmarks.py across checkpoint-* directories and "
            "save per-checkpoint JSON, stdout logs, and a sweep summary CSV."
        )
    )
    parser.add_argument("--checkpoints-root", type=str, default="", help="checkpoint-* 들이 들어있는 루트 디렉터리")
    parser.add_argument(
        "--single-ckpt-path",
        type=str,
        default="",
        help="단일 from_pretrained 경로/repo id를 sweep의 checkpoint 하나처럼 실행",
    )
    parser.add_argument(
        "--single-ckpt-name",
        type=str,
        default="",
        help="--single-ckpt-path 사용 시 JSON/대시보드에 표시할 checkpoint 이름",
    )
    parser.add_argument(
        "--single-ckpt-step",
        type=int,
        default=0,
        help="--single-ckpt-path 사용 시 sweep step 값",
    )
    parser.add_argument("--task", type=str, default="", help=f"단일 task 또는 콤마 목록. valid: {','.join(VALID_TASKS)}")
    parser.add_argument("--tasks", type=str, default="", help="여러 benchmark 콤마 목록. 예: mmlu,kmmlu,csatqa 또는 all")
    parser.add_argument("--checkpoint-pattern", type=str, default="checkpoint-*", help="체크포인트 디렉터리 glob")
    parser.add_argument("--step-interval", type=int, default=1000, help="N step 간격으로만 실행. 0이면 전체")
    parser.add_argument("--start-step", type=int, default=None, help="이 step 이상만 실행")
    parser.add_argument("--end-step", type=int, default=None, help="이 step 이하만 실행")
    parser.add_argument("--reverse", action="store_true", help="큰 step부터 실행")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="최대 실행 개수 제한. 0이면 전체")
    parser.add_argument("--result-root", type=str, default="", help="sweep 결과 루트 디렉터리")
    parser.add_argument("--run-name", type=str, default="", help="result-root 미지정 시 사용할 run 이름")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="하위 evaluator 실행에 사용할 python")
    parser.add_argument("--eval-script", type=str, default=str(DEFAULT_EVAL_SCRIPT), help="실행할 evaluator 스크립트 경로")
    parser.add_argument("--skip-existing-json", action="store_true", help="이미 JSON 결과가 있으면 해당 checkpoint 건너뜀")
    parser.add_argument("--stop-on-error", action="store_true", help="checkpoint 하나라도 실패하면 즉시 중단")
    parser.add_argument("--dry-run", action="store_true", help="실행 없이 계획만 출력")

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--cache_root", type=str, default="")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--space_variant_mode", type=str, default="both", choices=["auto", "both", "none"])
    parser.add_argument("--batch_scoring", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--continuation_scoring", type=str, default="dynamic", choices=["dynamic", "oneshot"])
    parser.add_argument("--dec_max_len", type=int, default=4096)
    parser.add_argument("--add_bos", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--kobest_tasks", type=str, default="")
    parser.add_argument("--kobest_split", type=str, default="test")
    parser.add_argument(
        "--benchmark_split",
        type=str,
        default="",
        help="(hellaswag/openbookqa) split override. 기본값은 evaluator task별 default",
    )
    parser.add_argument("--variant_data_root", type=str, default="")
    parser.add_argument("--variant_name", type=str, default="ko_spacing_stress")
    parser.add_argument("--rerank_data", type=str, default="")
    parser.add_argument("--rerank_max_candidates", type=int, default=100)
    parser.add_argument("--rerank_yes_label", type=str, default=" 예")
    parser.add_argument("--rerank_no_label", type=str, default=" 아니오")
    parser.add_argument("--rerank_score_mode", type=str, default="diff", choices=["diff", "norm_diff", "yes", "yes_norm"])
    parser.add_argument("--rerank_prompt_template", type=str, default="")
    parser.add_argument("--rerank_num_fewshot", type=int, default=0)
    parser.add_argument("--rerank_fewshot_data", type=str, default="")
    parser.add_argument("--rerank_fewshot_seed", type=int, default=42)
    parser.add_argument("--model_arch", type=str, default="")
    parser.add_argument("--model_label", type=str, default="")
    parser.add_argument("--log_group", type=str, default="")
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument("--disable_csv_log", action="store_true")

    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="추가 evaluator 인자. 예: -- --subjects history --dec_max_len 2048",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.tasks_list = _parse_task_list(parser, args)

    if any(task in VARIANT_TASKS for task in args.tasks_list) and args.eval_script == str(DEFAULT_EVAL_SCRIPT):
        args.eval_script = str(DEFAULT_VARIANT_EVAL_SCRIPT)

    if args.single_ckpt_path:
        checkpoints_root: Path | None = None
        checkpoints_root_name = args.single_ckpt_name or Path(args.single_ckpt_path.rstrip("/")).name or "single-model"
        checkpoints = [
            CheckpointInfo(
                step=args.single_ckpt_step,
                path=args.single_ckpt_path,
                display_name=args.single_ckpt_name or checkpoints_root_name,
            )
        ]
    else:
        if not args.checkpoints_root:
            parser.error("one of --checkpoints-root or --single-ckpt-path is required")
        checkpoints_root = Path(args.checkpoints_root).expanduser().resolve()
        if not checkpoints_root.is_dir():
            raise FileNotFoundError(f"Checkpoint root not found: {checkpoints_root}")
        checkpoints_root_name = checkpoints_root.name
        checkpoints = _discover_checkpoints(
            checkpoints_root,
            args.checkpoint_pattern,
            step_interval=args.step_interval,
            start_step=args.start_step,
            end_step=args.end_step,
            reverse=args.reverse,
            max_checkpoints=args.max_checkpoints,
        )
        if not checkpoints:
            raise ValueError("No checkpoints matched the requested filters.")

    args.eval_script = Path(args.eval_script).expanduser().resolve()
    if not args.eval_script.exists():
        raise FileNotFoundError(f"Evaluator script not found: {args.eval_script}")

    result_root = Path(args.result_root).expanduser().resolve() if args.result_root else _default_result_root(args, checkpoints_root_name)
    json_root = result_root / "json"
    stdout_root = result_root / "stdout"
    child_log_root = result_root / "csv_logs"
    summary_csv = result_root / "sweep_summary.csv"
    manifest_path = result_root / "sweep_manifest.json"

    manifest = {
        "created_at_utc": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoints_root": str(checkpoints_root) if checkpoints_root else "",
        "single_ckpt_path": args.single_ckpt_path,
        "single_ckpt_name": args.single_ckpt_name,
        "single_ckpt_step": args.single_ckpt_step,
        "task": args.tasks_list[0] if len(args.tasks_list) == 1 else ",".join(args.tasks_list),
        "tasks": list(args.tasks_list),
        "result_root": str(result_root),
        "checkpoint_pattern": args.checkpoint_pattern,
        "step_interval": args.step_interval,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "reverse": args.reverse,
        "max_checkpoints": args.max_checkpoints,
        "add_bos": args.add_bos,
        "continuation_scoring": args.continuation_scoring,
        "variant_data_root": args.variant_data_root,
        "variant_name": args.variant_name,
        "python_bin": args.python_bin,
        "eval_script": str(args.eval_script),
        "eval_args": list(args.eval_args),
        "model_arch": args.model_arch,
        "model_arch_auto_detect": not bool(args.model_arch),
        "checkpoints": [
            {
                "name": item.name,
                "step": item.step,
                "path": str(item.path),
                "model_arch": args.model_arch or _infer_model_arch_from_checkpoint(item.path),
            }
            for item in checkpoints
        ],
    }

    if args.dry_run:
        print(f"[dry-run] result_root={result_root}")
        print(f"[dry-run] checkpoints={len(checkpoints)}")
        print(f"[dry-run] tasks={','.join(args.tasks_list)}")
        for item in checkpoints:
            for task in args.tasks_list:
                result_json = json_root / task / f"{item.name}.json"
                cmd = _build_eval_command(args, item, task, result_json, child_log_root)
                print(f"[dry-run] {item.name} task={task}")
                print("  " + " ".join(shlex.quote(part) for part in cmd))
        return 0

    result_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)
    stdout_root.mkdir(parents=True, exist_ok=True)
    child_log_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks_list:
        (json_root / task).mkdir(parents=True, exist_ok=True)
        (stdout_root / task).mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, manifest)

    print(f"[sweep] checkpoints={len(checkpoints)}")
    print(f"[sweep] tasks={','.join(args.tasks_list)}")
    print(f"[sweep] total_runs={len(checkpoints) * len(args.tasks_list)}")
    print(f"[sweep] result_root={result_root}")
    print(f"[sweep] summary_csv={summary_csv}")

    failures = 0
    total_runs = len(checkpoints) * len(args.tasks_list)
    run_index = 0
    for checkpoint in checkpoints:
        for task in args.tasks_list:
            run_index += 1
            result_json = json_root / task / f"{checkpoint.name}.json"
            stdout_log = stdout_root / task / f"{checkpoint.name}.log"
            result_json.parent.mkdir(parents=True, exist_ok=True)
            stdout_log.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing_json and result_json.exists():
                print(f"[skip {run_index}/{total_runs}] {checkpoint.name} task={task} json exists: {result_json}")
                _append_summary_row(
                    summary_csv,
                    {
                        "checkpoint_name": checkpoint.name,
                        "step": str(checkpoint.step),
                        "task": task,
                        "status": "skipped_existing_json",
                        "returncode": "",
                        "started_at_utc": "",
                        "ended_at_utc": "",
                        "duration_sec": "",
                        "ckpt_path": str(checkpoint.path),
                        "result_json": str(result_json),
                        "stdout_log": str(stdout_log),
                        "command": "",
                    },
                )
                continue

            cmd = _build_eval_command(args, checkpoint, task, result_json, child_log_root)
            started_at = _utc_now()
            started_monotonic = time.monotonic()

            print(f"[run {run_index}/{total_runs}] {checkpoint.name} step={checkpoint.step} task={task}")
            print(f"[cmd] {' '.join(shlex.quote(part) for part in cmd)}")
            returncode = _run_with_tee(cmd, stdout_log)

            ended_at = _utc_now()
            duration_sec = time.monotonic() - started_monotonic
            status = "ok" if returncode == 0 else "failed"

            _append_summary_row(
                summary_csv,
                {
                    "checkpoint_name": checkpoint.name,
                    "step": str(checkpoint.step),
                    "task": task,
                    "status": status,
                    "returncode": str(returncode),
                    "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ended_at_utc": ended_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "duration_sec": f"{duration_sec:.3f}",
                    "ckpt_path": str(checkpoint.path),
                    "result_json": str(result_json),
                    "stdout_log": str(stdout_log),
                    "command": " ".join(shlex.quote(part) for part in cmd),
                },
            )

            if returncode == 0:
                print(f"[done] {checkpoint.name} task={task} -> {result_json}")
            else:
                failures += 1
                print(f"[error] {checkpoint.name} task={task} failed with code {returncode}. log={stdout_log}")
                if args.stop_on_error:
                    break

        if failures and args.stop_on_error:
            break

    if failures:
        print(f"[sweep] finished with failures={failures}")
        return 1

    print("[sweep] finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
