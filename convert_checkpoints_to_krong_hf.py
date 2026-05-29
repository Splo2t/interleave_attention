#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TemplateStyle:
    name: str
    default_template_dir: Path
    remote_code_files: tuple[str, ...]
    optional_template_files: tuple[str, ...]
    required_template_files: tuple[str, ...]
    generation_mode: str = "static"
    synthetic_processor_config: dict[str, Any] | None = None
    processor_force_keys: tuple[str, ...] = ()


STYLE_SPECS: dict[str, TemplateStyle] = {
    "checkpoint19000": TemplateStyle(
        name="checkpoint19000",
        default_template_dir=SCRIPT_DIR / "checkpoint-19000",
        remote_code_files=(
            "_configuration_krong.py",
            "_modeling_krong.py",
            "_processing_krong.py",
            "chat_template.jinja",
            "example_chat_hf.py",
        ),
        optional_template_files=("generation_config.json",),
        required_template_files=(
            "config.json",
            "processor_config.json",
            "_configuration_krong.py",
            "_modeling_krong.py",
            "_processing_krong.py",
            "krong_tokenizer",
        ),
        generation_mode="static",
        processor_force_keys=(
            "processor_class",
            "auto_map",
            "enc_subfolder",
            "enc_max_len",
            "min_prefix",
            "add_bos",
        ),
    ),
    "krong_llm": TemplateStyle(
        name="krong_llm",
        default_template_dir=SCRIPT_DIR / "checkpoint-19000",
        remote_code_files=(
            "_configuration_krong.py",
            "_modeling_krong.py",
            "_processing_krong.py",
            "chat_template.jinja",
            "example_chat_hf.py",
        ),
        optional_template_files=("generation_config.json",),
        required_template_files=(
            "config.json",
            "processor_config.json",
            "_configuration_krong.py",
            "_modeling_krong.py",
            "_processing_krong.py",
            "krong_tokenizer",
        ),
        generation_mode="incremental",
        processor_force_keys=(
            "processor_class",
            "auto_map",
            "enc_subfolder",
            "enc_max_len",
            "min_prefix",
            "add_bos",
        ),
    ),
}
DEFAULT_STYLE = "checkpoint19000"
DEFAULT_TEMPLATE_DIR = STYLE_SPECS[DEFAULT_STYLE].default_template_dir
TRAINING_ARTIFACT_PATTERNS = [
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
]


STATIC_WS_GENERATION_BLOCK = """        if ws_state is not None:
            if input_ids is None:
                raise ValueError(\"ws_state generation requires input_ids\")

            use_cache = False
            past_key_values = None
            cache_position = None

            dec_tok = ws_state[\"dec_tok\"]
            enc_tok = ws_state[\"enc_tok\"]
            min_prefix = int(ws_state[\"min_prefix\"])
            enc_max_len = int(ws_state[\"enc_max_len\"])
            dec_max_len = int(ws_state.get(\"dec_max_len\", 0) or 0)
            add_bos = bool(ws_state.get(\"add_bos\", False))

            dec_text_now = dec_tok.decode(input_ids[0].tolist(), skip_special_tokens=True)

            # Rebuild the full decoder-aligned schedule from decoded text each step so
            # generation follows the same text->tokenization->L path as preprocessing.
            rebuilt_ids, rebuilt_mask, rebuilt_l = build_prompt_ids_and_L_list(
                dec_text_now,
                dec_tok,
                enc_tok,
                dec_text_now,
                1_000_000,
                min_prefix,
                enc_max_len,
                device=input_ids.device,
                add_bos=add_bos,
            )

            if dec_max_len > 0 and rebuilt_ids.shape[1] > dec_max_len:
                offset = int(rebuilt_ids.shape[1] - dec_max_len)
                rebuilt_ids = rebuilt_ids[:, offset:]
                rebuilt_mask = rebuilt_mask[:, offset:]
                rebuilt_l = rebuilt_l[-dec_max_len:]

            dec_text_trim = dec_tok.decode(rebuilt_ids[0].tolist(), skip_special_tokens=True)

            enc_dev_raw = ws_state.get(\"enc_device\", input_ids.device)
            enc_dev = torch.device(enc_dev_raw) if not isinstance(enc_dev_raw, torch.device) else enc_dev_raw
            enc_ids, enc_mask = prepare_encoder_inputs(
                dec_text_trim,
                enc_tok,
                enc_max_len,
                device=enc_dev,
            )

            with torch.no_grad():
                enc_out = self.encoder(input_ids=enc_ids, attention_mask=enc_mask, causal=True)

            k_eff = int(enc_mask.sum().item())
            rebuilt_ids, rebuilt_mask, rebuilt_l = build_prompt_ids_and_L_list(
                dec_text_trim,
                dec_tok,
                enc_tok,
                dec_text_trim,
                k_eff,
                min_prefix,
                enc_max_len,
                device=input_ids.device,
                add_bos=add_bos,
            )

            ws_state[\"enc_text\"] = dec_text_trim
            ws_state[\"prev_len\"] = int(rebuilt_ids.shape[1])
            ws_state[\"L_per_token\"] = list(rebuilt_l)
            ws_state[\"L_cur\"] = max(min_prefix, min(k_eff, enc_max_len))
            ws_state[\"K_eff\"] = k_eff
            ws_state[\"encoder_hidden_states\"] = enc_out.last_hidden_state.to(input_ids.device, non_blocking=True)
            ws_state[\"encoder_attention_mask\"] = enc_mask.to(input_ids.device, non_blocking=True)
            ws_state[\"encoder_input_ids\"] = enc_ids
            ws_state[\"encoder_attention_mask_enc\"] = enc_mask

            attention_mask = rebuilt_mask
            input_ids = rebuilt_ids
            l_full = torch.tensor([rebuilt_l], device=input_ids.device, dtype=torch.long)

            return {
                \"input_ids\": input_ids,
                \"attention_mask\": attention_mask,
                \"past_key_values\": None,
                \"use_cache\": use_cache,
                \"cache_position\": cache_position,
                \"position_ids\": None,
                \"logits_to_keep\": 1,
                \"encoder_hidden_states\": ws_state.get(\"encoder_hidden_states\"),
                \"encoder_attention_mask\": ws_state.get(\"encoder_attention_mask\"),
                \"encoder_input_ids\": None,
                \"cross_k_allow_lens\": l_full,
                \"ws_state\": ws_state,
            }
"""


INCREMENTAL_WS_GENERATION_BLOCK = """        if ws_state is not None:
            if input_ids is None:
                raise ValueError(\"ws_state generation requires input_ids\")

            use_cache = False
            past_key_values = None
            cache_position = None

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

            cur_len = int(input_ids.shape[1])
            prev_len = int(ws_state[\"prev_len\"])
            if cur_len > prev_len:
                new_ids = input_ids[0, prev_len:cur_len].tolist()
                for _tid in new_ids:
                    ws_state[\"L_per_token\"].append(int(ws_state[\"L_cur\"]))
                ws_state[\"prev_len\"] = cur_len

            dec_tok = ws_state[\"dec_tok\"]
            dec_text_now = dec_tok.decode(input_ids[0].tolist(), skip_special_tokens=True)

            ws_pos = last_whitespace_boundary(dec_text_now)
            if ws_pos > len(ws_state[\"enc_text\"]):
                ws_state[\"enc_text\"] = dec_text_now[:ws_pos]

                enc_dev_raw = ws_state.get(\"enc_device\", input_ids.device)
                enc_dev = torch.device(enc_dev_raw) if not isinstance(enc_dev_raw, torch.device) else enc_dev_raw
                enc_ids, enc_mask = prepare_encoder_inputs(
                    ws_state[\"enc_text\"],
                    ws_state[\"enc_tok\"],
                    ws_state[\"enc_max_len\"],
                    device=enc_dev,
                )

                ws_state[\"K_eff\"] = int(enc_mask.sum().item())
                min_pref = int(ws_state[\"min_prefix\"])
                enc_max_len = int(ws_state[\"enc_max_len\"])
                ws_state[\"L_cur\"] = max(min_pref, min(ws_state[\"K_eff\"], enc_max_len))

                with torch.no_grad():
                    enc_out = self.encoder(input_ids=enc_ids, attention_mask=enc_mask, causal=True)

                ws_state[\"encoder_hidden_states\"] = enc_out.last_hidden_state.to(input_ids.device, non_blocking=True)
                ws_state[\"encoder_attention_mask\"] = enc_mask.to(input_ids.device, non_blocking=True)
                ws_state[\"encoder_input_ids\"] = enc_ids
                ws_state[\"encoder_attention_mask_enc\"] = enc_mask

            l_full = torch.tensor([ws_state[\"L_per_token\"]], device=input_ids.device, dtype=torch.long)

            return {
                \"input_ids\": input_ids,
                \"attention_mask\": attention_mask,
                \"past_key_values\": None,
                \"use_cache\": False,
                \"cache_position\": None,
                \"position_ids\": None,
                \"logits_to_keep\": 1,
                \"encoder_hidden_states\": ws_state.get(\"encoder_hidden_states\", None),
                \"encoder_attention_mask\": ws_state.get(\"encoder_attention_mask\", None),
                \"encoder_input_ids\": None,
                \"cross_k_allow_lens\": l_full,
                \"ws_state\": ws_state,
            }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more checkpoints into KRong HF remote-code packages "
            "using a reference checkpoint template such as checkpoint-19000."
        )
    )
    parser.add_argument(
        "--style",
        choices=sorted(STYLE_SPECS.keys()),
        default=DEFAULT_STYLE,
        help=(
            "Generation-policy preset to use while keeping the KRong HF package layout. "
            "`checkpoint19000` uses the static recompute helper, "
            "`krong_llm` uses the older incremental helper."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        help="Checkpoint directory to convert. Can be repeated.",
    )
    parser.add_argument(
        "--source-root",
        default="",
        help="Root directory that contains multiple checkpoint-* directories.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="checkpoint-*",
        help="Glob pattern used with --source-root.",
    )
    parser.add_argument(
        "--template-dir",
        default="",
        help=(
            "Optional override for the template directory. "
            "When omitted, the default template for --style is used."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="When not using --inplace, converted checkpoints are written under this directory.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Patch each checkpoint in place instead of copying to --output-root.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="When writing to --output-root, remove an existing destination directory first.",
    )
    parser.add_argument(
        "--exclude-training-artifacts",
        action="store_true",
        help="When copying to --output-root, skip trainer/resume state artifacts.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help=(
            "Override for attn_implementation/self_attn_backend/_attn_implementation. "
            "Use sdpa for maximum portability across servers."
        ),
    )
    parser.add_argument(
        "--enc-subfolder",
        default="",
        help="Override processor_config.enc_subfolder. Default: template value.",
    )
    parser.add_argument(
        "--enc-max-len",
        type=int,
        default=-1,
        help="Override processor_config.enc_max_len. Default: template value.",
    )
    parser.add_argument(
        "--min-prefix",
        type=int,
        default=-1,
        help="Override processor_config.min_prefix. Default: template value.",
    )
    parser.add_argument(
        "--add-bos",
        choices=["true", "false", ""],
        default="",
        help="Override processor_config.add_bos. Default: template value.",
    )
    parser.add_argument(
        "--copy-template-generation-config",
        action="store_true",
        help="Always overwrite generation_config.json from the template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying files.",
    )
    return parser.parse_args()


def resolve_style(args: argparse.Namespace) -> TemplateStyle:
    try:
        return STYLE_SPECS[args.style]
    except KeyError as exc:
        raise ValueError(f"Unsupported style: {args.style}") from exc


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        log(f"[dry-run] write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        log(f"[dry-run] copy {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        log(f"[dry-run] copy tree {src} -> {dst}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def write_text(path: Path, text: str, *, dry_run: bool) -> None:
    if dry_run:
        log(f"[dry-run] write text {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def remove_tree(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        log(f"[dry-run] remove tree {path}")
        return
    shutil.rmtree(path)


def ignore_for_training_artifacts(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if any(fnmatch.fnmatch(name, pattern) for pattern in TRAINING_ARTIFACT_PATTERNS):
            ignored.add(name)
    return ignored


def copy_checkpoint_tree(
    src: Path,
    dst: Path,
    *,
    exclude_training_artifacts: bool,
    overwrite_output: bool,
    dry_run: bool,
) -> None:
    if dst.exists():
        if not overwrite_output:
            raise FileExistsError(f"Destination already exists: {dst}")
        remove_tree(dst, dry_run=dry_run)

    ignore = ignore_for_training_artifacts if exclude_training_artifacts else None
    if dry_run:
        log(f"[dry-run] copy checkpoint tree {src} -> {dst}")
        return
    shutil.copytree(src, dst, ignore=ignore)


def collect_checkpoint_dirs(args: argparse.Namespace, template_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    for item in args.checkpoint_dir:
        path = Path(item).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")
        dirs.append(path)

    if args.source_root:
        source_root = Path(args.source_root).expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        for path in sorted(source_root.glob(args.checkpoint_pattern)):
            if path.is_dir():
                dirs.append(path.resolve())

    deduped: list[Path] = []
    seen = set()
    for path in dirs:
        if path == template_dir:
            continue
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    if not deduped:
        raise ValueError("No checkpoint directories were provided.")
    return deduped


def validate_template_dir(template_dir: Path, style: TemplateStyle) -> None:
    required = [template_dir / name for name in style.required_template_files]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Template directory is missing required files: {missing_text}")


def load_template_processor_config(template_dir: Path, style: TemplateStyle) -> dict[str, Any]:
    processor_path = template_dir / "processor_config.json"
    if processor_path.exists():
        return read_json(processor_path)
    if style.synthetic_processor_config is not None:
        return copy.deepcopy(style.synthetic_processor_config)
    raise FileNotFoundError(f"processor_config.json not found in template: {template_dir}")


def merge_config(template_config: dict[str, Any], source_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = copy.deepcopy(template_config)
    merged.update(source_config)

    for key in ("architectures", "model_type", "auto_map"):
        if key in template_config:
            merged[key] = copy.deepcopy(template_config[key])

    attn_override = (args.attn_implementation or "").strip()
    if attn_override:
        merged["attn_implementation"] = attn_override
        merged["_attn_implementation"] = attn_override
        merged["self_attn_backend"] = attn_override
    else:
        for key in ("attn_implementation", "_attn_implementation", "self_attn_backend"):
            if key not in merged and key in template_config:
                merged[key] = template_config[key]

    return merged


def merge_processor_config(
    template_config: dict[str, Any],
    existing_config: dict[str, Any],
    args: argparse.Namespace,
    style: TemplateStyle,
) -> dict[str, Any]:
    merged = copy.deepcopy(template_config)
    merged.update(existing_config)

    for key in style.processor_force_keys:
        if key in template_config:
            merged[key] = copy.deepcopy(template_config[key])

    if args.enc_subfolder:
        merged["enc_subfolder"] = args.enc_subfolder
    if args.enc_max_len >= 0:
        merged["enc_max_len"] = int(args.enc_max_len)
    if args.min_prefix >= 0:
        merged["min_prefix"] = int(args.min_prefix)
    if args.add_bos == "true":
        merged["add_bos"] = True
    elif args.add_bos == "false":
        merged["add_bos"] = False

    return merged


def merge_tokenizer_config(template_config: dict[str, Any], existing_config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(template_config)
    merged.update(existing_config)
    # Converted local checkpoints may carry the training-time TokenizersBackend
    # class name, which AutoTokenizer cannot import. Normalize them into a
    # vanilla HF fast-tokenizer package so evaluation loads without fallback.
    merged.pop("backend", None)
    merged.pop("is_local", None)
    merged["tokenizer_class"] = "PreTrainedTokenizerFast"
    merged["clean_up_tokenization_spaces"] = False
    merged.setdefault("model_input_names", ["input_ids", "attention_mask"])
    return merged


def build_special_tokens_map(tokenizer_config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token = tokenizer_config.get(key)
        if not token:
            continue
        out[key] = {
            "content": token,
            "lstrip": False,
            "normalized": True,
            "rstrip": False,
            "single_word": False,
        }
    return out


def write_conversion_metadata(
    dest_dir: Path,
    template_dir: Path,
    source_dir: Path,
    *,
    style: TemplateStyle,
    dry_run: bool,
) -> None:
    data = {
        "converted_at_utc": now_utc_iso(),
        "style": style.name,
        "generation_mode": style.generation_mode,
        "template_dir": str(template_dir),
        "source_dir": str(source_dir),
    }
    write_json(dest_dir / ".krong_hf_converted.json", data, dry_run=dry_run)


def apply_generation_mode_patch(dest_dir: Path, style: TemplateStyle, *, dry_run: bool) -> None:
    if style.generation_mode == "static":
        return

    modeling_path = dest_dir / "_modeling_krong.py"
    if dry_run:
        log(f"[dry-run] patch generation mode {style.generation_mode} in {modeling_path}")
        return
    if not modeling_path.exists():
        raise FileNotFoundError(f"_modeling_krong.py not found for generation patch: {modeling_path}")

    text = modeling_path.read_text(encoding="utf-8")
    old_import = "from ._processing_krong import build_prompt_ids_and_L_list, prepare_encoder_inputs\n"
    new_import = "from ._processing_krong import build_prompt_ids_and_L_list, last_whitespace_boundary, prepare_encoder_inputs\n"
    if old_import in text:
        text = text.replace(old_import, new_import, 1)

    if STATIC_WS_GENERATION_BLOCK not in text:
        raise RuntimeError(
            f"Could not find the static ws_state generation block in {modeling_path}. "
            "The template may have changed and the converter patch needs to be updated."
        )
    text = text.replace(STATIC_WS_GENERATION_BLOCK, INCREMENTAL_WS_GENERATION_BLOCK, 1)
    write_text(modeling_path, text, dry_run=dry_run)


def patch_checkpoint(
    source_dir: Path,
    dest_dir: Path,
    template_dir: Path,
    args: argparse.Namespace,
    style: TemplateStyle,
    template_config: dict[str, Any],
    template_processor_config: dict[str, Any],
    template_tokenizer_config: dict[str, Any],
) -> None:
    read_dir = dest_dir
    if args.dry_run and not args.inplace and not dest_dir.exists():
        read_dir = source_dir

    config_path = read_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in checkpoint: {read_dir}")

    source_config = read_json(config_path)
    merged_config = merge_config(template_config, source_config, args)
    write_json(dest_dir / "config.json", merged_config, dry_run=args.dry_run)

    processor_path = read_dir / "processor_config.json"
    existing_processor_config = read_json(processor_path) if processor_path.exists() else {}
    merged_processor_config = merge_processor_config(
        template_processor_config,
        existing_processor_config,
        args,
        style,
    )
    write_json(dest_dir / "processor_config.json", merged_processor_config, dry_run=args.dry_run)

    tokenizer_config_path = read_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists() or template_tokenizer_config:
        existing_tokenizer_config = read_json(tokenizer_config_path) if tokenizer_config_path.exists() else {}
        merged_tokenizer_config = merge_tokenizer_config(template_tokenizer_config, existing_tokenizer_config)
        write_json(dest_dir / "tokenizer_config.json", merged_tokenizer_config, dry_run=args.dry_run)
        special_tokens_map = build_special_tokens_map(merged_tokenizer_config)
        if special_tokens_map:
            write_json(dest_dir / "special_tokens_map.json", special_tokens_map, dry_run=args.dry_run)

    for name in style.remote_code_files:
        copy_file(template_dir / name, dest_dir / name, dry_run=args.dry_run)

    apply_generation_mode_patch(dest_dir, style, dry_run=args.dry_run)

    for name in style.optional_template_files:
        src = template_dir / name
        dst = dest_dir / name
        if src.exists() and (args.copy_template_generation_config or not dst.exists()):
            copy_file(src, dst, dry_run=args.dry_run)

    template_enc_subfolder = merged_processor_config.get("enc_subfolder", template_processor_config.get("enc_subfolder", "krong_tokenizer"))
    copy_tree(template_dir / "krong_tokenizer", dest_dir / str(template_enc_subfolder), dry_run=args.dry_run)

    write_conversion_metadata(
        dest_dir,
        template_dir,
        source_dir,
        style=style,
        dry_run=args.dry_run,
    )


def main() -> int:
    args = parse_args()
    style = resolve_style(args)
    template_dir = (
        Path(args.template_dir).expanduser().resolve()
        if args.template_dir
        else style.default_template_dir.resolve()
    )
    validate_template_dir(template_dir, style)

    if not args.inplace and not args.output_root:
        raise ValueError("Either --inplace or --output-root must be provided.")

    checkpoint_dirs = collect_checkpoint_dirs(args, template_dir)
    template_config = read_json(template_dir / "config.json")
    template_processor_config = load_template_processor_config(template_dir, style)
    template_tokenizer_config = read_json(template_dir / "tokenizer_config.json") if (template_dir / "tokenizer_config.json").exists() else {}

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = None

    log(f"[style] {style.name}")
    log(f"[template] {template_dir}")
    log(f"[checkpoints] {len(checkpoint_dirs)} item(s)")

    for source_dir in checkpoint_dirs:
        dest_dir = source_dir if args.inplace else output_root / source_dir.name
        log(f"[convert] {source_dir} -> {dest_dir}")

        if not args.inplace:
            copy_checkpoint_tree(
                source_dir,
                dest_dir,
                exclude_training_artifacts=args.exclude_training_artifacts,
                overwrite_output=args.overwrite_output,
                dry_run=args.dry_run,
            )

        patch_checkpoint(
            source_dir=source_dir,
            dest_dir=dest_dir,
            template_dir=template_dir,
            args=args,
            style=style,
            template_config=template_config,
            template_processor_config=template_processor_config,
            template_tokenizer_config=template_tokenizer_config,
        )

    log("[done] conversion finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
