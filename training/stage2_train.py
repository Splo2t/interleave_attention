#!/usr/bin/env python3
"""Stage-2 interleave/vanilla CPT training reference script.

This file is included as artifact code for the paper. It intentionally does not
contain credentials or private W&B tokens. Configure logging through environment
variables or command-line flags when running it locally.

Required local training modules:
  - kormo.train.arguments / kormo.train.trainer
  - krong_tokenizer
  - llama_interleave

The evaluation repository can be used without these training-only modules.
"""
from __future__ import annotations

import glob
import math
import os
import re
import time
from argparse import ArgumentParser
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from kiwipiepy import Kiwi
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizer, Trainer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def _repo_path_if_exists(*parts: str) -> Optional[str]:
    candidate = REPO_ROOT.joinpath(*parts)
    return str(candidate) if candidate.exists() else None


_KIWI_BY_PID: Dict[int, Kiwi] = {}
_JAMO_TABLE: Optional[List[str]] = None
STAGE2_EXTRA_BATCH_KEYS = (
    "encoder_input_ids",
    "encoder_attention_mask",
    "cross_k_allow_lens",
    "encoder_mlm_input_ids",
    "encoder_mlm_labels",
)


def get_kiwi() -> Kiwi:
    pid = os.getpid()
    kiwi = _KIWI_BY_PID.get(pid)
    if kiwi is None:
        kiwi = Kiwi(
            num_workers=1,
            model_path=None,
            load_default_dict=True,
            integrate_allomorph=True,
            model_type="sbg",
        )
        _KIWI_BY_PID[pid] = kiwi
    return kiwi


def _build_jamo_table() -> List[str]:
    base, cho, jung = 0xAC00, 588, 28
    chosung = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
    jungsung = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
    jongsung = ["⑨", 0x11A8, 0x11A9, 0x11AA, 0x11AB, 0x11AC, 0x11AD, 0x11AE, 0x11AF, 0x11B0, 0x11B1, 0x11B2, 0x11B3, 0x11B4, 0x11B5, 0x11B6, 0x11B7, 0x11B8, 0x11B9, 0x11BA, 0x11BB, 0x11BC, 0x11BD, 0x11BE, 0x11BF, 0x11C0, 0x11C1, 0x11C2]
    table = []
    for idx in range(0xD7A3 - base + 1):
        cp = base + idx
        x = cp - base
        c1 = x // cho
        c2 = (x - cho * c1) // jung
        c3 = x - cho * c1 - jung * c2
        j3 = "⑨" if c3 == 0 else chr(jongsung[c3])
        table.append(chosung[c1] + jungsung[c2] + j3)
    return table


def convert_fast(text: str) -> str:
    global _JAMO_TABLE
    if _JAMO_TABLE is None:
        _JAMO_TABLE = _build_jamo_table()
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3:
            out.append(_JAMO_TABLE[cp - 0xAC00])
        else:
            out.append(ch)
    return "".join(out)


def _ensure_str(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _fallback_prefix_map_from_whitespace(text: str, enc_tokenizer: PreTrainedTokenizer, reason: Optional[str] = None) -> Tuple[List[int], List[int], List[int]]:
    bos = 1 if enc_tokenizer.cls_token_id is not None else 0
    eos = 1 if getattr(enc_tokenizer, "sep_token_id", None) is not None else 0
    ws_ends = [m.end() for m in re.finditer(r"\S+\s*", text)]
    if reason is not None:
        print(f"[prefix-map] whitespace fallback: {reason}")

    full_conv = convert_fast(text)
    full_core = enc_tokenizer(full_conv, add_special_tokens=False)["input_ids"]
    full_ids = ([enc_tokenizer.cls_token_id] if bos else []) + full_core + ([enc_tokenizer.sep_token_id] if eos else [])

    enc_len_at_ws = []
    for we in ws_ends:
        pref = convert_fast(text[:we])
        pref_len = len(enc_tokenizer(pref, add_special_tokens=False)["input_ids"])
        enc_len_at_ws.append(bos + pref_len)
    return full_ids, ws_ends, enc_len_at_ws


def encode_full_and_build_prefix_map(text: str, enc_tokenizer: PreTrainedTokenizer) -> Tuple[List[int], List[int], List[int]]:
    """Build the encoder prefix schedule used by whitespace-aligned interleave."""
    kiwi = get_kiwi()
    text = _ensure_str(text).replace("\x00", "")
    if not text.strip():
        full_ids = []
        if enc_tokenizer.cls_token_id is not None:
            full_ids.append(enc_tokenizer.cls_token_id)
        if getattr(enc_tokenizer, "sep_token_id", None) is not None:
            full_ids.append(enc_tokenizer.sep_token_id)
        return full_ids, [], []

    try:
        toks_iter = kiwi.tokenize(text)
        try:
            toks = list(toks_iter) if toks_iter is not None else []
        except TypeError:
            toks = [toks_iter] if toks_iter is not None else []
        if toks and isinstance(toks[0], (list, tuple)):
            flat = []
            for seg in toks:
                flat.extend(seg if isinstance(seg, (list, tuple)) else [seg])
            toks = flat

        forms: List[str] = []
        ends: List[int] = []
        for tok in toks:
            form = getattr(tok, "form", None)
            if form is not None:
                forms.append(str(form))
                end = getattr(tok, "end", None)
                start = getattr(tok, "start", None)
                length = getattr(tok, "len", None)
                if end is not None:
                    ends.append(int(end))
                elif start is not None and length is not None:
                    ends.append(int(start) + int(length))
                else:
                    ends.append(-1)
            elif isinstance(tok, (list, tuple)):
                if len(tok) >= 4 and isinstance(tok[2], int) and isinstance(tok[3], int):
                    forms.append(str(tok[0]))
                    ends.append(int(tok[2]) + int(tok[3]))
                elif len(tok) >= 3 and isinstance(tok[1], int) and isinstance(tok[2], int):
                    forms.append(str(tok[0]))
                    ends.append(int(tok[2]))
                elif len(tok) >= 1:
                    forms.append(str(tok[0]))
                    ends.append(-1)
            else:
                forms.append(str(tok))
                ends.append(-1)
    except Exception as exc:  # pragma: no cover - defensive fallback path
        return _fallback_prefix_map_from_whitespace(text, enc_tokenizer, reason=f"kiwi tokenize error: {type(exc).__name__}")

    if not forms:
        return _fallback_prefix_map_from_whitespace(text, enc_tokenizer, reason="empty kiwi tokenization")

    if any(end < 0 for end in ends):
        ws_fix = [m.end() for m in re.finditer(r"\S+\s*", text)]
        for i in range(min(len(ends), len(ws_fix))):
            if ends[i] < 0:
                ends[i] = ws_fix[i]
        ends = [end if end >= 0 else len(text) for end in ends]

    conv = [convert_fast(form) for form in forms]
    enc_batch = enc_tokenizer(conv, add_special_tokens=False)
    ids_list = enc_batch.get("input_ids", []) if hasattr(enc_batch, "get") else enc_batch
    need_fallback = (not ids_list) or (len(ids_list) != len(forms)) or (sum(len(ids) for ids in ids_list) == 0)
    if need_fallback:
        return _fallback_prefix_map_from_whitespace(text, enc_tokenizer, reason="morph tokenization mismatch")

    bos = 1 if enc_tokenizer.cls_token_id is not None else 0
    eos = 1 if getattr(enc_tokenizer, "sep_token_id", None) is not None else 0
    per_len = [len(ids) for ids in ids_list]
    cumsum = [0]
    for length in per_len:
        cumsum.append(cumsum[-1] + length)

    full_ids: List[int] = []
    if bos:
        full_ids.append(enc_tokenizer.cls_token_id)
    for ids in ids_list:
        full_ids.extend(ids)
    if eos:
        full_ids.append(enc_tokenizer.sep_token_id)

    ws_ends = [m.end() for m in re.finditer(r"\S+\s*", text)]
    enc_len_at_ws = []
    nforms = len(forms)
    for we in ws_ends:
        idx = bisect_right(ends, we) - 1
        idx = min(max(idx, -1), nforms - 1)
        enc_len_at_ws.append(bos + (cumsum[idx + 1] if idx >= 0 else 0))
    return full_ids, ws_ends, enc_len_at_ws


def _extract_ends_from_offset_mapping(offset_mapping: Any) -> List[int]:
    offs = offset_mapping
    if offs is None:
        return []
    if isinstance(offs, torch.Tensor):
        offs = offs.tolist()
    try:
        if len(offs) > 0 and isinstance(offs[0], (list, tuple)) and len(offs[0]) > 0 and isinstance(offs[0][0], (list, tuple)):
            offs = offs[0]
    except TypeError:
        return []

    ends = []
    for item in offs:
        if isinstance(item, (list, tuple)):
            if len(item) >= 2 and item[1] is not None:
                ends.append(int(item[1]))
            elif len(item) == 1 and item[0] is not None:
                ends.append(int(item[0]))
            else:
                ends.append(0)
        else:
            try:
                ends.append(int(item))
            except Exception:
                ends.append(0)
    return ends


def use_pure_hf_vanilla(args: Any) -> bool:
    return getattr(args, "model_type", None) == "vanilla" and bool(getattr(args, "vanilla_pure_hf", False))


def list_data_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        if os.path.isdir(pattern):
            for ext in ("*.arrow", "*.parquet", "*.jsonl", "*.json"):
                files.extend(glob.glob(os.path.join(pattern, ext)))
        else:
            matches = glob.glob(pattern)
            if matches:
                files.extend(matches)
            elif os.path.isfile(pattern):
                files.append(pattern)
    return sorted(set(files))


def load_and_normalize(files_or_patterns: List[str], cache_dir: str):
    files = list_data_files(files_or_patterns)
    if not files:
        raise FileNotFoundError(f"No dataset files matched: {files_or_patterns}")

    by_ext = {
        ".arrow": [f for f in files if f.endswith(".arrow")],
        ".parquet": [f for f in files if f.endswith(".parquet")],
        ".jsonl": [f for f in files if f.endswith(".jsonl")],
        ".json": [f for f in files if f.endswith(".json")],
    }
    if by_ext[".arrow"]:
        kind, selected = "arrow", by_ext[".arrow"]
    elif by_ext[".parquet"]:
        kind, selected = "parquet", by_ext[".parquet"]
    elif by_ext[".jsonl"]:
        kind, selected = "json", by_ext[".jsonl"]
    else:
        kind, selected = "json", by_ext[".json"]

    ds = load_dataset(kind, data_files=selected, split="train", cache_dir=cache_dir)
    print(f"[Data] loaded {len(ds):,} rows with columns: {ds.column_names}")

    text_candidates = [c for c in ("text", "content", "contents", "body", "sentence", "raw", "document", "code") if c in ds.column_names]
    if text_candidates:
        if "text" not in ds.column_names:
            ds = ds.rename_column(text_candidates[0], "text")
    else:
        raise ValueError(f"No text-like column in dataset columns: {ds.column_names}")

    if "id" not in ds.column_names:
        if "hexsha" in ds.column_names:
            ds = ds.rename_column("hexsha", "id")
        else:
            ds = ds.add_column("id", [f"scd_{i}" for i in range(len(ds))])

    keep = [c for c in ("id", "text") if c in ds.column_names]
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def load_and_prepare_dataset(args: Any, dec_tok: PreTrainedTokenizer, enc_tok: Optional[PreTrainedTokenizer]):
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    decoder_only = use_pure_hf_vanilla(args)
    if not decoder_only and enc_tok is None:
        raise ValueError("Encoder tokenizer is required for interleave training.")

    if args.merged_dataset is not None:
        print(f"[Data] using merged dataset from {args.merged_dataset}")
        ds = load_and_normalize([args.merged_dataset], cache_dir=cache_dir).shuffle(seed=42)
    else:
        stage2_data_root = getattr(args, "stage2_data_root", None) or os.environ.get("KRONG_STAGE2_DATA_ROOT")
        if not stage2_data_root:
            default_merged = _repo_path_if_exists("merged_arrow") or _repo_path("merged_arrow")
            raise FileNotFoundError(
                "Local multi-domain dataset root is not configured. "
                f"Use --merged_dataset {default_merged!r}, or set --stage2_data_root / KRONG_STAGE2_DATA_ROOT."
            )
        stage2_data_root = os.path.abspath(os.path.expanduser(stage2_data_root))
        domain_specs = {
            "ko_cosmopedia": [
                os.path.join(stage2_data_root, "eos_ko_cosmopedia_packed_4096tok", "*.arrow"),
                os.path.join(stage2_data_root, "eos_ko_cosmopedia_packed_4096tok", "*.parquet"),
            ],
            "en_ufw": [
                os.path.join(stage2_data_root, "eos_ufw_en_3parquets_packed4096tok_parallel_text", "*.arrow"),
                os.path.join(stage2_data_root, "eos_ufw_en_3parquets_packed4096tok_parallel_text", "*.parquet"),
            ],
            "hindi_indicCorp": [
                os.path.join(stage2_data_root, "eos_IndicCorpV2_hin_Deva_samples_2Btok_packed4096tok_multiproc", "*.arrow"),
                os.path.join(stage2_data_root, "eos_IndicCorpV2_hin_Deva_samples_2Btok_packed4096tok_multiproc", "*.parquet"),
            ],
            "eus_craw": [
                os.path.join(stage2_data_root, "eos_euscrawl_full_packed4096tok", "*.arrow"),
                os.path.join(stage2_data_root, "eos_euscrawl_full_packed4096tok", "*.parquet"),
            ],
        }
        rng = np.random.default_rng(42)
        parts = []
        for name in reversed(list(domain_specs.keys())):
            dset = load_and_normalize(domain_specs[name], cache_dir=cache_dir)
            size = len(dset)
            print(f"[Data] sampling {size:,} docs from {name}")
            idx = rng.choice(len(dset), size=size, replace=False)
            parts.append(dset.select(idx))
        ds = concatenate_datasets(parts).shuffle(seed=42)

    def _maybe_add_row_boundary_tokens(dids: List[int], ends: List[int], text_len: int) -> Tuple[List[int], List[int]]:
        dids = list(dids)
        ends = list(ends)
        if not args.add_row_boundary_tokens:
            return dids, ends
        if dec_tok.bos_token_id is not None:
            dids = [dec_tok.bos_token_id] + dids
            ends = [0] + ends
        if dec_tok.eos_token_id is not None:
            dids = dids + [dec_tok.eos_token_id]
            ends = ends + [text_len]
        return dids, ends

    def _set_transform_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = examples.get("text", [])
        if not isinstance(texts, list):
            texts = [texts]
        out: Dict[str, List[Any]] = {"raw_text": [], "dec_input_ids": [], "dec_endpos": []}
        if not decoder_only:
            out.update({"enc_full_ids": [], "ws_ends": [], "enc_len_at_ws": []})

        for text in texts:
            text = _ensure_str(text)
            enc_dec = dec_tok(text, add_special_tokens=False, return_offsets_mapping=True)
            dids = enc_dec["input_ids"]
            ends = _extract_ends_from_offset_mapping(enc_dec.get("offset_mapping", []))
            dids, ends = _maybe_add_row_boundary_tokens(dids, ends, len(text))
            out["raw_text"].append(text)
            out["dec_input_ids"].append(dids)
            out["dec_endpos"].append(ends)
            if not decoder_only:
                full_ids, ws_ends, enc_len_at_ws = encode_full_and_build_prefix_map(text, enc_tok)
                out["enc_full_ids"].append(full_ids)
                out["ws_ends"].append(ws_ends)
                out["enc_len_at_ws"].append(enc_len_at_ws)
        return out

    ds.set_transform(_set_transform_batch)
    return ds


@dataclass
class PackedWSInterleaveCollator:
    dec_tokenizer: PreTrainedTokenizer
    enc_tokenizer: PreTrainedTokenizer
    dec_seq_len: int = 4096
    enc_seq_len: int = 4096
    sequences_per_batch: int = 4
    enc_min_prefix: int = 8
    boundary_silence: int = 0
    overfetch_factor: int = 8
    min_fill_ratio: float = 0.90
    make_mlm: bool = False
    _stash: Deque[Dict[str, Any]] = field(default_factory=deque, init=False, repr=False)
    rows_per_step: Optional[int] = None

    @staticmethod
    def _pad_to(ids: List[int], pad_id: int, tgt_len: int) -> torch.Tensor:
        out = torch.full((tgt_len,), pad_id, dtype=torch.long)
        if ids:
            n = min(tgt_len, len(ids))
            out[:n] = torch.tensor(ids[:n], dtype=torch.long)
        return out

    def _flatten_instances(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat = []
        for inst in instances:
            d_ids = inst.get("dec_input_ids", [])
            if d_ids and isinstance(d_ids[0], list):
                for i in range(len(d_ids)):
                    flat.append({
                        "raw_text": inst.get("raw_text", [""])[i] if isinstance(inst.get("raw_text"), list) else inst.get("raw_text", ""),
                        "dec_input_ids": inst["dec_input_ids"][i],
                        "dec_endpos": inst["dec_endpos"][i],
                        "enc_full_ids": inst["enc_full_ids"][i],
                        "ws_ends": inst["ws_ends"][i],
                        "enc_len_at_ws": inst["enc_len_at_ws"][i],
                    })
            else:
                flat.append(inst)
        return flat

    def _empty_row(self) -> Dict[str, torch.Tensor]:
        dec_pad = self.dec_tokenizer.pad_token_id or self.dec_tokenizer.eos_token_id
        enc_pad = self.enc_tokenizer.pad_token_id
        dec = torch.full((self.dec_seq_len,), dec_pad, dtype=torch.long)
        enc = torch.full((self.enc_seq_len,), enc_pad, dtype=torch.long)
        dec_mask = torch.zeros((self.dec_seq_len,), dtype=torch.bool)
        enc_mask = torch.zeros((self.enc_seq_len,), dtype=torch.bool)
        labels = dec.clone()
        labels[~dec_mask] = -100
        allow = torch.zeros((self.dec_seq_len,), dtype=torch.int32)
        out = {
            "input_ids": dec,
            "labels": labels,
            "attention_mask": dec_mask,
            "encoder_input_ids": enc,
            "encoder_attention_mask": enc_mask,
            "cross_k_allow_lens": allow,
        }
        if self.make_mlm:
            out["encoder_mlm_input_ids"] = enc.clone()
            out["encoder_mlm_labels"] = torch.full_like(enc, -100)
        return out

    def _pack_one_row(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        d_ids = list(sample["dec_input_ids"])[: self.dec_seq_len]
        d_ends = list(sample["dec_endpos"])[: len(d_ids)]
        e_ids = list(sample["enc_full_ids"])[: self.enc_seq_len]
        ws_ends = list(sample["ws_ends"])
        enc_len_at_ws = list(sample["enc_len_at_ws"])

        dec_pad = self.dec_tokenizer.pad_token_id or self.dec_tokenizer.eos_token_id
        enc_pad = self.enc_tokenizer.pad_token_id
        dec_arr = self._pad_to(d_ids, dec_pad, self.dec_seq_len)
        enc_arr = self._pad_to(e_ids, enc_pad, self.enc_seq_len)

        dec_mask = torch.zeros((self.dec_seq_len,), dtype=torch.bool)
        dec_mask[: len(d_ids)] = True
        enc_mask = torch.zeros((self.enc_seq_len,), dtype=torch.bool)
        enc_mask[: len(e_ids)] = True
        labels = dec_arr.clone()
        labels[~dec_mask] = -100

        if self.boundary_silence > 0:
            bos_id = getattr(self.dec_tokenizer, "bos_token_id", None)
            if bos_id is not None:
                for pos in [i for i, tid in enumerate(d_ids) if tid == bos_id][1:]:
                    labels[pos : min(pos + self.boundary_silence, self.dec_seq_len)] = -100

        if len(d_ids) == 0:
            allow_local: List[int] = []
        elif len(ws_ends) == 0:
            allow_local = [self.enc_min_prefix] * len(d_ids)
        else:
            ce = torch.tensor(d_ends, dtype=torch.int32)
            ws_t = torch.tensor(ws_ends, dtype=torch.int32)
            encws = torch.tensor(enc_len_at_ws, dtype=torch.int32)
            idxs = torch.searchsorted(ws_t, ce, right=True) - 1
            idxs = torch.clamp(idxs, 0, ws_t.numel() - 1)
            allow_local = encws[idxs].tolist()

        enc_cap_now = min(len(e_ids), self.enc_seq_len)
        cross_allow = [min(enc_cap_now, max(int(length), int(self.enc_min_prefix))) for length in allow_local]
        allow = torch.tensor(cross_allow + [0] * (self.dec_seq_len - len(cross_allow)), dtype=torch.int32)

        out = {
            "input_ids": dec_arr,
            "labels": labels,
            "attention_mask": dec_mask,
            "encoder_input_ids": enc_arr,
            "encoder_attention_mask": enc_mask,
            "cross_k_allow_lens": allow,
        }
        if self.make_mlm:
            mask_id = getattr(self.enc_tokenizer, "mask_token_id", None)
            if mask_id is None:
                raise ValueError("Encoder tokenizer must have mask_token_id for MLM.")
            ids = enc_arr.clone()
            mlm_labels = torch.full_like(ids, -100)
            valid = enc_mask.nonzero(as_tuple=False).squeeze(1)
            if valid.numel() > 0:
                k = max(1, int(valid.numel() * 0.15))
                pick = valid[torch.randperm(valid.numel())[:k]]
                probs = torch.rand(k)
                mask_sel = pick[probs < 0.8]
                rand_sel = pick[(probs >= 0.8) & (probs < 0.9)]
                mlm_labels[pick] = ids[pick]
                if mask_sel.numel():
                    ids[mask_sel] = mask_id
                if rand_sel.numel():
                    ids[rand_sel] = torch.randint(len(self.enc_tokenizer), (rand_sel.numel(),))
            out["encoder_mlm_input_ids"] = ids
            out["encoder_mlm_labels"] = mlm_labels
        return out

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        instances = self._flatten_instances(instances)
        need = self.rows_per_step or self.sequences_per_batch
        rows = [self._pack_one_row(sample) for sample in instances[:need]]
        while len(rows) < need:
            rows.append(self._empty_row())
        return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


@dataclass
class PackedDecoderOnlyCollator:
    dec_tokenizer: PreTrainedTokenizer
    dec_seq_len: int = 4096
    sequences_per_batch: int = 4
    boundary_silence: int = 0
    rows_per_step: Optional[int] = None

    @staticmethod
    def _pad_to(ids: List[int], pad_id: int, tgt_len: int) -> torch.Tensor:
        out = torch.full((tgt_len,), pad_id, dtype=torch.long)
        if ids:
            n = min(tgt_len, len(ids))
            out[:n] = torch.tensor(ids[:n], dtype=torch.long)
        return out

    def _flatten_instances(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat = []
        for inst in instances:
            d_ids = inst.get("dec_input_ids", [])
            if d_ids and isinstance(d_ids[0], list):
                flat.extend({"dec_input_ids": ids} for ids in d_ids)
            else:
                flat.append(inst)
        return flat

    def _empty_row(self) -> Dict[str, torch.Tensor]:
        dec_pad = self.dec_tokenizer.pad_token_id or self.dec_tokenizer.eos_token_id
        dec = torch.full((self.dec_seq_len,), dec_pad, dtype=torch.long)
        mask = torch.zeros((self.dec_seq_len,), dtype=torch.bool)
        labels = dec.clone()
        labels[~mask] = -100
        return {"input_ids": dec, "labels": labels, "attention_mask": mask}

    def _pack_one_row(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        d_ids = list(sample.get("dec_input_ids", []))[: self.dec_seq_len]
        dec_pad = self.dec_tokenizer.pad_token_id or self.dec_tokenizer.eos_token_id
        dec_arr = self._pad_to(d_ids, dec_pad, self.dec_seq_len)
        mask = torch.zeros((self.dec_seq_len,), dtype=torch.bool)
        mask[: len(d_ids)] = True
        labels = dec_arr.clone()
        labels[~mask] = -100
        if self.boundary_silence > 0:
            bos_id = getattr(self.dec_tokenizer, "bos_token_id", None)
            if bos_id is not None:
                for pos in [i for i, tid in enumerate(d_ids) if tid == bos_id][1:]:
                    labels[pos : min(pos + self.boundary_silence, self.dec_seq_len)] = -100
        return {"input_ids": dec_arr, "labels": labels, "attention_mask": mask}

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        instances = self._flatten_instances(instances)
        need = self.rows_per_step or self.sequences_per_batch
        rows = [self._pack_one_row(sample) for sample in instances[:need]]
        while len(rows) < need:
            rows.append(self._empty_row())
        return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def build_collator(args: Any, dec_tok: PreTrainedTokenizer, enc_tok: Optional[PreTrainedTokenizer]):
    if use_pure_hf_vanilla(args):
        return PackedDecoderOnlyCollator(
            dec_tokenizer=dec_tok,
            dec_seq_len=args.dec_len,
            sequences_per_batch=args.per_device_train_batch_size,
            boundary_silence=args.boundary_silence,
        )
    if enc_tok is None:
        raise ValueError("enc_tok is required for interleave training")
    return PackedWSInterleaveCollator(
        dec_tokenizer=dec_tok,
        enc_tokenizer=enc_tok,
        dec_seq_len=args.dec_len,
        enc_seq_len=args.enc_len,
        sequences_per_batch=args.per_device_train_batch_size,
        enc_min_prefix=args.min_prefix,
        overfetch_factor=6,
        boundary_silence=args.boundary_silence,
        make_mlm=(float(args.kd_mlm) > 0.0),
        min_fill_ratio=0.9,
    )


def force_untie_word_embeddings(model):
    model.config.tie_word_embeddings = False
    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    set_out = getattr(model, "set_output_embeddings", None)
    if get_in is None or get_out is None or set_out is None:
        return model
    input_emb = get_in()
    output_emb = get_out()
    if input_emb is None or output_emb is None:
        return model
    if not hasattr(input_emb, "weight") or not hasattr(output_emb, "weight"):
        return model
    if input_emb.weight.data_ptr() != output_emb.weight.data_ptr():
        return model
    new_output = nn.Linear(
        output_emb.in_features,
        output_emb.out_features,
        bias=output_emb.bias is not None,
        device=output_emb.weight.device,
        dtype=output_emb.weight.dtype,
    )
    with torch.no_grad():
        new_output.weight.copy_(output_emb.weight)
        if output_emb.bias is not None:
            new_output.bias.copy_(output_emb.bias)
    set_out(new_output)
    return model


def initialize_lm_head_from_input_embeddings(model):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is None or output_emb is None:
        raise ValueError("Model must expose input and output embeddings.")
    if input_emb.weight.shape != output_emb.weight.shape:
        raise ValueError(f"Cannot copy input embeddings to lm_head: {tuple(input_emb.weight.shape)} vs {tuple(output_emb.weight.shape)}")
    with torch.no_grad():
        output_emb.weight.copy_(input_emb.weight.to(device=output_emb.weight.device, dtype=output_emb.weight.dtype))


def initialize_lm_head_random(model):
    output_emb = model.get_output_embeddings()
    if output_emb is None or not hasattr(output_emb, "weight"):
        raise ValueError("Model must expose output embedding/lm_head weight.")
    std = float(getattr(model.config, "initializer_range", 0.02))
    with torch.no_grad():
        output_emb.weight.normal_(mean=0.0, std=std)
        if getattr(output_emb, "bias", None) is not None:
            output_emb.bias.zero_()


def apply_embedding_init_policy(model, args: Any):
    policy = getattr(args, "lm_head_init", "keep")
    if policy == "keep":
        return
    if policy == "copy_input":
        initialize_lm_head_from_input_embeddings(model)
    elif policy == "random":
        initialize_lm_head_random(model)
    elif policy == "tie_input":
        model.config.tie_word_embeddings = True
        model.tie_weights()
    else:
        raise ValueError(f"Unsupported lm_head_init: {policy}")
    model.config.lm_head_init = policy


class VanillaLlamaForStage2(LlamaForCausalLM):
    def forward(self, *args, **kwargs):
        for key in STAGE2_EXTRA_BATCH_KEYS:
            kwargs.pop(key, None)
        return super().forward(*args, **kwargs)


def resolve_stage1_model_path(args: Any) -> str:
    return args.stage1_model_path or args.dec_tokenizer_repo


def estimate_decoder_layer_params(hidden_size: int, intermediate_size: int, num_attention_heads: int, num_key_value_heads: int) -> int:
    kv_ratio = float(num_key_value_heads) / float(num_attention_heads)
    attn_params = (2.0 + 2.0 * kv_ratio) * hidden_size * hidden_size
    mlp_params = 3.0 * hidden_size * intermediate_size
    norm_params = 2.0 * hidden_size
    return int(attn_params + mlp_params + norm_params)


def estimate_interleave_extra_params(base_cfg: Any, args: Any, enc_vocab_size: int) -> Dict[str, int]:
    dec_h = int(base_cfg.hidden_size)
    dec_i = int(base_cfg.intermediate_size)
    dec_heads = int(base_cfg.num_attention_heads)
    dec_kv_heads = int(base_cfg.num_key_value_heads)
    enc_h = int(args.encoder_hidden_size)
    enc_layers = int(args.encoder_num_layers)
    enc_heads = int(args.encoder_num_attention_heads)
    enc_kv_heads = int(args.encoder_num_key_value_heads)
    enc_i = int(args.encoder_intermediate_size)

    interleave_block_params = estimate_decoder_layer_params(dec_h, dec_i, dec_heads, dec_kv_heads)
    encoder_layer_params = estimate_decoder_layer_params(enc_h, enc_i, enc_heads, enc_kv_heads)
    total_extra = (
        enc_vocab_size * enc_h
        + enc_layers * encoder_layer_params
        + enc_h
        + dec_h
        + enc_h * dec_h
        + enc_vocab_size * enc_h
        + int(args.interleave_n) * interleave_block_params
    )
    return {
        "decoder_layer_params": interleave_block_params,
        "encoder_layer_params": encoder_layer_params,
        "estimated_extra_params": int(total_extra),
    }


def choose_baseline_extra_layers(args: Any, base_cfg: Any, enc_vocab_size: int) -> Dict[str, Any]:
    dec_layer_params = estimate_decoder_layer_params(
        hidden_size=int(base_cfg.hidden_size),
        intermediate_size=int(base_cfg.intermediate_size),
        num_attention_heads=int(base_cfg.num_attention_heads),
        num_key_value_heads=int(base_cfg.num_key_value_heads),
    )
    if args.baseline_extra_layers is not None:
        return {
            "extra_layers": int(args.baseline_extra_layers),
            "decoder_layer_params": dec_layer_params,
            "estimated_extra_params": int(args.baseline_extra_layers) * dec_layer_params,
            "mode": "manual",
        }
    est = estimate_interleave_extra_params(base_cfg, args, enc_vocab_size)
    target = est["estimated_extra_params"]
    lower = max(0, target // dec_layer_params)
    upper = lower + 1
    lower_gap = abs(target - lower * dec_layer_params)
    upper_gap = abs(upper * dec_layer_params - target)
    chosen = max(1, upper if upper_gap < lower_gap else lower)
    est["extra_layers"] = int(chosen)
    est["mode"] = "auto"
    return est


def initialize_added_decoder_layers(model, base_num_layers: int, target_num_layers: int):
    extra_layers = target_num_layers - base_num_layers
    if extra_layers <= 0:
        return
    window = min(extra_layers, base_num_layers)
    start = base_num_layers - window
    for offset, new_idx in enumerate(range(base_num_layers, target_num_layers)):
        src_idx = start + (offset % window)
        model.model.layers[new_idx].load_state_dict(model.model.layers[src_idx].state_dict())


def _copy_tensor_overlap_(dst_tensor: torch.Tensor, src_tensor: torch.Tensor):
    if dst_tensor.ndim != src_tensor.ndim:
        return
    with torch.no_grad():
        src = src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype)
        if dst_tensor.shape == src.shape:
            dst_tensor.copy_(src)
            return
        slices = tuple(slice(0, min(d, s)) for d, s in zip(dst_tensor.shape, src.shape))
        dst_tensor[slices].copy_(src[slices])


def _copy_module_overlap_(dst_module, src_module):
    dst_named = dict(dst_module.named_parameters(recurse=True))
    dst_named.update(dict(dst_module.named_buffers(recurse=True)))
    src_named = dict(src_module.named_parameters(recurse=True))
    src_named.update(dict(src_module.named_buffers(recurse=True)))
    for name, dst_tensor in dst_named.items():
        src_tensor = src_named.get(name)
        if src_tensor is not None:
            _copy_tensor_overlap_(dst_tensor, src_tensor)


def _initialize_linear_identity_(linear_module):
    with torch.no_grad():
        linear_module.weight.zero_()
        rows, cols = linear_module.weight.shape
        diag = min(rows, cols)
        if diag > 0:
            idx = torch.arange(diag, device=linear_module.weight.device)
            linear_module.weight[idx, idx] = 1.0
        if getattr(linear_module, "bias", None) is not None:
            linear_module.bias.zero_()


def _copy_linear_from_linear_(dst_linear, src_linear):
    if dst_linear is None or src_linear is None:
        return
    if getattr(dst_linear, "weight", None) is not None and getattr(src_linear, "weight", None) is not None:
        _copy_tensor_overlap_(dst_linear.weight, src_linear.weight)
    if getattr(dst_linear, "bias", None) is not None and getattr(src_linear, "bias", None) is not None:
        _copy_tensor_overlap_(dst_linear.bias, src_linear.bias)


def _copy_norm_weight_(dst_norm, src_norm):
    if dst_norm is None or src_norm is None:
        return
    if getattr(dst_norm, "weight", None) is not None and getattr(src_norm, "weight", None) is not None:
        _copy_tensor_overlap_(dst_norm.weight, src_norm.weight)
    if getattr(dst_norm, "bias", None) is not None and getattr(src_norm, "bias", None) is not None:
        _copy_tensor_overlap_(dst_norm.bias, src_norm.bias)


def _initialize_interleave_layers_copy_top(model, base_num_layers: int):
    if base_num_layers <= 0:
        return
    interleave_layers = list(getattr(model, "interleave_layers", []))
    if interleave_layers:
        window = min(len(interleave_layers), base_num_layers)
        start = base_num_layers - window
        for offset, block in enumerate(interleave_layers):
            src_idx = start + (offset % window)
            block.layer.load_state_dict(model.model.layers[src_idx].state_dict())


def _initialize_interleave_layers_copy_low(model, base_num_layers: int):
    if base_num_layers <= 0:
        return
    interleave_layers = list(getattr(model, "interleave_layers", []))
    if interleave_layers:
        window = min(len(interleave_layers), base_num_layers)
        for offset, block in enumerate(interleave_layers):
            src_idx = offset % window
            block.layer.load_state_dict(model.model.layers[src_idx].state_dict())


def _initialize_encoder_from_decoder_overlap(model, base_num_layers: int):
    encoder = getattr(model, "encoder", None)
    if encoder is None or base_num_layers <= 0:
        return
    if len(getattr(encoder, "layers", [])) > 0:
        window = min(len(encoder.layers), base_num_layers)
        start = base_num_layers - window
        for offset, enc_layer in enumerate(encoder.layers):
            src_idx = start + (offset % window)
            _copy_module_overlap_(enc_layer, model.model.layers[src_idx])
    if hasattr(model.model, "norm"):
        _copy_module_overlap_(encoder.norm, model.model.norm)
        _copy_module_overlap_(encoder.dec_norm, model.model.norm)
    if hasattr(encoder, "enc_to_dec"):
        _initialize_linear_identity_(encoder.enc_to_dec)


def _detect_external_encoder_layout(source_model):
    if hasattr(source_model, "encoder") and hasattr(source_model.encoder, "layer"):
        return "bert_like", list(source_model.encoder.layer)
    if hasattr(source_model, "model") and hasattr(source_model.model, "layers"):
        return "llama_like", list(source_model.model.layers)
    if hasattr(source_model, "layers"):
        return "llama_like", list(source_model.layers)
    return "unknown", []


def _initialize_llama_encoder_layer_from_bert_layer(dst_layer, src_layer):
    try:
        src_self = src_layer.attention.self
        src_attn_out = src_layer.attention.output
        src_inter = src_layer.intermediate
        src_out = src_layer.output
    except Exception:
        return
    _copy_linear_from_linear_(dst_layer.self_attn.q_proj, src_self.query)
    _copy_linear_from_linear_(dst_layer.self_attn.k_proj, src_self.key)
    _copy_linear_from_linear_(dst_layer.self_attn.v_proj, src_self.value)
    _copy_linear_from_linear_(dst_layer.self_attn.o_proj, src_attn_out.dense)
    _copy_norm_weight_(dst_layer.input_layernorm, getattr(src_attn_out, "LayerNorm", None))
    _copy_norm_weight_(dst_layer.post_attention_layernorm, getattr(src_out, "LayerNorm", None))
    _copy_linear_from_linear_(dst_layer.mlp.up_proj, src_inter.dense)
    _copy_linear_from_linear_(dst_layer.mlp.gate_proj, src_inter.dense)
    _copy_linear_from_linear_(dst_layer.mlp.down_proj, src_out.dense)


def _initialize_encoder_from_external_model(model, source_model, copy_embeddings: bool = False):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    source_embeddings = None
    if hasattr(source_model, "get_input_embeddings"):
        try:
            source_embeddings = source_model.get_input_embeddings()
        except Exception:
            source_embeddings = None
    if copy_embeddings and source_embeddings is not None and getattr(source_embeddings, "weight", None) is not None:
        _copy_tensor_overlap_(encoder.embed_tokens.weight, source_embeddings.weight)

    layout, source_layers = _detect_external_encoder_layout(source_model)
    if source_layers:
        window = min(len(source_layers), len(encoder.layers))
        start = max(0, len(source_layers) - window)
        for offset, enc_layer in enumerate(encoder.layers[:window]):
            src_layer = source_layers[start + offset]
            if layout == "bert_like":
                _initialize_llama_encoder_layer_from_bert_layer(enc_layer, src_layer)
            else:
                _copy_module_overlap_(enc_layer, src_layer)
    if layout == "llama_like":
        src_norm = None
        if hasattr(source_model, "model") and hasattr(source_model.model, "norm"):
            src_norm = source_model.model.norm
        elif hasattr(source_model, "norm"):
            src_norm = source_model.norm
        _copy_norm_weight_(encoder.norm, src_norm)
        _copy_norm_weight_(encoder.dec_norm, src_norm)
    elif layout == "bert_like" and source_layers:
        last_ln = getattr(getattr(source_layers[-1], "output", None), "LayerNorm", None)
        _copy_norm_weight_(encoder.norm, last_ln)
        _copy_norm_weight_(encoder.dec_norm, last_ln)
    if hasattr(encoder, "enc_to_dec"):
        _initialize_linear_identity_(encoder.enc_to_dec)


def resolve_encoder_init_mode(args: Any) -> str:
    if args.encoder_init_mode != "auto":
        return args.encoder_init_mode
    if args.encoder_init_model_path:
        return "external_overlap"
    return "decoder_overlap"


def initialize_interleave_modules(model, base_num_layers: int, interleave_init_mode: str, encoder_init_mode: str, encoder_source_model=None, encoder_init_copy_embeddings: bool = False):
    if interleave_init_mode == "copy_top":
        _initialize_interleave_layers_copy_top(model, base_num_layers)
    elif interleave_init_mode == "copy_low":
        _initialize_interleave_layers_copy_low(model, base_num_layers)
    elif interleave_init_mode != "random":
        raise ValueError(f"Unsupported interleave_init_mode: {interleave_init_mode}")

    if encoder_init_mode == "decoder_overlap":
        _initialize_encoder_from_decoder_overlap(model, base_num_layers)
    elif encoder_init_mode == "external_overlap":
        if encoder_source_model is None:
            raise ValueError("encoder_init_mode='external_overlap' requires encoder_source_model")
        _initialize_encoder_from_external_model(model, encoder_source_model, copy_embeddings=encoder_init_copy_embeddings)
    elif encoder_init_mode == "random":
        return
    else:
        raise ValueError(f"Unsupported encoder_init_mode: {encoder_init_mode}")


def apply_attention_impl(config: Any, attn_impl: str):
    config.attn_implementation = attn_impl
    config._attn_implementation = attn_impl
    config.self_attn_backend = attn_impl
    return config


def load_tokenizers(args: Any, base_ckpt: str, load_encoder: bool = True):
    dec_tok = AutoTokenizer.from_pretrained(base_ckpt, use_fast=True)
    if dec_tok.pad_token_id is None:
        if dec_tok.eos_token_id is None:
            raise ValueError("Decoder tokenizer must have either pad_token_id or eos_token_id.")
        dec_tok.pad_token = dec_tok.eos_token
    if not load_encoder:
        return dec_tok, None
    try:
        from krong_tokenizer import KrongBertTokenizer
    except ImportError as exc:
        raise ImportError("Interleave training requires the local krong_tokenizer package on PYTHONPATH.") from exc
    enc_tok = KrongBertTokenizer.from_pretrained(args.enc_tokenizer_repo, strip_accents=False, lowercase=False)
    if hasattr(enc_tok, "setMorph"):
        enc_tok.setMorph(args.enc_tokenizer_repo)
    return dec_tok, enc_tok


def load_llama_with_interleave(args: Any):
    try:
        from llama_interleave import LlamaForCausalLMWithInterleave
    except ImportError as exc:
        raise ImportError("Interleave training requires llama_interleave.py on PYTHONPATH.") from exc

    base_ckpt = resolve_stage1_model_path(args)
    if float(args.kd_kl) > 0.0 and float(args.kd_T) <= 0.0:
        raise ValueError(f"KL distillation requires a positive temperature, got kd_T={args.kd_T}.")

    dec_tok, enc_tok = load_tokenizers(args, base_ckpt)
    enc_vocab_size = len(enc_tok) + 2
    enc_pad_id = enc_tok.pad_token_id
    if enc_pad_id is None:
        raise ValueError("Encoder tokenizer must have pad_token_id.")

    cfg = AutoConfig.from_pretrained(base_ckpt)
    base_num_layers = int(cfg.num_hidden_layers)
    cfg.tie_word_embeddings = False
    apply_attention_impl(cfg, args.attn_impl)

    cfg.enable_dual_stream = True
    cfg.interleave_num_layers = int(args.interleave_n)
    cfg.interleave_min_prefix = int(args.min_prefix)
    cfg.encoder_hidden_size = int(args.encoder_hidden_size)
    cfg.encoder_num_layers = int(args.encoder_num_layers)
    cfg.encoder_num_attention_heads = int(args.encoder_num_attention_heads)
    cfg.encoder_num_key_value_heads = int(args.encoder_num_key_value_heads)
    cfg.encoder_intermediate_size = int(args.encoder_intermediate_size)
    cfg.encoder_vocab_size = enc_vocab_size
    cfg.encoder_pad_token_id = enc_pad_id
    cfg.encoder_mse_weight = float(args.kd_mse)
    cfg.encoder_kl_weight = float(args.kd_kl)
    cfg.encoder_mlm_weight = float(args.kd_mlm)
    cfg.distill_temperature = float(args.kd_T)
    cfg.use_cache = False

    model = LlamaForCausalLMWithInterleave.from_pretrained(base_ckpt, config=cfg, torch_dtype=torch.bfloat16)
    force_untie_word_embeddings(model)

    encoder_init_mode = resolve_encoder_init_mode(args)
    encoder_source_model = None
    if encoder_init_mode == "external_overlap":
        if not args.encoder_init_model_path:
            raise ValueError("encoder_init_mode=external_overlap requires --encoder_init_model_path")
        encoder_source_model = AutoModel.from_pretrained(
            args.encoder_init_model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=args.encoder_init_trust_remote_code,
        )
    initialize_interleave_modules(
        model,
        base_num_layers=base_num_layers,
        interleave_init_mode=args.interleave_init_mode,
        encoder_init_mode=encoder_init_mode,
        encoder_source_model=encoder_source_model,
        encoder_init_copy_embeddings=args.encoder_init_copy_embeddings,
    )
    del encoder_source_model
    model.config.interleave_init_mode = args.interleave_init_mode
    model.config.encoder_init_mode = encoder_init_mode
    model.config.encoder_init_model_path = args.encoder_init_model_path
    model.config.pad_token_id = dec_tok.pad_token_id
    return model, dec_tok, enc_tok


def load_vanilla_llama(args: Any):
    base_ckpt = resolve_stage1_model_path(args)
    pure_hf = use_pure_hf_vanilla(args)
    dec_tok, enc_tok = load_tokenizers(args, base_ckpt, load_encoder=not pure_hf)
    base_cfg = AutoConfig.from_pretrained(base_ckpt)
    base_num_layers = int(base_cfg.num_hidden_layers)

    if pure_hf:
        if args.baseline_extra_layers != 0:
            raise ValueError("--vanilla_pure_hf is a no-growth diagnostic path. Pass --baseline_extra_layers 0.")
        dec_layer_params = estimate_decoder_layer_params(
            int(base_cfg.hidden_size),
            int(base_cfg.intermediate_size),
            int(base_cfg.num_attention_heads),
            int(base_cfg.num_key_value_heads),
        )
        growth = {"extra_layers": 0, "decoder_layer_params": dec_layer_params, "estimated_extra_params": 0, "mode": "pure_hf"}
        target_num_layers = base_num_layers
        model = LlamaForCausalLM.from_pretrained(base_ckpt, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl)
    else:
        growth = choose_baseline_extra_layers(args, base_cfg, len(enc_tok) + 2)
        target_num_layers = base_num_layers + int(growth["extra_layers"])
        cfg = AutoConfig.from_pretrained(base_ckpt)
        cfg.num_hidden_layers = target_num_layers
        cfg.tie_word_embeddings = False
        apply_attention_impl(cfg, args.attn_impl)
        model = VanillaLlamaForStage2.from_pretrained(base_ckpt, config=cfg, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl)
        if target_num_layers != base_num_layers and args.vanilla_init_mode == "copy_top":
            initialize_added_decoder_layers(model, base_num_layers=base_num_layers, target_num_layers=target_num_layers)
        force_untie_word_embeddings(model)

    apply_embedding_init_policy(model, args)
    model.config.use_cache = False
    model.config.pad_token_id = dec_tok.pad_token_id
    model.config.baseline_pure_hf_vanilla = bool(pure_hf)
    model.config.baseline_extra_decoder_layers = int(growth["extra_layers"])
    model.config.baseline_base_num_layers = base_num_layers
    model.config.baseline_target_num_layers = target_num_layers
    model.config.baseline_param_match_mode = growth["mode"]
    model.config.baseline_init_mode = args.vanilla_init_mode
    model.config.baseline_estimated_decoder_layer_params = int(growth["decoder_layer_params"])
    model.config.baseline_estimated_extra_params = int(growth["estimated_extra_params"])
    return model, dec_tok, enc_tok


def load_stage2_model(args: Any):
    if args.model_type == "interleave":
        return load_llama_with_interleave(args)
    if args.model_type == "vanilla":
        return load_vanilla_llama(args)
    raise ValueError(f"Unsupported model_type: {args.model_type}")


class CausalLMTrainerWithExtraLogs(Trainer):
    def __init__(self, *args, ignore_index: int = -100, extra_loss_names=("loss_ce", "loss_mlm", "loss_kd_mse", "loss_kd_kl"), **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.extra_loss_names = tuple(extra_loss_names)
        self.model_accepts_loss_kwargs = False
        self.debug_one_update = False
        self._debug_grad_logged = False
        self._throughput_started = False
        self._last_log_time = time.perf_counter()
        self._reset_train_log_buffer()

    def _reset_train_log_buffer(self):
        self._loss_sum = 0.0
        self._loss_count = 0
        self._token_correct = 0.0
        self._token_total = 0.0
        self._extra_loss_sums = {name: 0.0 for name in self.extra_loss_names}
        self._extra_loss_counts = {name: 0 for name in self.extra_loss_names}
        self._target_token_total = 0.0
        self._input_token_total = 0.0

    def _reduce_mean_scalar(self, value, device):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim > 0:
            tensor = tensor.mean()
        if hasattr(self, "accelerator") and self.accelerator is not None:
            tensor = self.accelerator.reduce(tensor, reduction="mean")
        return float(tensor.item())

    def _reduce_sum_pair(self, a, b, device):
        tensor = torch.tensor([a, b], dtype=torch.float32, device=device)
        if hasattr(self, "accelerator") and self.accelerator is not None:
            tensor = self.accelerator.reduce(tensor, reduction="sum")
        return float(tensor[0].item()), float(tensor[1].item())

    def _reduce_sum_scalar(self, value, device):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim > 0:
            tensor = tensor.sum()
        if hasattr(self, "accelerator") and self.accelerator is not None:
            tensor = self.accelerator.reduce(tensor, reduction="sum")
        return float(tensor.item())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        loss, outputs = super().compute_loss(model, dict(inputs), return_outputs=True, num_items_in_batch=num_items_in_batch)
        with torch.no_grad():
            loss_scalar = self._reduce_mean_scalar(loss.detach(), device=loss.device)
            self._loss_sum += loss_scalar
            self._loss_count += 1
            if labels is not None:
                self._target_token_total += self._reduce_sum_scalar(labels.ne(self.ignore_index).sum(), device=loss.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                self._input_token_total += self._reduce_sum_scalar(attention_mask.to(dtype=torch.float32).sum(), device=loss.device)
            for name in self.extra_loss_names:
                value = getattr(outputs, name, None)
                if value is None and isinstance(outputs, dict):
                    value = outputs.get(name, None)
                if value is not None:
                    value = self._reduce_mean_scalar(value, device=loss.device)
                    self._extra_loss_sums[name] += value
                    self._extra_loss_counts[name] += 1
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, dict):
                logits = outputs.get("logits", None)
            if logits is not None and labels is not None:
                preds = logits[:, :-1].argmax(dim=-1)
                gold = labels[:, 1:]
                mask = gold.ne(self.ignore_index)
                if mask.any():
                    correct = ((preds == gold) & mask).sum()
                    total = mask.sum()
                    correct, total = self._reduce_sum_pair(correct, total, device=loss.device)
                    self._token_correct += correct
                    self._token_total += total
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._throughput_started:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._last_log_time = time.perf_counter()
            self._throughput_started = True
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.debug_one_update and not self._debug_grad_logged:
            print_grad_finite_stats(model, prefix="[DebugOneUpdate] after backward")
            self._debug_grad_logged = True
        return loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        logs["global_step"] = int(self.state.global_step)
        if "loss" in logs:
            logs["loss_hf_raw"] = logs["loss"]
        if self._loss_count > 0:
            logs["loss"] = self._loss_sum / self._loss_count
        if self._token_total > 0:
            logs["mean_token_accuracy"] = self._token_correct / self._token_total
        for name in self.extra_loss_names:
            count = self._extra_loss_counts[name]
            if count > 0:
                logs[f"train_{name}"] = self._extra_loss_sums[name] / count
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = max(now - self._last_log_time, 1e-9)
        if self._target_token_total > 0:
            logs["train_target_tokens_per_second"] = self._target_token_total / elapsed
        if self._input_token_total > 0:
            logs["train_input_tokens_per_second"] = self._input_token_total / elapsed
        self._last_log_time = now
        self._reset_train_log_buffer()
        return super().log(logs, start_time=start_time)


def _rank_tag() -> str:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return f"rank={torch.distributed.get_rank()}"
    return "rank=0"


def _finite_stats_from_named_tensors(named_tensors, max_examples: int = 5) -> Dict[str, Any]:
    tensors = 0
    none_tensors = 0
    nonfinite_tensors = 0
    nonfinite_values = 0
    total_values = 0
    max_abs = 0.0
    norm_sq = 0.0
    examples = []
    for name, tensor in named_tensors:
        if tensor is None:
            none_tensors += 1
            continue
        tensors += 1
        with torch.no_grad():
            current = tensor.detach()
            finite = torch.isfinite(current)
            total_values += current.numel()
            if not bool(finite.all().item()):
                bad = int((~finite).sum().item())
                nonfinite_tensors += 1
                nonfinite_values += bad
                if len(examples) < max_examples:
                    examples.append(f"{name}: bad={bad}/{current.numel()}")
            finite_t = torch.where(finite, current, torch.zeros_like(current)).float()
            if finite_t.numel() > 0:
                max_abs = max(max_abs, float(finite_t.abs().max().item()))
                local_norm = torch.linalg.vector_norm(finite_t)
                norm_sq += float(local_norm.item()) ** 2
    return {
        "tensors": tensors,
        "none_tensors": none_tensors,
        "nonfinite_tensors": nonfinite_tensors,
        "nonfinite_values": nonfinite_values,
        "total_values": total_values,
        "max_abs": max_abs,
        "l2_norm": math.sqrt(norm_sq),
        "examples": examples,
    }


def _print_finite_stats(prefix: str, stats: Dict[str, Any]):
    print(
        f"{prefix} [{_rank_tag()}] tensors={stats['tensors']} none={stats['none_tensors']} "
        f"nonfinite_tensors={stats['nonfinite_tensors']} nonfinite_values={stats['nonfinite_values']}/{stats['total_values']} "
        f"max_abs={stats['max_abs']:.6g} l2_norm={stats['l2_norm']:.6g}"
    )
    for example in stats["examples"]:
        print(f"{prefix} [{_rank_tag()}] nonfinite_example {example}")


def print_grad_finite_stats(model, prefix: str):
    stats = _finite_stats_from_named_tensors((name, param.grad) for name, param in model.named_parameters())
    _print_finite_stats(prefix, stats)


def print_param_finite_stats(model, prefix: str):
    stats = _finite_stats_from_named_tensors(model.named_parameters())
    _print_finite_stats(prefix, stats)


def build_trainer(args: Any, model, dataset, dec_tok: PreTrainedTokenizer, collator):
    try:
        from kormo.train.arguments import KORMoTrainingArguments
    except ImportError as exc:
        raise ImportError("Training requires the local kormo package on PYTHONPATH.") from exc

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    device_batch = args.per_device_train_batch_size
    collator.rows_per_step = device_batch
    collator.sequences_per_batch = device_batch

    training_args = KORMoTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        logging_steps=args.logging_steps,
        bf16=True,
        fp16=False,
        deepspeed=args.deepspeed,
        report_to="none" if args.debug_one_update else args.report_to,
        remove_unused_columns=False,
        lr_scheduler_type="linear",
        max_grad_norm=args.max_grad_norm,
        max_steps=1 if args.debug_one_update else -1,
        save_strategy="no" if args.debug_one_update else "steps",
        save_steps=args.save_steps,
        seed=42,
        data_seed=42,
    )
    trainer = CausalLMTrainerWithExtraLogs(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=dec_tok,
        data_collator=collator,
    )
    trainer.debug_one_update = bool(args.debug_one_update)
    return trainer


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Stage-2 CPT training script for interleave and matched decoder-only controls.")
    parser.add_argument("--merged_dataset", type=str, default=_repo_path_if_exists("merged_arrow"), help="Merged packed dataset glob/dir/path.")
    parser.add_argument("--stage2_data_root", type=str, default=os.environ.get("KRONG_STAGE2_DATA_ROOT"), help="Packed multi-domain dataset root.")
    parser.add_argument("--cache_dir", type=str, default=_repo_path("stage2_cache"))
    parser.add_argument("--output_dir", type=str, default=_repo_path("KRong-1B-Stage2"))
    parser.add_argument("--stage1_model_path", type=str, default=None, help="Stage1 HF checkpoint path.")
    parser.add_argument("--model_type", choices=["interleave", "vanilla"], default="interleave")
    parser.add_argument("--baseline_extra_layers", type=int, default=None, help="Extra decoder layers for vanilla matched control.")
    parser.add_argument("--vanilla_pure_hf", action="store_true", help="Diagnostic no-growth vanilla path; requires --baseline_extra_layers 0.")
    parser.add_argument("--lm_head_init", choices=["keep", "copy_input", "random", "tie_input"], default="keep")
    parser.add_argument("--dec_tokenizer_repo", type=str, default=_repo_path("final"), help="Stage1 model/tokenizer path alias.")
    parser.add_argument("--enc_tokenizer_repo", type=str, default=_repo_path("krong_tokenizer"))
    parser.add_argument("--attn_impl", type=str, default="flash_attention_3", choices=["flash_attention_3", "sdpa", "eager"])
    parser.add_argument("--interleave_n", type=int, default=3)
    parser.add_argument("--min_prefix", type=int, default=1)
    parser.add_argument("--interleave_init_mode", choices=["random", "copy_top", "copy_low"], default="copy_top")
    parser.add_argument("--vanilla_init_mode", choices=["random", "copy_top"], default="copy_top")
    parser.add_argument("--encoder_init_mode", choices=["auto", "random", "decoder_overlap", "external_overlap"], default="auto")
    parser.add_argument("--encoder_init_model_path", type=str, default=None)
    parser.add_argument("--encoder_init_copy_embeddings", action="store_true")
    parser.add_argument("--encoder_init_trust_remote_code", action="store_true")
    parser.add_argument("--encoder_hidden_size", type=int, default=768)
    parser.add_argument("--encoder_num_layers", type=int, default=6)
    parser.add_argument("--encoder_num_attention_heads", type=int, default=8)
    parser.add_argument("--encoder_num_key_value_heads", type=int, default=8)
    parser.add_argument("--encoder_intermediate_size", type=int, default=4096)
    parser.add_argument("--kd_mse", type=float, default=0.00)
    parser.add_argument("--kd_kl", type=float, default=0.00)
    parser.add_argument("--kd_mlm", type=float, default=0.25)
    parser.add_argument("--kd_T", type=float, default=2.0)
    parser.add_argument("--dec_len", type=int, default=4096)
    parser.add_argument("--enc_len", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.015)
    parser.add_argument("--weight_decay", type=float, default=0.033)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--deepspeed", type=str, default=_repo_path("ds_config_zero2.json"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--boundary_silence", type=int, default=6)
    parser.add_argument("--debug_one_update", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--add_row_boundary_tokens", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none", help="Use 'wandb' after setting WANDB_API_KEY outside the repo.")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    return parser


def main():
    args = build_argparser().parse_args()
    base_ckpt = resolve_stage1_model_path(args)
    print(f"[Info] Loading {args.model_type} model from Stage1 checkpoint: {base_ckpt}")
    print(f"[Info] add_row_boundary_tokens={args.add_row_boundary_tokens}")

    model, dec_tok, enc_tok = load_stage2_model(args)
    print("[Info] Model loaded.")

    dataset = load_and_prepare_dataset(args, dec_tok, enc_tok)
    collator = build_collator(args, dec_tok, enc_tok)
    trainer = build_trainer(args, model, dataset, dec_tok, collator)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"[Model] total={total_params:,} trainable={trainable_params:,}")
    if args.model_type == "interleave":
        print(f"[Interleave] N={model.config.interleave_num_layers} min_prefix={model.config.interleave_min_prefix}")
        print(f"[EmbeddingInit] lm_head_init={getattr(model.config, 'lm_head_init', args.lm_head_init)}")
        print(
            f"[Init] interleave_init={getattr(model.config, 'interleave_init_mode', 'unknown')} "
            f"encoder_init={getattr(model.config, 'encoder_init_mode', 'unknown')} "
            f"encoder_init_model={getattr(model.config, 'encoder_init_model_path', None)}"
        )
        print(
            f"[KD] MSE={model.config.encoder_mse_weight} KL={model.config.encoder_kl_weight} "
            f"MLM={model.config.encoder_mlm_weight} T={model.config.distill_temperature}"
        )
    else:
        print("[Interleave] disabled (vanilla baseline)")
        print(f"[Vanilla] pure_hf={getattr(model.config, 'baseline_pure_hf_vanilla', False)}")
        print(f"[EmbeddingInit] lm_head_init={getattr(model.config, 'lm_head_init', args.lm_head_init)}")
        print(
            f"[BaselineMatch] mode={getattr(model.config, 'baseline_param_match_mode', 'n/a')} "
            f"init={getattr(model.config, 'baseline_init_mode', 'n/a')} "
            f"base_layers={getattr(model.config, 'baseline_base_num_layers', 'n/a')} "
            f"extra_layers={getattr(model.config, 'baseline_extra_decoder_layers', 0)} "
            f"target_layers={getattr(model.config, 'baseline_target_num_layers', 'n/a')}"
        )

    trainer.train(resume_from_checkpoint=args.resume_ckpt)
    if args.debug_one_update:
        print_param_finite_stats(trainer.model, prefix="[DebugOneUpdate] after optimizer step")
        print(f"[DebugOneUpdate] [{_rank_tag()}] finished one optimizer update; skip save_model")
        return
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
