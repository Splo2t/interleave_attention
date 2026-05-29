from __future__ import annotations

import inspect
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer_with_fallback(
    ckpt_path: str,
    *,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    cache_dir: str | None = None,
):
    """
    Local checkpoints may store tokenizer_class=TokenizersBackend with only tokenizer.json.
    In that case AutoTokenizer cannot resolve the class name, so fall back to a generic
    PreTrainedTokenizerFast initialized from tokenizer_config.json + tokenizer.json.
    """
    try:
        return AutoTokenizer.from_pretrained(
            ckpt_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            cache_dir=cache_dir,
        )
    except ValueError as e:
        cfg_path = os.path.join(ckpt_path, "tokenizer_config.json")
        tok_json_path = os.path.join(ckpt_path, "tokenizer.json")
        if not (os.path.isfile(cfg_path) and os.path.isfile(tok_json_path)):
            raise

        with open(cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)

        if tok_cfg.get("tokenizer_class") != "TokenizersBackend" and tok_cfg.get("backend") != "tokenizers":
            raise

        kwargs = {}
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ):
            if tok_cfg.get(key) is not None:
                kwargs[key] = tok_cfg[key]

        if tok_cfg.get("model_max_length") is not None:
            kwargs["model_max_length"] = tok_cfg["model_max_length"]
        kwargs["clean_up_tokenization_spaces"] = tok_cfg.get("clean_up_tokenization_spaces", True)

        print(
            f"[tokenizer] AutoTokenizer fallback for {ckpt_path}: "
            f"using PreTrainedTokenizerFast from tokenizer.json ({e})"
        )
        return PreTrainedTokenizerFast(tokenizer_file=tok_json_path, **kwargs)


def ensure_tokenizer_padding(tokenizer) -> None:
    if getattr(tokenizer, "pad_token", None) is not None:
        return

    fallback = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "unk_token", None)
    if fallback is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print("[tokenizer] added pad_token=[PAD]")
        return

    tokenizer.pad_token = fallback
    print(f"[tokenizer] pad_token was missing; using {fallback!r}")


def infer_model_arch_from_checkpoint(ckpt_path: str) -> str:
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        return ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
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


def resolve_dtype(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown dtype: {name}")


def parse_device_map_arg(s: str):
    """
    CLI에서 받은 device_map 문자열을 HF from_pretrained에 안전하게 전달하기 위한 헬퍼.

    - "auto" / "balanced" / "balanced_low_0" / "sequential" 는 그대로 사용
    - "cuda:0" / "cuda:1" / "cpu" / "mps" 같은 단일 디바이스 지정은 {"": device} 형태로 변환
    - "0" 같은 숫자면 {"": int}로 변환
    """
    if s is None:
        return "auto"
    s = str(s).strip()
    if not s:
        return "auto"

    presets = {"auto", "balanced", "balanced_low_0", "sequential"}
    if s in presets:
        return s

    if s.isdigit():
        return {"": int(s)}

    if s.startswith("cuda:"):
        try:
            idx = int(s.split(":", 1)[1])
            return {"": idx}
        except Exception:
            return {"": s}

    if s in {"cuda", "gpu"}:
        return {"": 0}

    if s in {"cpu", "mps"}:
        return {"": s}

    return s


def _split_for_tokenize(context: str, continuation: str) -> tuple[str, str]:
    """
    lm-eval 권장 규칙:
      context의 trailing whitespace는 continuation 앞으로 이동.
    """
    context = context or ""
    continuation = continuation or ""
    num_ws = len(context) - len(context.rstrip())
    if num_ws > 0:
        continuation = context[-num_ws:] + continuation
        context = context[:-num_ws]
    return context, continuation


def _longest_common_prefix(a: list[int], b: list[int]) -> int:
    i = 0
    m = min(len(a), len(b))
    while i < m and a[i] == b[i]:
        i += 1
    return i


def _prompt_and_cont_ids_lmeval(tokenizer, context: str, continuation: str) -> tuple[str, list[int], bool]:
    """
    lm-eval 방식:
      cont_ids = tok(context+cont) - tok(context)
    만약 ctx_ids가 full_ids의 prefix가 아니면(LCP fallback),
      prompt_text를 decode(full_ids[:lcp])로 바꿔서 alignment를 보장.
    return: (prompt_text, cont_ids, diverged)
    """
    context, continuation = _split_for_tokenize(context, continuation)

    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)

    lcp = _longest_common_prefix(ctx_ids, full_ids)
    diverged = lcp != len(ctx_ids)
    if diverged:
        prompt_ids = full_ids[:lcp]
        try:
            prompt_text = tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        cont_ids = full_ids[lcp:]
        return prompt_text, cont_ids, True

    cont_ids = full_ids[len(ctx_ids) :]
    return context, cont_ids, False


def _get_forward_accepts_kwargs(model) -> Tuple[Optional[set], bool]:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None, True

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return None, True
    return set(params.keys()), False


def _filter_forward_kwargs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    allowed, has_var_kw = _get_forward_accepts_kwargs(model)
    if has_var_kw or allowed is None:
        return inputs
    return {k: v for k, v in inputs.items() if k in allowed}


def _detect_dec_key(inputs: Dict[str, Any]) -> str:
    for key in ("input_ids", "decoder_input_ids"):
        if key in inputs:
            return key
    raise KeyError(f"Cannot find decoder ids key in inputs. keys={list(inputs.keys())}")


def _detect_attn_key(inputs: Dict[str, Any]) -> Optional[str]:
    for key in ("attention_mask", "decoder_attention_mask"):
        if key in inputs:
            return key
    return None


def _truncate_left_for_decoder_aligned_tensors(
    inputs: Dict[str, Any],
    *,
    max_len: Optional[int],
) -> Dict[str, Any]:
    if max_len is None or max_len <= 0:
        return inputs

    dec_key = _detect_dec_key(inputs)
    dec = inputs[dec_key]
    if not torch.is_tensor(dec) or dec.ndim != 2:
        return inputs

    seq_len = dec.size(1)
    if seq_len <= max_len:
        return inputs

    offset = seq_len - max_len
    out: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.ndim == 2 and value.size(0) == dec.size(0) and value.size(1) == seq_len:
            out[key] = value[:, offset:]
        else:
            out[key] = value
    return out


def _forward_logits(model, inputs: Dict[str, Any]) -> torch.Tensor:
    fwd_inputs = _filter_forward_kwargs(model, dict(inputs))
    fwd_inputs["use_cache"] = False
    out = model(**fwd_inputs)
    if isinstance(out, dict):
        logits = out.get("logits")
        if logits is None:
            raise ValueError("Model output dict does not contain 'logits'.")
        return logits
    if hasattr(out, "logits"):
        return out.logits
    raise ValueError("Model output has no logits attribute.")


def _select_last_valid_logits(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if attention_mask is None:
        return logits[:, -1, :]

    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D, got shape={tuple(attention_mask.shape)}")

    last_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_indices, last_indices, :]


def build_inputs_with_cont_ws(base_inputs: dict, cont_ids: list[int]) -> dict:
    if not cont_ids:
        return dict(base_inputs)

    inp = base_inputs["input_ids"]
    att = base_inputs["attention_mask"]
    allow_lens = base_inputs["cross_k_allow_lens"]

    device = inp.device
    cont = torch.tensor([cont_ids], device=device, dtype=inp.dtype)
    ones = torch.ones((1, len(cont_ids)), device=device, dtype=att.dtype)

    last_allow_len = int(allow_lens[0, -1].item())
    allow_add = torch.full((1, len(cont_ids)), last_allow_len, device=device, dtype=allow_lens.dtype)
    out = dict(base_inputs)
    out["input_ids"] = torch.cat([inp, cont], dim=1)
    out["attention_mask"] = torch.cat([att, ones], dim=1)
    out["cross_k_allow_lens"] = torch.cat([allow_lens, allow_add], dim=1)
    out["use_cache"] = False
    return out


def _materialize_ws_inputs_for_static_forward(base_inputs: dict) -> dict:
    """
    Convert processor ws_state into the explicit tensors accepted by model.forward.
    This mirrors the older lm-eval wrapper path: encoder state is fixed from the
    original prompt, while continuation tokens reuse the last prompt prefix length.
    """
    out = dict(base_inputs)
    ws_state = out.pop("ws_state", None)
    if ws_state is None:
        return out

    if "encoder_hidden_states" not in out and "encoder_hidden_states" in ws_state:
        out["encoder_hidden_states"] = ws_state["encoder_hidden_states"]
    if "encoder_attention_mask" not in out and "encoder_attention_mask" in ws_state:
        out["encoder_attention_mask"] = ws_state["encoder_attention_mask"]

    if "cross_k_allow_lens" not in out and "L_per_token" in ws_state:
        seq_len = int(out["input_ids"].size(1))
        allow_lens = list(ws_state["L_per_token"])[-seq_len:]
        if len(allow_lens) < seq_len:
            pad_value = int(allow_lens[0]) if allow_lens else 0
            allow_lens = [pad_value] * (seq_len - len(allow_lens)) + allow_lens
        out["cross_k_allow_lens"] = torch.tensor(
            [allow_lens],
            device=out["input_ids"].device,
            dtype=torch.int32,
        )

    out["use_cache"] = False
    return out


def build_inputs_with_cont_plain(base_inputs: dict, cont_ids: list[int]) -> dict:
    if not cont_ids:
        return dict(base_inputs)

    inp = base_inputs["input_ids"]
    att = base_inputs.get("attention_mask", torch.ones_like(inp))

    device = inp.device
    cont = torch.tensor([cont_ids], device=device, dtype=inp.dtype)
    ones = torch.ones((1, len(cont_ids)), device=device, dtype=att.dtype)

    out = dict(base_inputs)
    out.pop("ws_state", None)
    out["input_ids"] = torch.cat([inp, cont], dim=1)
    out["attention_mask"] = torch.cat([att, ones], dim=1)
    out["use_cache"] = False
    return out


@torch.inference_mode()
def loglikelihood_continuation_oneshot(model, base_inputs: dict, cont_ids: list[int]) -> float:
    if not cont_ids:
        return float("-inf")

    num_tokens = len(cont_ids)
    base_inputs = _materialize_ws_inputs_for_static_forward(base_inputs)
    base_len = int(base_inputs["input_ids"].size(1))

    if "cross_k_allow_lens" in base_inputs:
        full_inputs = build_inputs_with_cont_ws(base_inputs, cont_ids)
    else:
        full_inputs = build_inputs_with_cont_plain(base_inputs, cont_ids)

    full_inputs.pop("ws_state", None)
    logits = _forward_logits(model, full_inputs)
    prev_logits = logits[:, base_len - 1 : base_len - 1 + num_tokens, :]
    if prev_logits.size(1) != num_tokens:
        raise ValueError(
            f"oneshot logits length mismatch: expected {num_tokens}, got {prev_logits.size(1)}"
        )

    logprobs = F.log_softmax(prev_logits.float(), dim=-1)
    ids_t = torch.tensor(cont_ids, device=logprobs.device).view(1, num_tokens, 1)
    picked = torch.gather(logprobs, 2, ids_t).squeeze(-1)
    return float(picked.sum().item())


@torch.inference_mode()
def loglikelihood_continuation_ws(model, base_inputs: dict, cont_ids: list[int]) -> float:
    if not cont_ids:
        return float("-inf")

    num_tokens = len(cont_ids)
    full_inputs = build_inputs_with_cont_ws(base_inputs, cont_ids)
    full_inputs["logits_to_keep"] = num_tokens + 1
    full_inputs["use_cache"] = False

    out = model(**full_inputs)
    logits = out.logits
    slice_logits = logits[:, :-1, :]

    logprobs = F.log_softmax(slice_logits.float(), dim=-1)
    ids_t = torch.tensor(cont_ids, device=logprobs.device).view(1, num_tokens, 1)
    picked = torch.gather(logprobs, 2, ids_t).squeeze(-1)
    return float(picked.sum().item())


@torch.inference_mode()
def loglikelihood_continuation_ws_dynamic(model, base_inputs: dict, cont_ids: list[int]) -> float:
    if not cont_ids:
        return float("-inf")

    input_ids = base_inputs["input_ids"].clone()
    attention_mask = base_inputs.get("attention_mask", torch.ones_like(input_ids)).clone()

    if "ws_state" not in base_inputs:
        total_lp = 0.0
        for token_id in cont_ids:
            fwd = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            fwd["use_cache"] = False
            fwd["logits_to_keep"] = 1
            out = model(**fwd)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0, token_id].item()
            total_lp += float(lp)

            step = torch.tensor([[token_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, step], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(step)], dim=1)
    else:
        ws0 = base_inputs["ws_state"]
        ws_state = dict(ws0)
        ws_state["L_per_token"] = list(ws0["L_per_token"])
        ws_state["prev_len"] = int(ws0["prev_len"])
        ws_state["enc_text"] = str(ws0["enc_text"])
        ws_state["L_cur"] = int(ws0["L_cur"])

        total_lp = 0.0
        for token_id in cont_ids:
            fwd = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ws_state=ws_state,
                use_cache=False,
            )
            fwd["use_cache"] = False
            fwd["logits_to_keep"] = 1

            out = model(**fwd)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0, token_id].item()
            total_lp += float(lp)

            step = torch.tensor([[token_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, step], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(step)], dim=1)

    return total_lp


def continuation_ids_from_concat(tokenizer, prompt_text: str, prompt_ids: list[int], cand_text: str) -> list[int]:
    full = tokenizer(prompt_text + cand_text, add_special_tokens=False)["input_ids"]

    bos = tokenizer.bos_token_id
    if bos is not None and len(prompt_ids) > 0 and prompt_ids[0] == bos:
        if len(full) == 0 or full[0] != bos:
            full = [bos] + full

    if len(full) >= len(prompt_ids) and full[: len(prompt_ids)] == prompt_ids:
        return full[len(prompt_ids) :]

    return tokenizer(cand_text, add_special_tokens=False)["input_ids"]


@dataclass
class EvalScorerConfig:
    use_chat_template: bool = False
    system_prompt: str = ""
    enable_thinking: bool = False
    dec_max_len: Optional[int] = None
    space_variant_mode: str = "auto"
    batch_scoring: str = "auto"
    continuation_scoring: str = "dynamic"
    add_bos: str = "auto"


class HFEvalScorer:
    """
    - prompt 문자열 -> (옵션) chat_template로 감싸기
    - processor.prepare_generate_inputs(model, prompt) 로 base_inputs 생성
    - 후보 문자열들의 로그우도를 계산해 argmax 선택
    """

    def __init__(self, model, processor, tokenizer, cfg: EvalScorerConfig):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.cfg = cfg

    def format_prompt(self, raw_prompt: str) -> str:
        if not self.cfg.use_chat_template:
            return raw_prompt
        msgs = []
        if self.cfg.system_prompt:
            msgs.append({"role": "system", "content": self.cfg.system_prompt})
        msgs.append({"role": "user", "content": raw_prompt})
        try:
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )


    def _maybe_add_bos(self, ids: List[int]) -> List[int]:
        if self.cfg.add_bos != "true":
            return ids
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None or (ids and ids[0] == bos):
            return ids
        return [int(bos)] + ids

    def _tokenize_prompt_no_special(self, prompt: str) -> Dict[str, torch.Tensor]:
        ids = self._maybe_add_bos(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return self.tokenizer.pad(
            [{"input_ids": ids}],
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def _tokenize_prompts_no_special(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        rows = [
            {"input_ids": self._maybe_add_bos(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])}
            for prompt in prompts
        ]
        return self.tokenizer.pad(
            rows,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def prepare_base_inputs(self, prompt: str) -> Dict[str, Any]:
        if self.processor is not None and hasattr(self.processor, "prepare_generate_inputs"):
            inputs = self.processor.prepare_generate_inputs(self.model, prompt)
        else:
            tok = self._tokenize_prompt_no_special(prompt)
            dev = getattr(self.model, "device", None)
            if dev is None or (hasattr(dev, "type") and dev.type == "meta"):
                try:
                    dev = self.model._execution_device
                except Exception:
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(dev) for k, v in tok.items()}

        processor_dec_max_len = int(getattr(self.processor, "dec_max_len", 0) or 0) if self.processor is not None else 0
        if processor_dec_max_len <= 0:
            inputs = _truncate_left_for_decoder_aligned_tensors(inputs, max_len=self.cfg.dec_max_len)
        return inputs

    def prepare_base_inputs_batch(self, prompts: List[str]) -> Dict[str, Any]:
        if not prompts:
            raise ValueError("prompts must not be empty")

        if self.processor is not None and hasattr(self.processor, "prepare_generate_inputs_batch"):
            inputs = self.processor.prepare_generate_inputs_batch(self.model, prompts)
        else:
            tok = self._tokenize_prompts_no_special(prompts)
            dev = getattr(self.model, "device", None)
            if dev is None or (hasattr(dev, "type") and dev.type == "meta"):
                try:
                    dev = self.model._execution_device
                except Exception:
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(dev) for k, v in tok.items()}

        processor_dec_max_len = int(getattr(self.processor, "dec_max_len", 0) or 0) if self.processor is not None else 0
        if processor_dec_max_len <= 0:
            inputs = _truncate_left_for_decoder_aligned_tensors(inputs, max_len=self.cfg.dec_max_len)
        return inputs

    def _tokenize_no_special(self, s: str) -> List[int]:
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        return list(ids) if isinstance(ids, (list, tuple)) else []

    def _label_variants(self, label: str, prompt_text: str) -> List[str]:
        del prompt_text
        return [label or ""]

    @torch.inference_mode()
    def generate_text(
        self,
        raw_prompt: str,
        *,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        skip_special_tokens: bool = True,
    ) -> str:
        prompt = self.format_prompt(raw_prompt)
        inputs = self.prepare_base_inputs(prompt)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise KeyError(f"Cannot generate without input_ids. keys={list(inputs.keys())}")

        gen_inputs = _filter_forward_kwargs(self.model, dict(inputs))
        # Some tokenizer.json-only checkpoints emit token_type_ids, while many
        # causal LMs reject them during generate() validation.
        gen_inputs.pop("token_type_ids", None)
        gen_inputs["use_cache"] = False

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = int(pad_token_id)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = int(eos_token_id)
        if do_sample and temperature is not None:
            generation_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            generation_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            generation_kwargs["top_k"] = int(top_k)
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = float(repetition_penalty)

        out_ids = self.model.generate(**gen_inputs, **generation_kwargs)
        prompt_len = int(input_ids.shape[1])
        generated_ids = out_ids[0, prompt_len:]
        try:
            return self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def score_labels(self, raw_prompt: str, labels: List[str]) -> Dict[str, float]:
        prompt = self.format_prompt(raw_prompt)
        base_inputs = self.prepare_base_inputs(prompt)
        prompt_ids = base_inputs["input_ids"][0].tolist()

        ws_state = base_inputs.get("ws_state")
        if ws_state is not None:
            base_inputs["encoder_hidden_states"] = ws_state["encoder_hidden_states"]
            base_inputs["encoder_attention_mask"] = ws_state["encoder_attention_mask"]

            seq_len = int(base_inputs["input_ids"].size(1))
            allow_lens = ws_state["L_per_token"][-seq_len:]
            base_inputs["cross_k_allow_lens"] = torch.tensor(
                [allow_lens],
                device=base_inputs["input_ids"].device,
                dtype=torch.long,
            )

        base_inputs["use_cache"] = False
        continuation_mode = (self.cfg.continuation_scoring or "dynamic").lower()

        tokenized_variants: Dict[str, List[List[int]]] = {}
        for label in labels:
            vars_ = self._label_variants(label, prompt)
            ids_list = []
            seen = set()
            for variant in vars_:
                ids = tuple(continuation_ids_from_concat(self.tokenizer, prompt, prompt_ids, variant))
                if not ids or ids in seen:
                    continue
                seen.add(ids)
                ids_list.append(list(ids))
            tokenized_variants[label] = ids_list

        need_base = any(len(ids) == 1 for ids_list in tokenized_variants.values() for ids in ids_list)
        base_logprobs = None
        if need_base:
            logits = _forward_logits(self.model, base_inputs)
            base_logprobs = F.log_softmax(logits[:, -1, :].float(), dim=-1)[0]

        scores: Dict[str, float] = {}
        for label in labels:
            best = float("-inf")
            for ids in tokenized_variants.get(label, []):
                if len(ids) == 1 and base_logprobs is not None:
                    score = float(base_logprobs[ids[0]].item())
                else:
                    score = (
                        loglikelihood_continuation_oneshot(self.model, base_inputs, ids)
                        if continuation_mode == "oneshot"
                        else loglikelihood_continuation_ws_dynamic(self.model, base_inputs, ids)
                    )
                if score > best:
                    best = score
            scores[label] = best
        return scores

    @torch.inference_mode()
    def score_labels_ll_and_len(self, raw_prompt: str, labels: list[str]) -> dict[str, tuple[float, int]]:
        context = self.format_prompt(raw_prompt)

        groups = defaultdict(list)
        for label in labels:
            prompt_text, cont_ids, _ = _prompt_and_cont_ids_lmeval(self.tokenizer, context, label)
            groups[prompt_text].append((label, cont_ids))

        out: dict[str, tuple[float, int]] = {}
        continuation_mode = (self.cfg.continuation_scoring or "dynamic").lower()
        for prompt_text, items in groups.items():
            base_inputs = self.prepare_base_inputs(prompt_text)
            for label, cont_ids in items:
                ll = (
                    loglikelihood_continuation_oneshot(self.model, base_inputs, cont_ids)
                    if continuation_mode == "oneshot"
                    else loglikelihood_continuation_ws_dynamic(self.model, base_inputs, cont_ids)
                )
                out[label] = (ll, max(1, len(cont_ids)))

        return out

    def _score_labels_ll_and_len_serial_batch(
        self,
        raw_prompts: List[str],
        labels_batch: List[List[str]],
    ) -> List[dict[str, tuple[float, int]]]:
        return [
            self.score_labels_ll_and_len(raw_prompt, labels)
            for raw_prompt, labels in zip(raw_prompts, labels_batch)
        ]

    @torch.inference_mode()
    def score_labels_ll_and_len_batch(
        self,
        raw_prompts: List[str],
        labels_batch: List[List[str]],
    ) -> List[dict[str, tuple[float, int]]]:
        if len(raw_prompts) != len(labels_batch):
            raise ValueError("raw_prompts and labels_batch must have the same length")

        batch_mode = (self.cfg.batch_scoring or "auto").lower()
        if batch_mode == "off":
            return self._score_labels_ll_and_len_serial_batch(raw_prompts, labels_batch)
        if batch_mode == "auto" and self.processor is not None and not hasattr(self.processor, "prepare_generate_inputs_batch"):
            return self._score_labels_ll_and_len_serial_batch(raw_prompts, labels_batch)

        results: List[Optional[dict[str, tuple[float, int]]]] = [None] * len(raw_prompts)
        fast_indices: List[int] = []
        fast_prompt_texts: List[str] = []
        fast_token_maps: List[dict[str, int]] = []

        for idx, (raw_prompt, labels) in enumerate(zip(raw_prompts, labels_batch)):
            context = self.format_prompt(raw_prompt)
            prompt_text_ref: Optional[str] = None
            token_map: dict[str, int] = {}
            batchable = True

            for label in labels:
                prompt_text, cont_ids, _ = _prompt_and_cont_ids_lmeval(self.tokenizer, context, label)
                if prompt_text_ref is None:
                    prompt_text_ref = prompt_text
                elif prompt_text != prompt_text_ref:
                    batchable = False
                    break

                if len(cont_ids) != 1:
                    batchable = False
                    break
                token_map[label] = cont_ids[0]

            if batchable and prompt_text_ref is not None:
                fast_indices.append(idx)
                fast_prompt_texts.append(prompt_text_ref)
                fast_token_maps.append(token_map)
            else:
                results[idx] = self.score_labels_ll_and_len(raw_prompts[idx], labels)

        if fast_indices:
            base_inputs = self.prepare_base_inputs_batch(fast_prompt_texts)
            logits = _forward_logits(self.model, base_inputs)
            next_token_logits = _select_last_valid_logits(logits, base_inputs.get("attention_mask"))
            base_logprobs = F.log_softmax(next_token_logits.float(), dim=-1)

            for row_idx, result_idx in enumerate(fast_indices):
                result: dict[str, tuple[float, int]] = {}
                for label, token_id in fast_token_maps[row_idx].items():
                    result[label] = (float(base_logprobs[row_idx, token_id].item()), 1)
                results[result_idx] = result

        return [result if result is not None else {} for result in results]


def build_scorer_from_args(args) -> HFEvalScorer:
    model_arch = getattr(args, "model_arch", "") or infer_model_arch_from_checkpoint(args.ckpt_path)
    if not getattr(args, "model_arch", "") and model_arch:
        print(f"[model_arch] auto={model_arch}")

    if model_arch == "krong":
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            trust_remote_code=True,
            device_map=parse_device_map_arg(args.device_map),
            torch_dtype=resolve_dtype(args.dtype),
            cache_dir=getattr(args, "transformers_cache", None),
        )
        model.eval()

        processor_kwargs = {
            "trust_remote_code": True,
            "cache_dir": getattr(args, "transformers_cache", None),
            "dec_max_len": (args.dec_max_len if args.dec_max_len and args.dec_max_len > 0 else 0),
        }
        add_bos_override = getattr(args, "add_bos", "auto")
        if add_bos_override == "true":
            processor_kwargs["add_bos"] = True
        elif add_bos_override == "false":
            processor_kwargs["add_bos"] = False

        processor = AutoProcessor.from_pretrained(
            args.ckpt_path,
            **processor_kwargs,
        )
        if hasattr(processor, "add_bos"):
            print(f"[processor] add_bos={processor.add_bos}")
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = load_tokenizer_with_fallback(
                args.ckpt_path,
                trust_remote_code=True,
                use_fast=True,
                cache_dir=getattr(args, "transformers_cache", None),
            )
    else:
        tokenizer = load_tokenizer_with_fallback(
            args.ckpt_path,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=getattr(args, "transformers_cache", None),
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            trust_remote_code=True,
            device_map=parse_device_map_arg(args.device_map),
            torch_dtype=resolve_dtype(args.dtype),
            cache_dir=getattr(args, "transformers_cache", None),
        )
        model.eval()
        processor = None

    cfg = EvalScorerConfig(
        use_chat_template=bool(args.use_chat_template),
        system_prompt=args.system_prompt or "",
        enable_thinking=bool(args.enable_thinking),
        dec_max_len=(args.dec_max_len if args.dec_max_len and args.dec_max_len > 0 else None),
        space_variant_mode=args.space_variant_mode,
        batch_scoring=getattr(args, "batch_scoring", "auto"),
        continuation_scoring=getattr(args, "continuation_scoring", "dynamic"),
        add_bos=getattr(args, "add_bos", "auto"),
    )
    ensure_tokenizer_padding(tokenizer)
    return HFEvalScorer(model, processor, tokenizer, cfg)
