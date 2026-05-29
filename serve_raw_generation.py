#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from krong_eval.cache import DEFAULT_CACHE_ROOT, prepare_cache_paths
from krong_eval.scoring import build_scorer_from_args


SCRIPT_DIR = Path(__file__).resolve().parent


def _json_response(handler: BaseHTTPRequestHandler, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_model_path(path_text: str) -> str:
    text = (path_text or "").strip()
    if not text:
        raise ValueError("ckpt_path is required")
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    return text


def _infer_model_arch(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    config_path = path / "config.json"
    if not config_path.exists():
        return ""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    text = json.dumps(
        {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "auto_map": config.get("auto_map"),
        },
        ensure_ascii=False,
    ).lower()
    if "krong" in text or "kormo" in text:
        return "krong"
    return ""


@dataclass(frozen=True)
class ModelKey:
    ckpt_path: str
    model_arch: str
    dtype: str
    device_map: str
    dec_max_len: int
    add_bos: str


class GenerationManager:
    def __init__(self, *, cache_root: str) -> None:
        self.cache_paths = prepare_cache_paths(cache_root)
        self.lock = threading.Lock()
        self.key: ModelKey | None = None
        self.scorer = None
        self.loaded_at = ""

    def _make_args(self, key: ModelKey) -> argparse.Namespace:
        return argparse.Namespace(
            ckpt_path=key.ckpt_path,
            model_arch=key.model_arch,
            dtype=key.dtype,
            device_map=key.device_map,
            transformers_cache=self.cache_paths.transformers_cache,
            dec_max_len=key.dec_max_len,
            add_bos=key.add_bos,
            use_chat_template=False,
            system_prompt="",
            enable_thinking=False,
            space_variant_mode="both",
            batch_scoring="off",
            continuation_scoring="oneshot",
        )

    def unload(self) -> None:
        with self.lock:
            self.scorer = None
            self.key = None
            self.loaded_at = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, key: ModelKey) -> tuple[bool, float]:
        with self.lock:
            if self.scorer is not None and self.key == key:
                return False, 0.0

            self.scorer = None
            self.key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start = time.monotonic()
            self.scorer = build_scorer_from_args(self._make_args(key))
            self.key = key
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
            return True, time.monotonic() - start

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        ckpt_path = _resolve_model_path(str(payload.get("ckpt_path", "")))
        raw_arch = str(payload.get("model_arch", "auto") or "auto").strip().lower()
        model_arch = _infer_model_arch(ckpt_path) if raw_arch == "auto" else ("" if raw_arch in {"plain", "hf", "normal"} else raw_arch)
        if model_arch not in {"", "krong"}:
            raise ValueError("model_arch must be auto, plain, or krong")

        add_bos = str(payload.get("add_bos", "auto") or "auto").strip().lower()
        if add_bos not in {"auto", "true", "false"}:
            raise ValueError("add_bos must be auto, true, or false")

        key = ModelKey(
            ckpt_path=ckpt_path,
            model_arch=model_arch,
            dtype=str(payload.get("dtype", "bf16") or "bf16"),
            device_map=str(payload.get("device_map", "auto") or "auto"),
            dec_max_len=_as_int(payload.get("dec_max_len"), 4096),
            add_bos=add_bos,
        )

        prompt = str(payload.get("prompt", ""))
        if not prompt:
            raise ValueError("prompt is empty")

        loaded_new, load_sec = self.load(key)
        assert self.scorer is not None

        max_new_tokens = _as_int(payload.get("max_new_tokens"), 64)
        do_sample = _as_bool(payload.get("do_sample"), False)
        temperature = _as_float_or_none(payload.get("temperature"))
        top_p = _as_float_or_none(payload.get("top_p"))
        top_k = payload.get("top_k")
        top_k_value = None if top_k in (None, "") else _as_int(top_k, 0)
        repetition_penalty = _as_float_or_none(payload.get("repetition_penalty"))
        skip_special_tokens = _as_bool(payload.get("skip_special_tokens"), True)

        start = time.monotonic()
        with self.lock:
            text = self.scorer.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k_value,
                repetition_penalty=repetition_penalty,
                skip_special_tokens=skip_special_tokens,
            )
        gen_sec = time.monotonic() - start

        tokenizer = self.scorer.tokenizer
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        generated_tokens = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

        return {
            "ok": True,
            "loaded_new": loaded_new,
            "load_sec": load_sec,
            "generation_sec": gen_sec,
            "model": {
                "ckpt_path": ckpt_path,
                "model_arch": model_arch or "plain",
                "dtype": key.dtype,
                "device_map": key.device_map,
                "dec_max_len": key.dec_max_len,
                "add_bos": key.add_bos,
                "loaded_at": self.loaded_at,
            },
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "text": text,
        }


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Raw Model Generation</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101411;
      --panel: rgba(245, 239, 222, 0.08);
      --panel2: rgba(245, 239, 222, 0.13);
      --line: rgba(245, 239, 222, 0.16);
      --ink: #f5efde;
      --muted: #aeb7aa;
      --accent: #8dd9c1;
      --bad: #ff8b77;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 14% 8%, rgba(141, 217, 193, 0.25), transparent 32rem),
        radial-gradient(circle at 90% 10%, rgba(245, 190, 120, 0.16), transparent 28rem),
        linear-gradient(135deg, #101411, #1b241f 60%, #101411);
    }
    main { width: min(1280px, calc(100vw - 36px)); margin: 0 auto; padding: 30px 0 44px; }
    h1 { margin: 0; font-size: clamp(30px, 5vw, 58px); letter-spacing: -0.06em; }
    .sub { margin-top: 8px; color: var(--muted); }
    .grid { display: grid; grid-template-columns: 420px 1fr; gap: 16px; margin-top: 22px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 18px; backdrop-filter: blur(14px); }
    label { display: block; margin: 12px 0 6px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.09em; }
    input, select, textarea, button {
      width: 100%;
      color: var(--ink);
      background: rgba(0,0,0,0.26);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 11px 12px;
      outline: none;
      font: inherit;
    }
    textarea { min-height: 320px; resize: vertical; line-height: 1.45; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button { margin-top: 14px; background: linear-gradient(135deg, #8dd9c1, #f1c57d); color: #111713; font-weight: 800; cursor: pointer; }
    button.secondary { background: rgba(0,0,0,0.2); color: var(--ink); }
    pre {
      min-height: 300px;
      white-space: pre-wrap;
      overflow: auto;
      background: rgba(0,0,0,0.32);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      line-height: 1.5;
    }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
    .pill { border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; color: var(--muted); background: rgba(0,0,0,0.18); }
    .error { color: var(--bad); }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>Raw Model Generation</h1>
    <div class="sub">채팅 템플릿 없이 raw prompt를 그대로 넣어 모델 생성 상태를 확인합니다.</div>
    <div class="grid">
      <section class="card">
        <label>Checkpoint Path</label>
        <input id="ckpt_path" value="checkpoints-normal-copylayer-19000/" />
        <div class="row">
          <div>
            <label>Architecture</label>
            <select id="model_arch">
              <option value="auto">auto</option>
              <option value="plain">plain HF causal LM</option>
              <option value="krong">krong/interleave</option>
            </select>
          </div>
          <div>
            <label>Device Map</label>
            <input id="device_map" value="cuda:0" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>Dtype</label>
            <select id="dtype"><option>bf16</option><option>fp16</option><option>fp32</option></select>
          </div>
          <div>
            <label>Add BOS</label>
            <select id="add_bos"><option>auto</option><option>true</option><option>false</option></select>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Dec Max Len</label>
            <input id="dec_max_len" type="number" value="4096" />
          </div>
          <div>
            <label>Max New Tokens</label>
            <input id="max_new_tokens" type="number" value="64" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>Temperature</label>
            <input id="temperature" type="number" step="0.01" value="1.0" />
          </div>
          <div>
            <label>Top P</label>
            <input id="top_p" type="number" step="0.01" value="1.0" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>Top K</label>
            <input id="top_k" type="number" value="0" />
          </div>
          <div>
            <label>Repetition Penalty</label>
            <input id="repetition_penalty" type="number" step="0.01" value="" placeholder="optional" />
          </div>
        </div>
        <div class="row">
          <label><input id="do_sample" type="checkbox" style="width:auto;margin-right:8px" /> sampling</label>
          <label><input id="skip_special_tokens" type="checkbox" style="width:auto;margin-right:8px" checked /> skip special tokens</label>
        </div>
        <button onclick="generate()">Generate</button>
        <button class="secondary" onclick="unload()">Unload Model</button>
      </section>
      <section class="card">
        <label>Raw Prompt</label>
        <textarea id="prompt">다음 문장을 자연스럽게 이어서 작성하세요.

오늘 한국어 모델의 생성 품질을 테스트하기 위해</textarea>
        <div class="meta" id="meta"></div>
        <label>Generated Text</label>
        <pre id="output">대기 중...</pre>
      </section>
    </div>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    const val = id => $(id).value;
    const checked = id => $(id).checked;
    const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;" }[c]));
    function payload() {
      return {
        ckpt_path: val("ckpt_path"),
        model_arch: val("model_arch"),
        device_map: val("device_map"),
        dtype: val("dtype"),
        add_bos: val("add_bos"),
        dec_max_len: Number(val("dec_max_len")),
        max_new_tokens: Number(val("max_new_tokens")),
        temperature: val("temperature"),
        top_p: val("top_p"),
        top_k: val("top_k"),
        repetition_penalty: val("repetition_penalty"),
        do_sample: checked("do_sample"),
        skip_special_tokens: checked("skip_special_tokens"),
        prompt: val("prompt"),
      };
    }
    async function generate() {
      $("output").textContent = "모델 로드/생성 중...";
      $("meta").innerHTML = "";
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()),
      });
      const data = await res.json();
      if (!data.ok) {
        $("output").innerHTML = `<span class="error">${esc(data.error || "unknown error")}</span>\n\n${esc(data.traceback || "")}`;
        return;
      }
      $("output").textContent = data.text || "";
      const m = data.model || {};
      $("meta").innerHTML = [
        `model=${m.model_arch}`,
        `loaded_new=${data.loaded_new}`,
        `load=${Number(data.load_sec || 0).toFixed(2)}s`,
        `gen=${Number(data.generation_sec || 0).toFixed(2)}s`,
        `prompt_tok=${data.prompt_tokens}`,
        `gen_tok=${data.generated_tokens}`,
        `add_bos=${m.add_bos}`,
      ].map(x => `<span class="pill">${esc(x)}</span>`).join("");
    }
    async function unload() {
      const res = await fetch("/api/unload", { method: "POST" });
      const data = await res.json();
      $("meta").innerHTML = `<span class="pill">${esc(data.message || "unloaded")}</span>`;
    }
  </script>
</body>
</html>
"""


class RawGenerationHandler(BaseHTTPRequestHandler):
    server_version = "RawGenerationServer/1.0"

    @property
    def manager(self) -> GenerationManager:
        return self.server.manager  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            try:
                data = self.manager.generate(_read_json_body(self))
                _json_response(self, data)
            except Exception as exc:
                _json_response(
                    self,
                    {"ok": False, "error": str(exc), "traceback": traceback.format_exc()},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        if parsed.path == "/api/unload":
            self.manager.unload()
            _json_response(self, {"ok": True, "message": "model unloaded"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a raw-prompt text generation UI for local HF/KRong checkpoints.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--cache_root", type=str, default=DEFAULT_CACHE_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manager = GenerationManager(cache_root=args.cache_root)
    server = ThreadingHTTPServer((args.host, args.port), RawGenerationHandler)
    server.manager = manager  # type: ignore[attr-defined]
    print(f"[raw-gen] cache_root={manager.cache_paths.cache_root}")
    print(f"[raw-gen] http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[raw-gen] stopped")
    finally:
        manager.unload()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


