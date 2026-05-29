from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CACHE_ROOT = os.environ.get(
    "KRONG_EVAL_CACHE_ROOT",
    os.path.expanduser("~/.cache/krong_eval"),
)


@dataclass(frozen=True)
class CachePaths:
    cache_root: str
    hf_home: str
    transformers_cache: str
    datasets_cache: str


def prepare_cache_paths(cache_root: str) -> CachePaths:
    root = os.path.abspath(os.path.expanduser(cache_root or DEFAULT_CACHE_ROOT))
    hf_home = os.path.join(root, "huggingface")
    transformers_cache = os.path.join(hf_home, "transformers")
    datasets_cache = os.path.join(hf_home, "datasets")

    for path in (root, hf_home, transformers_cache, datasets_cache):
        os.makedirs(path, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache

    return CachePaths(
        cache_root=root,
        hf_home=hf_home,
        transformers_cache=transformers_cache,
        datasets_cache=datasets_cache,
    )
