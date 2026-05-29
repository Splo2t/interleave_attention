#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("convert_checkpoints_to_krong_hf.py")


def main() -> int:
    sys.argv = [str(SCRIPT), "--style", "krong_llm", *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
