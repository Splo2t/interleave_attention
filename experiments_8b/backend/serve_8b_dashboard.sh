#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

PY="${PY:-/mnt/nas_server_yhw/envs/eval_krong/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/sweep_results_8b}"
FRONTEND_HTML="${FRONTEND_HTML:-$REPO_ROOT/experiments_8b/frontend/dashboard_frontend_8b.html}"

exec "$PY" "$REPO_ROOT/serve_eval_dashboard.py" \
  --compare-root "$RESULT_ROOT" \
  --html-path "$FRONTEND_HTML" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7816}" \
  --refresh-ms "${REFRESH_MS:-3000}"
