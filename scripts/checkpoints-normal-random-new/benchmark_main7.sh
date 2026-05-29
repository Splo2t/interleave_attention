#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/../common.sh"
benchmark_tasks "main7" "$TASKS_MAIN7"
