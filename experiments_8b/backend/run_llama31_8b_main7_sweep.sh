#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

PY="${PY:-/mnt/nas_server_yhw/envs/eval_krong/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
MODEL_LABEL="${MODEL_LABEL:-llama31_8b}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nas_server_yhw/huggingface}"
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/sweep_results_8b}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_${MODEL_LABEL}_main7}"
TASKS="${TASKS:-mmlu,kmmlu,kobest,click,csatqa,arc_easy,arc_challenge,hellaswag,openbookqa}"

export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"

mkdir -p "$RESULT_BASE"

"$PY" "$REPO_ROOT/run_eval_checkpoint_sweep.py" \
  --single-ckpt-path "$MODEL" \
  --single-ckpt-name "$MODEL_LABEL" \
  --single-ckpt-step 0 \
  --tasks "$TASKS" \
  --result-root "$RESULT_BASE/$RUN_NAME" \
  --python-bin "$PY" \
  --dtype "${DTYPE:-bf16}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --cache_root "$CACHE_ROOT" \
  --k_shot "${K_SHOT:-5}" \
  --seed "${SEED:-42}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --space_variant_mode "${SPACE_VARIANT_MODE:-both}" \
  --batch_scoring "${BATCH_SCORING:-auto}" \
  --continuation_scoring "${CONTINUATION_SCORING:-oneshot}" \
  --dec_max_len "${DEC_MAX_LEN:-4096}" \
  --limit "${LIMIT:-0}" \
  --kobest_tasks "${KOBEST_TASKS:-boolq,copa,hellaswag,sentineg,wic}" \
  --kobest_split "${KOBEST_SPLIT:-test}" \
  --model_label "$MODEL_LABEL" \
  --log_group "${LOG_GROUP:-others}" \
  --experiment_tag "${EXPERIMENT_TAG:-8b_main7}" \
  --skip-existing-json
