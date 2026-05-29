#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$COMMON_DIR/.." && pwd -P)"

PY="${PY:-/mnt/nas_server_yhw/envs/eval_krong/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nas_server_yhw/huggingface}"
CONVERTED_BASE="${CONVERTED_BASE:-/mnt/nas_server_yhw/converted_checkpoints_for_experiments}"
MIRROR_BASE="${MIRROR_BASE:-/mnt/nas_server_yhw/krong_hf_mirrors}"
MIRROR_THIN_BASE="${MIRROR_THIN_BASE:-/mnt/nas_server_yhw/krong_hf_mirrors_thin}"
RESULT_BASE="${RESULT_BASE:-$REPO_ROOT/sweep_results}"

TASKS_MAIN7="${TASKS_MAIN7:-mmlu,kmmlu,kobest,click,csatqa,arc_easy,arc_challenge,hellaswag,openbookqa}"
TASKS_EXTRA="${TASKS_EXTRA:-click,csatqa,openbookqa}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[error] missing required env: $name" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="${2:-directory}"
  if [[ ! -d "$path" ]]; then
    echo "[error] $label not found: $path" >&2
    exit 2
  fi
}

checkpoint_pattern_from_env() {
  if [[ -n "${STEP:-}" ]]; then
    echo "checkpoint-${STEP}"
  else
    echo "${CHECKPOINT_PATTERN:-checkpoint-*}"
  fi
}

append_if_true() {
  local flag_value="$1"
  local arg="$2"
  if [[ "$flag_value" == "1" || "$flag_value" == "true" ]]; then
    printf '%s\n' "$arg"
  fi
}

convert_exp() {
  require_env EXP_NAME
  require_env CONVERT_KIND
  require_env SOURCE_ROOT

  local pattern
  pattern="$(checkpoint_pattern_from_env)"
  local experiments_root="${EXPERIMENTS_ROOT:-$CONVERTED_BASE/$EXP_NAME}"
  local mirror_root="${MIRROR_ROOT:-$MIRROR_BASE/$EXP_NAME}"
  local overwrite_mirror=()
  local overwrite_experiments=()
  local overwrite_vanilla=()

  if [[ "${OVERWRITE_MIRROR:-0}" == "1" || "${OVERWRITE_MIRROR:-0}" == "true" ]]; then
    overwrite_mirror=(--overwrite-mirror)
  fi
  if [[ "${OVERWRITE_EXPERIMENTS:-0}" == "1" || "${OVERWRITE_EXPERIMENTS:-0}" == "true" ]]; then
    overwrite_experiments=(--overwrite-experiments)
  fi
  if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-0}" == "true" ]]; then
    overwrite_vanilla=(--overwrite)
  fi

  require_dir "$SOURCE_ROOT" "source root"

  case "$CONVERT_KIND" in
    interleave)
      "$PY" "$REPO_ROOT/sync_local_interleave_checkpoints.py" \
        --source-root "$SOURCE_ROOT" \
        --mirror-root "$mirror_root" \
        --experiments-root "$experiments_root" \
        --checkpoint-pattern "$pattern" \
        --mirror-attn-implementation "${MIRROR_ATTN_IMPLEMENTATION:-flash_attention_2}" \
        --experiment-attn-implementation "${EXPERIMENT_ATTN_IMPLEMENTATION:-flash_attention_2}" \
        "${overwrite_mirror[@]}" \
        "${overwrite_experiments[@]}"
      ;;
    interleave_thin)
      mirror_root="${MIRROR_ROOT:-$MIRROR_THIN_BASE/$EXP_NAME}"
      "$PY" "$REPO_ROOT/sync_local_interleave_checkpoints.py" \
        --source-root "$SOURCE_ROOT" \
        --mirror-root "$mirror_root" \
        --experiments-root "$experiments_root" \
        --checkpoint-pattern "$pattern" \
        --mirror-attn-implementation "${MIRROR_ATTN_IMPLEMENTATION:-flash_attention_2}" \
        --experiment-attn-implementation "${EXPERIMENT_ATTN_IMPLEMENTATION:-flash_attention_2}" \
        --thin-mirror \
        "${overwrite_mirror[@]}" \
        "${overwrite_experiments[@]}"
      ;;
    vanilla)
      "$PY" "$REPO_ROOT/sync_local_vanilla_checkpoints.py" \
        --source-root "$SOURCE_ROOT" \
        --experiments-root "$experiments_root" \
        --checkpoint-pattern "$pattern" \
        --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
        --tie-word-embeddings "${TIE_WORD_EMBEDDINGS:-false}" \
        --exclude-training-artifacts \
        "${overwrite_vanilla[@]}"
      ;;
    none)
      echo "[convert] $EXP_NAME uses an existing converted root: $experiments_root"
      require_dir "$experiments_root" "converted root"
      ;;
    *)
      echo "[error] unsupported CONVERT_KIND=$CONVERT_KIND" >&2
      exit 2
      ;;
  esac
}

benchmark_tasks() {
  local task_group="$1"
  local tasks="$2"
  require_env EXP_NAME

  local experiments_root="${EXPERIMENTS_ROOT:-$CONVERTED_BASE/$EXP_NAME}"
  local result_root="${RESULT_ROOT:-$RESULT_BASE/${EXP_NAME}_${task_group}}"
  local ckpt_args=()
  local arch_args=()
  local skip_args=()

  if [[ -n "${STEP:-}" ]]; then
    local ckpt="$experiments_root/checkpoint-${STEP}"
    require_dir "$ckpt" "checkpoint"
    ckpt_args=(--single-ckpt-path "$ckpt" --single-ckpt-name "checkpoint-${STEP}" --single-ckpt-step "$STEP")
  else
    require_dir "$experiments_root" "converted root"
    ckpt_args=(
      --checkpoints-root "$experiments_root"
      --checkpoint-pattern "${CHECKPOINT_PATTERN:-checkpoint-*}"
      --step-interval "${STEP_INTERVAL:-1000}"
    )
    if [[ "${START_STEP:-0}" != "0" ]]; then
      ckpt_args+=(--start-step "$START_STEP")
    fi
    if [[ "${END_STEP:-0}" != "0" ]]; then
      ckpt_args+=(--end-step "$END_STEP")
    fi
    if [[ "${MAX_CHECKPOINTS:-0}" != "0" ]]; then
      ckpt_args+=(--max-checkpoints "$MAX_CHECKPOINTS")
    fi
  fi

  if [[ -n "${MODEL_ARCH:-}" ]]; then
    arch_args=(--model_arch "$MODEL_ARCH")
  fi
  if [[ "${SKIP_EXISTING_JSON:-0}" == "1" || "${SKIP_EXISTING_JSON:-0}" == "true" ]]; then
    skip_args=(--skip-existing-json)
  fi

  mkdir -p "$result_root"

  "$PY" "$REPO_ROOT/run_eval_checkpoint_sweep.py" \
    "${ckpt_args[@]}" \
    --tasks "$tasks" \
    --result-root "$result_root" \
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
    "${arch_args[@]}" \
    --model_label "${MODEL_LABEL:-$EXP_NAME}" \
    --log_group "${LOG_GROUP:-auto}" \
    --experiment_tag "${EXPERIMENT_TAG:-$task_group}" \
    "${skip_args[@]}"
}
