#!/usr/bin/env bash
set -euo pipefail

# Fill the paper-ready paired-flip core table for KoBEST stress tests.
# This intentionally re-runs KoBEST clean/stress with --save_item_predictions,
# because aggregate stress JSONs cannot be used for paired flip analysis.

cd -P /mnt/nas_server_yhw/eval_krong

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/mnt/nas_server_yhw/envs/eval_krong/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nas_server_yhw/huggingface}"
OUTROOT="${OUTROOT:-/mnt/nas_server_yhw/eval_krong/variant_eval_outputs_paper/paired_flip_core}"
LOGROOT="$OUTROOT/logs"

mkdir -p "$OUTROOT" "$LOGROOT"

KOBEST_TASKS="boolq,copa,hellaswag,sentineg,wic"
VARIANTS=(
  ko_random_p25
  ko_random_p50
  ko_josa_preserve_compaction_hard
)

run_clean() {
  local label="$1"
  local ckpt="$2"
  local arch="$3"
  local extra_args=()
  if [[ "$arch" == "krong" ]]; then
    extra_args+=(--model_arch krong)
  fi

  echo "[clean] $label"
  "$PY" eval_variant_hf_krong.py \
    --ckpt_path "$ckpt" \
    --model_label "$label" \
    --task kobest \
    --dtype bf16 \
    --device_map cuda:0 \
    --cache_root "$CACHE_ROOT" \
    --k_shot 5 \
    --seed 42 \
    --eval_batch_size 1 \
    --space_variant_mode both \
    --batch_scoring auto \
    --continuation_scoring oneshot \
    --dec_max_len 4096 \
    --limit 0 \
    --kobest_tasks "$KOBEST_TASKS" \
    --kobest_split test \
    --log_root "$LOGROOT" \
    --out_json "$OUTROOT/${label}_clean.json" \
    --save_item_predictions \
    "${extra_args[@]}"
}

run_stress() {
  local label="$1"
  local ckpt="$2"
  local arch="$3"
  local variant="$4"
  local extra_args=()
  if [[ "$arch" == "krong" ]]; then
    extra_args+=(--model_arch krong)
  fi

  echo "[stress] $label $variant"
  "$PY" eval_variant_hf_krong.py \
    --ckpt_path "$ckpt" \
    --model_label "$label" \
    --task kobest_variant \
    --variant_data_root "/mnt/nas_server_yhw/eval_krong/variant_benchmarks/$variant" \
    --variant_name "$variant" \
    --dtype bf16 \
    --device_map cuda:0 \
    --cache_root "$CACHE_ROOT" \
    --k_shot 5 \
    --seed 42 \
    --eval_batch_size 1 \
    --space_variant_mode both \
    --batch_scoring auto \
    --continuation_scoring oneshot \
    --dec_max_len 4096 \
    --limit 0 \
    --kobest_tasks "$KOBEST_TASKS" \
    --kobest_split test \
    --log_root "$LOGROOT" \
    --out_json "$OUTROOT/${label}_${variant}.json" \
    --save_item_predictions \
    "${extra_args[@]}"
}

run_paired() {
  local label="$1"
  local variant="$2"

  echo "[paired] $label $variant"
  "$PY" analyze_paired_flips.py \
    --original-json "$OUTROOT/${label}_clean.json" \
    --stress-json "$OUTROOT/${label}_${variant}.json" \
    --out-json "$OUTROOT/${label}_${variant}_paired_flips.json" \
    --out-csv "$OUTROOT/${label}_${variant}_paired_flips.csv"
}

run_model() {
  local label="$1"
  local ckpt="$2"
  local arch="$3"

  run_clean "$label" "$ckpt" "$arch"
  for variant in "${VARIANTS[@]}"; do
    run_stress "$label" "$ckpt" "$arch" "$variant"
    run_paired "$label" "$variant"
  done
}

# Paper core internal models.
run_model \
  stage2_mlm00_interleave_18000 \
  /mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints_interleave_full_enc4096_mlm00/checkpoint-18000 \
  krong

run_model \
  normal_random_new_18000 \
  /mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-normal-random-new/checkpoint-18000 \
  plain

run_model \
  token_only_1b_cpt_19000 \
  /mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-1b-cpt/checkpoint-19000 \
  plain

run_model \
  mbert5_encoder_interleave_18000 \
  /mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints_interleave_full_enc4096_mlm025_mbert/checkpoint-18000 \
  krong

echo "[done] paired flip core outputs are in $OUTROOT"
