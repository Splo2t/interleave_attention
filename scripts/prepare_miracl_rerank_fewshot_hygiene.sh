#!/usr/bin/env bash
set -euo pipefail

# Build fixed few-shot demos for korean_rerank.
#
# Note: datalama/miracl-hard-negatives provides Korean hard negatives only for
# the dev split. Therefore the default path below creates a disjoint-dev setup:
#   1. build the full dev candidate pool,
#   2. select fixed 2 positive + 2 negative demos,
#   3. remove those demo query IDs from the evaluation dev pool.
#
# If you later prepare a true train candidate file, pass it directly as
# --rerank_fewshot_data when running evaluation.

cd -P /mnt/nas_server_yhw/eval_krong

PY=${PY:-/mnt/nas_server_yhw/envs/eval_krong/bin/python}
OUT_DIR=${OUT_DIR:-rerank_candidates}
CACHE_ROOT=${CACHE_ROOT:-/mnt/nas_server_yhw/huggingface}

DEV_CANDIDATES_FULL=${DEV_CANDIDATES_FULL:-$OUT_DIR/miracl_ko_hardneg_dev_full.jsonl}
DEV_CANDIDATES=${DEV_CANDIDATES:-$OUT_DIR/miracl_ko_hardneg_dev_excl_fixed4_seed42.jsonl}
FIXED_FEWSHOT=${FIXED_FEWSHOT:-$OUT_DIR/miracl_ko_dev_fixed_4shot_seed42.jsonl}

mkdir -p "$OUT_DIR"

echo "[1/4] Build full dev controlled-hard-negative candidates"
"$PY" prepare_miracl_ko_hardneg_rerank.py \
  --split dev \
  --cache-dir "$CACHE_ROOT/datasets" \
  --out-jsonl "$DEV_CANDIDATES_FULL" \
  --max-candidates 100 \
  --require-positive

echo "[2/4] Create fixed 4-shot demos"
"$PY" prepare_rerank_fewshot_demos.py \
  --candidate-jsonl "$DEV_CANDIDATES_FULL" \
  --out-jsonl "$FIXED_FEWSHOT" \
  --num-positives 2 \
  --num-negatives 2 \
  --seed 42 \
  --prefer-low-rank

echo "[3/4] Remove demo query IDs from evaluation dev pool"
"$PY" filter_rerank_queries.py \
  --input-jsonl "$DEV_CANDIDATES_FULL" \
  --out-jsonl "$DEV_CANDIDATES" \
  --exclude-jsonl "$FIXED_FEWSHOT"

echo "[4/4] Verify fixed demo and eval files"
echo "[done] fixed few-shot demos: $FIXED_FEWSHOT"
echo "[done] eval candidates: $DEV_CANDIDATES"
