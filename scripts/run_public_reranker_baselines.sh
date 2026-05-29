#!/usr/bin/env bash
set -euo pipefail

cd -P /mnt/nas_server_yhw/eval_krong

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-/mnt/nas_server_yhw/envs/eval_krong/bin/python}
DATA=${DATA:-/mnt/nas_server_yhw/eval_krong/rerank_candidates/miracl_ko_hardneg_dev.jsonl}
OUTROOT=${OUTROOT:-/mnt/nas_server_yhw/eval_krong/public_reranker_results/$(date +%Y%m%d_%H%M%S)_miracl_ko_hardneg}
MAX_CANDIDATES=${MAX_CANDIDATES:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-512}

mkdir -p "$OUTROOT"

run_public_reranker () {
  local label="$1"
  local model="$2"
  local backend="$3"
  shift 3
  echo "[run] $label model=$model backend=$backend"
  "$PY" eval_public_reranker_baseline.py     --model "$model"     --backend "$backend"     --data "$DATA"     --out-json "$OUTROOT/${label}.json"     --max-candidates "$MAX_CANDIDATES"     --batch-size "$BATCH_SIZE"     --max-length "$MAX_LENGTH"     "$@"
}

# Strong public cross-encoder reranker baselines.
run_public_reranker   bge_reranker_v2_m3   BAAI/bge-reranker-v2-m3   flagembedding

run_public_reranker   jina_reranker_v2_base_multilingual   jinaai/jina-reranker-v2-base-multilingual   jina

run_public_reranker   qwen3_reranker_0p6b   Qwen/Qwen3-Reranker-0.6B   sentence-transformers   --trust-remote-code

# Optional dense retrieval-style baselines. These are not cross-encoder rerankers,
# but they are useful anchors for Korean query-passage matching.
if [ "${RUN_DENSE_BASELINES:-0}" = "1" ]; then
  run_public_reranker     bge_m3_biencoder     BAAI/bge-m3     st-biencoder

  run_public_reranker     multilingual_e5_large_biencoder     intfloat/multilingual-e5-large     st-biencoder     --query-prefix "query: "     --passage-prefix "passage: "
fi

printf '
[done] outputs: %s
' "$OUTROOT"
