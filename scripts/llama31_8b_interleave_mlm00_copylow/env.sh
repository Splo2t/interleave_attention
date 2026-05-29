#!/usr/bin/env bash
EXP_NAME="llama31_8b_interleave_mlm00_copylow"
SOURCE_ROOT="/mnt/nas_server_yhw/checkpoints_llama8b"
CONVERT_KIND="interleave_thin"
MODEL_ARCH="krong"
LOG_GROUP="krong"
RESULT_BASE="${RESULT_BASE:-/mnt/nas_server_yhw/eval_krong/sweep_results_8b}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
