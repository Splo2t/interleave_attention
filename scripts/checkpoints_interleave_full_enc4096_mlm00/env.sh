#!/usr/bin/env bash
EXP_NAME="checkpoints_interleave_full_enc4096_mlm00"
SOURCE_ROOT="/mnt/nas_server_yhw/checkpoints_interleave_full_enc4096_mlm00"
CONVERT_KIND="interleave"
MODEL_ARCH="krong"
LOG_GROUP="krong"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
