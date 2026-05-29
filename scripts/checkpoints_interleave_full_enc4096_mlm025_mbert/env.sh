#!/usr/bin/env bash
EXP_NAME="checkpoints_interleave_full_enc4096_mlm025_mbert"
SOURCE_ROOT="/mnt/nas_server_yhw/stage2_interleave_full_enc4096_mlm025_mbert"
CONVERT_KIND="interleave"
MODEL_ARCH="krong"
LOG_GROUP="krong"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
