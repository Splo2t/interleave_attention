#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_NAS_ROOT="/mnt/nas_server/KRongLLM/interleave_experiment"
DEFAULT_OUTPUT_PARENT="${DEFAULT_NAS_ROOT}/converted_checkpoints_for_experiments"
DEFAULT_PY="/mnt/nas_server/envs/eval_krong/bin/python"

usage() {
  cat <<'USAGE'
Convert training checkpoints into HF-evaluable checkpoint mirrors.

Usage:
  ./convert_checkpoints_for_eval.sh SOURCE_ROOT --arch krong|hf [options]

Required:
  SOURCE_ROOT
    Directory containing checkpoint-* subdirectories.

  --arch krong|hf
    Conversion path. `krong` adds KRong remote-code/processor files.
    `hf` keeps a normal AutoModelForCausalLM layout and normalizes config
    such as tie_word_embeddings=false.

Options:
  --output-parent PATH
    Parent directory for converted experiment mirrors.
    Default: /mnt/nas_server/KRongLLM/interleave_experiment/converted_checkpoints_for_experiments

  --experiment-name NAME
    Output mirror name under --output-parent.
    Default: basename of SOURCE_ROOT

  --pattern GLOB
    Checkpoint glob under SOURCE_ROOT.
    Default: checkpoint-*

  --style checkpoint19000|krong_llm
    HF KRong generation helper style.
    Default: checkpoint19000

  --template-dir PATH
    KRong HF template checkpoint directory.
    Default: ./checkpoint-19000

  --python PATH
    Python executable.
    Default: /mnt/nas_server/envs/eval_krong/bin/python if present, otherwise python3

  --attn-implementation NAME
    Optional override for attn_implementation/self_attn_backend.
    Default: flash_attention_2. Use sdpa for maximum portability.

  --tie-word-embeddings true|false
    For --arch hf, enforced in config.json.
    Default: false

  --enc-max-len N
    Optional processor_config.enc_max_len override.

  --min-prefix N
    Optional processor_config.min_prefix override.

  --add-bos true|false
    Optional processor_config.add_bos override.

  --overwrite
    Recreate existing converted checkpoint directories.

  --dry-run
    Print planned converter actions without writing files.

Examples:
  ./convert_checkpoints_for_eval.sh \
    /mnt/nas_server/KRongLLM/interleave_experiment/checkpoints-normal-copylayer \
    --arch hf \
    --overwrite

  ./convert_checkpoints_for_eval.sh \
    /mnt/nas_server/KRongLLM/interleave_experiment/checkpoints-interleave-random-enc4096-mlm05 \
    --arch krong \
    --overwrite

  ./convert_checkpoints_for_eval.sh \
    /mnt/nas_server/KRongLLM/interleave_experiment/checkpoints_1B_cpt \
    --experiment-name checkpoints-1b-cpt \
    --arch hf \
    --overwrite
USAGE
}

SOURCE_ROOT=""
OUTPUT_PARENT="${DEFAULT_OUTPUT_PARENT}"
EXPERIMENT_NAME=""
PATTERN="checkpoint-*"
ARCH=""
STYLE="checkpoint19000"
TEMPLATE_DIR="${SCRIPT_DIR}/checkpoint-19000"
PY="${PY:-}"
ATTN_IMPLEMENTATION="flash_attention_2"
TIE_WORD_EMBEDDINGS="false"
OVERWRITE=0
DRY_RUN=0
KRONG_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --output-parent)
      OUTPUT_PARENT="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --style)
      STYLE="$2"
      shift 2
      ;;
    --template-dir)
      TEMPLATE_DIR="$2"
      shift 2
      ;;
    --python)
      PY="$2"
      shift 2
      ;;
    --attn-implementation)
      ATTN_IMPLEMENTATION="$2"
      shift 2
      ;;
    --tie-word-embeddings)
      TIE_WORD_EMBEDDINGS="$2"
      shift 2
      ;;
    --enc-max-len)
      KRONG_EXTRA_ARGS+=(--enc-max-len "$2")
      shift 2
      ;;
    --min-prefix)
      KRONG_EXTRA_ARGS+=(--min-prefix "$2")
      shift 2
      ;;
    --add-bos)
      KRONG_EXTRA_ARGS+=(--add-bos "$2")
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --*)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${SOURCE_ROOT}" ]]; then
        echo "[error] multiple SOURCE_ROOT values: ${SOURCE_ROOT} and $1" >&2
        exit 2
      fi
      SOURCE_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "${SOURCE_ROOT}" ]]; then
  echo "[error] SOURCE_ROOT is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "${ARCH}" ]]; then
  echo "[error] --arch krong|hf is required" >&2
  usage >&2
  exit 2
fi

if [[ "${ARCH}" != "krong" && "${ARCH}" != "hf" ]]; then
  echo "[error] --arch must be krong or hf: ${ARCH}" >&2
  exit 2
fi

if [[ "${STYLE}" != "checkpoint19000" && "${STYLE}" != "krong_llm" ]]; then
  echo "[error] --style must be checkpoint19000 or krong_llm: ${STYLE}" >&2
  exit 2
fi

if [[ -z "${PY}" ]]; then
  if [[ -x "${DEFAULT_PY}" ]]; then
    PY="${DEFAULT_PY}"
  else
    PY="python3"
  fi
fi

if [[ "${TIE_WORD_EMBEDDINGS}" != "true" && "${TIE_WORD_EMBEDDINGS}" != "false" ]]; then
  echo "[error] --tie-word-embeddings must be true or false: ${TIE_WORD_EMBEDDINGS}" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "[error] source root not found: ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  EXPERIMENT_NAME="$(basename "${SOURCE_ROOT}")"
fi

OUTPUT_ROOT="${OUTPUT_PARENT%/}/${EXPERIMENT_NAME}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/convert}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/convert_${EXPERIMENT_NAME}_${TIMESTAMP}.log"

if [[ "${ARCH}" == "krong" && ! -d "${TEMPLATE_DIR}" ]]; then
  echo "[error] template dir not found: ${TEMPLATE_DIR}" >&2
  exit 1
fi

if [[ "${ARCH}" == "krong" ]]; then
  CMD=(
    "${PY}"
    "${SCRIPT_DIR}/convert_checkpoints_to_krong_hf.py"
    --style "${STYLE}"
    --source-root "${SOURCE_ROOT}"
    --checkpoint-pattern "${PATTERN}"
    --template-dir "${TEMPLATE_DIR}"
    --output-root "${OUTPUT_ROOT}"
    --exclude-training-artifacts
  )

  if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
    CMD+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
  fi

  CMD+=("${KRONG_EXTRA_ARGS[@]}")

  if [[ "${OVERWRITE}" -eq 1 ]]; then
    CMD+=(--overwrite-output)
  fi
else
  CMD=(
    "${PY}"
    "${SCRIPT_DIR}/sync_local_vanilla_checkpoints.py"
    --source-root "${SOURCE_ROOT}"
    --experiments-root "${OUTPUT_ROOT}"
    --checkpoint-pattern "${PATTERN}"
    --tie-word-embeddings "${TIE_WORD_EMBEDDINGS}"
    --normalize-tokenizer-config
    --exclude-training-artifacts
  )

  if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
    CMD+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
  fi

  if [[ "${OVERWRITE}" -eq 1 ]]; then
    CMD+=(--overwrite)
  fi
fi


if [[ "${DRY_RUN}" -eq 1 ]]; then
  CMD+=(--dry-run)
else
  mkdir -p "${OUTPUT_ROOT}"
  mkdir -p "${LOG_ROOT}"
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[source] ${SOURCE_ROOT}"
echo "[output] ${OUTPUT_ROOT}"
echo "[arch] ${ARCH}"
if [[ "${ARCH}" == "krong" ]]; then
  echo "[style] ${STYLE}"
  echo "[template] ${TEMPLATE_DIR}"
fi
if [[ "${ARCH}" == "hf" ]]; then
  echo "[tie_word_embeddings] ${TIE_WORD_EMBEDDINGS}"
fi
echo "[log] ${LOG_FILE}"
printf '[cmd]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${CMD[@]}"
else
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
