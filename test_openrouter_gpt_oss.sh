#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required" >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
INPUT_DATASET_DIR="${INPUT_DATASET_DIR:-./input/aime25}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./output}"
THREAD_NUM="${THREAD_NUM:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
OPENROUTER_PROVIDER_ORDER="${OPENROUTER_PROVIDER_ORDER:-fireworks}"
OPENROUTER_PROVIDER_QUANTIZATIONS="${OPENROUTER_PROVIDER_QUANTIZATIONS:-}"
RUN_ID="${RUN_ID:-openrouter_gptoss_fireworks_strict}"

cmd=(
  python bench.py
  --api_base_url "https://openrouter.ai/api/v1"
  --api_key "${OPENROUTER_API_KEY}"
  --model_name "${MODEL_NAME}"
  --input_dataset_dir "${INPUT_DATASET_DIR}"
  --output_base_dir "${OUTPUT_BASE_DIR}"
  --thread_num "${THREAD_NUM}"
  --temperature "${TEMPERATURE}"
  --reasoning_effort "${REASONING_EFFORT}"
  --openrouter_provider_order "${OPENROUTER_PROVIDER_ORDER}"
  --openrouter_disable_provider_fallbacks
  --run_id "${RUN_ID}"
)

if [[ -n "${OPENROUTER_PROVIDER_QUANTIZATIONS}" ]]; then
  cmd+=(--openrouter_provider_quantizations "${OPENROUTER_PROVIDER_QUANTIZATIONS}")
fi

"${cmd[@]}"
