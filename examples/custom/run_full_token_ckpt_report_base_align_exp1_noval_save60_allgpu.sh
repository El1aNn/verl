#!/usr/bin/env bash
set -euo pipefail

WORKING_DIR=${WORKING_DIR:-/root/rl/verl}
PYTHON_BIN=${PYTHON_BIN:-/home/vipuser/miniconda3/envs/verl/bin/python}

cd "${WORKING_DIR}"

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dsr_sub_baseline}
EXP_NAME=${EXP_NAME:-base_align_exp1_noval_save60_allgpu_20260425_151910}
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/ckpts/${PROJECT_NAME}/${EXP_NAME}"}
BASE_MODEL=${BASE_MODEL:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B}
VAL_PATH=${VAL_PATH:-"${WORKING_DIR}/data/testset"}
OVERLEAF_DIR=${OVERLEAF_DIR:-"${WORKING_DIR}/analysis/overleaf_bundle_20260406"}

STEPS=${STEPS:-0,60,120,180}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-32}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-4}
RUN_LABEL=${RUN_LABEL:-"${EXP_NAME}_subset${LIMIT_SAMPLES}_n${VAL_ROLLOUT_N}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${WORKING_DIR}/analysis/full_token_ckpt_reports/${RUN_LABEL}"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-8}
DIAGNOSTIC_TOPK=${DIAGNOSTIC_TOPK:-8}
DIAGNOSTIC_TOKEN_CHUNK_SIZE=${DIAGNOSTIC_TOKEN_CHUNK_SIZE:-0}
DTYPE=${DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
SEED=${SEED:-0}

cmd=(
    "${PYTHON_BIN}" scripts/offline_full_token_ckpt_report.py
    --ckpts-dir "${CKPTS_DIR}"
    --base-model "${BASE_MODEL}"
    --steps "${STEPS}"
    --val-path "${VAL_PATH}"
    --run-label "${RUN_LABEL}"
    --output-dir "${OUTPUT_DIR}"
    --merge-missing
    --n "${VAL_ROLLOUT_N}"
    --prompt-batch-size "${PROMPT_BATCH_SIZE}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-new-tokens "${MAX_RESPONSE_LENGTH}"
    --diagnostic-topk "${DIAGNOSTIC_TOPK}"
    --diagnostic-token-chunk-size "${DIAGNOSTIC_TOKEN_CHUNK_SIZE}"
    --dtype "${DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --overleaf-dir "${OVERLEAF_DIR}"
    --seed "${SEED}"
)

if [ -n "${LIMIT_SAMPLES}" ]; then
    cmd+=(--limit-samples "${LIMIT_SAMPLES}")
fi

if [ "${REPORT_ONLY:-false}" = "true" ]; then
    cmd+=(--report-only)
fi

if [ "${SKIP_EXISTING:-false}" = "true" ]; then
    cmd+=(--skip-existing)
fi

if [ "${FORCE_MERGE:-false}" = "true" ]; then
    cmd+=(--force-merge)
fi

if [ "${NO_SAMPLE:-false}" = "true" ]; then
    cmd+=(--no-do-sample)
fi

if [ "${EMBED_TOKEN_DETAILS:-false}" = "true" ]; then
    cmd+=(--embed-token-details)
fi

if [ "${NO_GZIP_TOKEN_RECORDS:-false}" = "true" ]; then
    cmd+=(--no-gzip-token-records)
fi

if [ "${NO_DECODE_TOKEN_TEXT:-false}" = "true" ]; then
    cmd+=(--no-decode-token-text)
fi

if [ "${EMPTY_CACHE_BETWEEN_CHUNKS:-false}" = "true" ]; then
    cmd+=(--empty-cache-between-chunks)
fi

echo "Running full-token checkpoint report:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"
