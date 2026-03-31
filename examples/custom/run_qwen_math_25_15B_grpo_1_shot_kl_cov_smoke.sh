#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test for end-to-end validation:
# - submit a lightweight training job
# - dump validation diagnostics
# - make it easy to render report artifacts

TRAIN_MODE=${TRAIN_MODE:-oneshot}

export EXP_NAME=${EXP_NAME:-"smoke_qwen_math_25_15B_${TRAIN_MODE}_kl_cov_$(date +%Y%m%d-%H%M%S)"}
export ENABLE_PAPER_STYLE_VIZ=${ENABLE_PAPER_STYLE_VIZ:-true}

# Keep smoke run small and fast
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
export ROLLOUT_N=${ROLLOUT_N:-4}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-4}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# Keep token diagnostics useful while avoiding huge dumps
export VAL_DIAG_MAX_TOKENS=${VAL_DIAG_MAX_TOKENS:-128}
export VAL_DIAG_DUMP_SAMPLES=${VAL_DIAG_DUMP_SAMPLES:-16}
export VAL_DIAG_DISTRIBUTION_TOPK=${VAL_DIAG_DISTRIBUTION_TOPK:-32}

# Reproducibility
export DATA_SEED=${DATA_SEED:-1}
export ROLLOUT_SEED=${ROLLOUT_SEED:-1}

echo "Smoke config:"
echo "  TRAIN_MODE=${TRAIN_MODE}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  TOTAL_EPOCHS=${TOTAL_EPOCHS}, TEST_FREQ=${TEST_FREQ}, SAVE_FREQ=${SAVE_FREQ}"
echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, ROLLOUT_N=${ROLLOUT_N}, VAL_ROLLOUT_N=${VAL_ROLLOUT_N}"

TRAIN_MODE="${TRAIN_MODE}" bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
