#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

export SAVE_RUN_SCRIPT=false
export EXP_NAME=${EXP_NAME:-base_c_n8_oneshot_swanlab_safe_$(date +%Y%m%d-%H%M%S)}
export TRAIN_MODE=${TRAIN_MODE:-oneshot}
export MODEL_PATH=${MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B}

# Keep dataset default same as failed run; override externally if needed.
export ONE_SHOT_TRAIN_FILE=${ONE_SHOT_TRAIN_FILE:-/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet}
export TRAIN_FILE=${TRAIN_FILE:-${ONE_SHOT_TRAIN_FILE}}

# Base-C-n8 core
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}

# OOM-safe adjustments
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
export ROLLOUT_LOG_PROB_MICRO_BSZ=${ROLLOUT_LOG_PROB_MICRO_BSZ:-8}
export REF_LOG_PROB_MICRO_BSZ=${REF_LOG_PROB_MICRO_BSZ:-16}
export ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true}
# vLLM memory pool is incompatible with expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF

# Training/eval cadence
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
export TEST_FREQ=${TEST_FREQ:-6}
export SAVE_FREQ=${SAVE_FREQ:-180}

# Logging
export LOGGER=${LOGGER:-'["swanlab","file"]'}
export ENABLE_VAL_DIAGNOSTICS=${ENABLE_VAL_DIAGNOSTICS:-true}
export ENABLE_PAPER_STYLE_VIZ=${ENABLE_PAPER_STYLE_VIZ:-false}

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh
