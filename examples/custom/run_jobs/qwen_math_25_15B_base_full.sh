#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

export SAVE_RUN_SCRIPT=false
export LOGGER=${LOGGER:-'["swanlab","file"]'}
export HOME_DIR=/root/rl
export TRAIN_MODE=full
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B
export EXP_NAME=${EXP_NAME:-qwen_math_25_15B_base_full_$(date +%Y%m%d-%H%M%S)}

# Formal-run defaults. Override in shell if needed.
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
export TEST_FREQ=${TEST_FREQ:-6}
export SAVE_FREQ=${SAVE_FREQ:-100}
export ENABLE_VAL_DIAGNOSTICS=${ENABLE_VAL_DIAGNOSTICS:-true}
export VAL_DIAG_MAX_SAMPLES=${VAL_DIAG_MAX_SAMPLES:-64}
export ENABLE_PAPER_STYLE_VIZ=${ENABLE_PAPER_STYLE_VIZ:-false}

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh
