#!/usr/bin/env bash
set -xeuo pipefail

# ============= 环境变量配置 =============
export WANDB_API_KEY="${WANDB_API_KEY:-f408d6f1e1f982b94e6034176c0cd1f72cf9ab62}"

# ============= Ray 配置 (可通过环境变量覆盖) =============
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"/root/autodl-tmp/verl/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# ============= 路径配置 (可通过环境变量覆盖) =============
HOME_DIR=${HOME_DIR:-"/root/autodl-tmp"}
LOG_DIR="${HOME_DIR}/verl/logs"
mkdir -p "${LOG_DIR}"

# ============= 日志文件配置 =============
project_name='DAPO'
exp_name='DAPO-Qwen2.5-0.5B'
LOG_FILE="${LOG_DIR}/${project_name}-${exp_name}-$(date +'%Y%m%d-%H%M%S').log"

# ============= 提交 Ray Job =============
echo "Submitting Ray job with YAML config file..."
submission_output=$(ray job submit --no-wait --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    --config-path=examples/custom/config \
    --config-name=dapo_qwen2.5 \
    ray.nnodes="${NNODES}")

# Extract the submission ID and clean it from ANSI color codes
submission_id_raw=$(echo "$submission_output" | grep -o "raysubmit_[^']*" | tail -n 1)
submission_id=$(echo "$submission_id_raw" | sed 's/\x1b\[[0-9;]*m//g')

if [ -n "$submission_id" ]; then
    echo "Job submitted with ID: ${submission_id}"
    echo "Streaming logs to ${LOG_FILE}"
    # Run the log streaming in the background
    ray job logs "${submission_id}" --follow > "${LOG_FILE}" 2>&1 &
    echo "Logs are being written in the background. You can check the file: ${LOG_FILE}"
else
    echo "Failed to get submission ID."
    echo "Submission output:"
    echo "$submission_output"
fi

