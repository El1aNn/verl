#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_API_KEY="f408d6f1e1f982b94e6034176c0cd1f72cf9ab62"
export SWANLAB_API_KEY="B2gwMFDhC9KMZAu6T8UXL"
ray stop
HOME_DIR=${HOME_DIR:-"/root/rl"}
project_name='DAPO'
exp_name=${EXP_NAME:-"DAPO-Qwen2.5-1.5B-base-1-shot_$(date +%Y%m%d-%H%M%S)"}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
max_response_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=32
ppo_micro_batch_size_per_gpu=8

# Ray
RAY_ADDRESS_INPUT=${RAY_ADDRESS:-""}
RAY_GCS_ADDRESS=${RAY_GCS_ADDRESS:-""}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-""}

if [ -n "${RAY_DASHBOARD_ADDRESS}" ] || [[ "${RAY_ADDRESS_INPUT}" =~ ^https?:// ]]; then
    # Dashboard address is explicitly provided
    if [ -n "${RAY_DASHBOARD_ADDRESS}" ]; then
        RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS}"
    else
        RAY_DASHBOARD_ADDRESS="${RAY_ADDRESS_INPUT}"
    fi
    # Derive host and default GCS port
    _ray_dash_no_scheme="${RAY_DASHBOARD_ADDRESS#http://}"
    _ray_dash_no_scheme="${_ray_dash_no_scheme#https://}"
    _ray_head_host="${_ray_dash_no_scheme%%:*}"
    if [ -z "${RAY_GCS_ADDRESS}" ]; then
        RAY_GCS_ADDRESS="${_ray_head_host}:6379"
    fi
else
    # Treat as GCS address (or default to local)
    if [ -n "${RAY_GCS_ADDRESS}" ]; then
        RAY_GCS_ADDRESS="${RAY_GCS_ADDRESS}"
    else
        RAY_GCS_ADDRESS="${RAY_ADDRESS_INPUT:-"127.0.0.1:6379"}"
    fi
    _ray_head_host="${RAY_GCS_ADDRESS%%:*}"
    if [ -z "${RAY_DASHBOARD_ADDRESS}" ]; then
        RAY_DASHBOARD_ADDRESS="http://${_ray_head_host}:8265"
    fi
fi

# 训练代码内部会读取环境变量 RAY_ADDRESS 作为 Ray Core(GCS) 地址
export RAY_ADDRESS="${RAY_GCS_ADDRESS}"

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${HOME_DIR}/verl/verl/trainer/runtime_env.yaml"}

# 自动检测 GPU 数量作为每节点的 GPU 数
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NNODES=${NNODES:-1}

echo "Detected ${N_GPUS} GPUs. Setting trainer.n_gpus_per_node=${N_GPUS}"

# 获取 ray 命令路径
RAY_CMD=$(which ray || true)
if [ -z "$RAY_CMD" ]; then
    PYTHON_PATH=$(which python3 || true)
    if [ -n "$PYTHON_PATH" ]; then
        RAY_CMD=$(dirname "$PYTHON_PATH")/ray
    fi
    # Fallback to known path if still not found
    if [ -z "$RAY_CMD" ] || [ ! -x "$RAY_CMD" ]; then
        RAY_CMD="/root/miniconda3/envs/verl_debug/bin/ray"
    fi
fi

if [ ! -x "$RAY_CMD" ]; then
    echo "Error: ray command not found. Please ensure ray is installed and in PATH."
    exit 1
fi

# 检查 Ray 是否运行，如果没有则启动（仅对本机 head 进行自动启动）
if ! "$RAY_CMD" status --address "${RAY_GCS_ADDRESS}" > /dev/null 2>&1; then
    if [ "${_ray_head_host}" = "127.0.0.1" ] || [ "${_ray_head_host}" = "localhost" ]; then
        echo "Ray is not running at ${RAY_GCS_ADDRESS}. Starting local Ray cluster..."
        # 清理可能存在的残留进程
        "$RAY_CMD" stop --force || true
        "$RAY_CMD" start --head --port 6379 --dashboard-host 0.0.0.0 --dashboard-port 8265 --num-gpus "${N_GPUS}" --disable-usage-stats
    else
        echo "Error: cannot reach Ray cluster at ${RAY_GCS_ADDRESS}."
        echo "- If your head node is ${_ray_head_host}, start Ray there (ensure dashboard listens on 0.0.0.0:8265)."
        echo "- Then set: RAY_GCS_ADDRESS='${_ray_head_host}:6379' and RAY_DASHBOARD_ADDRESS='http://${_ray_head_host}:8265'"
        exit 1
    fi
else
    echo "Ray is running at ${RAY_GCS_ADDRESS} (dashboard: ${RAY_DASHBOARD_ADDRESS})."
fi

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}/verl"}
MODEL_PATH=${MODEL_PATH:-"/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
DATA_BASE=${DATA_BASE:-"${HOME_DIR}/verl/data"}
# TRAIN_FILE=${TRAIN_FILE:-"${DATA_BASE}/dsr_sub/pi1_one_ans.parquet"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_BASE}/dsr_sub/pi1_one_ans.parquet"}

VAL_PATH=${VAL_PATH:-"${DATA_BASE}/testset"}

# 动态获取 VAL_PATH 下所有 parquet 文件
# 使用 find 命令查找所有 .parquet 文件
val_files_list=$(find "${VAL_PATH}" -type f -name "*.parquet")

# 将文件列表转换为 Python 列表格式的字符串: ['file1', 'file2', ...]
if [ -z "$val_files_list" ]; then
    VAL_FILES="[]"
    echo "Warning: No parquet files found in ${VAL_PATH}"
else
    formatted_val_files=$(echo "$val_files_list" | sed "s|^|'|;s|$|'|" | paste -sd, -)
    VAL_FILES="[${formatted_val_files}]"
fi
echo "VAL_FILES set to: ${VAL_FILES}"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1 # sequence parallel size 和 GPU数量有关，单卡建议设为1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1 # generation tensor parallel size， 单卡建议设为1

# Create a directory for logs if it doesn't exist
LOG_DIR="${HOME_DIR}/verl/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${project_name}-${exp_name}-$(date +'%Y%m%d-%H%M%S').log"

# Submit the job and capture the submission ID
submission_output=$(ray job submit --no-wait --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='swanlab' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=6 \
    trainer.save_freq=100 \
    trainer.total_epochs=30 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}/validation" \
    trainer.resume_mode=auto)

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

