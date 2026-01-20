#!/usr/bin/env bash
set -xeuo pipefail

# 可选：如需 WandB，请在此设置（或依赖外部已导出）
export WANDB_API_KEY="f408d6f1e1f982b94e6034176c0cd1f72cf9ab62"

# 基础路径与实验信息（来自 YAML）
HOME_DIR=${HOME_DIR:-"/root/rl"}
project_name='verl_grpo_dsr_sub'
exp_name='qwen_25_15B_dsr_grpo'

# Ray 相关（按需修改）
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
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

# 检查 Ray 是否运行，如果没有则启动
if ! $RAY_CMD status --address "$RAY_ADDRESS" > /dev/null 2>&1; then
    echo "Ray is not running at $RAY_ADDRESS. Starting local Ray cluster..."
    # 清理可能存在的残留进程
    $RAY_CMD stop --force || true
    $RAY_CMD start --head --port 6379 --dashboard-port 8265 --num-gpus ${N_GPUS} --disable-usage-stats
else
    echo "Ray is running at $RAY_ADDRESS."
fi

# 路径（来自 YAML 的 paths.*）
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}/verl"}
MODEL_PATH=${MODEL_PATH:-"/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct"}
DATA_BASE=${DATA_BASE:-"${HOME_DIR}/verl/data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_BASE}/dsr_sub/train.parquet"}
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

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# 数据与算法（来自 YAML）
#train_batch_size÷ppo_mini_batch_size=整数
# 您的配置: 
# 8 ÷ 4 = 2
# 结果: 每次采集数据后，会进行 2 次参数更新步骤。
train_batch_size=4
max_prompt_length=1024
max_response_length=1024
filter_overlong_prompts=true
truncation='error'

adv_estimator=grpo
use_kl_in_reward=false

# 模型/训练细节（来自 YAML）
lr=1e-6
ppo_mini_batch_size=2
ppo_micro_batch_size_per_gpu=1
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0
actor_param_offload=false
actor_optimizer_offload=false

# rollout / ref（来自 YAML）
rollout_log_prob_micro_bsz=16
ref_log_prob_micro_bsz=32
#((ppo_mini_batch_size × rollout_n)/ DP size) (mod ppo_micro_batch_size_per_gpu)==0

tp_size=1 #gpu数
rollout_name=vllm
gpu_mem_util=0.7
#每个 GPU 的样本数÷ppo_micro_batch_size_per_gpu=整数
rollout_n=5
ref_param_offload=true

# 训练器（来自 YAML）
logger=wandb
critic_warmup=0
log_val_generations=1
save_freq=20
test_freq=5
total_epochs=1

# 日志输出
LOG_DIR="${HOME_DIR}/verl/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${project_name}-${exp_name}-$(date +'%Y%m%d-%H%M%S').log"

# 提交 Ray 任务（使用 Hydra 覆盖键，等价于 YAML 中的配置）
# 注意：data.val_files 使用了双引号包裹的列表字符串
# 设置 PYTHONPATH 确保优先加载本地代码
export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH:-}"

submission_output=$(ray job submit --no-wait --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    trainer.logger=${logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes="${NNODES}" \
    trainer.critic_warmup=${critic_warmup} \
    trainer.log_val_generations=${log_val_generations} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto)

# 提取提交 ID 并打印日志
submission_id_raw=$(echo "$submission_output" | grep -o "raysubmit_[^']*" | tail -n 1)
submission_id=$(echo "$submission_id_raw" | sed 's/\x1b\[[0-9;]*m//g')

if [ -n "$submission_id" ]; then
    echo "Job submitted with ID: ${submission_id}"
    echo "Streaming logs to ${LOG_FILE}"
    ray job logs "${submission_id}" --follow > "${LOG_FILE}" 2>&1 &
    echo "Logs are being written in the background. You can check the file: ${LOG_FILE}"
else
    echo "Failed to get submission ID."
    echo "Submission output:"
    echo "$submission_output"
fi
