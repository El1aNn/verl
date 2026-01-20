#!/usr/bin/env bash
set -xeuo pipefail

# 可选：如需 WandB，请在此设置（或依赖外部已导出）
export WANDB_API_KEY="f408d6f1e1f982b94e6034176c0cd1f72cf9ab62"

# 基础路径与实验信息（来自 YAML）
HOME_DIR=${HOME_DIR:-"/root/autodl-tmp"}
project_name='verl_grpo_dsr_sub'
exp_name='1_shot_qwen_25_05B_dsr_grpo'

# Ray 相关（按需修改）
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"/root/autodl-tmp/verl/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# 路径（来自 YAML 的 paths.*）
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}/verl"}
MODEL_PATH=${MODEL_PATH:-"/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct"}
DATA_BASE=${DATA_BASE:-"/root/autodl-tmp/verl/data/gsm8k"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_BASE}/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_BASE}/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# 数据与算法（来自 YAML）
train_batch_size=16
max_prompt_length=512
max_response_length=512
filter_overlong_prompts=true
truncation='error'

adv_estimator=grpo
use_kl_in_reward=false

# 模型/训练细节（来自 YAML）
lr=1e-6
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=20
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0
actor_param_offload=false
actor_optimizer_offload=false

# rollout / ref（来自 YAML）
rollout_log_prob_micro_bsz=160
ref_log_prob_micro_bsz=160
tp_size=1
rollout_name=vllm
gpu_mem_util=0.6
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
submission_output=$(ray job submit --no-wait --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    trainer.logger=${logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=1 \
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
