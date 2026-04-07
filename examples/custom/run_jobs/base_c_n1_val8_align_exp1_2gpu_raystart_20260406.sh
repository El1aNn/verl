#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"

if [[ "${1:-}" != "--foreground" ]]; then
    SESSION_NAME="${SESSION_NAME:-base_c_n1_val8_align}"
    if screen -list | grep -q "[[:space:]]${SESSION_NAME}[[:space:]]"; then
        echo "screen session already exists: ${SESSION_NAME}"
        echo "attach: screen -r ${SESSION_NAME}"
        exit 1
    fi
    screen -dmS "${SESSION_NAME}" bash "${SCRIPT_PATH}" --foreground
    echo "started in screen: ${SESSION_NAME}"
    echo "attach: screen -r ${SESSION_NAME}"
    exit 0
fi

cd /root/rl/verl

if [[ -f /home/vipuser/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/vipuser/miniconda3/etc/profile.d/conda.sh
    conda activate verl
else
    export PATH="/home/vipuser/miniconda3/envs/verl/bin:${PATH}"
fi

if [[ -f /root/.bashrc ]]; then
    SWANLAB_API_KEY_FROM_BASHRC="$(sed -n "s/^export SWANLAB_API_KEY=['\"]\\(.*\\)['\"]$/\\1/p" /root/.bashrc | tail -n 1)"
    if [[ -n "${SWANLAB_API_KEY_FROM_BASHRC}" ]]; then
        export SWANLAB_API_KEY="${SWANLAB_API_KEY_FROM_BASHRC}"
    fi
    unset SWANLAB_API_KEY_FROM_BASHRC
fi

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_ADDRESS="127.0.0.1:6379"
export RAY_GCS_ADDRESS="127.0.0.1:6379"
export RAY_DASHBOARD_ADDRESS="http://127.0.0.1:8265"
unset PYTORCH_CUDA_ALLOC_CONF

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export RAY_DEBUG_POST_MORTEM=0

EXP_NAME="${EXP_NAME:-base_c_n1_val8_align_exp1_2gpu_$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet}"
VAL_PATH="${VAL_PATH:-/root/rl/verl/data/testset}"
RUNTIME_ENV="${RUNTIME_ENV:-/root/rl/verl/examples/custom/run_jobs/runtime_env_base_c_n1_group_ablation_rerun180_20260405_212235.yaml}"
CKPTS_DIR="${CKPTS_DIR:-/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/${EXP_NAME}}"
LOG_FILE="${LOG_FILE:-/root/rl/verl/logs/verl_grpo_dsr_sub_baseline-${EXP_NAME}.log}"
REPORT_DIR="${REPORT_DIR:-/root/rl/verl/reports/verl_grpo_dsr_sub_baseline/${EXP_NAME}}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/verl_torch_compile_cache/triton/${EXP_NAME}}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/verl_torch_compile_cache/inductor/${EXP_NAME}}"
RESOLVED_RUNTIME_ENV="/tmp/verl_runtime_env_${EXP_NAME}.yaml"

mkdir -p "$(dirname "${LOG_FILE}")" "${CKPTS_DIR}" "${REPORT_DIR}" \
    "$(dirname "${TRITON_CACHE_DIR}")" "$(dirname "${TORCHINDUCTOR_CACHE_DIR}")"

export TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-0}"
export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-verl.utils.triton_cache_manager:LenientFileCacheManager}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONPATH="/root/rl/verl:${PYTHONPATH:-}"

mapfile -t VAL_FILE_ARRAY < <(find "${VAL_PATH}" -type f -name '*.parquet' | sort)
if [[ "${#VAL_FILE_ARRAY[@]}" -eq 0 ]]; then
    echo "no validation parquet files found under ${VAL_PATH}"
    exit 1
fi

VAL_FILES="["
for idx in "${!VAL_FILE_ARRAY[@]}"; do
    if [[ "${idx}" -gt 0 ]]; then
        VAL_FILES+=","
    fi
    VAL_FILES+="'${VAL_FILE_ARRAY[$idx]}'"
done
VAL_FILES+="]"

grep -vE '^[[:space:]]+(WANDB_API_KEY|SWANLAB_API_KEY|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR|TORCHINDUCTOR_FX_GRAPH_CACHE|TRITON_CACHE_MANAGER|TORCHINDUCTOR_COMPILE_THREADS):' "${RUNTIME_ENV}" > "${RESOLVED_RUNTIME_ENV}"
printf '  TRITON_CACHE_DIR: "%s"\n' "${TRITON_CACHE_DIR}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TORCHINDUCTOR_CACHE_DIR: "%s"\n' "${TORCHINDUCTOR_CACHE_DIR}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TORCHINDUCTOR_FX_GRAPH_CACHE: "%s"\n' "${TORCHINDUCTOR_FX_GRAPH_CACHE}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TRITON_CACHE_MANAGER: "%s"\n' "${TRITON_CACHE_MANAGER}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TORCHINDUCTOR_COMPILE_THREADS: "%s"\n' "${TORCHINDUCTOR_COMPILE_THREADS}" >> "${RESOLVED_RUNTIME_ENV}"
if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
    printf '  SWANLAB_API_KEY: "%s"\n' "${SWANLAB_API_KEY}" >> "${RESOLVED_RUNTIME_ENV}"
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "==== launch info ===="
echo "time: $(date '+%F %T %Z')"
echo "exp_name: ${EXP_NAME}"
echo "log_file: ${LOG_FILE}"
echo "ckpts_dir: ${CKPTS_DIR}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "train_file: ${TRAIN_FILE}"
echo "val_files: ${#VAL_FILE_ARRAY[@]}"
echo "runtime_env: ${RUNTIME_ENV}"
echo "====================="

ray stop --force || true
ray start --head --port 6379 --dashboard-host 0.0.0.0 --dashboard-port 8265 --num-gpus 2 --disable-usage-stats

submission_output=$(
    ray job submit --no-wait \
        --address="${RAY_DASHBOARD_ADDRESS}" \
        --runtime-env="${RESOLVED_RUNTIME_ENV}" \
        --working-dir /root/rl/verl \
        -- python3 -m verl.trainer.main_ppo \
        "data.train_files=${TRAIN_FILE}" \
        "data.val_files=${VAL_FILES}" \
        "data.filter_overlong_prompts=true" \
        "data.truncation=error" \
        "data.max_prompt_length=1024" \
        "data.max_response_length=2048" \
        "data.train_batch_size=64" \
        "algorithm.adv_estimator=grpo" \
        "algorithm.use_kl_in_reward=false" \
        "actor_rollout_ref.model.path=${MODEL_PATH}" \
        "actor_rollout_ref.model.use_remove_padding=true" \
        "actor_rollout_ref.model.enable_gradient_checkpointing=true" \
        "actor_rollout_ref.actor.optim.lr=1e-6" \
        "actor_rollout_ref.actor.ppo_mini_batch_size=32" \
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2" \
        "actor_rollout_ref.actor.use_kl_loss=true" \
        "actor_rollout_ref.actor.kl_loss_coef=0.001" \
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl" \
        "actor_rollout_ref.actor.use_torch_compile=false" \
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true" \
        "actor_rollout_ref.actor.entropy_coeff=0.001" \
        "actor_rollout_ref.actor.fsdp_config.param_offload=false" \
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false" \
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8" \
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
        "actor_rollout_ref.rollout.name=vllm" \
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.8" \
        "actor_rollout_ref.rollout.calculate_log_probs=true" \
        "actor_rollout_ref.rollout.n=1" \
        "actor_rollout_ref.rollout.val_kwargs.n=8" \
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true" \
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0" \
        "actor_rollout_ref.ref.use_torch_compile=false" \
        "actor_rollout_ref.ref.entropy_from_logits_with_chunking=true" \
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16" \
        "actor_rollout_ref.ref.fsdp_config.param_offload=false" \
        'trainer.logger=["swanlab"]' \
        "trainer.project_name=verl_grpo_dsr_sub_baseline" \
        "trainer.experiment_name=${EXP_NAME}" \
        "trainer.n_gpus_per_node=2" \
        "trainer.nnodes=1" \
        "trainer.critic_warmup=0" \
        "trainer.log_val_generations=1" \
        "trainer.save_freq=180" \
        "trainer.test_freq=6" \
        "trainer.total_epochs=10" \
        "trainer.default_local_dir=${CKPTS_DIR}" \
        "trainer.validation_data_dir=${CKPTS_DIR}/validation" \
        "trainer.validation_diagnostics.enabled=true" \
        "trainer.validation_diagnostics.compare_to_previous_eval=true" \
        "++trainer.validation_diagnostics.max_samples=64" \
        "trainer.validation_diagnostics.samples_to_dump_token_details=8" \
        "trainer.validation_diagnostics.max_tokens_per_sample=96" \
        "trainer.validation_diagnostics.tail_tokens=32" \
        "trainer.validation_diagnostics.low_confidence_prob_threshold=0.2" \
        "trainer.validation_diagnostics.track_eos_probability=true" \
        "trainer.validation_diagnostics.eos_high_prob_threshold=0.1" \
        "trainer.validation_diagnostics.token_distribution_topk=5" \
        "trainer.resume_mode=auto"
)

echo "${submission_output}"
submission_id="$(echo "${submission_output}" | rg -o 'raysubmit_[A-Za-z0-9]+' | tail -n 1 || true)"
if [[ -z "${submission_id}" ]]; then
    echo "failed to parse submission id"
    exit 1
fi

echo "job submitted: ${submission_id}"
ray job logs --address="${RAY_DASHBOARD_ADDRESS}" "${submission_id}" --follow
