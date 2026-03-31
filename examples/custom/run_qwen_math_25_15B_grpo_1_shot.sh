#!/usr/bin/env bash
set -euo pipefail

if [ -f /root/.bashrc ]; then
    SWANLAB_API_KEY_FROM_BASHRC="$(sed -n "s/^export SWANLAB_API_KEY=['\"]\\(.*\\)['\"]$/\\1/p" /root/.bashrc | tail -n 1)"
    if [ -n "${SWANLAB_API_KEY_FROM_BASHRC}" ]; then
        export SWANLAB_API_KEY="${SWANLAB_API_KEY_FROM_BASHRC}"
    fi
    unset SWANLAB_API_KEY_FROM_BASHRC
fi

set -x

# 可选：如需 WandB / SwanLab，请在外部环境中显式导出对应 API Key。
ray stop
# 基础路径与实验信息（来自 YAML）
HOME_DIR=${HOME_DIR:-"/root/rl"}
project_name='verl_grpo_dsr_sub_baseline'
TRAIN_MODE=${TRAIN_MODE:-"oneshot"}
exp_name=${EXP_NAME:-"qwen_math_25_15B_dsr_grpo_${TRAIN_MODE}_entropy_$(date +%Y%m%d-%H%M%S)"}

# Ray 相关（按需修改）
# 说明：
# - Ray Core(GCS) 地址通常是 "<head_ip>:6379"，供 ray.init / ray status 使用
# - Ray Jobs/Dashboard 地址通常是 "http://<head_ip>:8265"，供 `ray job submit/logs` 使用
# 为了兼容旧脚本：
# - 若 RAY_ADDRESS 以 http(s) 开头，则视为 Dashboard 地址
# - 否则视为 GCS 地址
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
        if [ -z "${RAY_DASHBOARD_ADDRESS:-}" ] || [ "${RAY_DASHBOARD_ADDRESS}" = "http://127.0.0.1:8265" ] || [ "${RAY_DASHBOARD_ADDRESS}" = "http://localhost:8265" ]; then
            resolved_dashboard_host=${RAY_LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
            if [ -n "${resolved_dashboard_host}" ]; then
                RAY_DASHBOARD_ADDRESS="http://${resolved_dashboard_host}:8265"
            fi
        fi
    else
        echo "Error: cannot reach Ray cluster at ${RAY_GCS_ADDRESS}."
        echo "- If your head node is ${_ray_head_host}, start Ray there (ensure dashboard listens on 0.0.0.0:8265)."
        echo "- Then set: RAY_GCS_ADDRESS='${_ray_head_host}:6379' and RAY_DASHBOARD_ADDRESS='http://${_ray_head_host}:8265'"
        exit 1
    fi
else
    if { [ "${_ray_head_host}" = "127.0.0.1" ] || [ "${_ray_head_host}" = "localhost" ]; } && \
        { [ -z "${RAY_DASHBOARD_ADDRESS:-}" ] || [ "${RAY_DASHBOARD_ADDRESS}" = "http://127.0.0.1:8265" ] || [ "${RAY_DASHBOARD_ADDRESS}" = "http://localhost:8265" ]; }; then
        resolved_dashboard_host=${RAY_LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
        if [ -n "${resolved_dashboard_host}" ]; then
            RAY_DASHBOARD_ADDRESS="http://${resolved_dashboard_host}:8265"
        fi
    fi
    echo "Ray is running at ${RAY_GCS_ADDRESS} (dashboard: ${RAY_DASHBOARD_ADDRESS})."
fi

# 路径（来自 YAML 的 paths.*）
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}/verl"}
MODEL_PATH=${MODEL_PATH:-"/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct"}
DATA_BASE=${DATA_BASE:-"${HOME_DIR}/verl/data"}
ONE_SHOT_TRAIN_FILE=${ONE_SHOT_TRAIN_FILE:-"${DATA_BASE}/dsr_sub/pi1_one_ans.parquet"}
FULL_TRAIN_FILE=${FULL_TRAIN_FILE:-"${DATA_BASE}/dsr_sub/train.parquet"}
if [ -z "${TRAIN_FILE:-}" ]; then
    case "${TRAIN_MODE}" in
        oneshot|1shot|1-shot)
            TRAIN_FILE="${ONE_SHOT_TRAIN_FILE}"
            ;;
        full|all)
            TRAIN_FILE="${FULL_TRAIN_FILE}"
            ;;
        *)
            echo "Error: unsupported TRAIN_MODE='${TRAIN_MODE}'. Use oneshot or full."
            exit 1
            ;;
    esac
fi
VAL_PATH=${VAL_PATH:-"${DATA_BASE}/testset"}
echo "Using TRAIN_MODE=${TRAIN_MODE}, TRAIN_FILE=${TRAIN_FILE}"

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

RESUME_FROM=${RESUME_FROM:-""}
RESUME_FROM="" 
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from checkpoint: ${RESUME_FROM}"
    RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=${RESUME_FROM}"
else
    RESUME_ARGS="trainer.resume_mode=auto"
fi

# 数据与算法（来自 YAML）
# train_batch_size = 每次迭代处理的 Prompt 数量 * rollout_n
# 建议: 对于 1.5B 模型，设置 train_batch_size=64 (即 8个Prompt * 8个回复)
train_batch_size=${TRAIN_BATCH_SIZE:-64}
val_batch_size=${VAL_BATCH_SIZE:-""}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-true}
truncation=${TRUNCATION:-error}

adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-false}

# 模型/训练细节（来自 YAML）
lr=${LR:-1e-6}
# ppo_mini_batch_size: PPO 参数更新时的 batch 大小
# 必须能被 (N_GPUS * ppo_micro_batch_size_per_gpu) 整除
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}

# 确保 train_batch_size 是 ppo_mini_batch_size 的倍数
if [ $((train_batch_size % ppo_mini_batch_size)) -ne 0 ]; then
    train_batch_size=$(( (train_batch_size / ppo_mini_batch_size + 1) * ppo_mini_batch_size ))
    echo "Auto-adjusted train_batch_size to ${train_batch_size} to match ppo_mini_batch_size"
fi

# 1.5B 模型显存占用小，可以适当增大 micro_batch 以加速
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
use_kl_loss=${USE_KL_LOSS:-true}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
kl_loss_type=${KL_LOSS_TYPE:-low_var_kl}
use_torch_compile=${USE_TORCH_COMPILE:-false}
entropy_from_logits_with_chunking=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-false}
# 0 表示不使用熵正则化
entropy_coeff=${ENTROPY_COEFF:-0.001}
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-false}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false}

# rollout / ref（来自 YAML）
rollout_log_prob_micro_bsz=${ROLLOUT_LOG_PROB_MICRO_BSZ:-32}
ref_log_prob_micro_bsz=${REF_LOG_PROB_MICRO_BSZ:-64}
#((ppo_mini_batch_size × rollout_n)/ DP size) (mod ppo_micro_batch_size_per_gpu)==0

tp_size=${TP_SIZE:-1} #gpu数
rollout_name=${ROLLOUT_NAME:-vllm}
gpu_mem_util=${GPU_MEM_UTIL:-0.8}
# GRPO 算法核心参数：rollout_n 必须 > 1，通常设为 4-16
# 表示对每个 Prompt 采样多少个回复形成一个 Group
rollout_n=${ROLLOUT_N:-8}
val_rollout_n=${VAL_ROLLOUT_N:-8}
val_do_sample=${VAL_DO_SAMPLE:-true}
val_temperature=${VAL_TEMPERATURE:-1.0}
ref_param_offload=${REF_PARAM_OFFLOAD:-false}

# 训练器（来自 YAML）
# 1-shot 默认打开更完整的结果可视化与 token trace；如需减小产物体积可显式设为 false
enable_paper_style_viz=${ENABLE_PAPER_STYLE_VIZ:-true}
logger=${LOGGER:-""}
if [ -z "${logger}" ]; then
    if [ -n "${SWANLAB_API_KEY:-}" ]; then
        logger='["swanlab","file"]'
    else
        logger='["file"]'
    fi
fi
critic_warmup=0
log_val_generations=1
enable_val_diagnostics=${ENABLE_VAL_DIAGNOSTICS:-true}
rollout_calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS:-${enable_val_diagnostics}}
val_diag_tail_tokens=${VAL_DIAG_TAIL_TOKENS:-32}
val_diag_max_tokens_default=96
val_diag_dump_samples_default=8
val_diag_distribution_topk_default=5
if [ "${enable_paper_style_viz}" = "true" ]; then
    val_diag_max_tokens_default=192
    val_diag_dump_samples_default=-1
    val_diag_distribution_topk_default=64
fi
val_diag_max_tokens=${VAL_DIAG_MAX_TOKENS:-${val_diag_max_tokens_default}}
val_diag_dump_samples=${VAL_DIAG_DUMP_SAMPLES:-${val_diag_dump_samples_default}}
val_diag_max_samples=${VAL_DIAG_MAX_SAMPLES:-64}
val_diag_low_conf_prob_threshold=${VAL_DIAG_LOW_CONF_PROB_THRESHOLD:-0.2}
val_diag_track_eos_probability=${VAL_DIAG_TRACK_EOS_PROBABILITY:-true}
val_diag_eos_high_prob_threshold=${VAL_DIAG_EOS_HIGH_PROB_THRESHOLD:-0.1}
val_diag_distribution_topk=${VAL_DIAG_DISTRIBUTION_TOPK:-${val_diag_distribution_topk_default}}
trace_top_samples=${TRACE_TOP_SAMPLES:-6}
trace_uids_csv=${TRACE_UIDS:-""}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-6}
total_epochs=${TOTAL_EPOCHS:-10}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-""}
max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-""}

# 日志输出
LOG_DIR="${LOG_DIR:-"${HOME_DIR}/verl/logs"}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-"${LOG_DIR}/${project_name}-${exp_name}-$(date +'%Y%m%d-%H%M%S').log"}"
METRICS_DIR="${CKPTS_DIR}/metrics"
REPORT_ROOT="${REPORT_ROOT:-"${HOME_DIR}/verl/reports"}"
REPORT_DIR="${REPORT_DIR:-"${REPORT_ROOT}/${project_name}/${exp_name}"}"
TORCH_COMPILE_CACHE_ROOT="${TORCH_COMPILE_CACHE_ROOT:-"/tmp/verl_torch_compile_cache"}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-"${TORCH_COMPILE_CACHE_ROOT}/triton/${exp_name}"}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-"${TORCH_COMPILE_CACHE_ROOT}/inductor/${exp_name}"}"
TORCHINDUCTOR_FX_GRAPH_CACHE_VALUE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-0}"
TRITON_CACHE_MANAGER_VALUE="${TRITON_CACHE_MANAGER:-"verl.utils.triton_cache_manager:LenientFileCacheManager"}"
TORCHINDUCTOR_COMPILE_THREADS_VALUE="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
RESET_TORCH_COMPILE_CACHE="${RESET_TORCH_COMPILE_CACHE:-false}"
RUN_SCRIPT_DIR="${RUN_SCRIPT_DIR:-"${WORKING_DIR}/examples/custom/run_jobs"}"
SAVE_RUN_SCRIPT="${SAVE_RUN_SCRIPT:-true}"
RESOLVED_RUNTIME_ENV="${RUNTIME_ENV}"

mkdir -p "${RUN_SCRIPT_DIR}"

if [ "${SAVE_RUN_SCRIPT}" = "true" ]; then
    RUN_SCRIPT_PATH="${RUN_SCRIPT_DIR}/${exp_name}.sh"
    cat > "${RUN_SCRIPT_PATH}" <<EOF
#!/usr/bin/env bash
set -xeuo pipefail

cd ${WORKING_DIR@Q}
export SAVE_RUN_SCRIPT=false
export EXP_NAME=${exp_name@Q}
export HOME_DIR=${HOME_DIR@Q}
export TRAIN_MODE=${TRAIN_MODE@Q}
export RAY_GCS_ADDRESS=${RAY_GCS_ADDRESS@Q}
export RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS@Q}
export WORKING_DIR=${WORKING_DIR@Q}
export RUNTIME_ENV=${RUNTIME_ENV@Q}
export NNODES=${NNODES@Q}
export RAY_DATA_HOME=${RAY_DATA_HOME@Q}
export MODEL_PATH=${MODEL_PATH@Q}
export DATA_BASE=${DATA_BASE@Q}
export ONE_SHOT_TRAIN_FILE=${ONE_SHOT_TRAIN_FILE@Q}
export FULL_TRAIN_FILE=${FULL_TRAIN_FILE@Q}
export TRAIN_FILE=${TRAIN_FILE@Q}
export VAL_PATH=${VAL_PATH@Q}
export CKPTS_DIR=${CKPTS_DIR@Q}
export TRAIN_BATCH_SIZE=${train_batch_size@Q}
export VAL_BATCH_SIZE=${val_batch_size@Q}
export MAX_PROMPT_LENGTH=${max_prompt_length@Q}
export MAX_RESPONSE_LENGTH=${max_response_length@Q}
export FILTER_OVERLONG_PROMPTS=${filter_overlong_prompts@Q}
export TRUNCATION=${truncation@Q}
export ADV_ESTIMATOR=${adv_estimator@Q}
export USE_KL_IN_REWARD=${use_kl_in_reward@Q}
export LR=${lr@Q}
export PPO_MINI_BATCH_SIZE=${ppo_mini_batch_size@Q}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${ppo_micro_batch_size_per_gpu@Q}
export USE_KL_LOSS=${use_kl_loss@Q}
export KL_LOSS_COEF=${kl_loss_coef@Q}
export KL_LOSS_TYPE=${kl_loss_type@Q}
export USE_TORCH_COMPILE=${use_torch_compile@Q}
export ENTROPY_FROM_LOGITS_WITH_CHUNKING=${entropy_from_logits_with_chunking@Q}
export ENTROPY_COEFF=${entropy_coeff@Q}
export ACTOR_PARAM_OFFLOAD=${actor_param_offload@Q}
export ACTOR_OPTIMIZER_OFFLOAD=${actor_optimizer_offload@Q}
export ROLLOUT_LOG_PROB_MICRO_BSZ=${rollout_log_prob_micro_bsz@Q}
export REF_LOG_PROB_MICRO_BSZ=${ref_log_prob_micro_bsz@Q}
export TP_SIZE=${tp_size@Q}
export ROLLOUT_NAME=${rollout_name@Q}
export GPU_MEM_UTIL=${gpu_mem_util@Q}
export ROLLOUT_N=${rollout_n@Q}
export VAL_ROLLOUT_N=${val_rollout_n@Q}
export VAL_DO_SAMPLE=${val_do_sample@Q}
export VAL_TEMPERATURE=${val_temperature@Q}
export REF_PARAM_OFFLOAD=${ref_param_offload@Q}
export ENABLE_PAPER_STYLE_VIZ=${enable_paper_style_viz@Q}
export LOGGER=${logger@Q}
export ENABLE_VAL_DIAGNOSTICS=${enable_val_diagnostics@Q}
export ROLLOUT_CALCULATE_LOG_PROBS=${rollout_calculate_log_probs@Q}
export VAL_DIAG_TAIL_TOKENS=${val_diag_tail_tokens@Q}
export VAL_DIAG_MAX_TOKENS=${val_diag_max_tokens@Q}
export VAL_DIAG_DUMP_SAMPLES=${val_diag_dump_samples@Q}
export VAL_DIAG_MAX_SAMPLES=${val_diag_max_samples@Q}
export VAL_DIAG_LOW_CONF_PROB_THRESHOLD=${val_diag_low_conf_prob_threshold@Q}
export VAL_DIAG_TRACK_EOS_PROBABILITY=${val_diag_track_eos_probability@Q}
export VAL_DIAG_EOS_HIGH_PROB_THRESHOLD=${val_diag_eos_high_prob_threshold@Q}
export VAL_DIAG_DISTRIBUTION_TOPK=${val_diag_distribution_topk@Q}
export TRACE_TOP_SAMPLES=${trace_top_samples@Q}
export TRACE_UIDS=${trace_uids_csv@Q}
export SAVE_FREQ=${save_freq@Q}
export TEST_FREQ=${test_freq@Q}
export TOTAL_EPOCHS=${total_epochs@Q}
export MAX_ACTOR_CKPT_TO_KEEP=${max_actor_ckpt_to_keep@Q}
export MAX_CRITIC_CKPT_TO_KEEP=${max_critic_ckpt_to_keep@Q}
export LOG_DIR=${LOG_DIR@Q}
export LOG_FILE=${LOG_FILE@Q}
export REPORT_ROOT=${REPORT_ROOT@Q}
export REPORT_DIR=${REPORT_DIR@Q}
export TORCH_COMPILE_CACHE_ROOT=${TORCH_COMPILE_CACHE_ROOT@Q}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR@Q}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR@Q}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE_VALUE@Q}
export TRITON_CACHE_MANAGER=${TRITON_CACHE_MANAGER_VALUE@Q}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS_VALUE@Q}
export RESET_TORCH_COMPILE_CACHE=${RESET_TORCH_COMPILE_CACHE@Q}
export RUN_SCRIPT_DIR=${RUN_SCRIPT_DIR@Q}

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh
EOF
    chmod +x "${RUN_SCRIPT_PATH}"
    echo "Saved run script to ${RUN_SCRIPT_PATH}"
fi

# 提交 Ray 任务（使用 Hydra 覆盖键，等价于 YAML 中的配置）
# 注意：data.val_files 使用了双引号包裹的列表字符串
# 设置 PYTHONPATH 确保优先加载本地代码
export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${TRITON_CACHE_DIR}")" "$(dirname "${TORCHINDUCTOR_CACHE_DIR}")"
if [ "${RESET_TORCH_COMPILE_CACHE}" = "true" ]; then
    rm -rf "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
fi
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

_xtrace_was_enabled=0
case "$-" in
    *x*)
        _xtrace_was_enabled=1
        set +x
        ;;
esac

RESOLVED_RUNTIME_ENV="/tmp/verl_runtime_env_${exp_name}.yaml"
grep -vE '^[[:space:]]+(WANDB_API_KEY|SWANLAB_API_KEY|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR|TORCHINDUCTOR_FX_GRAPH_CACHE|TRITON_CACHE_MANAGER|TORCHINDUCTOR_COMPILE_THREADS):' "${RUNTIME_ENV}" > "${RESOLVED_RUNTIME_ENV}"
printf '  TRITON_CACHE_DIR: "%s"\n' "${TRITON_CACHE_DIR}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TORCHINDUCTOR_CACHE_DIR: "%s"\n' "${TORCHINDUCTOR_CACHE_DIR}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TORCHINDUCTOR_FX_GRAPH_CACHE: "%s"\n' "${TORCHINDUCTOR_FX_GRAPH_CACHE_VALUE}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TRITON_CACHE_MANAGER: "%s"\n' "${TRITON_CACHE_MANAGER_VALUE}" >> "${RESOLVED_RUNTIME_ENV}"
printf '  TORCHINDUCTOR_COMPILE_THREADS: "%s"\n' "${TORCHINDUCTOR_COMPILE_THREADS_VALUE}" >> "${RESOLVED_RUNTIME_ENV}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    printf '  WANDB_API_KEY: "%s"\n' "${WANDB_API_KEY}" >> "${RESOLVED_RUNTIME_ENV}"
fi
if [ -n "${SWANLAB_API_KEY:-}" ]; then
    printf '  SWANLAB_API_KEY: "%s"\n' "${SWANLAB_API_KEY}" >> "${RESOLVED_RUNTIME_ENV}"
fi

if [ "${_xtrace_was_enabled}" = "1" ]; then
    set -x
fi

checkpoint_keep_args=()
if [ -n "${max_actor_ckpt_to_keep}" ]; then
    checkpoint_keep_args+=("trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep}")
fi
if [ -n "${max_critic_ckpt_to_keep}" ]; then
    checkpoint_keep_args+=("trainer.max_critic_ckpt_to_keep=${max_critic_ckpt_to_keep}")
fi

data_override_args=()
if [ -n "${val_batch_size}" ]; then
    data_override_args+=("data.val_batch_size=${val_batch_size}")
fi

submission_output=$("$RAY_CMD" job submit --no-wait --address="${RAY_DASHBOARD_ADDRESS}" --runtime-env="${RESOLVED_RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- env VERL_FILE_LOGGER_ROOT="${METRICS_DIR}" python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    "${data_override_args[@]}" \
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
    actor_rollout_ref.actor.use_torch_compile=${use_torch_compile} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=${entropy_from_logits_with_chunking} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.calculate_log_probs=${rollout_calculate_log_probs} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.ref.use_torch_compile=${use_torch_compile} \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=${entropy_from_logits_with_chunking} \
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
    trainer.validation_data_dir="${CKPTS_DIR}/validation" \
    trainer.validation_diagnostics.enabled=${enable_val_diagnostics} \
    trainer.validation_diagnostics.compare_to_previous_eval=true \
    ++trainer.validation_diagnostics.max_samples=${val_diag_max_samples} \
    trainer.validation_diagnostics.samples_to_dump_token_details=${val_diag_dump_samples} \
    trainer.validation_diagnostics.max_tokens_per_sample=${val_diag_max_tokens} \
    trainer.validation_diagnostics.tail_tokens=${val_diag_tail_tokens} \
    trainer.validation_diagnostics.low_confidence_prob_threshold=${val_diag_low_conf_prob_threshold} \
    trainer.validation_diagnostics.track_eos_probability=${val_diag_track_eos_probability} \
    trainer.validation_diagnostics.eos_high_prob_threshold=${val_diag_eos_high_prob_threshold} \
    trainer.validation_diagnostics.token_distribution_topk=${val_diag_distribution_topk} \
    "${checkpoint_keep_args[@]}" \
    ${RESUME_ARGS})

# 提取提交 ID 并打印日志
submission_id_raw=$(echo "$submission_output" | grep -o "raysubmit_[^']*" | tail -n 1)
submission_id=$(echo "$submission_id_raw" | sed 's/\x1b\[[0-9;]*m//g')

if [ -n "$submission_id" ]; then
    echo "Job submitted with ID: ${submission_id}"
    echo "Streaming logs to ${LOG_FILE}"
    "$RAY_CMD" job logs --address="${RAY_DASHBOARD_ADDRESS}" "${submission_id}" --follow > "${LOG_FILE}" 2>&1 &
    echo "Logs are being written in the background. You can check the file: ${LOG_FILE}"
    if [ "${enable_paper_style_viz}" = "true" ]; then
        echo "1-shot report command:"
        echo "CKPTS_DIR=\"${CKPTS_DIR}\" REPORT_DIR=\"${REPORT_DIR}\" TRACE_TOP_SAMPLES=\"${trace_top_samples}\" TRACE_UIDS=\"${trace_uids_csv}\" bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh"
        echo "This will generate both the aggregate report and token_traces.html."
    fi
else
    echo "Failed to get submission ID."
    echo "Submission output:"
    echo "$submission_output"
fi
