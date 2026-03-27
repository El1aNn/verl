#!/usr/bin/env bash
set -xeuo pipefail

# 可选：如需 WandB，请在此设置（或依赖外部已导出）
export WANDB_API_KEY="f408d6f1e1f982b94e6034176c0cd1f72cf9ab62"
export SWANLAB_API_KEY="B2gwMFDhC9KMZAu6T8UXL"  # 添加这一行
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
    else
        echo "Error: cannot reach Ray cluster at ${RAY_GCS_ADDRESS}."
        echo "- If your head node is ${_ray_head_host}, start Ray there (ensure dashboard listens on 0.0.0.0:8265)."
        echo "- Then set: RAY_GCS_ADDRESS='${_ray_head_host}:6379' and RAY_DASHBOARD_ADDRESS='http://${_ray_head_host}:8265'"
        exit 1
    fi
else
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
train_batch_size=64
max_prompt_length=1024
max_response_length=2048
filter_overlong_prompts=true
truncation='error'

adv_estimator=grpo
use_kl_in_reward=false

# 模型/训练细节（来自 YAML）
lr=1e-6
# ppo_mini_batch_size: PPO 参数更新时的 batch 大小
# 必须能被 (N_GPUS * ppo_micro_batch_size_per_gpu) 整除
ppo_mini_batch_size=32

# 确保 train_batch_size 是 ppo_mini_batch_size 的倍数
if [ $((train_batch_size % ppo_mini_batch_size)) -ne 0 ]; then
    train_batch_size=$(( (train_batch_size / ppo_mini_batch_size + 1) * ppo_mini_batch_size ))
    echo "Auto-adjusted train_batch_size to ${train_batch_size} to match ppo_mini_batch_size"
fi

# 1.5B 模型显存占用小，可以适当增大 micro_batch 以加速
ppo_micro_batch_size_per_gpu=4
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
# 0 表示不使用熵正则化
entropy_coeff=0.001
actor_param_offload=false
actor_optimizer_offload=false

# rollout / ref（来自 YAML）
rollout_log_prob_micro_bsz=32
ref_log_prob_micro_bsz=64
#((ppo_mini_batch_size × rollout_n)/ DP size) (mod ppo_micro_batch_size_per_gpu)==0

tp_size=1 #gpu数
rollout_name=vllm
gpu_mem_util=0.8
# GRPO 算法核心参数：rollout_n 必须 > 1，通常设为 4-16
# 表示对每个 Prompt 采样多少个回复形成一个 Group
rollout_n=8
val_rollout_n=8
val_do_sample=true
val_temperature=1.0
ref_param_offload=false

# 训练器（来自 YAML）
# 1-shot 默认打开更完整的结果可视化与 token trace；如需减小产物体积可显式设为 false
enable_paper_style_viz=${ENABLE_PAPER_STYLE_VIZ:-true}
logger=${LOGGER:-swanlab}
if [ "${enable_paper_style_viz}" = "true" ] && [ -z "${LOGGER:-}" ]; then
    logger='["swanlab","file"]'
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
val_diag_low_conf_prob_threshold=${VAL_DIAG_LOW_CONF_PROB_THRESHOLD:-0.2}
val_diag_track_eos_probability=${VAL_DIAG_TRACK_EOS_PROBABILITY:-true}
val_diag_eos_high_prob_threshold=${VAL_DIAG_EOS_HIGH_PROB_THRESHOLD:-0.1}
val_diag_distribution_topk=${VAL_DIAG_DISTRIBUTION_TOPK:-${val_diag_distribution_topk_default}}
trace_top_samples=${TRACE_TOP_SAMPLES:-6}
trace_uids_csv=${TRACE_UIDS:-""}
save_freq=100
test_freq=6
total_epochs=10

# 日志输出
LOG_DIR="${HOME_DIR}/verl/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${project_name}-${exp_name}-$(date +'%Y%m%d-%H%M%S').log"
METRICS_DIR="${CKPTS_DIR}/metrics"
REPORT_DIR="${CKPTS_DIR}/paper_viz"

# 提交 Ray 任务（使用 Hydra 覆盖键，等价于 YAML 中的配置）
# 注意：data.val_files 使用了双引号包裹的列表字符串
# 设置 PYTHONPATH 确保优先加载本地代码
export PYTHONPATH="${WORKING_DIR}:${PYTHONPATH:-}"

submission_output=$("$RAY_CMD" job submit --no-wait --address="${RAY_DASHBOARD_ADDRESS}" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- env VERL_FILE_LOGGER_ROOT="${METRICS_DIR}" python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.calculate_log_probs=${rollout_calculate_log_probs} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
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
    trainer.validation_diagnostics.samples_to_dump_token_details=${val_diag_dump_samples} \
    trainer.validation_diagnostics.max_tokens_per_sample=${val_diag_max_tokens} \
    trainer.validation_diagnostics.tail_tokens=${val_diag_tail_tokens} \
    trainer.validation_diagnostics.low_confidence_prob_threshold=${val_diag_low_conf_prob_threshold} \
    trainer.validation_diagnostics.track_eos_probability=${val_diag_track_eos_probability} \
    trainer.validation_diagnostics.eos_high_prob_threshold=${val_diag_eos_high_prob_threshold} \
    trainer.validation_diagnostics.token_distribution_topk=${val_diag_distribution_topk} \
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
        echo "CKPTS_DIR=\"${CKPTS_DIR}\" TRACE_TOP_SAMPLES=\"${trace_top_samples}\" TRACE_UIDS=\"${trace_uids_csv}\" bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh"
        echo "This will generate both the aggregate report and token_traces.html."
    fi
else
    echo "Failed to get submission ID."
    echo "Submission output:"
    echo "$submission_output"
fi
