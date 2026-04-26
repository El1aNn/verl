#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl

source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
CUDA_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
unset PYTORCH_CUDA_ALLOC_CONF

export MANAGE_RAY_CLUSTER='false'
export SAVE_RUN_SCRIPT='false'
export EXP_NAME="${EXP_NAME:-base_align_exp1_noval_save60_allgpu_$(date +%Y%m%d_%H%M%S)}"
export HOME_DIR='/root/rl'
export TRAIN_MODE='oneshot'
export RAY_GCS_ADDRESS="${HOST_IP}:6379"
export RAY_DASHBOARD_ADDRESS="http://${HOST_IP}:8265"
export RAY_LOCAL_IP="${HOST_IP}"
export WORKING_DIR='/root/rl/verl'
export RUNTIME_ENV='/root/rl/verl/examples/custom/run_jobs/runtime_env_base_c_n1_group_ablation_rerun180_20260405_212235.yaml'
export NNODES='1'
export N_GPUS_OVERRIDE="${GPU_COUNT}"
export RAY_DATA_HOME='/root/rl/verl'
export MODEL_PATH='/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B'
export DATA_BASE='/root/rl/verl/data'
export ONE_SHOT_TRAIN_FILE='/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet'
export FULL_TRAIN_FILE='/root/rl/verl/data/dsr_sub/train.parquet'
export TRAIN_FILE='/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet'
export VAL_PATH='/root/rl/verl/data/testset'
export CKPTS_DIR="/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/${EXP_NAME}"
export TRAIN_BATCH_SIZE='64'
export VAL_BATCH_SIZE=''
export MAX_PROMPT_LENGTH='1024'
export MAX_RESPONSE_LENGTH='2048'
export FILTER_OVERLONG_PROMPTS='true'
export TRUNCATION='error'
export ADV_ESTIMATOR='grpo'
export USE_KL_IN_REWARD='false'
export LR='1e-6'
export PPO_MINI_BATCH_SIZE='32'
export PPO_MICRO_BATCH_SIZE_PER_GPU='2'
export USE_KL_LOSS='true'
export KL_LOSS_COEF='0.001'
export KL_LOSS_TYPE='low_var_kl'
export USE_TORCH_COMPILE='false'
export ENTROPY_FROM_LOGITS_WITH_CHUNKING='true'
export ENTROPY_COEFF='0.001'
export ACTOR_PARAM_OFFLOAD='false'
export ACTOR_OPTIMIZER_OFFLOAD='false'
export ROLLOUT_LOG_PROB_MICRO_BSZ='8'
export REF_LOG_PROB_MICRO_BSZ='16'
export TP_SIZE='1'
export ROLLOUT_NAME='vllm'
export GPU_MEM_UTIL='0.8'
export ROLLOUT_N='8'
export VAL_ROLLOUT_N='8'
export VAL_DO_SAMPLE='true'
export VAL_TEMPERATURE='1.0'
export REF_PARAM_OFFLOAD='false'
export ENABLE_PAPER_STYLE_VIZ='false'
export LOGGER='["swanlab","file"]'
export ENABLE_VAL_DIAGNOSTICS='false'
export VAL_BEFORE_TRAIN='false'
export ROLLOUT_CALCULATE_LOG_PROBS='false'
export VAL_DIAG_TAIL_TOKENS='32'
export VAL_DIAG_MAX_TOKENS='32'
export VAL_DIAG_DUMP_SAMPLES='0'
export VAL_DIAG_MAX_SAMPLES='0'
export VAL_DIAG_LOW_CONF_PROB_THRESHOLD='0.2'
export VAL_DIAG_TRACK_EOS_PROBABILITY='false'
export VAL_DIAG_EOS_HIGH_PROB_THRESHOLD='0.1'
export VAL_DIAG_DISTRIBUTION_TOPK='0'
export TRACE_TOP_SAMPLES='0'
export TRACE_UIDS=''
export SAVE_FREQ='60'
export TEST_FREQ='-1'
export TOTAL_EPOCHS='10'
export LOG_DIR='/root/rl/verl/logs'
export LOG_FILE="/root/rl/verl/logs/verl_grpo_dsr_sub_baseline-${EXP_NAME}.log"
export REPORT_ROOT='/root/rl/verl/reports'
export REPORT_DIR="/root/rl/verl/reports/verl_grpo_dsr_sub_baseline/${EXP_NAME}"
export TORCH_COMPILE_CACHE_ROOT='/tmp/verl_torch_compile_cache'
export TRITON_CACHE_DIR="/tmp/verl_torch_compile_cache/triton/${EXP_NAME}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/verl_torch_compile_cache/inductor/${EXP_NAME}"
export TORCHINDUCTOR_FX_GRAPH_CACHE='0'
export TRITON_CACHE_MANAGER='verl.utils.triton_cache_manager:LenientFileCacheManager'
export TORCHINDUCTOR_COMPILE_THREADS='1'
export RESET_TORCH_COMPILE_CACHE='false'
export RUN_SCRIPT_DIR='/root/rl/verl/examples/custom/run_jobs'

echo "submit_only exp=${EXP_NAME}"
echo "host_ip=${HOST_IP}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "ckpts_dir=${CKPTS_DIR}"
echo "log_file=${LOG_FILE}"

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh
