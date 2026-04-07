#!/usr/bin/env bash
set -euo pipefail

# Serial launcher for 6 predefined experiments.
# It submits one run, waits for Ray job completion, then starts next run.

cd /root/rl/verl

if [ -f /home/vipuser/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/vipuser/miniconda3/etc/profile.d/conda.sh
    conda activate verl
fi

RAY_CMD=$(command -v ray || true)
if [ -z "${RAY_CMD}" ]; then
    RAY_CMD="/home/vipuser/miniconda3/envs/verl/bin/ray"
fi
if [ ! -x "${RAY_CMD}" ]; then
    echo "[FATAL] ray command not found."
    exit 1
fi

SCRIPT_PATH=${SCRIPT_PATH:-examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}
STOP_ON_FAIL=${STOP_ON_FAIL:-true}
# Coarse watchdog mode by default: wake every 30 minutes.
POLL_SECONDS=${POLL_SECONDS:-1800}
QUEUE_TAG=${QUEUE_TAG:-$(date +%Y%m%d-%H%M%S)}
START_FROM_ORDER=${START_FROM_ORDER:-1}
WAIT_FOR_JOB_ID=${WAIT_FOR_JOB_ID:-}
QUEUE_LOG_DIR=${QUEUE_LOG_DIR:-/root/rl/verl/logs/queue}
mkdir -p "${QUEUE_LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG_DIR}/launch_6_exps_${QUEUE_TAG}.log"
QUEUE_STATUS="${QUEUE_LOG_DIR}/launch_6_exps_${QUEUE_TAG}.tsv"

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B}
INSTRUCT_MODEL_PATH=${INSTRUCT_MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct}
DATA_CLEAN=${DATA_CLEAN:-/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet}
DATA_SHUFFLE=${DATA_SHUFFLE:-/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_shuffle.parquet}

# OOM-safe defaults for all experiments.
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
export ROLLOUT_LOG_PROB_MICRO_BSZ=${ROLLOUT_LOG_PROB_MICRO_BSZ:-8}
export REF_LOG_PROB_MICRO_BSZ=${REF_LOG_PROB_MICRO_BSZ:-16}
export ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true}
unset PYTORCH_CUDA_ALLOC_CONF || true

# Shared defaults.
export TRAIN_MODE=${TRAIN_MODE:-oneshot}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
export TEST_FREQ=${TEST_FREQ:-6}
# Save checkpoint at step 180 to avoid excessive disk usage.
export SAVE_FREQ=${SAVE_FREQ:-180}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export ENABLE_VAL_DIAGNOSTICS=${ENABLE_VAL_DIAGNOSTICS:-true}
export ENABLE_PAPER_STYLE_VIZ=${ENABLE_PAPER_STYLE_VIZ:-false}
export LOGGER=${LOGGER:-'["swanlab","file"]'}
export DATA_SEED=${DATA_SEED:-1}
export ROLLOUT_SEED=${ROLLOUT_SEED:-1}

for f in "${SCRIPT_PATH}" "${DATA_CLEAN}" "${DATA_SHUFFLE}"; do
    if [ ! -f "${f}" ]; then
        echo "[FATAL] file not found: ${f}"
        exit 1
    fi
done

if command -v md5sum >/dev/null 2>&1; then
    clean_md5=$(md5sum "${DATA_CLEAN}" | awk '{print $1}')
    shuffle_md5=$(md5sum "${DATA_SHUFFLE}" | awk '{print $1}')
    if [ "${clean_md5}" = "${shuffle_md5}" ]; then
        echo "[WARN] DATA_SHUFFLE is byte-identical to DATA_CLEAN. reward-shuffle may not be a real ablation." | tee -a "${QUEUE_LOG}"
    fi
fi

printf "order\tlabel\texp_name\tjob_id\tfinal_status\tstart_time\tend_time\tmodel_path\ttrain_file\trollout_n\tval_rollout_n\tuse_kl_loss\tkl_loss_coef\tentropy_coeff\n" > "${QUEUE_STATUS}"

wait_for_job() {
    local job_id="$1"
    while true; do
        local out
        out=$("${RAY_CMD}" job status --address="${RAY_DASHBOARD_ADDRESS}" "${job_id}" 2>&1 || true)
        local now
        now=$(date '+%F %T')
        echo "[${now}] [${job_id}] ${out}" >> "${QUEUE_LOG}"
        if echo "${out}" | rg -q "SUCCEEDED"; then
            echo "SUCCEEDED"
            return 0
        fi
        if echo "${out}" | rg -q "FAILED|STOPPED"; then
            echo "FAILED"
            return 1
        fi
        echo "[${now}] [${job_id}] still running; sleep ${POLL_SECONDS}s before next check." >> "${QUEUE_LOG}"
        sleep "${POLL_SECONDS}"
    done
}

run_one() {
    local order="$1"
    local label="$2"
    local model_path="$3"
    local rollout_n="$4"
    local val_rollout_n="$5"
    local use_kl_loss="$6"
    local kl_loss_coef="$7"
    local entropy_coeff="$8"
    local train_file="$9"

    local exp_name="${label}_${QUEUE_TAG}"
    local start_time
    local end_time
    local submission_output
    local job_id
    local final_status

    start_time=$(date '+%F %T')
    echo "" | tee -a "${QUEUE_LOG}"
    echo "========== [${order}/6] ${label} ==========" | tee -a "${QUEUE_LOG}"
    echo "exp_name=${exp_name}" | tee -a "${QUEUE_LOG}"

    if ! submission_output=$( \
        EXP_NAME="${exp_name}" \
        MODEL_PATH="${model_path}" \
        ONE_SHOT_TRAIN_FILE="${train_file}" \
        TRAIN_FILE="${train_file}" \
        ROLLOUT_N="${rollout_n}" \
        VAL_ROLLOUT_N="${val_rollout_n}" \
        USE_KL_LOSS="${use_kl_loss}" \
        KL_LOSS_COEF="${kl_loss_coef}" \
        ENTROPY_COEFF="${entropy_coeff}" \
        bash "${SCRIPT_PATH}" 2>&1 \
    ); then
        end_time=$(date '+%F %T')
        echo "${submission_output}" | tee -a "${QUEUE_LOG}"
        final_status="SUBMIT_FAILED"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${order}" "${label}" "${exp_name}" "-" "${final_status}" "${start_time}" "${end_time}" \
            "${model_path}" "${train_file}" "${rollout_n}" "${val_rollout_n}" "${use_kl_loss}" "${kl_loss_coef}" "${entropy_coeff}" \
            >> "${QUEUE_STATUS}"
        echo "[ERROR] submit failed: ${label}" | tee -a "${QUEUE_LOG}"
        if [ "${STOP_ON_FAIL}" = "true" ]; then
            return 1
        fi
        return 0
    fi

    echo "${submission_output}" | tee -a "${QUEUE_LOG}"
    job_id=$(echo "${submission_output}" | rg -o "raysubmit_[A-Za-z0-9]+" | tail -n 1 || true)
    if [ -z "${job_id}" ]; then
        end_time=$(date '+%F %T')
        final_status="NO_JOB_ID"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${order}" "${label}" "${exp_name}" "-" "${final_status}" "${start_time}" "${end_time}" \
            "${model_path}" "${train_file}" "${rollout_n}" "${val_rollout_n}" "${use_kl_loss}" "${kl_loss_coef}" "${entropy_coeff}" \
            >> "${QUEUE_STATUS}"
        echo "[ERROR] no job id parsed: ${label}" | tee -a "${QUEUE_LOG}"
        if [ "${STOP_ON_FAIL}" = "true" ]; then
            return 1
        fi
        return 0
    fi

    echo "[INFO] waiting for ${job_id} ..." | tee -a "${QUEUE_LOG}"
    if final_status=$(wait_for_job "${job_id}"); then
        true
    else
        true
    fi

    end_time=$(date '+%F %T')
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${order}" "${label}" "${exp_name}" "${job_id}" "${final_status}" "${start_time}" "${end_time}" \
        "${model_path}" "${train_file}" "${rollout_n}" "${val_rollout_n}" "${use_kl_loss}" "${kl_loss_coef}" "${entropy_coeff}" \
        >> "${QUEUE_STATUS}"
    echo "[INFO] ${label} done with status=${final_status}" | tee -a "${QUEUE_LOG}"

    if [ "${final_status}" != "SUCCEEDED" ] && [ "${STOP_ON_FAIL}" = "true" ]; then
        return 1
    fi
    return 0
}

echo "[INFO] queue tag: ${QUEUE_TAG}" | tee -a "${QUEUE_LOG}"
echo "[INFO] queue log: ${QUEUE_LOG}" | tee -a "${QUEUE_LOG}"
echo "[INFO] queue status: ${QUEUE_STATUS}" | tee -a "${QUEUE_LOG}"
echo "[INFO] start from order: ${START_FROM_ORDER}" | tee -a "${QUEUE_LOG}"
if [ -n "${WAIT_FOR_JOB_ID}" ]; then
    echo "[INFO] wait for existing job first: ${WAIT_FOR_JOB_ID}" | tee -a "${QUEUE_LOG}"
    existing_status=$(wait_for_job "${WAIT_FOR_JOB_ID}" || true)
    echo "[INFO] existing job ${WAIT_FOR_JOB_ID} finished with status=${existing_status}" | tee -a "${QUEUE_LOG}"
    if [ "${existing_status}" != "SUCCEEDED" ] && [ "${STOP_ON_FAIL}" = "true" ]; then
        echo "[ERROR] existing job did not succeed; stop queue." | tee -a "${QUEUE_LOG}"
        exit 1
    fi
fi

[ "${START_FROM_ORDER}" -le 1 ] && run_one 1 "base_c_n8_baseline"         "${BASE_MODEL_PATH}"     8 8 true  0.001 0.001 "${DATA_CLEAN}"
[ "${START_FROM_ORDER}" -le 2 ] && run_one 2 "base_c_n1_group_ablation"   "${BASE_MODEL_PATH}"     1 1 true  0.001 0.001 "${DATA_CLEAN}"
[ "${START_FROM_ORDER}" -le 3 ] && run_one 3 "base_d_weak_constraint"     "${BASE_MODEL_PATH}"     8 8 false 0     0     "${DATA_CLEAN}"
[ "${START_FROM_ORDER}" -le 4 ] && run_one 4 "base_b_reward_shuffle"      "${BASE_MODEL_PATH}"     8 8 true  0.001 0.001 "${DATA_SHUFFLE}"
[ "${START_FROM_ORDER}" -le 5 ] && run_one 5 "instruct_b_reward_shuffle"  "${INSTRUCT_MODEL_PATH}" 8 8 true  0.001 0.001 "${DATA_SHUFFLE}"
[ "${START_FROM_ORDER}" -le 6 ] && run_one 6 "instruct_c_n8_baseline"     "${INSTRUCT_MODEL_PATH}" 8 8 true  0.001 0.001 "${DATA_CLEAN}"

echo "[INFO] all queued experiments finished." | tee -a "${QUEUE_LOG}"
echo "[INFO] status table: ${QUEUE_STATUS}" | tee -a "${QUEUE_LOG}"
