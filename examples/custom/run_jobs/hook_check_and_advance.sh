#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl

if [ -f /home/vipuser/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/vipuser/miniconda3/etc/profile.d/conda.sh
    conda activate verl || true
fi

RAY_CMD=${RAY_CMD:-/home/vipuser/miniconda3/envs/verl/bin/ray}
SCRIPT_PATH=${SCRIPT_PATH:-examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://192.168.122.6:8265}
STATE_FILE=${STATE_FILE:-/root/rl/verl/logs/queue/hook_state.env}
LOG_FILE=${LOG_FILE:-/root/rl/verl/logs/queue/hook_check_and_advance.log}
PLAN_MD=${PLAN_MD:-/root/rl/verl/logs/queue/hook_plan.md}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B}
INSTRUCT_MODEL_PATH=${INSTRUCT_MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct}
DATA_CLEAN=${DATA_CLEAN:-/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet}
DATA_SHUFFLE=${DATA_SHUFFLE:-/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_shuffle.parquet}
INITIAL_JOB_ID=${INITIAL_JOB_ID:-raysubmit_1khsbFk6tSWmQvbe}

mkdir -p "$(dirname "$STATE_FILE")"

ensure_plan_md() {
    if [ -f "$PLAN_MD" ]; then
        return
    fi
    mkdir -p "$(dirname "$PLAN_MD")"
    cat > "$PLAN_MD" <<'EOF'
# Hook Plan

## Config
<!-- HOOK_CONFIG_BEGIN -->
RAY_DASHBOARD_ADDRESS=http://192.168.122.6:8265
INITIAL_JOB_ID=raysubmit_1khsbFk6tSWmQvbe
SCRIPT_PATH=examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh

EXP2_LABEL=base_c_n1_group_ablation
EXP2_MODEL=BASE
EXP2_ROLLOUT_N=1
EXP2_VAL_ROLLOUT_N=1
EXP2_USE_KL_LOSS=true
EXP2_KL_LOSS_COEF=0.001
EXP2_ENTROPY_COEFF=0.001
EXP2_TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet

EXP3_LABEL=base_d_weak_constraint
EXP3_MODEL=BASE
EXP3_ROLLOUT_N=8
EXP3_VAL_ROLLOUT_N=8
EXP3_USE_KL_LOSS=false
EXP3_KL_LOSS_COEF=0
EXP3_ENTROPY_COEFF=0
EXP3_TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet

EXP4_LABEL=base_b_reward_shuffle
EXP4_MODEL=BASE
EXP4_ROLLOUT_N=8
EXP4_VAL_ROLLOUT_N=8
EXP4_USE_KL_LOSS=true
EXP4_KL_LOSS_COEF=0.001
EXP4_ENTROPY_COEFF=0.001
EXP4_TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_shuffle.parquet

EXP5_LABEL=instruct_b_reward_shuffle
EXP5_MODEL=INSTRUCT
EXP5_ROLLOUT_N=8
EXP5_VAL_ROLLOUT_N=8
EXP5_USE_KL_LOSS=true
EXP5_KL_LOSS_COEF=0.001
EXP5_ENTROPY_COEFF=0.001
EXP5_TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_shuffle.parquet

EXP6_LABEL=instruct_c_n8_baseline
EXP6_MODEL=INSTRUCT
EXP6_ROLLOUT_N=8
EXP6_VAL_ROLLOUT_N=8
EXP6_USE_KL_LOSS=true
EXP6_KL_LOSS_COEF=0.001
EXP6_ENTROPY_COEFF=0.001
EXP6_TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet
<!-- HOOK_CONFIG_END -->

## Execution Log
EOF
}

append_md_log() {
    local now msg
    now=$(date '+%F %T')
    msg="$*"
    echo "- [${now}] ${msg}" >> "$PLAN_MD"
}

load_plan_config() {
    local lines
    lines=$(awk '/HOOK_CONFIG_BEGIN/{f=1;next}/HOOK_CONFIG_END/{f=0}f' "$PLAN_MD" || true)
    if [ -z "$lines" ]; then
        return
    fi
    while IFS= read -r line; do
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        if [ -z "$line" ]; then
            continue
        fi
        if [[ "$line" =~ ^# ]]; then
            continue
        fi
        if [[ "$line" =~ ^[A-Z0-9_]+= ]]; then
            eval "export ${line}"
        fi
    done <<< "$lines"
}

log() {
    local now
    now=$(date '+%F %T')
    echo "[$now] $*" | tee -a "$LOG_FILE"
    append_md_log "$*"
}

save_state() {
    cat > "${STATE_FILE}.tmp" <<EOF
QUEUE_TAG='${QUEUE_TAG}'
CURRENT_ORDER='${CURRENT_ORDER}'
CURRENT_LABEL='${CURRENT_LABEL}'
CURRENT_JOB_ID='${CURRENT_JOB_ID}'
HALTED='${HALTED}'
DONE='${DONE}'
LAST_CHECK_TS='${LAST_CHECK_TS}'
EOF
    mv "${STATE_FILE}.tmp" "${STATE_FILE}"
}

init_state() {
    QUEUE_TAG="manual_$(date +%Y%m%d-%H%M%S)"
    CURRENT_ORDER=1
    CURRENT_LABEL='base_c_n8_baseline'
    CURRENT_JOB_ID="${INITIAL_JOB_ID}"
    HALTED=0
    DONE=0
    LAST_CHECK_TS=0
    save_state
    log "initialized state: order=${CURRENT_ORDER}, job=${CURRENT_JOB_ID}, tag=${QUEUE_TAG}"
}

ensure_plan_md
load_plan_config

if [ ! -f "$STATE_FILE" ]; then
    init_state
else
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi

if [ "${HALTED}" = "1" ]; then
    log "HALTED=1, skip."
    exit 0
fi

if [ "${DONE}" = "1" ]; then
    log "DONE=1, skip."
    exit 0
fi

status_out=$("$RAY_CMD" job status --address="${RAY_DASHBOARD_ADDRESS}" "${CURRENT_JOB_ID}" 2>&1 || true)
LAST_CHECK_TS=$(date +%s)
save_state
log "checked order=${CURRENT_ORDER}, job=${CURRENT_JOB_ID}"
echo "${status_out}" >> "$LOG_FILE"

if echo "${status_out}" | rg -q "Status for job '.*': RUNNING"; then
    log "job still RUNNING"
    exit 0
fi

if echo "${status_out}" | rg -q "Status for job '.*': SUCCEEDED"; then
    log "job SUCCEEDED"
elif echo "${status_out}" | rg -q "Status for job '.*': FAILED|Status for job '.*': STOPPED"; then
    HALTED=1
    save_state
    log "job FAILED/STOPPED; set HALTED=1"
    exit 0
else
    log "unknown status text; keep waiting"
    exit 0
fi

next_order=$((CURRENT_ORDER + 1))
if [ "${next_order}" -gt 6 ]; then
    DONE=1
    save_state
    log "all experiments finished; set DONE=1"
    exit 0
fi

eval "label=\${EXP${next_order}_LABEL:-}"
eval "model_selector=\${EXP${next_order}_MODEL:-}"
eval "rollout_n=\${EXP${next_order}_ROLLOUT_N:-}"
eval "val_rollout_n=\${EXP${next_order}_VAL_ROLLOUT_N:-}"
eval "use_kl_loss=\${EXP${next_order}_USE_KL_LOSS:-}"
eval "kl_loss_coef=\${EXP${next_order}_KL_LOSS_COEF:-}"
eval "entropy_coeff=\${EXP${next_order}_ENTROPY_COEFF:-}"
eval "train_file=\${EXP${next_order}_TRAIN_FILE:-}"

if [ -z "${label}" ] || [ -z "${model_selector}" ] || [ -z "${rollout_n}" ] || [ -z "${val_rollout_n}" ] || [ -z "${use_kl_loss}" ] || [ -z "${kl_loss_coef}" ] || [ -z "${entropy_coeff}" ] || [ -z "${train_file}" ]; then
    HALTED=1
    save_state
    log "missing EXP${next_order}_* config in plan md; set HALTED=1"
    exit 0
fi

case "${model_selector}" in
    BASE)
        model_path="${BASE_MODEL_PATH}"
        ;;
    INSTRUCT)
        model_path="${INSTRUCT_MODEL_PATH}"
        ;;
    /*)
        model_path="${model_selector}"
        ;;
    *)
        HALTED=1
        save_state
        log "invalid model selector '${model_selector}' for order=${next_order}; set HALTED=1"
        exit 0
        ;;
esac

exp_name="${label}_${QUEUE_TAG}"
log "submitting next order=${next_order}, exp_name=${exp_name}"

submission_output=$(
    EXP_NAME="${exp_name}" \
    MODEL_PATH="${model_path}" \
    ONE_SHOT_TRAIN_FILE="${train_file}" \
    TRAIN_FILE="${train_file}" \
    ROLLOUT_N="${rollout_n}" \
    VAL_ROLLOUT_N="${val_rollout_n}" \
    USE_KL_LOSS="${use_kl_loss}" \
    KL_LOSS_COEF="${kl_loss_coef}" \
    ENTROPY_COEFF="${entropy_coeff}" \
    bash "${SCRIPT_PATH}" 2>&1 || true
)
echo "${submission_output}" >> "$LOG_FILE"

new_job_id=$(echo "${submission_output}" | rg -o "raysubmit_[A-Za-z0-9]+" | tail -n 1 || true)
if [ -z "${new_job_id}" ]; then
    HALTED=1
    save_state
    log "submit failed/no job id parsed; set HALTED=1"
    exit 0
fi

CURRENT_ORDER="${next_order}"
CURRENT_LABEL="${label}"
CURRENT_JOB_ID="${new_job_id}"
save_state
log "submitted ok: order=${CURRENT_ORDER}, job=${CURRENT_JOB_ID}"
