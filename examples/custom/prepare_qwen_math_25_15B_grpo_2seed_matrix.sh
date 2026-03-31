#!/usr/bin/env bash
set -euo pipefail

# Prepare or submit the core matrix:
# - models: instruct, base
# - modes: oneshot, full
# - seeds: 2 runs by default
#
# Default behavior is safe: print commands and write a manifest.
# To actually submit jobs, set SUBMIT=true.

HOME_DIR=${HOME_DIR:-/root/rl}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dsr_sub_baseline}
BATCH_TAG=${BATCH_TAG:-$(date +%Y%m%d-%H%M%S)}
SUBMIT=${SUBMIT:-false}
SEEDS=${SEEDS:-"1 2"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TEST_FREQ=${TEST_FREQ:-6}
SAVE_FREQ=${SAVE_FREQ:-100}
ENABLE_PAPER_STYLE_VIZ=${ENABLE_PAPER_STYLE_VIZ:-true}
ENABLE_VAL_DIAGNOSTICS=${ENABLE_VAL_DIAGNOSTICS:-true}
SCRIPT_PATH=${SCRIPT_PATH:-examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh}
MANIFEST_DIR=${MANIFEST_DIR:-${HOME_DIR}/verl/ckpts/matrix_manifests}

INSTRUCT_MODEL_PATH=${INSTRUCT_MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B}

mkdir -p "${MANIFEST_DIR}"
MANIFEST_FILE="${MANIFEST_DIR}/qwen_math_25_15B_2seed_${BATCH_TAG}.tsv"

printf "label\texp_name\tmodel_tag\ttrain_mode\tseed\tmodel_path\tckpts_dir\tcommand\n" > "${MANIFEST_FILE}"

run_one() {
    local model_tag="$1"
    local model_path="$2"
    local train_mode="$3"
    local seed="$4"

    local exp_name="qwen_math_25_15B_${model_tag}_${train_mode}_s${seed}_${BATCH_TAG}"
    local label="${model_tag}_${train_mode}_s${seed}"
    local ckpts_dir="${HOME_DIR}/verl/ckpts/${PROJECT_NAME}/${exp_name}"
    local cmd
    cmd="DATA_SEED=${seed} ROLLOUT_SEED=${seed} EXP_NAME=${exp_name} MODEL_PATH=${model_path} TOTAL_EPOCHS=${TOTAL_EPOCHS} TEST_FREQ=${TEST_FREQ} SAVE_FREQ=${SAVE_FREQ} ENABLE_PAPER_STYLE_VIZ=${ENABLE_PAPER_STYLE_VIZ} ENABLE_VAL_DIAGNOSTICS=${ENABLE_VAL_DIAGNOSTICS} TRAIN_MODE=${train_mode} bash ${SCRIPT_PATH}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${label}" "${exp_name}" "${model_tag}" "${train_mode}" "${seed}" "${model_path}" "${ckpts_dir}" "${cmd}" \
        >> "${MANIFEST_FILE}"

    if [ "${SUBMIT}" = "true" ]; then
        echo "Submitting ${label}"
        env \
            DATA_SEED="${seed}" \
            ROLLOUT_SEED="${seed}" \
            EXP_NAME="${exp_name}" \
            MODEL_PATH="${model_path}" \
            TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
            TEST_FREQ="${TEST_FREQ}" \
            SAVE_FREQ="${SAVE_FREQ}" \
            ENABLE_PAPER_STYLE_VIZ="${ENABLE_PAPER_STYLE_VIZ}" \
            ENABLE_VAL_DIAGNOSTICS="${ENABLE_VAL_DIAGNOSTICS}" \
            TRAIN_MODE="${train_mode}" \
            bash "${SCRIPT_PATH}"
    else
        echo "${cmd}"
    fi
}

for seed in ${SEEDS}; do
    run_one "instruct" "${INSTRUCT_MODEL_PATH}" "oneshot" "${seed}"
    run_one "instruct" "${INSTRUCT_MODEL_PATH}" "full" "${seed}"
    run_one "base" "${BASE_MODEL_PATH}" "oneshot" "${seed}"
    run_one "base" "${BASE_MODEL_PATH}" "full" "${seed}"
done

echo
echo "Manifest written to: ${MANIFEST_FILE}"
echo "Batch tag: ${BATCH_TAG}"
echo "Submit mode: ${SUBMIT}"
