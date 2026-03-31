#!/usr/bin/env bash
set -euo pipefail

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
PYTHON_BIN=${PYTHON_BIN:-python3}
HOME_DIR=${HOME_DIR:-/root/rl}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dsr_sub_baseline}
MANIFEST_GLOB_DEFAULT="${HOME_DIR}/verl/ckpts/matrix_manifests/qwen_math_25_15B_2seed_*.tsv"
TRACE_TOP_SAMPLES=${TRACE_TOP_SAMPLES:-6}
RUN_REPORT=${RUN_REPORT:-true}
STRICT=${STRICT:-false}
SELECTED_STEPS=${SELECTED_STEPS:-""}
TRACE_UIDS=${TRACE_UIDS:-""}
TOKEN_GROUP_SPECS=${TOKEN_GROUP_SPECS:-""}

if [ $# -ge 1 ] && [ -n "${1:-}" ]; then
    MANIFEST_FILE="$1"
else
    latest_manifest=$(ls -1t ${MANIFEST_GLOB_DEFAULT} 2>/dev/null | head -n 1 || true)
    if [ -z "${latest_manifest}" ]; then
        echo "No manifest found. Pass one explicitly or run:"
        echo "  bash examples/custom/prepare_qwen_math_25_15B_grpo_2seed_matrix.sh"
        exit 1
    fi
    MANIFEST_FILE="${latest_manifest}"
fi

if [ ! -f "${MANIFEST_FILE}" ]; then
    echo "Manifest not found: ${MANIFEST_FILE}"
    exit 1
fi

manifest_basename=$(basename "${MANIFEST_FILE}")
batch_stem="${manifest_basename%.tsv}"
batch_tag="${batch_stem##*_}"
REPORT_ROOT=${REPORT_ROOT:-"${HOME_DIR}/verl/reports"}
OUTPUT_DIR_DEFAULT="${REPORT_ROOT}/compare/${batch_stem}"
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_DIR_DEFAULT}"}
COMMAND_FILE="${OUTPUT_DIR}/compare_report_command.sh"

mkdir -p "${OUTPUT_DIR}"

cmd=(
    "${PYTHON_BIN}" "scripts/validation_viz_report.py"
    "--output-dir" "${OUTPUT_DIR}"
    "--trace-top-samples" "${TRACE_TOP_SAMPLES}"
)

run_count=0
skip_count=0

while IFS=$'\t' read -r label exp_name model_tag train_mode seed model_path ckpts_dir command_text; do
    if [ "${label}" = "label" ]; then
        continue
    fi

    validation_dir="${ckpts_dir}/validation"
    metrics_file="${ckpts_dir}/metrics/${PROJECT_NAME}/${exp_name}.jsonl"

    has_validation=false
    if find "${validation_dir}" -maxdepth 1 -type f -name "*.jsonl" -print -quit >/dev/null 2>&1; then
        if [ -n "$(find "${validation_dir}" -maxdepth 1 -type f -name "*.jsonl" -print -quit 2>/dev/null)" ]; then
            has_validation=true
        fi
    fi

    if [ "${has_validation}" != "true" ]; then
        echo "Skipping ${label}: no validation jsonl under ${validation_dir}"
        skip_count=$((skip_count + 1))
        if [ "${STRICT}" = "true" ]; then
            echo "STRICT=true, aborting."
            exit 1
        fi
        continue
    fi

    if [ -f "${metrics_file}" ]; then
        run_spec="${label}=${validation_dir}::${metrics_file}"
    else
        echo "Including ${label} without metrics file: ${metrics_file}"
        run_spec="${label}=${validation_dir}"
    fi

    cmd+=("--run" "${run_spec}")
    run_count=$((run_count + 1))
done < "${MANIFEST_FILE}"

if [ "${run_count}" -eq 0 ]; then
    echo "No runnable runs found in manifest: ${MANIFEST_FILE}"
    exit 1
fi

if [ -n "${SELECTED_STEPS}" ]; then
    cmd+=("--selected-steps" "${SELECTED_STEPS}")
fi

if [ -n "${TRACE_UIDS}" ]; then
    IFS=',' read -r -a trace_uid_items <<< "${TRACE_UIDS}"
    for trace_uid in "${trace_uid_items[@]}"; do
        if [ -n "${trace_uid}" ]; then
            cmd+=("--trace-uid" "${trace_uid}")
        fi
    done
fi

if [ -n "${TOKEN_GROUP_SPECS}" ]; then
    IFS=';' read -r -a token_group_items <<< "${TOKEN_GROUP_SPECS}"
    for token_group_spec in "${token_group_items[@]}"; do
        if [ -n "${token_group_spec}" ]; then
            cmd+=("--token-group" "${token_group_spec}")
        fi
    done
fi

{
    printf "#!/usr/bin/env bash\n"
    printf "set -euo pipefail\n"
    printf "cd %q\n" "${WORKING_DIR}"
    printf "\n"
    printf "%q " "${cmd[@]}"
    printf "\n"
} > "${COMMAND_FILE}"
chmod +x "${COMMAND_FILE}"

echo "Manifest: ${MANIFEST_FILE}"
echo "Batch tag: ${batch_tag}"
echo "Runs included: ${run_count}"
echo "Runs skipped: ${skip_count}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Command file: ${COMMAND_FILE}"

if [ "${RUN_REPORT}" = "true" ]; then
    cd "${WORKING_DIR}"
    "${cmd[@]}"
    echo "Done. Open ${OUTPUT_DIR}/index.html and ${OUTPUT_DIR}/token_traces.html"
else
    echo "RUN_REPORT=false, generated command only."
    printf "%q " "${cmd[@]}"
    printf "\n"
fi
