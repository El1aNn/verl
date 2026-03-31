#!/usr/bin/env bash
set -euo pipefail

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
PYTHON_BIN=${PYTHON_BIN:-python3}
HOME_DIR=${HOME_DIR:-/root/rl}

if [ $# -ge 1 ] && [ -n "${1:-}" ]; then
    CKPTS_DIR="$1"
else
    CKPTS_DIR=${CKPTS_DIR:-""}
fi

if [ -z "${CKPTS_DIR}" ]; then
    echo "Usage: CKPTS_DIR=/path/to/ckpts/<project>/<exp> bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh"
    echo "   or: bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh /path/to/ckpts/<project>/<exp>"
    exit 1
fi

PROJECT_NAME=${PROJECT_NAME:-"$(basename "$(dirname "${CKPTS_DIR}")")"}
EXP_NAME=${EXP_NAME:-"$(basename "${CKPTS_DIR}")"}
RUN_LABEL=${RUN_LABEL:-"${EXP_NAME}"}
VALIDATION_DIR=${VALIDATION_DIR:-"${CKPTS_DIR}/validation"}
METRICS_FILE=${METRICS_FILE:-"${CKPTS_DIR}/metrics/${PROJECT_NAME}/${EXP_NAME}.jsonl"}
REPORT_ROOT=${REPORT_ROOT:-"${HOME_DIR}/verl/reports"}
REPORT_DIR=${REPORT_DIR:-"${REPORT_ROOT}/${PROJECT_NAME}/${EXP_NAME}"}
TRACE_TOP_SAMPLES=${TRACE_TOP_SAMPLES:-6}
TRACE_UIDS=${TRACE_UIDS:-""}
SELECTED_STEPS=${SELECTED_STEPS:-""}
TOKEN_GROUP_SPECS=${TOKEN_GROUP_SPECS:-""}

mkdir -p "${REPORT_DIR}"

cmd=(
    "${PYTHON_BIN}" "scripts/validation_viz_report.py"
    "--run" "${RUN_LABEL}=${VALIDATION_DIR}::${METRICS_FILE}"
    "--output-dir" "${REPORT_DIR}"
    "--trace-top-samples" "${TRACE_TOP_SAMPLES}"
)

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

echo "Rendering 1-shot report from:"
echo "  CKPTS_DIR=${CKPTS_DIR}"
echo "  VALIDATION_DIR=${VALIDATION_DIR}"
echo "  METRICS_FILE=${METRICS_FILE}"
echo "  REPORT_ROOT=${REPORT_ROOT}"
echo "  REPORT_DIR=${REPORT_DIR}"

cd "${WORKING_DIR}"
"${cmd[@]}"

echo "Done. Open ${REPORT_DIR}/index.html and ${REPORT_DIR}/token_traces.html"
