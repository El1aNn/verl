#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl

HOOK_SCRIPT=${HOOK_SCRIPT:-/root/rl/verl/examples/custom/run_jobs/hook_check_and_advance.sh}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}
RUNNER_LOG=${RUNNER_LOG:-/root/rl/verl/logs/queue/hook_runner_30m.log}

mkdir -p /root/rl/verl/logs/queue
touch "$RUNNER_LOG"

while true; do
    now=$(date '+%F %T')
    echo "[$now] runner tick" >> "$RUNNER_LOG"
    bash "$HOOK_SCRIPT" >> "$RUNNER_LOG" 2>&1 || true
    sleep "$INTERVAL_SECONDS"
done

