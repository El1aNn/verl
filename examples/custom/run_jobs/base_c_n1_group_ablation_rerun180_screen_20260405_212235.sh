#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl
bash /root/rl/verl/examples/custom/run_jobs/base_c_n1_group_ablation_rerun180_20260405_212235.sh
echo "[launcher done, keep screen alive]"
sleep 43200
