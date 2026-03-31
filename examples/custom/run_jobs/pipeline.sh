#!/usr/bin/env bash
set -euo pipefail
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP}" \
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP}" \
bash /root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh
