export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:-}"
set -x

python3 -m verl.trainer.main_ppo \
    --config-path=examples/custom/config \
    --config-name=qwen_25_05B_grpo \
    2>&1 | tee verl_grpo_demo.log