#!/bin/bash

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="f408d6f1e1f982b94e6034176c0cd1f72cf9ab62"
set -x

# Run the training using the yaml config
python3 -m verl.trainer.main_ppo --config-name=qwen_25_05B_grpo.yaml \
    --config-path=./recipe/grpo/config \
    2>&1 | tee verl_grpo_demo.log
