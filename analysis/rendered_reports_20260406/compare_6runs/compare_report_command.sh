#!/usr/bin/env bash
set -euo pipefail
cd /root/rl/verl

python3 scripts/validation_viz_report.py --output-dir /root/rl/verl/analysis/rendered_reports_20260406/compare_6runs --trace-top-samples 12 --run base_c_n8_baseline=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/base_align_exp1_20260401_141352/validation --run base_c_n1_group_ablation=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719/validation --run base_d_weak_constraint=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/base_d_weak_constraint_20260402_003007/validation --run base_b_reward_shuffle=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/base_b_reward_shuffle_20260402_132309/validation --run instruct_b_reward_shuffle=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/instruct_b_reward_shuffle_20260403_002727/validation --run instruct_c_n8_baseline=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/instruct_c_n8_baseline_rerun180_20260404_113902/validation 
