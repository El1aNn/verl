# Rendered Reports 2026-04-05

本目录基于以下脚本产出：

- `examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh`
- `examples/custom/render_qwen_math_25_15B_grpo_2seed_compare_report.sh`

## 主入口

五个完整 run 的总对比报告：

- [compare_5runs/index.html](/root/rl/verl/analysis/rendered_reports_20260405/compare_5runs/index.html)
- [compare_5runs/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/compare_5runs/token_traces.html)
- [compare_5runs/summary.json](/root/rl/verl/analysis/rendered_reports_20260405/compare_5runs/summary.json)

## 单实验报告

- [base_align_exp1_20260401_141352/index.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_align_exp1_20260401_141352/index.html)
- [base_align_exp1_20260401_141352/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_align_exp1_20260401_141352/token_traces.html)

- [base_d_weak_constraint_20260402_003007/index.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_d_weak_constraint_20260402_003007/index.html)
- [base_d_weak_constraint_20260402_003007/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_d_weak_constraint_20260402_003007/token_traces.html)

- [base_b_reward_shuffle_20260402_132309/index.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_b_reward_shuffle_20260402_132309/index.html)
- [base_b_reward_shuffle_20260402_132309/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_b_reward_shuffle_20260402_132309/token_traces.html)

- [instruct_b_reward_shuffle_20260403_002727/index.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/instruct_b_reward_shuffle_20260403_002727/index.html)
- [instruct_b_reward_shuffle_20260403_002727/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/instruct_b_reward_shuffle_20260403_002727/token_traces.html)

- [instruct_c_n8_baseline_rerun180_20260404_113902/index.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/instruct_c_n8_baseline_rerun180_20260404_113902/index.html)
- [instruct_c_n8_baseline_rerun180_20260404_113902/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/instruct_c_n8_baseline_rerun180_20260404_113902/token_traces.html)

## 补充：n=1 step0 报告

- [base_c_n1_group_ablation_screen_20260401_102318/index.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_c_n1_group_ablation_screen_20260401_102318/index.html)
- [base_c_n1_group_ablation_screen_20260401_102318/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260405/verl_grpo_dsr_sub_baseline/base_c_n1_group_ablation_screen_20260401_102318/token_traces.html)

## 说明

- 这批报告主要来自 `validation/*.jsonl`，当前大多数 run 没有本地 `metrics/*.jsonl`，因此训练期 actor entropy 一类曲线可能为空。
- 五个完整 run 的 token trace 覆盖率目前是每个 step `8 / 4000` 条样本，且集中在同一个 probe 题上；因此更适合做机制观察，不适合直接外推为全任务统计。
- `base_c_n1_group_ablation_screen_20260401_102318` 当前只有 `step 0`，该报告只能作为失败前快照使用，不应与五个完整 run 的全过程结论等量齐观。

## 复现命令

单实验报告脚本：

- [render_qwen_math_25_15B_grpo_1_shot_report.sh](/root/rl/verl/examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh)

本次五 run 对比使用的 manifest：

- [manifest_5runs.tsv](/root/rl/verl/analysis/rendered_reports_20260405/manifest_5runs.tsv)

本次 compare 命令记录：

- [compare_report_command.sh](/root/rl/verl/analysis/rendered_reports_20260405/compare_5runs/compare_report_command.sh)
