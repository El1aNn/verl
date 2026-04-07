# Rendered Reports 2026-04-06

这批报告是在 `n=1` 重跑完成之后重新生成的六实验版本。

## 主入口

- [compare_6runs/index.html](/root/rl/verl/analysis/rendered_reports_20260406/compare_6runs/index.html)
- [compare_6runs/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260406/compare_6runs/token_traces.html)
- [compare_6runs/summary.json](/root/rl/verl/analysis/rendered_reports_20260406/compare_6runs/summary.json)

## 新纳入的 n=1 正式版本

- [base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719/index.html](/root/rl/verl/analysis/rendered_reports_20260406/verl_grpo_dsr_sub_baseline/base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719/index.html)
- [base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719/token_traces.html](/root/rl/verl/analysis/rendered_reports_20260406/verl_grpo_dsr_sub_baseline/base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719/token_traces.html)

## 对比清单

- [manifest_6runs.tsv](/root/rl/verl/analysis/rendered_reports_20260406/manifest_6runs.tsv)
- [compare_report_command.sh](/root/rl/verl/analysis/rendered_reports_20260406/compare_6runs/compare_report_command.sh)

## 说明

- 本版已经把新的 `n=1` 重跑正式纳入六实验对比。
- `n=1` 版本使用 `val_rollout_n=1`，因此每个 step 约 `500` 条验证记录；`n=8` 版本每个 step 约 `4000` 条验证记录。对比时应说明评估口径差异。
- `reward shuffle` 数据文件当前仍与 clean 文件 `md5` 一致，因此相关实验仍只能按“名义负对照”来解释。
