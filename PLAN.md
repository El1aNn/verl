# 项目变更计划

记录规则：
- 以追加为主，记录本项目中的代码变更。
- 每条记录概述目标、范围、实现方案和验证计划。

---
## 2026-03-19 记录 1：1-shot 与全量训练的验证期 Reasoning 诊断

目标：
- 增加验证阶段的诊断能力，用于对比 1-shot 训练与全量训练，重点关注 reasoning 质量以及 token 级置信度行为。

预计修改的文件或区域：
- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/tracking.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/_generated_ppo_trainer.yaml`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 如有需要，也可能修改用于验证诊断的共享辅助模块

实现方案：
- 为验证样本增加稳定 ID，使同一道题可以在不同 eval step 之间对齐比较。
- 基于生成结果以及重新计算的策略 log-prob / entropy，计算 token 级验证诊断指标。
- 将面向 reasoning 的指标聚合进 validation 日志，并为选定样本导出精简的 token 轨迹。
- 暴露少量配置和脚本开关，让 1-shot 与全量训练共用同一套监控路径。

验证计划：
- 对修改后的 Python 文件执行有针对性的静态检查。
- 检查更新后运行脚本的 shell 语法是否正确。
- 审查新增指标名和导出字段，确保与现有 validation 日志兼容。

---
## 2026-03-19 记录 2：Token 分布与 EOS 概率监控

目标：
- 在验证阶段监控每个生成位置的 token 概率分布摘要，特别关注 EOS 结束符号概率随位置和训练步数的变化。

预计修改的文件或区域：
- `verl/workers/actor/dp_actor.py`
- `verl/workers/fsdp_workers.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/ppo_megatron_trainer.yaml`
- `examples/custom/run_qwen_math_25_15B_grpo.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

实现方案：
- 在 FSDP actor 的 log-prob 重计算路径中增加可选分布统计返回，包括 EOS 概率、top-1 概率以及每个位置的 top-k token 概率。
- 在 validation 诊断逻辑中汇总 EOS 相关标量指标，并把选中样本的逐 token top-k 分布与 EOS 概率写入导出文件。
- 通过配置和脚本开关控制 top-k 大小与 EOS 高概率阈值，避免默认导出过大。

验证计划：
- 对修改后的 shell 脚本执行语法检查。
- 对涉及的 Python 改动做针对性静态检查或人工审查，确认新增张量形状与现有 validation 流程兼容。

---
## 2026-03-23 记录 3：论文风格实验可视化与结果展示

目标：
- 为当前实验补充接近 `2510.03222v1.pdf` 风格的结果展示能力，能够从本地实验产物中生成训练动态、低概率 token 分布、概率-熵关系和代表性样本展示。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `scripts/` 下新增实验报告生成脚本
- `examples/custom/run_qwen_math_25_15B_grpo.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `tests/` 下新增针对报告脚本的轻量验证

实现方案：
- 新增离线分析脚本，读取 `trainer.validation_data_dir` 下的 validation JSONL，以及可选的本地 metrics 日志，生成论文风格的 SVG/HTML 报告。
- 报告默认支持多实验对比，重点覆盖验证集准确率或 reward 曲线、训练 entropy 曲线、指定 token 组的低概率分布、概率-熵散点和代表性样本表格。
- 为自定义训练脚本增加与报告生成更匹配的环境变量和注释，方便实验结束后直接产出结果页。

验证计划：
- 用合成的 validation JSONL 和 metrics 日志跑一次报告脚本，检查输出文件和关键统计是否正确。
- 对修改后的 shell 脚本执行语法检查。
- 对新增脚本执行一次帮助或最小输入验证，确保 CLI 可用且无明显语法错误。

---
## 2026-03-23 记录 4：1-shot 结果可视化与 Token 追踪脚本更新

目标：
- 让 1-shot 实验能更直接地产出结果可视化与样本级 token 追踪页面，减少训练后手工拼接命令和筛选样本的工作量。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `scripts/validation_viz_report.py`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 视情况新增 1-shot 报告生成辅助脚本
- `tests/utils/test_validation_viz_report_on_cpu.py`

实现方案：
- 在离线报告脚本中增加 token trace 能力，支持按样本 UID 跨 step 追踪，并自动生成样本级 token 轨迹页面。
- 为 1-shot 训练脚本补充更适合 token 追踪的默认导出配置，以及更直接的报告生成入口。
- 如有必要，新增一个 1-shot 专用的报告渲染辅助脚本，把常用参数收敛成一个入口。

验证计划：
- 对新增/更新的 Python 文件执行编译检查。
- 用合成的 validation JSONL 验证 token trace 页面和主报告都能生成。
- 对更新后的 1-shot shell 脚本执行语法检查。
