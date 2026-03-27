# 项目变更结果

记录规则：
- 以追加为主，记录本项目中的代码变更。
- 每条记录概述实际改动、涉及文件、验证情况以及后续说明。

---
## 2026-03-19 记录 1：1-shot 与全量训练的验证期 Reasoning 诊断

实际改动：
- 在 PPO 验证流程中新增了 reasoning 诊断能力，包括稳定的验证样本 ID、token 级置信度 / entropy 摘要、跨 eval 的变化跟踪，以及可选的 token 轨迹导出。
- 扩展了 validation JSONL 的导出内容，使选中的样本除了序列级摘要外，还能携带 token 级诊断信息。
- 为验证生成结果增加了简洁的诊断摘要，便于在 WandB / SwanLab 表格中直接查看。
- 更新了 actor 的 log-prob 重计算逻辑，使其能够遵循通过 `meta_info` 传入的验证期 temperature。
- 在 trainer 配置中新增了验证诊断相关开关。
- 更新了自定义 GRPO 启动脚本，使诊断默认开启，并让 1-shot 脚本可在 `TRAIN_MODE=oneshot` 与 `TRAIN_MODE=full` 之间切换。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/megatron_workers.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/ppo_megatron_trainer.yaml`
- `examples/custom/run_qwen_math_25_15B_grpo.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

执行的命令或检查：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 尝试执行 `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile ...`，但由于本地是 Python 3.9，而仓库中已有 Python 3.10+ 语法，因此未能完成。

结果与剩余注意事项：
- Shell 脚本已通过语法检查。
- Python 改动已做人工检查，但由于当前环境只有 Python 3.9，而仓库本身依赖更新语法，因此无法在本机完成完整编译校验。
- token 级导出默认会被配置限制，以避免 validation 产物过大；如果你希望保留更深的轨迹，可以调大 `trainer.validation_diagnostics.samples_to_dump_token_details` 或 `max_tokens_per_sample`。

---
## 2026-03-19 记录 2：Token 分布与 EOS 概率监控

实际改动：
- 在 FSDP actor 的验证期 log-prob 重计算路径中，增加了可选的 token 分布诊断返回，能够输出每个位置的 EOS 概率、top-1 token 及其概率、top-k token 及其概率质量。
- 在 PPO validation 诊断逻辑中，新增 EOS 相关聚合指标，包括 `eos_prob_mean`、`eos_prob_final`、`eos_prob_slope`、`eos_high_prob_ratio`、`eos_top1_ratio`，以及跨相邻 eval 的 EOS 概率变化量。
- 扩展 validation JSONL 导出内容，选中的样本现在会逐 token 写出 `eos_prob`、`top1_*`、`topk`、`eos_in_topk`、`eos_rank_in_topk` 等字段，方便直接分析结束倾向和候选分布变化。
- 在训练配置和自定义脚本中新增 EOS / top-k 监控开关，便于分别对 1-shot 与 full 训练复用同一套观测方式。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/fsdp_workers.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/ppo_megatron_trainer.yaml`
- `examples/custom/run_qwen_math_25_15B_grpo.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

执行的命令或检查：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 针对 `verl/trainer/ppo/ray_trainer.py` 与 `verl/workers/actor/dp_actor.py` 进行了人工代码审查，重点检查新增张量的形状传播与 JSONL 导出字段。

结果与剩余注意事项：
- 现在默认可以监控逐 token 的 top-k 分布摘要，并重点观察 EOS 概率是否提前抬升、是否进入 top-1 / top-k。
- FSDP 路径已接通该能力；如果走 fused kernels 或其他暂未返回分布张量的路径，会给出 warning，并跳过这部分细粒度指标。
- 当前环境仍只有 Python 3.9，因此无法对整个仓库做完整编译校验；但 shell 入口已通过语法检查，核心 Python 改动已完成针对性人工审查。

---
## 2026-03-23 记录 3：论文风格实验可视化与结果展示

实际改动：
- 新增 `scripts/validation_viz_report.py`，可以直接读取 `trainer.validation_data_dir` 下的 validation JSONL，以及可选的 `file` logger 指标日志，生成论文风格的 SVG/HTML 报告。
- 报告默认支持多实验对比，并覆盖验证指标轨迹、entropy 轨迹、低概率 token 的采样分布、概率-熵散点、低概率 top-k 候选均值趋势，以及代表性样本展示。
- 为三个自定义实验脚本增加了 `ENABLE_PAPER_STYLE_VIZ` 开关；开启后会自动把 logger 扩展为 `["swanlab","file"]`，并把 token 细节 dump 与 top-k 范围调到更适合离线出图的默认值。
- 自定义实验脚本现在会把本地 file logger 输出定向到 `${CKPTS_DIR}/metrics`，并在提交成功后打印一条可直接运行的报告生成命令。
- 新增了一个基于合成 validation 数据的轻量测试，用于校验报告脚本的核心产物和统计结果。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `scripts/validation_viz_report.py`
- `tests/utils/test_validation_viz_report_on_cpu.py`
- `examples/custom/run_qwen_math_25_15B_grpo.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

执行的命令或检查：
- `python3 scripts/validation_viz_report.py --help`
- `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile scripts/validation_viz_report.py tests/utils/test_validation_viz_report_on_cpu.py`
- 使用合成 validation JSONL 与 metrics JSONL 执行了一次 `scripts/validation_viz_report.py`，确认 `index.html`、`summary.json` 和各类 SVG 都能正常生成
- `bash -n examples/custom/run_qwen_math_25_15B_grpo.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 手动调用 `tests/utils/test_validation_viz_report_on_cpu.py` 中的测试函数完成一次本地校验；当前环境缺少 `pytest`，因此未能直接运行 pytest 命令

结果与剩余注意事项：
- 报告脚本本身不依赖 `matplotlib`、`numpy`、`pandas` 等额外绘图库，因此在较轻量的环境中也能直接出图。
- 若想更接近论文中的 token 分布图，建议在跑实验时开启 `ENABLE_PAPER_STYLE_VIZ=true`；这会显著增加 validation dump 体积，因为默认会导出全部样本的 token 细节并把 top-k 提高到 64。
- 如果某次实验的 token diagnostics 只覆盖了部分样本，报告会在 HTML 中给出 coverage warning，并提示继续增大 dump 范围。

---
## 2026-03-23 记录 4：1-shot 结果可视化与 Token 追踪脚本更新

实际改动：
- 扩展 `scripts/validation_viz_report.py`，新增 token trace 输出能力。主报告之外，现在还会生成 `token_traces.html`，用于按样本 UID 查看跨 validation step 的逐 token 概率、entropy、EOS 概率和 top-1 候选变化。
- 在报告脚本 CLI 中新增 `--trace-uid` 与 `--trace-top-samples`，既可以精确追踪指定 UID，也可以自动挑选最新 step 中最有代表性的样本做 token trace。
- 更新 `run_qwen_math_25_15B_grpo_1_shot.sh` 与 `run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`，让 1-shot 训练默认更偏向可视化/追踪场景：默认开启更完整的 report 友好模式，并把 token 细节导出长度提升到更适合追踪的范围。
- 新增 `examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh`，可以直接从 `CKPTS_DIR` 渲染 1-shot 主报告和 token trace 页面，不必手写长命令。
- 更新合成数据测试，确保 `token_traces.html` 也会被生成，且包含样本级 token 追踪内容。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `scripts/validation_viz_report.py`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh`
- `tests/utils/test_validation_viz_report_on_cpu.py`

执行的命令或检查：
- `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile scripts/validation_viz_report.py tests/utils/test_validation_viz_report_on_cpu.py`
- `python3 scripts/validation_viz_report.py --help`
- 手动调用 `tests/utils/test_validation_viz_report_on_cpu.py` 中的测试函数，确认主报告与 `token_traces.html` 都会生成
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh`
- 基于临时目录构造了一份最小 1-shot validation / metrics 结构，并成功执行 `render_qwen_math_25_15B_grpo_1_shot_report.sh`

结果与剩余注意事项：
- 现在 1-shot 训练结束后，可以直接渲染出总览报告和 token trace 页面，调试单个样本在不同 step 的行为会更直接。
- 1-shot 脚本为了方便可视化，默认会开启更重的 diagnostics 导出；如果更在意存储和 validation 速度，可以显式设置 `ENABLE_PAPER_STYLE_VIZ=false` 或下调 `VAL_DIAG_MAX_TOKENS`、`VAL_DIAG_DUMP_SAMPLES`、`VAL_DIAG_DISTRIBUTION_TOPK`。
- `token_traces.html` 依赖 validation dump 中包含 `token_diagnostics`；如果训练时关闭了这部分导出，主报告仍能生成，但 token trace 页面不会包含可追踪样本。
