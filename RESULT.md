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

---
## 2026-03-27 记录 5：本地 Smoke 训练排查、日志落盘与基础 GRPO 收敛

实际改动：
- 更新 `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`，支持通过环境变量显式指定 `LOG_DIR` / `LOG_FILE`，便于把 `ray job logs --follow` 直接落到固定文件中。
- 在本机自动启动 Ray 的分支里，补了 dashboard 地址的本地 IP 解析逻辑，避免脚本继续沿用 `http://127.0.0.1:8265` 而导致后续日志跟踪不稳定。
- 为训练脚本补充了更灵活的环境变量覆盖入口，并新增 smoke 运行、2-seed 矩阵生成、compare report 生成等辅助脚本，方便快速验证链路。
- 使用 `screen` 启动了多轮 smoke 训练，把提交阶段日志与训练阶段日志分别写入独立文件，以便定位故障点。
- 在 `recipe.entropy.main_entropy` 路径上逐步定位出当前阻塞点：`data.seed` 需要追加键、`actor_rollout_ref.rollout.seed` 虽可用追加键传入 Hydra，但会在 `RolloutConfig` dataclass 实例化时触发不兼容。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh`
- `examples/custom/TRAINING_PIPELINE_1SHOT_GRPO_LOGITS.md`
- `examples/custom/prepare_qwen_math_25_15B_grpo_2seed_matrix.sh`
- `examples/custom/render_qwen_math_25_15B_grpo_2seed_compare_report.sh`
- `REPORT_1SHOT_GRPO_LOGITS_PLAN_STORY.md`

执行的命令或检查：
- 在 `conda` 的 `verl` 环境里检查了 `Python 3.10`、`torch 2.6.0+cu124`、`CUDA=True` 和 `2 x A800 40GB`。
- 多次通过 `screen` 启动 smoke 训练，分别观察 launch log、train log、Ray 进程和 `recipe.entropy.main_entropy` 主进程状态。
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh`
- 并行检查了 Ray session 日志、`ray job submit` / `ray job logs` 进程、以及训练输出文件内容。

结果与剩余注意事项：
- 当前已经确认：日志可以稳定写入文件，`screen` 会话可持续保留，训练链路已经越过环境检查、Ray 启动、作业提交、数据集加载和 worker 初始化前半段。
- 当前 entropy 变体路径上最新的明确阻塞点是：`RolloutConfig.__init__()` 不接受 `seed` 字段，因此 `actor_rollout_ref.rollout.seed` 必须移除，而不是继续通过 Hydra 追加。
- 训练日志已经明确暴露了这一点，因此下一步应按用户要求去掉 `seed` 覆盖，并优先切回更基础的 `GRPO` 入口（`verl.trainer.main_ppo`）验证最小可运行版本。

---
## 2026-03-27 记录 6：移除 Seed 干扰并切回基础 GRPO smoke

实际改动：
- 更新 `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`，补齐与前序排查一致的 dashboard 地址修正、本地日志落盘以及常用训练参数环境变量覆盖能力。
- 新增 `examples/custom/run_qwen_math_25_15B_grpo_1_shot_smoke.sh`，用于以更轻量的 batch / rollout / epoch 配置启动基础 GRPO smoke。
- 将当前执行方向从 `recipe.entropy.main_entropy` + `kl_cov` 变体收敛回 `verl.trainer.main_ppo` 基础 GRPO 路径，减少结构性配置兼容风险。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_smoke.sh`

执行的命令或检查：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_smoke.sh`

结果与剩余注意事项：
- 当前已完成脚本层收敛，下一步是直接用基础 GRPO smoke 重跑，并验证其是否能够稳定进入运行态。
- 由于用户明确要求先去掉 `seed` 并跑最基础版本，本记录之后不再将 `seed` 作为默认覆盖项传入训练入口。

---
## 2026-03-27 记录 7：基础 GRPO smoke 重跑并确认进入正常运行阶段

实际改动：
- 未新增代码改动；本轮重点是继续盯住基础 GRPO smoke 重跑状态，并把现场信息补充进跟踪文档。
- 已确认当前运行使用的是基础入口 `python3 -m verl.trainer.main_ppo`，没有再传入 `seed` 覆盖。

修改的文件：
- `PLAN.md`
- `RESULT.md`

执行的命令或检查：
- `tail -n 120 logs/verl_grpo_smoke_20260327-205828.launch.log`
- `tail -n 220 logs/verl_grpo_smoke_20260327-205828.train.log`
- `screen -ls`
- `ps -ef | rg 'verl.trainer.main_ppo|raysubmit_xfneqvpJjxsppDDV|ray job logs --follow'`
- `ray job status --address=http://192.168.122.2:8265 raysubmit_xfneqvpJjxsppDDV`
- `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`

结果与剩余注意事项：
- 当前 `screen` 会话 `verl_grpo_smoke_20260327-205828` 正常存活，launch log 已确认 Ray 本地集群启动成功，job `raysubmit_xfneqvpJjxsppDDV` 提交成功，并持续向训练日志文件写入。
- `ray job status` 当前返回 `RUNNING`，说明作业仍处于正常执行状态，而不是提交后立即失败。
- `train.log` 已经越过配置校验、数据集加载、prompt 过滤、placement group 准备等阶段，进入 `actor_rollout_init_model` worker 的模型初始化流程。
- 两张 A800 当前均占用约 `26.9GB / 40GB` 显存，`actor_rollout_init_model` worker 保持高 CPU，占用特征与大模型初始化阶段一致，因此当前更像是“正常加载中”而不是“死掉但没报错”。
- 截至本记录，`ckpts/verl_grpo_dsr_sub_baseline/smoke_qwen_math_25_15B_instruct_grpo_20260327-205828` 目录尚未出现；这意味着训练还没走到首次持久化或 validation 输出落盘的位置，后续仍需继续盯日志，直到出现更明确的 step / eval / checkpoint 信号。

---
## 2026-03-27 记录 8：修复 SwanLab 登录阻塞并重新提交基础 GRPO smoke

实际改动：
- 更新 `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`，移除脚本顶部硬编码的 `WANDB_API_KEY` / `SWANLAB_API_KEY`，避免脚本默认携带失效凭证。
- 更新 `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh` 和 `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh` 的 logger 逻辑：未显式设置 `LOGGER` 时，仅在外部环境存在 `SWANLAB_API_KEY` 时启用 `["swanlab","file"]`，否则默认退回 `["file"]`。
- 重新以 `screen` 启动基础 GRPO smoke，新 run 显式清掉 `SWANLAB_API_KEY` / `WANDB_API_KEY`，避免再次触发第三方平台登录。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

执行的命令或检查：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_smoke.sh`
- 使用 `screen` 启动新会话 `verl_grpo_smoke_20260327-210328`
- 检查 `logs/verl_grpo_smoke_20260327-210328.launch.log`
- 检查 `logs/verl_grpo_smoke_20260327-210328.train.log`
- 检查 `ray job status --address=http://192.168.122.2:8265 raysubmit_r5jtfU92xsbadCMU`
- 检查 `ps -ef | rg 'main_ppo|raysubmit_r5jtfU92xsbadCMU|actor_rollout_init_model'`

结果与剩余注意事项：
- 上一轮基础 smoke 的真实失败原因已经确认，不是 GRPO 主体配置问题，而是 `swanlab` 登录失败导致训练在 `Tracking(...)` 初始化处退出。
- 新一轮基础 smoke 已成功提交，launch log 中已经明确显示 `trainer.logger=["file"]`，说明 logger 修复已生效。
- 当前新 job `raysubmit_r5jtfU92xsbadCMU` 状态为 `RUNNING`，`python3 -m verl.trainer.main_ppo` 主进程和后续 worker 初始化流程均已启动，且未再出现 SwanLab 登录异常。
- 截至本记录，新 run 已经越过配置校验、数据集过滤、placement group 准备，并进入模型初始化阶段；这说明我们已经把实验推进到了“能够正常运行”的位置。
- 仍需继续观察训练日志，等待首次 validation / step / checkpoint 输出，以确认 run 不会在后续 vLLM 或 FSDP 初始化阶段暴露新的运行时问题。

---
## 2026-03-27 记录 9：持续监控基础 GRPO smoke 并确认产出首个 validation 文件

实际改动：
- 本轮未新增代码改动；重点是继续监控当前基础 GRPO smoke 的运行状态，并补充“已经开始产出文件”的现场证据。

修改的文件：
- `PLAN.md`
- `RESULT.md`

执行的命令或检查：
- `tail -n 160 logs/verl_grpo_smoke_20260327-210328.train.log`
- `ray job status --address=http://192.168.122.2:8265 raysubmit_r5jtfU92xsbadCMU`
- `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`
- `find ckpts/verl_grpo_dsr_sub_baseline/smoke_qwen_math_25_15B_instruct_grpo_20260327-210328 -maxdepth 3 -type f`

结果与剩余注意事项：
- 当前 job `raysubmit_r5jtfU92xsbadCMU` 仍为 `RUNNING`。
- 当前两张 GPU 显存占用已提升到约 `34.5GB` 和 `36.1GB`，说明 worker 初始化和后续执行仍在持续推进。
- 关键进展是：`/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/smoke_qwen_math_25_15B_instruct_grpo_20260327-210328/validation/0.jsonl` 已经落盘。
- 这说明当前基础 GRPO smoke 已经越过“只有进程和显存占用”的阶段，真正跑到了 validation 结果写出的位置，可以视为训练链路已经基本跑通。
- 后续仍建议继续盯第一轮 step / checkpoint 是否正常出现，这样就能进一步确认训练、评估、保存三条链路都完整可用。

---
## 2026-03-27 记录 10：继续监控直到训练开始，并判断是否停在 validation 之后

实际改动：
- 本轮未新增代码改动；重点是继续监控当前 smoke run，确认是否已经出现第一条训练 step。
- 直接改用 Ray 的实时 job 日志与 `job-driver` 日志做判断，避免仅依赖本地 `train.log` 文件刷新情况。

修改的文件：
- `PLAN.md`
- `RESULT.md`

执行的命令或检查：
- `ray job logs --address=http://192.168.122.2:8265 raysubmit_r5jtfU92xsbadCMU | tail -n 160`
- `tail -n 160 /tmp/ray/session_latest/logs/job-driver-raysubmit_r5jtfU92xsbadCMU.log`
- `ps -o pid,etime,stat,%cpu,%mem,cmd -p 31814,32088,32089`
- `top -b -n 1 -H -p 31814`
- `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`
- `stat -c '%y %s %n' /tmp/ray/session_latest/logs/job-driver-raysubmit_r5jtfU92xsbadCMU.log .../validation/0.jsonl`
- `timeout 5 strace -f -p 31814 -s 80 -o /tmp/strace_31814.txt`

结果与剩余注意事项：
- 当前可以明确确认：run 已经完成 `val_before_train`，日志里出现了 `Training from scratch`、`validation generation end`，并成功写出了 `validation/0.jsonl`。
- 但截至本记录，仍未观察到第一条明确的训练 step / loss / rollout metric；因此严格来说，“训练优化阶段”还没有被日志证实已经开始。
- `job-driver` 日志和 `validation/0.jsonl` 的最后更新时间都停在 `2026-03-27 21:07` 左右，没有继续新增内容。
- 同时，`TaskRunner` 进程仍保持较高 CPU（约 75%~90%），`WorkerDict` 也持续占用 CPU，但两张 GPU 利用率接近 0，仅显存维持高占用。这更像是停在训练前或训练切换阶段的 CPU 侧处理，而不是已经进入稳定的 GPU 训练 step。
- 因此当前最准确的结论是：基础 smoke 已经跑通到“初始 validation 成功”这一步，但在真正开始训练 step 之前可能还存在卡点，需要继续专项排查这一段流程。

---
## 2026-03-27 记录 11：定位 validation 后慢点并修复训练阶段返回值解包错误

实际改动：
- 更新 `verl/workers/actor/dp_actor.py`，修复 `update_policy()` 中对 `_forward_micro_batch()` 返回值的解包错误：该函数现在返回 `(entropy, log_prob, distribution_tensors)` 三元组，训练路径此前仍按二元组解包。
- 更新 `verl/trainer/ppo/ray_trainer.py`，在 `process_validation_metrics(...)` 前后增加耗时打印，方便后续直接看出 validation 收尾是否在做长时间的 metrics 聚合。

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/ray_trainer.py`

执行的命令或检查：
- 阅读 `_validate()` 后半段代码，重点定位 [ray_trainer.py](/root/rl/verl/verl/trainer/ppo/ray_trainer.py#L1135) 到 [ray_trainer.py](/root/rl/verl/verl/trainer/ppo/ray_trainer.py#L1184) 的 validation 收尾逻辑
- 阅读 [metric_utils.py](/root/rl/verl/verl/trainer/ppo/metric_utils.py#L309) 和 [metric_utils.py](/root/rl/verl/verl/trainer/ppo/metric_utils.py#L437)，确认 `bootstrap_metric()` 与 `process_validation_metrics()` 的复杂度来源
- 阅读 [dp_actor.py](/root/rl/verl/verl/workers/actor/dp_actor.py#L124) 与 [dp_actor.py](/root/rl/verl/verl/workers/actor/dp_actor.py#L554)，确认训练报错的调用链
- `python3 -m py_compile verl/workers/actor/dp_actor.py verl/trainer/ppo/ray_trainer.py`
- 检查 `/tmp/ray/session_latest/logs/job-driver-raysubmit_r5jtfU92xsbadCMU.log`
- 检查 `validation/0.jsonl` 中的样本规模与字段数量

结果与剩余注意事项：
- “看起来卡住”的主因已经理清：当前 validation dump 一共有 `2000` 行（`500` 个 uid，每个 uid `4` 个 responses），并包含 `38` 个 numeric scalar metrics。按现有实现，`process_validation_metrics()` 会为这些变量按 uid 做 bootstrap 聚合，估算下来大约会触发 `38000` 次 bootstrap 调用、约 `76000000` 次 reduce 计算。
- 小样本基准也支持这个判断：仅 `10` 个 uid、`38` 个变量的 `process_validation_metrics()` 调用，就已经需要约 `18.4s`；线性外推到 `500` 个 uid 时，达到十几分钟是完全合理的。
- 因此此前的“卡住不动”，本质上不是进程死掉，而是 `_validate()` 在 `Dumped generations to .../validation/0.jsonl` 之后，还在做非常重的 validation metrics 聚合；这也是为什么日志迟迟没出现 `Initial validation metrics`。
- 同时，run 最终并不是停在这里结束，而是在 metrics 聚合完成、真正进入训练后，马上在 `actor.update_policy()` 处失败。最新 job 日志已经明确给出报错：`ValueError: too many values to unpack (expected 2)`，位置在 [dp_actor.py](/root/rl/verl/verl/workers/actor/dp_actor.py#L554)。
- 这个训练报错与 metrics 改动直接相关：`_forward_micro_batch()` 现在已经固定返回 3 个值，但训练更新路径还保留旧的 2 值解包逻辑。该问题已按最小改动修复。
- 现阶段的整体结论可以分成两层：
  1. validation 后“很久不动”主要是你新增 metrics 带来的 CPU 聚合成本；
  2. 真正进入训练后立即失败，则是 `_forward_micro_batch()` 返回值升级后，`update_policy()` 没同步解包导致的代码错误。

---
## 2026-03-27 记录 12：把 validation diagnostics 轻量化到仅分析部分 val 样本

实际改动：
- 更新 [ray_trainer.py](/root/rl/verl/verl/trainer/ppo/ray_trainer.py)，新增 `trainer.validation_diagnostics.max_samples` 的读取逻辑。
- 在 `_validate()` 中把限制前移到 diagnostics 的额外 `compute_log_prob()` 之前：现在只会对前 `N` 个 validation 样本执行额外 diagnostics 前向，其余样本仍然正常参与 reward/acc 验证，但不会再附带 token/logit diagnostics。
- 在 validation 日志中新增两条提示：
  - `Validation diagnostics sample budget: ...`
  - `Validation diagnostics analyzed ... samples in this eval pass`
- 更新两个启动脚本：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
  - [run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh)
  - 新增环境变量 `VAL_DIAG_MAX_SAMPLES`
  - 当前默认值设为 `64`

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/ppo/ray_trainer.py`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

执行的命令或检查：
- `python3 -m py_compile verl/trainer/ppo/ray_trainer.py`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

结果与剩余注意事项：
- 这次改动保留了全量 validation 的核心评估链路，不会影响 `reward/acc` 的全量统计；被裁掉的只是你新增的 diagnostics 前向与对应的 aux metrics 聚合。
- 相比只减少 `token_diagnostics` 导出条数，这种做法更有效，因为它同时降低了：
  - 额外 `compute_log_prob()` 的 GPU/前向成本
  - diagnostics scalar 的 bootstrap 聚合成本
  - token-level diagnostics JSON 的体积
- 默认 `VAL_DIAG_MAX_SAMPLES=64` 是一个偏 smoke/调试友好的选择；如果后面正式出 report，需要全量 diagnostics，可以显式设置 `VAL_DIAG_MAX_SAMPLES=-1` 恢复原行为。

---
## 2026-03-27 记录 13：回看上一次 smoke run 的最终失败原因

实际改动：
- 本轮未修改训练逻辑；重点是回看上一条 smoke run 的结束状态，并与当前工作区修复状态做对照。

修改的文件：
- `PLAN.md`
- `RESULT.md`

执行的命令或检查：
- `tail -n 120 logs/verl_grpo_smoke_20260327-210328.train.log`
- `tail -n 120 /tmp/ray/session_latest/logs/job-driver-raysubmit_r5jtfU92xsbadCMU.log`
- `nl -ba verl/workers/actor/dp_actor.py | sed -n '548,558p'`
- `stat -c '%y %n' verl/workers/actor/dp_actor.py verl/trainer/ppo/ray_trainer.py logs/verl_grpo_smoke_20260327-210328.train.log`

结果与剩余注意事项：
- 可以明确确认：上一条 smoke run 已经失败，不是“还在后台卡着”。
- 最终失败点仍然是 [dp_actor.py:554](/root/rl/verl/verl/workers/actor/dp_actor.py:554) 这一处训练阶段解包错误，日志中的报错是：
  - `ValueError: too many values to unpack (expected 2)`
- 这条 run 在失败前其实已经完成了 initial validation，并打印出了大量 `val-aux/*` metrics；真正崩掉是在进入 `Training Progress: 0/37` 之后第一次 `update_actor`。
- 当前工作区中的同一位置已经是修复后的代码：
  - `entropy, log_prob, _ = self._forward_micro_batch(...)`
- 时间戳也能对上：训练日志最后更新时间是 `2026-03-27 21:22:30`，而本地 `dp_actor.py` 修补时间是 `2026-03-27 21:27:11`。这说明失败发生在补丁之前，属于“旧 run 用旧代码失败”，不是当前修复后的代码再次复发。
- 结论上，现在不需要继续排查这条旧 run 本身；更合适的下一步是基于当前补丁后的代码重新启动一次 smoke/basic GRPO run。

---
## 2026-03-27 记录 14：重启基础 GRPO smoke，并修复 Hydra 对新增 diagnostics 键的覆盖方式

实际改动：
- 用 `screen` 启动了一条新的基础 GRPO smoke run，固定使用 `conda` 环境 `verl`。
- 首次重提时，job 很快失败在 Hydra 启动层，错误是：
  - `Could not override 'trainer.validation_diagnostics.max_samples'`
  - `To append to your config use +trainer.validation_diagnostics.max_samples=64`
- 因此更新了两个启动脚本，把新增键的覆盖方式从普通 override 改成兼容新增键的 `++` 写法：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
  - [run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh)

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

执行的命令或检查：
- 启动失败的第一次 smoke：
  - `screen -dmS verl_grpo_smoke_20260327-213840 ... bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_smoke.sh`
- 检查失败日志：
  - [verl_grpo_smoke_20260327-213840.launch.log](/root/rl/verl/logs/verl_grpo_smoke_20260327-213840.launch.log)
  - [verl_grpo_smoke_20260327-213840.train.log](/root/rl/verl/logs/verl_grpo_smoke_20260327-213840.train.log)
- 语法检查：
  - `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
  - `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 启动修复后的第二次 smoke：
  - `screen -dmS verl_grpo_smoke_20260327-214047 ... bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_smoke.sh`
- 持续检查：
  - `ray job status --address=http://192.168.122.2:8265 raysubmit_67zT2ZwLprkZvxzx`
  - `tail -n ... logs/verl_grpo_smoke_20260327-214047.train.log`
  - `nvidia-smi ...`

结果与剩余注意事项：
- 失败的第一次新 run 只是 Hydra 新增键覆盖方式不兼容，不是训练逻辑问题。
- 修复后，第二次新 run 已成功提交并处于 `RUNNING`：
  - `screen` 会话：`verl_grpo_smoke_20260327-214047`
  - Ray job：`raysubmit_67zT2ZwLprkZvxzx`
  - 启动日志：[verl_grpo_smoke_20260327-214047.launch.log](/root/rl/verl/logs/verl_grpo_smoke_20260327-214047.launch.log)
  - 训练日志：[verl_grpo_smoke_20260327-214047.train.log](/root/rl/verl/logs/verl_grpo_smoke_20260327-214047.train.log)
- 当前 train log 已确认：
  - 配置校验通过
  - `validation_diagnostics.max_samples: 64` 已成功出现在运行配置中
  - 数据集加载完成
  - placement group 已 ready
  - actor/ref worker 正在初始化模型
- 当前 GPU 显存已从空闲上升到约 `6 GB / GPU`，说明 worker/model 初始化在继续推进；截至本记录还未看到新的异常栈，也尚未到 `validation generation end` 或第一步训练指标。

---
## 2026-03-27 记录 15：持续监控直到训练进度条开始显示时长信息

实际改动：
- 本轮未修改训练逻辑；重点是持续监控当前 smoke run 的 driver 日志、CPU/GPU 状态和 validation 产物。

修改的文件：
- `PLAN.md`
- `RESULT.md`

执行的命令或检查：
- `tail -n ... /tmp/ray/session_latest/logs/job-driver-raysubmit_67zT2ZwLprkZvxzx.log`
- `rg -n "Training Progress|validation generation end|Processing validation metrics|Dumped generations|Training from scratch" /tmp/ray/session_latest/logs/job-driver-raysubmit_67zT2ZwLprkZvxzx.log`
- `stat -c '%y %s %n' ...job-driver... validation/0.jsonl`
- `ps -eo pid,ppid,etime,%cpu,%mem,stat,cmd | rg '43914|44178|44179|TaskRunner|WorkerDict'`
- `top -b -n 1 -H -p 43914`
- `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`

结果与剩余注意事项：
- 当前 smoke run 的关键路径已经明确走到了这一步：
  1. `Training from scratch`
  2. `validation generation end`
  3. `Dumped generations to .../validation/0.jsonl`
  4. `Processing validation metrics: samples=2000, uids=500, vars=35`
  5. `Training Progress: 0/37 [00:00<?, ?it/s]`
- 这说明它已经成功跨过了 initial validation 和 metrics 聚合，并正式进入训练阶段。
- 在 `Processing validation metrics` 阶段时，现场特征是：
  - `TaskRunner` 单核接近 `100% CPU`
  - GPU 利用率接近 `0`
  - `validation/0.jsonl` 已落盘
  这与此前判断一致，说明那一段确实是 CPU 聚合而不是死锁。
- 到最新一次检查时，`ray job status` 仍是 `RUNNING`，并且 GPU 又重新升到了高占用（约 `89% / 98%`），显存约 `12.6 GB x 2`，说明当前已经在做第一步训练相关计算。
- 不过截至本记录，tqdm 进度条还只显示了：
  - `Training Progress: 0/37 [00:00<?, ?it/s]`
  也就是第一步尚未完成，因此还没有出现可直接读取的剩余时间/ETA。
- 当前最保守的解释是：这条 smoke 已经进入第一个训练 step，但因为首步通常最慢（图编译、缓存、初始化开销更集中），所以 ETA 还没来得及稳定显示。

补充判断：关于 `3.0/32.0 CPU`
- `ray status` 里看到的 `3.0/32.0 CPU`，更接近“Ray 资源预留/占用记账”，不是说这条任务理论上应该把 32 个 CPU 核都打满。
- 当前现场更像是：
  - `TaskRunner` 约 1 个核
  - 两个 `WorkerDict` 各接近 1 个核
  - 合计约 3 个核左右
- 在训练 step 阶段同时观察到：
  - `ray status`: `3.0/32.0 CPU`
  - `nvidia-smi`: 2 张 GPU 满载
  - `top`: 两个 `WorkerDict` 约 `90%+ CPU`
- 这说明当前并不是“CPU 给少了导致跑不动”，而是这个阶段本来就主要由 GPU 主导，CPU 只需要少量线程喂数据和调度。
- 真正曾经明显吃 CPU 的阶段，是 validation metrics 聚合；那一段主要是 `TaskRunner` 单核接近 `100%`，属于实现本身偏单线程。单纯把可用 CPU 核数再加大，对这一段帮助也有限，除非继续把聚合逻辑并行化。

---
## 2026-03-27 记录 16：按用户要求终止当前训练

实际改动：
- 按用户要求停止当前 smoke run。
- 先尝试通过 `ray job stop` 终止 `raysubmit_67zT2ZwLprkZvxzx`。
- 由于 Ray dashboard 已不可达（`Connection refused`），转而用本机进程、GPU 进程和 screen 会话视角确认训练是否已退出。

修改的文件：
- `PLAN.md`
- `RESULT.md`

执行的命令或检查：
- `conda run -n verl ray job stop --address=http://192.168.122.2:8265 raysubmit_67zT2ZwLprkZvxzx`
- `ps -eo ... | rg 'TaskRunner|WorkerDict|raysubmit_67zT2ZwLprkZvxzx'`
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`
- `screen -S verl_grpo_smoke_20260327-214047 -X quit`
- `screen -S verl_grpo_smoke_20260327-210328 -X quit`
- `screen -S verl_grpo_smoke_20260327-205828 -X quit`
- `screen -wipe`
- `screen -ls`

结果与剩余注意事项：
- `ray job stop` 没成功执行，不是因为权限问题，而是因为 Ray dashboard 当时已经无法连接。
- 但从本机状态看，当前训练本体已经不在了：
  - `ps` 中已没有对应的 `TaskRunner/WorkerDict`
  - `nvidia-smi --query-compute-apps` 为空
  - 所有相关 `screen` 会话已清理干净
- 所以最终状态是：当前训练已停止，机器上没有残留的 GPU 训练进程。

---
## 2026-03-27 记录 17：把每次运行都固化成单独的 sh 文件

实际改动：
- 新建了固定目录：
  - [run_jobs/README.md](/root/rl/verl/examples/custom/run_jobs/README.md)
- 更新两个主训练脚本：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
  - [run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh)
- 现在每次运行这两个脚本时，都会自动先生成：
  - `examples/custom/run_jobs/<exp_name>.sh`
- 生成的 `sh` 文件里会保存本次运行的关键环境变量，并在最后直接调用对应主脚本。
- 为了避免你将来直接执行这些生成的 `sh` 文件时再次重复生成自身，文件里会自动加：
  - `export SAVE_RUN_SCRIPT=false`

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/TRAINING_PIPELINE_1SHOT_GRPO_LOGITS.md`
- `examples/custom/run_jobs/README.md`

执行的命令或检查：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `find examples/custom/run_jobs -maxdepth 2 -type f | sort`

结果与剩余注意事项：
- 现在运行逻辑已经更简单了：后面你不用再到处找命令，直接看 `examples/custom/run_jobs/` 就能找到每次运行对应的独立脚本。
- 这次还没有实际重新起实验，所以目录里目前只有说明文件；等你下次真正运行时，就会自动多出一份 `<exp_name>.sh`。

---
## 2026-03-27 记录 18：预置四组对照实验的独立运行脚本

实际改动：
- 在 `examples/custom/run_jobs/` 里新增了 4 个可直接执行的对照实验脚本：
  - [qwen_math_25_15B_instruct_oneshot.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh)
  - [qwen_math_25_15B_instruct_full.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh)
  - [qwen_math_25_15B_base_oneshot.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh)
  - [qwen_math_25_15B_base_full.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_full.sh)
- 这 4 个脚本统一使用基础 GRPO 主脚本：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
- 每个脚本都已经内置：
  - `conda activate verl`
  - 模型路径
  - `TRAIN_MODE`
  - 自动 `EXP_NAME`
  - 正式实验默认参数
  - `SAVE_RUN_SCRIPT=false`

修改的文件：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_jobs/README.md`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

执行的命令或检查：
- `chmod +x examples/custom/run_jobs/qwen_math_25_15B_*.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`
- `ls -l examples/custom/run_jobs`

结果与剩余注意事项：
- 这四个脚本现在已经是可执行状态，你可以直接 `bash <script>` 或 `./<script>` 跑。
- 当前默认把 `VAL_DIAG_MAX_SAMPLES` 设成 `64`，这是为了让对照实验更稳地先跑起来；如果你后面要做更完整的 logits/diagnostics 分析，可以在运行前临时改成：
  - `export VAL_DIAG_MAX_SAMPLES=-1`

---
## 2026-03-27 记录 19：让运行脚本默认接入 SwanLab

实际改动：
- 更新了两个主训练入口：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
  - [run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh)
- 这两个脚本现在启动时会先执行：
  - `source /root/.bashrc >/dev/null 2>&1 || true`
- 另外补了一层兼容处理：
  - 由于 `/root/.bashrc` 第 6 行有 ` [ -z "$PS1" ] && return`，非交互 shell 直接 `source ~/.bashrc` 时拿不到后面的 `SWANLAB_API_KEY`
  - 所以主训练脚本现在会在环境里还没有 `SWANLAB_API_KEY` 时，额外从 `/root/.bashrc` 中提取这一项并导出
- 同时也更新了它们内部自动生成 `run_jobs/<exp_name>.sh` 的模板，让后续新生成的运行脚本也会先读取 `~/.bashrc`。
- 更新了当前四个固定实验脚本：
  - [qwen_math_25_15B_instruct_oneshot.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh)
  - [qwen_math_25_15B_instruct_full.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh)
  - [qwen_math_25_15B_base_oneshot.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh)
  - [qwen_math_25_15B_base_full.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_full.sh)
- 这四个脚本现在默认会设置：
  - `LOGGER=${LOGGER:-'["swanlab","file"]'}`

执行的命令或检查：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

结果与剩余注意事项：
- 只要 `SWANLAB_API_KEY` 已经写在 `/root/.bashrc` 里，这 4 个脚本现在直接运行就会把日志同时写到 SwanLab 和本地文件。
- 后面如果你通过主训练脚本自动生成新的 `run_jobs/<exp_name>.sh`，这些新脚本也会继承同样的行为。

---
## 2026-03-27 记录 20：修复运行脚本无输出直接退出的问题

实际定位：
- 复现命令：
  - `bash -x /root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`
- 复现结果显示脚本停在：
  - `source /root/.bashrc`
- 根因不是训练本体，而是 shell 初始化阶段：
  - 这些 `run_jobs/*.sh` 脚本使用了 `set -euo pipefail`
  - `/root/.bashrc` 第 6 行会访问 `PS1`
  - 在非交互脚本里，`PS1` 通常未定义，和 `set -u` 组合后会直接退出
- 所以你看到的现象才会是：
  - 执行脚本后没有任何输出
  - 立刻回到 shell 提示符

实际改动：
- 去掉了这 4 个运行脚本里的 `source /root/.bashrc`：
  - [qwen_math_25_15B_instruct_oneshot.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh)
  - [qwen_math_25_15B_instruct_full.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh)
  - [qwen_math_25_15B_base_oneshot.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh)
  - [qwen_math_25_15B_base_full.sh](/root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_full.sh)
- 去掉了两个主训练脚本及其自动生成模板里的 `source /root/.bashrc`：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
  - [run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh)
- 保留了更安全的 SwanLab 兜底方式：
  - 主训练脚本如果发现当前环境里没有 `SWANLAB_API_KEY`，会直接从 `/root/.bashrc` 提取这一项并导出
  - 这样既不会触发 `PS1`/`set -u` 冲突，也还能正常登录 SwanLab

执行的命令或检查：
- `bash -x /root/rl/verl/examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

结果与剩余注意事项：
- 现在这个“无输出直接返回”的问题已经被修正。
- 后续如果脚本再次立即退出，优先用 `bash -x <script>` 看执行停在哪一行，定位会很快。

---
## 2026-03-27 记录 21：统计训练集里超过 1024 prompt tokens 的样本数

实际定位与统计口径：
- 先检查了训练代码中的过滤逻辑：
  - [rl_dataset.py](/root/rl/verl/verl/utils/dataset/rl_dataset.py)
- 当前过滤不是看字符数，而是看：
  - `tokenizer.apply_chat_template(doc["prompt"], add_generation_prompt=True)`
  之后的 token 数
- 当前 `full` 训练脚本使用的模型 tokenizer 是：
  - `/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B`

执行的命令或检查：
- 在 `verl` conda 环境中读取：
  - `/root/rl/verl/data/dsr_sub/train.parquet`
  - `/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet`
- 使用与训练一致的 `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` 和 `apply_chat_template(..., add_generation_prompt=True)` 逐条统计长度

统计结果：
- `full`:
  - 总样本数：`1209`
  - 超过 `1024` tokens 的样本数：`0`
  - 占比：`0.0000%`
  - 最大 prompt 长度：`857`
  - 平均 prompt 长度：`123.55`
  - `p95`：`229`
- `oneshot`:
  - 总样本数：`1209`
  - 超过 `1024` tokens 的样本数：`0`
  - 占比：`0.0000%`
  - 最大 prompt 长度：`140`
  - 平均 prompt 长度：`140.00`
  - `p95`：`140`

结果与剩余注意事项：
- 结论很明确：按当前数据和当前 tokenizer，`filter_overlong_prompts=true` 不会过滤掉任何训练样本。
- 所以至少在这批 `full/oneshot` 数据上，`max_prompt_length=1024` 不会造成训练样本损失，也不会影响你的对照结论。

---
## 2026-03-27 记录 22：验证更新后的 SwanLab API Key 是否可用

实际检查：
- 当前 `/root/.bashrc` 中的 key 已更新为：
  - [/root/.bashrc:121](/root/.bashrc:121)
- 在 `verl` conda 环境中执行了独立登录测试：
  - 从 `/root/.bashrc` 读取最新 `SWANLAB_API_KEY`
  - 调用一次 `swanlab.login(key)`

执行的命令或检查：
- `nl -ba /root/.bashrc | tail -n 12`
- 在 `conda activate verl` 环境中运行最小化 Python 登录脚本

结果：
- `SWANLAB_LOGIN_OK`

结果与剩余注意事项：
- 现在可以确认：新 API key 本身是可用的。
- 后续如果训练里再出现 SwanLab 相关报错，优先怀疑运行环境、网络或 logger 配置，而不是 key 失效。

---
## 2026-03-27 记录 23：修复 Ray runtime_env 覆盖新 SwanLab key 的问题

实际定位：
- 问题不在 SwanLab 本身，而在 Ray job 的 runtime env：
  - [runtime_env.yaml](/root/rl/verl/verl/trainer/runtime_env.yaml)
- 这个文件里原来硬编码了：
  - `WANDB_API_KEY`
  - `SWANLAB_API_KEY`
- 其中 `SWANLAB_API_KEY` 还是旧值，所以即使你已经更新了 `/root/.bashrc`，Ray job 启动后仍然会被 runtime env 里的旧 key 覆盖。
- 另外，主训练脚本此前只有在当前环境变量为空时才会去读 `/root/.bashrc`，这意味着如果 shell 里残留旧 key，也可能继续带着旧值提交 job。

实际改动：
- 清理了仓库里的静态密钥：
  - [runtime_env.yaml](/root/rl/verl/verl/trainer/runtime_env.yaml)
- 更新了两个主训练脚本：
  - [run_qwen_math_25_15B_grpo_1_shot.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh)
  - [run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh](/root/rl/verl/examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh)
- 新逻辑是：
  - 运行开始时优先读取 `/root/.bashrc` 中最新的 `SWANLAB_API_KEY`
  - 提交 Ray job 前，动态生成 `/tmp/verl_runtime_env_<exp>.yaml`
  - 把当前有效的 `SWANLAB_API_KEY` / `WANDB_API_KEY` 注入这个临时 runtime env
  - `ray job submit` 使用这个临时 runtime env，而不再直接用仓库里静态的旧密钥文件
- 还顺手修了一点安全性：
  - 在处理密钥时暂时关闭 `set -x`，避免把 API key 打进 shell trace

执行的命令或检查：
- `cat verl/trainer/runtime_env.yaml`
- `rg -n "SWANLAB_API_KEY|trainer.logger|ray job submit|runtime-env" ...`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

结果与剩余注意事项：
- 这次“明明 key 已更新，但训练里仍然报旧 key 无效”的根因已经被修正。
- 之后再提 job，Ray 侧拿到的应该就是你当前 `bashrc` 里的新 key，而不是 repo 里历史遗留的旧 key。

---
## 2026-03-27 记录 24：修复 validation diagnostics 在分布张量计算时的 OOM

实际定位：
- 当前 OOM 发生在 validation diagnostics 的分布张量计算路径：
  - [dp_actor.py](/root/rl/verl/verl/workers/actor/dp_actor.py)
- 栈里最关键的是：
  - `logits_fp32 = logits.to(torch.float32)`
  - `log_norm = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)`
- 这意味着代码会把整块 `logits[..., vocab]` 一次性升到 `fp32`
- 对 Qwen 这类大词表模型，这一步会瞬间申请一大块额外显存，所以即使模型前向本身已经勉强放得下，也会在 diagnostics 这里炸掉。

实际改动：
- 更新了 [dp_actor.py](/root/rl/verl/verl/workers/actor/dp_actor.py)
- `DataParallelPPOActor._compute_distribution_tensors_from_logits(...)` 现在改成按 rows 分块计算，默认 `chunk_size=512`
- 分块计算的内容仍然保持不变：
  - `dist_top1_token_ids`
  - `dist_top1_token_probs`
  - `dist_eos_probs`
  - `dist_topk_token_ids`
  - `dist_topk_token_probs`
  - `dist_topk_mass`
- 这样做的效果是：
  - 不再一次性把整块 logits 升到 fp32
  - 显存峰值显著下降
  - diagnostics 的定义不变

执行的命令或检查：
- `python -m py_compile verl/workers/actor/dp_actor.py`
- 在 `verl` conda 环境中构造随机 logits，对比旧实现和新实现

验证结果：
- `dist_eos_probs max_abs_diff 0.0`
- `dist_top1_token_ids exact_match True`
- `dist_top1_token_probs max_abs_diff 0.0`
- `dist_topk_mass max_abs_diff 0.0`
- `dist_topk_token_ids exact_match True`
- `dist_topk_token_probs max_abs_diff 0.0`

结果与剩余注意事项：
- 这次 OOM 的根因已经定位并修复为更省显存的实现。
- 旧的失败 run 不会自动恢复；需要基于当前代码重新提交一次训练。
- 如果后面仍然接近显存边界，优先再降这些 validation 参数：
  - `VAL_ROLLOUT_N`
  - `VAL_DIAG_MAX_SAMPLES`
  - `VAL_DIAG_DISTRIBUTION_TOPK`

---
## 2026-03-28 记录 25：解释 aux diagnostics 上的 best@k / worst@k / pass@k 含义

实际定位：
- `rollout_logprob_delta_abs_mean` 这类 aux diagnostics 的原始样本级定义在：
  - [ray_trainer.py](/root/rl/verl/verl/trainer/ppo/ray_trainer.py)
- 对于每个 response，它记录的是：
  - 当前 policy 的 token logprob 和 rollout 时 token logprob 的平均绝对差
- 也就是：
  - 值越小：当前 policy 与 rollout 时更一致
  - 值越大：差异越大

聚合逻辑来源：
- 所有 validation scalar 指标都会进入统一聚合：
  - [metric_utils.py](/root/rl/verl/verl/trainer/ppo/metric_utils.py)
- 这个聚合器会按 `uid` 把同一道题的多条 response 放在一起，然后统一计算：
  - `mean@N`
  - `std@N`
  - `best@N`
  - `worst@N`
  - `pass@N`（仅当它把该指标识别成“近似二值/正确率型”时）

关键解释：
- 这里的 `best@N` / `worst@N` 是通用模板，不知道每个指标的“优化方向”
- 在代码里：
  - `best@N` 实际就是 `max`
  - `worst@N` 实际就是 `min`
- 所以对 `rollout_logprob_delta_abs_mean` 这种“越小越好”的指标来说：
  - `best@4` 实际上是 4 条 response 里最大的那个值
  - `worst@4` 实际上是 4 条 response 里最小的那个值
- 也就是说，这两个名字对这种连续诊断量并不语义自洽，更准确的理解应该是：
  - `best@4` = `max@4`
  - `worst@4` = `min@4`

关于 `pass@k`：
- `pass@k` 在代码里只适合 binary-like 指标
- 但它的自动识别规则是启发式的：
  - `{0,1}`
  - `{-1,1}`
  - 或者数值刚好落在 `[0,1]` 时用 `>0.9` 当作“correct”
- 所以如果某个连续 aux metric 恰好数值范围落在 `[0,1]`，也可能被错误地算出 `pass@k`
- 这种 `pass@k` 对 diagnostics 通常没有稳定解释，不建议在 report 里重点使用

结果与建议：
- 对连续 diagnostics（例如 `rollout_logprob_delta_abs_mean`、`eos_prob_mean`、`topk_mass_mean`），最推荐看：
  - `mean@4`
  - 必要时再看 `best@4` / `worst@4` 作为 `max/min spread`
- 对 `pass@k`，建议主要只在：
  - `acc`
  - 明确二值正确性指标
  上使用

---
## 2026-03-28 记录 26：收窄 validation 聚合规则以减少 aux diagnostics 开销

实际改动：
- 更新了 [metric_utils.py](/root/rl/verl/verl/trainer/ppo/metric_utils.py)
- `process_validation_metrics()` 现在按变量类型分两类处理：
  - core vars：`acc`、`reward`
    - 保留 `mean/std`
    - 保留 `best/worst`
    - 保留 `pass`
  - aux diagnostics：
    - 只保留 `mean/std`
- 这意味着像：
  - `rollout_logprob_delta_abs_mean`
  - `eos_prob_mean`
  - `topk_mass_mean`
  这类指标后面不再生成 `best@k / worst@k / pass@k`

测试改动：
- 更新了 [test_metric_utils_on_cpu.py](/root/rl/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py)
- 新测试口径覆盖了两类情况：
  - aux metric 不再产生 `best/worst/pass`
  - `acc` 仍然保留 richer aggregation

执行的命令或检查：
- `python -m py_compile verl/trainer/ppo/metric_utils.py tests/trainer/ppo/test_metric_utils_on_cpu.py`
- `python -m unittest tests.trainer.ppo.test_metric_utils_on_cpu.TestProcessValidationMetrics -v`

验证结果：
- `TestProcessValidationMetrics` 目标测试通过：
  - `test_process_validation_metrics_basic ... ok`
  - `test_process_validation_metrics_core_metric ... ok`

额外说明：
- 全文件 `python tests/trainer/ppo/test_metric_utils_on_cpu.py` 里还有一个与本次改动无关的旧失败：
  - `TestComputeDataMetrics.test_compute_data_metrics_with_critic`
  - 报的是 `critic/rewards/mean` 断言不一致
- 这个失败不是本次 validation 聚合规则调整引入的。
