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

---
## 2026-03-27 记录 5：本地 Smoke 训练排查、日志落盘与基础 GRPO 收敛

目标：
- 先把本地 `conda` 环境中的 smoke 训练链路跑通，并确保训练/提交日志稳定落盘，便于持续排查。
- 在不大改训练结构的前提下，收敛掉当前 `recipe.entropy.main_entropy` 路径上的 Hydra 参数兼容问题。
- 根据最新排查结果，为后续切回更基础的 `GRPO` 入口做准备，优先移除会引起结构不兼容的 `seed` 覆盖。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 视排查结果可能会调整 `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`

实现方案：
- 为训练脚本增加可显式指定的 `LOG_FILE`，让 `ray job logs --follow` 可以稳定写入指定文件。
- 在本机自动启动 Ray 时，把 dashboard 地址解析到真实可访问的本地 IP，避免后续 `job logs` / `job status` 指向错误地址。
- 用 `screen` 持久会话启动 smoke 训练，并保留 launch log 与 train log 两份文件，区分“提交阶段”与“作业运行阶段”的问题。
- 对 `recipe.entropy.main_entropy` 路径上不兼容的 Hydra 覆盖逐项收敛，优先删除会改变配置结构或 dataclass 构造行为的 `seed` 字段。
- 如 entropy 变体仍持续暴露兼容问题，则转向更基础的 `verl.trainer.main_ppo` 路径验证最小 GRPO 能力。

验证计划：
- 对更新后的 shell 脚本执行语法检查。
- 在 `conda` 的 `verl` 环境中启动至少一轮 smoke 训练，确认 `screen` 会话、launch log、train log 都能生成。
- 检查训练日志中的报错栈，确认后续每次修复都能把失败点向前推进，而不是停留在同一配置错误。

---
## 2026-03-27 记录 6：移除 Seed 干扰并切回基础 GRPO smoke

目标：
- 去掉训练入口里的 `seed` 覆盖，避免继续触发 Hydra struct 或 dataclass 实例化兼容问题。
- 切回更基础的 `verl.trainer.main_ppo` GRPO 路径，优先验证最小 smoke 训练能够稳定进入运行态。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- 视需要新增基础 GRPO 的 smoke 脚本

实现方案：
- 为基础 GRPO 脚本补充与前序排查一致的日志落盘和 dashboard 地址修正能力。
- 新增基础 GRPO smoke 启动脚本，用小 batch / 小 rollout / 单 epoch 快速验证链路。
- 使用 `screen` 持久会话重跑，并观察日志是否越过作业提交、数据加载与 worker 初始化，进入正常训练流程。

验证计划：
- 对新增/更新的 shell 脚本执行语法检查。
- 用 `screen` 启动基础 GRPO smoke，并检查 launch log、train log 与相关进程状态。

---
## 2026-03-27 记录 7：基础 GRPO smoke 重跑并确认进入正常运行阶段

目标：
- 继续重跑最基础的 GRPO smoke 实验，确认任务不再停留在 Hydra 配置错误，而是稳定进入 `main_ppo` 主流程。
- 把当前 `screen` 会话、Ray job、训练日志与 GPU 占用状态固化到跟踪文档里，便于后续判断是“初始化中”还是“真正异常卡住”。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 复核当前 `screen` 会话中的 launch log、train log、Ray job status 和主进程状态，确认作业仍处于 `RUNNING`。
- 观察训练日志是否已经越过数据过滤、placement group 准备、配置校验等早期阶段，并进入 worker 模型初始化。
- 结合 `nvidia-smi` 与相关进程 CPU/显存占用，判断当前属于正常初始化等待，而不是作业失活或反复重启。

验证计划：
- 检查 `logs/verl_grpo_smoke_20260327-205828.launch.log` 与 `logs/verl_grpo_smoke_20260327-205828.train.log` 的最新输出。
- 检查 `ray job status raysubmit_xfneqvpJjxsppDDV` 是否仍为 `RUNNING`。
- 检查 `nvidia-smi` 和 `ps`，确认 `main_ppo` 与 `actor_rollout_init_model` worker 持续存在并占用资源。

---
## 2026-03-27 记录 8：修复 SwanLab 登录阻塞并重新提交基础 GRPO smoke

目标：
- 去掉当前基础 smoke 训练被 `swanlab.error.ValidationError: Login failed: Error api key` 阻塞的问题。
- 让训练脚本在没有有效 SwanLab 凭证时默认退回 `file` logger，优先保证本地训练与日志落盘链路可运行。
- 基于修复后的脚本重新提交基础 GRPO smoke，并确认新的 run 越过上一轮失败点。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

实现方案：
- 移除脚本顶部硬编码的 `SWANLAB_API_KEY`，避免脚本默认携带失效凭证。
- 调整 logger 选择逻辑：若用户显式设置 `LOGGER` 则尊重用户输入；否则仅在外部环境存在 `SWANLAB_API_KEY` 时启用 `["swanlab","file"]`，没有时默认 `["file"]`。
- 重新用 `screen` 提交基础 GRPO smoke，显式清掉外部 `SWANLAB_API_KEY` / `WANDB_API_KEY`，并观察新的 job 状态和训练日志。

验证计划：
- 对更新后的 shell 脚本执行 `bash -n` 语法检查。
- 检查新的 launch log 中是否出现 `trainer.logger=["file"]`。
- 检查新的 `ray job status` 是否为 `RUNNING`，并确认 train log 中不再出现 SwanLab 登录错误。

---
## 2026-03-27 记录 9：持续监控基础 GRPO smoke 并确认产出首个 validation 文件

目标：
- 持续确认基础 GRPO smoke 不只是停留在 worker 初始化，而是已经开始真实产出 validation 结果。
- 用首个 validation 文件作为“训练链路已跑通到产出阶段”的标记点。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 持续检查 train log、Ray job 状态、GPU 显存与 checkpoint 目录内容。
- 一旦发现 `validation/*.jsonl` 落盘，记录对应路径并把它作为本轮 smoke 已经跑通的直接证据。

验证计划：
- 检查 `logs/verl_grpo_smoke_20260327-210328.train.log` 最新输出。
- 检查 `ray job status --address=http://192.168.122.2:8265 raysubmit_r5jtfU92xsbadCMU`。
- 检查 `ckpts/verl_grpo_dsr_sub_baseline/smoke_qwen_math_25_15B_instruct_grpo_20260327-210328/validation/` 下是否已有 `jsonl` 文件。

---
## 2026-03-27 记录 10：继续监控直到训练开始，并判断是否停在 validation 之后

目标：
- 持续监控当前 smoke run，直到出现第一条明确的训练 step / metric 输出。
- 如果长时间未进入训练 step，则判断任务是否停在 `val_before_train` 之后的某个准备阶段。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 持续检查 Ray job 实时日志、job-driver 日志、worker 错误日志、validation 文件时间戳和相关进程状态。
- 结合 `TaskRunner` / `WorkerDict` 的 CPU 使用率、GPU 利用率和日志更新时间，判断是否出现“进程存活但训练未真正开始”的状态。
- 如确认长时间没有第一条训练 metric，则把当前结论记录为“基础链路跑到 validation 成功，但实际训练起步前仍可能存在卡点”。

验证计划：
- 检查 `ray job logs --address=http://192.168.122.2:8265 raysubmit_r5jtfU92xsbadCMU` 最新输出。
- 检查 `/tmp/ray/session_latest/logs/job-driver-raysubmit_r5jtfU92xsbadCMU.log` 的时间戳和尾部内容。
- 检查 `ps`、`top`、`nvidia-smi` 和 `validation/0.jsonl` 的时间戳是否继续推进。

---
## 2026-03-27 记录 11：定位 validation 后慢点并修复训练阶段返回值解包错误

目标：
- 理清当前 run 为什么会在 validation 之后长时间看起来“不动”。
- 判断用户新增 metrics 是否放大了 validation 收尾阶段的 CPU 开销。
- 修复 run 真正进入训练阶段后暴露出来的 actor 更新报错。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/actor/dp_actor.py`

实现方案：
- 对 `_validate()` 的后半段代码路径做静态阅读，重点查看 `_dump_generations()` 之后到 `return metric_dict` 之前的逻辑。
- 结合 `validation/0.jsonl` 的样本规模和字段数，估算 `process_validation_metrics()` 的复杂度，并用小样本基准验证 CPU 开销。
- 对训练报错 `ValueError: too many values to unpack (expected 2)` 做调用链排查，确认是否由 `_forward_micro_batch()` 返回值结构变化引起。
- 做最小修复：让训练路径正确忽略第三个返回值；并给 validation metric 聚合增加开始/结束耗时打印，方便后续直接观察慢点位置。

验证计划：
- 对 `verl/workers/actor/dp_actor.py` 和 `verl/trainer/ppo/ray_trainer.py` 做语法检查。
- 检查最新 `job-driver` 日志，确认 validation 卡点与训练报错都能在日志中对应到具体代码路径。
- 用 `validation/0.jsonl` 做规模估算，确认 metrics 聚合开销与现场现象一致。

---
## 2026-03-27 记录 12：把 validation diagnostics 轻量化到仅分析部分 val 样本

目标：
- 在不改动核心 GRPO 训练/验证结构的前提下，降低 validation diagnostics 的额外开销。
- 保留全量 val 的核心 `reward/acc` 统计，只把你新增的 token/logit diagnostics 限制到少量 val 样本上。
- 让这套轻量化行为能通过脚本环境变量控制，方便后续切回全量分析。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/ppo/ray_trainer.py`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

实现方案：
- 在 trainer 中新增 `trainer.validation_diagnostics.max_samples` 配置，表示每次 validation 最多对多少个样本执行额外 diagnostics。
- 把限制前移到 `compute_log_prob()` 之前：只对前 `N` 个 val 样本做 diagnostics 额外前向，而不是先全量前向、后面再裁掉结果。
- 保持全量 reward function 评估不变，只在 `batch_diag_infos` 与 `sample_metadata` 中对未分析的样本填空值。
- 在 train log 中打印本轮 validation diagnostics 的 sample budget 和实际分析样本数，便于后续核对是否生效。
- 在两个启动脚本里暴露 `VAL_DIAG_MAX_SAMPLES` 环境变量，并先给一个较轻的默认值。

验证计划：
- `python3 -m py_compile verl/trainer/ppo/ray_trainer.py`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 后续重跑时观察 train log 中是否出现 `Validation diagnostics sample budget` 与 `Validation diagnostics analyzed ... samples`。

---
## 2026-03-27 记录 13：回看上一次 smoke run 的最终失败原因

目标：
- 确认上一条基础 GRPO smoke run 是否已经失败。
- 区分“旧 run 仍在执行”与“run 已失败，但当前工作区代码已经修复”的状态。
- 给出下一步是否需要直接重跑的明确结论。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 读取最近一次 smoke 训练日志和对应的 Ray job driver 尾部日志。
- 提取最后的异常栈，确认失败位置与报错类型。
- 再对照当前工作区中的 `dp_actor.py` 修复状态和文件时间戳，判断这次失败是否发生在补丁之前。

验证计划：
- 检查 `logs/verl_grpo_smoke_20260327-210328.train.log` 尾部。
- 检查 `/tmp/ray/session_latest/logs/job-driver-raysubmit_r5jtfU92xsbadCMU.log` 尾部。
- 检查 [dp_actor.py](/root/rl/verl/verl/workers/actor/dp_actor.py) 当前第 554 行附近的代码。

---
## 2026-03-27 记录 14：重启基础 GRPO smoke，并修复 Hydra 对新增 diagnostics 键的覆盖方式

目标：
- 用当前已修补的代码重新拉起一条基础 GRPO smoke run。
- 使用 `screen` 和文件日志，让训练过程可持续跟踪。
- 若启动阶段再次失败，优先修复启动层/Hydra 覆盖问题，再继续重提。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

实现方案：
- 通过 `screen` 启动新的 smoke run，环境固定为 `conda activate verl`。
- 使用 `LOG_FILE` 把 Ray job follow 日志写入 `logs/verl_grpo_smoke_<timestamp>.train.log`。
- 启动参数中带上 `ENABLE_PAPER_STYLE_VIZ=false` 与 `VAL_DIAG_MAX_SAMPLES=64`，先用更轻的 smoke 验证主链路。
- 如果 Hydra 报 `Key ... is not in struct`，则把 `trainer.validation_diagnostics.max_samples=...` 改为 `++trainer.validation_diagnostics.max_samples=...`，兼容新增配置键。
- 重提作业后持续检查：job 状态、train log、GPU 显存、是否进入 validation/训练阶段。

验证计划：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 检查新的 `screen` 会话、launch log、train log 与 `ray job status`。

---
## 2026-03-27 记录 15：持续监控直到训练进度条开始显示时长信息

目标：
- 持续监控当前 smoke run，直到日志从 validation 聚合阶段进入真正的训练 step。
- 捕捉 `Training Progress` 的首次出现，并尽量拿到可用于估算总耗时的速度/时长信息。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 优先查看 Ray `job-driver` 日志，而不是只看本地 `train.log`，因为 driver 日志刷新更及时。
- 结合 `validation/0.jsonl` 落盘时间、`Processing validation metrics`、`Training Progress` 等关键行，判断 run 当前位于 validation 聚合还是训练 step。
- 配合 `ps/top/nvidia-smi` 观察：
  - GPU 高占用 -> 训练 step 或生成在跑
  - GPU 低占用 + TaskRunner 高 CPU -> validation metrics 聚合在跑

验证计划：
- 检查 `/tmp/ray/session_latest/logs/job-driver-raysubmit_67zT2ZwLprkZvxzx.log`
- 检查 `ray job status --address=http://192.168.122.2:8265 raysubmit_67zT2ZwLprkZvxzx`
- 检查 `nvidia-smi` 和 `top -H -p 43914`

---
## 2026-03-27 记录 16：按用户要求终止当前训练

目标：
- 立即停止当前正在运行的 smoke run。
- 确认 GPU 计算进程和对应的 screen 会话都已经清理掉。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 优先尝试通过 `ray job stop` 停止当前提交的 job。
- 如果 Ray dashboard 已不可达，则回退到本机进程与 GPU 视角确认训练是否已自然退出。
- 清理仍在残留的 `screen` 会话，避免后续误判。

验证计划：
- 检查 `nvidia-smi --query-compute-apps=...`
- 检查 `ps ... | rg 'TaskRunner|WorkerDict'`
- 检查 `screen -ls`

---
## 2026-03-27 记录 17：把每次运行都固化成单独的 sh 文件

目标：
- 简化运行方式，不再依赖散落的历史命令。
- 准备一个固定文件夹，每次运行前自动生成一个独立的 `sh` 文件，方便复跑和追溯。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/TRAINING_PIPELINE_1SHOT_GRPO_LOGITS.md`
- `examples/custom/run_jobs/README.md`

实现方案：
- 新建固定目录：`examples/custom/run_jobs/`
- 在两个主训练脚本里加入自动落盘逻辑：
  - 每次运行前，先生成 `examples/custom/run_jobs/<exp_name>.sh`
  - 文件内保存本次运行的关键环境变量
  - 文件最后直接调用对应主脚本
- 为避免复跑时无限重复生成，自动写入 `SAVE_RUN_SCRIPT=false`
- 在 pipeline 文档中补一句，后面统一从 `examples/custom/run_jobs/` 找每次运行的脚本

验证计划：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 检查 `examples/custom/run_jobs/` 目录是否存在并包含说明文件

---
## 2026-03-27 记录 18：预置四组对照实验的独立运行脚本

目标：
- 直接给出 `2 models x 2 train modes` 的四个独立 `sh` 文件。
- 让用户不需要再拼环境变量，直接执行对应脚本即可。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_jobs/README.md`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

实现方案：
- 基于基础 GRPO 主脚本 `run_qwen_math_25_15B_grpo_1_shot.sh` 预置四个独立的入口脚本。
- 统一在脚本里完成：
  - `cd /root/rl/verl`
  - `conda activate verl`
  - 固定 `TRAIN_MODE`
  - 固定 `MODEL_PATH`
  - 自动生成 `EXP_NAME`
  - 设定一组正式实验默认值（epochs / save_freq / test_freq）
- 为避免重复自动落盘，再统一设置 `SAVE_RUN_SCRIPT=false`

验证计划：
- 给四个脚本加执行权限
- `bash -n` 检查四个脚本
- 检查 `examples/custom/run_jobs/` 目录下文件列表

---
## 2026-03-27 记录 19：让运行脚本默认接入 SwanLab

目标：
- 让现在这 4 个对照实验脚本能直接读到 `~/.bashrc` 里的 `SWANLAB_API_KEY`。
- 让后续自动生成的 `run_jobs/<exp_name>.sh` 也默认继承相同行为。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

实现方案：
- 在两个主训练脚本开头先 `source /root/.bashrc`，保证非登录 shell 也能拿到 `SWANLAB_API_KEY`。
- 在自动生成的 run script 模板里也加入同样的 `source /root/.bashrc`。
- 在当前 4 个固定实验脚本里默认设置：
  - `LOGGER=${LOGGER:-'["swanlab","file"]'}`
  这样日志会同时写本地文件和 SwanLab。

验证计划：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

---
## 2026-03-27 记录 20：修复运行脚本无输出直接退出的问题

目标：
- 找到为什么执行 `run_jobs/qwen_math_25_15B_base_full.sh` 后没有任何输出就返回 shell。
- 修复启动入口，同时保留 SwanLab 自动读取 key 的能力。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

实现方案：
- 用 `bash -x` 复现入口脚本，确认脚本具体停在什么位置。
- 如果是 `source /root/.bashrc` 与 `set -u` 冲突，则去掉这一步。
- 保留主训练脚本里“直接从 `/root/.bashrc` 提取 `SWANLAB_API_KEY`”的兜底逻辑，避免重新引入静默退出。

验证计划：
- `bash -x examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_instruct_full.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_oneshot.sh`
- `bash -n examples/custom/run_jobs/qwen_math_25_15B_base_full.sh`

---
## 2026-03-27 记录 21：统计训练集里超过 1024 prompt tokens 的样本数

目标：
- 回答当前训练数据中有多少样本会被 `filter_overlong_prompts` 过滤掉。
- 按训练代码真实口径统计，而不是按字符数或粗略字段长度估计。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 先确认 `RLHFDataset.maybe_filter_out_long_prompts()` 的判定逻辑。
- 使用与训练一致的 tokenizer 和 `apply_chat_template(..., add_generation_prompt=True)` 统计 prompt token 长度。
- 统计当前 `full` 训练集，同时顺手统计 `oneshot` 训练集，方便后续对照。

验证计划：
- 检查 `verl/utils/dataset/rl_dataset.py`
- 在 `conda activate verl` 环境中读取 parquet 并统计长度分布

---
## 2026-03-27 记录 22：验证更新后的 SwanLab API Key 是否可用

目标：
- 不启动训练，单独验证 `/root/.bashrc` 中更新后的 `SWANLAB_API_KEY` 是否能成功登录 SwanLab。
- 排除“训练失败是因为 key 无效”这一层不确定性。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 读取 `/root/.bashrc` 中最新的 `SWANLAB_API_KEY`
- 在 `conda activate verl` 环境中执行一次最小 `swanlab.login(key)` 测试

验证计划：
- 检查 `/root/.bashrc` 尾部的 key 配置
- 执行一次独立登录测试

---
## 2026-03-27 记录 23：修复 Ray runtime_env 覆盖新 SwanLab key 的问题

目标：
- 找到为什么明明新 `SWANLAB_API_KEY` 已经可用，训练里仍然报旧的 `Error api key`。
- 修复 Ray job 提交链路，避免 runtime env 中的旧密钥覆盖当前环境。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/runtime_env.yaml`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

实现方案：
- 检查 `runtime_env.yaml` 是否存在硬编码的 `SWANLAB_API_KEY` / `WANDB_API_KEY`
- 若存在，移除仓库中的静态密钥
- 在主训练脚本里按运行时环境动态生成临时 `runtime_env`，把当前有效 key 注入 Ray job
- 同时优先使用 `/root/.bashrc` 中最新的 `SWANLAB_API_KEY`，避免旧 shell 环境变量残留

验证计划：
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh`
- `bash -n examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 检查 `verl/trainer/runtime_env.yaml`
- 干跑一次脚本，确认生成的 `/tmp/verl_runtime_env_<exp>.yaml` 使用的是当前 key 而不是旧 key

---
## 2026-03-27 记录 24：修复 validation diagnostics 在分布张量计算时的 OOM

目标：
- 定位为什么当前 run 在 `compute_log_prob_with_diagnostics()` 内部 OOM。
- 在尽量不改变 diagnostics 含义的前提下，把分布统计实现改成更低峰值显存。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `verl/workers/actor/dp_actor.py`

实现方案：
- 检查 `dp_actor.py` 中分布统计路径对 logits 的处理方式。
- 如果是整块 `logits.to(torch.float32)` 导致峰值显存过高，则改成按 token rows 分块计算：
  - `logsumexp`
  - top1 prob
  - eos prob
  - topk prob / topk mass
- 用小张量对比旧实现和新实现的输出，确认数值一致。

验证计划：
- `python -m py_compile verl/workers/actor/dp_actor.py`
- 在 `verl` 环境中构造随机 logits，对比旧实现与新实现输出

---
## 2026-03-28 记录 25：解释 aux diagnostics 上的 best@k / worst@k / pass@k 含义

目标：
- 解释为什么像 `rollout_logprob_delta_abs_mean` 这种辅助连续指标也会出现 `best@k`、`worst@k`、`pass@k`。
- 说明这些聚合项在代码里是怎么来的，以及哪些适合在 report 里重点使用。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`

实现方案：
- 检查 `ray_trainer.py` 中 diagnostics 指标的生成位置。
- 检查 `metric_utils.py` 中 validation 聚合逻辑，确认 `best@k / worst@k / pass@k` 的统一生成规则。
- 用代码口径解释这些后缀在连续指标上的真实含义。

验证计划：
- 检查 `verl/trainer/ppo/ray_trainer.py`
- 检查 `verl/trainer/ppo/metric_utils.py`

---
## 2026-03-28 记录 26：收窄 validation 聚合规则以减少 aux diagnostics 开销

目标：
- 按当前实验需求，把 validation 聚合规则改成：
  - `acc/reward` 保留 `best/worst/pass`
  - aux diagnostics 只保留 `mean/std`
- 减少无意义的 bootstrap 聚合和 `pass@k` 计算，降低 validation CPU 开销。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/ppo/metric_utils.py`
- `tests/trainer/ppo/test_metric_utils_on_cpu.py`

实现方案：
- 在 `process_validation_metrics()` 中只对 core vars (`acc`, `reward`) 计算 richer aggregation。
- 其他 numeric aux metrics 只计算 `mean@N` 和 `std@N`。
- 调整相关单测，让测试口径与新规则一致。

验证计划：
- `python -m py_compile verl/trainer/ppo/metric_utils.py tests/trainer/ppo/test_metric_utils_on_cpu.py`
- `python -m unittest tests.trainer.ppo.test_metric_utils_on_cpu.TestProcessValidationMetrics -v`

---
## 2026-03-31 记录 27：四次 validation 结果对比分析与初版报告

目标：
- 分析 `ALL_4_EXPERIMENTS_validation_only_20260331.zip` 中四次试验的 validation 结果。
- 抽取关键指标并做图像化对比。
- 产出初版分析报告，并补充一个可执行的 case 级别 LLM 打标方案。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `analysis/exp4_validation_20260331/`（新增分析脚本、图表、报告）

实现方案：
- 解压总包与四个子包，识别每个实验中的指标文件与 case 明细文件。
- 编写 Python 脚本统一读取四个实验数据，构建可比的指标表。
- 生成整体对比图（总体指标）与分组图（case 粒度统计）。
- 基于 case 明细抽样，形成问题模式并设计 LLM 打标 schema。
- 输出初版报告（含结论、图表解读、case 诊断建议、后续实验建议）。

验证计划：
- 运行分析脚本并确认图表和报告成功生成。
- 对比四个实验的样本量、字段覆盖和关键指标范围，确保无明显解析错误。

---
## 2026-03-31 记录 28：四种训练方式的 token 粒度影响分析

目标：
- 基于 `validation/*.jsonl` 中的 `token_diagnostics`，分析四种训练方式对 token 级行为的影响。
- 对比 `step 0` 与 `step 180`，并补充 final step 的 token 级结构化结论。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `analysis/exp4_validation_20260331/`（新增 token 分析脚本、表格、图表、补充报告）

实现方案：
- 流式读取 4 个实验在目标 step 的 `token_diagnostics`，聚合 token 级统计。
- 输出整体 token 指标、按正确/错误分组的 token 指标、按归一化位置 bucket 的曲线。
- 生成 token 级图表，并形成补充报告，说明训练对置信度、entropy、EOS 行为和输出结构的影响。

验证计划：
- 运行 token 分析脚本并确认表格、图像、报告成功生成。
- 校验样本量与 token 总数是否合理，并检查 step0/final 的趋势是否与已有 response 级指标一致。

---
## 2026-03-31 记录 29：训练中低成本 token 监控实现

目标：
- 在训练期 validation diagnostics 中加入低成本、可持续的 token 级监控。
- 支持稳定 probe 采样，并输出 step 级 token 聚合指标，而不依赖全量 raw token 明细。

预计修改的文件或区域：
- `PLAN.md`
- `RESULT.md`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/ppo_megatron_trainer.yaml`
- `tests/trainer/ppo/`（新增或补充 CPU 单测）

实现方案：
- 为 validation diagnostics 增加稳定 hash 采样配置，替代仅按顺序截前 N 条的监控方式。
- 在 token diagnostics 计算阶段为被采样样本构建轻量级聚合 payload。
- 在 validation 主循环中汇总 step 级 token 监控指标，并写入返回的 `metric_dict`。
- 保留原有 raw token trace 导出逻辑，作为少量排障样本。

验证计划：
- 运行针对新 helper 的 CPU 单测。
- 对修改后的 trainer 做语法检查，并确认新增 `val-token/...` 指标能够生成。
