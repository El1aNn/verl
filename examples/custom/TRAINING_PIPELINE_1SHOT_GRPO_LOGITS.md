# 1-shot GRPO Logits 实验训练 Pipeline（从跑通开始）

这份 pipeline 默认基于：

- 主脚本：`examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- 烟雾脚本：`examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh`
- 矩阵脚本：`examples/custom/prepare_qwen_math_25_15B_grpo_2seed_matrix.sh`
- 对比报告脚本：`examples/custom/render_qwen_math_25_15B_grpo_2seed_compare_report.sh`
- 报告脚本：`examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh`

另外，两个主训练脚本现在都会自动把“本次运行对应的独立启动脚本”保存到：

- `examples/custom/run_jobs/`

---

## 0. 预检查（5 分钟）

```bash
cd /root/rl/verl
nvidia-smi
python3 -V
ray --version
```

建议统一基础环境变量：

```bash
export HOME_DIR=/root/rl
export RAY_GCS_ADDRESS=127.0.0.1:6379
export RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265
export ENABLE_VAL_DIAGNOSTICS=true
export ENABLE_PAPER_STYLE_VIZ=true
```

---

## 1. 先跑通：1-shot 烟雾测试（推荐第一步）

```bash
cd /root/rl/verl
export EXP_NAME=smoke_qwen_math_15b_instruct_oneshot_$(date +%Y%m%d-%H%M%S)
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
TRAIN_MODE=oneshot bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh
```

这个脚本会自动使用轻量参数（`TOTAL_EPOCHS=1`、较小 batch/rollout）来验证整条链路。

---

## 2. 跑通后立刻验证产物

训练脚本是异步提交（`ray job submit --no-wait`），先看任务状态：

```bash
ray job list --address="$RAY_DASHBOARD_ADDRESS"
```

任务成功后渲染报告：

```bash
CKPTS_DIR=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/${EXP_NAME} \
bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh
```

检查两个文件是否生成：

- `${CKPTS_DIR}/paper_viz/index.html`
- `${CKPTS_DIR}/paper_viz/token_traces.html`

---

## 3. 正式训练：oneshot vs full（单模型）

### 3.1 oneshot（正式配置）

```bash
export EXP_NAME=qwen_math_15b_instruct_oneshot_$(date +%Y%m%d-%H%M%S)
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=100
TRAIN_MODE=oneshot bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
```

### 3.2 full（正式配置）

```bash
export EXP_NAME=qwen_math_15b_instruct_full_$(date +%Y%m%d-%H%M%S)
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=100
TRAIN_MODE=full bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
```

---

## 4. 跨模型扩展（用于 report 主结论）

如果时间有限，建议默认用 `2 seeds`。这样主矩阵一共是 `8 runs`：

1. Instruct + oneshot + seed 1
2. Instruct + full + seed 1
3. Base + oneshot + seed 1
4. Base + full + seed 1
5. Instruct + oneshot + seed 2
6. Instruct + full + seed 2
7. Base + oneshot + seed 2
8. Base + full + seed 2

你可以直接生成这 8 个命令：

```bash
bash examples/custom/prepare_qwen_math_25_15B_grpo_2seed_matrix.sh
```

如果确认要直接提交：

```bash
SUBMIT=true bash examples/custom/prepare_qwen_math_25_15B_grpo_2seed_matrix.sh
```

脚本会把 manifest 写到：

```bash
/root/rl/verl/ckpts/matrix_manifests/qwen_math_25_15B_2seed_<batch_tag>.tsv
```

建议先做这 4 个核心配置，然后补第 2 个 seed：

1. Instruct + oneshot
2. Instruct + full
3. Base + oneshot
4. Base + full

Base 模型只需切换：

```bash
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B
```

如需手工复现重复实验，设置：

```bash
export DATA_SEED=1
export ROLLOUT_SEED=1
```

并改成 `2` 复跑即可。

---

## 5. 对比报告（多 run 合并）

如果你已经用矩阵脚本生成过 manifest，直接跑：

```bash
bash examples/custom/render_qwen_math_25_15B_grpo_2seed_compare_report.sh
```

它会自动：

1. 找最新的 `2-seed manifest`
2. 收集已经完成且存在 validation JSONL 的 run
3. 生成 compare report
4. 在输出目录里写一份可复用命令：`compare_report_command.sh`

如果你想只生成命令、不立即运行：

```bash
RUN_REPORT=false bash examples/custom/render_qwen_math_25_15B_grpo_2seed_compare_report.sh
```

如果你想指定某个 manifest：

```bash
bash examples/custom/render_qwen_math_25_15B_grpo_2seed_compare_report.sh \
  /root/rl/verl/ckpts/matrix_manifests/qwen_math_25_15B_2seed_<batch_tag>.tsv
```

手工方式仍然保留如下：

```bash
python3 scripts/validation_viz_report.py \
  --run instruct_oneshot=/path/to/instruct_oneshot/validation::/path/to/instruct_oneshot/metrics/verl_grpo_dsr_sub_baseline/<exp>.jsonl \
  --run instruct_full=/path/to/instruct_full/validation::/path/to/instruct_full/metrics/verl_grpo_dsr_sub_baseline/<exp>.jsonl \
  --run base_oneshot=/path/to/base_oneshot/validation::/path/to/base_oneshot/metrics/verl_grpo_dsr_sub_baseline/<exp>.jsonl \
  --run base_full=/path/to/base_full/validation::/path/to/base_full/metrics/verl_grpo_dsr_sub_baseline/<exp>.jsonl \
  --output-dir /root/rl/verl/ckpts/compare_1shot_vs_full \
  --trace-top-samples 6
```

---

## 6. 你现在最常用的三条命令

1. 先跑通：

```bash
TRAIN_MODE=oneshot bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh
```

2. 正式跑：

```bash
TRAIN_MODE=oneshot bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
```

3. 出图：

```bash
CKPTS_DIR=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/<exp_name> \
bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh
```
