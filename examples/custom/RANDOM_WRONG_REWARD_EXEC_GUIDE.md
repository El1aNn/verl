# Random Wrong Reward 执行指南

这份文档覆盖你现在需要的完整流程：

1. 生成 `随机错误 reward` 数据。
2. 校验数据已生效。
3. 用 `screen + swanlab` 启动 `Base-C-n8`。
4. 可选：一键提交 6 组实验（2 模型 x 3 监督信号）。

---

## 1. 环境准备

```bash
cd /root/rl/verl

# 推荐激活 verl 环境（如已在该环境可跳过）
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

# 可选：先确认关键组件
nvidia-smi
ray --version
python -V
```

如果要上 SwanLab，请先设置 key：

```bash
export SWANLAB_API_KEY=<your_key>
```

---

## 2. 生成随机错误 Reward 数据

```bash
cd /root/rl/verl
python verl/utils/dataset/generate_random_wrong_reward.py \
  --input-file /root/rl/verl/data/dsr_sub/pi1_one_ans.parquet \
  --output-file /root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet \
  --seed 20260331 \
  --value-min -100 \
  --value-max 100 \
  --decimals 3
```

产物文件：

- `/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet`

---

## 3. 校验数据（必须）

```bash
cd /root/rl/verl
python - <<'PY'
import pandas as pd

src = "/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet"
dst = "/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet"

df_src = pd.read_parquet(src)
df_dst = pd.read_parquet(dst)

old = [x["ground_truth"] for x in df_src["reward_model"]]
new = [x["ground_truth"] for x in df_dst["reward_model"]]
changed = sum(str(a) != str(b) for a, b in zip(old, new))

print("rows_src:", len(df_src))
print("rows_dst:", len(df_dst))
print("changed_rows:", changed)
print("unique_old:", len(set(map(str, old))))
print("unique_new:", len(set(map(str, new))))
print("sample_pairs:", list(zip(old[:3], new[:3])))
PY
```

预期：

- `rows_src == rows_dst`
- `changed_rows` 应该接近总行数（当前数据通常是全变更）

---

## 4. 启动 Base-C-n8（screen + swanlab）

先写一个可复用启动脚本：

```bash
cat > /tmp/run_base_c_n8_wrong_reward.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd /root/rl/verl

export EXP_NAME=base_c_n8_wrong_reward_$(date +%Y%m%d-%H%M%S)
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B
export TRAIN_MODE=oneshot
export ONE_SHOT_TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet
export TRAIN_FILE=/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet

# Base-C-n8 核心设置
export ROLLOUT_N=8
export VAL_ROLLOUT_N=8
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE_PER_GPU=4
export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=180

# 有 SWANLAB_API_KEY 时会自动走 swanlab+file；这里显式指定
export LOGGER='["swanlab","file"]'

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh
EOF

chmod +x /tmp/run_base_c_n8_wrong_reward.sh
```

再用 `screen` 后台启动：

```bash
screen -dmS base_c_n8_wrong_reward /tmp/run_base_c_n8_wrong_reward.sh
```

---

## 5. 监控与排查命令

```bash
# 查看 screen
screen -ls

# 回连 screen
screen -r base_c_n8_wrong_reward

# 查看 Ray Job 状态
ray job list --address=http://127.0.0.1:8265

# 查看最近运行脚本（主脚本会自动保存一份到 run_jobs）
ls -lt /root/rl/verl/examples/custom/run_jobs | head

# 查看最近日志
ls -lt /root/rl/verl/logs | head
```

---

## 6. 可选：一键提交 6 组实验（2 模型 x 3 监督信号）

监督信号组合：

- `clean`: `pi1_one_ans.parquet`
- `shuffle`: `pi1_one_ans_reward_shuffle.parquet`
- `random_wrong`: `pi1_one_ans_reward_random_wrong.parquet`

模型组合：

- `base`: `Qwen2.5-Math-1.5B`
- `instruct`: `Qwen2.5-Math-1.5B-Instruct`

> 下面命令会提交 6 个 Ray Job（非阻塞提交），建议在资源允许时使用。

```bash
cat > /tmp/launch_6_supervision_runs.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /root/rl/verl

declare -A MODELS=(
  [base]="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B"
  [instruct]="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct"
)

declare -A DATASETS=(
  [clean]="/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet"
  [shuffle]="/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_shuffle.parquet"
  [random_wrong]="/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet"
)

for model_tag in base instruct; do
  for data_tag in clean shuffle random_wrong; do
    export EXP_NAME="${model_tag}_c_n8_${data_tag}_$(date +%Y%m%d-%H%M%S)"
    export MODEL_PATH="${MODELS[$model_tag]}"
    export TRAIN_MODE=oneshot
    export ONE_SHOT_TRAIN_FILE="${DATASETS[$data_tag]}"
    export TRAIN_FILE="${DATASETS[$data_tag]}"
    export ROLLOUT_N=8
    export VAL_ROLLOUT_N=8
    export TRAIN_BATCH_SIZE=64
    export PPO_MINI_BATCH_SIZE=32
    export PPO_MICRO_BATCH_SIZE_PER_GPU=4
    export TOTAL_EPOCHS=10
    export TEST_FREQ=6
    export SAVE_FREQ=180
    export LOGGER='["swanlab","file"]'

    echo "[submit] ${EXP_NAME}"
    bash examples/custom/run_qwen_math_25_15B_grpo_1_shot.sh
    sleep 5
  done
done
EOF

chmod +x /tmp/launch_6_supervision_runs.sh
screen -dmS oneshot_6runs /tmp/launch_6_supervision_runs.sh
```
