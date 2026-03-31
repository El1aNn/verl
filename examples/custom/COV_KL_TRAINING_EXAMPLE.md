# Cov-KL 训练启动示例

这是一份单独的启动示例文档，专门对应当前仓库里的 `cov-kl` 自定义训练链路。

适用脚本：

- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`
- `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh`

这条链路默认走的训练入口是：

- `recipe.entropy.main_entropy`

## 1. Cov-KL 在这套脚本里对应什么

当前 `cov-kl` 主要由下面 3 个参数控制：

- `LOSS_MODE=kl_cov`
- `KL_COV_RATIO=0.0002`
- `PPO_KL_COEF=1.0`

在 `examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh` 里，这 3 个变量会被传成：

- `actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov`
- `actor_rollout_ref.actor.policy_loss.kl_cov_ratio=<...>`
- `actor_rollout_ref.actor.policy_loss.ppo_kl_coef=<...>`

另外，这条脚本默认是：

- `USE_KL_IN_REWARD=false`
- `USE_KL_LOSS=false`
- `KL_LOSS_COEF=0.0`

也就是说，它默认跑的是 `KL-Cov`，不是“额外再叠一层常规 actor KL loss”。

如果你想跑“`KL-Cov + 常规 actor KL loss`”的混合版本，再额外打开：

- `USE_KL_LOSS=true`
- `KL_LOSS_COEF=0.001`
- `KL_LOSS_TYPE=low_var_kl`

## 2. 启动前必须知道的两点

### 2.1 这个脚本一开始会先 `ray stop`

`examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh` 开头会执行：

```bash
ray stop
```

所以：

- 不要和别的本地 Ray 训练并行跑
- 如果你机器上已经有另一条训练在跑，新开的这条会把它停掉

### 2.2 推荐用 `screen`

建议始终用 `screen` 启动，尤其是本机单卡/双卡本地 Ray。

一个稳定的启动模式是：

- 用 `screen -dmS ... bash -lc '...; sleep 12h'`
- 让 `screen` 本身多活一段时间，方便后续查看日志

## 3. 最小预检查

```bash
cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

nvidia-smi
ray --version
python3 -V
```

如果上一次跑挂了，先清一下本地 Ray：

```bash
ray stop --force
```

## 4. 推荐环境变量

如果你要跑 Instruct 模型，记得显式覆盖 `MODEL_PATH`。

因为 `run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh` 里的默认 `MODEL_PATH` 是 base 模型路径，不是 instruct。

推荐统一补这些：

```bash
export HOME_DIR=/root/rl
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export LOSS_MODE=kl_cov
export KL_COV_RATIO=0.0002
export PPO_KL_COEF=1.0
export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=180
export DATA_SEED=1
export ROLLOUT_SEED=1
```

如果你要保留当前 run 的启动脚本副本，也可以显式打开：

```bash
export SAVE_RUN_SCRIPT=true
```

生成的脚本会落到：

- `examples/custom/run_jobs/`

## 5. 烟雾测试

先跑一轮小配置 smoke，确认链路能通：

```bash
screen -dmS covkl_smoke bash -lc '
cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

export HOME_DIR=/root/rl
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export TRAIN_MODE=oneshot
export EXP_NAME=smoke_qwen_math_25_15B_instruct_oneshot_kl_cov_$(date +%Y%m%d-%H%M%S)
export LOSS_MODE=kl_cov
export KL_COV_RATIO=0.0002
export PPO_KL_COEF=1.0

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov_smoke.sh
echo "[smoke launcher finished] keep screen alive for inspection"
sleep 12h
'
```

这个脚本会自动使用更小的：

- `TOTAL_EPOCHS=1`
- `TRAIN_BATCH_SIZE=32`
- `ROLLOUT_N=4`
- `VAL_ROLLOUT_N=4`

适合先验证训练链路和日志是否正常。

## 6. 正式训练示例

### 6.1 Instruct + oneshot + Cov-KL

```bash
screen -dmS covkl_oneshot bash -lc '
cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

export HOME_DIR=/root/rl
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export TRAIN_MODE=oneshot
export EXP_NAME=qwen_math_25_15B_instruct_oneshot_kl_cov_$(date +%Y%m%d-%H%M%S)

export LOSS_MODE=kl_cov
export KL_COV_RATIO=0.0002
export PPO_KL_COEF=1.0

export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=180
export DATA_SEED=1
export ROLLOUT_SEED=1

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
echo "[formal oneshot launcher finished] keep screen alive for inspection"
sleep 12h
'
```

### 6.2 Instruct + full + Cov-KL

```bash
screen -dmS covkl_full bash -lc '
cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

export HOME_DIR=/root/rl
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export TRAIN_MODE=full
export EXP_NAME=qwen_math_25_15B_instruct_full_kl_cov_$(date +%Y%m%d-%H%M%S)

export LOSS_MODE=kl_cov
export KL_COV_RATIO=0.0002
export PPO_KL_COEF=1.0

export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=180
export DATA_SEED=1
export ROLLOUT_SEED=1

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
echo "[formal full launcher finished] keep screen alive for inspection"
sleep 12h
'
```

### 6.3 Base 模型版本

只需要把 `MODEL_PATH` 改成 base：

```bash
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B
```

## 7. 如果你想叠加常规 KL loss

默认 `cov-kl` 脚本里常规 actor KL 是关掉的。

如果你要跑：

- `KL-Cov`
- 再加上 actor KL loss

可以这样覆盖：

```bash
export USE_KL_LOSS=true
export KL_LOSS_COEF=0.001
export KL_LOSS_TYPE=low_var_kl
```

一个完整示例：

```bash
screen -dmS covkl_plus_actor_kl bash -lc '
cd /root/rl/verl
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate verl

export HOME_DIR=/root/rl
export MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct
export TRAIN_MODE=oneshot
export EXP_NAME=qwen_math_25_15B_instruct_oneshot_kl_cov_plus_actor_kl_$(date +%Y%m%d-%H%M%S)

export LOSS_MODE=kl_cov
export KL_COV_RATIO=0.0002
export PPO_KL_COEF=1.0
export USE_KL_LOSS=true
export KL_LOSS_COEF=0.001
export KL_LOSS_TYPE=low_var_kl

export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=180

bash examples/custom/run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh
echo "[cov-kl + actor-kl launcher finished] keep screen alive for inspection"
sleep 12h
'
```

## 8. 常用覆盖项

如果你要调小规模或控制随机性，最常用的是这些：

```bash
export TRAIN_MODE=oneshot           # 或 full
export EXP_NAME=your_exp_name
export MODEL_PATH=/path/to/model
export TOTAL_EPOCHS=10
export TEST_FREQ=6
export SAVE_FREQ=180
export DATA_SEED=1
export ROLLOUT_SEED=1

export LOSS_MODE=kl_cov
export KL_COV_RATIO=0.0002
export PPO_KL_COEF=1.0

export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE_PER_GPU=4
export ROLLOUT_N=8
export VAL_ROLLOUT_N=8
```

如果你想临时补额外 Hydra 参数，旧 `cov-kl` 脚本还支持：

```bash
export EXTRA_HYDRA_ARGS='trainer.save_freq=180 trainer.total_epochs=10'
```

## 9. 训练后怎么看日志

主日志会写到：

- `/root/rl/verl/logs/verl_grpo_dsr_sub_baseline-<exp_name>-<timestamp>.log`

screen 日志会写到你启动时指定的：

- `/root/rl/verl/logs/<screen_name>.screen.log`

重新连回 screen：

```bash
screen -r covkl_oneshot
```

查看当前 screen：

```bash
screen -ls
```

## 10. 训练后怎么出 report

训练完成后可以渲染单 run report：

```bash
CKPTS_DIR=/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline/${EXP_NAME} \
bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh
```

## 11. 最后再提醒一次

最容易踩的坑就是这 3 个：

1. `run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh` 一开始会 `ray stop`，不要并行跑第二个本地 Ray 实验。
2. 不覆盖 `MODEL_PATH` 的话，这条脚本默认会跑 base，不是 instruct。
3. 本机启动时优先用 `screen`，这样更方便保留本地 Ray、日志和后续排查现场。
