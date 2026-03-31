# Run Jobs

这个目录用来存每一次运行自动生成的独立 `sh` 脚本。

规则：

- 每次执行 `run_qwen_math_25_15B_grpo_1_shot.sh`
- 或 `run_qwen_math_25_15B_grpo_1_shot_kl_cov.sh`

都会自动生成一份：

- `examples/custom/run_jobs/<exp_name>.sh`

后面如果你想复跑某一次实验，直接执行对应的 `sh` 文件即可。

目前也预置了 4 个基础对照实验脚本：

- `qwen_math_25_15B_instruct_oneshot.sh`
- `qwen_math_25_15B_instruct_full.sh`
- `qwen_math_25_15B_base_oneshot.sh`
- `qwen_math_25_15B_base_full.sh`
