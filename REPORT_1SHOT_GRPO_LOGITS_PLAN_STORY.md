# 1-shot GRPO Reasoning Logits：训练计划与 Report Story

## 1. 报告目标

你这篇 report 聚焦三个问题：

1. `1-shot GRPO` 在 reasoning 过程中到底改变了哪些 token 分布行为（logits/probability dynamics）？
2. 这些变化和 `full` 训练相比有什么异同？在不同模型上是否一致？
3. 为什么 `1-shot` 这种极端低样本训练仍然能生效？

---

## 2. 核心假设（可证伪）

1. `H1: 快速重排而非重学知识`
   1-shot 的主要作用是快速重排已有推理模板对应 token 的相对 logits，而不是从零学习新知识。
2. `H2: 收敛信号集中在“关键分叉 token”`
   变化最大的位置不是全序列平均，而是 reasoning 分叉点（转折词、纠错词、EOS 竞争位点）。
3. `H3: 生效依赖预训练/指令先验`
   基座能力越强（尤其是 instruct/数学底座），1-shot 越像“对齐与校准”，而非“能力注入”。
4. `H4: 1-shot 的稳定性来自约束项`
   KL/entropy/kl-cov 等约束让策略更新成为“局部修正”，避免模式崩塌。

---

## 3. 实验矩阵（建议先跑最小可发表版本）

## 3.1 最小矩阵（优先）

| 维度 | 设置 |
|---|---|
| 模型 A | `Qwen2.5-Math-1.5B-Instruct` |
| 模型 B | `Qwen2.5-Math-1.5B` |
| 训练模式 | `oneshot` vs `full` |
| 目标函数 | `GRPO (main_ppo)` 与 `GRPO + kl_cov (main_entropy)`（可选） |
| 随机种子 | 每个配置 `2 seeds`（当前推荐） |

总 run 数（当前推荐）：`2 models x 2 modes x 2 seeds = 8 runs`。  
如果加 `kl_cov` 变体：翻倍为 `16 runs`。

## 3.2 控制变量（必须统一）

1. 相同验证集：`data/testset/*.parquet`
2. 相同采样参数：`rollout_n=8, val_rollout_n=8, val_temperature=1.0`
3. 相同诊断配置：`ENABLE_VAL_DIAGNOSTICS=true`, `ENABLE_PAPER_STYLE_VIZ=true`
4. 固定长度设置：`max_prompt_length=1024`, `max_response_length=2048`
5. 统一评估频率：`test_freq=6`（或你当前统一值）

注意：`oneshot` 与 `full` 的数据规模差异会导致 step 数不同。  
建议做两组比较：

1. `现实设置`：按默认配置直接对比（展示“实战可行性”）
2. `预算对齐`：按优化步数/总 token 对齐（做机制归因）

---

## 4. 训练执行计划（分阶段）

## 4.1 阶段 P0：烟雾测试（1 天）

目标：确认日志和诊断字段完整，不浪费大算力。

1. 每个模型先跑 `oneshot/full` 各 1 个短 run（少量 epoch）。
2. 检查 validation dump 是否包含：
   `token_entropy_mean`, `low_confidence_token_ratio`,
   `eos_prob_final`, `eos_prob_slope`,
   `topk_mass_mean`, `token_diagnostics`。

## 4.2 阶段 P1：主实验（2-4 天）

目标：产出可写主结论的主图和统计。

1. 跑完整矩阵（当前先用 2 seeds）。
2. 每个 run 结束后立即渲染报告：
   `bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh <CKPTS_DIR>`
3. 统一汇总到跨 run 对比报告（见第 6 节命令模板）。

## 4.3 阶段 P2：机制消融（可选但强烈建议）

目标：回答“为什么 1-shot 有效”而不是只描述现象。

1. 去约束：降低/关闭 KL 或 entropy，观察是否更不稳定。
2. 奖励打乱（负对照）：若性能仍升，说明不是有效学习而是偶然漂移。
3. 缩减 group size（例如降 `rollout_n`）：验证 GRPO 的相对排序信号是否关键。

---

## 5. 指标与图表清单（和你现有检测对齐）

## 5.1 主指标（结果层）

1. `primary_metric_mean`（按脚本自动选定）
2. `reward_mean`, `acc_mean`（若可用）

## 5.2 logits/probability 动态指标（机制层）

1. `token_entropy_mean`, `entropy_slope`
2. `low_confidence_token_ratio`
3. `eos_prob_mean`, `eos_prob_final`, `eos_prob_slope`, `eos_high_prob_ratio`
4. `topk_mass_mean`, `topk_mass_final`
5. 跨 eval 变化：`eos_prob_delta_prev_eval`, `eos_prob_final_delta_prev_eval`
6. 样本级轨迹：`token_diagnostics`（含 top-k、eos_rank_in_topk）

## 5.3 建议主图（报告正文）

1. 训练动态：`training_dynamics.svg`
2. 熵轨迹：`training_entropy.svg`
3. 概率-熵散点：`probability_entropy_scatter.svg`
4. 低概率 token 分布：`sampled_probability_distributions.svg`
5. 低概率 top-k 趋势：`low_prob_topk_trends.svg`
6. 样本级证据：`token_traces.html`（挑 3-6 个 UID）

---

## 6. 命令模板（多模型对比）

单个 run 渲染（你已有）：

```bash
CKPTS_DIR=/path/to/ckpts/<project>/<exp> \
bash examples/custom/render_qwen_math_25_15B_grpo_1_shot_report.sh
```

多 run 合并对比（核心）：

```bash
python3 scripts/validation_viz_report.py \
  --run m1_oneshot_s1=/path/m1_oneshot_s1/validation::/path/m1_oneshot_s1/metrics/<project>/<exp>.jsonl \
  --run m1_full_s1=/path/m1_full_s1/validation::/path/m1_full_s1/metrics/<project>/<exp>.jsonl \
  --run m2_oneshot_s1=/path/m2_oneshot_s1/validation::/path/m2_oneshot_s1/metrics/<project>/<exp>.jsonl \
  --run m2_full_s1=/path/m2_full_s1/validation::/path/m2_full_s1/metrics/<project>/<exp>.jsonl \
  --output-dir /path/to/compare_report \
  --trace-top-samples 6
```

---

## 7. Report Story（可直接当写作骨架）

## 7.1 一句话结论

`1-shot GRPO` 的有效性主要来自“对已有推理能力的快速 logit 重排与校准”，而不是新能力注入；  
这种重排在关键 reasoning token 与 EOS 竞争位点最明显，并且受 KL/entropy 约束稳定。

## 7.2 叙事结构

1. `Act 1: Paradox`
   数据极少（1-shot），却出现稳定收益，为什么？
2. `Act 2: What moves`
   用 token 级指标证明：并非全局均匀变化，而是关键位点的概率结构改变。
3. `Act 3: Cross-model consistency`
   不同模型都出现“方向一致但幅度不同”的变化。
4. `Act 4: Mechanism`
   GRPO 相对排序信号 + 预训练先验 + KL/entropy 约束，共同形成“低步数高收益”的快速校准。
5. `Act 5: Boundary`
   1-shot 不是万能：在先验不足或约束失衡时，容易出现不稳定或早停偏置。

## 7.3 为什么 1-shot 也能生效（建议写法）

1. `强先验假说`：模型已具备大量潜在推理模板，1-shot 只需“选优重排”。
2. `相对优势假说`：GRPO 组内比较减少绝对 reward 标定难度，少样本也能得到方向性梯度。
3. `局部更新假说`：KL/entropy/kl-cov 让更新集中在少量关键 logits，降低灾难性漂移风险。
4. `可检验预测`：
   有效 run 会表现为“关键 token 概率提升 + EOS 行为更合理 + 熵下降可控”，而非无差别熵塌缩。

---

## 8. 验收标准（你可以直接贴在实验看板）

1. 至少 2 个模型上，`oneshot` 对主指标相对初始点有一致提升（跨 seed）。
2. 关键 token 动态指标在 `oneshot` 与 `full` 间呈现可解释差异，而不是随机抖动。
3. 至少 3 个代表性 UID 的 token trace 能支持“关键位点重排”结论。
4. 负对照（如 reward 打乱）无法复现同等提升，排除偶然因素。

---

## 9. 风险与兜底

1. `风险`：oneshot 结果高方差。  
   `兜底`：固定 2 seeds，报告均值与区间，不只展示最佳 run。
2. `风险`：dump 不全导致 trace 结论不稳。  
   `兜底`：保持 `ENABLE_PAPER_STYLE_VIZ=true`，必要时提高 `VAL_DIAG_MAX_TOKENS`。
3. `风险`：full 与 oneshot 比较被步数混淆。  
   `兜底`：补一组预算对齐实验（step-matched）。
