# 六实验 Report 草稿（2026-04-05）

> 2026-04-06 更新说明：正式稿已经采用新的对齐版 `n=1` 消融实验
> `base_c_n1_val8_align_exp1_2gpu_20260406_195153`。
> 这版结果与 `Base baseline (n=8)` 在 `val_rollout_n=8` 上完全对齐：
> `step0=0.3630`，`final=0.5470`，`best step=174`，`best score=0.5590`。
> 因此，下面草稿中凡是提到旧 `n=1, val_n=1` 版本、`0.3580 -> 0.5900` 或“评估口径不一致”的部分，都应以 Overleaf 正式稿为准。

## 使用说明

这是一份基于当前仓库内可核验产物整理的正文草稿底稿，目标是支撑一篇`正文 25 页左右`的中文实验报告。

当前状态：

- 已根据本地 `ckpts/`、`validation/*.jsonl`、`run_jobs/*.sh` 和训练日志，整理出 6 组可写入正文的实验结果。
- `Base-C-n1 group ablation` 已经有新的完整重跑版本：`base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719`。不过它的 `val_rollout_n=1`，因此每个 step 只有 `500` 条验证记录，而 `n=8` 系列每个 step 是 `4000` 条记录；两者可比较，但需要在正文里说明口径差异。
- 所谓 `reward shuffle` 数据文件 `data/dsr_sub/pi1_one_ans_reward_shuffle.parquet` 与 `data/dsr_sub/pi1_one_ans.parquet` 当前 `md5` 完全一致，因此如果没有别的最终数据版本，这个负对照在当前证据下`不能被严格解释为真实 reward 打乱实验`。

建议你把这份稿子当作：

1. 正文主干；
2. 答辩/汇报版本的第一稿；
3. 后续补图、补表、补格式的统一底稿。

---

## 需要你补充或确认的信息

为了把这份稿子真正收束成可提交版本，我目前最需要你补这 8 项：

1. 报告的正式题目、课程/项目名称、作者信息、日期、单位。
2. 学校或课程是否有固定模板：页边距、字体、图表格式、参考文献格式、是否要求摘要中英双语。
3. 六个实验里你最终想认定为“正式 Exp1 基线”的目录到底是哪一个。
   当前本地脚本证据更支持 `base_align_exp1_20260401_141352` 是 `n=8` 基线，因为其运行脚本中明确写了 `ROLLOUT_N='8'`。
   如果你有外部记录证明正式基线其实是别的目录，请把最终实验名发我。
4. 如果 `Base-C-n1 group ablation` 还有比 `base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719` 更正式的最终版本，请告诉我；否则我就按这个目录写入正式稿。
5. `reward shuffle` 是否确实有另一份真正打乱过的训练集；如果有，请给我正确文件路径、日志或结果目录。
6. 这篇报告是更偏“课程实验报告”、还是更偏“论文式 research report”；两者写法会不一样。
7. 你是否需要我把图表也一起生成并插入正文。
8. 你希望最终语言风格更偏：
   - 保守学术型
   - 工程复现实验型
   - 论文投稿型

如果你现在不想一次补太多，优先补第 3、4、5 项就够我继续大幅推进。

---

## 正文页数规划建议

为了稳定达到`正文 25 页`，建议按下面的篇幅分配来写：

| 章节 | 建议页数 |
| --- | --- |
| 摘要 + 关键词 | 1 |
| 1. 引言 | 3 |
| 2. 方法与研究问题 | 3 |
| 3. 实验设置 | 4 |
| 4. 主结果 | 4 |
| 5. 机制分析 | 4 |
| 6. 消融与负对照 | 3 |
| 7. 讨论与局限性 | 2 |
| 8. 结论 | 1 |

如果学校把摘要、目录、参考文献不计入正文，那么上面这套结构仍然足够把正文撑到 25 页。

---

# 正文草稿

## 题目（待你确认）

`1-shot GRPO 在数学推理模型上的行为分析：基于六组对照实验的结果与机制研究`

备选题目：

- `面向数学推理的 1-shot GRPO 训练实验报告`
- `Qwen2.5-Math-1.5B 上 1-shot GRPO 的有效性、稳定性与局限性分析`
- `基于六组实验的 1-shot GRPO 数学推理对齐研究`

---

## 摘要

本报告围绕 `1-shot GRPO` 在数学推理任务上的训练效果展开，重点考察在极低样本条件下，强化学习式对齐是否仍能带来可观收益，以及这种收益来自能力注入还是来自对已有推理模式的快速校准。为此，我们基于 `Qwen2.5-Math-1.5B` 与 `Qwen2.5-Math-1.5B-Instruct` 构建了一组包含基线、弱约束、group size 消融与 reward shuffle 负对照在内的六实验矩阵，并统一使用 `MATH500` 验证集与 token-level diagnostics 对模型行为进行跟踪。

从当前仓库内可核验的结果看，Base 模型在标准 `n=8` 配置下由 `0.3745` 提升至约 `0.5733`，绝对提升接近 `0.20`；新完成的 `Base-C-n1 group ablation` 由 `0.3580` 提升至 `0.5900`；而 Instruct 模型初始性能已较高，从 `0.6837` 仅提升到 `0.6873` 到 `0.6933` 区间，收益显著小于 Base 模型。这表明 1-shot GRPO 对弱先验模型更像一次明显的对齐校正，而对强先验模型则更像局部微调。进一步地，Base 模型在训练后表现出更短的平均响应长度、更高的答案尾部置信度、更低的低置信 token 比例，以及显著下降的 token entropy，说明其改进并非简单来自“生成更长推理链”，而更可能来自关键推理位置的分布收缩与结束行为优化。

不过，当前实验也暴露出两个重要限制。第一，虽然 `Base-C-n1 group ablation` 已经完成重跑，但它使用 `val_rollout_n=1`，而其他主实验多为 `val_rollout_n=8`，因此其分数口径仍需谨慎比较。第二，当前 `reward shuffle` 训练文件与 clean 文件在字节级完全一致，导致这组实验暂不能被严格视为真实负对照。基于上述结果，本报告认为：1-shot GRPO 的有效性在当前设定下是成立的，但关于 group size 机制和负对照排除，需要补充更严格的对齐实验才能形成更强结论。

关键词：`GRPO`，`1-shot training`，`mathematical reasoning`，`Qwen2.5-Math`，`alignment`，`token diagnostics`

---

## 1. 引言

近年来，大语言模型在数学推理、代码生成与科学问答等复杂推理任务上取得了快速进展。相比纯监督微调，基于强化学习的对齐方法能够利用答案正确性、过程奖励或相对偏好信号，对模型的输出行为进行更细粒度的优化。尤其是在数学推理任务中，模型最终正确与否不仅取决于是否“知道”相关知识，还取决于是否能够在推理过程中选择合适的分解路径、控制中间步骤的稳定性，并在正确时机结束生成。正因如此，围绕推理模型的强化学习训练逐渐成为当前研究与工程实践的重点方向。

在众多算法中，`GRPO` 通过组内相对比较来构造优势估计，降低了对显式 critic 与高精度绝对 reward 标定的依赖，因此被认为特别适合用于大语言模型的推理优化。对于数学题这样的结构化任务，模型往往能够在多次采样中表现出不同质量的解法，GRPO 正好可以利用这种组内差异来放大“更优解”的训练信号。相比标准 PPO，GRPO 在工程实现与训练稳定性之间提供了一个较有吸引力的折中。

然而，一个值得进一步研究的问题是：当训练数据极少，甚至接近 `1-shot` 级别时，GRPO 是否仍然有效？如果有效，它到底改变了什么？直觉上，极少样本不太可能让模型真正学会一套全新的数学知识体系，因此更合理的假设是，训练主要在“重排”模型已有的推理模式，而不是注入新能力。换句话说，1-shot GRPO 可能是一种快速的对齐与校准机制，而非传统意义上的能力扩展方法。

从研究角度看，这个问题很有价值。第一，它能帮助我们理解小样本强化学习在推理模型上的真实作用边界。第二，它能解释为什么某些模型在极低数据预算下仍然能获得明显收益，而另一些模型几乎没有提升。第三，它能为后续设计更高效的推理对齐方案提供依据，例如应该更关注约束项、group size，还是更关注训练数据的排序与负对照构造。

基于此，本报告围绕以下三个核心问题展开：

1. 在数学推理任务上，1-shot GRPO 是否能给模型带来可重复的性能提升？
2. 训练后的性能变化是否伴随 token 级行为变化，例如 entropy、低置信 token 比例、EOS 概率与 top-1 概率结构的改变？
3. 在当前六组实验设定下，哪些因素可能是收益的来源，哪些因素又限制了结论的可靠性？

为了回答这些问题，我们构建了一组六实验矩阵。实验包括 Base 模型与 Instruct 模型的标准 `n=8` 基线、弱约束版本、group size 缩减版本以及 reward shuffle 负对照版本，并统一采用 `MATH500` 验证集进行多步评估。此外，我们不只看最终正确率，还引入 token-level diagnostics，用于追踪模型在训练前后推理分布的变化。

需要强调的是，这份报告并不试图宣称“六个实验都已形成无争议结论”。恰恰相反，我们在结果分析中会明确区分“当前证据足够支持的结论”和“受实验实现限制而只能谨慎讨论的结论”。这种写法虽然更保守，但对于课程报告或研究汇报而言反而更可靠，因为它能够同时呈现实验价值与实验边界。

---

## 2. 方法与研究问题

### 2.1 GRPO 的基本思想

本实验使用的训练框架基于 `verl.trainer.main_ppo` 中的 GRPO 路径。与传统 PPO 不同，GRPO 的核心不在于为每个样本估计一个高精度状态价值，而在于对同一输入下的多个采样结果做组内比较，并据此形成相对优势。对数学推理任务来说，这种设定很自然，因为同一道题目可以采样出多个候选推理链，其中往往既有完全正确的解法，也有部分正确但最终失败的解法，还有明显错误或过早结束的解法。GRPO 的训练目标，本质上就是推动模型在这些候选之间重新分配概率质量，让更优解法在下次采样时更容易被选中。

在当前实验中，标准配置采用 `rollout_n=8`，即对每个输入采样 8 个候选响应。对于 Base 模型和 Instruct 模型，我们统一采用 `train_batch_size=64`、`ppo_mini_batch_size=32`、`ppo_micro_batch_size_per_gpu=2`、学习率 `1e-6`，并使用 `test_freq=6` 对验证集进行定期评估。标准基线还开启了 `KL loss` 与 `entropy coefficient`，以控制训练过程中策略漂移过大或分布过快塌缩的问题。

### 2.2 我们关心的不是只有正确率

如果只看最终正确率，那么很容易得到一个过于粗糙的结论：Base 模型收益明显，Instruct 模型收益有限。但这样的结果并不能解释训练究竟改变了模型的什么行为。为此，本实验在验证阶段开启了较细粒度的 diagnostics，记录了若干关键指标，包括：

- `score`：最终是否答对的主指标；
- `response_length`：平均响应长度；
- `answer_tail_confidence`：答案尾部区域的平均置信度；
- `low_confidence_token_ratio`：低置信 token 的比例；
- `token_entropy_mean`：token 级平均熵；
- `eos_prob_final`：生成末尾位置的 EOS 概率；
- `top1_prob_mean`：各位置 top-1 token 的平均概率。

这些指标的作用不同。`score` 用来判断模型是否更准确；`response_length` 反映推理链是否变得更冗长或更简洁；`answer_tail_confidence` 与 `low_confidence_token_ratio` 则帮助判断模型在生成后段是否更稳定；`token_entropy_mean` 直接反映分布是否收缩；`eos_prob_final` 则用于观察模型是否学会在更合适的时机结束作答。通过把这些指标合在一起看，我们可以更有把握地回答“1-shot GRPO 到底改了什么”。

### 2.3 研究假设

结合实验设计与已有观察，我们提出如下假设：

1. `H1：1-shot GRPO 的主要作用是快速校准，而不是能力注入。`
2. `H2：Base 模型因先验较弱，因此收益空间更大；Instruct 模型因初始表现已高，收益会更小。`
3. `H3：训练收益会伴随 token entropy 降低、低置信 token 比例下降和答案尾部置信度上升。`
4. `H4：如果负对照设计有效，则 reward shuffle 不应复现 clean baseline 的收益。`
5. `H5：如果 group relative signal 很关键，则把 group size 从 8 降到 1 后，性能与行为变化应明显减弱。`

后文的分析将围绕这些假设展开。

---

## 3. 实验设置

### 3.1 模型与数据

本实验涉及两类模型：

- `Qwen2.5-Math-1.5B`
- `Qwen2.5-Math-1.5B-Instruct`

训练数据在脚本中分别对应：

- clean 数据：`data/dsr_sub/pi1_one_ans.parquet`
- reward-shuffle 数据：`data/dsr_sub/pi1_one_ans_reward_shuffle.parquet`

验证集统一来自：

- `data/testset/MATH500/test.parquet`

从命名上看，训练采用的是一个高度压缩的 one-shot 数据子集，这与本文关注的“极低样本强化学习”问题是一致的。训练模式统一为 `TRAIN_MODE=oneshot`。尽管这里的“1-shot”并不等同于只包含 1 条训练样本，但它代表一种极端低数据预算设定，这也是本文方法讨论的核心背景。

### 3.2 六实验矩阵

根据 `examples/custom/run_jobs/launch_6_exps_serial_queue.sh` 与各运行脚本，可还原出原始六实验矩阵如下：

| 编号 | 实验标签 | 模型 | 训练数据 | rollout_n | KL/Entropy 约束 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Exp1 | Base-C-n8 baseline | Base | clean | 8 | 开启 | 已完成，但最终正式目录需确认 |
| Exp2 | Base-C-n1 group ablation | Base | clean | 1 | 开启 | 已完成重跑，但验证口径与 n=8 版本不同 |
| Exp3 | Base-D weak constraint | Base | clean | 8 | 关闭 | 已完成 |
| Exp4 | Base-B reward shuffle | Base | reward-shuffle | 8 | 开启 | 已完成，但数据有效性待确认 |
| Exp5 | Instruct-B reward shuffle | Instruct | reward-shuffle | 8 | 开启 | 已完成，但数据有效性待确认 |
| Exp6 | Instruct-C-n8 baseline | Instruct | clean | 8 | 开启 | 已完成 |

结合现有目录，本报告暂使用如下产物作为分析对象：

| 报告内名称 | 当前对应目录 |
| --- | --- |
| Base-C-n8 baseline | `base_align_exp1_20260401_141352` |
| Base-C-n1 group ablation | `base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719` |
| Base-D weak constraint | `base_d_weak_constraint_20260402_003007` |
| Base-B reward shuffle | `base_b_reward_shuffle_20260402_132309` |
| Instruct-B reward shuffle | `instruct_b_reward_shuffle_20260403_002727` |
| Instruct-C-n8 baseline | `instruct_c_n8_baseline_rerun180_20260404_113902` |

这里有两点必须说明。第一，按当前本地运行脚本，`base_align_exp1_20260401_141352` 对应的是 `ROLLOUT_N=8`、`VAL_ROLLOUT_N=8`，因此更像正式的 `Base-C-n8 baseline`；而 `Base-C-n1 group ablation` 则对应单独的 `base_c_n1_group_ablation_*` 运行脚本。若你手头还有外部记录与此不一致，应以你的最终实验台账为准。第二，`reward-shuffle` 相关文件目前和 clean 文件 `md5` 完全一致，因此这部分在正文里只能先写成“名义上的负对照”，不能直接写成严格成立的负对照。

### 3.3 统一训练配置

从各运行脚本可见，标准实验配置具有较高一致性：

- 学习率：`1e-6`
- `train_batch_size=64`
- `ppo_mini_batch_size=32`
- `ppo_micro_batch_size_per_gpu=2`
- `test_freq=6`
- `total_epochs=10`
- 标准基线与 reward-shuffle 版本：`rollout_n=8`
- group ablation 版本：`rollout_n=1`
- 标准基线：`use_kl_loss=true`, `kl_loss_coef=0.001`, `entropy_coeff=0.001`
- weak constraint：`use_kl_loss=false`, `kl_loss_coef=0`, `entropy_coeff=0`

这种设计的优点是：我们能够把主要变量聚焦在三类因素上，即模型先验、group size、约束项与训练数据版本，而不用担心过多超参数同时变化。

### 3.4 指标口径

当前所有主要 run 的验证结果都覆盖 `step 0` 到 `step 180`，步长为 6。`rollout_n=8`、`val_rollout_n=8` 的实验，每个 step 对应约 `4000` 条验证记录；新完成的 `Base-C-n1 group ablation` 使用 `val_rollout_n=1`，因此每个 step 对应约 `500` 条验证记录。后文所有均值，均基于相应 step 的 case-level JSONL 统计得到，因此在比较 `n=1` 与 `n=8` 时应额外说明验证采样口径不同。

---

## 4. 主结果

### 4.1 总体结果表

表 1 汇总了当前最重要的主指标变化。

| 实验 | step0 score | final score | score 增量 | best step | best score |
| --- | --- | --- | --- | --- | --- |
| Base-C-n8 baseline | 0.3745 | 0.5733 | +0.1988 | 156 | 0.5753 |
| Base-C-n1 group ablation | 0.3580 | 0.5900 | +0.2320 | 180 | 0.5900 |
| Base-D weak constraint | 0.3745 | 0.5747 | +0.2002 | 174 | 0.5765 |
| Base-B reward shuffle | 0.3745 | 0.5745 | +0.2000 | 162 | 0.5755 |
| Instruct-B reward shuffle | 0.6837 | 0.6933 | +0.0095 | 174 | 0.6963 |
| Instruct-C-n8 baseline | 0.6837 | 0.6873 | +0.0035 | 162 | 0.6995 |

从这张表可以看出两个最直接的现象。第一，Base 模型在多个完整实验中都表现出近似 `+0.20` 的大幅提升，说明 1-shot GRPO 对基础模型的推理行为确实产生了显著影响。第二，Instruct 模型在相同训练框架下只出现了较小幅度的波动，提升量级仅为 `0.0035` 到 `0.0095`。这与我们的研究假设一致，即强先验模型的可提升空间更小。

### 4.2 Base 模型的收益最明显

Base-C-n8 baseline 从 `0.3745` 提升至 `0.5733`，说明在当前训练设定下，Base 模型从一个明显不够稳定的推理状态，被推向了一个更接近“可用”的水平。更重要的是，这种提升并不是孤例。无论是在弱约束版本还是在名义上的 reward-shuffle 版本中，最终分数都落在 `0.5745` 左右。这说明 Base 模型的收益具有很强的一致性。

如果只看这个结果，似乎可以得出一个乐观结论：1-shot GRPO 对 Base 模型非常稳健，甚至对约束和数据版本不太敏感。但这种解释需要保持谨慎。首先，reward-shuffle 数据目前被证实与 clean 数据完全一致，因此这组结果不能支持“打乱 reward 也能学到”的说法，反而只能说明这其实很可能还是 clean baseline 的一次重复。其次，弱约束版本虽然最终分数与标准基线接近，但这并不自动意味着约束项无效，因为约束可能主要影响训练过程中的稳定性、分布形态或最优 step 的出现位置，而不一定直接体现在 final score 上。

新完成的 `Base-C-n1 group ablation` 让讨论更加有意思。它从 `0.3580` 提升到 `0.5900`，最终甚至略高于当前 `Base-C-n8 baseline` 的 `0.5733`。如果只看这个数字，似乎会让人倾向于得出“group size 根本不重要”的结论。但这里必须非常克制：`n=1` 版本的 `val_rollout_n=1`，每个 step 只有 `500` 条验证记录，而 `n=8` 版本的 step 级均值来自 `4000` 条记录；此外，`n=1` 的正式终版是一次两卡重启后的新运行。因此，更稳妥的说法是：在当前实验里，把 group size 降到 1 并没有阻止模型获得显著提升，但是否因此可以否定 group-relative signal 的重要性，还需要进一步做严格口径对齐的补充实验。

### 4.3 Instruct 模型初始更强，但提升更小

Instruct-C-n8 baseline 在 step 0 就达到 `0.6837`，显著高于 Base 模型的 `0.3745`。这意味着 Instruct 模型本身已经具备更强的数学推理先验，因此 1-shot GRPO 能做的更像是微调与局部校准，而不是大幅纠偏。从最终结果看，Instruct baseline 的 final score 为 `0.6873`，最佳 step 可达 `0.6995`，但最终增益只有 `+0.0035`。Instruct-B reward shuffle 的 final score 为 `0.6933`，略高于 baseline，但考虑到 reward-shuffle 数据本身的有效性存在疑点，这个差异目前不能做过度解释。

这个现象对报告叙事很重要。它说明 1-shot GRPO 的收益大小，可能和模型初始先验密切相关。对于 Base 模型，训练主要在帮助它学会“更稳定地组织已有能力”；对于 Instruct 模型，训练更多是在已有高性能附近做细微修正。因此，我们不能用同一个收益尺度去评价两类模型。

### 4.4 最优 step 与 final step 不完全重合

值得注意的是，多个实验的最佳分数并不出现在最终 step 180，而出现在更早的 step。例如：

- Base-C-n8 baseline 的最佳分数出现在 `step 156`；
- Base-B reward shuffle 的最佳分数出现在 `step 162`；
- Instruct-C-n8 baseline 的最佳分数出现在 `step 162`；
- Instruct-B reward shuffle 的最佳分数出现在 `step 174`。

这说明在当前训练预算下，模型并不是单调提升的，而更像是在中后期进入波动区间。这一点后续可以在正式图中通过 `score-over-step` 曲线展示。对于正文而言，这个观察支持一个重要判断：在极低数据预算下，训练后期未必带来持续收益，因此如果把“最佳 checkpoint”与“最终 checkpoint”区别开来，可能更符合实际训练规律。

---

## 5. 机制分析：1-shot GRPO 到底改了什么

### 5.1 Base 模型从“高熵、低置信”走向“低熵、较稳定”

如果只看最终正确率，容易忽略行为层面的变化。以 Base-C-n8 baseline 为例，训练前后若干关键指标如下：

| 指标 | step 0 | final |
| --- | --- | --- |
| response_length | 812.5 | 603.1 |
| answer_tail_confidence | 0.8129 | 0.9331 |
| low_confidence_token_ratio | 0.1160 | 0.0325 |
| token_entropy_mean | 0.6752 | 0.1839 |
| eos_prob_final | 0.7340 | 0.9880 |
| top1_prob_mean | 0.8511 | 0.9354 |

这些数字非常有信息量。首先，`response_length` 明显缩短，说明训练后的模型不再需要像初始模型那样生成大量冗余内容来“摸索”答案。其次，`answer_tail_confidence` 明显上升，而 `low_confidence_token_ratio` 大幅下降，说明模型在后段生成时更加确定，不容易在关键推理位置出现犹豫。第三，`token_entropy_mean` 显著下降，意味着模型的输出分布更加集中。最后，`eos_prob_final` 上升到接近 1，说明模型在结束答案时更果断，减少了拖尾式生成。

综合这些变化，可以把 Base 模型的收益解释为：1-shot GRPO 并没有简单让模型“说更多”，而是让它“更快、更稳、更敢结束”。这非常符合“快速校准而非能力注入”的假设。

### 5.2 Weak constraint 版本的启示

Base-D weak constraint 的最终正确率与标准基线非常接近，但行为指标又有一些差异：

| 指标 | step 0 | final |
| --- | --- | --- |
| response_length | 812.5 | 578.6 |
| answer_tail_confidence | 0.8129 | 0.9437 |
| low_confidence_token_ratio | 0.1160 | 0.0294 |
| token_entropy_mean | 0.6752 | 0.1563 |
| eos_prob_final | 0.7340 | 0.9738 |
| top1_prob_mean | 0.8511 | 0.9437 |

可以看到，弱约束版本反而表现出更低的 entropy 和更高的 top-1 概率。这种现象至少说明一点：关闭 KL/entropy 约束后，模型依然能朝着更确定的方向收缩分布。问题在于，这种收缩究竟是“健康收缩”，还是更激进、风险更高的收缩？由于当前只看到了最终均值，还缺少更细粒度的 step 曲线和个案错误分析，因此我们不能仅凭 final score 就判断弱约束一定“不危险”。

更合理的写法是：在当前有限证据下，弱约束版本没有显著损害最终准确率，甚至在若干分布指标上表现得更激进；但是否因此意味着约束项可以省略，还需要更完整的 step-level 波动分析与失败样本分析来支撑。

### 5.3 Instruct 模型变化更小，说明其主要是局部校准

Instruct-C-n8 baseline 的 step 0 指标已经很强：

| 指标 | step 0 | final |
| --- | --- | --- |
| response_length | 473.2 | 470.3 |
| answer_tail_confidence | 0.9526 | 0.9571 |
| low_confidence_token_ratio | 0.0176 | 0.0187 |
| token_entropy_mean | 0.1126 | 0.1322 |
| eos_prob_final | 0.9947 | 0.9926 |
| top1_prob_mean | 0.9624 | 0.9634 |

与 Base 模型相比，Instruct 模型几乎没有“从混乱走向有序”的明显过程。它在起点就已经表现出较高置信度、较低 entropy 和极强的 EOS 结束能力。训练后的变化幅度很小，有些指标甚至略有反向波动，例如 `low_confidence_token_ratio` 与 `token_entropy_mean` 没有出现像 Base 模型那样大幅下降。这说明对 Instruct 模型而言，1-shot GRPO 的作用并不是大规模改变其生成分布，而是做少量局部调整。

这类结果在研究上很有启发性：同样的训练方法，对不同先验强度的模型，其作用机制可能完全不同。对于弱模型，训练更像“校正大方向”；对于强模型，训练更像“磨平局部毛刺”。

### 5.4 关于 EOS 与结束行为的解释

Base 模型训练后的 `eos_prob_final` 从约 `0.734` 提升到 `0.97` 到 `0.99`，这是一个非常显著的变化。直观地说，模型更愿意在生成末尾给出“我该结束了”的高置信信号。这通常意味着两种可能：

1. 模型更早形成了结构清晰、可以闭合的答案；
2. 模型不再依赖拖长推理链来弥补前面推理的不确定性。

由于 `response_length` 同时明显下降，因此第二种解释更有可能。也就是说，训练后的模型更擅长在合适的时机终止输出，而不是继续生成多余文本。对于数学题这类最终答案明确的任务，这是一个非常重要的改进，因为过长的后续推理往往会引入新的错误，甚至把原本正确的中间过程拖向错误结论。

### 5.5 Token 级追踪给出的直接证据

除了当前六实验的 case-level 指标外，仓库中还保留了一份更细粒度的 token 级补充分析，文件位于：

- `analysis/exp4_validation_20260331/REPORT_exp4_token_diagnostics_20260331_v1.md`
- `analysis/exp4_validation_20260331/token_outputs/tables/*.csv`
- `analysis/exp4_validation_20260331/token_outputs/plots/*.png`

这份材料并不对应当前六实验的全部 run，而是一次更早的四实验对比（`Base-Full`、`Base-OneShot`、`Instruct-Full`、`Instruct-OneShot`）。它的价值在于：虽然样本覆盖面有限，但它真的保留了 token 位置级的 `logprob`、`prob`、`entropy`、`eos_prob`、`top1_prob`、`top5_mass` 等轨迹，因此能更直接地支持“训练在改写早期 token 组织方式”的叙事。

需要先说明它的边界。结合当前运行脚本与 `validation/*.jsonl` 的实际落盘结果，更准确的说法是：每个 step 最多有 64 个样本参与 token-level diagnostics 统计，但只有其中 8 条样本会额外保存完整的 `token_diagnostics` 轨迹；在主 `n=8` 实验中，这 8 条完整轨迹通常对应同一固定 probe 题的 8 次采样，而每条记录只保留前缀大约前 96 个 token，`token_diagnostics_truncated=true`。因此，它更适合作为“机制证据”而非“全任务统计证据”。

尽管如此，它仍然提供了几个很强的观察。第一，instruct 系列在 step 0 就表现出更高 token confidence、更低 entropy 和更短输出；训练后变化很小，说明它们本来就处在较稳定的分布区域。第二，base 系列训练带来的变化更大，主要表现为前缀 token 的 logprob 提升、entropy 下降和输出缩短，说明训练确实在修正早期生成阶段的不稳定与冗长。第三，在 final step 上，`Base-OneShot` 仍然落后于 `Base-Full`，这说明对 base 模型而言，one-shot 方案能改善行为，但未必能把 token 级稳定化做到和 full 训练一样充分。

该补充分析中的若干具体数字也与本文主结论相互呼应。例如，在 traced probe 的 final step 上，`Base-Full` 的平均 token entropy 为 `0.2036`，而 `Instruct-Full` 为 `0.0778`；对应的 `top1_prob` 分别为 `0.9246` 与 `0.9705`。这说明 instruct 模型从起点到终点都更加“确信自己要输出什么”。从 `step 0 -> final` 的变化量看，`Base-Full` 的 `logprob` 增量为 `+0.3673`、`entropy` 下降 `-0.3908`；`Base-OneShot` 的 `logprob` 增量为 `+0.3849`、`entropy` 下降 `-0.3885`。相对地，`Instruct-Full` 的对应变化仅为 `+0.0744` 与 `-0.0760`，`Instruct-OneShot` 更小，仅为 `+0.0435` 与 `-0.0419`。这种“base 变化大、instruct 变化小”的格局，与我们在当前六实验均值表里看到的趋势高度一致。

因此，如果要把报告写得更有机制味道，可以把这份 token 级补充分析作为一个单独的小节来引用，其推荐表述是：

`现有 token 级追踪结果表明，训练对 base 模型的主要作用不是让其学会新的数学知识，而是让其在早期 token 组织、前缀分布收缩和答案收尾上变得更稳定；而 instruct 模型由于初始先验更强，其 token 级变化幅度明显更小。`

这个表述既能和当前六实验的统计结果对齐，又不会夸大单 probe token trace 的外推范围。

---

## 6. 消融与负对照分析

### 6.1 Group size 消融已完成，但当前仍不能做过强结论

原始实验设计里，`Base-C-n1 group ablation` 用于检验 `group relative signal` 是否关键。理论上，如果 `rollout_n=8` 的核心收益来自组内相对比较，那么把 `n` 降到 `1` 应当显著削弱 GRPO 的作用。当前这一部分已经有了新的完整重跑：`base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719` 覆盖 `step 0` 到 `step 180`，并成功保存了 `global_step_180` checkpoint。

从结果看，`Base-C-n1 group ablation` 的分数由 `0.3580` 提升到 `0.5900`，最佳 step 就在 `180`。这说明在当前设定下，即使把 `rollout_n` 和 `val_rollout_n` 都降到 `1`，训练依然能够学到东西，而且收益并不弱。

但这并不等于“group size 不重要”已经被证明。原因有三点：第一，`n=1` 与 `n=8` 的验证口径不同，前者每个 step 只有 `500` 条记录，后者是 `4000` 条。第二，`n=1` 的终版来自一次两卡重启后的新运行，其资源配置和运行轨迹与早期版本不同。第三，group size 影响的可能不仅是最终分数，还包括训练稳定性、方差和对不同样本的泛化质量。

因此，当前更准确的写法应当是：

`现有 n=1 重跑结果表明，group size 降到 1 并不会阻止模型获得显著收益；但由于验证采样口径与运行配置并未完全对齐，关于 group-relative signal 是否必要，仍需进一步做严格对齐实验才能下结论。`

### 6.2 Reward shuffle 负对照当前存在设计失效风险

原始设计中，reward shuffle 本应作为负对照：如果把训练信号打乱，模型理应无法获得与 clean baseline 同等质量的收益。这个设计非常重要，因为它直接关系到“训练收益是真学到了东西，还是只是偶然漂移”的解释力。

然而，当前仓库内的两个训练文件：

- `data/dsr_sub/pi1_one_ans.parquet`
- `data/dsr_sub/pi1_one_ans_reward_shuffle.parquet`

其 `md5` 值完全一致。这说明在当前文件层面，它们是字节级相同的文件。因此，现有 `Base-B reward shuffle` 与 `Instruct-B reward shuffle` 实际上不能被严格解释为“使用打乱奖励的数据训练得到”的结果。

这也解释了为什么 Base-B reward shuffle 的结果几乎与 clean baseline 等价。它更可能不是一个真正的负对照成立现象，而只是因为训练数据根本没有被打乱。对报告而言，这一点必须如实写出，否则会高估实验设计的说服力。

### 6.3 尽管如此，reward-shuffle 结果仍可作为“名义重复实验”参考

虽然 reward-shuffle 当前不能作为严格负对照，但在工程记录层面，它依然有一定参考价值。因为它至少表明：在同一训练管线下、在相近配置和不同运行时间点，Base 模型与 Instruct 模型都能复现与各自主实验相近的性能区间。这种“名义上的重复性”对于工程稳定性是有意义的。

但在正式论文式结论中，我们应该把它表述为：

`该实验在当前文件证据下更适合作为重复运行参考，而不宜作为严格负对照证据。`

---

## 7. 讨论与局限性

### 7.1 为什么 Base 模型收益远高于 Instruct 模型

最自然的解释是先验差异。Base 模型在初始状态下的推理模式更分散、更高熵，也更容易在关键位置犹豫，因此强化学习信号更容易带来方向性改进。相反，Instruct 模型已经在监督或指令对齐阶段获得了较强的行为先验，1-shot GRPO 只能在已有高质量分布周围进行微调，因此提升空间有限。这一现象本身并不意味着 Instruct 模型“不适合 RL”，而更可能说明：对高先验模型，低样本 RL 的边际收益天然较小。

### 7.2 这组实验最强的证据是什么

当前最强的证据并不是某一个 final score，而是 Base 模型在多个完整 run 中同时表现出的行为一致性：正确率提升、响应缩短、尾部置信度提升、低置信 token 减少、entropy 降低、EOS 结束更果断。这些现象共同构成了一个相互支持的证据链。它们共同指向同一个解释，即 1-shot GRPO 的收益很可能来自对已有推理行为的快速校准。

### 7.3 这组实验最弱的地方是什么

最弱的地方有两个。第一，虽然 `n=1` group ablation 现在已经有终版结果，但它与 `n=8` 主实验在验证采样口径上并未完全对齐，因此 group size 的机制解释仍不够干净。第二，reward-shuffle 负对照当前失效，削弱了“排除偶然收益”的说服力。因此，如果你后续还想把这篇报告往更论文式的方向推进，最优先补的仍然是这两部分。

### 7.4 对后续工作的直接建议

后续最值得补的工作有三项：

1. 补一组严格对齐的 `n=1` vs `n=8` 对比，尤其统一 `val_rollout_n` 和评估口径；
2. 确认真正的 reward-shuffle 数据，并重新执行 Exp4 与 Exp5；
3. 生成 step-level 曲线图与代表性 token trace 图，把“行为校准”的说法从数值表扩展到图形证据。

如果这三项补齐，这篇报告的说服力会显著提升。

---

## 8. 结论

本报告基于当前可核验的实验产物，对 1-shot GRPO 在数学推理模型上的作用进行了初步系统分析。结果显示，在 `Qwen2.5-Math-1.5B` 上，1-shot GRPO 可以带来接近 `+0.20` 的显著性能提升，同时伴随更短的响应、更高的尾部置信度、更低的 token entropy 与更稳定的结束行为。这说明其有效性更可能来自对已有推理能力的快速校准，而不是对新知识的注入。相比之下，`Qwen2.5-Math-1.5B-Instruct` 由于初始先验更强，提升空间明显更小，训练效果更接近局部微调。

与此同时，报告也指出了当前结论的边界。`n=1` group ablation 虽然已经完成，但由于与 `n=8` 主实验存在验证采样口径差异，关于 group relative signal 的机制论断仍需后续补充实验；reward-shuffle 负对照当前也缺乏真实有效的数据支撑。总体而言，当前证据已经足以支持“1-shot GRPO 对 Base 数学推理模型有效，且其收益伴随明显的分布层面收缩与行为稳定化”，但还不足以把所有机制问题完全盖棺定论。

---

## 附：当前可直接放入正文的数据表

### 表 A：行为指标汇总

| 实验 | final response length | final tail confidence | final low-conf ratio | final entropy | final EOS prob | final top1 prob |
| --- | --- | --- | --- | --- | --- | --- |
| Base-C-n8 baseline | 603.1 | 0.9331 | 0.0325 | 0.1839 | 0.9880 | 0.9354 |
| Base-C-n1 group ablation | 550.7 | 0.9163 | 0.0438 | 0.2411 | 0.9628 | 0.9205 |
| Base-D weak constraint | 578.6 | 0.9437 | 0.0294 | 0.1563 | 0.9738 | 0.9437 |
| Base-B reward shuffle | 531.3 | 0.9214 | 0.0357 | 0.1867 | 0.9688 | 0.9343 |
| Instruct-B reward shuffle | 486.1 | 0.9381 | 0.0323 | 0.2644 | 0.9929 | 0.9495 |
| Instruct-C-n8 baseline | 470.3 | 0.9571 | 0.0187 | 0.1322 | 0.9926 | 0.9634 |

### 表 B：需要在正式稿中加脚注说明的事实

1. `Base-C-n1 group ablation` 当前正式采用的是 `base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719`，它已经完整到 `global_step_180`，但 `val_rollout_n=1`，因此与 `n=8` 主实验的评估口径并不完全一致。
2. `pi1_one_ans.parquet` 与 `pi1_one_ans_reward_shuffle.parquet` 当前 `md5` 完全一致。
3. `Base-C-n8 baseline` 的最终正式目录尚待你确认是否就是 `base_align_exp1_20260401_141352`。
   但就当前仓库内脚本证据看，它更像 `n=8` 而不是 `n=1`。

---

## 我建议的下一步

如果你同意，我下一步可以直接继续做下面两件事中的任意一种：

1. 把这份草稿改写成`更像正式提交稿`的版本，语言更凝练、结构更论文化。
2. 继续本地生成图表与表格，把它扩成接近最终版的 25 页正文。
