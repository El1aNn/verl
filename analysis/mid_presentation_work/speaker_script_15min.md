# Midterm Presentation Speaker Script

对应文件：`slide.tex` 当前 21 页版本  
目标时长：约 10 分钟  
使用方式：每页只讲核心结论；图表页先说“怎么看”，再说“一句话结论”。中文和英文逐页对应，任选一种语言讲即可。

---

## 1. Title Page

### 中文讲法

各位老师好，我是 Zihang Xu。今天汇报我的 capstone midterm progress，题目是 Behavioral Calibration for Mathematical Reasoning under Low-Data Reinforcement Learning。

这次汇报围绕一个问题：1-shot GRPO 是否真的提升数学推理？如果有，它的收益来自哪里，机制又是什么？

### English version

Good morning or good afternoon everyone. My name is Zihang Xu. Today I will present my capstone midterm progress on behavioral calibration for mathematical reasoning under low-data reinforcement learning.

The talk focuses on one question: does 1-shot GRPO really help mathematical reasoning, where does the benefit come from, and what mechanism explains it?

---

## 2. Three Questions

### 中文讲法

我把整个汇报组织成三个问题：第一，1-shot GRPO 有没有收益；第二，收益来自 reward、data、group size 还是 model family；第三，它的机制是学习新知识，还是校准已有推理行为。

我的结论是：在 Base 模型上，1-shot GRPO 的收益是真实的，但更像 behavioral calibration，而不是从一个样本里学习全新数学知识。

### English version

I organize the talk around three questions. First, does 1-shot GRPO help? Second, does the benefit come from reward semantics, data budget, group size, or model family? Third, is the mechanism new knowledge learning or calibration of existing reasoning behavior?

My conclusion is that the gain is real for the Base model, but it is better explained as behavioral calibration rather than learning new mathematics from one example.

---

## 3. Evidence Design

### 中文讲法

为了回答这三个问题，我用了两层设计。第一层是 controlled comparisons：Base vs. Instruct、1-shot vs. full-data、n=8 vs. n=1、clean reward vs. random reward。

第二层是 evidence layers：score curve 说明有没有提升，control 说明收益来源，token metrics 和 full-token trace 说明机制。

### English version

To answer these questions, I use two layers of evidence. The first layer is controlled comparison: Base versus Instruct, 1-shot versus full-data, n=8 versus n=1, and clean reward versus random reward.

The second layer is evidence type: score curves test whether the gain exists, controls identify its source, and token metrics plus full-token traces test the mechanism.

---

## 4. Controls for the Three Questions

### 中文讲法

这七组设置分别对应不同解释。Base n=8 是主实验；Base full-data 看数据预算；Base n=1 看 group-relative signal；weak-reg 看是否只是 regularization；random-reward 是负对照；Instruct 两组看 model family。

所以这页的重点不是设置很多，而是每个设置都在排除一种可能解释。

### English version

The seven settings each test a possible explanation. Base n=8 is the main run; Base full-data tests data budget; Base n=1 tests group-relative signal; weak-reg tests regularization; random-reward is the negative control; and the Instruct runs test model family.

The point is not to add many experiments, but to isolate possible sources of the gain.

---

## 5. Question 1: Does 1-Shot GRPO Help?

### 中文讲法

这张图上半部分保留 random-reward，下半部分是 non-collapsed settings 的局部放大。

结论很直接：Base n=8 的 1-shot GRPO 明显提升，而 random-reward 很快崩溃。所以收益不是“跑 RL 本身”带来的，而依赖有意义的 reward signal。

### English version

This figure has a full view on top, including random reward, and a zoomed view of the non-collapsed settings at the bottom.

The conclusion is direct: Base n=8 improves clearly under 1-shot GRPO, while random reward collapses quickly. So the gain is not just a generic effect of running RL; it depends on meaningful reward signal.

---

## 6. Answer 1: The Gain Is Real but Conditional

### 中文讲法

从数字看，Base n=8 的 best score 是 0.603，final 是 0.583；Base full-data final 到 0.611，说明 full-data 仍然更强，但 1-shot 已经获得了相当一部分收益。

同时 Base n=1 更弱，random-reward final 为 0。这说明收益真实，但有条件：需要 reward 语义，也受 group signal 影响。

### English version

Numerically, Base n=8 reaches 0.603 best accuracy and 0.583 final accuracy. Base full-data ends at 0.611, so full-data is still stronger, but 1-shot captures a substantial part of the gain.

Base n=1 is weaker, and random reward ends at zero. So the gain is real, but conditional: it needs meaningful reward semantics and is affected by group signal.

---

## 7. Question 2: How Much Comes From Data Budget?

### 中文讲法

这里比较 one-shot 和 full-data。Base 上 full-data final 比 one-shot 高 0.0373；Instruct 上高 0.0145。

所以 one-shot 有价值，但不是 full-data 的替代。更准确地说，one-shot 能触发同方向的 calibration，而 full-data 进一步提高 endpoint。

### English version

This slide compares one-shot and full-data training. For the Base family, full-data is 0.0373 higher at the final checkpoint. For the Instruct family, the gap is 0.0145.

So one-shot is useful, but not a replacement for full-data training. A more accurate interpretation is that one-shot triggers the same calibration direction, while full-data further improves the endpoint.

---

## 8. Question 3: What Changes in the Output Distribution?

### 中文讲法

从这里开始看机制。成功设置不仅分数提高，token behavior 也变了：response 更短或更稳定，entropy 更低，top-1 probability 更高，低置信 token 更少。

这说明收益不只是多答对几题，而是生成分布变得更稳定。

### English version

From here, I move to mechanism. In successful settings, the score improves and token behavior also changes: responses become shorter or more stable, entropy decreases, top-1 probability increases, and low-confidence tokens decrease.

This means the gain is not only more correct answers; the generation distribution becomes more stable.

---

## 9. Mechanism Signal: Correct Responses Are Better Calibrated

### 中文讲法

这一页把 correct 和 wrong responses 分开。主要现象是：正确响应通常 entropy 更低、confidence 更高。

这把 accuracy gain 和 calibration 连接起来了：正确答案不是随机碰出来的，而是对应更稳定的 token distribution。

### English version

This slide separates correct and wrong responses. The main pattern is that correct responses usually have lower entropy and higher confidence.

This connects accuracy gain to calibration: correct answers are not just lucky samples; they correspond to more stable token distributions.

---

## 10. Full-Token Audit: Testing the Mechanism Directly

### 中文讲法

这里是 full-token audit：对 Base full-data 的 step 0、60、120、180，在 held-out problems 上记录每个生成 token 的指标。

最关键的是 step 0 到 60：score 从 0.305 到 0.617，length 和 entropy 都大幅下降。这说明主要变化很早出现，而且不是靠更长探索产生的。

### English version

This is the full-token audit. I evaluate Base full-data checkpoints at steps 0, 60, 120, and 180, and record metrics for every generated token on held-out problems.

The key transition is from step 0 to 60: score rises from 0.305 to 0.617, while length and entropy drop sharply. So the main change appears early and is not produced by longer exploration.

---

## 11. Mechanism Over Checkpoints

### 中文讲法

这页把 checkpoint dynamics 画出来。step 60 时，分数上升，同时 length、entropy 和 low-confidence mass 下降。

这更像快速校准已有行为，而不是从一个样本里慢慢学习大量新知识。

### English version

This slide visualizes checkpoint dynamics. By step 60, score increases while length, entropy, and low-confidence mass all decrease.

This looks more like rapid calibration of existing behavior than slow learning of broad new knowledge from one example.

---

## 12. Complete Entropy Trace

### 中文讲法

这是一个 score=1 的完整 trace。每个词块是真实生成文本，背景颜色是 token entropy，暖色代表更高不确定性。

它完整推理到 `\boxed{28}`。我想强调的是：full-token evidence 可以落到具体轨迹上看，而且高 entropy 更集中在转折和推理选择附近，而不是均匀分布。

### English version

This is a complete score-1 trace. Each word block is generated text, and the background color is token entropy. Warmer colors indicate higher uncertainty.

The trace reaches `\boxed{28}`. The point is that full-token evidence can be inspected inside an actual reasoning trajectory, and high entropy tends to appear around transitions and reasoning choices rather than uniformly across the response.

---

## 13. Mechanism Interpretation

### 中文讲法

full-token audit 给出四个机制信号：收益不是靠更长输出；同分 checkpoint 行为不同；正确响应更低熵更高 top-1；EOS 变化说明不是简单 early stopping。

所以机制不是单一指标，而是 confidence、entropy、length 和 stopping behavior 的共同变化。

### English version

The full-token audit gives four mechanism signals: the gain is not from longer outputs; checkpoints with the same score can behave differently; correct responses have lower entropy and higher top-1; and EOS changes show that it is not simply early stopping.

So the mechanism is not one metric, but a coordinated shift in confidence, entropy, length, and stopping behavior.

---

## 14. Where the Gain Appears in Training

### 中文讲法

这里看 gain 的时间位置。大部分有效提升出现在训练早期。

这支持 calibration 解释：训练很快把模型推向更稳定的生成区域，而不是从一个样本中逐步学出新能力。

### English version

This slide asks when the gain appears. Most useful improvement happens early in training.

This supports the calibration account: training quickly moves the model into a more stable generation region, rather than gradually learning a new capability from one sample.

---

## 15. Why Accuracy Alone Is Not Enough

### 中文讲法

这一页说明为什么不能只看 accuracy。两个 checkpoint 分数相近，但 entropy、low-confidence mass 和 length 可能不同。

所以好的 endpoint 应该同时高分、稳定、低不确定性。

### English version

This slide explains why accuracy alone is not enough. Two checkpoints can have similar scores but different entropy, low-confidence mass, and length.

So a good endpoint should be high-scoring, stable, and low-uncertainty at the same time.

---

## 16. Mechanism Evidence: Score Gain Couples With Calibration

### 中文讲法

这页看 score gain 和 calibration indicators 是否一起变化。

结果是，分数提升通常伴随 entropy、low-confidence ratio 和 length 的改善。它不是某个单一指标决定分数，而是一组行为指标共同移动。

### English version

This slide asks whether score gain moves together with calibration indicators.

The result is that score improvement often comes with improvements in entropy, low-confidence ratio, and length. No single metric explains accuracy, but the metrics move together as a behavioral pattern.

---

## 17. Source of the Gain: Reward Semantics Matter

### 中文讲法

这一页回到 reward semantics。clean runs 会降低不确定性，但 random-reward 直接崩溃。

同时 weak-reg 仍然有竞争力，所以机制不是简单压低 entropy，而是在有意义奖励下，把概率质量推向更可靠的 reasoning behavior。

### English version

This slide returns to reward semantics. Clean runs reduce uncertainty, while random reward collapses.

At the same time, weak regularization remains competitive. So the mechanism is not simply lowering entropy; it is using meaningful reward to move probability mass toward more reliable reasoning behavior.

---

## 18. Answering the Three Questions

### 中文讲法

现在回答三问。第一，1-shot GRPO 在 Base 模型上有明显收益。第二，收益依赖 reward semantics，也受 group signal 和 data budget 影响。第三，机制最像 behavioral calibration。

一句话总结：1-shot GRPO 不是从一个样本里教会模型数学，而是让模型更稳定地表达已有推理能力。

### English version

Now I answer the three questions. First, 1-shot GRPO gives a clear benefit for the Base model. Second, the benefit depends on reward semantics and is affected by group signal and data budget. Third, the mechanism looks like behavioral calibration.

In one sentence: 1-shot GRPO does not teach the model mathematics from one example; it helps the model express existing reasoning ability more stably.

---

## 19. Boundaries of the Claim

### 中文讲法

这个结论有边界：目前主要是 single-run evidence；random-reward 只在 Base family 做了匹配；token diagnostics 覆盖还不完全均匀；模型也集中在 Qwen2.5-Math-1.5B。

所以这是一个有范围的机制性结论，不是对所有 RLVR 的普遍声明。

### English version

The claim has boundaries. The current evidence is mainly single-run; the random-reward control is matched only for the Base family; token diagnostics are not equally complete for every setting; and the model is Qwen2.5-Math-1.5B.

So this is a scoped mechanistic claim, not a universal statement about all RLVR settings.

---

## 20. Next Steps: Strengthening the Mechanism Claim

### 中文讲法

下一步我会补 multi-seed，尤其是 n=1 vs. n=8 和 one-shot vs. full-data；然后按题目难度和 topic 分层；最后把 full-token audit 扩展到更多设置。

长期目标是看这个 calibration framework 能否扩展到更大模型、code reasoning 和 scientific QA。

### English version

Next, I will add multi-seed replications, especially for n=1 versus n=8 and one-shot versus full-data. I will also stratify problems by difficulty and topic, and expand full-token audits to more settings.

Longer term, I want to test whether this calibration framework scales to larger models, code reasoning, and scientific QA.

---

## 21. Thank You

### 中文讲法

我的汇报到这里结束。

总结一句话：1-shot GRPO 在 Base 数学模型上有真实收益，但最合理的解释是 behavioral calibration，也就是有意义的 reward signal 让模型更稳定地表达已有推理能力。谢谢各位老师，欢迎提问。

### English version

This concludes my presentation.

To summarize, 1-shot GRPO gives a real benefit for the Base math model, but the most plausible explanation is behavioral calibration: meaningful reward signal helps the model express existing reasoning ability more stably. Thank you, and I welcome your questions.

---

# Backup Q&A / 备用问答

## Q1. 为什么说是 behavioral calibration，而不是 new knowledge learning？ / Why call this behavioral calibration rather than new knowledge learning?

### 中文回答

因为训练只有一个样本，但提升出现在 held-out MATH500 上，并伴随 length、entropy、top-1 probability、low-confidence ratio 的系统变化。这更像重新校准已有能力的表达概率。

### English answer

Because the model sees only one training example, but the improvement appears on held-out MATH500 problems and comes with systematic changes in length, entropy, top-1 probability, and low-confidence ratio. This looks like recalibrating the expression of existing ability.

## Q2. one-shot 和 full-data 到底差多少？ / How different are one-shot and full-data training?

### 中文回答

matched protocol 下，Base full-data final 比 Base one-shot 高 0.0373，Instruct full-data 高 0.0145。所以 one-shot 有价值，但不是 full-data 的完全替代。

### English answer

Under the matched protocol, Base full-data is 0.0373 higher than Base one-shot at the final checkpoint, and Instruct full-data is 0.0145 higher than Instruct one-shot. So one-shot is useful, but not a complete substitute for full-data training.

## Q3. random reward control 说明什么？ / What does the random-reward control show?

### 中文回答

它说明 reward semantics 很重要。如果只是运行 RL 就能提升，random reward 也可能提升；但它崩溃了，说明 clean gain 依赖有意义的奖励。

### English answer

It shows that reward semantics matter. If simply running RL were enough, random reward might also improve. Instead, it collapses, so the clean gain depends on meaningful reward signal.

## Q4. 为什么要看 token entropy？ / Why analyze token entropy?

### 中文回答

accuracy 只告诉我们答对多少，不能说明生成过程是否稳定。token entropy 和 top-1 probability 能刻画生成时的概率分布，因此能支持机制解释。

### English answer

Accuracy tells us how many answers are correct, but not whether generation is stable. Token entropy and top-1 probability describe the probability distribution during generation, so they help explain the mechanism.

## Q5. 完整 trace 这一页怎么讲最省时间？ / How should I present the complete trace slide quickly?

### 中文回答

不用读完整推理。只说这是一个 score=1 的真实生成；背景是 entropy；它完整到 `\boxed{28}`；暖色主要出现在推理选择和转折附近。

### English answer

Do not read the whole trace. Say that this is a real score-1 generation, the background is entropy, it reaches `\boxed{28}`, and warm colors mainly appear around reasoning choices and transitions.
