# Final Project Speaker Script

对应文件：`slide.tex` 当前 17 页版本  
目标时长：约 10 分钟  
使用方式：每页只讲最核心的结论；图表页先说“这张图怎么看”，再说“一句话 takeaway”。中文和英文逐页对应，任选一种语言讲即可。

---

## 1. Title Page

### 中文讲法

各位老师好，我是 Zihang Xu。今天我想汇报一下我的 final project，题目是 Behavioral Calibration for Mathematical Reasoning under Low-Data Reinforcement Learning。

这次汇报其实围绕一个很简单的问题：1-shot GRPO 到底是不是真的能提升数学推理？如果能，它到底是怎么起作用的？

### English version

Good morning or good afternoon everyone. My name is Zihang Xu. Today I will present my final project, titled Behavioral Calibration for Mathematical Reasoning under Low-Data Reinforcement Learning.

The whole talk is built around one simple question: does 1-shot GRPO actually improve mathematical reasoning? If it does, what is really driving that improvement?

---

## 2. Three Questions

### 中文讲法

我把这个问题拆成三步来看。第一，1-shot GRPO 有没有真实收益；第二，这个收益来自 reward、data、group size，还是 model family；第三，它到底是在学新知识，还是在校准模型原本就有的推理行为。

我的主要结论是：在 Base 模型上，1-shot GRPO 的提升是存在的。但它更像是 behavioral calibration，而不是模型真的从一个样本里学会了新的数学知识。

### English version

I break the main question into three parts. First, does 1-shot GRPO give a real gain? Second, does that gain come from reward semantics, data budget, group size, or model family? Third, is the model learning new knowledge, or is it calibrating reasoning behavior it already has?

My main conclusion is that the gain is real for the Base model. But it looks more like behavioral calibration than learning new mathematics from one example.

---

## 3. Evidence Design

### 中文讲法

为了把这件事讲清楚，我用了两层证据。第一层是 controlled comparisons，比如 Base 和 Instruct、1-shot 和 full-data、n=8 和 n=1、clean reward 和 random reward。

第二层是不同类型的 evidence：score curve 告诉我们有没有提升，control 帮我们判断收益来自哪里，token metrics 和 full-token trace 用来进一步解释机制。

### English version

To make the story cleaner, I use two layers of evidence. The first layer is controlled comparison: Base versus Instruct, 1-shot versus full-data, n=8 versus n=1, and clean reward versus random reward.

The second layer is evidence type. Score curves tell us whether there is a gain. Controls help us locate where the gain comes from. Token metrics and full-token traces help us understand the mechanism.

---

## 4. Controls for the Three Questions

### 中文讲法

这里一共有七组设置，每一组都对应一种可能解释。Base n=8 是主实验；Base full-data 看数据量的影响；Base n=1 看 group-relative signal；weak-reg 看是不是只是 regularization；random-reward 是负对照；Instruct 两组看 model family。

所以这页我想强调的不是“实验很多”，而是每个实验都在排除一个可能的解释。

### English version

There are seven settings here, and each one tests a possible explanation. Base n=8 is the main run. Base full-data tests data budget. Base n=1 tests the group-relative signal. Weak-reg tests whether this is just regularization. Random-reward is the negative control. The two Instruct runs test model family.

So the point of this slide is not that I ran many settings. The point is that each setting rules out a different possible explanation.

---

## 5. Question 1: Does 1-Shot GRPO Help?

### 中文讲法

这张图上半部分保留了 random-reward，下半部分把没有崩掉的设置放大来看。

结论比较直接：Base n=8 的 1-shot GRPO 确实有明显提升，而 random-reward 很快就崩了。所以这个收益不是“只要跑 RL 就会有”，它依赖的是有意义的 reward signal。

### English version

The top part keeps the random-reward run in the plot, and the bottom part zooms in on the settings that do not collapse.

The takeaway is pretty direct: Base n=8 improves clearly with 1-shot GRPO, while random reward collapses quickly. So the gain is not just what happens when we run RL. It depends on a meaningful reward signal.

---

## 6. Answer 1: The Gain Is Real but Conditional

### 中文讲法

从数字上看，Base n=8 的 best score 是 0.603，final 是 0.583。Base full-data 的 final 到 0.611，说明 full-data 还是更强，但 1-shot 已经拿到了相当一部分收益。

另外，Base n=1 更弱，random-reward final 是 0。放在一起看，结论就是：收益是真实的，但它有条件，需要 reward 有语义，也会受到 group signal 的影响。

### English version

Numerically, Base n=8 reaches 0.603 best accuracy and 0.583 final accuracy. Base full-data ends at 0.611, so full-data is still stronger, but 1-shot already captures a large part of the gain.

At the same time, Base n=1 is weaker, and random reward ends at zero. Taken together, the gain is real, but conditional. It needs meaningful reward semantics, and it is also affected by the group signal.

---

## 7. Question 2: How Much Comes From Data Budget?

### 中文讲法

这一页主要比较 one-shot 和 full-data。Base 上，full-data final 比 one-shot 高 0.0373；Instruct 上，高 0.0145。

所以我的理解是，one-shot 是有用的，但它不是 full-data 的替代品。更准确地说，one-shot 能把模型往同一个 calibration 方向推，而 full-data 会把最终 endpoint 再往上推一点。

### English version

This slide compares one-shot and full-data training. For the Base family, full-data is 0.0373 higher at the final checkpoint. For the Instruct family, the gap is 0.0145.

So my interpretation is that one-shot is useful, but it is not a replacement for full-data training. More precisely, one-shot pushes the model in the same calibration direction, while full-data pushes the final endpoint further.

---

## 8. Question 3: What Changes in the Output Distribution?

### 中文讲法

从这里开始，我会看机制。成功的设置不只是分数提高，token-level behavior 也变了：response 变短或者更稳定，entropy 更低，top-1 probability 更高，低置信 token 也更少。

这说明模型不是简单多蒙对了几题，而是生成分布本身变得更稳定了。

### English version

From here, I move to the mechanism. In the successful settings, the score improves, but token-level behavior also changes: responses become shorter or more stable, entropy goes down, top-1 probability goes up, and there are fewer low-confidence tokens.

So the model is not just getting a few more answers right by chance. Its generation distribution itself becomes more stable.

---

## 9. Mechanism Signal: Correct Responses Are Better Calibrated

### 中文讲法

这一页把 correct responses 和 wrong responses 分开来看。主要现象是：正确答案通常 entropy 更低，confidence 更高。

这就把 accuracy gain 和 calibration 连起来了。也就是说，正确答案不是随机采样碰出来的，它们对应的是更稳定的 token distribution。

### English version

This slide separates correct and wrong responses. The main pattern is that correct responses usually have lower entropy and higher confidence.

This connects the accuracy gain to calibration. In other words, correct answers are not just lucky samples. They come from more stable token distributions.

---

## 10. Full-Token Audit: Testing the Mechanism Directly

### 中文讲法

这里是 full-token audit。我取的是 Base n=8 1-shot 这条 run 的 step 0、60、120、180，在 held-out problems 上记录每一个生成 token 的指标。

最关键的是 step 0 到 60：score 从 0.305 到 0.617，同时 length 和 entropy 都明显下降。这说明主要变化很早就发生了，而且不是靠生成更长、探索更多才得到的。

### English version

This is the full-token audit. I take checkpoints from the Base n=8 1-shot run at steps 0, 60, 120, and 180, and record metrics for every generated token on held-out problems.

The key transition is from step 0 to 60: score rises from 0.305 to 0.617, while length and entropy both drop sharply. So the main change happens early, and it is not coming from longer generation or more exploration.

---

## 11. Mechanism Over Checkpoints

### 中文讲法

这页把 checkpoint dynamics 画出来。到 step 60 的时候，分数上去了，同时 length、entropy 和 low-confidence mass 都下来了。

这个模式更像是模型很快把已有行为校准好了，而不是从一个样本里慢慢学出大量新知识。

### English version

This slide shows the checkpoint dynamics. By step 60, the score goes up, while length, entropy, and low-confidence mass all go down.

This pattern looks more like the model quickly calibrating existing behavior than slowly learning broad new knowledge from one example.

---

## 12. Complete Entropy Trace

### 中文讲法

这里是一个 score=1 的完整 trace。每个词块都是真实生成的文本，背景颜色表示 token entropy，颜色越暖，不确定性越高。

这个 trace 最后推到了 `\boxed{28}`。我想强调的是，full-token evidence 不是只能看平均数，也可以落到一条具体轨迹上看。而且高 entropy 主要出现在推理转折和选择附近，不是均匀散在整段回答里。

### English version

This is a complete score-1 trace. Each word block is real generated text, and the background color shows token entropy. Warmer colors mean higher uncertainty.

The trace reaches `\boxed{28}`. The point I want to make is that full-token evidence is not only about averages. We can inspect it inside an actual reasoning trajectory. Also, high entropy tends to appear around transitions and reasoning choices, not uniformly across the whole response.

---

## 13. Wrong Trace: High Entropy Marks the Failure Point

### 中文讲法

这一页换成一个错误 trace。这个例子没有 Python，也没有很奇怪的 verification，它就是一道正弦函数读图题，所以错误更清楚。

模型前面其实做对了：振幅是 2，周期对应 `b=3`，中线是 `d=1`。真正出错的是它说 “the graph reaches its maximum at x=0”。这里高 entropy 正好也集中在 `graph reaches`、`seems`、phase 这些读图和相位判断附近。

但实际函数在 `x=0` 的值是 `2 sin(pi)+1=1`，这是中线，不是最大值。模型一旦把这里看成最大值，就会推出 `sin(c)=1`，最后得到错误的 `pi/2`。所以这一页想说明：高熵有时会提前暴露“错误前提”在哪里，而不只是告诉我们最终答案错了。

### English version

This slide shows a wrong trace. It is cleaner than the previous example because there is no Python verification. It is just a sine-graph reading problem, so the failure is easier to locate.

The model gets several early facts right: the amplitude is 2, the period gives `b=3`, and the midline gives `d=1`. The real mistake is the sentence “the graph reaches its maximum at x=0.” The high-entropy words also concentrate around `graph reaches`, `seems`, and the phase or graph-reading choices.

But in the actual function, at `x=0`, we have `2 sin(pi)+1=1`, which is the midline, not the maximum. Once the model treats it as a maximum, it derives `sin(c)=1` and ends with the wrong answer `pi/2`. So this trace shows that high entropy can point to the mistaken premise before the final answer appears.

---

## 14. Mechanism Interpretation

### 中文讲法

full-token audit 给了几个机制信号：收益不是靠更长输出；分数差不多的 checkpoint 行为也可能不同；正确响应通常低熵、高 top-1；错误 trace 里，高熵也可以帮我们定位错误前提；EOS 的变化也说明这不是简单 early stopping。

所以这里的机制不是某一个指标单独在起作用，而是 confidence、entropy、length 和 stopping behavior 一起发生了变化。

### English version

The full-token audit gives several mechanism signals: the gain is not from longer outputs; checkpoints with similar scores can still behave differently; correct responses usually have lower entropy and higher top-1; wrong traces can reveal high-entropy mistaken premises; and EOS changes show that this is not simply early stopping.

So the mechanism is not a single metric acting alone. It is a coordinated shift in confidence, entropy, length, and stopping behavior.

---

## 15. Source of the Gain: Reward Semantics Matter

### 中文讲法

这一页回到 reward semantics。clean runs 会降低不确定性，但 random-reward 直接崩溃。

同时，weak-reg 仍然有竞争力。所以机制不是简单地把 entropy 压低，而是在有意义的奖励下，把概率质量推向更可靠的 reasoning behavior。

### English version

This slide returns to reward semantics. Clean runs reduce uncertainty, while random reward collapses.

At the same time, weak regularization remains competitive. So the mechanism is not just lowering entropy. With a meaningful reward, the model moves probability mass toward more reliable reasoning behavior.

---

## 16. Answering the Three Questions

### 中文讲法

现在回到开头的三个问题。第一，1-shot GRPO 在 Base 模型上确实有明显收益。第二，这个收益依赖 reward semantics，也会受到 group signal 和 data budget 的影响。第三，从证据看，机制最像 behavioral calibration。

用一句话总结就是：1-shot GRPO 不是从一个样本里教会模型数学，而是让模型更稳定地表达它原本已经有的推理能力。

### English version

Now I return to the three questions from the beginning. First, 1-shot GRPO gives a clear benefit for the Base model. Second, the benefit depends on reward semantics and is also affected by group signal and data budget. Third, the mechanism looks most like behavioral calibration.

In one sentence: 1-shot GRPO does not teach the model mathematics from one example. It helps the model express reasoning ability it already has more stably.

---

## 17. Thank You

### 中文讲法

我的汇报到这里就结束了。

最后再总结一句：1-shot GRPO 在 Base 数学模型上确实有收益，但最合理的解释是 behavioral calibration。也就是说，有意义的 reward signal 让模型更稳定地表达已有的推理能力。谢谢各位老师，欢迎提问。

### English version

That concludes my presentation.

To summarize one last time, 1-shot GRPO gives a real benefit for the Base math model, but the most plausible explanation is behavioral calibration. In other words, a meaningful reward signal helps the model express its existing reasoning ability more stably. Thank you, and I welcome your questions.

---

# Backup Q&A / 备用问答

## Q1. 为什么说是 behavioral calibration，而不是 new knowledge learning？ / Why call this behavioral calibration rather than new knowledge learning?

### 中文回答

因为训练里只有一个样本，但提升出现在 held-out MATH500 上，而且 length、entropy、top-1 probability、low-confidence ratio 都一起发生了系统变化。这更像是在重新校准模型已有能力的表达方式。

### English answer

Because the model sees only one training example, but the improvement appears on held-out MATH500 problems. At the same time, length, entropy, top-1 probability, and low-confidence ratio all change systematically. That looks more like recalibrating the expression of existing ability.

## Q2. one-shot 和 full-data 到底差多少？ / How different are one-shot and full-data training?

### 中文回答

在 matched protocol 下，Base full-data final 比 Base one-shot 高 0.0373，Instruct full-data 高 0.0145。所以 one-shot 有价值，但它不能完全替代 full-data。

### English answer

Under the matched protocol, Base full-data is 0.0373 higher than Base one-shot at the final checkpoint, and Instruct full-data is 0.0145 higher than Instruct one-shot. So one-shot is useful, but it is not a complete substitute for full-data training.

## Q3. random reward control 说明什么？ / What does the random-reward control show?

### 中文回答

它说明 reward semantics 很重要。如果只是“跑 RL”本身就能提升，那 random reward 也可能会提升；但它最后崩了，所以 clean gain 依赖的是有意义的奖励。

### English answer

It shows that reward semantics matter. If simply running RL were enough, random reward might also improve. Instead, it collapses, so the clean gain depends on a meaningful reward signal.

## Q4. 为什么要看 token entropy？ / Why analyze token entropy?

### 中文回答

accuracy 只能告诉我们答对了多少，但不能告诉我们生成过程稳不稳定。token entropy 和 top-1 probability 能描述生成时的概率分布，所以它们可以帮助解释机制。

### English answer

Accuracy tells us how many answers are correct, but it does not tell us whether generation is stable. Token entropy and top-1 probability describe the probability distribution during generation, so they help explain the mechanism.

## Q5. 完整 trace 这一页怎么讲最省时间？ / How should I present the complete trace slide quickly?

### 中文回答

不用读完整推理。只说这是一个 score=1 的真实生成；背景是 entropy；它最后到 `\boxed{28}`；暖色主要出现在推理选择和转折附近。

### English answer

Do not read the whole trace. Say that this is a real score-1 generation, the background is entropy, it reaches `\boxed{28}`, and warm colors mainly appear around reasoning choices and transitions.

## Q6. 错误 trace 这一页怎么讲？ / How should I present the wrong trace?

### 中文回答

重点讲错误发生在 `graph reaches its maximum at x=0` 这个前提上。高 entropy 也集中在 `graph reaches` 和 phase 判断附近，所以它不是只在最终答案处才显示不确定，而是在错误推理转折点就已经有信号。

### English answer

Focus on the mistaken premise: `the graph reaches its maximum at x=0`. The high-entropy tokens cluster around `graph reaches` and the phase decision, so uncertainty appears at the reasoning turn, not only at the final answer.
