# 4次实验 Validation 对比分析（初版）

## 1. 数据范围
- 分析对象：4 个实验 x 31 个 step（0 到 180）
- Case 粒度：每个 step 4000 条采样（约 500 题 x 8 次采样），final step 可对齐 case 数量：4000
- 统计口径：对每个 step 的 case 指标取均值；case 对比以 final step 为主

## 2. 总体结论
- final score 最优实验：**Instruct-Full**（0.7070）
- best score 最优实验：**Instruct-Full**（0.7070）
- 全部实验都答对的 case：1766
- 全部实验都答错的 case：752
- 结果分歧（mixed）case：1482

## 3. 图表
- ![score_over_steps](outputs/plots/01_score_over_steps.png)
- ![diagnostics](outputs/plots/02_diagnostics_over_steps.png)
- ![final_vs_best](outputs/plots/03_final_vs_best_score.png)
- ![final_metrics_heatmap](outputs/plots/04_final_metrics_heatmap.png)
- ![pairwise_winrate](outputs/plots/05_pairwise_winrate_heatmap.png)

## 4. 关键数值（summary）
| experiment_label | step0_score | final_score | best_step | best_score | score_gain_vs_step0 | final_response_length | final_answer_tail_confidence | final_low_confidence_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base-Full | 0.363 | 0.6052 | 174 | 0.6135 | 0.2422 | 577.9062 | 0.9248 | 0.0296 |
| Base-OneShot | 0.363 | 0.5752 | 156 | 0.5802 | 0.2123 | 546.1562 | 0.9291 | 0.0326 |
| Instruct-Full | 0.686 | 0.707 | 180 | 0.707 | 0.021 | 338.125 | 0.969 | 0.0177 |
| Instruct-OneShot | 0.686 | 0.6928 | 132 | 0.6985 | 0.0067 | 328.125 | 0.9725 | 0.0217 |

## 5. Case 级观察
### 5.1 分歧最大的 10 个 case（按 score_std）
| uid | score_mean | score_std | base_full_score | base_oneshot_score | instruct_full_score | instruct_oneshot_score | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 02bd68011db55aa3::0 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | Solve for $x: 3^{2x} + 19 = 10^x$. Let's think step by step and output the final answer within \boxed{}. |
| 02bd68011db55aa3::1 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | Solve for $x: 3^{2x} + 19 = 10^x$. Let's think step by step and output the final answer within \boxed{}. |
| 02bd68011db55aa3::6 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | Solve for $x: 3^{2x} + 19 = 10^x$. Let's think step by step and output the final answer within \boxed{}. |
| 0370ee7a527e24a9::3 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | If $\arccos x + \arccos 2x + \arccos 3x = \pi,$ then $x$ satisfies a cubic polynomial of the form \[ax^3 + bx^2 + cx + d |
| 0370ee7a527e24a9::4 | 0.5 | 0.5 | 1.0 | 0.0 | 0.0 | 1.0 | If $\arccos x + \arccos 2x + \arccos 3x = \pi,$ then $x$ satisfies a cubic polynomial of the form \[ax^3 + bx^2 + cx + d |
| 03a293c6dfb5cfa6::5 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | Find the modulo $7$ remainder of the sum $1+3+5+7+9+\dots+195+197+199.$ Let's think step by step and output the final an |
| 03a293c6dfb5cfa6::6 | 0.5 | 0.5 | 0.0 | 1.0 | 0.0 | 1.0 | Find the modulo $7$ remainder of the sum $1+3+5+7+9+\dots+195+197+199.$ Let's think step by step and output the final an |
| 0539aa3ac36b7ca3::0 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | In the diagram below, we have $\sin \angle RPQ = \frac{7}{25}$.  What is $\cos \angle RPS$?  [asy]  pair R,P,Q,SS;  SS = |
| 0539aa3ac36b7ca3::1 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | In the diagram below, we have $\sin \angle RPQ = \frac{7}{25}$.  What is $\cos \angle RPS$?  [asy]  pair R,P,Q,SS;  SS = |
| 0539aa3ac36b7ca3::6 | 0.5 | 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | In the diagram below, we have $\sin \angle RPQ = \frac{7}{25}$.  What is $\cos \angle RPS$?  [asy]  pair R,P,Q,SS;  SS = |

### 5.2 全实验都失败的 10 个 case
| uid | question |
| --- | --- |
| 0183cd3c82f57ad3::0 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::1 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::2 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::3 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::4 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::5 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::6 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 0183cd3c82f57ad3::7 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 019ca5bfafc7c4f4::0 | The distances from a point $P$ to five of the vertices of a regular octahedron are 3, 7, 8, 9, and 11.  Find the distance from $P$ to the sixth vertex.  [asy] i |
| 019ca5bfafc7c4f4::1 | The distances from a point $P$ to five of the vertices of a regular octahedron are 3, 7, 8, 9, and 11.  Find the distance from $P$ to the sixth vertex.  [asy] i |

### 5.3 最优实验（Instruct-Full）仍失败的 10 个 case
| uid | score_mean | question |
| --- | --- | --- |
| 0183cd3c82f57ad3::0 | 0.0 | Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by \[f(x) = \lambda x(1 - x).\]Find the values of $\lambda,$ $0 \le |
| 901ee76adc144638::2 | 0.0 | Let $G$ and $H$ denote the centroid and orthocenter of triangle $ABC,$ respectively.   Let $F$ be the midpoint of $\overline{GH}.$  Express $AF^2 + BF^2 + CF^2$ |
| 901ee76adc144638::3 | 0.0 | Let $G$ and $H$ denote the centroid and orthocenter of triangle $ABC,$ respectively.   Let $F$ be the midpoint of $\overline{GH}.$  Express $AF^2 + BF^2 + CF^2$ |
| 901ee76adc144638::4 | 0.0 | Let $G$ and $H$ denote the centroid and orthocenter of triangle $ABC,$ respectively.   Let $F$ be the midpoint of $\overline{GH}.$  Express $AF^2 + BF^2 + CF^2$ |
| 901ee76adc144638::5 | 0.0 | Let $G$ and $H$ denote the centroid and orthocenter of triangle $ABC,$ respectively.   Let $F$ be the midpoint of $\overline{GH}.$  Express $AF^2 + BF^2 + CF^2$ |
| 901ee76adc144638::6 | 0.0 | Let $G$ and $H$ denote the centroid and orthocenter of triangle $ABC,$ respectively.   Let $F$ be the midpoint of $\overline{GH}.$  Express $AF^2 + BF^2 + CF^2$ |
| 90731f8d19299b60::1 | 0.0 | In the diagram, four circles of radius 1 with centres $P$, $Q$, $R$, and $S$ are tangent to one another and to the sides of $\triangle ABC$, as shown. [asy] siz |
| 90731f8d19299b60::2 | 0.0 | In the diagram, four circles of radius 1 with centres $P$, $Q$, $R$, and $S$ are tangent to one another and to the sides of $\triangle ABC$, as shown. [asy] siz |
| 90731f8d19299b60::4 | 0.0 | In the diagram, four circles of radius 1 with centres $P$, $Q$, $R$, and $S$ are tangent to one another and to the sides of $\triangle ABC$, as shown. [asy] siz |
| 90731f8d19299b60::5 | 0.0 | In the diagram, four circles of radius 1 with centres $P$, $Q$, $R$, and $S$ are tangent to one another and to the sides of $\triangle ABC$, as shown. [asy] siz |

## 6. LLM 打标方案（建议）
建议把 `outputs/tables/llm_tagging_candidates.csv` 作为输入，按 `case_uid + 4个模型输出` 做错误标签标注。

推荐标签 schema：
- `math_reasoning_error`：推理链有明显数学错误
- `final_answer_extraction_error`：推导过程可能正确，但最终答案提取/格式错误
- `format_compliance_error`：未按 `\boxed{}` 或题目要求格式输出
- `instruction_following_error`：没有按题意约束（范围、单位、形式）
- `uncertainty_or_hallucination`：出现自相矛盾/明显不确定猜测
- `overlong_or_inefficient_reasoning`：推理过长且引入噪声导致错误

建议 LLM 标注 prompt 模板：
```text
你将看到一个数学题 case，以及 4 个模型在 final step 的输出与打分（0/1）。
请完成：
1) 给每个失败输出（score=0）打 1~2 个主标签（从给定 schema 里选）
2) 给出一句“最可能失败原因”
3) 给出“最小修复建议”（例如：答案提取、格式约束、减少冗余推理、强化中间校验）
输出 JSON，字段：uid, model, labels, root_cause, minimal_fix
```

## 7. 下一步建议
- 先对 all_wrong + mixed case 做一轮 LLM 打标，确认主失败类型分布
- 若 `final_answer_extraction_error/format_compliance_error` 占比高，优先加 answer-extractor 或格式约束
- 若 `math_reasoning_error` 占比高，优先做高价值 case 复训与难例采样