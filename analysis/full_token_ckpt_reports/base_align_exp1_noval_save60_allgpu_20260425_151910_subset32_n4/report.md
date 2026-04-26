# Full-token checkpoint validation report

Run label: `base_align_exp1_noval_save60_allgpu_20260425_151910_subset32_n4`

- Validation samples per step: 128
- Cases per step: 32
- Steps: 0, 60, 120, 180
- Final score: 0.6172
- Best score: 0.6172 at step 60
- Final response length: 662.9
- Final token entropy: 0.2188
- Final low-confidence ratio: 0.0398

Figures are in `figures/`; detailed sample and token records are in `records/`.

## Checkpoint-level summary

| step | score | length | entropy | low-conf. | top-1 | tail conf. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3047 | 812.1 | 0.6982 | 0.1254 | 0.8385 | 0.7755 |
| 60 | 0.6172 | 609.9 | 0.2394 | 0.0452 | 0.9184 | 0.9220 |
| 120 | 0.5859 | 608.9 | 0.2710 | 0.0492 | 0.9149 | 0.9066 |
| 180 | 0.6172 | 662.9 | 0.2188 | 0.0398 | 0.9256 | 0.9066 |

The main transition occurs by step 60: score rises sharply while average response length, entropy, and low-confidence mass all fall. Step 60 and step 180 tie in score but differ behaviorally, so the audit should be read as a joint score/length/confidence diagnostic rather than as a single-metric ranking.

## Correct vs. wrong samples

| step | group | n | length | entropy | low-conf. | tail conf. | top-1 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0 | correct | 39 | 482.4 | 0.3282 | 0.0619 | 0.8875 | 0.8978 |
| 0 | wrong | 89 | 956.5 | 0.8603 | 0.1533 | 0.7264 | 0.8125 |
| 60 | correct | 79 | 482.5 | 0.2231 | 0.0412 | 0.9286 | 0.9229 |
| 60 | wrong | 49 | 815.4 | 0.2657 | 0.0517 | 0.9114 | 0.9112 |
| 120 | correct | 75 | 473.1 | 0.2126 | 0.0388 | 0.9298 | 0.9268 |
| 120 | wrong | 53 | 801.2 | 0.3537 | 0.0639 | 0.8737 | 0.8980 |
| 180 | correct | 79 | 476.5 | 0.1765 | 0.0301 | 0.9381 | 0.9362 |
| 180 | wrong | 49 | 963.4 | 0.2871 | 0.0555 | 0.8559 | 0.9084 |

Correct samples are consistently shorter, lower-entropy, lower in low-confidence-token ratio, and higher in tail confidence/top-1 probability than wrong samples. At the final checkpoint, token-weighted correct/wrong entropy is 0.1850/0.2619 and token-weighted correct/wrong top-1 probability is 0.9345/0.9156.

## EOS behavior

The EOS metrics measure how much probability the model assigns to the EOS token at generated positions; they are not a direct count of whether sampled responses ended with EOS. Values below are raw ratios multiplied by 10,000 for readability.

| step | mean EOS | final EOS | EOS > 0.1 ratio | EOS top-1 ratio |
|---:|---:|---:|---:|---:|
| 0 | 2.613 | 8.906 | 2.547 | 2.399 |
| 60 | 0.827 | 0.408 | 1.488 | 0.548 |
| 120 | 0.098 | 0.230 | 0.069 | 0.108 |
| 180 | 0.806 | 2.586 | 2.121 | 0.961 |

EOS pressure is not monotonically increased by training. It falls strongly by step 60 and step 120, then partly rebounds at step 180, especially near the final generated position. This suggests that the score gain is not caused by simply making the model eager to stop everywhere; rather, the better checkpoints combine shorter correct responses with lower entropy, fewer low-confidence tokens, and controlled tail-end closure.
