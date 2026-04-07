# Legacy SwanLab References

This directory stores a supplementary comparison extracted from the SwanLab
project `El1an/verl_grpo_dsr_sub_baseline` via `swanlab.Api`.

The file:

- `legacy_reference_summary.csv`

collects a small set of earlier finished runs that are useful as supplementary
references for the paper, especially for:

- `1-shot` vs `full` training-mode comparison
- early Base / Instruct family contrast
- a format-constrained early reference run

Recommended positioning in the paper:

- Use them as `supplementary references` or `appendix comparisons`
- Do not merge them directly into the main six-run formal result table
- Explain that they come from an earlier experimental phase with partially
  different data scope and logging coverage

Representative takeaways from the extracted runs:

1. In the early Base-model runs, `full` training is stronger than `1-shot`.
   - `Base 1-shot`: `0.3630 -> 0.5753`, best `0.5803`
   - `Base full`: `0.3630 -> 0.6053`, best `0.6135`

2. In the early Instruct-model runs, `full` also exceeds `1-shot`, but the
   gap is much smaller in absolute terms.
   - `Instruct 1-shot`: `0.6860 -> 0.6928`, best `0.6985`
   - `Instruct full`: `0.6860 -> 0.7070`, best `0.7070`

3. The early Base format-constrained reference is much weaker than the main
   reward-based runs.
   - `Base format-only reference`: `0.3630 -> 0.4490`, best `0.4785`

4. An older Base entropy-oriented 1-shot run remains competitive with the
   later one-shot references, but still trails the early `full` run.
   - `Base 1-shot entropy`: `0.3630 -> 0.5698`, best `0.5835`

Interpretation guidance:

- These runs strengthen the paper's broader narrative that `1-shot` training is
  meaningful but not equivalent to `full` training.
- The Base family benefits much more in absolute score than the Instruct
  family.
- The format-constrained reference can serve as a weak auxiliary control,
  supporting the claim that limited reward structure leads to substantially
  smaller gains.

Important caveat:

- These historical runs should be cited as supplementary evidence rather than
  as part of the main formally aligned experiment set.
