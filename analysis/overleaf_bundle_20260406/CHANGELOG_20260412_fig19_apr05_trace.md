# Figure 19 Change Log

Date: 2026-04-12

## What changed

- Replaced Figure 19's data source from the older single-probe token analysis with the completed April 5, 2026 eight-question traced subset from:
  - `ckpts/verl_grpo_dsr_sub_baseline/base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719/validation/180.jsonl`
- Regenerated `figures/correct-vs-wrong-entropy.png` from that eight-question traced subset.
- Exported the per-question support data to:
  - `data/fig19-apr05-eight-question-trace.csv`
- Updated the Chinese and English report text around Figure 19 so the narrative matches the new scope:
  - `main_cn.tex`
  - `main_en.tex`
- Removed submission-facing process wording such as "completed April 5, 2026 rerun" and "2026 年 4 月 5 日完成版" from:
  - the Figure 19 discussion in `main_cn.tex`
  - the Figure 19 discussion in `main_en.tex`
  - the rendered text inside `figures/correct-vs-wrong-entropy.png`

## Why this changed

- The previous Figure 19 ultimately came from the older fixed single-question token supplement.
- That made the figure easy to misread as broader evidence than it really was.
- The April 5 traced subset is still small, but it is materially better aligned with the paper's intended claim because it covers 8 traced validation questions instead of only 1.

## Data summary used for the new figure

- Run: `base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719`
- Final step: `180`
- Traced questions: `8`
- Correct responses: `5`
- Wrong responses: `3`
- Mean traced-prefix token entropy:
  - correct: `0.238499`
  - wrong: `0.381102`

## Files changed

- `analysis/overleaf_bundle_20260406/scripts/build_report_assets.py`
- `analysis/overleaf_bundle_20260406/scripts/render_static_figures.py`
- `analysis/overleaf_bundle_20260406/scripts/render_fig19_apr05_trace.py`
- `analysis/overleaf_bundle_20260406/main_cn.tex`
- `analysis/overleaf_bundle_20260406/main_en.tex`

## Regeneration notes

- `render_fig19_apr05_trace.py` was added as a lightweight helper that can regenerate the Figure 19 CSV and PNG in environments where the system `python3` lacks the `matplotlib` dependency chain.
- `render_static_figures.py` was also updated so the bundle's normal static-figure pipeline points to the new Figure 19 CSV when a full matplotlib-capable environment is available.

## Verification

- Confirmed the new CSV contains `8` traced rows with a `5/3` correct-vs-wrong split.
- Rebuilt both reports with `latexmk -xelatex` after updating the figure source and neutralizing the submission-facing Figure 19 wording:
  - `analysis/overleaf_bundle_20260406/main_cn.tex`
  - `analysis/overleaf_bundle_20260406/main_en.tex`
