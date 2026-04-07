# SwanLab OpenAPI Export

This directory contains local exports of the six formal experiments in the
SwanLab project `El1an/verl_grpo_dsr_sub_baseline`.

The export script is:

- `fetch_metrics.py`

It uses `swanlab.Api` because the official `swanlab.OpenApi` docs note that
`OpenApi` will be deprecated in 0.8.0. The underlying cloud metric endpoints
are the same ones documented on the SwanLab OpenAPI page.

Outputs:

- `run_manifest.csv`: run list with formal labels and resolved metric counts
- `resolved_keys.json`: canonical metric names mapped to real SwanLab keys
- `combined_metrics_tidy.csv`: all six runs merged into one tidy CSV
- `training_curves.csv`: selected training metrics
- `validation_curves.csv`: selected validation and token-summary metrics
- `per_run/<run_name>/columns.txt`: full metric-key list for that run
- `per_run/<run_name>/resolved_keys.json`: actual keys chosen for export
- `per_run/<run_name>/metrics_raw.csv`: raw merged DataFrame from SwanLab
- `per_run/<run_name>/metrics_tidy.csv`: cleaned per-run CSV with canonical columns

Run it with:

```bash
SWANLAB_API_KEY=... python analysis/swanlab_openapi_export_20260406/fetch_metrics.py
```
