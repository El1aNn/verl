#!/usr/bin/env python3
"""Export SwanLab experiment curves for the six formal runs.

This script follows SwanLab's OpenAPI/Api flow:
1. authenticate with SWANLAB_API_KEY or local login state
2. query each run's available metric columns from the cloud
3. resolve a stable set of training/validation metrics
4. export both per-run CSVs and combined tidy CSVs locally

The docs page for `swanlab.OpenApi` notes that it will be deprecated in 0.8.0.
This script therefore uses `swanlab.Api`, which wraps the same cloud resources
and still relies on the documented metric endpoints.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from swanlab import Api


USERNAME = "El1an"
PROJECT = "verl_grpo_dsr_sub_baseline"
ROOT = Path("/root/rl/verl/analysis/swanlab_openapi_export_20260406")
PER_RUN_DIR = ROOT / "per_run"


@dataclass(frozen=True)
class RunSpec:
    formal_label: str
    run_name: str
    run_id: str


RUN_SPECS: List[RunSpec] = [
    RunSpec("Base baseline (n=8)", "base_align_exp1_20260401_141352", "refyqg089cx5qvp3g7zbo"),
    RunSpec(
        "Base small-group ablation (n=1)",
        "base_c_n1_val8_align_exp1_2gpu_20260406_195153",
        "04bw3n5xqk9h3gla7imi9",
    ),
    RunSpec("Base weak-regularization variant", "base_d_weak_constraint_20260402_003007", "rq4nirwa1qsaynipprodn"),
    RunSpec("Base nominal control condition", "base_b_reward_shuffle_20260402_132309", "n7tq5k0sid555howe8hmg"),
    RunSpec(
        "Instruct nominal control condition",
        "instruct_b_reward_shuffle_20260403_002727",
        "36jz6dyl5w510dkl2mff2",
    ),
    RunSpec("Instruct baseline (n=8)", "instruct_c_n8_baseline_rerun180_20260404_113902", "z0ub6abmgswdgbah08442"),
]


def _require_api_key() -> str:
    api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is not set; cannot export SwanLab metrics.")
    return api_key


def _ensure_dirs() -> None:
    PER_RUN_DIR.mkdir(parents=True, exist_ok=True)


def _pick_first(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    key_set = set(keys)
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    return None


def _pick_first_regex(keys: Iterable[str], pattern: str) -> Optional[str]:
    regex = re.compile(pattern)
    matched = sorted(key for key in keys if regex.fullmatch(key))
    return matched[0] if matched else None


def _dedupe_keep_order(items: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fetch_column_keys(run) -> List[str]:
    payload, _ = run._client.get(f"/experiment/{run.id}/column", params={"all": True})
    return [item["key"] for item in payload.get("list", [])]


def _resolve_keys(column_keys: List[str]) -> Dict[str, Optional[str]]:
    resolved = {
        "training_global_step": _pick_first(column_keys, ["training/global_step", "global_step"]),
        "actor_entropy": _pick_first(column_keys, ["actor/entropy"]),
        "actor_kl_loss": _pick_first(column_keys, ["actor/kl_loss"]),
        "actor_ppo_kl": _pick_first(column_keys, ["actor/ppo_kl"]),
        "actor_pg_loss": _pick_first(column_keys, ["actor/pg_loss"]),
        "train_response_length_mean": _pick_first(column_keys, ["response_length/mean"]),
        "val_reward_mean": _pick_first_regex(column_keys, r"val-core/math500/reward/mean@\d+"),
        "val_reward_pass1": _pick_first(column_keys, ["val-core/math500/reward/pass@1"]),
        "val_reward_pass8": _pick_first(column_keys, ["val-core/math500/reward/pass@8"]),
        "val_reward_best8_mean": _pick_first(column_keys, ["val-core/math500/reward/best@8/mean"]),
        "val_aux_response_length_mean": _pick_first_regex(column_keys, r"val-aux/math500/response_length/mean@\d+"),
        "val_token_entropy_mean": _pick_first(column_keys, ["val-token/math500/summary/token_entropy_mean"]),
        "val_low_confidence_ratio": _pick_first(column_keys, ["val-token/math500/summary/low_confidence_token_ratio"]),
        "val_token_response_length_mean": _pick_first(column_keys, ["val-token/math500/summary/response_length_mean"]),
        "val_top1_prob_mean": _pick_first(column_keys, ["val-token/math500/summary/top1_prob_mean"]),
        "val_eos_prob_mean": _pick_first(column_keys, ["val-token/math500/summary/eos_prob_mean"]),
        "val_topk_mass_mean": _pick_first(column_keys, ["val-token/math500/summary/topk_mass_mean"]),
        "val_analyzed_samples": _pick_first(column_keys, ["val-token/math500/summary/analyzed_samples"]),
        "val_analyzed_sample_fraction": _pick_first(column_keys, ["val-token/math500/summary/analyzed_sample_fraction"]),
    }
    return resolved


def _fetch_metrics_frame(run, resolved_keys: Dict[str, Optional[str]]) -> pd.DataFrame:
    metric_keys = _dedupe_keep_order(resolved_keys.values())
    df = run.metrics(keys=metric_keys)
    return df.reset_index()


def _canonicalize_frame(df: pd.DataFrame, resolved_keys: Dict[str, Optional[str]], run_spec: RunSpec) -> pd.DataFrame:
    renamed = {"step": "step"}
    for canonical_name, actual_key in resolved_keys.items():
        if actual_key and actual_key in df.columns:
            renamed[actual_key] = canonical_name
    clean = df.rename(columns=renamed)
    # Drop timestamp helpers from the tidy export; keep them in the raw export.
    clean = clean[[c for c in clean.columns if not c.endswith("_timestamp")]]
    clean.insert(0, "run_id", run_spec.run_id)
    clean.insert(0, "run_name", run_spec.run_name)
    clean.insert(0, "formal_label", run_spec.formal_label)
    return clean


def _write_run_outputs(run_spec: RunSpec, column_keys: List[str], resolved_keys: Dict[str, Optional[str]], raw_df: pd.DataFrame, tidy_df: pd.DataFrame) -> None:
    run_dir = PER_RUN_DIR / run_spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "columns.txt").write_text("\n".join(sorted(column_keys)) + "\n", encoding="utf-8")
    (run_dir / "resolved_keys.json").write_text(
        json.dumps(
            {
                "formal_label": run_spec.formal_label,
                "run_name": run_spec.run_name,
                "run_id": run_spec.run_id,
                "resolved_keys": resolved_keys,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    raw_df.to_csv(run_dir / "metrics_raw.csv", index=False)
    tidy_df.to_csv(run_dir / "metrics_tidy.csv", index=False)


def _write_root_outputs(run_results: List[Dict[str, object]]) -> None:
    manifest_rows = []
    combined_frames = []
    resolved_payload = {}

    for item in run_results:
        run_spec: RunSpec = item["run_spec"]  # type: ignore[assignment]
        column_keys: List[str] = item["column_keys"]  # type: ignore[assignment]
        resolved_keys: Dict[str, Optional[str]] = item["resolved_keys"]  # type: ignore[assignment]
        tidy_df: pd.DataFrame = item["tidy_df"]  # type: ignore[assignment]
        combined_frames.append(tidy_df)
        manifest_rows.append(
            {
                "formal_label": run_spec.formal_label,
                "run_name": run_spec.run_name,
                "run_id": run_spec.run_id,
                "column_count": len(column_keys),
                "resolved_metric_count": sum(1 for value in resolved_keys.values() if value),
            }
        )
        resolved_payload[run_spec.run_name] = {
            "formal_label": run_spec.formal_label,
            "run_id": run_spec.run_id,
            "resolved_keys": resolved_keys,
        }

    pd.DataFrame(manifest_rows).to_csv(ROOT / "run_manifest.csv", index=False)
    (ROOT / "resolved_keys.json").write_text(json.dumps(resolved_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    combined = pd.concat(combined_frames, ignore_index=True, sort=False).sort_values(["formal_label", "step"])
    combined.to_csv(ROOT / "combined_metrics_tidy.csv", index=False)

    training_cols = [
        "formal_label",
        "run_name",
        "run_id",
        "step",
        "training_global_step",
        "actor_entropy",
        "actor_kl_loss",
        "actor_ppo_kl",
        "actor_pg_loss",
        "train_response_length_mean",
    ]
    validation_cols = [
        "formal_label",
        "run_name",
        "run_id",
        "step",
        "val_reward_mean",
        "val_reward_pass1",
        "val_reward_pass8",
        "val_reward_best8_mean",
        "val_aux_response_length_mean",
        "val_token_entropy_mean",
        "val_low_confidence_ratio",
        "val_token_response_length_mean",
        "val_top1_prob_mean",
        "val_eos_prob_mean",
        "val_topk_mass_mean",
        "val_analyzed_samples",
        "val_analyzed_sample_fraction",
    ]
    combined[[col for col in training_cols if col in combined.columns]].to_csv(ROOT / "training_curves.csv", index=False)
    combined[[col for col in validation_cols if col in combined.columns]].to_csv(ROOT / "validation_curves.csv", index=False)


def main() -> None:
    api_key = _require_api_key()
    _ensure_dirs()
    api = Api(api_key=api_key)

    run_results: List[Dict[str, object]] = []
    for run_spec in RUN_SPECS:
        run = api.run(f"{USERNAME}/{PROJECT}/{run_spec.run_id}")
        column_keys = _fetch_column_keys(run)
        resolved_keys = _resolve_keys(column_keys)
        raw_df = _fetch_metrics_frame(run, resolved_keys)
        tidy_df = _canonicalize_frame(raw_df, resolved_keys, run_spec)
        _write_run_outputs(run_spec, column_keys, resolved_keys, raw_df, tidy_df)
        run_results.append(
            {
                "run_spec": run_spec,
                "column_keys": column_keys,
                "resolved_keys": resolved_keys,
                "tidy_df": tidy_df,
            }
        )

    _write_root_outputs(run_results)
    print(f"Exported SwanLab curves to {ROOT}")


if __name__ == "__main__":
    main()
