from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pandas as pd


ROOT = Path("/root/rl/verl")
BUNDLE = ROOT / "analysis" / "overleaf_bundle_20260406"
DATA_DIR = BUNDLE / "data"
TABLE_DIR = BUNDLE / "tables"
SWANLAB_EXPORT_DIR = ROOT / "analysis" / "swanlab_openapi_export_20260406"


RUNS = [
    {
        "slug": "base_c_n8_baseline",
        "plot_key": "BaseCN8",
        "paper_label": "Base baseline (n=8)",
        "family": "Base",
        "variant": "Baseline",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_align_exp1_20260401_141352" / "validation",
        "eval_samples_per_step": 4000,
    },
    {
        "slug": "base_c_n1_group_ablation",
        "plot_key": "BaseCN1",
        "paper_label": "Base small-group ablation (n=1)",
        "family": "Base",
        "variant": "Small-group ablation",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_c_n1_val8_align_exp1_2gpu_20260406_195153" / "validation",
        "eval_samples_per_step": 4000,
    },
    {
        "slug": "base_d_weak_constraint",
        "plot_key": "BaseWeak",
        "paper_label": "Base weak-regularization variant",
        "family": "Base",
        "variant": "Weak regularization",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_d_weak_constraint_20260402_003007" / "validation",
        "eval_samples_per_step": 4000,
    },
    {
        "slug": "base_b_reward_shuffle",
        "plot_key": "BaseShuffle",
        "paper_label": "Base nominal control condition",
        "family": "Base",
        "variant": "Nominal control",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_b_reward_shuffle_20260402_132309" / "validation",
        "eval_samples_per_step": 4000,
    },
    {
        "slug": "instruct_b_reward_shuffle",
        "plot_key": "InstrShuffle",
        "paper_label": "Instruct nominal control condition",
        "family": "Instruct",
        "variant": "Nominal control",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "instruct_b_reward_shuffle_20260403_002727" / "validation",
        "eval_samples_per_step": 4000,
    },
    {
        "slug": "instruct_c_n8_baseline",
        "plot_key": "InstrCN8",
        "paper_label": "Instruct baseline (n=8)",
        "family": "Instruct",
        "variant": "Baseline",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "instruct_c_n8_baseline_rerun180_20260404_113902" / "validation",
        "eval_samples_per_step": 4000,
    },
]


METRIC_KEYS = [
    "score",
    "response_length",
    "answer_tail_confidence",
    "low_confidence_token_ratio",
    "token_entropy_mean",
    "eos_prob_final",
    "top1_prob_mean",
]


TOKEN_SUPPLEMENT_SOURCE = {
    "final": ROOT / "analysis" / "exp4_validation_20260331" / "token_outputs" / "tables" / "traced_token_final_correctness.csv",
    "delta": ROOT / "analysis" / "exp4_validation_20260331" / "token_outputs" / "tables" / "traced_token_delta_step0_to_final.csv",
}

FIG19_TRACE_SOURCE = {
    "paper_label": "Base small-group ablation (n=1)",
    "run_slug": "base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719",
    "final_validation_file": ROOT
    / "ckpts"
    / "verl_grpo_dsr_sub_baseline"
    / "base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719"
    / "validation"
    / "180.jsonl",
}


def mean_of(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.fmean(values) if values else float("nan")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def fmt(num: float, digits: int = 4) -> str:
    if isinstance(num, int):
        return str(num)
    return f"{num:.{digits}f}"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def summarize_nonnull(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in df.columns:
        return {"start_step": None, "start": None, "final_step": None, "final": None, "delta": None}
    sub = df.dropna(subset=[column]).sort_values("step")
    if sub.empty:
        return {"start_step": None, "start": None, "final_step": None, "final": None, "delta": None}
    first = sub.iloc[0]
    last = sub.iloc[-1]
    start_val = float(first[column])
    final_val = float(last[column])
    return {
        "start_step": int(first["step"]),
        "start": start_val,
        "final_step": int(last["step"]),
        "final": final_val,
        "delta": final_val - start_val,
    }


def extract_user_prompt(input_text: str) -> str:
    marker = "user\n"
    if marker not in input_text:
        return input_text.strip()
    tail = input_text.split(marker, 1)[1]
    if "\nassistant\n" in tail:
        tail = tail.split("\nassistant\n", 1)[0]
    return tail.strip()


def build_fig19_trace_rows(path: Path, paper_label: str, run_slug: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        token_diag = row.get("token_diagnostics") or []
        if not token_diag:
            continue
        entropy_values = [item["entropy"] for item in token_diag if isinstance(item.get("entropy"), (int, float))]
        top1_values = [item["top1_prob"] for item in token_diag if isinstance(item.get("top1_prob"), (int, float))]
        eos_values = [item["eos_prob"] for item in token_diag if isinstance(item.get("eos_prob"), (int, float))]
        if not entropy_values:
            continue
        uid = str(row.get("uid", ""))
        uid_parts = uid.split("::")
        question_id = int(uid_parts[2]) if len(uid_parts) >= 3 and uid_parts[2].isdigit() else None
        rows.append(
            {
                "paper_label": paper_label,
                "run_slug": run_slug,
                "source_file": str(path),
                "final_step": int(row.get("step", 0)),
                "uid": uid,
                "question_id": question_id,
                "score": float(row.get("score", 0.0)),
                "correctness": "correct" if float(row.get("score", 0.0)) >= 0.5 else "wrong",
                "traced_tokens": len(token_diag),
                "mean_token_entropy": statistics.fmean(entropy_values),
                "mean_top1_prob": statistics.fmean(top1_values) if top1_values else float("nan"),
                "mean_eos_prob": statistics.fmean(eos_values) if eos_values else float("nan"),
                "response_length": int(row.get("response_length", 0)),
                "question": extract_user_prompt(str(row.get("input", ""))),
            }
        )
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict] = []
    run_rows: list[dict] = []
    per_run_steps: dict[str, list[dict]] = {}

    for run in RUNS:
        validation_files = sorted(run["validation_dir"].glob("*.jsonl"), key=lambda p: int(p.stem))
        steps: list[dict] = []

        for path in validation_files:
            step = int(path.stem)
            rows = read_jsonl(path)
            traced_count = sum(1 for row in rows if row.get("token_diagnostics"))
            record = {
                "slug": run["slug"],
                "paper_label": run["paper_label"],
                "plot_key": run["plot_key"],
                "family": run["family"],
                "variant": run["variant"],
                "step": step,
                "sample_count": len(rows),
                "traced_count": traced_count,
                "traced_ratio": traced_count / len(rows) if rows else 0.0,
            }
            for key in METRIC_KEYS:
                record[key] = mean_of(rows, key)
            steps.append(record)
            step_rows.append(record)

        per_run_steps[run["slug"]] = steps
        first = steps[0]
        last = steps[-1]
        best = max(steps, key=lambda item: item["score"])
        run_rows.append(
            {
                "slug": run["slug"],
                "paper_label": run["paper_label"],
                "plot_key": run["plot_key"],
                "family": run["family"],
                "variant": run["variant"],
                "eval_samples_per_step": run["eval_samples_per_step"],
                "step0_score": first["score"],
                "final_score": last["score"],
                "score_gain": last["score"] - first["score"],
                "best_step": best["step"],
                "best_score": best["score"],
                "final_response_length": last["response_length"],
                "final_answer_tail_confidence": last["answer_tail_confidence"],
                "final_low_confidence_token_ratio": last["low_confidence_token_ratio"],
                "final_token_entropy_mean": last["token_entropy_mean"],
                "final_eos_prob_final": last["eos_prob_final"],
                "final_top1_prob_mean": last["top1_prob_mean"],
                "validation_dir": str(run["validation_dir"]),
            }
        )

    write_csv(
        DATA_DIR / "run-summary.csv",
        [
            "slug",
            "paper_label",
            "plot_key",
            "family",
            "variant",
            "eval_samples_per_step",
            "step0_score",
            "final_score",
            "score_gain",
            "best_step",
            "best_score",
            "final_response_length",
            "final_answer_tail_confidence",
            "final_low_confidence_token_ratio",
            "final_token_entropy_mean",
            "final_eos_prob_final",
            "final_top1_prob_mean",
            "validation_dir",
        ],
        run_rows,
    )

    write_csv(
        DATA_DIR / "step-metrics-long.csv",
        [
            "slug",
            "paper_label",
            "plot_key",
            "family",
            "variant",
            "step",
            "sample_count",
            "traced_count",
            "traced_ratio",
            "score",
            "response_length",
            "answer_tail_confidence",
            "low_confidence_token_ratio",
            "token_entropy_mean",
            "eos_prob_final",
            "top1_prob_mean",
        ],
        step_rows,
    )

    plot_key_by_slug = {run["slug"]: run["plot_key"] for run in RUNS}

    def write_wide_metric_csv(filename: str, metric_key: str, run_slugs: list[str]) -> None:
        all_steps = sorted({row["step"] for row in step_rows})
        rows = []
        for step in all_steps:
            row = {"step": step}
            for slug in run_slugs:
                row[plot_key_by_slug[slug]] = next(
                    item[metric_key] for item in per_run_steps[slug] if item["step"] == step
                )
            rows.append(row)
        write_csv(DATA_DIR / filename, ["step", *[plot_key_by_slug[slug] for slug in run_slugs]], rows)

    all_slugs = [run["slug"] for run in RUNS]
    base_slugs = [run["slug"] for run in RUNS if run["family"] == "Base"]
    instruct_slugs = [run["slug"] for run in RUNS if run["family"] == "Instruct"]

    write_wide_metric_csv("score-curves-all.csv", "score", all_slugs)
    write_wide_metric_csv("score-curves-base.csv", "score", base_slugs)
    write_wide_metric_csv("score-curves-instruct.csv", "score", instruct_slugs)
    write_wide_metric_csv("base-entropy-curves.csv", "token_entropy_mean", base_slugs)
    write_wide_metric_csv("base-length-curves.csv", "response_length", base_slugs)
    write_wide_metric_csv("base-eos-curves.csv", "eos_prob_final", base_slugs)
    write_wide_metric_csv("base-low-conf-curves.csv", "low_confidence_token_ratio", base_slugs)
    write_wide_metric_csv("base-tail-conf-curves.csv", "answer_tail_confidence", base_slugs)
    write_wide_metric_csv("base-top1-curves.csv", "top1_prob_mean", base_slugs)

    swanlab_training = pd.read_csv(SWANLAB_EXPORT_DIR / "training_curves.csv")
    swanlab_validation = pd.read_csv(SWANLAB_EXPORT_DIR / "validation_curves.csv")
    formal_labels = [run["paper_label"] for run in RUNS]
    swanlab_training = swanlab_training[swanlab_training["formal_label"].isin(formal_labels)].copy()
    swanlab_validation = swanlab_validation[swanlab_validation["formal_label"].isin(formal_labels)].copy()
    swanlab_training.to_csv(DATA_DIR / "swanlab-training-curves.csv", index=False)
    swanlab_validation.to_csv(DATA_DIR / "swanlab-validation-curves.csv", index=False)

    dynamic_rows = []
    for run in RUNS:
        label = run["paper_label"]
        train_df = swanlab_training[swanlab_training["formal_label"] == label].copy()
        val_df = swanlab_validation[swanlab_validation["formal_label"] == label].copy()
        actor_entropy = summarize_nonnull(train_df, "actor_entropy")
        train_resp_len = summarize_nonnull(train_df, "train_response_length_mean")
        val_reward_mean = summarize_nonnull(val_df, "val_reward_mean")
        val_reward_pass1 = summarize_nonnull(val_df, "val_reward_pass1")
        val_reward_pass8 = summarize_nonnull(val_df, "val_reward_pass8")
        val_best8 = summarize_nonnull(val_df, "val_reward_best8_mean")
        val_length = summarize_nonnull(val_df, "val_aux_response_length_mean")
        val_token_entropy = summarize_nonnull(val_df, "val_token_entropy_mean")
        val_low_conf = summarize_nonnull(val_df, "val_low_confidence_ratio")
        val_token_resp_len = summarize_nonnull(val_df, "val_token_response_length_mean")
        val_top1 = summarize_nonnull(val_df, "val_top1_prob_mean")
        val_eos = summarize_nonnull(val_df, "val_eos_prob_mean")
        val_topk = summarize_nonnull(val_df, "val_topk_mass_mean")
        val_samples = summarize_nonnull(val_df, "val_analyzed_samples")
        val_sample_frac = summarize_nonnull(val_df, "val_analyzed_sample_fraction")
        best_val_reward = val_df.dropna(subset=["val_reward_mean"]).sort_values("val_reward_mean")
        if best_val_reward.empty:
            best_step = None
            best_score = None
        else:
            best_row = best_val_reward.iloc[-1]
            best_step = int(best_row["step"])
            best_score = float(best_row["val_reward_mean"])
        dynamic_rows.append(
            {
                "formal_label": label,
                "actor_entropy_start_step": actor_entropy["start_step"],
                "actor_entropy_start": actor_entropy["start"],
                "actor_entropy_final_step": actor_entropy["final_step"],
                "actor_entropy_final": actor_entropy["final"],
                "actor_entropy_delta": actor_entropy["delta"],
                "train_resp_len_start": train_resp_len["start"],
                "train_resp_len_final": train_resp_len["final"],
                "val_start_step": val_reward_mean["start_step"],
                "val_final_step": val_reward_mean["final_step"],
                "val_reward_mean_start": val_reward_mean["start"],
                "val_reward_mean_final": val_reward_mean["final"],
                "val_reward_mean_delta": val_reward_mean["delta"],
                "val_reward_pass1_start": val_reward_pass1["start"],
                "val_reward_pass1_final": val_reward_pass1["final"],
                "val_reward_pass1_delta": val_reward_pass1["delta"],
                "val_reward_pass8_start": val_reward_pass8["start"],
                "val_reward_pass8_final": val_reward_pass8["final"],
                "val_reward_pass8_delta": val_reward_pass8["delta"],
                "val_reward_best8_start": val_best8["start"],
                "val_reward_best8_final": val_best8["final"],
                "val_reward_best8_delta": val_best8["delta"],
                "val_aux_response_length_mean_start": val_length["start"],
                "val_aux_response_length_mean_final": val_length["final"],
                "val_aux_response_length_mean_delta": val_length["delta"],
                "val_token_entropy_mean_start": val_token_entropy["start"],
                "val_token_entropy_mean_final": val_token_entropy["final"],
                "val_token_entropy_mean_delta": val_token_entropy["delta"],
                "val_low_confidence_ratio_start": val_low_conf["start"],
                "val_low_confidence_ratio_final": val_low_conf["final"],
                "val_low_confidence_ratio_delta": val_low_conf["delta"],
                "val_token_response_length_mean_start": val_token_resp_len["start"],
                "val_token_response_length_mean_final": val_token_resp_len["final"],
                "val_token_response_length_mean_delta": val_token_resp_len["delta"],
                "val_top1_prob_mean_start": val_top1["start"],
                "val_top1_prob_mean_final": val_top1["final"],
                "val_top1_prob_mean_delta": val_top1["delta"],
                "val_eos_prob_mean_start": val_eos["start"],
                "val_eos_prob_mean_final": val_eos["final"],
                "val_eos_prob_mean_delta": val_eos["delta"],
                "val_topk_mass_mean_start": val_topk["start"],
                "val_topk_mass_mean_final": val_topk["final"],
                "val_topk_mass_mean_delta": val_topk["delta"],
                "val_analyzed_samples_start": val_samples["start"],
                "val_analyzed_samples_final": val_samples["final"],
                "val_analyzed_samples_delta": val_samples["delta"],
                "val_analyzed_sample_fraction_start": val_sample_frac["start"],
                "val_analyzed_sample_fraction_final": val_sample_frac["final"],
                "val_analyzed_sample_fraction_delta": val_sample_frac["delta"],
                "val_reward_best_step": best_step,
                "val_reward_best": best_score,
            }
        )

    dynamic_df = pd.DataFrame(dynamic_rows)
    dynamic_df.to_csv(DATA_DIR / "swanlab-dynamic-summary.csv", index=False)
    swanlab_row_lines = []
    for row in dynamic_rows:
        swanlab_row_lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["formal_label"]),
                    f"{row['actor_entropy_start']:.3f}$\\rightarrow${row['actor_entropy_final']:.3f}",
                    f"{row['val_token_entropy_mean_start']:.3f}$\\rightarrow${row['val_token_entropy_mean_final']:.3f}",
                    f"{row['val_low_confidence_ratio_start']:.3f}$\\rightarrow${row['val_low_confidence_ratio_final']:.3f}",
                    f"{row['val_aux_response_length_mean_start']:.1f}$\\rightarrow${row['val_aux_response_length_mean_final']:.1f}",
                    f"{row['val_top1_prob_mean_start']:.3f}$\\rightarrow${row['val_top1_prob_mean_final']:.3f}",
                ]
            )
            + r" \\"
        )
    write_text(TABLE_DIR / "swanlab-dynamics-rows.tex", "\n".join(swanlab_row_lines) + "\n")

    data_integrity_rows = [
        {
            "file": "data/dsr_sub/pi1_one_ans.parquet",
            "md5": md5sum(ROOT / "data" / "dsr_sub" / "pi1_one_ans.parquet"),
        },
        {
            "file": "data/dsr_sub/pi1_one_ans_reward_shuffle.parquet",
            "md5": md5sum(ROOT / "data" / "dsr_sub" / "pi1_one_ans_reward_shuffle.parquet"),
        },
    ]
    write_csv(DATA_DIR / "data-integrity.csv", ["file", "md5"], data_integrity_rows)

    with TOKEN_SUPPLEMENT_SOURCE["final"].open("r", encoding="utf-8") as f:
        token_final_rows = list(csv.DictReader(f))
    with TOKEN_SUPPLEMENT_SOURCE["delta"].open("r", encoding="utf-8") as f:
        token_delta_rows = list(csv.DictReader(f))
    fig19_trace_rows = build_fig19_trace_rows(
        FIG19_TRACE_SOURCE["final_validation_file"],
        paper_label=str(FIG19_TRACE_SOURCE["paper_label"]),
        run_slug=str(FIG19_TRACE_SOURCE["run_slug"]),
    )

    write_csv(
        DATA_DIR / "token-probe-final.csv",
        list(token_final_rows[0].keys()),
        token_final_rows,
    )
    write_csv(
        DATA_DIR / "token-probe-delta.csv",
        list(token_delta_rows[0].keys()),
        token_delta_rows,
    )
    write_csv(
        DATA_DIR / "fig19-apr05-eight-question-trace.csv",
        [
            "paper_label",
            "run_slug",
            "source_file",
            "final_step",
            "uid",
            "question_id",
            "score",
            "correctness",
            "traced_tokens",
            "mean_token_entropy",
            "mean_top1_prob",
            "mean_eos_prob",
            "response_length",
            "question",
        ],
        fig19_trace_rows,
    )

    main_result_lines = []
    for row in run_rows:
        main_result_lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["paper_label"]),
                    fmt(row["step0_score"]),
                    fmt(row["final_score"]),
                    f"{row['score_gain']:+.4f}",
                    str(int(row["best_step"])),
                    fmt(row["best_score"]),
                ]
            )
            + r" \\"
        )
    write_text(TABLE_DIR / "main-results-rows.tex", "\n".join(main_result_lines) + "\n")

    behavior_lines = []
    for row in run_rows:
        behavior_lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["paper_label"]),
                    fmt(row["final_response_length"], 1),
                    fmt(row["final_answer_tail_confidence"]),
                    fmt(row["final_low_confidence_token_ratio"]),
                    fmt(row["final_token_entropy_mean"]),
                    fmt(row["final_eos_prob_final"]),
                    fmt(row["final_top1_prob_mean"]),
                ]
            )
            + r" \\"
        )
    write_text(TABLE_DIR / "behavior-results-rows.tex", "\n".join(behavior_lines) + "\n")

    token_final_lines = []
    for row in token_final_rows:
        token_final_lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["experiment_label"]),
                    latex_escape(row["correctness"]),
                    row["traced_tokens"],
                    fmt(float(row["entropy"])),
                    fmt(float(row["top1_prob"])),
                    fmt(float(row["top5_mass"])),
                ]
            )
            + r" \\"
        )
    write_text(TABLE_DIR / "token-probe-final-rows.tex", "\n".join(token_final_lines) + "\n")

    token_delta_lines = []
    for row in token_delta_rows:
        token_delta_lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["experiment_label"]),
                    f"{float(row['logprob_delta_final_minus_step0']):+.4f}",
                    f"{float(row['entropy_delta_final_minus_step0']):+.4f}",
                    f"{float(row['top1_prob_delta_final_minus_step0']):+.4f}",
                ]
            )
            + r" \\"
        )
    write_text(TABLE_DIR / "token-probe-delta-rows.tex", "\n".join(token_delta_lines) + "\n")

    integrity_lines = []
    for row in data_integrity_rows:
        integrity_lines.append(
            "        " + " & ".join([latex_escape(row["file"]), latex_escape(row["md5"])]) + r" \\"
        )
    write_text(TABLE_DIR / "data-integrity-rows.tex", "\n".join(integrity_lines) + "\n")

    manifest = {
        "bundle_dir": str(BUNDLE),
        "runs": run_rows,
        "token_probe_source": {key: str(value) for key, value in TOKEN_SUPPLEMENT_SOURCE.items()},
        "fig19_trace_source": {
            key: str(value) for key, value in FIG19_TRACE_SOURCE.items()
        },
        "data_integrity": data_integrity_rows,
    }
    write_text(DATA_DIR / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
