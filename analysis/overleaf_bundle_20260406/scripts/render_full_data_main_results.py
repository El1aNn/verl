#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/root/rl/verl")
BUNDLE = ROOT / "analysis" / "overleaf_bundle_20260406"
DATA_DIR = BUNDLE / "data"
FIG_DIR = BUNDLE / "figures"


RUNS = [
    {
        "slug": "base_n8",
        "paper_label": "Base baseline (n=8)",
        "short_label": "Base n=8",
        "family": "Base",
        "kind": "one-shot",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_align_exp1_20260401_141352" / "validation",
        "color": "#1F5F8B",
        "linestyle": "-",
    },
    {
        "slug": "base_full",
        "paper_label": "Base full-data",
        "short_label": "Base full",
        "family": "Base",
        "kind": "full-data",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_align_exp1_full_20260413_215447" / "validation",
        "color": "#0B3954",
        "linestyle": "-",
    },
    {
        "slug": "base_n1",
        "paper_label": "Base small-group ablation (n=1)",
        "short_label": "Base n=1",
        "family": "Base",
        "kind": "one-shot",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_c_n1_val8_align_exp1_2gpu_20260406_195153" / "validation",
        "color": "#C56A2D",
        "linestyle": "--",
    },
    {
        "slug": "base_weak",
        "paper_label": "Base weak-regularization variant",
        "short_label": "Base weak-reg.",
        "family": "Base",
        "kind": "one-shot",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_d_weak_constraint_20260402_003007" / "validation",
        "color": "#2A7F62",
        "linestyle": "-.",
    },
    {
        "slug": "base_random",
        "paper_label": "Base corrupted-reward control",
        "short_label": "Base random-reward",
        "family": "Base",
        "kind": "control",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "base_c_n8_random_wrong_2gpu_20260407_014405" / "validation",
        "color": "#AA3F39",
        "linestyle": ":",
    },
    {
        "slug": "instr_n8",
        "paper_label": "Instruct baseline (n=8)",
        "short_label": "Instruct n=8",
        "family": "Instruct",
        "kind": "one-shot",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "instruct_c_n8_baseline_rerun180_20260404_113902" / "validation",
        "color": "#2C7A7B",
        "linestyle": "--",
    },
    {
        "slug": "instr_full",
        "paper_label": "Instruct full-data",
        "short_label": "Instruct full",
        "family": "Instruct",
        "kind": "full-data",
        "validation_dir": ROOT / "ckpts" / "verl_grpo_dsr_sub_baseline" / "instruct_align_exp1_full_20260414_101916" / "validation",
        "color": "#005F73",
        "linestyle": "-",
    },
]

METRICS = [
    "score",
    "response_length",
    "answer_tail_confidence",
    "low_confidence_token_ratio",
    "token_entropy_mean",
    "eos_prob_final",
    "top1_prob_mean",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_of(rows: list[dict], key: str) -> float:
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.fmean(vals) if vals else float("nan")


def base_axes(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor("#FCFBF8")
    ax.grid(True, color="#DDD7CD", linewidth=0.7)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    step_rows: list[dict] = []
    summary_rows: list[dict] = []

    for run in RUNS:
        validation_files = sorted(run["validation_dir"].glob("*.jsonl"), key=lambda p: int(p.stem))
        run_steps = []
        for path in validation_files:
            rows = read_jsonl(path)
            record = {
                "slug": run["slug"],
                "paper_label": run["paper_label"],
                "short_label": run["short_label"],
                "family": run["family"],
                "kind": run["kind"],
                "step": int(path.stem),
                "sample_count": len(rows),
            }
            for metric in METRICS:
                record[metric] = mean_of(rows, metric)
            step_rows.append(record)
            run_steps.append(record)

        first = run_steps[0]
        last = run_steps[-1]
        best = max(run_steps, key=lambda item: item["score"])
        summary_rows.append(
            {
                "slug": run["slug"],
                "paper_label": run["paper_label"],
                "short_label": run["short_label"],
                "family": run["family"],
                "kind": run["kind"],
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
            }
        )

    step_df = pd.DataFrame(step_rows)
    summary_df = pd.DataFrame(summary_rows)
    step_df.to_csv(DATA_DIR / "step-metrics-main-seven.csv", index=False)
    summary_df.to_csv(DATA_DIR / "run-summary-main-seven.csv", index=False)
    return step_df, summary_df


def plot_global_scores(step_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    base_axes(ax, "Validation score", "Matched score trajectories across seven settings")
    for run in RUNS:
        sub = step_df[step_df["paper_label"] == run["paper_label"]].sort_values("step")
        ax.plot(
            sub["step"],
            sub["score"],
            run["linestyle"],
            color=run["color"],
            linewidth=2.1,
            label=run["short_label"],
        )
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score-main-seven-settings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_detail_panel(step_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), sharex=True)

    ax = axes[0]
    base_axes(ax, "Validation score", "Base family")
    for slug in ["base_n8", "base_full", "base_n1", "base_weak"]:
        run = next(item for item in RUNS if item["slug"] == slug)
        sub = step_df[step_df["slug"] == slug].sort_values("step")
        ax.plot(sub["step"], sub["score"], run["linestyle"], color=run["color"], linewidth=2.2, label=run["short_label"])
    ax.set_ylim(0.35, 0.625)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    base_axes(ax, "Validation score", "Instruct family")
    for slug in ["instr_n8", "instr_full"]:
        run = next(item for item in RUNS if item["slug"] == slug)
        sub = step_df[step_df["slug"] == slug].sort_values("step")
        ax.plot(sub["step"], sub["score"], run["linestyle"], color=run["color"], linewidth=2.4, label=run["short_label"])
        ax.scatter(sub["step"], sub["score"], color=run["color"], s=12, alpha=0.55)
    ax.set_ylim(0.675, 0.7085)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    base_axes(ax, "Validation score", "Corrupted-reward control")
    run = next(item for item in RUNS if item["slug"] == "base_random")
    sub = step_df[step_df["slug"] == "base_random"].sort_values("step")
    ax.plot(sub["step"], sub["score"], run["linestyle"], color=run["color"], linewidth=2.4)
    ax.scatter(sub["step"], sub["score"], color=run["color"], s=12, alpha=0.55)
    ax.set_ylim(-0.02, 0.38)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "score-main-detail-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_matched_full_panel(summary_df: pd.DataFrame) -> None:
    pairs = [
        ("Base", "base_n8", "base_full"),
        ("Instruct", "instr_n8", "instr_full"),
    ]
    metrics = ["step0_score", "final_score", "best_score"]
    metric_titles = ["Initial", "Final", "Best"]
    colors = ["#D8E6EF", "#95BBD3", "#1F5F8B"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    for ax, (family, one_shot_slug, full_slug) in zip(axes, pairs):
        one = summary_df[summary_df["slug"] == one_shot_slug].iloc[0]
        full = summary_df[summary_df["slug"] == full_slug].iloc[0]
        x = [0, 1, 2]
        width = 0.34
        for idx, metric in enumerate(metrics):
            ax.bar(x[idx] - width / 2, one[metric], width=width, color=colors[idx], edgecolor="white")
            ax.bar(x[idx] + width / 2, full[metric], width=width, color="#0B3954" if family == "Base" else "#005F73", alpha=0.88, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_titles)
        ax.set_title(f"{family}: one-shot vs full-data")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylabel("Validation score")
        ax.legend(["one-shot", "full-data"], frameon=False, fontsize=8, loc="upper left")
        for idx, metric in enumerate(metrics):
            ax.text(x[idx] - width / 2, float(one[metric]) + 0.004, f"{float(one[metric]):.4f}", ha="center", va="bottom", fontsize=7)
            ax.text(x[idx] + width / 2, float(full[metric]) + 0.004, f"{float(full[metric]):.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matched-full-data-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    step_df, summary_df = build_tables()
    plot_global_scores(step_df)
    plot_detail_panel(step_df)
    plot_matched_full_panel(summary_df)
    print("Wrote main-result CSVs and figures with matched full-data runs.")


if __name__ == "__main__":
    main()
