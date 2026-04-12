#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/root/rl/verl/analysis/overleaf_bundle_20260406")
DATA = ROOT / "data"
FIGS = ROOT / "figures"

STEP_DF = pd.read_csv(DATA / "step-metrics-long-v2.csv")
TRAIN_DF = pd.read_csv(DATA / "swanlab-training-curves-v2.csv")
FIG19_TRACE_PATH = DATA / "fig19-apr05-eight-question-trace.csv"

PLOT_TO_LABEL = {
    "BaseCN8": "Base baseline",
    "BaseCN1": "Base small-group",
    "BaseWeak": "Base weak-reg.",
    "BaseCtrl": "Base corrupted-reward",
    "InstrCN8": "Instruct baseline",
}

LABEL_TO_PLOT = {
    "Base baseline (n=8)": "BaseCN8",
    "Base small-group ablation (n=1)": "BaseCN1",
    "Base weak-regularization variant": "BaseWeak",
    "Base corrupted-reward control": "BaseCtrl",
    "Instruct baseline (n=8)": "InstrCN8",
}

COLORS = {
    "BaseCN8": "#1F5F8B",
    "BaseCN1": "#C56A2D",
    "BaseWeak": "#2A7F62",
    "BaseCtrl": "#AA3F39",
    "InstrCN8": "#2C7A7B",
}

STYLES = {
    "BaseCN8": "-",
    "BaseCN1": "--",
    "BaseWeak": "-.",
    "BaseCtrl": ":",
    "InstrCN8": "-",
}

BASE_BEHAVIOR_LABELS = [
    "Base baseline (n=8)",
    "Base small-group ablation (n=1)",
    "Base weak-regularization variant",
    "Base corrupted-reward control",
]

CLEAN_BASE_LABELS = [
    "Base baseline (n=8)",
    "Base small-group ablation (n=1)",
    "Base weak-regularization variant",
]


def _base_axes(ax, ylabel: str, title: str):
    ax.set_facecolor("#FCFBF8")
    ax.grid(True, color="#DDD7CD", linewidth=0.7)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _series(label: str, metric: str) -> pd.DataFrame:
    return (
        STEP_DF[STEP_DF["paper_label"] == label][["step", metric]]
        .dropna(subset=[metric])
        .sort_values("step")
    )


def _train_series(label: str) -> pd.DataFrame:
    return (
        TRAIN_DF[TRAIN_DF["formal_label"] == label][["step", "actor_entropy"]]
        .dropna(subset=["actor_entropy"])
        .sort_values("step")
    )


def _fig19_trace_df() -> pd.DataFrame:
    if not FIG19_TRACE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FIG19_TRACE_PATH)


def plot_scores():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    _base_axes(ax, "Validation score", "All five formal experiments")
    for label, key in LABEL_TO_PLOT.items():
        sub = _series(label, "score")
        ax.plot(sub["step"], sub["score"], STYLES[key], color=COLORS[key], linewidth=2.0, label=PLOT_TO_LABEL[key])
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    _save(fig, "score-all-static.png")

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    _base_axes(ax, "Validation score", "Base-family trajectories")
    for label in [
        "Base baseline (n=8)",
        "Base small-group ablation (n=1)",
        "Base weak-regularization variant",
        "Base corrupted-reward control",
    ]:
        key = LABEL_TO_PLOT[label]
        sub = _series(label, "score")
        ax.plot(sub["step"], sub["score"], STYLES[key], color=COLORS[key], linewidth=2.2, label=PLOT_TO_LABEL[key])
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, "score-base-static.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _base_axes(ax, "Validation score", "Instruct baseline trajectory")
    label = "Instruct baseline (n=8)"
    key = LABEL_TO_PLOT[label]
    sub = _series(label, "score")
    ax.plot(sub["step"], sub["score"], STYLES[key], color=COLORS[key], linewidth=2.4, label=PLOT_TO_LABEL[key])
    ax.scatter(sub["step"], sub["score"], color=COLORS[key], s=14, alpha=0.65)
    ax.set_ylim(sub["score"].min() - 0.005, sub["score"].max() + 0.005)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "score-instruct-static.png")


def plot_score_detail_panel():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)

    # Successful Base conditions only.
    ax = axes[0]
    _base_axes(ax, "Validation score", "Base successful conditions")
    for label in [
        "Base baseline (n=8)",
        "Base small-group ablation (n=1)",
        "Base weak-regularization variant",
    ]:
        key = LABEL_TO_PLOT[label]
        sub = _series(label, "score")
        ax.plot(sub["step"], sub["score"], STYLES[key], color=COLORS[key], linewidth=2.2, label=PLOT_TO_LABEL[key])
    ax.set_ylim(0.35, 0.59)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # Instruct zoom.
    ax = axes[1]
    _base_axes(ax, "Validation score", "Instruct baseline (zoomed)")
    label = "Instruct baseline (n=8)"
    key = LABEL_TO_PLOT[label]
    sub = _series(label, "score")
    ax.plot(sub["step"], sub["score"], STYLES[key], color=COLORS[key], linewidth=2.4)
    ax.scatter(sub["step"], sub["score"], color=COLORS[key], s=14, alpha=0.65)
    ax.set_ylim(0.675, 0.702)

    # Corrupted-reward control.
    ax = axes[2]
    _base_axes(ax, "Validation score", "Corrupted-reward control")
    label = "Base corrupted-reward control"
    key = LABEL_TO_PLOT[label]
    sub = _series(label, "score")
    ax.plot(sub["step"], sub["score"], STYLES[key], color=COLORS[key], linewidth=2.4)
    ax.scatter(sub["step"], sub["score"], color=COLORS[key], s=14, alpha=0.65)
    ax.set_ylim(-0.02, 0.38)

    fig.tight_layout()
    fig.savefig(FIGS / "score-detail-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_base_behavior():
    specs = [
        ("response_length", "Tokens", "Base-family response length", "base-length-static.png"),
        ("token_entropy_mean", "Entropy", "Base-family token entropy", "base-entropy-static.png"),
        ("low_confidence_token_ratio", "Ratio", "Base-family low-confidence ratio", "base-lowconf-static.png"),
        ("eos_prob_final", "Probability", "Base-family final EOS probability", "base-eos-static.png"),
        ("answer_tail_confidence", "Probability", "Base-family tail confidence", "base-tailconf-static.png"),
        ("top1_prob_mean", "Probability", "Base-family top-1 probability", "base-top1-static.png"),
    ]
    for metric, ylabel, title, out_name in specs:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)

        ax = axes[0]
        _base_axes(ax, ylabel, f"{title}: full scale")
        for label in BASE_BEHAVIOR_LABELS:
            key = LABEL_TO_PLOT[label]
            sub = _series(label, metric)
            ax.plot(sub["step"], sub[metric], STYLES[key], color=COLORS[key], linewidth=2.2, label=PLOT_TO_LABEL[key])
        ax.legend(frameon=False, fontsize=8, loc="best")

        ax = axes[1]
        _base_axes(ax, ylabel, f"{title}: clean-region zoom")
        clean_values = []
        for label in CLEAN_BASE_LABELS:
            key = LABEL_TO_PLOT[label]
            sub = _series(label, metric)
            clean_values.extend(sub[metric].tolist())
            ax.plot(sub["step"], sub[metric], STYLES[key], color=COLORS[key], linewidth=2.4, label=PLOT_TO_LABEL[key])
        lo, hi = min(clean_values), max(clean_values)
        margin = (hi - lo) * 0.12 if hi > lo else max(abs(hi) * 0.05, 0.02)
        ax.set_ylim(lo - margin, hi + margin)
        if "Probability" in ylabel or "Ratio" in ylabel:
            low, high = ax.get_ylim()
            ax.set_ylim(max(0.0, low), min(1.02, high))
        ax.legend(frameon=False, fontsize=8, loc="best")

        _save(fig, out_name)


def plot_clean_behavior_detail_panel():
    specs = [
        ("token_entropy_mean", "Entropy", "Token entropy"),
        ("low_confidence_token_ratio", "Ratio", "Low-confidence ratio"),
        ("eos_prob_final", "Probability", "Final EOS probability"),
        ("top1_prob_mean", "Probability", "Top-1 probability"),
    ]
    labels = [
        "Base baseline (n=8)",
        "Base small-group ablation (n=1)",
        "Base weak-regularization variant",
        "Instruct baseline (n=8)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), sharex=True)
    axes = axes.flatten()
    for ax, (metric, ylabel, title) in zip(axes, specs):
        _base_axes(ax, ylabel, title)
        values = []
        for label in labels:
            key = LABEL_TO_PLOT[label]
            sub = _series(label, metric)
            values.extend(sub[metric].tolist())
            ax.plot(sub["step"], sub[metric], STYLES[key], color=COLORS[key], linewidth=2.1, label=PLOT_TO_LABEL[key])
        lo, hi = min(values), max(values)
        margin = (hi - lo) * 0.10 if hi > lo else 0.02
        ax.set_ylim(lo - margin, hi + margin)
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIGS / "behavior-detail-clean-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_actor_entropy():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)

    ax = axes[0]
    _base_axes(ax, "Actor entropy", "Training actor entropy: full scale")
    for label, key in LABEL_TO_PLOT.items():
        sub = _train_series(label)
        ax.plot(sub["step"], sub["actor_entropy"], STYLES[key], color=COLORS[key], linewidth=2.0, label=PLOT_TO_LABEL[key])
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right")

    ax = axes[1]
    _base_axes(ax, "Actor entropy", "Training actor entropy: non-collapse zoom")
    zoom_labels = [
        "Base baseline (n=8)",
        "Base small-group ablation (n=1)",
        "Base weak-regularization variant",
        "Instruct baseline (n=8)",
    ]
    values = []
    for label in zoom_labels:
        key = LABEL_TO_PLOT[label]
        sub = _train_series(label)
        values.extend(sub["actor_entropy"].tolist())
        ax.plot(sub["step"], sub["actor_entropy"], STYLES[key], color=COLORS[key], linewidth=2.2, label=PLOT_TO_LABEL[key])
    lo, hi = min(values), max(values)
    margin = (hi - lo) * 0.10 if hi > lo else 0.02
    ax.set_ylim(max(0.0, lo - margin), hi + margin)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right")

    _save(fig, "actor-entropy-static.png")


def plot_aligned_n1_vs_n8_panel():
    run_specs = [
        ("Base baseline (n=8)", "BaseCN8"),
        ("Base small-group ablation (n=1)", "BaseCN1"),
    ]
    metrics = [
        ("score", "Validation score", "Score"),
        ("response_length", "Tokens", "Response length"),
        ("token_entropy_mean", "Entropy", "Token entropy"),
        ("low_confidence_token_ratio", "Ratio", "Low-confidence ratio"),
        ("top1_prob_mean", "Probability", "Top-1 probability"),
        ("actor_entropy", "Entropy", "Actor entropy"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.0), sharex=True)
    axes = axes.flatten()
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        _base_axes(ax, ylabel, title)
        for label, key in run_specs:
            if metric == "actor_entropy":
                sub = _train_series(label)
                y = "actor_entropy"
            else:
                sub = _series(label, metric)
                y = metric
            ax.plot(sub["step"], sub[y], STYLES[key], color=COLORS[key], linewidth=2.2, label=PLOT_TO_LABEL[key])
    for ax in axes[-2:]:
        ax.set_xlabel("Training step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "aligned-n1-vs-n8-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correct_vs_wrong_entropy() -> None:
    df = _fig19_trace_df()
    if df.empty:
        return

    order = [label for label in ["correct", "wrong"] if label in set(df["correctness"])]
    colors = {
        "correct": "#2A7F62",
        "wrong": "#AA3F39",
    }
    label_map = {
        "correct": "Correct",
        "wrong": "Wrong",
    }
    summary = (
        df.groupby("correctness", as_index=False)
        .agg(
            mean_token_entropy=("mean_token_entropy", "mean"),
            question_count=("uid", "count"),
        )
        .set_index("correctness")
        .reindex(order)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_facecolor("#FCFBF8")
    ax.grid(axis="y", color="#DDD7CD", linewidth=0.7)
    x_positions = list(range(len(order)))
    heights = summary["mean_token_entropy"].tolist()
    bar_colors = [colors[label] for label in order]
    ax.bar(x_positions, heights, color=bar_colors, width=0.56, alpha=0.88)

    for xpos, label in zip(x_positions, order):
        sub = df[df["correctness"] == label].sort_values(["question_id", "uid"])
        count = len(sub)
        if count == 1:
            offsets = [0.0]
        else:
            step = 0.24 / (count - 1)
            offsets = [-0.12 + idx * step for idx in range(count)]
        xs = [xpos + offset for offset in offsets]
        ax.scatter(
            xs,
            sub["mean_token_entropy"],
            s=58,
            color=colors[label],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.text(
            xpos,
            heights[xpos] + 0.02,
            f"mean={heights[xpos]:.3f}\nn={count}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{label_map[label]}\n(n={int(summary.loc[idx, 'question_count'])})" for idx, label in enumerate(order)]
    )
    ax.set_ylabel("Mean token entropy over traced prefix")
    ax.set_title("Eight-question traced subset at the final step")
    ax.text(
        0.02,
        0.98,
        "Base n=1 traced subset (MATH500 question ids 0-7)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#4A4A4A",
    )

    _save(fig, "correct-vs-wrong-entropy.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_scores()
    plot_score_detail_panel()
    plot_base_behavior()
    plot_clean_behavior_detail_panel()
    plot_actor_entropy()
    plot_aligned_n1_vs_n8_panel()
    plot_correct_vs_wrong_entropy()


if __name__ == "__main__":
    main()
