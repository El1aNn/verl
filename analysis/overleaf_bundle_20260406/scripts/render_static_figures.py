#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/root/rl/verl/analysis/overleaf_bundle_20260406")
DATA = ROOT / "data"
FIGS = ROOT / "figures"

COLORS = {
    "BaseCN8": "#1F5F8B",
    "BaseCN1": "#C56A2D",
    "BaseWeak": "#2A7F62",
    "BaseShuffle": "#AA3F39",
    "InstrShuffle": "#B58227",
    "InstrCN8": "#2C7A7B",
}

LABELS = {
    "BaseCN8": "Base baseline",
    "BaseCN1": "Base small-group",
    "BaseWeak": "Base weak-reg.",
    "BaseShuffle": "Base nominal control",
    "InstrShuffle": "Instruct nominal control",
    "InstrCN8": "Instruct baseline",
}

STYLES = {
    "BaseCN8": "-",
    "BaseCN1": "--",
    "BaseWeak": "-.",
    "BaseShuffle": ":",
    "InstrShuffle": "--",
    "InstrCN8": "-",
}


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


def plot_scores():
    df = pd.read_csv(DATA / "score-curves-all.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    _base_axes(ax, "Validation score", "All six experiments")
    for key in ["BaseCN8", "BaseCN1", "BaseWeak", "BaseShuffle", "InstrShuffle", "InstrCN8"]:
        ax.plot(df["step"], df[key], STYLES[key], color=COLORS[key], linewidth=2.0, label=LABELS[key])
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    _save(fig, "score-all-static.png")

    df = pd.read_csv(DATA / "score-curves-base.csv")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    _base_axes(ax, "Validation score", "Base-family trajectories")
    for key in ["BaseCN8", "BaseCN1", "BaseWeak", "BaseShuffle"]:
        ax.plot(df["step"], df[key], STYLES[key], color=COLORS[key], linewidth=2.2, label=LABELS[key])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "score-base-static.png")

    df = pd.read_csv(DATA / "score-curves-instruct.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    _base_axes(ax, "Validation score", "Instruct-family trajectories")
    for key in ["InstrShuffle", "InstrCN8"]:
        ax.plot(df["step"], df[key], STYLES[key], color=COLORS[key], linewidth=2.2, label=LABELS[key])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "score-instruct-static.png")


def plot_base_behavior():
    specs = [
        ("base-length-curves.csv", "Tokens", "Base-family response length", "base-length-static.png"),
        ("base-entropy-curves.csv", "Entropy", "Base-family token entropy", "base-entropy-static.png"),
        ("base-low-conf-curves.csv", "Ratio", "Base-family low-confidence ratio", "base-lowconf-static.png"),
        ("base-eos-curves.csv", "Probability", "Base-family final EOS probability", "base-eos-static.png"),
        ("base-tail-conf-curves.csv", "Probability", "Base-family tail confidence", "base-tailconf-static.png"),
        ("base-top1-curves.csv", "Probability", "Base-family top-1 probability", "base-top1-static.png"),
    ]
    for csv_name, ylabel, title, out_name in specs:
        df = pd.read_csv(DATA / csv_name)
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        _base_axes(ax, ylabel, title)
        for key in ["BaseCN8", "BaseCN1", "BaseWeak", "BaseShuffle"]:
            ax.plot(df["step"], df[key], STYLES[key], color=COLORS[key], linewidth=2.2, label=LABELS[key])
        ax.legend(frameon=False, fontsize=9, loc="best")
        _save(fig, out_name)


def plot_actor_entropy():
    df = pd.read_csv(DATA / "swanlab-training-curves.csv")
    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    _base_axes(ax, "Actor entropy", "Training actor-entropy trajectories")
    for key in [
        ("Base baseline (n=8)", "BaseCN8"),
        ("Base small-group ablation (n=1)", "BaseCN1"),
        ("Base weak-regularization variant", "BaseWeak"),
        ("Base nominal control condition", "BaseShuffle"),
        ("Instruct nominal control condition", "InstrShuffle"),
        ("Instruct baseline (n=8)", "InstrCN8"),
    ]:
        sub = df[df["formal_label"] == key[0]].dropna(subset=["actor_entropy"])
        ax.plot(sub["step"], sub["actor_entropy"], STYLES[key[1]], color=COLORS[key[1]], linewidth=2.0, label=LABELS[key[1]])
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    _save(fig, "actor-entropy-static.png")


def plot_aligned_n1_vs_n8_panel():
    step_df = pd.read_csv(DATA / "step-metrics-long.csv")
    train_df = pd.read_csv(DATA / "swanlab-training-curves.csv")

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
        ax.set_facecolor("#FCFBF8")
        ax.grid(True, color="#DDD7CD", linewidth=0.7)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for label, key in run_specs:
            if metric == "actor_entropy":
                sub = train_df[train_df["formal_label"] == label].dropna(subset=[metric])
            else:
                sub = step_df[step_df["paper_label"] == label].dropna(subset=[metric])
            ax.plot(sub["step"], sub[metric], STYLES[key], color=COLORS[key], linewidth=2.2, label=LABELS[key])
    for ax in axes[-2:]:
        ax.set_xlabel("Training step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "aligned-n1-vs-n8-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_scores()
    plot_base_behavior()
    plot_actor_entropy()
    plot_aligned_n1_vs_n8_panel()


if __name__ == "__main__":
    main()
