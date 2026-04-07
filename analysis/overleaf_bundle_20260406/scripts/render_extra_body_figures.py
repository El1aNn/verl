from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/root/rl/verl/analysis/overleaf_bundle_20260406")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


LABEL_MAP = {
    "BaseCN8": "Base baseline (n=8)",
    "BaseCN1": "Base small-group ablation (n=1)",
    "BaseWeak": "Base weak-reg.",
    "BaseShuffle": "Base nominal control",
    "InstrShuffle": "Instruct nominal control",
    "InstrCN8": "Instruct baseline (n=8)",
}

COLOR_MAP = {
    "Base baseline (n=8)": "#1b5e20",
    "Base small-group ablation (n=1)": "#2e7d32",
    "Base weak-reg.": "#558b2f",
    "Base nominal control": "#7cb342",
    "Instruct nominal control": "#1565c0",
    "Instruct baseline (n=8)": "#3949ab",
}


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def render_phase_gain_breakdown() -> None:
    df = pd.read_csv(DATA_DIR / "score-curves-all.csv")
    phase_points = [0, 36, 96, 180]
    phase_labels = ["0→36", "36→96", "96→180"]
    rows = []
    for column, label in LABEL_MAP.items():
        vals = []
        for step in phase_points:
            vals.append(float(df.loc[df["step"] == step, column].iloc[0]))
        deltas = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        rows.append((label, deltas))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottoms = [0.0] * len(rows)
    phase_colors = ["#90caf9", "#42a5f5", "#1565c0"]
    for phase_idx, phase_label in enumerate(phase_labels):
        heights = [row[1][phase_idx] for row in rows]
        ax.bar(
            [row[0] for row in rows],
            heights,
            bottom=bottoms,
            color=phase_colors[phase_idx],
            label=phase_label,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_ylabel("Validation score gain")
    ax.set_title("Stage-wise decomposition of score gains")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase-gain-breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_best_final_gap() -> None:
    df = pd.read_csv(DATA_DIR / "run-summary.csv")
    label_col = "paper_label"
    df["best_final_gap"] = df["best_score"] - df["final_score"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)

    colors = [COLOR_MAP.get(x, "#666666") for x in df[label_col]]
    axes[0].barh(df[label_col], df["best_final_gap"], color=colors)
    axes[0].set_title("Best-score minus final-score gap")
    axes[0].set_xlabel("Gap")
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")

    axes[1].barh(df[label_col], df["best_step"], color=colors)
    axes[1].set_title("Best validation step")
    axes[1].set_xlabel("Step")
    axes[1].grid(axis="x", alpha=0.25, linestyle="--")

    fig.savefig(FIG_DIR / "best-final-gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_final_behavior_ranking() -> None:
    df = pd.read_csv(DATA_DIR / "run-summary.csv")
    label_col = "paper_label"
    renamed = {
        "final_token_entropy_mean": "Final token entropy",
        "final_low_confidence_token_ratio": "Final low-conf ratio",
        "final_top1_prob_mean": "Final top-1 probability",
        "final_response_length": "Final response length",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), constrained_layout=True)
    axes = axes.flatten()
    for ax, (column, title) in zip(axes, renamed.items()):
        sdf = df[[label_col, column]].sort_values(column, ascending=(column == "final_top1_prob_mean"))
        colors = [COLOR_MAP.get(x, "#666666") for x in sdf[label_col]]
        ax.barh(sdf[label_col], sdf[column], color=colors)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    fig.savefig(FIG_DIR / "final-behavior-ranking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_gain_calibration_coupling() -> None:
    df = pd.read_csv(DATA_DIR / "swanlab-dynamic-summary.csv")
    label_col = "formal_label"
    x_specs = [
        ("actor_entropy_delta", "Actor entropy reduction"),
        ("val_token_entropy_mean_delta", "Validation token-entropy reduction"),
        ("val_low_confidence_ratio_delta", "Low-confidence reduction"),
        ("val_token_response_length_mean_delta", "Response-length reduction"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), constrained_layout=True)
    axes = axes.flatten()
    for ax, (column, title) in zip(axes, x_specs):
        x = -df[column]
        y = df["val_reward_mean_delta"]
        colors = [COLOR_MAP.get(label, "#666666") for label in df[label_col]]
        ax.scatter(x, y, s=105, c=colors, edgecolor="white", linewidth=0.8)
        for _, row in df.iterrows():
            ax.annotate(
                row[label_col].replace(" baseline", "").replace(" nominal control condition", " nominal ctrl"),
                (-row[column] + 0.001, row["val_reward_mean_delta"] + 0.0015),
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Validation score gain")
        ax.grid(alpha=0.25, linestyle="--")

    fig.savefig(FIG_DIR / "gain-calibration-coupling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_endpoint_frontier_panel() -> None:
    df = pd.read_csv(DATA_DIR / "run-summary.csv")
    label_col = "paper_label"
    panels = [
        ("final_token_entropy_mean", "Final token entropy", True),
        ("final_low_confidence_token_ratio", "Final low-confidence ratio", True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    for ax, (column, xlabel, invert) in zip(axes, panels):
        x = df[column]
        y = df["final_score"]
        colors = [COLOR_MAP.get(label, "#666666") for label in df[label_col]]
        ax.scatter(x, y, s=120, c=colors, edgecolor="white", linewidth=0.9)
        for _, row in df.iterrows():
            short = (
                row[label_col]
                .replace("Base baseline (n=8)", "Base n=8")
                .replace("Base small-group ablation (n=1)", "Base n=1")
                .replace("Base weak-regularization variant", "Base weak-reg")
                .replace("Base nominal control condition", "Base nominal ctrl")
                .replace("Instruct nominal control condition", "Instr nominal ctrl")
                .replace("Instruct baseline (n=8)", "Instr n=8")
            )
            ax.annotate(
                short,
                (row[column] + 0.002, row["final_score"] + 0.0015),
                fontsize=8,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final validation score")
        ax.grid(alpha=0.25, linestyle="--")
        if invert:
            ax.invert_xaxis()

    fig.savefig(FIG_DIR / "endpoint-frontier-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir()
    render_phase_gain_breakdown()
    render_best_final_gap()
    render_final_behavior_ranking()
    render_gain_calibration_coupling()
    render_endpoint_frontier_panel()


if __name__ == "__main__":
    main()
