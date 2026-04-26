from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/root/rl/verl/analysis/overleaf_bundle_20260406")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

RUN_SUMMARY = pd.read_csv(DATA_DIR / "run-summary-main-seven.csv")
STEP_DF = pd.read_csv(DATA_DIR / "step-metrics-main-seven.csv")
TRAIN_DF = pd.read_csv(DATA_DIR / "swanlab-training-curves-v2.csv")


COLOR_MAP = {
    "Base baseline (n=8)": "#1F5F8B",
    "Base full-data": "#0B3954",
    "Base small-group ablation (n=1)": "#C56A2D",
    "Base weak-regularization variant": "#2A7F62",
    "Base corrupted-reward control": "#AA3F39",
    "Instruct baseline (n=8)": "#2C7A7B",
    "Instruct full-data": "#005F73",
}

CONTROL_LABEL = "Base corrupted-reward control"
NON_COLLAPSE_LABELS = [
    "Base baseline (n=8)",
    "Base small-group ablation (n=1)",
    "Base weak-regularization variant",
    "Instruct baseline (n=8)",
]


def _short_label(label: str) -> str:
    return (
        label.replace("Base baseline (n=8)", "Base n=8")
        .replace("Base full-data", "Base full")
        .replace("Base small-group ablation (n=1)", "Base n=1")
        .replace("Base weak-regularization variant", "Base weak-reg")
        .replace("Base corrupted-reward control", "Base ctrl")
        .replace("Instruct baseline (n=8)", "Instr n=8")
        .replace("Instruct full-data", "Instr full")
    )


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _step_series(label: str, metric: str) -> pd.Series:
    sub = STEP_DF[STEP_DF["paper_label"] == label].sort_values("step")
    return sub.set_index("step")[metric]


def _delta(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1] - series.iloc[0])


def render_phase_gain_breakdown() -> None:
    phase_points = [0, 36, 96, 180]
    phase_labels = ["0→36", "36→96", "96→180"]
    rows = []
    for label in RUN_SUMMARY["paper_label"]:
        score_series = _step_series(label, "score")
        vals = [float(score_series.loc[step]) for step in phase_points]
        deltas = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        rows.append((label, deltas))

    def draw_panel(ax, panel_rows, title):
        bottoms = [0.0] * len(panel_rows)
        phase_colors = ["#90caf9", "#42a5f5", "#1565c0"]
        x_labels = [_short_label(row[0]) for row in panel_rows]
        for phase_idx, phase_label in enumerate(phase_labels):
            heights = [row[1][phase_idx] for row in panel_rows]
            ax.bar(x_labels, heights, bottom=bottoms, color=phase_colors[phase_idx], label=phase_label)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_ylabel("Validation score gain")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    draw_panel(axes[0], rows, "All conditions")
    draw_panel(axes[1], [row for row in rows if row[0] != CONTROL_LABEL], "Non-collapse zoom")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(FIG_DIR / "phase-gain-breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_best_final_gap() -> None:
    df = RUN_SUMMARY.copy()
    df["best_final_gap"] = df["best_score"] - df["final_score"]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), constrained_layout=True)
    panels = [
        (axes[0, 0], df, "Best-score minus final-score gap: full scale", "best_final_gap", "Gap"),
        (axes[0, 1], df[df["paper_label"] != CONTROL_LABEL], "Best-score gap: non-collapse zoom", "best_final_gap", "Gap"),
        (axes[1, 0], df, "Best validation step: all conditions", "best_step", "Step"),
        (axes[1, 1], df[df["paper_label"] != CONTROL_LABEL], "Best validation step: non-collapse zoom", "best_step", "Step"),
    ]
    for ax, sdf, title, column, xlabel in panels:
        colors = [COLOR_MAP.get(x, "#666666") for x in sdf["paper_label"]]
        ax.barh([_short_label(x) for x in sdf["paper_label"]], sdf[column], color=colors)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    fig.savefig(FIG_DIR / "best-final-gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_final_behavior_ranking() -> None:
    df = RUN_SUMMARY.copy()
    renamed = {
        "final_token_entropy_mean": "Final token entropy",
        "final_low_confidence_token_ratio": "Final low-conf ratio",
        "final_top1_prob_mean": "Final top-1 probability",
        "final_response_length": "Final response length",
    }

    fig, axes = plt.subplots(4, 2, figsize=(12.6, 13.6), constrained_layout=True)
    for row_idx, (column, title) in enumerate(renamed.items()):
        for col_idx, sdf in enumerate([df, df[df["paper_label"] != CONTROL_LABEL]]):
            ax = axes[row_idx, col_idx]
            sdf = sdf[["paper_label", column]].sort_values(column, ascending=(column == "final_top1_prob_mean"))
            colors = [COLOR_MAP.get(x, "#666666") for x in sdf["paper_label"]]
            ax.barh([_short_label(x) for x in sdf["paper_label"]], sdf[column], color=colors)
            suffix = "full scale" if col_idx == 0 else "non-collapse zoom"
            ax.set_title(f"{title}: {suffix}")
            ax.grid(axis="x", alpha=0.25, linestyle="--")

    fig.savefig(FIG_DIR / "final-behavior-ranking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _dynamic_summary() -> pd.DataFrame:
    rows = []
    for label in RUN_SUMMARY["paper_label"]:
        step_sub = STEP_DF[STEP_DF["paper_label"] == label].sort_values("step")
        train_sub = TRAIN_DF[TRAIN_DF["formal_label"] == label].sort_values("step")
        rows.append(
            {
                "formal_label": label,
                "val_reward_mean_delta": _delta(step_sub["score"]),
                "actor_entropy_delta": _delta(train_sub["actor_entropy"]),
                "val_token_entropy_mean_delta": _delta(step_sub["token_entropy_mean"]),
                "val_low_confidence_ratio_delta": _delta(step_sub["low_confidence_token_ratio"]),
                "val_token_response_length_mean_delta": _delta(step_sub["response_length"]),
            }
        )
    return pd.DataFrame(rows)


def render_gain_calibration_coupling() -> None:
    df = _dynamic_summary()
    x_specs = [
        ("actor_entropy_delta", "Actor entropy reduction"),
        ("val_token_entropy_mean_delta", "Validation token-entropy reduction"),
        ("val_low_confidence_ratio_delta", "Low-confidence reduction"),
        ("val_token_response_length_mean_delta", "Response-length reduction"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12.8, 13.4), constrained_layout=True)
    for row_idx, (column, title) in enumerate(x_specs):
        for col_idx, sdf in enumerate([df, df[df["formal_label"] != CONTROL_LABEL]]):
            ax = axes[row_idx, col_idx]
            x = -sdf[column]
            y = sdf["val_reward_mean_delta"]
            colors = [COLOR_MAP.get(label, "#666666") for label in sdf["formal_label"]]
            ax.scatter(x, y, s=105, c=colors, edgecolor="white", linewidth=0.8)
            for _, row in sdf.iterrows():
                ax.annotate(_short_label(row["formal_label"]), (-row[column] + 0.001, row["val_reward_mean_delta"] + 0.0015), fontsize=8)
            suffix = "full scale" if col_idx == 0 else "non-collapse zoom"
            ax.set_title(f"{title}: {suffix}")
            ax.set_xlabel("Magnitude")
            ax.set_ylabel("Validation score gain")
            ax.grid(alpha=0.25, linestyle="--")

    fig.savefig(FIG_DIR / "gain-calibration-coupling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_endpoint_frontier_panel() -> None:
    df = RUN_SUMMARY.copy()
    panels = [
        ("final_token_entropy_mean", "Final token entropy", True),
        ("final_low_confidence_token_ratio", "Final low-confidence ratio", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    for col_idx, (column, xlabel, invert) in enumerate(panels):
        for row_idx, sdf in enumerate([df, df[df["paper_label"] != CONTROL_LABEL]]):
            ax = axes[row_idx, col_idx]
            x = sdf[column]
            y = sdf["final_score"]
            colors = [COLOR_MAP.get(label, "#666666") for label in sdf["paper_label"]]
            ax.scatter(x, y, s=120, c=colors, edgecolor="white", linewidth=0.9)
            for _, row in sdf.iterrows():
                ax.annotate(_short_label(row["paper_label"]), (row[column] + 0.002, row["final_score"] + 0.0015), fontsize=8)
            suffix = "full scale" if row_idx == 0 else "non-collapse zoom"
            ax.set_title(f"{xlabel}: {suffix}")
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
