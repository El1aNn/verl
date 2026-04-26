from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/root/rl/verl/analysis/overleaf_bundle_20260406")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
CKPT_ROOT = Path("/root/rl/verl/ckpts/verl_grpo_dsr_sub_baseline")

RUNS = {
    "Base baseline (n=8)": CKPT_ROOT / "base_align_exp1_20260401_141352" / "validation",
    "Base full-data": CKPT_ROOT / "base_align_exp1_full_20260413_215447" / "validation",
    "Base small-group ablation (n=1)": CKPT_ROOT / "base_c_n1_val8_align_exp1_2gpu_20260406_195153" / "validation",
    "Base weak-regularization variant": CKPT_ROOT / "base_d_weak_constraint_20260402_003007" / "validation",
    "Instruct baseline (n=8)": CKPT_ROOT / "instruct_c_n8_baseline_rerun180_20260404_113902" / "validation",
    "Instruct full-data": CKPT_ROOT / "instruct_align_exp1_full_20260414_101916" / "validation",
}

MATCHED_TRACKING_LABELS = [
    "Base baseline (n=8)",
    "Base full-data",
    "Instruct baseline (n=8)",
    "Instruct full-data",
]
FINAL_STEP = 180
STEP_RANGE = list(range(0, 181, 6))

COLOR_CORRECT = "#1F5F8B"
COLOR_WRONG = "#C56A2D"
TRACK_COLORS = {
    "Base baseline (n=8)": "#1F5F8B",
    "Base full-data": "#0B3954",
    "Instruct baseline (n=8)": "#2C7A7B",
    "Instruct full-data": "#005F73",
}
TRACK_STYLES = {
    "Base baseline (n=8)": "-",
    "Base full-data": "--",
    "Instruct baseline (n=8)": "-",
    "Instruct full-data": "--",
}
TEXT = "#222222"
GRID = "#DDD7CD"
BG = "#FCFBF8"


def _short_label(label: str) -> str:
    return (
        label.replace("Base baseline (n=8)", "Base n=8")
        .replace("Base full-data", "Base full")
        .replace("Base small-group ablation (n=1)", "Base n=1")
        .replace("Base weak-regularization variant", "Base weak-reg")
        .replace("Instruct baseline (n=8)", "Instr n=8")
        .replace("Instruct full-data", "Instr full")
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def build_final_summary() -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        "token_entropy_mean",
        "top1_prob_mean",
        "eos_prob_mean",
        "answer_tail_confidence",
    ]
    for label, run_dir in RUNS.items():
        grouped = {
            "correct": {metric: [] for metric in metrics},
            "wrong": {metric: [] for metric in metrics},
        }
        for row in _read_jsonl(run_dir / f"{FINAL_STEP}.jsonl"):
            entropy = row.get("token_entropy_mean")
            if not _is_number(entropy):
                continue
            group = "correct" if float(row.get("score", 0.0)) >= 0.5 else "wrong"
            for metric in metrics:
                value = row.get(metric)
                if _is_number(value):
                    grouped[group][metric].append(float(value))

        correct_n = len(grouped["correct"]["token_entropy_mean"])
        wrong_n = len(grouped["wrong"]["token_entropy_mean"])
        for metric in metrics:
            correct_vals = grouped["correct"][metric]
            wrong_vals = grouped["wrong"][metric]
            rows.append(
                {
                    "paper_label": label,
                    "metric": metric,
                    "correct_mean": _mean(correct_vals),
                    "wrong_mean": _mean(wrong_vals),
                    "correct_n": correct_n,
                    "wrong_n": wrong_n,
                }
            )
    return pd.DataFrame(rows)


def build_step_curves() -> pd.DataFrame:
    rows: list[dict] = []
    for label in MATCHED_TRACKING_LABELS:
        for step in STEP_RANGE:
            grouped = {
                "correct": {"token_entropy_mean": [], "top1_prob_mean": []},
                "wrong": {"token_entropy_mean": [], "top1_prob_mean": []},
            }
            for row in _read_jsonl(RUNS[label] / f"{step}.jsonl"):
                entropy = row.get("token_entropy_mean")
                top1 = row.get("top1_prob_mean")
                if not (_is_number(entropy) and _is_number(top1)):
                    continue
                group = "correct" if float(row.get("score", 0.0)) >= 0.5 else "wrong"
                grouped[group]["token_entropy_mean"].append(float(entropy))
                grouped[group]["top1_prob_mean"].append(float(top1))

            correct_entropy = _mean(grouped["correct"]["token_entropy_mean"])
            wrong_entropy = _mean(grouped["wrong"]["token_entropy_mean"])
            correct_top1 = _mean(grouped["correct"]["top1_prob_mean"])
            wrong_top1 = _mean(grouped["wrong"]["top1_prob_mean"])
            rows.append(
                {
                    "paper_label": label,
                    "step": step,
                    "correct_token_entropy_mean": correct_entropy,
                    "wrong_token_entropy_mean": wrong_entropy,
                    "entropy_gap_wrong_minus_correct": wrong_entropy - correct_entropy,
                    "correct_top1_prob_mean": correct_top1,
                    "wrong_top1_prob_mean": wrong_top1,
                    "top1_gap_correct_minus_wrong": correct_top1 - wrong_top1,
                    "correct_n": len(grouped["correct"]["token_entropy_mean"]),
                    "wrong_n": len(grouped["wrong"]["token_entropy_mean"]),
                }
            )
    return pd.DataFrame(rows)


def render_panel(final_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "token_entropy_mean": "Final-step grouped token entropy",
        "top1_prob_mean": "Final-step grouped top-1 probability",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
    fig.patch.set_facecolor(BG)

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color("#6F6A63")
            spine.set_linewidth(0.9)
        ax.tick_params(colors=TEXT)

    order = list(RUNS.keys())
    x = list(range(len(order)))
    width = 0.34

    for ax, metric in zip(axes[0], ["token_entropy_mean", "top1_prob_mean"]):
        sub = final_df[final_df["metric"] == metric].set_index("paper_label").loc[order].reset_index()
        ax.bar(
            [i - width / 2 for i in x],
            sub["correct_mean"],
            width=width,
            color=COLOR_CORRECT,
            label="Correct",
        )
        ax.bar(
            [i + width / 2 for i in x],
            sub["wrong_mean"],
            width=width,
            color=COLOR_WRONG,
            label="Wrong",
        )
        ax.set_xticks(x, [_short_label(label) for label in order], rotation=22)
        ax.set_title(metric_titles[metric], color=TEXT, fontsize=13)
        if metric == "token_entropy_mean":
            ax.set_ylim(0.0, max(sub["wrong_mean"].max(), sub["correct_mean"].max()) + 0.07)
        else:
            ax.set_ylim(0.80, 0.975)
        ax.set_xlabel("Formal condition", color=TEXT)

    axes[0, 0].set_ylabel("Mean entropy", color=TEXT)
    axes[0, 1].set_ylabel("Mean top-1 probability", color=TEXT)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")

    for ax, metric, ylabel, title in [
        (
            axes[1, 0],
            "entropy_gap_wrong_minus_correct",
            "Wrong - correct entropy",
            "Correctness gap in entropy",
        ),
        (
            axes[1, 1],
            "top1_gap_correct_minus_wrong",
            "Correct - wrong top-1 probability",
            "Correctness gap in top-1 probability",
        ),
    ]:
        for label in MATCHED_TRACKING_LABELS:
            sub = step_df[step_df["paper_label"] == label].sort_values("step")
            ax.plot(
                sub["step"],
                sub[metric],
                TRACK_STYLES[label],
                marker="o",
                markersize=3.5,
                linewidth=2.0,
                color=TRACK_COLORS[label],
                label=_short_label(label),
            )
        ax.set_title(title, color=TEXT, fontsize=13)
        ax.set_xlabel("Training step", color=TEXT)
        ax.set_ylabel(ylabel, color=TEXT)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.axhline(0.0, color="#6F6A63", linewidth=0.9, alpha=0.7)

    axes[1, 0].legend(frameon=False, ncol=2, loc="upper right", fontsize=8)

    fig.savefig(FIG_DIR / "correct-wrong-support-panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_df = build_final_summary()
    step_df = build_step_curves()
    final_df.to_csv(DATA_DIR / "correct-wrong-final-summary.csv", index=False)
    step_df.to_csv(DATA_DIR / "correct-wrong-step-curves.csv", index=False)
    render_panel(final_df, step_df)
    print("wrote", DATA_DIR / "correct-wrong-final-summary.csv")
    print("wrote", DATA_DIR / "correct-wrong-step-curves.csv")
    print("wrote", FIG_DIR / "correct-wrong-support-panel.png")


if __name__ == "__main__":
    main()
