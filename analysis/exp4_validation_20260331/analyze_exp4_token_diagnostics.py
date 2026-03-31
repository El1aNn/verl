#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "token_outputs"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"

EXPERIMENT_ORDER = [
    "base_full",
    "base_oneshot",
    "instruct_full",
    "instruct_oneshot",
]

EXPERIMENT_LABEL = {
    "base_full": "Base-Full",
    "base_oneshot": "Base-OneShot",
    "instruct_full": "Instruct-Full",
    "instruct_oneshot": "Instruct-OneShot",
}

EOS_TEXT = "<|endoftext|>"


def normalize_experiment_name(zip_name: str) -> str:
    name = zip_name.lower()
    if "base_full" in name:
        return "base_full"
    if "base_oneshot" in name:
        return "base_oneshot"
    if "instruct_full" in name:
        return "instruct_full"
    if "instruct_oneshot" in name:
        return "instruct_oneshot"
    raise ValueError(f"Cannot infer experiment type from: {zip_name}")


def step_from_path(path: str) -> int:
    return int(Path(path).stem)


def extract_user_question(input_text: str) -> str:
    marker = "\nuser\n"
    assistant_marker = "\nassistant"
    if marker in input_text:
        body = input_text.split(marker, 1)[1]
        if assistant_marker in body:
            body = body.split(assistant_marker, 1)[0]
        return body.strip()
    return input_text.strip()


def eos_is_top1(token_row: dict) -> int:
    return int(token_row.get("top1_text") == EOS_TEXT or token_row.get("top1_token") == EOS_TEXT)


def top5_mass(token_row: dict) -> float:
    return float(sum(item.get("prob", 0.0) for item in token_row.get("topk", [])))


def collect_records():
    response_rows = []
    token_rows = []

    zip_paths = sorted(RAW_DIR.glob("*_validation.zip"))
    path_by_exp = {normalize_experiment_name(p.name): p for p in zip_paths}

    for exp in EXPERIMENT_ORDER:
        zip_path = path_by_exp[exp]
        with zipfile.ZipFile(zip_path, "r") as zf:
            jsonl_paths = sorted([p for p in zf.namelist() if p.endswith(".jsonl")], key=step_from_path)
            for jsonl_path in jsonl_paths:
                step = step_from_path(jsonl_path)
                trace_idx = 0
                with zf.open(jsonl_path, "r") as f:
                    for raw in f:
                        rec = json.loads(raw.decode("utf-8", errors="replace"))
                        token_diag = rec.get("token_diagnostics")
                        if not token_diag:
                            continue

                        question = extract_user_question(rec.get("input", ""))
                        qhash = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:16]
                        sample_id = f"{qhash}::{trace_idx}"
                        trace_idx += 1

                        prefix_len = len(token_diag)
                        total_tokens = int(rec.get("token_diagnostics_total_tokens", prefix_len))
                        response_rows.append(
                            {
                                "experiment": exp,
                                "experiment_label": EXPERIMENT_LABEL[exp],
                                "step": step,
                                "sample_idx": trace_idx - 1,
                                "sample_id": sample_id,
                                "question_hash": qhash,
                                "question": question,
                                "score": float(rec.get("score", np.nan)),
                                "response_length": float(rec.get("response_length", np.nan)),
                                "token_prefix_len": prefix_len,
                                "token_total_len": total_tokens,
                                "token_truncated": bool(rec.get("token_diagnostics_truncated", False)),
                                "answer_tail_confidence": float(rec.get("answer_tail_confidence", np.nan)),
                                "low_confidence_token_ratio": float(rec.get("low_confidence_token_ratio", np.nan)),
                                "token_entropy_mean": float(rec.get("token_entropy_mean", np.nan)),
                                "token_logprob_mean": float(rec.get("token_logprob_mean", np.nan)),
                                "eos_prob_mean": float(rec.get("eos_prob_mean", np.nan)),
                                "eos_prob_final": float(rec.get("eos_prob_final", np.nan)),
                                "top1_prob_mean": float(rec.get("top1_prob_mean", np.nan)),
                                "topk_mass_mean": float(rec.get("topk_mass_mean", np.nan)),
                            }
                        )

                        denom_prefix = max(prefix_len - 1, 1)
                        denom_total = max(total_tokens - 1, 1)
                        for tok in token_diag:
                            position = int(tok["position"])
                            prefix_progress = position / denom_prefix
                            global_progress = position / denom_total
                            position_bucket = min(int(prefix_progress * 10), 9)
                            token_rows.append(
                                {
                                    "experiment": exp,
                                    "experiment_label": EXPERIMENT_LABEL[exp],
                                    "step": step,
                                    "sample_idx": trace_idx - 1,
                                    "sample_id": sample_id,
                                    "score": float(rec.get("score", np.nan)),
                                    "position": position,
                                    "position_bucket": position_bucket,
                                    "prefix_progress": prefix_progress,
                                    "global_progress": global_progress,
                                    "logprob": float(tok.get("logprob", np.nan)),
                                    "prob": float(tok.get("prob", np.nan)),
                                    "entropy": float(tok.get("entropy", np.nan)),
                                    "rollout_logprob": float(tok.get("rollout_logprob", np.nan)),
                                    "logprob_delta_from_rollout": float(tok.get("logprob_delta_from_rollout", np.nan)),
                                    "eos_prob": float(tok.get("eos_prob", np.nan)),
                                    "top1_prob": float(tok.get("top1_prob", np.nan)),
                                    "top5_mass": top5_mass(tok),
                                    "chosen_is_top1": int(tok.get("token_id") == tok.get("top1_token_id")),
                                    "eos_in_topk": int(bool(tok.get("eos_in_topk", False))),
                                    "eos_is_top1": eos_is_top1(tok),
                                }
                            )
                print(f"[{exp}] step={step} traced={trace_idx}")

    return pd.DataFrame(response_rows), pd.DataFrame(token_rows)


def build_tables(response_df: pd.DataFrame, token_df: pd.DataFrame):
    response_step = (
        response_df.groupby(["experiment", "experiment_label", "step"], as_index=False)
        .agg(
            traced_samples=("sample_id", "count"),
            score=("score", "mean"),
            response_length=("response_length", "mean"),
            token_total_len=("token_total_len", "mean"),
            answer_tail_confidence=("answer_tail_confidence", "mean"),
            low_confidence_token_ratio=("low_confidence_token_ratio", "mean"),
            token_entropy_mean=("token_entropy_mean", "mean"),
            token_logprob_mean=("token_logprob_mean", "mean"),
            eos_prob_mean=("eos_prob_mean", "mean"),
            eos_prob_final=("eos_prob_final", "mean"),
            top1_prob_mean=("top1_prob_mean", "mean"),
            topk_mass_mean=("topk_mass_mean", "mean"),
        )
    )

    token_step = (
        token_df.groupby(["experiment", "experiment_label", "step"], as_index=False)
        .agg(
            traced_tokens=("position", "count"),
            logprob=("logprob", "mean"),
            prob=("prob", "mean"),
            entropy=("entropy", "mean"),
            logprob_delta_from_rollout=("logprob_delta_from_rollout", "mean"),
            abs_logprob_delta_from_rollout=("logprob_delta_from_rollout", lambda s: np.mean(np.abs(s))),
            eos_prob=("eos_prob", "mean"),
            top1_prob=("top1_prob", "mean"),
            top5_mass=("top5_mass", "mean"),
            chosen_is_top1=("chosen_is_top1", "mean"),
            eos_in_topk=("eos_in_topk", "mean"),
            eos_is_top1=("eos_is_top1", "mean"),
        )
    )

    token_position_final = (
        token_df[token_df["step"] == token_df["step"].max()]
        .groupby(["experiment", "experiment_label", "position_bucket"], as_index=False)
        .agg(
            logprob=("logprob", "mean"),
            prob=("prob", "mean"),
            entropy=("entropy", "mean"),
            eos_prob=("eos_prob", "mean"),
            top1_prob=("top1_prob", "mean"),
            top5_mass=("top5_mass", "mean"),
        )
    )

    token_final_correctness = (
        token_df[token_df["step"] == token_df["step"].max()]
        .assign(correctness=lambda df: np.where(df["score"] >= 0.5, "correct", "wrong"))
        .groupby(["experiment", "experiment_label", "correctness"], as_index=False)
        .agg(
            traced_tokens=("position", "count"),
            logprob=("logprob", "mean"),
            prob=("prob", "mean"),
            entropy=("entropy", "mean"),
            eos_prob=("eos_prob", "mean"),
            top1_prob=("top1_prob", "mean"),
            top5_mass=("top5_mass", "mean"),
        )
    )

    first_step = int(token_step["step"].min())
    final_step = int(token_step["step"].max())
    delta = token_step[token_step["step"].isin([first_step, final_step])].copy()
    delta = delta.pivot(index=["experiment", "experiment_label"], columns="step")
    delta.columns = [f"{col}_{step}" for col, step in delta.columns]
    delta = delta.reset_index()
    for metric in ["logprob", "prob", "entropy", "eos_prob", "top1_prob", "top5_mass", "chosen_is_top1"]:
        delta[f"{metric}_delta_final_minus_step0"] = delta[f"{metric}_{final_step}"] - delta[f"{metric}_{first_step}"]

    return response_step, token_step, token_position_final, token_final_correctness, delta


def make_plots(response_step, token_step, token_position_final, token_final_correctness, delta):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    sns.lineplot(data=response_step, x="step", y="score", hue="experiment_label", marker="o", ax=axes[0])
    axes[0].set_title("Traced Probe Accuracy Over Steps")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean score")
    sns.lineplot(data=response_step, x="step", y="token_total_len", hue="experiment_label", marker="o", ax=axes[1], legend=False)
    axes[1].set_title("Traced Probe Total Response Length Over Steps")
    axes[1].set_ylabel("Tokens")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_traced_probe_over_steps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    metric_grid = [
        ("logprob", "Mean token logprob"),
        ("entropy", "Mean token entropy"),
        ("top1_prob", "Mean top1 prob"),
        ("eos_prob", "Mean EOS prob"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_grid):
        sns.lineplot(data=token_step, x="step", y=metric, hue="experiment_label", marker="o", ax=ax, legend=False)
        ax.set_title(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS_DIR / "02_token_step_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    curve_metrics = [
        ("logprob", "Final-step prefix logprob"),
        ("entropy", "Final-step prefix entropy"),
        ("eos_prob", "Final-step prefix EOS prob"),
    ]
    plot_df = token_position_final.copy()
    plot_df["bucket_label"] = plot_df["position_bucket"] + 1
    for ax, (metric, title) in zip(axes.flat, curve_metrics):
        sns.lineplot(data=plot_df, x="bucket_label", y=metric, hue="experiment_label", marker="o", ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("Prefix decile (stored 96-token prefix)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(PLOTS_DIR / "03_final_prefix_curves.png", dpi=180)
    plt.close(fig)

    delta_plot = delta[
        [
            "experiment_label",
            "logprob_delta_final_minus_step0",
            "entropy_delta_final_minus_step0",
            "top1_prob_delta_final_minus_step0",
            "eos_prob_delta_final_minus_step0",
        ]
    ].melt(id_vars="experiment_label", var_name="metric", value_name="delta")
    delta_plot["metric"] = delta_plot["metric"].map(
        {
            "logprob_delta_final_minus_step0": "logprob",
            "entropy_delta_final_minus_step0": "entropy",
            "top1_prob_delta_final_minus_step0": "top1_prob",
            "eos_prob_delta_final_minus_step0": "eos_prob",
        }
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(data=delta_plot, x="experiment_label", y="delta", hue="metric")
    plt.title("Final minus Step0 Token-Metric Delta (traced prefix)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_token_delta_step0_to_final.png", dpi=180)
    plt.close()

    if not token_final_correctness.empty and token_final_correctness["correctness"].nunique() > 1:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=token_final_correctness, x="experiment_label", y="entropy", hue="correctness")
        plt.title("Final-step Token Entropy: Correct vs Wrong Traces")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_correct_vs_wrong_entropy.png", dpi=180)
        plt.close()


def df_to_markdown_fallback(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |\n| (none) |"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            s = "" if pd.isna(v) else str(v)
            s = s.replace("\n", " ").replace("|", "\\|")
            cells.append(s)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def make_report(response_df, response_step, token_step, token_position_final, token_final_correctness, delta) -> str:
    final_step = int(response_step["step"].max())
    first_step = int(response_step["step"].min())
    final_resp = response_step[response_step["step"] == final_step].copy()
    final_tok = token_step[token_step["step"] == final_step].copy()
    question = response_df["question"].iloc[0]
    unique_questions = response_df["question_hash"].nunique()
    traced_per_step = int(response_step["traced_samples"].median())
    prefix_len = int(response_df["token_prefix_len"].median())

    merged_final = final_resp.merge(final_tok, on=["experiment", "experiment_label", "step"], suffixes=("_resp", "_tok"))
    merged_final = merged_final.sort_values("score", ascending=False)
    best_score = float(merged_final["score"].max())
    best_labels = merged_final.loc[merged_final["score"] == best_score, "experiment_label"].tolist()

    lines = []
    lines.append("# Token Diagnostics 补充分析（四种训练方式）")
    lines.append("")
    lines.append("## 1. 口径与限制")
    lines.append(f"- `token_diagnostics` 只存在于 traced subset：每个 step 每个实验 {traced_per_step} 条 traced rollout。")
    lines.append(f"- traced subset 实际只覆盖 **{unique_questions} 道 probe 题**，四个实验追踪的是同一题的 8 次采样。")
    lines.append(f"- 保存的是前缀 token 轨迹：每条只保留前 {prefix_len} 个 token 左右，且 `token_diagnostics_truncated=true`。")
    lines.append(f"- 因此本报告回答的是：**在固定 probe 题上，训练如何改变早期 token 行为**，不应直接外推到全部 500 题。")
    lines.append("")
    lines.append("Probe 题目：")
    lines.append(f"- `{question}`")
    lines.append("")
    lines.append("## 2. 总体观察")
    lines.append(f"- final step 上，traced probe score 最高的是 **{', '.join(best_labels)}**，并列达到 {best_score:.3f}。")
    lines.append("- instruct 系列从 step0 起就表现出更高 token confidence、更低 entropy、更短输出；训练后变化不大，说明它们一开始就在较稳定区域。")
    lines.append("- base 系列训练带来的变化更大：prefix token 的 logprob 提升、entropy 下降、输出明显缩短，说明训练主要在修正早期生成的不稳定与冗长。")
    lines.append("- `Base-OneShot` 在 final step 仍落后，说明 one-shot 方案对 base 模型的 token 级稳定化不如 full 训练充分。")
    lines.append("")
    lines.append("## 3. Final-step 指标")
    show_cols = [
        "experiment_label",
        "score",
        "token_total_len",
        "answer_tail_confidence",
        "low_confidence_token_ratio",
        "logprob",
        "entropy",
        "top1_prob",
        "eos_prob",
        "top5_mass",
        "chosen_is_top1",
    ]
    lines.append(df_to_markdown_fallback(merged_final[show_cols].round(4)))
    lines.append("")
    lines.append("## 4. Step0 -> Final 的 token 级变化")
    delta_cols = [
        "experiment_label",
        "logprob_delta_final_minus_step0",
        "entropy_delta_final_minus_step0",
        "top1_prob_delta_final_minus_step0",
        "eos_prob_delta_final_minus_step0",
    ]
    lines.append(df_to_markdown_fallback(delta[delta_cols].round(4)))
    lines.append("")
    if not token_final_correctness.empty and token_final_correctness["correctness"].nunique() > 1:
        lines.append("## 5. Final-step 正误分组")
        lines.append(df_to_markdown_fallback(token_final_correctness.round(4)))
        lines.append("")
    lines.append("## 6. 图表")
    lines.append("- ![traced_probe](token_outputs/plots/01_traced_probe_over_steps.png)")
    lines.append("- ![token_step](token_outputs/plots/02_token_step_metrics.png)")
    lines.append("- ![prefix_curves](token_outputs/plots/03_final_prefix_curves.png)")
    lines.append("- ![delta](token_outputs/plots/04_token_delta_step0_to_final.png)")
    if not token_final_correctness.empty and token_final_correctness["correctness"].nunique() > 1:
        lines.append("- ![correct_vs_wrong](token_outputs/plots/05_correct_vs_wrong_entropy.png)")
    lines.append("")
    lines.append("## 7. 结论建议")
    lines.append("- 如果你想用 token 级指标做训练诊断，当前 traced subset 更适合作为 **固定 probe**，用于监控训练是否让模型更短、更稳、更早收敛。")
    lines.append("- 如果你想把 token 分析推广到任务层结论，建议下一轮把 `token_diagnostics` 覆盖到更多 probe 题，而不是只保留 1 道题。")
    lines.append("- 对现有结果，最可信的结论是：**instruct 初始化已经把前缀 token 组织得更稳，base 训练主要是在补这个短板，而 one-shot 对 base 的修复不彻底。**")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    response_df, token_df = collect_records()
    response_step, token_step, token_position_final, token_final_correctness, delta = build_tables(response_df, token_df)

    response_df.to_csv(TABLES_DIR / "traced_response_rows.csv", index=False)
    token_df.to_csv(TABLES_DIR / "traced_token_rows.csv", index=False)
    response_step.to_csv(TABLES_DIR / "traced_response_step_summary.csv", index=False)
    token_step.to_csv(TABLES_DIR / "traced_token_step_summary.csv", index=False)
    token_position_final.to_csv(TABLES_DIR / "traced_token_position_final.csv", index=False)
    token_final_correctness.to_csv(TABLES_DIR / "traced_token_final_correctness.csv", index=False)
    delta.to_csv(TABLES_DIR / "traced_token_delta_step0_to_final.csv", index=False)

    make_plots(response_step, token_step, token_position_final, token_final_correctness, delta)

    report = make_report(response_df, response_step, token_step, token_position_final, token_final_correctness, delta)
    report_path = ROOT / "REPORT_exp4_token_diagnostics_20260331_v1.md"
    report_path.write_text(report, encoding="utf-8")

    print("=== Done ===")
    print(f"report: {report_path}")
    print(f"tables: {TABLES_DIR}")
    print(f"plots:  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
