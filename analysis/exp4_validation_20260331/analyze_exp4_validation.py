#!/usr/bin/env python3
import json
import math
import re
import zipfile
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"

METRIC_COLUMNS = [
    "score",
    "reward",
    "response_length",
    "token_logprob_mean",
    "token_entropy_mean",
    "low_confidence_token_ratio",
    "answer_tail_confidence",
    "rollout_logprob_delta_abs_mean",
    "eos_prob_mean",
    "eos_prob_final",
    "top1_prob_mean",
    "topk_mass_mean",
]

CASE_NUMERIC_COLUMNS = [
    "score",
    "response_length",
    "answer_tail_confidence",
    "low_confidence_token_ratio",
    "token_entropy_mean",
    "rollout_logprob_delta_abs_mean",
    "eos_prob_final",
]

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

TOKEN_DIAG_PATTERN = re.compile(r",\s*\"token_diagnostics\"\s*:")


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


def strip_token_diagnostics(json_line: str) -> str:
    match = TOKEN_DIAG_PATTERN.search(json_line)
    if not match:
        return json_line
    return json_line[: match.start()] + "}"


def parse_record(json_line: str) -> dict:
    compact = strip_token_diagnostics(json_line)
    return json.loads(compact)


def step_from_path(path: str) -> int:
    return int(Path(path).stem)


def extract_user_question(input_text: str) -> str:
    if not isinstance(input_text, str):
        return ""
    marker = "\nuser\n"
    assistant_marker = "\nassistant"
    if marker in input_text:
        body = input_text.split(marker, 1)[1]
        if assistant_marker in body:
            body = body.split(assistant_marker, 1)[0]
        return body.strip()
    return input_text.strip()


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def collect_from_zip(zip_path: Path, exp_name: str):
    step_acc: Dict[int, dict] = {}
    final_cases: Dict[str, dict] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        jsonl_paths = sorted([p for p in zf.namelist() if p.endswith(".jsonl")], key=step_from_path)
        if not jsonl_paths:
            raise ValueError(f"No jsonl files found in {zip_path}")

        final_step = step_from_path(jsonl_paths[-1])
        print(f"[{exp_name}] files={len(jsonl_paths)} final_step={final_step}")

        final_seen = Counter()
        for jsonl_path in jsonl_paths:
            step = step_from_path(jsonl_path)
            acc = step_acc.setdefault(step, {"n_cases": 0})
            for metric in METRIC_COLUMNS:
                acc.setdefault(f"{metric}_sum", 0.0)
                acc.setdefault(f"{metric}_cnt", 0)

            with zf.open(jsonl_path, "r") as f:
                for raw in f:
                    rec = parse_record(raw.decode("utf-8", errors="replace"))
                    acc["n_cases"] += 1

                    for metric in METRIC_COLUMNS:
                        value = rec.get(metric)
                        if is_number(value):
                            acc[f"{metric}_sum"] += float(value)
                            acc[f"{metric}_cnt"] += 1

                    if step == final_step:
                        raw_input = rec.get("input", "") or ""
                        qhash = hashlib.sha1(raw_input.encode("utf-8", errors="ignore")).hexdigest()[:16]
                        occ = final_seen[qhash]
                        final_seen[qhash] += 1
                        case_id = f"{qhash}::{occ}"
                        case_rec = {
                            "case_id": case_id,
                            "raw_uid": rec.get("uid"),
                            "question": extract_user_question(rec.get("input", "")),
                            "gts": rec.get("gts", ""),
                            "output": rec.get("output", ""),
                        }
                        for col in CASE_NUMERIC_COLUMNS:
                            value = rec.get(col)
                            case_rec[col] = float(value) if is_number(value) else np.nan
                        final_cases[case_id] = case_rec

    step_rows = []
    for step in sorted(step_acc):
        acc = step_acc[step]
        row = {
            "experiment": exp_name,
            "experiment_label": EXPERIMENT_LABEL[exp_name],
            "step": step,
            "n_cases": acc["n_cases"],
        }
        for metric in METRIC_COLUMNS:
            cnt = acc[f"{metric}_cnt"]
            row[metric] = (acc[f"{metric}_sum"] / cnt) if cnt > 0 else np.nan
        step_rows.append(row)

    return pd.DataFrame(step_rows), final_cases


def compute_case_matrix(final_cases_by_exp: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    shared = set.intersection(*[set(d.keys()) for d in final_cases_by_exp.values()])
    rows: List[dict] = []

    for case_id in sorted(shared):
        row = {"uid": case_id}
        ref = final_cases_by_exp[EXPERIMENT_ORDER[0]][case_id]
        row["raw_uid"] = ref.get("raw_uid")
        row["question"] = ref.get("question", "")
        row["gts"] = ref.get("gts", "")

        scores = []
        for exp in EXPERIMENT_ORDER:
            rec = final_cases_by_exp[exp][case_id]
            row[f"{exp}_score"] = rec.get("score", np.nan)
            row[f"{exp}_response_length"] = rec.get("response_length", np.nan)
            row[f"{exp}_answer_tail_confidence"] = rec.get("answer_tail_confidence", np.nan)
            row[f"{exp}_low_confidence_token_ratio"] = rec.get("low_confidence_token_ratio", np.nan)
            row[f"{exp}_token_entropy_mean"] = rec.get("token_entropy_mean", np.nan)
            row[f"{exp}_output"] = rec.get("output", "")
            score = rec.get("score", np.nan)
            scores.append(score if is_number(score) else np.nan)

        score_arr = np.array(scores, dtype=float)
        score_valid = score_arr[np.isfinite(score_arr)]
        row["score_mean"] = float(np.mean(score_valid)) if score_valid.size else np.nan
        row["score_std"] = float(np.std(score_valid)) if score_valid.size else np.nan
        row["n_correct"] = int(np.sum(score_valid >= 0.5)) if score_valid.size else 0

        bf = row["base_full_score"]
        bo = row["base_oneshot_score"]
        inf = row["instruct_full_score"]
        ino = row["instruct_oneshot_score"]

        row["delta_base_oneshot_minus_full"] = bo - bf
        row["delta_instruct_oneshot_minus_full"] = ino - inf
        row["delta_instruct_full_minus_base_full"] = inf - bf
        row["delta_instruct_oneshot_minus_base_oneshot"] = ino - bo

        if row["n_correct"] == 4:
            row["outcome_pattern"] = "all_correct"
        elif row["n_correct"] == 0:
            row["outcome_pattern"] = "all_wrong"
        else:
            row["outcome_pattern"] = "mixed"

        rows.append(row)

    return pd.DataFrame(rows)


def build_experiment_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for exp in EXPERIMENT_ORDER:
        sdf = step_df[step_df["experiment"] == exp].sort_values("step")
        first = sdf.iloc[0]
        final = sdf.iloc[-1]
        best_idx = sdf["score"].idxmax()
        best = sdf.loc[best_idx]

        tail = sdf.tail(5)
        slope = np.polyfit(tail["step"], tail["score"], 1)[0] if len(tail) >= 2 else np.nan

        summary_rows.append(
            {
                "experiment": exp,
                "experiment_label": EXPERIMENT_LABEL[exp],
                "step0_score": float(first["score"]),
                "final_step": int(final["step"]),
                "final_score": float(final["score"]),
                "best_step": int(best["step"]),
                "best_score": float(best["score"]),
                "score_gain_vs_step0": float(final["score"] - first["score"]),
                "late5_score_slope": float(slope),
                "final_response_length": float(final["response_length"]),
                "final_answer_tail_confidence": float(final["answer_tail_confidence"]),
                "final_low_confidence_ratio": float(final["low_confidence_token_ratio"]),
                "final_entropy": float(final["token_entropy_mean"]),
            }
        )

    return pd.DataFrame(summary_rows)


def build_pairwise_winrate(case_df: pd.DataFrame) -> pd.DataFrame:
    data = []
    for exp_i in EXPERIMENT_ORDER:
        row = []
        s_i = case_df[f"{exp_i}_score"].to_numpy(dtype=float)
        for exp_j in EXPERIMENT_ORDER:
            s_j = case_df[f"{exp_j}_score"].to_numpy(dtype=float)
            win = np.where(s_i > s_j, 1.0, np.where(s_i < s_j, 0.0, 0.5))
            row.append(float(np.mean(win)))
        data.append(row)

    return pd.DataFrame(data, index=[EXPERIMENT_LABEL[e] for e in EXPERIMENT_ORDER], columns=[EXPERIMENT_LABEL[e] for e in EXPERIMENT_ORDER])


def detect_failure_tags(row: pd.Series, p90_len: float) -> str:
    tags = set()
    for exp in EXPERIMENT_ORDER:
        score = row[f"{exp}_score"]
        if not is_number(score) or score >= 0.5:
            continue

        output = row.get(f"{exp}_output", "") or ""
        if "\\boxed" not in output:
            tags.add("missing_boxed")
        if is_number(row.get(f"{exp}_response_length")) and row[f"{exp}_response_length"] > p90_len:
            tags.add("overlong_reasoning")
        if is_number(row.get(f"{exp}_answer_tail_confidence")) and row[f"{exp}_answer_tail_confidence"] < 0.5:
            tags.add("low_tail_confidence")
        if is_number(row.get(f"{exp}_low_confidence_token_ratio")) and row[f"{exp}_low_confidence_token_ratio"] > 0.2:
            tags.add("high_token_uncertainty")
        if is_number(row.get(f"{exp}_token_entropy_mean")) and row[f"{exp}_token_entropy_mean"] > 0.8:
            tags.add("high_entropy")

    return ";".join(sorted(tags)) if tags else ""


def make_plots(step_df: pd.DataFrame, summary_df: pd.DataFrame, pairwise_df: pd.DataFrame):
    sns.set_theme(style="whitegrid")

    # 1. Accuracy over steps
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=step_df, x="step", y="score", hue="experiment_label", marker="o")
    plt.title("Validation Accuracy (score) Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Mean score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_score_over_steps.png", dpi=180)
    plt.close()

    # 2. Confidence diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    metric_grid = [
        ("answer_tail_confidence", "Answer Tail Confidence"),
        ("low_confidence_token_ratio", "Low-Confidence Token Ratio"),
        ("token_entropy_mean", "Token Entropy Mean"),
        ("rollout_logprob_delta_abs_mean", "Rollout LogProb Delta Abs Mean"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_grid):
        sns.lineplot(data=step_df, x="step", y=metric, hue="experiment_label", ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("Step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Validation Diagnostics Over Steps", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS_DIR / "02_diagnostics_over_steps.png", dpi=180)
    plt.close(fig)

    # 3. Final and best score bars
    bar_df = summary_df.melt(
        id_vars=["experiment_label"],
        value_vars=["final_score", "best_score"],
        var_name="metric",
        value_name="value",
    )
    bar_df["metric"] = bar_df["metric"].map({"final_score": "Final", "best_score": "Best"})

    plt.figure(figsize=(10, 6))
    sns.barplot(data=bar_df, x="experiment_label", y="value", hue="metric")
    plt.title("Final vs Best Validation Score")
    plt.xlabel("")
    plt.ylabel("Mean score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_final_vs_best_score.png", dpi=180)
    plt.close()

    # 4. Final metrics heatmap
    heat_cols = ["final_score", "final_response_length", "final_answer_tail_confidence", "final_low_confidence_ratio", "final_entropy"]
    heat_df = summary_df.set_index("experiment_label")[heat_cols].copy()
    norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)

    plt.figure(figsize=(10, 5))
    sns.heatmap(norm, annot=heat_df.round(4), fmt="", cmap="YlGnBu")
    plt.title("Final-Step Metrics (cell annotation=raw value, color=normalized)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_final_metrics_heatmap.png", dpi=180)
    plt.close()

    # 5. Pairwise winrate heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(pairwise_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Case-Level Pairwise Win Rate (row vs column)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_pairwise_winrate_heatmap.png", dpi=180)
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


def make_report(step_df: pd.DataFrame, summary_df: pd.DataFrame, case_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> str:
    total_cases = len(case_df)
    cases_per_step = int(round(float(step_df["n_cases"].median())))
    unique_questions = int(case_df["uid"].astype(str).str.split("::").str[0].nunique())
    pattern_counts = case_df["outcome_pattern"].value_counts().to_dict()

    mixed_df = case_df[case_df["outcome_pattern"] == "mixed"].copy()
    mixed_df = mixed_df.sort_values(["score_std", "score_mean"], ascending=[False, True]).head(10)

    all_wrong_df = case_df[case_df["outcome_pattern"] == "all_wrong"].copy().head(10)

    best_final_row = summary_df.sort_values("final_score", ascending=False).iloc[0]
    best_exp = best_final_row["experiment"]
    best_exp_label = best_final_row["experiment_label"]

    best_model_fail = case_df[case_df[f"{best_exp}_score"] < 0.5].copy()
    best_model_fail = best_model_fail.sort_values("score_mean", ascending=True).head(10)

    lines = []
    lines.append("# 4次实验 Validation 对比分析（初版）")
    lines.append("")
    lines.append("## 1. 数据范围")
    lines.append(f"- 分析对象：4 个实验 x 31 个 step（0 到 180）")
    lines.append(f"- Case 粒度：每个 step {cases_per_step} 条采样（约 {unique_questions} 题 x 8 次采样），final step 可对齐 case 数量：{total_cases}")
    lines.append("- 统计口径：对每个 step 的 case 指标取均值；case 对比以 final step 为主")
    lines.append("")
    lines.append("## 2. 总体结论")
    lines.append(f"- final score 最优实验：**{best_exp_label}**（{best_final_row['final_score']:.4f}）")
    lines.append(f"- best score 最优实验：**{summary_df.loc[summary_df['best_score'].idxmax(), 'experiment_label']}**（{summary_df['best_score'].max():.4f}）")
    lines.append(f"- 全部实验都答对的 case：{pattern_counts.get('all_correct', 0)}")
    lines.append(f"- 全部实验都答错的 case：{pattern_counts.get('all_wrong', 0)}")
    lines.append(f"- 结果分歧（mixed）case：{pattern_counts.get('mixed', 0)}")
    lines.append("")

    lines.append("## 3. 图表")
    lines.append("- ![score_over_steps](outputs/plots/01_score_over_steps.png)")
    lines.append("- ![diagnostics](outputs/plots/02_diagnostics_over_steps.png)")
    lines.append("- ![final_vs_best](outputs/plots/03_final_vs_best_score.png)")
    lines.append("- ![final_metrics_heatmap](outputs/plots/04_final_metrics_heatmap.png)")
    lines.append("- ![pairwise_winrate](outputs/plots/05_pairwise_winrate_heatmap.png)")
    lines.append("")

    lines.append("## 4. 关键数值（summary）")
    lines.append(summary_df[[
        "experiment_label",
        "step0_score",
        "final_score",
        "best_step",
        "best_score",
        "score_gain_vs_step0",
        "final_response_length",
        "final_answer_tail_confidence",
        "final_low_confidence_ratio",
    ]].round(4).pipe(df_to_markdown_fallback))
    lines.append("")

    lines.append("## 5. Case 级观察")
    lines.append("### 5.1 分歧最大的 10 个 case（按 score_std）")
    if len(mixed_df) > 0:
        tmp = mixed_df[[
            "uid",
            "score_mean",
            "score_std",
            "base_full_score",
            "base_oneshot_score",
            "instruct_full_score",
            "instruct_oneshot_score",
            "question",
        ]].copy()
        tmp["question"] = tmp["question"].str.slice(0, 120)
        lines.append(df_to_markdown_fallback(tmp.round(4)))
    else:
        lines.append("- 无 mixed case")
    lines.append("")

    lines.append("### 5.2 全实验都失败的 10 个 case")
    if len(all_wrong_df) > 0:
        tmp = all_wrong_df[["uid", "question"]].copy()
        tmp["question"] = tmp["question"].str.slice(0, 160)
        lines.append(df_to_markdown_fallback(tmp))
    else:
        lines.append("- 无 all_wrong case")
    lines.append("")

    lines.append(f"### 5.3 最优实验（{best_exp_label}）仍失败的 10 个 case")
    if len(best_model_fail) > 0:
        tmp = best_model_fail[["uid", "score_mean", "question"]].copy()
        tmp["question"] = tmp["question"].str.slice(0, 160)
        lines.append(df_to_markdown_fallback(tmp.round(4)))
    else:
        lines.append("- 最优实验在 final step 未出现失败 case")
    lines.append("")

    lines.append("## 6. LLM 打标方案（建议）")
    lines.append("建议把 `outputs/tables/llm_tagging_candidates.csv` 作为输入，按 `case_uid + 4个模型输出` 做错误标签标注。")
    lines.append("")
    lines.append("推荐标签 schema：")
    lines.append("- `math_reasoning_error`：推理链有明显数学错误")
    lines.append("- `final_answer_extraction_error`：推导过程可能正确，但最终答案提取/格式错误")
    lines.append("- `format_compliance_error`：未按 `\\boxed{}` 或题目要求格式输出")
    lines.append("- `instruction_following_error`：没有按题意约束（范围、单位、形式）")
    lines.append("- `uncertainty_or_hallucination`：出现自相矛盾/明显不确定猜测")
    lines.append("- `overlong_or_inefficient_reasoning`：推理过长且引入噪声导致错误")
    lines.append("")
    lines.append("建议 LLM 标注 prompt 模板：")
    lines.append("```text")
    lines.append("你将看到一个数学题 case，以及 4 个模型在 final step 的输出与打分（0/1）。")
    lines.append("请完成：")
    lines.append("1) 给每个失败输出（score=0）打 1~2 个主标签（从给定 schema 里选）")
    lines.append("2) 给出一句“最可能失败原因”")
    lines.append("3) 给出“最小修复建议”（例如：答案提取、格式约束、减少冗余推理、强化中间校验）")
    lines.append("输出 JSON，字段：uid, model, labels, root_cause, minimal_fix")
    lines.append("```")
    lines.append("")

    lines.append("## 7. 下一步建议")
    lines.append("- 先对 all_wrong + mixed case 做一轮 LLM 打标，确认主失败类型分布")
    lines.append("- 若 `final_answer_extraction_error/format_compliance_error` 占比高，优先加 answer-extractor 或格式约束")
    lines.append("- 若 `math_reasoning_error` 占比高，优先做高价值 case 复训与难例采样")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(RAW_DIR.glob("*_validation.zip"))
    if len(zip_paths) != 4:
        raise RuntimeError(f"Expected 4 zip files in {RAW_DIR}, found {len(zip_paths)}")

    path_by_exp = {normalize_experiment_name(p.name): p for p in zip_paths}
    missing = [e for e in EXPERIMENT_ORDER if e not in path_by_exp]
    if missing:
        raise RuntimeError(f"Missing expected experiments: {missing}")

    all_step_df = []
    final_cases_by_exp = {}

    for exp in EXPERIMENT_ORDER:
        sdf, fcases = collect_from_zip(path_by_exp[exp], exp)
        all_step_df.append(sdf)
        final_cases_by_exp[exp] = fcases

    step_df = pd.concat(all_step_df, ignore_index=True).sort_values(["experiment", "step"])
    summary_df = build_experiment_summary(step_df)
    case_df = compute_case_matrix(final_cases_by_exp)
    pairwise_df = build_pairwise_winrate(case_df)

    # Failure-tag candidates for LLM
    p90_len = np.nanpercentile(step_df["response_length"], 90)
    case_df["heuristic_failure_tags"] = case_df.apply(lambda r: detect_failure_tags(r, p90_len), axis=1)

    all_wrong = case_df[case_df["outcome_pattern"] == "all_wrong"].copy()
    mixed = case_df[case_df["outcome_pattern"] == "mixed"].copy().sort_values(["score_std", "score_mean"], ascending=[False, True])
    llm_candidates = pd.concat([all_wrong.head(120), mixed.head(120)], ignore_index=True).drop_duplicates(subset=["uid"])

    # Tag summary
    tag_counter = Counter()
    for tags in case_df["heuristic_failure_tags"].fillna(""):
        if not tags:
            continue
        for tag in tags.split(";"):
            if tag:
                tag_counter[tag] += 1
    tag_summary_df = pd.DataFrame(
        [{"tag": k, "count": v} for k, v in sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))]
    )

    # Persist tables
    step_df.to_csv(TABLES_DIR / "step_metrics.csv", index=False)
    summary_df.to_csv(TABLES_DIR / "experiment_summary.csv", index=False)
    output_cols = [f"{exp}_output" for exp in EXPERIMENT_ORDER]
    case_df.drop(columns=output_cols, errors="ignore").to_csv(TABLES_DIR / "final_case_matrix.csv", index=False)
    pairwise_df.to_csv(TABLES_DIR / "pairwise_winrate.csv")
    llm_candidates.to_csv(TABLES_DIR / "llm_tagging_candidates.csv", index=False)
    tag_summary_df.to_csv(TABLES_DIR / "failure_tag_summary.csv", index=False)

    # Plots and report
    make_plots(step_df, summary_df, pairwise_df)
    report_text = make_report(step_df, summary_df, case_df, pairwise_df)
    report_path = ROOT / "REPORT_exp4_validation_20260331_v1.md"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n=== Done ===")
    print(f"step_metrics: {TABLES_DIR / 'step_metrics.csv'}")
    print(f"summary:      {TABLES_DIR / 'experiment_summary.csv'}")
    print(f"case_matrix:  {TABLES_DIR / 'final_case_matrix.csv'}")
    print(f"llm_candidates:{TABLES_DIR / 'llm_tagging_candidates.csv'}")
    print(f"report:       {report_path}")


if __name__ == "__main__":
    main()
