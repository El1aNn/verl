#!/usr/bin/env python3
"""Render per-trajectory token entropy heatmaps across checkpoints.

The input is an `offline_full_token_ckpt_report.py` output directory containing:

- records/samples.step_XXXX.jsonl
- records/tokens.step_XXXX.jsonl or records/tokens.step_XXXX.jsonl.gz

Each trajectory is identified by the stable sample UID emitted by that script,
for example `val::math500::0::<hash>::sample2`. The script can either render
explicit UIDs or automatically pick trajectories with large entropy/correctness
changes across checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_RE = re.compile(r"samples\.step_(\d+)\.jsonl$")
TOKEN_RE = re.compile(r"tokens\.step_(\d+)\.jsonl(?:\.gz)?$")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: Any, digits: int = 3) -> str:
    value = _to_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _parse_step(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse step from {path}")
    return int(match.group(1))


def _safe_slug(value: str, max_len: int = 96) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_") or "trajectory"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len].strip("_") or "trajectory"


def _shorten(text: Any, limit: int = 220) -> str:
    text = str(text or "").replace("\r", "")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [_to_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_sample_files(report_dir: Path) -> list[tuple[int, Path]]:
    records_dir = report_dir / "records"
    files = []
    for path in records_dir.glob("samples.step_*.jsonl"):
        if SAMPLE_RE.match(path.name):
            files.append((_parse_step(path, SAMPLE_RE), path))
    return sorted(files)


def _find_token_files(report_dir: Path) -> list[tuple[int, Path]]:
    records_dir = report_dir / "records"
    files = []
    for path in list(records_dir.glob("tokens.step_*.jsonl")) + list(records_dir.glob("tokens.step_*.jsonl.gz")):
        if TOKEN_RE.match(path.name):
            files.append((_parse_step(path, TOKEN_RE), path))
    return sorted(files)


def load_samples(report_dir: Path, step_filter: Optional[set[int]] = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step, path in _find_sample_files(report_dir):
        if step_filter is not None and step not in step_filter:
            continue
        for record in _read_jsonl(path):
            uid = str(record.get("uid") or "")
            if not uid:
                continue
            rows.append(
                {
                    "step": int(record.get("step", step)),
                    "uid": uid,
                    "case_uid": str(record.get("case_uid") or uid.rsplit("::sample", 1)[0]),
                    "rollout_index": int(record.get("rollout_index", 0) or 0),
                    "sample_index": int(record.get("sample_index", 0) or 0),
                    "data_source": str(record.get("data_source") or ""),
                    "question": str(record.get("question") or ""),
                    "output": str(record.get("output") or ""),
                    "score": _to_float(record.get("score")),
                    "response_length": _to_float(record.get("response_length")),
                    "token_entropy_mean": _to_float(record.get("token_entropy_mean")),
                    "token_prob_mean": _to_float(record.get("token_prob_mean")),
                    "top1_prob_mean": _to_float(record.get("top1_prob_mean")),
                    "low_confidence_token_ratio": _to_float(record.get("low_confidence_token_ratio")),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No sample rows found under {report_dir / 'records'}")
    return pd.DataFrame(rows).sort_values(["uid", "step"])


def _uid_matches(query: str, uid: str) -> bool:
    query = str(query).strip()
    uid = str(uid)
    return uid == query or uid.endswith(query) or query in uid


def _explicit_uid_selection(samples: pd.DataFrame, queries: list[str], min_steps: int) -> list[str]:
    selected: list[str] = []
    all_uids = sorted(samples["uid"].unique())
    counts = samples.groupby("uid")["step"].nunique().to_dict()
    for query in queries:
        matches = [uid for uid in all_uids if _uid_matches(query, uid)]
        for uid in matches:
            if counts.get(uid, 0) >= min_steps and uid not in selected:
                selected.append(uid)
        if not matches:
            print(f"[warn] no UID matched query: {query}")
    return selected


def summarize_trajectories(samples: pd.DataFrame, min_steps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for uid, group in samples.groupby("uid"):
        group = group.sort_values("step")
        if group["step"].nunique() < min_steps:
            continue
        first = group.iloc[0]
        final = group.iloc[-1]
        entropies = pd.to_numeric(group["token_entropy_mean"], errors="coerce")
        scores = pd.to_numeric(group["score"], errors="coerce")
        lengths = pd.to_numeric(group["response_length"], errors="coerce")
        entropy_first = _to_float(first.get("token_entropy_mean"))
        entropy_final = _to_float(final.get("token_entropy_mean"))
        score_first = _to_float(first.get("score"))
        score_final = _to_float(final.get("score"))
        length_first = _to_float(first.get("response_length"))
        length_final = _to_float(final.get("response_length"))
        entropy_range = _to_float(entropies.max() - entropies.min()) or 0.0
        score_range = _to_float(scores.max() - scores.min()) or 0.0
        length_range = _to_float(lengths.max() - lengths.min()) or 0.0
        mixed_correctness = bool(scores.min() < 0.5 and scores.max() >= 0.5)
        selection_score = entropy_range + 0.35 * score_range + 0.10 * min(length_range / 1000.0, 1.0)
        if mixed_correctness:
            selection_score += 0.50
        rows.append(
            {
                "uid": uid,
                "case_uid": str(first.get("case_uid")),
                "rollout_index": int(first.get("rollout_index", 0) or 0),
                "data_source": str(first.get("data_source") or ""),
                "question": str(first.get("question") or ""),
                "n_steps": int(group["step"].nunique()),
                "steps": ",".join(str(int(step)) for step in group["step"].tolist()),
                "score_first": score_first,
                "score_final": score_final,
                "score_range": score_range,
                "entropy_first": entropy_first,
                "entropy_final": entropy_final,
                "entropy_range": entropy_range,
                "entropy_delta_final_minus_first": None
                if entropy_first is None or entropy_final is None
                else entropy_final - entropy_first,
                "response_length_first": length_first,
                "response_length_final": length_final,
                "response_length_range": length_range,
                "mixed_correctness": int(mixed_correctness),
                "selection_score": selection_score,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["selection_score", "entropy_range"], ascending=False)


def select_trajectories(
    samples: pd.DataFrame,
    top_n: int,
    min_steps: int,
    uid_queries: list[str],
    distinct_cases: bool,
) -> pd.DataFrame:
    summary = summarize_trajectories(samples, min_steps=min_steps)
    if summary.empty:
        return summary

    if uid_queries:
        selected_uids = _explicit_uid_selection(samples, queries=uid_queries, min_steps=min_steps)
        return summary[summary["uid"].isin(selected_uids)].copy()

    pools = [
        summary[summary["mixed_correctness"] == 1].sort_values(["entropy_range", "selection_score"], ascending=False),
        summary.sort_values("entropy_range", ascending=False),
        summary.sort_values("entropy_delta_final_minus_first", ascending=False),
        summary.sort_values("entropy_delta_final_minus_first", ascending=True),
        summary.sort_values("entropy_final", ascending=False),
        summary.sort_values("selection_score", ascending=False),
    ]

    selected_rows: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    selected_cases: set[str] = set()

    def add_row(row: pd.Series) -> bool:
        uid = str(row["uid"])
        case_uid = str(row["case_uid"])
        if uid in selected_uids:
            return False
        if distinct_cases and case_uid in selected_cases:
            return False
        selected_rows.append(row.to_dict())
        selected_uids.add(uid)
        selected_cases.add(case_uid)
        return len(selected_rows) >= top_n

    for pool in pools:
        for _, row in pool.iterrows():
            if add_row(row):
                return pd.DataFrame(selected_rows)

    if distinct_cases and len(selected_rows) < top_n:
        for _, row in summary.iterrows():
            if str(row["uid"]) not in selected_uids:
                selected_rows.append(row.to_dict())
                selected_uids.add(str(row["uid"]))
                if len(selected_rows) >= top_n:
                    break

    return pd.DataFrame(selected_rows)


def load_tokens_for_uids(
    report_dir: Path,
    selected_uids: set[str],
    metric: str,
    step_filter: Optional[set[int]] = None,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    token_files = _find_token_files(report_dir)
    if not token_files:
        raise FileNotFoundError(f"No token files found under {report_dir / 'records'}")

    tokens_by_uid_step: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    keep_fields = {
        "uid",
        "step",
        "position",
        "token",
        "text",
        "entropy",
        "prob",
        "logprob",
        "top1_prob",
        "eos_prob",
        "topk_mass",
        metric,
    }
    for step, path in token_files:
        if step_filter is not None and step not in step_filter:
            continue
        print(f"[read] tokens step={step} path={path}")
        for record in _read_jsonl(path):
            uid = str(record.get("uid") or "")
            if uid not in selected_uids:
                continue
            slim = {key: record.get(key) for key in keep_fields if key in record}
            slim["step"] = int(record.get("step", step))
            slim["position"] = int(record.get("position", len(tokens_by_uid_step[uid][step])))
            tokens_by_uid_step[uid][step].append(slim)

    for uid_steps in tokens_by_uid_step.values():
        for rows in uid_steps.values():
            rows.sort(key=lambda item: int(item.get("position", 0)))
    return {uid: dict(step_map) for uid, step_map in tokens_by_uid_step.items()}


def make_raw_matrix(
    step_rows: dict[int, list[dict[str, Any]]],
    steps: list[int],
    metric: str,
    max_tokens: int,
) -> np.ndarray:
    observed_max = max((len(step_rows.get(step, [])) for step in steps), default=0)
    width = min(max_tokens, max(1, observed_max))
    matrix = np.full((len(steps), width), np.nan, dtype=np.float64)
    for row_idx, step in enumerate(steps):
        values = [_to_float(token.get(metric)) for token in step_rows.get(step, [])[:width]]
        for col_idx, value in enumerate(values):
            if value is not None:
                matrix[row_idx, col_idx] = value
    return matrix


def make_normalized_matrix(
    step_rows: dict[int, list[dict[str, Any]]],
    steps: list[int],
    metric: str,
    columns: int,
) -> np.ndarray:
    columns = max(1, int(columns))
    matrix = np.full((len(steps), columns), np.nan, dtype=np.float64)
    for row_idx, step in enumerate(steps):
        rows = step_rows.get(step, [])
        if not rows:
            continue
        bins: list[list[float]] = [[] for _ in range(columns)]
        denom = max(len(rows), 1)
        for pos, token in enumerate(rows):
            value = _to_float(token.get(metric))
            if value is None:
                continue
            col_idx = min(int(pos * columns / denom), columns - 1)
            bins[col_idx].append(value)
        for col_idx, values in enumerate(bins):
            mean_value = _mean(values)
            if mean_value is not None:
                matrix[row_idx, col_idx] = mean_value
    return matrix


def _metric_limits(tokens_by_uid_step: dict[str, dict[int, list[dict[str, Any]]]], metric: str) -> tuple[float, float]:
    values: list[float] = []
    for step_map in tokens_by_uid_step.values():
        for rows in step_map.values():
            for token in rows:
                value = _to_float(token.get(metric))
                if value is not None:
                    values.append(value)
    if not values:
        return 0.0, 1.0
    low, high = np.nanpercentile(np.asarray(values, dtype=np.float64), [1, 99])
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def render_heatmap(
    matrix: np.ndarray,
    steps: list[int],
    samples_for_uid: pd.DataFrame,
    title: str,
    xlabel: str,
    metric: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    dpi: int,
) -> None:
    wrapped_title = "\n".join(textwrap.wrap(title, width=112))
    title_lines = max(1, wrapped_title.count("\n") + 1)
    height = max(2.6, 0.55 * len(steps) + 1.7)
    width = 12.5
    fig, ax = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    row_labels = []
    for step in steps:
        row = samples_for_uid[samples_for_uid["step"] == step]
        if row.empty:
            row_labels.append(f"step {step}")
            continue
        item = row.iloc[0]
        row_labels.append(
            f"step {step}  score={_fmt(item.get('score'))}  ent={_fmt(item.get('token_entropy_mean'))}"
        )
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(wrapped_title, fontsize=11, pad=12)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, max(0.90, 1.0 - 0.025 * (title_lines - 1))))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def collect_top_tokens(
    uid: str,
    step_rows: dict[int, list[dict[str, Any]]],
    metric: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, tokens in sorted(step_rows.items()):
        ranked = []
        for token in tokens:
            value = _to_float(token.get(metric))
            if value is None:
                continue
            ranked.append((value, token))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for rank, (value, token) in enumerate(ranked[:top_k], start=1):
            rows.append(
                {
                    "uid": uid,
                    "step": int(step),
                    "rank": rank,
                    "position": int(token.get("position", 0)),
                    metric: value,
                    "prob": _to_float(token.get("prob")),
                    "text": str(token.get("text") or token.get("token") or ""),
                    "token": str(token.get("token") or ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["uid", "case_uid", "step", "rank", "position", "score", "token_entropy_mean"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_index_html(
    output_dir: Path,
    report_dir: Path,
    selected: pd.DataFrame,
    samples: pd.DataFrame,
    figure_rows: list[dict[str, Any]],
    top_token_rows: list[dict[str, Any]],
    mode: str,
    metric: str,
) -> None:
    top_tokens_by_uid_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in top_token_rows:
        top_tokens_by_uid_step[(str(row["uid"]), int(row["step"]))].append(row)

    cards = []
    for _, selected_row in selected.iterrows():
        uid = str(selected_row["uid"])
        sample_rows = samples[samples["uid"] == uid].sort_values("step")
        fig_links = [
            row
            for row in figure_rows
            if row["uid"] == uid and (mode == "both" or row["mode"] == mode)
        ]
        step_cells = []
        for _, sample in sample_rows.iterrows():
            step = int(sample["step"])
            hot_tokens = top_tokens_by_uid_step.get((uid, step), [])
            hot_text = ", ".join(
                f"{_shorten(token.get('text'), 18)}@{token.get('position')}={_fmt(token.get(metric))}"
                for token in hot_tokens[:6]
            )
            step_cells.append(
                f"""
                <tr>
                  <td>{step}</td>
                  <td>{_fmt(sample.get('score'))}</td>
                  <td>{_fmt(sample.get('token_entropy_mean'))}</td>
                  <td>{_fmt(sample.get('response_length'), 1)}</td>
                  <td>{html.escape(hot_text)}</td>
                  <td>{html.escape(_shorten(sample.get('output'), 280))}</td>
                </tr>
                """
            )
        figure_html = "".join(
            f"""<figure><img src="{html.escape(row['path'])}" alt="{html.escape(row['mode'])} heatmap"><figcaption>{html.escape(row['mode'])}</figcaption></figure>"""
            for row in fig_links
        )
        cards.append(
            f"""
            <section class="card" id="{html.escape(_safe_slug(uid))}">
              <h2>{html.escape(uid)}</h2>
              <p class="meta">case: {html.escape(str(selected_row.get('case_uid')))} | entropy range: {_fmt(selected_row.get('entropy_range'))} | score range: {_fmt(selected_row.get('score_range'))}</p>
              <p>{html.escape(_shorten(selected_row.get('question'), 520))}</p>
              <div class="figures">{figure_html}</div>
              <table>
                <thead><tr><th>step</th><th>score</th><th>mean entropy</th><th>len</th><th>hottest tokens</th><th>output preview</th></tr></thead>
                <tbody>{''.join(step_cells)}</tbody>
              </table>
            </section>
            """
        )

    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trajectory Entropy Heatmaps</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5b6672;
      --line: #d8dde4;
      --panel: #ffffff;
      --bg: #f6f7f9;
      --accent: #a83f39;
    }}
    body {{
      margin: 0;
      padding: 28px;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 18px 0;
    }}
    .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      margin: 14px 0;
    }}
    figure {{ margin: 0; }}
    img {{
      width: 100%;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 650; }}
    code {{
      background: #eef0f3;
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>Trajectory Entropy Heatmaps</h1>
  <p class="meta">source: <code>{html.escape(str(report_dir))}</code> | metric: <code>{html.escape(metric)}</code></p>
  {''.join(cards)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def parse_steps(value: str) -> Optional[set[int]]:
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, help="Directory created by offline_full_token_ckpt_report.py.")
    parser.add_argument("--output-dir", default=None, help="Default: <report-dir>/trajectory_entropy_heatmaps")
    parser.add_argument("--steps", default="all", help="Comma-separated checkpoint steps to include, or 'all'.")
    parser.add_argument("--top-n", type=int, default=6, help="Number of trajectories to auto-select.")
    parser.add_argument("--uid", action="append", default=[], help="Explicit UID/suffix/substring to render. Repeatable.")
    parser.add_argument("--min-steps", type=int, default=2, help="Minimum checkpoints required for a trajectory.")
    parser.add_argument("--allow-same-case", action="store_true", help="Auto-selection may pick multiple rollouts per prompt.")
    parser.add_argument(
        "--metric",
        default="entropy",
        choices=["entropy", "prob", "logprob", "top1_prob", "eos_prob", "topk_mass"],
        help="Token metric to plot.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["normalized", "raw", "both"],
        help="Normalized bins compare relative response position; raw uses first N token positions.",
    )
    parser.add_argument("--columns", type=int, default=180, help="Columns for normalized-position heatmaps.")
    parser.add_argument("--max-raw-tokens", type=int, default=512, help="Max token positions for raw heatmaps.")
    parser.add_argument("--top-token-count", type=int, default=10, help="Top high-metric tokens to write per step.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report_dir = Path(args.report_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else report_dir / "trajectory_entropy_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    step_filter = parse_steps(args.steps)

    samples = load_samples(report_dir, step_filter=step_filter)
    selected = select_trajectories(
        samples=samples,
        top_n=int(args.top_n),
        min_steps=int(args.min_steps),
        uid_queries=list(args.uid or []),
        distinct_cases=not bool(args.allow_same_case),
    )
    if selected.empty:
        raise RuntimeError("No trajectories selected. Try lowering --min-steps or passing --uid.")

    selected_uids = set(str(uid) for uid in selected["uid"].tolist())
    tokens_by_uid_step = load_tokens_for_uids(
        report_dir=report_dir,
        selected_uids=selected_uids,
        metric=str(args.metric),
        step_filter=step_filter,
    )
    vmin, vmax = _metric_limits(tokens_by_uid_step, str(args.metric))

    selected.to_csv(output_dir / "selected_trajectories.csv", index=False)
    samples[samples["uid"].isin(selected_uids)].sort_values(["uid", "step"]).to_csv(
        output_dir / "selected_step_metrics.csv", index=False
    )

    figure_rows: list[dict[str, Any]] = []
    top_token_rows: list[dict[str, Any]] = []
    figures_dir = output_dir / "figures"
    for _, selected_row in selected.iterrows():
        uid = str(selected_row["uid"])
        step_rows = tokens_by_uid_step.get(uid, {})
        if not step_rows:
            print(f"[warn] no token rows loaded for {uid}")
            continue
        steps = sorted(step_rows.keys())
        uid_samples = samples[samples["uid"] == uid].sort_values("step")
        slug = _safe_slug(uid)
        question = _shorten(selected_row.get("question"), 150)
        title = f"{uid} | {question}"

        if args.mode in ("normalized", "both"):
            matrix = make_normalized_matrix(step_rows, steps=steps, metric=str(args.metric), columns=int(args.columns))
            rel_path = f"figures/{slug}.normalized.{args.metric}.png"
            render_heatmap(
                matrix=matrix,
                steps=steps,
                samples_for_uid=uid_samples,
                title=title,
                xlabel="Normalized response position",
                metric=str(args.metric),
                output_path=output_dir / rel_path,
                vmin=vmin,
                vmax=vmax,
                dpi=int(args.dpi),
            )
            figure_rows.append({"uid": uid, "mode": "normalized", "path": rel_path})

        if args.mode in ("raw", "both"):
            matrix = make_raw_matrix(
                step_rows,
                steps=steps,
                metric=str(args.metric),
                max_tokens=int(args.max_raw_tokens),
            )
            rel_path = f"figures/{slug}.raw.{args.metric}.png"
            render_heatmap(
                matrix=matrix,
                steps=steps,
                samples_for_uid=uid_samples,
                title=title,
                xlabel=f"Raw response token position, first {matrix.shape[1]} tokens",
                metric=str(args.metric),
                output_path=output_dir / rel_path,
                vmin=vmin,
                vmax=vmax,
                dpi=int(args.dpi),
            )
            figure_rows.append({"uid": uid, "mode": "raw", "path": rel_path})

        top_token_rows.extend(
            collect_top_tokens(
                uid=uid,
                step_rows=step_rows,
                metric=str(args.metric),
                top_k=int(args.top_token_count),
            )
        )

    write_csv(output_dir / "top_metric_tokens.csv", top_token_rows)
    write_csv(output_dir / "figures.csv", figure_rows)
    render_index_html(
        output_dir=output_dir,
        report_dir=report_dir,
        selected=selected,
        samples=samples,
        figure_rows=figure_rows,
        top_token_rows=top_token_rows,
        mode=str(args.mode),
        metric=str(args.metric),
    )

    print(f"[done] selected {len(selected)} trajectories")
    print(f"[done] output: {output_dir}")
    print(f"[done] open: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
