#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate paper-style validation visualizations from local experiment artifacts.

This script is designed for experiments that dump validation JSONL files through
`trainer.validation_data_dir`. When a local `file` logger is also enabled, the
script can additionally recover training-time metrics such as `actor/entropy`.

The report is intentionally lightweight:
- no third-party plotting dependency
- outputs self-contained SVG charts plus an HTML summary page
- supports multiple runs for side-by-side comparison
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TOKEN_GROUPS = {
    "reasoning_sparks": ["but", "wait", "perhaps", "alternatively", "however"],
    "noise": ["cost", "fine", "balanced", "ere", "trans"],
}

RUN_COLORS = [
    "#125B50",
    "#8E3200",
    "#0F3460",
    "#6B2737",
    "#607EAA",
    "#7A5C3E",
]

GROUP_COLORS = {
    "reasoning_sparks": "#125B50",
    "noise": "#8E3200",
}

CHART_BG = "#FBF7EF"
GRID_COLOR = "#D8D0C4"
TEXT_COLOR = "#1F2933"
MUTED_TEXT = "#5B6770"
PANEL_BORDER = "#D0C4B5"


@dataclass
class RunSpec:
    label: str
    validation_dir: Path
    metrics_file: Optional[Path] = None


@dataclass
class StepSummary:
    step: int
    sample_count: int
    primary_metric_key: str
    primary_metric_mean: Optional[float]
    score_mean: Optional[float]
    reward_mean: Optional[float]
    acc_mean: Optional[float]
    token_entropy_mean: Optional[float]
    response_length_mean: Optional[float]
    low_confidence_ratio_mean: Optional[float]
    actor_entropy: Optional[float] = None
    dumped_sample_count: int = 0
    dumped_sample_ratio: float = 0.0
    dumped_token_count: int = 0
    total_token_count: int = 0
    dumped_token_ratio: float = 0.0
    sampled_group_count: Dict[str, int] = field(default_factory=dict)
    sampled_group_frequency: Dict[str, float] = field(default_factory=dict)
    topk_low_prob_group_mean: Dict[str, Optional[float]] = field(default_factory=dict)
    token_frequency_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class RunData:
    spec: RunSpec
    primary_metric_key: str
    step_summaries: Dict[int, StepSummary]
    sampled_probabilities: Dict[str, Dict[int, List[float]]]
    scatter_points: Dict[str, List[Tuple[float, float]]]
    topk_low_prob_values: Dict[str, Dict[int, List[float]]]
    token_counts: Dict[str, Dict[int, Counter]]
    last_step_samples: List[Dict[str, Any]]
    trace_samples_by_uid: Dict[str, Dict[int, Dict[str, Any]]]
    last_step_trace_samples: List[Dict[str, Any]]
    warnings: List[str]

    @property
    def steps(self) -> List[int]:
        return sorted(self.step_summaries.keys())


def normalize_token_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", "")
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    text = text.strip().lower()
    return text


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [_to_float(item) for item in values]
    numeric = [item for item in numeric if item is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = min(max(fraction, 0.0), 1.0) * (len(sorted_values) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _series_bounds(series: Dict[str, Sequence[Tuple[float, float]]], pad_fraction: float = 0.08) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for points in series.values():
        for x_value, y_value in points:
            xs.append(float(x_value))
            ys.append(float(y_value))
    if not xs:
        return 0.0, 1.0, 0.0, 1.0

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        spread = abs(min_y) * 0.1 or 1.0
        min_y -= spread
        max_y += spread
    else:
        spread = (max_y - min_y) * pad_fraction
        min_y -= spread
        max_y += spread
    return min_x, max_x, min_y, max_y


def _select_nearest_step(available_steps: Sequence[int], target_step: int) -> Optional[int]:
    if not available_steps:
        return None
    return min(available_steps, key=lambda item: (abs(item - target_step), item))


def _pick_reference_steps(run_data_list: Sequence[RunData], count: int) -> List[int]:
    observed_steps = sorted({step for run_data in run_data_list for step in run_data.steps})
    if not observed_steps:
        return []
    if len(observed_steps) <= count:
        return observed_steps

    fractions = [0.0]
    if count > 2:
        for idx in range(1, count - 1):
            fractions.append(idx / float(count - 1))
    fractions.append(1.0)

    selected: List[int] = []
    for fraction in fractions:
        step = observed_steps[int(round((len(observed_steps) - 1) * fraction))]
        if step not in selected:
            selected.append(step)
    return selected


def _parse_run_spec(value: str) -> RunSpec:
    if "=" not in value:
        raise ValueError("run spec must look like label=/path/to/validation[::/path/to/metrics.jsonl]")
    label, payload = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label cannot be empty")
    if "::" in payload:
        validation_dir_str, metrics_file_str = payload.split("::", 1)
        metrics_file = Path(metrics_file_str).expanduser()
    else:
        validation_dir_str = payload
        metrics_file = None
    validation_dir = Path(validation_dir_str).expanduser()
    return RunSpec(label=label, validation_dir=validation_dir, metrics_file=metrics_file)


def _parse_token_group_spec(value: str) -> Tuple[str, List[str]]:
    if "=" not in value:
        raise ValueError("token group spec must look like name=token1,token2,token3")
    name, tokens_str = value.split("=", 1)
    name = name.strip()
    tokens = [item.strip() for item in tokens_str.split(",") if item.strip()]
    if not name or not tokens:
        raise ValueError("token group name and tokens must both be non-empty")
    return name, tokens


def _make_group_lookup(token_groups: Dict[str, Sequence[str]]) -> Dict[str, set]:
    return {name: {normalize_token_text(token) for token in tokens} for name, tokens in token_groups.items()}


def _load_metrics_file(metrics_file: Optional[Path]) -> Dict[int, Dict[str, float]]:
    if metrics_file is None or not metrics_file.exists():
        return {}

    metrics_by_step: Dict[int, Dict[str, float]] = {}
    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record.get("step", 0))
            payload = record.get("data", {})
            numeric_payload = {}
            for key, value in payload.items():
                numeric_value = _to_float(value)
                if numeric_value is not None:
                    numeric_payload[key] = numeric_value
            metrics_by_step[step] = numeric_payload
    return metrics_by_step


def _make_sample_preview(entry: Dict[str, Any], matched_tokens: Sequence[str], primary_metric_key: str) -> Dict[str, Any]:
    return {
        "uid": entry.get("uid"),
        "primary_metric_key": primary_metric_key,
        "primary_metric_value": _to_float(entry.get(primary_metric_key)),
        "input": entry.get("input", ""),
        "output": entry.get("output", ""),
        "response_length": entry.get("response_length"),
        "matched_tokens": list(matched_tokens),
        "acc": _to_float(entry.get("acc")),
        "reward": _to_float(entry.get("reward")),
        "score": _to_float(entry.get("score")),
    }


def _make_trace_sample(
    entry: Dict[str, Any],
    matched_tokens: Sequence[str],
    primary_metric_key: str,
    step: int,
    fallback_uid: str,
) -> Dict[str, Any]:
    trace_sample = _make_sample_preview(entry=entry, matched_tokens=matched_tokens, primary_metric_key=primary_metric_key)
    trace_sample.update(
        {
            "uid": str(trace_sample.get("uid") or fallback_uid),
            "step": int(step),
            "token_diagnostics": list(entry.get("token_diagnostics", []) or []),
            "token_diagnostics_total_tokens": int(
                entry.get("token_diagnostics_total_tokens")
                or entry.get("response_length")
                or len(entry.get("token_diagnostics", []) or [])
                or 0
            ),
            "token_diagnostics_truncated": bool(entry.get("token_diagnostics_truncated", False)),
            "token_entropy_mean": _to_float(entry.get("token_entropy_mean")),
            "token_logprob_mean": _to_float(entry.get("token_logprob_mean")),
            "answer_tail_confidence": _to_float(entry.get("answer_tail_confidence")),
            "eos_prob_final": _to_float(entry.get("eos_prob_final")),
        }
    )
    return trace_sample


def _render_text_preview(text: str, limit: int = 260) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _sanitize_slug(value: str) -> str:
    slug_chars = []
    for char in str(value):
        if char.isalnum():
            slug_chars.append(char.lower())
        else:
            slug_chars.append("-")
    slug = "".join(slug_chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "sample"


def _uid_matches(query: str, uid: str) -> bool:
    query = str(query or "").strip()
    uid = str(uid or "")
    if not query:
        return False
    return uid == query or uid.endswith(query) or uid.endswith(f"::{query}")


def _trace_sort_key(sample: Dict[str, Any]) -> Tuple[int, int, float, int, str]:
    metric_value = sample.get("primary_metric_value")
    metric_sort = float(metric_value) if metric_value is not None else float("-inf")
    return (
        0 if sample.get("matched_tokens") else 1,
        0 if metric_value is not None else 1,
        -metric_sort,
        -len(sample.get("matched_tokens", [])),
        str(sample.get("uid") or ""),
    )


def load_run_data(
    run_spec: RunSpec,
    token_groups: Dict[str, Sequence[str]],
    low_prob_min: float,
    low_prob_max: float,
) -> RunData:
    validation_files = sorted(
        run_spec.validation_dir.glob("*.jsonl"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if not validation_files:
        raise FileNotFoundError(f"no validation jsonl files found under {run_spec.validation_dir}")

    group_lookup = _make_group_lookup(token_groups)
    metrics_by_step = _load_metrics_file(run_spec.metrics_file)

    primary_metric_key = "score"
    step_summaries: Dict[int, StepSummary] = {}
    sampled_probabilities = {name: defaultdict(list) for name in token_groups}
    scatter_points = {name: [] for name in token_groups}
    topk_low_prob_values = {name: defaultdict(list) for name in token_groups}
    token_counts = {name: defaultdict(Counter) for name in token_groups}
    warnings: List[str] = []
    last_step_samples: List[Dict[str, Any]] = []
    trace_samples_by_uid: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    last_step_trace_samples: List[Dict[str, Any]] = []

    for file_path in validation_files:
        with file_path.open("r", encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
        if not entries:
            continue

        step = int(entries[0].get("step", file_path.stem))
        if any("acc" in entry for entry in entries):
            primary_metric_key = "acc"
        elif primary_metric_key != "acc" and any("reward" in entry for entry in entries):
            primary_metric_key = "reward"

        dumped_sample_count = 0
        total_token_count = 0
        dumped_token_count = 0
        sampled_group_count = {name: 0 for name in token_groups}
        token_counter_by_group = {name: Counter() for name in token_groups}
        matched_samples: List[Dict[str, Any]] = []
        trace_entries_for_step: List[Dict[str, Any]] = []

        for entry_idx, entry in enumerate(entries):
            token_details = entry.get("token_diagnostics", []) or []
            estimated_total_tokens = int(
                entry.get("token_diagnostics_total_tokens")
                or entry.get("response_length")
                or len(token_details)
                or 0
            )
            total_token_count += estimated_total_tokens
            if token_details:
                dumped_sample_count += 1
                dumped_token_count += len(token_details)

            matched_tokens_for_sample: List[str] = []

            for token_detail in token_details:
                token_text = normalize_token_text(token_detail.get("text") or token_detail.get("token"))
                sampled_probability = _to_float(token_detail.get("prob"))
                token_entropy = _to_float(token_detail.get("entropy"))
                if token_text and sampled_probability is not None:
                    for group_name, token_set in group_lookup.items():
                        if token_text in token_set:
                            sampled_probabilities[group_name][step].append(sampled_probability)
                            token_counter_by_group[group_name][token_text] += 1
                            sampled_group_count[group_name] += 1
                            matched_tokens_for_sample.append(token_text)
                            if token_entropy is not None:
                                scatter_points[group_name].append((sampled_probability, token_entropy))

                for topk_entry in token_detail.get("topk", []) or []:
                    topk_probability = _to_float(topk_entry.get("prob"))
                    if topk_probability is None or topk_probability <= low_prob_min or topk_probability >= low_prob_max:
                        continue
                    candidate_text = normalize_token_text(topk_entry.get("text") or topk_entry.get("token"))
                    if not candidate_text:
                        continue
                    for group_name, token_set in group_lookup.items():
                        if candidate_text in token_set:
                            topk_low_prob_values[group_name][step].append(topk_probability)

            if matched_tokens_for_sample:
                matched_samples.append(
                    _make_sample_preview(
                        entry=entry,
                        matched_tokens=sorted(set(matched_tokens_for_sample)),
                        primary_metric_key=primary_metric_key,
                    )
                )

            if token_details:
                fallback_uid = f"{run_spec.label}-step{step}-sample{entry_idx}"
                trace_sample = _make_trace_sample(
                    entry=entry,
                    matched_tokens=sorted(set(matched_tokens_for_sample)),
                    primary_metric_key=primary_metric_key,
                    step=step,
                    fallback_uid=fallback_uid,
                )
                trace_samples_by_uid[trace_sample["uid"]][step] = trace_sample
                trace_entries_for_step.append(trace_sample)

        sample_count = len(entries)
        dumped_sample_ratio = float(dumped_sample_count / sample_count) if sample_count else 0.0
        dumped_token_ratio = float(dumped_token_count / total_token_count) if total_token_count else 0.0

        if dumped_sample_ratio < 0.99:
            warnings.append(
                f"{run_spec.label} step {step}: token diagnostics cover only {dumped_sample_count}/{sample_count} samples"
            )

        token_frequency_by_group = {}
        sampled_group_frequency = {}
        for group_name in token_groups:
            group_counter = token_counter_by_group[group_name]
            token_counts[group_name][step] = group_counter
            sampled_group_frequency[group_name] = (
                float(sampled_group_count[group_name] / dumped_token_count) if dumped_token_count else 0.0
            )
            token_frequency_by_group[group_name] = {
                token_name: float(count / dumped_token_count) if dumped_token_count else 0.0
                for token_name, count in group_counter.items()
            }

        step_metrics = metrics_by_step.get(step, {})
        if "actor/entropy" not in step_metrics and metrics_by_step:
            nearest_metrics_step = _select_nearest_step(sorted(metrics_by_step.keys()), step)
            if nearest_metrics_step is not None:
                step_metrics = metrics_by_step[nearest_metrics_step]

        step_summaries[step] = StepSummary(
            step=step,
            sample_count=sample_count,
            primary_metric_key=primary_metric_key,
            primary_metric_mean=_mean(entry.get(primary_metric_key) for entry in entries),
            score_mean=_mean(entry.get("score") for entry in entries),
            reward_mean=_mean(entry.get("reward") for entry in entries),
            acc_mean=_mean(entry.get("acc") for entry in entries),
            token_entropy_mean=_mean(entry.get("token_entropy_mean") for entry in entries),
            response_length_mean=_mean(entry.get("response_length") for entry in entries),
            low_confidence_ratio_mean=_mean(entry.get("low_confidence_token_ratio") for entry in entries),
            actor_entropy=_to_float(step_metrics.get("actor/entropy")),
            dumped_sample_count=dumped_sample_count,
            dumped_sample_ratio=dumped_sample_ratio,
            dumped_token_count=dumped_token_count,
            total_token_count=total_token_count,
            dumped_token_ratio=dumped_token_ratio,
            sampled_group_count=sampled_group_count,
            sampled_group_frequency=sampled_group_frequency,
            topk_low_prob_group_mean={
                group_name: _mean(topk_low_prob_values[group_name].get(step, [])) for group_name in token_groups
            },
            token_frequency_by_group=token_frequency_by_group,
        )
        last_step_samples = matched_samples or [
            _make_sample_preview(entry=entry, matched_tokens=[], primary_metric_key=primary_metric_key) for entry in entries
        ]
        last_step_trace_samples = trace_entries_for_step

    return RunData(
        spec=run_spec,
        primary_metric_key=primary_metric_key,
        step_summaries=step_summaries,
        sampled_probabilities=sampled_probabilities,
        scatter_points=scatter_points,
        topk_low_prob_values=topk_low_prob_values,
        token_counts=token_counts,
        last_step_samples=last_step_samples,
        trace_samples_by_uid={uid: dict(step_map) for uid, step_map in trace_samples_by_uid.items()},
        last_step_trace_samples=last_step_trace_samples,
        warnings=sorted(set(warnings)),
    )


def _format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _line_chart_svg(
    title: str,
    y_label: str,
    series: Dict[str, Sequence[Tuple[float, float]]],
    width: int = 880,
    height: int = 340,
) -> str:
    margin_left = 72
    margin_right = 24
    margin_top = 44
    margin_bottom = 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_x, max_x, min_y, max_y = _series_bounds(series)

    def sx(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def sy(value: float) -> float:
        return margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{CHART_BG}" rx="18"/>',
        f'<text x="{margin_left}" y="26" font-size="18" font-weight="700" fill="{TEXT_COLOR}">{_escape(title)}</text>',
        f'<text x="18" y="{margin_top + plot_height / 2:.1f}" font-size="12" fill="{MUTED_TEXT}" '
        'transform="rotate(-90 18 '
        f'{margin_top + plot_height / 2:.1f})">{_escape(y_label)}</text>',
    ]

    for tick_idx in range(6):
        y_value = min_y + (max_y - min_y) * tick_idx / 5.0
        y_pos = sy(y_value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{width - margin_right}" y2="{y_pos:.2f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" font-size="11" text-anchor="end" '
            f'fill="{MUTED_TEXT}">{_escape(f"{y_value:.3f}")}</text>'
        )

    tick_steps = 6
    for tick_idx in range(tick_steps):
        x_value = min_x + (max_x - min_x) * tick_idx / max(tick_steps - 1, 1)
        x_pos = sx(x_value)
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{margin_top + plot_height}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{height - 16}" font-size="11" text-anchor="middle" '
            f'fill="{MUTED_TEXT}">{_escape(str(int(round(x_value))))}</text>'
        )

    parts.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
        f'fill="none" stroke="{PANEL_BORDER}" stroke-width="1.2" rx="14"/>'
    )

    legend_x = width - margin_right - 180
    legend_y = 24
    for idx, (label, points) in enumerate(series.items()):
        if not points:
            continue
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        polyline = " ".join(f"{sx(x_value):.2f},{sy(y_value):.2f}" for x_value, y_value in points)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" '
            f'stroke-linejoin="round" points="{polyline}"/>'
        )
        for x_value, y_value in points:
            parts.append(
                f'<circle cx="{sx(x_value):.2f}" cy="{sy(y_value):.2f}" r="3.6" fill="{color}" stroke="{CHART_BG}" stroke-width="1.2"/>'
            )
        legend_row_y = legend_y + idx * 18
        parts.append(f'<line x1="{legend_x}" y1="{legend_row_y}" x2="{legend_x + 16}" y2="{legend_row_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 22}" y="{legend_row_y + 4}" font-size="12" fill="{TEXT_COLOR}">{_escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _violin_polygon(values: Sequence[float], center_x: float, width: float, top: float, bottom: float, bins: int = 18) -> Optional[str]:
    if not values:
        return None
    counts = [0] * bins
    for value in values:
        clipped = min(max(float(value), 0.0), 0.999999)
        counts[min(bins - 1, int(clipped * bins))] += 1
    peak = max(counts)
    if peak <= 0:
        return None
    left_points = []
    right_points = []
    panel_height = bottom - top
    for idx, count in enumerate(counts):
        y_value = top + panel_height * (idx + 0.5) / bins
        half_width = (count / float(peak)) * (width / 2.0)
        left_points.append((center_x - half_width, y_value))
        right_points.append((center_x + half_width, y_value))
    polygon_points = left_points + list(reversed(right_points))
    return " ".join(f"{x_value:.2f},{y_value:.2f}" for x_value, y_value in polygon_points)


def _distribution_svg(
    token_groups: Dict[str, Sequence[str]],
    run_data_list: Sequence[RunData],
    selected_steps: Sequence[int],
    width: int = 1200,
    height_per_row: int = 280,
) -> str:
    if not run_data_list or not selected_steps:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="80"><text x="20" y="44">No distribution data available.</text></svg>'

    rows = len(token_groups)
    cols = len(selected_steps)
    chart_height = 54 + rows * height_per_row
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_height}" viewBox="0 0 {width} {chart_height}">',
        f'<rect x="0" y="0" width="{width}" height="{chart_height}" fill="{CHART_BG}" rx="20"/>',
        f'<text x="26" y="28" font-size="18" font-weight="700" fill="{TEXT_COLOR}">Sampled Probability Distributions</text>',
        f'<text x="26" y="46" font-size="12" fill="{MUTED_TEXT}">Paper-style violin summaries over selected validation steps.</text>',
    ]

    panel_width = (width - 44) / float(cols)
    run_count = max(len(run_data_list), 1)

    for row_idx, (group_name, _tokens) in enumerate(token_groups.items()):
        for col_idx, target_step in enumerate(selected_steps):
            panel_left = 22 + col_idx * panel_width
            panel_top = 58 + row_idx * height_per_row
            inner_left = panel_left + 58
            inner_right = panel_left + panel_width - 18
            inner_top = panel_top + 34
            inner_bottom = panel_top + height_per_row - 34
            plot_width = inner_right - inner_left
            plot_height = inner_bottom - inner_top

            parts.append(
                f'<rect x="{panel_left:.2f}" y="{panel_top:.2f}" width="{panel_width - 8:.2f}" height="{height_per_row - 10:.2f}" '
                f'fill="#FFFDFC" stroke="{PANEL_BORDER}" stroke-width="1.1" rx="16"/>'
            )

            title = f"{group_name} @ step~{target_step}"
            parts.append(
                f'<text x="{panel_left + 16:.2f}" y="{panel_top + 22:.2f}" font-size="13" font-weight="700" fill="{TEXT_COLOR}">{_escape(title)}</text>'
            )

            for tick_idx in range(5):
                y = inner_bottom - plot_height * tick_idx / 4.0
                prob_value = tick_idx / 4.0
                parts.append(
                    f'<line x1="{inner_left:.2f}" y1="{y:.2f}" x2="{inner_right:.2f}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{inner_left - 10:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="{MUTED_TEXT}">{prob_value:.2f}</text>'
                )

            for run_idx, run_data in enumerate(run_data_list):
                actual_step = _select_nearest_step(run_data.steps, target_step)
                values = run_data.sampled_probabilities[group_name].get(actual_step or -1, [])
                x_center = inner_left + plot_width * (run_idx + 0.5) / run_count
                color = RUN_COLORS[run_idx % len(RUN_COLORS)]
                polygon = _violin_polygon(values=values, center_x=x_center, width=42, top=inner_top, bottom=inner_bottom)
                if polygon is not None:
                    parts.append(
                        f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="1.4"/>'
                    )
                    sorted_values = sorted(values)
                    q1 = _percentile(sorted_values, 0.25)
                    q2 = _percentile(sorted_values, 0.5)
                    q3 = _percentile(sorted_values, 0.75)
                    def sy_local(value: float) -> float:
                        return inner_bottom - value * plot_height
                    parts.append(
                        f'<line x1="{x_center:.2f}" y1="{sy_local(q1):.2f}" x2="{x_center:.2f}" y2="{sy_local(q3):.2f}" stroke="{color}" stroke-width="3"/>'
                    )
                    parts.append(
                        f'<line x1="{x_center - 12:.2f}" y1="{sy_local(q2):.2f}" x2="{x_center + 12:.2f}" y2="{sy_local(q2):.2f}" stroke="{color}" stroke-width="3"/>'
                    )
                    parts.append(
                        f'<text x="{x_center:.2f}" y="{panel_top + height_per_row - 12:.2f}" text-anchor="middle" font-size="10" fill="{MUTED_TEXT}">n={len(values)}</text>'
                    )
                else:
                    parts.append(
                        f'<text x="{x_center:.2f}" y="{panel_top + height_per_row / 2:.2f}" text-anchor="middle" font-size="11" fill="{MUTED_TEXT}">no hits</text>'
                    )

                parts.append(
                    f'<text x="{x_center:.2f}" y="{inner_bottom + 18:.2f}" text-anchor="middle" font-size="11" fill="{TEXT_COLOR}">{_escape(run_data.spec.label)}</text>'
                )

    parts.append("</svg>")
    return "".join(parts)


def _scatter_svg(
    token_groups: Dict[str, Sequence[str]],
    run_data_list: Sequence[RunData],
    max_points_per_panel: int = 900,
    width: int = 1200,
    height_per_row: int = 280,
) -> str:
    if not run_data_list:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="80"><text x="20" y="44">No scatter data available.</text></svg>'

    rows = len(token_groups)
    cols = len(run_data_list)
    chart_height = 54 + rows * height_per_row
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_height}" viewBox="0 0 {width} {chart_height}">',
        f'<rect x="0" y="0" width="{width}" height="{chart_height}" fill="{CHART_BG}" rx="20"/>',
        f'<text x="26" y="28" font-size="18" font-weight="700" fill="{TEXT_COLOR}">Probability vs Entropy Scatter</text>',
        f'<text x="26" y="46" font-size="12" fill="{MUTED_TEXT}">Observed sampled-token instances aggregated across validation steps.</text>',
    ]

    panel_width = (width - 44) / float(max(cols, 1))
    entropy_max = 1.0
    for run_data in run_data_list:
        for group_name in token_groups:
            values = [item[1] for item in run_data.scatter_points[group_name]]
            if values:
                entropy_max = max(entropy_max, _percentile(sorted(values), 0.97))

    rng = random.Random(42)
    for row_idx, group_name in enumerate(token_groups):
        for col_idx, run_data in enumerate(run_data_list):
            panel_left = 22 + col_idx * panel_width
            panel_top = 58 + row_idx * height_per_row
            inner_left = panel_left + 52
            inner_right = panel_left + panel_width - 18
            inner_top = panel_top + 34
            inner_bottom = panel_top + height_per_row - 36
            plot_width = inner_right - inner_left
            plot_height = inner_bottom - inner_top
            color = GROUP_COLORS.get(group_name, RUN_COLORS[col_idx % len(RUN_COLORS)])

            parts.append(
                f'<rect x="{panel_left:.2f}" y="{panel_top:.2f}" width="{panel_width - 8:.2f}" height="{height_per_row - 10:.2f}" '
                f'fill="#FFFDFC" stroke="{PANEL_BORDER}" stroke-width="1.1" rx="16"/>'
            )
            parts.append(
                f'<text x="{panel_left + 16:.2f}" y="{panel_top + 22:.2f}" font-size="13" font-weight="700" fill="{TEXT_COLOR}">{_escape(group_name)} | {_escape(run_data.spec.label)}</text>'
            )

            for tick_idx in range(5):
                y = inner_bottom - plot_height * tick_idx / 4.0
                entropy_tick = entropy_max * tick_idx / 4.0
                parts.append(
                    f'<line x1="{inner_left:.2f}" y1="{y:.2f}" x2="{inner_right:.2f}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{inner_left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="{MUTED_TEXT}">{entropy_tick:.2f}</text>'
                )

            for tick_idx in range(6):
                x = inner_left + plot_width * tick_idx / 5.0
                prob_tick = tick_idx / 5.0
                parts.append(
                    f'<line x1="{x:.2f}" y1="{inner_top:.2f}" x2="{x:.2f}" y2="{inner_bottom:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{x:.2f}" y="{inner_bottom + 16:.2f}" text-anchor="middle" font-size="11" fill="{MUTED_TEXT}">{prob_tick:.1f}</text>'
                )

            values = list(run_data.scatter_points[group_name])
            if len(values) > max_points_per_panel:
                values = rng.sample(values, max_points_per_panel)

            for probability, entropy in values:
                x = inner_left + min(max(probability, 0.0), 1.0) * plot_width
                y = inner_bottom - min(max(entropy, 0.0), entropy_max) / entropy_max * plot_height
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.1" fill="{color}" fill-opacity="0.35"/>'
                )

            if values:
                probs = sorted(item[0] for item in values)
                entropies = sorted(item[1] for item in values)
                stat_text = (
                    f"n={len(values)} | mean p={statistics.mean(probs):.4f} | "
                    f"mean H={statistics.mean(entropies):.4f}"
                )
            else:
                stat_text = "no sampled token hits"
            parts.append(
                f'<text x="{panel_left + 16:.2f}" y="{panel_top + height_per_row - 12:.2f}" font-size="10.5" fill="{MUTED_TEXT}">{_escape(stat_text)}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def _topk_trend_svg(
    token_groups: Dict[str, Sequence[str]],
    run_data_list: Sequence[RunData],
    width: int = 1200,
    height_per_row: int = 280,
) -> str:
    if not run_data_list:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="80"><text x="20" y="44">No low-probability top-k data available.</text></svg>'

    rows = len(run_data_list)
    chart_height = 54 + rows * height_per_row
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_height}" viewBox="0 0 {width} {chart_height}">',
        f'<rect x="0" y="0" width="{width}" height="{chart_height}" fill="{CHART_BG}" rx="20"/>',
        f'<text x="26" y="28" font-size="18" font-weight="700" fill="{TEXT_COLOR}">Low-Probability Top-k Mean Trends</text>',
        f'<text x="26" y="46" font-size="12" fill="{MUTED_TEXT}">Average candidate probability inside the configured low-probability window.</text>',
    ]

    for row_idx, run_data in enumerate(run_data_list):
        panel_left = 22
        panel_top = 58 + row_idx * height_per_row
        panel_width = width - 44
        inner_left = panel_left + 64
        inner_right = panel_left + panel_width - 24
        inner_top = panel_top + 34
        inner_bottom = panel_top + height_per_row - 42
        plot_width = inner_right - inner_left
        plot_height = inner_bottom - inner_top

        series = {}
        for group_name in token_groups:
            points = []
            for step in run_data.steps:
                mean_value = run_data.step_summaries[step].topk_low_prob_group_mean.get(group_name)
                if mean_value is not None:
                    points.append((float(step), float(mean_value)))
            if points:
                series[group_name] = points

        min_x, max_x, min_y, max_y = _series_bounds(series or {"empty": [(0.0, 0.0), (1.0, 1.0)]}, pad_fraction=0.12)
        parts.append(
            f'<rect x="{panel_left:.2f}" y="{panel_top:.2f}" width="{panel_width:.2f}" height="{height_per_row - 10:.2f}" '
            f'fill="#FFFDFC" stroke="{PANEL_BORDER}" stroke-width="1.1" rx="16"/>'
        )
        parts.append(
            f'<text x="{panel_left + 16:.2f}" y="{panel_top + 22:.2f}" font-size="13" font-weight="700" fill="{TEXT_COLOR}">{_escape(run_data.spec.label)}</text>'
        )

        def sx(value: float) -> float:
            return inner_left + (value - min_x) / max(max_x - min_x, 1e-9) * plot_width

        def sy(value: float) -> float:
            return inner_top + plot_height - (value - min_y) / max(max_y - min_y, 1e-9) * plot_height

        for tick_idx in range(5):
            y_value = min_y + (max_y - min_y) * tick_idx / 4.0
            y_pos = sy(y_value)
            parts.append(
                f'<line x1="{inner_left:.2f}" y1="{y_pos:.2f}" x2="{inner_right:.2f}" y2="{y_pos:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{inner_left - 8:.2f}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="11" fill="{MUTED_TEXT}">{y_value:.4f}</text>'
            )

        for tick_idx in range(6):
            x_value = min_x + (max_x - min_x) * tick_idx / 5.0
            x_pos = sx(x_value)
            parts.append(
                f'<line x1="{x_pos:.2f}" y1="{inner_top:.2f}" x2="{x_pos:.2f}" y2="{inner_bottom:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x_pos:.2f}" y="{inner_bottom + 18:.2f}" text-anchor="middle" font-size="11" fill="{MUTED_TEXT}">{int(round(x_value))}</text>'
            )

        for idx, (group_name, points) in enumerate(series.items()):
            color = GROUP_COLORS.get(group_name, RUN_COLORS[idx % len(RUN_COLORS)])
            polyline = " ".join(f"{sx(x_value):.2f},{sy(y_value):.2f}" for x_value, y_value in points)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
            for x_value, y_value in points:
                parts.append(
                    f'<circle cx="{sx(x_value):.2f}" cy="{sy(y_value):.2f}" r="3.4" fill="{color}" stroke="{CHART_BG}" stroke-width="1"/>'
                )
            legend_x = panel_left + panel_width - 190
            legend_y = panel_top + 24 + idx * 18
            parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 16}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            parts.append(
                f'<text x="{legend_x + 22}" y="{legend_y + 4}" font-size="12" fill="{TEXT_COLOR}">{_escape(group_name)}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def _message_svg(title: str, message: str, width: int = 560, height: int = 220) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{CHART_BG}" rx="18"/>'
        f'<text x="24" y="34" font-size="16" font-weight="700" fill="{TEXT_COLOR}">{_escape(title)}</text>'
        f'<text x="24" y="{height / 2:.1f}" font-size="13" fill="{MUTED_TEXT}">{_escape(message)}</text>'
        "</svg>"
    )


def _token_metric_svg(snapshot: Dict[str, Any], key: str, title: str, y_label: str) -> str:
    token_details = snapshot.get("token_diagnostics", []) or []
    points = []
    for token_detail in token_details:
        metric_value = _to_float(token_detail.get(key))
        position = _to_float(token_detail.get("position"))
        if metric_value is None:
            continue
        points.append((position if position is not None else float(len(points)), metric_value))
    if not points:
        return _message_svg(title=title, message=f"No `{key}` values available for this sample.")
    return _line_chart_svg(
        title=title,
        y_label=y_label,
        series={title: points},
        width=560,
        height=220,
    )


def _format_topk_summary(token_detail: Dict[str, Any], limit: int = 5) -> str:
    entries = token_detail.get("topk", []) or []
    if not entries:
        return "-"
    rendered = []
    for entry in entries[:limit]:
        text = entry.get("text") or entry.get("token") or entry.get("token_id") or "?"
        prob = _to_float(entry.get("prob"))
        if prob is None:
            rendered.append(str(text))
        else:
            rendered.append(f"{text}({prob:.3f})")
    return " | ".join(rendered)


def _render_trace_step_summary_rows(snapshots: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for snapshot in snapshots:
        rows.append(
            f"""
            <tr>
              <td>{snapshot['step']}</td>
              <td>{_escape(_format_number(snapshot.get('primary_metric_value'), 4))}</td>
              <td>{_escape(str(snapshot.get('response_length') or '-'))}</td>
              <td>{_escape(_format_number(snapshot.get('token_entropy_mean'), 4))}</td>
              <td>{_escape(_format_number(snapshot.get('answer_tail_confidence'), 4))}</td>
              <td>{_escape(_format_number(snapshot.get('eos_prob_final'), 4))}</td>
              <td>{_escape(', '.join(snapshot.get('matched_tokens', [])) or '-')}</td>
            </tr>
            """
        )
    return "".join(rows)


def _select_trace_targets(
    run_data_list: Sequence[RunData],
    trace_uids: Sequence[str],
    trace_top_samples: int,
) -> List[Tuple[RunData, str]]:
    selected: List[Tuple[RunData, str]] = []
    seen = set()

    def add_target(run_data: RunData, uid: str):
        key = (run_data.spec.label, uid)
        if key in seen:
            return
        if uid not in run_data.trace_samples_by_uid:
            return
        selected.append((run_data, uid))
        seen.add(key)

    for query in trace_uids:
        for run_data in run_data_list:
            matched_uids = [uid for uid in run_data.trace_samples_by_uid.keys() if _uid_matches(query, uid)]
            for uid in sorted(matched_uids):
                add_target(run_data, uid)

    if trace_top_samples > 0:
        for run_data in run_data_list:
            candidates = sorted(run_data.last_step_trace_samples, key=_trace_sort_key)
            for sample in candidates[:trace_top_samples]:
                add_target(run_data, str(sample.get("uid")))

    return selected


def _render_token_traces_html(
    trace_targets: Sequence[Tuple[RunData, str]],
    output_dir: Path,
) -> str:
    nav_links = []
    sections = []
    for run_data, uid in trace_targets:
        trace_slug = f"{_sanitize_slug(run_data.spec.label)}-{_sanitize_slug(uid)}"
        nav_links.append(
            f'<a href="#{_escape(trace_slug)}">{_escape(run_data.spec.label)} / {_escape(uid)}</a>'
        )
        snapshots = [snapshot for _, snapshot in sorted(run_data.trace_samples_by_uid[uid].items())]
        if not snapshots:
            continue
        latest = snapshots[-1]
        summary_rows = _render_trace_step_summary_rows(snapshots)
        step_sections = []
        for snapshot in snapshots:
            charts = [
                _token_metric_svg(snapshot=snapshot, key="prob", title=f"Step {snapshot['step']} sampled probability", y_label="prob"),
                _token_metric_svg(snapshot=snapshot, key="entropy", title=f"Step {snapshot['step']} entropy", y_label="entropy"),
            ]
            if any(_to_float(item.get("eos_prob")) is not None for item in snapshot.get("token_diagnostics", []) or []):
                charts.append(
                    _token_metric_svg(snapshot=snapshot, key="eos_prob", title=f"Step {snapshot['step']} EOS probability", y_label="eos prob")
                )
            if any(_to_float(item.get("top1_prob")) is not None for item in snapshot.get("token_diagnostics", []) or []):
                charts.append(
                    _token_metric_svg(snapshot=snapshot, key="top1_prob", title=f"Step {snapshot['step']} top-1 probability", y_label="top1 prob")
                )

            token_rows = []
            for token_detail in snapshot.get("token_diagnostics", []) or []:
                token_rows.append(
                    f"""
                    <tr>
                      <td>{_escape(str(token_detail.get('position', '-')))}</td>
                      <td><code>{_escape(token_detail.get('text') or token_detail.get('token') or '-')}</code></td>
                      <td>{_escape(_format_number(_to_float(token_detail.get('prob')), 4))}</td>
                      <td>{_escape(_format_number(_to_float(token_detail.get('entropy')), 4))}</td>
                      <td>{_escape(_format_number(_to_float(token_detail.get('eos_prob')), 4))}</td>
                      <td><code>{_escape(token_detail.get('top1_text') or token_detail.get('top1_token') or '-')}</code></td>
                      <td>{_escape(_format_number(_to_float(token_detail.get('top1_prob')), 4))}</td>
                      <td>{_escape(_format_topk_summary(token_detail))}</td>
                    </tr>
                    """
                )

            step_sections.append(
                f"""
                <article class="trace-step-card">
                  <h3>Step {snapshot['step']}</h3>
                  <div class="trace-meta">
                    <span>{latest['primary_metric_key']}={_escape(_format_number(snapshot.get('primary_metric_value'), 4))}</span>
                    <span>len={_escape(str(snapshot.get('response_length') or '-'))}</span>
                    <span>token_count={_escape(str(snapshot.get('token_diagnostics_total_tokens') or '-'))}</span>
                    <span>matched={_escape(', '.join(snapshot.get('matched_tokens', [])) or '-')}</span>
                  </div>
                  <div class="trace-chart-grid">{''.join(charts)}</div>
                  <div class="trace-table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Pos</th>
                          <th>Token</th>
                          <th>Prob</th>
                          <th>Entropy</th>
                          <th>EOS Prob</th>
                          <th>Top1</th>
                          <th>Top1 Prob</th>
                          <th>Top-k</th>
                        </tr>
                      </thead>
                      <tbody>{''.join(token_rows) or '<tr><td colspan="8">No token diagnostics</td></tr>'}</tbody>
                    </table>
                  </div>
                </article>
                """
            )

        sections.append(
            f"""
            <section class="trace-card" id="{_escape(trace_slug)}">
              <div class="trace-card-header">
                <div>
                  <div class="trace-run">{_escape(run_data.spec.label)}</div>
                  <h2>{_escape(uid)}</h2>
                  <p class="trace-desc">available steps: {_escape(', '.join(str(snapshot['step']) for snapshot in snapshots))}</p>
                </div>
                <div class="trace-pill">{_escape(latest['primary_metric_key'])}: {_escape(_format_number(latest.get('primary_metric_value'), 4))}</div>
              </div>
              <div class="trace-io-grid">
                <div>
                  <h3>Prompt</h3>
                  <pre>{_escape(latest.get('input', ''))}</pre>
                </div>
                <div>
                  <h3>Latest output</h3>
                  <pre>{_escape(latest.get('output', ''))}</pre>
                </div>
              </div>
              <div class="trace-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>{_escape(latest['primary_metric_key'])}</th>
                      <th>Len</th>
                      <th>Token H</th>
                      <th>Tail P</th>
                      <th>EOS Final</th>
                      <th>Matched Tokens</th>
                    </tr>
                  </thead>
                  <tbody>{summary_rows}</tbody>
                </table>
              </div>
              <div class="trace-step-stack">{''.join(step_sections)}</div>
            </section>
            """
        )

    if not sections:
        sections_html = """
        <section class="trace-card">
          <h2>No token traces selected</h2>
          <p>No sample with token diagnostics matched the current selection.</p>
        </section>
        """
    else:
        sections_html = "".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Token Trace Report</title>
  <style>
    :root {{
      --bg: #f7f1e6;
      --card: rgba(255, 252, 247, 0.92);
      --ink: #1f2933;
      --muted: #59626b;
      --border: #d8cfc0;
      --accent: #125b50;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(18, 91, 80, 0.12), transparent 28rem),
        linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1380px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 42px;
      display: grid;
      gap: 18px;
    }}
    .hero, .trace-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 18px 42px rgba(31, 41, 51, 0.06);
      padding: 18px 20px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    .hero p, .trace-desc {{ color: var(--muted); }}
    .trace-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .trace-nav a, .trace-link {{
      color: white;
      text-decoration: none;
      background: var(--accent);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.92rem;
    }}
    .trace-card-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .trace-run {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.84rem;
      margin-bottom: 6px;
    }}
    .trace-pill {{
      color: var(--accent);
      font-weight: 700;
      white-space: nowrap;
    }}
    .trace-io-grid, .trace-chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
    }}
    .trace-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      margin: 10px 0 12px;
      font-size: 0.9rem;
    }}
    .trace-step-stack {{
      display: grid;
      gap: 18px;
      margin-top: 16px;
    }}
    .trace-step-card {{
      border-top: 1px solid rgba(216, 207, 192, 0.75);
      padding-top: 16px;
    }}
    .trace-table-wrap {{
      overflow: auto;
      margin-top: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid rgba(216, 207, 192, 0.7);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.84rem;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <a class="trace-link" href="{_escape((output_dir / 'index.html').name)}">Back to main report</a>
      <h1 style="margin-top: 14px;">Token Trace Report</h1>
      <p>Sample-level token tracking across validation steps for selected 1-shot examples.</p>
      <nav class="trace-nav">{''.join(nav_links) or '<span>No trace anchors</span>'}</nav>
    </section>
    {sections_html}
  </main>
</body>
</html>
"""


def _summary_cards(run_data_list: Sequence[RunData]) -> str:
    cards = []
    for run_data in run_data_list:
        if not run_data.steps:
            continue
        last_step = run_data.steps[-1]
        summary = run_data.step_summaries[last_step]
        coverage_text = f"{summary.dumped_sample_count}/{summary.sample_count} samples"
        cards.append(
            f"""
            <article class="summary-card">
              <div class="card-label">{_escape(run_data.spec.label)}</div>
              <div class="card-value">{_escape(_format_number(summary.primary_metric_mean, 4))}</div>
              <div class="card-meta">last-step {summary.primary_metric_key}</div>
              <dl>
                <dt>Step</dt><dd>{last_step}</dd>
                <dt>Actor entropy</dt><dd>{_escape(_format_number(summary.actor_entropy, 4))}</dd>
                <dt>Token entropy</dt><dd>{_escape(_format_number(summary.token_entropy_mean, 4))}</dd>
                <dt>Coverage</dt><dd>{_escape(coverage_text)}</dd>
              </dl>
            </article>
            """
        )
    return "".join(cards)


def _frequency_table_html(
    token_groups: Dict[str, Sequence[str]],
    run_data_list: Sequence[RunData],
    selected_steps: Sequence[int],
) -> str:
    sections = []
    for group_name, tokens in token_groups.items():
        header = ["<tr><th>Run</th><th>Step</th>"]
        for token in tokens:
            header.append(f"<th>{_escape(token)}</th>")
        header.append("</tr>")
        rows = []
        for run_data in run_data_list:
            for target_step in selected_steps:
                actual_step = _select_nearest_step(run_data.steps, target_step)
                if actual_step is None:
                    continue
                summary = run_data.step_summaries[actual_step]
                row = [f"<tr><td>{_escape(run_data.spec.label)}</td><td>{actual_step}</td>"]
                token_frequency = summary.token_frequency_by_group.get(group_name, {})
                for token in tokens:
                    row.append(f"<td>{_escape(_format_number(token_frequency.get(normalize_token_text(token)), 5))}</td>")
                row.append("</tr>")
                rows.append("".join(row))
        sections.append(
            f"""
            <section class="table-card">
              <h3>{_escape(group_name)} token frequency</h3>
              <table>
                <thead>{''.join(header)}</thead>
                <tbody>{''.join(rows) or '<tr><td colspan="99">No token frequency data</td></tr>'}</tbody>
              </table>
            </section>
            """
        )
    return "".join(sections)


def _sample_table_html(run_data_list: Sequence[RunData], max_samples_per_run: int = 6) -> str:
    sections = []
    for run_data in run_data_list:
        if not run_data.last_step_samples:
            continue
        samples = list(run_data.last_step_samples)
        samples.sort(key=lambda item: (item.get("primary_metric_value") is None, -(item.get("primary_metric_value") or -1e9)))
        selected_samples = samples[:max_samples_per_run]
        rows = []
        for sample in selected_samples:
            metric_value = sample.get("primary_metric_value")
            matched_tokens = ", ".join(sample.get("matched_tokens", [])) or "-"
            rows.append(
                f"""
                <tr>
                  <td>{_escape(sample.get('uid') or '-')}</td>
                  <td>{_escape(_format_number(metric_value, 4))}</td>
                  <td>{_escape(matched_tokens)}</td>
                  <td><details><summary>prompt</summary><pre>{_escape(_render_text_preview(sample.get('input', ''), 360))}</pre></details></td>
                  <td><details open><summary>output</summary><pre>{_escape(_render_text_preview(sample.get('output', ''), 520))}</pre></details></td>
                </tr>
                """
            )
        sections.append(
            f"""
            <section class="table-card">
              <h3>Representative samples: {_escape(run_data.spec.label)}</h3>
              <table class="sample-table">
                <thead>
                  <tr>
                    <th>UID</th>
                    <th>{_escape(run_data.primary_metric_key)}</th>
                    <th>Matched tokens</th>
                    <th>Prompt</th>
                    <th>Output</th>
                  </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            """
        )
    return "".join(sections)


def _warning_html(run_data_list: Sequence[RunData]) -> str:
    warnings = sorted({warning for run_data in run_data_list for warning in run_data.warnings})
    if not warnings:
        return ""
    items = "".join(f"<li>{_escape(item)}</li>" for item in warnings[:20])
    return f"""
    <section class="warning-card">
      <h3>Coverage warnings</h3>
      <ul>{items}</ul>
    </section>
    """


def _report_html(
    run_data_list: Sequence[RunData],
    token_groups: Dict[str, Sequence[str]],
    selected_steps: Sequence[int],
    output_dir: Path,
    has_token_traces: bool,
) -> str:
    token_trace_card = ""
    if has_token_traces:
        token_trace_card = (
            f'<article class="figure-card"><h2>Token traces</h2>'
            f'<p class="subtitle">Sample-level token tracking across validation steps for selected one-shot examples.</p>'
            f'<p><a class="trace-link" href="{_escape((output_dir / "token_traces.html").name)}">Open token trace report</a></p>'
            f"</article>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Validation Visualization Report</title>
  <style>
    :root {{
      --bg: #f7f1e6;
      --card: rgba(255, 252, 247, 0.9);
      --ink: #1f2933;
      --muted: #59626b;
      --border: #d8cfc0;
      --accent: #125b50;
      --accent-soft: #dcebe6;
      --warn: #8e3200;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(18, 91, 80, 0.12), transparent 28rem),
        radial-gradient(circle at top right, rgba(142, 50, 0, 0.08), transparent 24rem),
        linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1320px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      display: grid;
      gap: 14px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      max-width: 72rem;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .summary-card, .table-card, .warning-card, .figure-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 18px 42px rgba(31, 41, 51, 0.06);
      backdrop-filter: blur(10px);
    }}
    .summary-card {{
      padding: 18px 18px 14px;
    }}
    .summary-card .card-label {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .summary-card .card-value {{
      font-size: 2rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .summary-card .card-meta {{
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .summary-card dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 6px 12px;
      margin: 0;
      font-size: 0.92rem;
    }}
    .summary-card dt {{
      color: var(--muted);
    }}
    .summary-card dd {{
      margin: 0;
      font-variant-numeric: tabular-nums;
    }}
    .figure-grid {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .figure-card {{
      padding: 16px;
      overflow: auto;
    }}
    .figure-card h2 {{
      margin: 0 0 10px;
      font-size: 1.05rem;
    }}
    img, svg {{
      display: block;
      max-width: 100%;
      height: auto;
      border-radius: 16px;
    }}
    .table-stack {{
      display: grid;
      gap: 18px;
    }}
    .table-card {{
      padding: 16px 16px 18px;
      overflow: auto;
    }}
    .table-card h3 {{
      margin: 0 0 12px;
      font-size: 1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(216, 207, 192, 0.8);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.82rem;
      line-height: 1.45;
      color: #24313b;
    }}
    details summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }}
    .warning-card {{
      margin-bottom: 24px;
      border-color: rgba(142, 50, 0, 0.24);
      background: rgba(255, 249, 244, 0.92);
      padding: 16px 18px;
    }}
    .warning-card h3 {{
      margin: 0 0 10px;
      color: var(--warn);
    }}
    .warning-card ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .trace-link {{
      color: white;
      text-decoration: none;
      background: var(--accent);
      border-radius: 999px;
      display: inline-block;
      padding: 8px 12px;
      font-size: 0.92rem;
      margin-top: 4px;
    }}
    .footer-note {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">validation report</div>
      <h1>Paper-Style Experiment Visualization</h1>
      <p class="subtitle">
        Generated from local validation dumps and optional file logger metrics.
        Selected steps: {_escape(", ".join(str(step) for step in selected_steps) or "-")}.
        Token groups: {_escape(", ".join(f"{name}={','.join(tokens)}" for name, tokens in token_groups.items()))}.
      </p>
    </header>

    <section class="summary-grid">
      {_summary_cards(run_data_list)}
    </section>

    {_warning_html(run_data_list)}

    <section class="figure-grid">
      <article class="figure-card">
        <h2>Training dynamics</h2>
        <img src="{_escape((output_dir / 'training_metric.svg').name)}" alt="training metric curve" />
        <img src="{_escape((output_dir / 'training_entropy.svg').name)}" alt="entropy curve" style="margin-top: 14px;" />
      </article>
      <article class="figure-card">
        <h2>Sampled probability distributions</h2>
        <img src="{_escape((output_dir / 'sampled_probability_distributions.svg').name)}" alt="sampled probability distributions" />
      </article>
      <article class="figure-card">
        <h2>Probability-entropy scatter</h2>
        <img src="{_escape((output_dir / 'probability_entropy_scatter.svg').name)}" alt="probability entropy scatter" />
      </article>
      <article class="figure-card">
        <h2>Low-probability top-k trends</h2>
        <img src="{_escape((output_dir / 'low_prob_topk_trends.svg').name)}" alt="low probability topk trends" />
      </article>
      {token_trace_card}
    </section>

    <section class="table-stack">
      {_frequency_table_html(token_groups, run_data_list, selected_steps)}
      {_sample_table_html(run_data_list)}
    </section>

    <p class="footer-note">
      Files written under {_escape(str(output_dir))}. If token coverage warnings appear above, increase
      <code>trainer.validation_diagnostics.samples_to_dump_token_details</code> and
      <code>trainer.validation_diagnostics.token_distribution_topk</code> for the next run.
    </p>
  </main>
</body>
</html>
"""


def _build_training_series(run_data_list: Sequence[RunData], value_getter) -> Dict[str, List[Tuple[float, float]]]:
    series: Dict[str, List[Tuple[float, float]]] = {}
    for run_data in run_data_list:
        points = []
        for step in run_data.steps:
            value = value_getter(run_data.step_summaries[step])
            if value is not None:
                points.append((float(step), float(value)))
        series[run_data.spec.label] = points
    return series


def _summary_payload(run_data_list: Sequence[RunData], token_groups: Dict[str, Sequence[str]], selected_steps: Sequence[int]) -> Dict[str, Any]:
    payload = {
        "selected_steps": list(selected_steps),
        "token_groups": {name: list(tokens) for name, tokens in token_groups.items()},
        "runs": [],
    }
    for run_data in run_data_list:
        run_payload = {
            "label": run_data.spec.label,
            "validation_dir": str(run_data.spec.validation_dir),
            "metrics_file": str(run_data.spec.metrics_file) if run_data.spec.metrics_file else None,
            "primary_metric_key": run_data.primary_metric_key,
            "warnings": list(run_data.warnings),
            "steps": [],
        }
        for step in run_data.steps:
            summary = run_data.step_summaries[step]
            run_payload["steps"].append(
                {
                    "step": summary.step,
                    "sample_count": summary.sample_count,
                    "primary_metric_mean": summary.primary_metric_mean,
                    "actor_entropy": summary.actor_entropy,
                    "token_entropy_mean": summary.token_entropy_mean,
                    "dumped_sample_ratio": summary.dumped_sample_ratio,
                    "dumped_token_ratio": summary.dumped_token_ratio,
                    "sampled_group_count": summary.sampled_group_count,
                    "sampled_group_frequency": summary.sampled_group_frequency,
                    "topk_low_prob_group_mean": summary.topk_low_prob_group_mean,
                }
            )
        payload["runs"].append(run_payload)
    return payload


def generate_report(
    run_specs: Sequence[RunSpec],
    output_dir: Path,
    token_groups: Dict[str, Sequence[str]],
    selected_steps: Optional[Sequence[int]] = None,
    low_prob_min: float = 0.0,
    low_prob_max: float = 0.1,
    trace_uids: Optional[Sequence[str]] = None,
    trace_top_samples: int = 6,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_data_list = [
        load_run_data(run_spec=run_spec, token_groups=token_groups, low_prob_min=low_prob_min, low_prob_max=low_prob_max)
        for run_spec in run_specs
    ]

    resolved_steps = list(selected_steps or _pick_reference_steps(run_data_list, count=3))

    primary_metric_series = _build_training_series(run_data_list, lambda summary: summary.primary_metric_mean)
    actor_entropy_series = _build_training_series(
        run_data_list,
        lambda summary: summary.actor_entropy if summary.actor_entropy is not None else summary.token_entropy_mean,
    )

    (output_dir / "training_metric.svg").write_text(
        _line_chart_svg(
            title="Validation metric trajectory",
            y_label="metric",
            series=primary_metric_series,
        ),
        encoding="utf-8",
    )
    (output_dir / "training_entropy.svg").write_text(
        _line_chart_svg(
            title="Actor entropy trajectory",
            y_label="entropy",
            series=actor_entropy_series,
        ),
        encoding="utf-8",
    )
    (output_dir / "sampled_probability_distributions.svg").write_text(
        _distribution_svg(token_groups=token_groups, run_data_list=run_data_list, selected_steps=resolved_steps),
        encoding="utf-8",
    )
    (output_dir / "probability_entropy_scatter.svg").write_text(
        _scatter_svg(token_groups=token_groups, run_data_list=run_data_list),
        encoding="utf-8",
    )
    (output_dir / "low_prob_topk_trends.svg").write_text(
        _topk_trend_svg(token_groups=token_groups, run_data_list=run_data_list),
        encoding="utf-8",
    )

    trace_targets = _select_trace_targets(
        run_data_list=run_data_list,
        trace_uids=list(trace_uids or []),
        trace_top_samples=int(trace_top_samples),
    )
    has_token_traces = len(trace_targets) > 0
    if has_token_traces:
        (output_dir / "token_traces.html").write_text(
            _render_token_traces_html(trace_targets=trace_targets, output_dir=output_dir),
            encoding="utf-8",
        )

    summary = _summary_payload(run_data_list=run_data_list, token_groups=token_groups, selected_steps=resolved_steps)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _report_html(
            run_data_list=run_data_list,
            token_groups=token_groups,
            selected_steps=resolved_steps,
            output_dir=output_dir,
            has_token_traces=has_token_traces,
        ),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-style validation visualizations from validation JSONL dumps.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: label=/path/to/validation[::/path/to/metrics.jsonl]. Repeat for multi-run comparison.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to write SVG/HTML report artifacts.",
    )
    parser.add_argument(
        "--token-group",
        action="append",
        default=[],
        help="Token group spec: name=token1,token2,token3. If omitted, paper-inspired defaults are used.",
    )
    parser.add_argument(
        "--selected-steps",
        default="",
        help="Comma-separated target steps for distribution/frequency views. Defaults to first/middle/last observed steps.",
    )
    parser.add_argument(
        "--low-prob-max",
        type=float,
        default=0.1,
        help="Upper bound for the low-probability top-k trend view.",
    )
    parser.add_argument(
        "--low-prob-min",
        type=float,
        default=0.0,
        help="Lower bound for the low-probability top-k trend view.",
    )
    parser.add_argument(
        "--trace-uid",
        action="append",
        default=[],
        help="Sample UID to trace across validation steps. Repeat this flag to trace multiple UIDs.",
    )
    parser.add_argument(
        "--trace-top-samples",
        type=int,
        default=6,
        help="Automatically include top-N latest-step samples with token diagnostics in token trace output. Use 0 to disable.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_specs = [_parse_run_spec(item) for item in args.run]
    token_groups = dict(DEFAULT_TOKEN_GROUPS)
    for item in args.token_group:
        name, tokens = _parse_token_group_spec(item)
        token_groups[name] = tokens

    selected_steps = [int(item.strip()) for item in args.selected_steps.split(",") if item.strip()]
    output_dir = Path(args.output_dir).expanduser()
    generate_report(
        run_specs=run_specs,
        output_dir=output_dir,
        token_groups=token_groups,
        selected_steps=selected_steps or None,
        low_prob_min=float(args.low_prob_min),
        low_prob_max=float(args.low_prob_max),
        trace_uids=args.trace_uid,
        trace_top_samples=int(args.trace_top_samples),
    )
    print(f"Validation visualization report written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
