#!/usr/bin/env python3
"""Render readable token-entropy traces as text with heat-colored backgrounds."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from scripts.render_entropy_trajectory_heatmaps import (
    _metric_limits,
    _safe_slug,
    _to_float,
    load_samples,
    load_tokens_for_uids,
    select_trajectories,
)


DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LIGHT_ENTROPY_PALETTE = [
    (239, 246, 255),
    (219, 234, 254),
    (254, 243, 199),
    (253, 186, 116),
    (252, 165, 165),
]


def parse_steps(value: str) -> Optional[set[int]]:
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _display_text(text: str) -> str:
    text = text.replace("\r", "")
    text = text.replace("\t", " ")
    text = text.replace("\u00a0", " ")
    return text


def token_rows_to_word_units(
    rows: list[dict[str, Any]],
    metric: str,
    max_words: int,
    preserve_newlines: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    units: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_values: list[float] = []
    truncated = False

    def flush_word() -> None:
        nonlocal current_parts, current_values, truncated
        if not current_parts:
            return
        text = "".join(current_parts)
        value = sum(current_values) / len(current_values) if current_values else float("nan")
        units.append({"text": text, metric: value, "newline": False})
        current_parts = []
        current_values = []
        if len([unit for unit in units if not unit.get("newline")]) >= max_words:
            truncated = True

    for row in rows:
        if truncated:
            break
        raw_text = _display_text(str(row.get("text") or row.get("token") or ""))
        if not raw_text:
            continue
        value = _to_float(row.get(metric))
        if value is None:
            continue

        for segment in re.findall(r"\n+|[^\S\n]+|\S+", raw_text):
            if truncated:
                break
            if "\n" in segment:
                flush_word()
                if preserve_newlines and units and not units[-1].get("newline"):
                    units.append({"newline": True})
                continue
            if segment.isspace():
                flush_word()
                continue
            current_parts.append(segment)
            current_values.append(value)

    if not truncated:
        flush_word()
    return units, truncated


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.truetype(DEFAULT_FONT, size=size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _color_for_value(value: Any, vmin: float, vmax: float) -> tuple[int, int, int]:
    value = _to_float(value)
    if value is None:
        return (235, 235, 235)
    if vmax <= vmin:
        scale = 0.0
    else:
        scale = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    return _palette_rgb(scale)


def _palette_rgb(scale: float) -> tuple[int, int, int]:
    scale = min(max(float(scale), 0.0), 1.0)
    stops = LIGHT_ENTROPY_PALETTE
    if scale >= 1.0:
        return stops[-1]
    pos = scale * (len(stops) - 1)
    idx = int(math.floor(pos))
    frac = pos - idx
    left = stops[idx]
    right = stops[idx + 1]
    return tuple(int(round(left[channel] * (1.0 - frac) + right[channel] * frac)) for channel in range(3))


def _percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * min(max(percentile, 0.0), 100.0) / 100.0
    low_idx = int(math.floor(rank))
    high_idx = int(math.ceil(rank))
    if low_idx == high_idx:
        return values[low_idx]
    frac = rank - low_idx
    return values[low_idx] * (1.0 - frac) + values[high_idx] * frac


def _display_metric_limits(
    units_by_step: dict[int, tuple[list[dict[str, Any]], bool]],
    metric: str,
    fallback_vmin: float,
    fallback_vmax: float,
) -> tuple[float, float]:
    values: list[float] = []
    for units, _ in units_by_step.values():
        for unit in units:
            if unit.get("newline"):
                continue
            value = _to_float(unit.get(metric))
            if value is not None:
                values.append(value)
    if not values:
        return fallback_vmin, fallback_vmax
    low = _percentile(values, 5)
    high = _percentile(values, 95)
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        low = min(values)
        high = max(values)
    if low == high:
        high = low + 1.0
    return low, high


def _text_color_for_bg(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (17, 24, 39)


def _short_uid_label(uid: str) -> str:
    match = re.search(r"math500::(\d+).*::sample(\d+)", uid)
    if match:
        return f"MATH500 #{match.group(1)}, sample {match.group(2)}"
    return uid


def _summarize_step(samples_for_uid: pd.DataFrame, step: int) -> str:
    row = samples_for_uid[samples_for_uid["step"] == step]
    if row.empty:
        return f"step {step}"
    item = row.iloc[0]
    score = _to_float(item.get("score"))
    entropy = _to_float(item.get("token_entropy_mean"))
    length = _to_float(item.get("response_length"))
    return (
        f"step {step} | score={score:.3f} | mean entropy={entropy:.3f} | length={length:.0f}"
        if score is not None and entropy is not None and length is not None
        else f"step {step}"
    )


def render_text_trace(
    uid: str,
    step_rows: dict[int, list[dict[str, Any]]],
    samples_for_uid: pd.DataFrame,
    metric: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    font_path: str,
    font_size: int,
    max_words_per_step: int,
    max_lines_per_step: int,
    image_width: int,
    preserve_newlines: bool,
) -> None:
    margin_x = 34
    margin_top = 34
    margin_bottom = 52
    gutter = 14
    header_gap = 16
    token_pad_x = 5
    token_pad_y = 3
    line_gap = 10
    block_gap = 26
    legend_h = 26

    font = _font(font_path, font_size)
    header_font = _font(font_path, font_size + 4)
    small_font = _font(font_path, max(11, font_size - 4))
    title_font = _font(font_path, font_size + 8)

    scratch = Image.new("RGB", (image_width, 100), "white")
    draw = ImageDraw.Draw(scratch)
    _, token_h = _text_size(draw, "Ag", font)
    line_h = token_h + 2 * token_pad_y + line_gap
    _, header_h = _text_size(draw, "step 000", header_font)
    _, title_h = _text_size(draw, "title", title_font)
    steps = sorted(step_rows.keys())

    block_heights: list[int] = []
    units_by_step: dict[int, tuple[list[dict[str, Any]], bool]] = {}
    for step in steps:
        units, truncated = token_rows_to_word_units(
            step_rows[step],
            metric=metric,
            max_words=max_words_per_step,
            preserve_newlines=preserve_newlines,
        )
        units_by_step[step] = (units, truncated)
        block_heights.append(header_h + header_gap + max_lines_per_step * line_h + block_gap)
    display_vmin, display_vmax = _display_metric_limits(units_by_step, metric, vmin, vmax)

    height = margin_top + title_h + 20 + sum(block_heights) + legend_h + margin_bottom
    image = Image.new("RGB", (image_width, height), "white")
    draw = ImageDraw.Draw(image)

    title = f"Token entropy over generated reasoning: {_short_uid_label(uid)}"
    draw.text((margin_x, margin_top), title, fill=(20, 25, 32), font=title_font)
    y = margin_top + title_h + 20
    usable_width = image_width - 2 * margin_x

    for step in steps:
        header = _summarize_step(samples_for_uid, step)
        draw.text((margin_x, y), header, fill=(32, 42, 52), font=header_font)
        y += header_h + header_gap
        x = margin_x
        lines_used = 1
        units, truncated_by_words = units_by_step[step]
        truncated_by_lines = False

        for unit in units:
            if unit.get("newline"):
                x = margin_x
                y += line_h
                lines_used += 1
                if lines_used > max_lines_per_step:
                    truncated_by_lines = True
                    break
                continue

            text = str(unit.get("text") or "")
            if not text:
                continue
            word_w, word_h = _text_size(draw, text, font)
            box_w = word_w + 2 * token_pad_x
            if x + box_w > margin_x + usable_width:
                x = margin_x
                y += line_h
                lines_used += 1
                if lines_used > max_lines_per_step:
                    truncated_by_lines = True
                    break

            bg = _color_for_value(unit.get(metric), vmin=display_vmin, vmax=display_vmax)
            fg = _text_color_for_bg(bg)
            top = y
            bottom = y + word_h + 2 * token_pad_y
            draw.rounded_rectangle(
                (x, top, x + box_w, bottom),
                radius=4,
                fill=bg,
                outline=(209, 213, 219),
                width=1,
            )
            draw.text((x + token_pad_x, top + token_pad_y), text, fill=fg, font=font)
            x += box_w + gutter

        if truncated_by_words or truncated_by_lines:
            ellipsis = "... truncated for display"
            y += line_h if lines_used <= max_lines_per_step else 0
            draw.text((margin_x, y), ellipsis, fill=(95, 102, 112), font=small_font)
        y += (max_lines_per_step - min(lines_used, max_lines_per_step)) * line_h + block_gap

    legend_x = margin_x
    legend_y = height - margin_bottom
    legend_w = min(520, usable_width)
    for idx in range(legend_w):
        rgb = _palette_rgb(idx / max(legend_w - 1, 1))
        draw.line((legend_x + idx, legend_y, legend_x + idx, legend_y + 14), fill=rgb)
    draw.rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + 14), outline=(70, 70, 70), width=1)
    draw.text((legend_x, legend_y + 20), f"low {metric}", fill=(45, 52, 60), font=small_font)
    high_label = f"high {metric}"
    high_w, _ = _text_size(draw, high_label, small_font)
    draw.text((legend_x + legend_w - high_w, legend_y + 20), high_label, fill=(45, 52, 60), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", default="0,60,120,180")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--uid", action="append", default=[])
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--metric", default="entropy", choices=["entropy", "prob", "logprob", "top1_prob", "eos_prob", "topk_mass"])
    parser.add_argument("--max-words-per-step", type=int, default=180)
    parser.add_argument("--max-lines-per-step", type=int, default=10)
    parser.add_argument("--image-width", type=int, default=1900)
    parser.add_argument("--font-size", type=int, default=24)
    parser.add_argument("--font-path", default=DEFAULT_FONT)
    parser.add_argument(
        "--collapse-newlines",
        action="store_true",
        help="Treat generated newlines as spaces so a complete trace can fit in a compact slide.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report_dir = Path(args.report_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else report_dir / "trajectory_entropy_text_heatmaps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    step_filter = parse_steps(args.steps)
    samples = load_samples(report_dir, step_filter=step_filter)
    selected = select_trajectories(
        samples=samples,
        top_n=int(args.top_n),
        min_steps=int(args.min_steps),
        uid_queries=list(args.uid or []),
        distinct_cases=True,
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

    selected.to_csv(output_dir / "selected_text_trajectories.csv", index=False)
    written = []
    for uid in selected["uid"].tolist():
        uid = str(uid)
        step_rows = tokens_by_uid_step.get(uid, {})
        if not step_rows:
            print(f"[warn] no token rows loaded for {uid}")
            continue
        path = output_dir / f"{_safe_slug(uid)}.text_{args.metric}.png"
        render_text_trace(
            uid=uid,
            step_rows=step_rows,
            samples_for_uid=samples[samples["uid"] == uid].sort_values("step"),
            metric=str(args.metric),
            output_path=path,
            vmin=vmin,
            vmax=vmax,
            font_path=str(args.font_path),
            font_size=int(args.font_size),
            max_words_per_step=int(args.max_words_per_step),
            max_lines_per_step=int(args.max_lines_per_step),
            image_width=int(args.image_width),
            preserve_newlines=not bool(args.collapse_newlines),
        )
        written.append(path)
        print(f"[write] {path}")
    print(f"[done] rendered {len(written)} text heatmaps under {output_dir}")


if __name__ == "__main__":
    main()
