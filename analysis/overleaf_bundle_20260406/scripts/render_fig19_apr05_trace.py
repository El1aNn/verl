#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/rl/verl/analysis/overleaf_bundle_20260406")
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
TRACE_SOURCE = (
    Path("/root/rl/verl")
    / "ckpts"
    / "verl_grpo_dsr_sub_baseline"
    / "base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719"
    / "validation"
    / "180.jsonl"
)
CSV_PATH = DATA_DIR / "fig19-apr05-eight-question-trace.csv"
PNG_PATH = FIG_DIR / "correct-vs-wrong-entropy.png"

BG = "#FCFBF8"
GRID = "#DDD7CD"
TEXT = "#2F2F2F"
CORRECT = "#2A7F62"
WRONG = "#AA3F39"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_user_prompt(input_text: str) -> str:
    marker = "user\n"
    if marker not in input_text:
        return input_text.strip()
    tail = input_text.split(marker, 1)[1]
    if "\nassistant\n" in tail:
        tail = tail.split("\nassistant\n", 1)[0]
    return tail.strip()


def build_rows() -> list[dict]:
    rows = []
    for row in read_jsonl(TRACE_SOURCE):
        token_diag = row.get("token_diagnostics") or []
        if not token_diag:
            continue
        entropy_values = [item["entropy"] for item in token_diag if isinstance(item.get("entropy"), (int, float))]
        top1_values = [item["top1_prob"] for item in token_diag if isinstance(item.get("top1_prob"), (int, float))]
        eos_values = [item["eos_prob"] for item in token_diag if isinstance(item.get("eos_prob"), (int, float))]
        uid = str(row.get("uid", ""))
        parts = uid.split("::")
        question_id = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
        rows.append(
            {
                "paper_label": "Base small-group ablation (n=1)",
                "run_slug": "base_c_n1_group_ablation_rerun180_2gpu_restart_20260405_213719",
                "source_file": str(TRACE_SOURCE),
                "final_step": int(row.get("step", 0)),
                "uid": uid,
                "question_id": question_id,
                "score": float(row.get("score", 0.0)),
                "correctness": "correct" if float(row.get("score", 0.0)) >= 0.5 else "wrong",
                "traced_tokens": len(token_diag),
                "mean_token_entropy": statistics.fmean(entropy_values),
                "mean_top1_prob": statistics.fmean(top1_values) if top1_values else float("nan"),
                "mean_eos_prob": statistics.fmean(eos_values) if eos_values else float("nan"),
                "response_length": int(row.get("response_length", 0)),
                "question": extract_user_prompt(str(row.get("input", ""))),
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "paper_label",
                "run_slug",
                "source_file",
                "final_step",
                "uid",
                "question_id",
                "score",
                "correctness",
                "traced_tokens",
                "mean_token_entropy",
                "mean_top1_prob",
                "mean_eos_prob",
                "response_length",
                "question",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def render_png(rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1280, 820), BG)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(34, bold=True)
    subtitle_font = _load_font(22)
    axis_font = _load_font(20)
    tick_font = _load_font(18)
    note_font = _load_font(16)

    draw.text((70, 40), "Final-step token entropy by correctness", font=title_font, fill=TEXT)
    draw.text((70, 90), "Eight-question traced subset from Base n=1", font=subtitle_font, fill="#4A4A4A")

    left, top, right, bottom = 130, 180, 1180, 660
    chart_h = bottom - top
    chart_w = right - left
    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)

    values = [float(row["mean_token_entropy"]) for row in rows]
    y_min = min(values)
    y_max = max(values)
    lower = max(0.0, y_min - 0.03)
    upper = y_max + 0.06
    grid_steps = 5
    for idx in range(grid_steps + 1):
        frac = idx / grid_steps
        y_value = upper - (upper - lower) * frac
        y = top + chart_h * frac
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((50, y - 10), f"{y_value:.2f}", font=tick_font, fill="#5A5A5A")

    order = ["correct", "wrong"]
    group_x = {
        "correct": left + chart_w * 0.30,
        "wrong": left + chart_w * 0.72,
    }
    colors = {
        "correct": CORRECT,
        "wrong": WRONG,
    }
    labels = {
        "correct": "Correct",
        "wrong": "Wrong",
    }

    def y_of(value: float) -> float:
        return bottom - (value - lower) / (upper - lower) * chart_h

    for group in order:
        sub = sorted(
            [row for row in rows if row["correctness"] == group],
            key=lambda item: (item["question_id"] is None, item["question_id"], item["uid"]),
        )
        xs = []
        if len(sub) == 1:
            xs = [group_x[group]]
        else:
            span = 140
            start = group_x[group] - span / 2
            step = span / (len(sub) - 1)
            xs = [start + idx * step for idx in range(len(sub))]

        mean_val = statistics.fmean(float(row["mean_token_entropy"]) for row in sub)
        bar_top = y_of(mean_val)
        bar_left = group_x[group] - 70
        bar_right = group_x[group] + 70
        draw.rounded_rectangle((bar_left, bar_top, bar_right, bottom), radius=18, fill=colors[group])

        for xpos, row in zip(xs, sub):
            ypos = y_of(float(row["mean_token_entropy"]))
            draw.ellipse((xpos - 9, ypos - 9, xpos + 9, ypos + 9), fill=colors[group], outline="white", width=2)

        _text_center(draw, (group_x[group], bottom + 42), f"{labels[group]} (n={len(sub)})", font=axis_font, fill=TEXT)
        _text_center(draw, (group_x[group], bar_top - 28), f"mean={mean_val:.3f}", font=tick_font, fill=TEXT)

    draw.text((left, top - 34), "Mean token entropy", font=axis_font, fill=TEXT)
    draw.text((70, 720), "Traced MATH500 question ids 0-7", font=note_font, fill="#5A5A5A")
    draw.text((70, 748), "Correct responses: 5; wrong responses: 3; dots are per-question means", font=note_font, fill="#5A5A5A")
    image.save(PNG_PATH)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    render_png(rows)
    print(f"wrote {CSV_PATH}")
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
