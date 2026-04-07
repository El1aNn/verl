#!/usr/bin/env python3
"""
Generate a random-wrong reward parquet by rewriting reward_model.ground_truth.

The script keeps every column unchanged except reward_model.ground_truth.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd


def _format_float(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def _sample_wrong_scalar(
    rng: random.Random,
    original: Any,
    value_min: float,
    value_max: float,
    decimals: int,
    max_attempts: int = 1000,
) -> str:
    original_str = str(original)
    for _ in range(max_attempts):
        candidate = _format_float(rng.uniform(value_min, value_max), decimals)
        if candidate != original_str:
            return candidate
    raise RuntimeError(f"Failed to sample wrong ground_truth different from {original_str!r}")


def _rewrite_ground_truth(
    reward_model: dict[str, Any],
    rng: random.Random,
    value_min: float,
    value_max: float,
    decimals: int,
) -> tuple[dict[str, Any], Any, Any]:
    if not isinstance(reward_model, dict):
        raise TypeError(f"reward_model must be dict, got {type(reward_model)}")
    if "ground_truth" not in reward_model:
        raise KeyError("reward_model missing key: ground_truth")

    old_gt = reward_model["ground_truth"]
    if isinstance(old_gt, list):
        if len(old_gt) == 0:
            new_gt = [_sample_wrong_scalar(rng, "", value_min, value_max, decimals)]
        else:
            new_gt = [_sample_wrong_scalar(rng, item, value_min, value_max, decimals) for item in old_gt]
    elif isinstance(old_gt, tuple):
        if len(old_gt) == 0:
            new_gt = (_sample_wrong_scalar(rng, "", value_min, value_max, decimals),)
        else:
            new_gt = tuple(_sample_wrong_scalar(rng, item, value_min, value_max, decimals) for item in old_gt)
    else:
        new_gt = _sample_wrong_scalar(rng, old_gt, value_min, value_max, decimals)

    new_reward_model = dict(reward_model)
    new_reward_model["ground_truth"] = new_gt
    return new_reward_model, old_gt, new_gt


def _canon(x: Any) -> str:
    return repr(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a random-wrong reward parquet dataset.")
    parser.add_argument(
        "--input-file",
        default="/root/rl/verl/data/dsr_sub/pi1_one_ans.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output-file",
        default="/root/rl/verl/data/dsr_sub/pi1_one_ans_reward_random_wrong.parquet",
        help="Output parquet path.",
    )
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed for reproducibility.")
    parser.add_argument("--value-min", type=float, default=-100.0, help="Min random value.")
    parser.add_argument("--value-max", type=float, default=100.0, help="Max random value.")
    parser.add_argument("--decimals", type=int, default=3, help="Decimal digits in generated values.")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if args.value_min >= args.value_max:
        raise ValueError("--value-min must be smaller than --value-max")

    df = pd.read_parquet(input_file)
    if "reward_model" not in df.columns:
        raise KeyError("Column reward_model not found in input parquet")

    rng = random.Random(args.seed)
    new_reward_models: list[dict[str, Any]] = []
    old_ground_truths: list[Any] = []
    new_ground_truths: list[Any] = []

    for row_idx, reward_model in enumerate(df["reward_model"]):
        try:
            new_rm, old_gt, new_gt = _rewrite_ground_truth(
                reward_model=reward_model,
                rng=rng,
                value_min=args.value_min,
                value_max=args.value_max,
                decimals=args.decimals,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to process row {row_idx}") from exc
        new_reward_models.append(new_rm)
        old_ground_truths.append(old_gt)
        new_ground_truths.append(new_gt)

    df_out = df.copy()
    df_out["reward_model"] = new_reward_models

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_file, index=False)

    changed_rows = sum(_canon(a) != _canon(b) for a, b in zip(old_ground_truths, new_ground_truths))
    unique_old = len({_canon(x) for x in old_ground_truths})
    unique_new = len({_canon(x) for x in new_ground_truths})

    print(f"input_file={input_file}")
    print(f"output_file={output_file}")
    print(f"rows={len(df_out)}")
    print(f"seed={args.seed}")
    print(f"value_range=[{args.value_min}, {args.value_max}]")
    print(f"decimals={args.decimals}")
    print(f"changed_rows={changed_rows}")
    print(f"unique_ground_truth_before={unique_old}")
    print(f"unique_ground_truth_after={unique_new}")
    for i in range(min(5, len(df_out))):
        print(f"sample_{i}: old={old_ground_truths[i]!r} -> new={new_ground_truths[i]!r}")


if __name__ == "__main__":
    main()
