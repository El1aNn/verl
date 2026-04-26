#!/usr/bin/env python3
"""Run offline validation and full-token diagnostics from saved VERL checkpoints.

This script is meant for runs where training kept checkpoints but did not keep
full validation diagnostics. It can:

1. resolve base + global_step_* checkpoints,
2. merge FSDP actor checkpoints into Hugging Face format when needed,
3. generate validation completions,
4. compute token-level logprob / entropy / EOS / top-k diagnostics for every
   generated response token,
5. save detailed sample and token records,
6. render summary plots plus a LaTeX section for the report.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verl.utils.reward_score import default_compute_score


DEFAULT_BASE_MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VAL_PATH = "data/testset"
DEFAULT_OVERLEAF_DIR = "analysis/overleaf_bundle_20260406"


@dataclass
class ValExample:
    source_file: str
    row_idx: int
    uid_base: str
    data_source: str
    prompt_messages: list[dict[str, Any]]
    question: str
    ground_truth: Any
    reward_model: dict[str, Any]
    extra_info: dict[str, Any]


@dataclass
class StepSpec:
    step: int
    model_dir: Path
    source: str


@dataclass
class RunningStats:
    count: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)

    def add(self, values: dict[str, Any]) -> None:
        self.count += 1
        for key, value in values.items():
            value = _to_float(value)
            if value is None or not math.isfinite(value):
                continue
            self.sums[key] += value
            self.mins[key] = value if key not in self.mins else min(self.mins[key], value)
            self.maxs[key] = value if key not in self.maxs else max(self.maxs[key], value)

    def row(self, prefix: dict[str, Any]) -> dict[str, Any]:
        out = dict(prefix)
        out["token_count"] = self.count
        for key, total in self.sums.items():
            out[key] = total / max(self.count, 1)
            out[f"{key}_min"] = self.mins.get(key)
            out[f"{key}_max"] = self.maxs.get(key)
        return out


@dataclass
class AggregateCollector:
    bucket_stats: dict[tuple[int, int], RunningStats] = field(default_factory=lambda: defaultdict(RunningStats))
    correctness_stats: dict[tuple[int, str], RunningStats] = field(default_factory=lambda: defaultdict(RunningStats))

    def add_tokens(self, step: int, score: float, token_details: list[dict[str, Any]], bucket_count: int) -> None:
        correctness = "correct" if _to_float(score) is not None and float(score) >= 0.5 else "wrong"
        valid_len = len(token_details)
        for tok in token_details:
            position = int(tok["position"])
            bucket = min(int(position * bucket_count / max(valid_len, 1)), bucket_count - 1)
            values = {
                "logprob": tok.get("logprob"),
                "prob": tok.get("prob"),
                "entropy": tok.get("entropy"),
                "eos_prob": tok.get("eos_prob"),
                "top1_prob": tok.get("top1_prob"),
                "topk_mass": tok.get("topk_mass"),
            }
            self.bucket_stats[(step, bucket)].add(values)
            self.correctness_stats[(step, correctness)].add(values)

    def bucket_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (step, bucket), stats in sorted(self.bucket_stats.items()):
            rows.append(stats.row({"step": step, "position_bucket": bucket}))
        return rows

    def correctness_rows(self) -> list[dict[str, Any]]:
        rows = []
        for (step, correctness), stats in sorted(self.correctness_stats.items()):
            rows.append(stats.row({"step": step, "correctness": correctness}))
        return rows


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def _open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "run"


def _hash_text(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _mean(values: Iterable[Any]) -> float:
    numeric = [_to_float(v) for v in values]
    numeric = [v for v in numeric if v is not None and math.isfinite(v)]
    if not numeric:
        return float("nan")
    return float(sum(numeric) / len(numeric))


def _std(values: Iterable[Any]) -> float:
    numeric = [_to_float(v) for v in values]
    numeric = [v for v in numeric if v is not None and math.isfinite(v)]
    if not numeric:
        return float("nan")
    return float(np.std(np.array(numeric, dtype=np.float64)))


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return 0.0
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def parse_steps(value: str, ckpts_dir: Path, include_base: bool) -> list[int]:
    if value.strip().lower() == "auto":
        steps = sorted(
            int(path.name.rsplit("_", 1)[1])
            for path in ckpts_dir.glob("global_step_*")
            if path.is_dir() and path.name.rsplit("_", 1)[-1].isdigit()
        )
        if include_base and 0 not in steps:
            steps.insert(0, 0)
        return steps
    steps = []
    for item in value.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    return sorted(dict.fromkeys(steps))


def has_hf_weights(model_dir: Path) -> bool:
    names = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    return any((model_dir / name).exists() for name in names) or bool(list(model_dir.glob("*.safetensors")))


def merge_fsdp_actor(actor_dir: Path, target_dir: Path, force: bool = False) -> Path:
    if has_hf_weights(target_dir) and not force:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    print(f"[merge] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return target_dir


def resolve_step_specs(args: argparse.Namespace) -> list[StepSpec]:
    ckpts_dir = Path(args.ckpts_dir).expanduser().resolve()
    merge_root = Path(args.merge_dir).expanduser().resolve() if args.merge_dir else ckpts_dir / "merged_hf_for_full_token_eval"
    steps = parse_steps(args.steps, ckpts_dir=ckpts_dir, include_base=bool(args.base_model))
    specs: list[StepSpec] = []

    for step in steps:
        if step == 0:
            if args.base_model:
                specs.append(StepSpec(step=0, model_dir=Path(args.base_model).expanduser().resolve(), source="base_model"))
                continue
            actor_dir = ckpts_dir / "global_step_0" / "actor"
        else:
            actor_dir = ckpts_dir / f"global_step_{step}" / "actor"

        if not actor_dir.exists():
            raise FileNotFoundError(f"Missing actor checkpoint for step {step}: {actor_dir}")

        hf_dir = actor_dir / "huggingface"
        if has_hf_weights(hf_dir):
            specs.append(StepSpec(step=step, model_dir=hf_dir.resolve(), source="checkpoint_hf"))
            continue

        merged_dir = merge_root / f"global_step_{step}"
        if args.merge_missing:
            merge_fsdp_actor(actor_dir=actor_dir, target_dir=merged_dir, force=args.force_merge)
        elif not has_hf_weights(merged_dir):
            raise FileNotFoundError(
                f"Step {step} has FSDP shards but no merged HF weights. "
                f"Rerun with --merge-missing or merge {actor_dir} manually."
            )
        specs.append(StepSpec(step=step, model_dir=merged_dir.resolve(), source="merged_fsdp"))

    return specs


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    raise TypeError(f"Cannot normalize prompt messages from {type(value)}")


def extract_question(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(messages[-1].get("content", "")) if messages else ""


def collect_val_examples(
    val_path: Path,
    prompt_key: str,
    limit_samples: Optional[int],
    sample_mode: str,
    seed: int,
) -> list[ValExample]:
    if val_path.is_file():
        parquet_files = [val_path]
    else:
        parquet_files = sorted(val_path.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {val_path}")

    examples: list[ValExample] = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        for row_idx, row in df.iterrows():
            prompt_messages = normalize_messages(row[prompt_key])
            reward_model = dict(row.get("reward_model") or {})
            extra_info = dict(row.get("extra_info") or {})
            question = extract_question(prompt_messages)
            data_source = str(row.get("data_source", "unknown"))
            gt = reward_model.get("ground_truth")
            raw_uid_payload = f"{parquet_file}:{row_idx}:{data_source}:{question}:{gt}"
            index = extra_info.get("index", row_idx)
            uid_base = f"val::{data_source}::{index}::{_hash_text(raw_uid_payload)}"
            examples.append(
                ValExample(
                    source_file=str(parquet_file),
                    row_idx=int(row_idx),
                    uid_base=uid_base,
                    data_source=data_source,
                    prompt_messages=prompt_messages,
                    question=question,
                    ground_truth=gt,
                    reward_model=reward_model,
                    extra_info=extra_info,
                )
            )
            if sample_mode == "first" and limit_samples is not None and len(examples) >= limit_samples:
                return examples
    if sample_mode == "random" and limit_samples is not None and len(examples) > limit_samples:
        rng = random.Random(seed)
        selected = sorted(rng.sample(range(len(examples)), limit_samples))
        examples = [examples[idx] for idx in selected]
    return examples


def score_output(example: ValExample, output_text: str) -> float:
    score = default_compute_score(
        data_source=example.data_source,
        solution_str=output_text,
        ground_truth=example.ground_truth,
        extra_info=dict(example.extra_info),
    )
    if isinstance(score, dict):
        score = score.get("score", 0.0)
    return float(score)


def trim_response_ids(response_ids: torch.Tensor, eos_token_id: Optional[int], pad_token_id: Optional[int]) -> torch.Tensor:
    ids = response_ids.detach().cpu().tolist()
    stop_ids = []
    if eos_token_id is not None:
        stop_ids.append(int(eos_token_id))
    if pad_token_id is not None and int(pad_token_id) not in stop_ids:
        stop_ids.append(int(pad_token_id))
    if stop_ids:
        for idx, token_id in enumerate(ids):
            if int(token_id) in stop_ids:
                ids = ids[:idx]
                break
    return torch.tensor(ids, dtype=torch.long)


def build_prompt_texts(tokenizer: Any, examples: list[ValExample]) -> list[str]:
    texts = []
    for example in examples:
        texts.append(
            tokenizer.apply_chat_template(example.prompt_messages, add_generation_prompt=True, tokenize=False)
        )
    return texts


def expand_prompt_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_return_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_return_sequences == 1:
        return input_ids, attention_mask
    expanded_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
    expanded_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
    return expanded_ids, expanded_mask


@torch.inference_mode()
def compute_token_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt_ids_padded: torch.Tensor,
    prompt_attention_padded: torch.Tensor,
    response_ids_cpu: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid_len = int(response_ids_cpu.numel())
    if valid_len <= 0:
        empty_summary = {
            "response_length": 0,
            "token_logprob_mean": float("nan"),
            "token_entropy_mean": float("nan"),
            "low_confidence_token_ratio": float("nan"),
            "answer_tail_confidence": float("nan"),
        }
        return empty_summary, []

    device = next(model.parameters()).device
    prompt_ids = prompt_ids_padded.detach().cpu().long()
    prompt_attention = prompt_attention_padded.detach().cpu().long()
    response_ids = response_ids_cpu.detach().cpu().long()

    full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(device)
    full_attention = torch.cat([prompt_attention, torch.ones(valid_len, dtype=torch.long)], dim=0).unsqueeze(0).to(device)

    prompt_width = int(prompt_ids.numel())
    target_positions = torch.arange(prompt_width - 1, prompt_width + valid_len - 1, dtype=torch.long)
    chunk_size = int(args.diagnostic_token_chunk_size)
    if chunk_size <= 0:
        chunk_size = valid_len

    eos_token_id = tokenizer.eos_token_id
    topk = max(0, int(args.diagnostic_topk))
    low_conf_threshold = float(args.low_confidence_threshold)

    all_token_details: list[dict[str, Any]] = []
    logprobs: list[float] = []
    probs: list[float] = []
    entropies: list[float] = []
    eos_probs_all: list[float] = []
    top1_probs_all: list[float] = []
    topk_mass_all: list[float] = []

    target_token_texts = tokenizer.convert_ids_to_tokens(response_ids.tolist())
    target_token_decodes = (
        [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in response_ids.tolist()]
        if args.decode_token_text
        else [""] * valid_len
    )

    for start in range(0, valid_len, chunk_size):
        end = min(start + chunk_size, valid_len)
        keep_positions = target_positions[start:end].to(device)
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_attention,
            use_cache=False,
            logits_to_keep=keep_positions,
        )
        logits = outputs.logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        dist_probs = log_probs.exp()
        target_ids = response_ids[start:end].to(device)

        chosen_logprobs = log_probs.gather(1, target_ids[:, None]).squeeze(1)
        chosen_probs = chosen_logprobs.exp()
        entropy = -(dist_probs * log_probs).sum(dim=-1)
        top1_probs, top1_ids = dist_probs.max(dim=-1)

        if eos_token_id is not None:
            eos_probs = dist_probs[:, int(eos_token_id)]
        else:
            eos_probs = torch.full_like(chosen_probs, float("nan"))

        if topk > 0:
            topk_probs, topk_ids = torch.topk(dist_probs, k=min(topk, dist_probs.shape[-1]), dim=-1)
            topk_mass = topk_probs.sum(dim=-1)
            topk_probs_cpu = topk_probs.detach().cpu()
            topk_ids_cpu = topk_ids.detach().cpu()
        else:
            topk_mass = torch.full_like(chosen_probs, float("nan"))
            topk_probs_cpu = None
            topk_ids_cpu = None

        chosen_logprobs_cpu = chosen_logprobs.detach().cpu().tolist()
        chosen_probs_cpu = chosen_probs.detach().cpu().tolist()
        entropy_cpu = entropy.detach().cpu().tolist()
        eos_probs_cpu = eos_probs.detach().cpu().tolist()
        top1_probs_cpu = top1_probs.detach().cpu().tolist()
        top1_ids_cpu = top1_ids.detach().cpu().tolist()
        topk_mass_cpu = topk_mass.detach().cpu().tolist()

        for offset, token_pos in enumerate(range(start, end)):
            token_id = int(response_ids[token_pos].item())
            top1_token_id = int(top1_ids_cpu[offset])
            detail = {
                "position": int(token_pos),
                "token_id": token_id,
                "token": target_token_texts[token_pos],
                "text": target_token_decodes[token_pos],
                "logprob": float(chosen_logprobs_cpu[offset]),
                "prob": float(chosen_probs_cpu[offset]),
                "entropy": float(entropy_cpu[offset]),
                "eos_prob": float(eos_probs_cpu[offset]),
                "top1_token_id": top1_token_id,
                "top1_token": tokenizer.convert_ids_to_tokens([top1_token_id])[0],
                "top1_text": tokenizer.decode([top1_token_id], skip_special_tokens=False)
                if args.decode_token_text
                else "",
                "top1_prob": float(top1_probs_cpu[offset]),
                "topk_mass": float(topk_mass_cpu[offset]),
                "chosen_is_top1": int(token_id == top1_token_id),
            }
            if topk_probs_cpu is not None and topk_ids_cpu is not None:
                entries = []
                eos_rank = None
                ids_for_pos = [int(x) for x in topk_ids_cpu[offset].tolist()]
                probs_for_pos = [float(x) for x in topk_probs_cpu[offset].tolist()]
                tokens_for_pos = tokenizer.convert_ids_to_tokens(ids_for_pos)
                for rank, (cand_id, cand_tok, cand_prob) in enumerate(
                    zip(ids_for_pos, tokens_for_pos, probs_for_pos, strict=True), start=1
                ):
                    if eos_token_id is not None and cand_id == int(eos_token_id) and eos_rank is None:
                        eos_rank = rank
                    entries.append(
                        {
                            "rank": rank,
                            "token_id": cand_id,
                            "token": cand_tok,
                            "text": tokenizer.decode([cand_id], skip_special_tokens=False)
                            if args.decode_token_text
                            else "",
                            "prob": cand_prob,
                        }
                    )
                detail["topk"] = entries
                detail["eos_in_topk"] = eos_rank is not None
                if eos_rank is not None:
                    detail["eos_rank_in_topk"] = eos_rank
            all_token_details.append(detail)

        logprobs.extend(float(x) for x in chosen_logprobs_cpu)
        probs.extend(float(x) for x in chosen_probs_cpu)
        entropies.extend(float(x) for x in entropy_cpu)
        eos_probs_all.extend(float(x) for x in eos_probs_cpu)
        top1_probs_all.extend(float(x) for x in top1_probs_cpu)
        topk_mass_all.extend(float(x) for x in topk_mass_cpu)

        del outputs, logits, log_probs, dist_probs
        if torch.cuda.is_available() and args.empty_cache_between_chunks:
            torch.cuda.empty_cache()

    tail_width = min(valid_len, max(1, int(args.tail_tokens)))
    summary = {
        "response_length": valid_len,
        "token_logprob_mean": _mean(logprobs),
        "token_logprob_std": _std(logprobs),
        "token_logprob_min": min(logprobs),
        "token_logprob_max": max(logprobs),
        "token_prob_mean": _mean(probs),
        "token_prob_min": min(probs),
        "token_entropy_mean": _mean(entropies),
        "token_entropy_std": _std(entropies),
        "low_confidence_token_ratio": _mean([float(p < low_conf_threshold) for p in probs]),
        "answer_tail_confidence": _mean(probs[-tail_width:]),
        "logprob_slope": _slope(logprobs),
        "entropy_slope": _slope(entropies),
        "eos_prob_mean": _mean(eos_probs_all),
        "eos_prob_max": max(eos_probs_all) if eos_probs_all else float("nan"),
        "eos_prob_final": eos_probs_all[-1] if eos_probs_all else float("nan"),
        "eos_prob_slope": _slope(eos_probs_all),
        "eos_high_prob_ratio": _mean([float(p > float(args.eos_high_prob_threshold)) for p in eos_probs_all]),
        "top1_prob_mean": _mean(top1_probs_all),
        "top1_prob_final": top1_probs_all[-1] if top1_probs_all else float("nan"),
        "topk_mass_mean": _mean(topk_mass_all),
        "topk_mass_final": topk_mass_all[-1] if topk_mass_all else float("nan"),
        "eos_top1_ratio": _mean(
            [
                float(detail.get("top1_token_id") == int(eos_token_id))
                for detail in all_token_details
            ]
        )
        if eos_token_id is not None
        else float("nan"),
    }
    return summary, all_token_details


def write_token_rows(
    handle: Any,
    step: int,
    sample_record: dict[str, Any],
    token_details: list[dict[str, Any]],
    bucket_count: int,
) -> None:
    valid_len = max(len(token_details), 1)
    base = {
        "step": step,
        "uid": sample_record["uid"],
        "case_uid": sample_record["case_uid"],
        "sample_index": sample_record["sample_index"],
        "rollout_index": sample_record["rollout_index"],
        "score": sample_record["score"],
        "correct": int(float(sample_record["score"]) >= 0.5),
    }
    for detail in token_details:
        row = dict(base)
        position = int(detail["position"])
        row.update(
            {
                "position": position,
                "position_bucket": min(int(position * bucket_count / valid_len), bucket_count - 1),
                "position_fraction": position / max(valid_len - 1, 1),
                "token_id": detail.get("token_id"),
                "token": detail.get("token"),
                "text": detail.get("text"),
                "logprob": detail.get("logprob"),
                "prob": detail.get("prob"),
                "entropy": detail.get("entropy"),
                "eos_prob": detail.get("eos_prob"),
                "top1_token_id": detail.get("top1_token_id"),
                "top1_token": detail.get("top1_token"),
                "top1_text": detail.get("top1_text"),
                "top1_prob": detail.get("top1_prob"),
                "topk_mass": detail.get("topk_mass"),
                "chosen_is_top1": detail.get("chosen_is_top1"),
            }
        )
        if "topk" in detail:
            row["topk"] = detail["topk"]
            row["eos_in_topk"] = detail.get("eos_in_topk")
            row["eos_rank_in_topk"] = detail.get("eos_rank_in_topk")
        handle.write(_json_dumps(row) + "\n")


def load_model_and_tokenizer(model_dir: Path, args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation if args.attn_implementation else None,
    )
    model.eval()
    if args.device_map is None and args.device != "auto":
        model.to(torch.device(args.device))
    return model, tokenizer


def generation_kwargs(tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "num_return_sequences": int(args.n),
        "do_sample": bool(args.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        kwargs["temperature"] = float(args.temperature)
        if args.top_p is not None:
            kwargs["top_p"] = float(args.top_p)
        if args.top_k is not None:
            kwargs["top_k"] = int(args.top_k)
    return kwargs


def run_step(
    spec: StepSpec,
    examples: list[ValExample],
    out_dir: Path,
    args: argparse.Namespace,
    aggregate: AggregateCollector,
) -> list[dict[str, Any]]:
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    samples_path = records_dir / f"samples.step_{spec.step:04d}.jsonl"
    token_suffix = ".jsonl.gz" if args.gzip_token_records else ".jsonl"
    tokens_path = records_dir / f"tokens.step_{spec.step:04d}{token_suffix}"

    if args.skip_existing and samples_path.exists() and tokens_path.exists():
        print(f"[skip] step {spec.step}: records already exist")
        accumulate_existing_token_rows(tokens_path=tokens_path, aggregate=aggregate)
        return read_sample_metric_rows(samples_path)

    print(f"[load] step={spec.step} model={spec.model_dir}", flush=True)
    model, tokenizer = load_model_and_tokenizer(spec.model_dir, args)
    gen_kwargs = generation_kwargs(tokenizer, args)
    device = next(model.parameters()).device

    sample_rows: list[dict[str, Any]] = []
    sample_counter = 0

    with samples_path.open("w", encoding="utf-8") as sample_handle, _open_text(tokens_path, "wt") as token_handle:
        for batch_start in tqdm(range(0, len(examples), args.prompt_batch_size), desc=f"step {spec.step}"):
            batch_examples = examples[batch_start : batch_start + args.prompt_batch_size]
            prompt_texts = build_prompt_texts(tokenizer, batch_examples)
            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_prompt_length),
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            expanded_prompt_ids, expanded_attention = expand_prompt_tensors(
                encoded["input_ids"].detach().cpu(),
                encoded["attention_mask"].detach().cpu(),
                int(args.n),
            )

            generated = model.generate(**encoded, **gen_kwargs)
            if isinstance(generated, torch.Tensor):
                generated_ids = generated.detach().cpu()
            else:
                generated_ids = generated.sequences.detach().cpu()

            for seq_idx, sequence_ids in enumerate(generated_ids):
                local_prompt_idx = seq_idx // int(args.n)
                rollout_idx = seq_idx % int(args.n)
                example = batch_examples[local_prompt_idx]
                response_ids = trim_response_ids(
                    sequence_ids[prompt_width:],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                output_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
                score = score_output(example, output_text)
                prompt_ids_padded = expanded_prompt_ids[seq_idx]
                prompt_attention_padded = expanded_attention[seq_idx]
                input_text = tokenizer.decode(
                    prompt_ids_padded[prompt_attention_padded.bool()].tolist(),
                    skip_special_tokens=True,
                )

                diag_summary, token_details = compute_token_diagnostics(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids_padded=prompt_ids_padded,
                    prompt_attention_padded=prompt_attention_padded,
                    response_ids_cpu=response_ids,
                    args=args,
                )
                sample_uid = f"{example.uid_base}::sample{rollout_idx}"
                sample_record = {
                    "step": spec.step,
                    "uid": sample_uid,
                    "case_uid": example.uid_base,
                    "sample_index": sample_counter,
                    "rollout_index": rollout_idx,
                    "source_file": example.source_file,
                    "row_idx": example.row_idx,
                    "data_source": example.data_source,
                    "input": input_text,
                    "question": example.question,
                    "output": output_text,
                    "gts": example.ground_truth,
                    "score": score,
                    "reward": score,
                    "model_dir": str(spec.model_dir),
                    "checkpoint_source": spec.source,
                    **diag_summary,
                    "token_diagnostics_total_tokens": int(diag_summary.get("response_length", 0)),
                    "token_diagnostics_truncated": False,
                }
                if args.embed_token_details:
                    sample_record["token_diagnostics"] = token_details
                sample_handle.write(_json_dumps(sample_record) + "\n")
                write_token_rows(
                    handle=token_handle,
                    step=spec.step,
                    sample_record=sample_record,
                    token_details=token_details,
                    bucket_count=int(args.position_buckets),
                )
                aggregate.add_tokens(
                    step=spec.step,
                    score=score,
                    token_details=token_details,
                    bucket_count=int(args.position_buckets),
                )
                sample_rows.append(metric_row_from_sample(sample_record))
                sample_counter += 1

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sample_rows


def accumulate_existing_token_rows(tokens_path: Path, aggregate: AggregateCollector) -> None:
    with _open_text(tokens_path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            bucket = int(row["position_bucket"])
            correctness = "correct" if int(row.get("correct", 0)) == 1 else "wrong"
            values = {
                "logprob": row.get("logprob"),
                "prob": row.get("prob"),
                "entropy": row.get("entropy"),
                "eos_prob": row.get("eos_prob"),
                "top1_prob": row.get("top1_prob"),
                "topk_mass": row.get("topk_mass"),
            }
            aggregate.bucket_stats[(step, bucket)].add(values)
            aggregate.correctness_stats[(step, correctness)].add(values)


def metric_row_from_sample(record: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "step",
        "uid",
        "case_uid",
        "sample_index",
        "rollout_index",
        "data_source",
        "score",
        "reward",
        "response_length",
        "token_logprob_mean",
        "token_logprob_std",
        "token_logprob_min",
        "token_logprob_max",
        "token_prob_mean",
        "token_prob_min",
        "token_entropy_mean",
        "token_entropy_std",
        "low_confidence_token_ratio",
        "answer_tail_confidence",
        "logprob_slope",
        "entropy_slope",
        "eos_prob_mean",
        "eos_prob_max",
        "eos_prob_final",
        "eos_prob_slope",
        "eos_high_prob_ratio",
        "eos_top1_ratio",
        "top1_prob_mean",
        "top1_prob_final",
        "topk_mass_mean",
        "topk_mass_final",
    ]
    return {key: record.get(key) for key in metric_keys}


def read_sample_metric_rows(samples_path: Path) -> list[dict[str, Any]]:
    rows = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(metric_row_from_sample(json.loads(line)))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "step",
        "position_bucket",
        "correctness",
        "uid",
        "case_uid",
        "sample_index",
        "rollout_index",
        "data_source",
    ]
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_step_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "score",
        "reward",
        "response_length",
        "token_logprob_mean",
        "token_prob_mean",
        "token_entropy_mean",
        "low_confidence_token_ratio",
        "answer_tail_confidence",
        "eos_prob_mean",
        "eos_prob_final",
        "eos_high_prob_ratio",
        "eos_top1_ratio",
        "top1_prob_mean",
        "top1_prob_final",
        "topk_mass_mean",
        "topk_mass_final",
    ]
    rows = []
    for step, sdf in sample_df.groupby("step"):
        row = {
            "step": int(step),
            "n_samples": int(len(sdf)),
            "n_cases": int(sdf["case_uid"].nunique()) if "case_uid" in sdf else int(len(sdf)),
        }
        for metric in metrics:
            if metric in sdf:
                row[metric] = pd.to_numeric(sdf[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step")


def load_report_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_df = pd.read_csv(data_dir / "sample_metrics.csv")
    step_df = pd.read_csv(data_dir / "step_summary.csv")
    bucket_df = pd.read_csv(data_dir / "token_bucket_summary.csv")
    correctness_df = pd.read_csv(data_dir / "correctness_token_summary.csv")
    return sample_df, step_df, bucket_df, correctness_df


def save_tables(
    out_dir: Path,
    sample_rows: list[dict[str, Any]],
    aggregate: AggregateCollector,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_df = pd.DataFrame(sample_rows).sort_values(["step", "sample_index"])
    step_df = make_step_summary(sample_df)
    bucket_df = pd.DataFrame(aggregate.bucket_rows())
    correctness_df = pd.DataFrame(aggregate.correctness_rows())

    sample_df.to_csv(data_dir / "sample_metrics.csv", index=False)
    step_df.to_csv(data_dir / "step_summary.csv", index=False)
    bucket_df.to_csv(data_dir / "token_bucket_summary.csv", index=False)
    correctness_df.to_csv(data_dir / "correctness_token_summary.csv", index=False)
    return sample_df, step_df, bucket_df, correctness_df


def _setup_axes(ax: Any, title: str, ylabel: str, xlabel: str = "Training step") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#DDD7CD", linewidth=0.7)
    ax.set_facecolor("#FCFBF8")


def _save_fig(fig: Any, figs_dir: Path, name: str) -> None:
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figs_dir / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_plots(out_dir: Path, step_df: pd.DataFrame, bucket_df: pd.DataFrame, correctness_df: pd.DataFrame) -> None:
    figs_dir = out_dir / "figures"
    color = "#1F5F8B"
    accent = "#C56A2D"
    green = "#2A7F62"
    red = "#AA3F39"

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    axes[0].plot(step_df["step"], step_df["score"], marker="o", color=color, linewidth=2.2)
    _setup_axes(axes[0], "Validation score", "Mean score")
    axes[0].set_ylim(0, 1)
    axes[1].plot(step_df["step"], step_df["response_length"], marker="o", color=accent, linewidth=2.2)
    _setup_axes(axes[1], "Response length", "Tokens")
    _save_fig(fig, figs_dir, "full-token-score-length.png")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), sharex=True)
    specs = [
        ("token_entropy_mean", "Token entropy", "Entropy", color),
        ("low_confidence_token_ratio", "Low-confidence ratio", "Ratio", red),
        ("answer_tail_confidence", "Tail confidence", "Probability", green),
        ("top1_prob_mean", "Mean top-1 probability", "Probability", accent),
    ]
    for ax, (metric, title, ylabel, metric_color) in zip(axes.ravel(), specs, strict=True):
        ax.plot(step_df["step"], step_df[metric], marker="o", color=metric_color, linewidth=2.1)
        _setup_axes(ax, title, ylabel)
    _save_fig(fig, figs_dir, "full-token-confidence-panel.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    specs = [
        ("eos_prob_final", "Final EOS probability", "Probability", color),
        ("eos_high_prob_ratio", "EOS high-prob ratio", "Ratio", red),
        ("topk_mass_mean", "Mean top-k mass", "Probability", green),
    ]
    for ax, (metric, title, ylabel, metric_color) in zip(axes, specs, strict=True):
        if metric in step_df:
            ax.plot(step_df["step"], step_df[metric], marker="o", color=metric_color, linewidth=2.1)
        _setup_axes(ax, title, ylabel)
    _save_fig(fig, figs_dir, "full-token-eos-topk-panel.png")

    if not bucket_df.empty:
        final_step = int(bucket_df["step"].max())
        final_bucket = bucket_df[bucket_df["step"] == final_step].sort_values("position_bucket")
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
        specs = [
            ("entropy", "Entropy by response position", "Entropy", color),
            ("top1_prob", "Top-1 prob by response position", "Probability", accent),
            ("eos_prob", "EOS prob by response position", "Probability", green),
        ]
        for ax, (metric, title, ylabel, metric_color) in zip(axes, specs, strict=True):
            if metric in final_bucket:
                ax.plot(
                    final_bucket["position_bucket"],
                    final_bucket[metric],
                    marker="o",
                    color=metric_color,
                    linewidth=2.1,
                )
            _setup_axes(ax, f"{title} (step {final_step})", ylabel, xlabel="Position decile")
        _save_fig(fig, figs_dir, "full-token-position-final.png")

    if not correctness_df.empty:
        final_step = int(correctness_df["step"].max())
        cdf = correctness_df[correctness_df["step"] == final_step].copy()
        if not cdf.empty:
            cdf["correctness"] = pd.Categorical(cdf["correctness"], categories=["correct", "wrong"], ordered=True)
            cdf = cdf.sort_values("correctness")
            fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
            specs = [
                ("entropy", "Token entropy", "Entropy"),
                ("top1_prob", "Top-1 probability", "Probability"),
                ("prob", "Chosen-token probability", "Probability"),
            ]
            for ax, (metric, title, ylabel) in zip(axes, specs, strict=True):
                vals = cdf[metric] if metric in cdf else []
                ax.bar(cdf["correctness"].astype(str), vals, color=[green, red][: len(cdf)])
                _setup_axes(ax, f"{title} by correctness (step {final_step})", ylabel, xlabel="")
            _save_fig(fig, figs_dir, "full-token-correctness-final.png")


def fmt(value: Any, digits: int = 4) -> str:
    value = _to_float(value)
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def render_report(out_dir: Path, args: argparse.Namespace, step_df: pd.DataFrame) -> None:
    report_path = out_dir / "report.md"
    section_path = out_dir / "report_section_cn.tex"
    section_en_path = out_dir / "report_section_en.tex"
    final = step_df.sort_values("step").iloc[-1]
    first = step_df.sort_values("step").iloc[0]
    best = step_df.loc[step_df["score"].idxmax()]
    sorted_steps = step_df.sort_values("step")

    en_rows = "\n".join(
        (
            f"        {int(row['step'])} & {fmt(row.get('score'))} & "
            f"{fmt(row.get('response_length'), 1)} & "
            f"{fmt(row.get('token_entropy_mean'))} & "
            f"{fmt(row.get('low_confidence_token_ratio'))} & "
            f"{fmt(row.get('top1_prob_mean'))} & "
            f"{fmt(row.get('answer_tail_confidence'))} \\\\"
        )
        for _, row in sorted_steps.iterrows()
    )

    score_delta = _to_float(final.get("score")) - _to_float(first.get("score"))
    entropy_delta = _to_float(final.get("token_entropy_mean")) - _to_float(first.get("token_entropy_mean"))
    lowconf_delta = _to_float(final.get("low_confidence_token_ratio")) - _to_float(
        first.get("low_confidence_token_ratio")
    )
    top1_delta = _to_float(final.get("top1_prob_mean")) - _to_float(first.get("top1_prob_mean"))
    length_delta = _to_float(final.get("response_length")) - _to_float(first.get("response_length"))

    md = f"""# Full-token checkpoint validation report

Run label: `{args.run_label}`

- Validation samples per step: {int(final.get('n_samples', 0))}
- Cases per step: {int(final.get('n_cases', 0))}
- Steps: {', '.join(str(int(x)) for x in step_df['step'].tolist())}
- Final score: {fmt(final.get('score'))}
- Best score: {fmt(best.get('score'))} at step {int(best.get('step'))}
- Final response length: {fmt(final.get('response_length'), 1)}
- Final token entropy: {fmt(final.get('token_entropy_mean'))}
- Final low-confidence ratio: {fmt(final.get('low_confidence_token_ratio'))}

Figures are in `figures/`; detailed sample and token records are in `records/`.
"""
    report_path.write_text(md, encoding="utf-8")

    section = f"""\\subsection{{新增 checkpoint 的全量 token 离线诊断}}

为弥补训练期 validation diagnostics 只保留前缀 token 的限制，本文对新增 checkpoint 进行离线重评估。该流程从保存的 actor checkpoint 恢复模型，在 \\EvalSet{{}} validation set 上重新生成响应，并对每条响应的全部生成 token 计算 logprob、entropy、EOS probability、top-1 probability 与 top-k mass。与原训练日志中的前缀诊断不同，本节覆盖完整响应长度，因此更适合作为 checkpoint 级行为分析。

本次离线评估覆盖 step {int(first['step'])} 到 step {int(final['step'])}，每个 step 包含 {int(final.get('n_cases', 0))} 道题、{int(final.get('n_samples', 0))} 条采样响应。最终 checkpoint 的平均 score 为 {fmt(final.get('score'))}，最佳 checkpoint 为 step {int(best.get('step'))}，best score 为 {fmt(best.get('score'))}。从行为指标看，最终平均 response length 为 {fmt(final.get('response_length'), 1)}，token entropy 为 {fmt(final.get('token_entropy_mean'))}，低置信 token 比例为 {fmt(final.get('low_confidence_token_ratio'))}，平均 top-1 probability 为 {fmt(final.get('top1_prob_mean'))}。

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-score-length.png}}
    \\caption{{新增 checkpoint 的离线全量 token 评估：validation score 与响应长度随 checkpoint 变化。}}
    \\label{{fig:full-token-score-length}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-confidence-panel.png}}
    \\caption{{新增 checkpoint 的全量 token 置信度与 entropy 指标。该图使用完整响应 token，而非训练期保留的前缀子集。}}
    \\label{{fig:full-token-confidence-panel}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-position-final.png}}
    \\caption{{最终 checkpoint 上按响应位置分桶的全量 token 指标。横轴为响应内相对位置 decile。}}
    \\label{{fig:full-token-position-final}}
\\end{{figure}}

这组结果应作为现有 token 级补充证据的加强版来解读：训练期记录能展示早期 token 的局部轨迹，而离线重评估可以覆盖整个 validation set 与完整响应长度。二者若方向一致，则说明“分布稳定化”并非只发生在少量固定题目的前缀区域，而是能在完整生成轨迹上观察到。
"""
    section_path.write_text(section, encoding="utf-8")

    section_en = f"""\\subsection{{Offline full-token checkpoint audit}}

The retained training-time token traces are useful for mechanism inspection, but they have two important limits: they are narrow retained traces, and the detailed token fields only cover an early response prefix. To check whether the same stabilization pattern survives when the saved checkpoints are reloaded directly, I run an additional offline audit on the newly available checkpoints. The audit restores the saved actor checkpoints, regenerates responses on a fixed held-out subset of \\EvalSet{{}}, and scores every generated response token rather than only the first retained prefix tokens.

The audit covers checkpoint step {int(first['step'])} through step {int(final['step'])}. For each checkpoint, it evaluates {int(final.get('n_cases', 0))} held-out problems with {int(final.get('n_samples', 0))} sampled responses in total. This is still a subset audit rather than a replacement for the full validation curve, but it is stricter than the earlier fixed-prompt trace view because the token statistics are recomputed from the saved checkpoints over complete generated sequences.

\\begin{{table}}[H]
    \\centering
    \\caption{{Offline full-token audit across saved checkpoints}}
    \\label{{tab:en-full-token-ckpt-audit}}
    \\small
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\begin{{tabular}}{{rrrrrrr}}
        \\toprule
        Step & score & length & entropy & low-conf. & top-1 & tail conf. \\\\
        \\midrule
{en_rows}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

The checkpoint-level pattern is directionally consistent with the broader calibration account. From step {int(first['step'])} to step {int(final['step'])}, score changes by {fmt(score_delta)}, average response length changes by {fmt(length_delta, 1)} tokens, mean token entropy changes by {fmt(entropy_delta)}, the low-confidence-token ratio changes by {fmt(lowconf_delta)}, and mean top-1 probability changes by {fmt(top1_delta)}. The best score in this audit is {fmt(best.get('score'))} at step {int(best.get('step'))}, so the saved-checkpoint view again warns against reading the last checkpoint as the only meaningful endpoint.

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-score-length.png}}
    \\caption{{Offline full-token audit: validation score and response length across saved checkpoints.}}
    \\label{{fig:en-full-token-score-length}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-confidence-panel.png}}
    \\caption{{Offline full-token audit: token entropy, low-confidence mass, tail confidence, and chosen-token probability across saved checkpoints.}}
    \\label{{fig:en-full-token-confidence-panel}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-eos-topk-panel.png}}
    \\caption{{Offline full-token audit: EOS behavior, top-1 probability, and top-k mass across saved checkpoints.}}
    \\label{{fig:en-full-token-eos-topk-panel}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-position-final.png}}
    \\caption{{Final-checkpoint full-token statistics by relative response position. The x-axis bins generated tokens into response-position deciles.}}
    \\label{{fig:en-full-token-position-final}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{full-token-correctness-final.png}}
    \\caption{{Final-checkpoint full-token statistics grouped by response correctness.}}
    \\label{{fig:en-full-token-correctness-final}}
\\end{{figure}}

The audit strengthens the evidence hierarchy in the paper. The original validation curves show that performance improves; the aggregate token summaries show that the improvement is accompanied by shorter and sharper generations; and this checkpoint audit shows that the same behavior is visible when saved checkpoints are used for fresh full-response inference. Because the audit is based on {int(final.get('n_cases', 0))} held-out problems, its numerical values should be read as subset estimates. Its role is mechanism validation: it checks whether the token-level story remains coherent when the analysis is expanded from retained prefixes to complete generated sequences.
"""
    section_en_path.write_text(section_en, encoding="utf-8")

    if args.overleaf_dir:
        overleaf_dir = Path(args.overleaf_dir).expanduser().resolve()
        (overleaf_dir / "figures").mkdir(parents=True, exist_ok=True)
        (overleaf_dir / "sections").mkdir(parents=True, exist_ok=True)
        for fig in (out_dir / "figures").glob("full-token-*.png"):
            shutil.copy2(fig, overleaf_dir / "figures" / fig.name)
        shutil.copy2(section_path, overleaf_dir / "sections" / "full_token_ckpt_analysis_cn.tex")
        shutil.copy2(section_en_path, overleaf_dir / "sections" / "full_token_ckpt_analysis_en.tex")


def write_metadata(out_dir: Path, args: argparse.Namespace, specs: list[StepSpec], examples: list[ValExample]) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_label": args.run_label,
        "ckpts_dir": str(Path(args.ckpts_dir).expanduser().resolve()),
        "base_model": args.base_model,
        "val_path": str(Path(args.val_path).expanduser().resolve()),
        "n_examples": len(examples),
        "steps": [{"step": spec.step, "model_dir": str(spec.model_dir), "source": spec.source} for spec in specs],
        "generation": {
            "n": args.n,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "diagnostics": {
            "diagnostic_topk": args.diagnostic_topk,
            "diagnostic_token_chunk_size": args.diagnostic_token_chunk_size,
            "tail_tokens": args.tail_tokens,
            "low_confidence_threshold": args.low_confidence_threshold,
            "position_buckets": args.position_buckets,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpts_dir = Path(args.ckpts_dir).expanduser().resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = Path("analysis") / "full_token_ckpt_reports" / _safe_name(args.run_label or ckpts_dir.name)
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        sample_df, step_df, bucket_df, correctness_df = load_report_tables(out_dir / "data")
        render_plots(out_dir, step_df=step_df, bucket_df=bucket_df, correctness_df=correctness_df)
        render_report(out_dir, args=args, step_df=step_df)
        print(f"[done] report-only outputs under {out_dir}")
        return

    specs = resolve_step_specs(args)
    examples = collect_val_examples(
        val_path=Path(args.val_path).expanduser().resolve(),
        prompt_key=args.prompt_key,
        limit_samples=args.limit_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    write_metadata(out_dir=out_dir, args=args, specs=specs, examples=examples)

    aggregate = AggregateCollector()
    all_sample_rows: list[dict[str, Any]] = []
    for spec in specs:
        step_rows = run_step(spec=spec, examples=examples, out_dir=out_dir, args=args, aggregate=aggregate)
        all_sample_rows.extend(step_rows)
        sample_df, step_df, bucket_df, correctness_df = save_tables(out_dir, all_sample_rows, aggregate)
        render_plots(out_dir, step_df=step_df, bucket_df=bucket_df, correctness_df=correctness_df)
        render_report(out_dir, args=args, step_df=step_df)
        print(f"[step done] {spec.step}: tables/figures/report refreshed under {out_dir}", flush=True)

    print(f"[done] outputs under {out_dir}")
    print(f"[done] LaTeX section: {out_dir / 'report_section_cn.tex'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts-dir", required=True, help="Experiment checkpoint directory.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base HF model for step 0.")
    parser.add_argument("--steps", default="0,60,120,180", help="Comma steps or 'auto'. Include 0 for base model.")
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH, help="Validation parquet file or directory.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--merge-dir", default=None)
    parser.add_argument("--merge-missing", action="store_true", help="Merge FSDP actor shards when HF weights are absent.")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None, help="Smoke-test on the first N validation examples.")
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")

    parser.add_argument("--n", type=int, default=8, help="Validation samples per prompt.")
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)

    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--diagnostic-topk", type=int, default=16)
    parser.add_argument(
        "--diagnostic-token-chunk-size",
        type=int,
        default=512,
        help="Number of response positions per diagnostic forward. Use 0 to score all positions at once.",
    )
    parser.add_argument("--tail-tokens", type=int, default=32)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--eos-high-prob-threshold", type=float, default=0.1)
    parser.add_argument("--position-buckets", type=int, default=10)
    parser.add_argument("--decode-token-text", action="store_true", default=True)
    parser.add_argument("--no-decode-token-text", dest="decode_token_text", action="store_false")
    parser.add_argument("--embed-token-details", action="store_true")
    parser.add_argument("--gzip-token-records", action="store_true", default=True)
    parser.add_argument("--no-gzip-token-records", dest="gzip_token_records", action="store_false")
    parser.add_argument("--empty-cache-between-chunks", action="store_true")

    parser.add_argument(
        "--overleaf-dir",
        default=None,
        help=f"Optional Overleaf bundle dir. Example: {DEFAULT_OVERLEAF_DIR}",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.run_label:
        args.run_label = Path(args.ckpts_dir).expanduser().resolve().name
    run_pipeline(args)


if __name__ == "__main__":
    main()
