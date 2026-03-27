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

import json

from scripts.validation_viz_report import RunSpec, generate_report


def _write_jsonl(path, entries):
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_generate_validation_viz_report_from_synthetic_data(tmp_path):
    validation_dir = tmp_path / "validation"
    report_dir = tmp_path / "report"
    metrics_file = tmp_path / "metrics.jsonl"
    validation_dir.mkdir()

    _write_jsonl(
        validation_dir / "1.jsonl",
        [
            {
                "step": 1,
                "uid": "sample-1",
                "input": "Question 1",
                "output": "Wait, let's try again.",
                "score": 0.4,
                "reward": 0.4,
                "acc": 0.0,
                "response_length": 3,
                "token_entropy_mean": 0.9,
                "low_confidence_token_ratio": 0.5,
                "token_diagnostics_total_tokens": 3,
                "token_diagnostics": [
                    {
                        "text": "wait",
                        "token": "wait",
                        "prob": 0.08,
                        "entropy": 1.4,
                        "topk": [
                            {"text": "wait", "token": "wait", "prob": 0.08},
                            {"text": "cost", "token": "cost", "prob": 0.03},
                        ],
                    },
                    {
                        "text": "cost",
                        "token": "cost",
                        "prob": 0.02,
                        "entropy": 1.7,
                        "topk": [
                            {"text": "cost", "token": "cost", "prob": 0.02},
                            {"text": "however", "token": "however", "prob": 0.07},
                        ],
                    },
                    {
                        "text": "however",
                        "token": "however",
                        "prob": 0.15,
                        "entropy": 0.8,
                        "topk": [{"text": "however", "token": "however", "prob": 0.09}],
                    },
                ],
            },
            {
                "step": 1,
                "uid": "sample-2",
                "input": "Question 2",
                "output": "Fine.",
                "score": 0.2,
                "reward": 0.2,
                "acc": 0.0,
                "response_length": 2,
                "token_entropy_mean": 1.1,
                "low_confidence_token_ratio": 0.7,
                "token_diagnostics_total_tokens": 2,
                "token_diagnostics": [
                    {
                        "text": "fine",
                        "token": "fine",
                        "prob": 0.04,
                        "entropy": 1.5,
                        "topk": [{"text": "fine", "token": "fine", "prob": 0.04}],
                    },
                    {
                        "text": "but",
                        "token": "but",
                        "prob": 0.12,
                        "entropy": 0.7,
                        "topk": [{"text": "but", "token": "but", "prob": 0.06}],
                    },
                ],
            },
        ],
    )
    _write_jsonl(
        validation_dir / "10.jsonl",
        [
            {
                "step": 10,
                "uid": "sample-3",
                "input": "Question 3",
                "output": "Perhaps the answer is 3.",
                "score": 0.9,
                "reward": 0.9,
                "acc": 1.0,
                "response_length": 3,
                "token_entropy_mean": 0.4,
                "low_confidence_token_ratio": 0.2,
                "token_diagnostics_total_tokens": 3,
                "token_diagnostics": [
                    {
                        "text": "perhaps",
                        "token": "perhaps",
                        "prob": 0.11,
                        "entropy": 0.8,
                        "topk": [
                            {"text": "perhaps", "token": "perhaps", "prob": 0.09},
                            {"text": "balanced", "token": "balanced", "prob": 0.01},
                        ],
                    },
                    {
                        "text": "however",
                        "token": "however",
                        "prob": 0.13,
                        "entropy": 0.6,
                        "topk": [{"text": "however", "token": "however", "prob": 0.08}],
                    },
                    {
                        "text": "wait",
                        "token": "wait",
                        "prob": 0.09,
                        "entropy": 1.0,
                        "topk": [{"text": "wait", "token": "wait", "prob": 0.09}],
                    },
                ],
            }
        ],
    )
    _write_jsonl(
        metrics_file,
        [
            {"step": 1, "data": {"actor/entropy": 1.25}},
            {"step": 10, "data": {"actor/entropy": 0.55}},
        ],
    )

    summary = generate_report(
        run_specs=[RunSpec(label="smoke", validation_dir=validation_dir, metrics_file=metrics_file)],
        output_dir=report_dir,
        token_groups={
            "reasoning_sparks": ["but", "wait", "perhaps", "however"],
            "noise": ["cost", "fine", "balanced"],
        },
        selected_steps=[1, 10],
    )

    assert summary["selected_steps"] == [1, 10]
    assert summary["runs"][0]["label"] == "smoke"
    assert summary["runs"][0]["primary_metric_key"] == "acc"
    step_one = summary["runs"][0]["steps"][0]
    step_ten = summary["runs"][0]["steps"][1]
    assert step_one["sampled_group_count"]["reasoning_sparks"] == 3
    assert step_one["sampled_group_count"]["noise"] == 2
    assert step_ten["sampled_group_count"]["reasoning_sparks"] == 3
    assert step_ten["topk_low_prob_group_mean"]["reasoning_sparks"] == 0.08666666666666667
    assert (report_dir / "index.html").exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "training_metric.svg").exists()
    assert (report_dir / "training_entropy.svg").exists()
    assert (report_dir / "sampled_probability_distributions.svg").exists()
    assert (report_dir / "probability_entropy_scatter.svg").exists()
    assert (report_dir / "low_prob_topk_trends.svg").exists()
    assert (report_dir / "token_traces.html").exists()
    token_trace_html = (report_dir / "token_traces.html").read_text(encoding="utf-8")
    assert "sample-3" in token_trace_html
    assert "Step 10 sampled probability" in token_trace_html
