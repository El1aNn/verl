# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Tests for lightweight validation token diagnostics helpers in RayPPOTrainer.
"""

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _build_trainer(validation_diagnostics: dict) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {"validation_diagnostics": validation_diagnostics}})
    return trainer


def _build_batch(uids: list[str]) -> DataProto:
    batch_size = len(uids)
    return DataProto.from_dict(
        tensors={"input_ids": torch.zeros((batch_size, 1), dtype=torch.int64)},
        non_tensors={"uid": np.asarray(uids, dtype=object)},
    )


class TestValidationDiagSampling(unittest.TestCase):
    def test_select_validation_diag_indices_sequential_budget(self):
        trainer = _build_trainer({"enabled": True, "sampling_strategy": "sequential"})
        batch = _build_batch(["u0", "u1", "u2", "u3"])

        self.assertEqual(trainer._select_validation_diag_indices(batch, remaining_budget=2), [0, 1])

    def test_select_validation_diag_indices_hash_is_deterministic(self):
        sample_seed = 17
        sample_rate = 0.8
        remaining_budget = 3
        trainer = _build_trainer(
            {
                "enabled": True,
                "sampling_strategy": "hash",
                "sample_rate": sample_rate,
                "sample_seed": sample_seed,
            }
        )
        uids = ["u0", "u1", "u2", "u3", "u4", "u5"]
        batch = _build_batch(uids)

        expected_pairs = []
        for idx, uid in enumerate(uids):
            score = RayPPOTrainer._stable_unit_interval_from_key(uid, seed=sample_seed)
            if score < sample_rate:
                expected_pairs.append((idx, score))
        expected_pairs = sorted(expected_pairs, key=lambda item: (item[1], item[0]))[:remaining_budget]
        expected = sorted(idx for idx, _ in expected_pairs)

        first = trainer._select_validation_diag_indices(batch, remaining_budget=remaining_budget)
        second = trainer._select_validation_diag_indices(batch, remaining_budget=remaining_budget)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_select_validation_diag_indices_hash_zero_sample_rate(self):
        trainer = _build_trainer(
            {
                "enabled": True,
                "sampling_strategy": "hash",
                "sample_rate": 0.0,
                "sample_seed": 3,
            }
        )
        batch = _build_batch(["u0", "u1", "u2"])

        self.assertEqual(trainer._select_validation_diag_indices(batch, remaining_budget=None), [])


class TestValidationTokenAggregation(unittest.TestCase):
    def test_finalize_validation_token_aggregates(self):
        trainer = _build_trainer(
            {
                "enabled": True,
                "aggregate_token_metrics": True,
                "aggregate_position_buckets": 2,
            }
        )
        aggregate_states = {}
        aggregate_payloads = [
            {
                "response_length": 4,
                "token_count": 4,
                "sum_logprob": -2.0,
                "sum_prob": 2.0,
                "sum_entropy": 0.8,
                "low_conf_token_count": 1,
                "sum_abs_logprob_delta_from_rollout": 0.4,
                "sum_eos_prob": 0.6,
                "eos_high_token_count": 1,
                "sum_top1_prob": 2.8,
                "sum_topk_mass": 3.4,
                "chosen_is_top1_count": 3,
                "eos_in_topk_count": 2,
                "eos_top1_count": 1,
                "bucket_stats": [
                    {
                        "count": 2,
                        "sum_logprob": -1.0,
                        "sum_entropy": 0.4,
                        "sum_eos_prob": 0.2,
                        "sum_top1_prob": 1.8,
                    },
                    {
                        "count": 2,
                        "sum_logprob": -1.0,
                        "sum_entropy": 0.4,
                        "sum_eos_prob": 0.4,
                        "sum_top1_prob": 1.0,
                    },
                ],
            },
            {
                "response_length": 2,
                "token_count": 2,
                "sum_logprob": -1.0,
                "sum_prob": 1.0,
                "sum_entropy": 0.6,
                "low_conf_token_count": 0,
                "sum_abs_logprob_delta_from_rollout": None,
                "sum_eos_prob": None,
                "eos_high_token_count": None,
                "sum_top1_prob": None,
                "sum_topk_mass": None,
                "chosen_is_top1_count": None,
                "eos_in_topk_count": None,
                "eos_top1_count": None,
                "bucket_stats": [
                    {
                        "count": 1,
                        "sum_logprob": -0.4,
                        "sum_entropy": 0.2,
                        "sum_eos_prob": None,
                        "sum_top1_prob": None,
                    },
                    {
                        "count": 1,
                        "sum_logprob": -0.6,
                        "sum_entropy": 0.4,
                        "sum_eos_prob": None,
                        "sum_top1_prob": None,
                    },
                ],
            },
        ]
        trainer._accumulate_validation_token_aggregates(
            aggregate_states=aggregate_states,
            data_sources=np.asarray(["math", "code"], dtype=object),
            aggregate_payloads=aggregate_payloads,
        )

        metrics = trainer._finalize_validation_token_aggregates(
            aggregate_states=aggregate_states,
            total_counts={"all": 10, "math": 6, "code": 4},
        )

        self.assertAlmostEqual(metrics["val-token/all/summary/analyzed_samples"], 2.0)
        self.assertAlmostEqual(metrics["val-token/all/summary/analyzed_sample_fraction"], 0.2)
        self.assertAlmostEqual(metrics["val-token/all/summary/analyzed_tokens"], 6.0)
        self.assertAlmostEqual(metrics["val-token/all/summary/response_length_mean"], 3.0)
        self.assertAlmostEqual(metrics["val-token/all/summary/token_logprob_mean"], -0.5)
        self.assertAlmostEqual(metrics["val-token/all/summary/token_prob_mean"], 0.5)
        self.assertAlmostEqual(metrics["val-token/all/summary/token_entropy_mean"], 1.4 / 6.0)
        self.assertAlmostEqual(metrics["val-token/all/summary/low_confidence_token_ratio"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["val-token/all/summary/logprob_delta_abs_mean"], 0.1)
        self.assertAlmostEqual(metrics["val-token/all/summary/eos_prob_mean"], 0.15)
        self.assertAlmostEqual(metrics["val-token/all/summary/eos_high_prob_ratio"], 0.25)
        self.assertAlmostEqual(metrics["val-token/all/summary/top1_prob_mean"], 0.7)
        self.assertAlmostEqual(metrics["val-token/all/summary/topk_mass_mean"], 0.85)
        self.assertAlmostEqual(metrics["val-token/all/summary/chosen_is_top1_ratio"], 0.75)
        self.assertAlmostEqual(metrics["val-token/all/summary/eos_in_topk_ratio"], 0.5)
        self.assertAlmostEqual(metrics["val-token/all/summary/eos_top1_ratio"], 0.25)
        self.assertAlmostEqual(metrics["val-token/all/bucket_00/token_fraction"], 0.5)
        self.assertAlmostEqual(metrics["val-token/all/bucket_00/token_logprob_mean"], -1.4 / 3.0)
        self.assertAlmostEqual(metrics["val-token/all/bucket_00/token_entropy_mean"], 0.2)
        self.assertAlmostEqual(metrics["val-token/all/bucket_00/eos_prob_mean"], 0.1)
        self.assertAlmostEqual(metrics["val-token/all/bucket_00/top1_prob_mean"], 0.9)
        self.assertAlmostEqual(metrics["val-token/math/summary/analyzed_sample_fraction"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["val-token/math/summary/token_logprob_mean"], -0.5)
        self.assertAlmostEqual(metrics["val-token/math/bucket_01/token_fraction"], 0.5)
        self.assertAlmostEqual(metrics["val-token/code/summary/analyzed_sample_fraction"], 0.25)
        self.assertNotIn("val-token/code/summary/eos_prob_mean", metrics)


if __name__ == "__main__":
    unittest.main()
