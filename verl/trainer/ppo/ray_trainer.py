# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    # def _check_resource_available(self):
    #     """Check if the resource pool can be satisfied in this ray cluster."""
    #     node_available_resources = ray._private.state.available_resources_per_node()
    #     node_available_gpus = {
    #         node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
    #         for node, node_info in node_available_resources.items()
    #     }

    #     # check total required gpus can be satisfied
    #     total_available_gpus = sum(node_available_gpus.values())
    #     total_required_gpus = sum(
    #         [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
    #     )
    #     if total_available_gpus < total_required_gpus:
    #         raise ValueError(
    #             f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
    #         )
    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        nodes = ray.nodes()
        node_available_resources = [node["Resources"] for node in nodes if node["Alive"]]
        node_available_gpus = [
            node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node_info in node_available_resources
        ]

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus)
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self._prev_validation_diag_state: dict[str, dict[str, float | int]] = {}
        self._warned_missing_validation_distribution = False

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _get_validation_diagnostics_cfg(self):
        return self.config.trainer.get("validation_diagnostics", {}) or {}

    def _validation_diagnostics_enabled(self) -> bool:
        return bool(self._get_validation_diagnostics_cfg().get("enabled", False))

    def _validation_distribution_requested(self) -> bool:
        cfg = self._get_validation_diagnostics_cfg()
        if not cfg.get("enabled", False):
            return False
        return bool(cfg.get("track_eos_probability", True)) or int(cfg.get("token_distribution_topk", 0)) > 0

    @staticmethod
    def _safe_non_tensor_value(values, idx: int, default=None):
        if values is None:
            return default
        try:
            value = values[idx]
        except Exception:
            return default

        if isinstance(value, np.generic):
            return value.item()
        return value

    def _build_stable_validation_uid(self, batch: DataProto, idx: int) -> str:
        data_source = str(self._safe_non_tensor_value(batch.non_tensor_batch.get("data_source"), idx, "unknown"))
        index = self._safe_non_tensor_value(batch.non_tensor_batch.get("index"), idx, None)
        reward_model = self._safe_non_tensor_value(batch.non_tensor_batch.get("reward_model"), idx, {})
        ground_truth = reward_model.get("ground_truth", None) if isinstance(reward_model, dict) else None
        raw_prompt_ids = self._safe_non_tensor_value(batch.non_tensor_batch.get("raw_prompt_ids"), idx, None)

        prompt_ids_for_hash = raw_prompt_ids
        if prompt_ids_for_hash is None:
            prompt_ids_for_hash = batch.batch["input_ids"][idx].detach().cpu().tolist()
        elif isinstance(prompt_ids_for_hash, np.ndarray):
            prompt_ids_for_hash = prompt_ids_for_hash.tolist()

        payload = {
            "data_source": data_source,
            "index": index,
            "ground_truth": ground_truth,
            "prompt_ids": prompt_ids_for_hash,
        }
        payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha1(payload_str.encode("utf-8")).hexdigest()[:16]

        if index is not None:
            return f"val::{data_source}::{index}::{digest}"
        return f"val::{data_source}::{digest}"

    def _ensure_validation_uids(self, batch: DataProto):
        existing_uids = batch.non_tensor_batch.get("uid", None)
        if existing_uids is not None:
            batch.non_tensor_batch["uid"] = np.asarray([str(uid) for uid in existing_uids], dtype=object)
            return

        uids = [self._build_stable_validation_uid(batch, idx) for idx in range(len(batch))]
        batch.non_tensor_batch["uid"] = np.asarray(uids, dtype=object)

    @staticmethod
    def _compute_token_trend(values: torch.Tensor) -> float:
        if values.numel() <= 1:
            return 0.0

        positions = torch.arange(values.numel(), dtype=torch.float32)
        centered_positions = positions - positions.mean()
        denom = torch.sum(centered_positions.square()).item()
        if denom <= 0:
            return 0.0
        slope = torch.sum(centered_positions * values.to(torch.float32)).item() / denom
        return float(slope)

    @staticmethod
    def _extend_validation_infos(
        aggregated_infos: dict[str, list],
        batch_infos: dict[str, list],
        batch_size: int,
        num_previous_samples: int,
        reserved_keys: Optional[set[str]] = None,
    ):
        reserved_keys = reserved_keys or set()
        current_batch_keys = set(batch_infos.keys())

        for key, values in batch_infos.items():
            if key in reserved_keys:
                continue
            if key not in aggregated_infos:
                aggregated_infos[key] = [None] * num_previous_samples

            values = list(values)
            if len(values) < batch_size:
                values = values + [None] * (batch_size - len(values))
            aggregated_infos[key].extend(values[:batch_size])

        for key in list(aggregated_infos.keys()):
            if key in reserved_keys:
                continue
            if key not in current_batch_keys:
                aggregated_infos[key].extend([None] * batch_size)

    def _compute_validation_token_diagnostics(
        self,
        batch: DataProto,
        log_prob_batch: DataProto,
        next_validation_diag_state: dict[str, dict[str, float | int]],
    ) -> tuple[dict[str, list], list[dict[str, Any]]]:
        cfg = self._get_validation_diagnostics_cfg()
        if not cfg.get("enabled", False):
            return {}, [{} for _ in range(len(batch))]

        response_mask = batch.batch["response_mask"].detach().cpu().bool()
        response_ids = batch.batch["responses"].detach().cpu()
        policy_log_probs = log_prob_batch.batch["old_log_probs"].detach().cpu().to(torch.float32)
        entropys = log_prob_batch.batch["entropys"].detach().cpu().to(torch.float32)
        rollout_log_probs = None
        if "rollout_log_probs" in batch.batch.keys():
            rollout_log_probs = batch.batch["rollout_log_probs"].detach().cpu().to(torch.float32)
        eos_probs = log_prob_batch.batch["dist_eos_probs"].detach().cpu().to(torch.float32) if "dist_eos_probs" in log_prob_batch.batch.keys() else None
        top1_token_ids = (
            log_prob_batch.batch["dist_top1_token_ids"].detach().cpu().to(torch.int64)
            if "dist_top1_token_ids" in log_prob_batch.batch.keys()
            else None
        )
        top1_token_probs = (
            log_prob_batch.batch["dist_top1_token_probs"].detach().cpu().to(torch.float32)
            if "dist_top1_token_probs" in log_prob_batch.batch.keys()
            else None
        )
        topk_token_ids = (
            log_prob_batch.batch["dist_topk_token_ids"].detach().cpu().to(torch.int64)
            if "dist_topk_token_ids" in log_prob_batch.batch.keys()
            else None
        )
        topk_token_probs = (
            log_prob_batch.batch["dist_topk_token_probs"].detach().cpu().to(torch.float32)
            if "dist_topk_token_probs" in log_prob_batch.batch.keys()
            else None
        )
        topk_mass = (
            log_prob_batch.batch["dist_topk_mass"].detach().cpu().to(torch.float32)
            if "dist_topk_mass" in log_prob_batch.batch.keys()
            else None
        )

        compare_to_previous_eval = bool(cfg.get("compare_to_previous_eval", True))
        low_conf_prob_threshold = float(cfg.get("low_confidence_prob_threshold", 0.2))
        eos_high_prob_threshold = float(cfg.get("eos_high_prob_threshold", 0.1))
        tail_tokens = max(1, int(cfg.get("tail_tokens", 32)))
        max_tokens_per_sample = max(1, int(cfg.get("max_tokens_per_sample", 64)))
        samples_to_dump_token_details = int(cfg.get("samples_to_dump_token_details", 0))
        dump_token_details_unlimited = samples_to_dump_token_details < 0
        eos_token_id = self.tokenizer.eos_token_id

        if (
            self._validation_distribution_requested()
            and eos_probs is None
            and topk_token_ids is None
            and not self._warned_missing_validation_distribution
        ):
            print(
                "Warning: validation token distribution diagnostics were requested, "
                "but the actor did not return EOS/top-k distribution tensors on this path."
            )
            self._warned_missing_validation_distribution = True

        diag_metric_keys = [
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
            "rollout_logprob_delta_mean",
            "rollout_logprob_delta_abs_mean",
            "traj_mean_logprob_delta_prev_eval",
            "traj_tail_conf_delta_prev_eval",
            "response_length_delta_prev_eval",
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
            "eos_prob_delta_prev_eval",
            "eos_prob_final_delta_prev_eval",
        ]
        diag_infos = {key: [] for key in diag_metric_keys}
        sample_metadata: list[dict[str, Any]] = []
        dumped_token_detail_samples = 0

        for sample_idx in range(policy_log_probs.shape[0]):
            uid = str(self._safe_non_tensor_value(batch.non_tensor_batch.get("uid"), sample_idx, f"row-{sample_idx}"))
            mask = response_mask[sample_idx]
            valid_len = int(mask.sum().item())

            info: dict[str, Any] = {
                "uid": uid,
                "response_length": valid_len,
            }

            if valid_len > 0:
                token_ids = response_ids[sample_idx][mask]
                token_log_probs = policy_log_probs[sample_idx][mask]
                token_probs = token_log_probs.exp()
                token_entropys = entropys[sample_idx][mask]
                tail_width = min(valid_len, tail_tokens)
                sample_eos_probs = eos_probs[sample_idx][mask] if eos_probs is not None else None
                sample_top1_token_ids = top1_token_ids[sample_idx][mask] if top1_token_ids is not None else None
                sample_top1_token_probs = top1_token_probs[sample_idx][mask] if top1_token_probs is not None else None
                sample_topk_token_ids = topk_token_ids[sample_idx][mask] if topk_token_ids is not None else None
                sample_topk_token_probs = topk_token_probs[sample_idx][mask] if topk_token_probs is not None else None
                sample_topk_mass = topk_mass[sample_idx][mask] if topk_mass is not None else None

                info["token_logprob_mean"] = float(token_log_probs.mean().item())
                info["token_logprob_std"] = float(token_log_probs.std(unbiased=False).item())
                info["token_logprob_min"] = float(token_log_probs.min().item())
                info["token_logprob_max"] = float(token_log_probs.max().item())
                info["token_prob_mean"] = float(token_probs.mean().item())
                info["token_prob_min"] = float(token_probs.min().item())
                info["token_entropy_mean"] = float(token_entropys.mean().item())
                info["token_entropy_std"] = float(token_entropys.std(unbiased=False).item())
                info["low_confidence_token_ratio"] = float((token_probs < low_conf_prob_threshold).float().mean().item())
                info["answer_tail_confidence"] = float(token_probs[-tail_width:].mean().item())
                info["logprob_slope"] = self._compute_token_trend(token_log_probs)
                info["entropy_slope"] = self._compute_token_trend(token_entropys)

                if rollout_log_probs is not None:
                    rollout_sample_log_probs = rollout_log_probs[sample_idx][mask]
                    logprob_delta = token_log_probs - rollout_sample_log_probs
                    info["rollout_logprob_delta_mean"] = float(logprob_delta.mean().item())
                    info["rollout_logprob_delta_abs_mean"] = float(logprob_delta.abs().mean().item())

                if sample_eos_probs is not None:
                    info["eos_prob_mean"] = float(sample_eos_probs.mean().item())
                    info["eos_prob_max"] = float(sample_eos_probs.max().item())
                    info["eos_prob_final"] = float(sample_eos_probs[-1].item())
                    info["eos_prob_slope"] = self._compute_token_trend(sample_eos_probs)
                    info["eos_high_prob_ratio"] = float((sample_eos_probs > eos_high_prob_threshold).float().mean().item())

                if sample_top1_token_probs is not None:
                    info["top1_prob_mean"] = float(sample_top1_token_probs.mean().item())
                    info["top1_prob_final"] = float(sample_top1_token_probs[-1].item())
                if sample_top1_token_ids is not None and eos_token_id is not None:
                    info["eos_top1_ratio"] = float((sample_top1_token_ids == eos_token_id).float().mean().item())
                if sample_topk_mass is not None:
                    info["topk_mass_mean"] = float(sample_topk_mass.mean().item())
                    info["topk_mass_final"] = float(sample_topk_mass[-1].item())

                prev_state = self._prev_validation_diag_state.get(uid, None)
                if compare_to_previous_eval and prev_state is not None:
                    prev_mean_logprob = float(prev_state.get("token_logprob_mean", info["token_logprob_mean"]))
                    prev_tail_conf = float(prev_state.get("answer_tail_confidence", info["answer_tail_confidence"]))
                    prev_response_length = int(prev_state.get("response_length", info["response_length"]))
                    info["traj_mean_logprob_delta_prev_eval"] = info["token_logprob_mean"] - prev_mean_logprob
                    info["traj_tail_conf_delta_prev_eval"] = info["answer_tail_confidence"] - prev_tail_conf
                    info["response_length_delta_prev_eval"] = info["response_length"] - prev_response_length
                    if sample_eos_probs is not None:
                        prev_eos_prob_mean = float(prev_state.get("eos_prob_mean", info["eos_prob_mean"]))
                        prev_eos_prob_final = float(prev_state.get("eos_prob_final", info["eos_prob_final"]))
                        info["eos_prob_delta_prev_eval"] = info["eos_prob_mean"] - prev_eos_prob_mean
                        info["eos_prob_final_delta_prev_eval"] = info["eos_prob_final"] - prev_eos_prob_final

                next_validation_diag_state[uid] = {
                    "token_logprob_mean": info["token_logprob_mean"],
                    "answer_tail_confidence": info["answer_tail_confidence"],
                    "response_length": info["response_length"],
                }
                if sample_eos_probs is not None:
                    next_validation_diag_state[uid]["eos_prob_mean"] = info["eos_prob_mean"]
                    next_validation_diag_state[uid]["eos_prob_final"] = info["eos_prob_final"]

                should_dump_token_details = dump_token_details_unlimited or (
                    dumped_token_detail_samples < samples_to_dump_token_details
                )
                if should_dump_token_details and samples_to_dump_token_details != 0:
                    max_tokens = min(valid_len, max_tokens_per_sample)
                    token_id_list = token_ids[:max_tokens].tolist()
                    token_text_list = self.tokenizer.convert_ids_to_tokens(token_id_list)
                    token_details = []
                    for token_pos, (token_id, token_text) in enumerate(
                        zip(token_id_list, token_text_list, strict=True)
                    ):
                        token_detail = {
                            "position": token_pos,
                            "token_id": int(token_id),
                            "token": token_text,
                            "text": self.tokenizer.decode([int(token_id)], skip_special_tokens=False),
                            "logprob": float(token_log_probs[token_pos].item()),
                            "prob": float(token_probs[token_pos].item()),
                            "entropy": float(token_entropys[token_pos].item()),
                        }
                        if rollout_log_probs is not None:
                            rollout_token_logprob = float(rollout_sample_log_probs[token_pos].item())
                            token_detail["rollout_logprob"] = rollout_token_logprob
                            token_detail["logprob_delta_from_rollout"] = float(
                                token_log_probs[token_pos].item() - rollout_token_logprob
                            )
                        if sample_eos_probs is not None:
                            token_detail["eos_prob"] = float(sample_eos_probs[token_pos].item())
                        if sample_top1_token_ids is not None and sample_top1_token_probs is not None:
                            top1_token_id = int(sample_top1_token_ids[token_pos].item())
                            token_detail["top1_token_id"] = top1_token_id
                            token_detail["top1_token"] = self.tokenizer.convert_ids_to_tokens([top1_token_id])[0]
                            token_detail["top1_text"] = self.tokenizer.decode([top1_token_id], skip_special_tokens=False)
                            token_detail["top1_prob"] = float(sample_top1_token_probs[token_pos].item())
                        if sample_topk_token_ids is not None and sample_topk_token_probs is not None:
                            step_topk_ids = sample_topk_token_ids[token_pos].tolist()
                            step_topk_probs = sample_topk_token_probs[token_pos].tolist()
                            step_topk_tokens = self.tokenizer.convert_ids_to_tokens(step_topk_ids)
                            eos_rank_in_topk = None
                            topk_entries = []
                            for rank, (step_token_id, step_token, step_prob) in enumerate(
                                zip(step_topk_ids, step_topk_tokens, step_topk_probs, strict=True),
                                start=1,
                            ):
                                step_entry = {
                                    "rank": rank,
                                    "token_id": int(step_token_id),
                                    "token": step_token,
                                    "text": self.tokenizer.decode([int(step_token_id)], skip_special_tokens=False),
                                    "prob": float(step_prob),
                                }
                                if eos_token_id is not None and int(step_token_id) == eos_token_id and eos_rank_in_topk is None:
                                    eos_rank_in_topk = rank
                                topk_entries.append(step_entry)
                            token_detail["topk"] = topk_entries
                            token_detail["eos_in_topk"] = eos_rank_in_topk is not None
                            if eos_rank_in_topk is not None:
                                token_detail["eos_rank_in_topk"] = eos_rank_in_topk
                        token_details.append(token_detail)

                    info["token_diagnostics"] = token_details
                    info["token_diagnostics_total_tokens"] = valid_len
                    info["token_diagnostics_truncated"] = valid_len > max_tokens
                    dumped_token_detail_samples += 1

            for key in diag_metric_keys:
                diag_infos[key].append(info.get(key))

            sample_metadata.append(info)

        return diag_infos, sample_metadata

    @staticmethod
    def _attach_batch_extra_info_to_samples(sample_metadata: list[dict[str, Any]], batch_extra_info: dict[str, list]):
        if not sample_metadata:
            return

        for key, values in batch_extra_info.items():
            values = list(values)
            if len(values) < len(sample_metadata):
                values = values + [None] * (len(sample_metadata) - len(values))
            for sample_info, value in zip(sample_metadata, values[: len(sample_metadata)], strict=True):
                if isinstance(value, np.generic):
                    value = value.item()
                sample_info[key] = value

    @staticmethod
    def _format_validation_sample_annotation(sample_metadata: Optional[dict[str, Any]]) -> str:
        if not sample_metadata:
            return ""

        parts = []
        uid = sample_metadata.get("uid", None)
        if uid:
            parts.append(f"uid={str(uid).split('::')[-1]}")

        if sample_metadata.get("acc", None) is not None:
            parts.append(f"acc={float(sample_metadata['acc']):.3f}")
        elif sample_metadata.get("reward", None) is not None:
            parts.append(f"reward={float(sample_metadata['reward']):.3f}")

        if sample_metadata.get("response_length", None) is not None:
            parts.append(f"len={int(sample_metadata['response_length'])}")
        if sample_metadata.get("answer_tail_confidence", None) is not None:
            parts.append(f"tail_p={float(sample_metadata['answer_tail_confidence']):.3f}")
        if sample_metadata.get("low_confidence_token_ratio", None) is not None:
            parts.append(f"low_p={float(sample_metadata['low_confidence_token_ratio']):.3f}")
        if sample_metadata.get("token_logprob_mean", None) is not None:
            parts.append(f"logp={float(sample_metadata['token_logprob_mean']):.3f}")
        if sample_metadata.get("token_entropy_mean", None) is not None:
            parts.append(f"H={float(sample_metadata['token_entropy_mean']):.3f}")
        if sample_metadata.get("traj_mean_logprob_delta_prev_eval", None) is not None:
            parts.append(f"d_logp_prev={float(sample_metadata['traj_mean_logprob_delta_prev_eval']):+.3f}")
        if sample_metadata.get("rollout_logprob_delta_mean", None) is not None:
            parts.append(f"d_logp_roll={float(sample_metadata['rollout_logprob_delta_mean']):+.3f}")
        if sample_metadata.get("eos_prob_final", None) is not None:
            parts.append(f"eos_final={float(sample_metadata['eos_prob_final']):.3f}")
        if sample_metadata.get("eos_prob_final_delta_prev_eval", None) is not None:
            parts.append(f"d_eos_prev={float(sample_metadata['eos_prob_final_delta_prev_eval']):+.3f}")

        if not parts:
            return ""
        return " | ".join(parts)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        sample_metadata: Optional[list[dict[str, Any]]] = None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            if sample_metadata is not None and i < len(sample_metadata) and sample_metadata[i]:
                entry.update(sample_metadata[i])
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores, sample_metadata: Optional[list[dict[str, Any]]] = None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        samples = []
        for idx, (input_text, output_text, score) in enumerate(zip(inputs, outputs, scores, strict=True)):
            sample_info = sample_metadata[idx] if sample_metadata is not None and idx < len(sample_metadata) else None
            annotation = self._format_validation_sample_annotation(sample_info)
            if annotation:
                output_text = f"{output_text}\n\n[diag] {annotation}"
            samples.append((input_text, output_text, score))

        # Create tuples of (input, output, score) and sort by input text
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        core_reward_keys = {"reward", "reward_sum", "reward_token_std", "reward_token_min", "reward_token_max"}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_metadata = []
        next_validation_diag_state: dict[str, dict[str, float | int]] = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            self._ensure_validation_uids(test_batch)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            test_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.val_kwargs.temperature
            if self._validation_distribution_requested():
                test_batch.meta_info["return_distribution_diagnostics"] = True
                test_batch.meta_info["distribution_topk"] = int(
                    self._get_validation_diagnostics_cfg().get("token_distribution_topk", 0)
                )
                test_batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id

            if "response_mask" not in test_batch.batch.keys():
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            batch_diag_infos = {}
            batch_sample_metadata = [{} for _ in range(len(test_batch))]
            if self._validation_diagnostics_enabled():
                val_log_prob = self.actor_rollout_wg.compute_log_prob(test_batch)
                batch_diag_infos, batch_sample_metadata = self._compute_validation_token_diagnostics(
                    batch=test_batch,
                    log_prob_batch=val_log_prob,
                    next_validation_diag_state=next_validation_diag_state,
                )

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            # For validation display, we want token-granularity reward statistics.
            # Do NOT change reward assignment; only change how we aggregate/log it.
            response_mask = test_batch.batch["response_mask"].to(device=reward_tensor.device, dtype=reward_tensor.dtype)

            response_len = response_mask.sum(-1)
            valid = response_len > 0

            reward_sum = (reward_tensor * response_mask).sum(-1)
            reward_mean = reward_sum / response_len.clamp(min=1)

            # Per-token std/min/max for each sequence (masked by response_mask)
            reward_sq_mean = (reward_tensor.pow(2) * response_mask).sum(-1) / response_len.clamp(min=1)
            reward_var = (reward_sq_mean - reward_mean.pow(2)).clamp(min=0)
            reward_std = reward_var.sqrt()

            reward_min = reward_tensor.masked_fill(response_mask == 0, float("inf")).min(-1).values
            reward_max = reward_tensor.masked_fill(response_mask == 0, float("-inf")).max(-1).values

            # Use per-token mean as the canonical "reward" metric in validation.
            # This keeps reward-related metrics comparable across different response lengths.
            scores = reward_mean.detach().cpu().tolist()
            valid_cpu = valid.detach().cpu().tolist()
            sample_scores.extend(scores)
            for sample_info, score in zip(batch_sample_metadata, scores, strict=True):
                sample_info["reward"] = score

            reward_extra_infos_dict["reward"].extend([s if is_valid else None for s, is_valid in zip(scores, valid_cpu, strict=True)])
            reward_extra_infos_dict["reward_sum"].extend(
                [s if is_valid else None for s, is_valid in zip(reward_sum.detach().cpu().tolist(), valid_cpu, strict=True)]
            )
            reward_extra_infos_dict["reward_token_std"].extend(
                [s if is_valid else None for s, is_valid in zip(reward_std.detach().cpu().tolist(), valid_cpu, strict=True)]
            )
            reward_extra_infos_dict["reward_token_min"].extend(
                [s if is_valid else None for s, is_valid in zip(reward_min.detach().cpu().tolist(), valid_cpu, strict=True)]
            )
            reward_extra_infos_dict["reward_token_max"].extend(
                [s if is_valid else None for s, is_valid in zip(reward_max.detach().cpu().tolist(), valid_cpu, strict=True)]
            )
            
            batch_extra_info = result.get("reward_extra_info", {})
            num_previous_samples = len(sample_scores) - len(scores)
            self._extend_validation_infos(
                aggregated_infos=reward_extra_infos_dict,
                batch_infos=batch_extra_info,
                batch_size=len(scores),
                num_previous_samples=num_previous_samples,
                reserved_keys=core_reward_keys,
            )
            self._extend_validation_infos(
                aggregated_infos=reward_extra_infos_dict,
                batch_infos=batch_diag_infos,
                batch_size=len(scores),
                num_previous_samples=num_previous_samples,
                reserved_keys=core_reward_keys,
            )
            self._attach_batch_extra_info_to_samples(batch_sample_metadata, batch_extra_info)
            sample_metadata.extend(batch_sample_metadata)


            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        if self._validation_diagnostics_enabled():
            self._prev_validation_diag_state = next_validation_diag_state

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            sample_metadata=sample_metadata,
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                sample_metadata=sample_metadata,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "pass"])
                        and (f"@{n_max}" in metric_name or metric_name == "pass@1")
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # 初始化日志记录器 (例如 WandB, TensorBoard)
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # 在开始之前加载检查点
        self._load_checkpoint()

        # 在训练前执行验证
        # 目前，我们仅支持使用 reward_function 进行验证。
        # breakpoint()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # 添加进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # 我们从第 1 步开始
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # ----------------------------------------------------------------------
        # 主训练循环
        # ----------------------------------------------------------------------
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # 如果当前步骤启用了性能分析，则开始分析
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                # 1. 准备批次数据
                # 将批次字典转换为 DataProto 对象
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 为批次添加 uid，用于跟踪和优势计算
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # 提取生成批次 (input_ids, attention_mask 等)
                gen_batch = self._get_gen_batch(batch)

                # 传递 global_steps 以进行跟踪
                gen_batch.meta_info["global_steps"] = self.global_steps
                # 重复批次以进行每个提示的多次 rollout (n)
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # 2. Rollout (生成)
                    # 使用 Actor 模型生成序列
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # 3. 优势估计 (REMAX 特有)
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    
                    # 将原始批次与生成的输出合并
                    # 重复以与 rollout 中的重复响应对齐
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # 平衡 DP rank 之间的有效 token 数量。
                    # 注意：这通常会改变 `batch` 中数据的顺序，
                    # 这不会影响优势计算（因为它是基于 uid 的），
                    # 但可能会影响损失计算（由于 mini-batching 的变化）。
                    # TODO: 解耦 DP 平衡和 mini-batching。
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 计算全局有效 token 数
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # breakpoint()

                    # 4. 奖励计算
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # 计算奖励模型分数
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # 5. 旧对数概率计算
                    # 重新计算旧的 log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: 我们可能也想添加概率的差异。
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    # 6. 参考策略对数概率
                    if self.use_reference_policy:
                        # 计算参考 log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # 7. Critic 价值计算
                    # 计算价值
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # 8. 优势计算
                    with marked_timer("adv", timing_raw, color="brown"):
                        # 我们结合基于规则的 RM
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 计算奖励。如果可用，应用 KL 惩罚
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 计算优势，在驱动进程上执行
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO 优势归一化因子

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # 9. Critic 更新
                    # 更新 critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # 10. Actor 更新
                    # 实现 critic 预热
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新 actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # 如果启用，记录 rollout 生成
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # 11. 验证
                # 验证
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 12. 检查点保存
                # 检查 ESI (弹性服务器实例)/训练计划是否接近过期。
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # 检查是否满足保存检查点的条件。
                # 条件包括一个强制条件 (1) 和
                # 以下可选条件之一 (2/3/4):
                # 1. 保存频率设置为正值。
                # 2. 这是最后一个训练步骤。
                # 3. 当前步骤数是保存频率的倍数。
                # 4. ESI (弹性服务器实例)/训练计划接近过期。
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # 训练指标
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # 收集指标
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: 实现实际的 tflpo 和理论上的 tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # 这是实验性的，将来可能会更改/删除，以支持通用的采样器
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: 制作一个支持各种后端的规范记录器
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # 这是实验性的，将来可能会更改/删除
                # 以支持通用的数据缓冲池
                if hasattr(self.train_dataset, "on_batch_end"):
                    # 每次训练批次后数据集可能会更改
                    self.train_dataset.on_batch_end(batch=batch)
