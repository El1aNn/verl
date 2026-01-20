# Verl PPO Training Workflow Overview

本文档旨在帮助开发者理解 Verl 框架中 PPO 训练的整体流程与核心逻辑。

## 1. 核心架构

Verl 采用基于 Ray 的分布式架构，主要包含以下几个核心角色：

*   **RayPPOTrainer (Controller)**: 位于 `verl/trainer/ppo/ray_trainer.py`。它是整个训练过程的总指挥，负责协调各个 Worker，控制训练循环（Rollout -> Compute Reward -> Update）。
*   **ActorRolloutRefWorker**: 负责生成经验（Rollout）和计算参考策略的 Log Prob。通常包含 Actor 模型（策略网络）和 Reference 模型。
*   **CriticWorker**: 负责价值估计（Value Estimation）。包含 Critic 模型（价值网络）。
*   **RewardManager**: 负责计算奖励（Reward）。可以是基于规则的（如数学题判题），也可以是基于模型的（Reward Model）。

## 2. 训练流程 (Training Loop)

训练的主循环在 `RayPPOTrainer.fit()` 方法中定义，每个 Epoch 的核心步骤如下：

### Step 1: 生成经验 (Rollout)
*   **输入**: 从 `train_dataloader` 获取一批 Prompt。
*   **动作**: `ActorRolloutRefWorker` 根据当前的 Actor 策略，对 Prompt 进行补全，生成 Response。
*   **产出**: `DataProto` 对象，包含 Prompts, Responses, Log Probs 等信息。

### Step 2: 计算奖励与价值 (Evaluation & Value Estimation)
*   **计算奖励 (Reward)**:
    *   生成的 Response 被送往 `RewardManager`。
    *   如果是数学任务，会进行规则匹配判分；如果是 RLHF，会调用 Reward Model 打分。
    *   **关键代码**: `verl/workers/reward_manager/naive.py` 中的 `NaiveRewardManager`。
*   **计算价值 (Value)**:
    *   `CriticWorker` 接收 Prompt + Response，输出每个 Token 的价值估计（Values）。
*   **计算参考概率 (Ref Log Prob)**:
    *   `ActorRolloutRefWorker` 使用 Reference Model 计算生成文本的参考对数概率，用于计算 KL 散度。

### Step 3: 优势估计 (Advantage Estimation)
*   **处理**: Controller 收集所有数据（Reward, Values, Ref Log Probs）。
*   **计算**:
    *   **GAE (Generalized Advantage Estimation)**: 传统的 PPO 使用 GAE 计算优势函数。
    *   **GRPO (Group Relative Policy Optimization)**: 如果配置了 GRPO，会使用组内相对优势（Group Relative Advantage），通常不需要 Critic 模型，而是基于一组采样的平均 Reward 作为 Baseline。
*   **产出**: Advantages 和 Returns (Target Values)。

### Step 4: 策略更新 (Update)
*   **PPO Epochs**: 在收集到的这一批经验上，进行多次 PPO 更新（Inner Loop）。
*   **Actor 更新**:
    *   计算 PPO Loss（包含 Policy Loss, Entropy Bonus, KL Penalty）。
    *   `ActorRolloutRefWorker` 执行反向传播和参数更新。
*   **Critic 更新**:
    *   计算 Value Loss（预测值与 Target Values 的 MSE）。
    *   `CriticWorker` 执行反向传播和参数更新。

## 3. 关键代码路径

*   **入口**: `verl/trainer/main_ppo.py` -> `run_ppo`
*   **Trainer**: `verl/trainer/ppo/ray_trainer.py`
    *   `fit()`: 训练主循环。
    *   `_generate_rollouts()`: 生成经验。
*   **算法核心**: `verl/trainer/ppo/core_algos.py`
    *   `compute_gae_advantage_return()`: GAE 计算。
    *   `compute_grpo_outcome_advantage()`: GRPO 计算。
*   **Reward**: `verl/workers/reward_manager/naive.py`
    *   `__call__`: 计算 Reward 的入口。

## 4. 调试建议

*   **Reward 调试**: 关注 `verl/workers/reward_manager/naive.py`。
*   **PPO 逻辑调试**: 关注 `verl/trainer/ppo/ray_trainer.py` 中的 `fit` 方法。
*   **模型输出调试**: 可以在 `ActorRolloutRefWorker` 的 `generate_sequences` 方法中查看生成的文本。
