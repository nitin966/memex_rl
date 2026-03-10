"""GRPO Trainer skeleton for MemexRL.

Implements the Group Relative Policy Optimization algorithm (Section 3.3, Eq. 2-3):

    L = -E[min(r_t · A_hat, clip(r_t, 1-ε, 1+ε) · A_hat)] + λ_KL · KL(π_θ || π_ref)

Where:
  - r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) — importance ratio
  - A_hat = group-relative normalized advantage
  - ε = clip ratio (2.0 in paper for truncated importance sampling)
  - λ_KL = KL divergence penalty (0.001 in paper)

Paper hyperparameters:
  - Group size G = 8 (rollouts per prompt)
  - Batch size = 32 prompts per rollout step
  - LR = 5e-6, weight decay = 0.1
  - Context window = 32K, threshold τ = 8K
  - INT4 quantization for inference, QAT for backward pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.training.rewards import RewardEngine
from src.training.trajectory import TrajectoryProcessor, TrainingBatch, TrainingSegment


@dataclass
class GRPOConfig:
    """Configuration for the GRPO trainer.

    All defaults match the paper's hyperparameters.
    """
    # GRPO parameters
    group_size: int = 8               # G rollouts per prompt
    batch_size: int = 32              # Prompts per rollout step

    # Optimization
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    clip_ratio: float = 2.0           # Truncated importance sampling
    kl_penalty: float = 0.001         # λ_KL

    # Context budget
    context_window: int = 32_768
    threshold: int = 8_000

    # Reward weights
    alpha_context: float = 0.3
    alpha_redundancy: float = 0.3
    alpha_format: float = 0.3

    # Training
    max_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0


@dataclass
class TrainStepResult:
    """Result of a single GRPO training step."""
    loss: float = 0.0
    policy_loss: float = 0.0
    kl_loss: float = 0.0
    mean_reward: float = 0.0
    mean_advantage: float = 0.0
    num_segments: int = 0
    mean_clip_fraction: float = 0.0   # Fraction of ratios clipped


class GRPOTrainer:
    """GRPO trainer for MemexRL.

    This is a framework-agnostic skeleton that defines the training
    logic. The actual model forward/backward passes are delegated to
    a model wrapper (to be connected with HuggingFace/Slime in PR 10).

    The trainer handles:
      1. Rollout generation (via RolloutEngine)
      2. Reward computation (via RewardEngine)
      3. Trajectory segmentation (via TrajectoryProcessor)
      4. Group-relative advantage computation
      5. PPO-clipped surrogate loss + KL penalty

    Usage:
        trainer = GRPOTrainer(config, model_wrapper)
        for step in range(num_steps):
            result = trainer.train_step(prompts)
    """

    def __init__(
        self,
        config: GRPOConfig,
        model: Any = None,
        ref_model: Any = None,
    ) -> None:
        self.config = config
        self.model = model
        self.ref_model = ref_model

        self.reward_engine = RewardEngine(
            alpha_context=config.alpha_context,
            alpha_redundancy=config.alpha_redundancy,
            alpha_format=config.alpha_format,
        )
        self.trajectory_processor = TrajectoryProcessor(
            reward_engine=self.reward_engine,
            threshold=config.threshold,
        )
        self._step_count = 0

    def prepare_batch(
        self,
        rollout_groups: list[tuple[str, list[Any]]],
    ) -> TrainingBatch:
        """Process rollout groups into a training batch with advantages.

        Args:
            rollout_groups: List of (prompt_id, episodes) pairs.

        Returns:
            TrainingBatch ready for loss computation.
        """
        batch = self.trajectory_processor.build_batch(rollout_groups)
        return batch

    def compute_loss(self, batch: TrainingBatch) -> TrainStepResult:
        """Compute the GRPO loss for a training batch.

        This is a skeleton that computes the loss structure.
        Actual gradient computation requires a model wrapper.

        The loss (Eq. 3):
          L = -E[min(r_t · A_hat, clip(r_t, 1-ε, 1+ε) · A_hat)]
              + λ_KL · KL(π_θ || π_ref)
        """
        if batch.size == 0:
            return TrainStepResult()

        # Aggregate statistics
        total_reward = sum(seg.reward for seg in batch.segments)
        mean_reward = total_reward / batch.size
        mean_advantage = sum(seg.advantage for seg in batch.segments) / batch.size

        result = TrainStepResult(
            mean_reward=mean_reward,
            mean_advantage=mean_advantage,
            num_segments=batch.size,
        )

        # Actual loss computation requires model logprobs — would be:
        # for each token in each segment:
        #   r_t = π_θ(token|prefix) / π_θ_old(token|prefix)
        #   clipped = clip(r_t, 1 - ε, 1 + ε)
        #   policy_loss += -min(r_t * A_hat, clipped * A_hat)
        #   kl_loss += KL(π_θ(·|prefix) || π_ref(·|prefix))
        # result.loss = policy_loss + λ_KL * kl_loss

        if self.model is not None:
            # TODO: Connect to actual model in PR 10
            pass

        self._step_count += 1
        return result

    def train_step(
        self,
        rollout_groups: list[tuple[str, list[Any]]],
    ) -> TrainStepResult:
        """Execute one full GRPO training step.

        1. Process rollouts into segmented training batch
        2. Compute group-relative advantages
        3. Compute PPO-clipped loss + KL penalty
        4. (With model) perform backward pass + optimizer step

        Args:
            rollout_groups: Grouped rollout episodes from RolloutEngine.

        Returns:
            TrainStepResult with loss and metrics.
        """
        batch = self.prepare_batch(rollout_groups)
        result = self.compute_loss(batch)
        return result

    @property
    def step_count(self) -> int:
        return self._step_count
