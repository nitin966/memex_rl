"""Rollout engine for MemexRL training.

Generates G parallel rollout episodes per prompt using the current policy.
Feeds them to the TrajectoryProcessor for segmentation and GRPO training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.loop import AgentConfig, MemexAgent
from src.environments.base import Environment
from src.llm.backend import LLMBackend
from src.models.trajectory import Episode


@dataclass
class RolloutConfig:
    """Configuration for the rollout engine."""
    group_size: int = 8           # G rollouts per prompt
    max_steps: int = 100          # T_max per episode
    context_window: int = 32_768
    threshold: int = 8_000
    temperature: float = 0.7
    summary_max_tokens: int = 300


class RolloutEngine:
    """Generates rollout episodes for MemexRL training.

    For each prompt/task, runs G parallel episodes using the current
    policy (LLM) and the environment. Returns grouped episodes for
    GRPO training.

    Args:
        llm: Current policy LLM backend.
        env_factory: Callable that creates fresh environment instances.
        config: Rollout configuration.
    """

    def __init__(
        self,
        llm: LLMBackend,
        env_factory: Any,  # Callable[[], Environment]
        config: RolloutConfig | None = None,
    ) -> None:
        self.llm = llm
        self.env_factory = env_factory
        self.config = config or RolloutConfig()

    def generate_rollouts(
        self,
        task_ids: list[str],
    ) -> list[tuple[str, list[Episode]]]:
        """Generate G rollouts for each task.

        Args:
            task_ids: List of task identifiers to run.

        Returns:
            List of (task_id, episodes) pairs for GRPO training.
        """
        rollout_groups = []

        for task_id in task_ids:
            episodes = []
            for g in range(self.config.group_size):
                env = self.env_factory()
                agent_config = AgentConfig(
                    max_steps=self.config.max_steps,
                    context_window=self.config.context_window,
                    threshold=self.config.threshold,
                    temperature=self.config.temperature,
                    summary_max_tokens=self.config.summary_max_tokens,
                )
                agent = MemexAgent(
                    llm=self.llm,
                    environment=env,
                    config=agent_config,
                )
                episode = agent.run_episode(task_id=task_id)
                episodes.append(episode)

            rollout_groups.append((task_id, episodes))

        return rollout_groups
