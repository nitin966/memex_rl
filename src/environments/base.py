"""Abstract environment interface for Memex(RL).

Defines the contract that all task environments must implement
for integration with the Memex agent loop (Algorithm 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StepResult:
    """Result of an environment step."""
    observation: str
    done: bool = False
    reward: float = 0.0
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class Environment(ABC):
    """Abstract environment interface for Memex agent integration.

    All task environments (ALFWorld, stress test, etc.) must implement
    this interface to work with MemexAgent.run_episode().
    """

    @abstractmethod
    def reset(self, task_id: str | None = None) -> str:
        """Reset the environment and return the initial observation/task.

        Args:
            task_id: Optional task identifier to load a specific task.

        Returns:
            The task instruction string (becomes 'u' in the agent context).
        """
        ...

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Execute an action and return the result.

        Args:
            action: The action string from execute_action tool.

        Returns:
            StepResult with observation, done flag, and reward.
        """
        ...

    @abstractmethod
    def get_task_id(self) -> str:
        """Return the current task identifier."""
        ...

    @property
    def is_done(self) -> bool:
        """Whether the current episode has terminated."""
        return False
