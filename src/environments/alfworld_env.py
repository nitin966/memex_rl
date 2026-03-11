"""Modified ALFWorld environment for Memex(RL).

Implements the 4 modifications from the paper (Section 4.1):

  1. Hidden Commands: The agent's own commands are NOT echoed back.
     Reduces information leakage that makes compression trivial.

  2. Hidden Initial Observation: The initial "look" observation is
     NOT shown. Agent must explore to discover the environment.

  3. Limited Look: The "look" command is restricted to 1 use per
     location (instead of unlimited). Forces agent to use memory.

  4. Summary Truncation: Long observations are truncated to
     prevent single observations from filling the context window.

These modifications force the agent to rely on its Indexed Experience
Memory rather than brute-forcing through unlimited "look" commands.

Requires: pip install memex-rl[alfworld]
"""

from __future__ import annotations

import logging
from typing import Any

from src.environments.base import Environment, StepResult

logger = logging.getLogger(__name__)


class ALFWorldModifiedEnv(Environment):
    """Modified ALFWorld environment with paper's 4 difficulty enhancements.

    Args:
        max_obs_tokens: Maximum tokens per observation before truncation.
        look_limit: Max "look" commands per location (paper: 1).
        hide_initial_obs: Whether to hide the initial observation.
        hide_commands: Whether to hide echoed commands from observations.
    """

    def __init__(
        self,
        max_obs_tokens: int = 500,
        look_limit: int = 1,
        hide_initial_obs: bool = True,
        hide_commands: bool = True,
    ) -> None:
        self._max_obs_tokens = max_obs_tokens
        self._look_limit = look_limit
        self._hide_initial_obs = hide_initial_obs
        self._hide_commands = hide_commands

        # ALFWorld state
        self._env: Any = None
        self._task_id: str = ""
        self._task_desc: str = ""
        self._done: bool = False
        self._reward: float = 0.0

        # Modification tracking
        self._look_counts: dict[str, int] = {}  # location → look count
        self._current_location: str = "unknown"

        # Load ALFWorld
        try:
            self._init_alfworld()
        except ImportError:
            logger.warning(
                "ALFWorld not installed. You must install it to use this environment: "
                "pip install 'memex-rl[alfworld]'"
            )
            self._env = None

    def _init_alfworld(self) -> None:
        """Initialize the ALFWorld environment."""
        import alfworld.agents.environment as environment
        from alfworld.agents.utils.misc import add_task_to_grammar

        # ALFWorld config — uses default YAML
        import yaml
        import os

        config_path = os.environ.get(
            "ALFWORLD_CONFIG",
            os.path.join(os.path.dirname(__file__), "alfworld_config.yaml"),
        )
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {"env": {"type": "AlfredTWEnv"}}

        self._env = getattr(environment, config["env"]["type"])(config)

    def reset(self, task_id: str | None = None) -> str:
        """Reset and return the task description (hiding initial observation).

        Paper Modification 2: Initial observation hidden.
        """
        if self._env is None:
            raise RuntimeError(
                "ALFWorld is not installed. Cannot reset environment. "
                "Please install it via: pip install 'memex-rl[alfworld]'"
            )
            
        self._done = False
        self._reward = 0.0
        self._look_counts = {}
        self._current_location = "unknown"

        obs, info = self._env.reset()
        if isinstance(obs, list):
            obs = obs[0]
        self._task_desc = self._extract_task(obs)
        self._task_id = task_id or self._task_desc[:50]

        # Modification 2: Return only task description, not full observation
        if self._hide_initial_obs:
            return self._task_desc
        return self._task_desc

    def step(self, action: str) -> StepResult:
        """Execute action with all 4 paper modifications applied."""
        if self._env is None:
            raise RuntimeError(
                "ALFWorld is not installed. Cannot execute step. "
                "Please install it via: pip install 'memex-rl[alfworld]'"
            )
            
        if self._done:
            return StepResult(
                observation="Episode already finished.",
                done=True,
                reward=self._reward,
            )

        # Track location for look limiting
        if action.startswith("go to "):
            self._current_location = action[6:]

        # Modification 3: Limited look
        if action.strip().lower() == "look":
            count = self._look_counts.get(self._current_location, 0)
            if count >= self._look_limit:
                return StepResult(
                    observation=(
                        f"You have already looked here ({self._current_location}). "
                        f"Use ReadExperience to retrieve previously observed details."
                    ),
                    done=False,
                )
            self._look_counts[self._current_location] = count + 1

        # Execute in ALFWorld
        obs, reward, done, info = self._env.step([action])
        if isinstance(obs, list):
            obs = obs[0]
        if isinstance(reward, list):
            reward = reward[0]
        if isinstance(done, list):
            done = done[0]

        # Modification 1: Hide echoed commands
        if self._hide_commands:
            obs = self._strip_command_echo(obs, action)

        # Modification 4: Truncate long observations
        obs = self._truncate_observation(obs)

        self._done = bool(done)
        self._reward = float(reward)

        return StepResult(
            observation=obs,
            done=self._done,
            reward=self._reward,
        )

    def get_task_id(self) -> str:
        return self._task_id

    @property
    def is_done(self) -> bool:
        return self._done

    # ── Internal helpers ───────────────────────────────────────────────

    def _extract_task(self, obs: str) -> str:
        """Extract the task description from ALFWorld's initial observation."""
        lines = obs.strip().split("\n")
        # Task is typically the first line or after "Your task is to:"
        for line in lines:
            if "your task" in line.lower() or "you are" in line.lower():
                return line.strip()
        return lines[0] if lines else "Complete the household task."

    def _strip_command_echo(self, obs: str, action: str) -> str:
        """Remove the echoed command from the observation.

        ALFWorld echoes the command at the start of the observation.
        Modification 1 removes this to reduce information leakage.
        """
        if obs.startswith(f"> {action}"):
            obs = obs[len(f"> {action}"):].strip()
        elif obs.startswith(action):
            obs = obs[len(action):].strip()
        return obs

    def _truncate_observation(self, obs: str) -> str:
        """Truncate observation to max_obs_tokens.

        Modification 4: Prevents single observations from
        overwhelming the context window.
        """
        # Simple word-count approximation (fast, no tokenizer dep)
        words = obs.split()
        max_words = int(self._max_obs_tokens * 0.75)  # ~1.33 tokens/word
        if len(words) > max_words:
            truncated = " ".join(words[:max_words])
            return truncated + "...[observation truncated]"
        return obs
