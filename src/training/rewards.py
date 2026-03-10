"""Reward engine for MemexRL training.

Implements the episode-level return from Section 3.3, Equation 1:

    R = R_task - α₁·P_context - α₂·P_redundancy - α₃·P_format

Three memory-efficiency penalties, each normalized to [0, 1]:

  P_context:    Accumulated overflow beyond threshold τ across all steps.
                min(1, Σ max(0, C_t - τ) / (τ·T))

  P_redundancy: Fraction of tool calls with identical (name, args) signatures
                already seen, when no state-modifying operation occurred since.
                N_redundant / N_tool_call

  P_format:     Fraction of steps with malformed tool call output.
                N_malformed / N_tool_call
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.models.trajectory import Episode, Step


@dataclass
class RewardBreakdown:
    """Detailed breakdown of an episode's reward computation."""
    task_reward: float
    context_penalty: float
    redundancy_penalty: float
    format_penalty: float
    total_return: float

    # Raw counts for debugging
    total_steps: int = 0
    total_tool_calls: int = 0
    overflow_tokens: int = 0
    redundant_calls: int = 0
    malformed_calls: int = 0


class RewardEngine:
    """Computes episode-level returns for MemexRL training.

    Args:
        alpha_context: Weight for context overflow penalty.
        alpha_redundancy: Weight for redundant tool call penalty.
        alpha_format: Weight for format error penalty.
    """

    def __init__(
        self,
        alpha_context: float = 0.3,
        alpha_redundancy: float = 0.3,
        alpha_format: float = 0.3,
    ) -> None:
        self.alpha_context = alpha_context
        self.alpha_redundancy = alpha_redundancy
        self.alpha_format = alpha_format

    def compute_return(self, episode: Episode, threshold: int) -> float:
        """Compute the scalar return R for an episode (Eq. 1).

        Args:
            episode: Completed episode with segments and steps.
            threshold: Context threshold τ.

        Returns:
            Scalar return R ∈ [-1, 1].
        """
        return self.compute_breakdown(episode, threshold).total_return

    def compute_breakdown(self, episode: Episode, threshold: int) -> RewardBreakdown:
        """Compute the full reward breakdown for an episode.

        Returns:
            RewardBreakdown with all components and raw counts.
        """
        r_task = 1.0 if episode.task_success else 0.0
        all_steps = episode.all_steps()

        p_context, overflow = self._context_overflow_penalty(all_steps, threshold)
        p_redundancy, n_redundant, n_tools = self._redundancy_penalty(all_steps)
        p_format, n_malformed = self._format_error_penalty(all_steps)

        total = (
            r_task
            - self.alpha_context * p_context
            - self.alpha_redundancy * p_redundancy
            - self.alpha_format * p_format
        )

        return RewardBreakdown(
            task_reward=r_task,
            context_penalty=p_context,
            redundancy_penalty=p_redundancy,
            format_penalty=p_format,
            total_return=total,
            total_steps=len(all_steps),
            total_tool_calls=n_tools,
            overflow_tokens=overflow,
            redundant_calls=n_redundant,
            malformed_calls=n_malformed,
        )

    def _context_overflow_penalty(
        self, steps: list[Step], threshold: int
    ) -> tuple[float, int]:
        """P_context = min(1, Σ max(0, C_t - τ) / (τ·T))

        Excludes steps where compression was triggered (those steps
        intentionally have high context before rewriting).

        Returns:
            (penalty, total_overflow_tokens)
        """
        non_compress_steps = [
            s for s in steps if s.tool_call.name != "CompressExperience"
        ]
        T = len(non_compress_steps)
        if T == 0:
            return 0.0, 0

        overflow_sum = sum(
            max(0, step.context_tokens - threshold)
            for step in non_compress_steps
        )
        penalty = min(1.0, overflow_sum / (threshold * T))
        return penalty, overflow_sum

    def _redundancy_penalty(
        self, steps: list[Step]
    ) -> tuple[float, int, int]:
        """P_redundancy = N_redundant / N_tool_call

        A tool call is redundant if:
          - Same (name, arguments) signature was already called
          - No state-modifying operation occurred since the last identical call

        Memory tools (CompressExperience, ReadExperience, finish) and error
        steps are excluded from redundancy counting.

        Returns:
            (penalty, n_redundant, n_tool_calls)
        """
        MEMORY_TOOLS = {"CompressExperience", "ReadExperience", "finish", "_error"}

        seen_signatures: set[tuple[str, str]] = set()
        n_redundant = 0
        n_tool_calls = 0
        state_modified = False

        for step in steps:
            tc = step.tool_call
            if tc.name in MEMORY_TOOLS:
                # Memory/control tools modify state
                state_modified = True
                continue

            n_tool_calls += 1
            sig = tc.signature()

            if sig in seen_signatures and not state_modified:
                n_redundant += 1
            else:
                seen_signatures.add(sig)
                state_modified = False

        return (
            n_redundant / max(1, n_tool_calls),
            n_redundant,
            n_tool_calls,
        )

    def _format_error_penalty(
        self, steps: list[Step]
    ) -> tuple[float, int]:
        """P_format = N_malformed / N_tool_call

        Counts steps with at least one format error.

        Returns:
            (penalty, n_malformed)
        """
        n_total = len(steps)
        if n_total == 0:
            return 0.0, 0

        n_malformed = sum(1 for s in steps if len(s.format_errors) > 0)
        return n_malformed / max(1, n_total), n_malformed
