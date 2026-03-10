"""Trajectory data models for Memex(RL) training.

Implements the segmented trajectory structure from Section 3.3:
- Step: one agent step (thinking + tool call + observation)
- Segment: a trajectory segment between compression boundaries
- Episode: a complete episode with segmentation and terminal reward
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.models.tools import ToolCall


class Step(BaseModel):
    """One agent step: thinking z_t, tool call c_t, observation o_t.

    Corresponds to one iteration of the agent loop (Algorithm 1, lines 7-23).
    """
    thinking: str = Field(
        description="Agent's intermediate reasoning text z_t.",
    )
    tool_call: ToolCall = Field(
        description="Parsed tool invocation c_t.",
    )
    observation: str = Field(
        default="",
        description="Environment or tool response o_t.",
    )
    context_tokens: int = Field(
        default=0,
        description="Working context tokens C_t at this step (for reward computation).",
    )
    format_errors: list[str] = Field(
        default_factory=list,
        description="Format errors detected during tool call parsing.",
    )


class Segment(BaseModel):
    """A trajectory segment between compression boundaries.

    From Section 3.3 (Segmented Trajectory Processing):
      S_i = [system, task, summary_{i-1}, z_{i1}, c_{i1}, o_{i1}, ...]

    Each segment maintains its own context. S_0 contains full pre-compression
    history; subsequent segments contain the compressed summary plus new
    interactions. All segments from the same trajectory share the terminal
    reward R for credit assignment.
    """
    segment_idx: int = Field(description="Index of this segment within the episode.")
    prefix: str = Field(
        default="",
        description="Context prefix (system + task + inherited summary).",
    )
    steps: list[Step] = Field(
        default_factory=list,
        description="Sequence of reasoning-action-observation tuples in this segment.",
    )

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class Episode(BaseModel):
    """A complete agent episode with segmentation.

    If the agent compresses k times, the episode contains k+1 segments.
    The terminal_reward R is shared across all segments for GRPO training.
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique episode identifier.",
    )
    task_id: str = Field(
        default="",
        description="Task/prompt identifier from the environment.",
    )
    segments: list[Segment] = Field(
        default_factory=list,
        description="Ordered list of trajectory segments.",
    )
    terminal_reward: float = Field(
        default=0.0,
        description="Episode-level return R (Eq. 1).",
    )
    task_success: bool = Field(
        default=False,
        description="Whether the agent completed the task successfully.",
    )

    @property
    def total_steps(self) -> int:
        return sum(seg.num_steps for seg in self.segments)

    @property
    def num_compressions(self) -> int:
        """Number of CompressExperience calls = num_segments - 1."""
        return max(0, len(self.segments) - 1)

    @property
    def num_read_experience(self) -> int:
        """Count of ReadExperience calls across all segments."""
        return sum(
            1 for seg in self.segments for step in seg.steps
            if step.tool_call.name == "ReadExperience"
        )

    def all_steps(self) -> list[Step]:
        """Flatten all steps across segments."""
        return [step for seg in self.segments for step in seg.steps]
