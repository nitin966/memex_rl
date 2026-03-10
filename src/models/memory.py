"""Memory-related data models for Memex(RL).

Implements the core data structures from the paper:
- MemoryBlock: content to archive in the L2 external experience store
- IndexedSummary: the in-context indexed summary σ = (s, I)
- ContextStatus: deterministic context status message appended each step
- Message: a single message in the agent's context window M
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MessageRole(str, Enum):
    """Roles for messages in the context window."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the agent's context window M."""
    role: MessageRole
    content: str
    name: str | None = None  # Tool name for role=tool messages


class MemoryBlock(BaseModel):
    """A single block to archive in the external experience store D.

    Supports two modes (paper Section 3.2):
      (a) Explicit authoring: agent writes db_content directly
      (b) Anchor-based extraction: agent specifies start/mid/end anchors
          to extract a span verbatim from the conversation
    """
    db_index: str = Field(
        description="Stable index key, e.g. 'ctx_code_001'. Used as the key in D."
    )
    # Option A: explicit content
    db_content: str | None = Field(
        default=None,
        description="Content authored directly by the agent (summaries, reorganized notes).",
    )
    # Option B: anchor-based extraction
    start_anchor: str | None = Field(
        default=None,
        description="Exact text at the START of the span to extract.",
    )
    mid_anchor: str | None = Field(
        default=None,
        description="Exact text in the MIDDLE of the span (verification checkpoint).",
    )
    end_anchor: str | None = Field(
        default=None,
        description="Exact text at the END of the span to extract.",
    )

    @model_validator(mode="after")
    def validate_content_mode(self) -> MemoryBlock:
        """Ensure exactly one content mode is specified."""
        has_explicit = self.db_content is not None
        has_anchors = any(
            a is not None for a in (self.start_anchor, self.mid_anchor, self.end_anchor)
        )
        if not has_explicit and not has_anchors:
            raise ValueError(
                "MemoryBlock must specify either db_content (Option A) "
                "or all three anchors (Option B)."
            )
        if has_explicit and has_anchors:
            raise ValueError(
                "MemoryBlock cannot specify both db_content and anchors. "
                "Choose Option A (explicit) or Option B (anchor-based)."
            )
        if has_anchors:
            missing = []
            if self.start_anchor is None:
                missing.append("start_anchor")
            if self.mid_anchor is None:
                missing.append("mid_anchor")
            if self.end_anchor is None:
                missing.append("end_anchor")
            if missing:
                raise ValueError(
                    f"Anchor-based extraction requires all three anchors. "
                    f"Missing: {', '.join(missing)}"
                )
        return self


class IndexEntry(BaseModel):
    """A single (index, description) pair in the index map I."""
    index: str = Field(description="Stable index into D.")
    description: str = Field(description="Summarized descriptor of archived content.")


class IndexedSummary(BaseModel):
    """The in-context indexed summary σ = (s, I).

    Definition 1 from the paper:
      s = compact, actionable progress state
      I = {(index, description)} mapping indices to descriptors
    """
    summary: str = Field(
        description="Compact actionable progress state (verified info, plans, next steps).",
    )
    index_map: list[IndexEntry] = Field(
        default_factory=list,
        description="Set of (index, description) pairs pointing into D.",
    )

    def to_prompt_text(self) -> str:
        """Render as the text that goes into the working context."""
        lines = [self.summary, "", "Index map:"]
        for entry in self.index_map:
            lines.append(f"- {entry.index} - {entry.description}")
        return "\n".join(lines)

    @classmethod
    def from_summary_string(cls, summary: str) -> IndexedSummary:
        """Parse a summary string (as the agent writes it) into structured form.

        Expected format:
            Index map:
            - ctx_data_001 - Brief description
            - ctx_data_002 - Brief description
            Status: Current progress and next steps
        """
        index_map: list[IndexEntry] = []
        non_index_lines: list[str] = []

        for line in summary.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- ") and " - " in stripped[2:]:
                # Parse "- index - description"
                rest = stripped[2:]  # Remove leading "- "
                parts = rest.split(" - ", 1)
                if len(parts) == 2:
                    index_map.append(IndexEntry(index=parts[0].strip(), description=parts[1].strip()))
                    continue
            if stripped.lower() != "index map:":
                non_index_lines.append(line)

        return cls(
            summary="\n".join(non_index_lines).strip(),
            index_map=index_map,
        )


class ContextStatus(BaseModel):
    """Deterministic context status message appended at each agent step.

    From Algorithm 1, line 8:
      ContextStatus(M, τ) reports working context token usage and threshold.
      Format: "[Context Status: working tokens=X, threshold=Z]"
    """
    working_tokens: int = Field(description="Current working context token count.")
    total_tokens: int = Field(description="Total tokens in context window M.")
    threshold: int = Field(description="Compression threshold τ.")
    warning: str | None = Field(
        default=None,
        description="Warning string when working > threshold.",
    )

    def to_message_text(self) -> str:
        """Render as the text injected into the conversation."""
        text = (
            f"[Context Status: working tokens={self.working_tokens}, "
            f"threshold={self.threshold}]"
        )
        if self.warning:
            text += f"\n{self.warning}"
        return text

    @classmethod
    def compute(
        cls, working_tokens: int, total_tokens: int, threshold: int
    ) -> ContextStatus:
        """Factory that computes warning automatically."""
        warning = None
        if working_tokens > threshold:
            warning = (
                f"WARNING: working ({working_tokens}) > threshold ({threshold}). "
                f"Compress immediately to avoid penalty."
            )
        elif working_tokens > int(threshold * 0.8):
            warning = (
                f"NOTICE: working ({working_tokens}) approaching threshold ({threshold}). "
                f"Consider compressing soon."
            )
        return cls(
            working_tokens=working_tokens,
            total_tokens=total_tokens,
            threshold=threshold,
            warning=warning,
        )
