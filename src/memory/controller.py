"""Memory Controller for Memex(RL).

The central orchestrator managing the L1 (working context) ↔ L2 (experience store)
lifecycle. Implements Algorithm 1 from the paper:

  - Context window M = [m₀, u, M_work]
  - CompressExperience: archive blocks to D, rewrite M ← [m₀, u, IndexedSummary]
  - ReadExperience: dereference D[index], append to M
  - ContextStatus injection at each step
"""

from __future__ import annotations

from src.memory.anchor import AnchorExtractor, AnchorExtractionError
from src.memory.store import ExperienceStore
from src.memory.tokenizer import Tokenizer
from src.models.memory import (
    ContextStatus,
    IndexedSummary,
    MemoryBlock,
    Message,
    MessageRole,
)


class MemoryController:
    """Manages the Memex agent's context window and experience store.

    Responsible for:
      1. Maintaining context window M = [m₀, u, M_work]
      2. CompressExperience: archiving blocks to L2, rewriting working context
      3. ReadExperience: dereferencing L2 entries, injecting into context
      4. ContextStatus: monitoring token usage, injecting status messages
      5. Summary truncation to summary_max_tokens (300 in paper)

    Args:
        store: L2 ExperienceStore instance (per-episode).
        tokenizer: Tokenizer for counting tokens.
        context_window: Total context window size N (default 32768).
        threshold: Compression threshold τ (default 8000).
        summary_max_tokens: Max tokens for IndexedSummary (default 300, per paper).
    """

    def __init__(
        self,
        store: ExperienceStore,
        tokenizer: Tokenizer | None = None,
        context_window: int = 32_768,
        threshold: int = 8_000,
        summary_max_tokens: int = 300,
    ) -> None:
        self.store = store
        self.tokenizer = tokenizer or Tokenizer()
        self.context_window = context_window
        self.threshold = threshold
        self.summary_max_tokens = summary_max_tokens
        self._anchor_extractor = AnchorExtractor()

        # Context window M = [m₀, u, M_work]
        self._system_prompt: str = ""      # m₀ — never compressed
        self._task_instruction: str = ""   # u  — never compressed
        self._messages: list[Message] = [] # M_work (working context messages)

        # Token caches
        self._system_tokens: int = 0
        self._task_tokens: int = 0

    # ── Initialization ─────────────────────────────────────────────────

    def reset(self, system_prompt: str, task_instruction: str) -> None:
        """Initialize for a new episode: M ← [m₀, u], D ← ∅.

        Algorithm 1, lines 4-6.
        """
        self._system_prompt = system_prompt
        self._task_instruction = task_instruction
        self._messages = []
        self.store.clear()

        # Pre-compute fixed token counts
        self._system_tokens = self.tokenizer.count(system_prompt)
        self._task_tokens = self.tokenizer.count(task_instruction)

    # ── CompressExperience (Algorithm 1, lines 11-14) ──────────────────

    def compress_experience(
        self,
        summary: str,
        memory_blocks: list[MemoryBlock],
    ) -> str:
        """Archive blocks to L2 and rewrite working context.

        Algorithm 1 lines 11-14:
          - For each (index, content) in MemoryBlocks: D[index] ← content
          - M ← [m₀, u, IndexedSummary]

        Args:
            summary: The agent's indexed summary text.
            memory_blocks: List of MemoryBlocks to archive.

        Returns:
            Status message confirming the operation.
        """
        archived_count = 0
        errors: list[str] = []

        for block in memory_blocks:
            try:
                content = self._resolve_block_content(block)
                self.store.write(block.db_index, content)
                archived_count += 1
            except AnchorExtractionError as e:
                errors.append(f"Block '{block.db_index}': {e}")

        # Truncate summary to summary_max_tokens (paper: 300 tokens)
        truncated_summary = self.tokenizer.truncate(summary, self.summary_max_tokens)

        # Rewrite: M ← [m₀, u, IndexedSummary]
        self._messages = [
            Message(role=MessageRole.ASSISTANT, content=truncated_summary),
        ]

        result = f"Compression successful. {archived_count} block(s) archived."
        if errors:
            result += f" {len(errors)} block(s) failed: " + "; ".join(errors)
        return result

    # ── ReadExperience (Algorithm 1, lines 15-17) ──────────────────────

    def read_experience(self, db_index: str) -> str:
        """Dereference an index from L2 and inject into context.

        Algorithm 1 lines 15-17:
          - o ← D[index]
          - M ← M ⊕ [o]

        Args:
            db_index: The index to dereference.

        Returns:
            The retrieved content, or an error message.
        """
        content = self.store.read(db_index)
        if content is None:
            error_msg = (
                f"Error: index '{db_index}' not found in experience store. "
                f"Available indices: {self.store.list_indices()}"
            )
            self._messages.append(
                Message(role=MessageRole.TOOL, content=error_msg, name="ReadExperience")
            )
            return error_msg

        # M ← M ⊕ [o]
        self._messages.append(
            Message(role=MessageRole.TOOL, content=content, name="ReadExperience")
        )
        return content

    # ── Context Status (Algorithm 1, line 8) ───────────────────────────

    def get_context_status(self) -> ContextStatus:
        """Compute the deterministic context status for the current step."""
        working_tokens = self.working_token_count()
        total_tokens = self._system_tokens + self._task_tokens + working_tokens
        return ContextStatus.compute(
            working_tokens=working_tokens,
            total_tokens=total_tokens,
            threshold=self.threshold,
        )

    def inject_context_status(self) -> None:
        """Append ContextStatus to M (Algorithm 1, line 8)."""
        status = self.get_context_status()
        self._messages.append(
            Message(role=MessageRole.SYSTEM, content=status.to_message_text())
        )

    # ── Message Management ─────────────────────────────────────────────

    def append_assistant(self, content: str) -> None:
        """Append an assistant message (thinking + tool call) to M_work."""
        self._messages.append(Message(role=MessageRole.ASSISTANT, content=content))

    def append_tool_result(self, content: str, tool_name: str = "") -> None:
        """Append a tool result observation to M_work."""
        self._messages.append(
            Message(role=MessageRole.TOOL, content=content, name=tool_name or None)
        )

    def append_user(self, content: str) -> None:
        """Append a user message to M_work."""
        self._messages.append(Message(role=MessageRole.USER, content=content))

    # ── Context Queries ────────────────────────────────────────────────

    def get_messages(self) -> list[Message]:
        """Return the full context window M = [m₀, u, M_work]."""
        full: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=self._system_prompt),
            Message(role=MessageRole.USER, content=self._task_instruction),
        ]
        full.extend(self._messages)
        return full

    def get_messages_as_dicts(self) -> list[dict[str, str]]:
        """Return messages in the format expected by LLM APIs."""
        result = []
        for msg in self.get_messages():
            d: dict[str, str] = {"role": msg.role.value, "content": msg.content}
            if msg.name:
                d["name"] = msg.name
            result.append(d)
        return result

    def working_token_count(self) -> int:
        """Count tokens in M_work (excludes m₀ and u)."""
        return sum(self.tokenizer.count(msg.content) for msg in self._messages)

    def total_token_count(self) -> int:
        """Count total tokens in M = [m₀, u, M_work]."""
        return self._system_tokens + self._task_tokens + self.working_token_count()

    def last_observation(self) -> str:
        """Return the content of the last tool/observation message."""
        for msg in reversed(self._messages):
            if msg.role in (MessageRole.TOOL, MessageRole.USER):
                return msg.content
        return ""

    def get_conversation_text(self) -> str:
        """Return the full conversation as a single text string.

        Used by AnchorExtractor to search for anchor spans.
        """
        parts = []
        for msg in self._messages:
            parts.append(f"[{msg.role.value}] {msg.content}")
        return "\n".join(parts)

    def get_prefix(self) -> str:
        """Return the context prefix (system + task + current summary).

        Used by the trajectory processor for segmentation.
        """
        parts = [self._system_prompt, self._task_instruction]
        # If first working message is a summary from compression, include it
        if self._messages and self._messages[0].role == MessageRole.ASSISTANT:
            parts.append(self._messages[0].content)
        return "\n\n".join(parts)

    @property
    def num_working_messages(self) -> int:
        return len(self._messages)

    # ── Internal ───────────────────────────────────────────────────────

    def _resolve_block_content(self, block: MemoryBlock) -> str:
        """Resolve a MemoryBlock to its content string.

        Option A (explicit): return db_content directly.
        Option B (anchor): extract span from conversation using anchors.
        """
        if block.db_content is not None:
            return block.db_content

        # Option B: anchor-based extraction
        assert block.start_anchor and block.mid_anchor and block.end_anchor
        conversation = self.get_conversation_text()
        match = self._anchor_extractor.extract(
            conversation=conversation,
            start_anchor=block.start_anchor,
            mid_anchor=block.mid_anchor,
            end_anchor=block.end_anchor,
        )
        return match.content
