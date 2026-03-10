"""Anchor-based content extraction for Memex(RL).

Implements Option B from Section 3.2 of the paper:
  The model specifies three short text anchors (start_anchor, mid_anchor,
  end_anchor) that uniquely identify a span within the current conversation.
  The system locates the matching span and archives it verbatim, with the
  mid_anchor serving as a verification checkpoint to prevent false matches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnchorMatch:
    """Result of a successful anchor extraction."""
    content: str
    start_pos: int
    end_pos: int


class AnchorExtractionError(Exception):
    """Raised when anchor extraction fails."""
    pass


class AnchorExtractor:
    """Extracts verbatim spans from conversation text using three anchors.

    The dual-mode design (paper Section 3.2) gives the model flexibility:
      - Option A (explicit authoring): model writes content directly
      - Option B (anchor-based): model specifies anchors for verbatim extraction

    This class handles Option B. All three anchors are required:
      - start_anchor: unique text at the START of the target span
      - mid_anchor: text in the MIDDLE for verification (prevents false matches)
      - end_anchor: unique text at the END of the target span
    """

    def extract(
        self,
        conversation: str,
        start_anchor: str,
        mid_anchor: str,
        end_anchor: str,
    ) -> AnchorMatch:
        """Locate and extract a span from conversation text using three anchors.

        Args:
            conversation: The full conversation text to search within.
            start_anchor: Exact text at the start of the target span.
            mid_anchor: Exact text that must appear within the span (verification).
            end_anchor: Exact text at the end of the target span.

        Returns:
            AnchorMatch with the extracted verbatim content and positions.

        Raises:
            AnchorExtractionError: If any anchor is not found or verification fails.
        """
        if not start_anchor or not mid_anchor or not end_anchor:
            raise AnchorExtractionError(
                "All three anchors (start_anchor, mid_anchor, end_anchor) are required."
            )

        # Find start anchor
        start_idx = conversation.find(start_anchor)
        if start_idx == -1:
            raise AnchorExtractionError(
                f"start_anchor not found in conversation: '{start_anchor[:80]}...'"
            )

        # Find end anchor AFTER start
        end_search_start = start_idx + len(start_anchor)
        end_idx = conversation.find(end_anchor, end_search_start)
        if end_idx == -1:
            raise AnchorExtractionError(
                f"end_anchor not found after start_anchor: '{end_anchor[:80]}...'"
            )

        # Extract the full span (inclusive of both anchors)
        span_end = end_idx + len(end_anchor)
        span = conversation[start_idx:span_end]

        # Verify mid_anchor exists within the span
        if mid_anchor not in span:
            raise AnchorExtractionError(
                f"mid_anchor verification failed — not found within the extracted span: "
                f"'{mid_anchor[:80]}...'"
            )

        return AnchorMatch(
            content=span,
            start_pos=start_idx,
            end_pos=span_end,
        )

    def try_extract(
        self,
        conversation: str,
        start_anchor: str,
        mid_anchor: str,
        end_anchor: str,
    ) -> AnchorMatch | None:
        """Like extract() but returns None on failure instead of raising."""
        try:
            return self.extract(conversation, start_anchor, mid_anchor, end_anchor)
        except AnchorExtractionError:
            return None
