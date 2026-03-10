"""Unit tests for AnchorExtractor (PR 3)."""

from __future__ import annotations

import pytest

from src.memory.anchor import AnchorExtractionError, AnchorExtractor, AnchorMatch


class TestAnchorExtractor:
    def setup_method(self):
        self.extractor = AnchorExtractor()
        self.conversation = (
            "[assistant] Let me check the file.\n"
            "[tool] def calculate_sum(a, b):\n"
            "    result = a + b\n"
            "    print(f'Sum: {result}')\n"
            "    return result\n"
            "[assistant] The function looks correct.\n"
            "[tool] Test output: PASSED (3 tests)\n"
            "All assertions verified.\n"
        )

    # ── Successful Extractions ─────────────────────────────────────────

    def test_basic_extraction(self):
        match = self.extractor.extract(
            conversation=self.conversation,
            start_anchor="def calculate_sum",
            mid_anchor="result = a + b",
            end_anchor="return result",
        )
        assert "def calculate_sum(a, b):" in match.content
        assert "result = a + b" in match.content
        assert "return result" in match.content

    def test_extracts_verbatim(self):
        """Anchor extraction must preserve content verbatim."""
        match = self.extractor.extract(
            conversation=self.conversation,
            start_anchor="Test output: PASSED",
            mid_anchor="3 tests",
            end_anchor="assertions verified.",
        )
        assert match.content == "Test output: PASSED (3 tests)\nAll assertions verified."

    def test_positions_are_correct(self):
        match = self.extractor.extract(
            conversation=self.conversation,
            start_anchor="def calculate_sum",
            mid_anchor="a + b",
            end_anchor="return result",
        )
        # The extracted content should match conversation[start_pos:end_pos]
        assert self.conversation[match.start_pos:match.end_pos] == match.content

    # ── Failure Cases ──────────────────────────────────────────────────

    def test_start_anchor_not_found(self):
        with pytest.raises(AnchorExtractionError, match="start_anchor not found"):
            self.extractor.extract(
                conversation=self.conversation,
                start_anchor="NONEXISTENT_START",
                mid_anchor="something",
                end_anchor="something else",
            )

    def test_end_anchor_not_found(self):
        with pytest.raises(AnchorExtractionError, match="end_anchor not found"):
            self.extractor.extract(
                conversation=self.conversation,
                start_anchor="def calculate_sum",
                mid_anchor="a + b",
                end_anchor="NONEXISTENT_END",
            )

    def test_mid_anchor_verification_fails(self):
        with pytest.raises(AnchorExtractionError, match="mid_anchor verification"):
            self.extractor.extract(
                conversation=self.conversation,
                start_anchor="def calculate_sum",
                mid_anchor="WRONG_MIDDLE_TEXT",
                end_anchor="return result",
            )

    def test_empty_anchors_raise(self):
        with pytest.raises(AnchorExtractionError, match="required"):
            self.extractor.extract(
                conversation=self.conversation,
                start_anchor="",
                mid_anchor="mid",
                end_anchor="end",
            )

    # ── try_extract ────────────────────────────────────────────────────

    def test_try_extract_returns_match_on_success(self):
        match = self.extractor.try_extract(
            conversation=self.conversation,
            start_anchor="def calculate_sum",
            mid_anchor="a + b",
            end_anchor="return result",
        )
        assert match is not None
        assert "def calculate_sum" in match.content

    def test_try_extract_returns_none_on_failure(self):
        match = self.extractor.try_extract(
            conversation=self.conversation,
            start_anchor="NONEXISTENT",
            mid_anchor="mid",
            end_anchor="end",
        )
        assert match is None

    # ── Edge Cases ─────────────────────────────────────────────────────

    def test_anchors_at_boundaries(self):
        text = "START_HERE middle_content END_HERE"
        match = self.extractor.extract(
            conversation=text,
            start_anchor="START_HERE",
            mid_anchor="middle_content",
            end_anchor="END_HERE",
        )
        assert match.content == text

    def test_unicode_anchors(self):
        text = "日本語の開始 中間部分 終了部分"
        match = self.extractor.extract(
            conversation=text,
            start_anchor="日本語の開始",
            mid_anchor="中間部分",
            end_anchor="終了部分",
        )
        assert match.content == text
