"""Unit tests for core Pydantic models and tokenizer (PR 1)."""

from __future__ import annotations

import pytest

from src.models.memory import (
    ContextStatus,
    IndexedSummary,
    IndexEntry,
    MemoryBlock,
    Message,
    MessageRole,
)
from src.models.tools import (
    COMPRESS_EXPERIENCE_TOOL,
    FINISH_TOOL,
    MEMEX_TOOLS,
    READ_EXPERIENCE_TOOL,
    ToolCall,
    ToolDefinition,
)
from src.models.trajectory import Episode, Segment, Step
from src.memory.tokenizer import Tokenizer


# ── MemoryBlock Tests ──────────────────────────────────────────────────────

class TestMemoryBlock:
    def test_explicit_content(self):
        block = MemoryBlock(db_index="ctx_001", db_content="some content")
        assert block.db_content == "some content"
        assert block.start_anchor is None

    def test_anchor_content(self):
        block = MemoryBlock(
            db_index="ctx_002",
            start_anchor="start here",
            mid_anchor="middle part",
            end_anchor="end here",
        )
        assert block.start_anchor == "start here"
        assert block.db_content is None

    def test_no_content_raises(self):
        with pytest.raises(ValueError, match="must specify either"):
            MemoryBlock(db_index="ctx_003")

    def test_both_modes_raises(self):
        with pytest.raises(ValueError, match="cannot specify both"):
            MemoryBlock(
                db_index="ctx_004",
                db_content="explicit",
                start_anchor="also anchor",
                mid_anchor="mid",
                end_anchor="end",
            )

    def test_partial_anchors_raises(self):
        with pytest.raises(ValueError, match="Missing: mid_anchor"):
            MemoryBlock(
                db_index="ctx_005",
                start_anchor="start",
                end_anchor="end",
            )


# ── IndexedSummary Tests ──────────────────────────────────────────────────

class TestIndexedSummary:
    def test_to_prompt_text(self):
        summary = IndexedSummary(
            summary="Task in progress. Found object.",
            index_map=[
                IndexEntry(index="ctx_loc", description="All location IDs"),
                IndexEntry(index="ctx_obj", description="Found objects"),
            ],
        )
        text = summary.to_prompt_text()
        assert "Task in progress" in text
        assert "- ctx_loc - All location IDs" in text
        assert "- ctx_obj - Found objects" in text

    def test_from_summary_string(self):
        raw = (
            "Index map:\n"
            "- ctx_data_001 - Search results for KV stores\n"
            "- ctx_data_002 - API response data\n"
            "Status: Completed first phase"
        )
        parsed = IndexedSummary.from_summary_string(raw)
        assert len(parsed.index_map) == 2
        assert parsed.index_map[0].index == "ctx_data_001"
        assert parsed.index_map[0].description == "Search results for KV stores"
        assert "Completed first phase" in parsed.summary

    def test_empty_index_map(self):
        summary = IndexedSummary(summary="Just started.")
        assert len(summary.index_map) == 0
        assert "Just started" in summary.to_prompt_text()


# ── ContextStatus Tests ──────────────────────────────────────────────────

class TestContextStatus:
    def test_no_warning_below_threshold(self):
        status = ContextStatus.compute(
            working_tokens=3000, total_tokens=5000, threshold=8000
        )
        assert status.warning is None

    def test_notice_at_80_percent(self):
        status = ContextStatus.compute(
            working_tokens=6500, total_tokens=8000, threshold=8000
        )
        assert status.warning is not None
        assert "NOTICE" in status.warning

    def test_warning_above_threshold(self):
        status = ContextStatus.compute(
            working_tokens=9000, total_tokens=12000, threshold=8000
        )
        assert "WARNING" in status.warning

    def test_to_message_text(self):
        status = ContextStatus(
            working_tokens=5000, total_tokens=7000, threshold=8000
        )
        text = status.to_message_text()
        assert "working tokens=5000" in text
        assert "threshold=8000" in text


# ── ToolCall Tests ────────────────────────────────────────────────────────

class TestToolCall:
    def test_signature(self):
        tc = ToolCall(name="execute_action", arguments={"action": "go to desk 1"})
        sig = tc.signature()
        assert sig[0] == "execute_action"
        assert '"action": "go to desk 1"' in sig[1]

    def test_tool_definitions_exist(self):
        assert len(MEMEX_TOOLS) == 4
        names = [t.name for t in MEMEX_TOOLS]
        assert "CompressExperience" in names
        assert "ReadExperience" in names
        assert "execute_action" in names
        assert "finish" in names

    def test_json_schema_generation(self):
        schema = COMPRESS_EXPERIENCE_TOOL.to_json_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "CompressExperience"
        params = schema["function"]["parameters"]
        assert "summary" in params["properties"]
        assert "db_blocks" in params["properties"]
        assert "summary" in params["required"]


# ── Trajectory Tests ──────────────────────────────────────────────────────

class TestTrajectory:
    def _make_step(self, tool_name: str = "execute_action") -> Step:
        return Step(
            thinking="I should do something",
            tool_call=ToolCall(name=tool_name, arguments={"action": "look"}),
            observation="You see a room.",
            context_tokens=1000,
        )

    def test_episode_total_steps(self):
        ep = Episode(
            segments=[
                Segment(segment_idx=0, steps=[self._make_step(), self._make_step()]),
                Segment(segment_idx=1, steps=[self._make_step()]),
            ],
            task_success=True,
            terminal_reward=1.0,
        )
        assert ep.total_steps == 3
        assert ep.num_compressions == 1

    def test_episode_num_read_experience(self):
        ep = Episode(
            segments=[
                Segment(segment_idx=0, steps=[
                    self._make_step("execute_action"),
                    self._make_step("ReadExperience"),
                    self._make_step("ReadExperience"),
                ]),
            ],
        )
        assert ep.num_read_experience == 2

    def test_all_steps_flattens(self):
        ep = Episode(
            segments=[
                Segment(segment_idx=0, steps=[self._make_step()]),
                Segment(segment_idx=1, steps=[self._make_step(), self._make_step()]),
            ],
        )
        assert len(ep.all_steps()) == 3


# ── Tokenizer Tests ──────────────────────────────────────────────────────

class TestTokenizer:
    def test_count_basic(self):
        tok = Tokenizer()
        count = tok.count("Hello, world!")
        assert count > 0
        assert count < 10  # Should be ~4 tokens

    def test_count_empty(self):
        tok = Tokenizer()
        assert tok.count("") == 0

    def test_truncate_no_op(self):
        tok = Tokenizer()
        text = "short"
        assert tok.truncate(text, 100) == text

    def test_truncate_cuts(self):
        tok = Tokenizer()
        long_text = "word " * 500  # ~500 tokens
        result = tok.truncate(long_text, 50)
        assert result.endswith("...[truncated]")
        # Verify truncated version is within budget
        assert tok.count(result) <= 55  # Small margin for marker

    def test_truncate_empty(self):
        tok = Tokenizer()
        assert tok.truncate("", 10) == ""

    def test_count_messages(self):
        tok = Tokenizer()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        count = tok.count_messages(messages)
        assert count > 0

    def test_fallback_encoding(self):
        tok = Tokenizer(model="nonexistent-model-xyz")
        assert tok.encoding_name == "cl100k_base"
        assert tok.count("hello") > 0
