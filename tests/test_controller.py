"""Unit tests for MemoryController (PR 3)."""

from __future__ import annotations

import pytest

from src.memory.controller import MemoryController
from src.memory.store import DictStore
from src.memory.tokenizer import Tokenizer
from src.models.memory import MemoryBlock, MessageRole


class TestMemoryController:
    """Tests for the full MemoryController lifecycle."""

    def setup_method(self):
        self.store = DictStore()
        self.tokenizer = Tokenizer()
        self.ctrl = MemoryController(
            store=self.store,
            tokenizer=self.tokenizer,
            context_window=32_768,
            threshold=8_000,
            summary_max_tokens=300,
        )
        self.ctrl.reset(
            system_prompt="You are a helpful agent.",
            task_instruction="Find the butterknife and put it on the table.",
        )

    # ── Initialization ─────────────────────────────────────────────────

    def test_reset_clears_state(self):
        self.ctrl.append_assistant("thinking...")
        self.ctrl.reset("new prompt", "new task")
        assert self.ctrl.num_working_messages == 0
        assert self.store.size() == 0

    def test_initial_messages(self):
        msgs = self.ctrl.get_messages()
        assert len(msgs) == 2  # system + user
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[1].role == MessageRole.USER

    # ── CompressExperience ─────────────────────────────────────────────

    def test_compress_archives_blocks(self):
        """CompressExperience should write blocks to L2 store."""
        blocks = [
            MemoryBlock(db_index="ctx_loc", db_content="room1, room2, room3"),
            MemoryBlock(db_index="ctx_obj", db_content="butterknife at counter"),
        ]
        result = self.ctrl.compress_experience(
            summary="Index map:\n- ctx_loc - Location IDs\n- ctx_obj - Found objects\nStatus: searching",
            memory_blocks=blocks,
        )
        assert "2 block(s) archived" in result
        assert self.store.read("ctx_loc") == "room1, room2, room3"
        assert self.store.read("ctx_obj") == "butterknife at counter"

    def test_compress_rewrites_context(self):
        """After compression, M should be [m₀, u, summary]."""
        self.ctrl.append_assistant("some old reasoning")
        self.ctrl.append_tool_result("some old observation")
        assert self.ctrl.num_working_messages == 2

        self.ctrl.compress_experience(
            summary="Compressed state",
            memory_blocks=[MemoryBlock(db_index="x", db_content="data")],
        )

        # Working context should have exactly 1 message (the summary)
        assert self.ctrl.num_working_messages == 1
        msgs = self.ctrl.get_messages()
        assert len(msgs) == 3  # system + task + summary
        assert msgs[2].content == "Compressed state"

    def test_compress_truncates_long_summary(self):
        """Summary exceeding summary_max_tokens should be truncated."""
        ctrl = MemoryController(
            store=DictStore(),
            tokenizer=self.tokenizer,
            summary_max_tokens=10,  # Very small for testing
        )
        ctrl.reset("sys", "task")

        long_summary = "word " * 100  # Way more than 10 tokens
        ctrl.compress_experience(summary=long_summary, memory_blocks=[])

        msgs = ctrl.get_messages()
        summary_text = msgs[2].content
        assert summary_text.endswith("...[truncated]")
        assert self.tokenizer.count(summary_text) <= 15  # 10 + marker

    # ── ReadExperience ─────────────────────────────────────────────────

    def test_read_existing_index(self):
        """ReadExperience should return content and append to M."""
        self.store.write("ctx_data", "important evidence")
        content = self.ctrl.read_experience("ctx_data")
        assert content == "important evidence"

        # Should be appended to working context
        msgs = self.ctrl.get_messages()
        last = msgs[-1]
        assert last.role == MessageRole.TOOL
        assert last.content == "important evidence"

    def test_read_nonexistent_index(self):
        """ReadExperience with bad index should return error message."""
        content = self.ctrl.read_experience("missing_idx")
        assert "not found" in content
        assert "missing_idx" in content

    # ── Compress → Read Cycle ──────────────────────────────────────────

    def test_full_compress_read_cycle(self):
        """End-to-end: archive via compress, retrieve via read."""
        # Simulate some agent work
        self.ctrl.append_assistant("Let me look around")
        self.ctrl.append_tool_result("You see: desk, chair, lamp")
        self.ctrl.append_assistant("I see objects")
        self.ctrl.append_tool_result("butterknife is on desk")

        # Compress
        self.ctrl.compress_experience(
            summary="Index map:\n- ctx_room - Room contents\nStatus: Found butterknife",
            memory_blocks=[
                MemoryBlock(db_index="ctx_room", db_content="desk, chair, lamp, butterknife on desk"),
            ],
        )

        # Working context should be small now
        assert self.ctrl.num_working_messages == 1

        # Read back
        content = self.ctrl.read_experience("ctx_room")
        assert "butterknife on desk" in content

    # ── Context Status ─────────────────────────────────────────────────

    def test_context_status_below_threshold(self):
        status = self.ctrl.get_context_status()
        assert status.working_tokens >= 0
        assert status.threshold == 8_000
        assert status.warning is None  # Should be well below threshold

    def test_context_status_warning_above_threshold(self):
        # Stuff context with lots of text to exceed threshold
        for _ in range(50):
            self.ctrl.append_assistant("x " * 200)

        status = self.ctrl.get_context_status()
        assert status.warning is not None
        assert "WARNING" in status.warning

    def test_inject_context_status_appends_message(self):
        count_before = self.ctrl.num_working_messages
        self.ctrl.inject_context_status()
        assert self.ctrl.num_working_messages == count_before + 1
        msgs = self.ctrl.get_messages()
        last = msgs[-1]
        assert last.role == MessageRole.SYSTEM
        assert "[Context Status:" in last.content

    # ── Message Management ─────────────────────────────────────────────

    def test_append_assistant(self):
        self.ctrl.append_assistant("I'm thinking...")
        msgs = self.ctrl.get_messages()
        assert msgs[-1].role == MessageRole.ASSISTANT
        assert msgs[-1].content == "I'm thinking..."

    def test_append_tool_result(self):
        self.ctrl.append_tool_result("observation data", tool_name="execute_action")
        msgs = self.ctrl.get_messages()
        assert msgs[-1].role == MessageRole.TOOL
        assert msgs[-1].name == "execute_action"

    def test_last_observation(self):
        self.ctrl.append_assistant("thinking")
        self.ctrl.append_tool_result("tool output here")
        assert self.ctrl.last_observation() == "tool output here"

    def test_last_observation_empty(self):
        assert self.ctrl.last_observation() == ""

    def test_get_messages_as_dicts(self):
        self.ctrl.append_assistant("hello")
        dicts = self.ctrl.get_messages_as_dicts()
        assert all(isinstance(d, dict) for d in dicts)
        assert all("role" in d and "content" in d for d in dicts)

    # ── Anchor-Based Compression ──────────────────────────────────────

    def test_compress_with_anchor_blocks(self):
        """Option B: anchor-based extraction from conversation."""
        self.ctrl.append_tool_result(
            "def hello():\n    greeting = 'Hi there'\n    return greeting"
        )

        blocks = [
            MemoryBlock(
                db_index="ctx_code",
                start_anchor="def hello():",
                mid_anchor="greeting = 'Hi there'",
                end_anchor="return greeting",
            ),
        ]
        result = self.ctrl.compress_experience(summary="Code archived", memory_blocks=blocks)
        assert "1 block(s) archived" in result

        stored = self.store.read("ctx_code")
        assert "def hello():" in stored
        assert "return greeting" in stored

    def test_compress_with_bad_anchors_reports_error(self):
        """Failed anchor extraction should report error but not crash."""
        self.ctrl.append_tool_result("some content")
        blocks = [
            MemoryBlock(
                db_index="ctx_fail",
                start_anchor="NONEXISTENT",
                mid_anchor="nope",
                end_anchor="also nope",
            ),
        ]
        result = self.ctrl.compress_experience(summary="Partial", memory_blocks=blocks)
        assert "failed" in result
        assert self.store.read("ctx_fail") is None

    # ── Token Counting ─────────────────────────────────────────────────

    def test_working_token_count_increases(self):
        before = self.ctrl.working_token_count()
        self.ctrl.append_assistant("A significant amount of text. " * 50)
        after = self.ctrl.working_token_count()
        assert after > before

    def test_total_includes_system_and_task(self):
        total = self.ctrl.total_token_count()
        working = self.ctrl.working_token_count()
        # Total should be strictly greater than working (system + task > 0)
        assert total > working

    # ── get_prefix ─────────────────────────────────────────────────────

    def test_get_prefix_includes_summary_after_compression(self):
        self.ctrl.compress_experience(
            summary="Compressed summary here",
            memory_blocks=[],
        )
        prefix = self.ctrl.get_prefix()
        assert "You are a helpful agent" in prefix
        assert "butterknife" in prefix
        assert "Compressed summary here" in prefix
