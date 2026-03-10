"""Unit tests for ToolParser and prompt templates (PR 4)."""

from __future__ import annotations

import pytest

from src.agent.tool_parser import ToolParser, ParseResult
from src.agent.prompts import (
    build_system_prompt,
    ALFWORLD_ENVIRONMENT_PROMPT,
    ALFWORLD_MEMORY_ADDENDUM,
    MEMEX_SYSTEM_PROMPT,
)


class TestToolParser:
    """Tests for <tool_call> tag parsing and format error detection."""

    def setup_method(self):
        self.parser = ToolParser()

    # ── Well-Formed Tool Calls ─────────────────────────────────────────

    def test_parse_valid_tool_call(self):
        raw = (
            'I should navigate to the desk.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "go to desk 1"}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call is not None
        assert result.tool_call.name == "execute_action"
        assert result.tool_call.arguments == {"action": "go to desk 1"}
        assert "navigate to the desk" in result.thinking
        assert len(result.format_errors) == 0

    def test_parse_compress_experience(self):
        raw = (
            'Context is getting large, need to compress.\n'
            '<tool_call>\n'
            '{"name": "CompressExperience", "arguments": '
            '{"summary": "Index map:\\n- ctx_loc - Locations", '
            '"db_blocks": [{"db_index": "ctx_loc", "db_content": "room1, room2"}]}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call is not None
        assert result.tool_call.name == "CompressExperience"
        assert "db_blocks" in result.tool_call.arguments
        assert len(result.format_errors) == 0

    def test_parse_read_experience(self):
        raw = (
            'Let me retrieve the locations.\n'
            '<tool_call>\n'
            '{"name": "ReadExperience", "arguments": {"db_index": "ctx_loc"}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call.name == "ReadExperience"
        assert result.tool_call.arguments["db_index"] == "ctx_loc"

    def test_parse_finish(self):
        raw = (
            'Task is done.\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call.name == "finish"
        assert result.tool_call.arguments["success"] is True

    def test_parse_no_thinking_text(self):
        raw = (
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.thinking == ""
        assert result.tool_call is not None

    def test_parse_multiline_thinking(self):
        raw = (
            'First, I need to think about this.\n'
            'The task requires me to find the book.\n'
            'Let me go to the shelf.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "go to shelf 1"}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert "find the book" in result.thinking
        assert "go to the shelf" in result.thinking

    # ── Format Errors ──────────────────────────────────────────────────

    def test_error_empty_output(self):
        result = self.parser.parse("")
        assert result.tool_call is None
        assert len(result.format_errors) > 0

    def test_error_no_tool_call_tags(self):
        result = self.parser.parse("I'm just thinking out loud.")
        assert result.tool_call is None
        assert any("No <tool_call> tags" in e for e in result.format_errors)
        assert result.thinking == "I'm just thinking out loud."

    def test_error_open_without_close(self):
        raw = (
            'Thinking...\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n'
        )
        result = self.parser.parse(raw)
        assert any("mismatch" in e.lower() or "without closing" in e.lower()
                    for e in result.format_errors)

    def test_error_close_without_open(self):
        raw = (
            '{"name": "execute_action", "arguments": {"action": "look"}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert any("mismatch" in e.lower() or "without opening" in e.lower()
                    for e in result.format_errors)

    def test_error_invalid_json(self):
        raw = (
            '<tool_call>\n'
            '{name: execute_action, not valid json}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call is None
        assert any("Invalid JSON" in e for e in result.format_errors)

    def test_error_missing_name(self):
        raw = (
            '<tool_call>\n'
            '{"arguments": {"action": "look"}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert any("Missing required field 'name'" in e for e in result.format_errors)

    def test_error_missing_arguments(self):
        raw = (
            '<tool_call>\n'
            '{"name": "execute_action"}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert any("Missing required field 'arguments'" in e for e in result.format_errors)

    def test_error_name_not_string(self):
        raw = (
            '<tool_call>\n'
            '{"name": 42, "arguments": {}}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call is None
        assert any("non-empty string" in e for e in result.format_errors)

    def test_error_arguments_not_dict(self):
        raw = (
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": "not a dict"}\n'
            '</tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call is None
        assert any("must be an object" in e for e in result.format_errors)

    # ── count_malformed ────────────────────────────────────────────────

    def test_count_malformed_zero_for_valid(self):
        raw = (
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>'
        )
        assert self.parser.count_malformed(raw) == 0

    def test_count_malformed_nonzero_for_invalid(self):
        raw = '<tool_call>\nnot json\n</tool_call>'
        assert self.parser.count_malformed(raw) > 0

    # ── Edge Cases ─────────────────────────────────────────────────────

    def test_extra_whitespace_in_tags(self):
        raw = (
            '<tool_call>  \n'
            '  {"name": "execute_action", "arguments": {"action": "look"}}  \n'
            '  </tool_call>'
        )
        result = self.parser.parse(raw)
        assert result.tool_call is not None
        assert result.tool_call.name == "execute_action"

    def test_preserves_raw_text(self):
        raw = 'thinking\n<tool_call>\n{"name": "finish", "arguments": {"success": true}}\n</tool_call>'
        result = self.parser.parse(raw)
        assert result.raw_text == raw


class TestPrompts:
    """Tests for system prompt construction."""

    def test_build_default_prompt(self):
        prompt = build_system_prompt()
        assert "CompressExperience" in prompt
        assert "ReadExperience" in prompt
        assert "<tool_call>" in prompt
        assert "MANDATORY REQUIREMENTS" in prompt

    def test_build_with_environment(self):
        prompt = build_system_prompt(environment_prompt=ALFWORLD_ENVIRONMENT_PROMPT)
        assert "ALFWorld" in prompt
        assert "go to desk" in prompt

    def test_build_with_memory_addendum(self):
        prompt = build_system_prompt(memory_management_addendum=ALFWORLD_MEMORY_ADDENDUM)
        assert "MEMORY MANAGEMENT" in prompt
        assert "db_blocks" in prompt

    def test_build_alfworld_full(self):
        prompt = build_system_prompt(
            environment_prompt=ALFWORLD_ENVIRONMENT_PROMPT,
            memory_management_addendum=ALFWORLD_MEMORY_ADDENDUM,
        )
        assert "ALFWorld" in prompt
        assert "ReadExperience" in prompt
        assert "MEMORY MANAGEMENT" in prompt

    def test_base_prompt_has_all_sections(self):
        assert "execute_action" in MEMEX_SYSTEM_PROMPT
        assert "finish" in MEMEX_SYSTEM_PROMPT
        assert "Context Status" in MEMEX_SYSTEM_PROMPT
        assert "SEVERE PENALTIES" in MEMEX_SYSTEM_PROMPT
