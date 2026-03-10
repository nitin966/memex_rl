"""Tool call parser for Memex(RL).

Parses the agent's raw text output to extract:
  - Thinking/reasoning text (free-form text before the tool call)
  - Tool call (structured JSON inside <tool_call> tags)
  - Format errors (for the P_format penalty in MemexRL rewards)

Format errors detected (from paper Section 3.3):
  1. Tag mismatches: <tool_call> without closing </tool_call>
  2. Invalid JSON within tool call tags
  3. Missing required fields (name, arguments)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.models.tools import ToolCall


@dataclass
class ParseResult:
    """Result of parsing an agent's raw output."""
    thinking: str = ""
    tool_call: ToolCall | None = None
    format_errors: list[str] = field(default_factory=list)
    raw_text: str = ""


class ToolParser:
    """Parses <tool_call>...</tool_call> tags from agent output.

    Expected format (from paper Appendix B):
        Some reasoning text...
        <tool_call>
        {"name": "tool_name", "arguments": {"param1": "value1"}}
        </tool_call>

    The parser is intentionally lenient on whitespace/newlines within tags
    but strict on structural correctness (matching tags, valid JSON, required fields).
    """

    # Regex patterns
    _TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    _OPEN_TAG = "<tool_call>"
    _CLOSE_TAG = "</tool_call>"

    def parse(self, raw_output: str) -> ParseResult:
        """Parse agent output into thinking text, tool call, and format errors.

        Args:
            raw_output: The raw text output from the LLM.

        Returns:
            ParseResult with extracted components.
        """
        result = ParseResult(raw_text=raw_output)
        errors: list[str] = []

        if not raw_output or not raw_output.strip():
            errors.append("Empty output — no tool call found.")
            result.format_errors = errors
            return result

        # Check for tag presence and matching
        has_open = self._OPEN_TAG in raw_output
        has_close = self._CLOSE_TAG in raw_output

        if not has_open and not has_close:
            # No tool call tags at all — entire output is thinking
            result.thinking = raw_output.strip()
            errors.append("No <tool_call> tags found in output.")
            result.format_errors = errors
            return result

        if has_open and not has_close:
            # Small models often stop generating right before the closing tag.
            # We auto-append it to salvage the JSON silently without penalty.
            raw_output += f"\n{self._CLOSE_TAG}"
            has_close = True
        elif has_close and not has_open:
            errors.append("Tag mismatch: </tool_call> without opening <tool_call>.")

        # Try to extract tool call via regex
        match = self._TOOL_CALL_PATTERN.search(raw_output)

        if match:
            # Everything before the <tool_call> tag is thinking
            thinking_end = raw_output.find(self._OPEN_TAG)
            result.thinking = raw_output[:thinking_end].strip()

            # Parse the JSON inside the tags
            json_str = match.group(1).strip()
            # Clean involuntary Markdown JSON blocks from smaller models
            if json_str.startswith("```json") and json_str.endswith("```"):
                json_str = json_str[len("```json"):-3].strip()
            elif json_str.startswith("```") and json_str.endswith("```"):
                json_str = json_str[3:-3].strip()

            tool_call, json_errors = self._parse_tool_json(json_str)
            errors.extend(json_errors)
            if tool_call:
                result.tool_call = tool_call
        else:
            # Tags exist but regex didn't match (malformed)
            if has_open:
                thinking_end = raw_output.find(self._OPEN_TAG)
                result.thinking = raw_output[:thinking_end].strip()

                # Try to salvage JSON between open tag and end of string
                after_open = raw_output[thinking_end + len(self._OPEN_TAG):]
                # Strip a close tag if it appears (may be malformed)
                if self._CLOSE_TAG in after_open:
                    after_open = after_open[:after_open.find(self._CLOSE_TAG)]
                json_str = after_open.strip()
                # Clean involuntary Markdown blocks
                if json_str.startswith("```json") and json_str.endswith("```"):
                    json_str = json_str[len("```json"):-3].strip()
                elif json_str.startswith("```") and json_str.endswith("```"):
                    json_str = json_str[3:-3].strip()

                if json_str:
                    tool_call, json_errors = self._parse_tool_json(json_str)
                    errors.extend(json_errors)
                    if tool_call:
                        result.tool_call = tool_call
            else:
                result.thinking = raw_output.strip()

        result.format_errors = errors
        return result

    def _parse_tool_json(self, json_str: str) -> tuple[ToolCall | None, list[str]]:
        """Parse JSON string into a ToolCall, collecting errors."""
        errors: list[str] = []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in tool call: {e}")
            return None, errors

        if not isinstance(data, dict):
            errors.append(f"Tool call JSON must be an object, got {type(data).__name__}.")
            return None, errors

        # Check required fields
        if "name" not in data:
            errors.append("Missing required field 'name' in tool call.")
        if "arguments" not in data:
            errors.append("Missing required field 'arguments' in tool call.")

        name = data.get("name", "")
        arguments = data.get("arguments", {})

        if not isinstance(name, str) or not name:
            errors.append(f"Field 'name' must be a non-empty string, got: {name!r}")
            return None, errors

        if not isinstance(arguments, dict):
            errors.append(f"Field 'arguments' must be an object, got {type(arguments).__name__}.")
            return None, errors

        return ToolCall(name=name, arguments=arguments, raw_text=json_str), errors

    def count_malformed(self, raw_output: str) -> int:
        """Count the number of format errors in a raw output.

        Used directly by the reward engine for P_format computation.
        """
        result = self.parse(raw_output)
        return len(result.format_errors)
