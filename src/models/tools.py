"""Tool definitions and call models for Memex(RL).

Defines the JSON schemas for all tools available to the Memex agent:
- CompressExperience: archive content to L2, rewrite working context
- ReadExperience: dereference index from L2
- execute_action: generic environment interaction
- finish: episode termination
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A parsed tool invocation from the agent's output."""
    name: str = Field(description="Tool function name.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments as key-value pairs.",
    )
    raw_text: str = Field(
        default="",
        description="Original raw text of the tool call (for debugging).",
    )

    def signature(self) -> tuple[str, str]:
        """Return (name, sorted_args_json) for redundancy detection."""
        import json
        return (self.name, json.dumps(self.arguments, sort_keys=True))


class ToolParameter(BaseModel):
    """A single parameter in a tool's JSON schema."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: list[str] | None = None
    items: dict[str, Any] | None = None  # For array types


class ToolDefinition(BaseModel):
    """JSON Schema definition for a tool available to the agent."""
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function/tool JSON schema."""
        properties = {}
        required = []
        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Pre-defined tool definitions matching the paper's specification
# ---------------------------------------------------------------------------

COMPRESS_EXPERIENCE_TOOL = ToolDefinition(
    name="CompressExperience",
    description=(
        "Compress working context to database for later retrieval. "
        "Replaces all messages (except system prompt and task description) "
        "with your summary. Use when working context approaches threshold."
    ),
    parameters=[
        ToolParameter(
            name="summary",
            type="string",
            description=(
                "Index map listing ALL stored indices (old and new). "
                "Format: '- <db_index> - <what it contains>' per line. "
                "Include current status and next steps at the end."
            ),
            required=True,
        ),
        ToolParameter(
            name="db_blocks",
            type="array",
            description="List of content blocks to store in the experience database.",
            required=True,
            items={
                "type": "object",
                "properties": {
                    "db_index": {
                        "type": "string",
                        "description": "Unique key, e.g. 'ctx_code_001'",
                    },
                    "db_content": {
                        "type": "string",
                        "description": "Content you write/summarize (Option A)",
                    },
                    "start_anchor": {
                        "type": "string",
                        "description": "Exact text where extraction STARTS (Option B)",
                    },
                    "mid_anchor": {
                        "type": "string",
                        "description": "Exact text in MIDDLE for verification (Option B)",
                    },
                    "end_anchor": {
                        "type": "string",
                        "description": "Exact text where extraction ENDS (Option B)",
                    },
                },
                "required": ["db_index"],
            },
        ),
    ],
)

READ_EXPERIENCE_TOOL = ToolDefinition(
    name="ReadExperience",
    description=(
        "Retrieve previously compressed content by index. "
        "Use when you need exact details stored during compression. "
        "Always retrieve instead of re-running tools for the same information."
    ),
    parameters=[
        ToolParameter(
            name="db_index",
            type="string",
            description="The index to retrieve. Must match exactly.",
            required=True,
        ),
    ],
)

EXECUTE_ACTION_TOOL = ToolDefinition(
    name="execute_action",
    description="Execute an action in the environment.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="The action to execute in natural language.",
            required=True,
        ),
    ],
)

FINISH_TOOL = ToolDefinition(
    name="finish",
    description="Indicate task completion.",
    parameters=[
        ToolParameter(
            name="success",
            type="boolean",
            description="Whether the task was completed successfully.",
            required=True,
        ),
    ],
)

# All Memex tools in the order they should appear in the prompt
MEMEX_TOOLS: list[ToolDefinition] = [
    EXECUTE_ACTION_TOOL,
    FINISH_TOOL,
    COMPRESS_EXPERIENCE_TOOL,
    READ_EXPERIENCE_TOOL,
]
