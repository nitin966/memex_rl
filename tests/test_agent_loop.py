"""Integration tests for the MemexAgent loop (PR 5).

Uses mock LLM and mock environment to test the full Algorithm 1
execution loop without requiring any real LLM or env dependencies.
"""

from __future__ import annotations

import pytest

from src.agent.loop import AgentConfig, MemexAgent
from src.environments.base import Environment, StepResult
from src.llm.backend import LLMBackend
from src.models.memory import Message


# ── Mock LLM Backend ──────────────────────────────────────────────────────

class MockLLM(LLMBackend):
    """Mock LLM that returns pre-scripted responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_idx = 0

    def generate(self, messages: list[Message], temperature: float = 0.7,
                 max_tokens: int = 4096) -> str:
        if self._call_idx >= len(self._responses):
            # Auto-finish if we run out of scripted responses
            return (
                'No more scripted responses. Finishing.\n'
                '<tool_call>\n'
                '{"name": "finish", "arguments": {"success": false}}\n'
                '</tool_call>'
            )
        response = self._responses[self._call_idx]
        self._call_idx += 1
        return response

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def model_name(self) -> str:
        return "mock-llm"


# ── Mock Environment ──────────────────────────────────────────────────────

class MockEnvironment(Environment):
    """Mock household environment for testing."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._task_id = "test_task_001"
        self._task = "Put the book on the desk."
        self._done = False

    def reset(self, task_id: str | None = None) -> str:
        self._done = False
        if task_id:
            self._task_id = task_id
        return self._task

    def step(self, action: str) -> StepResult:
        obs = self._responses.get(action, f"You performed: {action}")
        return StepResult(observation=obs)

    def get_task_id(self) -> str:
        return self._task_id


# ── Tests ─────────────────────────────────────────────────────────────────

class TestMemexAgentLoop:
    """Integration tests for the full agent loop."""

    def test_simple_finish(self):
        """Agent immediately finishes — simplest possible episode."""
        llm = MockLLM([
            'Task is trivial.\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>'
        ])
        env = MockEnvironment()
        agent = MemexAgent(llm=llm, environment=env, config=AgentConfig(max_steps=10))

        episode = agent.run_episode()
        assert episode.task_success is True
        assert episode.terminal_reward == 1.0
        assert episode.total_steps == 1
        assert episode.num_compressions == 0

    def test_env_action_then_finish(self):
        """Agent explores then finishes."""
        llm = MockLLM([
            'Let me look around.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n'
            '</tool_call>',

            'I see the book on shelf 1. Let me get it.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "pick up book 1"}}\n'
            '</tool_call>',

            'Done!\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>',
        ])
        env = MockEnvironment(responses={
            "look": "You see: shelf 1, desk 1, lamp 1",
            "pick up book 1": "You picked up book 1.",
        })
        agent = MemexAgent(llm=llm, environment=env, config=AgentConfig(max_steps=10))

        episode = agent.run_episode()
        assert episode.task_success is True
        assert episode.total_steps == 3
        assert episode.num_compressions == 0
        assert len(episode.segments) == 1

    def test_compress_then_read_then_finish(self):
        """Agent compresses, reads back, then finishes — full Memex cycle."""
        llm = MockLLM([
            # Step 1: look
            'Look around.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n'
            '</tool_call>',

            # Step 2: compress
            'Context getting large, compressing.\n'
            '<tool_call>\n'
            '{"name": "CompressExperience", "arguments": '
            '{"summary": "Index map:\\n- ctx_loc - Location IDs\\nStatus: Explored room", '
            '"db_blocks": [{"db_index": "ctx_loc", "db_content": "shelf_1, desk_1, lamp_1"}]}}\n'
            '</tool_call>',

            # Step 3: read
            'Need locations back.\n'
            '<tool_call>\n'
            '{"name": "ReadExperience", "arguments": {"db_index": "ctx_loc"}}\n'
            '</tool_call>',

            # Step 4: finish
            'All done.\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>',
        ])
        env = MockEnvironment(responses={"look": "You see: shelf_1, desk_1, lamp_1"})
        agent = MemexAgent(llm=llm, environment=env, config=AgentConfig(max_steps=10))

        episode = agent.run_episode()
        assert episode.task_success is True
        assert episode.total_steps == 4
        assert episode.num_compressions == 1
        assert len(episode.segments) == 2  # Before and after compression
        assert episode.num_read_experience == 1

        # Check ReadExperience returned the right content
        # CompressExperience step is recorded as first step of new segment
        read_step = episode.segments[1].steps[1]  # Second step after compression

        assert read_step.tool_call.name == "ReadExperience"
        assert "shelf_1" in read_step.observation

    def test_failed_episode(self):
        """Agent fails the task."""
        llm = MockLLM([
            'I give up.\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": false}}\n'
            '</tool_call>'
        ])
        env = MockEnvironment()
        agent = MemexAgent(llm=llm, environment=env)

        episode = agent.run_episode()
        assert episode.task_success is False
        assert episode.terminal_reward == 0.0

    def test_max_steps_termination(self):
        """Agent hits T_max without finishing."""
        # LLM always does "look" — will hit max_steps
        llm = MockLLM([
            'Looking.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n'
            '</tool_call>'
        ] * 5)
        env = MockEnvironment(responses={"look": "Same room."})
        agent = MemexAgent(llm=llm, environment=env, config=AgentConfig(max_steps=3))

        episode = agent.run_episode()
        assert episode.task_success is False  # Never finished
        assert episode.total_steps == 3

    def test_malformed_tool_call_recovery(self):
        """Agent sends malformed output, then recovers with valid call."""
        llm = MockLLM([
            # Bad call: no tags
            'Just thinking out loud, no tool call here.',

            # Good call: proper format
            'OK let me try again.\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>',
        ])
        env = MockEnvironment()
        agent = MemexAgent(llm=llm, environment=env, config=AgentConfig(max_steps=10))

        episode = agent.run_episode()
        assert episode.task_success is True
        assert episode.total_steps == 2
        # First step should have format errors
        first_step = episode.all_steps()[0]
        assert len(first_step.format_errors) > 0
        assert first_step.tool_call.name == "_error"

    def test_multiple_compressions(self):
        """Agent compresses multiple times — creates multiple segments."""
        llm = MockLLM([
            # Step 1: action
            'Explore.\n<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n</tool_call>',

            # Step 2: compress #1
            'Compress.\n<tool_call>\n'
            '{"name": "CompressExperience", "arguments": '
            '{"summary": "Phase 1 done", "db_blocks": [{"db_index": "p1", "db_content": "data1"}]}}\n</tool_call>',

            # Step 3: more action
            'Continue.\n<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "go to desk 1"}}\n</tool_call>',

            # Step 4: compress #2
            'Compress again.\n<tool_call>\n'
            '{"name": "CompressExperience", "arguments": '
            '{"summary": "Phase 2 done", "db_blocks": [{"db_index": "p2", "db_content": "data2"}]}}\n</tool_call>',

            # Step 5: finish
            'Done.\n<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n</tool_call>',
        ])
        env = MockEnvironment(responses={"look": "room", "go to desk 1": "at desk"})
        agent = MemexAgent(llm=llm, environment=env, config=AgentConfig(max_steps=10))

        episode = agent.run_episode()
        assert episode.task_success is True
        assert episode.num_compressions == 2
        assert len(episode.segments) == 3  # k=2 compressions → k+1=3 segments

    def test_context_tokens_tracked(self):
        """Each step records its context token count."""
        llm = MockLLM([
            'Hello.\n<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "look"}}\n</tool_call>',

            'Done.\n<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n</tool_call>',
        ])
        env = MockEnvironment(responses={"look": "A room with things."})
        agent = MemexAgent(llm=llm, environment=env)

        episode = agent.run_episode()
        for step in episode.all_steps():
            assert step.context_tokens > 0

    def test_episode_has_task_id(self):
        """Episode records the task ID from the environment."""
        llm = MockLLM([
            'Finish.\n<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n</tool_call>'
        ])
        env = MockEnvironment()
        agent = MemexAgent(llm=llm, environment=env)

        episode = agent.run_episode(task_id="custom_task_42")
        assert episode.task_id == "custom_task_42"
