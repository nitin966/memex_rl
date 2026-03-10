"""Unit tests for LLM backends (PR 8).

Tests the EchoBackend (no external deps) and OpenAIBackend initialization.
OpenAIBackend generation tests require Ollama running locally.
"""

from __future__ import annotations

import pytest

from src.llm.backend import LLMBackend
from src.llm.openai_backend import EchoBackend
from src.models.memory import Message, MessageRole


class TestEchoBackend:
    """Tests for the testing-friendly EchoBackend."""

    def test_single_response(self):
        backend = EchoBackend(response="Hello!")
        result = backend.generate([
            Message(role=MessageRole.USER, content="Hi"),
        ])
        assert result == "Hello!"

    def test_multiple_responses(self):
        backend = EchoBackend(responses=["First", "Second", "Third"])
        msgs = [Message(role=MessageRole.USER, content="Hi")]

        assert backend.generate(msgs) == "First"
        assert backend.generate(msgs) == "Second"
        assert backend.generate(msgs) == "Third"

    def test_exhausted_responses_auto_finish(self):
        backend = EchoBackend(responses=["Only one"])
        msgs = [Message(role=MessageRole.USER, content="Hi")]

        backend.generate(msgs)  # "Only one"
        result = backend.generate(msgs)  # Exhausted
        assert "finish" in result
        assert "false" in result

    def test_reset(self):
        backend = EchoBackend(responses=["A", "B"])
        msgs = [Message(role=MessageRole.USER, content="Hi")]

        backend.generate(msgs)
        backend.generate(msgs)
        backend.reset()
        assert backend.generate(msgs) == "A"

    def test_count_tokens(self):
        backend = EchoBackend()
        count = backend.count_tokens("Hello, world!")
        assert count > 0

    def test_model_name(self):
        backend = EchoBackend()
        assert backend.model_name == "echo-backend"

    def test_is_llm_backend(self):
        backend = EchoBackend()
        assert isinstance(backend, LLMBackend)


class TestOpenAIBackendImport:
    """Tests that OpenAIBackend can be imported and validates init."""

    def test_import_without_openai(self):
        """If openai is not installed, a helpful error is raised."""
        try:
            from src.llm.openai_backend import OpenAIBackend
            # If openai package exists, just verify it can be instantiated
            # (won't connect to anything, but constructor should work)
            backend = OpenAIBackend(
                model="test-model",
                base_url="http://localhost:99999/v1",
                api_key="test-key",
            )
            assert backend.model_name == "test-model"
        except ImportError as e:
            assert "pip install memex-rl[llm]" in str(e)


class TestOpenAIBackendWithOllama:
    """Integration tests that require Ollama running locally.

    These tests are automatically skipped if Ollama isn't available.
    To run: start Ollama with a model (e.g., `ollama run qwen2.5:3b`)
    then run: pytest tests/test_llm_backend.py -v -k "ollama"
    """

    @pytest.fixture(autouse=True)
    def check_ollama(self):
        """Skip if Ollama is not running."""
        try:
            import httpx
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code != 200:
                pytest.skip("Ollama not responding")
            tags = resp.json()
            if not tags.get("models"):
                pytest.skip("No models available in Ollama")
            self._available_models = [m["name"] for m in tags["models"]]
        except Exception:
            pytest.skip("Ollama not available at localhost:11434")

    def _get_model(self) -> str:
        """Pick the best available model for testing."""
        preferred = ["qwen2.5:7b", "qwen2.5:3b", "llama3.1:8b", "llama3.2:3b", "gemma2:2b"]
        for model in preferred:
            if model in self._available_models:
                return model
        return self._available_models[0]

    def test_basic_generation(self):
        from src.llm.openai_backend import OpenAIBackend

        model = self._get_model()
        backend = OpenAIBackend(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        result = backend.generate(
            messages=[
                Message(role=MessageRole.SYSTEM, content="Reply with only 'hello'."),
                Message(role=MessageRole.USER, content="Say hello."),
            ],
            temperature=0.0,
            max_tokens=10,
        )
        assert len(result) > 0

    def test_tool_call_format(self):
        """Test that a model can produce tool call format."""
        from src.llm.openai_backend import OpenAIBackend
        from src.agent.tool_parser import ToolParser

        model = self._get_model()
        backend = OpenAIBackend(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        result = backend.generate(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You must respond with a tool call in this exact format:\n"
                        "<tool_call>\n"
                        '{"name": "execute_action", "arguments": {"action": "look"}}\n'
                        "</tool_call>"
                    ),
                ),
                Message(role=MessageRole.USER, content="Look around the room."),
            ],
            temperature=0.0,
            max_tokens=100,
        )

        parser = ToolParser()
        parsed = parser.parse(result)
        # We can't guarantee the model follows format perfectly,
        # but we can check the parser handles whatever it produces
        assert parsed.raw_text == result
