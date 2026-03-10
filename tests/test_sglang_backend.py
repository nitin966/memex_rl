"""Unit tests for SGLang backend."""

from __future__ import annotations

import pytest

from src.llm.backend import LLMBackend
from src.llm.sglang_backend import SGLangBackend
from src.models.memory import Message, MessageRole


class TestSGLangBackendImport:
    """Tests that SGLangBackend can be imported and validates init."""

    def test_import_without_openai(self):
        """If openai is not installed, a helpful error is raised."""
        try:
            # We assume openai is installed in the test env, so this should pass
            backend = SGLangBackend(
                model="test-model",
                base_url="http://localhost:30000/v1",
                api_key="EMPTY",
            )
            assert backend.model_name == "test-model"
            assert isinstance(backend, LLMBackend)
        except ImportError as e:
            assert "pip install openai" in str(e)


# Note: We do not run live SGLang integration tests by default like Ollama
# because setting up a local SGLang worker requires dedicated GPU resources.
# To test live, spawn: python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct
