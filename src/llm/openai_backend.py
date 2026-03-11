"""OpenAI-compatible LLM backend for Memex(RL).

Works out of the box with:
  - Ollama (http://localhost:11434/v1) — recommended for local testing
  - OpenAI API (https://api.openai.com/v1)
  - vLLM (http://localhost:8000/v1)
  - Any OpenAI-compatible server (LiteLLM, etc.)

For local testing with Ollama:
  backend = OpenAIBackend(
      model="qwen2.5:7b",
      base_url="http://localhost:11434/v1",
      api_key="ollama",  # Ollama ignores API key
  )
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.llm.backend import LLMBackend
from src.memory.tokenizer import Tokenizer
from src.models.memory import Message, MessageRole

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible LLM backend.

    Uses the OpenAI Python SDK to call any OpenAI-compatible endpoint.
    Works with Ollama, vLLM, OpenAI API, etc.

    Args:
        model: Model name (e.g. "qwen2.5:7b" for Ollama, "gpt-4" for OpenAI).
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var or OpenAI.
        api_key: API key. Defaults to OPENAI_API_KEY env var or "ollama".
        default_temperature: Default sampling temperature.
        default_max_tokens: Default max generation tokens.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str | None = None,
        api_key: str | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAIBackend requires the 'openai' package. "
                "Install with: pip install memex-rl[llm]"
            )

        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._tokenizer = Tokenizer(model=model)

        # Resolve connection parameters
        resolved_base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")

        self._client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )

        logger.info(
            f"OpenAIBackend initialized: model={model}, "
            f"base_url={resolved_base_url}"
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion from the model.

        Args:
            messages: Context window as Message objects.
            temperature: Sampling temperature (uses default if None).
            max_tokens: Max tokens to generate (uses default if None).

        Returns:
            Raw text output from the model.
        """
        # Convert Message objects to OpenAI-format dicts
        api_messages = []
        for msg in messages:
            d: dict[str, str] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            api_messages.append(d)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                temperature=temperature or self._default_temperature,
                max_tokens=max_tokens or self._default_max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                f"Generated {len(content)} chars, "
                f"usage: {response.usage}"
            )
            return content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (approximate for non-OpenAI models)."""
        return self._tokenizer.count(text)

    @property
    def model_name(self) -> str:
        return self._model


class EchoBackend(LLMBackend):
    """A testing backend that echoes a fixed response.

    Useful for unit/integration tests without any LLM dependency.
    """

    def __init__(self, response: str = "", responses: list[str] | None = None) -> None:
        self._responses = responses or [response]
        self._call_idx = 0
        self._tokenizer = Tokenizer()

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        if self._call_idx >= len(self._responses):
            # Auto-finish when responses exhausted
            return (
                'Responses exhausted.\n'
                '<tool_call>\n'
                '{"name": "finish", "arguments": {"success": false}}\n'
                '</tool_call>'
            )
        response = self._responses[self._call_idx]
        self._call_idx += 1
        return response

    def count_tokens(self, text: str) -> int:
        return self._tokenizer.count(text)

    @property
    def model_name(self) -> str:
        return "echo-backend"

    def reset(self) -> None:
        """Reset call index for reuse across episodes."""
        self._call_idx = 0
