"""SGLang-compatible LLM backend for Memex(RL).

Integrates with an SGLang server to leverage RadixAttention prefix caching.
This is highly recommended for MemexRL training, as the system prompt
and L2 experience memory keys remain static across execution frames,
allowing SGLang to serve generation requests with O(1) TTFT latency.

Requires an SGLang server running locally or remotely:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct --port 30000
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.llm.backend import LLMBackend
from src.memory.tokenizer import Tokenizer
from src.models.memory import Message, MessageRole

logger = logging.getLogger(__name__)


class SGLangBackend(LLMBackend):
    """SGLang-compatible API backend for Prefix Caching optimizations.

    Args:
        model: Model name registered on the SGLang server.
        base_url: API base URL (e.g., http://localhost:30000/v1).
        api_key: Optional API key (SGLang usually ignores this locally).
        default_temperature: Default sampling temperature.
        default_max_tokens: Default max generation tokens.
    """

    def __init__(
        self,
        model: str = "default",
        base_url: str | None = None,
        api_key: str | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ) -> None:
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "SGLangBackend uses the 'openai' client package to connect. "
                "Install with: pip install openai"
            )

        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._tokenizer = Tokenizer(model=model)

        # Resolve connection parameters
        resolved_base_url = base_url or os.environ.get(
            "SGLANG_BASE_URL", "http://localhost:30000/v1"
        )
        resolved_api_key = api_key or os.environ.get("SGLANG_API_KEY", "EMPTY")

        self._client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )

        logger.info(
            f"SGLangBackend initialized: model={model}, "
            f"base_url={resolved_base_url}"
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion from the SGLang server.

        Args:
            messages: Context window as Message objects.
            temperature: Sampling temperature (uses default if None).
            max_tokens: Max tokens to generate (uses default if None).

        Returns:
            Raw text output from the model.
        """
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                temperature=temperature or self._default_temperature,
                max_tokens=max_tokens or self._default_max_tokens,
                # SGLang specific extra body parameters could go here if needed
                # e.g., extra_body={"top_p": 0.95}
            )
            content = response.choices[0].message.content or ""
            
            # Extract caching metrics if returned by SGLang (often in usage context)
            usage = response.usage
            if usage and hasattr(usage, 'prompt_cache_hit_tokens'):
                logger.debug(
                    f"Prefix Cache Hit: {usage.prompt_cache_hit_tokens} / {usage.prompt_tokens} tokens"
                )
            
            return content

        except Exception as e:
            logger.error(f"SGLang generation failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return self._tokenizer.count(text)

    @property
    def model_name(self) -> str:
        return self._model
