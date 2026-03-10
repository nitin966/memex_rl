"""Abstract LLM backend interface for Memex(RL).

Defines the contract for LLM generation that the agent loop depends on.
Concrete implementations (OpenAI/Ollama, vLLM, SGLang) come in PR 8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.memory import Message


class LLMBackend(ABC):
    """Abstract LLM interface for the Memex agent.

    The agent loop calls generate() once per step to get the agent's
    thinking text and tool call. Implementations must handle the
    message format conversion internally.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion given the current context.

        Args:
            messages: The full context window M as a list of Messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw text output from the LLM (thinking + tool call).
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the model's tokenizer."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the underlying model."""
        ...
