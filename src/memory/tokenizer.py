"""tiktoken-based token counting for Memex(RL).

Used by the MemoryController to monitor context window usage and enforce
the compression threshold τ (paper Algorithm 1, line 8).
"""

from __future__ import annotations

import tiktoken


class Tokenizer:
    """Token counter and truncator using tiktoken.

    Args:
        model: Model name for tiktoken encoding lookup.
               Default "gpt-4" gives cl100k_base encoding.
               For Qwen models, override with the appropriate encoding.
        encoding_name: Direct encoding name (overrides model if set).
    """

    def __init__(
        self,
        model: str = "gpt-4",
        encoding_name: str | None = None,
    ) -> None:
        self.is_hf = False
        self._hf_name = ""
        if encoding_name:
            self._enc = tiktoken.get_encoding(encoding_name)
        else:
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                if "qwen" in model.lower():
                    from transformers import AutoTokenizer
                    # Map ollama tags to HF hub models if needed
                    hf_path = model
                    if ":" in model or "/" not in model:
                        hf_path = "Qwen/Qwen2.5-3B-Instruct"  # Safe default
                        if "7b" in model.lower():
                            hf_path = "Qwen/Qwen2.5-7B-Instruct"
                    
                    self._enc = AutoTokenizer.from_pretrained(hf_path)
                    self.is_hf = True
                    self._hf_name = hf_path
                else:
                    # Fallback to cl100k_base for unknown models
                    self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens across a list of chat messages.

        Approximates ChatML overhead: each message adds ~4 tokens
        for role/formatting markers.
        """
        total = 0
        for msg in messages:
            total += 4  # <|im_start|>role\n ... <|im_end|>\n
            total += self.count(msg.get("content", ""))
            if msg.get("name"):
                total += self.count(msg["name"])
        total += 2  # <|im_start|>assistant priming
        return total

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to at most max_tokens tokens.

        Truncates at token boundaries (no partial tokens).
        Appends '...[truncated]' if truncation occurred.
        """
        if not text:
            return text
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Reserve tokens for the truncation marker
        marker = "...[truncated]"
        marker_tokens = len(self._enc.encode(marker))
        keep = max(0, max_tokens - marker_tokens)
        truncated = self._enc.decode(tokens[:keep])
        return truncated + marker

    @property
    def encoding_name(self) -> str:
        """Name of the tiktoken encoding being used."""
        return self._hf_name if self.is_hf else self._enc.name
