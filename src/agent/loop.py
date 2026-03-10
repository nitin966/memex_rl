"""Memex Agent Loop — Algorithm 1 from the paper.

Implements the full execution loop with Indexed Experience Memory:
  1. Initialize M ← [m₀, u], D ← ∅
  2. At each step:
     a. Append ContextStatus(M, τ)
     b. Agent emits thinking z_t and tool call c_t
     c. If CompressExperience → archive and rewrite
     d. If ReadExperience → dereference and inject
     e. If finish → return
     f. Otherwise → execute environment tool
  3. Terminate after T_max steps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.agent.tool_parser import ToolParser
from src.agent.prompts import build_system_prompt
from src.environments.base import Environment, StepResult
from src.llm.backend import LLMBackend
from src.memory.controller import MemoryController
from src.memory.store import DictStore, ExperienceStore
from src.memory.tokenizer import Tokenizer
from src.models.memory import MemoryBlock
from src.models.tools import ToolCall
from src.models.trajectory import Episode, Segment, Step

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the Memex agent."""
    max_steps: int = 100              # T_max
    context_window: int = 32_768      # N
    threshold: int = 8_000            # τ
    summary_max_tokens: int = 300     # Paper default
    temperature: float = 0.7
    max_generation_tokens: int = 4096
    system_prompt: str | None = None  # Override default
    environment_prompt: str | None = None
    memory_addendum: str | None = None


class MemexAgent:
    """Memex agent implementing Algorithm 1 from the paper.

    Ties together the MemoryController, ToolParser, LLMBackend,
    and Environment into the full agent execution loop.

    Args:
        llm: LLM backend for generation.
        environment: Task environment.
        config: Agent configuration.
        store: Optional L2 store (DictStore created per-episode if None).
    """

    def __init__(
        self,
        llm: LLMBackend,
        environment: Environment,
        config: AgentConfig | None = None,
        store: ExperienceStore | None = None,
    ) -> None:
        self.llm = llm
        self.env = environment
        self.config = config or AgentConfig()
        self.parser = ToolParser()
        self.tokenizer = Tokenizer()

        # Store is created fresh per episode if not provided
        self._store_factory = store

    def run_episode(self, task_id: str | None = None) -> Episode:
        """Run a single episode following Algorithm 1.

        Args:
            task_id: Optional task identifier for the environment.

        Returns:
            Episode with segmented trajectory and terminal reward.
        """
        # ── Lines 4-6: Initialize ──────────────────────────────────────
        store = self._store_factory or DictStore()
        controller = MemoryController(
            store=store,
            tokenizer=self.tokenizer,
            context_window=self.config.context_window,
            threshold=self.config.threshold,
            summary_max_tokens=self.config.summary_max_tokens,
        )

        # Build system prompt
        system_prompt = self.config.system_prompt or build_system_prompt(
            environment_prompt=self.config.environment_prompt,
            memory_management_addendum=self.config.memory_addendum,
        )

        # Reset environment and get task instruction
        task_instruction = self.env.reset(task_id)

        # M ← [m₀, u]
        controller.reset(system_prompt=system_prompt, task_instruction=task_instruction)

        # Tracking
        answer: bool | None = None
        segments: list[Segment] = []
        current_segment_steps: list[Step] = []
        current_segment_idx = 0

        # ── Line 7: Main loop ──────────────────────────────────────────
        for t in range(self.config.max_steps):
            # Line 8: Append ContextStatus(M, τ)
            controller.inject_context_status()

            # Line 9: Agent emits thinking z_t and tool call c_t
            messages = controller.get_messages()
            raw_output = self.llm.generate(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_generation_tokens,
            )

            # Parse the output
            parse_result = self.parser.parse(raw_output)
            thinking = parse_result.thinking
            tool_call = parse_result.tool_call
            format_errors = parse_result.format_errors

            # Line 10: Append to working context
            controller.append_assistant(raw_output)

            # Handle no valid tool call
            if tool_call is None:
                error_msg = (
                    "Error: No valid tool call found in your response. "
                    "You MUST include a tool call in <tool_call> tags. "
                    f"Parse errors: {'; '.join(format_errors)}"
                )
                controller.append_tool_result(error_msg, tool_name="system")
                current_segment_steps.append(Step(
                    thinking=thinking,
                    tool_call=ToolCall(name="_error", arguments={}),
                    observation=error_msg,
                    context_tokens=controller.working_token_count(),
                    format_errors=format_errors,
                ))
                continue

            # ── Lines 11-14: CompressExperience ────────────────────────
            if tool_call.name == "CompressExperience":
                # Save current segment before compression
                segments.append(Segment(
                    segment_idx=current_segment_idx,
                    prefix=controller.get_prefix(),
                    steps=current_segment_steps,
                ))
                current_segment_steps = []
                current_segment_idx += 1

                # Parse arguments
                summary = tool_call.arguments.get("summary", "")
                db_blocks_raw = tool_call.arguments.get("db_blocks", [])
                memory_blocks = self._parse_memory_blocks(db_blocks_raw)

                result = controller.compress_experience(
                    summary=summary,
                    memory_blocks=memory_blocks,
                )
                controller.append_tool_result(result, tool_name="CompressExperience")

                current_segment_steps.append(Step(
                    thinking=thinking,
                    tool_call=tool_call,
                    observation=result,
                    context_tokens=controller.working_token_count(),
                    format_errors=format_errors,
                ))
                logger.info(f"Step {t}: CompressExperience — {result}")

            # ── Lines 15-17: ReadExperience ────────────────────────────
            elif tool_call.name == "ReadExperience":
                db_index = tool_call.arguments.get("db_index", "")
                content = controller.read_experience(db_index)

                current_segment_steps.append(Step(
                    thinking=thinking,
                    tool_call=tool_call,
                    observation=content,
                    context_tokens=controller.working_token_count(),
                    format_errors=format_errors,
                ))
                logger.info(f"Step {t}: ReadExperience({db_index})")

            # ── Lines 18-20: Finish ────────────────────────────────────
            elif tool_call.name == "finish":
                answer = tool_call.arguments.get("success", False)

                current_segment_steps.append(Step(
                    thinking=thinking,
                    tool_call=tool_call,
                    observation=f"Episode finished. success={answer}",
                    context_tokens=controller.working_token_count(),
                    format_errors=format_errors,
                ))
                logger.info(f"Step {t}: finish(success={answer})")
                break

            # ── Lines 21-23: Environment tool ──────────────────────────
            else:
                action = tool_call.arguments.get("action", str(tool_call.arguments))
                step_result = self.env.step(action)

                controller.append_tool_result(
                    step_result.observation, tool_name=tool_call.name
                )

                current_segment_steps.append(Step(
                    thinking=thinking,
                    tool_call=tool_call,
                    observation=step_result.observation,
                    context_tokens=controller.working_token_count(),
                    format_errors=format_errors,
                ))
                logger.debug(f"Step {t}: {tool_call.name}({action}) → {step_result.observation[:80]}")

                # Check if environment signals done
                if step_result.done:
                    answer = step_result.reward > 0
                    break

        # ── Line 24: Terminal segment ──────────────────────────────────
        segments.append(Segment(
            segment_idx=current_segment_idx,
            prefix=controller.get_prefix(),
            steps=current_segment_steps,
        ))

        return Episode(
            task_id=self.env.get_task_id(),
            segments=segments,
            terminal_reward=1.0 if answer else 0.0,
            task_success=bool(answer),
        )

    def _parse_memory_blocks(self, raw_blocks: list) -> list[MemoryBlock]:
        """Parse raw dict list into validated MemoryBlock objects."""
        blocks = []
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            try:
                blocks.append(MemoryBlock(**raw))
            except Exception as e:
                logger.warning(f"Skipping malformed memory block: {e}")
        return blocks
