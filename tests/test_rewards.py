"""Unit tests for the MemexRL Reward Engine (PR 6).

Tests each penalty individually with hand-calculated expected values,
plus combined return computation.
"""

from __future__ import annotations

import pytest

from src.models.tools import ToolCall
from src.models.trajectory import Episode, Segment, Step
from src.training.rewards import RewardEngine, RewardBreakdown


def _step(name: str = "execute_action", action: str = "look",
          tokens: int = 1000, errors: list[str] | None = None) -> Step:
    """Helper to create a Step with minimal boilerplate."""
    return Step(
        thinking="thinking",
        tool_call=ToolCall(name=name, arguments={"action": action}),
        observation="obs",
        context_tokens=tokens,
        format_errors=errors or [],
    )


def _episode(steps: list[Step], success: bool = False) -> Episode:
    """Helper to wrap steps into a single-segment Episode."""
    return Episode(
        segments=[Segment(segment_idx=0, steps=steps)],
        task_success=success,
        terminal_reward=1.0 if success else 0.0,
    )


class TestContextOverflowPenalty:
    """P_context = min(1, Σ max(0, C_t - τ) / (τ·T))"""

    def setup_method(self):
        self.engine = RewardEngine()

    def test_no_overflow(self):
        """All steps below threshold → P_context = 0."""
        ep = _episode([
            _step(tokens=3000),
            _step(tokens=5000),
            _step(tokens=7000),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.context_penalty == 0.0
        assert bd.overflow_tokens == 0

    def test_partial_overflow(self):
        """Some steps above threshold → 0 < P_context < 1."""
        # Step 1: 3000 (no overflow)
        # Step 2: 10000 (overflow = 2000)
        # Step 3: 9000 (overflow = 1000)
        # Total overflow = 3000, τ·T = 8000·3 = 24000
        # P_context = min(1, 3000/24000) = 0.125
        ep = _episode([
            _step(tokens=3000),
            _step(tokens=10000),
            _step(tokens=9000),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.context_penalty == pytest.approx(3000 / 24000)
        assert bd.overflow_tokens == 3000

    def test_severe_overflow_caps_at_1(self):
        """Massive overflow → P_context capped at 1.0."""
        ep = _episode([
            _step(tokens=100000),
            _step(tokens=100000),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.context_penalty == 1.0

    def test_compression_steps_excluded(self):
        """CompressExperience steps are excluded from penalty."""
        ep = _episode([
            _step(tokens=3000),
            _step(name="CompressExperience", tokens=50000),  # Huge but excluded
            _step(tokens=3000),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.context_penalty == 0.0  # Both non-compress steps are under threshold

    def test_empty_episode(self):
        ep = _episode([])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.context_penalty == 0.0


class TestRedundancyPenalty:
    """P_redundancy = N_redundant / N_tool_call"""

    def setup_method(self):
        self.engine = RewardEngine()

    def test_no_redundancy(self):
        """All unique calls → P_redundancy = 0."""
        ep = _episode([
            _step(action="look"),
            _step(action="go to desk 1"),
            _step(action="pick up book"),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.redundancy_penalty == 0.0
        assert bd.redundant_calls == 0

    def test_repeated_calls(self):
        """Identical calls without state change → penalized."""
        # 3 env tool calls: look, look (redundant), go
        # N_redundant = 1, N_tool = 3
        # P = 1/3
        ep = _episode([
            _step(action="look"),
            _step(action="look"),  # Redundant!
            _step(action="go to desk 1"),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.redundancy_penalty == pytest.approx(1 / 3)
        assert bd.redundant_calls == 1

    def test_state_change_resets(self):
        """ReadExperience between identical calls = not redundant."""
        ep = _episode([
            _step(action="look"),
            _step(name="ReadExperience", action="ctx_loc"),  # State change
            _step(action="look"),  # Same args but state changed → OK
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.redundancy_penalty == 0.0

    def test_memory_tools_excluded_from_count(self):
        """Memory tools (Compress/Read/finish) not counted as tool calls."""
        ep = _episode([
            _step(action="look"),
            _step(name="CompressExperience", action="summary"),
            _step(name="ReadExperience", action="ctx_loc"),
            _step(name="finish", action="true"),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.total_tool_calls == 1  # Only "look" counts


class TestFormatErrorPenalty:
    """P_format = N_malformed / N_tool_call"""

    def setup_method(self):
        self.engine = RewardEngine()

    def test_no_errors(self):
        ep = _episode([_step(), _step(), _step()])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.format_penalty == 0.0
        assert bd.malformed_calls == 0

    def test_some_errors(self):
        # 1 out of 3 steps has errors → P = 1/3
        ep = _episode([
            _step(),
            _step(errors=["Tag mismatch"]),
            _step(),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.format_penalty == pytest.approx(1 / 3)
        assert bd.malformed_calls == 1

    def test_all_errors(self):
        ep = _episode([
            _step(errors=["err1"]),
            _step(errors=["err2", "err3"]),
        ])
        bd = self.engine.compute_breakdown(ep, threshold=8000)
        assert bd.format_penalty == 1.0


class TestCombinedReturn:
    """R = R_task - α₁·P_context - α₂·P_redundancy - α₃·P_format"""

    def test_perfect_episode(self):
        """Success with no penalties → R = 1.0."""
        engine = RewardEngine(alpha_context=0.3, alpha_redundancy=0.3, alpha_format=0.3)
        ep = _episode([_step(tokens=3000)], success=True)
        r = engine.compute_return(ep, threshold=8000)
        assert r == 1.0

    def test_failed_no_penalties(self):
        """Failure with no penalties → R = 0.0."""
        engine = RewardEngine()
        ep = _episode([_step(tokens=3000)], success=False)
        r = engine.compute_return(ep, threshold=8000)
        assert r == 0.0

    def test_success_with_penalties(self):
        """Success but with penalties → 0 < R < 1."""
        engine = RewardEngine(alpha_context=0.3, alpha_redundancy=0.3, alpha_format=0.3)
        ep = _episode([
            _step(tokens=10000),  # Overflow of 2000
            _step(tokens=10000),  # Overflow of 2000
        ], success=True)
        bd = engine.compute_breakdown(ep, threshold=8000)
        assert bd.task_reward == 1.0
        assert bd.context_penalty > 0
        assert bd.total_return < 1.0
        assert bd.total_return > 0.0

    def test_negative_return_possible(self):
        """Heavy penalties on failed task → R < 0."""
        engine = RewardEngine(alpha_context=1.0, alpha_redundancy=1.0, alpha_format=1.0)
        ep = _episode([
            _step(tokens=100000, errors=["bad"]),
            _step(tokens=100000, action="look"),
            _step(tokens=100000, action="look"),  # Redundant
        ], success=False)
        r = engine.compute_return(ep, threshold=8000)
        assert r < 0

    def test_custom_weights(self):
        """Different alpha weights change the return."""
        ep = _episode([
            _step(tokens=10000),  # Some overflow
            _step(tokens=10000, errors=["err"]),
        ], success=True)

        engine_heavy = RewardEngine(alpha_context=1.0, alpha_redundancy=1.0, alpha_format=1.0)
        engine_light = RewardEngine(alpha_context=0.1, alpha_redundancy=0.1, alpha_format=0.1)

        r_heavy = engine_heavy.compute_return(ep, threshold=8000)
        r_light = engine_light.compute_return(ep, threshold=8000)
        assert r_light > r_heavy  # Light penalty = higher return

    def test_breakdown_has_all_fields(self):
        engine = RewardEngine()
        ep = _episode([_step()], success=True)
        bd = engine.compute_breakdown(ep, threshold=8000)
        assert isinstance(bd, RewardBreakdown)
        assert bd.total_steps == 1
        assert bd.task_reward == 1.0
