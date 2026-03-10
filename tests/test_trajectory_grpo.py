"""Unit tests for Trajectory Processor and GRPO Trainer (PR 7)."""

from __future__ import annotations

import pytest

from src.models.tools import ToolCall
from src.models.trajectory import Episode, Segment, Step
from src.training.rewards import RewardEngine
from src.training.trajectory import TrajectoryProcessor, TrainingBatch, TrainingSegment
from src.training.grpo import GRPOConfig, GRPOTrainer


# ── Helpers ────────────────────────────────────────────────────────────────

def _step(name: str = "execute_action", action: str = "look",
          tokens: int = 1000) -> Step:
    return Step(
        thinking="thinking",
        tool_call=ToolCall(name=name, arguments={"action": action}),
        observation="obs",
        context_tokens=tokens,
    )


def _episode(
    segments: list[Segment] | None = None,
    success: bool = False,
    episode_id: str = "ep1",
) -> Episode:
    if segments is None:
        segments = [Segment(segment_idx=0, steps=[_step()])]
    return Episode(
        id=episode_id,
        task_id="task_001",
        segments=segments,
        task_success=success,
        terminal_reward=1.0 if success else 0.0,
    )


# ── TrajectoryProcessor Tests ─────────────────────────────────────────────

class TestTrajectoryProcessor:

    def setup_method(self):
        self.processor = TrajectoryProcessor()

    def test_single_segment_episode(self):
        """No compression → 1 segment."""
        ep = _episode(success=True)
        segs = self.processor.process_episode(ep)
        assert len(segs) == 1
        assert segs[0].episode_id == ep.id
        assert segs[0].segment_idx == 0

    def test_multi_segment_episode(self):
        """k=2 compressions → 3 segments, all sharing reward."""
        ep = _episode(
            segments=[
                Segment(segment_idx=0, prefix="sys+task", steps=[_step()]),
                Segment(segment_idx=1, prefix="sys+task+summary1", steps=[_step()]),
                Segment(segment_idx=2, prefix="sys+task+summary2", steps=[_step()]),
            ],
            success=True,
        )
        segs = self.processor.process_episode(ep)
        assert len(segs) == 3
        # All share the same reward
        rewards = {s.reward for s in segs}
        assert len(rewards) == 1

    def test_prefix_preserved(self):
        ep = _episode(
            segments=[
                Segment(segment_idx=0, prefix="system + task", steps=[_step()]),
            ],
            success=True,
        )
        segs = self.processor.process_episode(ep)
        assert segs[0].prefix == "system + task"


# ── Advantage Computation Tests ───────────────────────────────────────────

class TestAdvantageComputation:

    def test_group_relative_advantage(self):
        """Advantages should be group-normalized: mean≈0, std≈1."""
        batch = TrainingBatch(segments=[
            TrainingSegment(episode_id="ep1", segment_idx=0, prefix="",
                           steps=[], reward=1.0, group_id="g1"),
            TrainingSegment(episode_id="ep2", segment_idx=0, prefix="",
                           steps=[], reward=0.0, group_id="g1"),
            TrainingSegment(episode_id="ep3", segment_idx=0, prefix="",
                           steps=[], reward=0.5, group_id="g1"),
        ])
        batch.compute_advantages()

        advantages = [s.advantage for s in batch.segments]
        mean_adv = sum(advantages) / len(advantages)
        assert mean_adv == pytest.approx(0.0, abs=0.01)

        # Highest reward → highest advantage
        assert batch.segments[0].advantage > batch.segments[1].advantage

    def test_shared_reward_same_advantage(self):
        """Segments from same episode should have identical advantage."""
        batch = TrainingBatch(segments=[
            # Two segments from ep1 (both have reward 1.0)
            TrainingSegment(episode_id="ep1", segment_idx=0, prefix="",
                           steps=[], reward=1.0, group_id="g1"),
            TrainingSegment(episode_id="ep1", segment_idx=1, prefix="",
                           steps=[], reward=1.0, group_id="g1"),
            # One segment from ep2 (reward 0.0)
            TrainingSegment(episode_id="ep2", segment_idx=0, prefix="",
                           steps=[], reward=0.0, group_id="g1"),
        ])
        batch.compute_advantages()

        assert batch.segments[0].advantage == batch.segments[1].advantage

    def test_separate_groups_independent(self):
        """Different groups compute advantages independently."""
        batch = TrainingBatch(segments=[
            TrainingSegment(episode_id="ep1", segment_idx=0, prefix="",
                           steps=[], reward=1.0, group_id="g1"),
            TrainingSegment(episode_id="ep2", segment_idx=0, prefix="",
                           steps=[], reward=0.0, group_id="g1"),
            TrainingSegment(episode_id="ep3", segment_idx=0, prefix="",
                           steps=[], reward=0.8, group_id="g2"),
            TrainingSegment(episode_id="ep4", segment_idx=0, prefix="",
                           steps=[], reward=0.2, group_id="g2"),
        ])
        batch.compute_advantages()

        # Within g1: ep1 should have positive advantage
        assert batch.segments[0].advantage > 0
        assert batch.segments[1].advantage < 0

        # Within g2: ep3 should have positive advantage
        assert batch.segments[2].advantage > 0
        assert batch.segments[3].advantage < 0

    def test_single_episode_group_zero_advantage(self):
        """A group with only one episode → advantage = 0."""
        batch = TrainingBatch(segments=[
            TrainingSegment(episode_id="ep1", segment_idx=0, prefix="",
                           steps=[], reward=1.0, group_id="g1"),
        ])
        batch.compute_advantages()
        assert batch.segments[0].advantage == 0.0


# ── TrainingBatch.build_batch Tests ───────────────────────────────────────

class TestBuildBatch:

    def test_build_batch_groups_correctly(self):
        proc = TrajectoryProcessor()
        ep1 = _episode(success=True, episode_id="ep1")
        ep2 = _episode(success=False, episode_id="ep2")

        rollout_groups = [("prompt_1", [ep1, ep2])]
        batch = proc.build_batch(rollout_groups)

        assert batch.size == 2
        assert all(s.group_id == "prompt_1" for s in batch.segments)

    def test_build_batch_multiple_prompts(self):
        proc = TrajectoryProcessor()
        rollout_groups = [
            ("prompt_A", [_episode(success=True, episode_id="a1")]),
            ("prompt_B", [_episode(success=False, episode_id="b1")]),
        ]
        batch = proc.build_batch(rollout_groups)
        assert batch.size == 2
        groups = {s.group_id for s in batch.segments}
        assert groups == {"prompt_A", "prompt_B"}


# ── GRPO Trainer Tests ────────────────────────────────────────────────────

class TestGRPOTrainer:

    def test_config_defaults_match_paper(self):
        cfg = GRPOConfig()
        assert cfg.group_size == 8
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 5e-6
        assert cfg.clip_ratio == 2.0
        assert cfg.kl_penalty == 0.001
        assert cfg.context_window == 32_768
        assert cfg.threshold == 8_000

    def test_prepare_batch(self):
        trainer = GRPOTrainer(GRPOConfig())
        ep1 = _episode(success=True, episode_id="ep1")
        ep2 = _episode(success=False, episode_id="ep2")

        batch = trainer.prepare_batch([("task_1", [ep1, ep2])])
        assert batch.size == 2

    def test_compute_loss_empty_batch(self):
        trainer = GRPOTrainer(GRPOConfig())
        batch = TrainingBatch()
        result = trainer.compute_loss(batch)
        assert result.num_segments == 0

    def test_compute_loss_with_data(self):
        trainer = GRPOTrainer(GRPOConfig())
        ep1 = _episode(success=True, episode_id="ep1")
        ep2 = _episode(success=False, episode_id="ep2")

        batch = trainer.prepare_batch([("task_1", [ep1, ep2])])
        result = trainer.compute_loss(batch)
        assert result.num_segments == 2
        assert result.mean_reward != 0  # One success, one failure

    def test_train_step_increments_counter(self):
        trainer = GRPOTrainer(GRPOConfig())
        assert trainer.step_count == 0

        ep = _episode(success=True, episode_id="ep1")
        trainer.train_step([("task_1", [ep])])
        assert trainer.step_count == 1

        trainer.train_step([("task_2", [ep])])
        assert trainer.step_count == 2
