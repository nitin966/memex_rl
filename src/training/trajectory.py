"""Segmented trajectory processor for MemexRL training.

Implements Section 3.3 (Segmented Trajectory Processing):
  When compression occurs, the trajectory is segmented at compression
  boundaries. Each segment S_i is processed as an independent training
  sample while preserving credit assignment via shared terminal reward R.

  S_0 = full pre-compression history
  S_i = [system, task, summary_{i-1}, z_{i1}, c_{i1}, o_{i1}, ...]

  All segments from the same trajectory share the identical terminal
  reward R, preserving credit assignment to earlier compression decisions
  through group-relative advantage estimation in GRPO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.trajectory import Episode, Segment, Step
from src.training.rewards import RewardEngine


@dataclass
class TrainingSegment:
    """A single training segment ready for the GRPO optimizer.

    Each segment is independently tokenized and optimized under its
    own context window, but shares the episode-level return.
    """
    episode_id: str
    segment_idx: int
    prefix: str                    # Context prefix (system + task + summary)
    steps: list[Step]
    reward: float                  # Shared terminal reward R
    advantage: float = 0.0        # Group-relative normalized advantage A_hat
    group_id: str = ""            # Groups segments from same prompt for GRPO


@dataclass
class TrainingBatch:
    """A batch of training segments flattened across episodes.

    All segments are processed independently during training,
    but advantage normalization happens within groups.
    """
    segments: list[TrainingSegment] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.segments)

    def compute_advantages(self) -> None:
        """Compute group-relative normalized advantages (Eq. 2).

        For each group (same prompt), compute:
          A_hat(g) = (R(g) - mean(R)) / std(R)

        This preserves credit assignment: all segments from the same
        trajectory share the same advantage.
        """
        # Group segments by group_id
        groups: dict[str, list[TrainingSegment]] = {}
        for seg in self.segments:
            groups.setdefault(seg.group_id, []).append(seg)

        for group_id, group_segs in groups.items():
            # Collect unique episode rewards within this group
            episode_rewards: dict[str, float] = {}
            for seg in group_segs:
                episode_rewards[seg.episode_id] = seg.reward

            rewards = list(episode_rewards.values())
            if len(rewards) <= 1:
                for seg in group_segs:
                    seg.advantage = 0.0
                continue

            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = max(var_r ** 0.5, 1e-8)  # Avoid division by zero

            for seg in group_segs:
                seg.advantage = (seg.reward - mean_r) / std_r


class TrajectoryProcessor:
    """Processes episodes into training segments for GRPO.

    Key design (from paper Section 3.3):
      - Segments at compression boundaries
      - Each segment has its own context (post-compression prefix)
      - All segments share the terminal reward
      - Flattened across batch for independent optimization
    """

    def __init__(self, reward_engine: RewardEngine | None = None,
                 threshold: int = 8000) -> None:
        self.reward_engine = reward_engine or RewardEngine()
        self.threshold = threshold

    def process_episode(self, episode: Episode) -> list[TrainingSegment]:
        """Convert an episode into training segments.

        If the agent compressed k times, produces k+1 segments,
        each sharing the same terminal reward.
        """
        # Compute the reward if not already set
        reward = episode.terminal_reward
        if reward == 0.0 and not episode.task_success:
            reward = self.reward_engine.compute_return(episode, self.threshold)
        elif episode.task_success:
            reward = self.reward_engine.compute_return(episode, self.threshold)

        segments = []
        for seg in episode.segments:
            segments.append(TrainingSegment(
                episode_id=episode.id,
                segment_idx=seg.segment_idx,
                prefix=seg.prefix,
                steps=seg.steps,
                reward=reward,
            ))
        return segments

    def build_batch(
        self,
        rollout_groups: list[tuple[str, list[Episode]]],
    ) -> TrainingBatch:
        """Build a training batch from grouped rollouts.

        Args:
            rollout_groups: List of (prompt_id, episodes) pairs.
                Each prompt has G rollout episodes (GRPO group).

        Returns:
            TrainingBatch with all segments flattened and advantages computed.
        """
        batch = TrainingBatch()

        for prompt_id, episodes in rollout_groups:
            for episode in episodes:
                training_segs = self.process_episode(episode)
                for seg in training_segs:
                    seg.group_id = prompt_id
                batch.segments.extend(training_segs)

        batch.compute_advantages()
        return batch
