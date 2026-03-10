"""Unit tests for environment integrations (PR 9).

Tests the ALFWorld modifications and stress test environment.
ALFWorld tests use the mock fallback (no alfworld package needed).
"""

from __future__ import annotations

import pytest

from src.environments.alfworld_env import ALFWorldModifiedEnv
from src.environments.stress_test import StressTestEnv
from src.environments.base import Environment


# ── ALFWorld Modified Environment ──────────────────────────────────────────

class TestALFWorldModifiedEnv:
    """Tests for the 4 paper modifications to ALFWorld."""

    def setup_method(self):
        self.env = ALFWorldModifiedEnv(
            max_obs_tokens=500,
            look_limit=1,
            hide_initial_obs=True,
            hide_commands=True,
        )

    def test_is_environment(self):
        assert isinstance(self.env, Environment)

    def test_reset_returns_task(self):
        task = self.env.reset(task_id="put the book on desk")
        assert isinstance(task, str)
        assert len(task) > 0

    def test_step_returns_step_result(self):
        self.env.reset(task_id="test_task")
        result = self.env.step("go to desk 1")
        assert result.observation is not None
        assert isinstance(result.done, bool)

    def test_modification_3_limited_look(self):
        """Look command limited to 1 per location."""
        self.env.reset(task_id="test_task")

        # First look at unknown location — should work
        r1 = self.env.step("look")
        assert "already looked" not in r1.observation

        # Second look at same location — should be blocked
        r2 = self.env.step("look")
        assert "already looked" in r2.observation

    def test_limited_look_resets_per_location(self):
        """Moving to a new location resets look availability."""
        self.env.reset(task_id="test_task")

        self.env.step("look")  # Look at initial location
        self.env.step("go to desk 1")  # Move to new location
        result = self.env.step("look")  # Should work at new location
        assert "already looked" not in result.observation

    def test_modification_4_truncation(self):
        """Long observations should be truncated."""
        env = ALFWorldModifiedEnv(max_obs_tokens=10)
        env.reset(task_id="test_task")
        # The mock won't produce a long observation, but let's test the
        # truncation method directly
        long_obs = " ".join(["word"] * 500)
        truncated = env._truncate_observation(long_obs)
        assert truncated.endswith("...[observation truncated]")

    def test_modification_1_strip_command_echo(self):
        """Command echoes should be stripped from observations."""
        env = ALFWorldModifiedEnv(hide_commands=True)
        # Test the internal stripping method
        obs = "> go to desk 1\nYou arrive at desk 1."
        stripped = env._strip_command_echo(obs, "go to desk 1")
        assert stripped == "You arrive at desk 1."

    def test_get_task_id(self):
        self.env.reset(task_id="specific_task_42")
        assert self.env.get_task_id() == "specific_task_42"


# ── Stress Test Environment ───────────────────────────────────────────────

class TestStressTestEnv:
    """Tests for the 1000-doc recursive file-search stress test."""

    def setup_method(self):
        self.env = StressTestEnv(
            num_files=100,  # Smaller for fast tests
            num_dirs=5,
            max_depth=3,
            target_depth=2,
            seed=42,
        )

    def test_is_environment(self):
        assert isinstance(self.env, Environment)

    def test_reset_returns_task_with_keyword(self):
        task = self.env.reset()
        assert "keyword" in task.lower() or "find" in task.lower()
        assert self.env._target_keyword in task

    def test_filesystem_generated(self):
        self.env.reset()
        assert len(self.env._files) == 100
        assert len(self.env._dirs) > 0
        assert self.env._target_path != ""

    def test_ls_root(self):
        self.env.reset()
        result = self.env.step("ls /")
        assert "dept_" in result.observation
        assert "📁" in result.observation

    def test_ls_nonexistent(self):
        self.env.reset()
        result = self.env.step("ls /nonexistent")
        assert "not found" in result.observation.lower()

    def test_cat_file(self):
        self.env.reset()
        # Cat the target file
        result = self.env.step(f"cat {self.env._target_path}")
        assert self.env._target_keyword in result.observation

    def test_cat_nonexistent(self):
        self.env.reset()
        result = self.env.step("cat /nonexistent.txt")
        assert "not found" in result.observation.lower()

    def test_cat_target_marks_found(self):
        self.env.reset()
        assert not self.env._found
        self.env.step(f"cat {self.env._target_path}")
        assert self.env._found is True

    def test_grep_finds_target(self):
        self.env.reset()
        target_dir = "/".join(self.env._target_path.split("/")[:-1])
        result = self.env.step(f"grep {self.env._target_keyword} {target_dir}")
        assert "Found" in result.observation
        assert self.env._target_path in result.observation

    def test_grep_no_match(self):
        self.env.reset()
        result = self.env.step("grep NONEXISTENT_KEYWORD_XYZ /")
        assert "No files" in result.observation

    def test_find_pattern(self):
        self.env.reset()
        result = self.env.step("find classified")
        assert "Found" in result.observation

    def test_unknown_command(self):
        self.env.reset()
        result = self.env.step("unknown_cmd foo")
        assert "Available commands" in result.observation

    def test_deterministic_with_seed(self):
        """Same seed produces same filesystem."""
        env1 = StressTestEnv(num_files=50, seed=123)
        env2 = StressTestEnv(num_files=50, seed=123)
        env1.reset()
        env2.reset()
        assert env1._target_keyword == env2._target_keyword
        assert env1._target_path == env2._target_path

    def test_full_size_generation(self):
        """Test with 1000 files (paper specification)."""
        env = StressTestEnv(num_files=1000, num_dirs=10, max_depth=5, seed=42)
        env.reset()
        assert len(env._files) == 1000

    def test_get_task_id(self):
        self.env.reset(task_id="custom_stress_test")
        assert self.env.get_task_id() == "custom_stress_test"
