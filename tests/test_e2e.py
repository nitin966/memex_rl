"""End-to-end integration test for Memex(RL).

Runs a full agent episode using the EchoBackend and StressTestEnv,
verifying the complete pipeline: LLM → parse → memory → env → rewards.
No external dependencies required (no Ollama, no GPU).
"""

from __future__ import annotations

import pytest

from src.agent.loop import AgentConfig, MemexAgent
from src.environments.stress_test import StressTestEnv
from src.llm.openai_backend import EchoBackend
from src.training.rewards import RewardEngine
from src.training.trajectory import TrajectoryProcessor


class TestEndToEndIntegration:
    """Full pipeline integration test."""

    def test_stress_test_with_compression_cycle(self):
        """Complete pipeline: agent explores, compresses, reads, finishes."""
        env = StressTestEnv(num_files=50, num_dirs=3, max_depth=2, seed=42)

        # Script the agent through a realistic scenario
        llm = EchoBackend(responses=[
            # Step 1: List root directory
            'Let me explore the filesystem.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "ls /"}}\n'
            '</tool_call>',

            # Step 2: List a subdirectory
            'Found departments. Let me check dept_00.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "ls /dept_00"}}\n'
            '</tool_call>',

            # Step 3: Compress to save context
            'Explored root and dept_00. Compressing findings.\n'
            '<tool_call>\n'
            '{"name": "CompressExperience", "arguments": {'
            '"summary": "Index map:\\n- ctx_dirs - Directory listing from root and dept_00\\n'
            'Status: Explored 2 directories, searching for target", '
            '"db_blocks": [{"db_index": "ctx_dirs", "db_content": '
            '"Root: dept_00, dept_01, dept_02. dept_00 has sub_1/ directory and files."}]'
            '}}\n'
            '</tool_call>',

            # Step 4: Read back the stored info
            'Let me retrieve my directory notes.\n'
            '<tool_call>\n'
            '{"name": "ReadExperience", "arguments": {"db_index": "ctx_dirs"}}\n'
            '</tool_call>',

            # Step 5: Try grep for the target
            'Let me search for the target keyword.\n'
            '<tool_call>\n'
            '{"name": "execute_action", "arguments": {"action": "find classified"}}\n'
            '</tool_call>',

            # Step 6: Finish
            'Found the target. Task complete.\n'
            '<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>',
        ])

        agent = MemexAgent(
            llm=llm,
            environment=env,
            config=AgentConfig(max_steps=20, threshold=8000),
        )

        # Run episode
        episode = agent.run_episode()

        # Verify episode structure
        assert episode.task_success is True
        assert episode.total_steps == 6
        assert episode.num_compressions == 1
        assert episode.num_read_experience == 1
        assert len(episode.segments) == 2

        # Verify reward computation
        engine = RewardEngine()
        breakdown = engine.compute_breakdown(episode, threshold=8000)
        assert breakdown.task_reward == 1.0
        assert breakdown.total_return > 0
        assert breakdown.format_penalty == 0.0

        # Verify trajectory processing
        processor = TrajectoryProcessor()
        training_segs = processor.process_episode(episode)
        assert len(training_segs) == 2  # One per segment
        assert all(s.reward == training_segs[0].reward for s in training_segs)

    def test_full_pipeline_with_batch(self):
        """Test batch processing across multiple episodes."""
        env_factory = lambda: StressTestEnv(num_files=20, num_dirs=2, seed=42)

        # Two episodes with different outcomes
        ep1_llm = EchoBackend(responses=[
            'Done.\n<tool_call>\n'
            '{"name": "finish", "arguments": {"success": true}}\n'
            '</tool_call>'
        ])
        ep2_llm = EchoBackend(responses=[
            'Failed.\n<tool_call>\n'
            '{"name": "finish", "arguments": {"success": false}}\n'
            '</tool_call>'
        ])

        agent1 = MemexAgent(llm=ep1_llm, environment=env_factory())
        agent2 = MemexAgent(llm=ep2_llm, environment=env_factory())

        ep1 = agent1.run_episode(task_id="task_1")
        ep2 = agent2.run_episode(task_id="task_1")

        # Build training batch
        processor = TrajectoryProcessor()
        batch = processor.build_batch([("task_1", [ep1, ep2])])

        assert batch.size == 2
        # After advantage computation, success should have positive advantage
        advantages = [s.advantage for s in batch.segments]
        assert advantages[0] > advantages[1]  # Success > failure

    def test_config_files_parseable(self):
        """Verify all YAML configs are valid."""
        import yaml
        import os

        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        for filename in ["default.yaml", "alfworld.yaml", "training.yaml"]:
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                with open(path) as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict), f"{filename} is not a valid YAML dict"

    def test_all_source_modules_importable(self):
        """Verify all core source modules can be imported."""
        import src.models.memory
        import src.models.tools
        import src.models.trajectory
        import src.memory.tokenizer
        import src.memory.store
        import src.memory.controller
        import src.agent.tool_parser
        import src.agent.prompts
        import src.agent.loop
        import src.environments.base
        import src.environments.stress_test
        import src.environments.alfworld_env
        import src.llm.backend
        import src.llm.openai_backend
        import src.training.rewards
        import src.training.trajectory
        import src.training.grpo
        import src.training.rollout
