"""Run a Memex agent episode on an environment.

Usage:
    # Stress test (default — no external deps)
    python scripts/run_agent.py

    # With specific model
    python scripts/run_agent.py --model qwen2.5:3b

    # ALFWorld (requires alfworld package)
    python scripts/run_agent.py --env alfworld --task "put the book on desk"

    # Custom Ollama URL
    python scripts/run_agent.py --base-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

# Add project root to path
sys.path.insert(0, ".")

from src.agent.loop import AgentConfig, MemexAgent
from src.agent.prompts import (
    ALFWORLD_ENVIRONMENT_PROMPT,
    ALFWORLD_MEMORY_ADDENDUM,
)
from src.environments.stress_test import StressTestEnv
from src.llm.openai_backend import OpenAIBackend


def load_config(config_path: str | None) -> dict:
    """Load YAML config, falling back to defaults."""
    if config_path:
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def build_environment(env_type: str, task_id: str | None, config: dict):
    """Create the appropriate environment."""
    if env_type == "stress_test":
        env_cfg = config.get("environment", {})
        return StressTestEnv(
            num_files=env_cfg.get("num_files", 1000),
            num_dirs=env_cfg.get("num_dirs", 10),
            max_depth=env_cfg.get("max_depth", 5),
            seed=env_cfg.get("seed", 42),
        )
    elif env_type == "alfworld":
        from src.environments.alfworld_env import ALFWorldModifiedEnv
        env_cfg = config.get("environment", {})
        return ALFWorldModifiedEnv(
            max_obs_tokens=env_cfg.get("max_obs_tokens", 500),
            look_limit=env_cfg.get("look_limit", 1),
            hide_initial_obs=env_cfg.get("hide_initial_obs", True),
            hide_commands=env_cfg.get("hide_commands", True),
        )
    else:
        raise ValueError(f"Unknown environment: {env_type}")


def main():
    parser = argparse.ArgumentParser(description="Run a Memex agent episode")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--env", type=str, default="stress_test",
                        choices=["stress_test", "alfworld"],
                        help="Environment to run (default: stress_test)")
    parser.add_argument("--model", type=str, default="qwen2.5:3b",
                        help="Ollama model name")
    parser.add_argument("--base-url", type=str,
                        default="http://localhost:11434/v1",
                        help="LLM API base URL")
    parser.add_argument("--task", type=str, default=None,
                        help="Task ID or description")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--threshold", type=int, default=8000,
                        help="Compression threshold τ")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config = load_config(args.config)

    # LLM config (CLI overrides YAML)
    llm_cfg = config.get("llm", {})
    model = args.model or llm_cfg.get("model", "qwen2.5:3b")
    base_url = args.base_url or llm_cfg.get("base_url", "http://localhost:11434/v1")

    print(f"🧠 Memex Agent — {model} via {base_url}")
    print(f"📦 Environment: {args.env}")
    print(f"⚙️  Max steps: {args.max_steps}, Threshold: {args.threshold}")
    print()

    # Build components
    llm = OpenAIBackend(model=model, base_url=base_url, api_key="ollama")
    env = build_environment(args.env, args.task, config)

    agent_config = AgentConfig(
        max_steps=args.max_steps,
        threshold=args.threshold,
        temperature=args.temperature,
    )

    if args.env == "alfworld":
        agent_config.environment_prompt = ALFWORLD_ENVIRONMENT_PROMPT
        agent_config.memory_addendum = ALFWORLD_MEMORY_ADDENDUM

    agent = MemexAgent(llm=llm, environment=env, config=agent_config)

    # Run episode
    print("▶️  Starting episode...")
    episode = agent.run_episode(task_id=args.task)

    # Report results
    print()
    print("=" * 60)
    print(f"✅ Task Success: {episode.task_success}")
    print(f"📊 Total Steps: {episode.total_steps}")
    print(f"🗜️  Compressions: {episode.num_compressions}")
    print(f"📖 ReadExperience calls: {episode.num_read_experience}")
    print(f"🏆 Terminal Reward: {episode.terminal_reward}")
    print(f"📦 Segments: {len(episode.segments)}")

    # Compute detailed reward breakdown
    from src.training.rewards import RewardEngine
    engine = RewardEngine()
    breakdown = engine.compute_breakdown(episode, threshold=args.threshold)
    print(f"\n📈 Reward Breakdown:")
    print(f"   Task reward:        {breakdown.task_reward}")
    print(f"   Context penalty:    {breakdown.context_penalty:.4f}")
    print(f"   Redundancy penalty: {breakdown.redundancy_penalty:.4f}")
    print(f"   Format penalty:     {breakdown.format_penalty:.4f}")
    print(f"   Total return:       {breakdown.total_return:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
