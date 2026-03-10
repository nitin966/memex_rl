"""Evaluate a Memex agent across multiple tasks.

Usage:
    # Evaluate on 5 stress test tasks
    python scripts/evaluate.py --env stress_test --num-tasks 5

    # With specific model
    python scripts/evaluate.py --model qwen2.5:3b --num-tasks 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import yaml

sys.path.insert(0, ".")

from src.agent.loop import AgentConfig, MemexAgent
from src.agent.prompts import ALFWORLD_ENVIRONMENT_PROMPT, ALFWORLD_MEMORY_ADDENDUM
from src.environments.stress_test import StressTestEnv
from src.llm.openai_backend import OpenAIBackend
from src.training.rewards import RewardEngine


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memex agent")
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--base-url", type=str,
                        default="http://localhost:11434/v1")
    parser.add_argument("--env", type=str, default="stress_test",
                        choices=["stress_test", "alfworld"])
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--threshold", type=int, default=8000)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"🧠 Memex Evaluation — {args.model}")
    print(f"📦 Environment: {args.env}, Tasks: {args.num_tasks}")
    print(f"⚙️  Max steps: {args.max_steps}, Threshold: {args.threshold}")
    print("=" * 60)

    llm = OpenAIBackend(
        model=args.model, base_url=args.base_url, api_key="ollama"
    )
    reward_engine = RewardEngine()

    results = []
    total_success = 0

    for i in range(args.num_tasks):
        seed = 42 + i
        env = StressTestEnv(num_files=1000, num_dirs=10, seed=seed)
        agent = MemexAgent(
            llm=llm,
            environment=env,
            config=AgentConfig(
                max_steps=args.max_steps,
                threshold=args.threshold,
            ),
        )

        task_id = f"stress_{seed}"
        print(f"\n[{i+1}/{args.num_tasks}] Task: {task_id}...", end=" ", flush=True)

        start = time.time()
        episode = agent.run_episode(task_id=task_id)
        elapsed = time.time() - start

        breakdown = reward_engine.compute_breakdown(episode, args.threshold)

        result = {
            "task_id": task_id,
            "success": episode.task_success,
            "steps": episode.total_steps,
            "compressions": episode.num_compressions,
            "reads": episode.num_read_experience,
            "return": round(breakdown.total_return, 4),
            "time_s": round(elapsed, 1),
        }
        results.append(result)

        if episode.task_success:
            total_success += 1
            print(f"✅ ({episode.total_steps} steps, {elapsed:.1f}s)")
        else:
            print(f"❌ ({episode.total_steps} steps, {elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Results: {total_success}/{args.num_tasks} success "
          f"({100*total_success/args.num_tasks:.0f}%)")
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_return = sum(r["return"] for r in results) / len(results)
    avg_compress = sum(r["compressions"] for r in results) / len(results)
    print(f"   Avg steps:        {avg_steps:.1f}")
    print(f"   Avg return:       {avg_return:.4f}")
    print(f"   Avg compressions: {avg_compress:.1f}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "environment": args.env,
                "num_tasks": args.num_tasks,
                "success_rate": total_success / args.num_tasks,
                "results": results,
            }, f, indent=2)
        print(f"\n📁 Results saved to {args.output}")


if __name__ == "__main__":
    main()
