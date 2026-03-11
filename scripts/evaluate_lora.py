"""Evaluate a LoRA-adapted model on the Memex stress test.

Loads the real model weights + trained LoRA adapters via mlx_lm,
generates real tokens through MLX inference, and runs the full
Memex agent loop to measure honest success rates.

Usage:
    python scripts/evaluate_lora.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --adapter checkpoints/adapters.safetensors \\
        --num-tasks 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

from src.agent.loop import AgentConfig, MemexAgent
from src.environments.stress_test import StressTestEnv
from src.llm.backend import LLMBackend
from src.memory.tokenizer import Tokenizer
from src.models.memory import Message
from src.training.rewards import RewardEngine

logger = logging.getLogger(__name__)


class MLXLoRABackend(LLMBackend):
    """Real MLX inference backend that loads actual model + LoRA adapters."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        from mlx_lm import load

        logger.info(f"Loading model: {model_path}")
        self._model, self._tokenizer_hf = load(model_path)

        if adapter_path:
            from mlx_lm.lora import load_adapters
            from pathlib import Path
            # load_adapters expects a directory, not a file
            adapter_dir = Path(adapter_path)
            if adapter_dir.is_file():
                adapter_dir = adapter_dir.parent
            logger.info(f"Loading LoRA adapters from: {adapter_dir}")
            load_adapters(self._model, adapter_dir)
            logger.info("Adapters applied successfully")

        self._model_path = model_path
        self._adapter_path = adapter_path
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._tokenizer = Tokenizer()

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response using real MLX inference."""
        from mlx_lm import generate as mlx_generate

        # Build the chat prompt using the HF tokenizer's chat template
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        prompt = self._tokenizer_hf.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = mlx_generate(
            self._model,
            self._tokenizer_hf,
            prompt=prompt,
            max_tokens=max_tokens or self._default_max_tokens,
            verbose=False,
        )

        return response

    def count_tokens(self, text: str) -> int:
        return self._tokenizer.count(text)

    @property
    def model_name(self) -> str:
        name = self._model_path.split("/")[-1]
        if self._adapter_path:
            name += " +LoRA"
        return name


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-adapted Memex agent")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to trained LoRA adapters (omit for baseline)")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--threshold", type=int, default=8000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mode = "LoRA-adapted" if args.adapter else "Baseline"
    print(f"🧠 Memex Evaluation ({mode}) — {args.model}")
    if args.adapter:
        print(f"🔧 Adapter: {args.adapter}")
    print(f"📦 Environment: stress_test, Tasks: {args.num_tasks}")
    print(f"⚙️  Max steps: {args.max_steps}, Threshold: {args.threshold}")
    print("=" * 60)

    # Load the real model (with or without adapters)
    llm = MLXLoRABackend(
        model_path=args.model,
        adapter_path=args.adapter,
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

        if episode.task_success:
            total_success += 1
            print(f"✅ ({episode.total_steps} steps, {elapsed:.1f}s)")
        else:
            print(f"❌ ({episode.total_steps} steps, {elapsed:.1f}s)")

        results.append({
            "task_id": task_id,
            "success": episode.task_success,
            "steps": episode.total_steps,
            "return": round(breakdown.total_return, 4),
            "time_s": round(elapsed, 1),
        })

    # Summary
    print("\n" + "=" * 60)
    success_rate = 100 * total_success / args.num_tasks
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_return = sum(r["return"] for r in results) / len(results)
    print(f"📊 Results ({mode}): {total_success}/{args.num_tasks} success ({success_rate:.0f}%)")
    print(f"   Avg steps:  {avg_steps:.1f}")
    print(f"   Avg return: {avg_return:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
