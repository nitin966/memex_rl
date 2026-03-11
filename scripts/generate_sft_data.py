"""Generate Supervised Fine-Tuning (SFT) data for Memex(RL).

This script runs the Memex Agent across randomized task instances.
It filters for perfectly successful trajectories (task success == True,
format penalty == 0.0) and exports the step-by-step history into
an OpenAI-compatible JSONL format suitable for SFT tools like Unsloth or MLX.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to python path to resolve src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.loop import MemexAgent
from src.environments.alfworld_env import ALFWorldModifiedEnv
from src.environments.stress_test import StressTestEnv
from src.llm.openai_backend import OpenAIBackend
from src.llm.sglang_backend import SGLangBackend
from src.training.rewards import RewardEngine

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data for MemexRL")
    parser.add_argument("--env", type=str, default="stress_test",
                        choices=["stress_test", "alfworld"],
                        help="Environment to use for data generation")
    parser.add_argument("--model", type=str, default="qwen2.5:3b",
                        help="Model to use for autonomous generation")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "sglang"])
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to attempt")
    parser.add_argument("--output", type=str, default="data/sft_trajectories.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--threshold", type=int, default=8000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Make output dir
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    if args.env == "alfworld":
        try:
            env = ALFWorldModifiedEnv()
        except ImportError:
            logger.error("ALFWorld not installed. Run: pip install -e '.[alfworld]'")
            return
    else:
        env = StressTestEnv()

    # Initialize Backend
    if args.backend == "sglang":
        llm = SGLangBackend(model=args.model, base_url=args.base_url)
    else:
        llm = OpenAIBackend(model=args.model, base_url=args.base_url)

    agent = MemexAgent(environment=env, llm=llm)
    agent.config.max_steps = args.max_steps
    agent.config.threshold = args.threshold
    
    reward_engine = RewardEngine()

    successful_trajectories = 0
    
    logger.info(f"Starting SFT generation: {args.num_episodes} attempts requested.")

    with open(out_path, "a" if out_path.exists() else "w", encoding="utf-8") as f:
        for i in range(args.num_episodes):
            logger.info(f"Episode {i+1}/{args.num_episodes}...")
            
            episode = agent.run_episode()
            breakdown = reward_engine.compute_breakdown(episode, agent.config.threshold)
            
            # STRCIT FILTER: Only keep perfect trajectories
            if not episode.task_success:
                logger.info("  -> Failed task. Discarding.")
                continue
            if breakdown.format_penalty > 0:
                logger.info("  -> Formatting errors detected. Discarding.")
                continue
                
            successful_trajectories += 1
            logger.info(f"  -> SUCCESS! Saving trajectory (Total steps: {breakdown.total_steps}).")
            
            # Reconstruct the conversation context for SFT
            # Based on how the MemoryController injects history
            all_steps = episode.all_steps()
            
            # The initial setup
            system_prompt = agent.controller._system_prompt
            task_instruction = agent.controller._task_instruction
            
            # We reconstruct the exact message stream that the LLM saw at each step
            # to provide identical sequence modeling targets.
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_instruction}
            ]
            
            # Replay the sequence
            working_tokens = 0
            for step in all_steps:
                # 1. System state injection
                status_msg = f"[Context Status: working tokens={working_tokens}, threshold={agent.config.threshold}]"
                messages.append({"role": "system", "content": status_msg})
                
                # 2. The Assistant's generation (Target for SFT)
                # We need the exact string it generated (thinking + tool call)
                target_text = ""
                if step.thinking:
                    target_text += step.thinking + "\n"
                if step.tool_call:
                    target_text += f"<tool_call>\n{step.tool_call.raw_text}\n</tool_call>"
                    
                messages.append({"role": "assistant", "content": target_text})
                
                # Save this exact slice as a valid SFT example if we want
                # step-level granularity, but standard SFT uses the full multi-turn dialog.
                
                # 3. Handle context modifications based on the tool
                if step.tool_call.name == "CompressExperience":
                    # In true Memex, the working context is rewritten here.
                    # For SFT data, we simulate the compression rewrite.
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_instruction},
                        {"role": "assistant", "content": step.observation} # The rewrite
                    ]
                    # Note: We don't save the full trajectory to disk if a compression occurred 
                    # in the middle of a string because OpenAI format represents a contiguous dialog. 
                    # Instead, we just append the whole episodic stream as one JSONL entry.
                else:
                    messages.append({"role": "tool", "content": step.observation})
                
                # Approximate token update
                working_tokens += len(target_text) // 4 + len(step.observation) // 4
                
            # Write the multi-turn trajectory as one JSONL line
            json_record = {"messages": messages}
            f.write(json.dumps(json_record) + "\n")
            f.flush()

    logger.info("=" * 50)
    logger.info("SFT Generation Complete")
    logger.info(f"Total attempts: {args.num_episodes}")
    logger.info(f"High-quality trajectories saved: {successful_trajectories}")
    logger.info(f"Output saved to: {out_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
