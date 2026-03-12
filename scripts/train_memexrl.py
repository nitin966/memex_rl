"""GRPO training loop for MemexRL.

Generates rollouts via the agent loop and computes group-relative advantages.
The --backend mlx path uses a placeholder model for the backward pass;
for real LoRA fine-tuning, use the mlx_lm CLI (see README).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to python path to resolve src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.environments.alfworld_env import ALFWorldModifiedEnv
from src.environments.stress_test import StressTestEnv
from src.llm.openai_backend import OpenAIBackend
from src.llm.sglang_backend import SGLangBackend
from src.training.grpo import GRPOConfig, GRPOTrainer
from src.training.mlx_grpo import MLXTrainerWrapper
from src.training.rollout import RolloutConfig, RolloutEngine

logger = logging.getLogger(__name__)


def get_env_factory(env_name: str):
    """Return a callable that creates a fresh environment."""
    if env_name == "alfworld":
        return lambda: ALFWorldModifiedEnv()
    elif env_name == "stress_test":
        return lambda: StressTestEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def main():
    parser = argparse.ArgumentParser(description="Run MemexRL GRPO Training Loop")
    parser.add_argument("--env", type=str, default="stress_test",
                        choices=["stress_test", "alfworld"],
                        help="Environment to use for training")
    parser.add_argument("--model", type=str, default="qwen2.5:3b",
                        help="Model to use as the policy")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "sglang", "mlx"])
    
    # GRPO Hyperparameters
    parser.add_argument("--group-size", type=int, default=8,
                        help="G rollouts per prompt (paper uses 8)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=10,
                        help="Number of GRPO updates per epoch")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Initializing MemexRL Training Pipeline...")
    
    # 1. Initialize the LLM Backend (Policy)
    if args.backend == "sglang":
        try:
            llm = SGLangBackend(model=args.model, base_url=args.base_url)
        except ImportError:
            logger.error("SGLang not installed. Defaulting to openai backend.")
            llm = OpenAIBackend(model=args.model, base_url=args.base_url)
    else:
        llm = OpenAIBackend(model=args.model, base_url=args.base_url)

    # 2. Environment Factory
    env_factory = get_env_factory(args.env)
    
    # 3. Rollout Engine
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_steps=100,
        context_window=32_768,
        threshold=8_000,
    )
    rollout_engine = RolloutEngine(
        llm=llm,
        env_factory=env_factory,
        config=rollout_config,
    )
    
    # 4. GRPO Trainer
    grpo_config = GRPOConfig(group_size=args.group_size)
    trainer = GRPOTrainer(config=grpo_config)

    # Wrap with MLX if requested
    if args.backend == "mlx":
        logger.info("Setting up Apple MLX LoRA backend...")
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten
            import mlx.nn as nn
            import mlx.optimizers as optim
            import mlx_lm
            import mlx_lm.lora

            logger.info(f"Loading base model {args.model} via mlx_lm...")
            
            # Map Ollama tags to HF hub for mlx_lm if needed
            hf_path = args.model
            if ":" in args.model or "/" not in args.model:
                hf_path = "Qwen/Qwen2.5-3B-Instruct"  # Fallback
                if "7b" in args.model.lower():
                    hf_path = "Qwen/Qwen2.5-7B-Instruct"
                    
            mlx_model, tokenizer = mlx_lm.load(hf_path)
            
            # Attach tokenizer to model instance so our MLXGRPOLoss can use it
            mlx_model.tokenizer = tokenizer
            
            # Convert linear layers to LoRA layers
            lora_config = {"keys": ["self_attn.q_proj", "self_attn.v_proj"]}
            mlx_lm.lora.linear_to_lora_layers(
                mlx_model, 
                num_layers=4, 
                config={"rank": 8, "alpha": 16, "scale": 16.0, "dropout": 0.0, **lora_config}
            )
            # Apply gradient checkpointing to reduce memory overhead
            from mlx_lm.tuner.trainer import grad_checkpoint
            if hasattr(mlx_model, "model") and hasattr(mlx_model.model, "layers"):
                logger.info("Enabling gradient checkpointing for MLX.")
                grad_checkpoint(mlx_model.model.layers[0])
            
            mlx_optimizer = optim.Adam(learning_rate=grpo_config.learning_rate)
            
            trainer = MLXTrainerWrapper(
                base_trainer=trainer,
                model=mlx_model,
                optimizer=mlx_optimizer
            )
        except ImportError:
            logger.error("Failed to import MLX. Ensure 'mlx' is installed via pip.")
            return

    logger.info(f"Configuration: Env={args.env}, Model={args.model}")
    logger.info(f"GRPO Group Size (G)={args.group_size}")

    # 5. Training Loop
    for epoch in range(args.epochs):
        logger.info(f"=== Starting Epoch {epoch + 1}/{args.epochs} ===")
        
        for step in range(args.steps_per_epoch):
            logger.info(f"--- Step {step + 1}/{args.steps_per_epoch} ---")
            
            # Form a batch of tasks/prompts.
            # In stress_test env, task generation is handled internally, 
            # so we just pass a generic task ID.
            task_ids = [f"task_epoch_{epoch}_step_{step}"]
            
            # Phase 1: Generate Rollouts (G trajectories per prompt)
            logger.info(f"Generating {args.group_size} rollouts...")
            rollout_groups = rollout_engine.generate_rollouts(task_ids)
            
            # Phase 2: Compute Advantages and Loss
            logger.info("Processing trajectories and computing group-relative advantages...")
            result = trainer.train_step(rollout_groups)
            
            logger.info(f"GRPO Step Complete: Mean Reward={result.mean_reward:.4f}, Mean Advantage={result.mean_advantage:.4f}, Segments={result.num_segments}")
            
    if args.backend == "mlx":
        save_path = "checkpoints/memex_qwen_lora.safetensors"
        Path("checkpoints").mkdir(exist_ok=True)
        trainer.save_adapters(save_path)
            
    logger.info("Training loop completed successfully.")


if __name__ == "__main__":
    main()
