"""Convert SFT trajectories to mlx_lm LoRA training format.

Reads the OpenAI-format JSONL from generate_sft_data.py and converts it
into the {train, valid, test}.jsonl directory format that mlx_lm.lora expects.

Each entry becomes: {"text": "<full chat-formatted conversation>"}

Usage:
    python scripts/convert_sft_to_mlx.py \\
        --input data/sft_trajectories.jsonl \\
        --output data/mlx_lora_data/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def messages_to_chat_text(messages: list[dict]) -> str:
    """Convert OpenAI message format to a single chat-formatted text string.
    
    Uses the ChatML format that Qwen 2.5 was trained on:
        <|im_start|>system\n...<|im_end|>\n
        <|im_start|>user\n...<|im_end|>\n
        <|im_start|>assistant\n...<|im_end|>\n
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Map 'tool' role to 'user' for compatibility
        if role == "tool":
            role = "user"
            content = f"Observation: {content}"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to mlx_lm format")
    parser.add_argument("--input", type=str, default="data/sft_trajectories.jsonl",
                        help="Input JSONL from generate_sft_data.py")
    parser.add_argument("--output", type=str, default="data/mlx_lora_data",
                        help="Output directory for {train, valid, test}.jsonl")
    parser.add_argument("--valid-split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run generate_sft_data.py first to collect golden trajectories.")
        return

    # Load all trajectories
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = messages_to_chat_text(data["messages"])
            records.append({"text": text})

    if not records:
        print("Error: No trajectories found in input file.")
        return

    print(f"Loaded {len(records)} golden trajectories")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_valid = max(1, int(len(records) * args.valid_split))
    valid_data = records[:n_valid]
    train_data = records[n_valid:]

    # Write files
    for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", valid_data)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for record in split_data:
                f.write(json.dumps(record) + "\n")
        print(f"  {split_name}.jsonl: {len(split_data)} examples")

    print(f"\nOutput written to: {output_dir}/")
    print("Ready for mlx_lm.lora training.")


if __name__ == "__main__":
    main()
