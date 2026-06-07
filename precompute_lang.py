"""
Precompute T5-XXL language embedding for a task instruction.
Run this on a machine with enough RAM (>=24GB) or on Colab (A100/T4).

Usage:
    python precompute_lang.py --task "pick up the workpiece and place it on the table"

Output:
    lang_embed.pt  (small file, load on any machine for inference)
"""
import argparse
import pathlib

import torch
from transformers import T5EncoderModel, T5Tokenizer

PRETRAINED_T5 = "google/t5-v1_1-xxl"
OUTPUT = pathlib.Path(__file__).parent / "lang_embed.pt"
MAX_LENGTH = 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        default="pick up the workpiece and place it on the table")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading T5-XXL tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_T5)

    print(f"Loading T5-XXL encoder (requires ~22GB RAM in bfloat16)...")
    encoder = T5EncoderModel.from_pretrained(
        PRETRAINED_T5, torch_dtype=torch.bfloat16
    ).to(args.device)
    encoder.eval()

    print(f"Encoding: '{args.task}'")
    inputs = tokenizer(
        args.task,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    ).to(args.device)

    with torch.no_grad():
        embed = encoder(**inputs).last_hidden_state  # (1, 1024, 4096)

    torch.save(embed.cpu(), OUTPUT)
    print(f"Saved to {OUTPUT}  shape={tuple(embed.shape)}")


if __name__ == "__main__":
    main()
