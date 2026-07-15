#!/usr/bin/env python3
"""Capture authoritative Qwen3 chat-template token IDs and padding masks."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default="a red fox under moonlight")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        text,
        return_tensors=None,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )
    args.output.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "chat_text": text,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
