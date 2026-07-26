#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Freeze a reproducible Gemma 4 calibration-token window from text files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


GEMMA4_TOKENIZER_SHA256 = (
    "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=2048)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_revision(model: Path) -> str | None:
    resolved = model.resolve()
    if resolved.parent.name == "snapshots":
        return resolved.name
    cache_ref = resolved.parent.parent / "refs" / "main"
    return cache_ref.read_text().strip() if cache_ref.is_file() else None


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.tokens <= 0:
        raise ValueError("--offset must be non-negative and --tokens must be positive")

    tokenizer_path = args.model / "tokenizer.json"
    tokenizer_sha256 = sha256_file(tokenizer_path)
    if tokenizer_sha256 != GEMMA4_TOKENIZER_SHA256:
        raise ValueError(
            "the selected tokenizer is not the pinned Gemma 4 tokenizer: "
            f"got {tokenizer_sha256}, expected {GEMMA4_TOKENIZER_SHA256}"
        )

    corpus_parts = [path.read_bytes() for path in args.text]
    corpus = b"\n\n".join(corpus_parts)
    text = corpus.decode("utf-8")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    all_ids = tokenizer.encode(text, add_special_tokens=True).ids
    end = args.offset + args.tokens
    input_ids = all_ids[args.offset:end]
    if len(input_ids) != args.tokens:
        raise ValueError(
            f"corpus yielded only {len(all_ids)} tokens; requested window {args.offset}:{end}"
        )

    manifest = {
        "schema": "hipfire.gemma4.imatrix-input.v1",
        "model": str(args.model.resolve()),
        "revision": model_revision(args.model),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256,
        "corpus_paths": [str(path.resolve()) for path in args.text],
        "corpus_sha256": sha256_bytes(corpus),
        "corpus_parts": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in args.text
        ],
        "separator": "two_newlines",
        "token_offset": args.offset,
        "token_count": len(input_ids),
        "add_special_tokens": True,
        "input_ids": input_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Gemma 4 calibration input: {len(input_ids)} tokens, "
        f"tokenizer={tokenizer_sha256}, output={args.output}"
    )


if __name__ == "__main__":
    main()
