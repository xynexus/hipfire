#!/usr/bin/env python3
"""
Verify per-bucket token counts by re-running each assembler and writing each
bucket to a temp file, tokenizing via llama-tokenize, and summing tokens.

This is for documentation accuracy in the README — the main corpus file
already exists; we don't rewrite it here.
"""
import os
import subprocess
import sys
from pathlib import Path

# Import the assembler functions from the build script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_calibration_mix_v1 import (  # noqa: E402
    BUDGET_CHARS, SEED, _load_hermes_rows, assemble_chat, assemble_code,
    assemble_tool_calls, assemble_wiki,
)
import random

REPO_ROOT = Path(__file__).resolve().parents[4]
GGUF = "/mnt/nas/kaden/hipfire/lucebox-quants/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_TOK = "/home/kaden/llama.cpp/build/bin/llama-tokenize"


def tok_count(text: str) -> int:
    """Tokenize text via llama-tokenize, return token count."""
    p = subprocess.run(
        [LLAMA_TOK, "-m", GGUF, "--stdin", "--show-count", "--log-disable", "--no-bos"],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    out = p.stdout.decode("utf-8", errors="replace") + p.stderr.decode("utf-8", errors="replace")
    for line in out.splitlines():
        if "Total number of tokens" in line:
            return int(line.split(":")[-1].strip())
    raise RuntimeError("llama-tokenize did not report token count: " + out[-500:])


def main() -> int:
    rng = random.Random(SEED)
    print("Loading hermes rows...", file=sys.stderr)
    hermes_rows = _load_hermes_rows()
    print(f"Loaded {len(hermes_rows)} hermes rows.", file=sys.stderr)

    # Use a null log file to avoid writing alongside the main build.
    devnull_path = Path("/tmp/calibration_mix_devnull.log")
    with open(devnull_path, "w") as devnull:
        wiki = assemble_wiki(BUDGET_CHARS["wiki"], devnull)
        tool = assemble_tool_calls(BUDGET_CHARS["tool"], list(hermes_rows), rng, devnull)
        chat = assemble_chat(BUDGET_CHARS["chat"], list(hermes_rows), rng, devnull)
        code = assemble_code(BUDGET_CHARS["code"], rng, devnull)

    counts = {
        "wiki": tok_count(wiki),
        "chat": tok_count(chat),
        "code": tok_count(code),
        "tool": tok_count(tool),
    }
    total_buckets = sum(counts.values())

    print("\nPer-bucket token counts:")
    for k, v in counts.items():
        bytes_count = {"wiki": len(wiki), "chat": len(chat), "code": len(code), "tool": len(tool)}[k]
        pct = v / total_buckets * 100
        print(f"  {k:6}: {v:>10,} tokens  ({bytes_count:>9,} B; {bytes_count/v:.2f} chars/tok; {pct:5.2f}% of total)")
    print(f"  TOTAL : {total_buckets:>10,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
