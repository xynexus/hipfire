"""Build calibration_corpus.jsonl for D3 activation surveys.

Composition (frozen as of 2026-05-06):
  7 prompts from the 2026-05-05 issue #171 reproducer matrix:
    1. agent_prompt — verbatim from
       docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/repro_failing.jsonl
    2-7. canonical reconstructions matching the matrix categories
       (sheep, capital, code_simple, code_complex, prose, math).
       The originals were not committed verbatim; reconstructions match
       the prompt-class fingerprint that exposed the 3.6-A3B cliff.
  25 chunks sampled from benchmarks/calib/calib-1m.txt
       (md5 c1879341cb2d4bcf06ead9d1c02ef5fa, wikitext-103-raw-v1).
       Deterministic seed=20260506 PRNG; each chunk = 600 chars
       starting at a sampled offset. Wikitext gives general-domain
       coverage outside the failure-prone agentic distribution.

Output format: JSONL, one record per line:
  {"id": "matrix_agent_prompt", "source": "issue-171-repro", "prompt": "..."}
  {"id": "wiki_00", "source": "calib-1m", "offset": N, "prompt": "..."}

The corpus md5 is recorded in d3_runner.py summary so cross-session
results can be verified against the same prompt bytes.

Reproducibility: rerun this script (no args) to regenerate
calibration_corpus.jsonl identically. Commit the jsonl. Do not edit
calibration_corpus.jsonl by hand.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts" / "quant-survey" / "calibration_corpus.jsonl"
WIKI = ROOT / "benchmarks" / "calib" / "calib-1m.txt"
EXPECTED_WIKI_MD5 = "c1879341cb2d4bcf06ead9d1c02ef5fa"

SEED = 20260506
N_WIKI_CHUNKS = 25
WIKI_CHUNK_CHARS = 600


# ---------------------------------------------------------------------------
# Matrix prompts
# ---------------------------------------------------------------------------

# 1. The long agentic prompt verbatim from repro_failing.jsonl. This is the
#    canonical 3.6-A3B attractor reproducer.
AGENT_PROMPT = (
    "You are an autonomous coding agent reviewing the current state of the hipfire repository before making a pull request. "
    "The hipfire project layout is:\n\n"
    "  crates/\n"
    "    rdna-compute/        - HIP runtime FFI + per-kernel BW profiler\n"
    "    hip-bridge/          - low-level HIP/HSA bindings\n"
    "    hipfire-runtime/     - inference orchestrator, daemon, KV cache, eviction\n"
    "    hipfire-arch-qwen35/ - Qwen3.5/3.6 dense + MoE forward path\n"
    "    hipfire-arch-llama/  - dense Llama-family forward path\n"
    "    hipfire-arch-qwen35-vl/ - VL variant\n"
    "    hipx/                - NPU (AIE-2P) bindings + dmabuf import\n"
    "  kernels/               - HIP source for every kernel; .hipfire_kernels/<gfx> hosts compiled blobs\n"
    "  scripts/               - coherence/speed/pflash gates, bench harnesses, install\n"
    "  benchmarks/            - corpora + saved bench results\n"
    "  docs/plans/             - PRDs (engine-modularization, ddtree-path-b, moe-egpu-offload)\n\n"
    "The most recent commits on master are:\n"
    "  fa0592d fix(qwen35-moe): split moe_softmax_topk_renorm into softmax + topk_renorm (#164)\n"
    "  7567e9d fix(install.ps1): honor CARGO_TARGET_DIR (#161)\n"
    "  f5ee068 Generalize to custom cargo target directories (#159)\n"
    "  c584605 fix(kernels/moe): clamp top-k winner to valid expert range (#156)\n"
    "  0005cc8 chore(release): finalize 0.1.20\n\n"
    "Your task: write a short summary (no more than 200 words) that explains, for a contributor unfamiliar with the project, "
    "what hipfire is, what crates own which responsibilities, and which two recent merges most directly affect MoE quality. "
    "End the summary with a single concrete recommendation for the next contributor task. "
    "Do not invent file paths or commit hashes that are not listed above. Keep the tone professional.\n"
)

MATRIX_PROMPTS = [
    ("matrix_agent_prompt", AGENT_PROMPT),
    ("matrix_sheep", "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Show your reasoning step by step, then give a final answer."),
    ("matrix_capital", "What is the capital of France? Answer in one sentence."),
    ("matrix_code_simple", "Write a Python one-line function that returns the square of its integer argument. Provide only the function, no explanation."),
    ("matrix_code_complex", "Write a complete, iterative Python implementation of the Fibonacci sequence. Include a docstring describing parameters and return value, and an example usage block under `if __name__ == \"__main__\":`. Do not use recursion."),
    ("matrix_prose", "Explain how an ATM machine works in plain English for someone who has never used one. Keep the explanation under 80 words and use a single paragraph."),
    ("matrix_math", "Carol bought 4 pizzas for a party. Each pizza is cut into 8 slices. If Carol ate 10 slices herself, how many slices remain for her guests? Show your arithmetic and state the final number."),
]


# ---------------------------------------------------------------------------
# Wikitext chunk sampling
# ---------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_wikitext_chunks(path: Path, n: int, chunk_chars: int, seed: int) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    rng = random.Random(seed)
    chunks: list[tuple[int, str]] = []
    max_off = len(text) - chunk_chars
    seen_offsets: set[int] = set()
    while len(chunks) < n:
        off = rng.randrange(0, max_off)
        if off in seen_offsets:
            continue
        seen_offsets.add(off)
        # Snap to the next paragraph boundary so a chunk doesn't start mid-word.
        nl = text.find("\n", off)
        if nl == -1 or nl - off > 200:
            start = off
        else:
            start = nl + 1
        chunk = text[start : start + chunk_chars].rstrip()
        if len(chunk) < chunk_chars // 2:
            continue
        chunks.append((start, chunk))
    return chunks


def main() -> int:
    if not WIKI.exists():
        raise SystemExit(f"missing wikitext source: {WIKI}")

    md5 = md5_file(WIKI)
    if md5 != EXPECTED_WIKI_MD5:
        raise SystemExit(
            f"calib-1m.txt md5 mismatch: got {md5}, expected {EXPECTED_WIKI_MD5}. "
            f"calibration corpus would not be reproducible."
        )

    chunks = sample_wikitext_chunks(WIKI, N_WIKI_CHUNKS, WIKI_CHUNK_CHARS, SEED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for pid, prompt in MATRIX_PROMPTS:
            rec = {"id": pid, "source": "issue-171-repro" if pid == "matrix_agent_prompt" else "issue-171-matrix-reconstruction", "prompt": prompt}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for i, (off, chunk) in enumerate(chunks):
            rec = {"id": f"wiki_{i:02d}", "source": "calib-1m", "offset": off, "prompt": chunk}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT}")
    print(f"  records: {len(MATRIX_PROMPTS) + len(chunks)} ({len(MATRIX_PROMPTS)} matrix + {len(chunks)} wikitext)")
    print(f"  source calib-1m.txt md5: {md5}")
    print(f"  output md5: {out_md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
