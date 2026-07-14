#!/usr/bin/env python3
"""Generate deterministic Sequential-NIAH fixtures.

Writes seqniah_{8k,16k}.jsonl. Sequential-NIAH embeds an ORDERED chain of K
numbered "step" needles at evenly spaced depths; the model must retrieve every
step of the sequence, not just the most recent one. Records reuse the
multi-needle NIAH schema (needles / expected_answer_substrings / min_recovered /
filler_text / question) so the eval materializer parses them unchanged.

PASS bar: min_recovered = K (the whole chain — a sequential task is only correct
if every step is recovered).

Scoring ceiling: the eval runner scores by substring recall, so it checks that
all K step secrets appear, not that they appear in order. Strict ordered scoring
is a follow-up; this v1 vendors a faithful ordered-chain haystack.

Re-running with no args is byte-identical (single seeded RNG).
"""

import json
import random
import sys
from pathlib import Path

SEED = 0x5407_5E00  # distinct from single (0x5407_FFFF) and multi (0x5407_FEEE)
RNG = random.Random(SEED)

WORDS = (
    "the quick brown fox jumps over the lazy dog and then settles down for a "
    "long restful nap on a warm patch of sunlit grass while a gentle breeze "
    "carries the scent of distant pine forests across the meadow where small "
    "wildflowers nod their heads in agreement with the rhythm of the afternoon "
    "and time itself seems to pause for just a moment so that everything alive "
    "can breathe in synchrony with the slow steady pulse of the surrounding "
    "world before the sun begins its descent toward the western horizon "
    "painting clouds in shades of amber rose and indigo as evening approaches"
).split()

# Ordered chain of K steps; each carries a unique secret token.
STEP_SECRETS = [
    "amber-lynx-3108",
    "cobalt-ibis-7742",
    "verdant-oryx-5519",
    "crimson-stork-2043",
    "onyx-marlin-8867",
]
K = len(STEP_SECRETS)
DEPTHS = [(i + 1) / (K + 1) for i in range(K)]  # 1/6, 2/6, ... evenly spaced

QUESTION = (
    "The document above describes a procedure as a numbered sequence of steps, "
    "each revealing a secret code. List the secret code for step 1 through "
    f"step {K}, in order, one per line."
)
EXPECTED_SUBSTRINGS = list(STEP_SECRETS)
MIN_RECOVERED = K  # sequential: the full chain must be recovered

CHARS_PER_TOKEN = 3.5
CONTEXT_SIZES = [8192, 16384]


def make_filler(approx_tokens: int) -> str:
    target_chars = int(approx_tokens * CHARS_PER_TOKEN)
    out = []
    chars = 0
    while chars < target_chars:
        w = WORDS[RNG.randrange(len(WORDS))]
        out.append(w)
        chars += len(w) + 1
    return " ".join(out)


def assemble(context_tokens: int) -> str:
    """Place K ordered step needles at DEPTHS inside context_tokens of filler."""
    fragments = []
    prev_depth = 0.0
    for i, depth in enumerate(DEPTHS):
        slice_tokens = int(context_tokens * (depth - prev_depth))
        fragments.append(make_filler(slice_tokens))
        fragments.append(
            f"\n\nStep {i + 1} of the procedure: the secret code is {STEP_SECRETS[i]}.\n\n"
        )
        prev_depth = depth
    tail_tokens = int(context_tokens * (1.0 - prev_depth))
    fragments.append(make_filler(tail_tokens))
    return "".join(fragments)


def main():
    out_dir = Path(__file__).parent
    for ctx in CONTEXT_SIZES:
        RNG.seed(SEED + ctx)
        filler_text = assemble(ctx)
        for sub in EXPECTED_SUBSTRINGS:
            n = filler_text.count(sub)
            assert n == 1, f"step secret {sub!r} appears {n}x at ctx={ctx}"
        record = {
            "context_tokens": ctx,
            "needles": [
                {"key": f"step {i + 1}", "value": STEP_SECRETS[i], "depth": DEPTHS[i]}
                for i in range(K)
            ],
            "question": QUESTION,
            "expected_answer_substrings": EXPECTED_SUBSTRINGS,
            "min_recovered": MIN_RECOVERED,
            "filler_text": filler_text,
        }
        out_path = out_dir / f"seqniah_{ctx // 1024}k.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
        approx_chars = len(filler_text)
        print(
            f"wrote {out_path.name}: {approx_chars} chars "
            f"(~{approx_chars / CHARS_PER_TOKEN:.0f} tokens, "
            f"{K} ordered steps at depths {[round(d, 3) for d in DEPTHS]})"
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
