#!/usr/bin/env python3
"""Generate deterministic RULER-style fixtures (vendored slices).

RULER (NVIDIA) is a synthetic long-context generator, not a static dataset, so
hipfire vendors generated slices of its two most load-bearing task families:

  * ruler_niah_{4k,8k}: S-NIAH — a "special magic number" needle inserted into a
    filler haystack; the model must retrieve the number for a given word.
  * ruler_vt_{4k,8k}: variable tracking — a chain of variable assignments
    (V1 = <num>; V2 = V1; ...) scattered through the haystack; the model must
    report every variable that ends up equal to the target value.

Records use the multi-needle NIAH schema (needles / expected_answer_substrings /
min_recovered / filler_text / question) so the eval materializer parses them
unchanged. Scoring is substring recall (RULER's recall-based metric).

Re-running with no args is byte-identical (single seeded RNG).
"""

import json
import random
import string
import sys
from pathlib import Path

SEED = 0x5407_2010  # distinct from niah/seqniah seeds
RNG = random.Random(SEED)

WORDS = (
    "the quick brown fox jumps over the lazy dog and then settles down for a "
    "long restful nap on a warm patch of sunlit grass while a gentle breeze "
    "carries the scent of distant pine forests across the meadow where small "
    "wildflowers nod their heads in agreement with the rhythm of the afternoon "
    "and time itself seems to pause for just a moment so that everything alive "
    "can breathe in synchrony with the slow steady pulse of the surrounding world"
).split()

CHARS_PER_TOKEN = 3.5
CONTEXT_SIZES = [4096, 8192]


def make_filler(approx_tokens: int) -> str:
    target_chars = int(approx_tokens * CHARS_PER_TOKEN)
    out, chars = [], 0
    while chars < target_chars:
        w = WORDS[RNG.randrange(len(WORDS))]
        out.append(w)
        chars += len(w) + 1
    return " ".join(out)


def insert_at_depths(context_tokens: int, inserts: list[tuple[float, str]]) -> str:
    """Insert (depth, sentence) fragments into filler of context_tokens size."""
    frags, prev = [], 0.0
    for depth, sentence in sorted(inserts):
        frags.append(make_filler(int(context_tokens * (depth - prev))))
        frags.append(f"\n\n{sentence}\n\n")
        prev = depth
    frags.append(make_filler(int(context_tokens * (1.0 - prev))))
    return "".join(frags)


def make_niah(ctx: int) -> dict:
    """S-NIAH: one magic-number needle for a keyword."""
    word = "".join(RNG.choice(string.ascii_lowercase) for _ in range(7))
    number = str(RNG.randint(1_000_000, 9_999_999))
    needle = f"One of the special magic numbers for {word} is {number}."
    filler = insert_at_depths(ctx, [(0.5, needle)])
    assert filler.count(number) == 1
    return {
        "context_tokens": ctx,
        "needles": [{"key": word, "value": number, "depth": 0.5}],
        "question": (
            f"What is the special magic number for {word} mentioned in the text above? "
            "Answer with only the number."
        ),
        "expected_answer_substrings": [number],
        "min_recovered": 1,
        "filler_text": filler,
    }


def make_vt(ctx: int) -> dict:
    """Variable tracking: a chain of assignments equal to one value."""
    value = str(RNG.randint(10_000, 99_999))
    n_vars = 4
    var_names = [f"VAR-{''.join(RNG.choice(string.ascii_uppercase) for _ in range(3))}" for _ in range(n_vars)]
    # First var is assigned the literal; each subsequent var copies the previous.
    sentences = [f"{var_names[0]} = {value}."]
    for i in range(1, n_vars):
        sentences.append(f"{var_names[i]} = {var_names[i - 1]}.")
    depths = [(i + 1) / (n_vars + 1) for i in range(n_vars)]
    filler = insert_at_depths(ctx, list(zip(depths, sentences)))
    return {
        "context_tokens": ctx,
        "needles": [
            {"key": var_names[i], "value": var_names[i], "depth": depths[i]}
            for i in range(n_vars)
        ],
        "question": (
            f"In the text above, some variables are assigned values through a chain of "
            f"assignments. List every variable name whose value is {value}, one per line."
        ),
        "expected_answer_substrings": var_names,
        "min_recovered": n_vars,  # recall over the full chain
        "filler_text": filler,
    }


def main():
    out_dir = Path(__file__).parent
    # Fixed per-task salt (Python's hash() is per-process randomized).
    tasks = [("niah", 101, make_niah), ("vt", 202, make_vt)]
    for name, salt, fn in tasks:
        for ctx in CONTEXT_SIZES:
            RNG.seed(SEED + ctx + salt)
            record = fn(ctx)
            out_path = out_dir / f"ruler_{name}_{ctx // 1024}k.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
            print(
                f"wrote {out_path.name}: {len(record['filler_text'])} chars, "
                f"{len(record['expected_answer_substrings'])} target(s)"
            )


if __name__ == "__main__":
    sys.exit(main() or 0)
