"""build_phase2_eval_corpus_agentic.py — assembles the agentic eval corpus
for Phase 2 SE-pinning rerun.

Composition:
  - 7 matrix-style prompts from `calibration_corpus.jsonl` (the issue #171
    reproducer family) replicated 8 each = 56 records. Replication is fine
    because the runner is bit-identical per (variant, prompt) — duplication
    weights the agentic shape across the per-token CE sum.
  - 30 hand-authored short agent-shape prompts (code requests, explanations,
    code review tasks, debugging etc.) — realistic agentic prompt shapes,
    NOT wikitext prose.

Total = 86 records. Output: phase2_eval_corpus_agentic.jsonl

Why duplicate the matrix prompts: PPL is mean per-token CE over the corpus.
Replicating concentrates statistical mass on the prompts that empirically
trigger the cliff per issue #171's matrix. Bit-identical trials see the
same prompt 8× — the variant-comparison still works because a difference
in PPL between V1/V2/V3 must come from variant-induced quant changes, not
from prompt sampling noise.
"""
from __future__ import annotations

import json
from pathlib import Path

_THIS = Path(__file__).resolve().parent
SRC = _THIS / "calibration_corpus.jsonl"
DST = _THIS / "phase2_eval_corpus_agentic.jsonl"

# Per the spec: replicate matrix prompts 8x.
MATRIX_REPLICATE = 8

# Hand-authored agentic-shape prompts. Realistic short turn-shape requests
# of the kind agentic users actually send. Mix: code request, debugging,
# code review, "explain" style, structured-output, multi-step.
AGENT_PROMPTS: list[str] = [
    "Write a Python function that takes a list of integers and returns the second-largest value. Handle the case where the list has fewer than two distinct elements by raising ValueError.",
    "Explain the difference between a process and a thread in two short paragraphs. Then give one example of when you'd prefer threads over processes.",
    "Review this Rust code and identify any issues:\n\n```rust\nfn main() {\n    let v = vec![1, 2, 3];\n    let r = &v;\n    drop(v);\n    println!(\"{:?}\", r);\n}\n```\n\nList each issue with a one-line explanation.",
    "Write a SQL query that returns the top 5 customers by total order amount over the last 90 days. Schema: orders(id, customer_id, amount, created_at). Use a CTE.",
    "I have a Python script that reads a CSV with pandas, but it's running out of memory on a 5GB file. Suggest three concrete strategies to fix this, ordered by least invasive first.",
    "Write a bash one-liner that finds all `.log` files modified in the last 24 hours under `/var/log/` and prints their total size in MB.",
    "Convert this JSON schema description into a TypeScript interface:\n\n- user object\n- fields: id (string, uuid), email (string), createdAt (ISO 8601 timestamp), roles (array of 'admin' | 'user' | 'guest'), profile (optional object with name and avatarUrl)",
    "Explain what `O(n log n)` means as if to someone who has never taken a CS course. Use a real-world analogy and keep it under 100 words.",
    "Write a Python decorator `@retry(n=3, backoff=2.0)` that retries the wrapped function up to n times on exception, doubling the sleep between attempts. Include a usage example.",
    "I need to add caching to a slow GraphQL resolver. The cache should be keyed by the query variables and have a 5-minute TTL. Sketch the design in pseudocode and call out the two biggest correctness pitfalls.",
    "Refactor this function to be more idiomatic Python:\n\n```python\ndef get_evens(lst):\n    result = []\n    for i in range(len(lst)):\n        if lst[i] % 2 == 0:\n            result.append(lst[i])\n    return result\n```",
    "Write the body of a function `parse_iso_duration(s: str) -> int` that converts an ISO 8601 duration string like `PT1H30M` to total seconds. Support hours, minutes, seconds. Raise ValueError on bad input.",
    "What's the difference between `git rebase` and `git merge`? When should I prefer one over the other? Answer in three short bullet points.",
    "Write a Rust function `fn moving_average(window: usize, data: &[f64]) -> Vec<f64>` that returns the trailing moving average of `data`. The output length should match `data.len()` (use partial windows at the start).",
    "Write a regex that matches a US phone number in any of these formats: `(555) 555-5555`, `555-555-5555`, `5555555555`, `+1 555 555 5555`. Provide the regex and three test cases that should match plus two that should not.",
    "I'm seeing this error when running `cargo build`: `error[E0382]: borrow of moved value: 's'`. Without seeing my code, what are the three most likely causes and how do I fix each?",
    "Write a Python function that takes a tree as a nested dict (each node has 'value' and 'children': list[node]) and returns the deepest value. If the tree has multiple equally deep values, return any one.",
    "Explain how a hash table handles collisions. Cover open addressing and chaining, and say which one Python's `dict` uses.",
    "I want to add rate limiting to an HTTP API. The limit should be 100 requests per minute per API key. Sketch the data structure, the Redis commands you'd use, and how you'd handle a key that's never been seen before.",
    "Write a TypeScript function that debounces an async function. Signature: `debounce<T extends (...args: any[]) => Promise<any>>(fn: T, wait: number): T`. Calling the returned function should resolve with the result of the latest invocation only.",
    "Compare REST and GraphQL for a mobile-first chat application backend. Two paragraphs maximum, end with a one-line recommendation.",
    "Write a Python script that takes a directory path as argv[1] and prints, for each file inside (recursively), `<size_in_bytes> <relative_path>` sorted by size descending. Skip symlinks.",
    "What are three subtle bugs that can occur when using `time.time()` for measuring elapsed time? Suggest the function I should use instead in Python.",
    "Write a function `fn longest_common_prefix(strs: &[&str]) -> &str` in Rust. Return an empty `&str` if the slice is empty or there's no common prefix.",
    "I have a function that calls an external API and sometimes returns stale data. The cache layer is in front of it and TTL is 60s. The stale data sometimes persists for 5+ minutes. List four likely causes ordered by probability.",
    "Write a SQL migration that adds a `status` enum column to a `tickets` table with values 'open', 'in_progress', 'closed'. Default to 'open'. Backfill existing rows. Use Postgres syntax.",
    "Sketch the architecture for a feature flag system that supports user-level overrides, percentage rollouts, and dependency between flags. Two paragraphs max.",
    "Review this Python code for security issues:\n\n```python\nimport pickle\ndef load_user_state(blob):\n    return pickle.loads(blob)\n```\n\nExplain the issue and give a concrete safer alternative.",
    "Write a function that takes a sentence and returns the longest word, breaking ties by returning the first occurrence. Punctuation should be stripped from word boundaries. Pick any language but explain your choice in one line.",
    "Explain in three sentences why floating-point arithmetic gives `0.1 + 0.2 != 0.3`, then suggest the right way to compare floats for approximate equality.",
]

assert len(AGENT_PROMPTS) == 30, f"agent prompt count: {len(AGENT_PROMPTS)}"


def main() -> None:
    # Read calibration corpus and pick out the 7 matrix prompts.
    matrix_records: list[dict] = []
    with SRC.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id", "").startswith("matrix_"):
                matrix_records.append(r)

    assert len(matrix_records) == 7, f"expected 7 matrix prompts, got {len(matrix_records)}"

    out_records: list[dict] = []
    # Replicate matrix prompts MATRIX_REPLICATE times.
    for r in matrix_records:
        for k in range(MATRIX_REPLICATE):
            out_records.append({
                "id": f"{r['id']}__rep{k:02d}",
                "source": r["source"],
                "prompt": r["prompt"],
            })

    # Append agent-shape prompts.
    for i, p in enumerate(AGENT_PROMPTS):
        out_records.append({
            "id": f"agent_{i:03d}",
            "source": "hand-authored-agentic-shape",
            "prompt": p,
        })

    DST.write_text("".join(json.dumps(r) + "\n" for r in out_records))
    print(f"wrote {len(out_records)} records to {DST}")


if __name__ == "__main__":
    main()
