#!/usr/bin/env python3
"""Build the grouped-document corpus for the MoE routing-locality probe.

Each group is 16 documents meant to be batched together as "similar subject"
streams. `wiki` (consecutive, unrelated wikitext articles) is the mixed-topic
control; `samedoc` is 16 consecutive chunks of ONE long document, the closest
stand-in for many jobs working inside one shared context.

    routing_locality_corpus.py <repo_root> <out.jsonl>

Writes `{"group", "id", "text"}` lines for `routing_trace_qwen4exp`.
"""
import json
import random
import re
import sys
from pathlib import Path

N = 16
CHARS = 6000  # comfortably > 512 tokens for English/code
CJK_CHARS = 2500

root = Path(sys.argv[1])
rng = random.Random(1234)
docs = []


def add(group, ident, text):
    docs.append({"group": group, "id": ident, "text": text})


# rust: the largest source file of 16 random crates, past the license header.
crates = sorted(p for p in (root / "crates").iterdir() if (p / "src").is_dir())
for c in rng.sample(crates, len(crates)):
    files = sorted((c / "src").rglob("*.rs"), key=lambda p: p.stat().st_size, reverse=True)
    if files and files[0].stat().st_size > CHARS + 400:
        add("rust", str(files[0].relative_to(root)), files[0].read_text()[400 : 400 + CHARS])
    if sum(d["group"] == "rust" for d in docs) == N:
        break

# docs: 16 random hipfire plan documents (same project, different topics).
plans = [p for p in sorted((root / "docs/plans").glob("*.md")) if p.stat().st_size > CHARS]
for p in rng.sample(plans, N):
    add("docs", str(p.relative_to(root)), p.read_text()[:CHARS])

# samedoc: 16 consecutive chunks of one long document.
long_doc = (root / "docs/architectural-review-2026-07-03-catalog.md").read_text()
for i in range(N):
    add("samedoc", f"catalog#{i}", long_doc[i * CHARS : (i + 1) * CHARS])

# wiki: consecutive wikitext-103 articles (unrelated topics) — the control.
wt = (root / "benchmarks/calib/calib-5m.txt").read_text()
arts = [a for a in re.split(r"\n = [^=\n]+ = \n", wt) if len(a) > CHARS]
for i, a in enumerate(arts[5 : 5 + N]):
    add("wiki", f"wikitext#{i}", a[:CHARS])

# math / pycode / chinese: domain-level groups from the labelled calib mix.
by = {}
for line in open(root / "benchmarks/calib/calib-multi-labelled.jsonl"):
    d = json.loads(line)
    by.setdefault(d["source"], []).append(d["text"])
for src, group in (("math", "math"), ("code", "pycode")):
    for i, t in enumerate([t for t in by[src] if len(t) > CHARS][:N]):
        add(group, f"{src}#{i}", t[:CHARS])
zh = "\n".join(by["chinese"])
for i in range(N):
    add("chinese", f"chinese#{i}", zh[i * CJK_CHARS : (i + 1) * CJK_CHARS])

with open(sys.argv[2], "w") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
for g in dict.fromkeys(d["group"] for d in docs):
    print(g, sum(d["group"] == g for d in docs))
