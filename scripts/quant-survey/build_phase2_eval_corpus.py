"""Build phase2_eval_corpus.jsonl for Phase 2 perplexity ablation.

Sampled from `benchmarks/calib/calib-5m.txt` (md5 5dc7dc29...), the
larger wikitext-103 train shard prefix. This is DISJOINT from
calibration_corpus.jsonl (which samples from calib-1m.txt) — different
source files, no shared bytes — so the eval set is guaranteed clean
relative to the D3 calibration distribution.

Composition:
  100 chunks × ~4000 chars each ≈ ~1024 tokens per chunk after
  tokenization (Qwen tokenizer ratio ~3.9 chars/token).

Output: phase2_eval_corpus.jsonl + summary including md5, source md5,
chunk count, total chars.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts" / "quant-survey" / "phase2_eval_corpus.jsonl"
SRC = ROOT / "benchmarks" / "calib" / "calib-5m.txt"
EXPECTED_SRC_MD5 = "5dc7dc29676eb591869378b3ddc17815"

SEED = 20260507
N_CHUNKS = 100
CHUNK_CHARS = 4000  # ~1024 tokens at Qwen ratio


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing wikitext source: {SRC}")
    src_md5 = md5_file(SRC)
    if src_md5 != EXPECTED_SRC_MD5:
        raise SystemExit(
            f"calib-5m.txt md5 mismatch: got {src_md5}, expected {EXPECTED_SRC_MD5}. "
            f"eval corpus would not be reproducible."
        )

    text = SRC.read_text(encoding="utf-8")
    rng = random.Random(SEED)
    chunks: list[tuple[int, str]] = []
    seen: set[int] = set()
    max_off = len(text) - CHUNK_CHARS
    while len(chunks) < N_CHUNKS:
        off = rng.randrange(0, max_off)
        if off in seen:
            continue
        seen.add(off)
        nl = text.find("\n", off)
        start = nl + 1 if (nl != -1 and nl - off < 200) else off
        chunk = text[start : start + CHUNK_CHARS].rstrip()
        if len(chunk) < CHUNK_CHARS // 2:
            continue
        chunks.append((start, chunk))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i, (off, chunk) in enumerate(chunks):
            rec = {"id": f"eval_{i:03d}", "source": "calib-5m", "offset": off, "prompt": chunk}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT}")
    print(f"  records: {len(chunks)}")
    print(f"  source calib-5m.txt md5: {src_md5}")
    print(f"  output md5: {out_md5}")
    print(f"  total chars: {sum(len(c) for _, c in chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
