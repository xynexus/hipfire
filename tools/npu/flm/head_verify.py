#!/usr/bin/env python3
"""The model head: final RMSNorm, lm_head, argmax — the phase with no evidence.

Everything else in this tree is verified. The sixteen decoder layers chain end to
end with P5 exact at thirteen of them (`chain_layers.sh 16`). But **nothing has
ever produced a token.** No design loads `model.norm` or `lm_head`; lm_head has
been TIMED at 163.7 MB / 54.7 GB/s and never RUN as part of the model. That is
2994 us of a 15.3 ms token — 20% — carried in every projection on no correctness
evidence at all.

This is the host half of closing that gap:

    x  ->  RMSNorm(model.norm)  ->  lm_head  ->  logits  ->  argmax

lm_head is [128256, 2048]. Dequantised whole that is a gigabyte of float32, so the
rows are walked in chunks and only the logits are kept.

    python3 head_verify.py --x x15.npy            # token from a hidden state
    python3 head_verify.py --token 128000         # embed a token, head it directly

The second form is a plumbing check, not a semantic one: it runs the embedding of
a token straight into the head with no layers, which exercises every step and
tells you nothing about whether the decoder is right.

Needs the q4nx container; no NPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
# NOT ffn_verify.load_linear — that returns the container's own row order, which
# measures corr 0.001 against the checkpoint. lm_head is 128256 rows of it.
def load_linear(c, name, N, K):
    return q4nx.q4nx_tensor_blocks(c, name, (N, K))
from qkv_verify import EPS, K_DIM  # noqa: E402

Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
VOCAB = 128256
CHUNK = 8192                    # lm_head rows per pass; 8192x2048 f32 = 64 MB


def rmsnorm(x, w, eps=EPS):
    """The same form the layer norms use, in float32."""
    return (x / np.sqrt((x.astype(np.float64) ** 2).mean() + eps)).astype(np.float32) * w


def logits_for(c, x):
    """lm_head @ x, in row chunks.

    `load_linear` returns the WHOLE tensor, so it is loaded once here and only the
    GEMV is chunked. The first version called it inside the loop, which loaded all
    128256 rows sixteen times over — the chunking bought nothing it was written to
    buy, since the peak allocation was the full tensor either way.
    """
    d, m, codes = load_linear(c, "lm_head.weight", VOCAB, K_DIM)
    out = np.empty(VOCAB, np.float32)
    for lo in range(0, VOCAB, CHUNK):
        hi = min(lo + CHUNK, VOCAB)
        out[lo:hi] = q4nx.gemv_reference_bf16(x, d[lo:hi], m[lo:hi], codes[lo:hi])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--x", help=".npy hidden state, K_DIM floats (e.g. layer 15's x_out)")
    g.add_argument("--token", type=int, help="embed this id and head it, no layers")
    p.add_argument("--top", type=int, default=5)
    o = p.parse_args()

    c = q4nx.Q4nx(str(Q4NX))
    if o.x:
        x = np.load(o.x).astype(np.float32)
        assert x.shape == (K_DIM,), x.shape
        src = o.x
    else:
        emb = c.bf16("model.embed_tokens.weight").astype(np.float32)
        x = emb.reshape(-1, K_DIM)[o.token].copy()
        src = f"embed_tokens[{o.token}]"

    nw = c.bf16("model.norm.weight").astype(np.float32)[:K_DIM]
    xn = rmsnorm(x, nw)
    lg = logits_for(c, xn)

    top = np.argsort(lg)[::-1][:o.top]
    print(f"head of {src}")
    print(f"  x      : mean|.| {np.abs(x).mean():.5f}  max|.| {np.abs(x).max():.5f}")
    print(f"  normed : mean|.| {np.abs(xn).mean():.5f}  max|.| {np.abs(xn).max():.5f}")
    print(f"  argmax : {top[0]}")
    for t in top:
        print(f"    {t:6d}  {lg[t]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
