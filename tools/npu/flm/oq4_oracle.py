#!/usr/bin/env python3
"""oq4 quantised from the ORIGINAL fp weights, against the fp32 oracle.

`oq4_accuracy.py` asked whether the shipped kernel's bf16 weight-fold survives
sixteen layers. It does — fold and no-fold were indistinguishable. But it
quantised the ALREADY-QUANTISED q4nx weights, so its absolute quality number
measured q4_1 -> oq4 DOUBLE quantisation (~14% added relative RMS) and says
nothing about oq4 itself.

This is the fair version: quantise `consolidated.00.pth` directly and run the
same fp32 forward that validated this whole tree, so the only variable is the
weight format.

    fp32        the oracle, unmodified
    oq4         symmetric int4, one bf16 scale per 256 along the input dim
    oq4_g32     the same at group 32, to separate "4 bits" from "coarse groups"

    python3 oq4_oracle.py --tokens 128000,791,4062,14198,39935

STILL NOT the oq4 CODEC: no FWHT, no clip-search, no LDLQ. All three IMPROVE on
this, and the FWHT is what makes a symmetric per-group codebook fit a weight
distribution at all, so this IS a genuine floor — unlike the double-quantised
version, this one sits where oq4 could actually sit and only gets better.

SEPARATE FILE deliberately: importing torch alongside the flm helpers segfaults,
because they pull `aie.iron` at module scope. Nothing here imports them.

Needs the original checkpoint and torch; no NPU.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import oracle_forward as of

GROUP = 256
# Every linear the forward multiplies by. Norms and embeddings stay fp32: they
# are a rounding error of the parameter count and quantising them would confound
# the measurement with a separate effect.
LINEARS = ("attention.wq", "attention.wk", "attention.wv", "attention.wo",
           "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")


def bf16(a):
    """f32 -> bf16 -> f32, round-to-nearest-even, without ml_dtypes."""
    u = np.asarray(a, np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def requant_oq4(W, group=GROUP):
    """[N, K] -> the matrix oq4 reconstructs, symmetric int4, bf16 group scale.

    The scale is folded into the weights and rounded to bf16, which is what the
    shipped kernel does; `oq4_accuracy.py` measured that choice as free over
    sixteen layers, so there is no reason to model the slower variant.
    """
    N, K = W.shape
    ng = K // group
    Wg = W.reshape(N, ng, group)
    s = bf16(np.abs(Wg).max(2) / np.float32(7.0))
    inv = np.where(s > 0, np.float32(1.0) / np.maximum(s, np.float32(1e-30)), 0.0)
    q = np.clip(np.rint(Wg * inv[:, :, None]), -7, 7).astype(np.float32)
    return bf16(q * s[:, :, None]).reshape(N, K)


def quantise(sd, cfg, group):
    import torch
    out = dict(sd)
    n = 0
    for L in range(cfg["num_hidden_layers"]):
        for nm in LINEARS:
            k = f"layers.{L}.{nm}.weight"
            W = sd[k].to(torch.float32).numpy()
            if W.shape[1] % group:
                raise ValueError(f"{k}: K={W.shape[1]} not a multiple of {group}")
            out[k] = torch.from_numpy(requant_oq4(W, group))
            n += 1
    return out, n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", default="128000,791,4062,14198,39935")
    p.add_argument("--groups", default="32,64,128,256",
                   help="group sizes to sweep; the coarsest that holds the token wins")
    p.add_argument("--top", type=int, default=5)
    o = p.parse_args()
    toks = [int(t) for t in o.tokens.split(",")]

    cfg = json.loads(of.CFG.read_text())
    sd = of.load(cfg)
    print(f"oq4 vs fp32 oracle, {len(toks)} tokens {toks}")

    groups = [int(g) for g in o.groups.split(",")]
    res = {}
    for name, group in [("fp32", None)] + [(f"g{g}", g) for g in groups]:
        w = sd
        if group:
            w, n = quantise(sd, cfg, group)
            print(f"  quantised {n} matrices at group {group}")
        x, _ = of.forward(toks, cfg=cfg, sd=w)
        lg = of.logits(x[-1], cfg=cfg, sd=w)
        res[name] = np.asarray(lg, np.float64)
        top = np.argsort(res[name])[::-1][:o.top]
        print(f"  {name:8s} argmax {int(top[0]):6d}  top{o.top} {[int(t) for t in top]}")

    base = res["fp32"]
    srt = np.sort(base)[::-1]
    print(f"\n  fp32 top-2 margin {srt[0]-srt[1]:.4f}")
    print(f"  {'group':>6} {'b/w':>7} {'cos':>9}  argmax   {'MB/token':>9}  {'tok/s':>6}")
    for g in groups:
        lg = res[f"g{g}"]
        cos = float(np.dot(base, lg) / (np.linalg.norm(base) * np.linalg.norm(lg)))
        ok = int(np.argmax(lg)) == int(np.argmax(base))
        bw = 4.0 + 16.0 / g                      # 4 bits a weight + one bf16 scale
        mb = 775.7 * bw / 5.0                    # q4nx is 5.00 b/w at 775.7 MB/token
        tps = 1e6 / (mb / 49.3e-3 + 10.5)        # 49.3 GB/s measured, + host term
        print(f"  {g:>6} {bw:>7.4f} {cos:>9.6f}  {'MATCH' if ok else 'DIFFER'}   "
              f"{mb:>9.1f}  {tps:>6.1f}")
    print("\n  b/w and tok/s assume no rotation: one bf16 scale per group, "
          "4 bits a weight.\n  The coarsest MATCH is the format to build if the "
          "FWHT codec is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
