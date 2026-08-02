#!/usr/bin/env python3
"""Does the FWHT rotation pay for itself on ACTIVATIONS? The W4A4 hypothesis, tested.

`quant_eval.py` measured something that looks wrong: FWHT + clip-search
reconstructs real weights ~21% BETTER than plain absmax oq4 and makes the model's
output 43-47% WORSE. Ablated with a control, both the rotation and the
clip-search cost ~44% each and do not compound.

The leading explanation is that the harness cannot see half the format. Oq4G256
is W4A**4**: rotation schemes (QuIP#, QuaRot, and hipfire's own FWHT path) earn
most of their advantage by flattening ACTIVATION outliers, and `quant_eval.py`
runs activations in fp32. The rotation's cost on weights is fully present there;
its benefit is entirely absent.

That is testable without rewriting the forward. `oracle_forward.forward` stashes
per-layer activations via `keep`, so this measures, on REAL activations:

    * how outlier-heavy they are (kurtosis, max/rms) against the weights
    * int4 quantisation error with and without the rotation

If the rotation slashes ACTIVATION error while barely touching weight error, the
W4A4 hypothesis holds and `quant_eval.py` structurally understates every rotated
format -- oq4+, oq3++ and qtip3 alike. If it does not, the rotation is simply not
paying here and that is a finding about the format on this model.

    python3 act_rotation.py --layers 0,8,15

Needs the original checkpoint and torch; no NPU.
"""

import argparse
import json

import numpy as np

import oracle_forward as of
from quant_eval import bf16, clipsearch, fwht

GROUP = 256


def q_err(A, group, rotate, clip, rng):
    """relRMS of int4 quantisation of A [.., K], per `group` along the last axis."""
    K = A.shape[-1]
    G = A.reshape(-1, K // group, group)
    if rotate:
        s1 = rng.choice(np.float32([-1, 1]), group)
        s2 = rng.choice(np.float32([-1, 1]), group)
        R = fwht(G, s1, s2)
    else:
        R = G
    sc = bf16(clipsearch(R) if clip else np.abs(R).max(-1) / np.float32(7.0))
    q = np.clip(np.rint(R / np.maximum(sc, 1e-30)[..., None]), -7, 7).astype(np.float32)
    deq = bf16(q * sc[..., None])
    rec = fwht(deq, s2, s1) if rotate else deq
    return float(np.sqrt(((rec - G) ** 2).mean()) / np.sqrt((G ** 2).mean()))


def stats(A):
    a = np.asarray(A, np.float64).ravel()
    return (float(((a - a.mean()) ** 4).mean() / ((a - a.mean()) ** 2).mean() ** 2),
            float(np.abs(a).max() / a.std()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", default="0,8,15")
    p.add_argument("--tokens", default="128000,791,4062,14198,39935")
    o = p.parse_args()
    import torch

    cfg = json.loads(of.CFG.read_text())
    sd = of.load(cfg)
    keep = {int(x) for x in o.layers.split(",")}
    toks = [int(t) for t in o.tokens.split(",")]
    _, stash = of.forward(toks, cfg=cfg, sd=sd, keep=keep)

    eps = cfg.get("rms_norm_eps", 1e-5)
    print("  ACTIVATIONS (the GEMV input: rmsnorm(x) at each layer)")
    print(f"  {'layer':>6} {'kurtosis':>9} {'max/rms':>8} | "
          f"{'plain':>8} {'+rot':>8} {'+clip':>8} {'+both':>8}")
    for L in sorted(keep):
        x = np.asarray(stash[L], np.float32)
        w = sd[f"layers.{L}.attention_norm.weight"].to(torch.float32).numpy()
        h = (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
             ).astype(np.float32) * w
        k, mr = stats(h)
        r = [q_err(h, GROUP, rot, cl, np.random.default_rng(0))
             for rot, cl in ((0, 0), (1, 0), (0, 1), (1, 1))]
        print(f"  {L:>6} {k:>9.2f} {mr:>8.1f} | {r[0]:>8.5f} {r[1]:>8.5f} "
              f"{r[2]:>8.5f} {r[3]:>8.5f}")

    print("\n  WEIGHTS, same treatment, for comparison")
    print(f"  {'layer':>6} {'kurtosis':>9} {'max/rms':>8} | "
          f"{'plain':>8} {'+rot':>8} {'+clip':>8} {'+both':>8}")
    for L in sorted(keep):
        W = np.array(sd[f"layers.{L}.attention.wq.weight"].to(torch.float32).numpy(),
                     np.float32)
        k, mr = stats(W)
        r = [q_err(W, GROUP, rot, cl, np.random.default_rng(0))
             for rot, cl in ((0, 0), (1, 0), (0, 1), (1, 1))]
        print(f"  {L:>6} {k:>9.2f} {mr:>8.1f} | {r[0]:>8.5f} {r[1]:>8.5f} "
              f"{r[2]:>8.5f} {r[3]:>8.5f}")

    print("\n  If +rot slashes the ACTIVATION column while barely moving the WEIGHT\n"
          "  one, the rotation is a W4A4 feature and quant_eval.py -- which runs\n"
          "  fp32 activations -- understates every rotated format by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
