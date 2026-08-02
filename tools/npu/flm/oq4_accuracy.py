#!/usr/bin/env python3
"""Does oq4 survive SIXTEEN CHAINED LAYERS? The accuracy gate the GEMV probe skipped.

`flm_gemv_oq4g256.cc` measured oq4 at the DMA ceiling (55.5 GB/s, ~78 tok/s
projected) and said nothing about accuracy — deliberately, and it says so. But
the shipped kernel folds the group scale into the WEIGHTS, which rounds
`scale * code` to bf16 BEFORE the MAC, and that was measured at 3.666e-03
relative against the accumulator-scaling variant's 1.651e-03 on ONE GEMV. Over
sixteen layers the errors compound, and nobody has looked.

Building sixteen layers of kernels to find out would be the expensive way round.
This is the cheap way: swap the weights on the HOST and run the validated
forward, which needs no NPU at all.

    q4nx        the container FLM ships, 5.00 b/w   -- the baseline this tree matches
    oq4         symmetric int4, per-256 group scale -- accumulator scaling (variant A)
    oq4_fold    the same, with scale*code rounded to bf16 -- what the kernel SHIPS

Everything goes through `host_forward.layer_parts`, which is externally validated
against an fp32 forward from `consolidated.00.pth`, so a token change here is a
change against a reference that shares no code with the quantiser.

    python3 oq4_accuracy.py --tokens 128000,791,4062,14198,39935

NOT the oq4 CODEC: no FWHT, no clip-search, no LDLQ, same as the GEMV probe. Those
IMPROVE accuracy (clip-search beats absmax, and the rotation is what makes low-bit
work), so this is a FLOOR on oq4's quality, not an estimate of it. If the floor
holds the tokens, the real codec certainly does.

Needs the q4nx container; no NPU.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

GROUP = 256
_CACHE = {}


def requant_oq4(W, fold):
    """[N, K] float32 -> the same matrix as oq4 would reconstruct it.

    Symmetric int4, one bf16 scale per 256-element group along K, q in [-7, 7].
    `fold` rounds `scale * code` to bf16, which is what the shipped kernel does
    when it folds the scale into the weights; without it the scale multiplies an
    f32 accumulator instead and only the accumulation rounds.
    """
    N, K = W.shape
    ng = K // GROUP
    Wg = W.reshape(N, ng, GROUP)
    s = (np.abs(Wg).max(2) / np.float32(7.0)).astype(np.float32)
    s = q4nx.bf16_to_f32(q4nx.f32_to_bf16(s))          # the tile carries bf16
    inv = np.where(s > 0, np.float32(1.0) / np.maximum(s, np.float32(1e-30)), 0.0)
    q = np.clip(np.rint(Wg * inv[:, :, None]), -7, 7).astype(np.float32)
    out = q * s[:, :, None]
    if fold:
        out = q4nx.bf16_to_f32(q4nx.f32_to_bf16(out))
    return out.reshape(N, K)


def patched_gemv(mode):  # noqa: C901
    """A drop-in for `q4nx.gemv_reference_bf16` that quantises to oq4 first.

    Keyed on the block arrays' identity: `host_forward` memoises its loaders, so
    each weight matrix is requantised once and reused for every token.
    """
    real = q4nx.gemv_reference_bf16

    def gemv(act, d, m, codes):
        if mode == "q4nx":
            return real(act, d, m, codes)
        # Keyed on id() AND holding a reference to `d` below, because id() is
        # reused after a temporary is freed: keying on the id of a chunk slice
        # returned a stale matrix of the wrong shape for a later chunk. Layer
        # weights are memoised by host_forward so their ids are stable; the
        # reference makes that guarantee instead of assuming it.
        key = (id(d), d.shape, mode)
        if key not in _CACHE:
            W = (d[:, :, None] * codes + m[:, :, None]).reshape(d.shape[0], -1)
            _CACHE[key] = (d, requant_oq4(W.astype(np.float32),
                                          fold=(mode == "oq4_fold")))
        return _CACHE[key][1] @ np.asarray(act, np.float32).astype(np.float64)

    return gemv


def run(mode, toks, nlay):
    import host_forward as hf
    from head_verify import logits_for, rmsnorm
    from qkv_verify import K_DIM

    real = q4nx.gemv_reference_bf16
    q4nx.gemv_reference_bf16 = patched_gemv(mode)
    hf._BLOCKS.clear()
    _CACHE.clear()
    c = q4nx.Q4nx(str(hf.Q4NX))
    emb = c.bf16("model.embed_tokens.weight").astype(np.float32).reshape(-1, K_DIM)
    pos = len(toks) - 1
    Kc, Vc = hf.prefill(c, toks[:pos], nlay=nlay)
    x = hf.rnd(emb[toks[pos]]).astype(np.float64)
    for L in range(nlay):
        x = hf.layer_parts(c, x, L, pos, (Kc[L], Vc[L]))[3]
    # THE HEAD STAYS q4nx in every mode. The question is what sixteen chained
    # LAYERS do; quantising lm_head as well would confound the comparison with a
    # separate effect, and the device runs the two-pass coarse head regardless.
    q4nx.gemv_reference_bf16 = real
    nw = c.bf16("model.norm.weight").astype(np.float32)[:K_DIM]
    lg = logits_for(c, rmsnorm(x.astype(np.float32), nw))
    return lg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", default="128000,791,4062,14198,39935")
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--top", type=int, default=5)
    o = p.parse_args()
    toks = [int(t) for t in o.tokens.split(",")]

    out = {}
    for mode in ("q4nx", "oq4", "oq4_fold"):
        t0 = time.time()
        lg = run(mode, toks, o.layers)
        out[mode] = lg
        top = np.argsort(lg)[::-1][:o.top]
        print(f"{mode:9s} argmax {int(top[0]):6d}  "
              f"top{o.top} {[int(t) for t in top]}  {time.time()-t0:.1f} s")

    base = out["q4nx"]
    print()
    for mode in ("oq4", "oq4_fold"):
        lg = out[mode]
        cos = float(np.dot(base, lg) / (np.linalg.norm(base) * np.linalg.norm(lg)))
        agree = int(np.argmax(lg)) == int(np.argmax(base))
        # The MARGIN matters as much as the match: a token that flips on a 0.05
        # logit gap is quantisation noise, one that flips on a 3-logit gap is a
        # broken forward. This tree has been fooled by that distinction before.
        srt = np.sort(base)[::-1]
        print(f"{mode:9s} vs q4nx: cos {cos:.6f}  argmax {'MATCH' if agree else 'DIFFER'}"
              f"   q4nx top-2 margin {srt[0]-srt[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
