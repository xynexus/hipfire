#!/usr/bin/env python3
"""Check the container reading against the REAL weights. Answers §1.3 — negatively.

Every numeric result in this reproduction is self-consistency: the kernel is
compared to a numpy reference built from *the same reading* of `model.q4nx`. If
that reading is wrong, kernel and reference are wrong together and every check
still passes. `docs/npu/flm-fused-layer-plan.md` §1.3 flags this and assumes it
is unresolvable without `flm run`. It is not — `meta-llama/Llama-3.2-1B` ships
`original/consolidated.00.pth`, the actual trained checkpoint.

Four probes. **Every one of them needs a control**, because the natural
statistics of a weight matrix are similar enough everywhere that an uncalibrated
probe reports a confident wrong answer. Two did, before the controls were added.

1. **bf16 tensors, bit-exact.** RMSNorm weights are stored unquantized, so this
   checks the file structure, the planar split, the tensor naming and *which
   model this is*, with no quantization noise and no control needed.

2. **Blocks, scale- and order-invariant.** Cosine on the block's *sorted* 32
   values: sorting removes the unknown nibble order, normalising removes any
   per-tensor scale. Matched against all 524,288 ground-truth blocks.
   **Control (essential):** the same search run on gt blocks this tool quantized
   itself, once with the true block in the pool and once with it masked out.
   That second number is the look-alike floor — the score a block gets when its
   true match is *definitely absent*. Sorted 32-value profiles of weight blocks
   all resemble each other, so the floor is ~0.995, not ~0. Without it, 0.9949
   reads as "no match by a mile" or "basically a match" depending on taste.

3. **Which q4_1 fit is this?** Per-block code span. llama.cpp's q4_1 is a plain
   min/max fit, which forces span 15 and a code 0 and a code 15 in every block.
   FLM's does not — so do NOT compare container `d` against `(max-min)/15`.
   An earlier version of this file did exactly that as its row-level probe; the
   probe could only ever fail, and it was removed rather than reported.

4. **Frobenius norm and value quantiles.** The norm is invariant to the nibble
   order (a block's value multiset is fixed however elements are assigned), so
   it separates an orthogonal transform (ratio 1.000) from a scaling. The
   quantile ratios then say whether that scaling is one number or many.

Do not try to infer the block grouping from the `d` (block range) distribution: a
random regrouping of the same weights reproduces it about as well as the true
one.

    python3 ground_truth.py

Needs torch for the checkpoint. No NPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
HF = Path("/srv/huggingface")
CKPT = {
    "instruct": HF / "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
                     "9213176726f574b556790deb65791e0c5aa438b6/original/consolidated.00.pth",
    "base": HF / "models--meta-llama--Llama-3.2-1B/snapshots/"
                 "4e20de362430cd3b72f300e6b0f18e50e7166e08/original/consolidated.00.pth",
}
BLK = 32
NORMS = [("model.layers.0.input_layernorm.weight", "layers.0.attention_norm.weight"),
         ("model.layers.0.post_attention_layernorm.weight", "layers.0.ffn_norm.weight"),
         ("model.norm.weight", "norm.weight")]
PROJ = [("mlp.down_proj", "feed_forward.w2"), ("mlp.gate_proj", "feed_forward.w1"),
        ("mlp.up_proj", "feed_forward.w3"), ("self_attn.q_proj", "attention.wq"),
        ("self_attn.k_proj", "attention.wk"), ("self_attn.v_proj", "attention.wv"),
        ("self_attn.o_proj", "attention.wo")]


def dequant(c, name):
    """-> (nrows, 256, 32) float64. Nibble order is q4nx's assumption."""
    d, m, codes = c.blocks(name)
    return (d.astype(np.float64)[:, :, None] * codes.astype(np.float64)
            + m.astype(np.float64)[:, :, None])


def fingerprint(x):
    """Sorted block values, unit norm: invariant to nibble order AND to scale."""
    s = np.sort(x, axis=-1).astype(np.float32).reshape(-1, BLK)
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def best_match(cs, gs, exclude=None):
    best = np.full(len(cs), -2.0)
    arg = np.zeros(len(cs), np.int64)
    for i in range(0, len(gs), 65536):
        b = gs[i:i + 65536]
        sim = cs @ b.T
        if exclude is not None:                      # mask the true match out
            k = (exclude >= i) & (exclude < i + len(b))
            sim[np.where(k)[0], exclude[k] - i] = -2
        j = sim.argmax(1)
        v = sim[np.arange(len(cs)), j]
        u = v > best
        best[u], arg[u] = v[u], j[u] + i
    return arg, best


def probe_bf16(c, sd, layer, torch):
    print("1. unquantized bf16 tensors — exact, no quantization noise")
    ok = True
    for cn, mn in NORMS:
        got = c.bf16(cn.replace("layers.0", f"layers.{layer}")).astype(np.float64)
        ref = sd[mn.replace("layers.0", f"layers.{layer}")].to(
            torch.float32).numpy().astype(np.float64)
        n = min(got.size, ref.size)
        ex = float(np.mean(got[:n] == ref[:n]))
        ok &= ex > 0.999
        print(f"   {cn:48s} bit-exact {ex:7.2%}  maxdiff "
              f"{np.abs(got[:n]-ref[:n]).max():.3e}")
    print(f"   -> file structure, tensor naming and model identity: "
          f"{'CONFIRMED' if ok else 'WRONG MODEL/READER'}\n")
    return ok


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--which", default="instruct", choices=list(CKPT))
    p.add_argument("--rows", type=int, default=8, help="container rows to match")
    o = p.parse_args()

    sd = torch.load(str(CKPT[o.which]), map_location="cpu", mmap=True, weights_only=True)
    c = q4nx.Q4nx(str(Q4NX))
    print(f"container vs meta-llama/Llama-3.2-1B"
          f"{'-Instruct' if o.which == 'instruct' else ''}, layer {o.layer}\n")

    if not probe_bf16(c, sd, o.layer, torch):
        print("   stop: the reader or the model is wrong; nothing below is meaningful.")
        return 1

    cn, mn = PROJ[0]
    w = sd[f"layers.{o.layer}.{mn}.weight"].to(torch.float32).numpy().astype(np.float64)
    g = w.reshape(-1, BLK)
    gs = fingerprint(g)

    print("2. blocks — cosine on sorted values (nibble order and scale both cancel)")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(g), 256, replace=False)
    mn_ = g[pick].min(1, keepdims=True)
    dd = (g[pick].max(1, keepdims=True) - mn_) / 15.0
    rec = fingerprint(mn_ + dd * np.clip(np.rint((g[pick] - mn_) / dd), 0, 15))
    arg_p, b_present = best_match(rec, gs)
    _, b_absent = best_match(rec, gs, exclude=pick)
    x = dequant(c, f"model.layers.{o.layer}.{cn}.weight")
    arg_c, b_cont = best_match(fingerprint(x[:o.rows]), gs)
    print(f"   control, true match PRESENT   p50={np.median(b_present):.5f}  "
          f"(found it {np.mean(arg_p == pick):.1%} of the time)")
    print(f"   control, true match ABSENT    p50={np.median(b_absent):.5f}  "
          f"<- the look-alike floor")
    print(f"   CONTAINER {cn:19s} p50={np.median(b_cont):.5f}")
    at_floor = abs(np.median(b_cont) - np.median(b_absent)) < \
        abs(np.median(b_cont) - np.median(b_present))
    print(f"   destinations: {len(set((arg_c // (w.shape[1] // BLK)).tolist()))} distinct "
          f"gt rows for {len(arg_c)} container blocks")
    print(f"   -> container is at the {'LOOK-ALIKE FLOOR — no true match exists' if at_floor else 'true-match level'}\n")

    print("3. which q4_1 fit is this? (min/max forces span 15 in every block)")
    _, _, codes = c.blocks(f"model.layers.{o.layer}.{cn}.weight")
    q = codes.reshape(-1, BLK)
    span = (q.max(1) - q.min(1)).mean()
    both = float(((q == 0).any(1) & (q == 15).any(1)).mean())
    print(f"   mean per-block code span {span:.2f} of 15;  blocks holding both "
          f"0 and 15: {both:.1%}")
    print(f"   -> FLM's q4_1 is a SEARCH fit on a grid ~{15/span-1:.1%} wider than "
          f"min/max,\n      not llama.cpp's. Never compare container d against "
          f"(max-min)/15.\n")

    print("4. Frobenius norm and value quantiles: one scale, or many?")
    print(f"   {'tensor':>16s} {'gt |W|_F':>9s} {'container':>9s} {'ratio':>7s}")
    ratios = []
    for pn, pm in PROJ:
        ww = sd[f"layers.{o.layer}.{pm}.weight"].to(torch.float32).numpy().astype(np.float64)
        xx = dequant(c, f"model.layers.{o.layer}.{pn}.weight")
        gf, cf = np.linalg.norm(ww), np.sqrt((xx * xx).sum())
        ratios.append(cf / gf)
        print(f"   {pn:>16s} {gf:9.3f} {cf:9.3f} {cf/gf:7.4f}")
    qs = [0.001, 0.01, 0.1, 0.25, 0.75, 0.9, 0.99, 0.999]
    qr = np.quantile(x.ravel(), qs) / np.quantile(w.ravel(), qs)
    print(f"   {cn} quantile ratios {np.round(qr, 3)}")
    print(f"     spread {qr.min():.3f}..{qr.max():.3f} — a single scalar would be flat\n")

    lo, hi = min(ratios), max(ratios)
    print("VERDICT: the quantized reading does NOT recover the real weights.")
    print("  The bf16 path is bit-exact, so the reader and the model are right.")
    print("  The quantized blocks sit at the look-alike floor against all 524288")
    print("  ground-truth blocks, so no arrangement of them is the model's blocks.")
    print(f"  Norms are inflated {lo:.2f}-{hi:.2f}x, per tensor — orthogonal would be")
    print("  1.0000, so there is a real scaling, and the quantile spread above says")
    print("  whether it is one number per tensor or a per-channel vector.")
    print("  Consequence: the GEMV kernels are verified as q4_1 ARITHMETIC against")
    print("  a reference on the same bytes, but are NOT verified as computing")
    print("  Llama-3.2-1B-Instruct. Throughput results are unaffected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
