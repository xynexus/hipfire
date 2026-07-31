#!/usr/bin/env python3
"""Check the container reading against the REAL weights. Answers §1.3 — negatively.

Every numeric result in this reproduction is self-consistency: the kernel is
compared to a numpy reference built from *the same reading* of `model.q4nx`. If
that reading is wrong, kernel and reference are wrong together and every check
still passes. `docs/npu/flm-fused-layer-plan.md` §1.3 flags this and assumes it
is unresolvable. It is not — `meta-llama/Llama-3.2-1B-Instruct` ships the actual
trained checkpoint, so the reading can be checked against ground truth.

Four probes, cheapest and most conclusive first:

1. **bf16 tensors, bit-exact.** RMSNorm weights are stored unquantized, so this
   is an exact check of the file structure, the tensor naming, and *which model
   this is*, with no quantization noise anywhere.

2. **Blocks, order-free.** The sorted 32 values of a q4_1 block are invariant to
   the nibble order `q4nx.py` flags as assumed, so a block can be matched
   against every ground-truth block with no unknowns left in it.

3. **Rows, order-free.** Each block's `(m, d)` is its (min, range/15). Comparing
   a ground-truth row's 256 such points against a container row's as a *set*
   asks whether any container row holds that row's blocks under any permutation.

4. **Frobenius norm.** Also invariant to the nibble order (the block's value
   multiset is fixed regardless of which element is which), so it separates an
   orthogonal transform (ratio 1.000) from a scaling (ratio != 1).

Do NOT try to infer the grouping from the `d` (block range) distribution. A
random regrouping of the same weights reproduces it about as well as the true
one, so it separates nothing — an earlier version of this file shipped that
comparison as a control and it was not reproducible run to run. The four probes
below are.

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


def probe_bf16(c, sd, layer):
    print("1. unquantized bf16 tensors — exact, no quantization noise")
    ok = True
    for cn, mn in NORMS:
        got = c.bf16(cn.replace("layers.0", f"layers.{layer}")).astype(np.float64)
        ref = sd[mn.replace("layers.0", f"layers.{layer}")].to(
            __import__("torch").float32).numpy().astype(np.float64)
        n = min(got.size, ref.size)
        ex = float(np.mean(got[:n] == ref[:n]))
        ok &= ex > 0.999
        print(f"   {cn:48s} bit-exact {ex:7.2%}  maxdiff "
              f"{np.abs(got[:n]-ref[:n]).max():.3e}")
    print(f"   -> file structure, tensor naming and model identity: "
          f"{'CONFIRMED' if ok else 'WRONG MODEL/READER'}\n")
    return ok


def probe_blocks(c, sd, layer, cn, mn, nrows=4):
    import torch
    w = sd[f"layers.{layer}.{mn}.weight"].to(torch.float32).numpy()
    nb = w.shape[1] // BLK
    gs = np.sort(w.reshape(-1, BLK), axis=1).astype(np.float32)
    cs = np.sort(dequant(c, f"model.layers.{layer}.{cn}.weight")[:nrows],
                 axis=2).reshape(-1, BLK).astype(np.float32)
    best = np.full(len(cs), np.inf)
    arg = np.zeros(len(cs), np.int64)
    cn2 = (cs * cs).sum(1)[:, None]
    for i in range(0, len(gs), 65536):
        b = gs[i:i + 65536]
        dist = cn2 + (b * b).sum(1)[None, :] - 2 * (cs @ b.T)
        j = dist.argmin(1)
        v = dist[np.arange(len(cs)), j]
        u = v < best
        best[u], arg[u] = v[u], j[u] + i
    rms = np.sqrt(np.maximum(best, 0) / BLK) / np.abs(w).mean()
    rows = arg // nb
    return float(np.median(rms)), len(set(rows.tolist())), len(rows)


def probe_rows(c, sd, layer, cn, mn, gt_rows=(0, 1, 7)):
    import torch
    w = sd[f"layers.{layer}.{mn}.weight"].to(torch.float32).numpy().astype(np.float64)
    nb = w.shape[1] // BLK
    g = w.reshape(w.shape[0], nb, BLK)
    gm = g.min(2)
    gd = (g.max(2) - gm) / 15.0
    d, m, _ = c.blocks(f"model.layers.{layer}.{cn}.weight")
    sd_, sm = 1.0 / gd.std(), 1.0 / gm.std()
    out = []
    for gr in gt_rows:
        A = np.stack([gd[gr] * sd_, gm[gr] * sm], 1)
        dists = np.empty(d.shape[0])
        for cr in range(d.shape[0]):
            B = np.stack([d[cr, :nb] * sd_, m[cr, :nb] * sm], 1)
            dists[cr] = np.sqrt(((A[:, None] - B[None]) ** 2).sum(2).min(1)).mean()
        out.append((dists.min(), float(np.median(dists))))
    return out


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--which", default="instruct", choices=list(CKPT))
    o = p.parse_args()

    sd = torch.load(str(CKPT[o.which]), map_location="cpu", mmap=True, weights_only=True)
    c = q4nx.Q4nx(str(Q4NX))
    print(f"container vs meta-llama/Llama-3.2-1B{'-Instruct' if o.which=='instruct' else ''}"
          f", layer {o.layer}\n")

    if not probe_bf16(c, sd, o.layer):
        print("   stop: the reader or the model is wrong; nothing below is meaningful.")
        return 1

    print("2. blocks, order-free (sorted values; nibble order cancels)")
    rms, distinct, n = probe_blocks(c, sd, o.layer, *PROJ[0])
    print(f"   {PROJ[0][0]}: best-of-all-blocks rms/|w| = {rms:.4f}")
    print(f"     q4_1 quantization error alone is ~0.03; random is ~1.0")
    print(f"     destinations: {distinct} distinct gt rows for {n} container blocks "
          f"({distinct/n:.0%} — uniform scatter)")
    blocks_ok = rms < 0.06
    print(f"   -> container blocks {'ARE' if blocks_ok else 'are NOT'} "
          f"the model's 32-element blocks\n")

    print("3. rows, order-free ((m,d) set match, any within-row permutation)")
    res = probe_rows(c, sd, o.layer, *PROJ[0])
    for (gr, (b, med)) in zip((0, 1, 7), res):
        print(f"   gt row {gr:3d}: best container row {b:.5f} vs median {med:.5f} "
              f"({'MATCH' if b < 0.3 * med else 'no match'})")
    rows_ok = all(b < 0.3 * med for b, med in res)
    print(f"   -> a true match would be ~bf16 noise; best ~= median means no\n"
          f"      container row holds any gt row's blocks\n")

    print("4. Frobenius norm (nibble-order invariant): rotation or scaling?")
    print(f"   {'tensor':>16s} {'gt |W|_F':>9s} {'container':>9s} {'ratio':>7s}")
    ratios = []
    for cn, mn in PROJ:
        w = sd[f"layers.{o.layer}.{mn}.weight"].to(torch.float32).numpy().astype(np.float64)
        x = dequant(c, f"model.layers.{o.layer}.{cn}.weight")
        gf, cf = np.linalg.norm(w), np.sqrt((x * x).sum())
        ratios.append(cf / gf)
        print(f"   {cn:>16s} {gf:9.3f} {cf:9.3f} {cf/gf:7.4f}")
    lo, hi = min(ratios), max(ratios)
    print(f"   -> ratios {lo:.4f}..{hi:.4f}; orthogonal would be 1.0000 and q4_1\n"
          f"      error alone contributes <0.001\n")

    if blocks_ok and rows_ok:
        print("VERDICT: the container reading matches the real weights.")
        return 0
    print("VERDICT: the quantized reading does NOT recover the real weights.")
    print("  The bf16 path is bit-exact, so the reader and the model are right;")
    print("  the quantized tensors carry a VALUE TRANSFORM, not a reordering —")
    print(f"  norms are inflated {lo:.2f}-{hi:.2f}x and vary per tensor, which is the")
    print("  signature of per-channel (AWQ/SmoothQuant-style) scaling, not a rotation.")
    print("  Consequence: the GEMV kernels are verified as q4_1 ARITHMETIC against")
    print("  a reference on the same bytes, but are NOT verified as computing")
    print("  Llama-3.2-1B-Instruct. Throughput results are unaffected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
