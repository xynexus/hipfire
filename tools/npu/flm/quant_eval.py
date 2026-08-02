#!/usr/bin/env python3
"""KLD and PPL against fp32, with q4nx as the BAR — the only comparison that matters.

Throughput is not an improvement on its own. A format is worth porting only if it
beats what FLM already ships -- q4nx, 5.00 b/w -- on quality, at fewer bits. Every
accuracy check in this tree so far has been argmax agreement on ONE token, which
is far too weak to decide that: it cannot see a format that gets the top token
right and the distribution wrong, and it cannot rank two formats that both agree.

So: run the same fp32 forward with three weight sets and score the DISTRIBUTIONS.

    fp32     the reference, unmodified `consolidated.00.pth`
    q4nx     FLM's container, dequantised into the same forward -- THE BAR
    oq4_gN   symmetric int4, one bf16 scale per N, quantised from fp32

    KLD  mean over positions of KL(fp32 || quant), in nats. Lower is better.
         This is the metric that sees a distribution shift the argmax hides.
    PPL  exp of the mean NLL of the actual next token. Lower is better.
    top1 fraction of positions whose argmax matches fp32's.

q4nx is loaded through the SAME container the device reads, so its row order and
its 5.00-bit q4_1 arithmetic are FLM's, not a reimplementation of them.

    python3 quant_eval.py --groups 32,64,128,256 --ntok 256

STILL NO FWHT / clip-search / LDLQ on the oq4 side: those only improve it, so a
group that already beats q4nx here beats it by more with the real codec, and a
group that loses may still win with one. That asymmetry is the whole reason to
report the number rather than a verdict.

Needs the original checkpoint, the q4nx container, torch and `tokenizers`; no NPU.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import oracle_forward as of
import q4nx                       # numpy only -- safe to import beside torch

Q4NX = Path.home() / ".config/flm/models/Llama-3.2-1B-NPU2/model.q4nx"
# oracle key -> q4nx container key. The container uses HF names, the checkpoint
# uses Meta's; the tree's q4nx_tensor_blocks already returns checkpoint row order.
KEYMAP = {"attention.wq": "self_attn.q_proj", "attention.wk": "self_attn.k_proj",
          "attention.wv": "self_attn.v_proj", "attention.wo": "self_attn.o_proj",
          "feed_forward.w1": "mlp.gate_proj", "feed_forward.w2": "mlp.down_proj",
          "feed_forward.w3": "mlp.up_proj"}

TEXT = ("The capital of France is Paris, a city known for its museums and its "
        "river. Machine learning models are trained on large corpora of text, "
        "and their quality is usually measured by perplexity on held-out data. "
        "A quantised model stores each weight in fewer bits, which reduces the "
        "memory traffic needed to run it. The trade is accuracy: coarser "
        "quantisation moves the output distribution away from the original "
        "model, and the usual way to measure that shift is the Kullback-Leibler "
        "divergence between the two distributions at every position. ")


def bf16(a):
    # np.array (a COPY), not np.asarray (a VIEW): `u + r` can elide into `u`,
    # and if `u` views the caller's array this writes uint32 bit patterns
    # straight through it. Every caller here passes a temp, but that is luck
    # rather than design and this function is cheap.
    u = np.array(a, np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def fwht(x, s1, s2):
    """hipfire's `signed_fwht`, vectorised over leading axes.

    signs1, Hadamard butterfly, orthonormal 1/sqrt(n), signs2 -- in that order,
    matching hipfire-primitives/src/fwht.rs. Because H/sqrt(n) is its own inverse
    and the sign vectors are +-1 (self-inverse), the INVERSE is the same transform
    with the sign vectors SWAPPED, which is what codecs.rs's decoder does.
    """
    # Flattened to 2-D for the butterfly. The earlier version reshaped to
    # (*lead, nblk, 2, h) and stacked, which was CORRECT for a 2-D input and
    # WRONG for 3-D -- round-trip error 7.6 and energy ratio 0.103 at
    # (64, 8, 256), against 3.6e-07 at (3, 256). It only ever ran on 3-D data
    # (rows x groups x 256), so every oq4+ number it produced was garbage while
    # the 2-D self-test passed. Flattening removes the axis bookkeeping that got
    # it wrong; the checks below run at BOTH ranks now.
    x = np.asarray(x, np.float32)
    n = x.shape[-1]
    y = (x * s1).reshape(-1, n).copy()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        u = y[:, :, 0, :].copy()
        v = y[:, :, 1, :].copy()
        # EXPLICIT out=, and it is load-bearing. Written as `y[...] = u + v`
        # followed by `y[...] = u - v`, numpy 2.1.3 on Python 3.14 elides the
        # first sum INTO `u` -- so the second line computes (u+v)-v = u and the
        # butterfly silently degenerates. Only above the 256 KB elision
        # threshold: (2,3,4,256) at 24 KB round-tripped to 4.8e-07 while
        # (512,256) at 512 KB gave energy 0.099 and error 7.1, from the SAME
        # code. Copying u and v protects `y` but not `u` itself; out= is what
        # actually stops it.
        np.add(u, v, out=y[:, :, 0, :])
        np.subtract(u, v, out=y[:, :, 1, :])
        y = y.reshape(-1, n)
        h *= 2
    return (y.reshape(x.shape) / np.float32(np.sqrt(n))) * s2


def clipsearch(g, qmax=7.0):
    """hipfire's `symmetric_clipsearch`: the scale over a 9-point grid that
    minimises squared reconstruction error, not simply amax/qmax."""
    amax = np.abs(g).max(-1)
    best_s = np.maximum(amax / qmax, 1e-12).astype(np.float32)
    best_e = np.full(amax.shape, np.inf, np.float32)
    for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
        sc = np.maximum(np.float32(c) * amax / qmax, 1e-12).astype(np.float32)
        q = np.clip(np.rint(g / sc[..., None]), -qmax, qmax)
        # OPERAND ORDER IS LOAD-BEARING. Written `g - q*sc`, numpy 2.1.3 on
        # 3.14 elides the subtraction INTO `g` -- the caller's rotated weights --
        # so clipsearch silently replaced R with its own residuals: R rms fell
        # 0.019971 -> 0.000108, every code quantised to zero, and the
        # reconstruction came back 26x too small. Reversed, the elision target
        # is the fresh `q*sc` temp instead, and squared error does not care.
        e = ((q * sc[..., None] - g) ** 2).sum(-1)
        take = e < best_e
        best_s = np.where(take, sc, best_s)
        best_e = np.where(take, e, best_e)
    return best_s


def requant_variant(W, group, rng, rotate, clip):
    """oq4 with the rotation and the clip-search independently switchable.

    `oq4+` bundles two changes and measured WORSE end-to-end than plain absmax
    oq4 despite reconstructing the weights ~21% better. Two very different
    explanations, so they are separated here:

      rotate only  -- if this is what hurts, the rotation's value is on the
                      ACTIVATION side (W4A4) and an fp32-activation eval cannot
                      see it, which is the leading hypothesis.
      clip only    -- if this is what hurts, MSE-optimal scaling is not
                      output-optimal: clip-search minimises per-group squared
                      error and in doing so discards outlier weights the model
                      actually depends on.
    """
    N, K = W.shape
    Wg = W.reshape(N, K // group, group)
    if rotate:
        s1 = rng.choice(np.float32([-1, 1]), group)
        s2 = rng.choice(np.float32([-1, 1]), group)
        R = fwht(Wg, s1, s2)
    else:
        R = Wg
    sc = bf16(clipsearch(R) if clip else np.abs(R).max(-1) / np.float32(7.0))
    q = np.clip(np.rint(R / sc[..., None]), -7, 7).astype(np.float32)
    deq = bf16(q * sc[..., None])
    return (fwht(deq, s2, s1) if rotate else deq).reshape(N, K)


def requant_oq4pp(W, group, rng):
    """oq4+ as hipfire actually encodes it: FWHT-rotate, clip-search, int4.

    The rotation is what makes a symmetric per-group codebook fit a weight
    distribution -- without it this tree measured 4.5-7.7x worse KLD than q4nx at
    every group size. Modelled here by rotating, quantising and rotating BACK, so
    the effective reconstructed weight is exact and the forward is untouched. On
    device the same identity is realised by rotating the ACTIVATION instead.
    """
    N, K = W.shape
    s1 = rng.choice(np.float32([-1, 1]), group)
    s2 = rng.choice(np.float32([-1, 1]), group)
    Wg = W.reshape(N, K // group, group)
    R = fwht(Wg, s1, s2)
    sc = bf16(clipsearch(R))
    q = np.clip(np.rint(R / sc[..., None]), -7, 7).astype(np.float32)
    return fwht(bf16(q * sc[..., None]), s2, s1).reshape(N, K)


def requant_oq4(W, group):
    N, K = W.shape
    Wg = W.reshape(N, K // group, group)
    s = bf16(np.abs(Wg).max(2) / np.float32(7.0))
    inv = np.where(s > 0, np.float32(1.0) / np.maximum(s, np.float32(1e-30)), 0.0)
    q = np.clip(np.rint(Wg * inv[:, :, None]), -7, 7).astype(np.float32)
    return bf16(q * s[:, :, None]).reshape(N, K)


def weights_variant(sd, cfg, group, rotate, clip):
    import torch
    out = dict(sd)
    rng = np.random.default_rng(0)
    for L in range(cfg["num_hidden_layers"]):
        for nm in KEYMAP:
            k = f"layers.{L}.{nm}.weight"
            W = np.array(sd[k].to(torch.float32).numpy(), np.float32)
            out[k] = torch.from_numpy(requant_variant(W, group, rng, rotate, clip))
    return out


def weights_oq4(sd, cfg, group, pp=False):
    import torch
    out = dict(sd)
    rng = np.random.default_rng(0)      # fixed signs: encode and decode must agree
    for L in range(cfg["num_hidden_layers"]):
        for nm in KEYMAP:
            k = f"layers.{L}.{nm}.weight"
            # np.array = an explicit COPY. `.to(float32).numpy()` returns a
            # VIEW sharing memory with the state dict when the tensor is already
            # float32, and elision inside the quantisers then writes through to
            # the ORIGINAL weights -- so every mode silently corrupted the
            # weights for every mode after it. oq4+ runs last and looked worst,
            # while measuring BETTER than oq4 in isolation on the same matrices.
            W = np.array(sd[k].to(torch.float32).numpy(), np.float32)
            out[k] = torch.from_numpy(requant_oq4pp(W, group, rng) if pp
                                      else requant_oq4(W, group))
    return out


def weights_q4nx(sd, cfg):
    """FLM's own container, dequantised into the oracle's state dict."""
    import torch
    c = q4nx.Q4nx(str(Q4NX))
    out = dict(sd)
    for L in range(cfg["num_hidden_layers"]):
        for nm, hf in KEYMAP.items():
            k = f"layers.{L}.{nm}.weight"
            N, K = tuple(sd[k].shape)
            d, m, codes = q4nx.q4nx_tensor_blocks(c, f"model.layers.{L}.{hf}.weight",
                                                  (N, K))
            W = (d[:, :, None] * codes + m[:, :, None]).reshape(N, K)
            out[k] = torch.from_numpy(W.astype(np.float32))
    return out


def all_logits(toks, cfg, sd):
    """[T, vocab] float64 — logits at every position, not just the last."""
    import torch
    x, _ = of.forward(toks, cfg=cfg, sd=sd)
    return np.stack([np.asarray(of.logits(x[t], cfg=cfg, sd=sd), np.float64)
                     for t in range(len(toks))])


def logsoftmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def score(ref, got, toks):
    """(KLD nats, PPL, top1) — ref and got are [T, vocab] logits."""
    lr, lg = logsoftmax(ref), logsoftmax(got)
    pr = np.exp(lr)
    kld = float((pr * (lr - lg)).sum(-1)[:-1].mean())      # last has no next token
    nxt = np.asarray(toks[1:], np.int64)
    ppl = float(np.exp(-lg[np.arange(len(nxt)), nxt].mean()))
    top1 = float((got.argmax(-1) == ref.argmax(-1)).mean())
    return kld, ppl, top1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", default="32,64,128,256")
    p.add_argument("--ntok", type=int, default=128)
    o = p.parse_args()

    cfg = json.loads(of.CFG.read_text())
    toks = ([128000] + of.encode(TEXT))[:o.ntok]
    sd = of.load(cfg)
    print(f"quant eval, {len(toks)} tokens, KLD vs fp32 in nats (lower better)")

    ref = all_logits(toks, cfg, sd)
    _, ppl0, _ = score(ref, ref, toks)
    print(f"  {'format':10s} {'b/w':>6} {'KLD':>10} {'PPL':>9} {'top1':>7}   vs q4nx")
    print(f"  {'fp32':10s} {16.0:>6.2f} {0.0:>10.5f} {ppl0:>9.4f} {1.0:>7.3f}")

    rows = []
    kld_bar = None
    gs = [int(x) for x in o.groups.split(",")]
    for name, w, bw in ([("q4nx", weights_q4nx(sd, cfg), 5.0)] +
                        [(f"oq4_g{g}", weights_oq4(sd, cfg, g), 4.0 + 16.0 / g)
                         for g in gs] +
                        [(f"oq4+_g{g}", weights_oq4(sd, cfg, g, pp=True),
                          4.0 + 16.0 / g) for g in gs] +
                        # THE CONTROL: requant_variant with BOTH switches off.
                        # It must land on plain oq4. If it does not, the ablation
                        # is comparing code paths rather than features -- the
                        # variant divides by the scale where requant_oq4
                        # multiplies by a precomputed reciprocal, which rounds
                        # differently at .5 boundaries.
                        [(f"base_g{g}", weights_variant(sd, cfg, g, False, False),
                          4.0 + 16.0 / g) for g in gs] +
                        [(f"rot_g{g}", weights_variant(sd, cfg, g, True, False),
                          4.0 + 16.0 / g) for g in gs] +
                        [(f"clip_g{g}", weights_variant(sd, cfg, g, False, True),
                          4.0 + 16.0 / g) for g in gs]):
        k, ppl, t1 = score(ref, all_logits(toks, cfg, w), toks)
        if kld_bar is None:
            kld_bar = k
        verdict = "" if name == "q4nx" else (
            f"  {'BETTER' if k < kld_bar else 'worse':>6}  {k/kld_bar:.2f}x KLD")
        print(f"  {name:10s} {bw:>6.4f} {k:>10.5f} {ppl:>9.4f} {t1:>7.3f}{verdict}")
        rows.append((name, bw, k, ppl))

    print("\n  The bar is q4nx, not fp32: a format is worth porting only if it beats\n"
          "  what FLM already ships, at fewer bits. No FWHT/clip-search/LDLQ on the\n"
          "  oq4 rows -- all three only improve them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
