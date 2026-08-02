#!/usr/bin/env python3
"""Read hipfire's REAL oq4++ artifact and score it against q4nx.

Everything measured in this tree says the lever is calibration, not the
container: q4_1 at group 32 -- q4nx's exact format and bit rate -- is 4.53x worse
KLD while only 5% worse on reconstruction. The open question was whether the full
codec closes that gap. It does not need reimplementing: hipfire has already
produced `Llama-3.2-1B-Instruct-nc--oq4++.hfq`, quantised from the Instruct bf16
weights with the Instruct Hessians (`llama-3.2-1b-inst.calib.hfq`) -- the same
model this tree's fp32 oracle loads.

WHAT oq4++ ACTUALLY IS, from the artifact's own index:

    qt 34   112 tensors   the weights, 4.0625 b/w   Oq4G256 blocks
    qt  1   112 tensors   `<name>.awq_scale.weight`, f16, length K
    qt 16    34 tensors   embeddings, norms, lm_head at 16 b/w
    qt 48     1 tensor    embed_tokens.coarse at 4.0078 b/w

so it is FWHT + clip-search + LDLQ + per-channel AWQ smoothing, and 973M of the
model's 1236M parameters are quantised (embeddings and the head are not).

RECOVERING THE EFFECTIVE WEIGHT. The quantiser stores `rot(W*s)`; the runtime
computes `rot(x/s)` and `(W*s).(x/s) = W.x` cancels. So:

    dequantise the int4 blocks        -> rot(W*s)
    inverse FWHT, signs SWAPPED       -> W*s
    divide by the AWQ scale per column-> W

Signs are `gen_fwht_signs(42, 256)` and `gen_fwht_signs(1042, 256)` -- a fixed
LCG, seeds baked into the codec, reproduced here exactly.

    python3 read_oq4.py --ntok 64

Needs the artifact, the q4nx container, the checkpoint and torch; no NPU.
"""

import argparse
import json
import mmap
from pathlib import Path

import numpy as np

import oracle_forward as of
import quant_eval as qe

ARTIFACT = Path.home() / ".hipfire/models/Llama-3.2-1B-Instruct-nc--oq4++.hfq"
GROUP = 256
BLOCK = 130                      # [f16 scale][128 nibbles] per 256 weights


def gen_fwht_signs(seed, n=GROUP):
    """hipfire-primitives `gen_fwht_signs`, exactly: an LCG whose bit 16 picks
    the sign. Reproduced rather than guessed -- the inverse rotation is wrong
    without the identical table, and wrong in a way that still looks like
    plausible weights."""
    out = np.empty(n, np.float32)
    state = np.uint32(seed)
    for i in range(n):
        state = np.uint32((int(state) * 1103515245 + 12345) & 0x7FFFFFFF)
        out[i] = 1.0 if (int(state) >> 16) & 1 else -1.0
    return out


def index(path):
    """-> (mmap, {name: (quant_type, shape, size, offset)}, metadata)."""
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert mm[:4] == b"HFQM"
    ver = int.from_bytes(mm[4:8], "little")
    n = int.from_bytes(mm[12:16], "little")
    moff = int.from_bytes(mm[16:24], "little")
    doff = int.from_bytes(mm[24:32], "little")
    meta, jend = json.JSONDecoder().raw_decode(
        mm[moff:doff].decode("utf-8", errors="ignore"))
    pos = moff + jend
    assert int.from_bytes(mm[pos:pos + 4], "little") == n
    pos += 4
    idx, cur = {}, doff
    for _ in range(n):
        nl = int.from_bytes(mm[pos:pos + 2], "little"); pos += 2
        name = mm[pos:pos + nl].decode(); pos += nl
        qt = mm[pos]; pos += 1
        nd = mm[pos]; pos += 1
        shape = [int.from_bytes(mm[pos + 4 * j:pos + 4 * j + 4], "little")
                 for j in range(nd)]
        pos += 4 * nd + 4
        dsz = int.from_bytes(mm[pos:pos + 8], "little"); pos += 8
        if ver >= 2:
            off = int.from_bytes(mm[pos:pos + 8], "little") * 32; pos += 8
        else:
            off = cur
        cur += dsz
        idx[name] = (qt, tuple(shape), dsz, off)
    return mm, idx, meta


def dequant_oq4(mm, ent, s1, s2, awq=None):
    """Oq4G256 blocks -> the effective [N, K] weight, un-rotated and un-scaled."""
    qt, (N, K), dsz, off = ent
    ng = K // GROUP
    raw = np.frombuffer(mm, np.uint8, count=N * ng * BLOCK, offset=off)
    raw = raw.reshape(N, ng, BLOCK)
    sc = raw[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(N, ng)
    nib = raw[:, :, 2:]                                   # [N, ng, 128]
    lo = (nib & 0x0F).astype(np.int8)
    hi = (nib >> 4).astype(np.int8)
    q = np.empty((N, ng, GROUP), np.int8)
    q[:, :, 0::2] = np.where(lo > 7, lo - 16, lo)         # 4-bit two's complement
    q[:, :, 1::2] = np.where(hi > 7, hi - 16, hi)
    rot = q.astype(np.float32) * sc[:, :, None]
    W = qe.fwht(rot, s2, s1).reshape(N, K)                # inverse: signs SWAPPED
    if awq is not None:
        W = W / awq[None, :]                              # undo (W*s)
    return W.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ntok", type=int, default=64)
    p.add_argument("--artifact", default=str(ARTIFACT))
    o = p.parse_args()
    import torch

    cfg = json.loads(of.CFG.read_text())
    sd = of.load(cfg)
    mm, idx, meta = index(o.artifact)
    print(f"  {meta.get('quant_format')} from {Path(meta['source_hfq']['path']).name}, "
          f"calib {meta['calibration']['source']}")
    s1, s2 = gen_fwht_signs(42), gen_fwht_signs(1042)

    out = dict(sd)
    nq = 0
    for L in range(cfg["num_hidden_layers"]):
        for nm, hf in qe.KEYMAP.items():
            wn = f"model.layers.{L}.{hf}.weight"
            if wn not in idx:
                continue
            aw = idx.get(f"model.layers.{L}.{hf}.awq_scale.weight")
            a = None
            if aw is not None:
                a = np.frombuffer(mm, np.float16, count=aw[1][0],
                                  offset=aw[3]).astype(np.float32)
            W = dequant_oq4(mm, idx[wn], s1, s2, a)
            k = f"layers.{L}.{nm}.weight"
            assert W.shape == tuple(sd[k].shape), (wn, W.shape, sd[k].shape)
            out[k] = torch.from_numpy(W)
            nq += 1
    print(f"  dequantised {nq} weight tensors")

    toks = ([128000] + of.encode(qe.TEXT))[:o.ntok]
    ref = qe.all_logits(toks, cfg, sd)
    rows = [("q4nx", qe.weights_q4nx(sd, cfg), 5.0),
            ("oq4 RTN", qe.weights_variant(sd, cfg, GROUP, False, True), 4.0625),
            ("oq4++ (hipfire)", out, 4.0625)]
    print(f"\n  {'format':16s} {'b/w':>6} {'KLD':>10} {'PPL':>9} {'top1':>7}   vs q4nx")
    bar = None
    for nm, w, bw in rows:
        k, ppl, t1 = qe.score(ref, qe.all_logits(toks, cfg, w), toks)
        if bar is None:
            bar = k
        v = "" if nm == "q4nx" else (
            f"   {'BETTER' if k < bar else 'worse':>6}  {k/bar:.2f}x")
        print(f"  {nm:16s} {bw:>6.4f} {k:>10.5f} {ppl:>9.4f} {t1:>7.3f}{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
