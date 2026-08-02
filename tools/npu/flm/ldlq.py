#!/usr/bin/env python3
"""LDLQ: Hessian error-feedback quantisation. Does it beat q4nx?

Everything measured here so far says the lever is CALIBRATION, not the container:
q4_1 at group 32 -- q4nx's exact format and bit rate -- scores 4.53x worse KLD
than q4nx while reconstructing only 5% worse. And reconstruction error has failed
to predict output quality five times, in both directions.

hipfire's own `ldlq.rs` header records the same conclusion from the other side:
"MSE-optimal QTIP-2 failed PPL (125.6 vs MQ4 14.0) because reconstruction-optimal
!= output-optimal; LDLQ minimizes the activation-weighted output error
||(W-W')*sqrt(H)|| via OBS error feedback."

So this ports that algorithm, faithfully:

    L with L.L^T = (H + lambda I)^-1     `inv_cholesky_lower`
    per 256-column block, per row:       clip-search scale, round to int4
    err[c] = (w[c] - w'[c]) / L[c,c]     OBS error
    residual[f] -= err[c] * L[f,c]       propagate to FUTURE blocks only

THE HESSIAN IS THE HARD PART, and it is why this is a separate file. H = X^T X
over the layer's real inputs must have rank >= K or the Cholesky is pure damping
and LDLQ degenerates to round-to-nearest. K is 2048 for six of the seven linears
and 8192 for down_proj, so 64 calibration tokens -- what the KLD harness uses --
would be rank-deficient by 32x. Accumulated in CHUNKS instead: each chunk is a
short forward whose [heads, T, T] attention stays small, and H accumulates across
them, so the token count is unbounded by memory.

Four distinct Hessians per layer, not seven: wq/wk/wv share the attention-norm
output, w1/w3 share the ffn-norm output, wo takes the attention output and w2 the
SwiGLU product.

    python3 ldlq.py --calib 2048 --chunk 256 --ntok 64

Needs the original checkpoint, the q4nx container and torch; no NPU.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

import oracle_forward as of
import quant_eval as qe

GROUP = 256
# Which weight matrices consume which hook point. The hook fires in this order
# inside each layer, so the capture below keys on the call index.
HOOK_OF = {"attention.wq": 0, "attention.wk": 0, "attention.wv": 0,
           "attention.wo": 1, "feed_forward.w1": 2, "feed_forward.w3": 2,
           "feed_forward.w2": 3}

CALIB = (
    "The history of computing is a history of tradeoffs between memory and time. "
    "Quantisation reduces the number of bits used to store each parameter of a "
    "neural network, which reduces the memory traffic required to evaluate it. "
    "Because inference at batch size one is bound by memory bandwidth rather than "
    "arithmetic, fewer bits per weight translates almost directly into more tokens "
    "per second. The difficulty is that coarser quantisation changes the model's "
    "output distribution. A calibrated quantiser uses statistics gathered from real "
    "activations to decide where to place its levels, so that the error it "
    "introduces falls in directions the model is least sensitive to. "
    "In practice this means minimising the activation-weighted error rather than "
    "the plain reconstruction error, because the two are not the same objective and "
    "optimising one can make the other worse. "
    "Paris is the capital of France, Rome is the capital of Italy, and Tokyo is the "
    "capital of Japan. Water freezes at zero degrees Celsius and boils at one "
    "hundred. The mitochondria are the powerhouse of the cell. In a market economy "
    "prices are set by the interaction of supply and demand, and a shortage raises "
    "the price until the quantity demanded falls to meet the quantity supplied. "
)


CALIB_HFQ = Path.home() / ".hipfire/calib/llama-3.2-1b.calib.hfq"


def read_calib(path=CALIB_HFQ):
    """-> ({name: H [K,K] float32}, metadata) from an HFQM calibration package.

    hipfire ships real Hessians: 112 of them (16 layers x 7 linears), collected
    over wikitext2 by the engine itself. That is strictly better than anything
    this tree could gather -- capturing them here would need thousands of
    calibration tokens through a reference forward, and would still be a
    different corpus from the one the production quantiser uses.

    Layout (hessian_io.rs): 32-byte header, self-delimited JSON, then a tensor
    index of [u16 name_len][name][u8 quant_type][u8 n_dims][u32 shape...]
    [u32 group_size][u64 data_size][u64 payload_offset_in_32B_units].

    quant_type 2 is dense row-major F32. Type 130 is the compact form: an exact
    F32 diagonal followed by a BF16 lower STRICT triangle, which halves a K x K
    package and is why a 1.5 GB file holds 112 Hessians of up to 8192 x 8192.
    """
    import mmap as _mm
    f = open(path, "rb")
    mm = _mm.mmap(f.fileno(), 0, access=_mm.ACCESS_READ)
    assert mm[:4] == b"HFQM", mm[:4]
    ver = int.from_bytes(mm[4:8], "little")
    n = int.from_bytes(mm[12:16], "little")
    moff = int.from_bytes(mm[16:24], "little")
    doff = int.from_bytes(mm[24:32], "little")
    meta, jend = json.JSONDecoder().raw_decode(
        mm[moff:doff].decode("utf-8", errors="ignore"))
    pos = moff + jend
    assert int.from_bytes(mm[pos:pos + 4], "little") == n
    pos += 4
    out = {}
    cur = doff
    for _ in range(n):
        nl = int.from_bytes(mm[pos:pos + 2], "little"); pos += 2
        name = mm[pos:pos + nl].decode(); pos += nl
        qt = mm[pos]; pos += 1
        nd = mm[pos]; pos += 1
        shape = [int.from_bytes(mm[pos + 4 * i:pos + 4 * i + 4], "little")
                 for i in range(nd)]
        pos += 4 * nd
        pos += 4                                        # group_size
        dsz = int.from_bytes(mm[pos:pos + 8], "little"); pos += 8
        if ver >= 2:
            off = int.from_bytes(mm[pos:pos + 8], "little") * 32
            pos += 8
        else:
            # v1 carries NO payload offset: payloads are contiguous from
            # data_offset in index order, so the cursor has to be carried.
            off = cur
        cur += dsz
        if not name.endswith(".hessian"):
            continue
        K = shape[0]
        buf = mm[off:off + dsz]
        if qt == 2:
            H = np.frombuffer(buf, np.float32, K * K).reshape(K, K).copy()
        elif qt == 130:
            diag = np.frombuffer(buf, np.float32, K).copy()
            tri = np.frombuffer(buf, np.uint16, offset=4 * K,
                                count=K * (K - 1) // 2).astype(np.uint32)
            low = (tri << 16).view(np.float32) if tri.dtype == np.uint32 else None
            H = np.zeros((K, K), np.float32)
            iy, ix = np.tril_indices(K, -1)
            H[iy, ix] = low
            H = H + H.T
            H[np.arange(K), np.arange(K)] = diag
        else:
            continue
        out[name[:-len(".hessian")]] = H
    return out, meta


def inv_chol_lower(H, damp_frac=0.01, tries=24):
    """L with L @ L.T = (H + lambda I)^-1. Escalates lambda until it factors.

    The shipped Hessians are NOT positive definite as stored: rank ~1060 of 2048
    from 128 calibration sequences, and the compact package keeps the strict
    lower triangle in BF16, which perturbs the spectrum. Layer 0 q_proj has a
    minimum eigenvalue of -6.7e-3 against a mean diagonal of ~5e-2, so the
    conventional 1%-of-mean-diagonal damping (5e-4) is an order of magnitude too
    small and the Cholesky fails outright.

    Escalation is x2, not x10, and that matters: damping pulls the solution
    toward plain round-to-nearest, so overshooting throws away the very thing
    LDLQ is for. Layer 0 q_proj needs lambda just above 6.7e-3; x10 from 5.1e-4
    lands on 5.1e-2 -- 100% of the mean diagonal, ~8x more damping than
    required -- while x2 lands on ~8e-3. The first lambda that FACTORS is the
    one to use.
    """
    K = H.shape[0]
    base = float(np.trace(H)) / K
    lam = damp_frac * base
    for i in range(tries):
        try:
            # The OBS loop needs the LOWER Cholesky factor of A^-1: it reads
            # L[f, c] for f > c to propagate error forward. C^-T satisfies
            # L L^T = A^-1 but is UPPER triangular, so every L[f, c] below the
            # diagonal is zero, no error propagates, and LDLQ silently becomes
            # plain RTN -- which is exactly what a 57-minute 16-layer run
            # produced, bit-identical to the RTN row. The equation was right and
            # the SHAPE was wrong, and a check that only verified
            # L L^T (H+lI) v == v passed it at 3.2e-12.
            #
            # So: invert the triangular factor (K^3/3), form A^-1 = X^T X as a
            # symmetric product, and take its lower Cholesky (K^3/3). Still well
            # under a general inverse plus a Cholesky.
            A = H + lam * np.eye(K)
            C = np.linalg.cholesky(A)
            X = solve_triangular(C, np.eye(K), lower=True)      # C^-1, lower
            Hinv = X.T @ X                                      # = A^-1, sym
            Hinv = 0.5 * (Hinv + Hinv.T)
            L = np.linalg.cholesky(Hinv)                        # LOWER
            assert abs(L[0, -1]) == 0.0 and np.count_nonzero(
                L[K // 2:, :K // 2]) > 0, "factor is not lower triangular"
            return L, lam / base
        except np.linalg.LinAlgError:
            lam *= 2.0
    raise np.linalg.LinAlgError(f"no damping in {tries} tries made H SPD")


_LCACHE = {}


def cached_factor(H, damp):
    """inv_chol_lower keyed on the Hessian object, because SEVEN matrices per
    layer share only FOUR Hessians: q/k/v all consume the attention-norm output
    and gate/up both consume the ffn-norm output. Refactorising per matrix
    repeats the two 2048 factorisations three and two times over."""
    key = (id(H), damp)
    if key not in _LCACHE:
        _LCACHE[key] = (H, inv_chol_lower(H, damp))     # hold H: id() is reused
    return _LCACHE[key][1]


def ldlq_quant(W, H, group=GROUP, clip=True, damp=0.01):
    """int4 symmetric, per-group scale, with OBS error feedback. -> dequantised W.

    Block-sequential exactly as ldlq.rs: the error of a 256-column block is
    propagated only to columns AFTER that block, never within it, so each block
    is quantised from a residual that already carries every earlier block's error.
    """
    N, K = W.shape
    L, used = cached_factor(H, damp)
    res = np.array(W, np.float64)
    out = np.empty_like(res)
    diag = np.diag(L).copy()
    for c0 in range(0, K, group):
        c1 = c0 + group
        g = res[:, c0:c1]
        sc = (qe.clipsearch(g.astype(np.float32)) if clip
              else np.abs(g).max(-1) / np.float32(7.0))
        sc = qe.bf16(np.maximum(sc, np.float32(1e-30))).astype(np.float64)
        q = np.clip(np.rint(g / sc[:, None]), -7, 7)
        deq = np.asarray(qe.bf16((q * sc[:, None]).astype(np.float32)), np.float64)
        out[:, c0:c1] = deq
        if c1 >= K:
            break
        d = diag[c0:c1]
        err = np.where(d > 0, (g - deq) / np.where(d > 0, d, 1.0), 0.0)
        # residual[:, f] -= sum_c err[:, c] * L[f, c]   for f >= c1
        res[:, c1:] -= err @ L[c1:, c0:c1].T
    return out.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ntok", type=int, default=64, help="eval tokens")
    p.add_argument("--group", type=int, default=GROUP)
    p.add_argument("--layers", default="", help="restrict layers (debug)")
    p.add_argument("--damp", type=float, default=0.01)
    o = p.parse_args()
    import torch

    cfg = json.loads(of.CFG.read_text())
    sd = of.load(cfg)
    nlay = cfg["num_hidden_layers"]
    layers = (set(range(nlay)) if not o.layers
              else {int(x) for x in o.layers.split(",")})

    t0 = time.time()
    H, meta = read_calib()
    print(f"  {len(H)} Hessians from {Path(meta.get('corpus','?')).name} "
          f"({time.time()-t0:.0f} s)")

    t0 = time.time()
    out = dict(sd)
    for L in sorted(layers):
        # Per LAYER, not global: L for an 8192 Hessian is 537 MB in f64, and
        # holding all sixteen layers' factorisations would be ~10 GB. Each
        # layer's four Hessians are used only within that layer.
        _LCACHE.clear()
        for nm, hf in qe.KEYMAP.items():
            k = f"layers.{L}.{nm}.weight"
            h = H.get(f"model.layers.{L}.{hf}")
            W = np.array(sd[k].to(torch.float32).numpy(), np.float32)
            if h is None:
                print(f"    no Hessian for {hf}, leaving fp32")
                continue
            out[k] = torch.from_numpy(ldlq_quant(W, h.astype(np.float64),
                                                 o.group, damp=o.damp))
        print(f"    layer {L} done ({time.time()-t0:.0f} s)")
    print(f"  LDLQ quantised {len(layers)*7} matrices ({time.time()-t0:.0f} s)")

    toks = ([128000] + of.encode(qe.TEXT))[:o.ntok]
    ref = qe.all_logits(toks, cfg, sd)
    # RTN over the SAME layers as LDLQ. Quantising all 16 for RTN while LDLQ
    # covers only the requested ones made LDLQ look 2.1x better in the layer-0
    # smoke test purely because fifteen of its layers were still fp32.
    rtn_all = qe.weights_variant(sd, cfg, o.group, False, True)
    rtn = dict(sd)
    for L in sorted(layers):
        for nm in qe.KEYMAP:
            k = f"layers.{L}.{nm}.weight"
            rtn[k] = rtn_all[k]
    rows = [("q4nx", qe.weights_q4nx(sd, cfg), 5.0),
            ("oq4 RTN", rtn, 4.0 + 16.0 / o.group),
            ("oq4 LDLQ", out, 4.0 + 16.0 / o.group)]
    print(f"\n  {'format':10s} {'b/w':>6} {'KLD':>10} {'PPL':>9} {'top1':>7}   vs q4nx")
    bar = None
    for nm, w, bw in rows:
        k, ppl, t1 = qe.score(ref, qe.all_logits(toks, cfg, w), toks)
        if bar is None:
            bar = k
        v = "" if nm == "q4nx" else (
            f"   {'BETTER' if k < bar else 'worse':>6}  {k/bar:.2f}x")
        print(f"  {nm:10s} {bw:>6.4f} {k:>10.5f} {ppl:>9.4f} {t1:>7.3f}{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
