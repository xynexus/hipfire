#!/usr/bin/env python3
"""Cross-check the coarse tier and the exact reference OUTSIDE numpy.

Why this file exists, separately from `lmhead_coarse.py --selfcheck`
-------------------------------------------------------------------
Three silent numpy faults have been measured in this venv (numpy 2.1.3 on
Python 3.14.4) in one day:

  1. temp elision writing into a named local, because 3.14's `LOAD_FAST_BORROW`
     leaves an operand at refcount 1 (found in this file's sibling, and it made
     a row cosine read 22.0);
  2. `(g / (1 + exp(-g))) * u` evaluating to exactly `u`, dropping a SwiGLU
     gate, with the same expression correct three lines later;
  3. a GEMM returning three different answers in one scope, one of them exactly
     the sum of two others.

All three are silent, none reproduce standalone, and 2 and 3 are SHAPE
dependent — they appear on wide shapes and vanish on narrow ones. The coarse
tier is the widest numpy work in this project: dequantise 128256 x 2048,
row-norm it, quantize it. A dropped or doubled term there does not raise; it
produces a plausible recall curve for a tier that is not the one specified.

So the tier is rebuilt here in **torch**, which shares no kernels with numpy,
and compared BIT FOR BIT — not to a tolerance, because the format is integers
and an f16 scale, and anything but exact equality is a defect.

`import aie.iron` followed by `import torch` segfaults in this venv, so this
file imports torch and never touches iron. Build the tier here, save it, run
the NPU from `lmhead_twostage.py` in a different process.

    python3 lmhead_coarse_xcheck.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

# NOT `import lmhead_coarse`. It reaches `qkv_verify` for K_DIM, and that module
# imports `aie.iron` at module scope -- which segfaults this process the moment
# torch is also loaded. The two constants and the one helper it would have
# supplied are restated here; the arrays come from the on-disk cache, which is
# the artefact under test anyway.
BLK = q4nx.BLK
K_DIM = 2048
CACHE = Path("/tmp/lmhead2p")


def rnd(x):
    return q4nx.bf16_to_f32(q4nx.f32_to_bf16(x))


def lmhead_blocks():
    z = np.load(CACHE / "lmhead_blocks.npz")
    return z["D"], z["M"], z["C"]


def coarse_tier():
    z = np.load(CACHE / "lmhead_coarse.npz")
    return z["nib"], z["scale"]


def load_probes():
    z = np.load(CACHE / "probes.npz")
    return list(zip(z["toks"].tolist(), z["xn"], z["amax"].tolist()))


def numpy_coarse_logits(nib, scale, act, chunk=16384):
    """`lmhead_coarse.coarse_logits`, restated (see the import note above)."""
    a = rnd(np.asarray(act, np.float32))
    rows = nib.shape[0]
    out = np.empty(rows, np.float32)
    lut = (np.arange(16, dtype=np.int8) << 4 >> 4).astype(np.float32)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        b = nib[lo:hi]
        q = np.empty((hi - lo, a.size), np.float32)
        q[:, 0::2] = lut[b & 0x0F]
        q[:, 1::2] = lut[b >> 4]
        out[lo:hi] = (q @ a) * scale[lo:hi].astype(np.float32)
    return out


def numpy_gemv_bf16(act, d, m, codes, chunk=4096):
    """`lmhead_coarse.gemv_bf16_fast`, restated (see the import note above)."""
    nrows, nb = d.shape
    a = np.asarray(act, np.float32).reshape(nb, BLK)
    asum = rnd(a.astype(np.float64).sum(1).astype(np.float32)).astype(np.float64)
    out = np.empty(nrows, np.float64)
    for lo in range(0, nrows, chunk):
        hi = min(lo + chunk, nrows)
        a_s = rnd(a[None, :, :] * rnd(d[lo:hi])[:, :, None]).astype(np.float64)
        out[lo:hi] = np.einsum("rbt,rbt->r", codes[lo:hi].astype(np.float64), a_s)
        out[lo:hi] += rnd(m[lo:hi]).astype(np.float64) @ asum
    return out


def torch_coarse(D, M, C):
    """`build_coarse_q4row` again, in torch, from the same q4_1 blocks.

    Deliberately NOT a transcription of the numpy version's expression tree:
    the dequant is a fused `addcmul`, the norm is `torch.linalg.vector_norm`
    (not sqrt-of-sum-of-squares), and the pack is integer shifts on int16. If
    both land on the same bytes, they are not sharing a bug.
    """
    rows, nb = D.shape
    cols = nb * BLK
    d = torch.from_numpy(np.ascontiguousarray(D)).to(torch.float32)
    m = torch.from_numpy(np.ascontiguousarray(M)).to(torch.float32)
    c = torch.from_numpy(np.ascontiguousarray(C)).to(torch.float32)
    W = torch.addcmul(m.unsqueeze(-1), d.unsqueeze(-1), c).reshape(rows, cols)

    unit = 3.0 / (7.0 * float(np.sqrt(cols)))
    norm = torch.linalg.vector_norm(W.to(torch.float64), dim=1).to(torch.float32)
    inv = torch.where(norm > 0, 1.0 / (unit * torch.clamp(norm, min=1e-30)),
                      torch.zeros_like(norm))
    q = torch.clamp(torch.round(W * inv.unsqueeze(1)), -7, 7).to(torch.int16)
    u = (q & 0x0F).to(torch.uint8)
    nib = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)
    scale = (norm * unit).to(torch.float16)
    return nib.numpy(), scale.numpy()


def torch_coarse_logits(nib, scale, act):
    """The coarse GEMV in torch, from the packed nibbles."""
    b = torch.from_numpy(np.ascontiguousarray(nib)).to(torch.int16)
    rows, half = b.shape
    q = torch.empty((rows, 2 * half), dtype=torch.float32)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    q[:, 0::2] = torch.where(lo > 7, lo - 16, lo).to(torch.float32)
    q[:, 1::2] = torch.where(hi > 7, hi - 16, hi).to(torch.float32)
    a = torch.from_numpy(np.ascontiguousarray(rnd(np.asarray(act, np.float32))))
    return (q @ a * torch.from_numpy(scale.astype(np.float32))).numpy()


def main():
    D, M, C = lmhead_blocks()
    nib, scale = coarse_tier()

    # --- the tier, bit for bit, on a wide slice ------------------------------
    # 16384 rows, not a few hundred: the faults are shape dependent and the
    # question is whether the WIDE path is right.
    for lo, hi in ((0, 16384), (60000, 76384), (111872, 128256)):
        tn, ts = torch_coarse(D[lo:hi], M[lo:hi], C[lo:hi])
        bad_n = int((tn != nib[lo:hi]).sum())
        bad_s = int((ts.view(np.uint16) != scale[lo:hi].view(np.uint16)).sum())
        print(f"rows {lo}-{hi}: nibble bytes differing {bad_n}/{tn.size}, "
              f"f16 scales differing {bad_s}/{ts.size}")
        assert bad_n == 0 and bad_s == 0, "numpy and torch built different tiers"

    # --- the coarse GEMV, torch vs numpy, full vocabulary --------------------
    probes = load_probes()
    xn = probes[0][1]
    a = numpy_coarse_logits(nib, scale, xn)
    b = torch_coarse_logits(nib, scale, xn)
    rel = float(np.abs(np.subtract(a, b)).max() / np.abs(b).max())
    print(f"coarse GEMV over {len(a)} rows: torch vs numpy rel {rel:.3e}, "
          f"argmax {int(np.argmax(a))} vs {int(np.argmax(b))}")
    assert rel < 1e-5 and np.argmax(a) == np.argmax(b)

    # --- the exact reference, wide ------------------------------------------
    # The 256-row bit-identity was under whatever width triggers the faults.
    # This is the array that decides every probe's ground-truth argmax.
    n = 8192
    ref = q4nx.gemv_reference_bf16(xn, D[:n], M[:n], C[:n])
    fast = numpy_gemv_bf16(xn, D[:n], M[:n], C[:n])
    err = float(np.abs(np.subtract(ref, fast)).max())
    print(f"exact GEMV over {n} rows: fast vs q4nx.gemv_reference_bf16 "
          f"max abs {err:.3e}")
    assert err == 0.0, "the vectorised exact reference is not bit-identical at width"

    # --- and the ground truth itself, against torch --------------------------
    # argmax over the WHOLE vocabulary is what the recall curve is measured
    # against, so recompute it a third way.
    d = torch.from_numpy(np.ascontiguousarray(D)).to(torch.float64)
    m = torch.from_numpy(np.ascontiguousarray(M)).to(torch.float64)
    c = torch.from_numpy(np.ascontiguousarray(C)).to(torch.float64)
    x = torch.from_numpy(np.ascontiguousarray(rnd(xn))).to(torch.float64)
    got = []
    for lo in range(0, D.shape[0], 16384):
        hi = min(lo + 16384, D.shape[0])
        # w = d*q + m, dotted with x directly -- no block-sum identity, no
        # bf16 rounding of the scaled activation. A DIFFERENT decomposition of
        # the same GEMV, so it cannot share the reference's arithmetic bugs.
        W = torch.addcmul(m[lo:hi].unsqueeze(-1), d[lo:hi].unsqueeze(-1),
                          c[lo:hi]).reshape(hi - lo, K_DIM)
        got.append(W @ x)
    tlog = torch.cat(got).numpy()
    nlog = numpy_gemv_bf16(xn, D, M, C)
    print(f"full-vocab argmax: numpy bf16 reference {int(np.argmax(nlog))}, "
          f"torch float64 exact-dequant {int(np.argmax(tlog))}, "
          f"probe records {probes[0][2]}")
    assert int(np.argmax(tlog)) == probes[0][2] == int(np.argmax(nlog))
    print("\nALL CROSS-CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
