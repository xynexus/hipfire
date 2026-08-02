#!/usr/bin/env python3
"""The coarse tier of a two-pass lm_head, and the recall gate that decides it.

Pass 1 (coarse, meant for the NPU) scores the WHOLE vocabulary from a 4-bit
row-normalised copy of `lm_head.weight`; pass 2 (fine, host) rescores only the
top-K of that shortlist with the exact q4nx arithmetic and takes the argmax of
the rescored set. The point is bytes, not arithmetic: lm_head is bandwidth
bound at 54.7 GB/s (97% of the 56.5 GB/s fabric roof), so the only lever is
streaming fewer of them. The exact q4nx tier costs 0.629 B/weight (4-bit code
plus a per-32 scale AND min); the coarse tier costs 0.5 B/weight plus 2 B a row.

The format is `hipfire_quantize::codecs::build_coarse_q4row`, transcribed, not
reinvented:

  * per row the exact L2 norm is factored out and becomes the row's f16 scale,
  * only the UNIT DIRECTION is quantized, symmetric Q4, one global 3-sigma step
    `unit_scale = 3 / (7*sqrt(cols))`, levels [-7, 7],
  * the stored scale is `norm * unit_scale`, folding the norm and the shared
    step together,
  * planar: `[rows*cols/2 nibble bytes][rows*2 f16 scale bytes]`, nibble 2i in
    the LOW half of byte i and 2i+1 in the HIGH half.

Row normalisation is the whole reason 4 bits is enough. A per-row-max Q4 lets a
handful of outlier channels set the step and crushes the rest of the direction;
against the norm every row spends its levels the same way.

The coarse tier does not have to be accurate. It has to keep the TRUE argmax
inside the top-K. `--recall` measures exactly that, on real hidden states rather
than random vectors -- the same quantity the Rust verifiers call recall@1.

    python3 lmhead_coarse.py --build              # one-time, ~2 min
    python3 lmhead_coarse.py --probes 24          # real hidden states, ~1/token
    python3 lmhead_coarse.py --recall             # the curve, and the K it picks

Needs the q4nx container; no NPU.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402
from head_verify import Q4NX, VOCAB, rmsnorm  # noqa: E402
from qkv_verify import K_DIM  # noqa: E402

BLK = q4nx.BLK
CACHE = Path("/tmp/lmhead2p")


def rnd(x):
    """f32 -> bf16 -> f32, the rounding the kernel actually does."""
    return q4nx.bf16_to_f32(q4nx.f32_to_bf16(x))


# !!! numpy 2.1.3 on Python 3.14 SILENTLY CORRUPTS NAMED LOCAL ARRAYS. !!!
#
# 3.14 emits `LOAD_FAST_BORROW`, which pushes a BORROWED reference, so a local
# ndarray still has refcount 1 while it is an operand. numpy's temp elision
# (`temp_elide.c`) reads refcount 1 as "this is a throwaway temporary" and
# writes the result of a binary op straight into it. So inside a FUNCTION
#
#     R = q * s[:, None]        # R was itself made by a binary op
#     num = (R * W).sum(1)      # <-- this OVERWRITES R with R*W
#     nrm = np.linalg.norm(R, axis=1)     # norm of the wrong array
#
# and it is silent. It does NOT happen at module scope (globals are not
# borrowed) and only for arrays over numpy's 256 KB elision threshold, which is
# exactly why it survives small unit tests and appears on real tensors. It cost
# an hour here: the first cosine diagnostic in this file read 22.0.
#
# The rule for every harness in this tree: an array that must SURVIVE its own
# use as an operand cannot be a bare local in a binary expression. Reduce with
# `np.einsum`/`np.dot` (which never elide) or take a `.copy()`.
def norm_rows(a):
    """Row L2 norms, without handing `a` to a binary op that may eat it."""
    return np.sqrt(np.einsum("rc,rc->r", a, a, dtype=np.float64))


# --------------------------------------------------------------------------
# The exact q4nx GEMV, vectorised.
#
# `q4nx.gemv_reference_bf16` is a python loop over (row, block) -- 8.2M
# iterations for one lm_head pass, ~50 s. Every probe in this file needs one of
# those for ground truth and the probe generator needs ~340 of them per token,
# so the reference is transcribed to numpy here. `--selfcheck` asserts it
# against the original; it is the same arithmetic in the same places, so any
# difference is float64 summation ORDER and nothing else.
# --------------------------------------------------------------------------
def gemv_bf16_fast(act, d, m, codes, chunk=4096):
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


# --------------------------------------------------------------------------
# The coarse tier.
# --------------------------------------------------------------------------
def build_coarse(D, M, CODES, chunk=8192):
    """(D, M, CODES) q4_1 blocks -> (nib uint8 [rows, cols//2], scale f16 [rows]).

    Dequantises in row chunks; the whole [128256, 2048] matrix as float32 is a
    gigabyte and there is no reason to hold it.
    """
    rows, nb = D.shape
    cols = nb * BLK
    unit_scale = np.float32(3.0 / (7.0 * np.sqrt(cols)))
    nib = np.empty((rows, cols // 2), np.uint8)
    scale = np.empty(rows, np.float32)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        W = (D[lo:hi, :, None] * CODES[lo:hi] + M[lo:hi, :, None]).reshape(-1, cols)
        norm = np.sqrt((W.astype(np.float64) ** 2).sum(1)).astype(np.float32)
        inv = np.where(norm > 0, 1.0 / (unit_scale * np.maximum(norm, 1e-30)), 0.0)
        q = np.clip(np.rint(W * inv[:, None]), -7, 7).astype(np.int8)
        u = q.view(np.uint8) & np.uint8(0x0F)
        nib[lo:hi] = u[:, 0::2] | (u[:, 1::2] << 4)
        scale[lo:hi] = norm * unit_scale
    # f16 is the stored width; round through it so host and device agree.
    return nib, scale.astype(np.float16)


def coarse_logits(nib, scale, act, chunk=16384):
    """The coarse GEMV, the arithmetic the NPU kernel has to reproduce.

    The activation is rounded to bf16 because the device's is: the codes go
    into a bf16 MAC. Leaving it in f32 here would make the host model slightly
    better than the thing it is modelling, and the shortlist is exactly where
    that would hide.
    """
    a = rnd(np.asarray(act, np.float32))
    rows = nib.shape[0]
    out = np.empty(rows, np.float32)
    lut = (np.arange(16, dtype=np.int8) << 4 >> 4).astype(np.float32)  # signed nibble
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        b = nib[lo:hi]
        q = np.empty((hi - lo, a.size), np.float32)
        q[:, 0::2] = lut[b & 0x0F]
        q[:, 1::2] = lut[b >> 4]
        out[lo:hi] = (q @ a) * scale[lo:hi].astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Cached artefacts.
# --------------------------------------------------------------------------
def q4nx_container():
    return q4nx.Q4nx(str(Q4NX))


def lmhead_blocks(c=None):
    """(D, M, CODES) for lm_head.weight in CHECKPOINT row order, cached."""
    p = CACHE / "lmhead_blocks.npz"
    if p.exists():
        z = np.load(p)
        return z["D"], z["M"], z["C"]
    c = c or q4nx_container()
    t0 = time.time()
    D, M, C = q4nx.q4nx_tensor_blocks(c, "lm_head.weight", (VOCAB, K_DIM))
    print(f"  decoded lm_head in {time.time() - t0:.1f} s")
    CACHE.mkdir(exist_ok=True)
    np.savez(p, D=D, M=M, C=C)
    return D, M, C


def coarse_tier():
    """(nib, scale), cached."""
    p = CACHE / "lmhead_coarse.npz"
    if p.exists():
        z = np.load(p)
        return z["nib"], z["scale"]
    D, M, C = lmhead_blocks()
    t0 = time.time()
    nib, scale = build_coarse(D, M, C)
    print(f"  built coarse tier in {time.time() - t0:.1f} s")
    np.savez(p, nib=nib, scale=scale)
    return nib, scale


def coarse_bytes(rows=VOCAB, cols=K_DIM):
    return rows * cols // 2 + rows * 2


# --------------------------------------------------------------------------
# Real hidden states.
# --------------------------------------------------------------------------
def make_probes(tokens):
    """Run host_forward's validated layer stack for each token -> (x, argmax).

    `host_forward.layer` reloads every weight from the container on every call;
    here the decode is cached across tokens, which is the only change.
    """
    import host_forward as hf

    c = q4nx_container()
    cache = {}

    def load_linear(cc, name, N, K):
        if name not in cache:
            cache[name] = q4nx.q4nx_tensor_blocks(cc, name, (N, K))
        return cache[name]

    # Only the quantized linears are worth caching; `c.bf16` re-reads a couple
    # of 2048-element norm vectors a layer, which is free.
    orig_ll, orig_ref = hf.load_linear, q4nx.gemv_reference_bf16
    hf.load_linear = load_linear
    q4nx.gemv_reference_bf16 = gemv_bf16_fast
    try:
        emb = c.bf16("model.embed_tokens.weight").astype(np.float32).reshape(-1, K_DIM)
        nw = c.bf16("model.norm.weight").astype(np.float32)[:K_DIM]
        D, M, C = lmhead_blocks(c)
        out = []
        for tok in tokens:
            t0 = time.time()
            x = emb[tok].astype(np.float64).copy()
            for L in range(hf.NLAY):
                x = hf.layer(c, x, L)
            xn = rmsnorm(x.astype(np.float32), nw)
            lg = gemv_bf16_fast(xn, D, M, C)
            out.append((tok, xn.astype(np.float32), int(np.argmax(lg))))
            print(f"  token {tok:6d} -> argmax {out[-1][2]:6d}   ({time.time()-t0:.1f} s)")
        return out
    finally:
        hf.load_linear, q4nx.gemv_reference_bf16 = orig_ll, orig_ref


def probes_path():
    return CACHE / "probes.npz"


def save_probes(probes):
    CACHE.mkdir(exist_ok=True)
    np.savez(probes_path(),
             toks=np.array([p[0] for p in probes]),
             xn=np.stack([p[1] for p in probes]),
             amax=np.array([p[2] for p in probes]))


def load_probes():
    z = np.load(probes_path())
    return list(zip(z["toks"].tolist(), z["xn"], z["amax"].tolist()))


# --------------------------------------------------------------------------
# The two-pass decode.
# --------------------------------------------------------------------------
def twopass_argmax(nib, scale, D, M, C, xn, topk):
    """Coarse shortlist, exact rescore of those K rows, argmax of the rescored."""
    cl = coarse_logits(nib, scale, xn)
    idx = np.argpartition(cl, -topk)[-topk:]
    fine = q4nx.gemv_reference_bf16(xn, D[idx], M[idx], C[idx])
    return int(idx[int(np.argmax(fine))])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true", help="decode lm_head, build the coarse tier")
    p.add_argument("--probes", type=int, default=0, help="generate N real hidden states")
    p.add_argument("--recall", action="store_true", help="the recall@K curve")
    p.add_argument("--selfcheck", action="store_true", help="fast reference vs q4nx's")
    p.add_argument("--seed", type=int, default=0)
    o = p.parse_args()

    if o.selfcheck:
        selfcheck()
    if o.build:
        nib, scale = coarse_tier()
        D, M, C = lmhead_blocks()
        exact = VOCAB * (2 * (K_DIM // BLK) * 2 + K_DIM // 2)   # q4_1 tile payload
        print(f"coarse tier: {nib.shape} nibbles + {scale.shape} f16 scales "
              f"= {coarse_bytes()/1e6:.1f} MB against {exact/1e6:.1f} MB exact "
              f"({100*(1-coarse_bytes()/exact):.1f}% fewer)")
        # Reconstruction quality, per row, on a sample -- not the gate, but a
        # cheap tell that the format was built right.
        rs = np.random.default_rng(0).choice(VOCAB, 512, replace=False)
        W = (D[rs, :, None] * C[rs] + M[rs, :, None]).reshape(len(rs), K_DIM)
        lut = (np.arange(16, dtype=np.int8) << 4 >> 4).astype(np.float32)
        b = nib[rs]
        q = np.empty((len(rs), K_DIM), np.float32)
        q[:, 0::2] = lut[b & 0x0F]
        q[:, 1::2] = lut[b >> 4]
        R = q * scale[rs].astype(np.float32)[:, None]
        cos = np.einsum("rc,rc->r", R, W) / (norm_rows(R) * norm_rows(W))
        clip = (np.abs(q) >= 7).mean()
        print(f"  row cosine vs exact: min {cos.min():.5f} mean {cos.mean():.5f}")
        print(f"  fraction of codes at the +-7 clip: {clip:.4%}")

    if o.probes:
        # Token ids spread over the vocabulary; deterministic.
        rng = np.random.default_rng(o.seed)
        toks = [128000] + sorted(rng.choice(128000, o.probes - 1, replace=False).tolist())
        print(f"generating {len(toks)} probes")
        save_probes(make_probes(toks))
        print(f"saved {probes_path()}")

    if o.recall:
        recall_curve()
    return 0


def selfcheck():
    """The vectorised reference against q4nx's, on real lm_head rows."""
    D, M, C = lmhead_blocks()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(K_DIM).astype(np.float32)
    n = 256
    a = q4nx.gemv_reference_bf16(x, D[:n], M[:n], C[:n])
    b = gemv_bf16_fast(x, D[:n], M[:n], C[:n])
    err = np.abs(a - b).max()
    print(f"selfcheck: fast vs q4nx.gemv_reference_bf16 over {n} real rows, "
          f"max abs diff {err:.3e} (mean |logit| {np.abs(a).mean():.4f})")
    assert err < 1e-9, "vectorised reference is not the same arithmetic"

    # The coarse GEMV against the format read back the long way round: unpack
    # the planar nibbles by hand, sign-extend, scale, dot. If the packing and
    # the GEMV ever disagree about which nibble is which, this is what says so.
    nib, scale = coarse_tier()
    cl = coarse_logits(nib[:n], scale[:n], x)
    ref = np.empty(n, np.float64)
    for r in range(n):
        by = nib[r].astype(np.int16)
        w = np.empty(K_DIM, np.float64)
        w[0::2] = (by & 0x0F)
        w[1::2] = (by >> 4)
        w[w > 7] -= 16
        ref[r] = float(np.dot(w, rnd(x).astype(np.float64))) * float(scale[r])
    rel = np.abs(cl - ref).max() / max(np.abs(ref).max(), 1e-30)
    print(f"selfcheck: coarse GEMV vs a hand unpack over {n} rows, "
          f"max rel diff {rel:.3e}")
    assert rel < 1e-5, "coarse GEMV disagrees with the planar nibble format"


def recall_curve(ks=(1, 2, 4, 8, 16, 32, 64)):
    probes = load_probes()
    nib, scale = coarse_tier()
    D, M, C = lmhead_blocks()
    kmax = max(ks)
    hit = {k: 0 for k in ks}
    two_ok = {k: 0 for k in ks}
    ranks = []
    for tok, xn, amax in probes:
        cl = coarse_logits(nib, scale, xn)
        order = np.argsort(cl)[::-1][:kmax]
        rank = int(np.where(order == amax)[0][0]) + 1 if amax in order else kmax + 1
        ranks.append(rank)
        for k in ks:
            if rank <= k:
                hit[k] += 1
            # The real gate is the two-pass ANSWER, not the shortlist: a miss
            # only matters if the rescored winner differs from the exact one.
            idx = order[:k]
            fine = gemv_bf16_fast(xn, D[idx], M[idx], C[idx])
            if int(idx[int(np.argmax(fine))]) == amax:
                two_ok[k] += 1
    n = len(probes)
    print(f"\nrecall on {n} real hidden states (host_forward, 16 layers, pos 0)")
    print(f"{'K':>5s} {'recall@K':>10s} {'two-pass argmax == exact':>26s}")
    print("-" * 44)
    for k in ks:
        print(f"{k:5d} {hit[k]}/{n} = {hit[k]/n:.3f}   {two_ok[k]}/{n} = {two_ok[k]/n:.3f}")
    bad = [r for r in ranks if r > kmax]
    print(f"\ncoarse rank of the true argmax: max {max(ranks)}, "
          f"median {int(np.median(ranks))}, outside top-{kmax}: {len(bad)}")
    return hit, two_ok, ranks


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# The oq4 tier — Opus Quant W4A4 (`Oq4G256`), the bandwidth probe's format.
# --------------------------------------------------------------------------
# Symmetric signed-INT4, one f16 scale per 256-element GROUP, against the coarse
# tier's one f32 scale per ROW. 4.0625 bits/weight either way to within 1.2%, so
# a device A/B between them isolates tile SHAPE from tile SIZE.
#
# NOT the full oq4++ recipe: hipfire's `quantize_oq4g256` FWHT-rotates the weights
# and calibrates with AWQ clip-search plus LDLQ error feedback. None of that
# changes the byte count or the streaming pattern, which is what the probe
# measures. It changes ACCURACY, so no accuracy claim may be made from this.
OQ4_GROUP = 256


def build_oq4(D, M, CODES, chunk=8192, group=OQ4_GROUP):
    """(D, M, CODES) q4_1 blocks -> (nib uint8 [rows, cols//2], f16 [rows, ng]).

    Chunked for the same reason `build_coarse` is: the whole [128256, 2048]
    matrix as float32 is a gigabyte and there is no reason to hold it.
    """
    rows, nb = D.shape
    cols = nb * BLK
    if cols % group:
        raise ValueError(f"cols {cols} is not a whole number of {group}-groups")
    ng = cols // group
    nib = np.empty((rows, cols // 2), np.uint8)
    gscale = np.empty((rows, ng), np.float32)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        W = (D[lo:hi, :, None] * CODES[lo:hi] + M[lo:hi, :, None]).reshape(-1, cols)
        Wg = W.reshape(-1, ng, group)
        # Symmetric: the scale is set by the group's peak magnitude over 7, and
        # codes land in [-7, 7]. int4 also holds -8, but a symmetric codebook
        # that used it would be asymmetric about zero by one step.
        amax = np.abs(Wg).max(2).astype(np.float32)
        s = amax / np.float32(7.0)
        inv = np.where(s > 0, np.float32(1.0) / np.maximum(s, np.float32(1e-30)), 0.0)
        q = np.clip(np.rint(Wg * inv[:, :, None]), -7, 7).astype(np.int8)
        u = q.reshape(-1, cols).view(np.uint8) & np.uint8(0x0F)
        nib[lo:hi] = u[:, 0::2] | (u[:, 1::2] << 4)
        gscale[lo:hi] = s
    # THE ON-DISK WIDTH IS f16; THE KERNEL TILE'S IS bf16. Both are 2 bytes, so
    # the byte count -- and every bandwidth number derived from it -- is
    # identical, but the exponent layouts are not: packing f16 and reading bf16
    # gave rel error 1.0 and argmax 128000 against 16309, with nothing to
    # complain about because the widths matched. Converting f16 -> bf16 is
    # precisely the job the format's per-arch loader repack exists to do, since
    # AIE2P's native 2-byte float is bf16.
    #
    # Returned as f32 holding bf16-REPRESENTABLE values, so the host reference
    # and the device tile carry the same numbers and a disagreement means the
    # kernel is wrong rather than the rounding.
    #
    # Cost: bf16 keeps 8 mantissa bits against f16's 11, so a group scale is
    # good to ~0.4% instead of ~0.05%. That is a real accuracy question for the
    # port -- f32 scales would fix it at 4.125 b/w instead of 4.0625 -- and it
    # is NOT settled by this probe, which measures bandwidth.
    return nib, q4nx.bf16_to_f32(q4nx.f32_to_bf16(gscale))


def oq4_unpack(nib):
    """[rows, cols//2] packed nibbles -> [rows, cols] int8 in [-7, 7].

    Byte j carries element 2j in its LOW half and 2j+1 in its HIGH half, which
    is what `build_oq4` packs and what the kernel's `int4` load expects. The
    `^8 - 8` is 4-bit two's-complement sign extension.
    """
    rows, half = nib.shape
    out = np.empty((rows, half * 2), np.int8)
    out[:, 0::2] = ((nib & np.uint8(0x0F)).astype(np.int8) ^ 8) - 8
    out[:, 1::2] = ((nib >> 4).astype(np.int8) ^ 8) - 8
    return out


def oq4_logits(nib, gscale, act, group=OQ4_GROUP, chunk=16384):
    """The oq4 GEMV, the arithmetic the NPU kernel has to reproduce.

    out[r] = sum_g scale[r,g] * (Q[r,g] . act[g]) — the group sum is formed
    first and scaled once, exactly as the kernel does it, because scaling each
    product instead would round in a different place.
    """
    rows = nib.shape[0]
    cols = nib.shape[1] * 2
    ng = cols // group
    a = np.asarray(act, np.float32).reshape(ng, group).astype(np.float64)
    out = np.empty(rows, np.float64)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        q = oq4_unpack(nib[lo:hi]).reshape(-1, ng, group).astype(np.float64)
        part = np.einsum("rgk,gk->rg", q, a)
        out[lo:hi] = (part * gscale[lo:hi].astype(np.float64)).sum(1)
    return out


def oq4_tier():
    """(nib, gscale), cached — the oq4 analogue of `coarse_tier`."""
    p = CACHE / "lmhead_oq4.npz"
    if p.exists():
        z = np.load(p)
        return z["nib"], z["gscale"]
    D, M, C = lmhead_blocks()
    t0 = time.time()
    nib, gscale = build_oq4(D, M, C)
    print(f"  built oq4 tier in {time.time() - t0:.1f} s")
    np.savez(p, nib=nib, gscale=gscale)
    return nib, gscale


def oq4_bytes(rows=VOCAB, cols=K_DIM, group=OQ4_GROUP):
    """4 bits a weight plus one f16 scale per group = 130 B per 256 weights."""
    return rows * cols // 2 + rows * (cols // group) * 2


# --------------------------------------------------------------------------
# The oq3 tier — Opus Quant W3A4 (`Oq3G256`), the memory-ceiling probe.
# --------------------------------------------------------------------------
# 3.0625 b/w against oq4's 4.0625: 98 B per 256-group. Codes are [-3, 3] stored
# as 3-bit two's complement in BIT-PLANES -- plane b of a 32-weight sub-block
# has bit i set iff bit b of `q & 7` is set for weight i.
#
# The NPU repack here is PLANE-MAJOR within a group, [8 p0][8 p1][8 p2], where
# the on-disk form interleaves p0,p1,p2 every 12 bytes. Same 96 B either way; 12
# is not a vector stride and this tree has already paid for a misaligned load
# that reads wrong bytes without faulting.
#
# NOT the full oq3 recipe: no FWHT, no clip-search, and critically no SpinQuant.
# codecs.rs is explicit that "3-bit is only viable with the SpinQuant learned
# rotation on top of the FWHT". Bytes are unaffected, which is all this measures.
OQ3_GROUP = 256


def build_oq3(D, M, CODES, chunk=8192, group=OQ3_GROUP):
    """(D, M, CODES) -> (planes uint32 [rows, ng*3*8], f32 [rows, ng] bf16-valued).

    `planes` is already in the kernel's plane-major order, so the packer is a
    reshape and the layout lives in exactly one place.
    """
    rows, nb = D.shape
    cols = nb * BLK
    ng, sub = cols // group, group // 32
    planes = np.empty((rows, ng * 3 * sub), np.uint32)
    gscale = np.empty((rows, ng), np.float32)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        W = (D[lo:hi, :, None] * CODES[lo:hi] + M[lo:hi, :, None]).reshape(-1, cols)
        Wg = W.reshape(-1, ng, group)
        s = (np.abs(Wg).max(2).astype(np.float32)) / np.float32(3.0)
        inv = np.where(s > 0, np.float32(1.0) / np.maximum(s, np.float32(1e-30)), 0.0)
        q = np.clip(np.rint(Wg * inv[:, :, None]), -3, 3).astype(np.int8)
        u = (q.astype(np.uint8) & np.uint8(7)).reshape(-1, ng, sub, 32)
        # bit b of every lane -> one u32 per (group, sub, plane), lane i at bit i
        # Weight i at bit i. The reversed order (31-i) was MEASURED and is
        # worse -- rel 1.52 against 0.28 -- so from_uint32 maps bit i to lane i,
        # as documented nowhere but now established.
        bits = np.arange(32, dtype=np.uint32)
        out = np.empty((hi - lo, ng, 3, sub), np.uint32)
        for b in range(3):
            out[:, :, b, :] = ((((u >> np.uint8(b)) & np.uint8(1)).astype(np.uint32)
                                << bits).sum(3, dtype=np.uint32))
        planes[lo:hi] = out.reshape(hi - lo, -1)
        gscale[lo:hi] = s
    # bf16-valued, for the same reason build_oq4 is: the kernel tile carries bf16
    # and the host reference must see the identical numbers.
    return planes, q4nx.bf16_to_f32(q4nx.f32_to_bf16(gscale))


def oq3_unpack(planes, ng, group=OQ3_GROUP):
    """plane-major u32 -> [rows, cols] int8 in [-3, 3]."""
    rows = planes.shape[0]
    sub = group // 32
    p = planes.reshape(rows, ng, 3, sub)
    bits = np.arange(32, dtype=np.uint32)   # see build_oq3
    u = np.zeros((rows, ng, sub, 32), np.uint8)
    for b in range(3):
        u |= (((p[:, :, b, :, None] >> bits) & np.uint32(1)).astype(np.uint8)
              << np.uint8(b))
    # 3-bit two's complement: u<4 -> u, u>=4 -> u-8
    q = u.astype(np.int8)
    return np.where(q >= 4, q - 8, q).astype(np.int8).reshape(rows, ng * group)


def oq3_logits(planes, gscale, act, group=OQ3_GROUP, chunk=8192):
    """The oq3 GEMV the NPU kernel has to reproduce (scale folded per group)."""
    ng = gscale.shape[1]
    rows = planes.shape[0]
    a = np.asarray(act, np.float32).reshape(ng, group).astype(np.float64)
    out = np.empty(rows, np.float64)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        q = oq3_unpack(planes[lo:hi], ng).reshape(-1, ng, group).astype(np.float64)
        part = np.einsum("rgk,gk->rg", q, a)
        out[lo:hi] = (part * gscale[lo:hi].astype(np.float64)).sum(1)
    return out


def oq3_tier():
    """(planes, gscale), cached."""
    p = CACHE / "lmhead_oq3.npz"
    if p.exists():
        z = np.load(p)
        return z["planes"], z["gscale"]
    D, M, C = lmhead_blocks()
    t0 = time.time()
    planes, gscale = build_oq3(D, M, C)
    print(f"  built oq3 tier in {time.time() - t0:.1f} s")
    np.savez(p, planes=planes, gscale=gscale)
    return planes, gscale
