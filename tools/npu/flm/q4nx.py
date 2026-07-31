#!/usr/bin/env python3
"""Reader for FLM's `model.q4nx` weight container, and the q4_1 reference math.

The container is a plain safetensors file. Every streamed weight tensor is
declared `I8` with a second dimension of **5120 bytes**, and each 5120-byte row
is *planar*, not an array of interleaved blocks:

    [   0: 512]  256 x bf16  d   scales, all positive
    [ 512:1024]  256 x bf16  m   mins,   all negative
    [1024:5120]  4096 B          packed 4-bit codes, 8192 of them

512 + 512 + 4096 = 5120 B for 8192 weights = **exactly 5.00 bits/weight**, which
is the bpw figure derived independently in `docs/npu/flm-layer-dataflow.md` §3.
8192 weights is four output rows at K=2048 or one at K=8192, which is why both
tensor families land on the same second dimension.

`m/d` is -7.4 to -7.5 with std 1.35 across every tensor and `m + 7.5d` centres
on zero, the signature of plain min/max asymmetric q4_1 (`d=(max-min)/15`,
`m=min`). Per-block scale spread matches K-major blocking (`q_proj` p99/p1 7.18
stored against 7.01 K-major and 31.77 N-major), so a block is **32 contiguous
input dims**.

**NOT established, and it gates end-to-end comparison against FLM**: which
`(output row, k-block)` each of the 256 slots in a row maps to. FLM's weights
are not a quantization of the published checkpoint — see the 2026-07-31 log
entry — so the mapping could not be recovered by matching against known
weights. The arithmetic below is exact for a given mapping; the indexing is not
yet pinned.
"""

import json
import struct

import numpy as np

ROW_BYTES = 5120
BLK = 32                       # weights per q4_1 block
BLOCKS_PER_ROW = 256
DM_BYTES = BLOCKS_PER_ROW * 2  # bf16
TILE_TRAILER = 64              # per-tile trailer: [f32 row_base][f32 flags][pad]


def tile_bytes(K, NROWS, trailer=True):
    """Bytes in one weight tile, including the trailer.

    [NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 codes][TILE_TRAILER]

    The trailer is 64 B and universal — every phase of the fused layer uses the
    same tile shape, which is what lets one operand fifo serve all of them. It
    costs +0.31% of weight traffic at K=2048/NROWS=16 and carries `row_base`,
    the global output-row index of the tile's first row. That one value replaces
    every per-core index the kernels would otherwise need (residual indexing,
    RoPE head identity, down-chunk accumulator slot) with no runtime scalar
    arguments and no static cursors. 64 also keeps the tile a multiple of 64, so
    both halves of a double-buffered fifo stay aligned — see the alternating
    even/odd tile corruption recorded in the log.
    """
    payload = 2 * NROWS * (K // BLK) * 2 + NROWS * (K // 2)
    return payload + (TILE_TRAILER if trailer else 0)


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    """round-to-nearest-even float32 -> bf16 bit pattern"""
    u = np.asarray(x, np.float32).view(np.uint32)
    r = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return (((u + r) >> 16) & np.uint32(0xFFFF)).astype(np.uint16)


class Q4nx:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def raw(self, name):
        t = self.header[name]
        a, b = t["data_offsets"]
        self.f.seek(self.base + a)
        return t, np.frombuffer(self.f.read(b - a), dtype=np.uint8)

    def rows(self, name):
        """-> (nrows, 5120) uint8"""
        t, raw = self.raw(name)
        return raw.reshape(t["shape"][0], ROW_BYTES)

    def bf16(self, name):
        t, raw = self.raw(name)
        return bf16_to_f32(raw.view(np.uint16)).reshape(t["shape"])

    def blocks(self, name):
        """-> d (nrows,256) f32, m (nrows,256) f32, codes (nrows,256,32) uint8"""
        r = self.rows(name)
        d = bf16_to_f32(r[:, 0:DM_BYTES].copy().view(np.uint16)).reshape(-1, BLOCKS_PER_ROW)
        m = bf16_to_f32(r[:, DM_BYTES:2 * DM_BYTES].copy().view(np.uint16)).reshape(-1, BLOCKS_PER_ROW)
        qs = r[:, 2 * DM_BYTES:].reshape(-1, BLOCKS_PER_ROW, BLK // 2)
        # Nibble order WITHIN one of FLM's blocks is not established. It was
        # once thought unestablishable for want of a reference; the running
        # `flm serve` is one (deterministic at temp 0, and the shipped
        # tokenizer reproduces its prompt tokens exactly), so a candidate can
        # now be judged end to end. Note the layout is NOT where the current
        # mismatch lives: the container's Frobenius norm exceeds the
        # checkpoint's by 1.16x, and permutation preserves that norm exactly. Split order is
        # assumed here so the codes are read consistently; every downstream use
        # repacks them anyway, so the choice cancels out.
        codes = np.concatenate([qs & 0x0F, qs >> 4], axis=2)
        return d, m, codes


def pack_tile(d, m, codes, row_base=0, flags=0.0, trailer=True):
    """Pack NROWS output rows into the kernel's tile layout.

    [NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 bytes], the same planar shape
    FLM uses per 5120-byte row, scaled to whatever NROWS/K the tile carries.

    d, m: (NROWS, NB) float32.  codes: (NROWS, NB, 32) uint8, values 0..15.
    """
    nrows, nb = d.shape
    assert codes.shape == (nrows, nb, BLK)
    assert nb % 2 == 0
    # Plain element order: byte j carries element 2j in its low nibble and
    # element 2j+1 in its high nibble. That is the layout a native uint4 vector
    # load expects, so the kernel gets the widening for free from the hardware
    # (`vldb.unpack` + `vups.4x`) instead of masking uint8 lanes.
    flat = codes.reshape(nrows, nb * BLK)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)
    parts = [f32_to_bf16(d).ravel().view(np.uint8),
             f32_to_bf16(m).ravel().view(np.uint8),
             packed.ravel()]
    if trailer:
        tr = np.zeros(TILE_TRAILER, np.uint8)
        tr[0:8].view(np.float32)[:] = (np.float32(row_base), np.float32(flags))
        parts.append(tr)
    return np.concatenate(parts)


def gemv_reference_bf16(act, d, m, codes):
    """The same GEMV, rounded exactly where the kernel rounds.

    The float64 reference is the wrong gate for correctness: bf16 operands are
    inherent to the format (FLM materialises `w = d*q + m` in bf16 before its
    own MAC), so a body that is exactly right still lands ~1% away on an output
    with heavy cancellation. This reproduces the kernel step for step — the
    activation scaled by the block's bf16 scale and rounded to bf16, the block
    sums rounded to bf16, everything accumulated in float — so a mismatch here
    is a defect.
    """
    nrows, nb = d.shape
    a = np.asarray(act, np.float32).reshape(nb, BLK)
    rnd = lambda x: bf16_to_f32(f32_to_bf16(x))

    asum = rnd(a.astype(np.float64).sum(1).astype(np.float32))
    d = rnd(d)
    out = np.zeros(nrows, np.float64)
    for r in range(nrows):
        acc = 0.0
        for b in range(nb):
            a_s = rnd(a[b] * d[r, b])
            acc += float(np.dot(codes[r, b].astype(np.float64),
                                a_s.astype(np.float64)))
        out[r] = acc + float(np.dot(rnd(m[r]).astype(np.float64),
                                    asum.astype(np.float64)))
    return out


def gemv_reference(act, d, m, codes):
    """out[r] = sum_b ( d[r,b] * sum_t codes[r,b,t]*a[b,t] + m[r,b] * sum_t a[b,t] )

    The identity the kernel uses: the zero-point term collapses to one scalar
    per block against an activation block-sum shared by every output row, so no
    weight is ever materialised. Accumulated in float64 here so the reference
    is not itself the thing under test.
    """
    nrows, nb = d.shape
    a = np.asarray(act, np.float64).reshape(nb, BLK)
    asum = a.sum(1)
    dot = np.einsum("rbt,bt->rb", codes.astype(np.float64), a)
    return (d.astype(np.float64) * dot + m.astype(np.float64) * asum).sum(1)


def flag_tag(flags):
    """A short token that makes a `compile_flags` set visible to iron.jit's cache.

    **iron.jit keys on the design's code object.** A `-D` value that reaches the
    kernel only through a runtime-built `compile_flags` list is invisible to
    that key, so two variants whose sources are byte-identical collide and the
    second silently runs the first's kernels. Nothing looks stale — the flag is
    right there in the file being read — and it has cost four debugging runs in
    this tree, twice producing a confidently wrong conclusion.

    Interpolate the tag into something inside the generated design source, most
    simply a fifo name:

        TAG = q4nx.flag_tag(flags)
        ...
        f_w = ObjectFifo(ty, name=f"wp{{i}}_{TAG}")

    Any value that varies with the flags works; this just makes it short and
    automatic rather than something each harness invents.
    """
    import hashlib
    return hashlib.sha1("|".join(map(str, flags)).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# The q4nx row layout, solved against FLM's own decoder and verified bit-exact
# (3/3 random payload+scale trials, maxdiff 0.0). See q4nx_contract_probe.py.
#
# A 5120-byte row carries 8192 elements as:
#   [0:512]     256 bf16 scales
#   [512:1024]  256 bf16 zero-points
#   [1024:5120] 4096 bytes of 4-bit codes
#
# BOTH the scales and the codes are 8-way transposed, which is what the library
# constant `group_size_bytes = 40960 = 8 * 5120` was pointing at. The decode is
#
#   w[i] = scale[block(i)] * (code[i] - zero[block(i)])
#
# i.e. a ZERO-POINT form, not llama.cpp's `d*q + m`. Reading region 1 as a min
# is what made every earlier arrangement fail: m = -d*z, so treating z as m is
# wrong by a factor of d.

def q4nx_maps():
    """-> (lo, hi, sidx): byte->element maps and the block->scale-slot map."""
    c = np.arange(4096)
    lo = 4096 * (c // 2048) + 512 * (c % 8) + ((c % 2048) // 8)
    hi = lo + 256
    b = np.arange(BLOCKS_PER_ROW)
    sidx = 32 * (b % 8) + (b // 8)
    return lo, hi, sidx


def q4nx_decode_row(row, zero_point=False):
    """One 5120-byte row -> (8192,) float32.

    zero_point=False gives the WEIGHTS, q4_1 `d*q + m` -- verified against the
    real Llama-3.2-1B-Instruct checkpoint at corr 0.99700, relative Frobenius
    error 0.0776 (i.e. 4-bit quantization error and nothing else).

    zero_point=True gives `d*(q - m)`, which is what FLM's own
    `Q4NX::q4nx_dequantize` returns. That is NOT the weight -- it is all
    non-negative -- and mistaking it for one cost a full retraction.
    """
    lo, hi, sidx = q4nx_maps()
    d = bf16_to_f32(row[0:512].copy().view(np.uint16))[sidx]
    z = bf16_to_f32(row[512:1024].copy().view(np.uint16))[sidx]
    by = row[1024:ROW_BYTES]
    q = np.empty(8192, np.float32)
    q[lo] = by & 0x0F
    q[hi] = by >> 4
    if zero_point:
        return d.repeat(BLK) * (q - z.repeat(BLK))
    return d.repeat(BLK) * q + z.repeat(BLK)


def q4nx_decode_tensor(c, name, out_shape):
    """Decode a whole quantized tensor to its true (rows, cols) layout.

    Container row `cr` is one 32x256 TILE: row-group `cr // 8`, column-group
    `cr % 8`. Within a tile, block b covers checkpoint

        row  = 64*(g//2) + 2*(b//8) + (g%2)        # rows interleave by 2
        cols = 32*(8*(cr%8) + b%8) .. +32

    The stride-2 row interleave and the 8-way scale/code transposes are all the
    same 8-lane structure; `group_size_bytes = 8 * ROW_BYTES` names it.
    """
    R, C = out_shape
    rows = c.rows(name)
    out = np.zeros((R, C), np.float32)
    b = np.arange(BLOCKS_PER_ROW)
    col = np.arange(BLK)
    for cr in range(rows.shape[0]):
        W = q4nx_decode_row(rows[cr]).reshape(BLOCKS_PER_ROW, BLK)
        g, cg = cr // 8, cr % 8
        r = 64 * (g // 2) + 2 * (b // 8) + (g % 2)
        k = 8 * cg + (b % 8)
        out[r[:, None], k[:, None] * BLK + col[None, :]] = W
    return out
