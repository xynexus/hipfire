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
RESIDUAL_BYTES = 64            # fixed, keeps the tile a multiple of 64


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
        # Nibble order WITHIN one of FLM's blocks is not established -- it
        # cannot be, while the block-to-(row, k) mapping is unknown, since
        # there is nothing to check a candidate order against. Split order is
        # assumed here so the codes are read consistently; every downstream use
        # repacks them anyway, so the choice cancels out.
        codes = np.concatenate([qs & 0x0F, qs >> 4], axis=2)
        return d, m, codes


def pack_tile(d, m, codes):
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
    return np.concatenate([
        f32_to_bf16(d).ravel().view(np.uint8),
        f32_to_bf16(m).ravel().view(np.uint8),
        packed.ravel(),
    ])


def pack_tile_residual(d, m, codes, residual):
    """Pack a tile for `flm_gemv_q4_1_residual`: the standard tile with the
    tile's NROWS residual values appended.

    The residual rides inside the weight tile because a core tile has only two
    input DMA channels and the activation and weight streams already use both.
    It costs a fixed RESIDUAL_BYTES per tile (0.3% at NROWS=16, K=2048) and
    removes a whole 92.9 us dispatch per residual add.

    The region is a fixed 64 bytes rather than nrows*2 so the tile stays a
    multiple of 64. At 20512 B the double-buffered ObjectFifo puts buffer 1 on a
    32-byte boundary and the vectorised residual load off it reads garbage —
    with the exact signature of EVEN tiles correct and ODD tiles wrong, because
    the fifo alternates buffers.
    """
    nrows = d.shape[0]
    assert residual.shape == (nrows,), residual.shape
    assert nrows * 2 <= RESIDUAL_BYTES
    pad = np.zeros(RESIDUAL_BYTES - nrows * 2, np.uint8)
    return np.concatenate([pack_tile(d, m, codes),
                           f32_to_bf16(residual).view(np.uint8), pad])


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
