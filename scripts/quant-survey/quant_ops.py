"""
Production-matched MQ4G256 + FWHT operations.

Mirrors crates/hipfire-quantize/src/main.rs:
  - gen_fwht_signs(seed, n)             at line 430
  - cpu_fwht_256(x, signs1, signs2)     at line 408
  - quantize_mq4g256(f32, signs1, s2)   at line 442

Production uses seeds 42 and 1042 (see main.rs:1530-1531). Any tool that
needs to reproduce hipfire's MQ4 reconstruction error MUST use these
exact seeds. The 2026-05-05 simulation under
docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/quant_recon_error.py
used numpy.random.default_rng(0xCAFEBABE) — that is NOT production-matched.
See docs/investigations/2026-05-06-moe-quant-cliff-survey/INVESTIGATION-LOG.md
for the audit history.
"""

from __future__ import annotations

import numpy as np

GROUP_SIZE = 256
BLOCK_BYTES = 136  # MQ4G256: 4 (scale f32) + 4 (min f32) + 128 (nibbles)
PRODUCTION_SIGNS1_SEED = 42
PRODUCTION_SIGNS2_SEED = 1042


def gen_fwht_signs(seed: int, n: int = GROUP_SIZE) -> np.ndarray:
    """LCG-generated +/-1 sign table matching hipfire-quantize:
       state = state * 1103515245 + 12345 & 0x7fffffff
       sign  = +1 if (state >> 16) & 1 else -1

    Returns float32 array of length n. Uses Python ints with explicit
    masking to avoid NumPy uint32 overflow warnings; output identical to
    the Rust wrapping-multiply implementation in main.rs:430-436.
    """
    state = int(seed) & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    LCG_MUL = 1103515245
    LCG_ADD = 12345
    MASK_31 = 0x7FFFFFFF
    MASK_32 = 0xFFFFFFFF
    for i in range(n):
        state = (((state * LCG_MUL) & MASK_32) + LCG_ADD) & MASK_31
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


def cpu_fwht_256(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard rotation of length 256 with two sign sequences.
       Matches main.rs:408 cpu_fwht_256:
         out = x * signs1
         for stride in 1, 2, 4, ..., 128:
             butterfly(out)
         out *= 0.0625 * signs2     (= 1/16 = 1/sqrt(256))
    """
    assert x.shape[0] == 256
    out = (x * signs1).astype(np.float32, copy=True)
    stride = 1
    while stride < 256:
        i = 0
        while i < 256:
            a = out[i:i + stride].copy()
            b = out[i + stride:i + 2 * stride]
            out[i:i + stride] = a + b
            out[i + stride:i + 2 * stride] = a - b
            i += 2 * stride
        stride <<= 1
    out *= np.float32(0.0625) * signs2
    return out


def inv_fwht_256(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Inverse of cpu_fwht_256. The forward is signs2 * (FWHT_raw(signs1 * x) / 16);
       the inverse is signs1 * (FWHT_raw(signs2 * x) / 16) since FWHT_raw is
       self-inverse up to the scale factor and signs are +/-1.
    """
    assert x.shape[0] == 256
    out = (x * signs2).astype(np.float32, copy=True)
    stride = 1
    while stride < 256:
        i = 0
        while i < 256:
            a = out[i:i + stride].copy()
            b = out[i + stride:i + 2 * stride]
            out[i:i + stride] = a + b
            out[i + stride:i + 2 * stride] = a - b
            i += 2 * stride
        stride <<= 1
    out *= np.float32(0.0625) * signs1
    return out


def quantize_mq4g256_fwht(group: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> bytes:
    """Quantize a single 256-element group as a 136-byte MQ4G256 block.
       Matches main.rs:442 quantize_mq4g256: FWHT, per-group min/max,
       4-bit asymmetric, store (scale_f32, min_f32, 128 nibble-bytes).

       Returns bytes of length 136.
    """
    assert group.shape[0] == 256
    rotated = cpu_fwht_256(group.astype(np.float32, copy=True), signs1, signs2)
    min_val = float(rotated.min())
    max_val = float(rotated.max())
    rng = max_val - min_val
    if rng > 0.0:
        scale = rng / 15.0
        inv_scale = 1.0 / scale
    else:
        scale = 1.0
        inv_scale = 0.0

    block = bytearray(BLOCK_BYTES)
    block[0:4] = np.float32(scale).tobytes()
    block[4:8] = np.float32(min_val).tobytes()

    # 128 bytes packed: byte i has lo nibble = q[2*i], hi nibble = q[2*i+1].
    for i in range(128):
        lo_q = int((rotated[2 * i] - min_val) * inv_scale + 0.5)
        hi_q = int((rotated[2 * i + 1] - min_val) * inv_scale + 0.5)
        if lo_q < 0:
            lo_q = 0
        elif lo_q > 15:
            lo_q = 15
        if hi_q < 0:
            hi_q = 0
        elif hi_q > 15:
            hi_q = 15
        block[8 + i] = (lo_q & 0xF) | ((hi_q & 0xF) << 4)
    return bytes(block)


def dequantize_mq4g256_fwht(block: bytes, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Dequantize a 136-byte MQ4G256 block back to float32[256].
       Reads (scale_f32, min_f32, 128 nibble bytes), reconstructs the
       rotated group, and applies inv_fwht_256 to recover original space.
    """
    assert len(block) == BLOCK_BYTES
    scale = float(np.frombuffer(block[0:4], dtype=np.float32)[0])
    min_val = float(np.frombuffer(block[4:8], dtype=np.float32)[0])
    rotated = np.empty(256, dtype=np.float32)
    for i in range(128):
        byte = block[8 + i]
        lo_q = byte & 0xF
        hi_q = (byte >> 4) & 0xF
        rotated[2 * i] = scale * lo_q + min_val
        rotated[2 * i + 1] = scale * hi_q + min_val
    return inv_fwht_256(rotated, signs1, signs2)


def quantize_then_dequantize_mq4g256_fwht(
    f32_data: np.ndarray,
    signs1: np.ndarray | None = None,
    signs2: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the full MQ4G256-FWHT round-trip to an arbitrary f32 1D array.
       Pads to a multiple of 256 with zeros, processes per group, then trims
       back to original length. Returns float32 array of the same length.

       Default signs1/signs2 are production-matched (seeds 42 / 1042).
    """
    if signs1 is None:
        signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    if signs2 is None:
        signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    n = f32_data.shape[0]
    n_padded = ((n + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    padded = np.zeros(n_padded, dtype=np.float32)
    padded[:n] = f32_data.astype(np.float32, copy=False)

    out = np.empty(n_padded, dtype=np.float32)
    for off in range(0, n_padded, GROUP_SIZE):
        block = quantize_mq4g256_fwht(padded[off:off + GROUP_SIZE], signs1, signs2)
        out[off:off + GROUP_SIZE] = dequantize_mq4g256_fwht(block, signs1, signs2)
    return out[:n]


# ---------------------------------------------------------------------------
# Vectorized batch path — process N groups in parallel via NumPy
#
# The scalar cpu_fwht_256 above is a faithful port of the Rust production
# code, but a Python-level `for off in range(0, n, 256)` loop costs roughly
# 5ms per group. For a 4096x12288 down_proj (197K groups per tensor) that's
# 16 minutes per tensor — infeasible at survey scale. The batch path below
# reshapes (n_groups, 256) and runs the butterfly + sign multiply in
# NumPy ops, giving 100-1000x throughput at identical numerical output.
# ---------------------------------------------------------------------------

def cpu_fwht_256_batch(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Vectorized FWHT on a batch of 256-element groups.

       Args:
         x: float32 array of shape (n_groups, 256). Modified out-of-place.
         signs1, signs2: float32 arrays of shape (256,).

       Returns: float32 array of shape (n_groups, 256), same as
         applying cpu_fwht_256 to each row independently.
    """
    if x.ndim != 2 or x.shape[1] != 256:
        raise ValueError(f"cpu_fwht_256_batch expects (N, 256); got {x.shape}")
    n_groups = x.shape[0]
    out = (x * signs1).astype(np.float32, copy=False)
    stride = 1
    while stride < 256:
        # Reshape so the butterfly runs over the inner pair-of-stride axis.
        # out shape: (n_groups, n_pairs, 2, stride) where 2*stride*n_pairs=256.
        n_pairs = 256 // (2 * stride)
        viewed = out.reshape(n_groups, n_pairs, 2, stride)
        a = viewed[:, :, 0, :]
        b = viewed[:, :, 1, :]
        # Allocate fresh buffer (in-place would alias a/b).
        new_view = np.empty_like(viewed)
        new_view[:, :, 0, :] = a + b
        new_view[:, :, 1, :] = a - b
        out = new_view.reshape(n_groups, 256)
        stride <<= 1
    out *= np.float32(0.0625) * signs2
    return out


def inv_fwht_256_batch(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Vectorized inverse FWHT on a batch of 256-element groups."""
    if x.ndim != 2 or x.shape[1] != 256:
        raise ValueError(f"inv_fwht_256_batch expects (N, 256); got {x.shape}")
    n_groups = x.shape[0]
    out = (x * signs2).astype(np.float32, copy=False)
    stride = 1
    while stride < 256:
        n_pairs = 256 // (2 * stride)
        viewed = out.reshape(n_groups, n_pairs, 2, stride)
        a = viewed[:, :, 0, :]
        b = viewed[:, :, 1, :]
        new_view = np.empty_like(viewed)
        new_view[:, :, 0, :] = a + b
        new_view[:, :, 1, :] = a - b
        out = new_view.reshape(n_groups, 256)
        stride <<= 1
    out *= np.float32(0.0625) * signs1
    return out


def quantize_then_dequantize_mq4g256_fwht_vectorized(
    f32_data: np.ndarray,
    signs1: np.ndarray | None = None,
    signs2: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized full round-trip: FWHT batch + per-group min/max + 4-bit
       quant + dequant + inv-FWHT batch. Same numerical output as the
       scalar version (modulo float32 rounding-direction ties at the +0.5
       quant cast, which agree at the 1e-6 level).
    """
    if signs1 is None:
        signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    if signs2 is None:
        signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    n = f32_data.shape[0]
    n_padded = ((n + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    n_groups = n_padded // GROUP_SIZE

    padded = np.zeros(n_padded, dtype=np.float32)
    padded[:n] = f32_data.astype(np.float32, copy=False)
    grouped = padded.reshape(n_groups, GROUP_SIZE)

    rotated = cpu_fwht_256_batch(grouped, signs1, signs2)

    # Per-group min/max -> per-group scale + zero. Shapes: (n_groups,)
    grp_min = rotated.min(axis=1)
    grp_max = rotated.max(axis=1)
    grp_range = grp_max - grp_min
    safe = grp_range > 0
    grp_scale = np.where(safe, grp_range / np.float32(15.0), np.float32(1.0))
    grp_inv_scale = np.where(safe, np.float32(1.0) / grp_scale, np.float32(0.0))

    # Quantize: q = round((rotated - min) * inv_scale), clamped to [0, 15].
    centered = rotated - grp_min[:, None]
    q = np.round(centered * grp_inv_scale[:, None] + np.float32(1e-6)).astype(np.int32)
    np.clip(q, 0, 15, out=q)

    # Dequant in rotated space, then inverse FWHT.
    deq = q.astype(np.float32) * grp_scale[:, None] + grp_min[:, None]
    recon = inv_fwht_256_batch(deq, signs1, signs2)

    return recon.reshape(n_padded)[:n]


def quantize_then_dequantize_mq4g256_fwht_2d(
    f32_data_2d: np.ndarray,
    signs1: np.ndarray | None = None,
    signs2: np.ndarray | None = None,
) -> np.ndarray:
    """2D-vectorized MQ4G256+FWHT round-trip. Treats every row's columns
    as independent groups of 256 — same layout as
    quantize_then_dequantize_mq4g256_fwht_vectorized but processes ALL
    rows in a single batched FWHT call. Critical for Phase 2 weight
    mutation walltime: the per-row Python loop costs tens of minutes
    per A3B model; this version runs in seconds.

    Requires f32_data_2d.shape[1] % 256 == 0. Returns same shape.
    """
    if signs1 is None:
        signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    if signs2 is None:
        signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    n_rows, n_cols = f32_data_2d.shape
    if n_cols % GROUP_SIZE != 0:
        # Fall back to per-row path (handles padding cleanly).
        out = np.empty_like(f32_data_2d, dtype=np.float32)
        for r in range(n_rows):
            out[r] = quantize_then_dequantize_mq4g256_fwht_vectorized(
                f32_data_2d[r], signs1, signs2)
        return out

    n_groups_per_row = n_cols // GROUP_SIZE
    n_total_groups = n_rows * n_groups_per_row

    # Reshape into [n_total_groups, GROUP_SIZE] for batched FWHT.
    grouped = f32_data_2d.reshape(n_total_groups, GROUP_SIZE).astype(np.float32, copy=False)
    rotated = cpu_fwht_256_batch(grouped, signs1, signs2)

    grp_min = rotated.min(axis=1)
    grp_max = rotated.max(axis=1)
    grp_range = grp_max - grp_min
    safe = grp_range > 0
    grp_scale = np.where(safe, grp_range / np.float32(15.0), np.float32(1.0))
    grp_inv_scale = np.where(safe, np.float32(1.0) / grp_scale, np.float32(0.0))

    centered = rotated - grp_min[:, None]
    q = np.round(centered * grp_inv_scale[:, None] + np.float32(1e-6)).astype(np.int32)
    np.clip(q, 0, 15, out=q)

    deq = q.astype(np.float32) * grp_scale[:, None] + grp_min[:, None]
    recon = inv_fwht_256_batch(deq, signs1, signs2)
    return recon.reshape(n_rows, n_cols)


def quantize_then_dequantize_q8g256_2d(f32_data_2d: np.ndarray) -> np.ndarray:
    """2D-vectorized Q8G256 round-trip. No FWHT. Treats every row's
    columns as independent groups of 256. Returns same shape.
    Symmetric with quantize_then_dequantize_mq4g256_fwht_2d.
    """
    n_rows, n_cols = f32_data_2d.shape
    if n_cols % GROUP_SIZE != 0:
        out = np.empty_like(f32_data_2d, dtype=np.float32)
        for r in range(n_rows):
            out[r] = quantize_then_dequantize_q8g256(f32_data_2d[r])
        return out

    n_groups_per_row = n_cols // GROUP_SIZE
    n_total_groups = n_rows * n_groups_per_row
    grouped = f32_data_2d.reshape(n_total_groups, GROUP_SIZE).astype(np.float32, copy=False)

    grp_min = grouped.min(axis=1)
    grp_max = grouped.max(axis=1)
    grp_range = grp_max - grp_min
    safe = grp_range > 0
    grp_scale = np.where(safe, grp_range / np.float32(255.0), np.float32(1.0))
    grp_inv_scale = np.where(safe, np.float32(1.0) / grp_scale, np.float32(0.0))

    centered = grouped - grp_min[:, None]
    q = np.round(centered * grp_inv_scale[:, None] + np.float32(1e-6)).astype(np.int32)
    np.clip(q, 0, 255, out=q)

    deq = q.astype(np.float32) * grp_scale[:, None] + grp_min[:, None]
    return deq.reshape(n_rows, n_cols)


def quantize_then_dequantize_q8g256(f32_data: np.ndarray) -> np.ndarray:
    """Vectorized Q8G256 round-trip: 8-bit per element in 256-element groups,
    per-group min/max scale. NO FWHT (8-bit headroom is sufficient — the
    rotation was needed only for the 4-bit budget).

    This is the All-Q8 "ceiling" precision used in Phase 2 ablation. A
    weight tensor round-tripped through this function has the noise floor
    set by 8-bit quant; on bf16 reference weights the resulting NRMSE is
    typically ~0.001-0.005 — close enough to bf16 reference that
    PPL_All-Q8 lands within ε of PPL_bf16.

    Symmetric API with quantize_then_dequantize_mq4g256_fwht_vectorized
    so both can be used interchangeably as round-trip operators in the
    Phase 2 weight-replacement loop.
    """
    n = f32_data.shape[0]
    n_padded = ((n + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    n_groups = n_padded // GROUP_SIZE

    padded = np.zeros(n_padded, dtype=np.float32)
    padded[:n] = f32_data.astype(np.float32, copy=False)
    grouped = padded.reshape(n_groups, GROUP_SIZE)

    grp_min = grouped.min(axis=1)
    grp_max = grouped.max(axis=1)
    grp_range = grp_max - grp_min
    safe = grp_range > 0
    grp_scale = np.where(safe, grp_range / np.float32(255.0), np.float32(1.0))
    grp_inv_scale = np.where(safe, np.float32(1.0) / grp_scale, np.float32(0.0))

    centered = grouped - grp_min[:, None]
    q = np.round(centered * grp_inv_scale[:, None] + np.float32(1e-6)).astype(np.int32)
    np.clip(q, 0, 255, out=q)

    deq = q.astype(np.float32) * grp_scale[:, None] + grp_min[:, None]
    return deq.reshape(n_padded)[:n]


def _benchmark_vectorized() -> int:
    """Compare scalar vs vectorized on a moderately large tensor."""
    import time
    rng = np.random.default_rng(0)
    n = 4096 * 1408  # one A3B expert down_proj
    x = rng.standard_normal(n).astype(np.float32) * 0.05

    s1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    s2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    t0 = time.time()
    rec_scalar = quantize_then_dequantize_mq4g256_fwht(x[:65536], s1, s2)
    dt_scalar_64k = time.time() - t0
    extrap_scalar = dt_scalar_64k * (n / 65536)

    t0 = time.time()
    rec_vec = quantize_then_dequantize_mq4g256_fwht_vectorized(x, s1, s2)
    dt_vec = time.time() - t0

    cos = mean_cosine_similarity(rec_scalar, quantize_then_dequantize_mq4g256_fwht_vectorized(x[:65536], s1, s2))
    print(f"[bench] scalar 64k took {dt_scalar_64k*1000:.0f}ms, extrap to {n} = {extrap_scalar:.1f}s")
    print(f"[bench] vectorized {n} took {dt_vec*1000:.0f}ms")
    print(f"[bench] speedup = {extrap_scalar/dt_vec:.0f}x")
    print(f"[bench] cos sim scalar vs vec on 64k: {cos:.6f}")
    return 0


def nrmse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Explicit NRMSE definition per
       docs/investigations/2026-05-06-moe-quant-cliff-survey/01-survey-runner-design.md D1:

           NRMSE = sqrt(MSE) / sqrt(var(reference))

       where MSE = mean((reference - reconstructed)^2)
             var = mean((reference - mean(reference))^2)

       Lower is better; 0 is perfect reconstruction. Returns float.
    """
    diff = reference.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return 0.0
    centered = reference.astype(np.float64) - float(np.mean(reference))
    var = float(np.mean(centered * centered))
    if var <= 0.0:
        return float("inf")
    return float(np.sqrt(mse) / np.sqrt(var))


def mean_cosine_similarity(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Cosine similarity over flattened arrays. Matches the 2026-05-05
       quant_recon_error.py 'cos' metric for cross-tool comparison.

       Returns float in [-1.0, 1.0]; 1.0 is perfect alignment.
    """
    a = reference.astype(np.float64).ravel()
    b = reconstructed.astype(np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------
# Self-test: run as `python quant_ops.py` to verify production parity.
# ----------------------------------------------------------------------

def _self_test() -> int:
    """Sanity checks: signs are deterministic, FWHT is its own inverse up
       to sign placement, MQ4 round-trip yields >= 0.99 cos sim on random
       Gaussian data with one outlier (the design's stated direction).
    """
    rng = np.random.default_rng(0)
    s1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    s2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)
    assert s1.shape == (256,) and s1.dtype == np.float32
    assert set(np.unique(s1).tolist()) == {-1.0, 1.0}
    assert s1.sum() != 0  # vanishingly unlikely for random sequence

    # Round-trip identity (FWHT . inv_FWHT = identity).
    x = rng.standard_normal(256).astype(np.float32)
    rotated = cpu_fwht_256(x, s1, s2)
    recovered = inv_fwht_256(rotated, s1, s2)
    err = float(np.max(np.abs(x - recovered)))
    assert err < 1e-4, f"FWHT round-trip max error too large: {err}"

    # MQ4 round-trip on a row with one big outlier among small values.
    row = rng.standard_normal(256).astype(np.float32) * 0.05
    row[42] = 0.8  # one outlier ~16x the bulk
    block = quantize_mq4g256_fwht(row, s1, s2)
    recon = dequantize_mq4g256_fwht(block, s1, s2)
    cos = mean_cosine_similarity(row, recon)
    nrm = nrmse(row, recon)
    print(f"[self-test] FWHT round-trip max err: {err:.3e}")
    print(f"[self-test] MQ4 round-trip cos sim:  {cos:.6f}")
    print(f"[self-test] MQ4 round-trip NRMSE:    {nrm:.6f}")
    assert cos >= 0.99, f"MQ4 round-trip cos sim too low: {cos}"
    print("[self-test] PASS")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(_benchmark_vectorized())
    sys.exit(_self_test())
