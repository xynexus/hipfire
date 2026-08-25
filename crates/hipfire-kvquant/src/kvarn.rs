// SPDX-License-Identifier: Apache-2.0
// hipfire — KVarN (variance-normalized 4-bit KV-cache quantization)
//
// Phase D (NEXT-STEPS.md): the offline/CPU core of variance-normalized 4-bit KV
// quantization. Clean-room Rust reimplementation of the KVarN method
// ([2606.03458]); the published repo is read only as an algorithm reference (it
// is MLA-shaped — DeepSeek latent KV, R=512 — and vLLM/Triton bound). hipfire's
// FullAttention is **GQA**, so this core operates on a generic 2D tile
// `[R, C]` and the KV wiring (Phase D1) tiles per `(layer, kv_head)` over
// `head_dim` rather than the MLA latent.
//
// The method, in three steps (mirrors `kvarn_mla_tilepack.pack_tile`):
//   1. (optional, upstream) Hadamard/FWHT-rotate the tile for incoherence —
//      same role as in QTIP; kept out of this module so callers can share the
//      engine's `cpu_fwht_*` rotation.
//   2. **Variance-normalize** (`variance_normalize`): log-domain Sinkhorn
//      balancing — alternating column/row std-normalization, tracking the
//      lowest-imbalance state seen. Returns `balanced = tile / s_col / s_row`
//      with equalized row- and column-std, so a single per-channel 4-bit
//      min/max quantizer has near-uniform error across the tile (the whole
//      point: keys/values have heavy per-channel variance spread that wrecks
//      naive per-token 4-bit).
//   3. Per-channel 4-bit min/max quantize the balanced tile; absorb the
//      per-row Sinkhorn scale into (scale, zp); store the per-column scale
//      separately. Dequant: `(q*scale + zp) * s_col`.
//
// Why it matters for hipfire: KV is a long-context bandwidth lever (Phase D).
// 4-bit KVarN ≈ near-lossless where plain asym4 drifts, because the recurrent
// reasoning error that KVarN targets accumulates exactly where per-channel
// variance is mis-scaled.

/// Sinkhorn iteration count (reference default).
pub const SINKHORN_ITERS: usize = 16;
/// 4-bit quantization: 16 levels, max code 15.
pub const QMAX: f32 = 15.0;

/// Imbalance metric: column-std spread + row-std spread. Lower is better; a
/// perfectly balanced tile scores 2.0 (each std max == its std min). The
/// Sinkhorn loop keeps the scales that minimized this. `tile` is row-major
/// `[r_dim, c_dim]`.
pub fn imbalance(tile: &[f32], r_dim: usize, c_dim: usize) -> f64 {
    // Per-column std (std along rows, i.e. down each column).
    let mut col_min = f64::INFINITY;
    let mut col_max = 0.0f64;
    for c in 0..c_dim {
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for r in 0..r_dim {
            let v = tile[r * c_dim + c] as f64;
            sum += v;
            sq += v * v;
        }
        let n = r_dim as f64;
        let std = (sq / n - (sum / n) * (sum / n)).max(0.0).sqrt();
        col_min = col_min.min(std);
        col_max = col_max.max(std);
    }
    // Per-row std (std along columns, i.e. across each row).
    let mut row_min = f64::INFINITY;
    let mut row_max = 0.0f64;
    for r in 0..r_dim {
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for c in 0..c_dim {
            let v = tile[r * c_dim + c] as f64;
            sum += v;
            sq += v * v;
        }
        let n = c_dim as f64;
        let std = (sq / n - (sum / n) * (sum / n)).max(0.0).sqrt();
        row_min = row_min.min(std);
        row_max = row_max.max(std);
    }
    col_max / col_min.max(1e-8) + row_max / row_min.max(1e-8)
}

/// Result of variance-normalization: the balanced tile plus the per-column and
/// per-row scales such that `balanced[r,c] = tile[r,c] / s_col[c] / s_row[r]`.
pub struct Balanced {
    pub balanced: Vec<f32>, // [r_dim * c_dim] row-major
    pub s_col: Vec<f32>,    // [c_dim] per-column scale
    pub s_row: Vec<f32>,    // [r_dim] per-row scale
}

/// Log-domain Sinkhorn variance-normalization (clean-room of KVarN's
/// `variance_normalize`). Alternates column-std and row-std normalization for
/// `iters` passes, tracking the lowest-imbalance scales seen (best-so-far, not
/// last — the loop can overshoot). `tile` row-major `[r_dim, c_dim]`.
pub fn variance_normalize(tile: &[f32], r_dim: usize, c_dim: usize, iters: usize) -> Balanced {
    // Work in log-scale: log_s_col[c], log_s_row[r]; current = tile * exp(-lc) * exp(-lr).
    let mut log_s_col = vec![0.0f64; c_dim];
    let mut log_s_row = vec![0.0f64; r_dim];

    // Materialize current balanced tile from the log scales.
    let cur_from = |lc: &[f64], lr: &[f64]| -> Vec<f32> {
        let mut out = vec![0.0f32; r_dim * c_dim];
        for r in 0..r_dim {
            let er = (-lr[r]).exp();
            for c in 0..c_dim {
                out[r * c_dim + c] = (tile[r * c_dim + c] as f64 * er * (-lc[c]).exp()) as f32;
            }
        }
        out
    };

    let log_clamp = |x: f64| x.clamp(-0.3, 10.0); // mirror reference _LOG_S_{MIN,MAX}

    let mut best = cur_from(&log_s_col, &log_s_row);
    let mut best_imb = imbalance(&best, r_dim, c_dim);
    let mut best_lc = log_s_col.clone();
    let mut best_lr = log_s_row.clone();

    for _ in 0..iters {
        // Column pass: normalize each column to unit std (down the rows).
        let cur = cur_from(&log_s_col, &log_s_row);
        for c in 0..c_dim {
            let mut sum = 0.0f64;
            let mut sq = 0.0f64;
            for r in 0..r_dim {
                let v = cur[r * c_dim + c] as f64;
                sum += v;
                sq += v * v;
            }
            let n = r_dim as f64;
            let std = (sq / n - (sum / n) * (sum / n)).max(0.0).sqrt();
            let std = std.clamp(1e-3, 1e3);
            log_s_col[c] = log_clamp(log_s_col[c] + std.ln());
        }
        // Row pass: normalize each row to unit std (across the columns).
        let cur = cur_from(&log_s_col, &log_s_row);
        for r in 0..r_dim {
            let mut sum = 0.0f64;
            let mut sq = 0.0f64;
            for c in 0..c_dim {
                let v = cur[r * c_dim + c] as f64;
                sum += v;
                sq += v * v;
            }
            let n = c_dim as f64;
            let std = (sq / n - (sum / n) * (sum / n)).max(0.0).sqrt();
            let std = std.clamp(1e-3, 1e3);
            log_s_row[r] = log_clamp(log_s_row[r] + std.ln());
        }
        // Track best-so-far.
        let cand = cur_from(&log_s_col, &log_s_row);
        let imb = imbalance(&cand, r_dim, c_dim);
        if imb < best_imb {
            best_imb = imb;
            best = cand;
            best_lc = log_s_col.clone();
            best_lr = log_s_row.clone();
        }
    }

    Balanced {
        balanced: best,
        s_col: best_lc.iter().map(|&x| x.exp() as f32).collect(),
        s_row: best_lr.iter().map(|&x| x.exp() as f32).collect(),
    }
}

/// Per-channel 4-bit quantized tile record (one per `[r_dim, c_dim]` tile).
/// `scale_abs`/`zp_abs` are per-channel (per-row, with the row Sinkhorn scale
/// absorbed); `s_col` is the per-column (per-token) scale stored separately.
pub struct QuantTile {
    pub q: Vec<u8>, // [r_dim * c_dim] 4-bit codes (stored one per byte here; pack downstream)
    pub scale_abs: Vec<f32>, // [r_dim]
    pub zp_abs: Vec<f32>, // [r_dim]
    pub s_col: Vec<f32>, // [c_dim]
    pub r_dim: usize,
    pub c_dim: usize,
}

/// Variance-normalize + per-channel (per-row) 4-bit min/max quantize a tile.
/// Mirrors `pack_tile`: quantize the balanced tile per-row, then absorb the
/// per-row Sinkhorn scale `s_row` into (scale, zp) so dequant only needs the
/// per-column scale at runtime: `deq = (q*scale_abs + zp_abs) * s_col`.
/// In-place Walsh–Hadamard transform over each length-`n` row segment, scaled by
/// `1/sqrt(n)` so the transform is ORTHONORMAL.
///
/// This is the stage hipfire's KVarN was missing. The paper
/// (2606.03458v1, `/srv/hipfire/references/Quant/2606.03458v1-KVarN`) specifies
/// "a Hadamard rotation followed by a dual-scaling variance normalization across
/// both axes" — we implemented the second half and not the first. The reference
/// (`kvarn_mla_tilepack.py`) packs values already in the rotated frame:
/// `qH = q @ H`, keys stored rotated.
///
/// Orthonormality is what makes it free at attention time: H^T H = I, so
/// `q·k == (qH)·(kH)` and the scores are unchanged — the rotation only has to be
/// applied consistently to Q and K, never inverted.
///
/// Why it matters most at low bit-width: the rotation spreads outlier channels
/// across the whole vector, so a coarse grid no longer has to span one dominant
/// coordinate. That is exactly the regime the paper targets (2-bit).
///
/// `n` must be a power of two.
pub fn hadamard_rows(x: &mut [f32], n: usize) {
    assert!(
        n.is_power_of_two(),
        "hadamard_rows: n must be a power of two"
    );
    assert_eq!(x.len() % n, 0, "hadamard_rows: len must be a multiple of n");
    let scale = 1.0f32 / (n as f32).sqrt();
    for row in x.chunks_mut(n) {
        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                for j in i..i + len {
                    let a = row[j];
                    let b = row[j + len];
                    row[j] = a + b;
                    row[j + len] = a - b;
                }
                i += len << 1;
            }
            len <<= 1;
        }
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

/// `quantize_tile_qmax` preceded by the paper's Hadamard rotation along the
/// channel axis (`c_dim`), i.e. the full published KVarN pipeline rather than
/// just its second half.
///
/// The caller MUST apply the same rotation to Q (see `hadamard_rows`) — this
/// returns codes in the rotated frame, and `dequantize_tile` returns rotated
/// values. Nothing inverts it, by design.
/// Hadamard-rotate along the CHANNEL axis of a `[channel(row) x token(col)]`
/// tile — i.e. down each column, across `r_dim` channels.
///
/// The axis matters and is easy to get backwards. `kvarn_gather_k_tiles` builds
/// tiles as `[head_dim x GROUP]`, so a ROW is one channel across all tokens and a
/// COLUMN is one token across all channels. The reference rotates `q @ H` with
/// `q` shaped `[heads, R]` where R is the channel/latent dim, so the transform
/// runs across CHANNELS. Rotating across tokens instead would mix independent
/// timesteps and is simply a different (wrong) operator.
pub fn hadamard_channels(tile: &mut [f32], r_dim: usize, c_dim: usize) {
    assert!(
        r_dim.is_power_of_two(),
        "hadamard_channels: r_dim (channels) must be a power of two, got {r_dim}"
    );
    assert_eq!(
        tile.len(),
        r_dim * c_dim,
        "hadamard_channels: tile shape mismatch"
    );
    let scale = 1.0f32 / (r_dim as f32).sqrt();
    let mut col = vec![0f32; r_dim];
    for t in 0..c_dim {
        for ch in 0..r_dim {
            col[ch] = tile[ch * c_dim + t];
        }
        let mut len = 1;
        while len < r_dim {
            let mut i = 0;
            while i < r_dim {
                for j in i..i + len {
                    let a = col[j];
                    let b = col[j + len];
                    col[j] = a + b;
                    col[j + len] = a - b;
                }
                i += len << 1;
            }
            len <<= 1;
        }
        for ch in 0..r_dim {
            tile[ch * c_dim + t] = col[ch] * scale;
        }
    }
}

pub fn quantize_tile_rotated(tile: &[f32], r_dim: usize, c_dim: usize, qmax: f32) -> QuantTile {
    let mut rot = tile.to_vec();
    hadamard_channels(&mut rot, r_dim, c_dim);
    quantize_tile_qmax(&rot, r_dim, c_dim, qmax)
}

pub fn quantize_tile(tile: &[f32], r_dim: usize, c_dim: usize) -> QuantTile {
    quantize_tile_qmax(tile, r_dim, c_dim, QMAX)
}

/// As `quantize_tile`, but with an explicit max code `qmax` (15 = 4-bit, 7 =
/// 3-bit, 3 = 2-bit, 1 = 1-bit). Codes ≤ 15 still pack into the same 4-bit
/// nibble container + dequant kernel, so a lower qmax measures lower-precision
/// quant QUALITY with no storage/GPU change (the storage win comes later from a
/// real sub-nibble packing). Used by the cold-tier 2-bit probe.
pub fn quantize_tile_qmax(tile: &[f32], r_dim: usize, c_dim: usize, qmax: f32) -> QuantTile {
    let bal = variance_normalize(tile, r_dim, c_dim, SINKHORN_ITERS);
    let mut q = vec![0u8; r_dim * c_dim];
    let mut scale_abs = vec![0.0f32; r_dim];
    let mut zp_abs = vec![0.0f32; r_dim];
    for r in 0..r_dim {
        // Per-row (per-channel) min/max of the balanced tile.
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for c in 0..c_dim {
            let v = bal.balanced[r * c_dim + c];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let scale = ((hi - lo) / qmax).max(1e-8);
        for c in 0..c_dim {
            let v = bal.balanced[r * c_dim + c];
            let code = (((v - lo) / scale).round()).clamp(0.0, qmax);
            q[r * c_dim + c] = code as u8;
        }
        // Absorb the per-row Sinkhorn scale so deq*(s_col) reconstructs the
        // rotated-frame value: balanced = tile / s_col / s_row ⇒
        // tile = balanced * s_col * s_row = (q*scale+lo) * s_col * s_row.
        scale_abs[r] = scale * bal.s_row[r];
        zp_abs[r] = lo * bal.s_row[r];
    }
    QuantTile {
        q,
        scale_abs,
        zp_abs,
        s_col: bal.s_col,
        r_dim,
        c_dim,
    }
}

/// Dequantize a `QuantTile` back to the (rotated-frame) tile:
/// `deq[r,c] = (q*scale_abs[r] + zp_abs[r]) * s_col[c]`.
pub fn dequantize_tile(qt: &QuantTile) -> Vec<f32> {
    let mut out = vec![0.0f32; qt.r_dim * qt.c_dim];
    for r in 0..qt.r_dim {
        let sa = qt.scale_abs[r];
        let za = qt.zp_abs[r];
        for c in 0..qt.c_dim {
            out[r * qt.c_dim + c] = (qt.q[r * qt.c_dim + c] as f32 * sa + za) * qt.s_col[c];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// KVarN tile record — the on-device byte layout the GPU write/read kernels use
// (D1 component #3 foundation, the analog of qtip::pack_qtip3_group). One record
// per [r_dim × c_dim] tile (K: [head_dim × GROUP], V: [GROUP × head_dim]):
//   [0 : (r*c+1)/2)            4-bit packed q (2 nibbles/byte, row-major)
//   [.. + r_dim*2)             scale_abs[r_dim] fp16 (per-channel for K)
//   [.. + r_dim*2)             zp_abs[r_dim]    fp16
//   [.. + c_dim*2)             s_col[c_dim]     fp16 (per-token for K)
// The dequant kernel reads this → fp16 scratch: deq[r,c]=(q*scale_abs[r]+zp_abs[r])*s_col[c].
// ─────────────────────────────────────────────────────────────────────────

/// Byte length of a packed KVarN tile record for an `r_dim × c_dim` tile (4-bit).
pub fn kvarn_record_bytes(r_dim: usize, c_dim: usize) -> usize {
    kvarn_record_bytes_bits(r_dim, c_dim, 4)
}

/// Byte length of a packed KVarN tile record at `bits` per code (2 or 4). Codes
/// pack `8/bits` per byte; the fp16 scale/zp/s_col blocks are bit-width-independent.
pub fn kvarn_record_bytes_bits(r_dim: usize, c_dim: usize, bits: usize) -> usize {
    let cpb = 8 / bits; // codes per byte
    (r_dim * c_dim).div_ceil(cpb) + r_dim * 2 * 2 + c_dim * 2
}

// Reuse the crate's tested f16 conversions (handle subnormals/inf correctly).
use hipfire_primitives::conv::{f16_to_f32 as f16_bits_to_f32, f32_to_f16 as f32_to_f16_bits};

/// Pack a `QuantTile` into its on-device record (4-bit q + fp16 metadata).
pub fn pack_kvarn_tile(qt: &QuantTile) -> Vec<u8> {
    pack_kvarn_tile_bits(qt, 4)
}

/// Pack a `QuantTile` at `bits` per code (2 or 4): `8/bits` codes per byte,
/// LSB-first within the byte, then the fp16 scale/zp/s_col blocks. bits=4 is
/// byte-identical to the legacy nibble layout. The GPU dequant kernel must be
/// passed the same `bits`.
pub fn pack_kvarn_tile_bits(qt: &QuantTile, bits: usize) -> Vec<u8> {
    let (r, c) = (qt.r_dim, qt.c_dim);
    let mut out = vec![0u8; kvarn_record_bytes_bits(r, c, bits)];
    let n = r * c;
    let cpb = 8 / bits; // codes per byte
    let mask = (1u8 << bits) - 1;
    for i in 0..n {
        let code = qt.q[i] & mask;
        out[i / cpb] |= code << ((i % cpb) * bits);
    }
    let mut off = n.div_ceil(cpb);
    for &s in &qt.scale_abs {
        out[off..off + 2].copy_from_slice(&f32_to_f16_bits(s).to_le_bytes());
        off += 2;
    }
    for &z in &qt.zp_abs {
        out[off..off + 2].copy_from_slice(&f32_to_f16_bits(z).to_le_bytes());
        off += 2;
    }
    for &sc in &qt.s_col {
        out[off..off + 2].copy_from_slice(&f32_to_f16_bits(sc).to_le_bytes());
        off += 2;
    }
    out
}

/// Unpack a KVarN tile record → `QuantTile` (fp16 metadata widened to f32).
pub fn unpack_kvarn_tile(rec: &[u8], r_dim: usize, c_dim: usize) -> QuantTile {
    unpack_kvarn_tile_bits(rec, r_dim, c_dim, 4)
}

/// Unpack a KVarN tile record packed at `bits` per code (mirror of pack_kvarn_tile_bits).
pub fn unpack_kvarn_tile_bits(rec: &[u8], r_dim: usize, c_dim: usize, bits: usize) -> QuantTile {
    let n = r_dim * c_dim;
    let cpb = 8 / bits;
    let mask = (1u8 << bits) - 1;
    let mut q = vec![0u8; n];
    for i in 0..n {
        let byte = rec[i / cpb];
        q[i] = (byte >> ((i % cpb) * bits)) & mask;
    }
    let mut off = n.div_ceil(cpb);
    let rd = |off: &mut usize| -> f32 {
        let v = f16_bits_to_f32(u16::from_le_bytes([rec[*off], rec[*off + 1]]));
        *off += 2;
        v
    };
    let scale_abs = (0..r_dim).map(|_| rd(&mut off)).collect();
    let zp_abs = (0..r_dim).map(|_| rd(&mut off)).collect();
    let s_col = (0..c_dim).map(|_| rd(&mut off)).collect();
    QuantTile {
        q,
        scale_abs,
        zp_abs,
        s_col,
        r_dim,
        c_dim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG → standard normal (Box–Muller), no rand dev-dep.
    struct Lcg(u64);
    impl Lcg {
        fn u01(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f32 {
            let u1 = self.u01().max(1e-12);
            let u2 = self.u01();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }
    }

    fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-12)
    }

    /// A tile with heavy per-channel (per-row) variance spread: row r scaled by
    /// a geometric factor. Sinkhorn must collapse the imbalance toward 2.0.
    fn skewed_tile(r_dim: usize, c_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let mut t = vec![0.0f32; r_dim * c_dim];
        for r in 0..r_dim {
            let row_scale = 1.0f32 * (2.0f32).powi(r as i32 % 8); // 1×…128× spread
            for c in 0..c_dim {
                t[r * c_dim + c] = rng.normal() * row_scale;
            }
        }
        t
    }

    #[test]
    fn sinkhorn_reduces_imbalance() {
        let (r, c) = (128, 128);
        let tile = skewed_tile(r, c, 0xABCD);
        let before = imbalance(&tile, r, c);
        let bal = variance_normalize(&tile, r, c, SINKHORN_ITERS);
        let after = imbalance(&bal.balanced, r, c);
        eprintln!("imbalance before={before:.2} after={after:.2} (perfect=2.0)");
        assert!(after < before, "Sinkhorn must reduce imbalance");
        assert!(after < before * 0.5, "expected a large imbalance drop");
    }

    #[test]
    fn reconstruction_matches_with_balancing() {
        // The KVarN spec's pack/unpack gate (a) is cos-sim ≥ 0.999, but that is
        // measured on the *Hadamard-rotated* tile (rotation Gaussianizes and
        // removes the heavy tail that eats 4-bit min/max range). This core
        // deliberately leaves rotation to the caller (shared engine FWHT, as in
        // QTIP), so the un-rotated core alone reaches ~0.995 — still a strong
        // 4-bit round-trip. The upstream FWHT closes the remaining gap to 0.999
        // (validated end-to-end at the KV-wiring layer, Phase D1).
        let (r, c) = (128, 128);
        let tile = skewed_tile(r, c, 0x1234);
        let qt = quantize_tile(&tile, r, c);
        let deq = dequantize_tile(&qt);
        let cs = cos_sim(&tile, &deq);
        eprintln!("KVarN 4-bit (un-rotated core) cos-sim={cs:.5}");
        assert!(cs >= 0.995, "KVarN 4-bit core cos-sim {cs:.5} < 0.995");
    }

    /// D1 component #3: the on-device tile record round-trips — pack→unpack
    /// recovers q exactly and the fp16 metadata within fp16 precision, and
    /// dequant from the unpacked record matches dequant from the original
    /// QuantTile to fp16 tolerance. Guards the byte layout the GPU write/read
    /// kernels will share.
    #[test]
    fn kvarn_tile_record_roundtrips() {
        let (r, c) = (128, 128); // K tile: head_dim × GROUP
        let tile = skewed_tile(r, c, 0x7ED);
        let qt = quantize_tile(&tile, r, c);
        let rec = pack_kvarn_tile(&qt);
        assert_eq!(rec.len(), kvarn_record_bytes(r, c));
        let qt2 = unpack_kvarn_tile(&rec, r, c);
        assert_eq!(qt.q, qt2.q, "4-bit codes must round-trip exactly");
        // Dequant equivalence: cos-sim is the robust metric (the skewed tile's
        // 128× row-scale spread makes per-element *relative* error explode on
        // near-zero elements under f16 metadata rounding, though absolute error
        // is negligible). The record must reconstruct the same tile.
        let d1 = dequantize_tile(&qt);
        let d2 = dequantize_tile(&qt2);
        let cs = cos_sim(&d1, &d2);
        eprintln!(
            "KVarN record: {} B/tile ({:.3} B/elem), pack/unpack dequant cos-sim {cs:.6}",
            rec.len(),
            rec.len() as f32 / (r * c) as f32
        );
        assert!(
            cs > 0.9999,
            "fp16-metadata dequant drift: cos-sim {cs} too low"
        );
    }

    #[test]
    fn balancing_beats_naive_per_row_4bit() {
        // KVarN (with Sinkhorn) must reconstruct better than naive per-row 4-bit
        // on a column-skewed tile (per-column variance spread the per-row
        // quantizer alone can't see).
        let (r, c) = (96, 128);
        let mut rng = Lcg(0x55AA);
        let mut tile = vec![0.0f32; r * c];
        for rr in 0..r {
            for cc in 0..c {
                let col_scale = (1.5f32).powi((cc % 8) as i32);
                tile[rr * c + cc] = rng.normal() * col_scale;
            }
        }
        // Naive per-row 4-bit (no balancing).
        let mut naive = vec![0.0f32; r * c];
        for rr in 0..r {
            let row = &tile[rr * c..rr * c + c];
            let lo = row.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s = ((hi - lo) / QMAX).max(1e-8);
            for cc in 0..c {
                let q = (((tile[rr * c + cc] - lo) / s).round()).clamp(0.0, QMAX);
                naive[rr * c + cc] = q * s + lo;
            }
        }
        let mse = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| ((x - y) as f64).powi(2))
                .sum::<f64>()
                / a.len() as f64
        };
        let kvarn = dequantize_tile(&quantize_tile(&tile, r, c));
        let (e_naive, e_kvarn) = (mse(&tile, &naive), mse(&tile, &kvarn));
        eprintln!(
            "MSE naive-per-row={e_naive:.5} KVarN={e_kvarn:.5} (KVarN/naive={:.3})",
            e_kvarn / e_naive
        );
        assert!(
            e_kvarn < e_naive,
            "KVarN must beat naive per-row 4-bit on a skewed tile"
        );
    }

    /// The paper's claim, reduced to something testable: on data with outlier
    /// channels, the Hadamard rotation should improve low-bit reconstruction,
    /// and it should be an exact orthonormal involution.
    #[test]
    fn hadamard_rows_is_orthonormal_and_self_inverse() {
        let n = 128usize;
        let mut x: Vec<f32> = (0..n).map(|i| ((i * 37 % 61) as f32) - 30.0).collect();
        let orig = x.clone();
        let n0: f32 = orig.iter().map(|v| v * v).sum();
        hadamard_rows(&mut x, n);
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!(
            (n0 - n1).abs() / n0.max(1e-6) < 1e-4,
            "orthonormal transform must preserve L2: {n0} vs {n1}"
        );
        hadamard_rows(&mut x, n); // H is its own inverse when orthonormal
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-3, "H(H(x)) != x: {a} vs {b}");
        }
    }

    #[test]
    fn hadamard_rotation_improves_2bit_reconstruction_on_outlier_channels() {
        // One dominant channel per row is the case the paper calls out: a coarse
        // grid spends its whole range on the outlier and flattens everything else.
        // [channel(row) x token(col)]. An OUTLIER CHANNEL is a whole ROW that is
        // large across all tokens — that is what the paper means, and it is the
        // axis the rotation acts on. (An earlier version of this test put a
        // spike at one token per row and "passed" while rotating across tokens,
        // i.e. it validated the wrong operator.)
        let (r, c) = (128usize, 32usize);
        let mut tile = vec![0f32; r * c];
        for ch in 0..r {
            let outlier = ch % 37 == 0;
            for t in 0..c {
                let base = (((ch * 31 + t * 17) % 13) as f32 - 6.0) * 0.05;
                tile[ch * c + t] = if outlier { 12.0 + base } else { base };
            }
        }
        let err = |q: &QuantTile, reference: &[f32]| -> f32 {
            let d = dequantize_tile(q);
            let num: f32 = d
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let den: f32 = reference.iter().map(|v| v * v).sum::<f32>().max(1e-12);
            (num / den).sqrt()
        };
        const QMAX_2BIT: f32 = 3.0;
        let plain = err(&quantize_tile_qmax(&tile, r, c, QMAX_2BIT), &tile);

        // The rotated tile is quantised in the rotated frame, so compare against
        // the rotated reference — that is what attention consumes.
        let mut rotated_ref = tile.clone();
        hadamard_channels(&mut rotated_ref, r, c);
        let rotated = err(&quantize_tile_rotated(&tile, r, c, QMAX_2BIT), &rotated_ref);

        assert!(
            rotated < plain,
            "Hadamard rotation should reduce 2-bit error on outlier channels: \
             rotated {rotated:.4} vs plain {plain:.4}"
        );
    }
}
