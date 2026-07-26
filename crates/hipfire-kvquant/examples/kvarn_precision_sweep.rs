// SPDX-License-Identifier: Apache-2.0
//! KVarN precision sweep — is 4-bit "lossless vs f16", and if not, is the gap a
//! precision ceiling (more bits close it) or structural (Sinkhorn/quantizer
//! floor that more bits won't fix)?
//!
//! Runs the KVarN codec (`variance_normalize` + per-channel min/max quantize) at
//! qmax = 3..255 (2..8 "bits") on a realistic K-shaped tile and reports
//! reconstruction error + an attention-weight KLD proxy, against the f16
//! round-trip floor (the bar a kvarn hot ring must clear). CPU-only, deterministic.
//!
//! Read the scaling: if rel-err/attn-KLD keep dropping ~2× per bit down to the
//! f16 floor, 8-bit kvarn ≈ f16 (precision-limited). If they plateau above the
//! f16 floor, the loss is structural — more bits won't help; fix the transform.
//!
//! `cargo run -q -p hipfire-kvquant --example kvarn_precision_sweep`

use hipfire_kvquant::kvarn::{dequantize_tile, quantize_tile_qmax};
use hipfire_primitives::conv::{f16_to_f32, f32_to_f16};

/// Deterministic xorshift64 + Box–Muller (no rng dep; reproducible).
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unif(&mut self) -> f32 {
        (self.u64() >> 11) as f32 / (1u64 << 53) as f32
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unif().max(1e-9);
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// K-shaped tile [r_dim=head_dim, c_dim=tokens]: heavy per-channel variance
/// spread (log-normal σ), a few extreme outlier channels, and a few outlier
/// tokens — the pathology KVarN's per-channel balancing targets.
fn make_k_tile(r_dim: usize, c_dim: usize, rng: &mut Rng) -> Vec<f32> {
    let sigma: Vec<f32> = (0..r_dim)
        .map(|i| {
            let mut s = (rng.normal() * 1.0).exp(); // log-normal channel std
            if i % 37 == 0 {
                s *= 12.0; // occasional extreme outlier channel
            }
            s
        })
        .collect();
    let mut tile = vec![0f32; r_dim * c_dim];
    for (r, &sr) in sigma.iter().enumerate() {
        for cc in 0..c_dim {
            tile[r * c_dim + cc] = sr * rng.normal();
        }
    }
    // A few outlier tokens with large spikes in the outlier channels.
    for cc in (0..c_dim).step_by(41) {
        for r in (0..r_dim).step_by(37) {
            tile[r * c_dim + cc] += 6.0 * sigma[r] * rng.normal().signum();
        }
    }
    tile
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|&x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

/// (relative RMSE = ‖err‖/‖x‖, cosine similarity).
fn recon_metrics(orig: &[f32], recon: &[f32]) -> (f32, f32) {
    let err: Vec<f32> = orig.iter().zip(recon).map(|(&a, &b)| a - b).collect();
    let rel = rms(&err) / rms(orig).max(1e-12);
    let dot: f32 = orig.iter().zip(recon).map(|(&a, &b)| a * b).sum();
    let na: f32 = orig.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let nb: f32 = recon.iter().map(|&b| b * b).sum::<f32>().sqrt();
    (rel, dot / (na * nb).max(1e-12))
}

/// Mean KL(softmax(qᵀK_orig) ‖ softmax(qᵀK_recon)) over random unit queries.
/// K is [r_dim(head_dim) × c_dim(tokens)]; attention is over the c_dim tokens.
fn attn_kld(orig: &[f32], recon: &[f32], r_dim: usize, c_dim: usize, rng: &mut Rng) -> f32 {
    let nq = 64;
    let scale = 1.0 / (r_dim as f32).sqrt();
    let mut total = 0.0f64;
    for _ in 0..nq {
        let q: Vec<f32> = (0..r_dim).map(|_| rng.normal()).collect();
        let softmax = |k: &[f32]| -> Vec<f32> {
            let mut s: Vec<f32> = (0..c_dim)
                .map(|t| scale * (0..r_dim).map(|i| q[i] * k[i * c_dim + t]).sum::<f32>())
                .collect();
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0;
            for x in &mut s {
                *x = (*x - m).exp();
                z += *x;
            }
            for x in &mut s {
                *x /= z;
            }
            s
        };
        let p = softmax(orig);
        let qd = softmax(recon);
        for t in 0..c_dim {
            if p[t] > 1e-9 {
                total += p[t] as f64 * ((p[t] as f64) / (qd[t] as f64).max(1e-12)).ln();
            }
        }
    }
    (total / nq as f64) as f32
}

/// Plain per-token (per-column) affine min/max quant → dequant, no Sinkhorn, no
/// rotation. tile is [r=head_dim × c=tokens]; each token (column) gets its own
/// min/max over the head_dim rows — the natural codec for a per-token hot ring.
/// This is the "does K need treatment at 8-bit?" candidate.
fn affine_per_token(tile: &[f32], r: usize, c: usize, qmax: f32) -> Vec<f32> {
    let mut recon = vec![0f32; r * c];
    for t in 0..c {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for row in 0..r {
            let v = tile[row * c + t];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let scale = ((hi - lo) / qmax).max(1e-8);
        for row in 0..r {
            let v = tile[row * c + t];
            let q = (((v - lo) / scale).round()).clamp(0.0, qmax);
            recon[row * c + t] = q * scale + lo;
        }
    }
    recon
}

/// Plain per-token (per-column) SYMMETRIC absmax int8 quant → dequant (no zero-
/// point), mirroring the GPU `kv_cache_write_q8` codec (scale = amax/127). This is
/// the hot-ring candidate: on FWHT-rotated K the distribution is centered, so
/// symmetric should ≈ affine. `qmax` here is the signed level count (127 for 8-bit).
fn sym_per_token(tile: &[f32], r: usize, c: usize, qlev: f32) -> Vec<f32> {
    let mut recon = vec![0f32; r * c];
    for t in 0..c {
        let mut amax = 0f32;
        for row in 0..r {
            amax = amax.max(tile[row * c + t].abs());
        }
        let scale = (amax / qlev).max(1e-8);
        for row in 0..r {
            let v = tile[row * c + t];
            let q = (v / scale).round().clamp(-qlev, qlev);
            recon[row * c + t] = q * scale;
        }
    }
    recon
}

/// Per-token FWHT/Hadamard rotation of a `[head_dim × tokens]` tile (rotate each
/// token's head_dim column). `signed_fwht` is orthonormal, so rotating both K and
/// the query preserves q·K — we measure the affine error in this rotated frame
/// (random queries are rotation-invariant), no inverse needed. Rotation spreads
/// the outlier-channel energy so a per-token min/max wastes fewer bits.
fn fwht_tile(tile: &[f32], r: usize, c: usize) -> Vec<f32> {
    use hipfire_primitives::fwht::{gen_fwht_signs, signed_fwht};
    let s1 = gen_fwht_signs(42, r);
    let s2 = gen_fwht_signs(1042, r);
    let mut rot = vec![0f32; r * c];
    for t in 0..c {
        let mut col: Vec<f32> = (0..r).map(|row| tile[row * c + t]).collect();
        signed_fwht(&mut col, &s1, &s2);
        for row in 0..r {
            rot[row * c + t] = col[row];
        }
    }
    rot
}

fn main() {
    let (r, c) = (256usize, 128usize); // head_dim × 128-token block
    let mut rng = Rng(0x5407_1234_5678);
    let tile = make_k_tile(r, c, &mut rng);

    // f16 round-trip floor — the bar a kvarn hot ring must clear.
    let f16: Vec<f32> = tile.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
    let (f16_rel, f16_cos) = recon_metrics(&tile, &f16);
    let f16_kld = attn_kld(&tile, &f16, r, c, &mut Rng(0xABCD));

    println!("KVarN precision sweep — tile {r}×{c} (head_dim × tokens), realistic K stats\n");
    println!(
        "{:>5}  {:>10}  {:>9}  {:>12}",
        "bits", "rel_rmse", "cos_sim", "attn_KLD"
    );
    println!("{}", "-".repeat(42));
    println!(
        "{:>5}  {:>10.2e}  {:>9.6}  {:>12.3e}   <- f16 floor",
        "f16", f16_rel, f16_cos, f16_kld
    );
    for &(bits, qmax) in &[
        (2u32, 3.0f32),
        (3, 7.0),
        (4, 15.0),
        (5, 31.0),
        (6, 63.0),
        (7, 127.0),
        (8, 255.0),
    ] {
        let qt = quantize_tile_qmax(&tile, r, c, qmax);
        let recon = dequantize_tile(&qt);
        let (rel, cos) = recon_metrics(&tile, &recon);
        let kld = attn_kld(&tile, &recon, r, c, &mut Rng(0xABCD));
        println!("{bits:>5}  {rel:>10.2e}  {cos:>9.6}  {kld:>12.3e}");
    }
    println!(
        "\ninterpret: error halving ~2×/bit toward the f16 floor ⇒ precision-limited \
         (8-bit kvarn ≈ f16);\nplateau above the floor ⇒ structural (Sinkhorn/min-max \
         ceiling — more bits won't fix it)."
    );

    // Per-token hot-ring codecs: does plain 8-bit affine on K need a treatment
    // (FWHT rotation) to handle the outlier channels, or is it already ~kvarn?
    println!("\nPer-token hot-ring codecs (does K need treatment?):");
    println!("  (FWHT rows measured in the rotated frame: q·K preserved, so the");
    println!("   rotated-frame quant error is what attention sees.)\n");
    println!(
        "{:>5}  {:>16}  {:>10}  {:>12}",
        "bits", "codec", "rel_rmse", "attn_KLD"
    );
    println!("{}", "-".repeat(50));
    // Precompute the rotated tile once (FWHT candidates quantize + score here).
    let rot = fwht_tile(&tile, r, c);
    for &(bits, qmax) in &[(4u32, 15.0f32), (8, 255.0)] {
        // kvarn (Sinkhorn) and plain affine: original frame.
        for (name, recon, refr) in [
            (
                "kvarn(Sinkhorn)",
                dequantize_tile(&quantize_tile_qmax(&tile, r, c, qmax)),
                &tile,
            ),
            ("affine plain", affine_per_token(&tile, r, c, qmax), &tile),
            // affine + FWHT: quantize the rotated tile, score vs the rotated ref.
            ("affine+FWHT", affine_per_token(&rot, r, c, qmax), &rot),
            // symmetric absmax (kv_cache_write_q8 codec) + FWHT — the hot-ring path.
            (
                "sym+FWHT",
                sym_per_token(&rot, r, c, (qmax - 1.0) / 2.0),
                &rot,
            ),
        ] {
            let (rel, _) = recon_metrics(refr, &recon);
            let kld = attn_kld(refr, &recon, r, c, &mut Rng(0xABCD));
            println!("{bits:>5}  {name:>16}  {rel:>10.2e}  {kld:>12.3e}");
        }
    }
}
