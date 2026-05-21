//! Pure-Rust AWQ scale arithmetic shared between the quantizer and the
//! iterative orchestrator (`bin/iterate_awq_gptq.rs`). Twins the small
//! Python helpers in `scripts/mq4_masked_calib.py`:
//!
//! - `compute_awq_scales` — `[K]` `in_sum2` → `[K]` geo-mean-normalized scales
//! - `compute_awq_scales_from_hessian_diag` — `[K,K]` Hessian → `[K]` scales
//!   (reads the diagonal in row-major order)
//! - `damp_awq_scales` — `(1 - beta) * prev + beta * raw`
//! - `relative_l2_delta` — sqrt(||cur - prev||² / ||prev||²)
//!
//! The quantizer's `main.rs` has its own copies of `compute_awq_scales`
//! and `compute_awq_scales_autoawq` (we don't touch those to keep the
//! existing single-pass quantize path bit-identical). Tests in this
//! module assert the shared helpers agree with the canonical formulas.

use byteorder::{ByteOrder, LittleEndian};

/// `compute_awq_scales(in_sum2, alpha)` — paper-formula scale derivation
/// in log space (numerically stable across the wide dynamic range of
/// activation-second-moment values). Mirrors
/// `scripts/mq4_masked_calib.py::compute_awq_scales` and the in-tree
/// `crates/hipfire-quantize/src/main.rs::compute_awq_scales`.
///
/// `in_sum2[j] = Σ_tok x_tok[j]²`. The factor of `1 / n_tok` cancels in
/// the geo-mean normalization, so we drop it.
///
/// Output: `s[j]` such that `geo_mean(s) = 1.0` (up to FP precision).
pub fn compute_awq_scales(in_sum2: &[f32], alpha: f32) -> Vec<f32> {
    let k = in_sum2.len();
    debug_assert!(k > 0, "empty in_sum2 vector");
    let half_alpha = (alpha as f64) * 0.5;
    let mut log_s = Vec::with_capacity(k);
    let mut sum_log = 0.0_f64;
    for &v in in_sum2 {
        let v_clamped = (v as f64).max(1.0e-12);
        let l = half_alpha * v_clamped.ln();
        log_s.push(l);
        sum_log += l;
    }
    let mean_log = sum_log / (k as f64);
    log_s
        .into_iter()
        .map(|l| ((l - mean_log).exp()) as f32)
        .collect()
}

/// `compute_awq_scales_from_hessian_diag(h_bytes, k, alpha)` — read the
/// diagonal of a K×K F32 row-major Hessian (HFHS payload format) and
/// derive AWQ scales from it. Mirrors Python's
/// `compute_awq_scales_from_hessian` for the 2D Hessian case.
///
/// The diagonal of `H = Σ_tok x · xᵀ` is exactly `Σ_tok x²` — the imatrix
/// `in_sum2`. So this is equivalent to reading the imatrix vector that
/// would have been emitted by an imatrix-only collector, but it lets the
/// orchestrator derive scales from a Hessian sidecar without re-running
/// a parallel imatrix collection.
///
/// `h_bytes` must be a row-major F32 K×K payload (raw bytes); we read
/// `H[i, i] = h_bytes[(i * k + i) * 4 .. + 4]`.
pub fn compute_awq_scales_from_hessian_diag(
    h_bytes: &[u8],
    k: usize,
    alpha: f32,
) -> Vec<f32> {
    debug_assert_eq!(
        h_bytes.len(),
        k * k * 4,
        "Hessian payload size {} != K*K*4 = {}",
        h_bytes.len(),
        k * k * 4
    );
    let mut diag = Vec::with_capacity(k);
    for i in 0..k {
        let off = (i * k + i) * 4;
        let v = LittleEndian::read_f32(&h_bytes[off..off + 4]);
        diag.push(v);
    }
    compute_awq_scales(&diag, alpha)
}

/// `damp_awq_scales(prev, raw, beta)` — round-to-round damped update:
/// `s_round = (1 - beta) * prev + beta * raw`. Mirrors Python's
/// `damp_awq_scale_dict`.
///
/// When `prev` is `None` (round 0), the raw scales pass through
/// unchanged (this is the same convention as Python; `beta` is ignored).
pub fn damp_awq_scales(prev: Option<&[f32]>, raw: &[f32], beta: f32) -> Vec<f32> {
    assert!(
        (0.0..=1.0).contains(&beta),
        "damping must be in [0, 1], got {beta}"
    );
    match prev {
        None => raw.to_vec(),
        Some(p) => {
            assert_eq!(
                p.len(),
                raw.len(),
                "damp_awq_scales: prev len {} != raw len {}",
                p.len(),
                raw.len()
            );
            p.iter()
                .zip(raw.iter())
                .map(|(&prev_v, &raw_v)| (1.0 - beta) * prev_v + beta * raw_v)
                .collect()
        }
    }
}

/// `relative_l2_delta(prev, cur)` — `sqrt(Σ‖cur - prev‖² / Σ‖prev‖²)`
/// summed over all tensors. Mirrors Python's `relative_l2_delta`.
///
/// `prev` and `cur` are aligned slices of per-tensor (name, scale)
/// pairs; the function only contributes terms for tensor names present
/// in both. Returns 0.0 when `prev` is empty (round 0).
pub fn relative_l2_delta(prev: &[(String, Vec<f32>)], cur: &[(String, Vec<f32>)]) -> f64 {
    if prev.is_empty() {
        return 0.0;
    }
    use std::collections::HashMap;
    let prev_map: HashMap<&str, &[f32]> =
        prev.iter().map(|(n, s)| (n.as_str(), s.as_slice())).collect();
    let mut numer = 0.0_f64;
    let mut denom = 0.0_f64;
    for (name, cur_v) in cur {
        let prev_v = match prev_map.get(name.as_str()) {
            Some(v) => *v,
            None => continue,
        };
        assert_eq!(
            prev_v.len(),
            cur_v.len(),
            "scale shape mismatch for {name}: prev {} vs cur {}",
            prev_v.len(),
            cur_v.len()
        );
        for (a, b) in prev_v.iter().zip(cur_v.iter()) {
            let d = (*b as f64) - (*a as f64);
            numer += d * d;
            denom += (*a as f64) * (*a as f64);
        }
    }
    if denom <= 0.0 {
        return 0.0;
    }
    (numer / denom).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo_mean(s: &[f32]) -> f64 {
        let mut log_sum = 0.0_f64;
        for &v in s {
            log_sum += (v as f64).max(1.0e-30).ln();
        }
        (log_sum / (s.len() as f64)).exp()
    }

    #[test]
    fn compute_awq_scales_geo_mean_is_one() {
        let in_sum2 = vec![1.0_f32, 4.0, 9.0, 16.0, 25.0];
        let s = compute_awq_scales(&in_sum2, 0.55);
        let gm = geo_mean(&s);
        assert!((gm - 1.0).abs() < 1.0e-5, "geo_mean = {gm}");
    }

    #[test]
    fn compute_awq_scales_alpha_zero_is_identity() {
        let in_sum2 = vec![1.0_f32, 10.0, 100.0, 1000.0];
        let s = compute_awq_scales(&in_sum2, 0.0);
        for v in &s {
            assert!((*v - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn from_hessian_diag_matches_in_sum2_path() {
        // Build a 4×4 row-major F32 Hessian with diag = [2, 8, 18, 32] and
        // arbitrary off-diagonal (the helper must ignore off-diagonals).
        let k = 4usize;
        let mut h = vec![0.0_f32; k * k];
        for i in 0..k {
            h[i * k + i] = 2.0 * ((i + 1) * (i + 1)) as f32;
        }
        // Sprinkle off-diagonal noise.
        h[0 * k + 1] = 99.0;
        h[3 * k + 2] = -77.0;
        let mut bytes = Vec::with_capacity(k * k * 4);
        for &v in &h {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let from_h = compute_awq_scales_from_hessian_diag(&bytes, k, 0.55);
        let from_diag = compute_awq_scales(&[2.0_f32, 8.0, 18.0, 32.0], 0.55);
        for (a, b) in from_h.iter().zip(from_diag.iter()) {
            assert!((a - b).abs() < 1.0e-6, "from_h {a} vs from_diag {b}");
        }
    }

    #[test]
    fn damp_round_zero_is_pass_through() {
        let raw = vec![1.0_f32, 2.0, 3.0];
        let s = damp_awq_scales(None, &raw, 0.5);
        assert_eq!(s, raw);
    }

    #[test]
    fn damp_convex_combination() {
        let prev = vec![1.0_f32, 1.0, 1.0];
        let raw = vec![3.0_f32, 5.0, 7.0];
        let s = damp_awq_scales(Some(&prev), &raw, 0.5);
        assert_eq!(s, vec![2.0_f32, 3.0, 4.0]);
    }

    #[test]
    fn rel_l2_delta_zero_for_identical() {
        let prev = vec![
            ("a".to_string(), vec![1.0_f32, 2.0]),
            ("b".to_string(), vec![3.0_f32, 4.0]),
        ];
        let cur = prev.clone();
        let d = relative_l2_delta(&prev, &cur);
        assert!(d.abs() < 1.0e-12);
    }

    #[test]
    fn rel_l2_delta_nonzero_for_perturbed() {
        let prev = vec![("a".to_string(), vec![1.0_f32, 1.0])];
        let cur = vec![("a".to_string(), vec![2.0_f32, 2.0])];
        let d = relative_l2_delta(&prev, &cur);
        // ||diff||^2 = 2, ||prev||^2 = 2 → sqrt(1) = 1
        assert!((d - 1.0).abs() < 1.0e-12, "delta = {d}");
    }

    #[test]
    fn rel_l2_delta_empty_prev_returns_zero() {
        let cur = vec![("a".to_string(), vec![1.0_f32, 2.0])];
        let d = relative_l2_delta(&[], &cur);
        assert_eq!(d, 0.0);
    }
}
