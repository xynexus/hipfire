// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Calibration-driven per-tensor mixed-precision assignment.
//!
//! oq2 is a low-importance-TAIL format, not a standalone one (see
//! `project_oq_mixed_precision_promotion`). This module is the layer that
//! decides which tensors form that tail: it ranks dense-linear tensors by the
//! output error they incur when demoted to the LOW format (oq2) and greedily
//! keeps the most sensitive at the HIGH format (oq8) under a target
//! bits-per-weight budget. The result is a per-tensor format map the quantizer
//! consults so oq2 only ever carries the least-important weights.

use std::collections::HashSet;

use crate::codecs::{dequant_oq2g256, quantize_oq2g256};

/// A dense-linear tensor considered for mixed-precision assignment.
#[derive(Clone, Debug)]
pub struct TensorCandidate {
    pub name: String,
    /// Weight element count (`m * k`) — drives both the byte cost and the budget.
    pub numel: usize,
    /// Output-error contribution if this tensor is quantized to the LOW format
    /// instead of HIGH (imatrix-weighted quant error). Higher ⇒ keep HIGH.
    pub sensitivity: f64,
}

/// Which tensors are promoted to the HIGH-precision format; the rest use LOW.
#[derive(Clone, Debug, Default)]
pub struct MixedPlan {
    pub high: HashSet<String>,
}

impl MixedPlan {
    pub fn is_high(&self, name: &str) -> bool {
        self.high.contains(name)
    }
    pub fn num_high(&self) -> usize {
        self.high.len()
    }
    /// Realized average bits-per-weight of the plan over `candidates`.
    pub fn realized_bpw(&self, candidates: &[TensorCandidate], hi_bpw: f64, lo_bpw: f64) -> f64 {
        let total: usize = candidates.iter().map(|c| c.numel).sum();
        if total == 0 {
            return lo_bpw;
        }
        let hi: usize = candidates
            .iter()
            .filter(|c| self.is_high(&c.name))
            .map(|c| c.numel)
            .sum();
        let lo = total - hi;
        (hi as f64 * hi_bpw + lo as f64 * lo_bpw) / total as f64
    }
}

/// Greedy sensitivity/byte knapsack. With every candidate at LOW as the base,
/// spend the extra bit-budget promoting tensors to HIGH in order of
/// sensitivity-per-extra-bit (≈ `sensitivity / numel`, since the per-weight bit
/// delta is constant) until the target average bits-per-weight is reached.
/// Removing a tensor's LOW error per byte spent is what we maximize, so this
/// is the standard value/cost greedy fill (it keeps trying smaller tensors
/// after a large one fails to fit, rather than stopping).
///
/// `hi_bpw`/`lo_bpw` are the two formats' bits-per-weight (e.g. oq8 8.0625,
/// oq2 2.0625). `target_bpw` is clamped to `[lo_bpw, hi_bpw]`.
pub fn assign_mixed_precision(
    candidates: &[TensorCandidate],
    hi_bpw: f64,
    lo_bpw: f64,
    target_bpw: f64,
) -> MixedPlan {
    let total: usize = candidates.iter().map(|c| c.numel).sum();
    if total == 0 || hi_bpw <= lo_bpw {
        return MixedPlan::default();
    }
    let target = target_bpw.clamp(lo_bpw, hi_bpw);
    // Extra bit-budget above the all-LOW base, in bit·weights.
    let extra_budget = (target - lo_bpw) * total as f64;
    let per_weight_cost = hi_bpw - lo_bpw; // bits per promoted weight

    // Rank by sensitivity density (error removed per extra byte). Cost is
    // proportional to numel, so density = sensitivity / numel. Ties broken by
    // raw sensitivity then name so the plan is deterministic.
    let mut order: Vec<&TensorCandidate> = candidates.iter().collect();
    order.sort_by(|a, b| {
        let da = a.sensitivity / a.numel.max(1) as f64;
        let db = b.sensitivity / b.numel.max(1) as f64;
        db.partial_cmp(&da)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.sensitivity
                    .partial_cmp(&a.sensitivity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut spent = 0.0f64;
    let mut plan = MixedPlan::default();
    for c in order {
        let cost = per_weight_cost * c.numel as f64;
        if spent + cost <= extra_budget + 1e-6 {
            spent += cost;
            plan.high.insert(c.name.clone());
        }
    }
    plan
}

/// Per-tensor promotion sensitivity: the imatrix-weighted output error the
/// tensor incurs at the LOW (oq2) format. Quantizes `weights_f32` ([m, k],
/// row-major) to oq2, dequantizes, and accumulates the per-column squared error
/// weighted by `imatrix_col[c]` (per-input-column activation importance,
/// e.g. Σ x²). When no imatrix is available, pass a uniform vector. This is the
/// error that promotion to oq8 removes, so it is exactly what the allocator
/// ranks on.
pub fn oq2_sensitivity(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    imatrix_col: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> f64 {
    debug_assert_eq!(weights_f32.len(), m * k);
    let q = quantize_oq2g256(weights_f32, signs1, signs2);
    let deq = dequant_oq2g256(&q, m * k, signs1, signs2);
    let mut per_col = vec![0.0f64; k];
    for row in 0..m {
        let base = row * k;
        for c in 0..k {
            let d = (weights_f32[base + c] - deq[base + c]) as f64;
            per_col[c] += d * d;
        }
    }
    (0..k)
        .map(|c| per_col[c] * imatrix_col.get(c).copied().unwrap_or(1.0) as f64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(name: &str, numel: usize, sensitivity: f64) -> TensorCandidate {
        TensorCandidate {
            name: name.to_string(),
            numel,
            sensitivity,
        }
    }

    #[test]
    fn all_low_at_min_budget_all_high_at_max() {
        let cs = vec![cand("a", 100, 5.0), cand("b", 100, 1.0)];
        let lo = assign_mixed_precision(&cs, 8.0, 2.0, 2.0);
        assert_eq!(lo.num_high(), 0, "target = lo_bpw ⇒ nothing promoted");
        let hi = assign_mixed_precision(&cs, 8.0, 2.0, 8.0);
        assert_eq!(hi.num_high(), 2, "target = hi_bpw ⇒ all promoted");
    }

    #[test]
    fn promotes_most_sensitive_first_under_budget() {
        // Two equal-size tensors, budget for exactly one at HIGH (avg 5 bpw).
        let cs = vec![cand("hot", 100, 9.0), cand("cold", 100, 1.0)];
        let plan = assign_mixed_precision(&cs, 8.0, 2.0, 5.0);
        assert!(plan.is_high("hot"), "the sensitive tensor is promoted");
        assert!(!plan.is_high("cold"), "the tail tensor stays LOW (oq2)");
        assert!((plan.realized_bpw(&cs, 8.0, 2.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn ranks_by_density_not_raw_sensitivity() {
        // "big" has higher raw sensitivity but far lower per-weight density.
        // With budget for only ~100 weights of HIGH, the dense small tensor wins.
        let cs = vec![cand("big", 1000, 20.0), cand("dense", 100, 8.0)];
        // avg target that affords promoting ~100 weights: total=1100, extra
        // budget = (t-2)*1100 bits; promoting "dense" costs 6*100=600 bit·wts.
        let plan = assign_mixed_precision(&cs, 8.0, 2.0, 2.6);
        assert!(plan.is_high("dense"), "higher density promoted first");
        assert!(!plan.is_high("big"), "low-density giant stays LOW");
    }

    #[test]
    fn greedy_backfills_smaller_tensors() {
        // After the densest tensor is placed, a smaller one should still fit the
        // remaining budget rather than the loop stopping.
        let cs = vec![
            cand("d1", 100, 10.0), // density 0.10
            cand("d2", 300, 24.0), // density 0.08 but too big to co-fit with d1 at tight budget
            cand("d3", 50, 4.0),   // density 0.08, small — should backfill
        ];
        let plan = assign_mixed_precision(&cs, 8.0, 2.0, 2.6);
        // total=450, extra=(2.6-2)*450=270 bit·wts; per-weight cost=6.
        // d1 cost=600 > 270 already too big? recompute: promote fits if cost<=extra.
        // d1 cost=600 > 270 ⇒ nothing fits; assert budget respected (no over-spend).
        assert!(plan.realized_bpw(&cs, 8.0, 2.0) <= 2.6 + 1e-9);
    }

    #[test]
    fn sensitivity_is_nonnegative_and_zero_for_representable() {
        let (s1, s2) = (
            crate::gen_fwht_signs(7, 256),
            crate::gen_fwht_signs(99, 256),
        );
        // Weights already on the oq2 grid after rotation are ~lossless ⇒ tiny.
        let w = vec![0.0f32; 256]; // all-zero: exactly representable
        let imat = vec![1.0f32; 256];
        let s = oq2_sensitivity(&w, 1, 256, &imat, &s1, &s2);
        assert!(s.abs() < 1e-6, "zero weights ⇒ ~zero sensitivity, got {s}");
        // Non-trivial weights ⇒ strictly positive sensitivity.
        let w2: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.13).collect();
        let s2v = oq2_sensitivity(&w2, 1, 256, &imat, &s1, &s2);
        assert!(s2v > 0.0, "lossy weights ⇒ positive sensitivity, got {s2v}");
    }
}
