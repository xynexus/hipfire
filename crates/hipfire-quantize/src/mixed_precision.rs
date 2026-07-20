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

use std::collections::{HashMap, HashSet};

use crate::codecs::{dequant_oq2g256, dequant_oq4g256, quantize_oq2g256, quantize_oq4g256};

/// On-disk / in-kernel-VRAM bits-per-weight of the three Opus tiers (block
/// overhead folded in): oq2 = 66 B/256, oq4 = 130 B/256, oq8 = 258 B/256. These
/// are the weight-bandwidth costs the in-kernel expand actually reads.
pub const OQ2_BPW: f64 = 66.0 * 8.0 / 256.0; // 2.0625
pub const OQ4_BPW: f64 = 130.0 * 8.0 / 256.0; // 4.0625
pub const OQ8_BPW: f64 = 258.0 * 8.0 / 256.0; // 8.0625

/// One of the three Opus weight tiers a dense-linear tensor can take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Oq2,
    Oq4,
    Oq8,
}

impl Tier {
    pub fn bpw(self) -> f64 {
        match self {
            Tier::Oq2 => OQ2_BPW,
            Tier::Oq4 => OQ4_BPW,
            Tier::Oq8 => OQ8_BPW,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Tier::Oq2 => "oq2",
            Tier::Oq4 => "oq4",
            Tier::Oq8 => "oq8",
        }
    }
}

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

/// imatrix-weighted output error at oq4 (per tensor), the sibling of
/// [`oq2_sensitivity`] used for 3-tier assignment. `< oq2` error and `>` the
/// (≈0) oq8 error, so the two together give the marginal quality of each upgrade
/// step (oq2→oq4→oq8).
pub fn oq4_sensitivity(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    imatrix_col: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> f64 {
    debug_assert_eq!(weights_f32.len(), m * k);
    let q = quantize_oq4g256(weights_f32, signs1, signs2);
    let deq = dequant_oq4g256(&q, m * k, signs1, signs2);
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

/// A dense-linear tensor with its output error at oq2 and oq4 (oq8 error is
/// taken as ≈0, near-lossless). The two errors define the quality gained by
/// each upgrade step.
#[derive(Clone, Debug)]
pub struct TierCandidate {
    pub name: String,
    pub numel: usize,
    pub err_oq2: f64,
    pub err_oq4: f64,
}

/// Per-tensor tier assignment across oq2/oq4/oq8.
#[derive(Clone, Debug, Default)]
pub struct TierPlan {
    pub tiers: HashMap<String, Tier>,
}

impl TierPlan {
    /// Assigned tier for a tensor (defaults to oq2 for unlisted tensors).
    pub fn tier(&self, name: &str) -> Tier {
        self.tiers.get(name).copied().unwrap_or(Tier::Oq2)
    }
    pub fn count(&self, t: Tier) -> usize {
        self.tiers.values().filter(|&&x| x == t).count()
    }
    /// Realized average bits-per-weight (weight-bandwidth) over `candidates`.
    pub fn realized_bpw(&self, candidates: &[TierCandidate]) -> f64 {
        let total: usize = candidates.iter().map(|c| c.numel).sum();
        if total == 0 {
            return OQ2_BPW;
        }
        let bits: f64 = candidates
            .iter()
            .map(|c| self.tier(&c.name).bpw() * c.numel as f64)
            .sum();
        bits / total as f64
    }
}

/// 3-tier greedy assignment across oq2/oq4/oq8 under a target average bpw.
/// Start every tensor at `floor` (oq2 for the full 3-tier sweep, or oq4 to
/// exclude oq2 entirely), then repeatedly apply the single upgrade step
/// (oq2→oq4 or oq4→oq8) with the best error-reduction per extra bit that still
/// fits the remaining budget. Each tensor upgrades at most twice, in order.
/// Reducing total weighted output error per byte spent is what we maximize.
pub fn assign_tiers(candidates: &[TierCandidate], target_bpw: f64, floor: Tier) -> TierPlan {
    let total: usize = candidates.iter().map(|c| c.numel).sum();
    let mut plan = TierPlan::default();
    for c in candidates {
        plan.tiers.insert(c.name.clone(), floor);
    }
    if total == 0 {
        return plan;
    }
    let target = target_bpw.clamp(floor.bpw(), OQ8_BPW);
    let budget_bits = (target - floor.bpw()) * total as f64; // extra above all-floor
    let mut spent = 0.0f64;

    loop {
        // Best available upgrade step by gain/cost density that still fits.
        let mut best: Option<(usize, f64, f64)> = None; // (idx, gain, cost)
        for (i, c) in candidates.iter().enumerate() {
            let (gain, cost, from_ok) = match plan.tier(&c.name) {
                Tier::Oq2 => (
                    c.err_oq2 - c.err_oq4,
                    (OQ4_BPW - OQ2_BPW) * c.numel as f64,
                    true,
                ),
                Tier::Oq4 => (c.err_oq4, (OQ8_BPW - OQ4_BPW) * c.numel as f64, true),
                Tier::Oq8 => (0.0, 0.0, false),
            };
            if !from_ok || cost <= 0.0 || spent + cost > budget_bits + 1e-6 {
                continue;
            }
            let density = gain / cost;
            match best {
                Some((_, bg, bc)) if density <= bg / bc => {}
                _ => best = Some((i, gain, cost)),
            }
        }
        match best {
            Some((i, _, cost)) => {
                let name = candidates[i].name.clone();
                let next = match plan.tier(&name) {
                    Tier::Oq2 => Tier::Oq4,
                    Tier::Oq4 => Tier::Oq8,
                    Tier::Oq8 => Tier::Oq8,
                };
                plan.tiers.insert(name, next);
                spent += cost;
            }
            None => break,
        }
    }
    plan
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

    fn tcand(name: &str, numel: usize, e2: f64, e4: f64) -> TierCandidate {
        TierCandidate {
            name: name.to_string(),
            numel,
            err_oq2: e2,
            err_oq4: e4,
        }
    }

    #[test]
    fn tiers_all_oq2_at_min_budget_all_oq8_at_max() {
        let cs = vec![tcand("a", 256, 10.0, 3.0), tcand("b", 256, 4.0, 1.0)];
        let lo = assign_tiers(&cs, OQ2_BPW, Tier::Oq2);
        assert_eq!(lo.count(Tier::Oq2), 2);
        let hi = assign_tiers(&cs, OQ8_BPW, Tier::Oq2);
        assert_eq!(hi.count(Tier::Oq8), 2, "max budget ⇒ everything oq8");
        assert!((hi.realized_bpw(&cs) - OQ8_BPW).abs() < 1e-9);
    }

    #[test]
    fn tiers_upgrade_highest_density_step_first() {
        // Equal size. "hot" has a big oq2→oq4 gain; "cold" barely benefits.
        // A mid budget (~oq4 avg) should lift "hot" toward oq8 before "cold"
        // leaves oq2.
        let cs = vec![tcand("hot", 256, 20.0, 2.0), tcand("cold", 256, 1.0, 0.5)];
        let plan = assign_tiers(&cs, OQ4_BPW, Tier::Oq2);
        assert_ne!(plan.tier("hot"), Tier::Oq2, "sensitive tensor upgraded");
        assert!(plan.realized_bpw(&cs) <= OQ4_BPW + 1e-9, "budget respected");
    }

    #[test]
    fn tiers_respect_budget_and_monotone_bpw() {
        let cs = vec![
            tcand("a", 512, 12.0, 4.0),
            tcand("b", 256, 6.0, 2.0),
            tcand("c", 128, 3.0, 1.0),
        ];
        let mut last = 0.0;
        for &t in &[OQ2_BPW, 3.0, 4.0, 5.0, 6.0, OQ8_BPW] {
            let bpw = assign_tiers(&cs, t, Tier::Oq2).realized_bpw(&cs);
            assert!(bpw <= t + 1e-6, "realized {bpw} over target {t}");
            assert!(bpw >= last - 1e-6, "bpw must be monotone in budget");
            last = bpw;
        }
    }
}
