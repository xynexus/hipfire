// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Online estimate of `r` — the marginal cost of one verify slot as a fraction
//! of a full AR step.
//!
//! `r` is the one term the acceptance curve cannot supply. Drafting a slot costs
//! the marginal compute of a wider verify, `c1`; an accepted token saves a whole
//! AR step, `c0 + c1`. So a slot pays exactly when `P(accept) > c1 / (c0 + c1)`,
//! and that right-hand side is `r`. See `NgramStats::marginal_acceptance`.
//!
//! ## Why this can be online, and still replay exactly
//!
//! `r` is a time ratio, so it can only come from timing — but timing only breaks
//! reproducibility if it varies *inside* the unit being replayed. The contract
//! is therefore **quantised and recorded**: fit between requests, hold the value
//! fixed for the duration of one request, and log it with the run. Replay with
//! the logged `r` is then bit-identical, exactly as a recorded RNG seed makes a
//! sampler reproducible. An RNG is not non-deterministic; an *unrecorded* seed
//! is.
//!
//! This type never reads a clock. The caller passes `elapsed_ns`, which keeps
//! the estimator pure and unit-testable, and keeps the crate free of a time
//! dependency it would otherwise carry into the draft path.
//!
//! ## It needs no probing
//!
//! Draft widths already vary in ordinary operation (the depth histogram spans
//! 316909 samples at depth 0 against 130207 at depth 14), so `(width, time)`
//! pairs arrive for free. There is no bandit, no exploration arm and no
//! perturbation: this is a regression over data the system produces anyway.
//!
//! ## Two traps this is built to avoid
//!
//! **Do not infer `r` from the operating point.** "Assume the current width is
//! optimal, so `r ~= marginal_acceptance(W)`" is self-referential: it ratifies
//! whatever the system started at and can never discover it was wrong. `r` comes
//! from the cost side only.
//!
//! **Batched steps are not clean samples.** In the daemon a step's elapsed time
//! reflects the whole batch, not this stream's draft width, so fitting `c1` to
//! them fits it to other tenants' work. [`observe`](SlotCostEstimator::observe)
//! drops anything with `batch_size != 1`.

/// ## The loop, closed
///
/// Feeding three synthetic parts to this estimator and reading
/// [`optimal_width`](crate::NgramStats::optimal_width) off the MEASURED
/// acceptance curve (400k tokens of hipfire's own Rust):
///
/// | part | recovered `r` | optimal width |
/// |---|---|---|
/// | bandwidth-bound (weights dominate) | 0.0020 | 16 (the cap) |
/// | balanced | 0.0909 | 16 |
/// | compute-bound (slots dear) | 0.4444 | 4 |
///
/// Neither `r` nor the width was configured — the first is regressed from
/// timings, the second read off measured acceptance. Note the practical
/// consequence: on anything but a compute-bound part this workload wants the
/// FULL draft width, which is the opposite of what `min_acceptance` does.
///
/// Streaming weighted least squares of `time ~= c0 + c1 * width`.
#[derive(Debug, Clone)]
pub struct SlotCostEstimator {
    n: f64,
    sum_w: f64,
    sum_t: f64,
    sum_ww: f64,
    sum_wt: f64,
    /// Per-sample decay, so the fit tracks rather than converges. `c0` moves
    /// when the model does, and a converged estimate would never notice.
    lambda: f64,
    /// Effective samples required before a fit is offered.
    min_samples: f64,
    /// Width variance required before a fit is offered. Without spread in
    /// `width` the normal equations are singular and the slope is noise
    /// amplified by a near-zero denominator — the failure mode that would
    /// otherwise produce a confident, meaningless `r`.
    min_width_var: f64,
}

impl Default for SlotCostEstimator {
    fn default() -> Self {
        Self {
            n: 0.0,
            sum_w: 0.0,
            sum_t: 0.0,
            sum_ww: 0.0,
            sum_wt: 0.0,
            // ~1000-sample memory: long enough to average scheduler noise,
            // short enough to follow a model swap.
            lambda: 0.999,
            min_samples: 64.0,
            min_width_var: 1.0,
        }
    }
}

impl SlotCostEstimator {
    pub fn new(lambda: f64, min_samples: f64, min_width_var: f64) -> Self {
        Self {
            lambda,
            min_samples,
            min_width_var,
            ..Default::default()
        }
    }

    /// Record one verify step. `batch_size != 1` is DROPPED — see the module
    /// note on contamination.
    pub fn observe(&mut self, width: usize, elapsed_ns: u64, batch_size: usize) {
        if batch_size != 1 || width == 0 || elapsed_ns == 0 {
            return;
        }
        let (w, t) = (width as f64, elapsed_ns as f64);
        let l = self.lambda;
        self.n = self.n * l + 1.0;
        self.sum_w = self.sum_w * l + w;
        self.sum_t = self.sum_t * l + t;
        self.sum_ww = self.sum_ww * l + w * w;
        self.sum_wt = self.sum_wt * l + w * t;
    }

    /// Effective sample count, after decay.
    pub fn samples(&self) -> f64 {
        self.n
    }

    /// Observed spread in width. Zero means every step drafted the same number
    /// of slots, which cannot identify a slope.
    pub fn width_variance(&self) -> f64 {
        if self.n <= 0.0 {
            return 0.0;
        }
        let mean = self.sum_w / self.n;
        (self.sum_ww / self.n - mean * mean).max(0.0)
    }

    /// The fitted `(c0, c1)`, or `None` when the data cannot support a fit.
    pub fn fit(&self) -> Option<(f64, f64)> {
        if self.n < self.min_samples || self.width_variance() < self.min_width_var {
            return None;
        }
        let denom = self.n * self.sum_ww - self.sum_w * self.sum_w;
        if denom.abs() < f64::EPSILON {
            return None;
        }
        let c1 = (self.n * self.sum_wt - self.sum_w * self.sum_t) / denom;
        let c0 = (self.sum_t - c1 * self.sum_w) / self.n;
        Some((c0, c1))
    }

    /// `r = c1 / (c0 + c1)`, or `None` when there is no usable fit.
    ///
    /// A fit implying a non-positive marginal slot cost, or a non-positive fixed
    /// cost, is REJECTED rather than clamped: both are physically impossible, so
    /// their appearance means the samples are contaminated (batching that slipped
    /// through, a preempted step, a model swap mid-window) and the honest answer
    /// is "no estimate", not a number pulled to a boundary.
    pub fn r(&self) -> Option<f32> {
        let (c0, c1) = self.fit()?;
        if c1 <= 0.0 || c0 <= 0.0 {
            return None;
        }
        let r = c1 / (c0 + c1);
        if !r.is_finite() || !(0.0..=1.0).contains(&r) {
            return None;
        }
        Some(r as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise steps from a known cost model and check `r` is recovered.
    fn feed(est: &mut SlotCostEstimator, c0: f64, c1: f64, widths: &[usize], reps: usize) {
        for _ in 0..reps {
            for &w in widths {
                est.observe(w, (c0 + c1 * w as f64) as u64, 1);
            }
        }
    }

    #[test]
    fn it_recovers_a_known_cost_ratio() {
        // c0 = 1_000_000 ns fixed, c1 = 50_000 ns per slot -> r = 50/1050.
        let mut est = SlotCostEstimator::default();
        feed(&mut est, 1_000_000.0, 50_000.0, &[1, 4, 8, 16], 40);
        let r = est.r().expect("a clean fit must produce r");
        let want = 50_000.0 / 1_050_000.0;
        assert!((r as f64 - want).abs() < 1e-3, "r={r} want~{want:.4}");
    }

    #[test]
    fn a_bandwidth_bound_part_wants_a_wide_draft() {
        // c0 dominates: a slot is nearly free, so r is small and the optimal
        // width is deep. This is the case the module header asserts is typical.
        let mut est = SlotCostEstimator::default();
        feed(&mut est, 10_000_000.0, 20_000.0, &[1, 4, 8, 16], 40);
        let r = est.r().unwrap();
        assert!(r < 0.01, "bandwidth-bound should give a tiny r, got {r}");
    }

    #[test]
    fn no_spread_in_width_means_no_estimate() {
        // Every step drafted 8 slots: the slope is unidentifiable, and the
        // honest answer is None rather than a confident number from a
        // near-singular denominator.
        let mut est = SlotCostEstimator::default();
        feed(&mut est, 1_000_000.0, 50_000.0, &[8], 200);
        assert!(est.samples() >= 64.0, "the samples are there");
        assert_eq!(est.width_variance(), 0.0);
        assert!(est.r().is_none(), "no width spread must yield no estimate");
    }

    #[test]
    fn thin_evidence_yields_no_estimate() {
        let mut est = SlotCostEstimator::default();
        feed(&mut est, 1_000_000.0, 50_000.0, &[1, 16], 4); // 8 samples
        assert!(est.r().is_none());
    }

    #[test]
    fn batched_steps_are_dropped_not_fitted() {
        // Contaminated samples must not reach the fit at all: a batched step's
        // elapsed time reflects other tenants' work, so fitting c1 to it fits
        // c1 to them.
        let mut est = SlotCostEstimator::default();
        for _ in 0..500 {
            est.observe(8, 999_999_999, 4);
        }
        assert_eq!(est.samples(), 0.0, "batch_size != 1 must not accumulate");
        assert!(est.r().is_none());
    }

    #[test]
    fn a_physically_impossible_fit_is_refused_not_clamped() {
        // Time FALLING with width implies a negative marginal slot cost. That
        // cannot happen, so it means the window is contaminated; return None
        // rather than a boundary value that would read as a real measurement.
        let mut est = SlotCostEstimator::default();
        feed(&mut est, 1_000_000.0, -20_000.0, &[1, 4, 8, 16], 40);
        assert!(est.r().is_none(), "a negative slope must be refused");
    }

    #[test]
    fn it_tracks_a_change_rather_than_converging() {
        // A model swap changes c0. With forgetting the estimate follows it;
        // a converged average never would.
        let mut est = SlotCostEstimator::new(0.99, 64.0, 1.0);
        feed(&mut est, 1_000_000.0, 50_000.0, &[1, 4, 8, 16], 100);
        let before = est.r().unwrap();
        feed(&mut est, 20_000_000.0, 50_000.0, &[1, 4, 8, 16], 400);
        let after = est.r().unwrap();
        assert!(
            after < before / 2.0,
            "r should fall sharply when c0 grows: {before} -> {after}"
        );
    }
}
