//! Pure (no-GPU) draft-block-size controller for DSpark spec-decode.
//!
//! Picks the block that maximizes decode throughput via a cost-model argmax:
//!   block = argmax_N  τ(N) / (t_ar + (N-1)·Δt)
//! where τ(N) = 1 + Σ_{k=1..N} S(k) is the expected committed tokens per window
//! (S(k) = P(accept_len ≥ k), the acceptance-depth survival), Δt is the marginal
//! per-position window cost (ms) and t_ar the single-block window cost (ms). Both
//! come from the live window-cost calibration — the FULL per-window wall-time
//! (draft+heads+verify) measured vs block, NOT verify alone (verify-only omits the
//! fixed drafter/launch overhead and so over-charges large blocks). The argmax thus
//! directly maximizes committed-tokens ÷ window-wall-time = tok/s.
//!
//! This auto-adapts across architectures without per-arch tuning:
//!   • DeepSeek4 (expensive MoE verify): Δt is large AND survival saturates early
//!     (S(3+)≈0) so τ stops growing → argmax settles at 2.
//!   • Qwen3 (cheap verify, fixed per-window overhead dominates): the window cost is
//!     ~flat in block (Δt≈0, clamped) so the argmax climbs toward the drafter's true
//!     acceptance depth (≈7), capped only by where survival runs out.
//!
//! Ramp phase (breaks the calibration deadlock): after WARMUP, each request runs a
//! linear sweep from min_block to max_block (RAMP_HOLD windows per step). The sweep
//! records window timing at ≥2 distinct n_verify so the cost curve can be calibrated,
//! and seeds the survival counts at every depth so the argmax has real signal. After
//! the ramp the argmax takes over for the remainder of the request.

/// Cost-model draft-block controller for the DSpark drafter.
///
/// Also driven by serving-core's DFlash decode loop (it is pure and
/// drafter-agnostic: everything it needs is per-window accept depth,
/// proposal count, and wall time).
pub struct BlockController {
    block: usize,
    default_block: usize,
    min_block: usize,
    max_block: usize,
    /// Deepest block ever drafted; upper bound of the argmax search. Grows during
    /// the ramp phase to max_block, then stays fixed — S(k) above it is unobservable
    /// (but after the ramp all depths have been tried).
    max_tried: usize,
    /// Per-depth acceptance-survival COUNTS (request-specific; reset each request).
    /// `s_hit[k]` = #windows with accept_len ≥ k; `s_tot[k]` = #windows that drafted
    /// ≥ k (so depth k was observable). The survival estimate is S(k)=s_hit[k]/s_tot[k],
    /// k ∈ 1..=MAX_DEPTH. Counts grow only for depths actually drafted, so once the block
    /// settles below k, s_tot[k] stops growing and S(k) retains its last value — the
    /// cap-trap fix (the old decaying P(accept==k) histogram forgot the ramp's
    /// deep-survival samples, so a low-settled block could never re-discover depth).
    /// Counts converge fast (a deterministic full-accept depth reads 2/2=1 after just
    /// the ramp), where a slow EMA would still read ≈0.1.
    s_hit: [f32; MAX_DEPTH + 1],
    s_tot: [f32; MAX_DEPTH + 1],
    windows_seen: u32,
    // ── live window-cost calibration (hardware cost; stable across requests) ──
    /// Total timing samples observed; gated to ≥TIMING_WARMUP before calibrating.
    timing_samples: u32,
    /// Per-n EMA of full-window wall-time in ms (draft+heads+verify), indexed by
    /// n_verify (= 1 + drafted block), slots 2..=MAX_DEPTH+1.
    t_window_by_n: [f32; MAX_DEPTH + 2],
    /// True once the window-cost curve has been fit; preserved across reset().
    calibrated: bool,
    /// Marginal per-position window cost (ms) = slope of the window-cost curve,
    /// clamped to ≥0 (a flat curve lets the block climb to the acceptance depth).
    dt: f32,
    /// Single-block window cost (ms) = intercept of the window-cost curve.
    t_ar: f32,
    /// True once dt/t_ar are usable (live-calibrated or test-seeded). Until then
    /// the argmax is disabled and the block stays at default_block.
    cost_ready: bool,
}

/// Deepest block the controller can model, and the size of every per-depth
/// array. This used to be a bare `8` written into three places at once --
/// `s_hit`/`s_tot` sized 9, the `(2..10)` timing window, and `max_tried.min(8)`
/// inside the argmax. The last of those is why `max_block` above 8 did nothing:
/// the argmax could not RETURN a larger block however wide the caller allowed.
///
/// That was invisible while only a DFlash drafter drove this, since trained
/// blocks are <= 8. The drafter-free n-gram path routinely wants 16, so it hit
/// all three caps at once and settled around 4. Raised to 32 (n_verify = block
/// + 1, so this covers a 31-token spine) and derived from one constant.
const MAX_DEPTH: usize = 32;

/// Skip the first few windows so the block doesn't react to bootstrap noise.
const WARMUP_WINDOWS: u32 = 6;
/// Minimum timing samples before attempting window-curve calibration.
const TIMING_WARMUP: u32 = 16;
/// Windows each ramp block is held so the histogram can collect survival samples
/// at that depth; after ramp_end the argmax takes over.
const RAMP_HOLD: u32 = 2;
/// Samples a depth needs before its survival estimate is trusted over the
/// shallower one. Below this the estimate is thin AND biased: the only windows
/// that reached the depth were ramp windows, when the draft source was cold.
const MIN_DEPTH_SAMPLES: f32 = 8.0;

impl BlockController {
    pub fn new(default_block: usize, min_block: usize, max_block: usize, p_star: f32) -> Self {
        let default_block = default_block.clamp(min_block, max_block);
        Self {
            block: default_block,
            default_block,
            min_block,
            max_block,
            max_tried: default_block,
            s_hit: [0.0f32; MAX_DEPTH + 1],
            s_tot: [0.0f32; MAX_DEPTH + 1],
            windows_seen: 0,
            timing_samples: 0,
            t_window_by_n: [0.0f32; MAX_DEPTH + 2],
            calibrated: false,
            // Dormant cost prior: only the dt/t_ar RATIO drives the argmax, and it
            // stays disabled (cost_ready=false) until live window timing refines
            // these into real milliseconds. Seeded from the caller's p* prior so the
            // ratio is sane if ever consulted before calibration (assume ~100ms AR).
            t_ar: 100.0,
            dt: p_star * 100.0,
            cost_ready: false,
        }
    }

    pub fn block(&self) -> usize {
        self.block
    }

    pub fn reset(&mut self) {
        // Reset only request-specific state. Calibration (dt, t_ar, cost_ready,
        // timing_samples, t_window_by_n, calibrated) is a thermal-invariant hardware
        // cost — calibrate once, reuse across requests.
        self.block = self.default_block;
        self.max_tried = self.default_block;
        self.windows_seen = 0;
        self.s_hit = [0.0f32; MAX_DEPTH + 1];
        self.s_tot = [0.0f32; MAX_DEPTH + 1];
    }

    /// Observe one window's full wall-time (draft+heads+verify). Accumulates per-n
    /// EMAs and fits the line `t_window(n) ≈ t_ar + (n−1)·Δt` once TIMING_WARMUP
    /// samples span ≥2 distinct n. The slope is clamped to ≥0: a flat or slightly
    /// negative measured slope means the per-window wall time barely grows with the
    /// block (a cheap-verify arch whose fixed drafter/launch overhead dominates), in
    /// which case the argmax should be free to climb toward the acceptance depth
    /// rather than be blocked by phantom marginal cost. Only a clearly-too-steep fit
    /// (Δt > t_ar/2, i.e. a thermal spike) is rejected. Preserved across reset().
    pub fn observe_timing(&mut self, t_window_ms: f32, n_verify: usize) {
        if (2..=MAX_DEPTH + 1).contains(&n_verify) && t_window_ms > 0.0 {
            let slot = &mut self.t_window_by_n[n_verify];
            *slot = if *slot == 0.0 {
                t_window_ms
            } else {
                0.7 * *slot + 0.3 * t_window_ms
            };
        }
        self.timing_samples = self.timing_samples.saturating_add(1);
        // Refit while the ramp is still widening the observed range, instead of
        // freezing the first fit that clears TIMING_WARMUP. That threshold lands
        // MID-ramp, so the frozen curve was fit over whatever narrow slice had
        // been swept -- one measured run calibrated off n2..n7, read a NEGATIVE
        // slope from 6 noisy samples, clamped it to dt=0, and carried that for
        // the whole request; the next got n3..n17 and a completely different
        // answer. Same workload, same hardware, different cost model, decided by
        // where the warmup boundary happened to fall.
        let ramp_done = self.windows_seen >= WARMUP_WINDOWS + 2 * self.max_block as u32;
        if (self.calibrated && ramp_done) || self.timing_samples < TIMING_WARMUP {
            return;
        }
        let lo = (2..=MAX_DEPTH + 1).find(|&n| self.t_window_by_n[n] > 0.0);
        let hi = (2..=MAX_DEPTH + 1)
            .rev()
            .find(|&n| self.t_window_by_n[n] > 0.0);
        if let (Some(n_lo), Some(n_hi)) = (lo, hi) {
            // If the controller never visits ≥2 distinct n_verify (block pinned),
            // n_hi > n_lo never holds and cost_ready stays false — the block then
            // safely stays at default_block for the process lifetime.
            if n_hi > n_lo {
                let slope =
                    (self.t_window_by_n[n_hi] - self.t_window_by_n[n_lo]) / (n_hi - n_lo) as f32;
                // Clamp negative/flat slope to 0 → "cost is flat in block", so the
                // argmax climbs to the survival-supported depth (a shallow-acceptance
                // arch still settles low because τ saturates). Anchor t_ar at the
                // cheapest measured point so the clamp keeps a sane intercept.
                let dt = slope.max(0.0);
                let t_ar = self.t_window_by_n[n_lo] - dt * (n_lo as f32 - 1.0);
                if t_ar > 0.0 && dt / t_ar <= 0.5 {
                    self.dt = dt;
                    self.t_ar = t_ar;
                    self.cost_ready = true;
                    self.calibrated = true;
                    eprintln!(
                        "[dspark] cost calibrated: dt={:.2}ms t_ar={:.1}ms (ratio={:.3}, n{}={:.1}ms n{}={:.1}ms)",
                        dt, t_ar, dt / t_ar, n_lo, self.t_window_by_n[n_lo], n_hi, self.t_window_by_n[n_hi]
                    );
                }
            }
        }
    }

    /// Observe one spec window's acceptance depth and re-decide the block via the
    /// cost-model argmax. During the ramp phase (post-warmup, pre-ramp_end) the block
    /// sweeps min→max to seed the window-cost calibration and the survival estimate.
    /// After ramp_end the argmax drives the block for the remainder of the request.
    pub fn observe(&mut self, accept_len: usize, n_proposed: usize) {
        // Accumulate survival counts ONLY for depths we actually drafted (k ≤
        // n_proposed): for those k, `accept_len ≥ k` is a real observation. Depths above
        // n_proposed are unobservable this window, so their counts don't grow — S(k)
        // retains its last value (the cap-trap fix; the old decaying P(accept==k)
        // histogram forgot the ramp's deep-survival samples so a low-settled block
        // could never re-discover that drafting deeper pays).
        let depth = n_proposed.min(MAX_DEPTH);
        for k in 1..=depth {
            self.s_tot[k] += 1.0;
            if accept_len >= k {
                self.s_hit[k] += 1.0;
            }
        }
        self.windows_seen += 1;
        if self.windows_seen < WARMUP_WINDOWS {
            return;
        }
        let ramp_end = WARMUP_WINDOWS + 2 * self.max_block as u32;
        if self.windows_seen < ramp_end {
            // Sweep block min→max to seed calibration (≥2 distinct n_verify) AND survival
            // at every depth. Without this the block never varies and calibration deadlocks.
            let step = (self.windows_seen - WARMUP_WINDOWS) / RAMP_HOLD;
            self.block = (self.min_block + step as usize).min(self.max_block);
            self.max_tried = self.block.max(self.max_tried);
            return;
        }
        // Settle at the cost-model optimum (calibration completed during the ramp).
        if self.cost_ready {
            self.block = self.argmax_block();
        } else {
            self.block = self.default_block; // fallback if calibration never completed
        }
    }

    /// Survival estimate at depth `k`, made monotone and optimistic-under-
    /// uncertainty. Both corrections fix measured failures:
    ///
    /// MONOTONE: S is `P(accept_len >= k)`, which cannot rise with k. The raw
    /// ratios do, because each depth conditions on a DIFFERENT subset (windows
    /// that drafted >= k), and the argmax then sums them as one curve. A real
    /// trace read `1:0.89 2:0.85 3:0.94` -- impossible for a survival function.
    ///
    /// OPTIMISTIC: a depth the block has stopped reaching stops accumulating
    /// samples and keeps whatever estimate it last had. During the ramp the
    /// n-gram store is still cold, so that frozen value is pessimistic, and it
    /// can never be revised because nothing drafts that deep again. That is a
    /// downward RATCHET, not the "cap-trap fix" the old comment claimed: a
    /// measured run walked the block 15 -> 14 -> 13 while acceptance at those
    /// depths was ~0.9. Under `MIN_DEPTH_SAMPLES`, carry the shallower estimate
    /// forward instead of trusting a thin one, so an unexplored depth is
    /// re-tried rather than condemned.
    fn survival_at(&self, k: usize, prev: f32) -> f32 {
        if k >= self.s_tot.len() || self.s_tot[k] < MIN_DEPTH_SAMPLES {
            return prev;
        }
        (self.s_hit[k] / self.s_tot[k]).min(prev)
    }

    /// argmax over N ∈ [min_block, max_block] of τ(N)/(t_ar + (N-1)·Δt).
    /// τ(N) = 1 + Σ_{k=1..N-1} S[k]: a block of N verifies the seed plus N-1
    /// DRAFTED positions, so the deepest observable depth is N-1.
    fn argmax_block(&self) -> usize {
        if self.t_ar <= 0.0 || self.dt < 0.0 {
            return self.default_block;
        }
        // The old loop ran `for n in 1..=max_tried` and indexed `s_hit[n]`,
        // conflating the BLOCK with the drafted DEPTH. `observe` is called with
        // `n_proposed = drafted.len() = block - 1`, so at `block == max_block`
        // the top index never accumulated a single sample: every trace line read
        // `16:0.00`, tau could not grow there, and the largest block was
        // structurally unable to win -- the fixed configuration it was being
        // compared against was not in its own reachable set.
        let mut surv = 1.0f32;
        let mut tau = 1.0f32;
        for k in 1..self.min_block {
            surv = self.survival_at(k, surv);
            tau += surv;
        }
        let mut best_n = self.min_block;
        let mut best_score = f32::MIN;
        for n in self.min_block..=self.max_block.min(MAX_DEPTH + 1) {
            if n > self.min_block {
                surv = self.survival_at(n - 1, surv);
                tau += surv;
            }
            let window_ms = self.t_ar + (n as f32 - 1.0) * self.dt; // > 0 by the guard above
            let score = tau / window_ms;
            if score > best_score {
                best_score = score;
                best_n = n;
            }
        }
        let chosen = best_n.clamp(self.min_block, self.max_block);
        if std::env::var("HIPFIRE_DSPARK_TRACE").is_ok() {
            let surv: Vec<String> = (1..=self.max_tried.min(MAX_DEPTH))
                .map(|n| {
                    if self.s_tot[n] > 0.0 {
                        format!("{n}:{:.2}", self.s_hit[n] / self.s_tot[n])
                    } else {
                        format!("{n}:--")
                    }
                })
                .collect();
            eprintln!(
                "[dspark-trace] w={} max_tried={} best_n={} chosen={} dt={:.3} t_ar={:.1} S=[{}]",
                self.windows_seen,
                self.max_tried,
                best_n,
                chosen,
                self.dt,
                self.t_ar,
                surv.join(" ")
            );
        }
        chosen
    }

    #[cfg(test)]
    fn set_cost_for_test(&mut self, dt: f32, t_ar: f32) {
        self.dt = dt;
        self.t_ar = t_ar;
        self.cost_ready = true;
    }

    #[cfg(test)]
    fn cost_for_test(&self) -> (f32, f32, bool) {
        (self.dt, self.t_ar, self.cost_ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // DS4-like: expensive verify (dt large) + survival that SATURATES at 2
    // (accept_len ∈ {0,1,2}). τ stops growing past 2, so the argmax settles low.
    // Run well past ramp_end (ramp sweeps min→max in 2*max_block=14 post-warmup
    // windows; we run 200 total so the argmax has long settled).
    #[test]
    fn settles_low_when_survival_saturates_and_verify_expensive() {
        let mut c = BlockController::new(2, 1, 7, 0.0);
        c.set_cost_for_test(15.0, 85.0); // DS4-like
        for i in 0..200 {
            c.observe([0, 1, 2, 2][i % 4], 5); // depth ~1.5, never > 2
        }
        // Depth saturates at 2 accepted DRAFTED tokens, and a block of N
        // verifies N-1 drafted positions, so the cheapest block that can still
        // collect both is 3 (seed + 2). This asserted `<=2` while `argmax_block`
        // summed survival over `1..=n` instead of `1..=n-1` -- the same
        // block-vs-depth conflation that made the top block unreachable in
        // production. The expectation moved by one because the model did.
        assert!((2..=3).contains(&c.block()), "got {}", c.block());
    }

    // qwen3-like: cheap verify (dt small) + a drafter that accepts the whole drafted
    // block (survival deep, cap always hit). The ramp seeds survival at all depths;
    // after ramp_end the cost model rewards over-drafting and settles high.
    #[test]
    fn climbs_high_when_survival_deep_and_verify_cheap() {
        let mut c = BlockController::new(2, 1, 7, 0.0);
        c.set_cost_for_test(4.0, 33.0); // qwen3-like
                                        // Always accept the whole drafted block, so all depths are rewarded.
        for _ in 0..200 {
            let b = c.block();
            c.observe(b.min(7), b);
        }
        assert!(
            c.block() >= 5,
            "cheap verify + deep accept should climb high, got {}",
            c.block()
        );
    }

    // Cost sensitivity: SAME (decreasing) survival, cheap vs expensive verify -> the
    // cheaper verify justifies a larger block. accept depth cycles 1..4 out of 7
    // drafted, so S saturates (S(5+)=0) → τ tops out ~4 and the marginal cost alone
    // decides how deep to go (cheap → 4, expensive → 1).
    #[test]
    fn cheaper_verify_picks_larger_block() {
        let mk = |dt: f32| {
            let mut c = BlockController::new(2, 1, 7, 0.0);
            c.set_cost_for_test(dt, 50.0);
            for i in 0..400 {
                c.observe([1, 2, 3, 4][i % 4], 7);
            }
            c.block()
        };
        let cheap = mk(2.0);
        let expensive = mk(20.0);
        assert!(
            cheap > expensive,
            "cheap {} should exceed expensive {}",
            cheap,
            expensive
        );
    }

    // Fit dt/t_ar from the window-cost curve: t_window(n) ≈ t_ar + (n-1)·Δt.
    // n=2 → 90ms, n=6 → 150ms: dt=(150-90)/4=15, t_ar=90-15=75; flips cost_ready.
    #[test]
    fn calibrates_dt_t_ar_from_window_curve() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        for _ in 0..30 {
            c.observe_timing(90.0, 2);
            c.observe_timing(150.0, 6);
        }
        let (dt, t_ar, ready) = c.cost_for_test();
        assert!(ready, "window-curve fit should flip cost_ready");
        assert!((dt - 15.0).abs() < 0.5, "dt={}", dt);
        assert!((t_ar - 75.0).abs() < 1.0, "t_ar={}", t_ar);
    }

    // A too-STEEP fit (Δt > t_ar/2, ratio > 0.5 — a thermal spike) must be REJECTED:
    // cost_ready stays false (the block then safely stays at default).
    #[test]
    fn rejects_out_of_range_fit() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        // t_w[2]=100, t_w[6]=300 → dt=50, t_ar=50, ratio=1.0 > 0.5 → reject.
        for _ in 0..30 {
            c.observe_timing(100.0, 2);
            c.observe_timing(300.0, 6);
        }
        assert!(
            !c.cost_for_test().2,
            "out-of-range fit must not flip cost_ready"
        );
    }

    // Cheap-verify arch (qwen3): window cost is ~flat/slightly-decreasing in block.
    // The slope clamps to 0 (NOT rejected by an old lower ratio floor), so cost_ready
    // flips with dt=0 and the argmax is free to climb. This is the fix for the qwen3
    // "stuck at min block" regression — a flat window curve must calibrate, not fall
    // back to default. t_w[2]=80, t_w[8]=68 → slope=−2 → dt clamped 0, t_ar=80.
    #[test]
    fn flat_window_cost_calibrates_dt_zero() {
        let mut c = BlockController::new(2, 1, 7, 0.05);
        for _ in 0..30 {
            c.observe_timing(80.0, 2);
            c.observe_timing(68.0, 8);
        }
        let (dt, t_ar, ready) = c.cost_for_test();
        assert!(ready, "flat window curve must calibrate (not fall back)");
        assert!(
            dt.abs() < 1e-6,
            "flat/negative slope must clamp to 0, got dt={dt}"
        );
        assert!((t_ar - 80.0).abs() < 1.0, "t_ar={t_ar}");
    }

    // reset() restores request state (block, histogram) but PRESERVES the calibrated
    // window cost (thermal-invariant hardware ratio).
    #[test]
    fn reset_preserves_calibration() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        for _ in 0..30 {
            c.observe_timing(90.0, 2);
            c.observe_timing(150.0, 6);
        }
        let cost = c.cost_for_test();
        assert!(cost.2, "should be calibrated");
        // Pollute request-specific state.
        for _ in 0..80 {
            c.observe(0, 4);
        }
        c.reset();
        assert_eq!(
            c.cost_for_test(),
            cost,
            "reset() must preserve dt/t_ar/cost_ready"
        );
        assert_eq!(c.block(), 3, "block returns to default after reset");
    }

    // n_proposed == 0 (degenerate window) must not panic, and all-reject settles the
    // block to the minimum without indexing OOB.
    #[test]
    fn zero_proposed_is_safe() {
        let mut c = BlockController::new(3, 1, 7, 0.18);
        c.set_cost_for_test(10.0, 50.0);
        for _ in 0..50 {
            c.observe(0, 0); // n_proposed=0; all-reject
        }
        assert!((1..=7).contains(&c.block()), "got {}", c.block());
    }
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    /// The widened `MAX_DEPTH` moved three array bounds and two loop bounds at
    /// once. Drive the controller across widths on both sides of `MAX_DEPTH`,
    /// with adversarial accept lengths, and assert it neither panics nor returns
    /// a block outside `[min_block, max_block]`. Indexing is the risk: `observe`
    /// writes `s_hit[k]`, `observe_timing` writes `t_window_by_n[n_verify]`, and
    /// the argmax reads `survival_at(n - 1)` — three different arrays whose
    /// lengths are all derived from one constant.
    #[test]
    fn never_panics_or_escapes_its_range() {
        for max_block in [2usize, 3, 8, 16, 32, MAX_DEPTH, MAX_DEPTH + 1, 64] {
            for min_block in [1usize, 2] {
                if min_block > max_block {
                    continue;
                }
                let mut c = BlockController::new(max_block, min_block, max_block, 0.2);
                c.set_cost_for_test(0.5, 20.0);
                for i in 0..400 {
                    // Accept lengths that sweep past the cap, including 0 and
                    // values above max_block, plus proposals above MAX_DEPTH.
                    let proposed = (i % (max_block + 8)).max(1);
                    let accepted = (i * 7) % (proposed + 3);
                    c.observe_timing((10 + i % 40) as f32, proposed + 1);
                    c.observe(accepted, proposed);
                    let b = c.block();
                    assert!(
                        b >= min_block && b <= max_block,
                        "block {b} escaped [{min_block}, {max_block}] at i={i}"
                    );
                }
                c.reset();
                assert!(c.block() >= min_block && c.block() <= max_block);
            }
        }
    }

    /// A spine longer than the controller can model must not index past the
    /// survival arrays. `observe` clamps with `.min(MAX_DEPTH)`; this pins it.
    #[test]
    fn proposals_far_above_max_depth_are_clamped() {
        let mut c = BlockController::new(16, 2, 16, 0.2);
        c.set_cost_for_test(0.5, 20.0);
        for _ in 0..100 {
            c.observe_timing(15.0, 4096);
            c.observe(4095, 4096);
        }
        assert!((2..=16).contains(&c.block()));
    }
}
