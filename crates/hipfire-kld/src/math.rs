// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure fp64 reduction + KLD scoring math.
//!
//! Faithful extraction of the two implementations that previously lived
//! (duplicated) in `build_kld_ref_hipfire::log_softmax_top_k_row` and
//! `eval_hipfire::score_position`. The reference reduction and the candidate
//! scoring now share this one module, so they cannot diverge.

use crate::refblock::RefBlock;

/// Log-partition `log Z = log Σ_i exp(logit_i)`, computed in fp64 with the
/// standard max-shift for numerical stability. The shift is folded back in, so
/// the result is the true (unshifted) log-sum-exp.
pub fn log_z(logits: &[f32]) -> f64 {
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_logit {
            max_logit = v;
        }
    }
    if !max_logit.is_finite() {
        // All-empty or all -inf: defined as the max (handles len()==0 → -inf).
        return max_logit as f64;
    }
    let mut sum_exp = 0.0f64;
    for &v in logits {
        sum_exp += ((v - max_logit) as f64).exp();
    }
    (max_logit as f64) + sum_exp.ln()
}

/// Top-K log-softmax reduction of a full logit row — the reference-side
/// representation written into a `.kldref`. Returns the K highest-logit token
/// ids (descending, ties broken by ascending id), their log-probabilities, and
/// the residual probability mass not captured by the top-K (`1 - Σ p_topk`,
/// clamped to ≥0).
#[derive(Debug, Clone, PartialEq)]
pub struct TopKReduction {
    pub indices: Vec<u32>,
    pub log_probs: Vec<f32>,
    pub residual_mass: f32,
}

pub fn top_k_log_softmax(logits: &[f32], top_k: usize) -> TopKReduction {
    let k = top_k.min(logits.len());
    let lz = log_z(logits);

    // Rank by logit descending, tie-break by id ascending (matches the
    // historical heap/sort order so byte-identical refs reproduce).
    //
    // Selection, not a sort. This runs once per scored position — 1023 per
    // 2048-token chunk — and fully ordering a 248320-entry vocabulary to keep
    // 256 of it cost ~5.8 ms each, ~6 s per chunk, which was 70% of a reference
    // build's wall time with the GPU idle (kernel trace: 3.21 s of GPU against
    // 10.5 s of chunk). Two costs: `sort_by` is the STABLE sort, so it also
    // allocates a scratch buffer per call, and the comparator indexes back into
    // `logits`, making every comparison two random loads over a 1 MB array.
    //
    // The comparator is a total order (`total_cmp` then id), so selecting the
    // first k and ordering only those reproduces the full sort's prefix
    // exactly — byte-identical references, no tie ambiguity to preserve.
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
    let mut order: Vec<(f32, u32)> = logits.iter().copied().zip(0u32..).collect();
    if k < order.len() {
        order.select_nth_unstable_by(k, &cmp);
        order.truncate(k);
    }
    order.sort_unstable_by(&cmp);

    let mut indices = vec![0u32; top_k];
    let mut log_probs = vec![f32::NEG_INFINITY; top_k];
    let mut top_p_sum = 0.0f64;
    for (i, &(logit, idx)) in order.iter().take(k).enumerate() {
        let log_p = ((logit as f64) - lz) as f32;
        indices[i] = idx;
        log_probs[i] = log_p;
        top_p_sum += (log_p as f64).exp();
    }
    let residual_mass = (1.0 - top_p_sum).max(0.0) as f32;
    TopKReduction {
        indices,
        log_probs,
        residual_mass,
    }
}

/// Per-position score: forward KLD `D(P_ref ‖ Q_cand)` plus the candidate NLL of
/// the actual next token.
///
/// **Wide accumulate, narrow store.** Both fields are computed via f64
/// accumulators (the partition sum and the K-term cross-entropy need the wide
/// accumulator — see [`log_z`]) but are returned as `f32`: a KLD of ~1e-3 needs
/// ~4 significant figures and f32 gives 7, so storing f64 here would only bloat
/// the high-volume per-position arrays (mean/p99 reductions, per-layer
/// divergence matrices) without adding information. f64 stays an accumulator
/// type, not a storage type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionScore {
    /// `Σ_{i∈topK(P_ref)} P_ref(i)·(log P_ref(i) − log Q_cand(i))` plus the
    /// residual cross-term. Clamped at 0 (Gibbs; tiny negatives are fp64 roundoff).
    pub kld: f32,
    /// `−log Q_cand(actual_next)`, or `None` if `actual_next` is out of range.
    pub nll: Option<f32>,
    /// Whether the candidate's argmax is the reference's argmax — the "did the
    /// greedily-decoded token survive quantization" question, which mean KLD
    /// cannot answer: a distribution can move a long way without the top token
    /// changing, and can barely move while flipping a near-tie.
    ///
    /// `None` when the reference block has no usable top-1 (empty or padded
    /// top-K, or a top-1 id outside the candidate's vocabulary), so "no
    /// reference to agree with" never counts as a disagreement.
    pub argmax_match: Option<bool>,
}

/// Index of the largest logit, ties broken by lowest id.
///
/// The tie-break MUST match [`top_k_log_softmax`]'s ordering (`total_cmp`
/// descending, then id ascending) or a model scored against its own reference
/// would report less than 100% agreement wherever two logits are exactly equal.
fn argmax_lowest_id(logits: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        // Strictly-greater only: iterating ascending, the first of a tie wins.
        if best.is_none_or(|(_, b)| v.total_cmp(&b).is_gt()) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

/// Score candidate logits against a reference block for one position.
///
/// Mirrors the historical `eval_hipfire::score_position` exactly: KLD over the
/// reference's top-K support with a single lumped residual cross-term, and NLL
/// from the candidate's log-softmax at `actual_next`. `log Z` of the candidate
/// is computed once here.
pub fn score_position(
    ref_block: &RefBlock,
    cand_logits: &[f32],
    actual_next: usize,
) -> PositionScore {
    let lz = log_z(cand_logits);

    let mut kld = 0.0f64;
    let mut sum_p_cand_at_ref_top = 0.0f64;
    for (j, &ref_idx) in ref_block.top_indices.iter().enumerate() {
        let ref_idx = ref_idx as usize;
        if ref_idx >= cand_logits.len() {
            continue;
        }
        let log_p_ref = ref_block.top_log_probs[j] as f64;
        if !log_p_ref.is_finite() {
            // Padding slot (fewer than top_k real entries): skip.
            continue;
        }
        let log_p_cand = (cand_logits[ref_idx] as f64) - lz;
        let p_ref = log_p_ref.exp();
        let p_cand = log_p_cand.exp();
        kld += p_ref * (log_p_ref - log_p_cand);
        sum_p_cand_at_ref_top += p_cand;
    }

    let sum_p_residual_ref = ref_block.residual_mass as f64;
    let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
    if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
        kld += sum_p_residual_ref * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
    }
    // KLD ≥ 0 by Gibbs' inequality. The residual cross-term plus ~K-term fp64
    // sums accumulate cancellation roundoff up to ~1e-8 when P_ref ≈ Q_cand
    // (the self-consistency limit); a real math bug yields O(0.1+) negatives.
    // Guard against the latter, clamp the former.
    debug_assert!(kld >= -1e-6, "negative KLD beyond fp roundoff: {kld}");

    // Narrow at the boundary: f64 was the accumulator, f32 is the stored value.
    let kld = kld.max(0.0) as f32;
    let nll = if actual_next < cand_logits.len() {
        Some((-((cand_logits[actual_next] as f64) - lz)) as f32)
    } else {
        None
    };
    // Top-1 agreement. The reference's argmax is slot 0 of its top-K (built
    // logit-descending), so this costs one linear pass over the candidate
    // logits — the same order as the `log_z` above, and far cheaper than the
    // top-K selection the reference build already pays per position.
    let argmax_match = match (
        ref_block.top_indices.first(),
        ref_block.top_log_probs.first(),
    ) {
        (Some(&ref_top), Some(&lp)) if lp.is_finite() && (ref_top as usize) < cand_logits.len() => {
            argmax_lowest_id(cand_logits).map(|c| c == ref_top as usize)
        }
        _ => None,
    };
    PositionScore {
        kld,
        nll,
        argmax_match,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the metric lives or dies by: scored against its OWN
    /// reference a model must agree with itself at every position, including
    /// where two logits tie. If the tie-break here ever drifts from
    /// `top_k_log_softmax`'s, a self-score silently reports < 100% and every
    /// comparison built on it is off by that amount.
    #[test]
    fn self_score_agrees_everywhere_including_exact_ties() {
        // Two exact ties at the top (ids 1 and 3), which the shared ordering
        // must resolve to the LOWER id on both sides.
        let logits = [0.5f32, 4.0, -1.0, 4.0, 2.0];
        let red = top_k_log_softmax(&logits, 3);
        assert_eq!(
            red.indices[0], 1,
            "reference argmax must be the lower tied id"
        );
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        let s = score_position(&rb, &logits, 0);
        assert_eq!(s.argmax_match, Some(true));
        assert!(s.kld.abs() < 1e-6, "self KLD should be ~0, got {}", s.kld);
    }

    /// A distribution that moves a long way without changing the argmax scores
    /// as a match; one that barely moves but flips a near-tie does not. This is
    /// the whole reason the metric exists next to mean KLD rather than instead
    /// of it.
    #[test]
    fn argmax_match_is_independent_of_how_far_the_distribution_moved() {
        let reference = [3.0f32, 1.0, 0.0, -1.0];
        let red = top_k_log_softmax(&reference, 4);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };

        // Mass redistributed hard, top token unchanged: big KLD, still a match.
        let flattened = [1.2f32, 1.0, 0.9, 0.5];
        let moved = score_position(&rb, &flattened, 0);
        assert_eq!(moved.argmax_match, Some(true));
        assert!(moved.kld > 0.1, "expected a large KLD, got {}", moved.kld);

        // Near-tie flipped: small KLD, not a match.
        let flipped = [2.99f32, 3.0, 0.0, -1.0];
        let flip = score_position(&rb, &flipped, 0);
        assert_eq!(flip.argmax_match, Some(false));
        assert!(
            flip.kld < moved.kld,
            "flip {} should move less than {}",
            flip.kld,
            moved.kld
        );
    }

    /// "No reference to agree with" must not be counted as a disagreement:
    /// padded top-K slots and out-of-vocabulary ids yield None, not Some(false).
    #[test]
    fn an_unusable_reference_top1_reports_none() {
        let cand = [1.0f32, 2.0, 3.0];
        let padded = score_position(
            &RefBlock {
                top_indices: &[0],
                top_log_probs: &[f32::NEG_INFINITY],
                residual_mass: 1.0,
            },
            &cand,
            0,
        );
        assert_eq!(padded.argmax_match, None);

        let out_of_range = score_position(
            &RefBlock {
                top_indices: &[99],
                top_log_probs: &[-0.1],
                residual_mass: 0.0,
            },
            &cand,
            0,
        );
        assert_eq!(out_of_range.argmax_match, None);

        let empty = score_position(
            &RefBlock {
                top_indices: &[],
                top_log_probs: &[],
                residual_mass: 1.0,
            },
            &cand,
            0,
        );
        assert_eq!(empty.argmax_match, None);
    }

    /// Selection must reproduce the full sort's prefix exactly — references
    /// built before and after the change have to stay byte-identical, so the
    /// tie order (logit desc, id asc) is part of the contract, not an artifact.
    #[test]
    fn top_k_log_softmax_matches_full_sort_prefix() {
        fn reference(logits: &[f32], top_k: usize) -> (Vec<u32>, Vec<f32>) {
            let k = top_k.min(logits.len());
            let lz = log_z(logits);
            let mut order: Vec<u32> = (0..logits.len() as u32).collect();
            order.sort_by(|&a, &b| {
                logits[b as usize]
                    .total_cmp(&logits[a as usize])
                    .then_with(|| a.cmp(&b))
            });
            let mut idx = vec![0u32; top_k];
            let mut lp = vec![f32::NEG_INFINITY; top_k];
            for i in 0..k {
                let id = order[i];
                idx[i] = id;
                lp[i] = ((logits[id as usize] as f64) - lz) as f32;
            }
            (idx, lp)
        }

        // Tie-heavy: 11 distinct values over 400 slots puts the k-boundary
        // inside a run of equal logits, where the id tiebreak is what decides.
        let tied: Vec<f32> = (0..400).map(|i| ((i * 31) % 11) as f32).collect();
        // Plus an all-distinct case and one shorter than k.
        let distinct: Vec<f32> = (0..300).map(|i| (i as f32) * -0.37 + 5.0).collect();
        let short = vec![1.0f32, -2.0, 0.5];

        for logits in [&tied, &distinct, &short] {
            for &k in &[1usize, 8, 256] {
                let got = top_k_log_softmax(logits, k);
                let (idx, lp) = reference(logits, k);
                assert_eq!(got.indices, idx, "indices k={k} len={}", logits.len());
                assert_eq!(got.log_probs, lp, "log_probs k={k} len={}", logits.len());
            }
        }
    }

    #[test]
    fn log_z_matches_naive() {
        let logits = [0.0f32, 1.0, 2.0, -1.0];
        // log(e^0 + e^1 + e^2 + e^-1)
        let naive = (1.0f64 + std::f64::consts::E + (2.0f64).exp() + (-1.0f64).exp()).ln();
        assert!((log_z(&logits) - naive).abs() < 1e-12);
    }

    #[test]
    fn top_k_picks_highest_and_orders_descending() {
        let logits = [0.0f32, 3.0, 1.0, 2.0];
        let r = top_k_log_softmax(&logits, 2);
        assert_eq!(r.indices, vec![1, 3]); // logits 3.0 then 2.0
                                           // log-probs descending, finite
        assert!(r.log_probs[0] > r.log_probs[1]);
        assert!(r.residual_mass > 0.0 && r.residual_mass < 1.0);
    }

    #[test]
    fn top_k_full_support_has_zero_residual() {
        let logits = [0.0f32, 1.0, 2.0];
        let r = top_k_log_softmax(&logits, 3);
        assert!(r.residual_mass < 1e-6, "residual {}", r.residual_mass);
    }

    #[test]
    fn top_k_breaks_ties_by_ascending_id() {
        let logits = [2.0f32, 2.0, 2.0, 0.0];
        let r = top_k_log_softmax(&logits, 2);
        assert_eq!(r.indices, vec![0, 1]); // equal logits → lowest ids first
    }

    /// THE invariant the whole refactor enforces: scoring a candidate against a
    /// reference built from *identical* logits yields KLD ≈ 0. This is the
    /// self-consistency property — at the math level it must be exact-to-roundoff.
    #[test]
    fn self_consistency_is_zero() {
        let logits: Vec<f32> = (0..512)
            .map(|i| ((i * 7 % 23) as f32) * 0.31 - 3.0)
            .collect();
        let red = top_k_log_softmax(&logits, 64);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        let s = score_position(&rb, &logits, 5);
        assert!(
            s.kld < 1e-9,
            "self-consistency KLD should be ~0, got {}",
            s.kld
        );
    }

    #[test]
    fn kld_is_positive_for_different_distribution() {
        let ref_logits: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05).collect();
        let red = top_k_log_softmax(&ref_logits, 32);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        // A clearly different candidate (reversed slope).
        let cand_logits: Vec<f32> = (0..128).map(|i| -(i as f32) * 0.05).collect();
        let s = score_position(&rb, &cand_logits, 0);
        assert!(s.kld > 0.1, "expected sizeable KLD, got {}", s.kld);
    }

    #[test]
    fn nll_of_argmax_is_smallest() {
        let logits = [0.0f32, 5.0, 1.0];
        let red = top_k_log_softmax(&logits, 3);
        let rb = RefBlock {
            top_indices: &red.indices,
            top_log_probs: &red.log_probs,
            residual_mass: red.residual_mass,
        };
        let nll_argmax = score_position(&rb, &logits, 1).nll.unwrap();
        let nll_other = score_position(&rb, &logits, 0).nll.unwrap();
        assert!(nll_argmax < nll_other);
        assert!(nll_argmax > 0.0);
    }
}
