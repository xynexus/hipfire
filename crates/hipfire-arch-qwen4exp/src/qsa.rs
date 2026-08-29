// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen Sparse Attention — the block SELECTION, as a CPU reference.
//!
//! The indexer does not produce activations; it produces a choice of which
//! micro-blocks of the KV cache the main attention may see. So its failure mode is
//! not a small numeric perturbation — it attends to the wrong tokens, and the
//! damage is unbounded and non-local. That is why the indexer's own weights are
//! kept at source fidelity, and why the selection has an oracle before it has a
//! kernel.
//!
//! Semantics from `Qwen4ExpTextQSAIndexer.forward`:
//!
//! ```text
//! block_topk = indexer_budget / compress_ratio          (2048 tokens / 4 = 512 blocks)
//! q          = rope(q_layernorm(q))                     at the QUERY position
//! block_key  = rope(k_layernorm(mean(K[block])))        at the block's FIRST position
//! score[b]   = sum_h relu(q[h] . block_key[b]) / sqrt(head_dim)
//! select     = topk(score, block_topk) -> each block's tokens
//!            + the ragged tail past the last complete block, ALWAYS visible
//! ```
//!
//! Four things here are silent-wrong if guessed: keys are **mean-pooled** over the
//! block (not the first token, not a learned pool); RoPE is applied to the pooled
//! key at the block's **first** position; the per-head scores are **ReLU'd before
//! being summed**, with no learned per-head weight; and the ragged tail is
//! **unconditionally visible** rather than competing for budget.
//!
//! The property this module exists to protect: **below the budget the selection is
//! everything**, so sparse attention is dense by construction and can be
//! differenced bit-for-bit against a dense reference. That is the only free exact
//! oracle in this port — see the scope doc's §8.

/// Indexer geometry, mirroring the config.
#[derive(Debug, Clone, Copy)]
pub struct QsaParams {
    pub n_heads: usize,
    pub head_dim: usize,
    /// Budget in TOKENS.
    pub budget: usize,
    /// Tokens per micro-block.
    pub compress_ratio: usize,
}

impl QsaParams {
    /// Blocks selected per query. The budget is in tokens; the selection is over
    /// blocks, and the two differ by exactly `compress_ratio`.
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }

    /// Below this many visible tokens the selection cannot exclude anything, so
    /// sparse and dense attention must agree EXACTLY.
    ///
    /// `block_topk` whole blocks plus the largest possible ragged tail, which is
    /// `compress_ratio - 1` tokens.
    pub fn dense_below(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }
}

/// Mean-pool a block of keys. `keys` is `[n_tokens, head_dim]` row-major.
pub fn pool_block(keys: &[f32], head_dim: usize, start: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; head_dim];
    for t in 0..len {
        let row = &keys[(start + t) * head_dim..(start + t + 1) * head_dim];
        for (o, v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
    for o in out.iter_mut() {
        *o /= len as f32;
    }
    out
}

/// Score one query against one pooled block key: ReLU per head, then sum.
///
/// The ReLU is INSIDE the sum — a sum of dots followed by one ReLU is a different
/// function, and there is no learned per-head weight here (DeepSeek V4's indexer
/// has one; this family's does not).
pub fn score_block(q: &[f32], block_key: &[f32], p: &QsaParams) -> f32 {
    let mut acc = 0.0f32;
    for h in 0..p.n_heads {
        let qh = &q[h * p.head_dim..(h + 1) * p.head_dim];
        let dot: f32 = qh.iter().zip(block_key).map(|(a, b)| a * b).sum();
        acc += dot.max(0.0);
    }
    acc / (p.head_dim as f32).sqrt()
}

/// Which cache positions one query may attend to.
///
/// `visible` is the causally-legal position list for this query, in increasing
/// order. `keys` is `[n_positions, head_dim]` indexed by cache position. Returns
/// the selected positions, sorted.
pub fn select(q: &[f32], keys: &[f32], visible: &[usize], p: &QsaParams) -> Vec<usize> {
    assert_eq!(q.len(), p.n_heads * p.head_dim);
    let n_blocks = visible.len() / p.compress_ratio;

    // The ragged tail past the last complete block is ALWAYS visible — it does not
    // compete for budget.
    let mut out: Vec<usize> = visible[n_blocks * p.compress_ratio..].to_vec();

    if n_blocks > 0 {
        let mut scored: Vec<(f32, usize)> = (0..n_blocks)
            .map(|b| {
                let start = b * p.compress_ratio;
                // Pool over the block's VISIBLE positions, which are not
                // necessarily contiguous cache indices.
                let mut pooled = vec![0.0f32; p.head_dim];
                for t in 0..p.compress_ratio {
                    let pos = visible[start + t];
                    let row = &keys[pos * p.head_dim..(pos + 1) * p.head_dim];
                    for (o, v) in pooled.iter_mut().zip(row) {
                        *o += v;
                    }
                }
                for o in pooled.iter_mut() {
                    *o /= p.compress_ratio as f32;
                }
                (score_block(q, &pooled, p), b)
            })
            .collect();
        // Descending by score; ties break to the lower block index.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for &(_, b) in scored.iter().take(p.block_topk().min(n_blocks)) {
            let start = b * p.compress_ratio;
            out.extend_from_slice(&visible[start..start + p.compress_ratio]);
        }
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> QsaParams {
        // The shipped geometry.
        QsaParams {
            n_heads: 4,
            head_dim: 128,
            budget: 2048,
            compress_ratio: 4,
        }
    }

    fn tiny() -> QsaParams {
        QsaParams {
            n_heads: 2,
            head_dim: 4,
            budget: 8,
            compress_ratio: 4,
        }
    }

    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// The budget is in tokens and the selection is over blocks.
    #[test]
    fn budget_is_tokens_and_topk_is_blocks() {
        let p = params();
        assert_eq!(p.block_topk(), 512, "2048 tokens / 4 per block");
        assert_eq!(p.dense_below(), 2051, "budget + compress_ratio - 1");
    }

    /// THE property that gives this port its only free exact oracle: at or below
    /// `dense_below` visible tokens, nothing can be excluded, so sparse attention
    /// IS dense attention and the two must agree bit-for-bit.
    #[test]
    fn below_the_budget_selection_is_everything() {
        let p = tiny();
        let q = seeded(p.n_heads * p.head_dim, 3);
        for n in 1..=p.dense_below() {
            let keys = seeded(n * p.head_dim, 5);
            let visible: Vec<usize> = (0..n).collect();
            let sel = select(&q, &keys, &visible, &p);
            assert_eq!(sel, visible, "n={n} must select every visible position");
        }
    }

    /// Above the budget it must actually exclude, otherwise the whole mechanism is
    /// a no-op and the test above would pass vacuously forever.
    #[test]
    fn above_the_budget_selection_excludes() {
        let p = tiny();
        let n = p.dense_below() + p.compress_ratio * 4;
        let q = seeded(p.n_heads * p.head_dim, 7);
        let keys = seeded(n * p.head_dim, 9);
        let visible: Vec<usize> = (0..n).collect();
        let sel = select(&q, &keys, &visible, &p);
        assert!(
            sel.len() < n,
            "must drop something at n={n}, kept {}",
            sel.len()
        );
        // Exactly block_topk whole blocks plus the ragged tail.
        let tail = n % p.compress_ratio;
        assert_eq!(sel.len(), p.block_topk() * p.compress_ratio + tail);
    }

    /// The ragged tail does not compete for budget — it is always present, even
    /// when every block slot is spent.
    #[test]
    fn ragged_tail_is_unconditional() {
        let p = tiny();
        let n = p.block_topk() * p.compress_ratio + 3; // 3-token tail
        let q = seeded(p.n_heads * p.head_dim, 11);
        let keys = seeded(n * p.head_dim, 13);
        let visible: Vec<usize> = (0..n).collect();
        let sel = select(&q, &keys, &visible, &p);
        for pos in n - 3..n {
            assert!(
                sel.contains(&pos),
                "tail position {pos} must always be visible"
            );
        }
    }

    /// Keys are MEAN-pooled over the block. Using only the first token of each
    /// block is the obvious cheaper alternative and gives different scores.
    #[test]
    fn block_keys_are_mean_pooled() {
        let head_dim = 4;
        // Block of 4 rows: 0, 2, 4, 6 -> mean 3.
        let keys: Vec<f32> = (0..4)
            .flat_map(|t| vec![(t * 2) as f32; head_dim])
            .collect();
        let pooled = pool_block(&keys, head_dim, 0, 4);
        assert!(pooled.iter().all(|v| (v - 3.0).abs() < 1e-6), "{pooled:?}");
        assert!(pooled[0] != 0.0, "must not be the FIRST token's value");
    }

    /// ReLU is applied per head, BEFORE the sum. A head with a strongly negative
    /// dot must contribute zero, not drag the score down.
    #[test]
    fn relu_is_per_head_before_the_sum() {
        let p = QsaParams {
            n_heads: 2,
            head_dim: 2,
            budget: 8,
            compress_ratio: 4,
        };
        let key = vec![1.0f32, 0.0];
        // head 0 dot = +3, head 1 dot = -100.
        let q = vec![3.0, 0.0, -100.0, 0.0];
        let s = score_block(&q, &key, &p);
        let want = 3.0f32 / (p.head_dim as f32).sqrt();
        assert!((s - want).abs() < 1e-6, "got {s}, expected {want}");
        // A sum-then-ReLU would give relu(3 - 100) = 0.
        assert!(
            s > 0.0,
            "the negative head must not cancel the positive one"
        );
    }

    /// Selection must be by score, not by recency: plant a high-scoring EARLY
    /// block and it must survive when the budget is tight.
    #[test]
    fn selection_follows_score_not_position() {
        let p = QsaParams {
            n_heads: 1,
            head_dim: 2,
            budget: 4,
            compress_ratio: 4,
        };
        let n = 16; // 4 blocks, budget 1 block
        let mut keys = vec![0.01f32; n * p.head_dim];
        // Block 0 (positions 0..4) aligned with the query; the rest are not.
        for t in 0..4 {
            keys[t * p.head_dim] = 10.0;
        }
        let q = vec![1.0f32, 0.0];
        let visible: Vec<usize> = (0..n).collect();
        let sel = select(&q, &keys, &visible, &p);
        assert_eq!(p.block_topk(), 1);
        assert_eq!(
            sel,
            vec![0, 1, 2, 3],
            "the highest-scoring block wins, not the newest"
        );
    }

    /// Non-contiguous visible sets (a masked or paged cache) must pool the actual
    /// visible positions rather than assuming a contiguous run.
    #[test]
    fn honours_a_non_contiguous_visible_set() {
        let p = QsaParams {
            n_heads: 1,
            head_dim: 2,
            budget: 4,
            compress_ratio: 4,
        };
        let keys = seeded(32 * p.head_dim, 21);
        let visible: Vec<usize> = (0..32).step_by(3).collect(); // 11 positions
        let q = seeded(p.n_heads * p.head_dim, 23);
        let sel = select(&q, &keys, &visible, &p);
        assert!(
            sel.iter().all(|s| visible.contains(s)),
            "never select a masked position"
        );
        let tail = visible.len() % p.compress_ratio;
        assert_eq!(sel.len(), p.block_topk() * p.compress_ratio + tail);
    }
}

/// Exact top-k over block scores by threshold search — the shape a GPU kernel can
/// follow, and the oracle it is differenced against.
///
/// The existing indexer top-k kernels in this tree are an O(N*K) serial selection
/// sort on decode and an O(N^2) rank-by-scan when batched, both written for
/// N <= 2048. QSA at the model's native context is top-512 of up to 65536 blocks,
/// where neither is viable — so selection is done by finding the k-th largest
/// value and partitioning around it, which is O(N) per probe and fully parallel.
///
/// Ties are resolved the way [`select`] resolves them: everything strictly above
/// the threshold is taken, then the remaining slots are filled from the tied
/// values in increasing index order. That rule has to match exactly, or the two
/// disagree only on inputs with duplicate scores — which is precisely where a
/// float-heavy kernel will produce them.
pub fn topk_by_threshold(scores: &[f32], k: usize) -> Vec<usize> {
    let n = scores.len();
    if k >= n {
        return (0..n).collect();
    }
    if k == 0 {
        return Vec::new();
    }
    // Binary search on the value, not the index: find the largest `t` with
    // `count(score > t) <= k`. 64 halvings is well past f32 resolution.
    let (mut lo, mut hi) = (
        scores.iter().cloned().fold(f32::INFINITY, f32::min),
        scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let above = scores.iter().filter(|s| **s > mid).count();
        if above > k {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = hi;
    let mut out: Vec<usize> = (0..n).filter(|&i| scores[i] > t).collect();
    // The search converges to a value strictly BETWEEN the k-th and (k+1)-th
    // distinct scores, not to one of them — so the tied band is not `== t`, it is
    // `== max{s : s <= t}`. Testing `== t` here matches nothing and silently
    // returns fewer than k, which is what the heavy-ties case exposed.
    if out.len() < k {
        let band = scores
            .iter()
            .cloned()
            .filter(|s| *s <= t)
            .fold(f32::NEG_INFINITY, f32::max);
        for i in 0..n {
            if out.len() == k {
                break;
            }
            if scores[i] == band {
                out.push(i);
            }
        }
    }
    out.truncate(k);
    out.sort_unstable();
    out
}

#[cfg(test)]
mod topk_tests {
    use super::*;

    fn brute(scores: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(k);
        idx.sort_unstable();
        idx
    }

    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 100_000) as f32 / 1000.0
            })
            .collect()
    }

    /// Matches an exact sort across sizes that span the real range, including the
    /// 65536-block case the existing kernels cannot handle.
    #[test]
    fn matches_a_full_sort() {
        for (n, k) in [(16usize, 4usize), (1000, 37), (4096, 512), (65536, 512)] {
            let s = seeded(n, n as u32);
            assert_eq!(topk_by_threshold(&s, k), brute(&s, k), "n={n} k={k}");
        }
    }

    /// Heavy ties are where a threshold method and a sort diverge if the
    /// tie-break rule is not identical.
    #[test]
    fn agrees_with_the_sort_under_heavy_ties() {
        // Only 4 distinct values across 200 entries.
        let s: Vec<f32> = (0..200).map(|i| (i % 4) as f32).collect();
        for k in [1usize, 7, 50, 51, 100, 199] {
            assert_eq!(topk_by_threshold(&s, k), brute(&s, k), "k={k}");
        }
    }

    /// All-equal scores are the degenerate tie case: it must still return exactly
    /// k, the lowest indices, rather than everything or nothing.
    #[test]
    fn all_equal_scores_return_exactly_k() {
        let s = vec![1.5f32; 64];
        let got = topk_by_threshold(&s, 10);
        assert_eq!(got, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn edges_hold() {
        let s = seeded(32, 5);
        assert_eq!(topk_by_threshold(&s, 0), Vec::<usize>::new());
        assert_eq!(topk_by_threshold(&s, 32), (0..32).collect::<Vec<_>>());
        assert_eq!(topk_by_threshold(&s, 99), (0..32).collect::<Vec<_>>());
    }
}
