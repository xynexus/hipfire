// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The hashed n-gram embedding: token ids → table row indices.
//!
//! This is the addressing half of the family's one genuinely new component. It is
//! pure integer math on already-committed token ids, which has two consequences
//! worth stating up front:
//!
//! * It is **fully testable without a GPU**, and every value here is pinned
//!   against the shipped checkpoint's own stored buffers.
//! * The row indices depend on nothing the forward pass produces, so they are
//!   known BEFORE the forward pass starts — which is what lets the 102 GB table
//!   live on disk and its reads overlap compute entirely.
//!
//! Semantics are the reference's (`Qwen4ExpTextNGramEmbedding.forward`), and were
//! independently confirmed against the merged llama.cpp port. Three details are
//! silent-wrong if guessed:
//!
//! * The mix is an **XOR of multiplied shifted tokens**, not a polynomial sum.
//! * The shift is **EOS-segment-aware**: an n-gram never spans a document
//!   boundary. A token's own EOS does NOT cut its own context, and a missing
//!   predecessor reads as EOS.
//! * Once the window is cut, every position further back is EOS too — the cut
//!   latches rather than being tested per position.

use crate::config::NgramConfig;

const MASK64: u64 = u64::MAX;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// Mixed into the seed so different PLE layers get different multipliers.
const PRIME_1: u64 = 10007;

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(SPLITMIX_GAMMA) & MASK64;
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1) & MASK64;
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2) & MASK64;
    (v ^ (v >> 31)) & MASK64
}

/// Derive the per-position hash multipliers, mirroring the reference's
/// `_build_layer_multipliers`.
///
/// The checkpoint STORES these as an i64 buffer, and a stored copy is
/// authoritative — but they are reproducible, which is what makes them checkable.
/// Verified: `(vocab 248320, ngram_size 3, layer 0, seed 1234)` reproduces the
/// shipped `layer_multipliers` exactly.
///
/// Each multiplier is forced odd (`2k + 1`) and bounded so `token * multiplier`
/// cannot exceed `i64::MAX`, which is what keeps the product well-defined on the
/// reference's signed-integer path.
pub fn build_layer_multipliers(
    unigram_vocab_size: u64,
    ngram_size: usize,
    ple_layer_index: usize,
    seed: u64,
) -> Vec<u64> {
    let max_long = (1u64 << 63) - 1;
    let multiplier_max = max_long / unigram_vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base = seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index as u64));
    (0..ngram_size)
        .map(|i| {
            let v = base.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i as u64 + 1)) & MASK64;
            2 * (splitmix64(v) % half_bound) + 1
        })
        .collect()
}

/// Turns token ids into the flat table rows one position needs.
#[derive(Debug, Clone)]
pub struct NgramHasher {
    multipliers: Vec<u64>,
    offsets: Vec<u64>,
    vocab_sizes: Vec<u64>,
    ngram_size: usize,
    heads_per_ngram: usize,
    eos: u32,
    shard_rows: u64,
    shards: usize,
}

/// Where one row lives on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLocation {
    pub shard: usize,
    pub row_in_shard: u64,
}

impl NgramHasher {
    /// Build from config alone, deriving the primes and multipliers.
    pub fn from_config(cfg: &NgramConfig, unigram_vocab: u64, eos: u32) -> Self {
        let (vocab_sizes, offsets, padded) = hipfire_arch_qwen4exp_spec::ngram_head_layout_at(
            cfg.vocab_size_base,
            cfg.heads(),
            cfg.divisible_by,
            cfg.ple_index,
        );
        Self {
            // `ple_index`, NOT `layer_idx` — the reference seeds from the PLE
            // block's ordinal. With `ple_layer_ids = [2]` those are 0 and 1.
            multipliers: build_layer_multipliers(
                unigram_vocab,
                cfg.ngram_size,
                cfg.ple_index,
                cfg.seed,
            ),
            offsets,
            vocab_sizes,
            ngram_size: cfg.ngram_size,
            heads_per_ngram: cfg.heads_per_ngram,
            eos,
            shard_rows: padded / cfg.shards as u64,
            shards: cfg.shards,
        }
    }

    /// Override the derived tables with the checkpoint's stored buffers. Prefer
    /// this whenever the artifact carries them: derivation is a reproduction, and
    /// the stored values are the ground truth.
    pub fn with_stored(
        mut self,
        multipliers: Option<Vec<u64>>,
        offsets: Option<Vec<u64>>,
        vocab_sizes: Option<Vec<u64>>,
    ) -> Result<Self, String> {
        if let Some(m) = multipliers {
            if m.len() != self.ngram_size {
                return Err(format!(
                    "stored layer_multipliers has {} entries, expected {}",
                    m.len(),
                    self.ngram_size
                ));
            }
            self.multipliers = m;
        }
        let heads = self.offsets.len();
        if let Some(o) = offsets {
            if o.len() != heads {
                return Err(format!(
                    "stored offsets has {} entries, expected {heads}",
                    o.len()
                ));
            }
            self.offsets = o;
        }
        if let Some(v) = vocab_sizes {
            if v.len() != heads {
                return Err(format!(
                    "stored vocab_sizes has {} entries, expected {heads}",
                    v.len()
                ));
            }
            self.vocab_sizes = v;
        }
        Ok(self)
    }

    pub fn heads(&self) -> usize {
        self.offsets.len()
    }
    pub fn multipliers(&self) -> &[u64] {
        &self.multipliers
    }
    pub fn vocab_sizes(&self) -> &[u64] {
        &self.vocab_sizes
    }
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Build the EOS-aware context window for one position.
    ///
    /// `current` is the token at this position; `predecessors` runs oldest-first
    /// and `None` means "no cached cell", i.e. before the sequence start. The
    /// returned window is `[current, prev, prev-1, ...]`, newest first, which is
    /// the order the multipliers index.
    ///
    /// A token's own EOS does NOT cut its own context — only positions strictly
    /// behind it are affected — and once cut, everything further back is EOS.
    pub fn context(&self, current: u32, predecessors: &[Option<u32>]) -> Vec<u64> {
        let mut ctx = Vec::with_capacity(self.ngram_size);
        ctx.push(current as u64);
        let mut cut = false;
        for s in 1..self.ngram_size {
            // `predecessors` is oldest-first, so `s` positions back is the s-th
            // from the end.
            let t = if cut {
                None
            } else {
                predecessors
                    .len()
                    .checked_sub(s)
                    .and_then(|i| predecessors.get(i).copied().flatten())
            };
            cut = cut || t.is_none() || t == Some(self.eos);
            ctx.push(if cut {
                self.eos as u64
            } else {
                t.unwrap() as u64
            });
        }
        ctx
    }

    /// The flat row index for every hash head at one position.
    ///
    /// Orders run `2..=ngram_size`; each order owns `heads_per_ngram` consecutive
    /// heads, and every head in an order hashes the SAME key under its own prime.
    pub fn rows(&self, current: u32, predecessors: &[Option<u32>]) -> Vec<u64> {
        let ctx = self.context(current, predecessors);
        let mut out = vec![0u64; self.heads()];
        for n in 2..=self.ngram_size {
            // XOR of multiplied shifted tokens — NOT a polynomial sum.
            let mut mixed = ctx[0].wrapping_mul(self.multipliers[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(self.multipliers[j]);
            }
            let base = (n - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                out[h] = mixed % self.vocab_sizes[h] + self.offsets[h];
            }
        }
        out
    }

    /// Map a flat row to its shard and offset within that shard. The shards are a
    /// uniform slice of one padded flat table, so this is plain division.
    pub fn locate(&self, row: u64) -> RowLocation {
        RowLocation {
            shard: (row / self.shard_rows) as usize,
            row_in_shard: row % self.shard_rows,
        }
    }

    pub fn shard_rows(&self) -> u64 {
        self.shard_rows
    }
    pub fn shards(&self) -> usize {
        self.shards
    }
}
