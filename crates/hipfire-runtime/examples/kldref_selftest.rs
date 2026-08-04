// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only self-test for an HFQM `*.kldref.hfq` reference: does block
//! `(chunk, j)` really describe the prediction of token
//! `tokens[chunk][scoring_start + j + 1]`?
//!
//! The check needs no model. A good reference's argmax agrees with the corpus's
//! actual next token a large fraction of the time (tens of percent on wikitext);
//! a misaligned block index drops that to chance (~0%). Running it per chunk
//! localizes a mapping bug to the chunk axis, which a single-chunk spot check
//! cannot do.
//!
//!   cargo run --release -p hipfire-runtime --example kldref_selftest -- <ref.hfq> [n_chunks]

use hipfire_runtime::hfq::HfqPackage;
use std::path::Path;

fn n_vocab_of(meta: &serde_json::Value) -> usize {
    meta.get("n_vocab").and_then(|v| v.as_u64()).unwrap() as usize
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: kldref_selftest <ref.kldref.hfq> [n]");
    let want: usize = std::env::args().nth(2).map(|v| v.parse().unwrap()).unwrap_or(8);

    let pkg = HfqPackage::open(Path::new(&path)).expect("open");
    let meta: serde_json::Value = serde_json::from_str(&pkg.metadata_json).expect("meta");
    let g = |k: &str| meta.get(k).and_then(|v| v.as_u64()).unwrap() as usize;
    let (n_ctx, n_chunk, spc, top_k, start) = (
        g("n_ctx"),
        g("n_chunk"),
        g("scored_per_chunk"),
        g("top_k"),
        g("scoring_start"),
    );
    println!(
        "ref: n_ctx={n_ctx} n_chunk={n_chunk} scored/chunk={spc} top_k={top_k} scoring_start={start}"
    );
    println!(
        "blobs: {:?}",
        pkg.entries().iter().map(|e| e.name.as_str()).collect::<Vec<_>>()
    );

    let u32s = |n: &str| -> Vec<u32> {
        pkg.blob_data(n)
            .unwrap_or_else(|| panic!("missing {n}"))
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let tokens = u32s("kldref.tokens");
    let top_indices = u32s("kldref.top_indices");
    println!(
        "tokens={} (expect {}), top_indices={} (expect {})",
        tokens.len(),
        n_chunk * n_ctx,
        top_indices.len(),
        n_chunk * spc * top_k
    );

    // The reference's OWN perplexity on chunk 0, from its stored top-k log-probs.
    // Gives the bf16 anchor a candidate's absolute KLD should be read against —
    // without it, "0.27 nats from bf16" has no scale.
    let f32s = |n: &str| -> Vec<f32> {
        pkg.blob_data(n)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let top_lp = f32s("kldref.top_log_probs");
    let _resid = f32s("kldref.residual_mass");
    let (mut nll, mut n_in, mut n_out) = (0.0f64, 0usize, 0usize);
    for j in 0..spc {
        let target = tokens[start + j + 1];
        let b = j * top_k;
        match (0..top_k).find(|&t| top_indices[b + t] == target) {
            Some(t) => {
                nll += -(top_lp[b + t] as f64);
                n_in += 1;
            }
            // Outside the stored top-k the true log-prob is not recoverable
            // (spreading the residual mass uniformly over 248k tokens charges
            // ~17 nats and blows the average up), so those positions are excluded
            // and the figure below is a RESTRICTED perplexity — optimistic, and
            // not directly comparable to a candidate's full-window PPL.
            None => n_out += 1,
        }
    }
    println!(
        "reference's OWN chunk-0 PPL over the {n_in}/{spc} targets inside top-{top_k}: {:.3} \
         ({n_out} excluded — restricted, optimistic, NOT comparable to a full-window PPL)",
        (nll / n_in.max(1) as f64).exp(),
    );
    let _ = n_vocab_of(&meta);

    // Is the block array actually distinct per chunk, or did the producer stall?
    let blk = |c: usize, j: usize| &top_indices[(c * spc + j) * top_k..(c * spc + j + 1) * top_k];
    let dup01 = (0..spc).filter(|&j| blk(0, j) == blk(1, j)).count();
    let zero1 = (0..spc).filter(|&j| blk(1, j).iter().all(|&v| v == 0)).count();
    let zero0 = (0..spc).filter(|&j| blk(0, j).iter().all(|&v| v == 0)).count();
    println!(
        "chunk1 blocks identical to chunk0: {dup01}/{spc};  all-zero blocks: chunk0 {zero0}, chunk1 {zero1}"
    );
    // Where does chunk 1's block set actually belong? Slide it over the token
    // stream and look for a position whose next-token sequence it predicts well.
    let probe = 256usize.min(spc);
    let mut best = (0usize, 0usize);
    for p in (0..(tokens.len() - probe - 1)).step_by(1) {
        let h = (0..probe).filter(|&j| blk(1, j)[0] == tokens[p + j]).count();
        if h > best.1 {
            best = (p, h);
        }
    }
    println!(
        "chunk1 blocks best-match token position {} ({}/{probe} = {:.1}%) — assumed position was {}",
        best.0,
        best.1,
        100.0 * best.1 as f64 / probe as f64,
        1 * n_ctx + start + 1
    );

    // Per-chunk agreement between the reference argmax and the corpus's next token,
    // under the assumed mapping block(c, j) -> predicts tokens[c][start + j + 1].
    for c in 0..want.min(n_chunk) {
        let mut hit = 0usize;
        for j in 0..spc {
            let pred = top_indices[(c * spc + j) * top_k];
            let target = tokens[c * n_ctx + start + j + 1];
            if pred == target {
                hit += 1;
            }
        }
        // Also probe a whole-chunk shift: does chunk c's block set match ANY chunk's
        // tokens better? (Only chunk c and its neighbours, to keep it cheap.)
        let mut best = (c, hit);
        for cc in c.saturating_sub(2)..(c + 3).min(n_chunk) {
            let mut h = 0usize;
            for j in 0..spc {
                if top_indices[(c * spc + j) * top_k] == tokens[cc * n_ctx + start + j + 1] {
                    h += 1;
                }
            }
            if h > best.1 {
                best = (cc, h);
            }
        }
        println!(
            "chunk {c}: top-1 agreement {:.1}% ({hit}/{spc}){}",
            100.0 * hit as f64 / spc as f64,
            if best.0 != c {
                format!("   <-- blocks match chunk {} better ({}/{})", best.0, best.1, spc)
            } else {
                String::new()
            }
        );
    }
}
