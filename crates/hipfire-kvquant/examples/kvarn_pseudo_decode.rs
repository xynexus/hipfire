//! Pseudo-decode proxy for KV-cache quantisation error ACCUMULATION.
//!
//! Implements the setting defined in 2606.03458v1 (KVarN), Fig. "pseudo-decode":
//!
//!   "We split the sequence into blocks of size b. After every block, the freshly
//!    produced K, V are quantized before being written back to the KV-cache.
//!    Subsequent blocks access a quantized cache, so quantization error
//!    accumulates over time."
//!
//! That feedback is the whole point and is what a single-shot reconstruction test
//! cannot see: prefill quantises once with no feedback, so it measures E(1 step).
//! Here the quantised run's FUTURE keys are generated from its OWN dequantised
//! history, so errors compound the way they do in real decoding.
//!
//! Two runs share an identical driving process; only the cache differs:
//!   exact  — cache holds f32 K
//!   quant  — cache holds dequant(quant(K)), and the next block's K is produced
//!            from a readout of THAT cache
//!
//! Reported per step: relative drift of the attention readout, which is what the
//! rest of the model actually consumes.
//!
//! Also reports the paper's error decomposition (eq. "decompose"):
//!   E_T = ||K - K_dq||^2  =  E_M (||K||-||K_dq||)^2  +  E_D 2||K||||K_dq||(1-cos)
//! The paper claims outlier error is "overwhelmingly caused by incorrect
//! magnitudes"; E_M/E_T tests that on our quantiser.
use hipfire_kvquant::kvarn::*;

const CH: usize = 128; // channels (head_dim) — the rotation axis
const BLK: usize = 32; // block size b

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*s >> 33) as f32) / (u32::MAX as f32 / 2.0)) - 1.0
}

/// Attention-like readout: softmax(q . k_t) weighted average over the cache.
fn readout(cache: &[Vec<f32>], q: &[f32], rotated_frame: bool) -> Vec<f32> {
    // A rotated cache MUST be queried with a rotated query. H is orthonormal so
    // q.k == (qH).(kH) and the scores are identical to the unrotated pair —
    // that identity is the entire reason the rotation is free at attention time.
    // Querying a rotated cache with a raw q instead produces meaningless scores
    // and the readout collapses (drift saturates at exactly 1.0).
    let mut qr;
    let q: &[f32] = if rotated_frame {
        qr = q.to_vec();
        hadamard_rows(&mut qr, CH);
        &qr
    } else {
        q
    };
    let mut logits = Vec::with_capacity(cache.len());
    for k in cache {
        let d: f32 = k.iter().zip(q).map(|(a, b)| a * b).sum();
        logits.push(d / (CH as f32).sqrt());
    }
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut w: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let z: f32 = w.iter().sum::<f32>().max(1e-30);
    for v in w.iter_mut() {
        *v /= z;
    }
    let mut out = vec![0f32; CH];
    for (wi, k) in w.iter().zip(cache) {
        for c in 0..CH {
            out[c] += wi * k[c];
        }
    }
    out
}

/// One decode block's keys, produced from the current hidden state. Deterministic
/// in (h, step) so both runs see the SAME process and differ only through h.
fn make_block(h: &[f32], step: usize, outlier_every: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(BLK);
    for t in 0..BLK {
        let mut k = vec![0f32; CH];
        for c in 0..CH {
            let mix = h[(c + t * 7 + step * 13) % CH];
            let base = ((((c * 31 + t * 17 + step * 11) % 13) as f32) - 6.0) * 0.05;
            let mag = if c % outlier_every == 0 { 9.0 } else { 1.0 };
            k[c] = (base + 0.35 * mix) * mag;
        }
        out.push(k);
    }
    out
}

fn quantize_block(block: &[Vec<f32>], qmax: f32, rotate: bool) -> Vec<Vec<f32>> {
    // tile is [channel(row) x token(col)] — matches kvarn_gather_k_tiles
    let mut tile = vec![0f32; CH * BLK];
    for (t, k) in block.iter().enumerate() {
        for c in 0..CH {
            tile[c * BLK + t] = k[c];
        }
    }
    let qt = if rotate {
        quantize_tile_rotated(&tile, CH, BLK, qmax)
    } else {
        quantize_tile_qmax(&tile, CH, BLK, qmax)
    };
    let deq = dequantize_tile(&qt);
    // The rotated path lives in the rotated frame; attention there uses rotated Q,
    // so scores match. To compare runs in ONE frame we rotate the exact side too
    // when `rotate` — handled by the caller for the reference. Here just return
    // the frame the cache holds.
    (0..BLK)
        .map(|t| (0..CH).map(|c| deq[c * BLK + t]).collect::<Vec<f32>>())
        .collect()
}

fn run(steps: usize, qmax: f32, rotate: bool, outlier_every: usize) -> (Vec<f32>, f64) {
    let mut s_exact = 0xDEADBEEFu64;
    let mut h_exact: Vec<f32> = (0..CH).map(|_| lcg(&mut s_exact)).collect();
    let mut h_quant = h_exact.clone();
    let mut cache_exact: Vec<Vec<f32>> = Vec::new();
    let mut cache_quant: Vec<Vec<f32>> = Vec::new();
    let mut drift = Vec::with_capacity(steps);
    let (mut em, mut et) = (0f64, 0f64);

    for step in 0..steps {
        // exact run
        let be = make_block(&h_exact, step, outlier_every);
        cache_exact.extend(be.iter().cloned());

        // quantised run: block produced from ITS OWN drifted state, then quantised
        let bq_raw = make_block(&h_quant, step, outlier_every);
        let bq = quantize_block(&bq_raw, qmax, rotate);

        // paper's decomposition, on this block
        for (k, kq) in bq_raw.iter().zip(&bq) {
            let nk: f64 = (k.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
            let nq: f64 = (kq.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
            let dot: f64 = k
                .iter()
                .zip(kq)
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let _cos = if nk * nq > 1e-30 {
                dot / (nk * nq)
            } else {
                1.0
            };
            let e_t: f64 = k
                .iter()
                .zip(kq)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            et += e_t;
            em += (nk - nq).powi(2);
        }
        cache_quant.extend(bq.into_iter());

        // readout drives the next block => errors compound
        let q_e: Vec<f32> = h_exact.clone();
        let q_q: Vec<f32> = h_quant.clone();
        h_exact = readout(&cache_exact, &q_e, false);
        h_quant = readout(&cache_quant, &q_q, rotate);
        // Only the CACHE lives in the rotated frame. A real implementation
        // un-rotates the attention output (`acc_rot @ H.t()`) before the rest of
        // the network sees it, so the model STATE is always in the natural basis.
        // Feeding a rotated h back into the next block would make the two runs
        // different processes rather than the same process plus quant error —
        // which showed up as drift pinned at exactly 1.0.
        if rotate {
            hadamard_rows(&mut h_quant, CH); // self-inverse at this scaling
        }

        let num: f32 = h_exact
            .iter()
            .zip(&h_quant)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f32 = h_exact.iter().map(|v| v * v).sum::<f32>().max(1e-20);
        drift.push((num / den).sqrt());
    }
    (drift, em / et.max(1e-30))
}

fn main() {
    let steps = 24usize;
    println!("pseudo-decode proxy: {CH} channels, block b={BLK}, {steps} blocks");
    println!("relative drift of the attention readout vs the unquantised run\n");
    for (bits, qmax) in [(2usize, 3.0f32), (4, 15.0)] {
        println!("--- {bits}-bit ---");
        println!(
            "{:>6} {:>12} {:>12} {:>10}",
            "step", "plain", "rotated", "rot/plain"
        );
        let (dp, emp) = run(steps, qmax, false, 8);
        let (dr, emr) = run(steps, qmax, true, 8);
        for i in [0usize, 3, 7, 11, 15, 19, 23] {
            if i < dp.len() {
                println!(
                    "{:>6} {:>12.5} {:>12.5} {:>10.3}",
                    i + 1,
                    dp[i],
                    dr[i],
                    dr[i] / dp[i].max(1e-12)
                );
            }
        }
        println!(
            "  final: plain {:.5}  rotated {:.5}  ({:.2}x)",
            dp[steps - 1],
            dr[steps - 1],
            dr[steps - 1] / dp[steps - 1].max(1e-12)
        );
        println!("  E_M/E_T (magnitude share of error): plain {emp:.3}  rotated {emr:.3}\n");
    }
}
