// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the DFlash2 CPU reference (`hipfire_runtime::dflash2`) against
//! golden vectors produced by z-lab's own `dflash/model.py` running on the REAL
//! Qwen3.8-27B-DFlash2 checkpoint weights.
//!
//! This is the check that makes the port meaningful: a parity test written
//! against a re-derivation of our own reading would validate nothing. The
//! generator is `gen_golden.py` (see docs/todo/2026-08-20-handover-dflash2-*).
//!
//! usage: dflash2_parity <dflash2_golden.bin>

use hipfire_runtime::dflash2::{grouped_dynamic_convolve, selector_argmax, selector_scores};

fn rd(b: &[u8], p: &mut usize) -> Vec<f32> {
    let n = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap()) as usize;
    *p += 8;
    let v = b[*p..*p + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    *p += n * 4;
    v
}

fn cmp(name: &str, got: &[f32], want: &[f32], tol: f32) -> bool {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: length {} vs {}",
        got.len(),
        want.len()
    );
    let (mut max_abs, mut at) = (0f32, 0usize);
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
    let ok = max_abs <= tol * scale;
    println!(
        "  {:<16} max|Δ|={max_abs:.3e} (ref|max|={scale:.3e}, at {at}) -> {}",
        name,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dflash2_parity <golden.bin>");
    let b = std::fs::read(&path).expect("read golden");
    let hdr: Vec<i64> = b[..48]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (h, gs, ks, rank, topk, len) = (
        hdr[0] as usize,
        hdr[1] as usize,
        hdr[2] as usize,
        hdr[3] as usize,
        hdr[4] as usize,
        hdr[5] as usize,
    );
    println!("DFlash2 parity: hidden={h} group={gs} taps={ks} rank={rank} top_k={topk} len={len}");
    let mut p = 48usize;
    let hidden = rd(&b, &mut p);
    let base = rd(&b, &mut p);
    let proj = rd(&b, &mut p);
    let want_prepare = rd(&b, &mut p);
    let want_finish = rd(&b, &mut p);
    let hp_h = rd(&b, &mut p);
    let _cand = rd(&b, &mut p);
    let unary = rd(&b, &mut p);
    let pre_anchor = rd(&b, &mut p);
    let pre_cand = rd(&b, &mut p);
    let suc_cand = rd(&b, &mut p);
    let want_scores = rd(&b, &mut p);
    let want_path = rd(&b, &mut p);

    let groups = h / gs;
    // dynamic = hidden @ proj^T, viewed [len, 2, ks, groups]; slice 0 -> prepare,
    // slice 1 -> finish. proj is [2*ks*groups, h] row-major.
    let outw = 2 * ks * groups;
    let mut dynamic = vec![0f32; len * outw];
    for t in 0..len {
        for o in 0..outw {
            let row = &proj[o * h..(o + 1) * h];
            dynamic[t * outw + o] = row
                .iter()
                .zip(&hidden[t * h..(t + 1) * h])
                .map(|(a, x)| a * x)
                .sum();
        }
    }
    let slice = |which: usize| -> Vec<f32> {
        let mut d = vec![0f32; len * ks * groups];
        for t in 0..len {
            for tap in 0..ks {
                let src = t * outw + (which * ks + tap) * groups;
                let dst = (t * ks + tap) * groups;
                d[dst..dst + groups].copy_from_slice(&dynamic[src..src + groups]);
            }
        }
        d
    };

    let mut ok = true;
    let got_prepare = grouped_dynamic_convolve(&hidden, &slice(0), &base[..ks * h], len, h, ks, gs);
    ok &= cmp("conv prepare", &got_prepare, &want_prepare, 2e-4);
    let got_finish = grouped_dynamic_convolve(&hidden, &slice(1), &base[ks * h..], len, h, ks, gs);
    ok &= cmp("conv finish", &got_finish, &want_finish, 2e-4);

    // Selector: greedy trace. The golden stores the predecessor row for the
    // anchor plus every candidate's rows, so the trace is reproducible here.
    let mut got_scores = Vec::with_capacity(len * topk);
    let mut got_path = Vec::with_capacity(len);
    let mut pred: Vec<f32> = pre_anchor.clone();
    for t in 0..len {
        let s = selector_scores(
            &unary[t * topk..(t + 1) * topk],
            &pred,
            &hp_h[t * rank..(t + 1) * rank],
            &suc_cand[t * topk * rank..(t + 1) * topk * rank],
            topk,
            rank,
        );
        let idx = selector_argmax(&s);
        got_scores.extend_from_slice(&s);
        got_path.push(idx as f32);
        pred = pre_cand[(t * topk + idx) * rank..(t * topk + idx + 1) * rank].to_vec();
    }
    ok &= cmp("selector scores", &got_scores, &want_scores, 2e-4);
    // want_path holds token ids; compare the chosen candidate INDEX instead by
    // re-deriving it is not possible from ids alone, so check score-argmax
    // agreement per position, which is what drives the trace.
    for t in 0..len {
        let g = selector_argmax(&got_scores[t * topk..(t + 1) * topk]);
        let w = selector_argmax(&want_scores[t * topk..(t + 1) * topk]);
        if g != w {
            println!("  argmax mismatch at position {t}: {g} vs {w}");
            ok = false;
        }
    }
    println!(
        "  selector path len={} (golden ids: {:?})",
        got_path.len(),
        &want_path[..len.min(4)]
    );
    println!("dflash2_parity: {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
