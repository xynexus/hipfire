// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the chunkwise-parallel gated DeltaNet kernel.
//!
//! Two claims, checked separately, because they can fail independently:
//!
//!  1. the chunk kernel reproduces the f64 CHUNKED reference
//!     (`gdn_chunk::gdn_chunked_f64`) — i.e. the kernel implements the algorithm;
//!  2. it reproduces the f64 SEQUENTIAL reference (`gdn_sequential_f64`) — i.e.
//!     the algorithm is still the serving recurrence.
//!
//! (2) is the one that matters for shipping, but a failure in (1) alone points at
//! the kernel and a failure in both points at the derivation, so it is worth
//! knowing which. The equivalence of the two references is itself a unit test in
//! `gdn_chunk`, run without a GPU.
//!
//! The serial kernel is measured alongside on the same inputs, so the error bar
//! is a comparison and not an absolute: both are f32 accumulating over head_dim,
//! and the chunk form should land in the same neighbourhood. It will not be
//! bit-identical — it sums the same terms in a different order.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gdn_chunk

use hipfire_rdna::gdn_chunk::{gdn_chunked_f64, gdn_sequential_f64, GdnDims};
use hipfire_rdna::{DType, Gpu};

const HD: usize = 128;
const N_HEADS: usize = 4;
const N_TOKENS: usize = 24; // > CMAX, so the multi-chunk carry is covered

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 + 0.5) / 2_147_483_648.0) * 2.0 - 1.0
        })
        .collect()
}

fn report(name: &str, got: &[f32], want: &[f64]) -> f64 {
    let (mut max_abs, mut at) = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g as f64 - w).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!("  {name:<34} max|Δ|={max_abs:.3e} (ref|max|={scale:.3e}, at {at})");
    max_abs
}

fn main() {
    let dims = GdnDims {
        n_tokens: N_TOKENS,
        n_heads: N_HEADS,
        head_dim: HD,
    };
    let n = N_TOKENS * N_HEADS * HD;
    let (q, k, v) = (lcg(1, n), lcg(2, n), lcg(3, n));
    // gate is a log-decay, so strictly <= 0.
    let gate: Vec<f32> = lcg(4, N_TOKENS * N_HEADS)
        .iter()
        .map(|x| -x.abs() * 0.5)
        .collect();
    let beta: Vec<f32> = lcg(5, N_TOKENS * N_HEADS).iter().map(|x| x.abs()).collect();
    let state0: Vec<f32> = lcg(6, N_HEADS * HD * HD).iter().map(|x| x * 0.05).collect();

    let s0_f64: Vec<f64> = state0.iter().map(|&x| x as f64).collect();
    let (mut s_seq, mut s_chk) = (s0_f64.clone(), s0_f64.clone());
    let want_seq = gdn_sequential_f64(dims, &q, &k, &v, &gate, &beta, &mut s_seq);
    let want_chk = gdn_chunked_f64(dims, 16, &q, &k, &v, &gate, &beta, &mut s_chk);

    let mut gpu = Gpu::init().expect("gpu");
    let up = |g: &mut Gpu, d: &[f32]| g.upload_f32(d, &[d.len()]).expect("upload");
    let (qt, kt, vt) = (up(&mut gpu, &q), up(&mut gpu, &k), up(&mut gpu, &v));
    let (gt, bt) = (up(&mut gpu, &gate), up(&mut gpu, &beta));
    let st = up(&mut gpu, &state0);
    let out = gpu.alloc_tensor(&[n], DType::F32).expect("out");
    gpu.gated_delta_net_f32_chunk(&qt, &kt, &vt, &gt, &bt, &st, &out, N_TOKENS, N_HEADS, HD)
        .expect("chunk kernel");
    let got_out = gpu.download_f32(&out).expect("dl out");
    let got_state = gpu.download_f32(&st).expect("dl state");

    // Same inputs through the serial kernel, for a same-precision yardstick.
    let st2 = up(&mut gpu, &state0);
    let out2 = gpu.alloc_tensor(&[n], DType::F32).expect("out2");
    gpu.gated_delta_net_f32_batch_seq(&qt, &kt, &vt, &gt, &bt, &st2, &out2, N_TOKENS, N_HEADS, HD)
        .expect("serial kernel");
    let seq_out = gpu.download_f32(&out2).expect("dl out2");
    let seq_state = gpu.download_f32(&st2).expect("dl state2");

    println!("parity_gdn_chunk tokens={N_TOKENS} heads={N_HEADS} head_dim={HD} CMAX=16");
    println!(" chunk kernel vs f64 chunked reference (does the kernel implement it?)");
    let a1 = report("output", &got_out, &want_chk);
    let a2 = report("state", &got_state, &s_chk);
    println!(" chunk kernel vs f64 SEQUENTIAL reference (is it still the recurrence?)");
    let b1 = report("output", &got_out, &want_seq);
    let b2 = report("state", &got_state, &s_seq);
    println!(" serial kernel vs the same sequential reference (yardstick)");
    let c1 = report("output", &seq_out, &want_seq);
    let c2 = report("state", &seq_state, &s_seq);

    // f32 accumulation over head_dim 128 with |state| ~ 0.05 and |q,k,v| ~ 1
    // lands around 1e-6; the bound is set from the SERIAL kernel's own error so
    // this fails when the chunk form is meaningfully worse, not when f32 is f32.
    let floor = 2e-5f64;
    let tol = floor.max(4.0 * c1.max(c2));
    let worst = a1.max(a2).max(b1).max(b2);
    println!("  tol={tol:.3e} (4x serial kernel error, floor {floor:.0e})");
    if worst <= tol {
        println!("parity_gdn_chunk -> PASS");
    } else {
        println!("parity_gdn_chunk -> FAIL (worst {worst:.3e} > tol {tol:.3e})");
        std::process::exit(1);
    }
}
