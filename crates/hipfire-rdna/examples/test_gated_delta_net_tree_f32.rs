#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::useless_vec
)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for the non-Q8 tree DeltaNet kernels.
//!
//! These exist so DDTree tree-replay does not require Q8 state, which
//! `qwen35/state.rs` forbids. Before the Q8 tree kernel can be retired, its
//! replacements have to be shown to actually replay a tree correctly — a kernel
//! that merely compiles proves nothing.
//!
//! Two claims are checked, both against the SAME reference: the per-token linear
//! FP32 kernel called N times on a rolling state, which is what tree spine
//! semantics are defined to reproduce.
//!
//!   1. `gated_delta_net_f32_tree` on a spine is BYTE-EXACT against it. That is
//!      why the kernel uses the `col = tid * 4` lane mapping of the FP32 linear
//!      kernel rather than the Q8 family's stride-32 layout: summation order
//!      fixes the low bits, so exactness requires matching the kernel you claim
//!      exactness against.
//!   2. `gated_delta_net_f16_tree` is close but NOT exact — one f16 rounding per
//!      tape round-trip. Asserting a bound rather than equality is the point;
//!      if it ever came back byte-exact the tape would not be f16.
//!
//! Build: `cargo run --release --features deltanet \
//!   --example test_gated_delta_net_tree_f32 -p hipfire-rdna`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_gated_delta_net_tree_f32 requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_rdna::{DType, Gpu};
    const HD: usize = 128;
    const N_HEADS: usize = 4;
    const N_TOKENS: usize = 5;

    let mut gpu = Gpu::init().expect("GPU init");

    let n_el = N_TOKENS * N_HEADS * HD;
    let q: Vec<f32> = (0..n_el).map(|i| sin_det(i, 3)).collect();
    let k: Vec<f32> = (0..n_el).map(|i| sin_det(i, 5)).collect();
    let v: Vec<f32> = (0..n_el).map(|i| sin_det(i, 7)).collect();
    let gate: Vec<f32> = (0..N_TOKENS * N_HEADS)
        .map(|i| sin_det(i, 11) * 0.1 - 0.5)
        .collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_HEADS)
        .map(|i| sigmoid(sin_det(i, 13)))
        .collect();
    // FP32 initial state, deterministic and O(1e-2) so the recurrence neither
    // saturates nor sits at zero (a zero state would hide tape bugs entirely).
    let s_init: Vec<f32> = (0..N_HEADS * HD * HD)
        .map(|i| ((i * 5381) % 251) as f32 * 1e-4 - 0.0125)
        .collect();

    // ---- Reference: N successive linear FP32 calls, n_tokens=1, rolling state.
    let state_ref = gpu.upload_f32(&s_init, &[N_HEADS * HD * HD]).unwrap();
    let out_ref = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();
    for t in 0..N_TOKENS {
        let lo = t * N_HEADS * HD;
        let hi = (t + 1) * N_HEADS * HD;
        let q1 = gpu.upload_f32(&q[lo..hi], &[1, N_HEADS * HD]).unwrap();
        let k1 = gpu.upload_f32(&k[lo..hi], &[1, N_HEADS * HD]).unwrap();
        let v1 = gpu.upload_f32(&v[lo..hi], &[1, N_HEADS * HD]).unwrap();
        let g1 = gpu
            .upload_f32(&gate[t * N_HEADS..(t + 1) * N_HEADS], &[1, N_HEADS])
            .unwrap();
        let b1 = gpu
            .upload_f32(&beta[t * N_HEADS..(t + 1) * N_HEADS], &[1, N_HEADS])
            .unwrap();
        let o1 = gpu.zeros(&[1, N_HEADS * HD], DType::F32).unwrap();
        gpu.gated_delta_net_f32_batch_seq(&q1, &k1, &v1, &g1, &b1, &state_ref, &o1, 1, N_HEADS, HD)
            .unwrap();
        let row_bytes = N_HEADS * HD * 4;
        gpu.hip
            .memcpy_dtod_at(&out_ref.buf, t * row_bytes, &o1.buf, 0, row_bytes)
            .unwrap();
        for t in [q1, k1, v1, g1, b1, o1] {
            gpu.free_tensor(t).unwrap();
        }
    }
    let ref_host = gpu.download_f32(&out_ref).unwrap();

    // ---- Tree kernels on a spine: parents = [-1, 0, 1, 2, ...].
    let parents: Vec<i32> = (0..N_TOKENS as i32).map(|t| t - 1).collect();
    let q_gpu = gpu.upload_f32(&q, &[N_TOKENS, N_HEADS * HD]).unwrap();
    let k_gpu = gpu.upload_f32(&k, &[N_TOKENS, N_HEADS * HD]).unwrap();
    let v_gpu = gpu.upload_f32(&v, &[N_TOKENS, N_HEADS * HD]).unwrap();
    let g_gpu = gpu.upload_f32(&gate, &[N_TOKENS, N_HEADS]).unwrap();
    let b_gpu = gpu.upload_f32(&beta, &[N_TOKENS, N_HEADS]).unwrap();
    let parents_gpu = upload_i32(&mut gpu, &parents);

    // f32 tape: N_TOKENS x N_HEADS x HD x HD floats.
    let init_f32 = gpu.upload_f32(&s_init, &[N_HEADS * HD * HD]).unwrap();
    let tape_f32 = gpu
        .zeros(&[N_TOKENS * N_HEADS * HD * HD], DType::F32)
        .unwrap();
    let out_f32 = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();
    gpu.gated_delta_net_f32_tree_batch_seq(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &g_gpu,
        &b_gpu,
        &init_f32,
        &tape_f32,
        &parents_gpu,
        &out_f32,
        N_TOKENS,
        N_HEADS,
        HD,
    )
    .unwrap();
    let f32_host = gpu.download_f32(&out_f32).unwrap();

    // f16 tape: HALF the bytes — 2 per element, so allocate as Raw.
    let init_f16 = gpu.upload_f32(&s_init, &[N_HEADS * HD * HD]).unwrap();
    let tape_f16 = gpu
        .alloc_tensor(&[N_TOKENS * N_HEADS * HD * HD * 2], DType::Raw)
        .unwrap();
    let out_f16 = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();
    gpu.gated_delta_net_f16_tree_batch_seq(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &g_gpu,
        &b_gpu,
        &init_f16,
        &tape_f16,
        &parents_gpu,
        &out_f16,
        N_TOKENS,
        N_HEADS,
        HD,
    )
    .unwrap();
    let f16_host = gpu.download_f32(&out_f16).unwrap();

    // ---- Verdicts.
    let mut exact = 0usize;
    let mut max_abs_f32 = 0.0f32;
    for i in 0..ref_host.len() {
        if ref_host[i].to_bits() == f32_host[i].to_bits() {
            exact += 1;
        }
        max_abs_f32 = max_abs_f32.max((ref_host[i] - f32_host[i]).abs());
    }
    let denom: f32 = ref_host.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let mut max_abs_f16 = 0.0f32;
    for i in 0..ref_host.len() {
        max_abs_f16 = max_abs_f16.max((ref_host[i] - f16_host[i]).abs());
    }

    println!("elements: {}", ref_host.len());
    println!(
        "f32 tree vs per-token linear: {}/{} byte-exact, max|diff|={:.3e}",
        exact,
        ref_host.len(),
        max_abs_f32
    );
    println!(
        "f16 tree vs per-token linear: max|diff|={:.3e} (rel {:.3e}), ref max|x|={:.3e}",
        max_abs_f16,
        max_abs_f16 / denom.max(1e-30),
        denom
    );

    let mut failed = false;
    if exact != ref_host.len() {
        eprintln!(
            "FAIL: f32 tree must be byte-exact on a spine ({} of {} matched)",
            exact,
            ref_host.len()
        );
        failed = true;
    }
    // f16 keeps ~11 mantissa bits; over a 5-deep spine the round-trips stay well
    // inside 1e-2 relative. Loose enough not to be flaky, tight enough that a
    // genuinely wrong tape (wrong stride, wrong parent) blows straight past it.
    let rel_f16 = max_abs_f16 / denom.max(1e-30);
    if !(rel_f16 < 1e-2) {
        eprintln!("FAIL: f16 tree relative error {rel_f16:.3e} exceeds 1e-2");
        failed = true;
    }
    // If f16 came back byte-exact the tape is not actually f16 — that would mean
    // the kernel silently kept f32 storage and the memory saving is imaginary.
    if max_abs_f16 == 0.0 {
        eprintln!("FAIL: f16 tree is byte-exact; the tape is not narrowing to f16");
        failed = true;
    }

    if failed {
        std::process::exit(1);
    }
    println!("PASS");
}

#[cfg(feature = "deltanet")]
fn upload_i32(gpu: &mut hipfire_rdna::Gpu, data: &[i32]) -> hipfire_rdna::GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len() * 4], hipfire_rdna::DType::Raw)
        .unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}

#[cfg(feature = "deltanet")]
fn sin_det(i: usize, mul: usize) -> f32 {
    ((((i * mul * 2654435761) % 10007) as f32 / 10007.0) - 0.5) * 0.25
}

#[cfg(feature = "deltanet")]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
