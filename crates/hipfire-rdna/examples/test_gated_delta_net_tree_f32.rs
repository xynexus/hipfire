#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::useless_vec
)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for the non-Q8 DeltaNet kernels: the FP32/FP16 tree
//! variants and the FP16 state kernel.
//!
//! These exist so DDTree tree-replay does not require Q8 state, which
//! `qwen35/state.rs` forbids, and so "never Q8" stops meaning "always FP32".
//! Before the Q8 kernels can be retired, the replacements have to be shown to
//! actually work — a kernel that merely compiles proves nothing.
//!
//! The reference throughout is the per-token linear kernel called N times on a
//! rolling state, which is what tree spine semantics are defined to reproduce.
//! Four things are checked:
//!
//!   1. `gated_delta_net_f32_tree` on a spine is BYTE-EXACT against FP32 linear.
//!      That is why it uses the `col = tid * 4` lane mapping of the FP32 linear
//!      kernel rather than the Q8 family's stride-32 layout: summation order
//!      fixes the low bits, so exactness requires matching the kernel you claim
//!      exactness against.
//!   2. `gated_delta_net_f16` (f16 STATE) vs FP32 linear — the measurement of
//!      what half-precision state actually costs, with both arms started from
//!      bit-identical, exactly-f16-representable values so the comparison is of
//!      storage format and not of initial conditions.
//!   3. `gated_delta_net_f16_tree` on a spine is BYTE-EXACT against the f16
//!      LINEAR kernel. This is the strong check on the tree: same storage, same
//!      lane mapping, same summation order, so a wrong tape stride or a
//!      mis-resolved parent shows up here even though the loose FP32-relative
//!      bound in (4) would absorb it.
//!   4. Neither f16 arm is byte-exact against FP32. If either ever were, the
//!      storage would not be narrowing and the memory saving would be fictional
//!      — so equality is a FAILURE here, not a pass.
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
    // Deterministic initial state, O(1e-2) so the recurrence neither saturates
    // nor sits at zero (a zero state would hide tape bugs entirely).
    //
    // Chosen to be EXACTLY f16-representable — a small integer over 2^13 — so
    // the FP32 and FP16 runs start from bit-identical numbers. Otherwise the
    // f16 arm would begin from a slightly different state and the measurement
    // would conflate "cost of f16 storage" with "different initial condition".
    let s_init: Vec<f32> = (0..N_HEADS * HD * HD)
        .map(|i| ((((i * 5381) % 251) as i32) - 125) as f32 / 8192.0)
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

    // ---- f16 LINEAR, per-token, rolling f16 state. This is what the f16 tree
    // must reproduce on a spine, and — compared against `ref_host` — it is the
    // measurement of what f16 STATE actually costs.
    let state_f16 = upload_f16(&mut gpu, &s_init);
    let out_lin16 = gpu.zeros(&[N_TOKENS, N_HEADS * HD], DType::F32).unwrap();
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
        gpu.gated_delta_net_f16_batch_seq(&q1, &k1, &v1, &g1, &b1, &state_f16, &o1, 1, N_HEADS, HD)
            .unwrap();
        let row_bytes = N_HEADS * HD * 4;
        gpu.hip
            .memcpy_dtod_at(&out_lin16.buf, t * row_bytes, &o1.buf, 0, row_bytes)
            .unwrap();
        for t in [q1, k1, v1, g1, b1, o1] {
            gpu.free_tensor(t).unwrap();
        }
    }
    let lin16_host = gpu.download_f32(&out_lin16).unwrap();

    // f16 tree: init AND tape are f16 — HALF the bytes each, so allocate as Raw
    // with 2 bytes per element.
    let init_f16 = upload_f16(&mut gpu, &s_init);
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

    // f16 STATE cost: f16 linear vs f32 linear, same inputs, same start values.
    let mut max_abs_lin16 = 0.0f32;
    for i in 0..ref_host.len() {
        max_abs_lin16 = max_abs_lin16.max((ref_host[i] - lin16_host[i]).abs());
    }
    // f16 tree vs f16 linear: the parity that characterises the tree kernel.
    let mut exact16 = 0usize;
    let mut max_abs_tree16 = 0.0f32;
    for i in 0..ref_host.len() {
        if lin16_host[i].to_bits() == f16_host[i].to_bits() {
            exact16 += 1;
        }
        max_abs_tree16 = max_abs_tree16.max((lin16_host[i] - f16_host[i]).abs());
    }

    println!("elements: {}", ref_host.len());
    println!(
        "f32 tree  vs f32 linear: {}/{} byte-exact, max|diff|={:.3e}",
        exact,
        ref_host.len(),
        max_abs_f32
    );
    println!(
        "f16 STATE vs f32 linear: max|diff|={:.3e} (rel {:.3e})   <- cost of half-precision state",
        max_abs_lin16,
        max_abs_lin16 / denom.max(1e-30)
    );
    println!(
        "f16 tree  vs f16 linear: {}/{} byte-exact, max|diff|={:.3e}",
        exact16,
        ref_host.len(),
        max_abs_tree16
    );
    println!(
        "f16 tree  vs f32 linear: max|diff|={:.3e} (rel {:.3e}), ref max|x|={:.3e}",
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
    // If f16 came back byte-exact against FP32 the storage is not actually f16 —
    // the kernel kept f32 and the memory saving is imaginary.
    if max_abs_f16 == 0.0 {
        eprintln!("FAIL: f16 tree matches FP32 bit-for-bit; storage is not narrowing to f16");
        failed = true;
    }
    if max_abs_lin16 == 0.0 {
        eprintln!("FAIL: f16 linear matches FP32 bit-for-bit; state is not narrowing to f16");
        failed = true;
    }
    // The f16 tree on a spine must reproduce the f16 LINEAR kernel exactly —
    // same storage format, same lane mapping, same summation order. This is the
    // check that would catch a wrong tape stride or a mis-resolved parent, which
    // the loose FP32-relative bound above could absorb.
    if exact16 != ref_host.len() {
        eprintln!(
            "FAIL: f16 tree must be byte-exact against f16 linear on a spine ({} of {} matched, max|diff|={:.3e})",
            exact16,
            ref_host.len(),
            max_abs_tree16
        );
        failed = true;
    }
    // Half-precision state has to actually be usable, not merely different.
    let rel_lin16 = max_abs_lin16 / denom.max(1e-30);
    if !(rel_lin16 < 1e-2) {
        eprintln!("FAIL: f16 state relative error {rel_lin16:.3e} exceeds 1e-2");
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

/// f32 -> IEEE binary16 bits, round-to-nearest-even. Rust has no stable f16, and
/// pulling a crate in for a test that needs one conversion is not worth it.
/// Subnormals flush to zero: the test's values are all well inside normal range,
/// and a silent wrong answer there would show up as a failed exactness check
/// rather than passing quietly.
#[cfg(feature = "deltanet")]
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

#[cfg(feature = "deltanet")]
fn upload_f16(gpu: &mut hipfire_rdna::Gpu, data: &[f32]) -> hipfire_rdna::GpuTensor {
    let halves: Vec<u16> = data.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let t = gpu
        .alloc_tensor(&[data.len() * 2], hipfire_rdna::DType::Raw)
        .unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(halves.as_ptr() as *const u8, halves.len() * 2) };
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
