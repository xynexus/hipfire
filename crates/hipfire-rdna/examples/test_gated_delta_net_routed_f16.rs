#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for `gated_delta_net_f16_routed_batch_seq`.
//!
//! The routed kernel exists to fix a specific bug: sharing one DeltaNet state
//! across independent request sessions. So the test that matters is not "does
//! it run" but "are the sessions actually isolated" — a kernel that ignored
//! `row_session_indices` entirely would still produce plausible numbers.
//!
//! Rows are INTERLEAVED between two sessions (s0, s1, s0, s1, ...) so a kernel
//! that leaked state between them, or applied rows in flat order, cannot pass.
//! Each session's rows are then replayed independently through the f16 LINEAR
//! kernel, and the routed output must match BYTE-EXACTLY: same storage format,
//! same lane mapping, same per-session recurrence order, so there is no slack to
//! hide a difference in.
//!
//! Build: `cargo run --release --features deltanet \
//!   --example test_gated_delta_net_routed_f16 -p hipfire-rdna`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_gated_delta_net_routed_f16 requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_rdna::{DType, Gpu};
    const HD: usize = 128;
    const N_HEADS: usize = 4;
    const N_SESSIONS: usize = 2;
    const ROWS: usize = 6; // interleaved: s0 s1 s0 s1 s0 s1

    let mut gpu = Gpu::init().expect("GPU init");

    let n_el = ROWS * N_HEADS * HD;
    let q: Vec<f32> = (0..n_el).map(|i| sin_det(i, 3)).collect();
    let k: Vec<f32> = (0..n_el).map(|i| sin_det(i, 5)).collect();
    let v: Vec<f32> = (0..n_el).map(|i| sin_det(i, 7)).collect();
    let gate: Vec<f32> = (0..ROWS * N_HEADS)
        .map(|i| sin_det(i, 11) * 0.1 - 0.5)
        .collect();
    let beta: Vec<f32> = (0..ROWS * N_HEADS)
        .map(|i| sigmoid(sin_det(i, 13)))
        .collect();
    let row_session: Vec<i32> = (0..ROWS as i32).map(|b| b % N_SESSIONS as i32).collect();

    // Distinct initial state PER SESSION, exactly f16-representable. Distinct
    // matters: identical states would let a leak between sessions go unnoticed.
    let s_init: Vec<Vec<f32>> = (0..N_SESSIONS)
        .map(|s| {
            (0..N_HEADS * HD * HD)
                .map(|i| ((((i * 5381 + s * 977) % 251) as i32) - 125) as f32 / 8192.0)
                .collect()
        })
        .collect();

    // ---- Routed: one launch, both sessions, interleaved rows.
    let q_gpu = gpu.upload_f32(&q, &[ROWS, N_HEADS * HD]).unwrap();
    let k_gpu = gpu.upload_f32(&k, &[ROWS, N_HEADS * HD]).unwrap();
    let v_gpu = gpu.upload_f32(&v, &[ROWS, N_HEADS * HD]).unwrap();
    let g_gpu = gpu.upload_f32(&gate, &[ROWS, N_HEADS]).unwrap();
    let b_gpu = gpu.upload_f32(&beta, &[ROWS, N_HEADS]).unwrap();
    let rs_gpu = upload_i32(&mut gpu, &row_session);

    let states: Vec<_> = (0..N_SESSIONS)
        .map(|s| upload_f16(&mut gpu, &s_init[s]))
        .collect();
    // One pointer per (session, layer); a single layer here, so stride 1.
    let ptrs: Vec<u64> = states.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let ptrs_gpu = upload_u64(&mut gpu, &ptrs);
    let out_routed = gpu.zeros(&[ROWS, N_HEADS * HD], DType::F32).unwrap();

    gpu.gated_delta_net_f16_routed_batch_seq(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &g_gpu,
        &b_gpu,
        &ptrs_gpu,
        &rs_gpu,
        &out_routed,
        1, // ptr_layer_stride
        0, // delta_layer_index
        ROWS,
        N_HEADS,
        HD,
        N_SESSIONS,
    )
    .unwrap();
    let routed_host = gpu.download_f32(&out_routed).unwrap();

    // ---- Reference: each session's rows GATHERED into a contiguous batch and
    // replayed in ONE f16 linear call.
    //
    // One call per session, not one per row. Both kernels widen the f16 state
    // into FP32 LDS once per launch and narrow once at the end, so a per-ROW
    // reference would round-trip through f16 at every step while the routed
    // kernel does not — a real difference in precision, not a bug, and it shows
    // up as ~1e-6 disagreement on every row after the first. (The existing Q8
    // tree example flags the same trap: the batched kernel holds S in f32 across
    // the batch, so it is not interchangeable with N single-token calls.)
    //
    // Gathering also means the reference never sees the other session's rows,
    // which is exactly the isolation property under test.
    let mut ref_host = vec![0.0f32; ROWS * N_HEADS * HD];
    for s in 0..N_SESSIONS {
        let rows: Vec<usize> = (0..ROWS)
            .filter(|&b| row_session[b] as usize == s)
            .collect();
        let n = rows.len();
        let mut qs = Vec::with_capacity(n * N_HEADS * HD);
        let mut ks = Vec::with_capacity(n * N_HEADS * HD);
        let mut vs = Vec::with_capacity(n * N_HEADS * HD);
        let mut gs = Vec::with_capacity(n * N_HEADS);
        let mut bs = Vec::with_capacity(n * N_HEADS);
        for &b in &rows {
            qs.extend_from_slice(&q[b * N_HEADS * HD..(b + 1) * N_HEADS * HD]);
            ks.extend_from_slice(&k[b * N_HEADS * HD..(b + 1) * N_HEADS * HD]);
            vs.extend_from_slice(&v[b * N_HEADS * HD..(b + 1) * N_HEADS * HD]);
            gs.extend_from_slice(&gate[b * N_HEADS..(b + 1) * N_HEADS]);
            bs.extend_from_slice(&beta[b * N_HEADS..(b + 1) * N_HEADS]);
        }
        let state = upload_f16(&mut gpu, &s_init[s]);
        let qg = gpu.upload_f32(&qs, &[n, N_HEADS * HD]).unwrap();
        let kg = gpu.upload_f32(&ks, &[n, N_HEADS * HD]).unwrap();
        let vg = gpu.upload_f32(&vs, &[n, N_HEADS * HD]).unwrap();
        let gg = gpu.upload_f32(&gs, &[n, N_HEADS]).unwrap();
        let bg = gpu.upload_f32(&bs, &[n, N_HEADS]).unwrap();
        let og = gpu.zeros(&[n, N_HEADS * HD], DType::F32).unwrap();
        gpu.gated_delta_net_f16_batch_seq(&qg, &kg, &vg, &gg, &bg, &state, &og, n, N_HEADS, HD)
            .unwrap();
        let got = gpu.download_f32(&og).unwrap();
        // Scatter back to the interleaved row positions.
        for (i, &b) in rows.iter().enumerate() {
            let dst = b * N_HEADS * HD;
            let src = i * N_HEADS * HD;
            ref_host[dst..dst + N_HEADS * HD].copy_from_slice(&got[src..src + N_HEADS * HD]);
        }
        for t in [qg, kg, vg, gg, bg, og, state] {
            gpu.free_tensor(t).unwrap();
        }
    }

    // ---- Verdict.
    let mut exact = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..ref_host.len() {
        if ref_host[i].to_bits() == routed_host[i].to_bits() {
            exact += 1;
        }
        max_abs = max_abs.max((ref_host[i] - routed_host[i]).abs());
    }
    println!(
        "rows={ROWS} sessions={N_SESSIONS} interleaved, elements={}",
        ref_host.len()
    );
    println!(
        "routed f16 vs per-session f16 linear: {}/{} byte-exact, max|diff|={:.3e}",
        exact,
        ref_host.len(),
        max_abs
    );
    if exact != ref_host.len() {
        // WORK BACKWARDS: localise the mismatch instead of reading the kernels.
        // Which rows/heads/columns differ says immediately whether it is
        // accumulation, routing, or (as it was) a per-kernel compile difference.
        let mut per_row = vec![0usize; ROWS];
        let mut per_head = vec![0usize; N_HEADS];
        let mut per_col = vec![0usize; HD];
        for b in 0..ROWS {
            for h in 0..N_HEADS {
                for c in 0..HD {
                    let i = b * N_HEADS * HD + h * HD + c;
                    if ref_host[i].to_bits() != routed_host[i].to_bits() {
                        per_row[b] += 1;
                        per_head[h] += 1;
                        per_col[c] += 1;
                    }
                }
            }
        }
        eprintln!(
            "mismatches by ROW (session): {:?}",
            (0..ROWS)
                .map(|b| (b, row_session[b], per_row[b]))
                .collect::<Vec<_>>()
        );
        eprintln!("mismatches by HEAD: {per_head:?}");
        let cols: Vec<usize> = (0..HD).filter(|&c| per_col[c] > 0).collect();
        eprintln!(
            "mismatching COLS ({} of {HD}): {:?}",
            cols.len(),
            &cols[..cols.len().min(40)]
        );
        let lanes16: Vec<usize> = (0..8)
            .filter(|&l| (0..16).any(|j| per_col[l * 16 + j] > 0))
            .collect();
        eprintln!("affected 16-col lane groups: {lanes16:?}");
    }

    // A leak between sessions, or flat-order application, shows up here — the
    // two sessions start from deliberately different states, so any crosstalk
    // moves the numbers well outside a rounding difference.
    if exact != ref_host.len() {
        eprintln!(
            "FAIL: routed must be byte-exact against independent per-session replay \
             ({} of {} matched, max|diff|={max_abs:.3e}) — sessions are not isolated",
            exact,
            ref_host.len()
        );
        std::process::exit(1);
    }
    println!("PASS");
}

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
fn upload_u64(gpu: &mut hipfire_rdna::Gpu, data: &[u64]) -> hipfire_rdna::GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len() * 8], hipfire_rdna::DType::Raw)
        .unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
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
