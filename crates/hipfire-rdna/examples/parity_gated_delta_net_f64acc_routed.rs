// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Routed sibling of `parity_gated_delta_net_f64acc`: validates
//! `gated_delta_net_f64acc_routed_batch_seq` against an independent f64 CPU
//! reference, and the f32 routed kernel against the same one.
//!
//! This is the kernel that produced the "FP32 reference drifts more than FP16
//! does" measurement, because BATCHED decode dispatches the routed variant. The
//! plain oracle has a parity test; without this one the headline number rests on
//! an oracle whose sibling is the only validated part — which is why that entry
//! in BUGS.md is marked PROVISIONAL.
//!
//! Routing semantics, from the kernel: each session walks ALL batch rows in
//! order and skips the ones that are not its own (`row_session_indices[b] !=
//! session_idx`), so a session sees its rows in batch order. Sessions are
//! interleaved deliberately here — a reference that processed each session's
//! rows contiguously would agree with a kernel that ignored routing entirely.
//!
//! Expected, mirroring the plain test: the oracle sits at the FP32 STORAGE floor
//! (~1e-8, since it accumulates in double but stores f32), and the f32 kernel an
//! order of magnitude worse. If they matched, the oracle would be measuring
//! nothing.
//!
//!   cargo run --release -p hipfire-rdna --features deltanet \
//!       --example parity_gated_delta_net_f64acc_routed

use hipfire_rdna::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const N_HEADS: usize = 2;
const N_SESSIONS: usize = 3;
const BATCH_ROWS: usize = 12;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 + 0.5) / 2_147_483_648.0) * 2.0 - 1.0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    states0: &[Vec<f32>],
    row_session: &[i32],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let stride = N_HEADS * HD;
    let mut s: Vec<Vec<f64>> = states0
        .iter()
        .map(|st| st.iter().map(|&x| x as f64).collect())
        .collect();
    let mut out = vec![0.0f64; BATCH_ROWS * stride];
    for sess in 0..N_SESSIONS {
        for b in 0..BATCH_ROWS {
            if row_session[b] as usize != sess {
                continue;
            }
            for h in 0..N_HEADS {
                let alpha = (gate[b * N_HEADS + h] as f64).exp();
                let beta_v = beta[b * N_HEADS + h] as f64;
                let base = b * stride + h * HD;
                for r in 0..HD {
                    let row = h * HD * HD + r * HD;
                    let mut kv = 0.0f64;
                    for c in 0..HD {
                        kv += s[sess][row + c] * k[base + c] as f64;
                    }
                    let delta = (v[base + r] as f64 - alpha * kv) * beta_v;
                    let mut acc = 0.0f64;
                    for c in 0..HD {
                        s[sess][row + c] = alpha * s[sess][row + c] + k[base + c] as f64 * delta;
                        acc += s[sess][row + c] * q[base + c] as f64;
                    }
                    out[base + r] = acc;
                }
            }
        }
    }
    (s, out)
}

fn up_f32(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_raw(
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        &[v.len()],
    )
    .unwrap()
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((*g as f64) - w).powi(2);
        den += w * w;
    }
    (num / den.max(1e-300)).sqrt()
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let stride = N_HEADS * HD;
    let q = lcg(11, BATCH_ROWS * stride);
    let k = lcg(12, BATCH_ROWS * stride);
    let v = lcg(13, BATCH_ROWS * stride);
    let gate: Vec<f32> = lcg(14, BATCH_ROWS * N_HEADS)
        .iter()
        .map(|x| x * 0.02)
        .collect();
    let beta: Vec<f32> = lcg(15, BATCH_ROWS * N_HEADS)
        .iter()
        .map(|x| x * 0.5)
        .collect();
    // Interleaved on purpose — see the module doc.
    let row_session: Vec<i32> = vec![0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1, 2];
    let states0: Vec<Vec<f32>> = (0..N_SESSIONS)
        .map(|i| {
            lcg(20 + i as u32, N_HEADS * HD * HD)
                .iter()
                .map(|x| x * 0.1)
                .collect()
        })
        .collect();

    let (ref_states, ref_out) = cpu_reference(&q, &k, &v, &gate, &beta, &states0, &row_session);

    let qd = up_f32(&mut gpu, &q);
    let kd = up_f32(&mut gpu, &k);
    let vd = up_f32(&mut gpu, &v);
    let gd = up_f32(&mut gpu, &gate);
    let bd = up_f32(&mut gpu, &beta);
    let od = up_f32(&mut gpu, &vec![0.0f32; BATCH_ROWS * stride]);
    let sds: Vec<GpuTensor> = states0.iter().map(|st| up_f32(&mut gpu, st)).collect();
    let rsi = gpu
        .upload_raw(
            &row_session
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
            &[BATCH_ROWS],
        )
        .unwrap();
    // Session-major pointer table, ptr_layer_stride = 1, layer 0. u64 per entry,
    // uploaded as an F32-typed buffer (two elements per pointer) — the same
    // convention the batch pointer tables use.
    let ptr_bytes: Vec<u8> = sds
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as usize as u64).to_le_bytes())
        .collect();
    let sptr = gpu.upload_raw(&ptr_bytes, &[N_SESSIONS * 2]).unwrap();
    let _ = DType::F32;

    gpu.gated_delta_net_f32_routed_batch_seq(
        &qd, &kd, &vd, &gd, &bd, &sptr, &rsi, &od, 1, 0, BATCH_ROWS, N_HEADS, HD, N_SESSIONS,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();

    let oracle_on = std::env::var("HIPFIRE_DN_STATE_F64_ORACLE").ok().as_deref() == Some("1");
    let label = if oracle_on {
        "f64acc ORACLE (routed)"
    } else {
        "f32 kernel (routed)"
    };

    let mut worst_state = 0.0f64;
    for (i, sd) in sds.iter().enumerate() {
        let got = gpu.download_f32(sd).unwrap();
        let e = rel_err(&got, &ref_states[i]);
        worst_state = worst_state.max(e);
        println!("  session {i} state rel L2 err: {e:.6e}");
    }
    let out_err = rel_err(&gpu.download_f32(&od).unwrap(), &ref_out);
    println!("{label}: rows={BATCH_ROWS} sessions={N_SESSIONS} heads={N_HEADS} hd={HD}");
    println!("  worst state rel L2 err vs f64 CPU reference: {worst_state:.6e}");
    println!("  output      rel L2 err vs f64 CPU reference: {out_err:.6e}");

    let bound = if oracle_on { 1e-7 } else { 1e-2 };
    if worst_state < bound && out_err < bound {
        println!("PASS (bound {bound:.0e})");
    } else {
        println!("FAIL: exceeded {bound:.0e}");
        std::process::exit(1);
    }
}
