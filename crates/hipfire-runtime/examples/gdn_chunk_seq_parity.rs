// SPDX-License-Identifier: Apache-2.0
// hipfire — is the DeltaNet recurrence chunk-invariant?
//
//! `gated_delta_net_*_batch_seq` advances the GDN state for N tokens in one
//! launch; the per-token path advances it one token at a time. Batched prefill
//! and the speculative verify use the first, plain decode uses the second, so if
//! they disagree the same tokens leave the model in a different state — which is
//! what `docs/bugs/2026-09-01-spec-decode-not-output-equivalent-to-ar.md`
//! measures, and where `FIRST DIVERGING LAYER: 0` points (layer 0 is
//! `linear_attn`, and carries no KV at all).
//!
//! `speculative.rs` already treats these as NON-equivalent for FP16 — it replays
//! token by token because "f16 narrows the state once per launch" — while using
//! the batched call for FP32. This tests whether that FP32 assumption holds.
//!
//!   gdn_chunk_seq_parity [--n 64] [--chunk 17] [--heads 16] [--head-dim 128] [--f16]
use hipfire_rdna::{DType, Gpu};

fn arg(name: &str, default: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Minimal IEEE half decode; the workspace has no `half` dependency and adding
/// one for a debug example is not worth it.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // subnormal: renormalise
            let mut e = -1i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let e = (127 - 15 + e + 1) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

fn synth(len: usize, salt: u32, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut s = 0x9e3779b9u32 ^ salt;
    for _ in 0..len {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(((s >> 8) as f32 / 8388608.0 - 1.0) * scale);
    }
    v
}

fn main() {
    let n = arg("--n", 64);
    let chunk = arg("--chunk", 17);
    let n_heads = arg("--heads", 16);
    let head_dim = arg("--head-dim", 128);
    let vd = n_heads * head_dim;

    let mut gpu = Gpu::init().expect("gpu init");
    println!("gdn parity: n={n} chunk={chunk} heads={n_heads} head_dim={head_dim}");

    let q = synth(n * vd, 1, 1.0);
    let k = synth(n * vd, 2, 1.0);
    let v = synth(n * vd, 3, 1.0);
    // gate in (0,1) — it is a decay; beta small and positive.
    let gate: Vec<f32> = synth(n * n_heads, 4, 1.0)
        .iter()
        .map(|x| 0.5 + x * 0.25)
        .collect();
    let beta: Vec<f32> = synth(n * n_heads, 5, 1.0)
        .iter()
        .map(|x| 0.5 + x * 0.25)
        .collect();

    let up = |g: &mut Gpu, d: &[f32]| g.upload_f32(d, &[d.len()]).expect("upload");
    let qt = up(&mut gpu, &q);
    let kt = up(&mut gpu, &k);
    let vt = up(&mut gpu, &v);
    let gt = up(&mut gpu, &gate);
    let bt = up(&mut gpu, &beta);

    let f16 = std::env::args().any(|a| a == "--f16");
    let s_size = n_heads * head_dim * head_dim;
    let sdt = if f16 { DType::F16 } else { DType::F32 };
    let state_a = gpu.zeros(&[s_size], sdt).unwrap();
    let state_b = gpu.zeros(&[s_size], sdt).unwrap();
    println!("  state dtype: {}", if f16 { "F16" } else { "F32" });
    let out_a = gpu.zeros(&[n * vd], DType::F32).unwrap();
    let out_b = gpu.zeros(&[n * vd], DType::F32).unwrap();

    // A: chunked — `chunk` tokens per launch (batched prefill / spec verify).
    let mut p = 0usize;
    while p < n {
        let rows = chunk.min(n - p);
        let call = if f16 {
            Gpu::gated_delta_net_f16_batch_seq
        } else {
            Gpu::gated_delta_net_f32_batch_seq
        };
        call(
            &mut gpu,
            &qt.sub_offset(p * vd, rows * vd),
            &kt.sub_offset(p * vd, rows * vd),
            &vt.sub_offset(p * vd, rows * vd),
            &gt.sub_offset(p * n_heads, rows * n_heads),
            &bt.sub_offset(p * n_heads, rows * n_heads),
            &state_a,
            &out_a.sub_offset(p * vd, rows * vd),
            rows,
            n_heads,
            head_dim,
        )
        .expect("batch_seq");
        p += rows;
    }

    // B: one token per launch (plain decode).
    for t in 0..n {
        // The FP16 per-token path in speculative.rs is `f16_batch_seq` with
        // n_steps=1 in a loop -- mirror it exactly.
        let call1 = if f16 {
            Gpu::gated_delta_net_f16_batch_seq
        } else {
            Gpu::gated_delta_net_f32
        };
        call1(
            &mut gpu,
            &qt.sub_offset(t * vd, vd),
            &kt.sub_offset(t * vd, vd),
            &vt.sub_offset(t * vd, vd),
            &gt.sub_offset(t * n_heads, n_heads),
            &bt.sub_offset(t * n_heads, n_heads),
            &state_b,
            &out_b.sub_offset(t * vd, vd),
            1,
            n_heads,
            head_dim,
        )
        .expect("per-token");
    }

    let dls = |g: &Gpu, t: &hipfire_rdna::GpuTensor| -> Vec<f32> {
        if f16 {
            let mut b = vec![0u8; t.buf.size()];
            g.hip.memcpy_dtoh(&mut b, &t.buf).expect("dtoh");
            b.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        } else {
            g.download_f32(t).expect("dl")
        }
    };
    let sa = dls(&gpu, &state_a);
    let sb = dls(&gpu, &state_b);
    let oa = gpu.download_f32(&out_a).unwrap();
    let ob = gpu.download_f32(&out_b).unwrap();

    let relmax = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (x.abs().max(y.abs()).max(1e-6)))
            .fold(0.0f32, f32::max)
    };
    let nz = |a: &[f32]| a.iter().filter(|x| **x != 0.0).count();

    // Non-vacuity: a zero state or zero output means the kernel did nothing and
    // any "match" is meaningless. This investigation has been fooled by that
    // three times.
    if nz(&sa) == 0 || nz(&oa) == 0 {
        println!("INCONCLUSIVE — state or output is all zero; the recurrence did not run");
        std::process::exit(2);
    }
    let ds = relmax(&sa, &sb);
    let dof = relmax(&oa, &ob);
    let sbytes = sa
        .iter()
        .zip(&sb)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let obytes = oa
        .iter()
        .zip(&ob)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "  state : worst |rel| {ds:.3e}   {sbytes}/{} elements differ",
        sa.len()
    );
    println!(
        "  output: worst |rel| {dof:.3e}   {obytes}/{} elements differ",
        oa.len()
    );

    if sbytes == 0 && obytes == 0 {
        println!(
            "PASS — the {} GDN recurrence is chunk-invariant at these sizes",
            if f16 { "FP16" } else { "FP32" }
        );
    } else {
        println!(
            "FAIL — chunking changes the recurrence. Batched prefill and the speculative \
             verify advance the state differently from plain decode."
        );
        std::process::exit(1);
    }
}
