// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-check the rest of the `linear_attn` core against the inference
//! kernels, the way `verify_deltanet_vs_kernel` does for the recurrence.
//!
//! Those two together cover the whole core. What each of these settles is a
//! LAYOUT or FORMULA question that no self-consistent host test can reach:
//!
//!   * **conv1d + SiLU + split.** Which tap is "now" and which is three tokens
//!     back, and whether the channel block order is `[Q|K|V]`. Reversing the
//!     taps is still a valid causal conv over the same weights, and swapping Q
//!     with K is shape-identical whenever `hd_k == hd_v` — which the 35B has.
//!   * **gated norm.** Whether `weight` multiplies the normalised value or the
//!     gate, and whether the gate is `silu(z)` or `sigmoid(z)`. Both variants
//!     are smooth, both train, and the wrong one is only visibly wrong in the
//!     logits.
//!   * **alpha / beta.** That `alpha = exp(softplus(a + dt_bias) * -exp(A_log))`
//!     and `beta = sigmoid(b)`, against the kernel that computes them.
//!   * **q/k L2 norm + q scale.** Whether that stage exists at all. It was
//!     missing from this core until a real model at seq 64 produced NaN and a
//!     worse-than-uniform loss — the state is unbounded without it. Nothing
//!     self-consistent could have found that: the layer gradchecked clean,
//!     stayed causal, and ran.
//!
//! Run: cargo run --release -p hipfire-train --features deltanet \
//!        --example verify_la_core_vs_kernels

use hipfire_rdna::{DType, Gpu};
use hipfire_train::ops::deltanet::{linear_attn_core_forward, LinearAttnCore, LinearAttnDims};

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn worst(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut w = 0.0f32;
    let mut mag = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        w = w.max((x - y).abs());
        mag = mag.max(y.abs());
    }
    (w, w / mag.max(1e-6))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    // head_dim 128 for both k and v: the real geometry, and the width the
    // recurrence kernel is compiled for.
    // Configurable so the REAL geometries can be checked, not just a toy one.
    // The 35B is 32 value heads against 16 key heads; every earlier run of this
    // file used 2/2, which exercises neither the head count nor the repeat.
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let d = LinearAttnDims {
        seq: *a.first().unwrap_or(&5),
        h: *a.get(3).unwrap_or(&64),
        n_heads: *a.get(1).unwrap_or(&2),
        n_k_heads: *a.get(2).unwrap_or(&0),
        hd_k: 128,
        hd_v: 128,
        conv_k: 4,
        eps: 1e-6,
    };
    let (seq, nh, hk, hv) = (d.seq, d.n_heads, d.hd_k, d.hd_v);
    let nk = d.nk();
    let qkv_dim = 2 * nk * hk + nh * hv;
    let (k_dim, v_dim) = (nk * hk, nh * hv);
    println!(
        "  geometry: seq={seq} v_heads={nh} k_heads={nk} hd={hk} h={} qkv={qkv_dim}",
        d.h
    );

    let mut s = 0xc0ffee_u64;
    let rnd = |n: usize, s: &mut u64, k: f32| (0..n).map(|_| k * lcg(s)).collect::<Vec<f32>>();

    let qkv = rnd(seq * qkv_dim, &mut s, 0.5);
    let a_raw = rnd(seq * nh, &mut s, 0.5);
    let b_raw = rnd(seq * nh, &mut s, 0.5);
    let z = rnd(seq * nh * hv, &mut s, 0.5);
    let conv1d = rnd(qkv_dim * d.conv_k, &mut s, 0.5);
    let a_log = rnd(nh, &mut s, 0.3);
    let dt_bias = rnd(nh, &mut s, 0.3);
    let norm: Vec<f32> = (0..hv).map(|_| 1.0 + 0.2 * lcg(&mut s)).collect();

    let core = LinearAttnCore {
        conv1d: &conv1d,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm: &norm,
    };
    let (normed, acts) = linear_attn_core_forward(&qkv, &a_raw, &b_raw, &z, &core, &d);

    println!("linear_attn core vs inference kernels: seq={seq} heads={nh} hd={hk}");
    let mut ok = true;

    // ── conv1d + SiLU + [Q|K|V] split ────────────────────────────────────
    {
        let inp = gpu.upload_f32(&qkv, &[seq * qkv_dim])?;
        let w = gpu.upload_f32(&conv1d, &[qkv_dim * d.conv_k])?;
        // Ring buffer of conv_k-1 past samples per channel, zero for a fresh
        // sequence — which is what the host's left zero-padding means.
        let state = gpu.zeros(&[qkv_dim * (d.conv_k - 1)], DType::F32)?;
        let qo = gpu.zeros(&[seq * k_dim], DType::F32)?;
        let ko = gpu.zeros(&[seq * k_dim], DType::F32)?;
        let vo = gpu.zeros(&[seq * v_dim], DType::F32)?;
        gpu.conv1d_silu_split_f32_n(&qo, &ko, &vo, &inp, &w, &state, k_dim, v_dim, seq)?;
        let (gq, gk, gv) = (
            gpu.download_f32(&qo)?,
            gpu.download_f32(&ko)?,
            gpu.download_f32(&vo)?,
        );
        // acts.q/k are POST-L2-norm; the conv kernel emits the pre-norm value,
        // which the core keeps as q_raw/k_raw.
        for (name, host, dev) in [
            ("q", &acts.q_raw, &gq),
            ("k", &acts.k_raw, &gk),
            ("v", &acts.v, &gv),
        ] {
            let (abs, rel) = worst(host, dev);
            println!("  conv+silu+split {name}: worst {abs:.3e} (rel {rel:.3e})");
            ok &= rel < 1e-5;
        }
        for t in [inp, w, state, qo, ko, vo] {
            gpu.free_tensor(t)?;
        }
    }

    // ── q/k L2 norm + q scale ────────────────────────────────────────────
    {
        // The stage between the conv split and the recurrence. Omitting it
        // entirely still gradchecks and still runs; it shows up only as an
        // unbounded state on a real model at real sequence length.
        let qt = gpu.upload_f32(&acts.q_raw, &[seq * nk * hk])?;
        let kt = gpu.upload_f32(&acts.k_raw, &[seq * nk * hk])?;
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &qt,
            &kt,
            nk,
            hk,
            1.0 / (hk as f32).sqrt(),
            d.eps,
            seq,
        )?;
        let (gq, gk) = (gpu.download_f32(&qt)?, gpu.download_f32(&kt)?);
        // acts.q/k are POST-repeat (nh heads). The repeat is
        // dst[kh*ratio + r] = src[kh], so key head kh lands at value head
        // kh*ratio — gather those back out rather than slicing the front,
        // which would only be right when nk == nh.
        let rep = nh / nk;
        let gather = |v: &[f32]| -> Vec<f32> {
            let mut o = Vec::with_capacity(seq * nk * hk);
            for t in 0..seq {
                for kh in 0..nk {
                    let src = t * nh * hk + (kh * rep) * hk;
                    o.extend_from_slice(&v[src..src + hk]);
                }
            }
            o
        };
        let (aq, rq) = worst(&gather(&acts.q), &gq);
        let (ak, rk) = worst(&gather(&acts.k), &gk);
        println!("  l2norm(q)*1/sqrt(hd): worst {aq:.3e} (rel {rq:.3e})");
        println!("  l2norm(k):            worst {ak:.3e} (rel {rk:.3e})");
        ok &= rq < 1e-5 && rk < 1e-5;
        gpu.free_tensor(qt)?;
        gpu.free_tensor(kt)?;
    }

    // ── alpha / beta activations ─────────────────────────────────────────
    {
        // The kernel transforms in place, one token at a time.
        let mut host_alpha = Vec::with_capacity(seq * nh);
        let mut host_beta = Vec::with_capacity(seq * nh);
        let dtb = gpu.upload_f32(&dt_bias, &[nh])?;
        let alg = gpu.upload_f32(&a_log, &[nh])?;
        for t in 0..seq {
            let bt = gpu.upload_f32(&b_raw[t * nh..(t + 1) * nh], &[nh])?;
            let at = gpu.upload_f32(&a_raw[t * nh..(t + 1) * nh], &[nh])?;
            gpu.fused_sigmoid_alpha_gate_f32(&bt, &at, &dtb, &alg, nh)?;
            host_beta.extend(gpu.download_f32(&bt)?);
            // The kernel leaves the GATE in the alpha buffer; the recurrence
            // exponentiates it. The host stores the same pre-exp quantity.
            host_alpha.extend(gpu.download_f32(&at)?);
            gpu.free_tensor(bt)?;
            gpu.free_tensor(at)?;
        }
        gpu.free_tensor(dtb)?;
        gpu.free_tensor(alg)?;
        let (ab, rb) = worst(&acts.beta, &host_beta);
        let (aa, ra) = worst(&acts.gate, &host_alpha);
        println!("  beta = sigmoid(b): worst {ab:.3e} (rel {rb:.3e})");
        println!("  gate = softplus(a+dt)*-exp(A_log): worst {aa:.3e} (rel {ra:.3e})");
        ok &= rb < 1e-5 && ra < 1e-5;
    }

    // ── gated norm ───────────────────────────────────────────────────────
    {
        let x = gpu.upload_f32(&acts.dn_out, &[seq * nh * hv])?;
        let zt = gpu.upload_f32(&z, &[seq * nh * hv])?;
        let w = gpu.upload_f32(&norm, &[hv])?;
        let out = gpu.zeros(&[seq * nh * hv], DType::F32)?;
        gpu.gated_norm_f32_batched(&x, &zt, &w, &out, nh, hv, d.eps, seq)?;
        let g = gpu.download_f32(&out)?;
        let (abs, rel) = worst(&normed, &g);
        println!("  gated norm rmsnorm(x)*w*silu(z): worst {abs:.3e} (rel {rel:.3e})");
        ok &= rel < 1e-5;
        for t in [x, zt, w, out] {
            gpu.free_tensor(t)?;
        }
    }

    if ok {
        println!("\nPASS — the core's layouts and formulas are the inference path's");
        Ok(())
    } else {
        println!("\nFAIL");
        std::process::exit(1)
    }
}
