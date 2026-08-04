#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Finite-difference gradcheck for the ASSEMBLED linear-attention block:
//! `norm1 -> linear_attn -> residual -> norm2 -> MoE -> residual`.
//!
//! `gradcheck_linear_attn` covers the layer's math on the host and
//! `gradcheck_moe` covers the experts. This covers the assembly, which can be
//! wrong while both halves are right — and specifically covers the two joints
//! that only exist here:
//!
//!   * the GPU/host boundary. The projections run on device and the core on
//!     the host, so every adjoint crosses back and forth. A transposed upload
//!     or a stale download breaks `d_x` while both halves still gradcheck
//!     alone.
//!   * `d_xn2` threading. The MoE's input gradient is the only route from the
//!     MLP back into the attention half; dropping it leaves the MoE gradient
//!     perfect and the linear_attn gradient silently truncated.
//!
//! Run: cargo run --release -p hipfire-train --example gradcheck_la_block

use hipfire_rdna::{DType, Gpu};
use hipfire_train::la_block::{
    free_la_block_acts, la_block_backward, la_block_forward, LinearAttnBlockWeights,
};
use hipfire_train::ops::deltanet::LinearAttnDims;
use hipfire_train::ops::moe::{
    free_moe_acts, moe_backward, moe_forward, ExpertWeights, MoeDims, MoeWeights, SharedExpert,
};

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let d = LinearAttnDims {
        seq: 5,
        h: 16,
        n_heads: 2,
        hd_k: 5,
        hd_v: 3,
        conv_k: 4,
        eps: 1e-6,
    };
    let (seq, h, nh) = (d.seq, d.h, d.n_heads);
    let qkv_dim = nh * (2 * d.hd_k + d.hd_v);
    let vd = nh * d.hd_v;
    let md = MoeDims {
        seq,
        h,
        inter: 20,
        n_experts: 4,
        top_k: 2,
    };
    let (ne, inter) = (md.n_experts, md.inter);

    let mut s = 0xf00d5_u64;
    let rnd = |n: usize, s: &mut u64, k: f32| (0..n).map(|_| k * lcg(s)).collect::<Vec<f32>>();

    let x_host = rnd(seq * h, &mut s, 0.5);
    let seed = rnd(seq * h, &mut s, 1.0);

    let up = |gpu: &mut Gpu, v: &[f32]| gpu.upload_f32(v, &[v.len()]).unwrap();
    let norm1 = up(&mut gpu, &vec![1.0f32; h]);
    let norm2 = up(&mut gpu, &vec![1.0f32; h]);
    let wqkv = up(&mut gpu, &rnd(qkv_dim * h, &mut s, 0.3));
    // Larger, for the same reason as in gradcheck_linear_attn: the alpha and
    // beta paths are otherwise too small a share of d_x to test anything.
    let wa = up(&mut gpu, &rnd(nh * h, &mut s, 1.5));
    let wb = up(&mut gpu, &rnd(nh * h, &mut s, 1.5));
    let wz = up(&mut gpu, &rnd(vd * h, &mut s, 0.3));
    let wo = up(&mut gpu, &rnd(h * vd, &mut s, 0.3));
    let conv1d = rnd(qkv_dim * d.conv_k, &mut s, 0.5);
    let a_log = rnd(nh, &mut s, 0.3);
    let dt_bias = rnd(nh, &mut s, 0.3);
    let norm: Vec<f32> = (0..d.hd_v).map(|_| 1.0 + 0.2 * lcg(&mut s)).collect();

    let w = LinearAttnBlockWeights {
        norm1: &norm1,
        in_proj_qkv: &wqkv,
        in_proj_a: &wa,
        in_proj_b: &wb,
        in_proj_z: &wz,
        out_proj: &wo,
        norm2: &norm2,
        conv1d: &conv1d,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm: &norm,
    };

    let router = up(&mut gpu, &rnd(ne * h, &mut s, 0.3));
    let eg: Vec<_> = (0..ne)
        .map(|_| up(&mut gpu, &rnd(inter * h, &mut s, 0.3)))
        .collect();
    let eu: Vec<_> = (0..ne)
        .map(|_| up(&mut gpu, &rnd(inter * h, &mut s, 0.3)))
        .collect();
    let ed: Vec<_> = (0..ne)
        .map(|_| up(&mut gpu, &rnd(h * inter, &mut s, 0.3)))
        .collect();
    let sinter = 12usize;
    let sg = up(&mut gpu, &rnd(sinter * h, &mut s, 0.3));
    let su = up(&mut gpu, &rnd(sinter * h, &mut s, 0.3));
    let sd = up(&mut gpu, &rnd(h * sinter, &mut s, 0.3));
    let ssg = up(&mut gpu, &rnd(h, &mut s, 0.3));
    let moe_w = MoeWeights {
        router: &router,
        experts: (0..ne)
            .map(|e| ExpertWeights {
                wgate: &eg[e],
                wup: &eu[e],
                wdown: &ed[e],
            })
            .collect(),
        shared: Some(SharedExpert {
            w_scalar_gate: &ssg,
            wgate: &sg,
            wup: &su,
            wdown: &sd,
            inter: sinter,
        }),
    };

    // Full block: attention half, then MoE on xn2, added to x_mid.
    let forward =
        |gpu: &mut Gpu, xh: &[f32]| -> Result<(f64, Vec<u32>), Box<dyn std::error::Error>> {
            let xt = gpu.upload_f32(xh, &[seq * h])?;
            let acts = la_block_forward(gpu, &xt, &w, &d)?;
            let (moe_out, macts) = moe_forward(gpu, &acts.xn2, &moe_w, &md)?;
            let out = gpu.zeros(&[seq * h], DType::F32)?;
            gpu.add_f32(&acts.x_mid, &moe_out, &out)?;
            let l: f64 = gpu
                .download_f32(&out)?
                .iter()
                .zip(seed.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let idx = macts.idx.clone();
            free_moe_acts(gpu, macts)?;
            free_la_block_acts(gpu, acts)?;
            for t in [moe_out, out, xt] {
                gpu.free_tensor(t)?;
            }
            Ok((l, idx))
        };

    // Analytic pass.
    let xt = gpu.upload_f32(&x_host, &[seq * h])?;
    let acts = la_block_forward(&mut gpu, &xt, &w, &d)?;
    let (moe_out, macts) = moe_forward(&mut gpu, &acts.xn2, &moe_w, &md)?;
    let routing = macts.idx.clone();
    let d_out = gpu.upload_f32(&seed, &[seq * h])?;
    let (d_xn2, _moe_adj) = moe_backward(&mut gpu, &d_out, &moe_w, &macts, &md)?;
    let (d_x, adj) = la_block_backward(&mut gpu, &d_out, &d_xn2, &xt, &w, &acts, &d)?;
    let d_x_host = gpu.download_f32(&d_x)?;
    free_moe_acts(&mut gpu, macts)?;
    free_la_block_acts(&mut gpu, acts)?;
    for t in [moe_out, d_xn2, d_x, d_out, xt] {
        gpu.free_tensor(t)?;
    }

    println!("linear_attn + MoE block gradcheck: seq={seq} h={h} heads={nh} experts={ne}");
    println!(
        "  adjoints captured: d_qkv {} d_z {} d_out_proj {}",
        adj.d_qkv.len(),
        adj.d_z.len(),
        adj.d_out_proj.len()
    );

    // eps 1e-2, the value swept in gradcheck_moe_block: small enough that the
    // routing rarely flips, large enough to clear f32 cancellation in the loss.
    let eps = 1e-2f32;
    let (mut worst, mut worst_i, mut flips, mut checked) = (0.0f64, 0usize, 0usize, 0usize);
    for i in (0..seq * h).step_by(3) {
        let (mut a, mut b) = (x_host.clone(), x_host.clone());
        a[i] += eps;
        b[i] -= eps;
        let (lp, f1) = forward(&mut gpu, &a)?;
        let (lm, f2) = forward(&mut gpu, &b)?;
        if f1 != routing || f2 != routing {
            flips += 1;
            continue;
        }
        let num = (lp - lm) / (2.0 * eps as f64);
        let ana = d_x_host[i] as f64;
        let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-4);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
        checked += 1;
    }

    println!("  checked {checked} coords, {flips} skipped (routing flip)");
    println!("  worst relative error {worst:.3e} at coord {worst_i}");
    if worst < 1e-2 && checked > 0 {
        println!("\nPASS — linear_attn and MoE halves compose across the GPU/host boundary");
        Ok(())
    } else {
        println!("\nFAIL");
        std::process::exit(1)
    }
}
