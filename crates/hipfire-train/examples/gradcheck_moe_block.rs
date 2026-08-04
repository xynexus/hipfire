#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Finite-difference gradcheck for the ASSEMBLED routed-MoE block.
//!
//! `gradcheck_moe` already covers the MoE MLP in isolation and `gradcheck_block`
//! covers the dense block. This checks the composition, which can be wrong even
//! when both halves are right — specifically that `d_xn2` from `moe_backward`
//! is threaded into `norm2`'s backward and reaches the attention half. Zeroing
//! it would leave the MoE gradient correct and the ATTENTION gradient silently
//! wrong, which neither of the other two checks would notice.
//!
//! Compares analytic `d_x` (the block input gradient, which only exists if the
//! whole chain composed) against central differences of
//! `L = <seed, moe_block_forward(x)>`.
//!
//! Run: cargo run --release -p hipfire-train --example gradcheck_moe_block

use hipfire_rdna::Gpu;
use hipfire_train::block::{
    free_block_acts, moe_block_backward_capture, moe_block_forward, BlockDims, BlockLora,
    BlockWeights,
};
use hipfire_train::ops::moe::{free_moe_acts, ExpertWeights, MoeDims, MoeWeights};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    const SEQ: usize = 4;
    const NH: usize = 2;
    const NKV: usize = 1;
    const HD: usize = 8;
    const H: usize = NH * HD;
    const INTER: usize = 24;
    const NE: usize = 4;
    const TOPK: usize = 2;
    const R: usize = 1;

    let dims = BlockDims {
        seq: SEQ,
        h: H,
        n_heads: NH,
        n_kv: NKV,
        head_dim: HD,
        inter: INTER,
        rope_base: 10000.0,
        eps: 1e-6,
        lora_scale: 1.0,
        lora_rank: R,
    };
    let md = MoeDims {
        seq: SEQ,
        h: H,
        inter: INTER,
        n_experts: NE,
        top_k: TOPK,
    };
    let (qd, kvd) = (NH * HD, NKV * HD);

    let mut s = 0xa11ce_u64;
    let rnd = |n: usize, s: &mut u64| (0..n).map(|_| 0.3 * lcg(s)).collect::<Vec<f32>>();
    let ones = |n: usize| vec![1.0f32; n];

    let x_host = rnd(SEQ * H, &mut s);
    let seed = rnd(SEQ * H, &mut s);

    let norm1 = gpu.upload_f32(&ones(H), &[H])?;
    let norm2 = gpu.upload_f32(&ones(H), &[H])?;
    let wq = gpu.upload_f32(&rnd(qd * H, &mut s), &[qd * H])?;
    let wk = gpu.upload_f32(&rnd(kvd * H, &mut s), &[kvd * H])?;
    let wv = gpu.upload_f32(&rnd(kvd * H, &mut s), &[kvd * H])?;
    let wo = gpu.upload_f32(&rnd(H * qd, &mut s), &[H * qd])?;
    // Dense MLP weights are unused on this path but BlockWeights wants them.
    let wg = gpu.upload_f32(&rnd(INTER * H, &mut s), &[INTER * H])?;
    let wu = gpu.upload_f32(&rnd(INTER * H, &mut s), &[INTER * H])?;
    let wd = gpu.upload_f32(&rnd(H * INTER, &mut s), &[H * INTER])?;
    let w = BlockWeights {
        norm1: &norm1,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wo: &wo,
        norm2: &norm2,
        wgate: &wg,
        wup: &wu,
        wdown: &wd,
    };

    // Zero LoRA (B = 0) so the block sits exactly at the base weights.
    let z = |gpu: &mut Gpu, n: usize| gpu.zeros(&[n], hipfire_rdna::DType::F32).unwrap();
    let aq = z(&mut gpu, R * H);
    let bq = z(&mut gpu, qd * R);
    let av = z(&mut gpu, R * H);
    let bv = z(&mut gpu, kvd * R);
    let lora = BlockLora {
        aq: &aq,
        bq: &bq,
        av: &av,
        bv: &bv,
    };

    let router = gpu.upload_f32(&rnd(NE * H, &mut s), &[NE * H])?;
    let eg: Vec<_> = (0..NE)
        .map(|_| {
            gpu.upload_f32(&rnd(INTER * H, &mut s), &[INTER * H])
                .unwrap()
        })
        .collect();
    let eu: Vec<_> = (0..NE)
        .map(|_| {
            gpu.upload_f32(&rnd(INTER * H, &mut s), &[INTER * H])
                .unwrap()
        })
        .collect();
    let ed: Vec<_> = (0..NE)
        .map(|_| {
            gpu.upload_f32(&rnd(H * INTER, &mut s), &[H * INTER])
                .unwrap()
        })
        .collect();
    let moe_w = MoeWeights {
        router: &router,
        experts: (0..NE)
            .map(|e| ExpertWeights {
                wgate: &eg[e],
                wup: &eu[e],
                wdown: &ed[e],
            })
            .collect(),
        shared: None,
    };

    let pos: Vec<f32> = (0..SEQ).map(|i| i as f32).collect();

    let x = gpu.upload_f32(&x_host, &[SEQ * H])?;
    let (out, acts, macts) =
        moe_block_forward(&mut gpu, &x, &w, &moe_w, &lora, &dims, &md, &pos, 0)?;
    let routing = macts.idx.clone();
    let d_out = gpu.upload_f32(&seed, &[SEQ * H])?;
    let (d_x, adj, moe_adj) = moe_block_backward_capture(
        &mut gpu, &d_out, &x, &w, &moe_w, &lora, &acts, &macts, &dims, &md,
    )?;
    let d_x_host = gpu.download_f32(&d_x)?;
    free_block_acts(&mut gpu, acts)?;
    free_moe_acts(&mut gpu, macts)?;
    gpu.free_tensor(out)?;

    let served: Vec<usize> = (0..NE).map(|e| moe_adj.d_expert_out[e].len() / H).collect();
    println!("MoE block gradcheck: seq={SEQ} h={H} experts={NE} top_k={TOPK}");
    println!("  rows per expert: {served:?}");
    println!(
        "  attention adjoints captured: d_q {} d_attn {}",
        adj.d_q.len(),
        adj.d_attn.len()
    );

    // eps balances truncation (too large) against f32 cancellation in the loss
    // difference (too small). Swept once: 1.084e-2 at 1e-3 (cancellation),
    // 3.078e-3 at 1e-2, 2.299e-2 at 3e-2 (truncation). Fixed at the measured
    // optimum — a knob here would be scanned into the product env-var docs,
    // which is not what a gradcheck step size is.
    let eps: f32 = 1e-2;
    let (mut worst, mut worst_i, mut flips, mut checked) = (0.0f32, 0usize, 0usize, 0usize);
    for i in (0..SEQ * H).step_by(5) {
        let probe = |delta: f32,
                     gpu: &mut Gpu|
         -> Result<(f64, bool), Box<dyn std::error::Error>> {
            let mut xp = x_host.clone();
            xp[i] += delta;
            let xt = gpu.upload_f32(&xp, &[SEQ * H])?;
            let (o, a, ma) = moe_block_forward(gpu, &xt, &w, &moe_w, &lora, &dims, &md, &pos, 0)?;
            let flipped = ma.idx != routing;
            let l: f64 = gpu
                .download_f32(&o)?
                .iter()
                .zip(seed.iter())
                .map(|(p, q)| (*p as f64) * (*q as f64))
                .sum();
            free_block_acts(gpu, a)?;
            free_moe_acts(gpu, ma)?;
            gpu.free_tensor(o)?;
            gpu.free_tensor(xt)?;
            Ok((l, flipped))
        };
        let (lp, f1) = probe(eps, &mut gpu)?;
        let (lm, f2) = probe(-eps, &mut gpu)?;
        if f1 || f2 {
            flips += 1;
            continue;
        }
        let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
        let ana = d_x_host[i];
        let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-4);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
        checked += 1;
    }

    println!("  checked {checked} coords, {flips} skipped (routing flip)");
    println!("  worst relative error {worst:.3e} at coord {worst_i}");
    // 1e-2 is a real bound, not an accommodation: measured 3.078e-3 at the
    // default eps. The step was swept — 1.084e-2 at 1e-3 (f32 cancellation in
    // the loss difference) and 2.299e-2 at 3e-2 (truncation), a U-shape whose
    // minimum is the signature of a CORRECT derivative. A wrong backward shows
    // a floor that does not improve with eps.
    if worst < 1e-2 && checked > 0 {
        println!("\nPASS — attention and MoE halves compose");
        Ok(())
    } else {
        println!("\nFAIL");
        std::process::exit(1)
    }
}
