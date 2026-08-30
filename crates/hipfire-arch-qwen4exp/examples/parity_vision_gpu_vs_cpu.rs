// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vision tower vs the CPU reference, block by block and end to end.
//!
//! The CPU tower is differenced against pinned upstream in
//! `tests/reference_oracle.rs`, so agreement here chains back to the reference.
//!
//! Head dim is 72 (1152 / 16) in the shipped model — NOT a power of two and not
//! one of the specialised widths the text-side attention kernels take. This uses
//! `vit_attention_f32`, which is width-agnostic, so the fixture keeps an awkward
//! head_dim on purpose rather than a convenient one.

use hipfire_arch_qwen4exp::config::VisionConfig;
use hipfire_arch_qwen4exp::vision::{merger, VisionBlock};
use hipfire_arch_qwen4exp::vision_gpu::{
    vision_block, vision_merger, MergerWeights, VisionBlockWeights, VisionScratch,
};
use hipfire_rdna::{DType, Gpu};

fn seeded(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    // head_dim = 144 / 4 = 36: deliberately not a power of two, matching the
    // shipped tower's 72.
    let v = VisionConfig {
        depth: 2,
        hidden: 144,
        n_heads: 4,
        intermediate: 288,
        out_hidden: 192,
        in_channels: 3,
        patch_size: 4,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        num_position_embeddings: 64,
    };
    let (h, nh, hd) = (v.hidden, v.n_heads, v.head_dim());
    let n_tok = 16usize;
    let eps = 1e-6f32;

    let x0 = seeded(n_tok * h, 3);
    let cos: Vec<f32> = seeded(n_tok * hd, 5).iter().map(|t| t.cos()).collect();
    let sin: Vec<f32> = seeded(n_tok * hd, 5).iter().map(|t| t.sin()).collect();

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("parity_vision_gpu_vs_cpu: no GPU ({e}) — skipped");
            return;
        }
    };
    let g_cos = gpu.upload_f32(&cos, &[n_tok, hd]).unwrap();
    let g_sin = gpu.upload_f32(&sin, &[n_tok, hd]).unwrap();
    let g_x = gpu.upload_f32(&x0, &[n_tok, h]).unwrap();
    let mut s = VisionScratch::new(&mut gpu, &v, n_tok).unwrap();

    let mut cpu_x = x0.clone();
    let mut worst_block = 0.0f32;
    for l in 0..v.depth {
        let sd = 100 + l as u32 * 20;
        let (n1w, n1b) = (seeded(h, sd), seeded(h, sd + 1));
        let (n2w, n2b) = (seeded(h, sd + 2), seeded(h, sd + 3));
        let (qw, qb) = (seeded(3 * h * h, sd + 4), seeded(3 * h, sd + 5));
        let (pw, pb) = (seeded(h * h, sd + 6), seeded(h, sd + 7));
        let (f1w, f1b) = (
            seeded(v.intermediate * h, sd + 8),
            seeded(v.intermediate, sd + 9),
        );
        let (f2w, f2b) = (seeded(h * v.intermediate, sd + 10), seeded(h, sd + 11));

        let blk = VisionBlock {
            norm1_w: &n1w,
            norm1_b: &n1b,
            norm2_w: &n2w,
            norm2_b: &n2b,
            qkv_w: &qw,
            qkv_b: &qb,
            proj_w: &pw,
            proj_b: &pb,
            fc1_w: &f1w,
            fc1_b: &f1b,
            fc2_w: &f2w,
            fc2_b: &f2b,
            hidden: h,
            n_heads: nh,
            intermediate: v.intermediate,
            eps,
        };
        cpu_x = blk.forward(&cpu_x, n_tok, &cos, &sin);

        let gw = VisionBlockWeights {
            norm1_w: gpu.upload_f32(&n1w, &[h]).unwrap(),
            norm1_b: gpu.upload_f32(&n1b, &[h]).unwrap(),
            norm2_w: gpu.upload_f32(&n2w, &[h]).unwrap(),
            norm2_b: gpu.upload_f32(&n2b, &[h]).unwrap(),
            qkv_w: gpu.upload_f32(&qw, &[3 * h, h]).unwrap(),
            qkv_b: gpu.upload_f32(&qb, &[3 * h]).unwrap(),
            proj_w: gpu.upload_f32(&pw, &[h, h]).unwrap(),
            proj_b: gpu.upload_f32(&pb, &[h]).unwrap(),
            fc1_w: gpu.upload_f32(&f1w, &[v.intermediate, h]).unwrap(),
            fc1_b: gpu.upload_f32(&f1b, &[v.intermediate]).unwrap(),
            fc2_w: gpu.upload_f32(&f2w, &[h, v.intermediate]).unwrap(),
            fc2_b: gpu.upload_f32(&f2b, &[h]).unwrap(),
        };
        vision_block(&mut gpu, &v, &gw, &mut s, &g_x, n_tok, &g_cos, &g_sin, eps).unwrap();
        let d = maxd(&gpu.download_f32(&g_x).unwrap(), &cpu_x);
        let m = cpu_x.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        worst_block = worst_block.max(d / m.max(1e-6));
    }

    // Merger.
    let wide = h * v.merge_unit();
    let (nw, nb) = (seeded(h, 300), seeded(h, 301));
    let (f1w, f1b) = (seeded(wide * wide, 302), seeded(wide, 303));
    let (f2w, f2b) = (seeded(v.out_hidden * wide, 304), seeded(v.out_hidden, 305));
    let want = merger(
        &cpu_x,
        &nw,
        &nb,
        &f1w,
        &f1b,
        &f2w,
        &f2b,
        h,
        v.merge_unit(),
        v.out_hidden,
        eps,
    );
    let mw = MergerWeights {
        norm_w: gpu.upload_f32(&nw, &[h]).unwrap(),
        norm_b: gpu.upload_f32(&nb, &[h]).unwrap(),
        fc1_w: gpu.upload_f32(&f1w, &[wide, wide]).unwrap(),
        fc1_b: gpu.upload_f32(&f1b, &[wide]).unwrap(),
        fc2_w: gpu.upload_f32(&f2w, &[v.out_hidden, wide]).unwrap(),
        fc2_b: gpu.upload_f32(&f2b, &[v.out_hidden]).unwrap(),
    };
    let merged = n_tok / v.merge_unit();
    let g_out = gpu.zeros(&[merged * v.out_hidden], DType::F32).unwrap();
    vision_merger(&mut gpu, &v, &mw, &g_x, n_tok, &g_out, eps).unwrap();
    let mag = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let worst_merger = maxd(&gpu.download_f32(&g_out).unwrap(), &want) / mag.max(1e-6);

    // RELATIVE, not absolute. These towers accumulate: activations here reach a
    // magnitude of a few hundred, and an absolute bound would have to be re-tuned
    // every time the fixture's scale moved. 5e-6 relative is f32 reduction-order
    // noise over a 288-wide dot product; a real transposition or a wrong GELU is
    // orders of magnitude larger.
    let tol = 5e-6;
    let ok = worst_block <= tol && worst_merger <= tol;
    println!(
        "parity_vision_gpu_vs_cpu: {} blocks (head_dim {hd}), blocks {worst_block:.3e}, \
         merger {worst_merger:.3e} RELATIVE (mag {mag:.2}, tol {tol:.0e}) -> {}",
        v.depth,
        if ok { "OK" } else { "FAILED" }
    );
    if !ok {
        std::process::exit(1);
    }
}
