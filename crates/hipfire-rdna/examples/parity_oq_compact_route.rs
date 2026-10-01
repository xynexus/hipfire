// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity of the batched-serving routes of `gemm_oq_compact_act_batched` —
//! the wide multicol (<= 24 rows) and the BN=64 wave64 tile (25..=64 rows) —
//! against the default BN=128 tile, which the same rows padded to B=100 take.
//! The narrow tile must match bit for bit (same per-element K order); the
//! multicol sums in a different order, so it gets a rounding tolerance.
//!
//!   cargo run --release -p hipfire-rdna --example parity_oq_compact_route

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x2468_ACE1u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let (group, n_out) = (256usize, 3usize);
    let stride = 2 + group / 2 + 2 * n_out;
    const PAD: usize = 100;
    let mut fail = false;
    for &(m, k) in &[(1536usize, 5120usize), (1024, 17408)] {
        let ng = k / group;
        let nblk = m * ng;
        let mut blocks = vec![0u8; nblk * stride];
        for blk in 0..nblk {
            let off = blk * stride;
            let bits = (((12 + rnd() % 4) as u16) << 10) | (rnd() % 1024) as u16;
            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for i in 0..group / 2 {
                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
            }
            let hdr = 2 + group / 2;
            let mut used = vec![false; group];
            for s in 0..n_out {
                let mut idx = (rnd() % group as u32) as usize;
                while used[idx] {
                    idx = (idx + 1) % group;
                }
                used[idx] = true;
                blocks[off + hdr + 2 * s] = idx as u8;
                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                blocks[off + 2 + idx / 2] &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }
        let side = stride - group / 2;
        let mut dev = vec![0u8; blocks.len()];
        for blk in 0..nblk {
            let src = blk * stride;
            dev[blk * (group / 2)..blk * (group / 2) + group / 2]
                .copy_from_slice(&blocks[src + 2..src + 2 + group / 2]);
            let d = nblk * (group / 2) + blk * side;
            dev[d..d + 2].copy_from_slice(&blocks[src..src + 2]);
            dev[d + 2..d + side].copy_from_slice(&blocks[src + 2 + group / 2..src + stride]);
        }
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let x: Vec<f32> = (0..PAD * k)
            .map(|_| (rnd() % 2000) as f32 * 1e-3 - 1.0)
            .collect();
        let xb = gpu.upload_f32(&x, &[x.len()]).expect("x");
        let yref = gpu.alloc_tensor(&[PAD * m], DType::F32).expect("y");
        gpu.set_oq_batch_serving(true);
        gpu.gemm_oq_compact_act_batched(&wb, &xb, &yref, m, k, PAD, stride)
            .unwrap();
        let reference = gpu.download_f32(&yref).unwrap();
        for &b in &[1usize, 8, 20, 24, 25, 33, 40, 64] {
            let xs = gpu.upload_f32(&x[..b * k], &[b * k]).expect("x");
            let y = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
            gpu.gemm_oq_compact_act_batched(&wb, &xs, &y, m, k, b, stride)
                .unwrap();
            let got = gpu.download_f32(&y).unwrap();
            let (mut max_rel, mut exact) = (0f32, true);
            for i in 0..b * m {
                let (g, r) = (got[i], reference[i]);
                exact &= g.to_bits() == r.to_bits();
                max_rel = max_rel.max((g - r).abs() / r.abs().max(1.0));
            }
            let tiled = b > 24;
            let pass = if tiled { exact } else { max_rel < 1e-3 };
            fail |= !pass;
            println!(
                "M={m} K={k} B={b:>2} {}: max_rel={max_rel:.2e} bit_exact={exact} -> {}",
                if tiled { "n64 tile " } else { "multicol " },
                if pass { "PASS" } else { "FAIL" }
            );
            let _ = gpu.free_tensor(xs);
            let _ = gpu.free_tensor(y);
        }
    }
    if fail {
        std::process::exit(1);
    }
}
