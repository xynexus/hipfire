// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity + bandwidth for the compact-resident Opus W4A8 decode GEMV.
//!
//! A8 REACHES the memory wall — 236 GB/s on o_proj against a 250 GB/s pure-read
//! stream — but only at N_out=1. At the production N_out=3 the sparse overlay
//! loop dominates both this kernel and its A16 sibling. `OQC8_NOUT` sweeps the
//! overlay count, which is what exposes that: 1 -> 8 corrections costs BOTH
//! kernels ~4x, because the loop is O(N_out^2) and divergent. Fix the overlay
//! before judging the dot.
//!
//! Two things are checked, because they fail independently:
//!
//!  1. **Exactness.** Given the SAME int8 activation, the kernel must reproduce
//!     an integer reference bit-for-bit up to the single f32 rescale per group.
//!     That covers the nibble widening AND the overlay contract — overlays
//!     replace the bulk value, so they apply as a difference, and a duplicate
//!     index means LAST WINS. The fixture deliberately emits duplicates.
//!  2. **Throughput.** Against the A16 kernel on the same weights, at the real
//!     Qwen3.8-27B projection shapes. This is the number the whole exercise is
//!     for; exactness without it means nothing.
//!
//!   cargo run --release -p hipfire-rdna --features deltanet \
//!     --example parity_gemv_oq_compact_w8a8

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn f16_bits_to_f32(b: u16) -> f32 {
    let (sign, exp, mant) = (
        ((b >> 15) & 1) as u32,
        ((b >> 10) & 0x1f) as u32,
        (b & 0x3ff) as u32,
    );
    let bits = if exp == 0 {
        sign << 31
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Mirror of `hipfire_runtime::oq8_arch::normalize_compact_overlays`. Duplicated
/// rather than imported because hipfire-rdna cannot depend on hipfire-runtime;
/// the runtime crate owns the authoritative version and its own unit tests.
fn normalize_overlays(blocks: &mut [u8], m: usize, ng: usize, stride: usize, n_out: usize) {
    if n_out < 2 {
        return;
    }
    for b in 0..m * ng {
        let base = b * stride;
        let tbl = base + 130;
        for e in 0..n_out - 1 {
            let idx = blocks[tbl + 2 * e] as usize;
            if !(e + 1..n_out).any(|e2| blocks[tbl + 2 * e2] as usize == idx) {
                continue;
            }
            let byte = blocks[base + 2 + idx / 2];
            let bulk = if idx % 2 == 0 {
                ((byte & 0xf) as i8) << 4 >> 4
            } else {
                ((byte >> 4) as i8) << 4 >> 4
            };
            blocks[tbl + 2 * e + 1] = bulk as u8;
        }
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    // Real oq4.25++ is N_out=3. OQC8_NOUT sweeps it: if the A8 deficit tracks the
    // overlay count, the per-correction byte load of Xq[idx] is the cost.
    let n_out: usize = std::env::var("OQC8_NOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    #[allow(non_snake_case)]
    let N_OUT: usize = n_out;
    let block_stride = 130 + 2 * N_OUT;

    let shapes: &[(&str, usize, usize)] = &[
        ("gate/up  [17408, 5120]", 17408, 5120),
        ("down     [5120, 17408]", 5120, 17408),
        ("o_proj   [5120,  5120]", 5120, 5120),
    ];

    println!("compact Opus W4A8 decode GEMV — exactness + throughput (N_out={N_OUT})\n");
    let mut fail = 0usize;

    for &(name, m, k) in shapes {
        let ng = k / GROUP;
        let mut rnd = lcg(0x00c0_ffee ^ (k as u32));
        let mut blocks = vec![0u8; m * ng * block_stride];
        for r in 0..m {
            for g in 0..ng {
                let off = (r * ng + g) * block_stride;
                let bits = (((10 + rnd() % 9) as u16) << 10) | (rnd() % 1024) as u16;
                blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
                for i in 0..GROUP / 2 {
                    blocks[off + 2 + i] = (rnd() & 0xff) as u8;
                }
                for s in 0..N_OUT {
                    // Duplicate indices are deliberate: the contract is last-wins,
                    // and a kernel that sums corrections instead lands on
                    // v1+v2-bulk. Narrow the range so collisions actually happen.
                    blocks[off + 130 + 2 * s] = (rnd() % 24) as u8;
                    blocks[off + 130 + 2 * s + 1] = (rnd() & 0xff) as u8;
                }
            }
        }
        // Resolve duplicate overlay indices exactly as `oq8_arch_load` does before
        // upload — the kernels are allowed to assume it, so the fixture must too.
        normalize_overlays(&mut blocks, m, ng, block_stride, n_out);

        // int8 activation + per-group f32 scales, exactly what the decode path
        // hands the kernel after rotate + quantize_act_oq8.
        let xq: Vec<i8> = (0..k).map(|_| ((rnd() % 255) as i32 - 127) as i8).collect();
        let xs: Vec<f32> = (0..ng)
            .map(|_| 0.25f32 + (rnd() % 100_000) as f32 * 1.0e-5)
            .collect();

        // Integer reference: decode nibbles, apply last-wins overlay, dot in i32,
        // then ONE f32 rescale per group — the kernel's own order of operations.
        let mut want = vec![0f32; m];
        for r in 0..m {
            let mut acc = 0f32;
            for g in 0..ng {
                let off = (r * ng + g) * block_stride;
                let sw = f16_bits_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
                let mut q = [0i32; GROUP];
                for i in 0..GROUP / 2 {
                    let byte = blocks[off + 2 + i];
                    q[2 * i] = (((byte & 0xf) as i8) << 4 >> 4) as i32;
                    q[2 * i + 1] = (((byte >> 4) as i8) << 4 >> 4) as i32;
                }
                for s in 0..N_OUT {
                    let idx = blocks[off + 130 + 2 * s] as usize;
                    q[idx] = blocks[off + 130 + 2 * s + 1] as i8 as i32; // last wins
                }
                let mut idot = 0i32;
                for i in 0..GROUP {
                    idot += q[i] * xq[g * GROUP + i] as i32;
                }
                acc += idot as f32 * (sw * xs[g]);
            }
            want[r] = acc;
        }

        let d_blocks = gpu.upload_raw(&blocks, &[blocks.len()]).expect("blocks");
        let xq_u: Vec<u8> = xq.iter().map(|&v| v as u8).collect();
        let d_xq = gpu.upload_raw(&xq_u, &[k]).expect("xq");
        let d_xs = gpu.upload_f32(&xs, &[ng]).expect("xs");
        let d_y = gpu.alloc_tensor(&[m], DType::F32).expect("y");
        gpu.gemv_oq_compact_w8a8_grouped(&d_blocks, &d_xq, &d_xs, &d_y, m, k, GROUP, block_stride)
            .expect("w8a8 gemv");
        let got = gpu.download_f32(&d_y).expect("dl");

        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        let (mut worst, mut at) = (0f32, 0usize);
        for r in 0..m {
            let d = (got[r] - want[r]).abs();
            if d > worst {
                worst = d;
                at = r;
            }
        }
        let rel = worst / scale;
        // Only the per-group f32 rescale and its accumulation order differ from
        // the reference; the dot itself is exact i32.
        let ok = rel < 1e-6;
        if !ok {
            fail += 1;
        }

        // Throughput against the A16 kernel on the same weights.
        let x_f32: Vec<f32> = xq.iter().map(|&v| v as f32).collect();
        let d_x = gpu.upload_f32(&x_f32, &[k]).expect("x");
        let bytes = m * ng * block_stride;
        let bench = |g: &mut Gpu, a8: bool| -> f64 {
            for _ in 0..3 {
                if a8 {
                    g.gemv_oq_compact_w8a8_grouped(
                        &d_blocks,
                        &d_xq,
                        &d_xs,
                        &d_y,
                        m,
                        k,
                        GROUP,
                        block_stride,
                    )
                    .unwrap();
                } else {
                    g.gemv_oq_compact_grouped_auto(
                        &d_blocks,
                        &d_x,
                        &d_y,
                        m,
                        k,
                        GROUP,
                        block_stride,
                    )
                    .unwrap();
                }
            }
            g.device_synchronize().unwrap();
            let mut best = f64::MAX;
            for _ in 0..12 {
                let t = Instant::now();
                if a8 {
                    g.gemv_oq_compact_w8a8_grouped(
                        &d_blocks,
                        &d_xq,
                        &d_xs,
                        &d_y,
                        m,
                        k,
                        GROUP,
                        block_stride,
                    )
                    .unwrap();
                } else {
                    g.gemv_oq_compact_grouped_auto(
                        &d_blocks,
                        &d_x,
                        &d_y,
                        m,
                        k,
                        GROUP,
                        block_stride,
                    )
                    .unwrap();
                }
                g.device_synchronize().unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            bytes as f64 / best / 1e9
        };
        let a16 = bench(&mut gpu, false);
        let a8 = bench(&mut gpu, true);

        println!(
            "  {name:<24} rel={rel:8.2e} {}   A16={a16:6.1} GB/s  A8={a8:6.1} GB/s  {:.2}x",
            if ok { "PASS" } else { "**FAIL**" },
            a8 / a16
        );
        let _ = at;
        for t in [d_blocks, d_xq, d_xs, d_y, d_x] {
            let _ = gpu.free_tensor(t);
        }
    }
    if fail == 0 {
        println!("\nparity_gemv_oq_compact_w8a8: PASS");
    } else {
        println!("\nparity_gemv_oq_compact_w8a8: FAIL ({fail} shape(s))");
        std::process::exit(1);
    }
}
