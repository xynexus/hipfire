// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity oracle for the compact-resident Opus W8A8 GEMM.
//!
//! `gemm_oq_compact_grouped_wmma` reads OqPlusCompact (qt=36) blocks directly so
//! oq4.25++ stays ~4.25 bits/weight resident. This checks it against the path it
//! is meant to replace: expand the SAME blocks with the host oracle
//! (`oqplus_compact_to_oq8_combined`, byte-for-byte the load-time expansion),
//! run `gemm_oq8_grouped_wmma` on the dense result, and compare.
//!
//! The bar is BIT-IDENTICAL, not approximate. Both kernels do the same
//! int8xint8 WMMA in the same order over the same values; the only difference is
//! where the weight bytes and the per-group scale come from. f16->f32 is exact,
//! so any difference at all means a decode bug — an epsilon would hide exactly
//! the nibble-order and overlay-precedence errors this is built to catch.
//!
//! Run: cargo run -p hipfire-rdna --example parity_gemm_oq_compact

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    // Only used for small, exactly-representable scales here.
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mant as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Host expansion, mirroring `hipfire_runtime::oq8_arch::oqplus_compact_to_oq8_combined`
/// exactly: int4 bulk sign-extended to int8, then overlays applied in table
/// order (last duplicate wins), scales split into a trailing f32 plane.
#[allow(non_snake_case)]
fn expand(
    blocks: &[u8],
    m: usize,
    k: usize,
    block_stride: usize,
    group: usize,
) -> (Vec<i8>, Vec<f32>) {
    let GROUP: usize = group;
    let ng = k / GROUP;
    let header = 2 + GROUP / 2;
    let n_out = (block_stride - header) / 2;
    let mut w = vec![0i8; m * k];
    let mut ws = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * block_stride;
            let dst = r * k + g * GROUP;
            for i in 0..GROUP / 2 {
                let byte = blocks[src + 2 + i];
                w[dst + 2 * i] = (((byte & 0xf) as i8) << 4) >> 4;
                w[dst + 2 * i + 1] = ((byte >> 4) as i8) << 4 >> 4;
            }
            let tbl = src + header;
            for s in 0..n_out {
                let idx = blocks[tbl + 2 * s] as usize;
                w[dst + idx] = blocks[tbl + 2 * s + 1] as i8;
            }
            ws[r * ng + g] = f16_bits_to_f32(u16::from_le_bytes([blocks[src], blocks[src + 1]]));
        }
    }
    (w, ws)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("GPU: {}", gpu.arch);

    let mut fail = 0usize;
    // Shapes chosen to exercise ragged M/B (not multiples of 16) and several
    // overlay counts: 3 = oq4.25++, 7 = oq4.5++, 1 = the format minimum, 16 =
    // well above either. N_out = 0 is NOT tested: the on-disk contract requires
    // block_bytes >= 132, so a zero-overlay block is not a valid OqPlusCompact
    // block and the host oracle rejects it too.
    // The last four are real Qwen3.5-0.8B--oq4.25 projection shapes at B=1. B=1
    // is the decode case and was NOT covered before — every earlier shape used
    // B>=5, so a bug that only shows with a single active batch row (lane
    // clamping via safe_x, the n=1 tail) could not have been caught here.
    // G=128 exists to fit K values that 256 cannot divide; both must decode.
    for &group in &[256usize, 128] {
        for &(m, k, b) in &[
            (16usize, 256usize, 16usize),
            (48, 512, 8),
            (33, 768, 5),
            (2048, 1024, 1),
            (1024, 3584, 1),
            (512, 1024, 1),
            (512, 1024, 2),
            // Qwen3.8-27B down_proj: K = the FFN intermediate, 17408. Every
            // shape above tops out at K=3584, so the whole large-K regime was
            // untested — and down_proj is the ONE class that serves garbage
            // under HIPFIRE_OQ_COMPACT_RESIDENT while gate/up, qkv and lm_head
            // (all K=5120) are fine. M is trimmed; K is the axis under test.
            (512, 17408, 1),
            (512, 17408, 9),
            // Same M, ordinary K — the control that separates "large K" from
            // "this M".
            (512, 5120, 1),
        ] {
            for &n_out in &[1usize, 3, 7, 16] {
                // Power-of-two scales are exact in f16 AND exact under f32 multiply, so
                // they hide any difference in rounding or multiply order between the two
                // paths. A real artifact's scales are arbitrary f16, so sweep both:
                // `exact` reproduces the original coverage, `!exact` is the realistic case.
                for &exact in &[true, false] {
                    let block_stride = 2 + group / 2 + 2 * n_out;
                    let ng = k / group;
                    let mut rnd =
                        lcg(0x51ee_d00d ^ (m * k * b + n_out) as u32 ^ (exact as u32) << 20);

                    // Build compact blocks.
                    let mut blocks = vec![0u8; m * ng * block_stride];
                    for r in 0..m {
                        for g in 0..ng {
                            let off = (r * ng + g) * block_stride;
                            // Arbitrary case: build the f16 bit pattern directly (random
                            // 10-bit mantissa, exponent ~2^-5..2^3) so no f32->f16 rounding
                            // is needed and the stored scale is exactly what both paths see.
                            let bits = if exact {
                                f32_to_f16_bits(1.0f32 / (1 << (1 + (rnd() % 4))) as f32)
                            } else {
                                (((10 + rnd() % 9) as u16) << 10) | (rnd() % 1024) as u16
                            };
                            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
                            for i in 0..group / 2 {
                                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
                            }
                            for s in 0..n_out {
                                // Deliberately allow duplicate indices so last-wins is exercised.
                                let hdr = 2 + group / 2;
                                blocks[off + hdr + 2 * s] = (rnd() % group as u32) as u8;
                                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                            }
                        }
                    }

                    // Activations: int8 + per-group f32 scales, shared by both paths.
                    let xq: Vec<i8> = (0..b * k)
                        .map(|_| ((rnd() % 255) as i32 - 127) as i8)
                        .collect();
                    let xs: Vec<f32> = (0..b * ng)
                        .map(|_| {
                            if exact {
                                1.0f32 / (1 << (1 + (rnd() % 3))) as f32
                            } else {
                                // Arbitrary positive f32, same order of magnitude.
                                0.25f32 + (rnd() % 100_000) as f32 * 1.0e-5
                            }
                        })
                        .collect();

                    let (w_dense, w_scales) = expand(&blocks, m, k, block_stride, group);

                    let d_blocks = gpu
                        .upload_raw(&blocks, &[blocks.len()])
                        .expect("upload blocks");
                    let d_wdense = gpu
                        .upload_raw(
                            unsafe {
                                std::slice::from_raw_parts(
                                    w_dense.as_ptr() as *const u8,
                                    w_dense.len(),
                                )
                            },
                            &[w_dense.len()],
                        )
                        .expect("upload dense");
                    let d_wscales = gpu
                        .upload_f32(&w_scales, &[w_scales.len()])
                        .expect("upload wscales");
                    let d_xq = gpu
                        .upload_raw(
                            unsafe {
                                std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len())
                            },
                            &[xq.len()],
                        )
                        .expect("upload xq");
                    let d_xs = gpu.upload_f32(&xs, &[xs.len()]).expect("upload xs");
                    let d_y_ref = gpu.zeros(&[b * m], DType::F32).expect("y ref");
                    let d_y_cmp = gpu.zeros(&[b * m], DType::F32).expect("y cmp");

                    gpu.gemm_oq8_grouped_wmma(
                        &d_wdense, &d_wscales, &d_xq, &d_xs, &d_y_ref, m, k, b, group,
                    )
                    .expect("dense gemm");
                    gpu.gemm_oq_compact_grouped_wmma(
                        &d_blocks,
                        &d_xq,
                        &d_xs,
                        &d_y_cmp,
                        m,
                        k,
                        b,
                        group,
                        block_stride,
                    )
                    .expect("compact gemm");

                    let y_ref = gpu.download_f32(&d_y_ref).expect("dl ref");
                    let y_cmp = gpu.download_f32(&d_y_cmp).expect("dl cmp");

                    let mut bad = 0usize;
                    let mut worst = 0.0f32;
                    for (a, c) in y_ref.iter().zip(y_cmp.iter()) {
                        if a.to_bits() != c.to_bits() {
                            bad += 1;
                            worst = worst.max((a - c).abs());
                        }
                    }
                    let scales = if exact {
                        "pow2-scales"
                    } else {
                        "arbitrary-scales"
                    };
                    let tag = format!("G={group} M={m} K={k} B={b} N_out={n_out} {scales}");
                    if bad == 0 {
                        println!("  ok   {tag}: bit-identical over {} outputs", y_ref.len());
                    } else {
                        fail += 1;
                        println!(
                            "  FAIL {tag}: {bad}/{} outputs differ, worst |delta| {worst:.6e}",
                            y_ref.len()
                        );
                    }
                }
            }
        }
    }

    if fail == 0 {
        println!("parity_gemm_oq_compact: PASS");
    } else {
        println!("parity_gemm_oq_compact: FAIL ({fail} case(s))");
        std::process::exit(1);
    }
}
