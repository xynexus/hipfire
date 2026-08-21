// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemm_oq_compact_iu4_wmma` against a CPU oracle.
//!
//! CONTRACT UNDER TEST: the kernel computes the BULK term only —
//! `Y[b,m] = Σ_g sw[m,g]·sx[b,g]·Σ_{p∈g} nibble[m,p]·x4[b,p]` — with the
//! sparse int8 overlay deliberately NOT applied. That is well defined rather
//! than approximate, because the loader zeroes the bulk nibble under every
//! overlay entry, so those positions contribute exactly 0. The oracle mirrors
//! that, and the separate correction pass is tested on its own.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_oq_compact_iu4

use hipfire_rdna::{DType, Gpu};

const GROUP: usize = 256;

fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let m = (bits & 0x3ff) as u32;
    let out = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while (m2 & 0x400) == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            m2 &= 0x3ff;
            (s << 31) | (e2 << 23) | (m2 << 13)
        }
    } else if e == 0x1f {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(out)
}

#[inline]
fn sext4(v: u8) -> i32 {
    (((v & 0xf) as i32) << 28) >> 28
}

/// Split planes: all nibble groups first, then all [f16 scale][overlay table].
fn split_planes(blocks: &[u8], nblk: usize, stride: usize) -> Vec<u8> {
    let nib = GROUP / 2;
    let side = stride - nib;
    let mut out = vec![0u8; nblk * stride];
    for b in 0..nblk {
        let src = b * stride;
        out[b * nib..b * nib + nib].copy_from_slice(&blocks[src + 2..src + 2 + nib]);
        let d = nblk * nib + b * side;
        out[d..d + 2].copy_from_slice(&blocks[src..src + 2]);
        out[d + 2..d + side].copy_from_slice(&blocks[src + 2 + nib..src + stride]);
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x5EED_1234u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u32
    };
    let mut fail = 0usize;
    println!("gemm_oq_compact_iu4_wmma vs CPU oracle (BULK term, overlay excluded)\n");
    println!("      M      K    B  N_out    max|rel|   verdict");

    for &(m, k, b, n_out) in &[
        (16usize, 256usize, 16usize, 1usize),
        (32, 512, 16, 3),
        (256, 1024, 32, 3),
        (512, 5120, 128, 3),
        (272, 2560, 256, 7),
    ] {
        let ng = k / GROUP;
        let stride = 2 + GROUP / 2 + 2 * n_out;
        let nblk = m * ng;
        let mut blocks = vec![0u8; nblk * stride];
        for blk in 0..nblk {
            let off = blk * stride;
            let bits = (((14 + rnd() % 3) as u16) << 10) | (rnd() % 1024) as u16;
            blocks[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for i in 0..GROUP / 2 {
                blocks[off + 2 + i] = (rnd() & 0xff) as u8;
            }
            let hdr = 2 + GROUP / 2;
            let mut used = [false; GROUP];
            for s in 0..n_out {
                let mut idx = (rnd() % GROUP as u32) as usize;
                while used[idx] {
                    idx = (idx + 1) % GROUP;
                }
                used[idx] = true;
                blocks[off + hdr + 2 * s] = idx as u8;
                blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                // The loader ZEROES the bulk nibble under every overlay.
                let nb = &mut blocks[off + 2 + idx / 2];
                *nb &= if idx % 2 == 0 { 0xf0 } else { 0x0f };
            }
        }

        // Packed signed int4 activations + per-(b, group) scales.
        let x4: Vec<u8> = (0..b * k / 2).map(|_| (rnd() & 0xff) as u8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| (rnd() % 1000) as f32 * 1e-5 + 1e-4)
            .collect();

        let mut want = vec![0f32; b * m];
        for row in 0..m {
            for g in 0..ng {
                let off = (row * ng + g) * stride;
                let sw = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
                for bb in 0..b {
                    let mut acc = 0i32;
                    for i in 0..GROUP {
                        let p = blocks[off + 2 + i / 2];
                        let qw = if i % 2 == 0 {
                            sext4(p & 0xf)
                        } else {
                            sext4(p >> 4)
                        };
                        let xp = x4[(bb * k + g * GROUP + i) / 2];
                        let qx = if i % 2 == 0 {
                            sext4(xp & 0xf)
                        } else {
                            sext4(xp >> 4)
                        };
                        acc += qw * qx;
                    }
                    want[bb * m + row] += acc as f32 * sw * xs[bb * ng + g];
                }
            }
        }

        let dev = split_planes(&blocks, nblk, stride);
        let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
        let xb = gpu.upload_raw(&x4, &[x4.len()]).expect("x4");
        let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).expect("y");
        gpu.gemm_oq_compact_iu4_wmma(&wb, &xb, &xsb, &yb, m, k, b, GROUP, stride)
            .expect("launch");
        let got = gpu.download_f32(&yb).expect("dl");

        // Scale-relative, not per-element-relative: with random signed data many
        // outputs land near zero by cancellation, and dividing by those reports a
        // huge "relative" error for a tiny absolute one. This is the metric
        // parity_gemv_oq_compact uses.
        let mut max_abs = 0f32;
        let mut max_ref = 0f32;
        for i in 0..b * m {
            max_abs = max_abs.max((got[i] - want[i]).abs());
            max_ref = max_ref.max(want[i].abs());
        }
        let worst = max_abs / max_ref.max(1e-30);
        let ok = worst < 1e-5;
        if !ok {
            fail += 1;
        }
        println!(
            "  {m:>5} {k:>6} {b:>4} {n_out:>6}    {worst:>8.2e}   {}   (max|d|={max_abs:.3e} |ref|max={max_ref:.3e})",
            if ok { "PASS" } else { "FAIL" }
        );
        let _ = gpu.free_tensor(wb);
        let _ = gpu.free_tensor(xb);
        let _ = gpu.free_tensor(xsb);
        let _ = gpu.free_tensor(yb);
    }
    if fail == 0 {
        println!("\nparity_gemm_oq_compact_iu4: PASS");
    } else {
        println!("\nparity_gemm_oq_compact_iu4: FAIL ({fail} shape(s))");
        std::process::exit(1);
    }
}
