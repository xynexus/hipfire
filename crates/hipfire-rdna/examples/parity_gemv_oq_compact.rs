// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the compact-resident DECODE GEMV, which had none.
//!
//! `parity_gemm_oq_compact` covers `gemm_oq_compact_grouped_wmma` — the batched
//! kernel prefill and spec-verify use. Decode goes somewhere else entirely:
//! `GemvOqCompactG256Prerotated` -> `gemv_oq_compact_grouped_auto` -> the v2
//! GEMV. Nothing checked that path, and with `HIPFIRE_OQ_COMPACT_RESIDENT=1` on
//! Qwen3.8-27B it emits one token and stops — bisected to `down_proj` alone
//! (`[5120, 17408]`), while `gate_proj` `[17408, 5120]` and `lm_head`
//! `[248320, 5120]` are both fine. Same block count, same 136 B stride; the only
//! difference is which of M and K is large.
//!
//! So the shapes here are the real Qwen3.8-27B projection classes, and the point
//! is the K sweep: if `[5120, 17408]` fails while `[17408, 5120]` passes, the
//! fault is in the kernel's handling of K, not in residency or dispatch.
//!
//! Reference is a straight f32 dot over the EXPANDED weights — the same
//! expansion `oq8_arch_load` does when compact residency is off, so agreement
//! here is exactly the invariant the flag is supposed to preserve.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_oq_compact

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn f16_bits_to_f32(b: u16) -> f32 {
    let sign = ((b >> 15) & 1) as u32;
    let exp = ((b >> 10) & 0x1f) as u32;
    let mant = (b & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127 - 15) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Expand compact blocks to dense int8 + per-group f32 scales — the same thing
/// `oq8_arch_load` does when compact residency is off.
fn expand(
    blocks: &[u8],
    m: usize,
    k: usize,
    block_stride: usize,
    group: usize,
) -> (Vec<i8>, Vec<f32>) {
    let ng = k / group;
    let header = 2 + group / 2;
    let n_out = (block_stride - header) / 2;
    let mut w = vec![0i8; m * k];
    let mut ws = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * block_stride;
            let dst = r * k + g * group;
            for i in 0..group / 2 {
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
    const GROUP: usize = 256;
    const N_OUT: usize = 3; // the real Qwen3.8-27B oq4.25++ block: 130 + 2*3 = 136 B
    let block_stride = 2 + GROUP / 2 + 2 * N_OUT;

    // Real Qwen3.8-27B OqPlusCompact projection classes. M is trimmed where the
    // full size would only slow the CPU reference down; K is NOT, because K is
    // the axis under suspicion.
    let shapes: &[(&str, usize, usize)] = &[
        ("gate/up   [17408, 5120]", 2048, 5120),
        ("qkv       [10240, 5120]", 2048, 5120),
        ("lm_head  [248320, 5120]", 2048, 5120),
        ("attn_out  [5120,  6144]", 2048, 6144),
        ("down      [5120, 17408]", 2048, 17408),
        ("down x2   [5120, 34816]", 512, 34816),
    ];

    println!(
        "compact DECODE GEMV parity (group={GROUP}, block_stride={block_stride}, N_out={N_OUT})\n"
    );
    let mut fail = 0usize;
    for &(name, m, k) in shapes {
        let ng = k / GROUP;
        let mut rnd = lcg(0x00c0_de01_u32 ^ (k as u32));
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
                    let hdr = 2 + GROUP / 2;
                    blocks[off + hdr + 2 * s] = (rnd() % GROUP as u32) as u8;
                    blocks[off + hdr + 2 * s + 1] = (rnd() & 0xff) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..k).map(|_| (rnd() % 2000) as f32 * 1e-3 - 1.0).collect();
        let (w, ws) = expand(&blocks, m, k, block_stride, GROUP);

        // f32 reference over the expanded weights.
        let mut want = vec![0f32; m];
        for r in 0..m {
            let mut acc = 0f32;
            for g in 0..ng {
                let s = ws[r * ng + g];
                let mut gs = 0f32;
                for i in 0..GROUP {
                    gs += w[r * k + g * GROUP + i] as f32 * x[g * GROUP + i];
                }
                acc += gs * s;
            }
            want[r] = acc;
        }

        let d_blocks = gpu.upload_raw(&blocks, &[blocks.len()]).expect("blocks");
        let d_x = gpu.upload_f32(&x, &[k]).expect("x");
        let d_y = gpu.alloc_tensor(&[m], DType::F32).expect("y");
        gpu.gemv_oq_compact_grouped_auto(&d_blocks, &d_x, &d_y, m, k, GROUP, block_stride)
            .expect("gemv");
        let got = gpu.download_f32(&d_y).expect("dl");

        let (mut worst, mut at) = (0f32, 0usize);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        for r in 0..m {
            let d = (got[r] - want[r]).abs();
            if d > worst {
                worst = d;
                at = r;
            }
        }
        let rel = worst / scale;
        let ok = rel < 2e-3;
        if !ok {
            fail += 1;
        }
        println!(
            "  {name:<26} ng={ng:<4} max|Δ|={worst:9.3e} rel={rel:8.2e} (|ref|max={scale:.3e}, row {at})  {}",
            if ok { "PASS" } else { "**FAIL**" }
        );
        for t in [d_blocks, d_x, d_y] {
            let _ = gpu.free_tensor(t);
        }
    }
    if fail == 0 {
        println!("\nparity_gemv_oq_compact: PASS");
    } else {
        println!("\nparity_gemv_oq_compact: FAIL ({fail} shape(s))");
        std::process::exit(1);
    }
}
