// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Time `attention_prefill_kvarn_wmma` alone at Qwen3.8-27B shapes (24 query
//! heads, 4 KV heads, head_dim 256): `rows` query rows ending at position `ctx`,
//! K in real KVarN records (quantized on the GPU, as the parity harness does) plus
//! the window tail, V Q8_0. Reports ms per call and causal TFLOPS.
//!
//!   cargo run --release -p hipfire-rdna --example bench_kvarn_prefill_wmma [ctx rows]
use hipfire_rdna::{DType, Gpu};

fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) & 1;
    let e = (bits >> 10) & 0x1f;
    let m = bits & 0x3ff;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = (mant >> shift) as u16;
        if (mant >> (shift - 1)) & 1 == 1 {
            let sticky = mant & ((1 << (shift - 1)) - 1);
            if sticky != 0 || (h & 1) == 1 {
                h += 1;
            }
        }
        return sign | h;
    }
    let mut h_mant = (mant >> 13) as u16;
    if (mant >> 12) & 1 == 1 {
        let sticky = mant & 0xfff;
        if sticky != 0 || (h_mant & 1) == 1 {
            h_mant += 1;
            if h_mant == 0x400 {
                h_mant = 0;
                exp += 1;
            }
        }
    }
    sign | ((exp as u16) << 10) | h_mant
}

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    (0..n)
        .map(|_| {
            let u1 = u().max(1e-7);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (ctx, rows) = (p(1, 16384), p(2, 2048));
    let (head_dim, n_heads, n_kv_heads, group) = (256usize, 24usize, 4usize, 128usize);
    let mut gpu = Gpu::init().unwrap();
    // BENCH_ATTN_SRC=<file.hip>: time a variant of the kernel, registered under its
    // name before the dispatcher's first call (which then finds it loaded).
    if let Ok(path) = std::env::var("BENCH_ATTN_SRC") {
        let src = std::fs::read_to_string(&path).unwrap();
        let module = format!("attn_variant_{:x}", src.len() * 2654435761 % 0xffff_ffff);
        gpu.ensure_kernel_public(&module, &src, "attention_prefill_kvarn_wmma")
            .unwrap();
        eprintln!("variant: {path}");
    }
    let kv_dim = n_kv_heads * head_dim;
    let q_dim = n_heads * head_dim;
    let n_full_blocks = ctx / group;
    let n_full = n_full_blocks * group;
    let tail = ctx - n_full;
    let tile_elems = head_dim * group;
    let record_bytes = tile_elems.div_ceil(2) + head_dim * 2 * 2 + group * 2;
    let rec_buf_bytes = (n_full_blocks.max(1) * n_kv_heads * record_bytes).next_multiple_of(4);
    let rd = gpu
        .upload_raw(&vec![0u8; rec_buf_bytes], &[rec_buf_bytes / 4])
        .unwrap();
    if n_full_blocks > 0 {
        let k = lcg(11, n_full * kv_dim);
        let kd = gpu
            .upload_raw(
                &k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                &[n_full * kv_dim],
            )
            .unwrap();
        let n_tiles = n_full_blocks * n_kv_heads;
        let td = gpu
            .upload_raw(
                &vec![0u8; n_tiles * tile_elems * 4],
                &[n_tiles * tile_elems],
            )
            .unwrap();
        gpu.kvarn_gather_k_tiles(&kd, &td, n_full_blocks, n_kv_heads, head_dim, group)
            .unwrap();
        gpu.kvarn_quantize_tile(&td, &rd, n_tiles, head_dim, group, record_bytes, 4)
            .unwrap();
        let _ = gpu.free_tensor(kd);
        let _ = gpu.free_tensor(td);
    }
    let w = lcg(23, group * kv_dim);
    let wd = gpu
        .upload_raw(
            &w.iter()
                .map(|v| v * 0.3)
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[group * kv_dim],
        )
        .unwrap();
    let bph = head_dim / 32;
    let v_row = n_kv_heads * bph * 34;
    let mut v_cache = vec![0u8; ctx * v_row];
    let mut s = 7u32;
    for (i, b) in v_cache.iter_mut().enumerate() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *b = if i % 34 < 2 {
            [0x00u8, 0x20][i % 34]
        } else {
            (s >> 16) as u8
        };
    }
    let vd = gpu.upload_raw(&v_cache, &[ctx * v_row]).unwrap();
    let q = lcg(5, rows * q_dim);
    let qd = gpu
        .upload_raw(
            &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[rows * q_dim],
        )
        .unwrap();
    let positions: Vec<i32> = (0..rows).map(|r| (ctx - rows + r) as i32).collect();
    let posd = gpu
        .upload_raw(
            &positions
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[rows],
        )
        .unwrap();
    let out = gpu.zeros(&[rows * q_dim], DType::F32).unwrap();
    let call = |gpu: &mut Gpu| {
        gpu.attention_prefill_kvarn_wmma(
            &qd,
            &rd,
            &wd,
            &vd,
            &out,
            &posd,
            n_heads,
            n_kv_heads,
            rows,
            n_full_blocks,
            record_bytes,
            4,
        )
        .unwrap()
    };
    call(&mut gpu);
    call(&mut gpu);
    gpu.device_synchronize().unwrap();
    let iters = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        call(&mut gpu);
    }
    gpu.device_synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let pairs: f64 = positions.iter().map(|&p| p as f64 + 1.0).sum();
    let flops = 4.0 * head_dim as f64 * n_heads as f64 * pairs;
    let o = gpu.download_f32(&out).unwrap();
    let finite = o.iter().all(|v| v.is_finite());
    let checksum: f64 = o.iter().step_by(97).map(|v| *v as f64).sum();
    println!(
        "ctx {ctx} rows {rows}: {ms:.2} ms  {:.2} TFLOPS  (finite {finite}, checksum {checksum:.6})",
        flops / (ms * 1e-3) / 1e12
    );
}
