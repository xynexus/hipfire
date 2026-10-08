// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity of the causal KVarN prefill WMMA kernel (`attention_prefill_kvarn_wmma`)
//! against an f64 host reference (exact dequantized records + window + Q8 V) and
//! the per-row tile+reduce path it replaces (`HIPFIRE_KVARN_PREFILL_WMMA=0`).
//! Many query rows at causal positions: a cold prefill from 0 and an attached
//! tail starting mid-context, across full record blocks and the window tail.
//!
//!   cargo run --release -p hipfire-rdna --example parity_kvarn_prefill_wmma

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
    let mut gpu = Gpu::init().unwrap();
    let mut all_pass = true;
    // (n_full_blocks, tail_len, rows): rows end at the last position.
    for &(nfb, tail, rows) in &[
        (0usize, 70usize, 70usize), // cold, window only
        (2, 37, 293),               // cold prefill of everything
        (3, 0, 64),                 // attached tail ending on a block boundary
        (4, 90, 130),               // attached tail spanning records + window
        (1, 5, 33),                 // short, partial workgroup
        // A continuation's chunk that starts mid-block opens with a short segment
        // attended while its block is still f32: rows end the window.
        (8, 128, 22),   // short head segment, window just filled
        (3, 20, 1),     // single row
        (2, 45, 7),     // a few rows, partial window
        (104, 128, 22), // the 27B's 13.3K-prefix head segment
    ] {
        // Every KVarN K width: the kernel compiles one variant per width.
        for bits in [4usize, 2, 8] {
            all_pass &= run_case(&mut gpu, nfb, tail, rows, bits);
        }
    }
    if !all_pass {
        std::process::exit(1);
    }
    println!("ALL PASS");
}

fn run_case(
    gpu: &mut Gpu,
    n_full_blocks: usize,
    tail_len: usize,
    rows: usize,
    bits: usize,
) -> bool {
    let head_dim = 256usize;
    let n_heads = 6usize;
    let n_kv_heads = 2usize;
    let group = 128usize;
    let kv_dim = n_kv_heads * head_dim;
    let n_full = n_full_blocks * group;
    let seq_len = n_full + tail_len;
    let max_seq = seq_len.next_multiple_of(group).max(1024);
    let blocks_per_head = head_dim / 32;
    let v_row_stride = n_kv_heads * blocks_per_head * 34;
    let tile_elems = head_dim * group;
    let record_bytes = (tile_elems * bits).div_ceil(8) + head_dim * 2 * 2 + group * 2;

    let kbase = lcg(11, n_full.max(1) * kv_dim);
    let mut k = vec![0.0f32; n_full * kv_dim];
    for t in 0..n_full {
        for j in 0..kv_dim {
            let ch_scale = 0.05f32 * 40f32.powf((j % head_dim) as f32 / head_dim as f32);
            k[t * kv_dim + j] = kbase[t * kv_dim + j] * ch_scale;
        }
    }
    let n_blocks_alloc = max_seq.div_ceil(group);
    let rec_buf_bytes = (n_blocks_alloc * n_kv_heads * record_bytes).next_multiple_of(4);
    let rd = gpu
        .upload_raw(&vec![0u8; rec_buf_bytes], &[rec_buf_bytes / 4])
        .unwrap();
    if n_full_blocks > 0 {
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
        gpu.kvarn_quantize_tile(&td, &rd, n_tiles, head_dim, group, record_bytes, bits)
            .unwrap();
    }
    let recs = gpu.download_raw(&rd, rec_buf_bytes).unwrap();

    let mut window = vec![0.0f32; group * kv_dim];
    let wbase = lcg(23, tail_len.max(1) * kv_dim);
    for t in 0..tail_len {
        for j in 0..kv_dim {
            window[t * kv_dim + j] = wbase[t * kv_dim + j] * 0.3;
        }
    }
    let wd = gpu
        .upload_raw(
            &window
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[group * kv_dim],
        )
        .unwrap();

    let vbase = lcg(7, seq_len * kv_dim);
    let mut v_cache = vec![0u8; max_seq * v_row_stride];
    let mut v_deq = vec![0.0f32; seq_len * kv_dim];
    for t in 0..seq_len {
        for kvh in 0..n_kv_heads {
            for b in 0..blocks_per_head {
                let at = |e: usize| vbase[t * kv_dim + kvh * head_dim + b * 32 + e] * 0.3;
                let amax = (0..32).fold(0.0f32, |m, e| m.max(at(e).abs()));
                let scale = (amax / 127.0).max(1e-8);
                let blk = t * v_row_stride + (kvh * blocks_per_head + b) * 34;
                v_cache[blk..blk + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
                for e in 0..32 {
                    let qd = (at(e) / scale).round().clamp(-127.0, 127.0) as i8;
                    v_cache[blk + 2 + e] = qd as u8;
                    v_deq[t * kv_dim + kvh * head_dim + b * 32 + e] = scale * qd as f32;
                }
            }
        }
    }
    let vd = gpu.upload_raw(&v_cache, &[max_seq * v_row_stride]).unwrap();

    let qbytes = (tile_elems * bits).div_ceil(8);
    let (off_scale, off_zp) = (qbytes, qbytes + head_dim * 2);
    let off_scol = off_zp + head_dim * 2;
    let mut k_host = vec![0.0f32; seq_len * kv_dim];
    for b in 0..n_full_blocks {
        for kvh in 0..n_kv_heads {
            let rec = &recs[(b * n_kv_heads + kvh) * record_bytes..];
            let rd16 = |o: usize| f16_to_f32(u16::from_le_bytes([rec[o], rec[o + 1]]));
            for ch in 0..head_dim {
                let (sa, za) = (rd16(off_scale + ch * 2), rd16(off_zp + ch * 2));
                for c in 0..group {
                    let gi = ch * group + c;
                    let per = 8 / bits;
                    let byte = rec[gi / per];
                    let q = ((byte as u32 >> ((gi % per) * bits)) & ((1u32 << bits) - 1)) as f32;
                    k_host[(b * group + c) * kv_dim + kvh * head_dim + ch] =
                        (q * sa + za) * rd16(off_scol + c * 2);
                }
            }
        }
    }
    for t in 0..tail_len {
        for j in 0..kv_dim {
            k_host[(n_full + t) * kv_dim + j] = window[t * kv_dim + j];
        }
    }

    // Query rows at the last `rows` positions.
    let first = seq_len - rows;
    let positions: Vec<i32> = (first..seq_len).map(|p| p as i32).collect();
    let q_dim = n_heads * head_dim;
    let q: Vec<f32> = lcg(3, rows * q_dim).iter().map(|v| v * 0.5).collect();
    let kv_group = n_heads / n_kv_heads;
    let scale_attn = 1.0f64 / (head_dim as f64).sqrt();
    let mut ref_out = vec![0.0f32; rows * q_dim];
    for r in 0..rows {
        let len = positions[r] as usize + 1;
        for h in 0..n_heads {
            let kvh = h / kv_group;
            let qv = &q[r * q_dim + h * head_dim..][..head_dim];
            let sc: Vec<f64> = (0..len)
                .map(|t| {
                    let kt = &k_host[t * kv_dim + kvh * head_dim..][..head_dim];
                    qv.iter()
                        .zip(kt)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * scale_attn
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let sum: f64 = e.iter().sum();
            for d in 0..head_dim {
                let acc: f64 = (0..len)
                    .map(|t| e[t] * v_deq[t * kv_dim + kvh * head_dim + d] as f64)
                    .sum();
                ref_out[r * q_dim + h * head_dim + d] = (acc / sum) as f32;
            }
        }
    }

    let qd = gpu
        .upload_raw(
            &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[rows * q_dim],
        )
        .unwrap();
    let posd = gpu
        .upload_raw(
            &positions
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[rows],
        )
        .unwrap();
    let max_tiles = max_seq.div_ceil(group);
    let partials = gpu
        .zeros(&[rows * n_heads * max_tiles * (2 + head_dim)], DType::F32)
        .unwrap();
    // The WMMA arm calls the kernel directly: routing by row count would send a
    // short segment (< 32 rows) to the tile path, and a prefill chunk's short
    // head segment now takes this kernel whatever its own width.
    let run = |gpu: &mut Gpu, wmma: bool| {
        std::env::set_var("HIPFIRE_KVARN_PREFILL_WMMA", "0");
        let out = gpu.zeros(&[rows * q_dim], DType::F32).unwrap();
        if wmma {
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
                bits,
            )
            .unwrap();
            gpu.device_synchronize().unwrap();
            return gpu.download_f32(&out).unwrap();
        }
        gpu.attention_flash_kvarn_batched_masked(
            &qd,
            &rd,
            &wd,
            &vd,
            &out,
            &posd,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            seq_len,
            rows,
            &partials,
            None,
            0,
            0,
            n_full_blocks,
            record_bytes,
            bits,
        )
        .unwrap();
        gpu.device_synchronize().unwrap();
        gpu.download_f32(&out).unwrap()
    };
    let old = run(gpu, false);
    let new = run(gpu, true);
    let err = |x: &[f32]| {
        x.iter()
            .zip(&ref_out)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
    };
    let (e_old, e_new) = (err(&old), err(&new));
    let ref_max = ref_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let nan = new.iter().any(|v| !v.is_finite());
    // f16 Q/K/V/P in the WMMA path vs f32 in the old one: allow a few f16 ulps of
    // the output scale.
    let pass = !nan && e_new < 4e-3 && e_new <= ref_max * 2e-2;
    println!(
        "  bits={bits} n_full={n_full_blocks} tail={tail_len} rows={rows} seq={seq_len}: old-vs-host={e_old:.2e} wmma-vs-host={e_new:.2e} (|ref|max {ref_max:.3}) -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}
