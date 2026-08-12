// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//!
//! Channel parity for native packed DFLASH qt=45/46/47 production and
//! reference GEMMs.
//! Uses ragged K (groups span row boundaries), multi-batch inputs, extreme
//! signed codes, and a duplicate mixed-overlay position whose last entry must
//! win exactly as it does in the host expansion oracle.
//!
//!   cargo run --release -p hipfire-rdna --example parity_dflash_oq_plain

use hipfire_primitives::conv::{f16_to_f32, f32_to_f16};
use hipfire_rdna::{DType, Gpu};

const GROUP: usize = 256;

fn activation(seed: usize) -> f32 {
    ((seed.wrapping_mul(37).wrapping_add(11) % 257) as f32 - 128.0) / 47.0
}

// GPU `_Float16` casts use round-to-nearest-even. The repository's generic
// f32_to_f16 codec intentionally truncates and is correct for reproducing the
// stored scale bytes, but it is not the activation-cast oracle used here.
fn activation_f16(value: f32) -> f32 {
    if value == 0.0 {
        return value;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mut exp16 = exp32 - 127 + 15;
    assert!(
        (1..31).contains(&exp16),
        "fixture activation must be normal f16"
    );
    let frac = bits & 0x7f_ffff;
    let mut mantissa = frac >> 13;
    let remainder = frac & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && mantissa & 1 != 0) {
        mantissa += 1;
        if mantissa == 0x400 {
            mantissa = 0;
            exp16 += 1;
        }
    }
    f16_to_f32(sign | ((exp16 as u16) << 10) | mantissa as u16)
}

fn code4(flat: usize) -> i8 {
    ((flat.wrapping_mul(13).wrapping_add(5) % 15) as i8) - 7
}

fn code8(flat: usize) -> i8 {
    ((flat.wrapping_mul(29).wrapping_add(17) % 255) as i16 - 127) as i8
}

fn scale(block: usize) -> f32 {
    // Round through f16 because that is the exact on-disk contract.
    f16_to_f32(f32_to_f16(0.0025 + (block % 9) as f32 * 0.00375))
}

fn build_blocks(dtype: DType, elements: usize) -> (Vec<u8>, usize) {
    let stride = match dtype {
        DType::DflashOq8Plain => 258,
        DType::DflashOq4Plain => 130,
        DType::DflashOq4MixedPlain => 136, // three overlays = oq4.25+
        _ => unreachable!(),
    };
    let blocks = elements.div_ceil(GROUP);
    let mut bytes = vec![0u8; blocks * stride];
    for block in 0..blocks {
        let off = block * stride;
        bytes[off..off + 2].copy_from_slice(&f32_to_f16(scale(block)).to_le_bytes());
        match dtype {
            DType::DflashOq8Plain => {
                for pos in 0..GROUP {
                    bytes[off + 2 + pos] = code8(block * GROUP + pos) as u8;
                }
            }
            DType::DflashOq4Plain | DType::DflashOq4MixedPlain => {
                for pos in (0..GROUP).step_by(2) {
                    let lo = code4(block * GROUP + pos) as u8 & 0xf;
                    let hi = code4(block * GROUP + pos + 1) as u8 & 0xf;
                    bytes[off + 2 + pos / 2] = lo | (hi << 4);
                }
                if dtype == DType::DflashOq4MixedPlain {
                    // Position 3 occurs twice. The final replacement (17) wins.
                    bytes[off + 130..off + 136].copy_from_slice(&[
                        3,
                        60u8,
                        71,
                        (-40i8) as u8,
                        3,
                        17u8,
                    ]);
                }
            }
            _ => unreachable!(),
        }
    }
    (bytes, stride)
}

fn decode(block: &[u8], stride: usize, dtype: DType, pos: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
        * decode_q(block, stride, dtype, pos) as f32
}

fn decode_q(block: &[u8], stride: usize, dtype: DType, pos: usize) -> i32 {
    match dtype {
        DType::DflashOq8Plain => block[2 + pos] as i8 as i32,
        DType::DflashOq4Plain | DType::DflashOq4MixedPlain => {
            let packed = block[2 + pos / 2];
            let nibble = if pos & 1 == 0 {
                packed & 0xf
            } else {
                packed >> 4
            };
            let mut value = ((nibble as i8) << 4 >> 4) as i32;
            if dtype == DType::DflashOq4MixedPlain {
                for entry in 0..(stride - 130) / 2 {
                    if block[130 + 2 * entry] as usize == pos {
                        value = block[131 + 2 * entry] as i8 as i32;
                    }
                }
            }
            value
        }
        _ => unreachable!(),
    }
}

fn run(gpu: &mut Gpu, dtype: DType, m: usize, k: usize, batch: usize) {
    let (weights, stride) = build_blocks(dtype, m * k);
    let x: Vec<f32> = (0..batch * k).map(activation).collect();
    let mut expected = vec![0.0f32; batch * m];
    for b in 0..batch {
        for row in 0..m {
            // Mirror the kernel's lane-strided accumulation and shuffle tree.
            // A sequential CPU dot can differ materially on cancellation-heavy
            // inputs even when every decoded code is correct.
            let mut lanes = [0.0f32; 32];
            for (lane, lane_sum) in lanes.iter_mut().enumerate() {
                for col in (lane..k).step_by(32) {
                    let flat = row * k + col;
                    let block_index = flat / GROUP;
                    let pos = flat % GROUP;
                    let block = &weights[block_index * stride..(block_index + 1) * stride];
                    let x_f16 = activation_f16(x[b * k + col]);
                    *lane_sum += decode(block, stride, dtype, pos) * x_f16;
                }
            }
            for offset in [16usize, 8, 4, 2, 1] {
                let before = lanes;
                for lane in 0..32 - offset {
                    lanes[lane] = before[lane] + before[lane + offset];
                }
            }
            expected[b * m + row] = lanes[0];
        }
    }

    let wd = gpu.upload_raw(&weights, &[weights.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[batch, k]).unwrap();
    let yd_ref = gpu
        .upload_raw(&vec![0u8; batch * m * 4], &[batch, m])
        .unwrap();
    let yd = gpu
        .upload_raw(&vec![0u8; batch * m * 4], &[batch, m])
        .unwrap();
    gpu.gemm_dflash_oq_plain_ref(dtype, &wd, &xd, &yd_ref, m, k, batch, stride)
        .unwrap();
    gpu.gemm_dflash_oq_plain(dtype, &wd, &xd, &yd, m, k, batch, stride)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let reference = gpu.download_f32(&yd_ref).unwrap();
    let actual = gpu.download_f32(&yd).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    let mut max_ref_delta = 0.0f32;
    for ((&got, &reference), &want) in actual.iter().zip(&reference).zip(&expected) {
        max_abs = max_abs.max((got - want).abs());
        max_ref_delta = max_ref_delta.max((got - reference).abs());
        max_mag = max_mag.max(want.abs());
    }
    let tolerance = 2e-5 * max_mag.max(1.0);
    println!(
        "{dtype:?} M={m} K={k} B={batch} max_abs={max_abs:.6} ref_delta={max_ref_delta:.6} tol={tolerance:.6} -> {}",
        if max_abs <= tolerance && max_ref_delta == 0.0 { "PASS" } else { "FAIL" }
    );
    assert!(max_abs <= tolerance, "{dtype:?} parity failed");
    assert_eq!(max_ref_delta, 0.0, "{dtype:?} production/reference drift");
}

fn run_aligned_dp4a(gpu: &mut Gpu, dtype: DType, m: usize, k: usize, batch: usize) {
    assert_eq!(k % GROUP, 0);
    let (weights, stride) = build_blocks(dtype, m * k);
    let x: Vec<f32> = (0..batch * k).map(activation).collect();
    let groups = k / GROUP;
    let mut expected = vec![0.0f32; batch * m];
    for b in 0..batch {
        let mut xq = vec![0i8; k];
        let mut xs = vec![0.0f32; groups];
        for group in 0..groups {
            let values = &x[b * k + group * GROUP..b * k + (group + 1) * GROUP];
            let max_abs = values
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            xs[group] = scale;
            for (index, value) in values.iter().enumerate() {
                xq[group * GROUP + index] = (value / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        for row in 0..m {
            let mut lanes = [0.0f32; 32];
            for group in 0..groups {
                let block_index = row * groups + group;
                let block = &weights[block_index * stride..(block_index + 1) * stride];
                let weight_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for (lane, lane_sum) in lanes.iter_mut().enumerate() {
                    let base = group * GROUP + lane * 8;
                    let mut dot = 0i32;
                    for offset in 0..8 {
                        dot += decode_q(block, stride, dtype, lane * 8 + offset)
                            * xq[base + offset] as i32;
                    }
                    *lane_sum += dot as f32 * weight_scale * xs[group];
                }
            }
            for offset in [16usize, 8, 4, 2, 1] {
                let before = lanes;
                for lane in 0..32 - offset {
                    lanes[lane] = before[lane] + before[lane + offset];
                }
            }
            expected[b * m + row] = lanes[0];
        }
    }

    let wd = gpu.upload_raw(&weights, &[weights.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[batch, k]).unwrap();
    let yd_ref = gpu
        .upload_raw(&vec![0u8; batch * m * 4], &[batch, m])
        .unwrap();
    let yd = gpu
        .upload_raw(&vec![0u8; batch * m * 4], &[batch, m])
        .unwrap();
    gpu.gemm_dflash_oq_plain_ref(dtype, &wd, &xd, &yd_ref, m, k, batch, stride)
        .unwrap();
    gpu.gemm_dflash_oq_plain(dtype, &wd, &xd, &yd, m, k, batch, stride)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let reference = gpu.download_f32(&yd_ref).unwrap();
    let actual = gpu.download_f32(&yd).unwrap();
    let max_mag = expected
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let max_abs = actual
        .iter()
        .zip(&expected)
        .fold(0.0f32, |max, (got, want)| max.max((got - want).abs()));
    let max_f16_delta = actual
        .iter()
        .zip(&reference)
        .fold(0.0f32, |max, (got, want)| max.max((got - want).abs()));
    let tolerance = 3e-5 * max_mag.max(1.0);
    println!(
        "{dtype:?} A8 M={m} K={k} B={batch} max_abs={max_abs:.6} f16_delta={max_f16_delta:.6} tol={tolerance:.6} -> {}",
        if max_abs <= tolerance { "PASS" } else { "FAIL" }
    );
    assert!(max_abs <= tolerance, "{dtype:?} A8 parity failed");
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    // K=385 intentionally crosses G256 boundaries within a logical row.
    for dtype in [
        DType::DflashOq8Plain,
        DType::DflashOq4Plain,
        DType::DflashOq4MixedPlain,
    ] {
        run(&mut gpu, dtype, 19, 385, 3);
        run_aligned_dp4a(&mut gpu, dtype, 19, 512, 3);
    }
    println!("DFLASH plain Opus packed parity on {}: PASS", gpu.arch);
}
