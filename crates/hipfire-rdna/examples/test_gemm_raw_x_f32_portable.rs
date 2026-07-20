// SPDX-License-Identifier: Apache-2.0
//! Channel test for the family-neutral raw F16/BF16 x F32 fallback GEMM.
//!
//! This deliberately calls the portable backend even on WMMA-capable hosts so
//! the scalar correctness path can be checked before running it on RDNA2/CDNA.

use hipfire_rdna::{DType, Gpu};

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        if unbiased < -24 {
            return sign;
        }
        let normalized = mantissa | 0x80_0000;
        let shift = (-unbiased - 1) as u32;
        let mut rounded = (normalized >> shift) as u16;
        let remainder = normalized & ((1u32 << shift) - 1);
        let midpoint = 1u32 << (shift - 1);
        if remainder > midpoint || (remainder == midpoint && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded;
    }
    let mut half_exponent = (unbiased + 15) as u16;
    let mut half_mantissa = (mantissa >> 13) as u16;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 != 0) {
        half_mantissa += 1;
        if half_mantissa == 0x400 {
            half_mantissa = 0;
            half_exponent += 1;
        }
    }
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | (half_exponent << 10) | half_mantissa
}

fn f16_bits_to_f32(value: u16) -> f32 {
    let sign = ((value >> 15) & 1) as u32;
    let exponent = ((value >> 10) & 0x1f) as u32;
    let mantissa = (value & 0x03ff) as u32;
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            let mut normalized = mantissa;
            let mut unbiased = -14i32;
            while normalized & 0x400 == 0 {
                normalized <<= 1;
                unbiased -= 1;
            }
            normalized &= 0x3ff;
            (sign << 31) | (((unbiased + 127) as u32) << 23) | (normalized << 13)
        }
    } else if exponent == 31 {
        (sign << 31) | 0x7f80_0000 | (mantissa << 13)
    } else {
        (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn raw_bytes(values: &[f32], dtype: DType) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| match dtype {
            DType::F16 => f32_to_f16_bits(*value).to_le_bytes(),
            DType::BF16 => f32_to_bf16_bits(*value).to_le_bytes(),
            _ => unreachable!(),
        })
        .collect()
}

fn rounded_weight(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => f16_bits_to_f32(f32_to_f16_bits(value)),
        DType::BF16 => f32::from_bits((f32_to_bf16_bits(value) as u32) << 16),
        _ => unreachable!(),
    }
}

fn run(gpu: &mut Gpu, dtype: DType) {
    let (m, k, batch) = (37usize, 53usize, 7usize);
    let weights: Vec<f32> = (0..m * k)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.0078125)
        .collect();
    let activations: Vec<f32> = (0..batch * k)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.005859375)
        .collect();
    let actual = {
        let weight = gpu
            .alloc_owned(&[m, k], dtype)
            .expect("allocate raw weight");
        gpu.hip
            .memcpy_htod(&weight.buf, &raw_bytes(&weights, dtype))
            .expect("upload raw weight");
        let x = gpu
            .upload_owned_f32(&activations, &[batch, k])
            .expect("upload x");
        let y = gpu
            .alloc_owned(&[batch, m], DType::F32)
            .expect("allocate y");

        gpu.gemm_raw_x_f32_portable(&weight, &x, &y, m, k, batch)
            .expect("portable raw GEMM");
        gpu.device_synchronize().expect("synchronize");
        gpu.download_f32(&y).expect("download y")
    };
    gpu.reclaim_pending();

    let mut max_abs = 0.0f32;
    for batch_index in 0..batch {
        for output in 0..m {
            let mut expected = 0.0f32;
            for column in 0..k {
                expected = rounded_weight(weights[output * k + column], dtype)
                    .mul_add(activations[batch_index * k + column], expected);
            }
            max_abs = max_abs.max((actual[batch_index * m + output] - expected).abs());
        }
    }
    println!("dtype={dtype:?} max_abs={max_abs:.6e}");
    assert!(
        max_abs <= 1.0e-6,
        "portable {dtype:?} mismatch: {max_abs:.6e}"
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);
    run(&mut gpu, DType::F16);
    run(&mut gpu, DType::BF16);
}
