// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Locked F32 CPU/GPU goldens for Gemma 4's generic mathematical primitives.

use hipfire_rdna::Gpu;

const QWEN_Q_BITS: [u32; 32] = [
    0xbf4cd1f1, 0xbeb5723e, 0xbf488a1b, 0xbf3c288a, 0xbf4e3417, 0xbf7dddfa, 0xbf12e536, 0xbf001cf2,
    0xbee1e1e2, 0xbec3c3c4, 0xbea5a5a6, 0xbe878788, 0xbe52d2d3, 0xbe169697, 0xbdb4b4b5, 0xbcf0f0f1,
    0xbc2d91c1, 0xbde185d4, 0x3e0f1f59, 0x3e528d54, 0x3e88416b, 0x3ea22d14, 0x3ec528ed, 0x3ee1f216,
    0x3f000000, 0x3f0f0f0f, 0x3f1e1e1e, 0x3f2d2d2d, 0x3f3c3c3c, 0x3f4b4b4b, 0x3f5a5a5a, 0x3f696969,
];
const QWEN_K_BITS: [u32; 16] = [
    0x3f07b1ae, 0x3e9b8093, 0x3ed77433, 0x3eb1380b, 0x3eb48f81, 0x3ee1f11e, 0x3dfcba11, 0x3d1e63ec,
    0xbd1d89d9, 0xbdec4ec5, 0xbe44ec4f, 0xbe89d89e, 0xbeb13b14, 0xbed89d8a, 0xbf000000, 0xbf13b13b,
];

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max <= tolerance, "{label}: max abs {max} > {tolerance}");
}

fn rope_cpu(
    values: &[f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    basis_dim: usize,
    position: usize,
    theta: f32,
) -> Vec<f32> {
    let mut out = values.to_vec();
    let pairs = rotary_dim / 2;
    let half = basis_dim / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for i in 0..pairs {
            let frequency = 1.0 / theta.powf((2 * i) as f32 / basis_dim as f32);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let first = values[base + i];
            let second = values[base + i + half];
            out[base + i] = first * cos - second * sin;
            out[base + i + half] = first * sin + second * cos;
        }
    }
    out
}

fn rms_cpu(x: &[f32], rows: usize, width: usize, eps: f32, weight: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for row in 0..rows {
        let values = &x[row * width..(row + 1) * width];
        let inv = (values.iter().map(|v| v * v).sum::<f32>() / width as f32 + eps)
            .sqrt()
            .recip();
        for col in 0..width {
            out[row * width + col] = values[col] * inv * weight.map_or(1.0, |w| w[col]);
        }
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let pos = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos, &19_i32.to_ne_bytes()).unwrap();

    let q_host = (0..32)
        .map(|i| ((i as f32) - 15.5) / 17.0)
        .collect::<Vec<_>>();
    let k_host = (0..16)
        .map(|i| ((i as f32) - 7.5) / -13.0)
        .collect::<Vec<_>>();
    let q = gpu.upload_f32(&q_host, &[2, 16]).unwrap();
    let k = gpu.upload_f32(&k_host, &[1, 16]).unwrap();
    gpu.rope_partial_interleaved_f32(&q, &k, &pos, 2, 1, 16, 8, 8, 1_000_000.0)
        .unwrap();
    assert_eq!(
        gpu.download_f32(&q)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        QWEN_Q_BITS,
        "basis_dim == rotary_dim must preserve the frozen Qwen Q bytes"
    );
    assert_eq!(
        gpu.download_f32(&k)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        QWEN_K_BITS,
        "basis_dim == rotary_dim must preserve the frozen Qwen K bytes"
    );
    let _ = gpu.free_tensor(q);
    let _ = gpu.free_tensor(k);

    for (label, rotary_dim, basis_dim, theta) in [
        ("full/local RoPE", 16, 16, 10_000.0),
        ("proportional RoPE", 8, 16, 1_000_000.0),
    ] {
        let q = gpu.upload_f32(&q_host, &[2, 16]).unwrap();
        let k = gpu.upload_f32(&k_host, &[1, 16]).unwrap();
        gpu.rope_partial_interleaved_f32(&q, &k, &pos, 2, 1, 16, rotary_dim, basis_dim, theta)
            .unwrap();
        assert_close(
            &format!("{label} Q"),
            &gpu.download_f32(&q).unwrap(),
            &rope_cpu(&q_host, 2, 16, rotary_dim, basis_dim, 19, theta),
            2e-6,
        );
        assert_close(
            &format!("{label} K"),
            &gpu.download_f32(&k).unwrap(),
            &rope_cpu(&k_host, 1, 16, rotary_dim, basis_dim, 19, theta),
            2e-6,
        );
        let _ = gpu.free_tensor(q);
        let _ = gpu.free_tensor(k);
    }

    let norm_input = (0..16)
        .map(|i| ((i as f32) - 7.0) / 5.0)
        .collect::<Vec<_>>();
    let norm_weight = (0..8).map(|i| 0.7 + i as f32 * 0.05).collect::<Vec<_>>();
    let x = gpu.upload_f32(&norm_input, &[2, 8]).unwrap();
    let w = gpu.upload_f32(&norm_weight, &[8]).unwrap();
    let weighted = gpu.zeros(&[2, 8], hipfire_rdna::DType::F32).unwrap();
    let weightless = gpu.zeros(&[2, 8], hipfire_rdna::DType::F32).unwrap();
    gpu.rmsnorm_batched(&x, &w, &weighted, 2, 8, 1e-6).unwrap();
    gpu.rmsnorm_weightless_f32(&x, &weightless, 1e-6).unwrap();
    assert_close(
        "weighted Q/K norm",
        &gpu.download_f32(&weighted).unwrap(),
        &rms_cpu(&norm_input, 2, 8, 1e-6, Some(&norm_weight)),
        2e-6,
    );
    assert_close(
        "weightless V norm",
        &gpu.download_f32(&weightless).unwrap(),
        &rms_cpu(&norm_input, 2, 8, 1e-6, None),
        2e-6,
    );
    for tensor in [x, w, weighted, weightless] {
        let _ = gpu.free_tensor(tensor);
    }

    let values = vec![-90.0, -30.0, -3.0, 0.0, 2.0, 30.0, 90.0];
    let scaled = gpu.upload_f32(&values, &[values.len()]).unwrap();
    gpu.scale_f32(&scaled, 4.0).unwrap();
    assert_close(
        "Q scaling",
        &gpu.download_f32(&scaled).unwrap(),
        &values.iter().map(|v| v * 4.0).collect::<Vec<_>>(),
        0.0,
    );
    gpu.scale_f32(&scaled, 0.125).unwrap();
    assert_close(
        "layer scalar",
        &gpu.download_f32(&scaled).unwrap(),
        &values.iter().map(|v| v * 0.5).collect::<Vec<_>>(),
        0.0,
    );
    let softcap = gpu.upload_f32(&values, &[values.len()]).unwrap();
    gpu.vector_softcap_f32(&softcap, &softcap, values.len(), 30.0)
        .unwrap();
    assert_close(
        "final vector softcap",
        &gpu.download_f32(&softcap).unwrap(),
        &values
            .iter()
            .map(|value| 30.0 * (value / 30.0).tanh())
            .collect::<Vec<_>>(),
        3e-6,
    );
    let _ = gpu.free_tensor(scaled);
    let _ = gpu.free_tensor(softcap);
    gpu.hip.free(pos).unwrap();
    gpu.drain_pool();
    println!("gemma4_operator_goldens: PASS");
}
