// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixed-seed CPU/GPU parity gate for the shared configurable top-k sampler.

use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::sampler::{sample_top_k_top_p, SamplerRng};

fn main() {
    const SEED: u32 = 0x1357_9bdf;
    const TOP_K: usize = 64;
    const TOP_P: f32 = 0.95;
    const TEMPERATURE: f32 = 1.0;

    // Unique, smoothly varying values avoid tie-order ambiguity while keeping
    // substantial probability mass across all 64 retained candidates.
    let logits: Vec<f32> = (0..96)
        .map(|index| index as f32 * 0.03125 + (index % 7) as f32 * 0.0001)
        .collect();

    // The CPU stream is now an owned value seeded exactly as the GPU kernel is
    // below, so this gate compares two RNGs that were pinned together by
    // construction rather than by a global both happened to read.
    let mut cpu_stream = SamplerRng::from_seed(SEED);
    let cpu_token = sample_top_k_top_p(&logits, TEMPERATURE, TOP_K, TOP_P, &mut cpu_stream);
    let cpu_rng = cpu_stream.state();

    let mut gpu = Gpu::init().expect("initialize GPU");
    let logits_gpu = gpu
        .upload_f32(&logits, &[logits.len()])
        .expect("upload logits");
    let result_gpu = gpu.alloc_tensor(&[2], DType::F32).expect("result scratch");
    let repeat_gpu = gpu.alloc_tensor(&[1], DType::F32).expect("repeat scratch");
    let (gpu_token, gpu_rng) = gpu
        .sample_top_p_pf(
            &logits_gpu,
            &result_gpu,
            &repeat_gpu,
            logits.len(),
            TEMPERATURE,
            TOP_P,
            TOP_K,
            SEED,
            0,
            1.0,
            0.0,
            0.0,
        )
        .expect("GPU top-k sample");

    assert_eq!(gpu_token, cpu_token, "CPU/GPU top-k token mismatch");
    assert_eq!(gpu_rng, cpu_rng, "CPU/GPU sampler RNG mismatch");
    println!("PASS top_k={TOP_K} token={gpu_token} rng=0x{gpu_rng:08x}");
}
