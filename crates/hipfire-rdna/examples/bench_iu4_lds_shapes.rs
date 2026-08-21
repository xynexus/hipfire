// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Prices the tuned `gemm_iu4_i32_wmma_lds` (wave64, BK=64, N-heavy 2x8) at the
//! SAME shapes the compact iu4 GEMM is benched on, to see how much of the gap to
//! the 99.2 TOPS peak is structure rather than the compact format's extras.

use hipfire_rdna::Gpu;
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x1234_5678u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u8
    };
    let iters = 10usize;
    println!("gemm_iu4_i32_wmma_lds (pure int4 core, no scales) at compact shapes\n");
    println!("  proj             M      K    B      ms     TOPS   % of 99.2");
    for &(name, m, k, b) in &[
        ("gate/up", 17408usize, 5120usize, 256usize),
        ("down", 5120, 17408, 256),
        ("qkv", 6144, 5120, 256),
        ("wo", 5120, 4096, 256),
    ] {
        let a: Vec<u8> = (0..m * k / 2).map(|_| rnd()).collect();
        let x: Vec<u8> = (0..b * k / 2).map(|_| rnd()).collect();
        let ab = gpu.upload_raw(&a, &[a.len()]).expect("a");
        let xb = gpu.upload_raw(&x, &[x.len()]).expect("x");
        let yb = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).expect("y");
        if gpu.gemm_iu4_i32_wmma_lds(&ab, &xb, &yb, m, k, b).is_err() {
            println!("  {name:<12} launch failed");
            continue;
        }
        gpu.device_synchronize().expect("sync");
        let t = Instant::now();
        for _ in 0..iters {
            gpu.gemm_iu4_i32_wmma_lds(&ab, &xb, &yb, m, k, b)
                .expect("run");
        }
        gpu.device_synchronize().expect("sync");
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tops = 2.0 * (m as f64) * (k as f64) * (b as f64) / (ms * 1e-3) / 1e12;
        println!(
            "  {name:<12} {m:>6} {k:>6} {b:>4} {ms:>7.3} {tops:>8.2} {:>9.1}%",
            100.0 * tops / 99.2
        );
        for t in [ab, xb, yb] {
            let _ = gpu.free_tensor(t);
        }
    }
}
