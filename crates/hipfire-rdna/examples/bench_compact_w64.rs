// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Compact W4A4: the wave32 kernel vs the wave64 port, at real prefill shapes.
//! Same block bytes, same result — only the kernel STRUCTURE differs (wave64,
//! BK=64 K-strip, N-heavy BM=64/BN=256, register-staged double buffer).

use hipfire_rdna::Gpu;
use std::time::Instant;

/// Swept: the compact dispatches accept 256 or 128, and `OqPlusCompactG128`
/// (qt=52) is a real on-disk format, but the timing benches were 256-only.
const GROUPS: [usize; 2] = [256, 128];

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let mut seed = 0x2468_1357u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 16) as u8
    };
    let iters = 10usize;
    println!("compact W4A4: wave32 vs wave64 structure (iu4 peak = 99.2 TOPS)\n");
    println!("  proj             M      K    B    w32ms w32TOPS    w64ms w64TOPS  speedup  %peak");

    for group in GROUPS {
        println!("-- group={group} --");
        for &(name, m, k, b) in &[
            ("gate/up", 17408usize, 5120usize, 256usize),
            ("down", 5120, 17408, 256),
            ("qkv", 6144, 5120, 256),
            ("wo", 5120, 4096, 256),
            ("gate/up B=512", 17408, 5120, 512),
        ] {
            let ng = k / group;
            let stride = 2 + group / 2 + 2 * 3; // N_out=3 => 4.25 bits
            let nblk = m * ng;
            let mut dev = vec![0u8; nblk * stride];
            for v in dev.iter_mut() {
                *v = rnd();
            }
            let x: Vec<u8> = (0..b * k / 2).map(|_| rnd()).collect();
            let xs: Vec<f32> = (0..b * ng).map(|_| 0.01f32).collect();
            let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
            let xb = gpu.upload_raw(&x, &[x.len()]).expect("x");
            let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
            let yb = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).expect("y");

            for _ in 0..2 {
                gpu.gemm_oq_compact_iu4_wmma(&wb, &xb, &xsb, &yb, m, k, b, group, stride)
                    .expect("w32");
                gpu.gemm_oq_compact_iu4_w64(&wb, &xb, &xsb, &yb, m, k, b, stride)
                    .expect("w64");
            }
            gpu.device_synchronize().expect("sync");

            let t0 = Instant::now();
            for _ in 0..iters {
                gpu.gemm_oq_compact_iu4_wmma(&wb, &xb, &xsb, &yb, m, k, b, group, stride)
                    .expect("w32");
            }
            gpu.device_synchronize().expect("sync");
            let ms32 = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

            let t1 = Instant::now();
            for _ in 0..iters {
                gpu.gemm_oq_compact_iu4_w64(&wb, &xb, &xsb, &yb, m, k, b, stride)
                    .expect("w64");
            }
            gpu.device_synchronize().expect("sync");
            let ms64 = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

            let tops = |ms: f64| 2.0 * (m as f64) * (k as f64) * (b as f64) / (ms * 1e-3) / 1e12;
            println!(
            "  {name:<12} {m:>6} {k:>6} {b:>4} {ms32:>8.3} {:>7.2} {ms64:>8.3} {:>7.2} {:>8.2}x {:>5.1}%",
            tops(ms32),
            tops(ms64),
            ms32 / ms64,
            100.0 * tops(ms64) / 99.2
        );
            for t in [wb, xb, xsb, yb] {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}
