// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! How close is the SHIPPING exact-W4A8 GEMM to the gfx1151 iu4 issue ceiling?
//!
//! Reports two numbers per shape, because the kernel does TWO iu4 WMMA passes
//! per useful MAC (x = 16*x_hi + x_lo):
//!   hw   TOPS = iu4 ops the matrix unit actually retires -> compare to peak
//!   useful TOPS = the GEMM the caller asked for -> half of hw, by construction
//!
//! Peak: 80 SIMD * (8192 iu4 ops / 16 cyc) * 2.9 GHz.

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
    let peak = 80.0 * (8192.0 / 16.0) * 2.9e9 / 1e12;
    println!("gemm_oq_compact_iu4x2_w64 vs iu4 issue ceiling ({peak:.1} TOPS)\n");
    println!("  proj                  M      K    B      ms   hwTOPS  %peak  usefulTOPS");

    for group in GROUPS {
        println!("-- group={group} --");
        for &(name, m, k, b) in &[
            ("gate/up", 17408usize, 5120usize, 512usize),
            ("down", 5120, 17408, 512),
            ("qkv", 6144, 5120, 512),
            ("wo", 5120, 4096, 512),
            ("gate/up B=2048", 17408, 5120, 2048),
        ] {
            let ng = k / group;
            let stride = 2 + group / 2 + 2 * 3;
            let mut dev = vec![0u8; m * ng * stride];
            for v in dev.iter_mut() {
                *v = rnd();
            }
            let x: Vec<u8> = (0..b * k).map(|_| rnd()).collect();
            let xs: Vec<f32> = (0..b * ng).map(|_| 0.01f32).collect();
            let wb = gpu.upload_raw(&dev, &[dev.len()]).expect("w");
            let xb = gpu.upload_raw(&x, &[x.len()]).expect("x");
            let xsb = gpu.upload_f32(&xs, &[xs.len()]).expect("xs");
            let yb = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).expect("y");

            for _ in 0..3 {
                gpu.gemm_oq_compact_iu4x2_w64(&wb, &xb, &xsb, &yb, m, k, b, stride)
                    .expect("warm");
            }
            gpu.device_synchronize().expect("sync");
            let iters = 20usize;
            let t = Instant::now();
            for _ in 0..iters {
                gpu.gemm_oq_compact_iu4x2_w64(&wb, &xb, &xsb, &yb, m, k, b, stride)
                    .expect("run");
            }
            gpu.device_synchronize().expect("sync");
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            // 2 ops per MAC, and the kernel retires 2 iu4 passes per MAC.
            let useful = 2.0 * m as f64 * k as f64 * b as f64 / (ms * 1e-3) / 1e12;
            let hw = 2.0 * useful;
            println!(
                "  {name:<16} {m:6} {k:6} {b:4}  {ms:6.2}  {hw:7.1}  {:5.1}%  {useful:10.1}",
                hw / peak * 100.0
            );
        }
    }
}
