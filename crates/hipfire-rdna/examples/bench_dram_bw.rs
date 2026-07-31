// SPDX-License-Identifier: Apache-2.0
//! Achievable streaming DRAM bandwidth on this GPU — the roofline input for the
//! oq4 int4-act throughput analysis. Times bandwidth-heavy, compute-light ops on
//! a large buffer: scale_f32 (1 read + 1 write = 2 B/elem) and add_inplace_f32
//! (2 read + 1 write = 3 B/elem). Reports GB/s = bytes_moved / time.

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let n = 64 * 1024 * 1024usize; // 64M f32 = 256 MiB per buffer
    let bytes_buf = (n * 4) as f64;
    let a = gpu.alloc_tensor(&[n], DType::F32).unwrap();
    let b = gpu.alloc_tensor(&[n], DType::F32).unwrap();
    gpu.fill_f32(&a, 1.0).unwrap();
    gpu.fill_f32(&b, 2.0).unwrap();
    gpu.device_synchronize().unwrap();

    let iters = 30usize;
    macro_rules! med_gbps {
        ($call:expr, $bytes_moved:expr) => {{
            for _ in 0..5 {
                $call;
            }
            gpu.device_synchronize().unwrap();
            let mut gbps = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                $call;
                gpu.device_synchronize().unwrap();
                let s = t.elapsed().as_secs_f64();
                gbps.push(($bytes_moved) / s / 1e9);
            }
            gbps.sort_by(|x, y| x.partial_cmp(y).unwrap());
            gbps[iters / 2]
        }};
    }

    let copy_gbps = med_gbps!(gpu.copy_d2d(&b, &a, n * 4).unwrap(), 2.0 * bytes_buf);
    let add_gbps = med_gbps!(gpu.add_inplace_f32(&a, &b).unwrap(), 3.0 * bytes_buf);

    println!("achievable DRAM bandwidth  arch={}  buf=256MiB  iters={iters}", gpu.arch);
    println!("  copy_d2d (2 B/elem):         {copy_gbps:7.1} GB/s");
    println!("  add_inplace_f32 (3 B/elem):  {add_gbps:7.1} GB/s");
    println!("  peak achieved:               {:7.1} GB/s", copy_gbps.max(add_gbps));
}
