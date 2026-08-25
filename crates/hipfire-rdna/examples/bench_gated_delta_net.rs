// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Microbench for the live decode DeltaNet recurrence, at Qwen3.8-27B shapes.
//!
//! The kernel had never been tuned: `perf(qwen35): chunkwise-parallel gated
//! DeltaNet` dismissed it at "0.7%" of a prefill-shaped profile, but in spec
//! decode it is 7% of the cycle at B=8 and 14% at B=24, because 48 of the 27B's
//! 64 layers are DeltaNet and the recurrence is sequential in tokens.
//!
//! ENV: GDN_HEADS (default 32), GDN_TOKENS (sweep when unset).

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    println!("gated_delta_net_f32 on {}", gpu.arch);
    let hd = 128usize;
    let n_heads: usize = std::env::var("GDN_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let toks: Vec<usize> = match std::env::var("GDN_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(t) => vec![t],
        None => vec![1, 2, 4, 8, 10, 16, 24],
    };
    println!("  heads={n_heads} head_dim={hd}");
    println!(
        "  {:>7} {:>10} {:>12} {:>12}",
        "tokens", "ms", "us/token", "vs t=1"
    );

    let mut base = 0.0f64;
    for (i, &nt) in toks.iter().enumerate() {
        let rows = nt * n_heads * hd;
        let mut mk = |n: usize| {
            let host: Vec<f32> = (0..n).map(|j| ((j % 17) as f32 - 8.0) * 0.01).collect();
            gpu.upload_f32(&host, &[n]).expect("upload")
        };
        let q = mk(rows);
        let k = mk(rows);
        let v = mk(rows);
        let gate = mk(nt * n_heads);
        let beta = mk(nt * n_heads);
        let state = mk(n_heads * hd * hd);
        let out = gpu.alloc_tensor(&[rows], DType::F32).expect("alloc");

        for _ in 0..3 {
            gpu.gated_delta_net_f32(&q, &k, &v, &gate, &beta, &state, &out, nt, n_heads, hd)
                .expect("run");
        }
        gpu.device_synchronize().expect("sync");
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gpu.gated_delta_net_f32(&q, &k, &v, &gate, &beta, &state, &out, nt, n_heads, hd)
                .expect("run");
        }
        gpu.device_synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        if i == 0 {
            base = ms;
        }
        println!(
            "  {:>7} {:>10.4} {:>12.2} {:>12.2}",
            nt,
            ms,
            ms * 1000.0 / nt as f64,
            ms / base
        );
        for t in [q, k, v, gate, beta, state, out] {
            let _ = gpu.free_tensor(t);
        }
    }
}
