//! Decode-shaped (GEMV) throughput on the NPU with DEVICE-RESIDENT weights.
//!
//! Decode is one token through every linear: `C[1,N] = A[1,K] · W[K,N]`. It is
//! bandwidth-bound, not compute-bound — the whole weight matrix streams from DRAM per
//! token — so the figure of merit is achieved W bytes/s against the ~55 GB/s fabric, not
//! TOPS. Padding M=1 up to the kernel's `block_m` costs nothing in bandwidth terms
//! (the extra rows ride along on weight traffic that had to happen anyway), which is why
//! the prefill kernel can serve decode at all.
//!
//! Weights are uploaded once via `NpuGemm::upload_weights`, so the timed loop has zero
//! host weight traffic — the condition any real decode path must meet.
//!
//! Run: cargo run --release -p hipfire-xdna --example npu_decode_bench -- \
//!        <dir> MT KCHUNK GROUPS NB ROUNDS [K N iters]
fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemm;
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (mt, kc, g, nb, rounds) = (p(2, 8), p(3, 32), p(4, 64), p(5, 8), p(6, 4));
        let (k, n) = (p(7, 2048), p(8, 8192));
        let iters = p(9, 50);

        let x = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let i = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let mut gemm = NpuGemm::load_rounds(&x, &i, mt, 4, kc, g, nb, rounds).expect("load");
        let (bm, bn, bk) = (gemm.block_m(), gemm.block_n(), gemm.block_k());
        assert!(
            n % bn == 0 && k % bk == 0,
            "K/N must tile (block {bm}x{bn}x{bk})"
        );

        let rnd = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0xf) as i32 - 8) as i8
        };
        // One token, padded to the kernel's M block. Only row 0 is the real activation.
        let av: Vec<i8> = (0..bm * k)
            .map(|i| if i < k { rnd(i) } else { 0 })
            .collect();
        let wv: Vec<i8> = (0..k * n).map(|i| rnd(7_777_777 + i)).collect();
        let mut cv = vec![0i32; bm * n];

        let tup = Instant::now();
        let weights = gemm.upload_weights(k, n, &wv).expect("upload");
        let up_ms = tup.elapsed().as_secs_f64() * 1e3;
        let w_bytes = k * n / 2; // int4

        gemm.run_resident(bm, k, n, &av, &weights, &mut cv)
            .expect("run");
        let mut bad = 0usize;
        for nn in 0..n {
            let acc: i32 = (0..k)
                .map(|kk| av[kk] as i32 * wv[kk * n + nn] as i32)
                .sum();
            if cv[nn] != acc {
                bad += 1;
            }
        }

        let t = Instant::now();
        for _ in 0..iters {
            gemm.run_resident(bm, k, n, &av, &weights, &mut cv)
                .expect("run");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "K={k} N={n} block {bm}x{bn}x{bk}  {} dispatches  (upload {:.1} ms once, {} MB resident)",
            (k / bk) * (n / bn),
            up_ms,
            weights.len() * 0 + w_bytes / (1 << 20)
        );
        println!(
            "  {:.3} ms/token-linear  =>  {:.1} GB/s W stream  ({})",
            per * 1e3,
            w_bytes as f64 / per / 1e9,
            if bad == 0 {
                "row 0 correct".to_string()
            } else {
                format!("{bad} MISMATCHES")
            }
        );
    }
}
