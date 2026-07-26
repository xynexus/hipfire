//! Validate the productionized NpuGemmMp primitive: prepack W once, load it, run a full
//! row-major W4A8 GEMM tiled over M-parallel dispatches, compare to a CPU reference. Point
//! it at an M-parallel xclbin (r6_gen_mp.py, ROUNDS=1) built for (COLS, MT, KCHUNK, NB).
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run -p hipfire-xdna --example npu_gemm_mp_verify -- <dir>  (config from dir name)

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuGemmMp;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        // Self-describing: config (COLS/MT/KCHUNK/NB) is parsed from the cache dir name.
        let mut g = NpuGemmMp::load_cached(dir).unwrap();
        let (k, n, rows_per) = (g.k(), g.n(), g.rows_per_dispatch());

        let weight_bits = g.weight_bits();
        let pattern = std::env::var("HIPFIRE_NPU_GEMM_VERIFY_PATTERN").unwrap_or_default();
        let basis_k = std::env::var("HIPFIRE_NPU_GEMM_BASIS_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let basis_col0 = std::env::var("HIPFIRE_NPU_GEMM_BASIS_COL0")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let rnd_a = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            (((s >> 13) & 0x7f) as i32 - 63) as i8
        };
        let rnd_w = |i: usize| -> i8 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            if weight_bits == 8 {
                (((s >> 9) & 0xff) as i32 - 128) as i8
            } else {
                (((s >> 13) & 0xf) as i32 - 8) as i8
            }
        };
        let wv: Vec<i8> = if pattern == "basis" {
            let mut wv = vec![0i8; k * n];
            for offset in 0..64.min(n.saturating_sub(basis_col0)) {
                let nn = basis_col0 + offset;
                wv[basis_k * n + nn] = (offset as i8).saturating_add(1);
            }
            wv
        } else {
            (0..k * n).map(|i| rnd_w(7_777_777 + i)).collect()
        };
        let packed_w = g.prepack_weights(k, n, &wv);
        if pattern == "basis" {
            let nonzero_chunks: Vec<_> = packed_w
                .chunks(64)
                .enumerate()
                .filter_map(|(idx, chunk)| {
                    let nz = chunk.iter().filter(|&&v| v != 0).count();
                    (nz != 0).then_some((idx, nz, chunk[0], chunk[n.min(8).min(63)]))
                })
                .take(32)
                .collect();
            eprintln!("basis packed_w nonzero 64B chunks={nonzero_chunks:?}");
        }
        g.load_weights(&packed_w);

        // Two sizes: one dispatch, and a 3-tile M (exercises the M-loop).
        for &m in &[rows_per, rows_per * 3] {
            let av: Vec<i8> = if pattern == "basis" {
                let mut av = vec![0i8; m * k];
                av[basis_k] = 1;
                av
            } else {
                (0..m * k).map(rnd_a).collect()
            };
            let mut cv = vec![0i32; m * n];
            g.run(m, k, n, &av, &mut cv).unwrap();
            if pattern == "basis" {
                eprintln!("basis row0 first64 got={:?}", &cv[0..n.min(64)]);
                let nz: Vec<_> = cv
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &v)| (v != 0).then_some((idx / n, idx % n, v)))
                    .take(128)
                    .collect();
                eprintln!("basis first nonzero outputs={nz:?}");
            }
            // CPU reference over the full output. Sampling rows missed W8 row leakage bugs.
            let mut bad = 0usize;
            let mut samples = Vec::new();
            for mm in 0..m {
                for nn in 0..n {
                    let acc: i32 = (0..k)
                        .map(|kk| av[mm * k + kk] as i32 * wv[kk * n + nn] as i32)
                        .sum();
                    if cv[mm * n + nn] != acc {
                        bad += 1;
                        if samples.len() < 8 {
                            samples.push((mm, nn, cv[mm * n + nn], acc));
                        }
                    }
                }
            }
            println!(
                "M={m} K={k} N={n} ({} dispatches): {bad} mismatches",
                m / rows_per
            );
            if bad != 0 {
                for (mm, nn, got, want) in samples {
                    eprintln!("  mismatch row={mm} col={nn}: got={got} want={want}");
                }
                eprintln!("NpuGemmMp WRONG");
                std::process::exit(4);
            }
        }
        println!("NpuGemmMp W{weight_bits}A8 GEMM CORRECT — M-parallel W-broadcast, row-major, weights broadcast once");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
