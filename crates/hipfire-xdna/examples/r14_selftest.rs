//! r14 array-GEMM layout selftest + dispatch timing.
//!
//! Validates the host packers in `gemm_r14.rs` against a CPU reference on random
//! data — an all-ones probe cannot distinguish a transposed base tile, so this
//! uses distinct random values per (row, col, k) and requires an EXACT match on
//! every C element. Then times steady-state dispatch to get the array's real
//! byte rate at DFlash shapes.
//!
//! Usage: r14_selftest [--dir ~/.hipfire/npu/r14_1x2x128_nb128] [--iters 20]

#[cfg(target_os = "linux")]
fn main() {
    use hipfire_xdna::gemm_r14::NpuGemmR14;
    use std::time::Instant;

    let argv: Vec<String> = std::env::args().collect();
    let arg = |k: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == k)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let dir = arg("--dir").unwrap_or_else(|| {
        format!(
            "{}/.hipfire/npu/r14_1x2x128_nb128",
            std::env::var("HOME").unwrap()
        )
    });
    let iters: usize = arg("--iters").and_then(|v| v.parse().ok()).unwrap_or(20);

    let mut g = NpuGemmR14::load_dir(&dir).expect("load r14");
    let geom = g.geom();
    let (mt, nt, kc) = (geom.m_tile(), geom.n_tile(), geom.k_chunk());
    println!(
        "[r14_selftest] {dir}\n  LM={} LN={} KT={} NBLK={}  =>  M_TILE={mt} N_TILE={nt} K_CHUNK={kc}",
        geom.lm, geom.ln, geom.kt, geom.nblk
    );
    println!(
        "  per-dispatch bytes: A={:.2} MiB  W={:.2} MiB  C={:.2} MiB",
        geom.a_bytes() as f64 / 1048576.0,
        geom.w_bytes() as f64 / 1048576.0,
        geom.c_bytes() as f64 / 1048576.0
    );

    // Deterministic pseudo-random A (int8) and W (int4 codes), one iteration's worth.
    let mut rng: u64 = 0x243f6a8885a308d3;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let a: Vec<i8> = (0..mt * kc).map(|_| (next() % 51) as i8 - 25).collect();
    let w: Vec<i8> = (0..nt * kc).map(|_| (next() % 15) as i8 - 7).collect(); // [N][K]

    // Pack slot 0 only; zero the rest so the other iterations are harmless.
    let mut wbuf = g.alloc_weights().expect("alloc w");
    wbuf.as_mut_slice().fill(0);
    geom.pack_w_slot(wbuf.as_mut_slice(), 0, &w, kc, 0, 0);
    g.sync_weights(&wbuf).expect("sync w");
    {
        let geom = geom;
        let abuf = g.a_mut();
        abuf.fill(0);
        geom.pack_a_slot(abuf, 0, &a, kc, 0, 0);
    }
    g.dispatch(&wbuf).expect("dispatch");
    let c = g.read_c().expect("read c");

    // Reference: C[r][n] = sum_k A[r][k] * W[n][k]
    let mut refc = vec![0i32; mt * nt];
    for r in 0..mt {
        for n in 0..nt {
            let mut acc = 0i32;
            for k in 0..kc {
                acc += a[r * kc + k] as i32 * w[n * kc + k] as i32;
            }
            refc[r * nt + n] = acc;
        }
    }
    let mut bad = 0usize;
    let mut first = None;
    geom.each_c(c, 0, |row, col, v| {
        let e = refc[row * nt + col];
        if v != e {
            if first.is_none() {
                first = Some((row, col, v, e));
            }
            bad += 1;
        }
    });
    if bad == 0 {
        println!("  LAYOUT: EXACT on all {} C elements", mt * nt);
    } else {
        println!(
            "  LAYOUT: MISMATCH on {bad}/{} elements; first {:?} (got, want)",
            mt * nt,
            first
        );
        std::process::exit(1);
    }

    // Timing: fill every slot so the dispatch moves its full byte budget.
    {
        let geom = geom;
        let abuf = g.a_mut();
        for b in 1..geom.nblk {
            geom.pack_a_slot(abuf, b, &a, kc, 0, 0);
        }
    }
    for b in 1..geom.nblk {
        geom.pack_w_slot(wbuf.as_mut_slice(), b, &w, kc, 0, 0);
    }
    g.sync_weights(&wbuf).expect("sync w");
    // Rotating over N distinct resident weight BOs, as the real body does (one
    // per dispatch): isolates any per-BO residency cost from the stream rate.
    let nbufs: usize = arg("--wbufs").and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut bufs = vec![wbuf];
    for _ in 1..nbufs {
        let mut b = g.alloc_weights().expect("alloc w");
        b.as_mut_slice().copy_from_slice(bufs[0].as_slice());
        g.sync_weights(&b).expect("sync w");
        bufs.push(b);
    }
    println!(
        "  weight BOs = {nbufs} ({:.0} MiB resident)",
        (nbufs * geom.w_bytes()) as f64 / 1048576.0
    );
    for b in &bufs {
        g.dispatch(b).expect("warm");
    }
    let mut ts = Vec::new();
    for i in 0..iters {
        let t = Instant::now();
        g.dispatch(&bufs[i % nbufs]).expect("timed");
        ts.push(t.elapsed().as_secs_f64());
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (min, med, max) = (ts[0], ts[iters / 2], ts[iters - 1]);
    let wgb = geom.w_bytes() as f64 / 1e9;
    let agg = (geom.w_bytes() + geom.a_bytes() + geom.c_bytes()) as f64 / 1e9;
    println!(
        "  dispatch n={iters}: min={:.0} us  med={:.0} us  max={:.0} us  (spread {:.1}%)",
        min * 1e6,
        med * 1e6,
        max * 1e6,
        (max - min) / med * 100.0
    );
    println!(
        "  W-path = {:.2} GB/s (med)   aggregate = {:.2} GB/s (med)",
        wgb / med,
        agg / med
    );
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
