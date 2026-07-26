//! Correctness check for the r5 K-SPLIT cascade (`r5_ksplit_gen.py`): proves each
//! cascade core consumes a DIFFERENT K-slice (a real GEMM), not the replicated
//! broadcast r5_gen.py uses. Differential test: activations = 1, and core `ri`'s
//! weight tiles = `ri+1`; a correct K-split gives C[0] = Σ_ri KSLICE·16·(ri+1)
//! (each core does KSLICE mmuls × 16 K, A=1 · W=ri+1), summed over the cascade.
//! Replication (all cores same W) would give a different value.
//!
//! Usage: `npu_cascade_verify DIR XSZ CSZ AW WW ROWS KSLICE`   (hold hipfire lock)

#[cfg(target_os = "linux")]
fn main() {
    use hipfire_xdna::NpuKernel;

    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        eprintln!("usage: npu_cascade_verify DIR XSZ CSZ AW WW ROWS KSLICE");
        std::process::exit(2);
    }
    let dir = &a[1];
    let xsz: usize = a[2].parse().unwrap();
    let csz: usize = a[3].parse().unwrap();
    let aw: usize = a[4].parse().unwrap();
    let ww: usize = a[5].parse().unwrap();
    let rows: usize = a[6].parse().unwrap();
    let kslice: usize = a[7].parse().unwrap();
    let xe = aw + ww; // combined A|W bytes per core

    let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("final.xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts.bin");
    let kernel = NpuKernel::load(&xclbin, &insts).expect("load");
    let mut x = kernel.alloc_arg(xsz).expect("X");
    let mut c = kernel.alloc_arg(csz).expect("C");

    // One column's X = [core0_A, core0_W, core1_A, core1_W, ...]; fill A=1, W(core ri)=ri+1.
    x.as_mut_slice().fill(0);
    let ncols = xsz / (rows * xe);
    for col in 0..ncols {
        for ri in 0..rows {
            let base = (col * rows + ri) * xe;
            for b in &mut x.as_mut_slice()[base..base + aw] {
                *b = 1; // int8 activation = 1
            }
            let v = (ri as u8 + 1) & 0x0f;
            for b in &mut x.as_mut_slice()[base + aw..base + xe] {
                *b = v | (v << 4); // two int4 = ri+1
            }
        }
    }
    c.as_mut_slice().fill(0);
    kernel.dispatch(&[&x, &c]).expect("dispatch");

    // With INNER>1 the kernel recomputes its slice INNER times, so C scales by INNER;
    // pass INNER as a8 (default 1). Correctness = each tile's [0] == base*inner.
    let inner: i32 = a.get(8).and_then(|v| v.parse().ok()).unwrap_or(1);
    let base: i32 = (0..rows).map(|ri| (kslice * 16 * (ri + 1)) as i32).sum();
    let expected = base * inner;
    // Every output tile (CW=64 int32 = 4x16) is fed the same differential pattern, so
    // each tile's first element must equal `expected`. Checking ALL tiles proves NB
    // streaming produced every tile (not just the first) — the M/N-tiling correctness.
    const CW: usize = 64;
    let cbuf = c.as_slice();
    let n_tiles = (csz / 4) / CW;
    let mut bad = 0usize;
    let mut first_bad = None;
    for t in 0..n_tiles {
        let v = i32::from_le_bytes(cbuf[t * CW * 4..t * CW * 4 + 4].try_into().unwrap());
        if v != expected {
            bad += 1;
            first_bad.get_or_insert((t, v));
        }
    }
    let c0 = i32::from_le_bytes(cbuf[0..4].try_into().unwrap());
    let ok = bad == 0;
    println!(
        "npu_cascade_verify {dir}: C[0]={c0} expected={expected} tiles={n_tiles} bad={bad}{} (K-split {}) cols={ncols} rows={rows}",
        first_bad.map(|(t, v)| format!(" first_bad=tile{t}:{v}")).unwrap_or_default(),
        if ok { "CORRECT" } else { "WRONG — streaming/K-split/layout bug" }
    );
    // Optional timing: pass MACS_PER_DISPATCH as a9 (+ ITERS a10) to report GMAC/s.
    if let Some(macs) = a.get(9).and_then(|v| v.parse::<f64>().ok()) {
        use std::time::Instant;
        let iters: usize = a.get(10).and_then(|v| v.parse().ok()).unwrap_or(200);
        for _ in 0..8 {
            kernel.dispatch(&[&x, &c]).expect("warmup");
        }
        let t = Instant::now();
        for _ in 0..iters {
            kernel.dispatch(&[&x, &c]).expect("dispatch");
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "  device_us={us:.2} GMAC/s={:.1} iters={iters}",
            macs / (us * 1e-6) / 1e9
        );
    }
    if !ok {
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
