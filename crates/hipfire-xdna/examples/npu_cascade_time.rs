//! Device-time / GMAC-per-second for a raw NPU xclbin dispatched through NpuKernel.
//! Used to measure the r5 cascade GEMM array (`benchmarks/npu_gemm_tuning/r5`) vs the
//! r6 fullk baseline (~210 GMAC/s). Prep-once, timed dispatch loop.
//!
//! Usage: `npu_cascade_time DIR ASZ WSZ CSZ MACS_PER_DISPATCH [ITERS]`  (hold hipfire lock)

#[cfg(target_os = "linux")]
fn main() {
    use std::time::Instant;

    use hipfire_xdna::NpuKernel;

    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: npu_cascade_time DIR ASZ WSZ CSZ MACS_PER_DISPATCH [ITERS]");
        std::process::exit(2);
    }
    let dir = &a[1];
    let asz: usize = a[2].parse().expect("ASZ");
    let wsz: usize = a[3].parse().expect("WSZ");
    let csz: usize = a[4].parse().expect("CSZ");
    let macs: f64 = a[5].parse().expect("MACS");
    let iters: usize = a.get(6).and_then(|v| v.parse().ok()).unwrap_or(300);

    let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("final.xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts.bin");
    let kernel = NpuKernel::load(&xclbin, &insts).expect("load");
    let mut abuf = kernel.alloc_arg(asz).expect("A");
    let mut wbuf = kernel.alloc_arg(wsz).expect("W");
    let mut cbuf = kernel.alloc_arg(csz).expect("C");
    abuf.as_mut_slice().fill(1);
    wbuf.as_mut_slice().fill(0x11);
    cbuf.as_mut_slice().fill(0);

    for _ in 0..8 {
        kernel.dispatch(&[&abuf, &wbuf, &cbuf]).expect("warmup");
    }
    let c0 = unsafe { *(cbuf.as_slice().as_ptr() as *const i32) };
    let started = Instant::now();
    for _ in 0..iters {
        kernel.dispatch(&[&abuf, &wbuf, &cbuf]).expect("dispatch");
    }
    let us = started.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "npu_cascade_time {dir}: device_us={us:.2} GMAC/s={:.1} C[0]={c0} iters={iters}",
        macs / (us * 1e-6) / 1e9
    );
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
