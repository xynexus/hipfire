//! Measure aggregate W4A8 GEMM throughput on the NPU through the hipfire NpuKernel
//! dispatch path (no XRT). Loads a compiled mlir-aie kernel from a cache dir, binds
//! A/W/C, validates the all-ones result, then times a dispatch loop for TOPS.
//!
//! Run: cargo run -p hipfire-xdna --example npu_gemm_bench -- <cache-dir> \
//!        <asize> <wsize> <csize> <macs-per-dispatch> [iters] [expect_c0]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        if a.len() < 6 {
            eprintln!("usage: npu_gemm_bench <dir> <asz> <wsz> <csz> <macs> [iters] [expect_c0]");
            std::process::exit(2);
        }
        let dir = &a[1];
        let asz: usize = a[2].parse().unwrap();
        let wsz: usize = a[3].parse().unwrap();
        let csz: usize = a[4].parse().unwrap();
        let macs: f64 = a[5].parse().unwrap();
        let iters: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(500);
        let expect: Option<i32> = a.get(7).and_then(|s| s.parse().ok());

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");

        // C is over-allocated by GUARD bytes and the whole thing is pre-filled with a
        // sentinel. C[0] alone is NOT a sufficient gate: it only exercises the (i=0, j=0)
        // register-tile element, so it stays correct even when the kernel walks off the end
        // of its buffers. That is exactly how an LM < MT misconfiguration (r11_gemm's
        // register tile defaults to 3x3, and LM=1 is forced at M=16) hides -- it reads and
        // writes past A and C while C[0] still reads back clean. So: check EVERY element,
        // count how many were never written, and canary the bytes past the end.
        const GUARD: usize = 4096;
        const SENTINEL: i32 = -559038737; // 0xDEADBEEF

        let mut aw = k.alloc_arg(asz).expect("A");
        let mut ww = k.alloc_arg(wsz).expect("W");
        let mut cw = k.alloc_arg(csz + GUARD).expect("C");
        aw.as_mut_slice().fill(1);
        ww.as_mut_slice().fill(0x11);
        for b in cw.as_mut_slice().chunks_exact_mut(4) {
            b.copy_from_slice(&SENTINEL.to_ne_bytes());
        }

        // Correctness gate: all-ones W4A8 compute. Every C element must equal K, because
        // A is all 1 and each weight decodes to 1 (int4 nibble of 0x11, or int8 17).
        k.dispatch(&[&aw, &ww, &cw]).expect("dispatch");
        let c: &[i32] = unsafe {
            std::slice::from_raw_parts(cw.as_slice().as_ptr() as *const i32, (csz + GUARD) / 4)
        };
        let n = csz / 4;
        let c0 = c[0];
        println!(
            "all-ones C[0] = {c0}{}",
            match expect {
                Some(e) => format!(" (expect {e})"),
                None => String::new(),
            }
        );

        // Canary: nothing may touch the bytes past the declared C length.
        let clobbered = c[n..].iter().filter(|&&v| v != SENTINEL).count();
        if clobbered > 0 {
            let first = n + c[n..].iter().position(|&v| v != SENTINEL).unwrap();
            eprintln!(
                "OOB WRITE: {clobbered} of {} guard words past C[{n}] clobbered; \
                 first at C[{first}] = {}",
                GUARD / 4,
                c[first]
            );
            std::process::exit(5);
        }

        if let Some(e) = expect {
            let unwritten = c[..n].iter().filter(|&&v| v == SENTINEL).count();
            let bad = c[..n].iter().filter(|&&v| v != e).count();
            if bad > 0 {
                let first = c[..n].iter().position(|&v| v != e).unwrap();
                eprintln!(
                    "correctness FAIL: {bad}/{n} elements != {e} \
                     ({unwritten} never written); first bad at C[{first}] = {}",
                    c[first]
                );
                std::process::exit(4);
            }
            println!("  full-C gate: {n}/{n} elements == {e}, guard clean");
        } else if c[..n].iter().all(|&v| v == SENTINEL) {
            println!("  (C untouched — feed-only probe, no compute; guard clean)");
        }

        // Warm up, then time the dispatch loop.
        for _ in 0..20 {
            k.dispatch(&[&aw, &ww, &cw]).expect("warmup");
        }
        let t = Instant::now();
        for _ in 0..iters {
            k.dispatch(&[&aw, &ww, &cw]).expect("bench");
        }
        let dt = t.elapsed().as_secs_f64();
        let per = dt / iters as f64;
        let tops = 2.0 * macs / per / 1e12;
        println!(
            "iters={iters} total={:.3}s per_dispatch={:.1}us  MACs/dispatch={macs:.0}  => {tops:.2} TOPS",
            dt,
            per * 1e6
        );
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
