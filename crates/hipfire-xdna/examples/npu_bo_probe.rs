//! R134 -- BO-size scaling probe. Like `npu_gemm_bench`, but (a) runs several
//! independent timing passes so run-to-run spread is visible, and (b) can skip the
//! per-dispatch `sync_bo` cache flush on the argument buffers.
//!
//! `npu_gemm_bench` calls `NpuKernel::dispatch`, which flushes EVERY argument buffer
//! to the device on EVERY iteration (`submit_synced(args, None)` ->
//! `sync_bo(TO_DEVICE, a.len())`). That is a full-buffer host cache maintenance op
//! whose cost is linear in BO SIZE, not in bytes the NPU actually reads -- exactly the
//! confound that R133's "1.77x BO-size penalty" cannot distinguish from a real DMA
//! bandwidth change. `sync=0` here flushes once before the loop and then uses
//! `dispatch_synced(args, &[false, false, false])`, isolating device-side time.
//!
//! Run: cargo run -p hipfire-xdna --example npu_bo_probe -- \
//!        <dir> <asz> <wsz> <csz> <w_read_bytes> <iters> <passes> <sync 0|1> [expect_c0]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        if a.len() < 9 {
            eprintln!(
                "usage: npu_bo_probe <dir> <asz> <wsz> <csz> <w_read> <iters> <passes> <sync> [expect_c0]"
            );
            std::process::exit(2);
        }
        let dir = &a[1];
        let asz: usize = a[2].parse().unwrap();
        let wsz: usize = a[3].parse().unwrap();
        let csz: usize = a[4].parse().unwrap();
        let wread: f64 = a[5].parse().unwrap();
        let iters: u32 = a[6].parse().unwrap();
        let passes: u32 = a[7].parse().unwrap();
        let syncmode: u32 = a[8].parse().unwrap();
        let expect: Option<i32> = a.get(9).and_then(|s| s.parse().ok());

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");

        let mut aw = k.alloc_arg(asz).expect("A");
        let mut ww = k.alloc_arg(wsz).expect("W");
        let mut cw = k.alloc_arg(csz).expect("C");
        aw.as_mut_slice().fill(1);
        ww.as_mut_slice().fill(0x11);
        cw.as_mut_slice().fill(0);

        // Correctness gate (always with a full flush so the W bytes are certainly live).
        k.dispatch(&[&aw, &ww, &cw]).expect("dispatch");
        let c0 = unsafe { *(cw.as_slice().as_ptr() as *const i32) };
        println!(
            "C[0] = {c0}{}",
            match expect {
                Some(e) => format!(" (expect {e})"),
                None => String::new(),
            }
        );
        if let Some(e) = expect {
            if c0 != e {
                eprintln!("correctness FAIL");
                std::process::exit(4);
            }
        }

        let nosync = [false, false, false];
        let run_one = |k: &NpuKernel| {
            if syncmode == 1 {
                k.dispatch(&[&aw, &ww, &cw]).expect("bench")
            } else {
                k.dispatch_synced(&[&aw, &ww, &cw], &nosync).expect("bench")
            }
        };

        for _ in 0..20 {
            run_one(&k);
        }

        let mut us: Vec<f64> = Vec::new();
        for _ in 0..passes {
            let t = Instant::now();
            for _ in 0..iters {
                run_one(&k);
            }
            us.push(t.elapsed().as_secs_f64() / iters as f64 * 1e6);
        }
        let mut s = us.clone();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = s[s.len() / 2];
        let spread = (s[s.len() - 1] - s[0]) / med * 100.0;
        let gbs = wread / (med * 1e-6) / 1e9;
        let each: Vec<String> = us.iter().map(|v| format!("{v:.1}")).collect();
        println!(
            "sync={syncmode} w_bo={wsz} w_read={wread:.0} passes=[{}] median={med:.1}us spread={spread:.1}% => {gbs:.2} GB/s",
            each.join(", ")
        );
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
