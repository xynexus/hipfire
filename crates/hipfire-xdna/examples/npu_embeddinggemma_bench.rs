//! embeddinggemma-300m NPU efficiency bench.
//!
//! Runs the r14 W4A8 GEMM (int8 act × int4 weight → int32) on the Phoenix NPU,
//! samples NPU power via `XdnaDevice::sensors()`, and reports the metrics that
//! decide the bulk-embedding question: GMAC/s, watts, tokens/s, and tokens/joule.
//!
//! embeddinggemma-300m does ~0.11 GMAC of matmul per token (≈95% of its compute:
//! 101M layer projections + 4.7M Dense + ~9M attention), so
//! `tok/s = GMAC_per_s / 0.11` and `tok/J = tok/s / watts`. This is the
//! **GEMM-ceiling upper bound** — it does not count attention, norms, or the
//! ~78 µs inter-op dispatch floor a full NPU forward would pay.
//!
//! Run: build the r14 xclbin (benchmarks/npu_gemm_tuning/r14/r14_cache.sh), then
//!   cargo run --release -p hipfire-xdna --example npu_embeddinggemma_bench -- \
//!     <cache-dir> <asz> <wsz> <csz> <macs-per-dispatch> [iters] [expect_c0]

/// embeddinggemma-300m matmul MACs per token (see module doc).
const MAC_PER_TOKEN: f64 = 0.11e9;

/// Find the amdgpu hwmon `power1_average` (SoC "Socket Graphics Package Power" in
/// µW). On this Phoenix APU the NPU and iGPU share this rail, so it is the fair
/// package-level energy meter for an NPU-vs-GPU tok/joule comparison. (The NPU's
/// dedicated sensor via `amd_pmf` is not loaded on this box.)
#[cfg(target_os = "linux")]
fn amdgpu_power_path() -> Option<std::path::PathBuf> {
    for e in std::fs::read_dir("/sys/class/hwmon").ok()?.flatten() {
        let p = e.path();
        let is_amdgpu = std::fs::read_to_string(p.join("name"))
            .map(|s| s.trim() == "amdgpu")
            .unwrap_or(false);
        if is_amdgpu {
            let pw = p.join("power1_average");
            if pw.exists() {
                return Some(pw);
            }
        }
    }
    None
}

/// Read the package power rail in watts.
#[cfg(target_os = "linux")]
fn read_watts(path: &std::path::Path) -> Option<f64> {
    let uw: f64 = std::fs::read_to_string(path).ok()?.trim().parse().ok()?;
    Some(uw / 1e6)
}

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::{NpuKernel, XdnaDevice};
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        if a.len() < 6 {
            eprintln!(
                "usage: npu_embeddinggemma_bench <dir> <asz> <wsz> <csz> <macs> [iters] [expect_c0]"
            );
            std::process::exit(2);
        }
        let dir = &a[1];
        let asz: usize = a[2].parse().unwrap();
        let wsz: usize = a[3].parse().unwrap();
        let csz: usize = a[4].parse().unwrap();
        let macs: f64 = a[5].parse().unwrap();
        let iters: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(200);
        let expect: Option<i32> = a.get(7).and_then(|s| s.parse().ok());

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");

        let mut aw = k.alloc_arg(asz).expect("A");
        let mut ww = k.alloc_arg(wsz).expect("W");
        let mut cw = k.alloc_arg(csz).expect("C");
        aw.as_mut_slice().fill(1);
        ww.as_mut_slice().fill(0x11);
        cw.as_mut_slice().fill(0);

        // Correctness gate (all-ones W4A8 compute).
        k.dispatch(&[&aw, &ww, &cw]).expect("dispatch");
        let c0 = unsafe { *(cw.as_slice().as_ptr() as *const i32) };
        println!(
            "all-ones C[0] = {c0}{}",
            expect.map(|e| format!(" (expect {e})")).unwrap_or_default()
        );
        if let Some(e) = expect {
            assert_eq!(c0, e, "correctness FAIL");
        }

        let _ = XdnaDevice::open_default(); // touch the crate (telemetry path unused: amd_pmf absent)
        let pw_path = amdgpu_power_path();
        if pw_path.is_none() {
            eprintln!("warning: amdgpu power1_average not found; power will be N/A");
        }

        // Idle baseline (GPU + NPU quiescent) so we can report dynamic (delta) power.
        let mut idle_w = f64::NAN;
        if let Some(p) = &pw_path {
            let mut s = 0.0;
            let mut n = 0.0;
            for _ in 0..15 {
                if let Some(w) = read_watts(p) {
                    s += w;
                    n += 1.0;
                }
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            if n > 0.0 {
                idle_w = s / n;
            }
        }

        for _ in 0..20 {
            k.dispatch(&[&aw, &ww, &cw]).expect("warmup");
        }

        // Timed loop; sample package power once per dispatch (~8 ms each → plenty).
        let mut pw_sum: f64 = 0.0;
        let mut pw_n: f64 = 0.0;
        let mut pw_peak: f64 = 0.0;
        let t = Instant::now();
        for _ in 0..iters {
            k.dispatch(&[&aw, &ww, &cw]).expect("bench");
            if let Some(p) = &pw_path {
                if let Some(w) = read_watts(p) {
                    pw_sum += w;
                    pw_n += 1.0;
                    pw_peak = pw_peak.max(w);
                }
            }
        }
        let dt = t.elapsed().as_secs_f64();
        let per = dt / iters as f64;

        let mac_per_s = macs / per;
        let gmac_s = mac_per_s / 1e9;
        let tops = 2.0 * mac_per_s / 1e12; // MAC = 2 ops
        let tok_s = mac_per_s / MAC_PER_TOKEN;

        let pkg_w = if pw_n > 0.0 { pw_sum / pw_n } else { f64::NAN };
        let dyn_w = pkg_w - idle_w;

        println!(
            "iters={iters} per_dispatch={:.1}us  {gmac_s:.1} GMAC/s  ({tops:.2} TOPS)",
            per * 1e6
        );
        println!(
            "SoC package power: idle={idle_w:.2} W  active={pkg_w:.2} W  peak={pw_peak:.2} W  \
             (NPU-dynamic ≈ {dyn_w:.2} W)"
        );
        println!(
            "embeddinggemma-300m @ {:.2} GMAC/token (GEMM-ceiling upper bound):",
            MAC_PER_TOKEN / 1e9
        );
        println!("  throughput      = {tok_s:.0} tok/s");
        println!("  efficiency (pkg)= {:.0} tok/joule  (active package power)", tok_s / pkg_w);
        println!("  efficiency (dyn)= {:.0} tok/joule  (idle-subtracted NPU power)", tok_s / dyn_w);
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
