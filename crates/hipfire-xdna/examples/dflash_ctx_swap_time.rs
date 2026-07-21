//! DFlash native driver: what does a hardware-context swap cost?
//!
//! npu1 (Phoenix) admits only 6 concurrent hardware contexts (measured:
//! `dflash_manifest_load --hold` loads 6/14, the 7th CREATE_HWCTX returns
//! EINVAL — the same budget the Python harness's LRU-of-6 was built around).
//! The DFlash body uses 12 distinct kernels per layer, so the native driver
//! cannot keep them all resident and must evict/reload inside the block. This
//! measures what that costs, which decides whether the 57 ms budget survives.
//!
//! Two load paths are timed:
//!   * `NpuKernel::load`      — opens its own DRM file + 64 MiB device heap.
//!   * `NpuKernel::load_peer` — reuses an anchor kernel's file + heap, so only
//!     the hwctx + PDI/instruction upload is paid.
//!
//! Usage: `dflash_ctx_swap_time MANIFEST.json [ITERS]`   (hold the hipfire lock)

#[cfg(target_os = "linux")]
fn main() {
    use hipfire_xdna::NpuKernel;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dflash_ctx_swap_time MANIFEST.json [ITERS]");
        std::process::exit(2);
    }
    let iters: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20);

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[1]).expect("read manifest"))
            .expect("parse manifest");
    let kernels = manifest["kernels"].as_object().expect("kernels");

    // Load every artifact's bytes up front: the swap cost we care about is the
    // ioctl/PDI-upload path, not page-cache misses on the xclbin file.
    let arts: Vec<(String, Vec<u8>, Vec<u8>)> = kernels
        .iter()
        .map(|(n, s)| {
            (
                n.clone(),
                std::fs::read(s["xclbin"].as_str().unwrap()).expect("xclbin"),
                std::fs::read(s["insts"].as_str().unwrap()).expect("insts"),
            )
        })
        .collect();
    println!("artifacts: {}", arts.len());

    // Anchor stays resident so load_peer has a device + heap to share.
    let anchor = NpuKernel::load(&arts[0].1, &arts[0].2).expect("anchor load");

    for (label, peer) in [("load", false), ("load_peer", true)] {
        // Warm once so first-touch cost doesn't land in the mean.
        {
            let (_n, x, i) = &arts[1];
            let _ = if peer {
                NpuKernel::load_peer(&anchor, x, i)
            } else {
                NpuKernel::load(x, i)
            };
        }
        let mut total = std::time::Duration::ZERO;
        let mut worst = std::time::Duration::ZERO;
        for it in 0..iters {
            // Cycle through the artifacts so we exercise different PDI sizes,
            // and drop each kernel before the next load — that drop is the
            // eviction an LRU would perform.
            let (_n, x, i) = &arts[1 + (it % (arts.len() - 1))];
            let t0 = Instant::now();
            let k = if peer {
                NpuKernel::load_peer(&anchor, x, i).expect("peer load")
            } else {
                NpuKernel::load(x, i).expect("load")
            };
            let dt = t0.elapsed();
            drop(k);
            total += dt;
            worst = worst.max(dt);
        }
        println!(
            "  {:10} mean={:.0} us  worst={:.0} us  over {iters} loads",
            label,
            total.as_secs_f64() * 1e6 / iters as f64,
            worst.as_secs_f64() * 1e6
        );
    }

    // What the body actually pays: 12 distinct kernels per layer, 5 layers,
    // against a 6-context budget under LRU.
    println!(
        "\n  body access pattern: 12 distinct kernels/layer x 5 layers, budget 6\n  \
         => an LRU of 6 over a 12-kernel cycle misses EVERY access (cycle > capacity)"
    );
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
