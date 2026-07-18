//! DFlash native driver, step 1: verify every kernel in the Python harness's
//! artifact manifest loads through `NpuKernel::load`.
//!
//! The manifest (from `dflash_body_npu.py --dump-manifest`) is the contract
//! between the Python reference and the native driver: it resolves the
//! hash-keyed `@iron.jit` cache paths (`~/.npu/cache/<hash>/`) that the native
//! side cannot compute, alongside the plain `target/npu/*.xclbin` primitives.
//!
//! Usage: `dflash_manifest_load MANIFEST.json`   (hold the hipfire lock)

#[cfg(target_os = "linux")]
fn main() {
    use hipfire_xdna::NpuKernel;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dflash_manifest_load MANIFEST.json");
        std::process::exit(2);
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[1]).expect("read manifest"))
            .expect("parse manifest");
    let kernels = manifest["kernels"].as_object().expect("kernels object");

    let mut ok = 0usize;
    let mut failed = Vec::new();
    for (name, spec) in kernels {
        let xclbin_path = spec["xclbin"].as_str().expect("xclbin path");
        let insts_path = spec["insts"].as_str().expect("insts path");
        let xclbin = match std::fs::read(xclbin_path) {
            Ok(b) => b,
            Err(e) => {
                failed.push(format!("{name}: xclbin unreadable: {e}"));
                continue;
            }
        };
        let insts = match std::fs::read(insts_path) {
            Ok(b) => b,
            Err(e) => {
                failed.push(format!("{name}: insts unreadable: {e}"));
                continue;
            }
        };
        // Load and drop immediately: npu1 (Phoenix) can only keep a handful of
        // hardware contexts resident, so holding all 14 at once would exhaust
        // the budget. This step only proves each artifact is loadable.
        match NpuKernel::load(&xclbin, &insts) {
            Ok(_k) => {
                ok += 1;
                println!("  [OK]   {name}  (xclbin {} B, insts {} B)", xclbin.len(), insts.len());
            }
            Err(e) => failed.push(format!("{name}: NpuKernel::load failed: {e:?}")),
        }
    }

    println!("\ndflash_manifest_load: {ok}/{} kernels loaded", kernels.len());
    for f in &failed {
        println!("  [FAIL] {f}");
    }
    if !failed.is_empty() {
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
