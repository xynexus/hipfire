// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Device vs host repack of routed-expert modules, byte for byte.
//!
//! The weight pager now builds a module's per-expert device form ON THE GPU
//! (`kernels/src/pager_repack.hip`) instead of on the host. This samples
//! modules evenly across an artifact, repacks each both ways, and requires every
//! tensor byte to match — the device form is read by the same GEMV/GEMM kernels
//! as before, so "close" is not a pass.
//!
//!     pager_repack_parity <model.hfq> [n_modules]
//!
//! Exit 1 on any differing byte, or if no module was eligible (a vacuous pass).

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_pager::device_repack_parity;
use std::os::unix::fs::FileExt;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: pager_repack_parity <model.hfq> [n_modules]");
    let n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);
    let hfq = HfqFile::open(Path::new(&path)).expect("open artifact");
    let mods = hfq.modules();
    if mods.is_empty() {
        eprintln!("pager_repack_parity: {path} has no module table");
        std::process::exit(1);
    }
    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("pager_repack_parity: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let file = std::fs::File::open(&path).expect("open for reads");
    let step = (mods.len() / n.max(1)).max(1);
    let (mut eligible, mut declined, mut bytes, mut differ) = (0usize, 0usize, 0usize, 0usize);
    for module in mods.iter().step_by(step).take(n) {
        let mut disk = vec![0u8; module.data_size];
        file.read_exact_at(&mut disk, module.data_offset as u64)
            .expect("read module");
        match device_repack_parity(&mut gpu, module, &disk) {
            Ok(Some((c, d))) => {
                eligible += 1;
                bytes += c;
                differ += d;
                if d > 0 {
                    eprintln!("  {}: {d} of {c} bytes differ", module.module_id);
                }
            }
            Ok(None) => declined += 1,
            Err(e) => {
                eprintln!("pager_repack_parity: {}: {e}", module.module_id);
                std::process::exit(1);
            }
        }
    }
    println!(
        "pager_repack_parity: {eligible} modules device-repacked ({declined} declined -> host path), \
         {bytes} bytes compared, {differ} differ"
    );
    if eligible == 0 {
        eprintln!("pager_repack_parity: no module took the device path — nothing was tested");
        std::process::exit(1);
    }
    if differ > 0 {
        std::process::exit(1);
    }
    println!("pager_repack_parity: OK");
}
