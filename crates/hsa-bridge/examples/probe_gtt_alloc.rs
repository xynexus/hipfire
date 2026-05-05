use hip_bridge::HipRuntime;
use hsa_bridge::HsaRuntime;

fn mib(n: usize) -> f64 {
    n as f64 / (1024.0 * 1024.0)
}

fn main() {
    let chunk_gib: usize = std::env::var("CHUNK_GIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let chunk_mib: usize = std::env::var("CHUNK_MIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if chunk_gib == 0 { 178 } else { 0 });
    let max_gib: usize = std::env::var("MAX_GIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let chunk = if chunk_gib > 0 {
        chunk_gib * 1024 * 1024 * 1024
    } else {
        chunk_mib * 1024 * 1024
    };
    let max = max_gib * 1024 * 1024 * 1024;

    eprintln!("=== HIP hipMalloc probe ===");
    let hip = HipRuntime::load().expect("load HIP");
    hip.set_device(0).expect("set HIP device 0");
    match hip.get_vram_info() {
        Ok((free, total)) => {
            eprintln!(
                "hipMemGetInfo: free={:.1} MiB total={:.1} MiB",
                mib(free),
                mib(total)
            );
        }
        Err(e) => eprintln!("hipMemGetInfo failed: {e}"),
    }

    let mut hip_bufs = Vec::new();
    let mut total = 0usize;
    while total + chunk <= max {
        match hip.malloc(chunk) {
            Ok(buf) => {
                if let Err(e) = hip.memset(&buf, 0, buf.size()) {
                    eprintln!(
                        "hipMemset failed after {:.1} GiB reserved: {e}",
                        total as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                    let _ = hip.free(buf);
                    break;
                }
                total += chunk;
                eprintln!(
                    "hipMalloc+memset ok: {:.1} GiB total",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                hip_bufs.push(buf);
            }
            Err(e) => {
                eprintln!(
                    "hipMalloc failed after {:.1} GiB: {e}",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                break;
            }
        }
    }
    for buf in hip_bufs {
        let _ = hip.free(buf);
    }

    eprintln!("\n=== HSA GPU coarse pool probe ===");
    let hsa = HsaRuntime::load().expect("load HSA");
    let gpu = hsa.find_gpu_agent(Some("gfx1151")).expect("find gfx1151");
    let cpu = hsa.find_cpu_agent().expect("find CPU agent");
    let coarse = gpu
        .find_coarse_grained_pool()
        .expect("find GPU coarse pool");
    let mut hsa_ptrs = Vec::new();
    let mut total = 0usize;
    while total + chunk <= max {
        match coarse.allocate(chunk) {
            Ok(ptr) => {
                if let Err(e) = coarse.allow_access(&[&gpu, &cpu], ptr) {
                    eprintln!(
                        "hsa allow_access failed at {:.1} GiB: {e}",
                        total as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                    let _ = coarse.free(ptr);
                    break;
                }
                total += chunk;
                eprintln!(
                    "HSA coarse ok: {:.1} GiB total",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                hsa_ptrs.push(ptr);
            }
            Err(e) => {
                eprintln!(
                    "HSA coarse failed after {:.1} GiB: {e}",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                break;
            }
        }
    }
    for ptr in hsa_ptrs {
        let _ = coarse.free(ptr);
    }

    eprintln!("\n=== HSA GPU fine pool probe ===");
    let fine = gpu.find_fine_grained_pool().expect("find GPU fine pool");
    let mut hsa_ptrs = Vec::new();
    let mut total = 0usize;
    while total + chunk <= max {
        match fine.allocate(chunk) {
            Ok(ptr) => {
                if let Err(e) = fine.allow_access(&[&gpu, &cpu], ptr) {
                    eprintln!(
                        "hsa fine allow_access failed at {:.1} GiB: {e}",
                        total as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                    let _ = fine.free(ptr);
                    break;
                }
                total += chunk;
                eprintln!(
                    "HSA fine ok: {:.1} GiB total",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                hsa_ptrs.push(ptr);
            }
            Err(e) => {
                eprintln!(
                    "HSA fine failed after {:.1} GiB: {e}",
                    total as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                break;
            }
        }
    }
    for ptr in hsa_ptrs {
        let _ = fine.free(ptr);
    }
}
