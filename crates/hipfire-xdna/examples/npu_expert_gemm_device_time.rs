//! Clean **device-time** for a W4A8 op4++ expert GEMM via the resident-scaled
//! full-K path — prep once, dispatch many. Isolates the on-array dispatch from
//! the host activation-prep (AWQ/FWHT/int8-quant) and int32 partial readback that
//! dominate `run_f32`'s wall time, giving the real PP contribution per expert GEMM.
//!
//! `run_resident_scaled` reconstructs the scaled f32 output on-device (no partial
//! readback) and takes pre-quantised int8 activations, so the timed loop is
//! dispatch + cheap tile copies. Data is arbitrary-but-valid: the dispatch cost is
//! data-independent, so this measures device time, not numerics (see
//! `npu_expert_ffn_w4_parity` for correctness).
//!
//! Default cache is the Qwen3.5-A3B fused gate_up shape (K=2048, N=1536).
//! Hold `hipfire lock` while running.
//!
//! Usage: `npu_expert_gemm_device_time [CACHE] [--cols C] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::NpuGemmFullK;

    let home = std::env::var("HOME")?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let opt_usize = |key: &str| -> Option<usize> {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let cache = args
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .next()
        .cloned()
        .unwrap_or_else(|| {
            format!("{home}/.hipfire/npu/embgemma_aie2p_fullk_submit_w4-scaled_m256_kg8_n1536")
        });
    let cols = opt_usize("--cols").unwrap_or(8);
    let iterations = opt_usize("--iters").unwrap_or(200);

    let mut gemm = NpuGemmFullK::load_cached(&cache, cols)?;
    let (rows, k, n) = (gemm.rows(), gemm.k(), gemm.n());
    let groups = k / 256;

    // Arbitrary-but-valid resident weights: groups × [256, N] int8 + per-group
    // [N] f32 scales.
    let base_bufs: Vec<Vec<i8>> = (0..groups)
        .map(|g| {
            (0..256 * n)
                .map(|i| (((i + g * 7) % 15) as i8) - 7)
                .collect()
        })
        .collect();
    let scale_bufs: Vec<Vec<f32>> = (0..groups)
        .map(|g| {
            (0..n)
                .map(|c| 0.008 + ((c + g) % 17) as f32 * 0.0002)
                .collect()
        })
        .collect();
    let base_refs: Vec<&[i8]> = base_bufs.iter().map(Vec::as_slice).collect();
    let scale_refs: Vec<&[f32]> = scale_bufs.iter().map(Vec::as_slice).collect();
    let packed = gemm.prepack_weights_with_scales(&base_refs, &[], &scale_refs)?;
    let weights = gemm.upload_resident_weights(&packed)?;

    // Pre-quantised int8 activations [rows*k] + per-group per-row scales.
    let acts: Vec<i8> = (0..rows * k)
        .map(|i| (((i * 13) % 255) as i16 - 127) as i8)
        .collect();
    let act_scales: Vec<f32> = (0..groups * rows)
        .map(|i| 0.01 + (i % 29) as f32 * 0.0001)
        .collect();
    let mut out = vec![0.0f32; rows * n];

    // Warm up (context bring-up, first-dispatch cold cost).
    for _ in 0..5 {
        gemm.run_resident_scaled(&weights, &acts, &act_scales, &mut out)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        gemm.run_resident_scaled(&weights, &acts, &act_scales, &mut out)?;
    }
    let us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
    let finite = out.iter().all(|v| v.is_finite());
    let macs = rows as f64 * k as f64 * n as f64;
    println!(
        "npu_expert_gemm_device_time M={rows} K={k} N={n} (W4A8, 1 dispatch): device_us={us:.2} \
         GMAC/s={:.1} finite={finite} iters={iterations}",
        macs / (us * 1e-6) / 1e9
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("resident AIE2P device-time measurement is Linux-only");
}
