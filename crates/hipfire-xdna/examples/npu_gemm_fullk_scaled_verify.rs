//! Verify the fused f32-scale full-K W4 path with resident weights.
//!
//! Usage: `npu_gemm_fullk_scaled_verify CACHE COLS`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::NpuGemmFullK;

    let mut args = std::env::args().skip(1);
    let cache = args.next().ok_or("CACHE")?;
    let cols: usize = args.next().ok_or("COLS")?.parse()?;
    let iterations: usize = args.next().map_or(Ok(10), |value| value.parse())?;
    let mut gemm = NpuGemmFullK::load_cached(&cache, cols)?;
    if !gemm.scaled_output() {
        return Err("cache is not scaled-f32".into());
    }
    let rows = gemm.rows();
    let groups = gemm.k() / 256;
    let n = gemm.n();
    let base = vec![vec![1i8; 256 * n]; groups];
    let scales = vec![vec![1.0f32; n]; groups];
    let base_refs: Vec<&[i8]> = base.iter().map(Vec::as_slice).collect();
    let scale_refs: Vec<&[f32]> = scales.iter().map(Vec::as_slice).collect();
    let packed = gemm.prepack_weights_with_scales(&base_refs, &[], &scale_refs)?;
    let resident = gemm.upload_resident_weights(&packed)?;
    let activations = vec![1i8; rows * gemm.k()];
    let activation_scales = vec![1.0f32; groups * rows];
    let mut output = vec![0.0f32; rows * n];
    gemm.run_resident_scaled(&resident, &activations, &activation_scales, &mut output)?;
    let expected = (groups * 256) as f32;
    let mismatches = output.iter().filter(|&&value| value != expected).count();
    println!(
        "scaled-fullk M={rows} K={} N={n}: mismatches={mismatches} first={:?} expected={expected}",
        gemm.k(),
        &output[..8.min(output.len())],
    );
    println!(
        "slab_heads={:?}",
        (0..n / 64)
            .map(|slab| output[slab * 64])
            .collect::<Vec<_>>()
    );
    if mismatches != 0 {
        return Err("scaled full-K parity failed".into());
    }
    for _ in 0..2 {
        gemm.run_resident_scaled(&resident, &activations, &activation_scales, &mut output)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        gemm.run_resident_scaled(&resident, &activations, &activation_scales, &mut output)?;
    }
    let seconds = started.elapsed().as_secs_f64() / iterations as f64;
    let macs = rows as f64 * gemm.k() as f64 * n as f64;
    println!(
        "iters={iterations} scaled_projection_ms={:.4} logical_tops={:.4}",
        seconds * 1e3,
        2.0 * macs / seconds / 1e12,
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {}
