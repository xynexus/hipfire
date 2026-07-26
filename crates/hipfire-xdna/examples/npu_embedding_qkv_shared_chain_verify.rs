//! Zero-copy R68 projection -> pack chain using one XDNA SHMEM BO in two NPU
//! contexts that share a DRM file-description and GEM-handle namespace.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;

    const ACTIVATION_BYTES: usize = 589_824;
    const WEIGHT_BYTES: usize = 2_359_296;
    const STAGE_BYTES: usize = 1_517_568;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(5..=6).contains(&args.len()) {
        return Err("usage: npu_embedding_qkv_shared_chain_verify PROJECTION_CACHE PACK_CACHE ACTIVATIONS.bin WEIGHTS.bin STAGE_SEED.bin [ITERS]".into());
    }
    let iterations = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    if iterations == 0 {
        return Err("shared-chain verifier needs at least one iteration".into());
    }
    let read = |path: &str, expected: usize| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(Path::new(path))?;
        if bytes.len() != expected {
            return Err(format!("{path} has {} bytes, expected {expected}", bytes.len()).into());
        }
        Ok(bytes)
    };
    let activations = read(&args[2], ACTIVATION_BYTES)?;
    let weights = read(&args[3], WEIGHT_BYTES)?;
    let stage_seed = read(&args[4], STAGE_BYTES)?;

    let projection = load_kernel(&args[0])?;
    let pack = load_peer_kernel(&projection, &args[1])?;
    let mut activation_bo = projection.alloc_arg(ACTIVATION_BYTES)?;
    let mut weight_bo = projection.alloc_arg(WEIGHT_BYTES)?;
    activation_bo.as_mut_slice().copy_from_slice(&activations);
    weight_bo.as_mut_slice().copy_from_slice(&weights);

    let mut projection_stage = projection.alloc_arg(STAGE_BYTES)?;
    projection_stage.as_mut_slice().copy_from_slice(&stage_seed);

    let mut control_stage = pack.alloc_arg(STAGE_BYTES)?;
    let control_q = pack.alloc_arg(Layout::Q_BYTES)?;
    let control_kv = pack.alloc_arg(Layout::KV_BYTES)?;
    let chained_q = pack.alloc_arg(Layout::Q_BYTES)?;
    let chained_kv = pack.alloc_arg(Layout::KV_BYTES)?;

    // Materialize a host-copy control from a completed projection.
    projection.dispatch_synced(
        &[&activation_bo, &weight_bo, &projection_stage],
        &[true, true, true],
    )?;
    projection.sync_output(&projection_stage)?;
    control_stage
        .as_mut_slice()
        .copy_from_slice(projection_stage.as_slice());
    pack.dispatch_synced(
        &[&control_stage, &control_q, &control_kv],
        &[true, false, false],
    )?;
    pack.sync_output(&control_q)?;
    pack.sync_output(&control_kv)?;

    // Run the real cross-context no-copy path. Projection rewrites only the
    // mutable value regions; the seed's cos/sin and norm tails stay resident.
    projection.dispatch_synced(
        &[&activation_bo, &weight_bo, &projection_stage],
        &[false, false, false],
    )?;
    projection.sync_output(&projection_stage)?;
    pack.dispatch_synced(
        &[&projection_stage, &chained_q, &chained_kv],
        &[false, false, false],
    )?;
    pack.sync_output(&chained_q)?;
    pack.sync_output(&chained_kv)?;
    let q_mismatches = byte_mismatches(control_q.as_slice(), chained_q.as_slice());
    let kv_mismatches = byte_mismatches(control_kv.as_slice(), chained_kv.as_slice());
    if q_mismatches != 0 || kv_mismatches != 0 {
        return Err(format!(
            "shared chain differs from host-copy control: q={q_mismatches} kv={kv_mismatches}"
        )
        .into());
    }

    for _ in 0..2 {
        projection.dispatch_synced(
            &[&activation_bo, &weight_bo, &projection_stage],
            &[false, false, false],
        )?;
        projection.sync_output(&projection_stage)?;
        pack.dispatch_synced(
            &[&projection_stage, &chained_q, &chained_kv],
            &[false, false, false],
        )?;
    }
    let started = Instant::now();
    let mut projection_ns = 0u128;
    let mut handoff_sync_ns = 0u128;
    let mut pack_ns = 0u128;
    for _ in 0..iterations {
        let phase = Instant::now();
        projection.dispatch_synced(
            &[&activation_bo, &weight_bo, &projection_stage],
            &[false, false, false],
        )?;
        projection_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        projection.sync_output(&projection_stage)?;
        handoff_sync_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        pack.dispatch_synced(
            &[&projection_stage, &chained_q, &chained_kv],
            &[false, false, false],
        )?;
        pack_ns += phase.elapsed().as_nanos();
    }
    let chain_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let projection_ms = projection_ns as f64 * 1e-6 / iterations as f64;
    let handoff_sync_ms = handoff_sync_ns as f64 * 1e-6 / iterations as f64;
    let pack_ms = pack_ns as f64 * 1e-6 / iterations as f64;
    pack.sync_output(&chained_q)?;
    pack.sync_output(&chained_kv)?;
    let final_q_mismatches = byte_mismatches(control_q.as_slice(), chained_q.as_slice());
    let final_kv_mismatches = byte_mismatches(control_kv.as_slice(), chained_kv.as_slice());
    if final_q_mismatches != 0 || final_kv_mismatches != 0 {
        return Err(format!(
            "timed shared chain differs from host-copy control: q={final_q_mismatches} kv={final_kv_mismatches}"
        )
        .into());
    }
    println!(
        "r69-shared-drmfile-qkv-chain q_mismatches={q_mismatches} kv_mismatches={kv_mismatches} final_q_mismatches={final_q_mismatches} final_kv_mismatches={final_kv_mismatches} stage_bytes={STAGE_BYTES} iterations={iterations} projection_ms={projection_ms:.4} handoff_sync_ms={handoff_sync_ms:.4} pack_ms={pack_ms:.4} chain_ms={chain_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn load_kernel(cache: &str) -> Result<hipfire_xdna::NpuKernel, Box<dyn std::error::Error>> {
    Ok(hipfire_xdna::NpuKernel::load(
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?)
}

#[cfg(target_os = "linux")]
fn load_peer_kernel(
    peer: &hipfire_xdna::NpuKernel,
    cache: &str,
) -> Result<hipfire_xdna::NpuKernel, Box<dyn std::error::Error>> {
    Ok(hipfire_xdna::NpuKernel::load_peer(
        peer,
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?)
}

#[cfg(target_os = "linux")]
fn byte_mismatches(left: &[u8], right: &[u8]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a != b).count()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_qkv_shared_chain_verify is Linux-only");
}
