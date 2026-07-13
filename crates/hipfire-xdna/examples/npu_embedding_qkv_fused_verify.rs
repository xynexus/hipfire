//! R70 single-context W4 projection + headnorm/RoPE pack verifier.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;

    const SOURCE_ACTIVATION_BYTES: usize = 589_824;
    const ACTIVATION_BYTES: usize = 737_280;
    const WEIGHT_BYTES: usize = 2_359_296;
    const STAGE_BYTES: usize = 2_457_600;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(7..=9).contains(&args.len()) {
        return Err("usage: npu_embedding_qkv_fused_verify FUSED_CACHE R65_CACHE R66_CACHE SOURCE_ACTIVATIONS.bin PADDED_ACTIVATIONS.bin REFERENCE_WEIGHTS.bin STAGE_SEED.bin [ITERS] [FUSED_WEIGHTS.bin]".into());
    }
    let iterations = args
        .get(7)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    if iterations == 0 {
        return Err("R70 verifier needs at least one iteration".into());
    }
    let read = |path: &str, expected: usize| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(Path::new(path))?;
        if bytes.len() != expected {
            return Err(format!("{path} has {} bytes, expected {expected}", bytes.len()).into());
        }
        Ok(bytes)
    };
    let read_weight = |path: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if path.ends_with(".rdna2.hfp") {
            let raw = std::fs::read(Path::new(path))?;
            if raw.len() != 192 + WEIGHT_BYTES || &raw[..8] != b"HFOPHFP2" {
                return Err(format!("{path} is not an R81-sized Opus HFP").into());
            }
            Ok(raw[192..].to_vec())
        } else {
            read(path, WEIGHT_BYTES)
        }
    };
    let source_activations = read(&args[3], SOURCE_ACTIVATION_BYTES)?;
    let activations = read(&args[4], ACTIVATION_BYTES)?;
    let weights = read_weight(&args[5])?;
    let fused_weights = args
        .get(8)
        .map(|path| read_weight(path))
        .transpose()?
        .unwrap_or_else(|| weights.clone());
    let stage_seed = read(&args[6], STAGE_BYTES)?;

    let reference_stage = {
        let projection = load_kernel(&args[1])?;
        let mut activation = projection.alloc_arg(SOURCE_ACTIVATION_BYTES)?;
        let mut weight = projection.alloc_arg(WEIGHT_BYTES)?;
        let mut stage = projection.alloc_arg(STAGE_BYTES)?;
        activation
            .as_mut_slice()
            .copy_from_slice(&source_activations);
        weight.as_mut_slice().copy_from_slice(&weights);
        stage.as_mut_slice().copy_from_slice(&stage_seed);
        projection.dispatch_synced(&[&activation, &weight, &stage], &[true, true, true])?;
        projection.sync_output(&stage)?;
        stage.as_slice().to_vec()
    };

    let (reference_q, reference_kv) = {
        let pack = load_kernel(&args[2])?;
        let mut stage = pack.alloc_arg(STAGE_BYTES)?;
        let mut q = pack.alloc_arg(Layout::Q_BYTES)?;
        let mut kv = pack.alloc_arg(Layout::KV_BYTES)?;
        stage.as_mut_slice().copy_from_slice(&reference_stage);
        pack.dispatch_synced(&[&stage, &q, &kv], &[true, false, false])?;
        q.as_mut_slice().fill(0);
        kv.as_mut_slice().fill(0);
        pack.sync_to_device(&q)?;
        pack.sync_to_device(&kv)?;
        pack.dispatch_synced(&[&stage, &q, &kv], &[false, false, false])?;
        pack.sync_output(&q)?;
        pack.sync_output(&kv)?;
        (q.as_slice().to_vec(), kv.as_slice().to_vec())
    };

    let fused = load_kernel(&args[0])?;
    let mut activation = fused.alloc_arg(ACTIVATION_BYTES)?;
    let mut weight = fused.alloc_arg(WEIGHT_BYTES)?;
    let mut stage = fused.alloc_arg(STAGE_BYTES)?;
    let mut q = fused.alloc_arg(Layout::Q_BYTES)?;
    let mut kv = fused.alloc_arg(Layout::KV_BYTES)?;
    activation.as_mut_slice().copy_from_slice(&activations);
    weight.as_mut_slice().copy_from_slice(&fused_weights);
    stage.as_mut_slice().copy_from_slice(&stage_seed);

    fused.dispatch_synced(
        &[&activation, &weight, &stage, &q, &kv],
        &[true, true, true, false, false],
    )?;
    q.as_mut_slice().fill(0);
    kv.as_mut_slice().fill(0);
    fused.sync_to_device(&q)?;
    fused.sync_to_device(&kv)?;
    fused.dispatch_synced(
        &[&activation, &weight, &stage, &q, &kv],
        &[false, false, false, false, false],
    )?;
    fused.sync_output(&stage)?;
    fused.sync_output(&q)?;
    fused.sync_output(&kv)?;
    verify("stage", &reference_stage, stage.as_slice())?;
    verify("Q", &reference_q, q.as_slice())?;
    verify("KV", &reference_kv, kv.as_slice())?;

    for _ in 0..2 {
        fused.dispatch_synced(
            &[&activation, &weight, &stage, &q, &kv],
            &[false, false, false, false, false],
        )?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        fused.dispatch_synced(
            &[&activation, &weight, &stage, &q, &kv],
            &[false, false, false, false, false],
        )?;
    }
    let fused_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    fused.sync_output(&stage)?;
    fused.sync_output(&q)?;
    fused.sync_output(&kv)?;
    verify("timed stage", &reference_stage, stage.as_slice())?;
    verify("timed Q", &reference_q, q.as_slice())?;
    verify("timed KV", &reference_kv, kv.as_slice())?;

    println!(
        "r70-fused-qkv projection_stage_mismatches=0 q_mismatches=0 kv_mismatches=0 iterations={iterations} fused_ms={fused_ms:.4}"
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
fn verify(label: &str, expected: &[u8], actual: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mismatches = expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left != right)
        .count();
    if mismatches == 0 {
        return Ok(());
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap();
    Err(format!(
        "R70 {label} has {mismatches} byte mismatches; first offset {first}: expected={} actual={}",
        expected[first], actual[first]
    )
    .into())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_qkv_fused_verify is Linux-only");
}
