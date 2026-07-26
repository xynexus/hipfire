//! R129 batched staged-full-K weight-reuse parity, isolation, stability, and timing gate.
//! Usage: `npu_embedding_opus_staged_fullk_batched_verify R121_CACHE R129_CACHE [REUSED] [FRESH]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_primitives::conv::f32_to_f16;
    use hipfire_xdna::{NpuOpusExecutor, OpusMatrixEncoding};

    const DOCUMENT_ROWS: usize = 256;
    const K: usize = 768;
    const N: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;
    const BLOCK_BYTES: usize = 258;
    const WEIGHT_BYTES: usize = 7_987_200;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(2..=4).contains(&args.len()) {
        return Err("usage: npu_embedding_opus_staged_fullk_batched_verify R121_CACHE R129_CACHE [REUSED] [FRESH]".into());
    }
    let reused = args
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(10usize);
    let fresh = args
        .get(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(3usize);
    if reused < 10 || fresh == 0 {
        return Err("R129 gate requires REUSED>=10 and FRESH>=1".into());
    }
    let r121_cache = &args[0];
    let r129_cache = &args[1];
    let r129_manifest = std::fs::read_to_string(Path::new(r129_cache).join("shape.txt"))?;
    let batch = manifest_value(&r129_manifest, "batch")?;
    if !matches!(batch, 2 | 4) {
        return Err(format!("R129 verifier supports batch=2 or batch=4, got {batch}").into());
    }
    let m = batch * DOCUMENT_ROWS;
    let activation_bytes = batch * 589_824;
    let output_bytes = m * N * std::mem::size_of::<f32>();
    require_manifest(&r129_manifest, "m", m)?;
    require_manifest(&r129_manifest, "weight-bytes", WEIGHT_BYTES)?;
    require_manifest(&r129_manifest, "weight-dma-passes", 1)?;
    require_manifest(&r129_manifest, "weight-batch-replicas", 0)?;

    let mut payload = vec![0u8; N * GROUPS * BLOCK_BYTES];
    for col in 0..N {
        for group in 0..GROUPS {
            let offset = (col * GROUPS + group) * BLOCK_BYTES;
            let scale = 0.0029 + ((group * N + col) % 13) as f32 * 0.000_037;
            payload[offset..offset + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for inner in 0..GROUP {
                payload[offset + 2 + inner] =
                    ((((group * GROUP + inner) * 13 + col * 7) % 23) as i8 - 11) as u8;
            }
        }
    }
    assert_eq!(
        OpusMatrixEncoding::classify(35, payload.len(), K, N)?,
        OpusMatrixEncoding::W8
    );
    let awq_scale = (0..K)
        .map(|index| 0.91 + (index % 17) as f32 * 0.007)
        .collect::<Vec<_>>();
    let document = |document: usize, variant: usize| {
        (0..DOCUMENT_ROWS * K)
            .map(|index| {
                let mixed = index * (19 + 2 * document)
                    + index / (31 + document)
                    + document * 37
                    + variant * (index % 29 + 1);
                (mixed % 101) as f32 / 100.0 - 0.5
            })
            .collect::<Vec<_>>()
    };
    let documents = (0..batch)
        .map(|document_index| document(document_index, 0))
        .collect::<Vec<_>>();
    let combined = documents.concat();
    let mut alternate_documents = documents.clone();
    alternate_documents[batch - 1] = document(batch - 1, 7);
    let combined_alternate = alternate_documents.concat();

    let mut r121 = NpuOpusExecutor::load_staged_fullk_cached(&[r121_cache], N)?;
    let r121_artifact = Path::new(r121_cache).join("runtime-oq8.staged-fullk.rdna2.hfp");
    let r121_matrix =
        r121.pack_matrix_prepacked(35, K, N, &payload, Some(awq_scale.clone()), &r121_artifact)?;
    let mut references = Vec::with_capacity(batch);
    for (document_index, input) in documents.iter().enumerate() {
        r121.recreate_staged_fullk_context(&r121_matrix)?;
        let mut reference = vec![0.0f32; DOCUMENT_ROWS * N];
        r121.run_f32(&r121_matrix, DOCUMENT_ROWS, input, &mut reference)?;
        let cpu = r121.reference_f32(&r121_matrix, DOCUMENT_ROWS, input)?;
        require_close(
            &format!("R121 doc{document_index} absolute"),
            metrics(&cpu, &reference),
        )?;
        references.push(reference);
    }

    let mut r129 = NpuOpusExecutor::load_staged_fullk_cached(&[r129_cache], N)?;
    let r129_artifact = Path::new(r129_cache).join("runtime-oq8.staged-fullk.rdna2.hfp");
    let r129_matrix =
        r129.pack_matrix_prepacked(35, K, N, &payload, Some(awq_scale), &r129_artifact)?;
    r129.recreate_staged_fullk_context(&r129_matrix)?;
    let mut batched = vec![0.0f32; m * N];
    r129.run_f32(&r129_matrix, m, &combined, &mut batched)?;
    let mut batch_metrics = Vec::with_capacity(batch);
    for (document_index, reference) in references.iter().enumerate() {
        let range = document_range(document_index, N);
        let result = metrics(reference, &batched[range]);
        require_close(&format!("R129 doc{document_index} versus R121"), result)?;
        batch_metrics.push(result);
    }

    let mut alternate = vec![0.0f32; m * N];
    r129.run_f32(&r129_matrix, m, &combined_alternate, &mut alternate)?;
    let doc0_cross_mismatches = batched[..DOCUMENT_ROWS * N]
        .iter()
        .zip(&alternate[..DOCUMENT_ROWS * N])
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    if doc0_cross_mismatches != 0 {
        return Err(format!(
            "R129 cross-document contamination: doc0 changed in {doc0_cross_mismatches} values"
        )
        .into());
    }

    for fresh_index in 0..fresh {
        r129.recreate_staged_fullk_context(&r129_matrix)?;
        r129.run_f32(&r129_matrix, m, &combined, &mut batched)?;
        for (document_index, reference) in references.iter().enumerate() {
            require_close(
                &format!("R129 fresh-{fresh_index} doc{document_index}"),
                metrics(reference, &batched[document_range(document_index, N)]),
            )?;
        }
    }
    for reused_index in 0..reused {
        r129.run_f32(&r129_matrix, m, &combined, &mut batched)?;
        for (document_index, reference) in references.iter().enumerate() {
            require_close(
                &format!("R129 reused-{reused_index} doc{document_index}"),
                metrics(reference, &batched[document_range(document_index, N)]),
            )?;
        }
    }

    for _ in 0..2 {
        r121.recreate_staged_fullk_context(&r121_matrix)?;
        r121.run_f32(
            &r121_matrix,
            DOCUMENT_ROWS,
            &documents[0],
            &mut references[0],
        )?;
        r129.run_f32(&r129_matrix, m, &combined, &mut batched)?;
    }
    r121.recreate_staged_fullk_context(&r121_matrix)?;
    let timing_iterations = reused.max(20);
    let started = Instant::now();
    for _ in 0..timing_iterations {
        r121.run_f32(
            &r121_matrix,
            DOCUMENT_ROWS,
            &documents[0],
            &mut references[0],
        )?;
    }
    let r121_ms = started.elapsed().as_secs_f64() * 1e3 / timing_iterations as f64;
    let started = Instant::now();
    for _ in 0..timing_iterations {
        r129.run_f32(&r129_matrix, m, &combined, &mut batched)?;
    }
    let r129_command_ms = started.elapsed().as_secs_f64() * 1e3 / timing_iterations as f64;
    let throughput_gain = batch as f64 * r121_ms / r129_command_ms;
    let min_cosine = batch_metrics
        .iter()
        .map(|result| result.cosine)
        .fold(f64::INFINITY, f64::min);
    let max_abs = batch_metrics
        .iter()
        .map(|result| result.max_abs)
        .fold(0.0f32, f32::max);

    println!(
        "embedding-opus-staged-fullk-batched batch={batch} R121_M256_ms={r121_ms:.6} R129_M{m}_command_ms={r129_command_ms:.6} R129_ms_per_document={:.6} row_throughput_gain={throughput_gain:.4} min_cosine={min_cosine:.9} max_abs={max_abs:.9} doc0_cross_mismatches={doc0_cross_mismatches} fresh={fresh} reused={reused} weight_bytes={WEIGHT_BYTES} weight_dma_passes=1 weight_batch_replicas=0 activation_bytes={activation_bytes} output_bytes={output_bytes}",
        r129_command_ms / batch as f64,
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn document_range(document: usize, n: usize) -> std::ops::Range<usize> {
    let start = document * 256 * n;
    start..start + 256 * n
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug)]
struct Metrics {
    cosine: f64,
    max_abs: f32,
    mismatches: usize,
}

#[cfg(target_os = "linux")]
fn metrics(expected: &[f32], actual: &[f32]) -> Metrics {
    let mut dot = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut mismatches = 0usize;
    for (&want, &got) in expected.iter().zip(actual) {
        let error = (got - want).abs();
        max_abs = max_abs.max(error);
        let tolerance = 3.0e-5f32.max(want.abs() * 3.0e-6);
        if !got.is_finite() || error > tolerance {
            mismatches += 1;
        }
        dot += want as f64 * got as f64;
        expected_norm += (want as f64).powi(2);
        actual_norm += (got as f64).powi(2);
    }
    Metrics {
        cosine: dot / (expected_norm.sqrt() * actual_norm.sqrt()),
        max_abs,
        mismatches,
    }
}

#[cfg(target_os = "linux")]
fn require_close(label: &str, metrics: Metrics) -> Result<(), Box<dyn std::error::Error>> {
    if metrics.mismatches != 0 || metrics.cosine < 0.999 {
        return Err(format!("{label} failed: {metrics:?}").into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn require_manifest(
    manifest: &str,
    key: &str,
    expected: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = manifest_value(manifest, key)?;
    if actual != expected {
        return Err(format!("R129 manifest {key}={actual}, expected {expected}").into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn manifest_value(manifest: &str, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(manifest
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .ok_or_else(|| format!("R129 manifest missing {key}"))?
        .parse::<usize>()?)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P Opus staged full-K batched verification is Linux-only");
}
