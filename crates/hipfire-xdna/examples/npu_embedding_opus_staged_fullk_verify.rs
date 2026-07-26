//! End-to-end generic Opus OQ8 gate for the R121 staged full-K runtime.
//! Usage: `npu_embedding_opus_staged_fullk_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_primitives::conv::f32_to_f16;
    use hipfire_xdna::{NpuOpusExecutor, OpusMatrixEncoding};

    const M: usize = 256;
    const K: usize = 768;
    const N: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;
    const BLOCK_BYTES: usize = 258;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_opus_staged_fullk_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(20usize);
    let cache = &args[0];
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
    let x = (0..M * K)
        .map(|index| ((index * 19 + index / 31) % 101) as f32 / 100.0 - 0.5)
        .collect::<Vec<_>>();

    let mut executor = NpuOpusExecutor::load_staged_fullk_cached(&[cache], N)?;
    let artifact = Path::new(cache).join("runtime-oq8.staged-fullk.rdna2.hfp");
    let matrix = executor.pack_matrix_prepacked(35, K, N, &payload, Some(awq_scale), &artifact)?;
    executor.recreate_staged_fullk_context(&matrix)?;
    let mut output = vec![0.0f32; M * N];
    executor.run_f32(&matrix, M, &x, &mut output)?;
    let expected = executor.reference_f32(&matrix, M, &x)?;

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    let mut mismatches_by_block = vec![0usize; N / 32];
    let mut nonzero_by_block = vec![0usize; N / 32];
    let mut mismatches_by_half = [0usize; 2];
    let mut nonzero_by_half = [0usize; 2];
    for (index, (&got, &want)) in output.iter().zip(&expected).enumerate() {
        let col = index % N;
        let row = index / N;
        if got != 0.0 {
            nonzero_by_block[col / 32] += 1;
            nonzero_by_half[row / 128] += 1;
        }
        let error = (got - want).abs();
        max_abs = max_abs.max(error);
        let tolerance = 3.0e-5f32.max(want.abs() * 3.0e-6);
        if !got.is_finite() || error > tolerance {
            mismatches += 1;
            mismatches_by_block[col / 32] += 1;
            mismatches_by_half[row / 128] += 1;
            first.get_or_insert((index / N, index % N, got, want, error, tolerance));
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "Opus staged full-K parity failed: mismatches={mismatches} max_abs={max_abs:.9} first={first:?} by_block={mismatches_by_block:?} nonzero_by_block={nonzero_by_block:?} by_half={mismatches_by_half:?} nonzero_by_half={nonzero_by_half:?}"
        )
        .into());
    }

    for _ in 0..2 {
        executor.run_f32(&matrix, M, &x, &mut output)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        executor.run_f32(&matrix, M, &x, &mut output)?;
    }
    let runtime_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-opus-staged-fullk encoding=oq8 awq=true M={M} K={K} N={N}: mismatches={mismatches} max_abs={max_abs:.9} runtime_ms={runtime_ms:.6} artifact={} activation_dma_passes=1 nmacro_replicas=0",
        artifact.display()
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P Opus staged full-K verification is Linux-only");
}
