//! Hardware gate for R118's activation-once, two-N32-block consumer.
//! Usage: `npu_embedding_r113_staged_fullk_n64_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const K: usize = 768;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;
    const N: usize = 64;
    const COLS: usize = 8;
    const N_BLOCKS: usize = 2;
    const SLOT: usize = 6_144;
    const JOIN: usize = 4 * SLOT;
    const A_BYTES: usize = 4 * 2 * GROUPS * JOIN;
    const W_RECORD: usize = 8_320;
    const W_BYTES: usize = COLS * N_BLOCKS * GROUPS * W_RECORD;
    const O_BYTES: usize = M * N * size_of::<f32>();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_r113_staged_fullk_n64_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(100usize);
    let cache = &args[0];
    let manifest = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    let repeat_output_task = manifest
        .lines()
        .any(|line| line == "op=embeddinggemma-r113-staged-fullk-n64-repeat-output");
    let expected_op = if repeat_output_task {
        "op=embeddinggemma-r113-staged-fullk-n64-repeat-output"
    } else {
        "op=embeddinggemma-r113-staged-fullk-n64"
    };
    for field in [
        expected_op,
        "mode=w8-scaled",
        "m=256",
        "k=768",
        "n=64",
        "activation-stage-bytes-per-core=6336",
        "activation-stage-group-stride=2112",
        "activation-stage-group-alignment=64",
        "activation-physical-bytes=589824",
        "activation-dma-passes=1",
        "n32-output-blocks=2",
        "nmacro-materialized-replicas=0",
        "immutable-tensor-reorder=none",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("R118 cache missing {field}").into());
        }
    }
    if repeat_output_task
        && !manifest
            .lines()
            .any(|line| line == "output-task-repeat-count=1")
    {
        return Err("R119 cache missing output-task-repeat-count=1".into());
    }

    let activations = (0..M * K)
        .map(|index| (((index * 17 + index / 29) % 31) as i8) - 15)
        .collect::<Vec<_>>();
    let activation_scales = (0..GROUPS * M)
        .map(|index| 0.0037 + (index % 19) as f32 * 0.000_041)
        .collect::<Vec<_>>();
    let weights = (0..K * N)
        .map(|index| (((index * 13 + index / 17) % 23) as i8) - 11)
        .collect::<Vec<_>>();
    let weight_scales = (0..GROUPS * N)
        .map(|index| 0.0029 + (index % 13) as f32 * 0.000_037)
        .collect::<Vec<_>>();

    let mut packed_a = vec![0u8; A_BYTES];
    for token in 0..M {
        let half = token / 128;
        let within_half = token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        for group in 0..GROUPS {
            let record = (core_row * 2 + half) * GROUPS + group;
            let base = record * JOIN + local_col * SLOT;
            for inner in 0..GROUP {
                let kt = inner / 8;
                let kk = inner % 8;
                packed_a[base + kt * 64 + local_row * 8 + kk] =
                    activations[token * K + group * GROUP + inner] as u8;
            }
            packed_a[base + 2_048 + local_row * 4..base + 2_052 + local_row * 4]
                .copy_from_slice(&activation_scales[group * M + token].to_le_bytes());
        }
    }

    let mut packed_w = vec![0u8; W_BYTES];
    for physical_col in 0..COLS {
        for n_block in 0..N_BLOCKS {
            for group in 0..GROUPS {
                let base = (physical_col * N_BLOCKS * GROUPS + n_block * GROUPS + group) * W_RECORD;
                for slice in 0..2 {
                    for kt in 0..32 {
                        for kk in 0..8 {
                            for local_col in 0..16 {
                                let col = n_block * 32 + slice * 16 + local_col;
                                let target = base
                                    + slice * 4_096
                                    + kt * 128
                                    + (local_col / 8) * 64
                                    + kk * 8
                                    + local_col % 8;
                                let source = (group * GROUP + kt * 8 + kk) * N + col;
                                packed_w[target] = weights[source] as u8;
                            }
                        }
                    }
                }
                for local_col in 0..32 {
                    let col = n_block * 32 + local_col;
                    packed_w[base + 8_192 + local_col * 4..base + 8_196 + local_col * 4]
                        .copy_from_slice(&weight_scales[group * N + col].to_le_bytes());
                }
            }
        }
    }

    let kernel = NpuKernel::load(
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?;
    let mut a = kernel.alloc_arg(A_BYTES)?;
    let mut w = kernel.alloc_arg(W_BYTES)?;
    let mut o = kernel.alloc_arg(O_BYTES)?;
    a.as_mut_slice().copy_from_slice(&packed_a);
    w.as_mut_slice().copy_from_slice(&packed_w);
    o.as_mut_slice().fill(0);
    kernel.dispatch_synced(&[&a, &w, &o], &[true, true, false])?;
    kernel.sync_output(&o)?;

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    let mut mismatches_by_block = [0usize; N_BLOCKS];
    for row in 0..M {
        for col in 0..N {
            let expected = (0..GROUPS)
                .map(|group| {
                    let dot = (0..GROUP)
                        .map(|inner| {
                            let inner_k = group * GROUP + inner;
                            activations[row * K + inner_k] as i32
                                * weights[inner_k * N + col] as i32
                        })
                        .sum::<i32>();
                    dot as f32 * activation_scales[group * M + row] * weight_scales[group * N + col]
                })
                .sum::<f32>();
            let offset = (row * N + col) * 4;
            let got = f32::from_le_bytes(o.as_slice()[offset..offset + 4].try_into()?);
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            let tolerance = 3.0e-5f32.max(expected.abs() * 3.0e-6);
            if !got.is_finite() || error > tolerance {
                mismatches += 1;
                mismatches_by_block[col / 32] += 1;
                first.get_or_insert((row, col, got, expected, error, tolerance));
            }
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "R118 parity failed: mismatches={mismatches} max_abs={max_abs:.9} first={first:?} by_block={mismatches_by_block:?}"
        )
        .into());
    }

    for _ in 0..2 {
        kernel.dispatch_synced(&[&a, &w, &o], &[false; 3])?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(&[&a, &w, &o], &[false; 3])?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-r113-staged-fullk-n64 M={M} K={K} N={N}: mismatches={mismatches} max_abs={max_abs:.9} dispatch_ms={dispatch_ms:.6} activation_physical_bytes=589824 activation_unique_bytes=199680 activation_dma_passes=1 nmacro_replicas=0"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P R118 verification is Linux-only");
}
