//! Hardware gate for R116's full-K direct R113 compact consumer.
//! Usage: `npu_embedding_r113_compact_fullk_n16_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const GROUP: usize = 256;
    const FULL_GROUPS: usize = 3;
    const N: usize = 16;
    const COLS: usize = 8;
    const SLOT: usize = 6_144;
    const JOIN: usize = 4 * SLOT;
    const A_BYTES: usize = 4 * 2 * FULL_GROUPS * JOIN;
    const W_RECORD: usize = 4_160;
    const O_BYTES: usize = M * N * size_of::<f32>();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_r113_compact_fullk_n16_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(100usize);
    let unit_scales = std::env::var_os("HIPFIRE_R116_UNIT_SCALES").is_some();
    let cache = &args[0];
    let manifest = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    let groups = manifest
        .lines()
        .find_map(|line| line.strip_prefix("activation-groups="))
        .ok_or("R116 cache missing activation-groups")?
        .parse::<usize>()?;
    if !(2..=FULL_GROUPS).contains(&groups) {
        return Err(format!("R116 diagnostic group count must be 2 or 3, got {groups}").into());
    }
    let total_k = groups * GROUP;
    let w_bytes = COLS * groups * W_RECORD;
    for field in [
        "op=embeddinggemma-r113-compact-fullk-n16",
        "mode=w8-scaled",
        "m=256",
        "n=16",
        "activation-input=r113-per-core-diagnostic-slots",
        "accumulation=local-f32",
        "nmacro-materialized-replicas=0",
        "immutable-tensor-reorder=none",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("R116 cache missing {field}").into());
        }
    }
    if !manifest.lines().any(|line| line == format!("k={total_k}")) {
        return Err(format!("R116 cache missing k={total_k}").into());
    }

    let activations = (0..M * total_k)
        .map(|index| (((index * 17 + index / 29) % 31) as i8) - 15)
        .collect::<Vec<_>>();
    let activation_scales = (0..groups * M)
        .map(|index| {
            if unit_scales {
                1.0
            } else {
                0.0037 + (index % 19) as f32 * 0.000_041
            }
        })
        .collect::<Vec<_>>();
    let weights = (0..total_k * N)
        .map(|index| (((index * 13 + index / 17) % 23) as i8) - 11)
        .collect::<Vec<_>>();
    let weight_scales = (0..groups * N)
        .map(|index| {
            if unit_scales {
                1.0
            } else {
                0.0029 + (index % 13) as f32 * 0.000_037
            }
        })
        .collect::<Vec<_>>();

    let mut packed_a = vec![0u8; A_BYTES];
    for token in 0..M {
        let half = token / 128;
        let within_half = token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        for group in 0..groups {
            let record = (core_row * 2 + half) * FULL_GROUPS + group;
            let base = record * JOIN + local_col * SLOT;
            for inner in 0..GROUP {
                let kt = inner / 8;
                let kk = inner % 8;
                packed_a[base + kt * 64 + local_row * 8 + kk] =
                    activations[token * total_k + group * GROUP + inner] as u8;
            }
            packed_a[base + 2_048 + local_row * 4..base + 2_052 + local_row * 4]
                .copy_from_slice(&activation_scales[group * M + token].to_le_bytes());
        }
    }

    let mut packed_w = vec![0u8; w_bytes];
    for physical_col in 0..COLS {
        for group in 0..groups {
            let base = (physical_col * groups + group) * W_RECORD;
            for kt in 0..32 {
                for kk in 0..8 {
                    for col in 0..N {
                        let target = base + kt * 128 + (col / 8) * 64 + kk * 8 + col % 8;
                        let source = (group * GROUP + kt * 8 + kk) * N + col;
                        packed_w[target] = weights[source] as u8;
                    }
                }
            }
            for col in 0..N {
                packed_w[base + 4_096 + col * 4..base + 4_100 + col * 4]
                    .copy_from_slice(&weight_scales[group * N + col].to_le_bytes());
            }
        }
    }

    let kernel = NpuKernel::load(
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?;
    let mut a = kernel.alloc_arg(A_BYTES)?;
    let mut w = kernel.alloc_arg(w_bytes)?;
    let mut o = kernel.alloc_arg(O_BYTES)?;
    a.as_mut_slice().copy_from_slice(&packed_a);
    w.as_mut_slice().copy_from_slice(&packed_w);
    o.as_mut_slice().fill(0);
    kernel.dispatch_synced(&[&a, &w, &o], &[true, true, false])?;
    kernel.sync_output(&o)?;

    let mut mismatches = 0usize;
    let mut mismatches_by_col = [0usize; N];
    let mut max_abs = 0.0f32;
    let mut first = None;
    for row in 0..M {
        for col in 0..N {
            let expected = (0..groups)
                .map(|group| {
                    let dot = (0..GROUP)
                        .map(|inner| {
                            let inner_k = group * GROUP + inner;
                            activations[row * total_k + inner_k] as i32
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
                mismatches_by_col[col] += 1;
                first.get_or_insert((row, col, got, expected, error, tolerance));
            }
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "R116 parity failed: mismatches={mismatches} max_abs={max_abs:.9} first={first:?} by_col={mismatches_by_col:?}"
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
        "embedding-r113-compact-fullk-n16 M={M} K={total_k} N={N}: mismatches={mismatches} max_abs={max_abs:.9} dispatch_ms={dispatch_ms:.6} activation_physical_bytes={} activation_unique_bytes={} nmacro_replicas=0",
        196_608 * groups,
        66_560 * groups,
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P R116 verification is Linux-only");
}
