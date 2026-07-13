//! Hardware gate for R115's direct R113 compact-chunk consumer.
//! Usage: `npu_embedding_r113_compact_group_n16_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const K: usize = 256;
    const N: usize = 16;
    const COLS: usize = 8;
    const GROUPS: usize = 3;
    const SLOT: usize = 6_144;
    const JOIN: usize = 4 * SLOT;
    const A_BYTES: usize = 4 * 2 * GROUPS * JOIN;
    const W_RECORD: usize = 4_160;
    const W_BYTES: usize = COLS * W_RECORD;
    const O_BYTES: usize = M * N * size_of::<f32>();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_r113_compact_group_n16_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(100usize);
    let cache = &args[0];
    let manifest = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    for field in [
        "op=embeddinggemma-r113-compact-group-n16",
        "mode=w8-scaled",
        "m=256",
        "k=256",
        "n=16",
        "activation-input=r113-per-core-diagnostic-slots",
        "nmacro-materialized-replicas=0",
        "immutable-tensor-reorder=none",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("R115 cache missing {field}").into());
        }
    }

    let activations = (0..M * K)
        .map(|index| (((index * 17 + index / 29) % 31) as i8) - 15)
        .collect::<Vec<_>>();
    let activation_scales = (0..M)
        .map(|row| 0.0037 + (row % 19) as f32 * 0.000_041)
        .collect::<Vec<_>>();
    let weights = (0..K * N)
        .map(|index| (((index * 13 + index / 17) % 23) as i8) - 11)
        .collect::<Vec<_>>();
    let weight_scales = (0..N)
        .map(|col| 0.0029 + (col % 13) as f32 * 0.000_037)
        .collect::<Vec<_>>();

    let mut packed_a = vec![0u8; A_BYTES];
    for token in 0..M {
        let half = token / 128;
        let within_half = token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        let record = (core_row * 2 + half) * GROUPS;
        let base = record * JOIN + local_col * SLOT;
        for inner in 0..K {
            let kt = inner / 8;
            let kk = inner % 8;
            packed_a[base + kt * 64 + local_row * 8 + kk] = activations[token * K + inner] as u8;
        }
        packed_a[base + 2_048 + local_row * 4..base + 2_052 + local_row * 4]
            .copy_from_slice(&activation_scales[token].to_le_bytes());
    }

    let mut weight_record = vec![0u8; W_RECORD];
    for kt in 0..32 {
        for kk in 0..8 {
            for col in 0..N {
                let index = kt * 128 + (col / 8) * 64 + kk * 8 + col % 8;
                weight_record[index] = weights[(kt * 8 + kk) * N + col] as u8;
            }
        }
    }
    for (col, scale) in weight_scales.iter().copied().enumerate() {
        weight_record[4_096 + col * 4..4_100 + col * 4].copy_from_slice(&scale.to_le_bytes());
    }
    let mut packed_w = vec![0u8; W_BYTES];
    for col in 0..COLS {
        packed_w[col * W_RECORD..(col + 1) * W_RECORD].copy_from_slice(&weight_record);
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
    for row in 0..M {
        for col in 0..N {
            let dot = (0..K)
                .map(|inner| activations[row * K + inner] as i32 * weights[inner * N + col] as i32)
                .sum::<i32>();
            let expected = dot as f32 * activation_scales[row] * weight_scales[col];
            let offset = (row * N + col) * 4;
            let got = f32::from_le_bytes(o.as_slice()[offset..offset + 4].try_into()?);
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            let tolerance = 2.0e-5f32.max(expected.abs() * 2.0e-6);
            if !got.is_finite() || error > tolerance {
                mismatches += 1;
                first.get_or_insert((row, col, got, expected, error, tolerance));
            }
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "R115 parity failed: mismatches={mismatches} max_abs={max_abs:.9} first={first:?}"
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
        "embedding-r113-compact-group-n16 M={M} K={K} N={N}: mismatches={mismatches} max_abs={max_abs:.9} dispatch_ms={dispatch_ms:.6} activation_physical_bytes=196608 activation_unique_bytes=66560 nmacro_replicas=0"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P R115 verification is Linux-only");
}
