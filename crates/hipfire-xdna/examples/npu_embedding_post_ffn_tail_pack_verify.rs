//! Hardware oracle for fused post-FFN tail and next-layer W8 pack rungs.
//! Usage: `npu_embedding_post_ffn_tail_pack_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const PAD_M: usize = 288;
    const HIDDEN: usize = 768;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;
    const CORES: usize = 32;
    const PARAM_RECORD: usize = 9_216;
    const D_BYTES: usize = PAD_M * HIDDEN * 3 * size_of::<u16>();
    const P_BYTES: usize = CORES * PARAM_RECORD;
    const N_BYTES: usize = CORES * PARAM_RECORD;
    const O_BYTES: usize = PAD_M * HIDDEN * 2 * size_of::<u16>();
    const CORE_OUTPUT_BYTES: usize = 2 * HIDDEN * 2 * size_of::<u16>();
    const OUTPUT_JOIN_BYTES: usize = 4 * CORE_OUTPUT_BYTES;
    const DIAGNOSTIC_Q_BYTES: usize = 4 * 2 * GROUPS * OUTPUT_JOIN_BYTES;
    const R34_COMPACT_Q_BYTES: usize = 4 * GROUPS * 2 * OUTPUT_JOIN_BYTES;
    const CHUNK_Q_BYTES: usize = 8 * GROUP;
    const CHUNK_BYTES: usize = CHUNK_Q_BYTES + 8 * size_of::<f32>();
    const EPSILON: f32 = 1.0e-6;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_post_ffn_tail_pack_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(0usize);
    let cache = &args[0];
    let manifest = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    let r34_compact_output = manifest
        .lines()
        .any(|line| line == "op=embeddinggemma-post-ffn-tail-r34-compact-pack");
    let expected_op = if r34_compact_output {
        "op=embeddinggemma-post-ffn-tail-r34-compact-pack"
    } else {
        "op=embeddinggemma-post-ffn-tail-next-pack"
    };
    for field in [
        expected_op,
        "mode=bf16x2-resident",
        "m=256",
        "k=768",
        "token-owner-order=contiguous-eight-per-core",
        "completed-input-passes-for-pack=0",
        "immutable-tensor-reorder=none",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("fused tail-pack cache missing {field}").into());
        }
    }
    if r34_compact_output {
        for field in [
            "r34-output-bytes=589824",
            "r34-prefix-bytes=6240",
            "r34-materialized-nmacro-replicas=0",
            "r34-compact-memory-tiles=2,3,6,7",
            "r34-shim-route=reuse-completed-output",
            "token-owner-routing=adjacent-triomino-dma",
            "chunk-assembly=neighbor-memory-objectfifo",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(format!("R114 cache missing {field}").into());
            }
        }
    }

    let post_norm = (0..HIDDEN)
        .map(|hidden| f32_to_bf16_bits(0.87 + (hidden % 31) as f32 * 0.0018))
        .collect::<Vec<_>>();
    let residual = (0..M * HIDDEN)
        .map(|index| f32_to_bf16_bits(((index * 37 % 257) as f32 - 128.0) * 0.0017))
        .collect::<Vec<_>>();
    let exact_ffn = (0..M * HIDDEN)
        .map(|index| ((index * 23 % 193) as f32 - 96.0) * 0.0911 + 0.00317)
        .collect::<Vec<_>>();
    let (ffn_high, ffn_low): (Vec<_>, Vec<_>) = exact_ffn
        .iter()
        .map(|&value| {
            let high = f32_to_bf16_bits(value);
            let low = f32_to_bf16_bits(value - bf16_bits_to_f32(high));
            (high, low)
        })
        .unzip();
    let input_norm = (0..HIDDEN)
        .map(|col| 0.8 + (col % 29) as f32 * 0.011)
        .collect::<Vec<_>>();
    let awq = (0..HIDDEN)
        .map(|col| 0.7 + (col % 23) as f32 * 0.017)
        .collect::<Vec<_>>();
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);

    let kernel = NpuKernel::load(
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?;
    let mut d = kernel.alloc_arg(D_BYTES)?;
    let mut p = kernel.alloc_arg(P_BYTES)?;
    let mut n = kernel.alloc_arg(N_BYTES)?;
    let mut o = kernel.alloc_arg(O_BYTES)?;
    let mut q = kernel.alloc_arg(if r34_compact_output {
        R34_COMPACT_Q_BYTES
    } else {
        DIAGNOSTIC_Q_BYTES
    })?;
    d.as_mut_slice().fill(0);
    p.as_mut_slice().fill(0);
    n.as_mut_slice().fill(0);
    o.as_mut_slice().fill(0);
    q.as_mut_slice().fill(0);
    write_joined_rows(d.as_mut_slice(), &ffn_high, &ffn_low, &residual, M, HIDDEN);
    pack_tail_params(p.as_mut_slice(), &post_norm, EPSILON, PARAM_RECORD);
    pack_next_params(
        n.as_mut_slice(),
        &input_norm,
        &awq,
        &signs1,
        &signs2,
        PARAM_RECORD,
    );

    kernel.dispatch_synced(&[&d, &p, &n, &o, &q], &[true, true, true, false, false])?;
    kernel.sync_output(&o)?;
    kernel.sync_output(&q)?;

    let completed = read_completed(o.as_slice(), M, HIDDEN);
    let expected_completed = tail_reference(&ffn_high, &ffn_low, &residual, &post_norm, EPSILON);
    let (cosine, max_abs) = metrics(&completed, &expected_completed);
    if !cosine.is_finite() || cosine < 0.99999 || max_abs > 0.025 {
        return Err(format!(
            "R113 tail parity failed: cosine={cosine:.8} max_abs={max_abs:.7} got={:?} expected={:?}",
            &completed[..8],
            &expected_completed[..8]
        )
        .into());
    }

    let mut mismatches = 0usize;
    let mut mismatches_by_group = [0usize; GROUPS];
    let mut mismatches_by_owner = [0usize; CORES];
    let mut mismatches_by_lm = [0usize; 3];
    let mut max_q_delta = 0i16;
    let mut max_scale_abs = 0.0f32;
    let mut first = None;
    let mut best_q0_match = [(0usize, 0usize, usize::MAX); 8];
    for token in 0..M {
        let row = &completed[token * HIDDEN..(token + 1) * HIDDEN];
        let inverse = (row.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32
            + EPSILON)
            .sqrt()
            .recip();
        let half = token / 128;
        let within_half = token % 128;
        let core_row = within_half / 32;
        let within_row = within_half % 32;
        let local_col = within_row / 8;
        let local_row = within_row % 8;
        let owner = half * 16 + core_row * 4 + local_col;
        for group in 0..GROUPS {
            let mut rotated = [0.0f32; GROUP];
            for inner in 0..GROUP {
                let col = group * GROUP + inner;
                rotated[inner] = row[col] * inverse * input_norm[col] / awq[col];
            }
            cpu_fwht_256(&mut rotated, &signs1, &signs2);
            let max_value = rotated
                .iter()
                .fold(0.0f32, |value, item| value.max(item.abs()));
            let scale = if max_value > 0.0 {
                max_value / 127.0
            } else {
                0.0
            };
            let (q_base, scale_offset) = if r34_compact_output {
                let block = token / 24;
                let lm = (token % 24) / 8;
                let (packer_col, packer_row) = match block {
                    0..=7 => (block, 2),
                    8 => (2, 3),
                    9 => (5, 3),
                    10 => (7, 3),
                    _ => unreachable!(),
                };
                let memory_tile = packer_row + (packer_col / 4) * 4;
                let compact_index = match memory_tile {
                    2 => 0,
                    3 => 1,
                    6 => 2,
                    7 => 3,
                    _ => unreachable!(),
                };
                let slot = (packer_col % 4) * CORE_OUTPUT_BYTES;
                let q_plane = (compact_index * GROUPS * 2 + group * 2) * OUTPUT_JOIN_BYTES;
                (
                    q_plane + slot + lm * CHUNK_Q_BYTES,
                    q_plane + OUTPUT_JOIN_BYTES + slot + lm * 8 * size_of::<f32>(),
                )
            } else {
                let q_base = ((core_row * 2 + half) * GROUPS + group) * OUTPUT_JOIN_BYTES
                    + local_col * CORE_OUTPUT_BYTES;
                (q_base, q_base + CHUNK_Q_BYTES)
            };
            let got_scale = read_f32(q.as_slice(), scale_offset + local_row * 4)?;
            max_scale_abs = max_scale_abs.max((got_scale - scale).abs());
            let mut q0_probe_mismatches = 0usize;
            for inner in 0..GROUP {
                let kt = inner / 8;
                let kk = inner % 8;
                let got = q.as_slice()[q_base + kt * 64 + local_row * 8 + kk] as i8;
                let expected = if scale > 0.0 {
                    (rotated[inner] / scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
                if r34_compact_output {
                    let probe = q.as_slice()[kt * 64 + local_row * 8 + kk] as i8;
                    q0_probe_mismatches += usize::from(probe != expected);
                }
                if got != expected {
                    mismatches += 1;
                    mismatches_by_group[group] += 1;
                    mismatches_by_owner[owner] += 1;
                    mismatches_by_lm[(token % 24) / 8] += 1;
                    max_q_delta = max_q_delta.max((got as i16 - expected as i16).abs());
                    first.get_or_insert((token, group, inner, got, expected));
                }
            }
            if r34_compact_output && q0_probe_mismatches < best_q0_match[local_row].2 {
                best_q0_match[local_row] = (token, group, q0_probe_mismatches);
            }
        }
    }
    if mismatches > 16 || max_q_delta > 1 || max_scale_abs > 1.0e-6 {
        return Err(format!(
            "{} pack parity failed: mismatches={mismatches} max_q_delta={max_q_delta} max_scale_abs={max_scale_abs:.9} first={first:?} by_group={mismatches_by_group:?} by_owner={mismatches_by_owner:?} by_lm={mismatches_by_lm:?} best_q0_match={best_q0_match:?} q0={:?} scales0={:?}",
            if r34_compact_output { "R114" } else { "R113" },
            &q.as_slice()[..32],
            if r34_compact_output {
                &q.as_slice()[OUTPUT_JOIN_BYTES..OUTPUT_JOIN_BYTES + 96]
            } else {
                &q.as_slice()[CHUNK_Q_BYTES..CHUNK_BYTES]
            },
        )
        .into());
    }

    let dispatch_ms = if iterations > 0 {
        for _ in 0..2 {
            kernel.dispatch_synced(&[&d, &p, &n, &o, &q], &[false; 5])?;
        }
        let started = Instant::now();
        for _ in 0..iterations {
            kernel.dispatch_synced(&[&d, &p, &n, &o, &q], &[false; 5])?;
        }
        Some(started.elapsed().as_secs_f64() * 1e3 / iterations as f64)
    } else {
        None
    };
    println!(
        "embedding-post-ffn-tail-pack mode={} M={M} K={HIDDEN}: cosine={cosine:.8} max_abs={max_abs:.7} q_mismatches={mismatches} max_q_delta={max_q_delta} max_scale_abs={max_scale_abs:.9} dispatch_ms={dispatch_ms:?}",
        if r34_compact_output {
            "r34-compact"
        } else {
            "diagnostic"
        }
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_joined_rows(
    destination: &mut [u8],
    high: &[u16],
    low: &[u16],
    residual: &[u16],
    rows: usize,
    hidden: usize,
) {
    let row_bytes = 3 * hidden * size_of::<u16>();
    for row in 0..rows {
        let target = &mut destination[row * row_bytes..(row + 1) * row_bytes];
        let source = row * hidden;
        for col in 0..hidden {
            let offset = col * 4;
            target[offset..offset + 2].copy_from_slice(&high[source + col].to_le_bytes());
            target[offset + 2..offset + 4].copy_from_slice(&low[source + col].to_le_bytes());
        }
        for col in 0..hidden {
            let offset = 2 * hidden * 2 + col * 2;
            target[offset..offset + 2].copy_from_slice(&residual[source + col].to_le_bytes());
        }
    }
}

#[cfg(target_os = "linux")]
fn pack_tail_params(destination: &mut [u8], norm: &[u16], epsilon: f32, record: usize) {
    for core in 0..32 {
        let base = core * record;
        for (col, bits) in norm.iter().copied().enumerate() {
            destination[base + col * 2..base + col * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let offset = base + norm.len() * 2;
        destination[offset..offset + 4].copy_from_slice(&epsilon.to_le_bytes());
    }
}

#[cfg(target_os = "linux")]
fn pack_next_params(
    destination: &mut [u8],
    norm: &[f32],
    awq: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    record: usize,
) {
    use hipfire_primitives::conv::f32_to_bf16_bits;

    const GROUP: usize = 256;
    const GROUP_PARAM: usize = 3_072;
    for core in 0..32 {
        let record_base = core * record;
        for group in 0..3 {
            let base = record_base + group * GROUP_PARAM;
            for inner in 0..GROUP {
                let col = group * GROUP + inner;
                destination[base + inner * 4..base + inner * 4 + 4]
                    .copy_from_slice(&norm[col].to_le_bytes());
                let awq_offset = base + GROUP * 4 + inner * 4;
                destination[awq_offset..awq_offset + 4].copy_from_slice(&awq[col].to_le_bytes());
                let sign1_offset = base + 2 * GROUP * 4 + inner * 2;
                destination[sign1_offset..sign1_offset + 2]
                    .copy_from_slice(&f32_to_bf16_bits(signs1[inner]).to_le_bytes());
                let sign2_offset = base + 2 * GROUP * 4 + GROUP * 2 + inner * 2;
                destination[sign2_offset..sign2_offset + 2]
                    .copy_from_slice(&f32_to_bf16_bits(signs2[inner]).to_le_bytes());
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn read_completed(bytes: &[u8], rows: usize, hidden: usize) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;

    let mut output = vec![0.0f32; rows * hidden];
    for row in 0..rows {
        let base = row * 2 * hidden * 2;
        for col in 0..hidden {
            let high = u16::from_le_bytes(
                bytes[base + col * 2..base + col * 2 + 2]
                    .try_into()
                    .unwrap(),
            );
            let low_offset = base + hidden * 2 + col * 2;
            let low = u16::from_le_bytes(bytes[low_offset..low_offset + 2].try_into().unwrap());
            output[row * hidden + col] = bf16_bits_to_f32(high) + bf16_bits_to_f32(low);
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn tail_reference(
    high: &[u16],
    low: &[u16],
    residual: &[u16],
    norm: &[u16],
    epsilon: f32,
) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};

    const M: usize = 256;
    const HIDDEN: usize = 768;
    let mut output = vec![0.0f32; M * HIDDEN];
    for token in 0..M {
        let base = token * HIDDEN;
        let sum = (0..HIDDEN)
            .map(|col| {
                let value = bf16_bits_to_f32(high[base + col]) + bf16_bits_to_f32(low[base + col]);
                value * value
            })
            .sum::<f32>();
        let inverse = (sum / HIDDEN as f32 + epsilon).sqrt().recip();
        for col in 0..HIDDEN {
            let ffn = bf16_bits_to_f32(high[base + col]) + bf16_bits_to_f32(low[base + col]);
            let value = bf16_bits_to_f32(residual[base + col])
                + ffn * bf16_bits_to_f32(norm[col]) * inverse;
            let result_high = f32_to_bf16_bits(value);
            let result_low = f32_to_bf16_bits(value - bf16_bits_to_f32(result_high));
            output[base + col] = bf16_bits_to_f32(result_high) + bf16_bits_to_f32(result_low);
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, Box<dyn std::error::Error>> {
    Ok(f32::from_le_bytes(bytes[offset..offset + 4].try_into()?))
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32) {
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&got, &expected) in got.iter().zip(expected) {
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        max_abs = max_abs.max((got - expected).abs());
    }
    (dot / (got_norm.sqrt() * expected_norm.sqrt()), max_abs)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P R113 verification is Linux-only");
}
