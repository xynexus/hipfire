//! Verify and time all K=256 group partials in one AIE2P/XRT dispatch.
//!
//! This proves the full-K submission schedule only. It deliberately keeps the
//! group outputs separate; scale reconstruction and accumulation are not part
//! of this benchmark and must not be inferred from its timing.
//!
//! Usage: `npu_gemm_fullk_verify CACHE COLS MT NB KGROUPS MODE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{NpuGemmFullK, NpuKernel};

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(6..=7).contains(&args.len()) {
        return Err("usage: npu_gemm_fullk_verify CACHE COLS MT NB KGROUPS MODE [ITERS]".into());
    }
    let cache = &args[0];
    let direct_output = std::fs::read_to_string(format!("{cache}/output-layout.txt"))
        .is_ok_and(|layout| layout.trim() == "direct");
    let cols: usize = args[1].parse()?;
    let mt: usize = args[2].parse()?;
    let nb: usize = args[3].parse()?;
    let groups: usize = args[4].parse()?;
    let mode = args[5].as_str();
    let weight_bits = match mode {
        "w4" | "mixed" => 4,
        "w8" => 8,
        _ => return Err("MODE must be w4, mixed, or w8".into()),
    };
    let iterations: usize = args
        .get(6)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(20);

    let mr = if weight_bits == 4 { 4 } else { 8 };
    let rows_per_core = mt * mr;
    let rows = cols * rows_per_core;
    let group_k = 256usize;
    let slab_n = 64usize;
    let n = nb * slab_n;
    let activation_bytes = rows_per_core * group_k;
    let weight_bytes = if mode == "mixed" || weight_bits == 8 {
        16384
    } else {
        8192
    };
    let weight_entries = if mode == "mixed" { 2 } else { 1 };
    let output_components = 1usize;
    let output_elements = rows_per_core * slab_n;

    let xclbin = std::fs::read(format!("{cache}/final.xclbin"))?;
    let instructions = std::fs::read(format!("{cache}/insts.bin"))?;
    let kernel = NpuKernel::load(&xclbin, &instructions)?;
    let mut input = kernel.alloc_arg(cols * groups * activation_bytes)?;
    let mut weights = kernel.alloc_arg(groups * nb * weight_entries * weight_bytes)?;
    let mut output =
        kernel.alloc_arg(cols * groups * nb * output_components * output_elements * 4)?;
    weights.as_mut_slice().fill(0);
    for group in 0..groups {
        for slab in 0..nb {
            let entry = (group * nb + slab) * weight_entries;
            let base =
                &mut weights.as_mut_slice()[entry * weight_bytes..(entry + 1) * weight_bytes];
            if weight_bits == 4 {
                base[..8192].fill(0x11);
            } else {
                base.fill(0x01);
            }
            if mode == "mixed" {
                let residual = &mut weights.as_mut_slice()
                    [(entry + 1) * weight_bytes..(entry + 2) * weight_bytes];
                fill_dense_residual(residual);
            }
        }
    }
    output.as_mut_slice().fill(0);

    let mut expected_dots = vec![0i32; groups * rows];
    let mut canonical_activations = vec![0i8; rows * groups * group_k];
    for core in 0..cols {
        for group in 0..groups {
            let block_index = core * groups + group;
            let block = &mut input.as_mut_slice()
                [block_index * activation_bytes..(block_index + 1) * activation_bytes];
            for local_row in 0..rows_per_core {
                let global_row = core * rows_per_core + local_row;
                let mut row_values = vec![0i8; group_k];
                let mut dot = 0i32;
                for inner in 0..group_k {
                    let value = ((global_row * 3 + group * 5 + inner) % 15) as i8 - 7;
                    row_values[inner] = value;
                    dot += value as i32;
                }
                for (inner, value) in row_values.iter().enumerate() {
                    block[local_row * group_k + inner] = *value as u8;
                    canonical_activations
                        [global_row * groups * group_k + group * group_k + inner] = *value;
                }
                expected_dots[group * rows + global_row] = dot;
            }
        }
    }

    kernel.dispatch(&[&input, &weights, &output])?;
    let output_values: &[i32] = unsafe {
        std::slice::from_raw_parts(
            output.as_slice().as_ptr().cast::<i32>(),
            cols * groups * nb * output_components * output_elements,
        )
    };
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for core in 0..cols {
        for group in 0..groups {
            for slab in 0..nb {
                for local_row in 0..rows_per_core {
                    let global_row = core * rows_per_core + local_row;
                    for local_col in 0..slab_n {
                        let mut expected = expected_dots[group * rows + global_row];
                        if mode == "mixed" {
                            let row_start = global_row * groups * group_k + group * group_k;
                            expected += sparse_correction_for_column(
                                as_u8(&canonical_activations[row_start..row_start + group_k]),
                                local_col,
                            );
                        }
                        let col = slab * slab_n + local_col;
                        let physical = if direct_output {
                            (group * rows + global_row) * n + col
                        } else {
                            ((core * groups + group) * nb + slab) * output_elements
                                + local_row * slab_n
                                + local_col
                        };
                        let got = output_values[physical];
                        if got != expected {
                            mismatches += 1;
                            first_mismatch.get_or_insert((global_row, group, col, got, expected));
                        }
                    }
                }
            }
        }
    }
    println!(
        "fullk-submit-{mode} M={rows} K={} N={n} groups={groups}: mismatches={mismatches}",
        groups * group_k
    );
    if let Some(sample) = first_mismatch {
        println!(
            "first_mismatch row={} group={} col={} got={} expected={}",
            sample.0, sample.1, sample.2, sample.3, sample.4
        );
    }
    if mismatches != 0 {
        return Err(format!("full-K {mode} AIE2P submission parity failed").into());
    }

    for _ in 0..3 {
        kernel.dispatch(&[&input, &weights, &output])?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        kernel.dispatch(&[&input, &weights, &output])?;
    }
    let elapsed = started.elapsed().as_secs_f64() / iterations as f64;
    let macs = rows as f64 * (groups * group_k) as f64 * n as f64;
    println!(
        "iters={iterations} submission_ms={:.4} tops={:.4}",
        elapsed * 1e3,
        2.0 * macs / elapsed / 1e12
    );

    // Full-column AIE programs own the complete partition. Tear down the raw
    // verifier context before loading the production wrapper's context.
    let packed_runtime_weights = weights.as_slice().to_vec();
    drop(output);
    drop(input);
    drop(weights);
    drop(kernel);
    let mut runtime = NpuGemmFullK::load_cached(cache, cols)?;
    let resident = runtime.upload_resident_weights(&packed_runtime_weights)?;
    let mut runtime_partials = vec![0i32; groups * rows * n];
    runtime.run_resident(&resident, &canonical_activations, &mut runtime_partials)?;
    let mut runtime_mismatches = 0usize;
    let mut runtime_first = None;
    for group in 0..groups {
        for row in 0..rows {
            let activations = &canonical_activations[row * groups * group_k + group * group_k
                ..row * groups * group_k + (group + 1) * group_k];
            for col in 0..n {
                let mut expected = expected_dots[group * rows + row];
                if mode == "mixed" {
                    expected += sparse_correction_for_column(as_u8(activations), col % slab_n);
                }
                if runtime_partials[(group * rows + row) * n + col] != expected {
                    runtime_mismatches += 1;
                    runtime_first.get_or_insert((
                        group,
                        row,
                        col,
                        runtime_partials[(group * rows + row) * n + col],
                        expected,
                    ));
                }
            }
        }
    }
    println!("runtime-layout-{mode}: mismatches={runtime_mismatches}");
    if let Some((group, row, col, got, expected)) = runtime_first {
        println!("runtime_first group={group} row={row} col={col} got={got} expected={expected}");
    }
    if runtime_mismatches != 0 {
        return Err(format!("full-K {mode} runtime layout parity failed").into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn as_u8(values: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len()) }
}

#[cfg(target_os = "linux")]
const SPARSE_INDICES: [usize; 5] = [0, 47, 94, 141, 188];
#[cfg(target_os = "linux")]
const SPARSE_DELTAS: [i8; 5] = [2, -3, 4, -5, 6];

#[cfg(target_os = "linux")]
fn fill_dense_residual(residual: &mut [u8]) {
    const NT: usize = 4;
    const KCHUNK: usize = 32;
    for nt in 0..NT {
        for k_tile in 0..KCHUNK {
            for n_half in 0..2 {
                for kk in 0..8 {
                    for nn in 0..8 {
                        let k = k_tile * 8 + kk;
                        let n = nt * 16 + n_half * 8 + nn;
                        let value = SPARSE_INDICES
                            .iter()
                            .zip(SPARSE_DELTAS)
                            .find_map(|(&index, delta)| (index == k).then_some(delta))
                            .map_or(0, |delta| if n % 3 == 0 { delta } else { -delta });
                        let packed = ((nt * KCHUNK + k_tile) * 2 + n_half) * 64 + kk * 8 + nn;
                        residual[packed] = value as u8;
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn sparse_correction_for_column(activations: &[u8], column: usize) -> i32 {
    let correction: i32 = SPARSE_INDICES
        .iter()
        .zip(SPARSE_DELTAS)
        .map(|(&index, delta)| activations[index] as i8 as i32 * delta as i32)
        .sum();
    if column % 3 == 0 {
        correction
    } else {
        -correction
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("amdxdna is Linux-only");
}
