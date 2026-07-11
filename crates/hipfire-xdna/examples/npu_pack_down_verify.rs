//! Hardware parity and timing for combined vector-pack + W4/W8 down projection.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::fwht::gen_fwht_signs;
    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const PAD_M: usize = 288;
    const K: usize = 1152;
    const PAD_K: usize = 1280;
    const N: usize = 768;
    const GROUP: usize = 256;
    const GROUPS: usize = 5;
    const COLS: usize = 8;
    const ROW_STRIPES: usize = 4;
    const ROWS_PER_STRIPE: usize = 24;
    const W_DATA: usize = 12288;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_pack_down_verify CACHE [ITERS]".into());
    }
    let iterations = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(20);
    let shape = std::fs::read_to_string(format!("{}/shape.txt", args[0]))?;
    let w8 = shape.lines().any(|line| line == "mode=w8");
    let mixed = shape.lines().any(|line| line == "mode=mixed");
    let identity = std::env::var_os("HIPFIRE_PACK_IDENTITY").is_some();
    let overlays = shape
        .lines()
        .find_map(|line| line.strip_prefix("overlays="))
        .map(str::parse)
        .transpose()?
        .unwrap_or(0usize);
    let (outblocks, n_macros, cols_stripe, ln, inner_k) = if w8 {
        (6, 2, 48, 3, 8)
    } else {
        (3, 1, 96, 6, 16)
    };
    let xblocks = (outblocks / n_macros) * GROUPS * ROWS_PER_STRIPE;
    let wblocks = outblocks * GROUPS;
    let param_offset = W_DATA + cols_stripe * size_of::<f32>();
    let mixed_offset = param_offset + 3 * GROUP * size_of::<f32>();
    let used = mixed_offset + 2 * cols_stripe * overlays;
    let wb = 16384.max(used.div_ceil(256) * 256);

    let mut input = vec![0.0f32; PAD_M * PAD_K];
    for row in 0..M {
        for col in 0..K {
            input[row * PAD_K + col] = ((row * 29 + col * 17) as f32 * 0.0027).sin() * 3.25
                + ((row + col) % 9) as f32 * 0.031;
        }
    }
    let mut awq = vec![1.0f32; PAD_K];
    for (col, scale) in awq[..K].iter_mut().enumerate() {
        *scale = 0.7 + (col % 23) as f32 * 0.027;
    }
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);

    let weights: Vec<Vec<i8>> = (0..GROUPS)
        .map(|group| {
            (0..GROUP * N)
                .map(|index| {
                    if identity {
                        i8::from(index / N == index % N % GROUP)
                    } else {
                        ((index * 11 + index / N * 5 + group * 7) % 15) as i8 - 7
                    }
                })
                .collect()
        })
        .collect();
    let mut effective_weights = weights.clone();
    if mixed {
        for group in 0..GROUPS {
            for col in 0..N {
                for entry in 0..overlays {
                    let inner = (entry * 37 + col * 13 + group * 11) % GROUP;
                    effective_weights[group][inner * N + col] =
                        (((entry * 19 + col * 7 + group * 5) % 201) as i16 - 100) as i8;
                }
            }
        }
    }
    let weight_scales: Vec<Vec<f32>> = (0..GROUPS)
        .map(|group| {
            (0..N)
                .map(|col| {
                    if identity {
                        1.0
                    } else {
                        0.007 + ((col + group * 3) % 31) as f32 * 0.00009
                    }
                })
                .collect()
        })
        .collect();
    let (quantized, activation_scales) = prepare(&input, &awq, &signs1, &signs2);

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut x = kernel.alloc_arg(ROW_STRIPES * xblocks * GROUP * size_of::<f32>())?;
    let mut w = kernel.alloc_arg(COLS * wblocks * wb)?;
    let c = kernel.alloc_arg(PAD_M * N * size_of::<f32>())?;

    for stripe in 0..ROW_STRIPES {
        for m_macro in 0..outblocks / n_macros {
            for group in 0..GROUPS {
                for local_row in 0..ROWS_PER_STRIPE {
                    let row = m_macro * 96 + stripe * ROWS_PER_STRIPE + local_row;
                    let block = (m_macro * GROUPS + group) * ROWS_PER_STRIPE + local_row;
                    let destination = (stripe * xblocks + block) * GROUP * size_of::<f32>();
                    let source = row * PAD_K + group * GROUP;
                    x.as_mut_slice()[destination..destination + GROUP * size_of::<f32>()]
                        .copy_from_slice(unsafe { as_bytes(&input[source..source + GROUP]) });
                }
            }
        }
    }

    w.as_mut_slice().fill(0);
    for stripe in 0..COLS {
        for outblock in 0..outblocks {
            let m_macro = outblock / n_macros;
            let n_macro = outblock % n_macros;
            for group in 0..GROUPS {
                let block = (m_macro * GROUPS + group) * n_macros + n_macro;
                let base = (stripe * wblocks + block) * wb;
                for ln_index in 0..ln {
                    for kt in 0..GROUP / inner_k {
                        for kk in 0..inner_k {
                            for nn in 0..16 {
                                let col = n_macro * COLS * cols_stripe
                                    + stripe * cols_stripe
                                    + ln_index * 16
                                    + nn;
                                let value = weights[group][(kt * inner_k + kk) * N + col];
                                if w8 {
                                    let index = (ln_index * 32 + kt) * 128
                                        + (nn / 8) * 64
                                        + kk * 8
                                        + nn % 8;
                                    w.as_mut_slice()[base + index] = value as u8;
                                } else {
                                    let index = (ln_index * 16 + kt) * 256 + kk * 16 + nn;
                                    let nibble = (value & 0x0f) as u8;
                                    w.as_mut_slice()[base + index / 2] |=
                                        if index % 2 == 0 { nibble } else { nibble << 4 };
                                }
                            }
                        }
                    }
                }
                for local_col in 0..cols_stripe {
                    let col = n_macro * COLS * cols_stripe + stripe * cols_stripe + local_col;
                    let offset = base + W_DATA + local_col * size_of::<f32>();
                    w.as_mut_slice()[offset..offset + 4]
                        .copy_from_slice(&weight_scales[group][col].to_ne_bytes());
                }
                let mut params = Vec::with_capacity(3 * GROUP);
                params.extend_from_slice(&awq[group * GROUP..(group + 1) * GROUP]);
                params.extend_from_slice(&signs1);
                params.extend_from_slice(&signs2);
                w.as_mut_slice()[base + param_offset..base + param_offset + 3 * GROUP * 4]
                    .copy_from_slice(unsafe { as_bytes(&params) });
                if mixed {
                    for local_col in 0..cols_stripe {
                        let col = stripe * cols_stripe + local_col;
                        for entry in 0..overlays {
                            let inner = (entry * 37 + col * 13 + group * 11) % GROUP;
                            let offset = base + mixed_offset + (local_col * overlays + entry) * 2;
                            w.as_mut_slice()[offset] = inner as u8;
                            w.as_mut_slice()[offset + 1] = (effective_weights[group]
                                [inner * N + col]
                                - weights[group][inner * N + col])
                                as u8;
                        }
                    }
                }
            }
        }
    }

    kernel.dispatch_synced(&[&x, &w, &c], &[true, true, true])?;
    kernel.sync_output(&c)?;
    let output = unsafe { as_f32(c.as_slice()) };
    let mut mismatches = 0usize;
    let mut first = None;
    let mut max_abs = 0.0f32;
    for row in 0..M {
        for col in 0..N {
            let mut expected = 0.0f32;
            for group in 0..GROUPS {
                let dot: i32 = (0..GROUP)
                    .map(|inner| {
                        quantized[row * PAD_K + group * GROUP + inner] as i32
                            * effective_weights[group][inner * N + col] as i32
                    })
                    .sum();
                expected += dot as f32
                    * activation_scales[row * GROUPS + group]
                    * weight_scales[group][col];
            }
            let got = output[row * N + col];
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            if error > 1e-5 + expected.abs() * 1e-5 {
                mismatches += 1;
                first.get_or_insert((row, col, got, expected));
            }
        }
    }
    if let Some((row, col, got, expected)) = first {
        eprintln!("first mismatch row={row} col={col} got={got} expected={expected}");
    }
    if mismatches != 0 {
        let nonzero = output.iter().filter(|&&value| value != 0.0).count();
        eprintln!("nonzero physical outputs: {nonzero}/{}", output.len());
        eprintln!("first physical outputs: {:?}", &output[..16]);
        return Err(format!(
            "combined pack/down parity failed: mismatches={mismatches} max_abs={max_abs:.7}"
        )
        .into());
    }

    for _ in 0..3 {
        kernel.dispatch_synced(&[&x, &w, &c], &[false, false, true])?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(&[&x, &w, &c], &[false, false, true])?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let mode = if w8 {
        "W8".to_string()
    } else if mixed {
        format!("mixed-o{overlays}")
    } else {
        "W4".to_string()
    };
    println!(
        "vector-pack-down-{mode} M={M} K={K} padded_K={PAD_K} N={N}: mismatches=0 max_abs={max_abs:.7} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn prepare(input: &[f32], awq: &[f32], signs1: &[f32], signs2: &[f32]) -> (Vec<i8>, Vec<f32>) {
    use hipfire_primitives::fwht::cpu_fwht_256;

    const M: usize = 256;
    const PAD_K: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 5;
    let mut quantized = vec![0i8; M * PAD_K];
    let mut scales = vec![0.0f32; M * GROUPS];
    for row in 0..M {
        for group in 0..GROUPS {
            let mut rotated = vec![0.0f32; GROUP];
            for inner in 0..GROUP {
                let col = group * GROUP + inner;
                rotated[inner] = input[row * PAD_K + col] / awq[col];
            }
            cpu_fwht_256(&mut rotated, signs1, signs2);
            let max_abs = rotated
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            scales[row * GROUPS + group] = scale;
            if scale > 0.0 {
                for inner in 0..GROUP {
                    quantized[row * PAD_K + group * GROUP + inner] =
                        (rotated[inner] / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (quantized, scales)
}

#[cfg(target_os = "linux")]
unsafe fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(target_os = "linux")]
unsafe fn as_f32(values: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() / 4) }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("R21 AIE2P verification is Linux-only");
}
