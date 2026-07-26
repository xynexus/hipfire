//! Exact scaled parity and timing for the R15 grouped whole-array W4 path.
//! Usage: `npu_gemm_whole_scaled_verify CACHE [ITERS]`
//! Set `HIPFIRE_NPU_VERIFY_UNIT_SCALES=1` to isolate the integer contract.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::NpuKernel;

    const ARRAY: usize = 4;
    const LM: usize = 6;
    const LN: usize = 6;
    const GROUP_K: usize = 256;
    const ROWS_STRIPE: usize = 24;
    const COLS_STRIPE: usize = 96;
    const MACRO_M: usize = 96;
    const MACRO_N: usize = 384;
    const A0: usize = 6144;
    const W0: usize = 12288;
    const AB: usize = 8192;
    const WB: usize = 16384;
    const CB: usize = 2304;
    const CJ: usize = 9216;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_gemm_whole_scaled_verify CACHE [ITERS]".into());
    }
    let cache = &args[0];
    let iterations = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(10);
    let shape = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    let get = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
        let value = shape
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .ok_or_else(|| format!("missing {key}"))?;
        Ok(value.parse()?)
    };
    let (m, k, n) = (get("m")?, get("k")?, get("n")?);
    let (mm, nm, groups) = (get("mm")?, get("nm")?, get("kg")?);
    let outblocks = get("outblocks")?;
    assert_eq!(
        (mm, nm, groups),
        (m.div_ceil(MACRO_M), n.div_ceil(MACRO_N), k / GROUP_K)
    );
    assert_eq!(outblocks, mm * nm);

    let activations: Vec<i8> = (0..m * k)
        .map(|index| (((index * 17 + index / k * 3) % 255) as i16 - 127) as i8)
        .collect();
    let unit_scales = std::env::var_os("HIPFIRE_NPU_VERIFY_UNIT_SCALES").is_some();
    let activation_scales: Vec<f32> = (0..groups * m)
        .map(|index| {
            if unit_scales {
                1.0
            } else {
                0.003 + (index % 29) as f32 * 0.00007
            }
        })
        .collect();
    let weights: Vec<Vec<i8>> = (0..groups)
        .map(|group| {
            (0..GROUP_K * n)
                .map(|index| ((index * 11 + index / n * 5 + group * 7) % 15) as i8 - 7)
                .collect()
        })
        .collect();
    let weight_scales: Vec<Vec<f32>> = (0..groups)
        .map(|group| {
            (0..n)
                .map(|col| {
                    if unit_scales {
                        1.0
                    } else {
                        0.007 + ((col + group * 3) % 31) as f32 * 0.00009
                    }
                })
                .collect()
        })
        .collect();

    let xclbin = std::fs::read(format!("{cache}/final.xclbin"))?;
    let insts = std::fs::read(format!("{cache}/insts.bin"))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let inblocks = outblocks * groups;
    let mut a = kernel.alloc_arg(ARRAY * inblocks * AB)?;
    let mut w = kernel.alloc_arg(ARRAY * inblocks * WB)?;
    let c = kernel.alloc_arg(ARRAY * outblocks * CJ * 4)?;
    a.as_mut_slice().fill(0);
    w.as_mut_slice().fill(0);

    for stripe in 0..ARRAY {
        for m_macro in 0..mm {
            for n_macro in 0..nm {
                let outblock = m_macro * nm + n_macro;
                for group in 0..groups {
                    let block = outblock * groups + group;
                    let abase = (stripe * inblocks + block) * AB;
                    for lm in 0..LM {
                        for kt in 0..16 {
                            for local_row in 0..4 {
                                let row =
                                    m_macro * MACRO_M + stripe * ROWS_STRIPE + lm * 4 + local_row;
                                if row < m {
                                    let src = row * k + group * GROUP_K + kt * 16;
                                    let dst = abase + (lm * 16 + kt) * 64 + local_row * 16;
                                    a.as_mut_slice()[dst..dst + 16]
                                        .copy_from_slice(as_bytes(&activations[src..src + 16]));
                                }
                            }
                        }
                    }
                    for local_row in 0..ROWS_STRIPE {
                        let row = m_macro * MACRO_M + stripe * ROWS_STRIPE + local_row;
                        let scale = if row < m {
                            activation_scales[group * m + row]
                        } else {
                            0.0
                        };
                        let offset = abase + A0 + local_row * 4;
                        a.as_mut_slice()[offset..offset + 4].copy_from_slice(&scale.to_ne_bytes());
                    }
                    let wbase = (stripe * inblocks + block) * WB;
                    for ln in 0..LN {
                        for kt in 0..16 {
                            for kk in 0..16 {
                                for nn in 0..16 {
                                    let col =
                                        n_macro * MACRO_N + stripe * COLS_STRIPE + ln * 16 + nn;
                                    let value = if col < n {
                                        weights[group][(kt * 16 + kk) * n + col]
                                    } else {
                                        0
                                    };
                                    let index = (ln * 16 + kt) * 256 + kk * 16 + nn;
                                    let nibble = (value & 0x0f) as u8;
                                    w.as_mut_slice()[wbase + index / 2] |=
                                        if index % 2 == 0 { nibble } else { nibble << 4 };
                                }
                            }
                        }
                    }
                    for local_col in 0..COLS_STRIPE {
                        let col = n_macro * MACRO_N + stripe * COLS_STRIPE + local_col;
                        let scale = if col < n {
                            weight_scales[group][col]
                        } else {
                            0.0
                        };
                        let offset = wbase + W0 + local_col * 4;
                        w.as_mut_slice()[offset..offset + 4].copy_from_slice(&scale.to_ne_bytes());
                    }
                }
            }
        }
    }

    kernel.dispatch_synced(&[&a, &w, &c], &[true, true, true])?;
    kernel.dispatch_synced(&[&a, &w, &c], &[false, false, true])?;
    kernel.dispatch_synced(&[&a, &w, &c], &[false, false, true])?;
    kernel.sync_output(&c)?;
    let physical = as_f32(c.as_slice());
    let mut output = vec![0.0f32; m * n];
    for col_stripe in 0..ARRAY {
        for m_macro in 0..mm {
            for n_macro in 0..nm {
                let outblock = m_macro * nm + n_macro;
                for row_stripe in 0..ARRAY {
                    let core = (col_stripe * outblocks + outblock) * CJ + row_stripe * CB;
                    for lm in 0..LM {
                        for ln in 0..LN {
                            for rr in 0..4 {
                                let row =
                                    m_macro * MACRO_M + row_stripe * ROWS_STRIPE + lm * 4 + rr;
                                if row >= m {
                                    continue;
                                }
                                for cc in 0..16 {
                                    let col =
                                        n_macro * MACRO_N + col_stripe * COLS_STRIPE + ln * 16 + cc;
                                    if col < n {
                                        output[row * n + col] =
                                            physical[core + (lm * LN + ln) * 64 + rr * 16 + cc];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for row in 0..m {
        for col in 0..n {
            let mut expected = 0.0f32;
            for group in 0..groups {
                let dot: i32 = (0..GROUP_K)
                    .map(|inner| {
                        activations[row * k + group * GROUP_K + inner] as i32
                            * weights[group][inner * n + col] as i32
                    })
                    .sum();
                expected +=
                    dot as f32 * activation_scales[group * m + row] * weight_scales[group][col];
            }
            let got = output[row * n + col];
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            if error > 1e-5 + expected.abs() * 1e-5 {
                mismatches += 1;
                first.get_or_insert((row, col, got, expected));
            }
        }
    }
    println!("whole-scaled-W4 M={m} K={k} N={n}: mismatches={mismatches} max_abs={max_abs:.7}");
    if let Some((row, col, got, expected)) = first {
        println!("first_mismatch row={row} col={col} got={got} expected={expected}");
    }
    if mismatches != 0 {
        return Err("R15 scaled parity failed".into());
    }
    for _ in 0..3 {
        kernel.dispatch_synced(&[&a, &w, &c], &[false, false, true])?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(&[&a, &w, &c], &[false, false, true])?;
    }
    let seconds = started.elapsed().as_secs_f64() / iterations as f64;
    println!(
        "iters={iterations} dispatch_ms={:.4} logical_tops={:.4}",
        seconds * 1e3,
        2.0 * m as f64 * k as f64 * n as f64 / seconds / 1e12
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn as_bytes(values: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

#[cfg(target_os = "linux")]
fn as_f32(values: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() / 4) }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
