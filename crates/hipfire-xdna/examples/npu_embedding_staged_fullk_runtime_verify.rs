//! Hardware gate for the production R121 staged full-K runtime wrapper.
//! Usage: `npu_embedding_staged_fullk_runtime_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
    use hipfire_xdna::NpuGemmStagedFullK;

    const M: usize = 256;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_staged_fullk_runtime_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(100usize);
    let mut runtime = NpuGemmStagedFullK::load_cached(&args[0])?;
    let n = runtime.n();

    let x = (0..M * runtime.k())
        .map(|index| ((index * 19 + index / 31) % 101) as f32 / 100.0 - 0.5)
        .collect::<Vec<_>>();
    let awq_scale = (0..runtime.k())
        .map(|index| 0.91 + (index % 17) as f32 * 0.007)
        .collect::<Vec<_>>();
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);
    let mut activations = vec![vec![0i8; M * GROUP]; GROUPS];
    let mut activation_scales = vec![vec![1.0f32; M]; GROUPS];
    for group in 0..GROUPS {
        for row in 0..M {
            let mut rotated = [0.0f32; GROUP];
            for inner in 0..GROUP {
                let col = group * GROUP + inner;
                rotated[inner] = x[row * runtime.k() + col] / awq_scale[col];
            }
            cpu_fwht_256(&mut rotated, &signs1, &signs2);
            let scale = rotated
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()))
                / 127.0;
            activation_scales[group][row] = if scale > 0.0 { scale } else { 1.0 };
            for inner in 0..GROUP {
                activations[group][row * GROUP + inner] =
                    (rotated[inner] / activation_scales[group][row])
                        .round()
                        .clamp(-127.0, 127.0) as i8;
            }
        }
    }
    let weights = (0..GROUPS)
        .map(|group| {
            (0..GROUP * n)
                .map(|index| {
                    let inner = index / n;
                    let col = index % n;
                    ((((group * GROUP + inner) * 13 + col * 7) % 23) as i8) - 11
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let weight_scales = (0..GROUPS)
        .map(|group| {
            (0..n)
                .map(|col| 0.0029 + ((group * n + col) % 13) as f32 * 0.000_037)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let activation_refs = activations.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let activation_scale_refs = activation_scales
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let weight_refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let weight_scale_refs = weight_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let packed_weights = runtime.prepack_weights(&weight_refs, &weight_scale_refs)?;
    let resident = runtime.upload_resident_weights(&packed_weights)?;
    let mut output = vec![0.0f32; M * n];
    runtime.run_resident_scaled(
        &resident,
        &activation_refs,
        &activation_scale_refs,
        &mut output,
    )?;

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for row in 0..M {
        for col in 0..n {
            let expected = (0..GROUPS)
                .map(|group| {
                    let dot = (0..GROUP)
                        .map(|inner| {
                            activations[group][row * GROUP + inner] as i32
                                * weights[group][inner * n + col] as i32
                        })
                        .sum::<i32>();
                    dot as f32 * activation_scales[group][row] * weight_scales[group][col]
                })
                .sum::<f32>();
            let got = output[row * n + col];
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            let tolerance = 3.0e-5f32.max(expected.abs() * 3.0e-6);
            if !got.is_finite() || error > tolerance {
                mismatches += 1;
                first.get_or_insert((row, col, got, expected, error, tolerance));
            }
        }
    }
    if mismatches != 0 {
        return Err(format!(
            "staged full-K runtime parity failed: mismatches={mismatches} max_abs={max_abs:.9} first={first:?}"
        )
        .into());
    }

    for _ in 0..2 {
        runtime.run_resident_scaled(
            &resident,
            &activation_refs,
            &activation_scale_refs,
            &mut output,
        )?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        runtime.run_resident_scaled(
            &resident,
            &activation_refs,
            &activation_scale_refs,
            &mut output,
        )?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-staged-fullk-runtime M={M} K={} N={n}: mismatches={mismatches} max_abs={max_abs:.9} runtime_ms={dispatch_ms:.6} activation_dma_passes=1 nmacro_replicas=0",
        runtime.k()
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P staged full-K runtime verification is Linux-only");
}
