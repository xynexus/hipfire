//! Hardware parity and resident-dispatch timing for the R17 EmbeddingGemma GeGLU stage.
//! Usage: `npu_geglu_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const INTER: usize = 1152;
    const COMBINED: usize = 2 * INTER;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_geglu_verify CACHE [ITERS]".into());
    }
    let iterations = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(20);
    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut input_bo = kernel.alloc_arg(M * COMBINED * size_of::<f32>())?;
    let output_bo = kernel.alloc_arg(M * INTER * size_of::<f32>())?;

    let probe = std::env::var("HIPFIRE_R17_PROBE").ok();
    let mut input = vec![0.0f32; M * COMBINED];
    for row in 0..M {
        for col in 0..INTER {
            match probe.as_deref() {
                Some("constant") => {
                    input[row * COMBINED + col] = 1.0;
                    input[row * COMBINED + INTER + col] = 1.0;
                }
                Some("layout") => {
                    input[row * COMBINED + col] = 1.0;
                    input[row * COMBINED + INTER + col] = row as f32 + col as f32 / 4096.0;
                }
                _ => {
                    input[row * COMBINED + col] =
                        ((row * 17 + col * 13) as f32 * 0.0031).sin() * 3.5;
                    input[row * COMBINED + INTER + col] =
                        ((row * 11 + col * 19) as f32 * 0.0027).cos() * 2.25;
                }
            }
        }
    }
    input_bo
        .as_mut_slice()
        .copy_from_slice(unsafe { as_bytes(&input) });
    kernel.dispatch_synced(&[&input_bo, &output_bo], &[true, true])?;
    kernel.sync_output(&output_bo)?;
    let output = unsafe { as_f32(output_bo.as_slice()) };

    let mut reference = vec![0.0f32; M * INTER];
    for row in 0..M {
        for col in 0..INTER {
            let gate = input[row * COMBINED + col];
            let up = input[row * COMBINED + INTER + col];
            let gelu =
                0.5 * gate * (1.0 + (0.797_884_6 * (gate + 0.044_715 * gate * gate * gate)).tanh());
            reference[row * INTER + col] = gelu * up;
        }
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut ref_norm = 0.0f64;
    for (&got, &expected) in output.iter().zip(&reference) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        sum_abs += error as f64;
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        ref_norm += (expected as f64).powi(2);
    }
    let mean_abs = sum_abs / output.len() as f64;
    let cosine = dot / (got_norm.sqrt() * ref_norm.sqrt());
    if cosine < 0.9999 || max_abs > 0.05 {
        eprintln!(
            "probe={} cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8}",
            probe.as_deref().unwrap_or("random")
        );
        for row in [0, 1, 2, 7, 8, 9, 15, 16, 24, 31, 32, 63, 64, 127, 128, 255] {
            eprintln!(
                "row={row:3} got=[{:10.5}, {:10.5}, {:10.5}] ref=[{:10.5}, {:10.5}, {:10.5}]",
                output[row * INTER],
                output[row * INTER + 1],
                output[row * INTER + INTER - 1],
                reference[row * INTER],
                reference[row * INTER + 1],
                reference[row * INTER + INTER - 1],
            );
        }
        if probe.as_deref() == Some("layout") {
            let scale = 0.839_843_75f32;
            eprintln!("row map from output[:,0]:");
            for base in (0..M).step_by(32) {
                let mapped = (base..base + 32)
                    .map(|row| format!("{:3}", (output[row * INTER] / scale).round() as i32))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!("out {base:3}..{:3}: {mapped}", base + 31);
            }
        }
        return Err(
            format!("R17 GeGLU parity failed: cosine={cosine:.8} max_abs={max_abs:.7}").into(),
        );
    }

    for _ in 0..2 {
        kernel.dispatch_synced(&[&input_bo, &output_bo], &[false, true])?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(&[&input_bo, &output_bo], &[false, true])?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "resident-geglu M={M} I={INTER}: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
unsafe fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(target_os = "linux")]
unsafe fn as_f32(values: &[u8]) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast(),
            values.len() / std::mem::size_of::<f32>(),
        )
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("XDNA GeGLU verification is Linux-only");
}
