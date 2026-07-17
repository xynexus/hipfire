//! Live AIE2P parity for Qwen3 BF16 SwiGLU.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuQwen3SwiGlu;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: npu_qwen3_swiglu_verify XCLBIN INSTRUCTIONS M I".into());
    }
    let rows = args[2].parse::<usize>()?;
    let intermediate = args[3].parse::<usize>()?;
    let elements = rows * intermediate;
    let gate = (0..elements)
        .map(|index| {
            // Cover the range observed in a traced Qwen3-Embedding-0.6B
            // forward pass, not just the near-zero interval where every
            // sigmoid approximation looks accurate.
            let phase = (index as f32 * 0.001_731).sin();
            f32_to_bf16_bits((phase + 1.0) * 175.0 - 11.0)
        })
        .collect::<Vec<_>>();
    let up = (0..elements)
        .map(|index| {
            let phase = (index as f32 * 0.002_117).cos();
            f32_to_bf16_bits((phase + 1.0) * 154.0 - 264.0)
        })
        .collect::<Vec<_>>();
    let reference = gate
        .iter()
        .zip(&up)
        .map(|(&gate, &up)| {
            let gate = bf16_bits_to_f32(gate);
            let up = bf16_bits_to_f32(up);
            let silu = bf16_bits_to_f32(f32_to_bf16_bits(gate / (1.0 + (-gate).exp())));
            bf16_bits_to_f32(f32_to_bf16_bits(silu * up))
        })
        .collect::<Vec<_>>();
    let xclbin = std::fs::read(&args[0])?;
    let instructions = std::fs::read(&args[1])?;
    let mut op = NpuQwen3SwiGlu::load(&xclbin, &instructions, rows, intermediate)?;
    let actual_bits = op.run(&gate, &up)?;
    let repeated = op.run(&gate, &up)?;
    if actual_bits != repeated {
        return Err("Qwen3 SwiGLU changed across repeated dispatches".into());
    }
    let independently_loaded =
        NpuQwen3SwiGlu::load(&xclbin, &instructions, rows, intermediate)?.run(&gate, &up)?;
    if independently_loaded != actual_bits {
        return Err("Qwen3 SwiGLU changed across independent image loads".into());
    }
    let actual = actual_bits
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    if let Some(index) = actual.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "Qwen3 SwiGLU produced non-finite output at element {index}: gate={} up={} actual={}",
            bf16_bits_to_f32(gate[index]),
            bf16_bits_to_f32(up[index]),
            actual[index]
        )
        .into());
    }
    let (cosine, max_abs, max_relative) = metrics(&actual, &reference);
    if cosine < 0.9999 || max_relative > 0.01 {
        return Err(
            format!(
                "Qwen3 SwiGLU parity failed: cosine={cosine:.8} max_abs={max_abs:.7} max_relative={max_relative:.7}"
            )
            .into(),
        );
    }
    println!(
        "qwen3-swiglu M={rows} I={intermediate}: cosine={cosine:.8} max_abs={max_abs:.7} max_relative={max_relative:.7} repeat_stable=true"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn metrics(actual: &[f32], reference: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut reference_peak = 0.0f32;
    for (&actual, &reference) in actual.iter().zip(reference) {
        dot += actual as f64 * reference as f64;
        actual_norm += (actual as f64).powi(2);
        reference_norm += (reference as f64).powi(2);
        max_abs = max_abs.max((actual - reference).abs());
        reference_peak = reference_peak.max(reference.abs());
    }
    (
        (dot / (actual_norm.sqrt() * reference_norm.sqrt())) as f32,
        max_abs,
        max_abs / reference_peak.max(f32::MIN_POSITIVE),
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_swiglu_verify is Linux-only");
}
