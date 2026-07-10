//! Hardware parity and timing for the R18 fused gate/up projection + GeGLU.
//! Usage: `npu_fused_gateup_geglu_verify FUSED_CACHE PACKING_CACHE w4|w8 [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_xdna::{NpuGemmWholeScaled, NpuKernel};

    const M: usize = 256;
    const K: usize = 768;
    const GROUP: usize = 256;
    const INTER: usize = 1152;
    const PHYSICAL_N: usize = 2 * INTER;
    const PAD_M: usize = 288;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err(
            "usage: npu_fused_gateup_geglu_verify FUSED_CACHE PACKING_CACHE w4|w8 [ITERS]".into(),
        );
    }
    let w4 = match args[2].as_str() {
        "w4" => true,
        "w8" => false,
        mode => return Err(format!("mode must be w4 or w8, got {mode}").into()),
    };
    let iterations = args
        .get(3)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(20);
    let half_stripe = if w4 { 48 } else { 24 };
    let groups = K / GROUP;

    let activations: Vec<i8> = (0..M * K)
        .map(|index| signed_sample(index as u64 ^ 0xa17e_5eed, 63))
        .collect();
    let activation_scales: Vec<f32> = (0..groups)
        .flat_map(|group| (0..M).map(move |row| 0.006 + ((row + group * 7) % 19) as f32 * 0.00013))
        .collect();
    let mut weights = vec![vec![0i8; GROUP * PHYSICAL_N]; groups];
    let mut weight_scales = vec![vec![0.0f32; PHYSICAL_N]; groups];
    for group in 0..groups {
        for physical_col in 0..PHYSICAL_N {
            let (role, logical_col) = decode_physical_col(physical_col, half_stripe);
            weight_scales[group][physical_col] =
                0.004 + ((logical_col + role * 11 + group * 5) % 23) as f32 * 0.00009;
            for kk in 0..GROUP {
                let modulus = if w4 { 15 } else { 127 };
                let seed = kk as u64
                    ^ (logical_col as u64).wrapping_mul(0x9e37_79b9)
                    ^ (role as u64) << 47
                    ^ (group as u64) << 53;
                weights[group][kk * PHYSICAL_N + physical_col] = signed_sample(seed, modulus);
            }
        }
    }

    // Reuse the production physical packer from the equivalent R16 projection
    // shape. It is dropped before the fused kernel is loaded/timed.
    let packer = NpuGemmWholeScaled::load_cached(&args[1])?;
    let packed_a = packer.prepack_activations(&activations, &activation_scales)?;
    let weight_refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let scale_refs = weight_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let packed_w = packer.prepack_weights(&weight_refs, &scale_refs)?;
    let resident_weights = packer.upload_resident_weights(&packed_w)?;
    let mut projected = vec![0.0f32; M * PHYSICAL_N];
    let mut packer = packer;
    packer.run_resident(
        &resident_weights,
        &activations,
        &activation_scales,
        &mut projected,
    )?;
    drop(packer);

    let cpu_reference = reference(
        &activations,
        &activation_scales,
        &weights,
        &weight_scales,
        half_stripe,
    );
    let reference = projected_reference(&projected, half_stripe);
    let (reference_cosine, reference_max_abs, _, _) = metrics(&reference, &cpu_reference);
    if reference_cosine < 0.999999 || reference_max_abs > 1e-4 {
        return Err(format!(
            "packing projection disagrees with CPU oracle: cosine={reference_cosine:.8} max_abs={reference_max_abs:.7}"
        )
        .into());
    }
    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut a = kernel.alloc_arg(packed_a.len())?;
    let mut w = kernel.alloc_arg(packed_w.len())?;
    let padded_n = if w4 { INTER } else { 1536 };
    let o = kernel.alloc_arg(PAD_M * padded_n * size_of::<f32>())?;
    a.as_mut_slice().copy_from_slice(&packed_a);
    w.as_mut_slice().copy_from_slice(&packed_w);
    kernel.sync_to_device(&w)?;
    kernel.dispatch_synced(&[&a, &w, &o], &[true, false, true])?;
    kernel.sync_output(&o)?;
    let output = deblock_output(unsafe { as_f32(o.as_slice()) }, w4);
    let (cosine, max_abs, mean_abs, max_reference) = metrics(&output, &reference);
    let max_allowed = 0.01 + max_reference * 0.01;
    if cosine < 0.9999 || max_abs > max_allowed {
        for index in [0, 1, 7, 8, 15, 16, 23, 24, INTER - 1, INTER, M * INTER - 1] {
            let row = index / INTER;
            let col = index % INTER;
            let gate = projected[row * PHYSICAL_N + physical_col(0, col, half_stripe)];
            let up = projected[row * PHYSICAL_N + physical_col(1, col, half_stripe)];
            eprintln!(
                "output[{index}] got={:.7} ref={:.7} gate={gate:.7} up={up:.7}",
                output[index], reference[index],
            );
        }
        if let Some((index, value)) = output
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        {
            eprintln!("largest output[{index}]={value:.7}");
        }
        return Err(format!(
            "fused gate/up GeGLU parity failed: cosine={cosine:.8} max_abs={max_abs:.7} allowed={max_allowed:.7}"
        )
        .into());
    }

    for _ in 0..2 {
        kernel.dispatch_synced(&[&a, &w, &o], &[false, false, false])?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(&[&a, &w, &o], &[false, false, false])?;
    }
    let fused_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    kernel.sync_output(&o)?;
    println!(
        "fused-gateup-geglu-{} M={M} K={K} I={INTER}: cosine={cosine:.8} max_abs={max_abs:.7} max_allowed={max_allowed:.7} mean_abs={mean_abs:.8} fused_ms={fused_ms:.4}",
        args[2]
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn decode_physical_col(physical: usize, half_stripe: usize) -> (usize, usize) {
    let stripe = physical / (2 * half_stripe);
    let local = physical % (2 * half_stripe);
    if half_stripe == 24 {
        return match local {
            0..=15 => (0, stripe * 24 + local),
            16..=31 => (1, stripe * 24 + local - 16),
            32..=39 => (0, stripe * 24 + 16 + local - 32),
            _ => (1, stripe * 24 + 16 + local - 40),
        };
    }
    if local < half_stripe {
        (0, stripe * half_stripe + local)
    } else {
        (1, stripe * half_stripe + local - half_stripe)
    }
}

#[cfg(target_os = "linux")]
fn signed_sample(mut value: u64, modulus: i16) -> i8 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value % modulus as u64) as i16 - modulus / 2) as i8
}

#[cfg(target_os = "linux")]
fn physical_col(role: usize, logical: usize, half_stripe: usize) -> usize {
    let stripe = logical / half_stripe;
    if half_stripe == 24 {
        let local = logical % 24;
        return stripe * 48
            + if local < 16 {
                role * 16 + local
            } else {
                32 + role * 8 + local - 16
            };
    }
    stripe * 2 * half_stripe + role * half_stripe + logical % half_stripe
}

#[cfg(target_os = "linux")]
fn reference(
    activations: &[i8],
    activation_scales: &[f32],
    weights: &[Vec<i8>],
    weight_scales: &[Vec<f32>],
    half_stripe: usize,
) -> Vec<f32> {
    const M: usize = 256;
    const K: usize = 768;
    const GROUP: usize = 256;
    const INTER: usize = 1152;
    const PHYSICAL_N: usize = 2304;
    let mut output = vec![0.0f32; M * INTER];
    for row in 0..M {
        for col in 0..INTER {
            let mut projected = [0.0f32; 2];
            for (role, value) in projected.iter_mut().enumerate() {
                let pcol = physical_col(role, col, half_stripe);
                for group in 0..K / GROUP {
                    let mut dot = 0i32;
                    for kk in 0..GROUP {
                        dot += activations[row * K + group * GROUP + kk] as i32
                            * weights[group][kk * PHYSICAL_N + pcol] as i32;
                    }
                    *value += dot as f32
                        * activation_scales[group * M + row]
                        * weight_scales[group][pcol];
                }
            }
            let gate = projected[0];
            let up = projected[1];
            let gelu =
                0.5 * gate * (1.0 + (0.797_884_6 * (gate + 0.044_715 * gate.powi(3))).tanh());
            output[row * INTER + col] = gelu * up;
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn projected_reference(projected: &[f32], half_stripe: usize) -> Vec<f32> {
    const M: usize = 256;
    const INTER: usize = 1152;
    const PHYSICAL_N: usize = 2304;
    let mut output = vec![0.0f32; M * INTER];
    for row in 0..M {
        for col in 0..INTER {
            let gate = projected[row * PHYSICAL_N + physical_col(0, col, half_stripe)];
            let up = projected[row * PHYSICAL_N + physical_col(1, col, half_stripe)];
            let gelu =
                0.5 * gate * (1.0 + (0.797_884_6 * (gate + 0.044_715 * gate.powi(3))).tanh());
            output[row * INTER + col] = gelu * up;
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn deblock_output(physical: &[f32], w4: bool) -> Vec<f32> {
    const M: usize = 256;
    const INTER: usize = 1152;
    if w4 {
        return physical[..M * INTER].to_vec();
    }
    const LOGICAL_MACRO_N: usize = 8 * 24;
    const PHYSICAL_MACRO_N: usize = 8 * 32;
    const PADDED_N: usize = 6 * PHYSICAL_MACRO_N;
    let mut output = vec![0.0f32; M * INTER];
    for row in 0..M {
        for col in 0..INTER {
            let n_macro = col / LOGICAL_MACRO_N;
            let within = col % LOGICAL_MACRO_N;
            let stripe = within / 24;
            let local = within % 24;
            output[row * INTER + col] =
                physical[row * PADDED_N + n_macro * PHYSICAL_MACRO_N + stripe * 32 + local];
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64, f32) {
    let mut dot = 0.0;
    let mut got_norm = 0.0;
    let mut ref_norm = 0.0;
    let mut max_abs = 0.0f32;
    let mut max_reference = 0.0f32;
    let mut sum_abs = 0.0;
    for (&got, &expected) in got.iter().zip(expected) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        max_reference = max_reference.max(expected.abs());
        sum_abs += error as f64;
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        ref_norm += (expected as f64).powi(2);
    }
    (
        dot / (got_norm.sqrt() * ref_norm.sqrt()),
        max_abs,
        sum_abs / got.len() as f64,
        max_reference,
    )
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
    eprintln!("fused AIE2P gate/up GeGLU verification is Linux-only");
}
