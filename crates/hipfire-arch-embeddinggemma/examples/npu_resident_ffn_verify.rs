//! Verify the resident AIE2P gate/up projection -> GeGLU handoff.
//!
//! The two kernels import the same row-major dma-buf for the intermediate, so
//! no CPU or GPU operation materializes gate/up between the AIE dispatches.
//! Usage: `npu_resident_ffn_verify PROJECTION_CACHE GEGLU_CACHE w4|w8 [--awq] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_rdna::{DType, Gpu};
    use hipfire_runtime::quant::f32_to_f16;
    use hipfire_xdna::{NpuGeGlu, NpuOpusGemmMp};

    const M: usize = 256;
    const K: usize = 768;
    const INTER: usize = 1152;
    const N: usize = 2 * INTER;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "usage: npu_resident_ffn_verify PROJECTION_CACHE GEGLU_CACHE w4|w8 [--awq] [--iters N]"
                .into(),
        );
    }
    let (quant_type, block_bytes) = match args[2].as_str() {
        "w4" => (34u8, 130usize),
        "w8" => (35u8, 258usize),
        mode => return Err(format!("mode must be w4 or w8, got {mode}").into()),
    };
    let use_awq = args.iter().any(|arg| arg == "--awq");
    let iterations = args
        .iter()
        .position(|arg| arg == "--iters")
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);

    let groups = K.div_ceil(256);
    let mut payload = vec![0u8; N * groups * block_bytes];
    for col in 0..N {
        for group in 0..groups {
            let block = &mut payload
                [(col * groups + group) * block_bytes..(col * groups + group + 1) * block_bytes];
            let scale = 0.006 + ((col + group * 5) % 29) as f32 * 0.00011;
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            if quant_type == 35 {
                for inner in 0..256 {
                    block[2 + inner] =
                        (((inner * 31 + col * 3 + group * 7) % 251) as i16 - 125) as i8 as u8;
                }
            } else {
                for packed in 0..128 {
                    let low = ((packed + col + group) % 15) as i8 - 7;
                    let high = ((packed * 3 + col + group * 2) % 15) as i8 - 7;
                    block[2 + packed] = (low as u8 & 0x0f) | ((high as u8 & 0x0f) << 4);
                }
            }
        }
    }
    let awq = use_awq.then(|| {
        (0..K)
            .map(|index| 0.75 + (index % 17) as f32 * 0.025)
            .collect::<Vec<_>>()
    });
    let mut projection =
        NpuOpusGemmMp::load_whole_scaled_only(&args[0], quant_type, K, N, &payload, awq.clone())?;
    let layout = projection
        .whole_scaled_io_layout()
        .ok_or("projection cache did not select scaled whole-array execution")?;
    if !layout.row_major_output() || layout.n() != N || layout.padded_n() != N {
        return Err("projection cache must emit contiguous row-major [M,2304]".into());
    }
    let mut geglu = NpuGeGlu::load_cached(&args[1], M, INTER)?;

    let input: Vec<f32> = (0..M * K)
        .map(|index| ((index as f32 * 0.0091).sin() * 1.7) + (index % 11) as f32 * 0.025)
        .collect();
    let gate_up_reference = projection.reference_f32(M, &input)?;
    let reference = geglu_reference(&gate_up_reference, M, INTER);

    let mut gpu = Gpu::init()?;
    let input_gpu = gpu.upload_owned_f32(&input, &[M * K])?;
    let awq_gpu = awq
        .as_deref()
        .map(|scale| gpu.upload_owned_f32(scale, &[scale.len()]))
        .transpose()?;
    let mut packed_input = gpu.alloc_shared_gtt(layout.input_bytes())?;
    let mut gate_up = gpu.alloc_shared_gtt(layout.output_bytes())?;
    let mut ffn = gpu.alloc_shared_gtt(geglu.output_bytes())?;
    packed_input.as_mut_slice().fill(0);
    gate_up.as_mut_slice().fill(0);
    ffn.as_mut_slice().fill(0);
    let packed_input_gpu = gpu.import_dmabuf(
        packed_input.dmabuf_fd(),
        layout.input_bytes(),
        &[layout.input_bytes()],
        DType::Raw,
    )?;
    let ffn_gpu = gpu.import_dmabuf(ffn.dmabuf_fd(), ffn.len(), &[M * INTER], DType::F32)?;
    projection.attach_whole_scaled_shared_io(
        packed_input.dmabuf_fd(),
        layout.input_bytes(),
        gate_up.dmabuf_fd(),
        layout.output_bytes(),
    )?;
    gpu.pack_opus_npu_activations(
        &input_gpu,
        awq_gpu.as_ref().map(|scale| scale.view()),
        &packed_input_gpu.view(),
        M,
        K,
        rdna_layout(layout),
    )?;
    gpu.device_synchronize()?;
    projection.run_whole_scaled_shared_to_device()?;
    geglu.attach_shared_io(
        gate_up.dmabuf_fd(),
        layout.output_bytes(),
        ffn.dmabuf_fd(),
        ffn.len(),
    )?;
    geglu.run_shared()?;
    let output = gpu.download_f32(&ffn_gpu.view())?;
    let (cosine, max_abs, mean_abs) = metrics(&output, &reference);
    let max_reference = reference
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let max_allowed = 0.01 + max_reference * 0.01;
    if cosine < 0.9999 || max_abs > max_allowed {
        let gate_up_output = &unsafe { as_f32(gate_up.as_slice()) }[..M * N];
        let (projection_cosine, projection_max_abs, projection_mean_abs) =
            metrics(gate_up_output, &gate_up_reference);
        eprintln!(
            "projection intermediate: cosine={projection_cosine:.8} max_abs={projection_max_abs:.7} mean_abs={projection_mean_abs:.8}"
        );
        for index in [0, 1, INTER - 1, INTER, N - 1, N, M * N - 1] {
            eprintln!(
                "gate_up[{index}] got={:.7} ref={:.7}",
                gate_up_output[index], gate_up_reference[index]
            );
        }
        for index in [0, 1, INTER - 1, INTER, M * INTER - 1] {
            eprintln!(
                "ffn[{index}] got={:.7} ref={:.7}",
                output[index], reference[index]
            );
        }
        let nonzero = output.iter().filter(|value| value.abs() > 1e-12).count();
        let first_nonzero = output
            .iter()
            .enumerate()
            .filter(|(_, value)| value.abs() > 1e-12)
            .take(16)
            .map(|(index, value)| format!("{index}:{value:.5}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "ffn nonzero={nonzero}/{} first={first_nonzero}",
            output.len()
        );
        return Err(format!(
            "resident FFN handoff parity failed: cosine={cosine:.8} max_abs={max_abs:.7} allowed={max_allowed:.7}"
        )
        .into());
    }

    // Change every producer value and rerun through the same imported BOs. This
    // catches a falsely-correct chain that only rereads the first coherent
    // cache image across the two XDNA contexts.
    let input_refresh: Vec<f32> = (0..M * K)
        .map(|index| ((index as f32 * 0.0073).cos() * 1.3) - (index % 13) as f32 * 0.019)
        .collect();
    let refresh_gate_up = projection.reference_f32(M, &input_refresh)?;
    let refresh_reference = geglu_reference(&refresh_gate_up, M, INTER);
    let refresh_gpu = gpu.upload_owned_f32(&input_refresh, &[M * K])?;
    gpu.pack_opus_npu_activations(
        &refresh_gpu,
        awq_gpu.as_ref().map(|scale| scale.view()),
        &packed_input_gpu.view(),
        M,
        K,
        rdna_layout(layout),
    )?;
    gpu.device_synchronize()?;
    projection.run_whole_scaled_shared_to_device()?;
    geglu.run_shared()?;
    let refresh_output = gpu.download_f32(&ffn_gpu.view())?;
    let (refresh_cosine, refresh_max_abs, _) = metrics(&refresh_output, &refresh_reference);
    let refresh_max_reference = refresh_reference
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let refresh_allowed = 0.01 + refresh_max_reference * 0.01;
    if refresh_cosine < 0.9999 || refresh_max_abs > refresh_allowed {
        return Err(format!(
            "resident FFN refresh parity failed: cosine={refresh_cosine:.8} max_abs={refresh_max_abs:.7} allowed={refresh_allowed:.7}"
        )
        .into());
    }

    for _ in 0..2 {
        projection.run_whole_scaled_shared_to_device()?;
        geglu.run_shared()?;
    }
    let started = Instant::now();
    let mut projection_seconds = 0.0;
    let mut geglu_seconds = 0.0;
    for _ in 0..iterations {
        let stage = Instant::now();
        projection.run_whole_scaled_shared_to_device()?;
        projection_seconds += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        geglu.run_shared()?;
        geglu_seconds += stage.elapsed().as_secs_f64();
    }
    let chain_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let projection_ms = projection_seconds * 1e3 / iterations as f64;
    let geglu_ms = geglu_seconds * 1e3 / iterations as f64;
    println!(
        "resident-ffn-{}{} M={M} K={K} gate_up_N={N}: cosine={cosine:.8} max_abs={max_abs:.7} refresh_cosine={refresh_cosine:.8} refresh_max_abs={refresh_max_abs:.7} max_allowed={max_allowed:.7} mean_abs={mean_abs:.8} projection_ms={projection_ms:.4} geglu_ms={geglu_ms:.4} chain_ms={chain_ms:.4}",
        args[2],
        if use_awq { "+/++" } else { "" },
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn rdna_layout(layout: hipfire_xdna::NpuWholeScaledIoLayout) -> hipfire_rdna::OpusNpuIoLayout {
    hipfire_rdna::OpusNpuIoLayout::new(
        layout.mode() == hipfire_xdna::NpuWholeMode::W8,
        layout.cols(),
        layout.rows(),
        layout.groups(),
        layout.n(),
        layout.n_macros(),
        layout.outblocks(),
        8192,
        layout.input_bytes(),
        layout.output_bytes(),
        layout.row_major_output(),
        layout.padded_n(),
    )
}

#[cfg(target_os = "linux")]
fn geglu_reference(gate_up: &[f32], rows: usize, intermediate: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * intermediate];
    for row in 0..rows {
        for col in 0..intermediate {
            let gate = gate_up[row * 2 * intermediate + col];
            let up = gate_up[row * 2 * intermediate + intermediate + col];
            let gelu =
                0.5 * gate * (1.0 + (0.797_884_6 * (gate + 0.044_715 * gate.powi(3))).tanh());
            output[row * intermediate + col] = gelu * up;
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let mut dot = 0.0;
    let mut got_norm = 0.0;
    let mut ref_norm = 0.0;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0;
    for (&got, &expected) in got.iter().zip(expected) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        sum_abs += error as f64;
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        ref_norm += (expected as f64).powi(2);
    }
    (
        dot / (got_norm.sqrt() * ref_norm.sqrt()),
        max_abs,
        sum_abs / got.len() as f64,
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
    eprintln!("resident AIE2P FFN verification is Linux-only");
}
