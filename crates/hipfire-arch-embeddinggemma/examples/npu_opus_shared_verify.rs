//! Exact GPU -> shared dma-buf -> AIE2P -> shared dma-buf -> GPU Opus parity.
//!
//! Usage: `npu_opus_shared_verify CACHE w4|w8 K N [--awq] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_primitives::fwht::gen_fwht_signs;
    use hipfire_rdna::{DType, Gpu};
    use hipfire_runtime::quant::f32_to_f16;
    use hipfire_xdna::NpuOpusGemmMp;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err("usage: npu_opus_shared_verify CACHE w4|w8 K N [--awq] [--iters N]".into());
    }
    let cache = &args[0];
    if !Path::new(cache).join("final.xclbin").is_file() {
        return Err(format!("missing cache {cache}").into());
    }
    let (encoding, quant_type, block_bytes) = match args[1].as_str() {
        "w4" => ("w4", 34u8, 130usize),
        "w8" => ("w8", 35u8, 258usize),
        other => return Err(format!("encoding must be w4 or w8, got {other}").into()),
    };
    let k: usize = args[2].parse()?;
    let n: usize = args[3].parse()?;
    let use_awq = args.iter().any(|arg| arg == "--awq");
    let iterations = args
        .iter()
        .position(|arg| arg == "--iters")
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10);

    let groups = k.div_ceil(256);
    let mut payload = vec![0u8; n * groups * block_bytes];
    for col in 0..n {
        for group in 0..groups {
            let block = &mut payload
                [(col * groups + group) * block_bytes..(col * groups + group + 1) * block_bytes];
            let scale = 0.007 + ((col + group * 3) % 31) as f32 * 0.00009;
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            if quant_type == 35 {
                for inner in 0..256 {
                    block[2 + inner] =
                        (((inner * 29 + col + group * 7) % 241) as i16 - 120) as i8 as u8;
                }
            } else {
                for packed in 0..128 {
                    let low = ((packed + col + group) % 15) as i8 - 7;
                    let high = ((packed * 3 + col + group) % 15) as i8 - 7;
                    block[2 + packed] = (low as u8 & 0x0f) | ((high as u8 & 0x0f) << 4);
                }
            }
        }
    }
    let awq = use_awq.then(|| {
        (0..k)
            .map(|index| 0.75 + (index % 17) as f32 * 0.025)
            .collect::<Vec<_>>()
    });
    let mut gemm =
        NpuOpusGemmMp::load_whole_scaled_only(cache, quant_type, k, n, &payload, awq.clone())?;
    let layout = gemm
        .whole_scaled_io_layout()
        .ok_or("cache did not select scaled whole-array execution")?;
    let gpu_layout = rdna_layout(layout);
    let m = layout.rows();
    let input: Vec<f32> = (0..m * k)
        .map(|index| ((index as f32 * 0.013).sin() * 2.0) + ((index % 7) as f32 - 3.0) * 0.1)
        .collect();
    let reference = gemm.reference_f32(m, &input)?;

    let mut gpu = Gpu::init()?;
    let input_gpu = gpu.upload_owned_f32(&input, &[m * k])?;
    let awq_gpu = awq
        .as_deref()
        .map(|scale| gpu.upload_owned_f32(scale, &[scale.len()]))
        .transpose()?;
    let output_gpu = gpu.alloc_owned(&[m * n], DType::F32)?;
    let mut input_shared = gpu.alloc_shared_gtt(layout.input_bytes())?;
    let mut output_shared = gpu.alloc_shared_gtt(layout.output_bytes())?;
    input_shared.as_mut_slice().fill(0);
    output_shared.as_mut_slice().fill(0);
    let input_import = gpu.import_dmabuf(
        input_shared.dmabuf_fd(),
        layout.input_bytes(),
        &[layout.input_bytes()],
        DType::Raw,
    )?;
    let output_import = gpu.import_dmabuf(
        output_shared.dmabuf_fd(),
        layout.output_bytes(),
        &[layout.output_bytes()],
        DType::Raw,
    )?;
    gemm.attach_whole_scaled_shared_io(
        input_shared.dmabuf_fd(),
        layout.input_bytes(),
        output_shared.dmabuf_fd(),
        layout.output_bytes(),
    )?;

    let run = |gpu: &mut Gpu, gemm: &mut NpuOpusGemmMp| -> Result<(), Box<dyn std::error::Error>> {
        gpu.pack_opus_npu_activations(
            &input_gpu,
            awq_gpu.as_ref().map(|scale| scale.view()),
            &input_import.view(),
            m,
            k,
            gpu_layout,
        )?;
        gpu.device_synchronize()?;
        gemm.run_whole_scaled_shared()?;
        gpu.unpack_opus_npu_output(
            &output_import.view(),
            &output_gpu,
            n,
            None,
            None,
            m,
            gpu_layout,
        )?;
        gpu.device_synchronize()?;
        Ok(())
    };
    run(&mut gpu, &mut gemm)?;
    validate_packed_activations(
        input_shared.as_slice(),
        &input,
        awq.as_deref(),
        k,
        layout,
        &gen_fwht_signs(42, 256),
        &gen_fwht_signs(1042, 256),
    );
    let output = gpu.download_f32(&output_gpu)?;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_mismatch = None;
    for (index, (got, expected)) in output.iter().zip(&reference).enumerate() {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        if error > 1e-4 + expected.abs() * 1e-5 {
            mismatches += 1;
            first_mismatch.get_or_insert((index, *got, *expected, error));
        }
    }
    if let Some((index, got, expected, error)) = first_mismatch {
        eprintln!(
            "first mismatch index={index} got={got:.7} expected={expected:.7} abs={error:.7}"
        );
    }

    for _ in 0..2 {
        run(&mut gpu, &mut gemm)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        run(&mut gpu, &mut gemm)?;
    }
    let wrapper_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "shared-opus-{encoding}{} M={m} K={k} padded_K={} N={n}: mismatches={mismatches} max_abs={max_abs:.7} wrapper_ms={wrapper_ms:.4}",
        if use_awq { "+/++" } else { "" },
        layout.k(),
    );
    assert_eq!(mismatches, 0, "shared Opus projection parity failed");
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_packed_activations(
    packed: &[u8],
    input: &[f32],
    awq: Option<&[f32]>,
    k: usize,
    layout: hipfire_xdna::NpuWholeScaledIoLayout,
    signs1: &[f32],
    signs2: &[f32],
) {
    use hipfire_primitives::fwht::cpu_fwht_256;

    let (inner_k, mr) = match layout.mode() {
        hipfire_xdna::NpuWholeMode::W4 => (16, 4),
        hipfire_xdna::NpuWholeMode::W8 => (8, 8),
    };
    let mut mismatches = 0usize;
    let mut first = None;
    let mut max_scale_error = 0.0f32;
    let inblocks = layout.outblocks() * layout.groups();
    for row in 0..layout.rows() {
        for group in 0..layout.groups() {
            let mut rotated = [0.0f32; 256];
            for (inner, value) in rotated.iter_mut().enumerate() {
                let col = group * 256 + inner;
                if col < k {
                    *value = awq.map_or(input[row * k + col], |scale| {
                        input[row * k + col] / scale[col]
                    });
                }
            }
            cpu_fwht_256(&mut rotated, signs1, signs2);
            let scale = rotated.iter().fold(0.0f32, |a, b| a.max(b.abs())) / 127.0;
            let scale = if scale > 0.0 { scale } else { 1.0 };
            let m_macro = row / 96;
            let within_macro = row % 96;
            let stripe = within_macro / 24;
            let within_stripe = within_macro % 24;
            let lm = within_stripe / mr;
            let local_row = within_stripe % mr;
            let outblock = m_macro * layout.n_macros();
            let block = outblock * layout.groups() + group;
            let base = (stripe * inblocks + block) * 8192;
            let got_scale = f32::from_ne_bytes(
                packed[base + 6144 + within_stripe * 4..base + 6144 + within_stripe * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            max_scale_error = max_scale_error.max((got_scale - scale).abs());
            for inner in 0..256 {
                let kt = inner / inner_k;
                let kk = inner % inner_k;
                let target = (lm * (256 / inner_k) + kt) * 64 + local_row * inner_k + kk;
                let got = packed[base + target] as i8;
                let expected = (rotated[inner] / scale).round().clamp(-127.0, 127.0) as i8;
                if got != expected {
                    mismatches += 1;
                    first.get_or_insert((row, group, inner, got, expected));
                }
            }
        }
    }
    eprintln!(
        "packed activation oracle: mismatches={mismatches} max_scale_abs={max_scale_error:.9} first={first:?}"
    );
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

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("EmbeddingGemma XDNA shared Opus verification is Linux-only");
}
