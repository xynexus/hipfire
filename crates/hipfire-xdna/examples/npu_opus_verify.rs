//! Hardware parity for generic W4, mixed, and W8 Opus AIE2P caches.
//!
//! Usage:
//! `npu_opus_verify <w4-cache> <w8-cache> <sparse3-cache> <N> \
//!    [--encoding w4|mixed|w8] [--outliers N] [--awq]`

#[cfg(target_os = "linux")]
fn main() {
    use hipfire_primitives::conv::f32_to_f16;
    use hipfire_xdna::NpuOpusGemmMp;

    let mut args = std::env::args().skip(1);
    let w4_cache = args.next().expect("w4 cache path");
    let w8_cache = args.next().expect("w8 cache path");
    let sparse3_cache = args.next().expect("sparse3 cache path");
    let n: usize = args.next().expect("N").parse().expect("numeric N");
    let options: Vec<String> = args.collect();
    let use_awq = options.iter().any(|arg| arg == "--awq");
    let encoding = options
        .iter()
        .position(|arg| arg == "--encoding")
        .and_then(|index| options.get(index + 1))
        .map(String::as_str)
        .unwrap_or("mixed");
    let outlier_count = options
        .iter()
        .position(|arg| arg == "--outliers")
        .and_then(|index| options.get(index + 1))
        .map(|value| value.parse::<usize>().expect("numeric outlier count"))
        .unwrap_or(3);
    assert!((1..=255).contains(&outlier_count));
    let k = 256usize;
    let (quant_type, block_bytes) = match encoding {
        "w4" => (34u8, 130usize),
        "mixed" => (36u8, 130 + 2 * outlier_count),
        "w8" => (35u8, 258usize),
        other => panic!("unknown --encoding {other}; want w4|mixed|w8"),
    };
    let mut payload = vec![0u8; n * block_bytes];
    for col in 0..n {
        let block = &mut payload[col * block_bytes..(col + 1) * block_bytes];
        let scale = 0.01 + (col % 11) as f32 * 0.0005;
        block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        if encoding == "w8" {
            for inner in 0..256 {
                block[2 + inner] = (((inner * 29 + col) % 241) as i16 - 120) as i8 as u8;
            }
        } else {
            for packed_idx in 0..128 {
                let low = ((packed_idx + col) % 15) as i8 - 7;
                let high = ((packed_idx * 3 + col) % 15) as i8 - 7;
                block[2 + packed_idx] = (low as u8 & 0x0f) | ((high as u8 & 0x0f) << 4);
            }
            if encoding == "mixed" {
                for index in 0..outlier_count {
                    let position = (index * 47 % 256) as u8;
                    let replacement = ((index * 29 + col) % 201) as i16 - 100;
                    block[130 + 2 * index] = position;
                    block[131 + 2 * index] = replacement as i8 as u8;
                }
            }
        }
    }
    let awq_scale = use_awq.then(|| {
        (0..k)
            .map(|index| 0.75 + (index % 17) as f32 * 0.025)
            .collect::<Vec<_>>()
    });
    let mut gemm = NpuOpusGemmMp::load_cached(
        &w4_cache,
        &w8_cache,
        &sparse3_cache,
        quant_type,
        k,
        n,
        &payload,
        awq_scale,
    )
    .expect("load Opus matrix");
    let m = gemm.rows_per_dispatch();
    let x: Vec<f32> = (0..m * k)
        .map(|index| ((index as f32 * 0.013).sin() * 2.0) + ((index % 7) as f32 - 3.0) * 0.1)
        .collect();
    let reference = gemm.reference_f32(m, &x).expect("CPU reference");
    let mut output = vec![0.0f32; m * n];
    gemm.run_f32(m, &x, &mut output).expect("NPU mixed Opus");

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for (got, expected) in output.iter().zip(&reference) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        let tolerance = 1e-4 + expected.abs() * 1e-5;
        if error > tolerance {
            mismatches += 1;
        }
    }
    println!(
        "opus-{encoding}{} bits={:.4} outliers={} sparse_dispatches={} M={m} K={k} N={n}: mismatches={mismatches} max_abs={max_abs:.6}",
        if use_awq { "+/++" } else { "" },
        block_bytes as f32 * 8.0 / 256.0,
        if encoding == "mixed" { outlier_count } else { 0 },
        if encoding == "mixed" { outlier_count.div_ceil(3) } else { 0 },
    );
    assert_eq!(mismatches, 0, "NPU mixed Opus parity failed");
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("amdxdna is Linux-only");
}
