//! Exact hardware gate for resident AWQ/FWHT/int8 activation preprocessing.
//! Usage: `npu_fwht_quant_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::fwht::gen_fwht_signs;
    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const K: usize = 1152;
    const PAD_K: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 5;
    const PARAM: usize = PAD_K + 2 * GROUP;
    const ROW_OUT: usize = PAD_K + 8 * size_of::<f32>();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_fwht_quant_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(20);

    let mut input = vec![0.0f32; M * PAD_K];
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
    let mut param = Vec::with_capacity(PARAM);
    param.extend_from_slice(&awq);
    param.extend_from_slice(&signs1);
    param.extend_from_slice(&signs2);

    let (reference_q, reference_scales) = reference(&input, &awq, &signs1, &signs2);
    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut x = kernel.alloc_arg(input.len() * size_of::<f32>())?;
    let mut p = kernel.alloc_arg(param.len() * size_of::<f32>())?;
    let o = kernel.alloc_arg(M * ROW_OUT)?;
    x.as_mut_slice()
        .copy_from_slice(unsafe { as_bytes(&input) });
    p.as_mut_slice()
        .copy_from_slice(unsafe { as_bytes(&param) });
    kernel.sync_to_device(&p)?;
    kernel.dispatch_synced(&[&x, &p, &o], &[true, false, true])?;
    kernel.sync_output(&o)?;

    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    let mut max_scale_abs = 0.0f32;
    for row in 0..M {
        let base = row * ROW_OUT;
        for col in 0..PAD_K {
            let got = o.as_slice()[base + col] as i8;
            let expected = reference_q[row * PAD_K + col];
            if got != expected {
                mismatches += 1;
                first_mismatch.get_or_insert((row, col, got, expected));
            }
        }
        for group in 0..GROUPS {
            let offset = base + PAD_K + group * size_of::<f32>();
            let got = f32::from_ne_bytes(o.as_slice()[offset..offset + 4].try_into()?);
            let expected = reference_scales[row * GROUPS + group];
            max_scale_abs = max_scale_abs.max((got - expected).abs());
        }
    }
    if let Some((row, col, got, expected)) = first_mismatch {
        eprintln!("first quant mismatch row={row} col={col} got={got} expected={expected}");
    }
    if mismatches != 0 || max_scale_abs > 1e-7 {
        eprintln!("row0 q bytes: {:?}", &o.as_slice()[..32]);
        let scale_bytes = &o.as_slice()[PAD_K..PAD_K + 8 * size_of::<f32>()];
        let scale_values = scale_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        eprintln!("row0 scales: {scale_values:?}");
        let nonzero = o.as_slice().iter().filter(|&&value| value != 0).count();
        eprintln!("nonzero output bytes: {nonzero}/{}", o.len());
        return Err(format!(
            "R19 parity failed: q_mismatches={mismatches} max_scale_abs={max_scale_abs:.9}"
        )
        .into());
    }

    for _ in 0..2 {
        kernel.dispatch_synced(&[&x, &p, &o], &[false, false, false])?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(&[&x, &p, &o], &[false, false, false])?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "resident-awq-fwht-quant M={M} K={K} padded_K={PAD_K}: q_mismatches=0 max_scale_abs={max_scale_abs:.9} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn reference(input: &[f32], awq: &[f32], signs1: &[f32], signs2: &[f32]) -> (Vec<i8>, Vec<f32>) {
    use hipfire_primitives::fwht::cpu_fwht_256;

    const M: usize = 256;
    const PAD_K: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 5;
    let mut q = vec![0i8; M * PAD_K];
    let mut scales = vec![0.0f32; M * GROUPS];
    for row in 0..M {
        for group in 0..GROUPS {
            let base = group * GROUP;
            let mut rotated = vec![0.0f32; GROUP];
            for i in 0..GROUP {
                rotated[i] = input[row * PAD_K + base + i] / awq[base + i];
            }
            cpu_fwht_256(&mut rotated, signs1, signs2);
            let max_abs = rotated
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            scales[row * GROUPS + group] = scale;
            if scale > 0.0 {
                for i in 0..GROUP {
                    q[row * PAD_K + base + i] =
                        (rotated[i] / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (q, scales)
}

#[cfg(target_os = "linux")]
unsafe fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P FWHT quant verification is Linux-only");
}
