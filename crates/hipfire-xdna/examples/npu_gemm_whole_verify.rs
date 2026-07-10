//! Exact patterned parity and timing for the AIE2P 4x4 whole-array W4 path.
//!
//! Usage: `npu_gemm_whole_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::NpuGemmWholeArray;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_gemm_whole_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(20usize);
    let mut gemm = NpuGemmWholeArray::load_cached(&args[0])?;
    let (m, k, n) = (gemm.rows(), gemm.k(), gemm.n());
    let groups = k / 256;

    let activations: Vec<i8> = (0..m * k)
        .map(|index| ((index * 17 + index / k * 3) % 15) as i8 - 7)
        .collect();
    let matrices: Vec<Vec<i8>> = (0..groups)
        .map(|group| {
            (0..256 * n)
                .map(|index| ((index * 11 + index / n * 5 + group * 7) % 15) as i8 - 7)
                .collect()
        })
        .collect();
    let refs: Vec<&[i8]> = matrices.iter().map(Vec::as_slice).collect();
    let packed = gemm.prepack_weights(&refs)?;
    let resident = gemm.upload_resident_weights(&packed)?;
    let mut partials = vec![0i32; groups * m * n];
    gemm.run_resident(&resident, &activations, &mut partials)?;

    let mut mismatches = 0usize;
    let mut first = None;
    for group in 0..groups {
        for row in 0..m {
            for col in 0..n {
                let expected = (0..256)
                    .map(|inner| {
                        activations[row * k + group * 256 + inner] as i32
                            * matrices[group][inner * n + col] as i32
                    })
                    .sum::<i32>();
                let got = partials[(group * m + row) * n + col];
                if got != expected {
                    mismatches += 1;
                    first.get_or_insert((group, row, col, got, expected));
                }
            }
        }
    }
    println!(
        "whole-{:?} M={m} K={k} N={n}: mismatches={mismatches}",
        gemm.mode()
    );
    if let Some((group, row, col, got, expected)) = first {
        println!("first_mismatch group={group} row={row} col={col} got={got} expected={expected}");
    }
    if mismatches != 0 {
        return Err("whole-array AIE2P patterned parity failed".into());
    }

    for _ in 0..3 {
        gemm.run_resident(&resident, &activations, &mut partials)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        gemm.run_resident(&resident, &activations, &mut partials)?;
    }
    let seconds = started.elapsed().as_secs_f64() / iterations as f64;
    let logical_macs = m as f64 * k as f64 * n as f64;
    let physical_macs = m.div_ceil(96) as f64 * 96.0 * k as f64 * n.div_ceil(384) as f64 * 384.0;
    println!(
        "iters={iterations} wrapper_ms={:.4} logical_tops={:.4} physical_tops={:.4}",
        seconds * 1e3,
        2.0 * logical_macs / seconds / 1e12,
        2.0 * physical_macs / seconds / 1e12
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_gemm_whole_verify is Linux-only");
}
