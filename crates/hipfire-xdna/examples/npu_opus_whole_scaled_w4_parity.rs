//! Hardware parity for a resident **W4A8 op4++ projection GEMM** at an arbitrary
//! `(K, N)` via the `whole8_w4-scaled` array kernel — the NPU parity leg (L2) at
//! shapes the baked EmbeddingGemma FFN cannot reach.
//!
//! Milestone M2/path-a (`docs/plans/2026-07-17-npu-w4a8-op4pp-moe-qwen35.md`):
//! `NpuOpusExecutor::run_f32` does `C[M,N] = X[M,K]·Wᵀ` with on-device AWQ →
//! FWHT-256 → int8-activation quant → integer dot → scale (whole8 output
//! deblocking handled internally), and is checked against the shared
//! `OpusPackedMatrix::reference_f32` op4++ oracle. Point it at
//! `embgemma_aie2p_whole8_w4-scaled_m256_kg3_n2304` to verify the **Qwen3.5-A3B
//! down projection** (K=768 → N=2048, covered by N=2304): a `--down` view reports
//! parity over just the first 2048 output columns.
//!
//! Serialize with the NPU: hold `hipfire lock` while running.
//!
//! Usage: `npu_opus_whole_scaled_w4_parity CACHE [--down N] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::f32_to_f16;
    use hipfire_xdna::{NpuOpusExecutor, OpusPackedMatrix};

    const GROUP: usize = 256;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let cache = args
        .first()
        .filter(|value| !value.starts_with("--"))
        .ok_or("usage: npu_opus_whole_scaled_w4_parity CACHE [--down N] [--iters N]")?;
    let opt = |key: &str| -> Option<usize> {
        args.iter()
            .position(|value| value == key)
            .and_then(|index| args.get(index + 1))
            .and_then(|value| value.parse().ok())
    };
    let iterations = opt("--iters").unwrap_or(5);

    let shape = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    let field = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
        shape
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .ok_or_else(|| format!("cache shape.txt missing {key}").into())
            .and_then(|value| value.parse::<usize>().map_err(Into::into))
    };
    let (m, k, n) = (field("m")?, field("k")?, field("n")?);
    if !shape.lines().any(|line| line == "mode=w4-scaled") {
        return Err("cache is not a whole-array w4-scaled kernel".into());
    }
    let down = opt("--down").unwrap_or(n).min(n);
    assert_eq!(k % GROUP, 0, "K must be % 256");
    let groups = k / GROUP;

    // op4++ weights in the qt=33 W4 on-disk layout (130-byte blocks: fp16 scale
    // + 128 nibbles, low = inner 2j, high = inner 2j+1), block order col-major.
    let mut payload = vec![0u8; n * groups * 130];
    for col in 0..n {
        for g in 0..groups {
            let block = &mut payload[(col * groups + g) * 130..(col * groups + g + 1) * 130];
            let scale = 0.012 * (1.0 + ((col + 3 * g) % 7) as f32 * 0.03);
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for j in 0..128 {
                let val = |inner: usize| -> u8 {
                    let mixed = (inner as u64).wrapping_mul(0x9e37_79b1)
                        ^ (col as u64).wrapping_mul(0x85eb_ca77)
                        ^ (g as u64).wrapping_mul(0xc2b2_ae3d);
                    (((mixed % 15) as i8 - 7) as u8) & 0x0f
                };
                block[2 + j] = val(g * GROUP + 2 * j) | (val(g * GROUP + 2 * j + 1) << 4);
            }
        }
    }

    let x = (0..m * k)
        .map(|i| ((i * 19 + i / 31) % 101) as f32 / 100.0 - 0.5)
        .collect::<Vec<_>>();

    let mut executor = NpuOpusExecutor::load_whole_scaled_cached(&[cache], n)?;
    let matrix = executor.pack_matrix(33, k, n, &payload, None)?;
    // Independent oracle from the same nibbles+scales.
    let oracle_matrix = OpusPackedMatrix::from_payload(33, k, n, &payload, None)?;
    let reference = oracle_matrix.reference_f32(m, &x)?;

    let mut output = vec![0.0f32; m * n];
    executor.run_f32(&matrix, m, &x, &mut output)?;

    let view = |buf: &[f32]| -> Vec<f32> {
        // First `down` of `n` columns, all m rows.
        (0..m)
            .flat_map(|r| buf[r * n..r * n + down].to_vec())
            .collect()
    };
    let (cosine, max_abs, db) = metrics(&view(&output), &view(&reference));
    let pass = cosine >= 0.999 && db > 30.0;
    if !pass {
        for idx in [0usize, 1, down - 1, n, m * n - 1] {
            eprintln!(
                "out[{idx}] got={:.6} ref={:.6}",
                output[idx], reference[idx]
            );
        }
        return Err(format!("W4 whole-scaled parity FAILED: cosine={cosine:.8} db={db:.2}").into());
    }

    for _ in 0..2 {
        executor.run_f32(&matrix, m, &x, &mut output)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        executor.run_f32(&matrix, m, &x, &mut output)?;
    }
    let ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "npu_opus_whole_scaled_w4 M={m} K={k} N={n} (view first {down} cols): cosine={cosine:.8} max_abs={max_abs:.6} SQNR={db:.2} dB dispatch_ms={ms:.4} -> PASS"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let mut dot = 0.0;
    let (mut gn, mut en) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&g, &e) in got.iter().zip(expected) {
        max_abs = max_abs.max((g - e).abs());
        dot += g as f64 * e as f64;
        gn += (g as f64).powi(2);
        en += (e as f64).powi(2);
        sig += (e as f64).powi(2);
        noise += ((e - g) as f64).powi(2);
    }
    (
        dot / (gn.sqrt() * en.sqrt()),
        max_abs,
        10.0 * (sig / noise.max(1e-30)).log10(),
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("resident AIE2P W4 verification is Linux-only");
}
