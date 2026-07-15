//! Batched-FFN correctness check: the un-pinned resident dense-W8 canonical FFN.
//!
//! The FFN is per-row independent, so a M512 dispatch fed two identical documents
//! [X; X] must produce two identical output blocks, each equal to the M256 result
//! for X. This runs both the M256 and M512 (batch=2) caches with the SAME logical
//! weights and asserts out512[doc0] == out512[doc1] == out256(X).
//!
//! Usage: `npu_resident_ffn_w8_batched_verify [M256_CACHE] [M512_CACHE]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::{NpuResidentFfnDenseW8, NpuResidentFfnDenseW8IoMode, OpusPackedMatrix};

    const M: usize = 256;
    const K: usize = 768;
    const INTERMEDIATE: usize = 1152;
    const OUTPUT: usize = 768;

    let home = std::env::var("HOME").expect("HOME");
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let cache256 = args.first().cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16_m256_k768_i1152_o768")
    });
    let cache512 = args.get(1).cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16_m512_k768_i1152_o768")
    });

    // One logical set of gate/up/down weights, shared by both M256 and M512.
    let gate = OpusPackedMatrix::from_payload(
        35,
        K,
        INTERMEDIATE,
        &w8_payload(K, INTERMEDIATE, 3, 0.0060),
        None,
    )?;
    let up = OpusPackedMatrix::from_payload(
        35,
        K,
        INTERMEDIATE,
        &w8_payload(K, INTERMEDIATE, 11, 0.0055),
        None,
    )?;
    let down = OpusPackedMatrix::from_payload(
        35,
        INTERMEDIATE,
        OUTPUT,
        &w8_payload(INTERMEDIATE, OUTPUT, 23, 0.0040),
        None,
    )?;

    // One document X (256 rows), token-major bf16.
    let x_bf16 = (0..M * K)
        .map(|i| f32_to_bf16_bits((i as f32 * 0.0037).sin() * 0.7 + (i % 19) as f32 * 0.009 - 0.08))
        .collect::<Vec<u16>>();

    // --- M256 reference on hardware ---
    let mut ffn256 = NpuResidentFfnDenseW8::load_cached(&cache256)?;
    if !matches!(
        ffn256.io_mode(),
        NpuResidentFfnDenseW8IoMode::CanonicalBf16
            | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
    ) {
        return Err("M256 cache is not canonical-BF16".into());
    }
    let w256 = ffn256.upload_weights(&gate, &up, &down)?;
    let y256 = ffn256.run_canonical_bf16(&w256, &x_bf16)?; // [256*768]

    // --- M512 = two identical documents [X; X] ---
    let mut ffn512 = NpuResidentFfnDenseW8::load_cached(&cache512)?;
    let mut xx = x_bf16.clone();
    xx.extend_from_slice(&x_bf16); // [512*768]
    let w512 = ffn512.upload_weights(&gate, &up, &down)?;
    let y512 = ffn512.run_canonical_bf16(&w512, &xx)?; // [512*768]

    if y512.len() != 2 * y256.len() {
        return Err(format!(
            "M512 output {} != 2x M256 output {}",
            y512.len(),
            y256.len()
        )
        .into());
    }
    let (doc0, doc1) = y512.split_at(y256.len());

    let (cos00, max00, _) = metrics(doc0, &y256); // M512 doc0 vs M256
    let (cos11, max11, _) = metrics(doc1, &y256); // M512 doc1 vs M256
    let (cos01, max01, _) = metrics(doc0, doc1); // doc0 vs doc1 (batch self-consistency)

    println!("M512 doc0 vs M256(X): cosine={cos00:.8} max_abs={max00:.7}");
    println!("M512 doc1 vs M256(X): cosine={cos11:.8} max_abs={max11:.7}");
    println!("M512 doc0 vs doc1   : cosine={cos01:.8} max_abs={max01:.7}");

    let ok = cos00 > 0.9999
        && cos11 > 0.9999
        && cos01 > 0.99999
        && max00 < 0.02
        && max11 < 0.02
        && max01 < 1e-4;
    if !ok {
        for i in [0usize, 1, OUTPUT, y256.len() - 1] {
            eprintln!(
                "  [{i}] m256={:.6} doc0={:.6} doc1={:.6}",
                y256[i], doc0[i], doc1[i]
            );
        }
        return Err("BATCHED FFN MISMATCH: M512 != 2x M256".into());
    }
    println!("BATCHED FFN OK: M512 = 2x M256 (per-row independent, docs identical)");

    // Floor amortization: if the ~fixed per-dispatch cost dominates, M512 (2x the
    // rows) should take far less than 2x the M256 time -> throughput gain.
    let iters = 20usize;
    for _ in 0..3 {
        ffn256.run_canonical_bf16(&w256, &x_bf16)?;
        ffn512.run_canonical_bf16(&w512, &xx)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        ffn256.run_canonical_bf16(&w256, &x_bf16)?;
    }
    let ms256 = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        ffn512.run_canonical_bf16(&w512, &xx)?;
    }
    let ms512 = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "timing: M256={ms256:.3} ms ({:.0} rows/ms)  M512={ms512:.3} ms ({:.0} rows/ms)  throughput_gain={:.2}x  (2x-linear would be {:.3} ms)",
        256.0 / ms256,
        512.0 / ms512,
        (512.0 / ms512) / (256.0 / ms256),
        2.0 * ms256
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn w8_payload(k: usize, n: usize, seed: usize, base_scale: f32) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_f16;
    const GROUP: usize = 256;
    const BLOCK: usize = 258;
    let groups = k.div_ceil(GROUP);
    let mut payload = vec![0u8; n * groups * BLOCK];
    for col in 0..n {
        for group in 0..groups {
            let block =
                &mut payload[(col * groups + group) * BLOCK..(col * groups + group + 1) * BLOCK];
            let scale = base_scale * (1.0 + ((col + 3 * group + seed) % 7) as f32 * 0.025);
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for inner in 0..GROUP {
                let mixed = (inner as u64).wrapping_mul(0x9e37_79b1)
                    ^ (col as u64).wrapping_mul(0x85eb_ca77)
                    ^ (group as u64).wrapping_mul(0xc2b2_ae3d)
                    ^ (seed as u64).wrapping_mul(0x27d4_eb2f);
                block[2 + inner] = ((mixed % 15) as i8 - 7) as u8;
            }
        }
    }
    payload
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let (mut dot, mut gn, mut en, mut max_abs, mut sum) = (0.0, 0.0, 0.0, 0.0f32, 0.0);
    for (&g, &e) in got.iter().zip(expected) {
        max_abs = max_abs.max((g - e).abs());
        sum += (g - e).abs() as f64;
        dot += g as f64 * e as f64;
        gn += (g as f64).powi(2);
        en += (e as f64).powi(2);
    }
    (
        dot / (gn.sqrt() * en.sqrt()),
        max_abs,
        sum / got.len() as f64,
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("resident AIE2P FFN verification is Linux-only");
}
