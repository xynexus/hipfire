// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Verify `Gpu::bf16_huff_decode` against the CPU reference decoder.
//!
//! The check is BYTE EQUALITY against `bf16_huff::decode`, not a tolerance: the
//! format is lossless, so any difference is a bug. It matters that this is
//! exact — a decoder that diverges produces plausible-but-wrong weights (right
//! magnitude, wrong values), which no magnitude or NaN check catches. That is
//! the same failure mode the v0/v1 chunk-offset note in `bf16_huff` describes.
//!
//! Run:
//!   cargo run --release -p hipfire-rdna --example verify_bf16_huff_decode

use hipfire_primitives::bf16_huff;
use hipfire_rdna::{DType, Gpu};

/// Deterministic bf16 weight-ish values: a few dominant exponents plus a heavy
/// tail, so the payload exercises BOTH the primary-table fast path and the
/// escape path. A uniform-random buffer would be all escapes and a constant
/// buffer would be all fast path; neither alone proves the decoder.
fn synth_bf16(n: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    let mut z = seed;
    for _ in 0..n {
        z = z
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (z >> 33) as u32;
        // ~94% of values land in a narrow exponent band (the common case that
        // gets short codes); the rest spread wide and force escapes.
        let exp: u32 = if r % 16 != 0 {
            118 + (r >> 8) % 6
        } else {
            (r >> 12) % 256
        };
        let sign = (r >> 20) & 1;
        let mant = (r >> 3) & 0x7f;
        let bits = ((sign << 15) | (exp << 7) | mant) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut failures = 0usize;

    // Sizes chosen around the chunk boundary: below one chunk, exactly one,
    // several plus a partial tail. The partial tail is where an off-by-one in
    // the `end` clamp shows up.
    for &n in &[1usize, 4095, 8192, 8193, 20_000, 8192 * 5 + 77] {
        let raw = synth_bf16(n, 0x9E37_79B9_7F4A_7C15 ^ n as u64);
        // `encode` always produces a payload; `encode_if_smaller` is the one
        // that declines. Decode must round-trip either way.
        let packed = bf16_huff::encode(&raw);
        let want = bf16_huff::decode(&packed, n).expect("cpu decode");
        assert_eq!(want, raw, "n={n}: cpu decode is not lossless — bad fixture");

        let n_chunks = n.div_ceil(bf16_huff::CHUNK);
        let d_packed = gpu
            .upload_raw(&packed, &[packed.len()])
            .expect("upload packed");
        let d_out = gpu.alloc_owned(&[n], DType::BF16).expect("alloc out");
        gpu.bf16_huff_decode(&d_packed, &d_out, n_chunks, n)
            .expect("launch");
        gpu.device_synchronize().expect("sync");
        let got = gpu.download_raw(&d_out, n * 2).expect("download");

        if got == want {
            println!(
                "n={n:<8} chunks={n_chunks:<4} packed={:<9} OK",
                packed.len()
            );
        } else {
            failures += 1;
            let first = got
                .iter()
                .zip(want.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            println!(
                "n={n:<8} chunks={n_chunks:<4} MISMATCH at byte {first} (elem {}) \
                 gpu={:#06x} cpu={:#06x}",
                first / 2,
                u16::from_le_bytes([got[first & !1], got[(first & !1) + 1]]),
                u16::from_le_bytes([want[first & !1], want[(first & !1) + 1]]),
            );
        }
    }

    // Throughput at a realistic size: 262M elements is the Llama-3.2-1B tied
    // embed table (128256 x 2048), the tensor this format exists for. The
    // comparison that matters is against the HOST path this replaces —
    // `decode_par` on all cores — since correctness is already established.
    // Scaling probe first: if device time is linear in chunk count the kernel is
    // simply slow per element; if it is superlinear something is contending.
    for &probe in &[262_144usize, 2_097_152, 16_777_216] {
        let raw = synth_bf16(probe, 0xABCD);
        let packed = bf16_huff::encode(&raw);
        let nc = probe.div_ceil(bf16_huff::CHUNK);
        let dp = gpu.upload_raw(&packed, &[packed.len()]).expect("up");
        let dout = gpu.alloc_owned(&[probe], DType::BF16).expect("alloc");
        gpu.bf16_huff_decode(&dp, &dout, nc, probe).expect("warm");
        gpu.device_synchronize().expect("sync");
        let t = std::time::Instant::now();
        gpu.bf16_huff_decode(&dp, &dout, nc, probe).expect("go");
        gpu.device_synchronize().expect("sync");
        let s = t.elapsed().as_secs_f64();
        println!(
            "probe n={probe:<9} chunks={nc:<5} {s:.4} s  ({:.1} ns/elem)",
            s * 1e9 / probe as f64
        );
    }

    let n = 128_256usize * 2048;
    println!(
        "\n--- throughput, n={n} ({:.2} GB as BF16) ---",
        (n * 2) as f64 / 1e9
    );
    let raw = synth_bf16(n, 0x1234_5678);
    let packed = bf16_huff::encode(&raw);
    println!(
        "packed {:.2} GB ({:.4}x)",
        packed.len() as f64 / 1e9,
        raw.len() as f64 / packed.len() as f64
    );

    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8);
    let t0 = std::time::Instant::now();
    let cpu = bf16_huff::decode_par(&packed, n, threads).expect("cpu decode_par");
    let cpu_s = t0.elapsed().as_secs_f64();
    println!(
        "cpu  decode_par({threads} threads): {cpu_s:.3} s  ({:.2} GB/s out)",
        (n * 2) as f64 / 1e9 / cpu_s
    );

    let n_chunks = n.div_ceil(bf16_huff::CHUNK);
    let d_packed = gpu.upload_raw(&packed, &[packed.len()]).expect("upload");
    let d_out = gpu.alloc_owned(&[n], DType::BF16).expect("alloc");
    // Warm the kernel so the measurement excludes first-call JIT.
    gpu.bf16_huff_decode(&d_packed, &d_out, n_chunks, n)
        .expect("warm");
    gpu.device_synchronize().expect("sync");
    let t1 = std::time::Instant::now();
    gpu.bf16_huff_decode(&d_packed, &d_out, n_chunks, n)
        .expect("launch");
    gpu.device_synchronize().expect("sync");
    let gpu_s = t1.elapsed().as_secs_f64();
    println!(
        "gpu  bf16_huff_decode:            {gpu_s:.3} s  ({:.2} GB/s out)",
        (n * 2) as f64 / 1e9 / gpu_s
    );
    println!("gpu is {:.2}x the cpu path", cpu_s / gpu_s);

    let got = gpu.download_raw(&d_out, n * 2).expect("download");
    if got != cpu {
        println!("MISMATCH at the large size");
        failures += 1;
    }

    if failures == 0 {
        println!("\nbf16_huff_decode: GPU output is byte-identical to the CPU reference");
    } else {
        println!("\nbf16_huff_decode: {failures} size(s) MISMATCHED");
        std::process::exit(1);
    }
}
