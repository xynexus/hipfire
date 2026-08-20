// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Compact-vs-expanded Opus parity on REAL artifact blocks.
//!
//! `hipfire-rdna`'s `parity_gemm_oq_compact` compares the two kernels on
//! SYNTHETIC blocks expanded by a local mirror of
//! `oqplus_compact_to_oq8_combined`. Two assumptions ride on that, and neither
//! was ever executed: that the mirror matches the real expander, and that real
//! artifact blocks look like the generated ones. This example removes both — it
//! reads an actual `.hfq`, takes its OqPlusCompact tensors verbatim, and expands
//! them with the REAL `oqplus_compact_to_oq8_combined` that the loader calls.
//!
//! It exists because the model shows a small but reproducible compact-vs-expanded
//! logit difference that the synthetic oracle cannot reproduce: bit-identical
//! 56/56 there, yet in-model EVERY (M, K) projection class diverges on its own —
//! down to a single 16-row tensor. Kernels, scales, rotation and activation
//! quantization are all eliminated, so the remaining suspects are the two
//! assumptions above. See docs/plans/2026-08-05-opus-across-model-families.md.
//!
//! The bar is BIT-IDENTICAL, as in the synthetic oracle: both paths do the same
//! int8xint8 WMMA over the same values, and f16->f32 is exact.
//!
//! Run: cargo run --release -p hipfire-runtime --example parity_oq_compact_real \
//!        -- /srv/hipfire/models/Qwen3.5-0.8B--oq4.25.hfq

use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::oq8_arch::oqplus_compact_to_oq8_combined;
use hipfire_runtime::quant::QuantType;
use std::path::Path;

const GROUP: usize = 256;

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: parity_oq_compact_real <model.hfq> [max_tensors]");
        std::process::exit(2);
    });
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    let f = HfqFile::open(Path::new(&path)).expect("open hfq");
    let compact_qt = QuantType::OqPlusCompact.code();

    // One representative per (M, K) class: the in-model bisection is per (M, K),
    // so matching that granularity makes a failure here line up with a failing
    // class there instead of naming an arbitrary tensor.
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let mut picks: Vec<(String, usize, usize)> = Vec::new();
    for t in f.tensors() {
        if t.quant_type != compact_qt || t.shape.len() != 2 {
            continue;
        }
        let (m, k) = (t.shape[0] as usize, t.shape[1] as usize);
        if k % GROUP != 0 || seen.contains(&(m, k)) {
            continue;
        }
        seen.push((m, k));
        picks.push((t.name.clone(), m, k));
        if picks.len() >= limit {
            break;
        }
    }
    if picks.is_empty() {
        eprintln!("no OqPlusCompact (qt {compact_qt}) tensors in {path}");
        std::process::exit(2);
    }

    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("GPU: {}", gpu.arch);
    println!(
        "parity_oq_compact_real: {} (M, K) class(es) from {path}",
        picks.len()
    );

    let mut fail = 0usize;
    for (name, m, k) in &picks {
        let (_, blocks) = f.tensor_data(name).expect("tensor payload");
        let (m, k) = (*m, *k);
        let ng = k / GROUP;
        let n_blocks = m * ng;
        if n_blocks == 0 || blocks.len() % n_blocks != 0 {
            println!(
                "  SKIP {name}: {} bytes over {n_blocks} blocks",
                blocks.len()
            );
            continue;
        }
        let block_stride = blocks.len() / n_blocks;

        // The REAL expander the loader uses, on the REAL bytes.
        let combined = oqplus_compact_to_oq8_combined(&blocks, m, k);

        // Batch of 4 covers the ragged-lane tail without a large upload; the
        // activations are shared byte-for-byte by both paths.
        const B: usize = 4;
        let mut rnd = lcg(0x0425_beefu32 ^ (m * k) as u32);
        let xq: Vec<i8> = (0..B * k)
            .map(|_| ((rnd() % 255) as i32 - 127) as i8)
            .collect();
        let xs: Vec<f32> = (0..B * ng)
            .map(|_| 0.25f32 + (rnd() % 100_000) as f32 * 1.0e-5)
            .collect();

        let d_blocks = gpu.upload_raw(&blocks, &[blocks.len()]).expect("up blocks");
        let d_comb = gpu
            .upload_raw(&combined, &[combined.len()])
            .expect("up combined");
        let d_xq = gpu
            .upload_raw(
                unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len()) },
                &[xq.len()],
            )
            .expect("up xq");
        let d_xs = gpu.upload_f32(&xs, &[xs.len()]).expect("up xs");
        let d_ref = gpu.zeros(&[B * m], DType::F32).expect("y ref");
        let d_cmp = gpu.zeros(&[B * m], DType::F32).expect("y cmp");

        // Same split the oq8 dispatch does: scales are the tail of the combined
        // buffer, at m*k.
        let d_ws = d_comb.sub_offset(m * k, m * ng * 4);

        gpu.gemm_oq8_grouped_wmma(&d_comb, &d_ws, &d_xq, &d_xs, &d_ref, m, k, B, GROUP)
            .expect("expanded gemm");
        gpu.gemm_oq_compact_grouped_wmma(
            &d_blocks,
            &d_xq,
            &d_xs,
            &d_cmp,
            m,
            k,
            B,
            GROUP,
            block_stride,
        )
        .expect("compact gemm");

        let y_ref = gpu.download_f32(&d_ref).expect("dl ref");
        let y_cmp = gpu.download_f32(&d_cmp).expect("dl cmp");

        let mut bad = 0usize;
        let mut worst = 0.0f32;
        for (a, c) in y_ref.iter().zip(y_cmp.iter()) {
            if a.to_bits() != c.to_bits() {
                bad += 1;
                worst = worst.max((a - c).abs());
            }
        }
        let tag = format!(
            "M={m} K={k} stride={block_stride} N_out={}",
            (block_stride - 130) / 2
        );
        if bad == 0 {
            println!(
                "  ok   {tag}: bit-identical over {} outputs  [{name}]",
                y_ref.len()
            );
        } else {
            fail += 1;
            println!(
                "  FAIL {tag}: {bad}/{} differ, worst |delta| {worst:.6e}  [{name}]",
                y_ref.len()
            );
        }
    }

    if fail == 0 {
        println!("parity_oq_compact_real: PASS");
    } else {
        println!("parity_oq_compact_real: FAIL ({fail} class(es))");
        std::process::exit(1);
    }
}
