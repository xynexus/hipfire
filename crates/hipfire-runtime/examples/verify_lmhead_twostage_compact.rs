// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Is the two-stage lm_head GREEDY-EXACT on a COMPACT-RESIDENT (qt 36) head?
//!
//! The two-stage path is only sound if the coarse q2 shortlist always contains
//! the true argmax; otherwise it silently returns a different token. This
//! compares, over many decode-shaped probes:
//!
//!   full = argmax( gemv_oq_compact_grouped_auto(W, x) )   [every vocab row]
//!   two  = argmax( lmhead_twostage_serve_compact(W, x) )  [coarse -> rescore]
//!
//! Both sides are fed the SAME x, so the FWHT rotation is irrelevant here and is
//! deliberately omitted — this isolates the shortlist question, which is the only
//! thing the two-stage path can get wrong.
//!
//!   ./target/release/examples/verify_lmhead_twostage_compact <model.hfq> [topk] [n]

use hipfire_rdna::lmhead_twostage::lmhead_twostage_serve_compact;
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::build_coarse_from_compact;
use hipfire_runtime::oq8_arch::normalize_compact_overlays;
use std::path::Path;

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: verify_lmhead_twostage_compact <model.hfq> [fnorm.bin]");
    // Real decode states, captured with
    //   HIPFIRE_DUMP_HIDDEN=<prefix> HIPFIRE_DUMP_HIDDEN_ALL=1 \
    //   HIPFIRE_DUMP_HIDDEN_LAYER=<n_layers>
    // (the fnorm dump is tagged with layer_idx == n_layers). Without it this
    // falls back to random probes, which are a HARSHER and less relevant
    // distribution — the shipped default should be chosen on real states.
    let fnorm_path = args.get(2).cloned();

    let mut gpu = Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let (info, bytes) = hfq
        .tensor_data_cow("lm_head.weight")
        .expect("no lm_head.weight tensor");
    assert_eq!(
        info.quant_type, 36,
        "this verifier needs an OqPlusCompact lm_head (quant_type 36); got {}",
        info.quant_type
    );
    let vocab = info.shape[0] as usize;
    let hidden = info.shape[1] as usize;
    let block_stride = bytes.len() / (vocab * (hidden / 256));
    let base_mb = bytes.len() as f64 / 1e6;

    // Exactly what the loader does before the weight reaches any kernel.
    let mut owned = bytes.to_vec();
    normalize_compact_overlays(&mut owned, vocab, hidden, 256);
    let mut buf = gpu.upload_raw(&owned, &[owned.len()]).expect("upload");
    buf.dtype = DType::OqCompactG256;
    drop(owned);

    // Probe set.
    let probes: Vec<Vec<f32>> = match &fnorm_path {
        Some(fp) => {
            let raw = std::fs::read(fp).expect("read fnorm");
            let n = raw.len() / (hidden * 4);
            assert!(n > 0, "fnorm file holds no [{hidden}] f32 vectors");
            (0..n)
                .map(|i| {
                    (0..hidden)
                        .map(|j| {
                            let o = (i * hidden + j) * 4;
                            f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
                        })
                        .collect()
                })
                .collect()
        }
        None => {
            let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut next = || {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };
            (0..256)
                .map(|_| (0..hidden).map(|_| next()).collect())
                .collect()
        }
    };
    println!(
        "lm_head [{vocab} x {hidden}] OqPlusCompact, block_stride={block_stride}, {base_mb:.0} MB\n\
         probes: {} {}\n",
        probes.len(),
        if fnorm_path.is_some() {
            "REAL decode states (FWHT-rotated below, as serving does)"
        } else {
            "random (no fnorm.bin given — harsher than real decode)"
        }
    );

    // Rotate every probe once: the stored compact weights live in the
    // FWHT-rotated basis, so this is the vector serving actually scores.
    let full = gpu.zeros(&[vocab], DType::F32).expect("full");
    let two = gpu.zeros(&[vocab], DType::F32).expect("two");
    let mut rot: Vec<hipfire_rdna::GpuTensor> = Vec::with_capacity(probes.len());
    let mut ref_argmax: Vec<usize> = Vec::with_capacity(probes.len());
    for h in &probes {
        let hg = gpu.upload_f32(h, &[hidden]).expect("upload h");
        let xr = gpu.alloc_tensor(&[hidden], DType::F32).expect("xr");
        gpu.rotate_x_mq(&hg, &xr, hidden).expect("rotate");
        let _ = gpu.free_tensor(hg);
        // Reference: the exact full-vocab compact GEMV this path replaces.
        gpu.gemv_oq_compact_grouped_auto(&buf, &xr, &full, vocab, hidden, 256, block_stride)
            .expect("full gemv");
        let fv = gpu.download_f32(&full).expect("dl full");
        ref_argmax.push(argmax(&fv).0);
        rot.push(xr);
    }

    println!("  bits     K   coarse MB   fine MB   total MB  vs full   recall@1");
    let ks = [32usize, 128, 512, 2048, 8192, 32768];
    let mut best: Option<(usize, usize, f64)> = None;
    for bits in [2usize, 4] {
        let coarse =
            build_coarse_from_compact(&mut gpu, &buf, vocab, hidden, bits).expect("coarse tier");
        let coarse_mb = (vocab * hidden * bits / 8) as f64 / 1e6 + (vocab * 4) as f64 / 1e6;
        for &k in &ks {
            let mut hits = 0usize;
            for (i, xr) in rot.iter().enumerate() {
                lmhead_twostage_serve_compact(
                    &mut gpu,
                    &buf,
                    &coarse,
                    xr,
                    &two,
                    vocab,
                    hidden,
                    block_stride,
                    k,
                )
                .expect("two-stage");
                let tv = gpu.download_f32(&two).expect("dl two");
                if argmax(&tv).0 == ref_argmax[i] {
                    hits += 1;
                }
            }
            let fine_mb = (k.min(vocab) * (hidden / 256) * block_stride) as f64 / 1e6;
            let total = coarse_mb + fine_mb;
            let recall = 100.0 * hits as f64 / rot.len() as f64;
            println!(
                "  {bits:>4}  {k:>5}   {coarse_mb:>9.1} {fine_mb:>9.1} {total:>10.1}  {:>6.2}x   {hits}/{} ({recall:.2}%)",
                base_mb / total,
                rot.len()
            );
            // "Ideal" = fewest bytes among the configs that lose no argmax.
            if hits == rot.len() && best.map(|(_, _, t)| total < t).unwrap_or(true) {
                best = Some((bits, k, total));
            }
        }
        gpu.free_tensor(coarse.q4).ok();
        gpu.free_tensor(coarse.scales).ok();
    }
    for xr in rot {
        let _ = gpu.free_tensor(xr);
    }
    match best {
        Some((bits, k, total)) => println!(
            "\nCHEAPEST LOSSLESS CONFIG: q{bits}, K={k} — {total:.0} MB/token vs {base_mb:.0} MB              ({:.2}x fewer bytes), argmax identical on every probe",
            base_mb / total
        ),
        None => println!("\nNo config reproduced the full-vocab argmax on every probe."),
    }
}
