// SPDX-License-Identifier: Apache-2.0
// hipfire — real-weight recall@1 check for the generic two-stage lm_head.
//
//! Loads a model's REAL bf16 `lm_head.weight` (or tied `model.embed_tokens.weight`)
//! and measures whether the coarse-shortlist + fine-rescore two-stage decode keeps
//! the true argmax inside the shortlist — i.e. greedy-exactness (recall@1).
//!
//! For each random hidden vector it compares:
//!   full   = argmax( gemv_bf16_f32(lm_head, h) )        [exact, all rows]
//!   two    = argmax( lmhead_twostage_serve_bf16(...) )  [coarse top-K -> fine rescore]
//! recall@1 == 1.0 => the shortlist never drops the winner => lossless for greedy decode.
//!
//! Random hidden is a valid probe here: the test measures whether the per-row Q4
//! DIRECTION quant preserves the dot-product ranking, which is independent of how
//! "realistic" h is. Run:
//!   ./target/release/examples/verify_lmhead_twostage_real <model.bf16.hfq> [topk=32] [n=256]

use hipfire_rdna::lmhead_twostage::{build_lmhead_coarse_bf16, lmhead_twostage_serve_bf16};
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::hfq::HfqFile;
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
        .expect("usage: verify_lmhead_twostage_real <model.hfq> [topk] [n]");
    let topk: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let nprobe: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);

    let hfq = HfqFile::open(Path::new(path)).expect("open hfq");
    let (info, bytes) = hfq
        .tensor_data("lm_head.weight")
        .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
        .expect("no lm_head.weight / embed_tokens.weight tensor");
    assert_eq!(
        info.quant_type, 16,
        "this verifier needs a bf16 lm_head (quant_type 16); got {}",
        info.quant_type
    );
    let vocab = info.shape[0] as usize;
    let hidden = info.shape[1] as usize;
    println!(
        "lm_head [{vocab} x {hidden}] bf16, {} MB",
        bytes.len() / 1_000_000
    );
    assert_eq!(bytes.len(), vocab * hidden * 2, "bf16 byte count mismatch");

    // Upload the raw bf16 weight as a BF16 GpuTensor.
    let mut lmhead = gpu
        .upload_raw(bytes, &[bytes.len()])
        .expect("upload lm_head");
    lmhead.dtype = DType::BF16;

    let coarse =
        build_lmhead_coarse_bf16(&mut gpu, &lmhead, vocab, hidden, 4).expect("build coarse tier");
    println!("coarse tier built (Q4 row-norm), topk={topk}, probes={nprobe}");

    // Deterministic splitmix64 -> f32 in [-1, 1].
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };

    let logits_full = gpu.zeros(&[vocab], DType::F32).expect("logits_full");
    let logits_two = gpu.zeros(&[vocab], DType::F32).expect("logits_two");

    let mut matches = 0usize;
    let mut worst_gap = 0.0f32;
    let mut misses: Vec<(usize, usize, usize, f32)> = Vec::new();
    for p in 0..nprobe {
        let h: Vec<f32> = (0..hidden).map(|_| next()).collect();
        let hbytes: &[u8] =
            unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 4) };
        let mut hgpu = gpu.upload_raw(hbytes, &[hidden]).expect("upload h");
        hgpu.dtype = DType::F32;

        // Full reference: same bf16(x) arithmetic the two-stage fine pass uses, over
        // ALL rows. gemv_bf16_f32 needs a BF16 input (y=Σ W_bf16·x_bf16).
        let xb = gpu.alloc_tensor(&[hidden], DType::BF16).expect("xb");
        gpu.cast_f32_to_bf16(&hgpu, &xb).expect("cast x");
        gpu.gemv_bf16_f32(&lmhead, &xb, &logits_full, vocab, hidden)
            .expect("full gemv");
        let _ = gpu.free_tensor(xb);
        let (a_full, v_full) = argmax(&gpu.download_f32(&logits_full).expect("dl full"));

        lmhead_twostage_serve_bf16(
            &mut gpu,
            &lmhead,
            &coarse,
            &hgpu,
            &logits_two,
            vocab,
            hidden,
            topk,
        )
        .expect("two-stage");
        let two = gpu.download_f32(&logits_two).expect("dl two");
        let (a_two, _v_two) = argmax(&two);

        if a_full == a_two {
            matches += 1;
        } else {
            // gap = how far below the true winner the two-stage's pick scored (in full space)
            let gap = v_full - two[a_full].max(f32::NEG_INFINITY);
            worst_gap = worst_gap.max(gap.abs());
            if misses.len() < 8 {
                misses.push((p, a_full, a_two, gap));
            }
        }
        let _ = gpu.free_tensor(hgpu);
    }

    let recall = matches as f64 / nprobe as f64;
    println!("recall@1 = {matches}/{nprobe} = {recall:.4}");
    if !misses.is_empty() {
        println!("misses (probe, argmax_full, argmax_two, full-space gap):");
        for (p, af, at, g) in &misses {
            println!("  probe {p}: full={af} two={at} gap={g:.4}");
        }
        println!("worst full-space gap on a miss: {worst_gap:.4}");
    }
    if recall == 1.0 {
        println!("LOSSLESS: shortlist kept the true argmax on every probe (greedy-exact).");
    }
}
