// SPDX-License-Identifier: Apache-2.0
//! Phase 0c throughput gate — kernel-level. The full-model harnesses run
//! qwen3.5 prefill per-token (W4A16), so they cannot compare the batched oq4
//! activation-precision GEMMs. This times the three head-to-head on the same
//! x_rot[N×K] + weight, at a prefill-shaped (M,K,N), to answer the ONLY question
//! that decides whether A4 quality work is worth doing:
//!   is int4-act (iu4 WMMA) actually FASTER than W4A8-MMQ / W4A16-f16-WMMA?
//!
//!   act4  = gemm_oq4_grouped_act_batched   (quantize_act_oq4 + iu4·iu4 WMMA)
//!   act8  = gemm_oq4_residual_mmq          (q8_1 quantize + int8 MMQ)
//!   act16 = gemm_oq4_grouped_f16_wmma      (dequant int4→f16 + f16 WMMA)
//! All three consume the same f32 rotated activation and produce y[N×M], so the
//! measured wall is the fair per-projection prefill cost at that precision.

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

fn lcgf(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (((s as f32 / 2_147_483_648.0) - 0.5) * 2.0).to_le_bytes()
        })
        .collect()
}
fn nib(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: no wmma");
        return;
    }
    gpu.ensure_mq_signs().unwrap();
    const GROUP: usize = 256;
    let iters = 100usize;

    // qwen3.5-0.8b-ish projection shapes; N = prefill batch.
    let shapes = [
        ("qkv   M=1536 K=1024", 1536usize, 1024usize),
        ("o/down M=1024 K=1024", 1024usize, 1024usize),
        ("gate  M=3072 K=1024", 3072usize, 1024usize),
    ];
    let batches = [128usize, 512usize];

    println!("oq4 activation-GEMM throughput  arch={}  iters={iters}", gpu.arch);
    println!("{:<22} {:>5} {:>12} {:>12} {:>12}   winner", "shape", "N", "act4 ms", "act8 ms", "act16 ms");

    for (label, m, k) in shapes {
        let ng = k / GROUP;
        // Combined weight buffer [nibbles m*k/2 | f32 scales m*ng], Raw.
        let mut wbuf = nib(7, m * (k / 2));
        wbuf.extend_from_slice(&lcgf(8, m * ng));
        let w = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();

        for &n in &batches {
            let xr = gpu
                .upload_f32(
                    &lcgf(100, n * k)
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect::<Vec<_>>(),
                    &[n, k],
                )
                .unwrap();
            let y = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

            // median ms over `iters` timed launches (after 5 warm), macro so each
            // kernel call site is inlined (no closure capture of !Clone GpuTensor).
            macro_rules! med_ms {
                ($call:expr) => {{
                    for _ in 0..5 {
                        $call;
                    }
                    gpu.device_synchronize().unwrap();
                    let mut ms = Vec::with_capacity(iters);
                    for _ in 0..iters {
                        let t = Instant::now();
                        $call;
                        gpu.device_synchronize().unwrap();
                        ms.push(t.elapsed().as_secs_f32() * 1e3);
                    }
                    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    ms[iters / 2]
                }};
            }

            // Pre-quantize the activation ONCE so we can time the pure iu4 GEMM
            // (gemm_oq4_grouped_wmma) in isolation, separate from the per-call
            // quantize_act_oq4 launch that gemm_oq4_grouped_act_batched folds in.
            let ws = w.sub_offset(m * (k / 2), m * ng * 4);
            let xq = gpu.alloc_tensor(&[n * (k / 2)], DType::Raw).unwrap();
            let xs = gpu.alloc_tensor(&[n * ng], DType::F32).unwrap();
            gpu.quantize_act_oq4(&xr, &xq, &xs, n, k, GROUP).unwrap();
            gpu.device_synchronize().unwrap();

            // bf16-output variant of the pure iu4 GEMM (output-memory lever): halves
            // the output write (2 B vs 4 B/elem). y_bf16 is a Raw buffer of n*m*2 B.
            let y_bf16 = gpu.alloc_tensor(&[n * m * 2], DType::Raw).unwrap();

            // baseline (unoptimized) iu4 GEMM — measured same-run for context.
            let _t4_gemm =
                med_ms!(gpu.gemm_oq4_grouped_wmma(&w, &ws, &xq, &xs, &y, m, k, n, GROUP).unwrap());
            // LDS-staged optimized iu4 GEMM (Stream A): f32 + bf16 output.
            let t4_lds =
                med_ms!(gpu.gemm_oq4_grouped_wmma_lds(&w, &ws, &xq, &xs, &y, m, k, n, GROUP).unwrap());
            let t4_lds_bf16 = med_ms!(gpu
                .gemm_oq4_grouped_wmma_lds_bf16out(&w, &ws, &xq, &xs, &y_bf16, m, k, n, GROUP)
                .unwrap());
            let _t4 = med_ms!(gpu.gemm_oq4_grouped_act_batched(&w, &xr, &y, m, k, n).unwrap());
            let t8 = med_ms!(gpu.gemm_oq4_residual_mmq(&w, &xr, &y, m, k, n, false).unwrap());
            let t16 =
                med_ms!(gpu.gemm_oq4_grouped_f16_wmma(&w, &xr, &y, m, k, n, GROUP).unwrap());

            // FAIR full-path comparison: MMQ (t8) re-quantizes the activation every
            // call (ensure_q8_1_mmq_x must_convert=true), so time the iu4-lds FULL
            // path too = quantize_act_oq4 + LDS GEMM, matching MMQ's quant+GEMM.
            let t4_lds_full = med_ms!({
                gpu.quantize_act_oq4(&xr, &xq, &xs, n, k, GROUP).unwrap();
                gpu.gemm_oq4_grouped_wmma_lds(&w, &ws, &xq, &xs, &y, m, k, n, GROUP).unwrap();
            });

            // Winner over the "pure GEMM" iu4 variants vs MMQ vs f16 (excludes the
            // fused-quant act4 which includes the quantize_act launch).
            let best = t4_lds.min(t4_lds_bf16).min(t8).min(t16);
            let win = if best == t4_lds_bf16 {
                "iu4-lds/bf16"
            } else if best == t4_lds {
                "iu4-lds/f32"
            } else if best == t8 {
                "act8-MMQ"
            } else {
                "act16"
            };
            println!(
                "{label:<22} {n:>5}  iu4-lds/f32={t4_lds:>6.3} iu4-lds/bf16={t4_lds_bf16:>6.3}  \
                 lds-FULL(q+gemm)={t4_lds_full:>6.3}  MMQ(q+gemm)={t8:>6.3} act16={t16:>6.3}  \
                 best={win:<12} (lds-GEMM vs MMQ {:+.0}%; lds-FULL vs MMQ {:+.0}%)",
                (t4_lds / t8 - 1.0) * 100.0,
                (t4_lds_full / t8 - 1.0) * 100.0
            );
        }
    }
}
