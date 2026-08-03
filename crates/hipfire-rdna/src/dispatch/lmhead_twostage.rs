// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic, arch-neutral two-stage ("coarse shortlist + fine rescore") lm_head
//! decode.
//!
//! The math is arch-agnostic: it operates on a final hidden state (`fnorm`,
//! f32 `[H]`) and the model's bf16 lm_head weight (`[V, H]`) and produces
//! logits `[V]`. Stage 1 scores every vocab row cheaply with a per-row
//! symmetric Q`bits` GEMV; Stage 2 selects a device-side top-`K` shortlist and
//! rescores exactly those rows at bf16, scattering into a `-inf`-masked logit
//! buffer. When the coarse recall@1 at `K` is 100%, the argmax is exact
//! (greedy-lossless).
//!
//! This is the generalized hoist of the private zaya implementation
//! (`crates/hipfire-arch-zaya/src/gpu.rs`). v1 is the MINIMAL path: `bits ∈
//! {2,4}`, no low-rank projection, no residual correction. The zaya-specific
//! `ZayaGpuWeights` / `embed.wt_mk()` plumbing is gone — callers pass the bf16
//! `&GpuTensor` and `(vocab, hidden)` directly.

use super::{DType, Gpu, GpuTensor};

/// Built coarse tier: per-row symmetric Q`bits` codes + per-row f32 scale.
///
/// `q4` holds `vocab * kdim * bits / 8` packed bytes (the name is historical —
/// it carries Q2 packing too when `bits == 2`). `scales[v]` is the row's L2
/// norm folded with the shared unit dequant scale, so `scale * dequant(code) ≈
/// W[v]`. `kdim == hidden` in the minimal (no-projection) path.
pub struct LmheadCoarse {
    pub q4: GpuTensor,
    pub scales: GpuTensor,
    pub kdim: usize,
    pub bits: usize, // 2 or 4
}

/// Build the coarse tier from the model's bf16 lm_head weight `[vocab, hidden]`.
///
/// Downloads the bf16 rows once, computes each row's L2 norm → f32 scale, and
/// quantizes the unit direction to symmetric Q`bits` (`bits ∈ {2, 4}`). No
/// projection, no residual correction (v1 minimal path). Mirrors
/// `build_lmhead_coarse` in the zaya source with `proj_r = 0`, `correct_r = 0`.
pub fn build_lmhead_coarse_bf16(
    gpu: &mut Gpu,
    lmhead_bf16: &GpuTensor,
    vocab: usize,
    hidden: usize,
    bits: usize,
) -> Result<LmheadCoarse, String> {
    if bits != 2 && bits != 4 {
        return Err(format!("lmhead coarse: bits must be 2 or 4, got {bits}"));
    }
    let kdim = hidden; // no low-rank projection in the minimal path.
    let bytes = gpu
        .download_raw(lmhead_bf16, vocab * hidden * 2)
        .map_err(|e| format!("lmhead coarse download: {e:?}"))?;
    // Global symmetric quant range + packing density for the chosen bit-width.
    let (lo, hi, max_mag, per_byte) = if bits == 2 {
        (-2.0f32, 1.0f32, 2.0f32, 4usize)
    } else {
        (-7.0f32, 7.0f32, 7.0f32, 2usize)
    };
    // The unit direction has L2 norm 1; a 3σ clip over `kdim` roughly Gaussian
    // components lands at ±3/√kdim, so scale so that maps to the code extreme.
    let unit_scale = 3.0f32 / (max_mag * (kdim as f32).sqrt());
    let inv = 1.0 / unit_scale;
    let kb = kdim / per_byte;
    let mut q4 = vec![0u8; vocab * kb];
    let mut scales = vec![0f32; vocab];
    let mut wv = vec![0f32; hidden];
    for v in 0..vocab {
        let row = &bytes[v * hidden * 2..(v + 1) * hidden * 2];
        for i in 0..hidden {
            let u = u16::from_le_bytes([row[2 * i], row[2 * i + 1]]);
            // bf16 → f32: the bf16 bits are the high 16 bits of the f32.
            wv[i] = f32::from_bits((u as u32) << 16);
        }
        let norm = wv.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let q = &mut q4[v * kb..(v + 1) * kb];
        if norm > 0.0 {
            let ni = inv / norm;
            let qz = |d: usize| ((wv[d] * ni).round().clamp(lo, hi) as i32) as u8;
            if bits == 2 {
                for i in 0..kb {
                    q[i] = (qz(4 * i) & 0x3)
                        | ((qz(4 * i + 1) & 0x3) << 2)
                        | ((qz(4 * i + 2) & 0x3) << 4)
                        | ((qz(4 * i + 3) & 0x3) << 6);
                }
            } else {
                for i in 0..kb {
                    q[i] = (qz(2 * i) & 0xF) | ((qz(2 * i + 1) & 0xF) << 4);
                }
            }
        }
        scales[v] = norm * unit_scale;
    }
    let q4buf = gpu
        .upload_raw(&q4, &[q4.len()])
        .map_err(|e| format!("lmhead coarse upload q: {e:?}"))?;
    let scbuf = gpu
        .upload_f32(&scales, &[vocab])
        .map_err(|e| format!("lmhead coarse upload scales: {e:?}"))?;
    Ok(LmheadCoarse {
        q4: q4buf,
        scales: scbuf,
        kdim,
        bits,
    })
}

/// Coarse-score every vocab row: `coarse[v] = scale[v] · <dequant(code[v]),
/// fnorm>`. Allocates and returns the `[vocab]` f32 score buffer (caller frees).
/// Minimal path: no projection, no low-rank correction.
pub fn coarse_score(
    gpu: &mut Gpu,
    c: &LmheadCoarse,
    fnorm: &GpuTensor,
    vocab: usize,
    _hidden: usize,
) -> Result<GpuTensor, String> {
    let kdim = c.kdim;
    let coarse = gpu
        .zeros(&[vocab], DType::F32)
        .map_err(|e| format!("lmhead coarse alloc: {e:?}"))?;
    if c.bits == 2 {
        gpu.gemv_q2sym_f32(&c.q4, &c.scales, fnorm, &coarse, vocab, kdim)
            .map_err(|e| format!("lmhead coarse gemv q2: {e:?}"))?;
    } else {
        gpu.gemv_q4sym_f32(&c.q4, &c.scales, fnorm, &coarse, vocab, kdim)
            .map_err(|e| format!("lmhead coarse gemv q4: {e:?}"))?;
    }
    Ok(coarse)
}

/// Device top-K over `coarse` [V]: min/max → histogram → threshold scan →
/// compact. Returns `(idx buffer, cap)` where `idx` is `-1`-sentinel-filled
/// (`0xFFFFFFFF`) `cap` slots holding a SUPERSET of the exact top-`kk` rows (the
/// fine bf16 pass rescores exactly, so extra candidates are harmless). Only
/// three tiny scalars (min/max + histogram) cross to the host. Caller frees the
/// returned buffer.
pub fn gpu_topk(
    gpu: &mut Gpu,
    coarse: &GpuTensor,
    vocab: usize,
    kk: usize,
) -> Result<(GpuTensor, usize), String> {
    const NBINS: usize = 4096;
    let kk = kk.min(vocab).max(1);
    // Folded stats buffer: [0..NBINS) histogram bins (zeroed) | [NBINS] min key
    // | [NBINS+1] max key. minmax writes the tail; hist reads lo/hi from it
    // on-device, so the whole top-K needs ONE host download (the histogram).
    let stats = gpu
        .zeros(&[NBINS + 2], DType::F32)
        .map_err(|e| format!("lmhead topk stats: {e:?}"))?;
    // Init the min slot to 0xFFFFFFFF so atomicMin reduces it (max slot stays 0).
    let lo_slot = stats.sub_offset(NBINS, 1);
    gpu.hip
        .memset(&lo_slot.buf, 0xFF, 4)
        .map_err(|e| format!("lmhead topk min-init: {e:?}"))?;
    gpu.lmhead_coarse_minmax(coarse, &stats, vocab, NBINS)
        .map_err(|e| format!("lmhead topk minmax: {e:?}"))?;
    gpu.lmhead_coarse_hist(coarse, &stats, vocab, NBINS)
        .map_err(|e| format!("lmhead topk hist: {e:?}"))?;
    let _ = gpu.hip.device_synchronize();
    let sb = gpu
        .download_raw(&stats, (NBINS + 2) * 4)
        .map_err(|e| format!("lmhead topk stats download: {e:?}"))?;
    let _ = gpu.free_tensor(stats);
    let rd =
        |i: usize| u32::from_le_bytes([sb[i * 4], sb[i * 4 + 1], sb[i * 4 + 2], sb[i * 4 + 3]]);
    let lo = rd(NBINS);
    let hi = rd(NBINS + 1);
    // Scan bins top-down until the cumulative count reaches kk → threshold τ.
    let mut acc = 0usize;
    let mut boundary = 0usize;
    for b in (0..NBINS).rev() {
        acc += rd(b) as usize;
        if acc >= kk {
            boundary = b;
            break;
        }
    }
    let range = (hi as u64) - (lo as u64) + 1;
    let tau = (lo as u64 + (boundary as u64) * range / (NBINS as u64)) as u32;
    let cap = acc + 512; // count == acc up to integer-division rounding; +slack.
                         // Sentinel-fill idx (0xFFFFFFFF), compact key≥τ rows into it, and let the
                         // fine gather run over all `cap` slots skipping sentinels.
    let idxbuf = gpu
        .zeros(&[cap], DType::F32)
        .map_err(|e| format!("lmhead topk idx alloc: {e:?}"))?;
    gpu.hip
        .memset(&idxbuf.buf, 0xFF, cap * 4)
        .map_err(|e| format!("lmhead topk idx-init: {e:?}"))?;
    let counter = gpu
        .zeros(&[1], DType::F32)
        .map_err(|e| format!("lmhead topk counter: {e:?}"))?; // device write-cursor (not read back)
    gpu.lmhead_coarse_compact(coarse, &idxbuf, &counter, vocab, tau, cap)
        .map_err(|e| format!("lmhead topk compact: {e:?}"))?;
    let _ = gpu.free_tensor(counter);
    Ok((idxbuf, cap))
}

/// Full two-stage serving path. Coarse-score all `vocab` rows with the per-row
/// Q`bits` scorer, device-select the top-`topk` shortlist, cast `fnorm` to
/// bf16, `-inf`-mask `logits_out`, then rescore exactly the shortlisted rows at
/// bf16 and scatter into `logits_out`. Greedy-exact when the coarse recall@1 =
/// 100% at `topk`. Mirrors the GPU-topk default arm of `lmhead_twostage_serve`.
pub fn lmhead_twostage_serve_bf16(
    gpu: &mut Gpu,
    lmhead_bf16: &GpuTensor,
    coarse: &LmheadCoarse,
    fnorm: &GpuTensor,
    logits_out: &GpuTensor,
    vocab: usize,
    hidden: usize,
    topk: usize,
) -> Result<(), String> {
    let kk = topk.min(vocab).max(1);
    let scores = coarse_score(gpu, coarse, fnorm, vocab, hidden)?;
    let (idxbuf, count) = gpu_topk(gpu, &scores, vocab, kk)?;
    let _ = gpu.free_tensor(scores);
    let xb = gpu
        .alloc_tensor(&[hidden], DType::BF16)
        .map_err(|e| format!("lmhead fine xb: {e:?}"))?;
    gpu.cast_f32_to_bf16(fnorm, &xb)
        .map_err(|e| format!("lmhead fine cast: {e:?}"))?;
    gpu.fill_f32(logits_out, f32::NEG_INFINITY)
        .map_err(|e| format!("lmhead mask: {e:?}"))?;
    gpu.gemv_bf16_gather_f32(lmhead_bf16, &idxbuf, &xb, logits_out, count, hidden)
        .map_err(|e| format!("lmhead fine gather: {e:?}"))?;
    let _ = gpu.free_tensor(idxbuf);
    let _ = gpu.free_tensor(xb);
    Ok(())
}
