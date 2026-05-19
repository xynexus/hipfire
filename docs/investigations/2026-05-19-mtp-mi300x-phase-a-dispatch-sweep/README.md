# MI300x Phase A — dispatch sweep on rocBLAS threshold

**Date:** 2026-05-19
**Hardware:** AMD Instinct MI300X VF / gfx942 / ROCm 7.0.0

## Goal

Profile what's left on the table for MI300x as a first-class target.
Phase 1 closed the per-token MTP lm_head gap (10.83 → 45.14 tok/s solo).
This phase looks for the *next* dispatch-side lever.

## rocprof finding on DFlash solo (dflash_spec_demo, B=17 verify)

Top kernels by GPU time on a 60-token DFlash spec-decode run:

| Rank | Kernel | % of GPU time | Class |
|---|---|---:|---|
| 1 | `gemm_gate_up_hfq4g256_fp16_wave64` | 16.6% | ✅ native wave64 |
| 2 | `gemm_hfq4g256_residual_fp16_wave64` | 14.5% | ✅ native wave64 |
| 3 | `Cijk_..._MT256x256x32_MI32x32x8x1_..._WGM18` | **10.5%** | ❌ **rocBLAS Tensile** |
| 4 | `Cijk_..._WGM4` | 10.4% | ❌ rocBLAS Tensile |
| 5 | `Cijk_..._WGM6` | 9.5% | ❌ rocBLAS Tensile |
| 8 | `hfq4g256_dequantize_to_f16` | 4.4% | ⚠️ pre-pass that feeds Tensile |

~32% of GPU time was in rocBLAS Tensile MFMA + 4.4% in the FP16 shadow
pre-pass that feeds it. The comment at `dispatch.rs:11974-11975` claimed
"20-100x Tensile over wave64 GEMV". Reality at this shape regime:
**Tensile is *competitive but not 20-100x faster*.**

## min_batch sweep (gfx942)

The `rocblas_min_batch` threshold decides B above which `gemm_hfq4g256`
takes the dequant-to-FP16-shadow + rocBLAS path. Phase 1 set it to 16
under `HIPFIRE_GFX942_NATIVE_LM_HEAD=1`. Sweep:

| min_batch | DFlash B=17 prefill | DFlash B=17 decode | Trunk prefill=256 |
|---:|---:|---:|---:|
| 16 (Phase 1) | 130 | 54.9 | 1303 |
| **32 (Phase A)** | **350 (+170%)** | **62.3 (+13.5%)** | **1303** (unchanged) |
| 64 | 352 | 62.4 | 1304 |
| 128 | 351 | 62.4 | 1303 |

**Crossover is sharp at exactly B=17.** At B=17 native wave64 wins
substantially (per-call 246µs vs Tensile's 256µs avg, but with no
dequant-to-shadow pre-pass overhead). At B=256 Tensile takes over
decisively (1303 vs 424 tok/s — 3.07x). Byte-identical acceptance
between configs (decode_tau = 2.5882 across all).

32, 64, and 128 are all equivalent — Tensile is dominant at B≥32 on
these shapes. 32 is the smallest crossover value, preserving the most
of the rocBLAS prefill amortization while rescuing DFlash verify.

## Fix shipped

One-line change in `crates/rdna-compute/src/dispatch.rs::rocblas_min_batch`:

```rust
// gfx94x default under HIPFIRE_GFX942_NATIVE_LM_HEAD=1
- return 16;
+ return 32;
```

Verified post-commit (no env override): dflash_spec_demo B=17 prefill
130 → 348 tok/s, decode 54.9 → 62.3 tok/s. Trunk prefill=256 unchanged
at 1303 tok/s (Tensile still chosen at this batch).

## What's still on the table

This sweep was scoped to the existing rocBLAS-vs-wave64 binary choice.
Bigger Phase B/C levers not yet probed:

1. **Tensile workgroup-mapping (WGM4/6/18) selection:** rocBLAS picks
   different Tensile kernels for different (M, N, K) shapes. WGM4 (696µs
   avg) is 3.5x slower per call than WGM18 (256µs). Could be wrong
   selection for some prefill shapes.
2. **FP16 shadow pre-pass (`hfq4g256_dequantize_to_f16`, 4.4% of GPU
   time):** caching the dequantized FP16 weights once per layer would
   eliminate redundant dequant work across batched calls. The Tensile
   path already does this (per the `ensure_fp16_shadow` helper), but
   re-checks per-call.
3. **DeltaNet projections (`w_z`, `w_beta`, `w_alpha`):** not visible
   in the top-10. Either small enough to be ignorable or routed through
   a path that wasn't profiled.
4. **MFMA-direct quantized kernels:** the current "native wave64" path
   for HFQ4 uses GEMV-style scalar dispatch. A real MFMA kernel that
   reads HFQ4 directly (skipping the dequant-to-FP16 pre-pass + the
   rocBLAS launch) could close the gap to gfx1100's 199 tok/s reference.
   This is Phase C work (kernel writing) not Phase B (dispatch).

## Cost

~$5 incremental rental for this sweep. Cumulative session ~$27.

## Recommendation

Phase A shipped. Phase B candidates (dispatch fixes that don't require
kernel writes): WGM auto-selection investigation, FP16 shadow caching
audit, DeltaNet projection dispatch check. Phase C (MFMA-direct
quantized kernels) is the long-horizon win.
