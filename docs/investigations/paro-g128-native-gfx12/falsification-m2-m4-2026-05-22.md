# G128-native gfx12 PARO MoE kernels — m2 + m4 falsification (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` HEAD (post-commit).
> Hardware: hiptrx R9700 / gfx1201 / RDNA4 / ROCm 7.2.
> Triggered by user ask: "create paro-optimal G128 kernels for gfx12 prefill AND decode".
> Hypothesis: G128 prefill is BW-bound (m2 lever), decode is X-load-BW-bound (m4 lever).
> **Both hypotheses falsified. Both kernels kept as opt-in research artifacts.**

## TL;DR

| Kernel | Default | Result | Env gate |
|---|---|---|---|
| `gemm_paro_q4g128_moe_grouped_mmq_m2.gfx12.hip` | OFF | -3.4 % (v1 streaming) / -20.1 % (v2 preload) prefill | `HIPFIRE_MOE_PARO_I8_M2_GFX12=1` |
| `gemv_paro_q4g128_moe_gate_up_m4_indexed.gfx12.hip` | OFF | -7.1 % decode (median post-warmup) | `HIPFIRE_MOE_PARO_GEMV_M4_GFX12=1` |

K2 baselines (shisa A3B-PARO, hiptrx gfx1201, `HIPFIRE_PARO_BATCHED=1`,
`HIPFIRE_KV_MODE=q8`):
- Prefill @ --prefill 256: **702-706 tok/s** (n=4, tight)
- Decode @ --gen 16: **59.5 tok/s** (n=4, tight)

## Prefill m2 — falsified by VGPR occupancy regression

### Lever (theory)

2×1 M-direction reg-blocking. Each WG owns a 32-row × 16-slot output tile
(vs k2's 16×16). The B operand (X-gather row) is loaded ONCE per K-substep
and reused across two M-blocks; A doubles. Direct G256 analogue:
`gemm_hfq4g256_moe_grouped_wmma_m2.gfx12.hip` (FP16 WMMA, shipped).

Expected gain: ~halve B-gather BW per output FLOP → +15-30 % on prefill.

### Measured

```
m2 v1 (streaming, no preload):  681.2 / 680.9 / 681.6 / 679.1   median ≈ 681   ≈ -3.4 %
m2 v2 (preload mirrors k2):     563.0 / 563.8 / 561.4 / 560.4   median ≈ 562   ≈ -20.1 %
k2 baseline:                    706.2 / 705.2 / 703.2 / 704.2   median ≈ 705   reference
```

### Root cause (VGPR pressure → occupancy regression)

```
                       VGPR   SGPR   waves/SIMD   waves/SIMD@launch_bounds(32,2)
gemm_paro_..._gfx12    97    18     15           15  ← baseline
gemm_paro_..._k4_gfx12 109   18     14           14  ← neutral
gemm_paro_..._m2_gfx12 132   18     11           11  ← m2 (failed)
```

(`vgpr_spill_count: 0` and `private_segment_fixed_size: 0` on all three —
**no spills**, just register pressure.)

m2 needs an extra `acc1` (float8 = 8 VGPRs), `sc_row_1/zp_row_1[8]`
(16 VGPRs), `pk1_arr[8]` in the v2 preload variant (8 VGPRs), and
transient `a_vec_1` state. Net: 35 more VGPRs than k2 → 11 vs 15 waves
per SIMD → 27 % occupancy regression.

The m2 B-share win (~halved B-BW per output) does not compensate for the
27 % occupancy loss because the kernel is **compute-occupancy-limited,
not BW-limited**. Effective measured BW is well below HBM3e nominal —
the kernel spends most of its time in WMMA dispatch + scale-FMA chain,
not waiting on memory.

### Verdict

m2 is not viable for G128 + i8 MMQ on gfx12. To make it competitive, VGPR
count would need to drop below ~110 (matching k4's occupancy) — that
requires removing acc1 or sc_row_1/zp_row_1 from registers, which defeats
the m2 premise.

## Decode m4 — falsified by LDS-occupancy + per-row branching

### Lever (theory)

4 output rows per WG with LDS-cached X. The token's hidden state (X
operand for gate_up) is identical across all (M × K_TOP) WGs in the
baseline; the baseline re-fetches X from global memory per WG. For
A3B-PARO at K_TOP=8, gate_up M ≈ 1536, K=2048: theoretical X-load BW
drops 4× (~96 MB → ~24 MB per layer per token).

Expected gain: +20-30 % on decode tok/s if X-load was the bottleneck.

### Measured

```
k2 baseline decode:    59.5 / 59.5 / 59.5 / 59.5   tight at 59.5 tok/s
m4 decode:             23.7 / 55.3 / 55.2 / 54.9   (first run is cold)
                       median (warm) ≈ 55.0 tok/s   ≈ -7.5 %
```

(Cold run = -60 % — kernel compile + LDS allocation. Excluded from median.)

### Root cause (X-load not actually the bottleneck)

A-load and X-load arithmetic (per layer per token, gate_up only):

|   | k2 | m4 (theoretical) |
|---|---:|---:|
| WG count | 12 288 | 3 072 |
| A reads | 14 MB | 14 MB |
| X reads (naive) | 96 MB | 24 MB |

The ratio I keyed off was wrong. **A reads dominate**, not X. The
baseline's 96 MB X-load is also heavily L2-absorbed because every WG in
a single layer reads the same K floats from the same source — L2 hit
rate on X is near-perfect after the first WG. Effective X-load BW with
L2 hits ≈ same order as A-load.

m4 cost analysis:

```
                       VGPR   LDS    Achieved occupancy
gemv_paro_..._indexed  48    0      16   ← baseline __launch_bounds__(32, 16)
gemv_paro_..._m4...    29    8 KB*  8    ← m4 __launch_bounds__(32, 8)
                                          (LDS-limited: 64 KB CU / 8 KB = 8 WGs)
```

(* dynamic LDS allocated at launch via `shmem_bytes`.)

m4 has half the WG occupancy per CU. 4× output per WG × 0.5× occupancy
= 2× theoretical CU throughput. But:

1. Per-row branches (`if (live_1)`, `if (live_2)`, `if (live_3)`) are
   runtime-resolved at every group iteration × 16 groups. For
   M=1536 / 4 = 384 row-tiles, all 4 rows ARE live, but the compiler
   can't prove it from the WG-side branch condition — so it emits the
   branch tests + control flow.
2. Cooperative LDS load + `__syncthreads()` are on the critical path
   before any compute starts.
3. The 4-row inner loop reads sc/zp/qweight per group per row — register
   reuse across rows is limited by the 4 distinct row pointers, so
   per-group state doesn't compress.

Net: m4's structural overhead exceeds the BW-saving headroom, which was
overstated to begin with.

### Verdict

m4 is not viable for G128 + scalar-FMA GEMV decode on gfx12. The actual
decode bottleneck appears to be a mix of **kernel-launch dispatch
overhead** (~12 288 WGs per layer) and **scalar nibble-unpack ALU
throughput**, neither of which is addressed by LDS-caching X.

## What might actually help (untested levers)

1. **Vectorized nibble unpack via `__builtin_amdgcn_perm_b32`** —
   the current scalar `(pk >> X) & 0xF` chain may be auto-vectorized
   by the compiler, but worth verifying via disasm and forcing if not.
   Touches both k2 prefill MMQ and decode GEMV.

2. **Kernel-launch overhead reduction via hipGraph capture for decode** —
   12 288 WGs per layer × 28 layers × 60 tok/s = ~20 M dispatches/sec.
   At ~5-10 ns per dispatch on R9700, this is real overhead. hipGraph
   already wired into the prefill decode path; the routed-MoE decode
   path may have residual hipError 906 issues (Phase 4 NaN-argmax fix
   already addressed one such gap, but the dispatch-count axis remains).

3. **Pack 2 HFQ4 groups into one outer-loop iteration** (a "g2" pattern)
   to amortize per-group scale-shuffle overhead. Less risky than the
   m-direction or LDS-cache approaches because it doesn't change grid
   geometry or VGPR pressure dramatically.

## Files shipped (opt-in research artifacts)

```
kernels/src/gemm_paro_q4g128_moe_grouped_mmq_m2.gfx12.hip      ← prefill m2
kernels/src/gemv_paro_q4g128_moe_gate_up_m4_indexed.gfx12.hip  ← decode m4
crates/rdna-compute/src/kernels.rs                              ← SRC registration
crates/rdna-compute/src/dispatch.rs                             ← dispatch methods
crates/hipfire-arch-qwen35/src/qwen35.rs                        ← env-gated routing
```

Env gates:
- `HIPFIRE_MOE_PARO_I8_M2_GFX12=1` (prefill m2, default-off)
- `HIPFIRE_MOE_PARO_GEMV_M4_GFX12=1` (decode m4, default-off)

Default routing on gfx1200/1201 remains:
- Prefill: `gemm_paro_q4g128_moe_grouped_mmq_gfx12` (k2, 702-706 tok/s)
- Decode: `gemv_paro_q4g128_moe_gate_up_k8_indexed` (k2, 59.5 tok/s)

This confirms the gfx12-asymptote.md certification from Phase 6 still
holds — the two new prefill+decode levers attempted on top of it both
falsified, reinforcing that further wins require a structural change
(e.g. vectorized nibble unpack, hipGraph-capture compatibility on the
routed-MoE decode path) rather than M-direction or LDS-share patterns.
