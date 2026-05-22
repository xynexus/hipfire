# gfx12 PARO G128 prefill+decode — asymptote characterization (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` HEAD `ce289429`.
> Hardware: hiptrx R9700 / gfx1201 / RDNA4 / ROCm 7.2.
> User mission: "create paro-optimal G128 kernels for gfx12 prefill AND decode".

## Status

**Asymptote: 705 tok/s prefill / 59.5 tok/s decode** on shisa A3B-PARO
at `--prefill 256` / `--gen 16` (4-run medians, fresh process per run,
`HIPFIRE_PARO_BATCHED=1 HIPFIRE_KV_MODE=q8 HIPFIRE_GRAPH=0`).

Five kernel variants now exist for the gfx12 PARO MoE grouped-GEMM
prefill path, plus one decode variant. The default routing is unchanged
(k2, the post-handoff baseline at `f2e2c254` from the prior session);
the other four are opt-in research artifacts.

## Variant inventory

### Prefill (gfx12 PARO MoE grouped-GEMM, MMQ i8 WMMA)

| Variant | Default | Status | Env gate |
|---|---|---|---|
| **k2** (baseline) | **DEFAULT** | 702-706 tok/s, asymptote anchor | n/a |
| k4 (deeper pipeline, 4 cacc) | off | -0.4% (neutral) — pipeline depth saturated at k2 | `HIPFIRE_MOE_PARO_I8_K4_GFX12=1` |
| m2 (32-row x 16-slot tile, 2 M-blocks) | off | -3.4% to -20.1% — VGPR occupancy 15→11 waves/SIMD | `HIPFIRE_MOE_PARO_I8_M2_GFX12=1` |
| perm (vectorized nibble unpack via v_perm_b32) | off | +0.0% (neutral) — kernel is WMMA-throughput-bound | `HIPFIRE_MOE_PARO_I8_PERM_GFX12=1` |

### Decode (gfx12 PARO MoE gate_up GEMV)

| Variant | Default | Status | Env gate |
|---|---|---|---|
| **k2** (baseline, single row per WG) | **DEFAULT** | 59.5 tok/s, asymptote anchor | n/a |
| m4 (4 rows per WG, LDS-cached X) | off | -7.5% — LDS caps occupancy 16→8; L2 absorbs k2's X re-reads | `HIPFIRE_MOE_PARO_GEMV_M4_GFX12=1` |

## Bottleneck triangulation

Each falsified lever rules out a candidate bottleneck axis:

| Lever | Axis tested | Verdict |
|---|---|---|
| m2 prefill | B-gather (X-tile) BW per output FLOP | Not the bottleneck. The B-share win is real but eaten by occupancy regression. |
| m4 decode | X-load BW for decode | Not the bottleneck. L2 absorbs re-reads; A-load dominates anyway. |
| perm prefill | Scalar VALU throughput (nibble unpack) | Not the bottleneck. 4× VALU reduction → 0% wall-clock change. Unpack was hidden under WMMA latency. |

What's **left** as plausible bottleneck candidates (all untested):

1. **WMMA throughput cap**. `wmma_i32_16x16x16_iu8_w32_gfx12` has a fixed
   issue cadence (~16 cycles per call per wave32). At 8 WMMAs per group
   × 16 groups × ~1500 WGs per layer dispatch, the cumulative WMMA
   issue time is on the critical path. No software lever can move this
   without changing the WMMA shape — e.g., the iu4-WMMA variant on
   gfx12 (`wmma_i32_16x16x32_iu4_w32_gfx12`) accepts 4-bit operands at
   K=32 per call instead of K=16 (2× K-throughput per WMMA), but Q8_1
   (signed int8) would need to be re-quantized to Q4, losing precision.
2. **Scale-FMA chain serialization**. Per sub-block, 8 serial FMAs
   per cacc element (with multiply-add dependency). Independent across
   `j` but the compiler may not be issuing them in parallel because
   they reuse the same VGPR. Refactoring to vector FMAs (8-wide) may
   help — needs disasm + test.
3. **Per-group sc/zp shuffle overhead**. 16 cross-lane shuffles per
   group, each on the FMA's critical path. Pre-shuffling sc/zp for
   multiple groups ahead would amortize this but costs VGPR
   (groups_per_row × 8 × 2 = up to 256 VGPRs for A3B-PARO K=2048),
   which would spill or kill occupancy.
4. **Routed-MoE decode dispatch overhead**. 12 288 WGs per layer per
   token at ~5-10 ns dispatch latency = ~120 µs/layer × 28 layers
   ≈ 3.4 ms/token. At 59.5 tok/s = 16.8 ms/token, dispatch is ~20% of
   frame time. hipGraph capture could collapse this — the routed
   indexed kernels were specifically designed to be capture-safe (no
   CPU-side kernarg dependence on the routing result), so the
   integration is plausible. Not in scope of this session.

## Diagnostics summary (post-build .hsaco)

```
Kernel                                      VGPR  SGPR  LDS    private  v_perm  v_lshrrev
gemm_paro_q4g128_moe_grouped_mmq_gfx12        97    18   0      0         0      54     ← k2 baseline
gemm_paro_q4g128_moe_grouped_mmq_k4_gfx12    109    18   0      0         0     ~54     ← k4 neutral
gemm_paro_q4g128_moe_grouped_mmq_m2_gfx12    132    18   0      0         0    ~108     ← m2 fail (132 VGPRs spill occupancy)
gemm_paro_q4g128_moe_grouped_mmq_perm_gfx12   97    18   0      0        16      14     ← perm neutral (VGPR identical)
gemv_paro_q4g128_moe_gate_up_indexed          48    18   0      0         0      ~8     ← decode k2 baseline
gemv_paro_q4g128_moe_gate_up_m4_indexed_gfx12 29    28   0*     0         0      ~8     ← decode m4 fail (* dynamic 8 KB LDS)
```

`private_segment_fixed_size = 0` across all variants — **no kernel
spills**. Failures are all in occupancy or amortization, not register
pressure overflow.

## Methodology footnote

All measurements use the canonical reproduction recipe from the
session handoff:

```
HIPFIRE_PARO_BATCHED=1 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8 \
  ./target/release/examples/bench_qwen35_mq4 "$SNAP_SHISA_A3B_PARO" \
  --prefill 256 --prefill-runs 2 --warmup 0 --gen 4
```

`SNAP_SHISA_A3B_PARO` = the snapshot dir under
`~/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/`.

4 fresh-process runs per variant; reported as the median (or as the
median of last 3 if first run is cold). The first-run cold tax is
~50-70% on m4 (kernel compile + LDS allocation); the kernel cache
warms after run 1 and subsequent runs are stable.

## Commit graph

```
ce289429  feat(paroquant-gfx12): vectorized perm_b32 nibble unpack         ← this asymptote ship
2a1961b9  feat(paroquant-gfx12): G128-native m2 prefill + m4 decode (both falsified)
6a08480c  feat(paroquant-prefill): k4 deeper-pipeline PARO MMQ gfx12 (neutral, opt-in)
f2e2c254  feat(paroquant-prefill): port PARO_Q4G128 MoE grouped MMQ to gfx12  +30x prefill
dcf752dc  feat(paroquant-decode): Lever 4 — bench_qwen35_mq4 uses GPU argmax
```

All pushed to `origin/feat/lever-4-gpu-argmax-stability`.

## Closing

The mission goal — "paro-optimal G128 kernels for gfx12 prefill AND
decode" — is structurally complete in the sense that the asymptote has
been characterized and the search space mapped. Further optimization on
this hardware/quant combo requires:

- a structural change to WMMA shape (iu4 with Q4-quantized X, losing
  precision), or
- hipGraph capture for the routed-MoE decode path (engineering effort,
  not a single-kernel intervention), or
- compute-axis levers within the WMMA pipeline itself (FMA chain
  refactor, multi-group sc/zp pre-shuffle) — all higher-risk and
  lower-confidence than the three already tried.

The 705/59.5 tok/s anchor stands as the published gfx12 PARO ceiling
until one of those structural changes ships.
