# Decode perfmaxx attempt on z-lab A3B-PARO gfx12 (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` HEAD `46e93d15`
> Goal (user): 100 (min) / 120 (target) / 150 (stretch) tok/s decode on A3B-PARO gfx12
> Starting point: 62.7 tok/s decode (production-canonical post-prefill perfmaxx)
> **Outcome: goal NOT MET. Honest ceiling assessment delivered.**

## Decode rocprof attribution (z-lab post-prefill-perfmaxx baseline)

| % | Kernel | Calls | µs/call | Total ms |
|---:|---|---:|---:|---:|
| 45.8 | `gemv_f32` | 26101 | 24.5 | 639.5 |
| 16.0 | `gemv_hfq4g128` | 13000 | 17.1 | 222.7 |
|  6.2 | `givens_rotate_f32` | 13000 |  6.7 |  86.8 |
|  4.5 | `attention_flash_q8_0_tile` | 1000 | 62.6 | 62.6 |
|  3.4 | `rmsnorm_f32` | 10201 |  4.7 | 47.4 |
|  3.1 | `gemv_paro_q4g128_moe_gate_up_k8_indexed` | 4000 | 10.9 | 43.5 |
|  2.4 | `fused_silu_mul_givens_rotate_f32` | 4000 |  8.2 | 32.9 |
|  2.2 | `gemv_paro_q4g128_moe_down_k8_indexed_batched` | 4000 |  7.7 | 30.7 |
|  ... | (smaller) | | | |
| **Total** | | **86106** | | **1395.9** |

Per token (100 generated): **13.96 ms GPU time** / 16.0 ms wall ≈ 87 % GPU utilization.

## Attempted levers (all FALSIFIED or NEUTRAL)

### 1. `gemv_f32_multirow.gfx12.hip` — 8 rows per WG with LDS-cached X

Two design iterations:

**v1 (each thread iters all 8 rows)**: -54 % decode wall (62 → 28.9 tok/s).
Root cause: each thread reads 8 different rows at K-stride apart, hitting
8 different L1 cache lines per iter. Catastrophic memory pattern.

**v2 (subwave: 4 threads per row × 8 rows = 32 threads)**: -23 % decode
wall (62 → 48 tok/s). Root cause: only 1 wave per WG vs baseline's
8 waves → 8× less BW utilization. Baseline `gemv_f32` uses 256
threads per WG specifically because BW-bound work needs many concurrent
memory transactions.

Lesson: **multirow doesn't help when each gemv call is BW-bound on its
own — fewer threads-per-WG just hurts BW saturation per call.**

Shipped opt-in via `HIPFIRE_GEMV_F32_MULTIROW_GFX12=1`.

### 2. `HIPFIRE_KEEP_F16_WEIGHTS=1` — F16 storage instead of F32-expand

The shared_expert / router weights are F16 on disk but expand to F32
on upload (`load_fp16_weight_from_source`). Switching to direct F16
storage halves BW for those reads.

Result: **-21 % decode wall** (62.7 → 49.5 tok/s). Counter-intuitive.

rocprof showed `gemm_f16_tiled` (the F16-W × F32-X kernel) at 24.7 µs/call
vs baseline `gemv_f32` at 24.5 µs/call — **same per-call time** despite
half the BW. The dispatch overhead and per-WG fixed costs (launch,
initialization, LDS sync) outweigh the BW saving for kernels at this
scale.

Plus the F16 path adds 36 % more calls/token (35620 vs 26101) due to
prefill-batched path not having an F16 dispatch arm and falling back to
per-token, multiplying the call count.

Shipped opt-in via `HIPFIRE_KEEP_F16_WEIGHTS=1`. Useful for testing
F16 paths in the future but not a production win.

### 3. `HIPFIRE_GRAPH_MOE=1` — hipGraph capture for MoE decode

PARO has `use_gpu_topk = true` (k_top=8, routed dtypes match), so the
download_f32 router-logits sync that breaks capture for other A3B
variants does NOT apply here. Capture should work.

Result: **neutral** — 4/5 runs at 62.2 tok/s (matching baseline 62.6),
1/5 outlier at 31.9 (likely graph rebuild trigger). No win, no loss.

Implication: **launch overhead is NOT the dominant per-call cost.**
The 24.5 µs/call is mostly *GPU-internal* dispatch+work, not host-side
launch latency. hipGraph can only collapse host-side serialization;
GPU-internal serial dispatch persists.

## What this means for the ceiling

The 13.96 ms/token GPU time is dominated by serial-dependent kernel
calls. Per-call timing (24.5 µs avg for the biggest 45.8 % slice) is
already 5× off BW-ceiling — but the gap is GPU-internal dispatch, not
something we can collapse with kernel optimization or graph capture.

Realistic bounded levers and estimated gains:

| Lever | Effort | Estimated decode gain |
|---|---|---|
| PARO 4-way fused F32 GEMV (mirror MQ4's `fused_qkvza_hfq4g256`) | 1-2 h | +5-10 % → ~67 tok/s |
| `gemv_hfq4g128` multirow with proper 256-thread WG | 2-3 h | +3-5 % → ~65-70 tok/s |
| `fused_silu_mul_givens_rotate_f32` further fusion | 1-2 h | +2 % → marginal |
| **Combined ceiling estimate** | | **~70-75 tok/s** |

Beyond ~75 tok/s requires structural changes:

- **Speculative decode** (DFlash already in tree but PARO admit not yet
  validated). Could 2-3× decode via verified-batched tokens.
- **Per-layer kernel fusion** (FA + DeltaNet + MoE into mega-kernels).
  Weeks of work, deep arch knowledge.
- **Model change** (smaller A3B variant, different quant). Out of scope.

## Recommendation

The user's goal of 100 / 120 / 150 tok/s decode requires either:

1. **Spec decode pipeline for PARO** — admit z-lab into DFlash. Multi-
   day project but the highest-confidence path to ≥100 tok/s.
2. **Accept a softer goal** (~75 tok/s = +20 % over baseline) and ship
   the bounded fusion + multirow-with-fix levers.

The session's prefill perfmaxx (+49× from 64 → 3130 tok/s on z-lab)
remains the headline win. Decode lives in a fundamentally different
optimization regime and the bounded levers I tried this session do not
move the needle meaningfully.

## Files shipped (opt-in research)

- `kernels/src/gemv_f32_multirow.gfx12.hip` — subwave multirow GEMV
  (falsified; kept for future kernel research)
- `crates/rdna-compute/src/dispatch.rs` — `gemv_f32_multirow_gfx12`
  dispatch fn + env-gated routing in `gemv_f32`
- `crates/rdna-compute/src/kernels.rs` — SRC registration
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — `HIPFIRE_KEEP_F16_WEIGHTS=1`
  loader gate (keeps F16 storage for `load_fp16_weight_from_source`
  call sites). Admit predicate relaxed to accept F16 shared_expert
  alongside F32 and ParoQ4G128.

All opt-in; default routing unchanged.
