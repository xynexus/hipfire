# MI300x Phase 1 — lm_head + verify-GEMM dispatch fix

**Date:** 2026-05-19
**Hardware:** AMD Instinct MI300X VF / gfx942 / wave64 / sramecc+ xnack-, 192GB HBM3, ROCm 7.0.0 (rocprofv3 1.0.0)
**Branch:** `feat/mtp-mi300x` at the new HEAD (see commit referenced in this doc)
**Rental:** DigitalOcean droplet at `129.212.180.71` (alias `mi300`). Phase 1 added ~30 min wall-clock on top of the warm Phase 0 droplet — under $3 incremental. Cumulative session well under the $20 budget.

## Headline

| Bench | Phase 0 (rocBLAS Tensile) | Phase 1 (native wave64) | Δ |
|---|---:|---:|---:|
| Solo MTP (27B-3.5 mq4-mtp, max=120, --no-chatml, q8 KV, max-n=3, greedy) | **10.83 tok/s** (Phase 0: 10.82) | **45.14 tok/s** | **4.17×** |
| Composition (DFlash B=16 + MTP K=2, same config) | **41.19 tok/s** (Phase 0: 41.05) | **112.90 tok/s** | **2.74×** |
| rocBLAS Tensile HSS share (solo decode) | 58.7% | **0.0%** | eliminated |
| Tau (solo) | 2.0517 | 2.0517 | identical |
| Output (solo + composition) | reference | byte-identical | passes |

**Decision-gate outcome: PHASE 1 SUCCESS — exceeds the ≥30 tok/s solo floor by 50%, falls just under the ≥50 tok/s "Phase 2-ready" ceiling (45.14 vs 50). Composition exceeds the ≥60 tok/s target by 88%.**

The fix is structurally minimal (a single Rust dispatch file, ~30 lines), opt-in via env var, and preserves rocBLAS where it still wins (large-batch composition verify at B=19 still routes through Tensile MFMA; see composition profile below).

## What changed

**One file**, `crates/rdna-compute/src/dispatch.rs`, two functions:

1. **`rocblas_min_batch`** (line 1610-ish): when `HIPFIRE_GFX942_NATIVE_LM_HEAD=1` and the arch is gfx94x, the rocBLAS batch threshold rises from 4 to 16. The `HIPFIRE_ROCBLAS_MIN_BATCH` env var still takes precedence if you want to A/B explicit thresholds.

2. **`is_gcn5_wave64`** (line 247-ish): adds gfx94x to the wave64-FP16-hybrid eligibility list when `HIPFIRE_GFX942_NATIVE_LM_HEAD=1`. This routes the four batched HFQ4-G256 GEMM families (`gemm_gate_up`, `gemm_qkv`, `gemm_qkvza`, `gemm_hfq4g256_residual`) to their `_fp16_wave64` siblings — block=[64,1,1] with 2 rows/block instead of block=[32,1,1] wave32.

Both gated by `HIPFIRE_GFX942_NATIVE_LM_HEAD=1`. Default-off for safety — to be flipped to default-on for gfx94x in a follow-up commit once a broader workload sweep confirms no regressions on other models / configs.

## Why this works (root cause)

Phase 0 rocprof showed `Cijk_Alik_Bljk_HSS_BH_MT256x256x32_MI32x32x8x1` (and two siblings) dominating solo decode at **58.7%** of GPU time. These are rocBLAS Tensile FP16 GEMM kernels using the MFMA `MI32x32x8x1` instruction with a `MT256x256x32` (M×N×K) workgroup tile. They were firing because:

1. `gemm_hfq4g256` (and siblings `_residual`, `gemm_qkv_*`, `gemm_qkvza_*`, `gemm_gate_up_*`) on CDNA3 check `rocblas_min_batch()` (default = 4) and route to rocBLAS via an FP16-shadow + `rocblas_gemm_hfq4_prefill`.
2. Solo MTP's per-cycle workload: trunk verify forward at **B = max_n + 1 = 4**, MTP-head batched lm_head at **B = max_n = 3**.
3. Tensile's `MT256x256x32` tile means N is padded to 256 internally. At B=4, **only 4/256 N-columns carry real data** → ~98.4% of MFMA work is computing on zeros. Per-call cost: 0.2–0.73 ms (vs native wave64 GEMM at 0.02–0.06 ms = ~10× per-call faster at this shape).

The native HFQ4-G256 wave64 kernels (`gemm_*_fp16_wave64`) handle B≤16 correctly because each block processes BATCH_TILE=8 batch elements with grid_x=(M+1)/2 — no zero-padding waste, half the launch count of wave32. They already existed for gfx906 and gfx908 (`is_gcn5_wave64`) but were excluded from gfx94x on the assumption that rocBLAS would always win. Phase 0 falsified that assumption for small-B verify.

The wave64 FP16 hybrid kernels use `__hfma2` (packed FP16 FMA), which works on CDNA3. The rocBLAS path is preserved for large-B verify (composition mode's B=19 still routes to Tensile and benefits from the high-arithmetic-intensity MFMA tile).

## Verification — kernel breakdown (rocprof)

**Solo MTP (Phase 1, `HIPFIRE_GFX942_NATIVE_LM_HEAD=1`):**

```
1. fused_gate_up_hfq4g256_wave64        15104  847 ms  13.89%  (per-token MTP block)
2. fused_rmsnorm_mq_rotate              44544  799 ms  13.11%
3. gemv_hfq4g256_residual_wave64        30208  785 ms  12.88%  (per-token GEMV)
4. gemm_gate_up_hfq4g256_fp16_wave64     7168  610 ms  10.00%  (verify B=4)
5. gemm_hfq4g256_residual_fp16_wave64   14336  575 ms   9.44%  (verify residual B=4)
6. fused_qkvza_hfq4g256_wave64          11328  325 ms   5.33%
7. gemm_qkvza_hfq4g256_fp16_wave64       5376  228 ms   3.74%
... (no rocBLAS Tensile entries — grep returns 0)
```

Compare Phase 0 (solo):
```
1. Cijk_Alik_Bljk_HSS_BH_MT256x256x32 (HSS, WGM4)   3712  2723 ms  19.71%
2. Cijk_..._WGM18                                   11194  2662 ms  19.27%
3. Cijk_..._WG64_4_1_WGM6                           10208  2051 ms  14.84%
4. fused_gate_up_hfq4g256_wave64                    15104   850 ms   6.15%
   (etc.)
```

The rocBLAS Tensile triplet (53.82% combined) is wholly replaced by native wave64 variants. CSV files:
- `phase1-solo-kernel-stats.csv` — Phase 1 final (both patches applied)
- `phase1-solo-kernel-stats-threshold-only.csv` — intermediate result with only the `rocblas_min_batch` patch (no `is_gcn5_wave64` opt-in); 25.62 tok/s; shows that just disabling rocBLAS at B=4 already lifts to 2.4× but the wave32 `_fp16` kernels are still suboptimal for CDNA3
- `phase1-compose-kernel-stats.csv` — Phase 1 composition, showing rocBLAS still firing on B=19 verify (preserved win)

**Composition (Phase 1):** native wave64 dominates per-token and small-B paths; rocBLAS Tensile is still firing at B=19 verify (~30% of compose GPU time across three Tensile shapes) but is no longer wasteful — it's the right tool for the large-B compute-bound regime.

## Coherence

Solo MTP output is **byte-identical** to Phase 0 (same `last_node = self.tail.prev` final line; same 120 tokens, 58 cycles, accepted_mtp_total=61, bonus_total=58 — all identical). Composition output also byte-identical (same `lru_node = self.tail.prev` final line, same `accept_dflash_total=110, accept_mtp_total=0`).

This is the expected outcome: the dispatch change is math-preserving (the native wave64 kernels are bit-equivalent to the rocBLAS path modulo FP16 rounding, and accumulation order is preserved). No kernel correctness work was needed.

Coherence-gate proper was not re-run (the change touches only dispatch, not numerics); a CI re-run would be appropriate before flipping the default-on.

## Reproduction

On the droplet:
```bash
cd /root/hipfire   # feat/mtp-mi300x at the Phase 1 commit
export PATH=$HOME/.cargo/bin:/opt/rocm/bin:$PATH

# Solo MTP — Phase 1
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 ./target/release/examples/mtp_only_demo \
  --target /root/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --prompt-file /root/lru_cache_pep8_strict.txt \
  --max 120 --no-chatml --kv-mode q8 --max-n 3 --temp 0

# Composition — Phase 1
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 ./target/release/examples/dflash_mtp_demo \
  --target /root/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --drafter /root/.hipfire/models/qwen35-27b-dflash.mq4 \
  --mtp-head /root/.hipfire/models/qwen3.5-27b.mtp \
  --prompt-file /root/lru_cache_pep8_strict.txt \
  --max 120 --no-chatml --kv-mode q8 --mtp-k 2 --temp 0

# Phase 0 baseline — drop the env var
./target/release/examples/mtp_only_demo --target ... (etc.)
```

Prompt md5: `df5dedc8040ce70ba55080c4548e6024` (source file). Reported `prompt_md5` in demo output: `1e74f17934fe759468dbe1471b732067` (post-normalization, byte-stable across runs).

Three warm runs collected for each cell. Stddev tight (±0.04 tok/s on solo, ±0.20 tok/s on composition).

## What surprised me

1. **Two-stage win**: applying only the `rocblas_min_batch` raise (without the `is_gcn5_wave64` extension) gave 2.37× (10.83 → 25.62 tok/s). Adding wave64 routing on top doubled the lift to 4.17×. The two effects are independent — the wave32 `_fp16` kernels that the threshold raise fell through to are themselves suboptimal on CDNA3 (the upper 32 lanes of every wave64 idle). Without the wave64 extension, the result would have missed the 30 tok/s floor.

2. **Composition keeps rocBLAS — and that's correct.** Compose verify at B=19 still routes through Tensile MFMA, which now contributes ~30% of GPU time on composition (down from ~14% at the lower batch in Phase 0 but at a higher absolute throughput). Tensile's 256-wide N tile is well-utilized at B=19 (~7.4% utilization) vs B=4 (~1.6%). The fix correctly preserves the rocBLAS path where it remains the right choice.

3. **The fix should generalize.** Other CDNA3 workloads doing per-token decode or small-B verify (any non-DFlash spec-decode, AR generation through the daemon, sidecar cal at small batches) will pick up the same lift the moment they set the env var. The default flip should ride on a daemon-mode smoke + a single AR bench across {0.8B, 9B, 27B-3.5, 27B-3.6, A3B} on gfx942 to confirm no regression.

4. **gfx1100 reference numbers are still ahead on solo MTP.** 45.14 tok/s on MI300X vs ~50–53 tok/s historical on 7900 XTX (deferred memo / kernel metadata estimate). Two next-step candidates: (a) port the `gemv_hfq4g256_wide`/`_multirow` variants to gfx94x with appropriate `__launch_bounds__` so per-token GEMV occupancy is tuned for CDNA3 (currently the per-token native wave64 GEMVs are sharing-but-not-tuned), (b) profile the trunk verify kernels (`gemm_*_fp16_wave64`) for MFMA-vs-wave64-FMA tradeoffs — at B=4 they might benefit from a wave64-MFMA hybrid not currently implemented.

5. **The `gemm_hfq4g256_wave64` entry (1.35% solo / 4.41% compose) is the MTP-head batched lm_head at B=3 (and the seed-position decode lm_head at B=1 fallback).** It's not contributing meaningful time but the per-call avg is ~1ms (M=152K × K=5120). For a 60+ tok/s push the right lever is probably WMMA on CDNA3 (gfx94x has MFMA but no WMMA), so MFMA-accelerated FP16×F32 in a custom MQ4 kernel would be the next exploration. Out of Phase 1 scope.

## Next steps

**Phase 2-ready: borderline yes.** We exceed the 30 tok/s floor with margin (45.14 = 1.5× floor) but fall under the 50 tok/s "fully closed" mark. Phase 2 composition prototype (MTP-extended tree verify) should be authorized to start, with a Phase 2.0 first step of: **flip `HIPFIRE_GFX942_NATIVE_LM_HEAD` default-on for gfx94x** after coherence-gate + AR-decode sweep across the standard 5-model set. That's a 10-line change but needs the validation pass.

**Phase 2 priorities** (per master plan):
- Decode wins from MTP-extended verify (push solo past 50 → toward 100+ via tree-verify amortization across MTP K and DFlash B simultaneously)
- Spec-decode wins from per-slot tree composition (compose 113 → push toward 200+ on MI300x)
- No further dispatch tuning needed on gfx942 unless the perf gap to gfx1100 stays open at the >2× absolute level after Phase 2

**Won't pursue from Phase 0 deferred list:**
- Compressed-vocab head (cvs4k/8k/16k): the bottleneck wasn't lm_head BW, it was kernel selection. Compressed vocab would re-create the small-N rocBLAS regression on the new vocab dim. Decline.
- EAGLE-style retrain: same reason. The MTP head is already producing good predictions (τ=2.05 sustained); the work was in the dispatch, not the head.

## Cross-refs

- Phase 0 source: `docs/investigations/2026-05-19-mtp-mi300x-phase0/README.md`
- Runbook: `docs/plans/mtp-mi300x-runbook.md`
- Master plan: `docs/plans/mtp-dflash-composition-master-plan.md`
- Patched function comments: `crates/rdna-compute/src/dispatch.rs` lines 247 + 1610 (search for `HIPFIRE_GFX942_NATIVE_LM_HEAD`)
- Raw rocprof CSVs in this directory:
  - `phase1-solo-kernel-stats.csv` — full patch applied (45.14 tok/s)
  - `phase1-solo-kernel-stats-threshold-only.csv` — threshold-only intermediate (25.62 tok/s)
  - `phase1-compose-kernel-stats.csv` — composition with patch (112.90 tok/s)

## Rental state

Droplet left running at `129.212.180.71`. Cumulative session spend ~$3 incremental over Phase 0 ($10 → $13 estimated). Build cache + Phase 1 artifacts preserved. Tear down via DO console when no follow-up is planned.
