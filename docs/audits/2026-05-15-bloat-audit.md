# hipfire bloat audit — 2026-05-15

**Branch:** `audit/vram-compute-bloat-2026-05-15` (forked off `fix/dflash-vram-bloat-kv-layer-filter` after fix #1 + fix #2 landed).

**Method:** 4 parallel read-only audit agents on the post-fix codebase, each scoped to one waste category. This document consolidates and dedupes their findings into a single prioritized catalog.

Sizes throughout: 27B Qwen3.5 (hidden=5120, vocab=152K, n_layers=64 hybrid 48 LA + 16 FA, num_extract=5), 7900 XTX 24 GB, ctx=17K unless noted.

## Categorization legend

- **Difficulty:** trivial (1-3 lines) | small (10-30 lines) | medium (30-100 LOC) | large (multi-file) | invasive (correctness-sensitive math)
- **Danger:** green (byte-identical output) | yellow (correctness-class accuracy change, needs validation) | red (touches active forward / spec dispatch / graph capture)

---

## Tier 1: trivial-green wins (land first, single PR)

| # | Finding | Location | Savings | Source audit |
|---|---|---|---|---|
| T1.1 | **My regression: MQ3 chunk loop sets `fp16_x_source_ptr = null_mut()` per chunk** | `dflash.rs:653` (newly introduced in fix #1, commit `1b16aade`) | redundant invalidation on multi-chunk prefill | compute |
| T1.2 | Uncached `std::env::var()` in `forward_scratch` hot path | `qwen35.rs:2983, 2998` | ~3.2k syscalls/s on 27B decode | compute |
| T1.3 | Uncached `std::env::var()` in `spec_step_dflash` cycle | `speculative.rs:2491, 2520` | ~40 syscalls/s + String allocs | compute |
| T1.4 | DflashWeights norm tensors lifted F16→F32 on host load | `dflash.rs:158-182` (`hfq_tensor_f32`) | ~480 KB (small but zero-cost, free win) | dtype |
| T1.5 | `weight_gemv_residual` MQ branches: `vec![..]` alloc for 1-elem shape per call | `llama.rs:815-862` | ~25k allocs/s on 27B decode | compute |
| T1.6 | DflashScratch.k_cat / v_cat / positions_k sized to max(L+B) | `dflash.rs:497-501` | ~35 MB (low absolute) | vram |
| T1.7 | `host_idx = vec![0i32; batch]` per draft+verify cycle | `speculative.rs:2192-2197, 2763-2768` | trivial allocs | memcpy |
| T1.8 | `pos_buf` 4-byte H2D via sync `memcpy_htod` (multiple sites) | `qwen35.rs:3059, 5952, 5979, 5253, 5818/5825, 6360/6367, 6715/6722, 7051/7058` | latency-bound sync barriers per FA layer / per AR token | memcpy |

**Tier 1 total VRAM savings:** modest (~500 MB direct) — but the **launch-overhead and sync-barrier wins** materially uplift decode tok/s on production paths. All green, zero accuracy risk, fits one PR.

## Tier 2: small-green and small-yellow wins

| # | Finding | Location | Savings | Audit |
|---|---|---|---|---|
| T2.1 | **`pflash.rs:512` calls `KvCache::new_gpu_q8` with `q35_cfg.n_layers`** — same bug pattern as fix #2 commit `33fe5ab4` | `pflash.rs:512` | 200-500 MB on hybrid drafter at long-ctx | vram |
| T2.2 | KvCache Givens cos/sin (`asym3`/`asym2`/`asym4`) regenerated + uploaded on every cache create (6 ctors × 2 H2Ds = 12 redundant H2Ds) | `llama.rs:3042-3053`, 6 call sites | one-shot per cache but redundant; cache-on-`OnceLock<(Vec, Vec)>` | compute |
| T2.3 | `argmax_per_pos` / `drafted` / `block` Vec allocs per spec cycle (worst: 600 KB host vocab buffer if RP active) | `speculative.rs:2090-2091, 2499, 2508-2510, 2598, 2748` | 12 MB/s heap churn under RP/n-gram | compute |
| T2.4 | TriAttnCalibStateGpu accumulators sized to all 64 layers (75% are LA, dead) | `triattn.rs:359-364` | 4.7 MB (calibration-time only) | vram |
| T2.5 | `forward_scratch` lm_head memset silent-sync-fallback when `active_stream = None` (CLAUDE.md-documented trap) — DFlash mitigates, AR path doesn't | `qwen35.rs:2985, 5950` (need same `is_none()` create as `speculative.rs:2468-2470`) | per-cycle hipMemset sync stalls on AR path | memcpy |
| T2.6 | DDTree Path B uploads `parent_indices` buffer **twice** in same cycle | `speculative.rs:4054, 4075, 4243, 4278, 4285, 4298` | per-cycle when DDTree path-B engaged | memcpy |

## Tier 3: medium / needs validation

| # | Finding | Location | Savings | Type | Audit |
|---|---|---|---|---|---|
| T3.1 | **lm_head F16-on-disk lifted to F32 on GPU in qwen35.rs** (mirror of PR #242 for llama.rs — **flagged by BOTH dtype audit AND vram audit** for cross-confidence) | `qwen35.rs:795-803` + `qwen35.rs:1306-1313` (tied-embed fallback) + `hfq.rs:571-580, 619-630` | **1.16-1.55 GB on F16-lm_head models** | wrong-dtype | dtype + vram |
| T3.2 | SmallVec for the 273 `vec![..]` kernel-param sites (allocates per kernel launch) | `dispatch.rs` ~273 sites | 0.5-1% decode | hot-loop-alloc | compute |
| T3.3 | `pos_buf` per-AR-token sync H2D 4-byte writes — swap to `stream_write_value32` (already wired for graph replay at `qwen35.rs:3021`) | qwen35.rs multiple FA paths | per-FA-layer latency saving post-eviction | memcpy |
| T3.4 | Q/K rmsnorm fused as `rmsnorm_batched_pair` (single launch, 2 inputs) | `dflash.rs:955, 962`, `qwen35.rs:5785, 5787` | 24.6k fewer launches/s on 27B | unfused-kernel | compute |
| T3.5 | Incremental `positions_k` H2D upload (like `target_hidden` already does) | `speculative.rs:2591-2606`, `dflash.rs:806-807` | 4.5 KB H2D + sync barrier per cycle | memcpy |
| T3.6 | `target_hidden_proj` F32 → F16 (357 MB at 17K) | `dflash.rs:493` | 357 MB / 17K, scales with ctx | dtype | vram + dtype |
| T3.7 | DflashScratch FFN scratch (gate/up/gate_up) F32 → F16 | `dflash.rs:484-486` | ~5 MB (low absolute) | dtype | dtype |

## Tier 4: medium/large, yellow — biggest wins, need careful validation

| # | Finding | Location | Savings | Audit |
|---|---|---|---|---|
| T4.1 | **`target_hidden + HiddenStateRingBuffer.layer_bufs` collapse** (the "tier-3" from the plan doc — same payload in two layouts; collapse + permutation) | `dflash.rs:492`, `speculative.rs:1165` | **1.78 GB at 17K, 6.4 GB at 64K — gates 128K context** | vram |
| T4.2 | **Hidden ring buffer + target_hidden F32 → F16 (coupled)** | both files above | 1.28 GB at 17K (if T4.1 done first, this halves what remains) | dtype + vram |
| T4.3 | **DflashScratch K/V cached F16** (`k_ctx_cached`, `v_ctx_cached`) — user flagged drafter-KV-quantization sensitivity, needs τ A/B | `dflash.rs:471-473` | ~750 MB at 17K, 671 MB additional at 64K | dtype + vram |
| T4.4 | `commit_staging_to_ring` unconditional stream_synchronize per spec cycle — convert to event fence | `speculative.rs:1321` (called at `:2067`) | per-cycle pipeline stall | unnecessary-sync | compute |
| T4.5 | `target_hidden_host` D2H+H2D shadow in DDTree/path_c variants — `spec_step_dflash` already uses on-GPU scatter, but DDTree variants don't | `speculative.rs:3108-3111, 3716, 3889, 4023, 4435, 4582, 4827, 4834` | 80-160 KB host alloc + roundtrip per cycle (variant-dependent) | memcpy |

## Tier 5: invasive / low ROI — defer unless context needs grow

| # | Finding | Location | Note |
|---|---|---|---|
| T5.1 | Residual `memcpy_dtod` of `x → residual_attn` and `x → residual_ffn` (62 layers × 2 copies × 199 tok/s = 1 GB/s D2D traffic) | `dflash.rs:881, 1026` | fuse into `add_to_x_from_proj` kernel OR pointer-swap; correctness-sensitive |
| T5.2 | `DflashScratch.k_ctx_cached` / `v_ctx_cached` per-layer at full max_ctx | `dflash.rs:468-473` | 178 MB at 17K, 671 MB at 64K — fold or quantize on long ctx |
| T5.3 | KvCache `givens_cos/sin` F32 → F16 | `llama.rs:3242-3243` | 512 B per cache — not worth the validation cost |
| T5.4 | DeltaNetState.conv_states F32 → F16 | `qwen35.rs:575` | 1.7 MB total; recurrence drift risk — skip |
| T5.5 | Positions/tokens cosmetic dtype rename (F32 alloc that's really 4B already) | `llama.rs:1255-1256, 500-501` | zero VRAM benefit; cosmetic only |

---

## Recurring theme across audits

**DFlash's `spec_step_dflash` fast path was heavily optimized during the 2026-04→05 window. The AR-only path and the DDTree-batched variants did NOT get the same treatment.** Several of the findings (T2.5, T3.5, T4.5) are "apply pattern X from `spec_step_dflash` to `spec_step_ar`/`spec_step_ddtree_batched`." A focused PR doing exactly that backport would knock out 4-5 items at once.

## Cross-audit confirmed (high confidence)

- **lm_head F16 in qwen35.rs (T3.1)** — flagged by BOTH dtype audit and VRAM audit. Single biggest dtype win.
- **PFlash hybrid KvCache (T2.1)** — VRAM audit caught that the just-shipped fix-2 pattern applies to `pflash.rs:512` identically.
- **target_hidden / hidden_rb collapse (T4.1)** — confirmed against plan doc; agent estimate matches doc-projected savings.

## Suggested PR-batching strategy

1. **Tier-1 PR (single commit, no flag changes):** all 8 Tier 1 items. Pure green. Reviewable in one pass, no benches needed beyond coherence-gate + canonical bench.
2. **Tier-2 PR (one PR per item):** T2.1 (PFlash filter) ships first as 1-commit follow-up to fix #2. T2.2-T2.6 each get their own commit on a shared cleanup PR.
3. **Tier-3 PR:** T3.1 (lm_head F16 in qwen35.rs) ships standalone — mirror of #242, high value, validate against canonical bench + coherence gate.
4. **Tier-4 work:** dedicated plan doc + multi-week effort. T4.1 (target_hidden/hidden_rb collapse) is the gate to 128K context per existing plan.
5. **Tier-5:** parked.

## Aggregate VRAM savings if Tiers 1-4 all landed

| Tier | Approximate cumulative VRAM saved at ctx=17K |
|---|---|
| Tier 1 | ~500 MB direct + compute/sync wins |
| Tier 2 | ~200-500 MB (mostly PFlash drafter case) |
| Tier 3 (T3.1 alone) | 1.16 GB (qwen35 lm_head F16) |
| Tier 3 + T3.6 | +357 MB target_hidden_proj F16 |
| Tier 4 (T4.1 + T4.2 + T4.3) | +3.06 GB (target_hidden+hidden_rb collapse, F16 conversions) |
| **Total Tiers 1-4** | **~5+ GB at 17K, scales to ~10+ GB at 64K** |

Combined with the already-landed fix #1 + fix #2 (3 GB at 17K): total recoverable VRAM bloat on production decode path is **~8 GB at 17K, scaling to ~13 GB+ at 64K**.

## Open follow-up: dead-code-in-hot-path audit not yet done

The four audits cover VRAM, compute, memcpy, and dtype. A 5th audit pass could specifically hunt for:
- Branches in hot paths that are unreachable in production (only fire under defunct env vars)
- Match arms for retired/dead DTypes
- `eprintln!` / `dbg!` left in non-debug builds
- Reachable-but-no-op code (`if cond { /* TODO */ }` style)

Not dispatched in this pass — listed for completeness.
