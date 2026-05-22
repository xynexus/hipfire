# Session-final summary: PARO gfx12 perfmaxxing (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` HEAD `6b992b49`
> Hardware: hiptrx R9700 / gfx1201 / RDNA4 / ROCm 7.2
> Model: z-lab/Qwen3.6-35B-A3B-PARO (canonical), shisa-ai variant (research)

## Headline

**z-lab A3B-PARO prefill: 64 → 3130 tok/s (+48.9× speedup)** with clean
coherence (0 hard fails) on all 3 humaneval prompts. Beats shisa-A3B-PARO's
2660 tok/s ceiling AND avoids shisa's humaneval_3 attractor.

## Session arc

| Stage | Commit | tok/s | Speedup | Coherence |
|---|---|---:|---|---|
| Session start (admit broken on z-lab) | (pre-session) | 64 | 1× | per-token fallback |
| F32 batched shared_expert dispatch | `f4681367` | 697 | 10.9× | 0 hard / 1 soft |
| ↳ silent-corruption layout fix | `d6e66cb7` | 697 | 10.9× | 0 hard / 1 soft (math now correct) |
| + HFQ4-WMMA-gfx12 (non-grouped LA path) | `6c097ad5` | 1547 | 24.2× | 0 hard / 1 soft |
| + shared_expert F32-WMMA per-call-site | `5a556225` | **3130** | **48.9×** | **0 hard / 1 soft (production)** |
| + broad BF16-WMMA (opt-in, shorter-EOS) | `6b992b49` | 3660 | 57.2× | 0 hard / 1 soft (EOS earlier) |

## Three pivotal moments

### 1. Rocprof revealed the "hidden lever" (`6c097ad5`)

User pushback: "if 1500 is your estimated ceiling then it is still only
50% of mq4." HIPFIRE_PROFILE-based attribution was wrong. Switching to
rocprof showed `gemm_hfq4g128` consuming 70 % of GPU time, completely
invisible to the internal serialized profiler (no `begin_timer` wrapper).
Porting it to FP16 WMMA on gfx12 was a +250 % prefill win on shisa
in one commit.

### 2. Methodology fix (`4a9bcc2c`)

Same "hidden lever" pattern as 2026-05-19's `gemm_q8_0_batched` discovery.
Baked `rocprofv3` auto-rerun into `HIPFIRE_PROFILE=1` so this can't recur:
the bench self-execs under `rocprofv3` with `HIPFIRE_ROCPROF_CSV` pre-wired.
Opt-out: `HIPFIRE_PROFILE_NO_ROCPROF=1`.

### 3. The silent-corruption near-miss (`d6e66cb7`)

User pushback: "ah you used asym3? that would do it. try fwht3 instead."
(I'd actually used q8, but the broader point landed.) Investigating the
remaining attractor on shisa, I discovered z-lab passes coherence cleanly
but admit-fails (shared_expert is F16 dense, not PARO). I added F32
shared_expert dispatch — and ALMOST shipped silent-corrupted output.

`gemm_f32_batched` writes `Y[m*N + n]` ([M × N] layout) but the
shared_expert downstream consumed [N × M] (batch-major, matching shisa's
HFQ4 sister). The math was producing scrambled values that the routed
experts absorbed into still-non-attractor output — but humaneval_2/3
only emitted 11 tokens instead of 100+. Caught it because the
coherence_probe binary was stale (built before my changes), and when I
rebuilt it the token counts dropped. Fix: swap A↔B in the gemm call to
flip output layout. The win evaporated to "+250%" → "0%" then recovered
to "+102%" with CORRECT math.

## Production-canonical configuration

```
HIPFIRE_PARO_BATCHED=1
HIPFIRE_HFQ4G128_WMMA_GFX12=1
HIPFIRE_F32_SHARED_EXPERT_WMMA_GFX12=1
HIPFIRE_MOE_PARO_FP16_GFX12=1
HIPFIRE_KV_MODE=q8

Result: 3130 tok/s prefill, 57 tok/s decode, clean coherence
```

Opt-out flips (default behavior changes):
- Without `HIPFIRE_HFQ4G128_WMMA_GFX12=1` → scalar baseline (~700 tok/s)
- Without `HIPFIRE_F32_SHARED_EXPERT_WMMA_GFX12=1` → scalar F32 GEMM (~1547 tok/s)
- Without `HIPFIRE_MOE_PARO_FP16_GFX12=1` → i8 MMQ routed (~regression, has admit-required side effects)

Opt-in researcher flags (not for production):
- `HIPFIRE_GEMM_F32_WMMA_GFX12=1` → broad FP16 (breaks router)
- `HIPFIRE_GEMM_F32_WMMA_BF16_GFX12=1` → broad BF16 (also breaks router but cleaner failure mode)
- `HIPFIRE_MOE_PARO_I8_K4_GFX12=1` → k4 deeper-pipeline (neutral)
- `HIPFIRE_MOE_PARO_I8_M2_GFX12=1` → m2 reg-blocked (regression -25%)
- `HIPFIRE_MOE_PARO_I8_PERM_GFX12=1` → perm_b32 nibble unpack (neutral)
- `HIPFIRE_MOE_PARO_GEMV_M4_GFX12=1` → m4 decode (regression -7.5%)

## What's left on the table

Levers explored but not delivering:
- **m2 M-direction reg-blocking** on G128 — VGPR occupancy cliff. Falsified
  on both i8 MMQ and FP16 WMMA variants. G128's short K-amortization
  doesn't tolerate the VGPR cost.
- **Broad BF16/FP16 F32-WMMA** — router top-K is precision-sensitive
  beyond just exponent range. Mantissa precision also matters for the
  6-of-256 expert selection.
- **k4 deeper-pipeline** on G128 — measured neutral. Pipeline depth
  capped by the 8 K-tiles per group.
- **DeltaNet register-array variant** on gfx12 — wired the pre-existing
  `gated_delta_net_q8.gfx1200.hip` (commit `afc88620`). Falsified at
  -34 % on prefill: 4 waves/head vs baseline 32 waves/head = 8× less
  parallelism. The kernel was designed for single-token decode where
  baseline does per-token requant noise; the batched-seq prefill path
  already avoids that noise, so register variant has no upside on this
  axis. Opt-in via `HIPFIRE_GDN_Q8_GFX12_REGISTER=1` for decode-path
  experimentation. Future work: split S across multiple WGs to recover
  parallelism while keeping register hot-path.

Levers untouched (estimated effort vs reward):
- **hipGraph capture for prefill** — would collapse ~30k WG dispatches per
  layer into one graph replay. Could be +20-30 % on top of current. Complex
  (CASK-style state machine for routed MoE). 2-3 days.
- **DeltaNet attention parallelism redesign** — 11.9 % of GPU time. The
  register-array variant fix needs S split across multiple WGs to recover
  the 8× parallelism lost in the single-WG-per-head pattern. Algorithm
  work, deep DeltaNet expertise required. (Plain register port to gfx12
  was tried at `afc88620` and falsified.)
- **lm_head GEMV WMMA-ization** — 4.4 % at 2 calls / 7.3 ms total. Already
  at 523 GiB/s, probably near ceiling.
- **F32→F16 pre-conversion cache in LDS** for gemm_f32_wmma — would
  amortize the inline downcast. Possibly +5-10 % on the F32-WMMA slot.
- **Fused gemm_f32_wmma + sigmoid_scaled residual** for the down step —
  saves a kernel launch + a scratch buffer pass. ~1-2 ms total.
- **PARO4G256 pivot** — would unlock m2 on G256 layout (per `docs/plans/
  paroquant-g256-pivot-2026-05-22.md`). Earlier-recommended; lower
  priority now that 3130 tok/s is achieved on G128.

## Lessons (for future kernel-perf sessions)

1. **HIPFIRE_PROFILE alone is unreliable** for attribution. The
   serialized-launch timer misses kernels that lack `begin_timer`
   wrappers — exactly the kernels that need optimization. Baked into the
   methodology now (commit `4a9bcc2c`).

2. **Rebuild coherence_probe after every library change.** Cargo doesn't
   auto-rebuild all examples even when the library changes. Stale
   coherence binaries can mask correctness regressions in fresh kernels.

3. **Layout matters more than math.** The biggest near-miss this session
   was silent layout corruption that the model partially absorbed —
   "0 hard fails" coherence was meaningless when the math was wrong.
   Rebuild + re-verify on every dispatch change.

4. **Precision-sensitivity is call-site specific.** Router top-K (small
   margin of victory between expert logits) is far more precision-
   sensitive than the bulk-FFN shared_expert path. Per-call-site WMMA
   gates dramatically beat broad auto-dispatch.

5. **The "asymptote" you measure depends on the kernel you fail to
   profile.** I called 705 tok/s the asymptote based on the wrong
   bottleneck attribution. The actual ceiling was 4× higher and visible
   only via rocprof. Don't anchor to "asymptotes" derived from incomplete
   measurement.
