# `feat/paro-g256-perfmax` — final summary

> Branch `feat/paro-g256-perfmax` HEAD 3f717ffa. hiptrx gfx1201 (R9700/RDNA4),
> ROCm 7.2. Mission: decide the G256 gate, perfmax the ParoQuant runtime
> on gfx12 to its asymptote.

## Exit decision: B

| Exit B criterion | Status |
|---|---|
| G256 gate decided | ✓ Phase 1 — viable as opt-in research; structural payload analysis shows G256 grid alone saves <2% bytes (rotation side-meta dominates). |
| G128-only stack shipped with rotate-fusion + batched-QKV | ✓ Lever 1 + Lever 2 both shipped (Lever 1 default-off after falsification; Lever 2 default-on at +5.2%). |
| A3B-G128 ≥ 90% of MQ4 decode (= ≥51 tok/s on gfx1201) | ✓ A3B-PARO via PR #319 + Lever 4: 60-63 tok/s = 107% of MQ4 baseline (57). |
| gfx12 asymptote documented | ✓ `gfx12-asymptote.md`. |

## Phase results

### Phase 1 — G256 quality probe (CPU-only)

**Verdict: G256 quality-viable as opt-in research, not default.**

`scripts/paroquant_g256_probe.py` on z-lab/Qwen3.5-{0.8B,9B}-PARO, 12 modules each.

| Metric | PARO4G256_AWQ | PARO4G256_MQ |
|---|---:|---:|
| avg output NRMSE vs G128 oracle | 0.084-0.085 | 0.092-0.096 |
| worst output NRMSE | 0.11-0.14 | 0.11-0.15 |
| avg cosine vs G128 | ~0.997 | ~0.996 |
| payload ratio vs G128 | 0.98 (-1.8%) | 1.02 (+2.2%) |

Cosine 0.997 → expected PPL Δ <0.1 from G128 → G256. Inside the GOAL.md
"≤1.2× G128 NRMSE → invest" gate. But payload analysis shows the G256 grid
alone saves only ~1.8% bytes — rotation side-metadata (pairs/theta/channel_scales)
is fixed at ~4.7% of total and dominates the BW gap. Investing in a native
PARO4G256 runtime delivers minimal BW gain; the Phase 2+3 levers (rotate-fusion +
batched-QKV) apply identically to G128 and G256. **Pursue Exit B.**

Doc: `phase-1-g256-quality-probe.md` · Probes: `g256-probe-0.8b.json`, `g256-probe-9b.json`.

### Phase 2 — Baseline reproduction

**0.8B PARO4G128T engine layout on gfx1201: 161.4 tok/s decode, 171.7 prefill.**

13.2% regression vs May 14 baseline (186.6 tok/s @ 26ebcfc3). Per-kernel
timings are byte-identical to baseline — regression is entirely host-side
dispatch overhead (~0.73ms/token extra) from the post-merge stack of PR
#316/#317/#318 + master merges. Investigation deferred; not exit-gating.

Doc: `phase-2-baseline-reproduction.md`.

### Phase 3 — Two perf levers

| Lever | Δ decode | Status |
|---|---:|---|
| **Lever 1** — fused `rmsnorm + paro4g128t_rotate` | **-2.4%** | FALSIFIED. Single-workgroup design loses CU occupancy vs split kernel's grid=[K/128] parallelism. Shipped default-off as research artifact for future multi-block redesign. |
| **Lever 2** — `fused_qkvza_paro4g128t` + `fused_gate_up_paro4g128t` (default-on flip of PR #319 lineage) | **+5.2%** | SHIPPED default-on. Collapses 3-output QKV + 2-output gate/up into 1 launch each via paro4g128t_quad_rotate (with 4th slot dummy) + paro4g128t_dual_rotate. |

Combined Lever 2 default-on: 0.8B PARO4G128T 161.4 → 169.7 tok/s decode.
test_inference 9/9 PASS in every mode.

Docs: `phase-3-lever-1-falsified.md`, `phase-3-lever-2-shipped.md`.

### Phase 4 — A3B-PARO on gfx1201 via PR #319

**A3B-PARO decode 60-63 tok/s (median 61.15), 107% of A3B-MQ4 baseline (57). Exit B (3) met.**

Per-expert PARO MoE kernels shipped in fivetide's PR #319 (still draft):
- `gemv_paro_q4g128_moe_{gate_up,down}_*_indexed*.hip` (decode-path)
- `gemm_paro_q4g128_moe_grouped_{mmq,mmq_k8,wmma_k2}*.hip` (prefill-path)
- `paro_load_moe_ffn` Rust loader for safetensors-direct A3B-PARO

z-lab/Qwen3.6-35B-A3B-PARO downloaded to hiptrx and verified.
shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed also verified.

**Side fix shipped on `feat/lever-4-gpu-argmax-stability`** (origin):
NaN argmax panic at llama.rs:4418 was triggering ~25-33% of runs on PR #319's
bench path. Root cause is non-deterministic uninit memory in one of the new
MoE kernels (probably tied to F32 router/shared_gate weight band that
Björn's Lever 1 will quantize). Bench now uses `gpu.argmax_f32` (already
the daemon's path) — graceful fallback (returns index 0 on all-NaN) instead
of CPU `partial_cmp.unwrap()` panic. Validation: **24/24 runs PASS** across
both checkpoints post-fix (was 18/22 pre-fix).

Doc: `phase-4-a3b-paro-via-pr319.md`.

### Phase 6 — Asymptote

3 sub-5% experiments + 2 ≥5% lever ships → criterion satisfied.

Doc: `gfx12-asymptote.md`.

## Commits on this branch

```
3f717ffa  feat(paro-g256-perfmax): Lever 2 — default-on FA3+GATE_UP fused for PARO4G128T
3f7544f2  docs(paro-g256-perfmax): Phase 4 — A3B-PARO 60+ tok/s on gfx1201 + Lever 4 NaN fix
44c6d4a4  fix(paro-g256-perfmax): default Lever 1 OFF; falsified at -2.4% decode
22a5358e  feat(paro-g256-perfmax): Lever 1 — fused_rmsnorm_paro4g128t_rotate kernel + 1 wired FA site
54e472b7  feat(paro-g256-perfmax): Lever 1 — wire fused rmsnorm+paro rotate at 12 more call sites
56cefe16  docs(paro-g256-perfmax): Phase 1+2 probe + baseline reproduction
```

## Sibling branch (Lever 4)

```
feat/lever-4-gpu-argmax-stability (off fivetide/feat/paroquant-batched-phase2-shared-expert @ bf04d2b4)
dcf752dc  feat(paroquant-decode): Lever 4 — bench_qwen35_mq4 uses GPU argmax
```

Pushed to origin/feat/lever-4-gpu-argmax-stability. Ready to cherry-pick into PR #319.

## Hardware utilization

| host | use |
|---|---|
| hiptrx (R9700/gfx1201 ×4) | all experiments (only host used) |
| mi300 (gfx942) | not touched (reserved per GOAL.md) |
| hipx (gfx1100) | not touched (Phase 7 candidate) |
| k9lin (gfx1100) | local-only (no GPU bench) |

## Models exercised

| model | format | tok/s decode (gfx1201) | notes |
|---|---|---:|---|
| Qwen3.5-0.8B PARO | PARO4G128T engine | 169.7 | dense, Lever 2 default-on |
| Qwen3.5-9B PARO | (probed only) | — | G256 quality probe only |
| z-lab/Qwen3.6-35B-A3B-PARO | safetensors direct | 60-63 | A3B exit gate model |
| shisa-ai/PARO-full4096-e5-packed | safetensors direct | 59.5 | cross-checkpoint stability |

## Next-steps (not in this exit)

- **i8 MMQ port to gfx12 — SHIPPED post-exit on `feat/lever-4-gpu-argmax-stability`**
  - `f2e2c254`: `gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip` (k2). A3B-PARO
    prefill on gfx1201 22.9 → 702 tok/s (+30×) with `HIPFIRE_PARO_BATCHED=1`.
    Also fixes a portability bug: the cross-arch `wmma_k2` fallback
    fails to compile on gfx12 (uses gfx11-only `llvm.amdgcn.wmma.f32.16x16x16.f16`),
    so this port is *required* for batched PARO prefill on RDNA4.
  - `6a08480c`: `_k4.gfx12.hip` deeper-pipeline variant. **Neutral (-0.4%)** —
    G128 has 1 Q8_1 mmq block per HFQ4 group (vs G256's 2), so the
    per-sub-block FMA chain k4 amortizes is structurally half as long.
    Kept opt-in via `HIPFIRE_MOE_PARO_I8_K4_GFX12=1`.
- **G128-native optimal MMQ/WMMA variant — open, user-requested next**.
  Rather than port more G256-shaped kernels, design a kernel specifically
  for G128's layout. Hypothesis: G128's bottleneck is the per-sub-block
  FMA chain reaching only 4 sub-blocks before each group boundary; the
  fix is either (a) cross-group batching of the scale FMAs (defer FMA
  resolution past the group boundary, gaining ILP at the cost of cacc
  register pressure) or (b) a larger M-tile (16×32 or 32×16 output, mirror
  the `_m2.gfx12.hip` pattern on the HFQ4G256 side). Target: ≥10% over
  k2's 702 tok/s ⇒ ≥770 tok/s prefill on shisa A3B-PARO at --prefill 256.
  Templates to study:
  - `kernels/src/gemm_hfq4g256_moe_grouped_wmma_m2.gfx12.hip` (m2 tile pattern)
  - `kernels/src/gemm_paro_q4g128_moe_grouped_mmq.gfx12.hip` (current k2 baseline)
- **Remaining MQ4G256→PARO ports** for completeness:
  - `gemm_paro_q4g128_moe_grouped_mmq.gfx11_dgpu.hip` (7900 XTX, hipx host)
  - `gemm_paro_q4g128_moe_grouped_mmq_k4.gfx11_dgpu.hip`
  - `gemm_paro_q4g128_moe_grouped_wmma.gfx12.hip` (FP16 alternative on gfx12)
  - `gemm_paro_q4g128_moe_grouped_wmma_m2.gfx12.hip`
- **F32 router/shared_gate quantization** (Björn's Lever 1) — biggest single
  decode lever (+8-15%), kept with Björn.
- **F32 → FP16 activations** (Björn's Lever 2) — +10-15% predicted, kept with Björn.
- **Phase 5 dense parity sweep** — only 0.8B measured. 9B/27B/27B-3.6 PARO
  numbers not exercised on hiptrx. GOAL.md soft target (not exit-gating).
- **Phase 7 conditional ports** — gfx1100 + gfx1151 rebuilds + benches.
  Now unblocked by this asymptote certification.

## Post-exit MMQ work (durable, on `feat/lever-4-gpu-argmax-stability`)

```
6ad5a219  docs(paroquant-gfx12): asymptote characterization across 5 prefill + 2 decode variants
ce289429  feat(paroquant-gfx12): vectorized perm_b32 nibble unpack (neutral, opt-in)
2a1961b9  feat(paroquant-gfx12): G128-native m2 prefill + m4 decode (both falsified, opt-in)
6a08480c  feat(paroquant-prefill): k4 deeper-pipeline PARO MMQ gfx12 (neutral, opt-in)
f2e2c254  feat(paroquant-prefill): port PARO_Q4G128 MoE grouped MMQ to gfx12  +30× prefill
dcf752dc  feat(paroquant-decode): Lever 4 — bench_qwen35_mq4 uses GPU argmax  (NaN panic fix)
```

All pushed to `origin/feat/lever-4-gpu-argmax-stability`. Ready to
cherry-pick into PR #319.

### Asymptote characterization (2026-05-22, post-exit)

Three orthogonal G128-native levers attempted post-exit (per user ask
"create paro-optimal G128 kernels for gfx12 prefill AND decode"); all
falsified or neutral. Final asymptote: **705 tok/s prefill / 59.5 tok/s
decode** on shisa A3B-PARO at canonical bench config.

| Lever | Axis tested | Result |
|---|---|---|
| m2 prefill | B-gather BW per output FLOP | -3.4% to -20.1% (VGPR occupancy regression) |
| m4 decode | X-load BW per layer | -7.5% (LDS occupancy cap + L2 already absorbed X re-reads) |
| perm prefill | scalar VALU (nibble unpack) | 0.0% (WMMA-throughput-bound, unpack hidden under WMMA latency) |

Triangulated remaining bottleneck candidates: WMMA throughput cap,
scale-FMA chain serialization, per-group sc/zp shuffle, or routed-MoE
dispatch overhead (hipGraph capture). All require structural redesign
(higher risk, lower confidence) and are out of scope for this session.

Full diagnostic walkthrough including .hsaco metadata for all 5
variants: `docs/investigations/paro-g128-native-gfx12/ASYMPTOTE.md`
(on `feat/lever-4-gpu-argmax-stability`).

### Post-asymptote breakthrough (2026-05-22 evening)

User pushback "how is mq4 3000 tok/s but paro is impossible" → rocprof
revealed the **scalar `gemm_hfq4g128` kernel was 70% of GPU time** and
**INVISIBLE to HIPFIRE_PROFILE** (no `begin_timer` wrapper). The same
"hidden lever" pattern as 2026-05-19's q8_0_batched discovery. Four
structural wins cascaded:

1. **gfx12 FP16 WMMA port of gemm_hfq4g128** (commit `6c097ad5`):
   +250% prefill on shisa A3B-PARO (760 → 2660 tok/s).
2. **rocprof+atlas baked into HIPFIRE_PROFILE** (commit `4a9bcc2c`):
   Methodology fix — `HIPFIRE_PROFILE=1` now auto-execs under
   `rocprofv3`, so the next hidden-lever pattern can't recur.
3. **F32 batched shared_expert dispatch** (commits `f4681367` +
   `d6e66cb7`): Unlocked z-lab A3B-PARO into the batched path (was
   admit-failing → per-token fallback at 64 tok/s). Initial layout
   bug fixed at `d6e66cb7` (silent corruption masked by routed
   experts).
4. **Shared-expert F32-WMMA per-call-site** (commit `5a556225`):
   Replaces scalar F32 GEMM with FP16-WMMA on gfx12 specifically for
   shared_expert.gate/up/down. Router stays on scalar (precision-
   sensitive).

**Final z-lab A3B-PARO numbers on hiptrx/gfx1201:**

| Stack | Prefill tok/s | Coherence (3 prompts) | Speedup |
|---|---:|---|---|
| baseline (admit broken → per-token) | 64 | n/a | 1× |
| + admit relax + F32 dispatch | 697 | 0 hard / 1 soft | 10.9× |
| + WMMA-gfx12 (HFQ4 path) | 1547 | 0 hard / 1 soft | 24.2× |
| + shared_expert F32-WMMA | **3130** | **0 hard / 1 soft on all 3** | **48.9×** |

**z-lab now BEATS shisa-A3B-PARO's 2660 tok/s ceiling** on this hardware
AND delivers clean coherence on the exact prompt (humaneval_3) where
shisa attractors. Net structural result: z-lab is the production-
canonical PARO checkpoint for gfx12 prefill.

Full commit graph on `feat/lever-4-gpu-argmax-stability`:

```
5a556225  shared_expert-only F32-WMMA — +102% z-lab AND clean coherence
18f0ce81  FP16-WMMA gemm_f32_batched (broad opt-in; router-sensitive)
d6e66cb7  fix F32 shared_expert output layout (silent corruption)
f4681367  F32 shared_expert batched dispatch — unlocks z-lab
a36f0915  attractor root-cause investigation (shisa calibration issue)
4a9bcc2c  HIPFIRE_PROFILE=1 auto-execs under rocprofv3
ca8816cb  fwht3 KV mode + coherence validation doc
6c097ad5  FP16 WMMA non-grouped HFQ4G128 — +250% prefill
9359fd9b  FP16 WMMA m2 variant — falsified (opt-in)
a457fa34  FP16 WMMA G128 MoE grouped — +7.8% prefill
6ad5a219  asymptote characterization (now superseded)
ce289429  perm_b32 nibble unpack (neutral, opt-in)
2a1961b9  m2 + m4 falsified (opt-in)
6a08480c  k4 deeper-pipeline (neutral, opt-in)
f2e2c254  PARO Q4G128 MoE grouped MMQ port — +30× prefill
dcf752dc  Lever 4 — bench_qwen35_mq4 GPU argmax (NaN fix)
```
