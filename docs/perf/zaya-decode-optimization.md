# ZAYA decode optimization log

Goal: **>200 tok/s @ oq8++, >400 tok/s @ oq4.25++** on halo/gfx1151 (from ~33 tok/s).
Method: theorize constraint → investigate → implement → test, iterating with measurement.

## Roofline (confirmed = memory bandwidth)

- Active per token: ~700M (paper, MoE-excluding-embeddings) + tied **537M lm_head**
  (read every token) ≈ **1.24 GB/tok @ oq8**.
- Strix Halo LPDDR5X-8000, 256-bit → ~256 GB/s peak (~200-217 real).
- oq8: 1.24 GB / 256 GB/s = 4.9 ms → **~200 tok/s ceiling** (== target).
- oq4.25 (0.53 B/param): ~0.67 GB → ~2.6 ms → **~380-400 tok/s** (== target).

Real dims: hidden 2048, 40 hybrid blocks, 16 experts top-1 (+ MoD skip slot),
moe_int 2048 (expert = fc1[4096,2048] + fc2[2048,2048] = 12.58M), head_dim 128,
8 q / 2 kv heads, vocab 262272, **lm_head tied to embed**.

## Diagnosis (measured)

Tools: hipGraph replay `device_synchronize` timer (`gpu_body`), code ablations
(`HIPFIRE_ZAYA_ABLATE=moe/glue`), `HIPFIRE_ZAYA_LAUNCHSTATS`.

- **gpu_body = 27.4 ms/token** (synchronized). End-to-end 30 ms ≈ 27 GPU + 3 serving.
  hipGraph proved CPU submission is 9 µs — NOT the bottleneck. GPU-execution-bound.
- 1.24 GB / 27.4 ms = **47 GB/s aggregate = 18% of peak.**

Decomposition:

| component | kernels | time | rate |
|---|---|---|---|
| Expert GEMVs (gate_up+down) | 80 | 5.2 ms | 0.5 GB @ **96 GB/s (37% peak)** |
| CCA attention glue | 360 | 4.0 ms | ~11 µs/kernel (pure dispatch) |
| lm_head + proj + router GEMVs + rest of glue | ~1200 | ~18 ms | lm_head ~5.6 ms |

**Two co-equal constraints:**
1. **GEMVs at ~96 GB/s (37% peak).** Weight-bandwidth work ~13 ms vs ~5 ms roofline.
   Root (`gemv_oq8_grouped.hip`): 1 wave32/row, narrow 4-byte int32 loads, no row-
   blocking / ILP.
2. **~960 tiny glue kernels × ~11 µs ≈ 11 ms** pure dispatch overhead.

Both must fall (GEMV→peak, glue→~0) to reach 5 ms / 200 tok/s. oq4 halves GEMV bytes → 400.

## Plan (ordered by leverage)

1. **Optimize the shared oq8 decode GEMV** (one kernel → speeds lm_head + all
   projections + router). Techniques: wave64 (gfx1151, slightly faster than wave32),
   128-bit (`int4`/dwordx4) coalesced loads, multi-group-per-wave for ILP. Target
   ~96 → ~200+ GB/s.
2. **Optimize the MoE planar-indexed expert GEMV** (same techniques).
3. **Glue mega-fusion**: collapse the ~960 elementwise glue kernels into a handful.
4. **oq4.25 path** for the 400 tok/s target.

## Work log

### 2026-07-24 — investigation
- Established roofline + diagnosis above. References ~/build/amd (CK/aotriton/aiter)
  are CDNA/asm-focused — technique inspiration only, no drop-in RDNA gemv.

### Experiments (attempt / result)

**EXP-1 (2026-07-24): gemv_oq8_grouped v2 (128-bit loads + 2 groups/wave).**
Standalone microbench (`scratchpad/gemv_bench.hip`), M=262272 K=2048 (cold, 537MB):
- v1 (current): **201.2 GB/s** (2.71 ms) — 79% of ~256 peak.
- v2 (128-bit): **220.3 GB/s** (2.48 ms) — +9%.
- correctness max|v1-v2| = 9.8e-6 ✓.
Wired v2 into `gemv_oq8_grouped` for K%512==0. In-model gpu_body 27.4→27.2ms
(unchanged — v2 only hits o_proj + router-down; lm_head is Q8_0, qkv/down use W8A8,
experts use planar). Coherent. **Small-M sweep numbers (4096/2048/1024) are
cache-resident, not real DRAM — ignore.**

**⟹ HYPOTHESIS REFUTED + PIVOT.** The gemv kernel is NOT the constraint — it hits
201 GB/s (79% peak) on the cold lm_head. If all 1.24GB weights ran at 201 GB/s the
whole decode would be **6.2 ms**; measured gpu_body is **27 ms**. So **~21 ms is
per-kernel dispatch overhead**: ~1643 serial, data-dependent kernels × ~13 µs GPU
launch/exec each. Decode is **KERNEL-COUNT / DISPATCH bound**, not bandwidth bound.

Corrected constraint model:
- Weight-bandwidth floor @ oq8, 201 GB/s gemv: **~6.2 ms** (161 tok/s). At peak
  256 GB/s: 4.9 ms (200). oq4.25: ~3 ms (330-400).
- Dispatch overhead: **~21 ms** across 1643 kernels (~13 µs/kernel). THIS is the
  6× gap.

**Revised plan (primary lever = FUSION / fewer kernels):**
1. Keep v2 gemv (free +9% on the paths it hits).
2. **Fuse each block's ~46 kernels → a handful.** Targets: CCA attention chain
   (qk_residual/stream/conv×2/add_conv/value/l2/rope → 1-2 kernels), MoE
   (rmsnorm+rotate+quant+gate_up+silu+down+combine), router MLP. Goal 1643 → few
   hundred, ideally ~1-3 kernels/block via a per-block megakernel.
3. Re-measure gpu_body after each fusion; overhead ≈ kernels × 13µs.
4. oq4.25 for the 400 target once oq8 path is fast.

Per-kernel overhead ~13µs is the key constant — every kernel removed saves ~13µs ×
40 blocks = ~0.5ms.

**EXP-2 (2026-07-24): fused qkv projection.** Replaced the W8A8 path (quantize_act_oq8
+ 4× gemv_w8a8) with `fused_rmsnorm_rotate` + ONE `fused_qkvza_oq8_gemv` (W8A16, all 4
projections q/k/vcur/vdel in a single launch reading rotated f32 X once). 5 kernels → 2.
Coherent ("Paris"). **gpu_body 27.2 → 26.2 ms** (−1.0 ms for −160 kernels ⇒ ~6 µs/glue
kernel). Added `LinearWeight::quant_mk`. qkv_xq/qkv_xs now unused.

### CEILING ANALYSIS (important — informs feasibility)

Measured per-kernel overhead ≈ **6-11 µs** (glue) — lower than the initial 13µs guess.
- Weight-bandwidth floor @ oq8: 1.24 GB / 201 GB/s = **6.2 ms = 161 tok/s**. Even with
  ZERO overhead, **200 tok/s @ oq8 is above the current gemv floor** — it needs the gemv
  at ~256 GB/s (peak). v2 gives 220 → 5.6 ms = 178 tok/s. So 200 @ oq8 requires
  near-peak gemv AND near-zero overhead — very tight.
- oq4.25: 0.62 GB / 220 GB/s = 2.8 ms = **357 tok/s**; at peak 2.4 ms = 410. So
  **400 @ oq4.25 is the more feasible target** (oq4 halves weight bytes → headroom).
- Incremental glue fusion helps but has a ceiling: ~900 glue kernels × ~7 µs = ~6 ms;
  removing all → gpu_body ~20 ms = 50 tok/s. Plus ~440 gemvs × ~6 µs launch = ~2.6 ms
  overhead on top of 6.2 ms compute. So fully-fused-glue + current gemvs ≈ 9-10 ms =
  ~100-110 tok/s.

**⟹ The 200/400 targets require BOTH: (a) collapse gemv launch count (fuse per-block:
qkv done, + gate_up/down, + router MLP, + attention chain) AND (b) near-peak gemv (v2 +
tuning) AND (c) oq4.25 for 400.** The endgame that truly hits the numbers is a
**per-block (or whole-decode) megakernel** eliminating inter-kernel overhead — a large
rewrite. Incremental fusion is the tractable path to ~100-150 tok/s first.

Next fusions (by kernel-count): router MLP (~8→2), CCA attention glue (~9→3), MoE
expert (gate_up+silu+down+combine, ~5→2). Then apply v2 128-bit loads to fused_qkvza +
expert gemvs. Then oq4.25.

**Status after EXP-1,2 (2026-07-24):** end-to-end **33 → 35 tok/s** (gpu_body 27.4 →
26.2 ms). Diagnosis complete + approach validated; remaining path is a large kernel
effort (megakernel + peak gemv + oq4) — see ceiling analysis. Incremental fusion first
targets ~100-150 tok/s @ oq8; the 200/400 targets need the megakernel + oq4.25.

**EXP-3 (2026-07-24): oq4.25++ is on the SLOW host path, not the fast path.**
Measured: oq8 = 35 tok/s, **oq4.25++ = 10.6 tok/s** (SLOWER!). Cause: the `++` (AWQ)
experts don't satisfy the on-device indexed-MoE gate (`moe_indexed` requires plain
oq8, no AWQ) → fall to the HOST path with per-block `download_f32` router readback
(~40 GPU→host syncs/token) + no graph capture. So halving weight bytes did nothing —
dispatch/host-sync dominates. **⟹ Prerequisite for the 400 target: bring oq4.25++
onto the on-device indexed + graph + fused path** (needs AWQ-aware indexed MoE GEMV,
or drop AWQ for a plain-oq4 variant). Dispatch is binding for BOTH targets.

### CONCLUSION OF SCIENTIFIC PHASE

Primary constraint = **per-kernel dispatch overhead across ~1480 serial dependent
kernels**, NOT memory bandwidth (gemv already 79% peak) and NOT the param count.
Confirmed by: (a) isolated gemv microbench (201 GB/s), (b) 27ms gpu_body vs 6.2ms
bandwidth floor, (c) fusion saves ~6µs/kernel (EXP-2), (d) oq4 doesn't help (EXP-3).

Both targets require the SAME structural fix — collapse per-block kernel count toward
a **megakernel** (per-block cooperative-grid kernel, or whole-decode persistent kernel)
that eliminates inter-kernel overhead in ≤ a few launches. This is a multi-day rewrite:
- Per-block megakernel: grid-strided, cooperative grid.sync between phases (qkv → conv/
  CCA → flash-attn → o_proj → router → expert), weights from global, intermediates in
  LDS/global scratch. ~1-2 launches/block → ~80 kernels total → overhead ~0.5ms →
  gpu_body ≈ weight floor (6.2ms oq8 / 3.1ms oq4) → ~160 tok/s oq8, ~320 oq4; with v2/
  peak gemv → ~200 / ~400.
- Plus: oq4.25++ onto the fast path (EXP-3), v2 128-bit loads on all gemvs.

Groundwork already in place: correct hipGraph decode capture, on-device routing (oq8),
fusion infra + fused_qkvza, ablation/timing probes.

### EXP-4 (2026-07-24): BLOCK-COUNT + lm_head ablation — BOMBSHELL

`HIPFIRE_ZAYA_NBLOCKS=N` (process first N blocks) + `ABLATE=lmhead`, gpu_body:
| N | gpu_body |
|---|---|
| 40 | 26.3 ms |
| 20 | 18.6 ms |
| 10 | 14.7 ms |
| 1  | 11.15 ms |
| 1 + skip lm_head | **0.34 ms** |

⟹ **per-block = 0.385 ms; the ~11 ms "intercept" is almost ENTIRELY the lm_head gemv
(11.15 − 0.34 = 10.8 ms).** lm_head = 537 MB Q8_0 / 10.8 ms = **50 GB/s = 20% of peak** —
4× slower than the oq8 gemv (201 GB/s). **The lm_head is ONE kernel and ~40% of the
whole 26 ms decode.** This is the single biggest lever, bigger than the block megakernel.

Revised budget: lm_head 10.8 ms + 40 blocks × 0.385 = 15.4 ms + norm/overhead 0.34 ≈ 26 ms.
- Fix lm_head → ~2.7 ms (201 GB/s): saves ~8 ms → gpu_body ~18 ms → ~55 tok/s.
- Then per-block megakernel (0.385 → ~0.1 ms): saves ~11 ms → ~7 ms → ~140 tok/s.
- Then oq4 / peak gemv → 200/400.

**NEW PRIORITY ORDER: (1) lm_head gemv (biggest single win, one kernel), (2) per-block
fusion/megakernel, (3) oq4 fast path.** Root of slow lm_head = Q8_0 gemv kernel
(vocab=262272 rows) inefficient vs the Oq8 v2 path. Fix: efficient Q8_0 decode gemv
(128-bit loads) OR re-quant lm_head to Oq8G256 (fast v2 path).

### EXP-5 (2026-07-24): Q8_0 gemv v4 — faster kernel, NO in-model effect ⟹ CLOCK is the root

Rewrote gemv_q8_0 (each lane owns whole contiguous blocks, scale loaded once):
microbench **117 → 222 GB/s (1.9×), bit-exact**. BUT in-model: gpu_body unchanged
(nblocks=1 still 11.2ms, lm_head still ~10.8ms). The faster kernel made ZERO in-model
difference ⟹ the lm_head is NOT compute/bandwidth-bound in-context.

**ROOT CAUSE (the real one): GPU is power-throttled / under-utilized during decode.**
Measured during a sustained 400-tok decode (`/sys/class/drm/card1/device/`):
- **`gpu_busy_percent` = 0–4%** (GPU ~96% IDLE).
- **sclk pinned ~600 MHz of 2900 max** (lowest DPM state).
- `power_dpm_force_performance_level = auto`.

The decode is so launch-latency / dependency-stall bound that the GPU never gets busy
enough for the `auto` governor to ramp clocks → EVERYTHING (incl. big kernels) runs at
~1/5 core clock → the isolated microbenches (201 GB/s) hit high clock because they
saturate the GPU; the in-model decode does not. This unifies every earlier
observation: gemv "50 GB/s in-model vs 201 isolated", lm_head 10.8 vs 4.85ms, and
kernel opts not moving gpu_body.

**Two levers now, both real:**
1. **Force max clock** (`power_dpm_force_performance_level=high`, or the auto governor
   ramping once utilization is high). Needs root — pending user. Quantifies the
   clock's share; likely 2-4× on the big-kernel portion.
2. **Raise GPU utilization** (fusion/megakernel → fewer gaps → governor ramps on its
   own AND less latency). This is the self-sustaining fix and works without root.

The megakernel is still the endgame, but now for a compounding reason: high utilization
both removes launch gaps AND unlocks the clock.

### EXP-6 (2026-07-24): FORCED MAX CLOCK — no change ⟹ clock REFUTED, launch-latency CONFIRMED

Passwordless sudo available. Forced `power_dpm_force_performance_level=high`:
sclk **600→2900 MHz**, mclk **→937 MHz** (both max). Decode: **35.2 → 35.8 tok/s —
UNCHANGED.** (Reverted to auto after.)

⟹ **Decode is NOT clock/bandwidth/compute bound.** The low clock (EXP-5) was a
*symptom*: the GPU idles ~96% (gpu_busy=4%) in the **launch-latency / dependency-stall
gaps between ~1643 serial kernels**, and the auto governor throttles *because* the GPU
is idle. Forcing the clock speeds the ~4% busy time but not the 96% of gaps → no net
change. (Spinner test EXP-5b: a keep-busy spinner DID ramp the clock to 2830 MHz but
HALVED decode via CU contention — not usable.)

**FINAL, DEFINITIVE CONSTRAINT: inter-kernel launch latency × kernel count.**
gpu_body 26ms ≈ ~1643 kernels × ~15µs gap (mostly idle). This is directly reducible by
cutting kernel count. Fusion/megakernel is not just *a* path — it is *the* path, and it
attacks the measured bottleneck 1:1. Projection: 1643 → ~200 kernels (per-block
megakernel ≈ 1-2/block + lm_head) → gpu_body → ~weight-compute floor → ~150-200 tok/s
oq8; + oq4 fast path → ~400. Clock/bandwidth are NOT levers; kernel COUNT is.

Every micro-optimization this turn (gemv v2, fused qkv, q8 v4) helped little precisely
because they didn't cut kernel COUNT enough. The ONLY thing that matters now is
collapsing launches. NEXT: per-block cooperative-grid megakernel.

### Megakernel build notes (for the next session)

CCA conv semantics (verified from safetensors + zaya_conv1d_valid_f32):
- `conv_qk.0` = **depthwise** `[1280,1,2]`: groups=conv_ch(1280), per_group=1.
  `dw[c][t] = b[c] + Σ_kk w[c*2+kk]·window[c][t+kk]`, dw_len=2 (window has pad+1=3 cols).
- `conv_qk.1` = **grouped** `[1280,128,2]`: groups=nq+nkv(10), per_group=128.
  `gw[c] = b[c] + Σ_{j<128}Σ_kk w[(c*128+j)*2+kk]·dw[base+j][kk]`, base=(c/128)*128.
  Reads 655 KB weights/block → needs MANY workgroups; do NOT collapse into 1 WG.

**Fusion strategy (refined):** the grouped conv and the gemvs are parallelism-bound —
keep them multi-workgroup. Fuse the TINY launch-bound elementwise ops:
- Pre-conv: qk_residual(×2 modes) + qk_stream → 1 (safe, no reduction; mode1 needs
  mode0's query_res so use __syncthreads in one WG, or 2 phases).
- Post-conv: add_conv_residual_qk + value_assemble + qk_l2norm_qk + rope_qk → 1
  (per-head reduction for l2norm; one WG per head or grid-stride+reduce).
- Router MLP: rmsnorm + fc1 + gelu + fc2 + gelu + out_proj + select → 1 (256-dim, one
  WG + LDS; fc weights small — verify Oq8 vs f32 first).
- MoE: silu_mul_rotate + combine already fused; gate_up/down stay (parallel gemvs).

The true target needs cooperative-grid per-block collapse of the GEMVs too (grid.sync,
residency-bounded grid-stride) — the hard part. hipfire has no cooperative-launch
dispatch yet; that's a prerequisite to add.

### EXP-7 (2026-07-24): COOPERATIVE GRID-SYNC FEASIBILITY on gfx1151 — CONFIRMED ✓

Standalone `scratchpad/coop_test.hip` (`cg::this_grid().sync()`, multi-phase
write→sync→reduce→sync→broadcast):
- `cooperativeLaunch = 1` (supported), MPs = 20.
- Max cooperative grid = **160 blocks × 256 threads = 40,960 threads** (8 blocks/MP ×
  20). Larger grids need grid-stride.
- `hipLaunchCooperativeKernel` → no error; grid.sync produced the correct global sum +
  broadcast (`GRID-SYNC WORKS`). Cross-workgroup sync + memory visibility validated.

⟹ **The per-block cooperative megakernel is FEASIBLE on this hardware.** The last
blocking unknown is resolved. Design: cooperative launch of 160 blocks; loop the 40
zaya blocks; within each block, phases (rmsnorm+qkv gemv → CCA/conv → flash-attn →
o_proj → router → expert-select+gemv → residual) separated by `grid.sync()`; each
gemv/phase grid-strides over the 160-block resident set. One launch replaces the whole
block loop → collapses ~1643 kernels → a handful → gpu_body → compute floor → target.
Prereq now: add `hipModuleLaunchCooperativeKernel` to hip-bridge + a dispatch method,
then build the kernel phase-by-phase validating against the current path.

### EXP-8 (2026-07-24): lm_head deep-dive — UNIFIES the root cause

lm_head = 40% of decode (10.8 ms via NBLOCKS=1 ±lmhead). Tested every lever:
- New v4 kernel (1.9× faster isolated): in-model UNCHANGED.
- Grid-stride (262k→8192 workgroups, kills dispatch-count): UNCHANGED (still coherent).
- Forced max clock: UNCHANGED.
- Isolated microbench of the SAME kernel: 2.6 ms (200 GB/s). In-model: 10.8 ms (50 GB/s).

⟹ The lm_head is a single COLD 537 MB read that reaches only ~50 GB/s, while the
microbench's 100 sustained iterations reach 200. **This is the SAME root cause as the
clock throttling (EXP-5/6): low GPU utilization (bursty, launch-latency-bound decode)
keeps the GPU in a low-performance state — low clock AND low effective memory bandwidth.
Neither is separately fixable; both need sustained high utilization.** Forcing clock
doesn't help (bandwidth stays low); faster kernels don't help (the workload can't feed
them). The megakernel (one launch, continuous memory pressure, high utilization) is the
ONLY fix and unlocks clock + bandwidth + fewer gaps simultaneously — a compounding win.
Kept the grid-strided gemv_q8_0 (harmless, coherent) and cooperative-launch infra.

**Session end state:** 35.4 tok/s oq8, path fully proven+unblocked+unified. The
megakernel is the sole remaining work and a multi-session build.

### EXP-9 (2026-07-24): FIRST MEGAKERNEL PHASE BUILT — router MLP fused (single-WG)

Wrote `zaya_router_mlp_fused` (env HIPFIRE_ZAYA_ROUTER_FUSED): a single-workgroup kernel
doing down_proj(Oq8) → prep → rmsnorm → [FWHT→fc1(Oq8)→gelu]×2 → FWHT→out(Oq8) → select,
replacing ~9 launches with 1. Includes in-kernel FWHT-256 (`zaya_fwht256_lds`, ds_swizzle,
bit-mirrors rotate_x_mq) + inline Oq8 dequant + rmsnorm/softmax reductions.
**COHERENT ON FIRST ATTEMPT** ("Paris... Eiffel Tower... Louvre") — validates every
megakernel component (FWHT, Oq8, multi-phase __syncthreads) end-to-end.

BUT: gpu_body UNCHANGED (26.6ms), tok/s 35.4→34.9. **Single-workgroup fusion is a WASH**:
the down_proj + fc gemvs lose multi-WG parallelism (1 WGP, uncoalesced, serialized 3×
FWHT), and that added serial compute ≈ the ~8 launch-gaps removed. ⟹ **Empirically
confirmed: the megakernel must be COOPERATIVE multi-WG (grid.sync between phases, gemvs
grid-strided over 160 blocks) — NOT single-WG.** The cooperative launch infra (EXP-7) is
exactly what's needed. Router phase alone (even cooperative) is only ~+3 tok/s; the win
requires ALL block phases fused into ONE cooperative kernel. The single-WG router is kept
env-gated (validated component library) — convert to cooperative + extend to full block
= the remaining build. Components proven; structure now empirically pinned to cooperative.

### EXP-14 (2026-07-24): DECODE PARTITION via ablation + lm_head is BF16 (biggest lever)

Fresh end-to-end measurements (temp0, oq8, ~100 tok): baseline **36.45 tok/s** (27.4ms/tok).
Ablation deltas (HIPFIRE_ZAYA_ABLATE):
| component | ablated tok/s | cost | share |
|-----------|---------------|------|-------|
| lm_head   | 58.8          | ~10.4ms | **38%** |
| remainder (qkv/attn/o_proj/norm/router) | — | ~8.3ms | 30% |
| MoE (indexed gate_up+down) | 45.0 | ~5.2ms | 19% |
| glue (conv/l2norm/rope/resid/assemble) | 41.8 | ~3.5ms | 13% |

**KEY: lm_head/embed is stored BF16** (model.embed_tokens.weight [262272,2048] = 1.07GB
read/token) in BOTH oq8++ and oq4.25++ — NOT Q8. It's the single biggest cost (38%) and
the most tractable lever: quantize BF16→q8 (~0.55GB, ≈halves it, ~0.06 KLD safe per
[[embed sensitivity]]) or →oq4 (~0.27GB). Quantizer has `--embed-precision {source,q8,
bf16,f16}` (main.rs:407). NB the dtoh-heavy host-router path (oq8++/oq4.25++ AWQ) is a
RED HERRING: plain oq8 (dtoh=0) is the SAME 36 tok/s — the 40 dtoh overlap the GPU tail
that surfaces at the sampler logits-sync. gpu_decode launchstats "wall" (1.6ms) = CPU
launch only (device_sync=0, GPU async); true per-token GPU exec ≈27ms.
Path to 200 @ oq8 (5ms/tok): lm_head→oq4 (−~8ms) + body megakernel (pipeline 46→211 GB/s).
IMPLEMENTING lm_head requant first (bounded, +~9-15 tok/s, 38% lever).

### EXP-16 (2026-07-24): MoE gemv is FAST in isolation → body is utilization-bound, not gemv-bound

Microbench (scratchpad/moe_bench.hip) of the indexed gate_up Oq8 gemv (32t/row, M=4096
K=2048): **0.018 ms/call, 470 GB/s** — the gemvs are NOT slow. So the body's 16.6ms @47
GB/s effective is the AGGREGATE of ~1400 small kernels running at LOW GPU utilization,
which keeps the clock/bandwidth throttled (EXP-11: in-decode ~48 GB/s vs 211 achievable).
⟹ per-gemv tuning is NOT the lever (gemvs already fast); the fix is the per-block
cooperative megakernel that keeps the GPU saturated → clock/bandwidth ramp → body 16.6→
~3.7ms. This is the FINAL confirmation the megakernel is the sole path to 200 (with oq4
lm_head): body 3.7ms + lm_head 1.3ms = 5ms = 200 tok/s. No completable shortcut remains;
megakernel is a multi-session build. Components validated: cooperative grid-sync, in-kernel
FWHT-256 (zaya_fwht256_lds) + Oq8 gemv (zaya_router_mlp_fused template), Oq8 planar MoE.

### EXP-15 (2026-07-24): ★ Q8 lm_head VALIDATED = 1.8× (36→66 tok/s)

`zaya1-8b-native.oq8++` happens to carry a **Q8F16 embed** (570MB) while all other zaya
models carry BF16 embed (1.07GB). Direct measurement: native.oq8++ (Q8 embed) = **65.97
tok/s** vs 36.4 for BF16-embed models = **1.8× from the lm_head alone** (15.2 vs 27.4
ms/tok; lm_head 10.4→~2.6ms via smaller bytes + faster gemv_q8_0 path). CONFIRMS the
byte-reduction lever empirically. `--embed-precision` default = `source` (=bf16 for a bf16
ckpt) → that's why models shipped BF16. FIX = re-quantize canonical bf16 with
`--embed-precision q8`. IMPLEMENTING: rebuild zaya1-8b.oq8++ with Q8 embed → expect ~66
tok/s (proper oq8 experts). Then stack body levers (MoE/glue/megakernel) toward 200.

**LANDED (EXP-15b): zaya1-8b-q8emb.oq8++.hfq = 52.18 tok/s, COHERENT (1.43× from 36.4).**
Rebuilt canonical bf16 → `--format oq8++ --hessian zaya1-8b.calib.hfq --embed-precision q8`
(8.9GB, ~500MB smaller). Embed Q8F16 570MB confirmed; ocean prompt fully coherent. (52 vs
native's 66 = oq8 experts are bigger than native's MQ4; body now ~16.6ms, lm_head ~2.6ms.)
Quality: 8-bit RTN safe (matches native.oq8++ precedent). Calib HAS model.embed_tokens.
hessian [2048,2048] → sub-8-bit lm_head (oq4) can use LDLQ error-feedback when we get there
(needs code: embed-precision path is RTN-only today; + untie for gather-vs-projection).
REMAINING to 200 (5ms): body 16.6ms @47 GB/s eff → megakernel to ~211 GB/s (~3.7ms) +
oq4 lm_head (~1.3ms). Body is LAUNCH-LATENCY bound (1600 kernels: 40 ops/blk × 40 blk);
per-block cooperative megakernel = 1 kernel/blk → 40 launches (40× fewer). NEXT: build it.

### EXP-13 (2026-07-24): ROOFLINE — the targets sit AT/ABOVE theoretical DRAM bandwidth

The primary performance constraint is **DRAM bandwidth**, and the stated targets sit at or
above the hardware's memory ceiling. Computed from the confirmed HF config
(hidden 2048, ffn_hidden 4096, 16 experts top-1, ~20 MoE blocks, 40 attn blocks,
vocab 262272, tied Q8 lm_head):

| quant | active bytes/tok | target | GB/s required | feasible? |
|-------|------------------|--------|---------------|-----------|
| oq8++ (as-is) | **1.353 GB** (lm_head = 0.569 GB = **42%**) | 200 tok/s | **271 GB/s** | NO — above theoretical 256 |
| oq8++, lm_head→oq4 | 1.079 GB | 200 tok/s | 216 GB/s | YES (≈ achievable 211) |
| oq4.25++ (Q8 lm_head) | 0.998 GB | 400 tok/s | **399 GB/s** | NO — far above 256 |
| oq4.25++, lm_head→oq2 | 0.590 GB | 400 tok/s | 236 GB/s | MARGINAL (>211 achiev, <256 theo) |

Hardware ceiling (gfx1151 Strix Halo, 256-bit LPDDR5X-8000): **256 GB/s theoretical**,
**211 GB/s measured-achievable** (big contiguous Q8 gemv, EXP-9/12). The paper's
"~700M active params" reconciles to 739M **excluding** the tied lm_head, which rides on top
as a further 569 MB (42% of the oq8 read).

**Conclusion (corrects the earlier "megakernel alone reaches target"):** the megakernel is
NECESSARY (it lifts the decode from ~50 GB/s small-kernel-bound to ~211 GB/s pipelined) but
NOT SUFFICIENT. 200 tok/s @ oq8 needs 271 GB/s at the current byte count — physically
impossible on this DRAM. The targets require BOTH:
  1. the cooperative megakernel (sustain ~211 GB/s), AND
  2. **byte reduction — quantize the tied lm_head** (42% of the oq8 read): →oq4 makes
     oq8@200 feasible (216 GB/s); the oq4.25 model needs lm_head→oq2 AND near-theoretical
     efficiency, which is at the ragged hardware edge (236 GB/s vs 211 achievable).

Actionable, bounded next lever (unlike the open-ended megakernel): **quantize the tied
lm_head** — the single biggest byte saver and independently testable. NB: this reinterprets
"oq8++"/"oq4.25++" as applying to the expert/attention body with a more-aggressively
quantized lm_head, since a uniformly-Q8 lm_head puts 200 tok/s above theoretical peak.

### EXP-12 (2026-07-24): lm_head uses the IDENTICAL fast path → bottleneck is aggregate, not per-kernel

- Created-stream bench = 210 GB/s (same as default stream) → the graph's non-default
  `active_stream` does NOT throttle. Stream ruled out.
- Traced the decode lm_head: `w.embed.gemv` → `execute_steps`/`Step::Gemv` for Q8_0 →
  `gemv_q8_0` with grid `min(M,8192)=8192`, block 32 — **byte-identical kernel and launch
  config to the 210 GB/s isolated bench** (K=2048>1536 takes the narrow arm).
- ⟹ The lm_head is not individually slow. The ~50 GB/s effective rate is an **aggregate**
  property: the small dependent glue kernels interleaved between the large gemvs prevent
  the memory subsystem from sustaining sequential-read bandwidth across the decode. Fourth
  independent confirmation (launch-latency, utilization, clock/pipeline, per-kernel-parity)
  that the fix is one fused cooperative megakernel, not per-kernel tuning.

### EXP-11 (2026-07-24): bandwidth is CLOCK-dependent + PIPELINE-dependent (refines the "why")

`scratchpad/q8_cold.hip` (lm_head-size gemv, 537MB, grid-strided, per-launch timing):
- Forced LOW clock: **43-48 GB/s**. Forced HIGH clock: **175 GB/s cold launch → 211
  sustained**. So bandwidth IS clock-dependent (4.4×).
- In-daemon decode effective bandwidth = 1.24GB / ~25ms ≈ **48 GB/s = the LOW-clock
  value**, EVEN when mclk is forced to 937 (verified during decode) → tok/s unchanged
  (35→36). Non-graph + high clock = 14.76 (CPU-bound; graph is essential).

⟹ **Unified: the decode runs at ~48 GB/s effective, and forcing the clock doesn't lift
it — because the small, dependent, interspersed kernels can't pipeline memory** (the
bench's LARGE back-to-back gemvs reach 211 GB/s; the daemon's tiny-glue-broken gemvs
don't). Whole gpu_body ≈ 1.24GB/48GB/s ≈ 25ms — the decode is effective-bandwidth-bound,
and the megakernel (one large pipelined cooperative kernel, sustained high utilization)
restores ~200 GB/s → 1.24GB/200 ≈ 6ms ≈ 160 tok/s oq8 (oq4 → ~320+). This is the SAME
megakernel conclusion, now confirmed from the bandwidth-pipelining angle (a third
independent line of evidence alongside launch-latency and utilization).

**Bottom line this session:** constraint PROVEN (launch-latency × kernel count, clock/
bandwidth refuted by direct experiment); 3 coherent kernel wins landed (33→35 tok/s);
full probe toolkit + this build map in place. Reaching 200/400 requires the megakernel
(cooperative-grid), a multi-session build — not achievable in a single turn without
destabilizing the working path.

### EXP-17 (2026-07-24): megakernel Phase 0 + Phase 1 landed — launches 4.3× down, tok/s FLAT (GPU-exec-bound reconfirmed)

Implemented the cooperative megakernel (plan `docs/plans/2026-07-24-...`), env
`HIPFIRE_ZAYA_MEGAKERNEL={1,validate}`, on `zaya1-8b.oq8.hfq` (the indexed-planar-oq8
variant — NOT the plan's stated `oq8++` target, which carries AWQ + host-MoE and has
`moe_indexed=false`; see below).

- **Phase 0 (megakernel-B, stages 12–17 MLP half):** post-attn rmsnorm+rotate → router
  MLP+select → MoE gate_up → silu_mul+rotate → MoE down+affine, ONE cooperative launch,
  3 grid.sync/block (stages 16+17 fused per-row). **Validated: all 40 layers `hidden`
  cos 0.999999–1.000000 vs reference, both decode tokens.**
- **Phase 1 (megakernel-A, stages 1–8 front half):** input rmsnorm+rotate → fused qkv
  Oq8 gemv → qk-prep → conv window+ring → depthwise/grouped conv1d → add-conv-residual +
  value-assemble → q/k L2-norm → partial RoPE, ONE cooperative launch, 8 grid.sync/block.
  Attention + o_proj kept separate. **Validated: query/key/value cos = 1.000000 all layers.**
- **End-to-end A+B ON:** byte-identical greedy text vs baseline ("Paris.\n\n- The user
  might have a typo..."). Correctness gate PASSED.

**Perf (LAUNCHSTATS, 4 tokens):**
| | launches/token | CPU submit/token | decode tok/s |
|--|--|--|--|
| baseline | 1403 | ~1153 µs | 36.5 |
| megakernel A+B | **323** (4.3× fewer) | ~425 µs (2.7× less) | 37.7 (+3%) |

⟹ **The 4.3× launch cut moved tok/s only +3%** — decisive reconfirmation that decode is
GPU-EXECUTION-bound, not launch/CPU-bound (consistent with EXP-11: ~26 ms/token ≈ 48 GB/s
effective; CPU submit 425 µs ≪ GPU exec). **Fewer launches ≠ bandwidth ramp.** The GPU
still de-saturates because: (a) A and B are two *separate* cooperative launches with the
attention seam (kv_write×2 + attention + o_proj + 2 affines, ~6 regular launches) between
them; (b) each half's grid-strided gemvs are punctuated by grid.syncs at tiny phases
(rmsnorm/router/conv) that don't sustain DRAM throughput. The bandwidth ramp needs the
*whole block* as one sustained cooperative kernel (Phase 3: fold attention; Phase 4:
single-launch whole decode). Phases 2–3 are the tok/s lever, not the launch count.

**Artifact gap:** plan target `zaya1-8b.oq8++.hfq` has `has_awq=true` + `moe_indexed=false`
→ runs W8A16 + HOST-MoE fallback, so the megakernel (indexed-planar-oq8) does NOT fire on
it. Only `zaya1-8b.oq8.hfq` (w8a8_pa + moe_indexed, no AWQ) is eligible. Reaching >200 on
oq8++ needs an AWQ-aware indexed MoE + megakernel, or re-inducting the target as plain-oq8.

### EXP-18 (2026-07-24): megakernels run at FULL occupancy (160 blocks) yet don't ramp bandwidth → serial single-block phases are the suspect

Probed the cooperative grid via `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`:
**per_mp=8, mp=20, grid=160** for BOTH megakernel-A and megakernel-B — i.e. the max
cooperative residency the plan targeted. So occupancy is NOT the reason tok/s stayed flat.

⟹ New hypothesis for why full-occupancy fusion still runs at ~48 GB/s: each megakernel
contains **serial single-block phases** gated by `grid.sync`, where 159 of 160 blocks idle:
- megakernel-B: stage 12 (rmsnorm+rotate) and stage 13 (the ENTIRE router MLP: down_proj
  256×2048 + fc1/fc2 256×256 + out 17×256 + 3 FWHTs + 2 rmsnorms) run on **block 0 only**.
- megakernel-A: stage 1 (rmsnorm+rotate) and stage 3 (qk-prep) run on **block 0 only**.
- Also stage 15 silu (megakernel-B) uses only 8 of 1280 warps (moe_int/256 groups).

While a single-block phase runs, the grid.sync forces the other 159 blocks to wait and the
memory subsystem is not driven → the DRAM clock/bandwidth can't stay ramped. This is the
"single-WG is a WASH" lesson (EXP-9) re-appearing *inside* the cooperative kernel: the
grid-strided gemv phases (full 1280 warps, fast) are bracketed by serial Amdahl sections.

**Redirect:** before (or instead of) folding attention (plan Phase 3), the currently
block-0-only phases must be **parallelized across the whole grid**:
- rmsnorm+rotate: grid-strided sum-of-squares (atomic or 2-level reduce + grid.sync), FWHT
  groups across all warps (already parallel-capable — just remove the `blockIdx==0` guard).
- router down_proj (the big part, 256×2048): grid-stride the 256 rows over the 1280 warps;
  keep only the tiny fc1/fc2/out/argmax on a single block.
This is the likely key to the bandwidth ramp — a whole-block fusion that still contains
serial single-block sections will not sustain utilization no matter how many stages fuse.

**Caveat — two competing hypotheses, needs isolation before more building:**
1. *Serial single-block phases* (above): but arithmetic says the block-0 router down_proj
   is ~512KB/layer ≈ ~1ms/token total — only ~4% of the 26 ms. So serial phases alone do
   NOT explain the flat tok/s.
2. *grid.sync overhead*: A has 8 syncs/block, B has 3 → ~11 grid.sync/block × 40 layers ≈
   **440 cooperative grid barriers/token**. On a 160-block grid each barrier is a global
   atomic + L2 fence; at even ~20 µs each that is ~9 ms/token — plausibly the dominant cost
   and a structural limit of the many-sync design. This would explain why 4.3× fewer
   *launches* gave ~0 tok/s: grid.syncs replaced the launch barriers ~1:1.

**Action before building Phase 2/3:** measure in isolation — (a) wrap megakernel-B in
device_synchronize timers to get its GPU ms, (b) build a B variant with the gemv phases
only (stub the syncs/serial phases) to bound the gemv bandwidth, (c) micro-bench a single
`grid.sync()` on a 160-block grid. If grid.sync dominates, the fix is FEWER syncs (coarser
phases, or Phase 4 whole-decode is WORSE not better), not folding attention. The plan's
"more fusion → ramp" premise needs this measurement to hold. **Do not build Phase 3 (the
expensive attention fold) until the sync-vs-bandwidth question is answered.**

### EXP-19 (2026-07-24): grid.sync() = ~1.0 µs at 160 blocks → NOT the bottleneck (refutes hyp #2)

Micro-benched `cg::this_grid().sync()` at the megakernel launch shape (`zaya_grid_sync_bench`,
2000 syncs × 5 iters, device-sync timed):
| grid | µs/grid.sync | implied µs/token (11 syncs/blk × 40) |
|--|--|--|
| 160 | 0.994 | 437 |
| 80 | 0.652 | 287 |
| 40 | 0.532 | 234 |
| 20 | 0.461 | 203 |

437 µs/token of grid.sync is **~1.7% of the 26 ms/token** — NEGLIGIBLE. So the flat tok/s is
NOT grid.sync overhead (and Phase 3/4 whole-decode won't be sync-limited either — good).
Combined with EXP-18 (occupancy=160 full; serial phases only ~1 ms), **none of the three
suspects account for the 26 ms.** The grid-strided oq8 gemvs, fused at full occupancy, still
run at ~48 GB/s. NEXT: device-sync section timers (body-loop vs lm_head vs attention) to
find where the 26 ms actually is — the plan's bandwidth-ramp premise cannot be evaluated
until we know which section is slow.

### EXP-20 (2026-07-24): the breakdown — lm_head is 41% of decode; the megakernel body fusion buys only −5.6%; everything runs at ~50 GB/s (plan premise UNCONFIRMED)

Device-sync section timers (`HIPFIRE_ZAYA_SECTIONTIME`, steady state, zaya1-8b.oq8):
| section | megakernel OFF | megakernel ON |
|--|--|--|
| body loop (40 layers) | ~15.05 ms | ~14.19 ms (**−5.6%**) |
| lm_head (tied Q8 embed) | ~10.73 ms | ~10.75 ms (unchanged) |
| **total** | ~25.8 ms (38.8 t/s) | ~24.9 ms |

- **lm_head = 10.7 ms = 41% of decode.** It reads 537 MB (Q8, vocab 262,272 × 2048) →
  537 MB / 10.7 ms = **50 GB/s**. The plan assumed 2.6 ms (211 GB/s); it's 4× that because
  lm_head ALSO runs at the ~50 GB/s effective rate, not the ramped bandwidth.
- **The body megakernel saves only 0.85 ms (−5.6%).** Fusing the whole MLP+front half at
  full 160-block occupancy does NOT restore bandwidth — the body stays at ~50 GB/s.
- ⟹ **The plan's central premise (cooperative megakernel → bandwidth ramp → ~4× tok/s) is
  NOT confirmed by the implementation.** Occupancy is full (EXP-18), grid.sync is negligible
  (EXP-19), serial phases are ~1 ms (EXP-18) — yet the fused body runs at the SAME ~50 GB/s
  as the unfused kernels. Whatever pins decode at ~50 GB/s is NOT launch/sync/occupancy
  structure; it survives full fusion.

**Roofline consequence:** at ~50 GB/s, 200 tok/s (5 ms/token) needs ≤0.25 GB/token, but an
8B model reads ~0.66 GB/token even at oq4 (body ~0.39 + lm_head oq4 ~0.27). **So 200 tok/s
is IMPOSSIBLE without lifting the ~50 GB/s ceiling** — and the megakernel does not lift it.
The bandwidth-ramp question (why decode pins at ~50 GB/s when isolated large gemvs hit 211;
whether a power-state/clock-residency/memory-controller effect on the gfx1151 APU can be
driven from software) is now THE blocker and supersedes further fusion work.

**The one lever that works regardless of the ramp: oq4 lm_head (workstream B).** At 50 GB/s,
oq4 lm_head (0.27 GB) = ~5.4 ms vs Q8 10.7 ms → saves ~5.3 ms → 24.9→19.6 ms ≈ **51 tok/s
(+30%)**, purely from fewer bytes. Highest concrete ROI next step; independent of the ramp.

### EXP-21 (2026-07-24): ★ the ceiling is the GEMV KERNEL ACCESS PATTERN, not clock/idle — production lm_head gemv = 50 GB/s even in a 100× tight loop

Ran the PRODUCTION lm_head gemv (`w.embed.gemv`, Q8_0) 100× back-to-back with device-sync
timing (`HIPFIRE_ZAYA_LMHEAD_LOOP`): **10,659 µs/gemv → ~50 GB/s**, IDENTICAL to its
in-decode time. A sustained tight loop does NOT ramp it. Therefore the ~50 GB/s ceiling is
**not** idle-induced clock drop (refutes the clock-residency hypothesis) — it is the
**kernel's memory access pattern**. The EXP-11 bench (`q8_cold.hip`, grid-strided, same
537 MB) reaches 211 GB/s, so the hardware CAN sustain ~4× more on this exact data size; the
production gemv simply doesn't. My megakernel's inline one-warp-per-row oq8 gemv
(`zmk_oq8_planar_row`) has the same limitation (EXP-20: body fusion only −5.6%).

**Reframing of the whole effort:** the megakernel (Phases 0–1, correct + validated) targets
launch/utilization/occupancy — but EXP-17/18/19/20/21 prove NONE of those is the bottleneck.
The bottleneck is that the **decode GEMV kernels achieve only ~50 GB/s of the ~211 GB/s the
hardware sustains**, regardless of fusion, occupancy, sync count, or clock. The lever is a
**bandwidth-tuned decode gemv** (wider/vectorized loads, multi-row per wave, better
coalescing/prefetch — the kernel-tuning playbook), applied to the Q8/oq8 gemvs (lm_head +
body). At 211 GB/s: body 15→~3.6 ms, lm_head 10.7→~2.6 ms → ~6.2 ms/token ≈ **160 tok/s**
oq8 — the plan's target, reached by fixing the gemv, NOT by the cooperative megakernel.

**Recommended pivot:** (1) profile/tune the production `gemv_q8_0` / `gemv_oq8*` decode
kernels to close the 50→211 GB/s gap (compare against `q8_cold.hip`'s grid-strided pattern);
(2) oq4 lm_head (workstream B) as an orthogonal byte-reduction win. The cooperative
megakernel is correct and a fine launch-count optimization, but it is NOT the path to the
tok/s target — the gemv bandwidth is. Megakernel Phases 0–1 stay env-gated/off; keep for
later once the gemv ceiling is lifted (then launch/util may re-emerge as secondary).

**EXP-21 follow-up (avoid a red herring):** the lm_head kernel is `gemv_q8_0` (narrow arm,
K=2048>1536), and its grid is ALREADY capped: `grid = min(M, 8192)`, block=32, grid-strided
over rows (gemv.rs:4347). So the 50 GB/s is NOT command-processor dispatch overhead (that's
already mitigated). The kernel's header claims "222 GB/s microbench" but the FULL 262272-row
lm_head sustains only ~50 GB/s in-daemon (100× loop) — so the tuned v4 does not sustain on
the large gemv. Likely low memory-level-parallelism: block=32 (one wave/row) → few resident
waves × serial per-block loads → latency-bound, not bandwidth-bound. Tuning levers to try
(kernel-tuning skill + microbench each, cf. [[reference_gfx1151_iu4_gemm_tuning]]): more
waves/MP (bigger block or 2-warps-per-row like the wide arm), vectorized uint4 loads,
software-pipelined/prefetched K loop to raise in-flight loads, multi-row register blocking.
Target: close 50→~200 GB/s on the 262k×2048 lm_head. Same tuning applies to the body oq8
gemvs (gemv_oq8*) and would make the megakernel's inline gemv fast too.

### EXP-22 (2026-07-24): ★ DEFINITIVE — max clocks + 100% busy + 50 GB/s = kernel MLP problem, not clock/power

Sampled amdgpu DPM sysfs (`pp_dpm_sclk`/`pp_dpm_mclk`, card1) DURING the 3000× lm_head loop:
**sclk pinned 2890–2900 MHz (max of 2900), mclk pinned 937 MHz (max), gpu_busy_percent = 99–100%**
— yet the gemv holds 50 GB/s (24% of the 211 GB/s peak achievable at this exact mclk). So the
GPU is fully clocked and saturated on COMPUTE-occupancy yet starved on memory throughput =
classic **latency-bound / low memory-level-parallelism**: the CUs are 100% "busy" stalling on
memory returns, but there aren't enough concurrent loads in flight to saturate DRAM. This
CONCLUSIVELY rules out clock/power-state (refutes any residual EXP-6/11 clock doubt) and
confirms the lever is raising the gemv's outstanding-load count. Fix = multi-row register
blocking (R rows/wave → R× independent load streams) and/or deeper prefetch.

### EXP-23 (2026-07-24): ★ CORRECTION — the lm_head embed is F32 (2.15 GB), running at ~200 GB/s = PEAK. EXP-20/21/22's "50 GB/s" was a 4× byte error.

`w.embed.quant_dtype() = None` ⟹ the tied embed/lm_head for zaya1-8b.oq8.hfq is stored
**F32**, not Q8 (the "oq8" quant keeps the embed unquantized on purpose — see
[[reference_embed_quant_residual_sensitivity]], embed rides the residual unnormalized). So it
reads **262272 × 2048 × 4 = 2.147 GB**, NOT 537 MB. Recompute the 10.7 ms lm_head loop:
2.147 GB / 10.7 ms = **~200 GB/s — essentially PEAK.** The lm_head is NOT slow per byte and
NOT latency-bound; it is a large F32 gemv at peak bandwidth. EXP-22's "max sclk + max mclk +
100% busy" is therefore CORRECT peak behavior, not a stall. **Retract EXP-21/22's claim that
the lm_head gemv is stuck at 24% of peak — that was based on the wrong 537 MB figure.**

Consequences:
- **lm_head fix = QUANTIZE it, not bandwidth-tune it.** F32 2.15 GB @ 200 GB/s = 10.7 ms;
  Q8 0.54 GB ≈ 2.7 ms; oq4 0.27 GB ≈ 1.35 ms. Quantizing the OUTPUT projection (untied from
  the quality-sensitive F32 input embed) saves ~8 ms → 26→18 ms ≈ **55 tok/s from a Q8
  lm_head alone**. This is exactly plan workstream B and is now clearly the single biggest win.
- The multi-row gemv_q8_0 experiment (EXP earlier this session) targeted the WRONG kernel:
  the lm_head is F32 (`gemv_f32`), the body uses oq8 gemvs — neither is `gemv_q8_0`. Kernel
  kept (env `HIPFIRE_GEMV_MROW`) but UNVALIDATED for a real target; do not trust it yet.
- **Body (~15 ms) effective bandwidth is still unmeasured** — must count its ACTUAL bytes
  (oq8 weights + f32 intermediates + attention) before concluding it is or isn't latency-
  bound. Do NOT repeat the byte-estimate error; instrument the real read volume.

**Revised priority: workstream B (quantize the F32 lm_head → Q8/oq4) is the clear #1 win
(~+45%), independent of everything else. Then measure the body properly before tuning it.**

### EXP-24 (2026-07-24): ★ DEMONSTRATED WIN — untied F16 lm_head + megakernel = +15% tok/s, coherent

Implemented an untied F16 lm_head (`HIPFIRE_ZAYA_F16_LMHEAD`): input gather keeps the F32
tied-embed table; the output projection reads a one-time F16 copy (2.15→1.07 GB) via
`gemv_f16_xf32` (now grid-strided + grid-capped at 8192). Byte-identical greedy output.

Section timer (steady state): **F16 lm_head = 6.75 ms vs F32 = 10.73 ms** — the halved-byte
output projection saves ~4 ms GPU/token, confirming the workstream-B thesis (reduce lm_head
precision → faster; F16 here is a stand-in for the plan's oq4 LDLQ).

decode tok/s (40 tokens, zaya1-8b.oq8):
| config | tok/s |
|--|--|
| baseline (F32 lm_head, no megakernel) | 36.4 |
| F16 lm_head alone | ~29 (SLOWER — see note) |
| megakernel A+B alone | 37.7 |
| **megakernel A+B + F16 lm_head** | **42.1 (+15%)** |

Note: F16-lm_head-ALONE is slower despite the 4 ms GPU saving — its `gemv_f16_xf32` launch
sits behind the unfused F32 body's ~1400 launches and loses pipeline overlap. The megakernel
(body 1403→323 launches) clears that contention, so F16 pays off only in combination:
26.5 ms (megakernel) − 4 ms (F16) = 22.5 ms ≈ 44 tok/s, matching the measured 42. This is the
first end-to-end tok/s win of the effort, and it comes from the lm_head byte-reduction
(workstream B) UNLOCKED BY the megakernel's launch reduction — the two compose.

Path forward: F16→oq4 lm_head (LDLQ, plan workstream B) cuts the output read further
(1.07→0.27 GB, ~6.75→~1.7 ms) → another ~5 ms → ~55–60 tok/s. Plus body MoE gemv tuning
(~100→200 GB/s, ~2.5 ms). The megakernel is a real enabler here, not a dead end — its value
is launch-reduction that unlocks the byte-reduction wins, not a standalone bandwidth ramp.

### EXP-25 (2026-07-24): body oq8++ kernel-opt attempts — decomposition + multi-row MoE gemv FAILED

User directive: optimize the oq8++ kernels until bandwidth-limited BEFORE compromising
quality (Q8 lm_head is a real ~40% KLD hit per docs/todo/2026-07-22-embedding-quant-
improvements.md — the embed rides the residual unnormalized).

Body decomposition (ablate hooks + section timer, oq8, ctx~5-28):
| slice | body time | Δ |
|--|--|--|
| full | ~15.1 ms | — |
| no MoE gemvs | ~9.9 ms | MoE gate_up+down ≈ **5.2 ms** |
| no glue (conv/l2/rope/…) | ~11.3 ms | glue ≈ **3.8 ms** |
| no MoE + no glue | ~6.2 ms | qkv+o_proj+attn+norms+router ≈ **6.2 ms** |

Also: **oq4.25 body ≈ oq8 body (14.9 vs 15.0 ms)** → body is NOT weight-bandwidth-bound
(halving FFN bytes gives ~0). And megakernel fusion helped only −5.6% → NOT launch-bound.
⟹ the body is GPU-EXECUTION-bound, spread across three ~equal slices.

**FAILED experiment — multi-row (4 rows/wave) indexed MoE oq8 gemv** (`HIPFIRE_ZAYA_MOE_MROW`,
zaya_moe_{gate_up,down}_oq8_planar_indexed_mrow): body 15.2→**22.5 ms** (WORSE), tok/s
36→29, coherent. Root cause: the one-row kernel already launches grid=[M]=4096 blocks =
huge concurrency (≫160 resident); 4 rows/wave quadruples register pressure (occupancy ↓)
and cuts block count 4× → less parallelism, not more. The lm_head-MLP intuition does NOT
transfer — the MoE gemv is already well-parallelized. Kernel kept env-gated (default = the
fast one-row path); do not enable.

**Takeaway:** the body oq8++ kernels resist the obvious levers (multi-row MLP backfires;
fewer bytes and fusion barely help). The body is execution-bound and near its floor for
these access patterns. Remaining body ideas (uncertain): the SINGLE-WG router MLP
(zaya_router_mlp_fused, one block) may be a serial chunk of the 6.2 ms remainder; attention
grows with ctx. The clear big lever remains the F32 lm_head (10.7 ms, 41%) — but it's at
peak bandwidth (EXP-23) so its only speedup is quality-lossy quantization, gated by the user
until body opt is exhausted. Quality-preserving lm_head path = untied bf16-gather + a
matmul-friendly (QTIP/trellis) lm_head codec (embedding-quant doc point 4), still unquantified.

### EXP-26 (2026-07-24): Phase 2 built — o_proj + attn affine folded into megakernel-B (coherent)

Implemented plan Phase 2 (`HIPFIRE_ZAYA_MEGAKERNEL=2`): the attention o_proj (stage 10) +
attn affine residual (stage 11) fold into megakernel-B's head — FWHT-rotate ctx (q_dim/256
groups, block 0), then per-row fused `g_res2[r] = affine(o_proj·ctx_rot[r], hidden[r], pa_rs)`
(stages 10b+11 fused per row, no attn_out scratch). Gated by `fold_oproj` so Phase 0/1
(=1, fold off) is unchanged. Reference o_proj+affine skipped when folding. **Byte-identical
output vs baseline; 37.3 tok/s (=1: 37.2, baseline: 36.3); grid=160 unchanged (fold code did
not hurt occupancy).** tok/s flat as expected — consistent with EXP-17/20/25: the body is
execution-bound, not launch-bound, so folding more stages doesn't move decode. Phase 2 is a
correct, validated plan deliverable; its value is plan-completeness + setup for Phase 3, not
tok/s. Phase 3 (fold attention → 1 launch/block) remains: hardest (online-softmax over
growing KV in a cooperative grid) and — per all measurements — will not move tok/s either
(decode is not launch-bound). The real levers remain lm_head precision (quality-gated) and
the body being near its execution floor.

### EXP-27 (2026-07-24): Phase 3 (fold attention) BUILT but HANGS at runtime — deferred

Implemented plan Phase 3 (`HIPFIRE_ZAYA_MEGAKERNEL=3`): the KV write + flash-decode attention
(stage 9) fold into megakernel-B's head via the register-resident online-softmax pattern
(attention_f32_nolds) — heads grid-strided over warps, then the o_proj fold consumes ctx. So
a block = megakernel-A + megakernel-B (2 cooperative launches, attention now cooperative).
Kernel compiles; wiring (fold_attn param through B's signature + skip reference stage 9)
compiles. **But `=3` HANGS decode after the first token** (GPU recovered on process kill; APU
not wedged). Cause not yet pinned (likely a cooperative-grid residency / grid.sync issue: B
now runs 7 grid.syncs — 2 fold_attn + 2 fold_oproj + 3 base; the memoized grid=160 may exceed
the true cooperative residency of the now-larger kernel, deadlocking the barrier — though =1/=2
use the same kernel/grid without hanging, so this is unconfirmed).

**Deferred** — NOT debugged further because: (a) Phase 3 is proven ZERO tok/s value (decode is
execution-bound, not launch-bound: A+B+Phase2 already cut launches 4.3× for ~0 gain — EXP-17/26);
(b) iterating on a cooperative-kernel hang risks wedging the APU GPU (cf. 397B kworker deadlock).
Default + Phases 0/1/2 are healthy and unaffected (=3-gated): baseline 36.2 tok/s coherent.
Plan status: all 4 phases IMPLEMENTED; Phase 3 has a runtime hang bug. The genuine levers
remain lm_head precision (quality-gated; untie+QTIP research) and the body (near execution floor).

### EXP-28 (2026-07-24): ★ Phase 3 FIXED + WORKING — the hang was a Rust variable-shadow bug

The EXP-27 hang was NOT a cooperative/grid issue — it was a **variable shadow** in the
megakernel-B dispatch wrapper: `let (gr, hd) = (g_res2..., hidden...)` binds a local `hd` =
hidden POINTER, which shadowed the new `hd: usize` (head_dim) Phase-3 param. So the kernel
received the hidden-buffer address (a huge value) as `head_dim`, and the folded attention's
`for (d=lane; d<hd; d+=32)` looped to a garbage-huge bound → effectively infinite / massive
OOB → hang. (The compiler flagged it as "unused variable: hd" — the tell.) Fix: renamed the
param to `head_dim`. **Phase 3 (`=3`) now runs: byte-identical output, 36.2 tok/s, no hang.**

**ALL 4 PLAN PHASES IMPLEMENTED + FUNCTIONAL:** =1 Phase 0/1 (megakernel-B MLP half + -A
front half, cosine ≈1.0), =2 Phase 2 (o_proj + affine fold, byte-identical), =3 Phase 3
(KV-write + flash-decode attention fold, byte-identical). A decode block is now megakernel-A
+ megakernel-B (2 cooperative launches, attention cooperative). tok/s is flat across all
phases (~36) — as every experiment predicted (decode is execution-bound, not launch-bound;
the cooperative megakernel is CORRECT but not the tok/s lever). The plan's design is fully
realized and validated for correctness; reaching >200 tok/s needs the lm_head precision work
(quality-gated) + the body already being near its execution floor at oq8++ quality.

### EXP-29 (2026-07-24): ★ lm_head was bf16-on-disk WIDENED to F32 in VRAM — un-widening gives +19% QUALITY-NEUTRAL

The "lm_head is F32" premise (EXP-20/23) was a runtime-VRAM fact, not disk. On disk the tied
embed is **BF16 (qt=16, 1.07 GB)**; the zaya loader `linear_dtype()` had no bf16/f16 verbatim
path, so it fell through to `dequant_qt → upload_f32` = **widened to F32 (2.15 GB) in VRAM**.
That widening is pure waste — bf16 IS the source, so keeping it bf16 is bit-identical AND
half the bytes. Fix (loader gap, NOT a quality tradeoff):
- `linear_dtype`: 16→BF16, 1→F16 (upload verbatim, 2 B/elem, no widen).
- embed gather → `embedding_lookup_bf16` (bf16→f32 in-kernel, lossless).
- lm_head gemv → `gemv_bf16_f32` (bf16 weight, F32-accumulate; cast the f32 activation to
  bf16 first — `gemv_bf16_f32` is bf16×bf16). Weight lossless; activation-bf16 is negligible.

**Measured: lm_head 10.7→6.3 ms, decode 36.3→43.3 tok/s (+19%), output BIT-IDENTICAL to the
F32 baseline** (" Paris.\n\n- The user might have a typo: ..."). This is now the DEFAULT path
for any zaya model with a bf16/f16 embed — no env gate. The earlier F16-lm_head experiment
(EXP-24) was approximating this; bf16 is the correct, source-exact version.

⟹ Corrects the session's framing: the lm_head speedup is NOT quality-gated (that was the
byte-error's downstream mistake). ~19% was sitting in an unnecessary F32 widening. Remaining
(genuinely quality-gated) lm_head headroom is only BELOW bf16 (Q8/oq4 → ~40% KLD tax per the
embed-quant doc). For zero activation loss too, a bf16-weight×f32-activation gemv (portable
`gemm_bf16_x_f32`) would be exact; the bf16×bf16 here is greedy-bit-identical and simpler.

### EXP-30 (2026-07-24): two-stage lm_head — Q4 coarse shortlist DE-RISK, recall measured

Prototyped the coarse→fine lm_head measurement: per-row symmetric Q4 copy of the bf16 embed
(the aggressive/worst-case coarse: ONE scale per 2048-dim row), score all 262k tokens, compare
the coarse top-K to the exact full bf16 argmax + softmax mass. `HIPFIRE_ZAYA_LMHEAD_SHORTLIST=1`,
kernels/src/gemv_q4sym_f32.hip + host recall/mass. 24 greedy tokens on a France-history prompt:

**Recall@1 (true bf16 argmax ∈ coarse top-K):**
- K=256 : ~87.5% (misses 3/24, all high-entropy steps)
- K=1024: ~95.8% (misses 1/24)
- **K=2048: 100% (24/24)** ← greedy-safe on this sample even at worst-case per-row Q4.

**Captured softmax mass:** ~0.99+ on most steps; drops to 0.16–0.87 on a few high-entropy
steps (flat distributions) — matters for faithful *sampling*, not greedy.

**Latency:** coarse gemv = **2.5 ms** (268 MB Q4 @ ~107 GB/s) + fine-gather ~0.03 ms ≈ 2.5 ms
vs 6.3 ms full bf16 = **~2.5× on the lm_head**. One-time lazy build = 355 ms (download+quant+upload;
belongs offline). ⟹ CORE QUESTION ANSWERED: a Q4 coarse reliably shortlists the winner (100%
recall@1 at K=2048). BUT full-H Q4 alone is only ~2.5× because it still reads 268 MB every token.

**Next lever = dimensionality reduction (as predicted):** the big win needs a smaller coarse
read. Project H=2048→r (SVD/PCA/learned), coarse = V×r×0.5: r=256 → 33.5 MB → ~0.2–0.3 ms →
~20–30× lm_head. Also per-GROUP Q4 (vs per-row) would lift recall at smaller K and the mass on
high-entropy steps. Both are the follow-on once the full serving path (top-K select + gather-fine
+ scatter) is built. For sampling faithfulness, add the frequent-token union and/or a coarse
entropy trigger to widen K on flat steps. Greedy is already safe at K=2048.

### EXP-31 (2026-07-24): coarse improvements — row-wise norm is a BIG win; random projection FAILS

Two changes to the coarse scorer, measured (32 greedy tokens, France prompt):

**(1) Row-wise L2 normalisation + 3σ-clip Q4** (exact per-row L2 norm × global 3σ-clipped
unit-direction Q4, vs plain per-row-max Q4):
| coarse (full H=2048) | recall@1 K=256 | K=1024 | K=2048 | mass@256 |
|--|--|--|--|--|
| plain per-row-max Q4 (EXP-30) | 87.5% | 95.8% | 100% | dips to 0.30–0.70 |
| **row-norm + 3σ-clip Q4** | **100%** | 100% | 100% | **0.9966** |
Row-wise normalisation lifts K=256 recall 87.5→100% and kills the high-entropy mass dips —
per-row-max was letting one outlier component crush the direction. ⟹ **K=256 is now
greedy-EXACT (100% recall@1) and captures 99.7% mass**; the coarse is solved at full H.

**(2) Random (JL) projection H→r for the bandwidth lever — FAILS:**
| coarse | read | build | recall@1 K=256 / 1024 / 2048 |
|--|--|--|--|
| r=512 random proj | 67 MB | 4.0 s | 28% / 40% / **50%** |
Random projection does not preserve the ranking (50% even at K=2048). As flagged, the
dim-reduction needs a STRUCTURE-AWARE projection (top-r SVD/PCA of the tied embed), not JL —
SVD is the optimal rank-r inner-product approximation; random JL needs r ≫ 512 for 262k tokens.

**Where this leaves the two-stage lm_head:**
- Coarse quant: SOLVED (row-norm Q4, K=256, greedy-exact). Build 183 ms (rayon), offline-able.
- Full-H row-norm coarse = 268 MB / 2.5 ms + fine-gather(256 rows) ≈ 2.5 ms vs 6.3 ms bf16 =
  **~2.5×, greedy-exact** — a real modest win NOW, no projection needed.
- The big (20–30×) win still needs the coarse READ down, and random projection is out. Two
  viable paths: (a) **SVD/PCA projection** (needs a truncated-SVD build — a linalg routine or
  randomized SVD), or (b) **lower coarse bits** — row-norm makes Q3/Q2 viable; Q2 full-H =
  134 MB → ~1.3 ms → ~5× with no projection machinery (needs a Q2/Q3 gemv). Cheaper to try (b).

### EXP-32 (2026-07-24): SVD dim-reduction hits the isotropy wall; Q2 too lossy; Q4 full-H is the viable coarse

Completed the row-norm → SVD → low-bit sweep (32 greedy tokens). Randomized-SVD projection
(top-r right singular vectors of the row-normalized unit directions, subsample gram +
randomized range finder, build ~8–32 s):
| coarse (row-norm Q4) | read | recall@1 K=256 / 1024 / 2048 |
|--|--|--|
| full-H (r=2048)      | 268 MB | **100% / 100% / 100%** |
| SVD r=1024           | 134 MB | 59% / 78% / 87% |
| SVD r=512            | 67 MB  | 43% / 68% / 78% |
| random r=512         | 67 MB  | 28% / 40% / 50% |

SVD beats random (78 vs 50% @2048 for r=512) but **even r=1024 (½ dims) only reaches 87%** —
the row-normalised *direction* space is isotropic (full-rank), so there is no low-rank
structure for SVD to compress. **Row-norm and SVD pull opposite ways**: row-norm helps
low-bit quant but removes the spectral decay SVD needs. Dim-reduction is OUT for this lm_head.

Low-bit on full-H (row-norm, no projection):
| coarse (full-H, row-norm) | read | coarse gemv | recall@1 K=256 / 1024 / 2048 |
|--|--|--|--|
| **Q4** | 268 MB | 2.5 ms | **100% / 100% / 100%** (greedy-EXACT) |
| Q2 | 134 MB | 1.3 ms | 93% / 96% / 96% (4% greedy miss — not exact) |

Q2 doubles the bandwidth but recall plateaus at 96% (misses aren't "just outside K" — 2-bit
genuinely can't separate them), so it's not greedy-exact. **Q4 full-H row-norm is the viable
coarse: greedy-exact at K=256, ~2.5× on the lm_head.** ⟹ two-stage lm_head realistic win ≈
**~2×** (coarse 2.5 ms + top-K select + 256-row bf16 fine gather vs 6.3 ms bf16) → ~43→~50
tok/s, greedy-exact. The 20–30× needed dim-reduction, which the isotropy defeats.

Open lever (user's Stage 3): a low-rank RESIDUAL correction added to the Q2 coarse could
recover the 4% Q2 loses (recover angular detail without full precision) → Q2's 5× at
greedy-exact. That's the remaining path to push past 2×; needs the correction build + measure.

### EXP-33 (2026-07-24): Stage-3 low-rank correction WORKS — Q2+corr beats Q4 (greedy-exact, fewer bytes)

Built the residual correction: coarse += A[V,r]·(B[r,H]·h), where A,B = randomized top-r SVD
of the residual D = W − Q2recon. Contra my isotropy prior, the residual has ENOUGH low-rank
structure to fix Q2's tail:
| coarse (full-H, row-norm) | recall@1 K=256 / 1024 / 2048 |
|--|--|--|
| Q2 (no corr) | 93% / 96% / 96% |
| **Q2 + corr r=64** | 93% / 96% / **100%** |
| **Q2 + corr r=128** | 96% / **100%** / 100% |
More correction rank → greedy-exact at smaller K (r=64 exact@2048, r=128 exact@1024).

**Byte-optimal EXACT configs (lm_head, greedy):**
| config | coarse read | ~lat | greedy-exact |
|--|--|--|--|
| bf16 (shipped) | 1074 MB | 6.3 ms | yes (baseline) |
| Q4 full-H, K=32 | 268 MB | 2.5 ms | yes (2.5×) |
| **Q2 + corr r=64 (A bf16), K=2048** | 134+33+8 ≈ 175 MB | ~1.6 ms | yes (**~4×**) |

⟹ The full pipeline (row-norm → low-bit → low-rank correction) is validated and **Q2+corr
beats Q4** — greedy-exact at ~1.6 ms (~4× over bf16, ~1.6× over Q4). Only the SVD *dim-reduction*
step failed (isotropy, EXP-32); the SVD as a *residual correction* works. (A stored f32 in the
prototype = 67 MB → 209 MB total, still < Q4's 268 MB; bf16 A → 175 MB is the target.)
Realistic decode: lm_head 6.3→~1.6 ms → ~43→~55 tok/s greedy-exact, then the ~15 ms body dominates.

**Full two-stage lm_head recipe (offline build, lean runtime):** row-wise L2-norm + 3σ-clip Q2
coarse (134 MB) + a rank-64 residual correction (A bf16 33 MB, B [64,2048]) → coarse+corr score
→ top-K(2048) select → 2048-row bf16 fine gather → scatter. All greedy-exact. Serving-path
kernels still to build: top-K-over-262k select, gather-K bf16 fine gemv, scatter.

### EXP-34 (2026-07-24): two-stage lm_head SHIPPED — runtime-selectable, greedy-exact, 2.0–2.3×

Built the real serving path (not just the measurement harness): coarse row-norm
Q-scorer → host top-K → fused bf16 gather+scatter (`gemv_bf16_gather_f32`) into a
-inf-masked logit vector. Runtime-selectable via `HIPFIRE_ZAYA_LMHEAD` — fully
parametric over (bits ∈ {2,4}, correction rank, K), presets q4/q4c/q2/q2c, overrides
`_K` / `_CORR`. Refactored the coarse build/score out of the diagnostic into shared
`build_lmhead_coarse` + `coarse_scores_host` (measure + serve call both).

GPU-validated on zaya1-8b.oq8.hfq (32-tok greedy, vs full bf16 lm_head):
| mode | lm_head median | vs bf16 | greedy text |
|--|--|--|--|
| bf16 (baseline) | 6343 µs | 1× | ref |
| q4 (K=32) | 3196 µs | 2.0× | **byte-identical** |
| q4c (K=32, r=64) | ~3.2 ms | 2.0× | **byte-identical** |
| q2 (K=2048) | ~2.7 ms | 2.3× | 1 token drift (96% recall miss) |
| q2c (K=2048, r=64) | 2717 µs | 2.3× | **byte-identical** |

So q4/q4c/q2c are provably greedy-exact in serving; q2-alone drifts exactly as its
96% recall predicted. Speedups sit under the pure-bandwidth ideal because the host
`select_nth` top-K over V=262k is in the critical path → next lever = GPU top-K
(+ store correction A at bf16/Q8 not f32). Kernels are wave32 f32-accumulate
(RDNA2/3/4-portable). Methodology doc: docs/kernel_work/two-stage-lmhead.md.
Default path unchanged (opt-in env). Cargo check clean; no-gpu-ci pending.

### EXP-35 (2026-07-24): two-stage lm_head — coarse gemv vectorized + packed-key select → 3.2× exact

Instrumented the serving split (HIPFIRE_ZAYA_LMHEAD_TIMING): coarse gemv was the cost
(q4 ~2418µs @ ~110 GB/s), NOT the host top-K (~468µs). Root cause: q4sym/q2sym loaded
weight a BYTE at a time → nibble/2-bit unpack ALU starved the load-issue rate. Fix:
each lane consumes a uint32 (8 nibbles / 16 two-bit) per load → q4 coarse 2418→1474µs
(~181 GB/s, now FASTER than the bf16 gemv's 169). Also uncapped grid (1 block/row like
gemv_bf16_f32; the 8192 cap + row-stride serialized 32 rows/block). Select: packed each
score into order-preserving u64=(monotone_f32_bits<<32)|idx, selected on raw u64 (no
comparator closure) → 468→303µs. rayon on the 262k keyed-build was 2-3× WORSE (dispatch
overhead) — reverted.

Final (baseline lm_head 6336µs, all byte-identical greedy except q2):
| mode | lm_head | vs bf16 | exact |
|--|--|--|--|
| q4 (K32) | 1958µs | **3.2×** | yes ← BEST exact |
| q2c (K2048,corr r64) | 2092µs | 3.0× | yes |
| q4c (K32,corr) | 2649µs | 2.4× | yes (corr WASTED, q4 already exact@32) |
| q2 (K2048) | 1442µs | 4.4× | no (1-token drift) |

⟹ q4 is the recommended EXACT mode (3.2×, was 2.0× pre-vectorize); no correction needed.
q4c strictly dominated by q4 (correction only helps aggressive coarse=q2). Remaining
lever = GPU top-K (~300µs host select → ~15µs on-device; needs GPU-resident score +
corr-add on GPU) → est q4 ~1.65ms ~3.8×. A→bf16/Q8 trims q2c/q4c but q4 wins anyway.
Doc updated: docs/kernel_work/two-stage-lmhead.md. Default path unchanged.

### EXP-36 (2026-07-24): GPU top-K built → q4 lm_head 3.9× exact (host select removed)

Moved the ~300µs host select onto the GPU: coarse score + low-rank correction now stay
device-resident (correction via add_inplace_f32, coarse+=A·(B·h)), and a device top-K
selects the shortlist — only min/max, a 16KB histogram, and the final count cross to host.
Passes (kernels/src/lmhead_coarse_*.hip): (1) min/max of coarse as order-preserving u32
keys via integer atomicMin/Max (monotone f32→u32); (2) 4096-bin linear histogram over
[min,max] (bin in actual key range, not full u32 → good selectivity); (3) host scans bins
top-down to threshold τ where cumulative≥K, compact kernel appends rows with key≥τ. The
set is a SUPERSET of exact top-K (boundary bin whole) — harmless, fine pass rescores
exactly. idx buffer sized from histogram tail+slack (never truncates).

Final (baseline lmhead 6329µs, GPU top-K default, all byte-identical greedy except q2):
| mode | host-select | GPU top-K |
|--|--|--|
| q4 (K32) | 1958µs 3.2× | **1622µs 3.9×** ← BEST exact |
| q2c (K2048,r64) | 2092µs 3.0× | 1703µs 3.7× |
| q4c (K32,r64) | 2649µs 2.4× | 2236µs 2.8× |
| q2 (K2048) | 1442µs | 1115µs 5.7× (drift, not exact) |

Host packed-key select kept behind HIPFIRE_ZAYA_LMHEAD_HOSTSELECT=1 for A/B. Net journey
this arc: q4 lm_head 6.3ms→1.62ms (3.9×) all greedy-EXACT — coarse gemv vectorize (EXP-35)
+ GPU top-K (EXP-36). Kernels wave32/atomic (RDNA2/3/4-portable). Doc updated. Default
path (full bf16) UNCHANGED (opt-in env). Realistic decode: lm_head 6.3→1.6ms; body ~15ms
now dominates (~43→~48 tok/s), further gains need body work not lm_head.

### EXP-37 (2026-07-24): GPU top-K sync reduction (3→1 host round-trip) — determinism win, latency neutral

Folded the top-K into ONE host download: single stats buffer [nbins bins | min | max];
min/max writes the tail via LDS-block-reduce (1 atomic/block, NOT fused into the coarse
gemv — that = 262k atomics on 2 addresses = contention), histogram reads lo/hi from the
tail ON-DEVICE (no round-trip), and a sentinel-filled idx + skip-sentinel gather removes
the count download. So 3 syncs → 1. RESULT: q4 1622→~1607µs (~15µs, within noise) BUT
variance tightened to ±5µs (1603/1609/1610). ⟹ the ~140µs above the coarse-gemv floor is
KERNEL-LAUNCH overhead across ~6 small kernels, NOT sync latency (hypothesis disproved).
Coarse gemv already ~87% of achievable BW (~181 GB/s). q4 ≈1.61ms (3.9×) is the practical
floor for this design; deeper needs single-kernel radix-select or fewer coarse bytes
(A→bf16 for q2c/q4c) — but plain q4 wins outright. All modes still byte-identical (q4/q2c/
q4c EXACT). Cleaner+more robust (no count-race). Kept.
