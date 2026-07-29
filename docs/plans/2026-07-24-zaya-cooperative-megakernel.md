# ZAYA decode cooperative megakernel — implementation plan

**Goal:** get `zaya1-8b.oq8++` decode from **52 tok/s → >200 tok/s** on halo / gfx1151.

**Status when this plan was written (2026-07-24):** diagnosis complete (16 experiments,
`docs/perf/zaya-decode-optimization.md`), Q8 lm_head landed (36.76→52.33 tok/s). The
megakernel is the sole remaining lever to reach the target. This is a multi-session build.

---

## 1. Why a megakernel (the confirmed diagnosis)

Decode is **GPU-execution-bound**, and the body runs at only **~47 GB/s effective** vs the
**211 GB/s** the hardware sustains for a large contiguous gemv. Root cause is NOT any single
slow kernel:

- Individual gemvs are fast in isolation (MoE gate_up = 470 GB/s, EXP-16).
- hipGraph gives ~0 (EXP-12/15) → not CPU-submission-bound.
- Forcing max clock gives ~0 (EXP-6) → clock is a *symptom*.

The body issues **~1400 small, data-dependent kernels/token** (~40 ops × 40 blocks). Each
is individually fast but the GPU sits at low utilization between them, so the DRAM
clock/bandwidth never ramps (EXP-11: in-decode ~48 GB/s). The fix is to **keep the GPU
saturated across the whole block** — one cooperative kernel per block, grid-strided gemvs
over all 160 resident workgroups, `grid.sync()` between dependent phases. Fewer launches is
secondary; **sustained utilization → bandwidth ramp** is the primary win.

**Roofline math for the target:**
- Body weight bytes/token @ oq8 ≈ 0.78 GB → at 211 GB/s = **3.7 ms**.
- lm_head: Q8 (current) 0.55 GB ≈ 2.6 ms; **oq4** 0.27 GB ≈ **1.3 ms**.
- Target 200 tok/s = 5.0 ms/token ⟹ **body megakernel (3.7 ms) + oq4 lm_head (1.3 ms)**.
- Megakernel alone (Q8 lm_head): 3.7 + 2.6 = 6.3 ms ≈ **158 tok/s** (milestone).

So there are **two workstreams**: (A) the body megakernel, (B) the oq4 lm_head with LDLQ.
Both are required for >200. A is the hard one.

---

## 2. Validated building blocks (already in tree)

- **Cooperative launch:** `hip_bridge::launch_cooperative_kernel` (ffi.rs) →
  `hipModuleLaunchCooperativeKernel` (symbol verified). Grid-sync validated on gfx1151:
  max cooperative grid = **160 blocks × 256 threads** (8 blk/MP × 20 MP), `cg::this_grid()
  .sync()` correct across multi-phase (EXP-7, scratchpad/coop_test.hip).
- **In-kernel FWHT-256:** `zaya_fwht256_lds` (kernels/src/zaya_cca.hip:593) — wave32 local
  butterfly + `ds_swizzle` wave butterfly + signs. Used by `zaya_router_mlp_fused`.
- **In-kernel Oq8 planar gemv:** pattern in `zaya_moe_{gate_up,down}_oq8_planar_indexed`
  (zaya_cca.hip:744/796) and `zaya_router_mlp_fused` (zaya_cca.hip:622) — 8 int8/lane, two
  int32 loads, per-group f32 scale, shfl reduce. Planar layout `[int8 M*K | f32 M*(K/256)]`.
- **Full-block megakernel template (single-WG):** `zaya_router_mlp_fused` proves the
  multi-phase in-kernel pattern (down_proj→prep→rmsnorm→[FWHT→fc→gelu]×2→FWHT→out→select)
  is coherent. **Lesson (EXP-9): single-WG is a WASH** — it loses gemv multi-WG parallelism.
  The megakernel MUST be cooperative multi-WG with grid-strided gemvs.
- **On-device MoE routing / indexed experts:** `moe_indexed` pointer tables + `sel_idx`/
  `sel_gate` already device-resident (gpu.rs), no host round-trip.
- **Env-gated decode wrapper:** `gpu_decode` / `gpu_decode_body` split, `HIPFIRE_ZAYA_*`
  probes (ABLATE, NBLOCKS, LAUNCHSTATS, GRAPH) — reuse for MEGAKERNEL gating.

---

## 3. Per-block op inventory (what must be fused)

From `gpu_decode_body` (crates/hipfire-arch-zaya/src/gpu.rs ~1640–1985), per block `li`:

| # | op | current kernel(s) | shape | notes |
|---|----|-------------------|-------|-------|
| 1 | input rmsnorm (+rotate) | `fused_rmsnorm_rotate_mq_plain` | h=2048 | feeds qkv |
| 2 | qkv proj | `fused_qkvza_oq8_gemv` | q 1024, k 256, vcur/vdel 128 | 4 gemvs, 1 launch |
| 3 | qk prep + conv window | `zaya_qk_prep_decode_f32`, `zaya_conv_window_f32` | conv_ch=1280 | ring state |
| 4 | conv depthwise + grouped | `zaya_conv1d_valid_f32` ×2 | dw[1280,1,2], gr[1280,128,2] | small |
| 5 | add-conv-residual q+k | `zaya_add_conv_residual_qk_f32` | | |
| 6 | value assemble | `zaya_value_assemble_decode_f32` | v_half=128 | delayed-v state |
| 7 | q+k l2norm+scale | `zaya_qk_l2norm_qk_f32` | | key per-head temp |
| 8 | q+k partial-RoPE | `zaya_rope_partial_qk_posbuf_f32` | n_rot=64 | device pos_buf |
| 9 | KV write + flash attn | `kv_cache_write` ×2, `attention_f32` | nq=8 nkv=2 hd=128 | grows w/ ctx |
| 10 | o_proj | `LinearWeight::gemv` | 2048←1024 | |
| 11 | affine residual (attn) | `zaya_affine_residual_f32` | h | pa_rs scales |
| 12 | post-attn rmsnorm+rotate | `fused_rmsnorm_rotate_mq_plain` | h | feeds down+gate_up |
| 13 | router MLP + select | `zaya_router_mlp_fused` (already fused!) | rh=256, n_route=17 | → sel_idx, sel_gate |
| 14 | MoE gate_up (indexed) | `zaya_moe_gate_up_oq8_planar_indexed` | M=4096 K=2048 | data-dep expert |
| 15 | silu_mul + rotate | `fused_silu_mul_rotate_mq` | moe_int=2048 | FWHT rotate |
| 16 | MoE down (indexed) + combine | `zaya_moe_down_oq8_planar_indexed`, `moe_down_combine` | M=2048 K=2048 | += sel_gate·down |
| 17 | affine residual (mlp) | `zaya_affine_residual_f32` | h | pm_rs scales |

~17 logical stages, ~30–40 kernel launches/block after existing fusion.

---

## 4. Design

### 4.1 Granularity: per-block cooperative kernel (first target)

Launch **one cooperative kernel per block** (40 launches/token) rather than one monolithic
kernel for all 40 blocks. Rationale: per-block keeps KV/conv/delayed-v state handling and
the data-dependent expert selection simple, and still collapses ~1600 → ~40–120 launches.
A whole-decode single-launch kernel (loop over 40 blocks inside) is a **later** optimization
once the per-block kernel is correct and fast.

### 4.2 Grid / occupancy

- Grid = **160 blocks × 256 threads** (max cooperative residency). All gemvs grid-stride
  their output rows over the 160×(256/32)=1280 waves.
- Scratch (normed, q/k/v, ctx, gate/up, act, down, moe_out) lives in **global** device
  buffers (LDS cannot be shared across cooperative blocks). Pre-allocate once on the
  `ZayaDecodeState` (persistent addresses — required and already done for graph capture).
- `grid.sync()` between every dependent stage (~13–15 syncs/block).

### 4.3 The attention seam

Flash attention (stage 9) is the one stage that is awkward inside a grid-strided cooperative
kernel (per-head online softmax over growing KV, complex reductions). **Phase 1 keeps
attention as a separate launch:** megakernel-A does stages 1–8, then the existing
`attention_f32` launches, then megakernel-B does stages 10–17. That's **3 launches/block =
120/token** (vs 1600) — already most of the utilization win. Folding attention into the
cooperative kernel (heads grid-strided over block groups) is a Phase-3 refinement.

### 4.4 Correctness anchors

- Every Oq8 gemv MUST rotate its activation into the FWHT-256 basis first (weights live
  rotated). Reuse `zaya_fwht256_lds`. Getting the rotate wrong = garbage (this is the #1
  risk — cf. the AWQ/rotate bugs in the journal).
- The MoE down_proj activation must be FWHT-rotated after silu_mul (stage 15), exactly as
  `fused_silu_mul_rotate_mq` does.
- Router uses prob+balancing_bias argmax with the 17th null (MoD skip) slot → gate 0.
- Residuals use the per-block affine scales (`pa_rs`, `pm_rs`, in/out scales).

---

## 5. Phased implementation with validation gates

Each phase is **env-gated** (`HIPFIRE_ZAYA_MEGAKERNEL=<phase>`) and validated by
**per-token bit/cosine diff vs the current path** before moving on. The working 52 tok/s
path is never removed until the megakernel beats it end-to-end and passes coherence.

- **Phase 0 — scaffold.** Add `zaya_decode_megakernel_b` cooperative kernel (stages 10–17
  only: post-attn rmsnorm+rotate → router → MoE → residual), launched via
  `launch_cooperative_kernel`. Pre-allocate global scratch on `ZayaDecodeState`. Validate
  MoE-block output `hidden` matches the reference path (cosine > 0.9999) on 1 block.
  *Gate: numerically identical MLP half.*

- **Phase 1 — front half.** Add megakernel-A (stages 1–8: rmsnorm+qkv → conv → qk-prep →
  l2norm → rope). Keep `attention_f32` + o_proj as separate launches between A and B.
  Validate q/k/value and post-o_proj `hidden` vs reference. *Gate: full decode coherent
  (Fibonacci/Paris temp0), 3 launches/block.* **Measure tok/s** — expect first real jump
  from utilization ramp.

- **Phase 2 — o_proj + attn glue into B/A.** Move o_proj and the two affine residuals into
  the cooperative kernels (o_proj at head of B, attn-residual after attention). *Gate:
  coherent; measure.*

- **Phase 3 — fold attention in.** Grid-stride the 8 q-heads over block groups inside a
  single cooperative kernel so a block becomes **1 launch** (stages 1–17). Requires
  device-pos KV append + online-softmax reduction across the block's threads. *Gate:
  context-flat coherence to ctx 2000; measure — target body ≈ 3.7 ms.*

- **Phase 4 — whole-decode single launch (optional).** Loop 40 blocks inside one
  cooperative kernel (weights indexed by layer), 1 launch/token. Only if Phase 3 hasn't
  already hit the target.

**Parallel workstream B — oq4 lm_head (needed for >200, independent of the megakernel):**
- The calib already has `model.embed_tokens.hessian` [2048,2048].
- The `--embed-precision` path is RTN-only today (main.rs:3728) → add an LDLQ branch that
  quantizes the lm_head to oq4 using that Hessian (GPTQ/OBS error feedback).
- **Untie** for quality: keep a cheap high-precision input-embed for the gather; use the
  oq4 LDLQ copy only for the output projection (the 1.07 GB read). Gate on KLD vs bf16.
- Expected: lm_head 2.6 → 1.3 ms.

---

## 6. Risks & mitigations

| risk | mitigation |
|------|-----------|
| FWHT/rotate basis mismatch → garbage | reuse `zaya_fwht256_lds` verbatim; diff each Oq8 gemv output vs reference before wiring |
| grid.sync deadlock / occupancy < 160 | query `hipOccupancyMaxActiveBlocksPerMultiprocessor`; cap grid to actual residency; coop launch fails loudly if grid too big |
| data-dependent expert ptr inside coop kernel | `sel_idx`/`sel_gate` computed in the router stage (13) *before* the grid.sync into gate_up (14); pointer table already device-resident |
| breaks working decode | strictly env-gated; reference path stays until megakernel wins + passes coherence |
| attention complexity | deferred to Phase 3; separate launch until then |
| non-linear payoff (partial shows no gain) | expect the first measurable jump only at Phase 1 (front half fused); don't judge Phase 0 on tok/s, judge on correctness |

## 7. Success criteria

- **Correctness:** temp0 coherence (Paris, Fibonacci) + context-flat to ctx≥2000; per-stage
  cosine > 0.9999 vs reference during bring-up.
- **Milestone:** Phase 1 shows a measurable tok/s jump (utilization ramp) — proves thesis.
- **Sub-target:** megakernel + Q8 lm_head ≈ 158 tok/s.
- **Target:** megakernel + oq4 lm_head **> 200 tok/s @ oq8++**.

## 8. Pointers

- Journal / evidence: `docs/perf/zaya-decode-optimization.md` (EXP-1…16).
- Decode body: `crates/hipfire-arch-zaya/src/gpu.rs` `gpu_decode_body`.
- Kernels: `kernels/src/zaya_cca.hip` (fwht, router, MoE), dispatch in
  `crates/hipfire-rdna/src/dispatch/zaya_cca.rs`.
- Cooperative launch: `crates/hip-bridge/src/ffi.rs` `launch_cooperative_kernel`.
- References (technique-only, CDNA/asm): `~/build/amd/{composable_kernel,aotriton,aiter}`.
