# DFlash Phase D — step 2: fuse the block body (dispatch reduction)

Baseline: the UNFUSED body (`tools/npu/dflash_body_npu.py`, Gate D step 1) is
numerically correct on nix1's NPU — full-body cos 0.99902 vs f16 golden — but at
**88 logical / 2048 raw dispatches, ~23 s wall**. Gate D wants **≤ ~3
dispatches/layer (~15/block)** and block wall-time **< 57 ms** (the 9B verify
budget). This step cuts dispatches while holding full-body cos > 0.99.

## Where the dispatches go (from the step-1 run)

- **Per-group int8 projection = the dominant driver.** Each projection runs one
  `matmul_npu` per 256-group: q/k/v/o over K=4096 = 16 groups, gate/up over K=4096
  = 16, down over K=12288 = 48, fc over K=20480 = 80. ×5 layers → most of the 2048.
- **Per-row norm/rope/swiglu.** The rmsnorm/headnorm/rope/swiglu xclbins are invoked
  per row (16 block rows, or per-head) in the host loop.

## Fusion levers, highest-value first

### Lever 1 — batch the per-row ops into one dispatch each (likely FREE)
The `_transform_gen` primitive kernels tile over rows (`N_div_n = rows`), so feeding
the whole `[16, H]` (or `[16*heads, head_dim]`) buffer runs all rows in ONE dispatch
instead of 16. Check `build_qwen35_rmsnorm.py` / headnorm / rope / swiglu tiling and
the run wrapper: pass the batched buffer, confirm one dispatch, re-validate parity.
This alone removes most of the norm/rope/swiglu dispatch multiplier with no new
kernel.

### Lever 2 — collapse the per-group projection to ONE dispatch/projection
Two options; measure both:
- **2a (simple, pragmatic): full-K per-ROW int8 matmul.** One `matmul_npu` over the
  whole K (single scale per output row / per activation row), rescale rank-1
  `C·(a_scale[m]⊗w_scale[n])`. 1 dispatch/projection. Costs ~2–4 dB vs per-group
  (measured earlier: per-row W8A8 ≈ 26 dB vs 30 dB per-group) — but the drafter is
  VERIFIED downstream, so check whether full-body cos stays > 0.99. If yes, ship it
  (biggest dispatch win for least effort).
- **2b (if 2a drops parity below gate): in-core per-group scale epilogue.** A custom
  @iron.jit int8 GEMM that streams all K groups, applies `w_scale[g]·a_scale[g]` to
  each group's int32 partial, accumulates in f32, one dispatch. This is the
  `opus_lowbit::dot_offset_fold` structure in-core. More work; keeps per-group
  quality. Reuse the `oq_gemm_design` int8 mmul; add the scale+accumulate epilogue.

### Lever 3 — stage fusion (toward ~3 dispatches/layer)
Per the plan: fuse **stage 1** (rmsnorm → qkv proj → q/k-norm → rope) and **stage 3**
(o-proj → resid → rmsnorm → gate/up → silu → down → resid) into one dispatch each;
attention stays its own dispatch (different dataflow). This is the asymptote and the
hardest — only pursue after levers 1–2 land and if the dispatch/time budget still
isn't met. A fused stage keeps the `[16,H]` activation resident across its ops.

### Lever 4 — host-overhead (for the 57 ms wall, not dispatch count)
23 s is dominated by Python + per-dispatch XRT/context latency, NOT NPU compute. Even
at ~15 dispatches/block the wall won't hit 57 ms if each dispatch pays ms of host
round-trip. To approach 57 ms the activations must stay NPU-resident across ops
(that's what stage fusion buys) and the driver loop must be tight (reuse loaded
xclbins/contexts via the shared `CachedXRTRuntime`, avoid re-quantising/re-uploading
weights each step — upload int8 weights ONCE). Separate "dispatch count" (Gate D
structural target) from "wall < 57 ms" (needs residency + a lean driver); report both.

## Method

1. Instrument `dflash_body_npu.py` to print logical + raw dispatch count and wall.
2. Land lever 1, re-validate (cos > 0.99), record new counts.
3. Land lever 2a, measure full-body cos; if < 0.99 fall back to 2b.
4. Re-measure dispatches/layer + wall. If > 3/layer or ≫ 57 ms, do lever 3 stage
   fusion for the worst layer stage, then generalize.
5. Update `docs/npu/dflash-drafter-npu-plan.md` Gate D with final counts + wall +
   parity. Keep the bf16/int8-precision reference as the honest gate (no loosening).

## Guardrails

- Validate against the Phase-A golden + the int8/bf16-precision numpy reference after
  EVERY lever (regression check). Reuse the golden dump + `run_body.sh` from step 1
  (scratchpad), or regenerate via `dflash_ref_dump` under the GPU lock.
- Weights uploaded/quantised ONCE (not per dispatch) — a correctness-neutral speed win.
- Env: fork `~/mlir-aie-312/venv312`; all NPU loads through the shared
  `CachedXRTRuntime` (`aie.utils._get_default_npu_runtime()`) — mixing private
  XRTHostRuntime with iron.jit exhausts Phoenix hw-contexts (err=-22).
