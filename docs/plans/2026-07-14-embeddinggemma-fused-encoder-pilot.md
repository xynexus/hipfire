# EmbeddingGemma fused-encoder pilot — scoping + plan

Date: 2026-07-14. Follows the top-down NPU investigation
(`benchmarks/npu_gemm_tuning/iron_ctx_probe/`) that fixed the mlir-aie
`--generate-full-elf` bug (Xilinx/mlir-aie#3337) and got the reference IRON
Llama-3.2-1B **fused decode to 8 tok/s** (1 dispatch/token). Goal: decide whether
the same compiler-managed fused-graph approach beats hipfire's hand-authored
amdxdna-direct resident EmbeddingGemma encode (currently ~336 tok/s at M256, the
subject of NPU-RESULTS R57–R121).

## The premise had to be reframed

The obvious pitch — "fuse per-layer to cut dispatches" — is **already done**.
`NpuOpusProjector::project_layer` runs a *resident context* that fuses attention
(`NpuEmbeddingLayerAttentionDenseW8`) + FFN + tail into one per-layer NPU unit.
hipfire's resident path is not dispatch-count-bound within a layer.

The real overhead is **cross-layer**, documented in NPU-RESULTS:

- host readback / normalization / output-deblocking ≈ **10–12 ms/layer** (R100);
- distinct next-layer / residual preparation contexts ≈ **7–9 ms/layer**
  (R104/R108/R109, down from 9–12);
- context-alternation tax ≈ **27%** of wall time (R91).

At M256 the layer compute itself is real (256×768 GEMMs, 256×256 attention) — this
is **compute-bound, not dispatch-bound** (unlike the M1 Llama decode). So the win
is bounded by how much of the 749 ms (24 layers) is *overhead* vs *compute*.
Rough envelope: (10–12 + 7–9) ms/layer × 24 ≈ **400–500 ms of the 749 ms** is
cross-layer overhead. Removing most of it → ~300–450 ms → **~570–850 tok/s, i.e.
~1.7–2.5×**. A meaningful but not order-of-magnitude win.

## Why the resident path pays that overhead — and why IRON might not

hipfire's amdxdna-direct path was *forced* into per-layer host-bridged contexts by
the exact limits R59–R121 fought: the **5-data-argument DPU regmap**, the **16 KiB
core-text program store**, and cross-context state handoff (PRIME `EINVAL`, etc.).
That is why it cannot keep 24 layers resident in one context and round-trips
through the host between layers.

The IRON `--generate-full-elf` mechanism is *different*: it stitches per-op
sub-ELFs together via a **fused `main` device** whose DMA orchestration addresses
one flat byte **arena** (the `memref.view` slices that PR #3337 fixes). That arena
is how cross-op/cross-layer state stays on-device without per-context host bridges
or the 5-arg limit. The Llama fused decode **proves the mechanism carries 24 layers
in one dispatch** (at M1). The open question is purely: **does it hold at M256**
(bigger tiles, same program), giving a whole-encoder ELF that eliminates the
per-layer host bridges hipfire cannot avoid?

That is the pilot's real hypothesis:
> The fixed IRON full-ELF can fuse the whole EmbeddingGemma encoder into one
> on-device dispatch where hipfire's direct path was forced into 24 host-bridged
> per-layer contexts — removing the ~400–500 ms cross-layer overhead.

## EmbeddingGemma-300M shapes (for the IRON graph)

hidden 768 · 24 layers · 3 Q-heads + 1 KV-head × head_dim 256 (QKV = 1280, matches
R-log N1280) · FFN 1152 (GeGLU) · attention 5 sliding : 1 global (window pattern 6)
· **encoder: bidirectional, no KV cache, no causal mask** · M256 · mean-pool tail.
Note: encode is *simpler* than the Llama decode it's modeled on — no KV-cache ELF
patching — which removes the app's most fragile piece.

## Build plan

1. **Baseline** — run hipfire's resident encode with `HIPFIRE_EMBED_TRACE_RESIDENT=1`
   to get the real per-layer overhead vs compute split (confirm the envelope above).
2. **Rung A — one fused layer at EmbeddingGemma dims.** Author a single bidirectional
   Gemma3 layer (RMSNorm ×2, QKV 768→1280, QK-norm, no-RoPE-for-global/RoPE-for-
   sliding, softmax attention, O-proj, GeGLU FFN 768→1152→768) as an IRON graph at
   M256, compile with the fixed `build/bin/aiecc`, dispatch, verify numerics vs the
   GPU reference, and measure one-dispatch wall time. Reuse the Llama app's GEMM /
   RMSNorm / softmax / SiLU operators; add the bidirectional attention + GeGLU.
3. **Rung B — stack to whole-encoder fused ELF.** Chain 24 layers through the arena
   (the full-ELF path), add the pooling tail, and measure end-to-end M256 encode
   tok/s vs the 336 baseline. This is the go/no-go on the hypothesis.
4. **Decide.** If ≥1.7× and numerically faithful, it's a real alternative to the
   resident path and worth productionizing a hipfire↔IRON bridge (author graph in
   IRON, compile offline, dispatch via amdxdna or XRT). If it hits the M256 program
   wall, that's a clean negative that validates the hand-authored per-layer design.

## Risks

- ~~M256 whole-encoder ELF may exceed program store~~ — **largely retired.** The
  fused ELF uses **temporal streaming, not spatial packing**: the 24-layer Llama
  decode fuses into ONE ELF with only ~13 distinct operator sub-devices (RMSNorm,
  GEMV, Transpose, …), each core's program **2.5–6.3 KB — well under the 16 KiB
  limit**. Layers are DMA-schedule iterations over a fixed operator set; the
  program does not grow with depth. hipfire hit 16 KiB because it *spatially packs*
  attention+FFN+tail into each core; IRON runs one small op/core and streams. The
  real M256 gate is therefore the **DMA / static BD-ID schedule** (R119/R120
  regime), mitigable with `repeat_count`/outer-tiling DMA, BD reuse across layers,
  and finer core tiling; per-layer fused ELFs are the graceful fallback.
- Numerical fidelity of the IRON operators at Gemma3 specifics (query pre-attn
  scalar, QK-norm, GeGLU, dual RMSNorm) must match the GPU reference before any
  perf number counts.
- Even at 2×, this is a *second* NPU execution path; productionizing it is a real
  cost that only pays off if the win is robust across shapes.

## Progress

- **Rung 0 (verification foundation) — DONE.** `benchmarks/npu_gemm_tuning/embeddinggemma_pilot/`:
  `dump_reference.py` captures HF's exact layer-0 I/O + per-submodule intermediates
  (full 256-token no-pad input → pure bidirectional attention). `numpy_reference.py`
  reimplements the layer per the spec and **verifies against HF stage-by-stage**
  (input_layernorm / q,k,v_proj / self_attn / mlp / final: rel error 1e-6–1e-7).
  This numpy layer is the executable oracle the IRON graph must match.
  Confirmed math: RMSNorm `rsqrt(mean(x²)+1e-6)·(1+w)`; QK-norm per-head(256) then
  RoPE rotate-half base 1e4 (layer 0 local); GQA rep 3; softmax scale 0.0625;
  o_proj; sandwich residual; GeGLU tanh (0.7978845608, 0.044715); dual norms.
- **Next — Rung A (IRON graph).** Author the layer as an IRON graph at M256, one
  operator at a time verified against `numpy_reference` intermediates, compile with
  the fixed `build/bin/aiecc`, dispatch, then fuse to one ELF. Then Rung B stacks 24.
- **Rung A de-risked — it's composition, not authoring.** The vendored IRON
  operator library (`third_party/IRON/iron/operators/`) already provides every
  building block the layer needs: `rms_norm` (weighted), `gemm`, `gemv`, `rope`,
  `mha`/`softmax`, `gelu`, `elementwise_add`/`elementwise_mul`, `swiglu`. So Rung A
  is: instantiate each at EmbeddingGemma dims → verify each vs `numpy_reference`
  intermediates on the NPU (single-op xclbin, no full-ELF needed) → compose into a
  fused layer via `FusedMLIROperator` (the same runlist path the Llama decode uses)
  → compile with the fixed `build/bin/aiecc`. Gemma specifics map onto library ops:
  RMSNorm weight passed as `(1+w)`; QK-norm = rms_norm at tile 256; GeGLU = gelu +
  elementwise_mul; separate q/k/v = three gemm instances.
  Order of work: (1) rms_norm@[256,768] vs oracle input_layernorm; (2) the 3 QKV
  gemms; (3) qk-norm + rope + mha attention block; (4) o_proj + residual;
  (5) GeGLU FFN + residual; (6) fuse the 6 into one layer ELF; (7) Rung B: stack 24.
- **Rung A step 1 — DONE on hardware.** `step1_rmsnorm.py` (via `run_step1.sh`):
  IRON weighted `RMSNorm(size=256*768, tile_size=768, weighted=True)` compiled and
  dispatched on the aie2p NPU, verified vs the HF oracle `input_layernorm` at
  **rel 1.1e-2** (bf16 tolerance — per-element ~4e-3 over the 768 reduction). Proves
  the operator-reuse → compile → NPU dispatch → verify loop at EmbeddingGemma dims.
  Weight fed as `(1+w)`; kernel eps 1e-5 negligible (post-embed-scale mean(x²)≈700).
  Next: steps 2–5 (QKV gemms, QK-norm+RoPE+mha, o_proj+residual, GeGLU FFN) same
  pattern; step 6 fuse; Rung B stack 24 + check final embedding cosine vs f32.
- **Rung A steps 2–3 (2026-07-14).** Step 2 ✅: the three QKV GEMMs on NPU vs HF
  (q 8.2e-3 / k 9.0e-3 / v 6.8e-3, bf16 accum tol) — `b_col_maj=True` feeds HF
  `[out,in]` directly; `tile_n=64 * 4 cols` → min_N=256 divides 768 & 256. Import
  order aie/iron-before-torch avoids the duelling-LLVM segfault.
  Step 3 **redirected**: the IRON `MHA` operator is **head_dim=64 only** (Embedding-
  Gemma head_dim=256), so attention is composed from verified primitives instead —
  per q-head: scores = GEMM(Q, K, b_col_maj=True)·(1/√256) → `softmax` op → context
  = GEMM(P, V). GQA rep 3 (3 q-heads share the 1 kv-head), bidirectional (no mask),
  then o_proj GEMM + residual (step 4). This matches how the Llama fused decode does
  attention (GEMV+softmax+GEMV), so it composes cleanly into the fused ELF.
- **Rung A COMPLETE — full layer on NPU matches HF (2026-07-14).** `full_layer.py`
  runs the whole Gemma3 encoder layer with heavy ops on the aie2p NPU (RMSNorm ×4,
  q/k/v/o/gate/up/down GEMMs, GQA softmax attention, GeGLU = gelu[tanh]·mul) and
  glue (residuals, qk-norm, RoPE, reshape) in numpy; verified vs the HF oracle's
  final layer-0 output at **rel 1.06e-2, cos 0.99993** (every intermediate cos
  >0.999). Numeric fidelity of the composed operators is proven on hardware. Bug
  caught+fixed: down_proj is a weight [out,in] so `b_col_maj=True` (not False).
  All Gemma3 specifics land correctly (query scale 0.0625 folded into Q; per-head
  qk-norm; RoPE base 1e4; gelu tanh-approx; sandwich residuals).
- **Next — step 6 (fuse) + Rung B (stack 24).** Compose the layer's ops into ONE
  ELF via FusedMLIROperator + the fixed `build/bin/aiecc` (tests the M256 DMA/BD
  schedule, the remaining feasibility gate), move the numpy glue on-device, stack
  24 layers + pooling tail, and measure end-to-end M256 tok/s vs the 336 baseline
  plus final-embedding cosine vs f32.
