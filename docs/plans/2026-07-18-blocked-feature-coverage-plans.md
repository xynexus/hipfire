# Blocked feature-coverage plans (2026-07-18)

Two feature-coverage gaps hit blockers during the map→fix→validate→push loop and are
too deep for a single loop tick. This doc captures the root cause + a concrete path
forward for each so the work is not lost. See the `/goal` memory
`project_feature_coverage_buildout` and `project_zaya_serving_loader_broken`.

Status of the triggering work: the **zaya qt-36 OQ loader fix is DONE + unit-tested**
(route `oq_repack` through the shared `oq8_arch_load`; net −121 lines) and rides the
already-serving-proven shared helper (gemma3/medgemma qt 36). It is being landed
separately; it is NOT blocked. The two items below ARE blocked.

---

## 1. zaya serving loader ↔ real ZAYA1-8B residual-scale mismatch (BLOCKS all zaya serving)

### Root cause
`crates/hipfire-arch-zaya/src/weights.rs` `ZayaWeights::load_host` (L146-221) was
written against a residual-scale layout that the shipped checkpoint does not use.

- **Loader expects:** model-level `model.input_hidden_states_scale` (L148) + per-layer
  TWO blocks `{prefix}.post_attention_residual_scale.{hidden_states,residual}_{scale,bias}`
  and `{prefix}.post_mlp_residual_scale.*` (via `ZayaResidualScale::load`, L107-113).
- **Real ZAYA1-8B has** (from model.safetensors.index.json):
  - model-level: `model.res_scale.{hidden_states,residual}_{scale,bias}` (4 tensors)
  - per-layer: `model.layers.N.res_scale.{hidden_states,residual}_{scale,bias}`
  - config: `residual_in_fp32: true`, `scale_residual_merge: true`, `num_hidden_layers: 80`.
- So the real model uses ONE **merged** `res_scale` block per layer (+ one at model
  level), not a model-level input scale + two post-op blocks. Load fails at the first
  missing tensor (`model.input_hidden_states_scale`), before any weight, on ANY quant
  format (bf16 included). The zaya serving path has never loaded the real checkpoint;
  only the config-parse test (`parses_zaya1_8b`) exercises real data.

### Plan
1. **Nail the real forward semantics first.** Read Zyphra's ZAYA1 modeling code (HF
   `Zyphra/ZAYA1-8B/*.py` or config `auto_map`) to learn exactly where the single
   `res_scale` applies and what `scale_residual_merge` means (likely: one affine
   `h = hidden_states_scale*h + hidden_states_bias`, `residual = residual_scale*res +
   residual_bias`, merged into a single residual-combine per layer, vs the loader's
   two-op post-attn/post-mlp assumption). Cross-check against `cpu.rs` reference math
   (L50, L67-71) which currently applies `input_hidden_states_scale` then per-op
   `residual_scale(...)` with the two blocks.
2. **Reshape the loader** (`weights.rs`): replace `input_hidden_states_scale` +
   `post_{attention,mlp}_residual_scale` with a single per-layer `res_scale` block
   (`ZayaResidualScale::load(src, "{prefix}.res_scale")`) + a model-level
   `model.res_scale` block. Add `residual_bias` (the real model has it; current
   `ZayaResidualScale` has scale/bias for hidden_states + residual_scale but check
   residual_bias coverage at L99-113).
3. **Reconcile the forward math** in `cpu.rs` and `gpu.rs` to the merged single-scale
   layout so the reference (cpu) and GPU paths agree; keep `residual_in_fp32`.
4. **Validate:** the ZAYA1-8B.oq4.125.hfq artifact already exists
   (scratchpad/, 5.25GB) — once loading works, `hipfire chat` coherence check at
   temperature 0 (base model → factual continuation). Also validates the qt-36 OQ
   path end-to-end on GPU (currently only unit-tested). A bf16 zaya quant is the
   cleaner first load target (isolates the structural fix from quant).

### Risk / unknowns
Medium-large. Requires reading the upstream modeling code (the exact merge semantics
are the crux). Touches loader + both forward paths. Not a rename — an architecture
reconciliation. ~half-day with the reference code in hand.

---

## 2. gemma4 P3: q8 / KVarN KV-quant via the shared `LayeredKvArena` (P3, DEEP)

### Root cause
gemma4 discards `kv_mode` (`crates/hipfire-arch-gemma4/src/arch.rs:290`,
`let _ = options.kv_mode`) and its state builds `LayeredKvArena::new_fp32`
(`forward.rs:54`). The shared `LayeredKvArena`
(`crates/hipfire-runtime/src/layered_kv.rs:344`) is **fp32-only**: only `new_fp32`
(L363, builds `KvCache::new_gpu_capped_filtered`) + `homogeneous_fp32_cache`; its
`store_f32` (L434), `view` (L418), and attend paths all assume plain-f32 `KvCache`
groups. gemma3's WORKING quant KV does NOT go through this — it threads a
`KvQuantMode` enum (Q8/Kvarn) into `Gemma3State` (`gemma3_kv_mode` load.rs:310 →
`Gemma3State::new_with_max_seq` gemma3/forward.rs:103). So there is no existing
quant path in `LayeredKvArena` to simply call.

### Plan
1. **Add a KvQuantMode-aware constructor to `LayeredKvArena`:**
   `new_quant(gpu, plan, mode: KvQuantMode, kvarn_bits: usize)` that builds each group
   via `KvCache::new_gpu_q8_capped` / `new_gpu_kvarn_capped(.., bits)` instead of
   `new_gpu_capped_filtered`. Store the per-group `KvQuantMode` on the arena.
2. **Make store/view/attend quant-aware.** This is the hard part: `store_f32` and the
   attention read must dispatch on the group's storage dtype. **Target the KVarN tier**
   (variance-normalized K + Q8 V, block-record + window ring — mirror the kvarn_attend
   path), NOT plain Q8 KV: Q8 (weight and KV) is being deprecated per the 2026-07-18
   directive (see memory `project_mq_q8_deprecation`), so Q8-KV wiring here would be
   throwaway. KVarN is the harder path but the go-forward one; reuse the qwen3.5/gemma3
   kvarn store/attend kernels and the 082ee3da4 batched-prefill kvarn fix.
3. **Plumb gemma4:** add `gemma4_kv_mode` (mirror `gemma3_kv_mode` load.rs:310), thread
   `options.kv_mode` at arch.rs:290 into `Gemma4DenseState::new` (forward.rs:50) →
   `LayeredKvArena::new_quant`.
4. **Guard the shared change:** gemma3 also constructs `LayeredKvArena` in some paths —
   confirm the fp32 default is untouched and only the new `new_quant` opt-in changes
   behavior. head_dim 256 → later kvarn tier needs no new rotation kernel.
5. **Validate:** gemma4 DOES serve. Use a small on-disk variant (gemma-4-E2B / E4B),
   quantize with an OQ format, `HIPFIRE_KV_MODE=kvarn hipfire chat` coherence vs fp32-KV.

### Risk / unknowns
Medium-large. Shared-infra change (`LayeredKvArena`) that gemma3 may also ride → needs
GPU-validation on both. Targeting KVarN (not the deprecated Q8) raises the difficulty:
the block-record + window-ring layout is the same hazard as the qwen3.5 kvarn
batched-prefill OOB fix (082ee3da4), so port that store/attend carefully into the
layered arena.

---

## Sequencing recommendation
By the user's axis priority (P1 OQ > P2 > P3), the next CHEAP P1 item —
**dots-ocr OQ loader reroute** (hand-rolled panic at
`crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:660/753` → route through shared
`oq8_arch_load`/`oq4_arch_load`, same pattern as zaya) — should land before either of
the above deep items. It has no small model on disk to serving-validate (VL), so land
it unit-tested + unvalidated per the gemma3/gemma4 reroute precedent, document, move on.
Then tackle item 1 (zaya, unblocks a whole family + validates qt-36 end-to-end) or
item 2 (gemma4 Q8-KV) as dedicated non-loop sessions.
