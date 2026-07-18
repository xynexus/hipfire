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

### ⚠️ SCOPE CORRECTION (2026-07-18, after checking the checkpoint)
The mismatch is NOT limited to residual scaling — the ENTIRE zaya crate targets a
phantom/preview layout. Real ZAYA1-8B = 80 layers ALTERNATING half-layers: even idx =
attention (`model.layers.2k.self_attn`), odd idx = MoE (`model.layers.2k+1.zaya_block`
with `router` + `experts.local_experts.{E}.linear_fc1/linear_fc2`), one `input_norm` +
one `res_scale` each. The crate instead reads 40 COMBINED blocks with
`self_attn`+`mlp.experts`(sliced 3D)+two layernorms+two residual-scale blocks per index.
⇒ Fixing zaya serving is a **full loader + forward BRING-UP against the real checkpoint**
(needs the zaya1_vl modeling code studied for CCA attention / EDA router / zaya_block MoE
shapes), a LARGE dedicated task — not the "mechanical residual-scale rewrite" the steps
below assume. The residual-scale detail below is CORRECT but is only one of ~4 layers of
mismatch. Deprioritize vs tractable serveable work (gemma4 KVarN). The landed qt-36 OQ
fix is unaffected.

### Root cause (residual-scale layer — one of several)
`crates/hipfire-arch-zaya/src/weights.rs` `ZayaWeights::load_host` (L146-221) was
written against a residual-scale layout that the shipped checkpoint does not use.

- **Loader expects:** model-level `model.input_hidden_states_scale` (L148) + per-layer
  TWO blocks `{prefix}.post_attention_residual_scale.{hidden_states,residual}_{scale,bias}`
  and `{prefix}.post_mlp_residual_scale.*` (via `ZayaResidualScale::load`, L107-113).
- **Real ZAYA1-8B has** (CONFIRMED via Zyphra/transformers `zaya1-vl` modeling code +
  model.safetensors.index.json inventory):
  - per-half-layer `model.layers.N.res_scale.*` (N in 0..80): `hidden_states_scale`
    + `hidden_states_bias` ALWAYS (80 each); `residual_scale` + `residual_bias` only
    if not first layer (79 each — layer 0 omits them).
  - model-level `model.res_scale.{hidden_states,residual}_{scale,bias}` (1 each).
  - config: `residual_in_fp32: true`, `scale_residual_merge: true`,
    `num_hidden_layers: 80`, hidden 2048.
- **Confirmed semantics** (upstream `ResidualScaling` module): ZAYA alternates ATT and
  MLP *half-layers* (matches the zaya crate's `half_layer_roles_alternate` test and the
  `blocks=40` load log → 40 att + 40 mlp = 80). Each half-layer applies its `res_scale`
  ONCE: `hidden_states = (hidden_states + hidden_states_bias) * hidden_states_scale`,
  and (layer>0) `residual = (residual + residual_bias) * residual_scale`, gated on
  `scale_residual_merge`. So per full transformer layer the scaling happens TWICE (once
  in the att half, once in the mlp half) — the loader's `post_attention_residual_scale`
  + `post_mlp_residual_scale` intuition was directionally right about "twice per layer"
  but modeled it as two blocks on ONE layer index instead of one block on each of two
  half-layer indices, and it invented a nonexistent `model.input_hidden_states_scale`.
- Load fails at the first missing tensor (`model.input_hidden_states_scale`), before any
  weight, on ANY quant format (bf16 included). The zaya serving path has never loaded
  the real checkpoint; only the config-parse test (`parses_zaya1_8b`) exercises real data.

### Plan (semantics now RESOLVED — see above; implementation is mechanical-ish)
1. **Reshape the loader** (`weights.rs`): per half-layer, load ONE `res_scale` block via
   `ZayaResidualScale::load(src, "{prefix}.res_scale")` (currently at L107-113); drop the
   `post_attention_residual_scale`/`post_mlp_residual_scale` two-block model (L124-125,
   L208-215) and the model-level `input_hidden_states_scale` (L148). Make
   `residual_{scale,bias}` OPTIONAL (layer 0 omits them → `Option<Vec<f32>>` or default
   to identity scale=1/bias=0). Add a model-level `model.res_scale` block.
2. **Reconcile the forward math** in `cpu.rs` (L50, L67-71) and `gpu.rs`: each half-layer
   applies its single res_scale ONCE — `h = (h + hidden_states_bias) * hidden_states_scale`
   and (layer>0) `residual = (residual + residual_bias) * residual_scale`, gated on
   `scale_residual_merge`. Remove the `input_hidden_states_scale` pre-scale. Keep
   `residual_in_fp32`. The half-layer ATT/MLP alternation already exists (tests pass), so
   this is a residual-scale rewrite, not a layer-structure change.
3. **Validate:** the ZAYA1-8B.oq4.125.hfq artifact already exists
   (scratchpad/, 5.25GB) — once loading works, `hipfire chat` coherence check at
   temperature 0 (base model → factual continuation). Also validates the qt-36 OQ
   path end-to-end on GPU (currently only unit-tested). A bf16 zaya quant is the
   cleaner first load target (isolates the structural fix from quant).

### Risk / unknowns
Medium (downgraded — semantics now RESOLVED above from upstream code + checkpoint
inventory). Touches the loader + cpu/gpu forward residual-scale math; the half-layer
structure is already correct. Main remaining care: get the model-level `model.res_scale`
application point right (upstream applies it within the layer loop, not at the model
boundary — verify against a bf16 load before trusting) and keep cpu/gpu in agreement.
~half-day; validatable on the existing artifact.

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

### ✅ kvarn head_dim-512 kernel DONE + GPU-validated (2026-07-18)
The head_dim-512 blocker below is RESOLVED. The kvarn kernels were almost entirely
head_dim-generic already (strided `dpt = head_dim/32` loops); the only 256 caps were
LDS/register sizing: `kvarn_quantize_tile` LDS `KVARN_RMAX` 256→512; and the flash
tile + asym reduce kernels' per-thread register arrays (`mq/sa/za/out_vec` and the
reduce's dim split), which were templated on a `MAXDPT`/`DPT` parameter — the default
entry points stay MAXDPT=8 (BYTE-IDENTICAL for gemma3/qwen35 @256) and new `_hd512`
entry points use 16, selected in `attention_flash_kvarn_batched_masked` when
head_dim>256. Asserts relaxed to allow 512 on the two kvarn ctors + quantize dispatch.
`kvarn_gather_k_tiles`/`kvarn_dequant_tile` were already generic. Rotation stays
256-only (FWHT-256; gemma4 @512 runs kvarn unrotated). VALIDATED: extended
`parity_kvarn_fused_flash` to sweep head_dim {256,512} vs an f64 host oracle — 256
still matches (regression guard for the template refactor) AND 512 matches at ~1e-5
(same precision). So gemma4 KVarN is now kernel-unblocked; the remaining gap is only
the separate gemma4-dense-loader PLE issue (E2B/E4B don't load) for full serve-validation.

### ⚠️ (historical) BUILT + BLOCKED: gemma4 global layers are head_dim 512
The full KVarN wiring was implemented and compiles + unit-tests clean:
`LayeredKvArena::new_kvarn` + `LayerCacheView` kvarn fields (layered_kv.rs), a
`new_with_kv_mode` state ctor + KVarN branch in the Full attention path + kvarn scratch
(gemma4/forward.rs), and `gemma4_kv_mode` plumb (gemma4/arch.rs). BUT it is INERT on
shipped gemma4: gemma4's **Full/global attention layers use `global_head_dim` = 512**
(config.rs:294-303; only the SlidingWindow/local layers are 256), and the KVarN kernels
HARD-CAP at head_dim ≤ 256 (`kvarn_quantize_tile: r,c must be <= 256`, kv.rs:1397; FWHT
rotation is 256; block/window layout sized ≤256). So `new_kvarn` safely falls back to
F32 for the 512 global groups → KVarN is a no-op on gemma4. The F32 path is unchanged
by construction (the head_dim filter yields 0 kvarn groups → `new_fp32`, no scratch), so
the refactor is a safe no-op, not a regression.
Second blocker (validation): no small DENSE gemma4 on disk. gemma-4-E2B/E4B use
Per-Layer-Embeddings / shared-tail KV → the gemma4 **dense** loader rejects them
("requires no PLE, KV sharing, or routed experts"); only plain-dense gemma4 (31B) loads,
too heavy for a loop tick. So the refactor could only be checked by construction + unit
tests (fp32 path: head_dim filter → 0 kvarn groups → `new_fp32`, no scratch, unchanged)
— a live gemma4 serve was not possible here.

**To actually deliver gemma4 KVarN needs a kvarn head_dim-512 kernel** (tile 512→2×256 in
`kvarn_quantize_tile`/`kvarn_build_kcache`/flash, extend the window ring + FWHT) — a real
kernel task, NOT config. That is the true remaining blocker; the arena/forward/plumb
above is ready and activates the moment kvarn supports 512 (or for any head_dim-256
layered model).

### Feasibility CONFIRMED (2026-07-18) — de-risked, turnkey (superseded by the block above)
- **gemma3 is NOT at risk.** gemma3's working KVarN uses plain `KvCache::new_gpu_kvarn_filtered`
  + separate `swa_k/swa_v` F32 rings (forward.rs:142-159), NOT `LayeredKvArena`. gemma3 only
  touches the arena via `homogeneous_fp32_cache` (returns a plain `KvCache`). So adding a quant
  path to `LayeredKvArena` is **gemma4-only** — no gemma3 regression surface.
- **Kernels + KvCache ctors already exist:** `KvCache::new_gpu_kvarn_capped_filtered(gpu,
  is_kv_layer, n_kv_heads, head_dim, max_seq, physical_cap, bits)` (kv.rs:1337) is the exact
  analog of the `new_gpu_capped_filtered` that `new_fp32` (layered_kv.rs:363) already calls per
  group; plus `kvarn_attend` / `attention_flash_kvarn_batched_masked` for the read side.
- **gemma4 has MIXED storage** (Full global + SlidingWindow local, forward.rs:70-78) — apply
  KVarN to the **Full-storage groups only** (head_dim 256 ∈ {128,256} ✓), keep SlidingWindow
  groups F32 (mirrors gemma3's local-rings-stay-F32 choice).
- ⚠️ **ATOMIC change:** ctor + store-routing + forward attend must land together; a partial is
  dead/broken code (can't validate). Do it as ONE focused unit, not a tick fragment.

### Plan (turnkey)
1. **Add `LayeredKvArena::new_kvarn(gpu, plan, bits)`** mirroring `new_fp32` (L363) but, per
   group, dispatch on `group.storage`: `KvStorageKind::Full` + head_dim∈{128,256} →
   `KvCache::new_gpu_kvarn_capped_filtered(gpu, &owned, group.kv_heads, group.head_dim,
   plan.max_seq, group.storage.physical_cap(plan.max_seq), bits)`; else (SlidingWindow /
   incompatible) → the existing `new_gpu_capped_filtered` (F32). Tag each group's mode so
   store/view/attend can branch.
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
4. **Guard:** keep `new_fp32` the default path (gemma4 fp32-KV unchanged when kv_mode is
   unset). gemma3 confirmed NOT on the arena quant path, so no gemma3 regression surface;
   head_dim 256 → no new rotation kernel needed.
5. **Validate:** gemma4 DOES serve. Use a small on-disk variant (gemma-4-E2B / E4B),
   quantize with an OQ format, `HIPFIRE_KV_MODE=kvarn hipfire chat` coherence vs fp32-KV.

### Risk / unknowns
Medium-large. Shared-infra change (`LayeredKvArena`) that gemma3 may also ride → needs
GPU-validation on both. Targeting KVarN (not the deprecated Q8) raises the difficulty:
the block-record + window-ring layout is the same hazard as the qwen3.5 kvarn
batched-prefill OOB fix (082ee3da4), so port that store/attend carefully into the
layered arena.

---

## Sequencing recommendation (revised 2026-07-18)
Cheap loader-reroute wins are exhausted. Ranking the remaining deep work by
tractability × value:
1. **gemma4 KVarN KV (item 2) — DO FIRST.** gemma4 loads + serves today, so it is
   validatable; the change is bounded to the shared `LayeredKvArena` + reusing existing
   kvarn kernels. Highest tractability among the deep items.
2. **zaya full bring-up (item 1) — LARGE, deprioritize.** After the scope correction
   above it is a from-scratch loader+forward reimplementation against the real ZAYA1-8B
   layout (needs the zaya1_vl modeling code studied), not a mechanical fix. Best as a
   dedicated session with the user engaged; the landed qt-36 OQ fix does not depend on it.
3. dots-ocr OQ: SKIP — not the cheap zaya-style reroute (F16-dequant vision loader,
   unvalidatable, author-deferred). Documented in the scratchpad; not worth churn now.
Deeper still: W2A8/oq2 codec, P2 batched prefill per family, P4 spec-decode.
