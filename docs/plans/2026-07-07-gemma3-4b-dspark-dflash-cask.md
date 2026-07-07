# gemma3-4b (text) speculative decoding: DSpark + DFlash sidecars + CASK enablement

- Date: 2026-07-07
- Status: DESIGN / not started
- Target SKU (first): `medgemma-1.5-4b-it` text decoder (arch gemma3, `head_dim=256`).
  Present on the mount: `/srv/huggingface/models--google--medgemma-1.5-4b-it`.
- Scope: three deliverables for the gemma3-4b text path —
  1. **DSpark** drafter sidecar (trained) — block speculative decode.
  2. **DFlash** drafter sidecar (trained) — hidden-conditioned chain speculative decode.
  3. **CASK** KV-compression *enablement* for gemma3 (runtime/config, **no training** —
     `CaskCtx` is a weightless policy).
  Plus: locate + download a general instruct training corpus for the drafter labels.

## Background — what exists today

- **Arch-agnostic spec-decode core** lives in `crates/hipfire-specdecode-dspark/`
  (`spec.rs`, `dspark_core.rs`). It owns the verifier boundary (`SpecTarget`), the
  drafter interfaces (`Speculator` / `MtpDrafter` / `DsparkBody`), the greedy accept
  rule, and the sidecar `DsparkConfig`/`DsparkWeights` format.
- **`SpecTarget` is the single shared target-side seam** for *both* DSpark and DFlash:
  a target implements batched `verify_block` + extract-layer hidden capture
  (`dflash_extract_layers` + `hidden_out` sinks in `spec_advance`/`verify_block`) and
  both drafter families can drive it. This is why the gemma3 target work is a shared
  foundation, not per-drafter.
- **Reference impls today:** `SpecTarget for LlamaBackend`
  (`hipfire-arch-llama/src/spec_impl.rs`) + the qwen3 `DsparkBody`
  (`hipfire-arch-llama/src/dspark_body.rs`); deepseek4 has its own MTP body. DFlash
  currently has arch-side verify in qwen35 (`speculative.rs::spec_step_dflash`,
  `verify_dflash_block`) and lfm2moe. **Gemma3 implements none of these.**
- **Training:** DSpark drafter training = `hipfire-train/src/dspark_train.rs` +
  `dspark_drafter` (fwd/bwd + heads gradchecked), consuming a `DSLB v1` label cache
  from `examples/dspark_labels.rs`, exported and packed to a `.dspark.hfq` sidecar via
  `crates/hipfire-quantize/src/bin/dspark_convert.rs`. Label-gen currently loads the
  target via `load_llama_from_hfq` (**llama-family only** today).
- **CASK / TriAttention** (`runtime/src/cask.rs`, `runtime/src/triattn.rs`) are
  **weightless** runtime KV eviction/compression policies (greedy L2 grouping + Q8
  weighted-avg of KV rows). Nothing to train; enablement is config + sliding-window
  KV compatibility.

## Gemma3-specific facts that shape the work (`arch-gemma3/src/config.rs`)

- **Interleaved attention:** 5 local sliding-window : 1 global full-causal
  (`sliding_window_pattern=6`), with **dual RoPE theta** (`rope_theta` global vs
  `rope_local_base_freq` local). qwen3/llama are uniform full-causal — the *easy* case.
- `head_dim=256` (4b) decoupled from hidden (2560); **qk-norm** present;
  `query_pre_attn_scalar` score scaling (`attn_scale = s^-0.5`); sandwich norms; GeGLU.
- **√hidden embed scaling** and **tied lm_head** (embedding reused as output proj).
- **Vocab 262144** — large. `markov_head [vocab, rank]` and lm_head/embed dominate
  drafter memory; a real departure from qwen3's smaller vocab.
- Gemma3 is **stateless / pure attention** → `commit_prefix` is a no-op (the cheap
  case, like llama). No recurrent snapshot/rewind.

## Shared foundation — Leg A: `SpecTarget for Gemma3Backend`

Serves DSpark **and** DFlash. Mirror `hipfire-arch-llama/src/spec_impl.rs`.

1. **Extract-layer hidden capture in gemma3 forward.** llama has
   `forward_prefill_batch_capture` (`runtime/src/llama.rs:656`); gemma3 has
   `forward_prefill_batch` (`arch-gemma3/src/forward.rs:684`) but **no capture
   variant**. Add a per-position residual-hidden tap at the target extract layers,
   honoring the local/global block layout. This is the main new kernel-adjacent work.
2. **`impl SpecTarget for Gemma3Backend`** — thin shell:
   - `new_spec_scratch` → a gemma3 `PrefillBatchScratch` sized to block.
   - `verify_block` / `verify_block_logits` → one batched forward, per-position
     argmax / full logits, with `hidden_out` capture.
   - `commit_prefix` → **no-op** (stateless).
   - `spec_advance` (chunked/abortable) with optional `hidden_out`.
   - `eos_token` (Gemma3 `<end_of_turn>`=106, from config), `ctx_capacity`,
     `kv_cache_mut` (gemma3 uses the shared `KvCache` → return `Some`).
   - `dflash_extract_layers` → `Some(layers)`.
3. **Sliding-window reality (investigated 2026-07-07 — resizes M1).**
   gemma3-4b runs SWA by default (5 local : 1 global). gemma3's
   `forward_prefill_batch` (`forward.rs:684`) has an **SWA safety-net that falls
   back to per-token** whenever `swa_window > 0` — the true batched attention path
   only runs with SWA *off*. Consequences:
   - **Per-token verify gives NO spec-decode speedup** (it costs the same as AR),
     so a correctness-only per-token `SpecTarget` is a baseline, not the goal.
   - SWA can't just be disabled for verify — gemma3 was *trained* with windowed
     local attention, so full-context local layers would change outputs.
   - **BUT the batched SWA primitives already support batch>1**:
     `kernels/src/swa_visibility_stage_batched.hip` (grid `[head_dim, batch, 1]`)
     correctly stages each batch position's visible window from the pre-chunk ring
     + the chunk KV, respecting intra-chunk causality. They were built for
     deepseek4's Phase-B2 batched chunked prefill and are called at batch=1 by
     gemma3's per-token path. So batched SWA verify is **wiring existing
     primitives, not new kernels.**

   → **M1 splits:**
   - **M1a (critical path, unblocks M2 training):** per-token gemma3 forward with
     `HiddenCaptureSink` capture + per-position logits, and a correctness-baseline
     `SpecTarget for Gemma3Backend` via the per-token path (greedy-equivalent by
     construction — same kernel as AR decode). Offline label-gen and training need
     ONLY this; no new kernels, low risk.
   - **M1b (serving speedup, follow-on):** a batched gemma3 forward wiring the
     `swa_*_batched` primitives at batch=block_size (local layers) + batched
     causal attention over the KvCache (global layers) + per-position logits +
     capture. This is what makes serving-time spec decode actually faster. Feasible
     (primitives exist) but a substantial GPU implementation; gate with
     `coherence-gate-dflash` for batched==per-token parity.

   gemma3 has its OWN forward (not `runtime::llama`), so the capture/verify helpers
   live in the gemma3 crate (`arch-gemma3/src/{spec,spec_impl}.rs`), not
   `runtime::llama_spec`. Add a `hipfire-specdecode-dspark` dep to the crate.

## Workstream 1 — Training corpus

- **Locate/download a general instruct mix.** Local mount has only QA/medical sets
  (`datasets--google-research-datasets--nq_open`, `datasets--kroshan--BioASQ`,
  `structured-wikipedia`) — **not** a general instruct corpus. Download one
  (candidates: `allenai/tulu-3-sft-mixture`, `HuggingFaceH4/ultrachat_200k`,
  `teknium/OpenHermes-2.5`); check `/srv/huggingface` first, cache there.
- **Render with gemma3 chat template** (`<start_of_turn>`/`<end_of_turn>`) and
  tokenize with the medgemma tokenizer so label positions match serving. Size the
  corpus for drafter training (10s–100s M tokens is typical; start small to validate
  the pipeline end-to-end, then scale).

## Workstream 2 — DSpark drafter (body + training + serving)

1. **`Gemma3DsparkBody`** (`impl DsparkBody`), analog of `Qwen3DsparkBody`
   (`arch-llama/src/dspark_body.rs`): sidecar loader + block-attention body forward.
   - Drafter body dim must match `main_proj` ingest of gemma3 hidden (2560) and use
     gemma3's embed scaling + tied lm_head; RoPE base per design choice.
   - **Design decision:** the 5-layer drafter body need not replicate gemma3's
     sliding-window; simplest is an all-global dense-GQA body (train against gemma3
     hidden). Sliding-window in the drafter is an optimization, not required. **Decide
     before training** — it fixes the sidecar tensor layout.
2. **Training path:** extend `dspark_labels` to generate labels from a **gemma3
   target** (extract-layer hidden + soft logits). Today it uses `load_llama_from_hfq`;
   gemma3 label-gen should route through the runtime gemma3 forward (depends on Leg A's
   capture hook) rather than adding a second gemma3 loader to `hipfire-train`.
   Parameterize `dspark_train`/`dspark_drafter` to gemma3 dim/vocab/embed-scaling.
3. **Export/pack:** add a gemma3 arm to `dspark_convert.rs` (or generalize) → emit
   `medgemma-1.5-4b-it.dspark.hfq` with the arch-agnostic `DsparkConfig` metadata
   (`dspark_block_size`, `dspark_target_layer_ids`, `dspark_markov_rank`,
   `dspark_confidence_uses_normed`, `norm_eps`).
4. **Serving wiring:** `maybe_load_dspark`/`load_dspark_state` arm in
   `hipfire-serving-core/src/load.rs` for gemma3; `DsparkState` in `model.rs`.

## Workstream 3 — DFlash drafter

- **Rides the same `SpecTarget` seam** (Leg A gives it `dflash_extract_layers` +
  hidden capture + `verify_block`). Sidecar loader is `runtime/src/dflash.rs`
  (`DflashWeights::load`), served via `generate_dflash`.
- **Verify item / risk:** the existing DFlash *verify* helpers live per-arch in
  qwen35/lfm2moe (`spec_step_dflash`, `verify_dflash_block`), and a dedicated DFlash
  *training* driver in `hipfire-train` is **not confirmed to exist** (the `drafter.rs`
  there is the PFlash importance-scorer, not DFlash). **Early task:** confirm how the
  existing `.dflash.hfq` sidecars were produced and whether that path is arch-generic
  or must be lifted onto the `SpecTarget` seam for gemma3. This determines whether
  DFlash is "reuse the trainer with a gemma3 target" or "build the trainer."
- Reuse the Workstream 1 corpus + Leg A capture for labels; wire `DflashState` +
  `generate_dflash` gating for gemma3 in `load.rs`.

## Workstream 4 — CASK enablement (no training)

- Verify `CaskCtx`/`EvictionCtx` operate correctly over gemma3's KV, especially the
  **sliding-window local layers** (evicting/compressing rows the local mask already
  drops must not corrupt the global layers). Confirm the eviction hooks fire on the
  gemma3 serving path (`model.rs::Eviction::maybe_evict`).
- Enablement is config (`HipfireConfig`) + making sure CASK composes with the DSpark/
  DFlash decode loops. No dataset, no weights.

## Workstream 5 — Gating, model-support, gates

- `model-support.toml` entries + `gen_model_support` projections for gemma3 DSpark/
  DFlash/CASK; `dflash_gfx_supported` gating (`runtime/src/transformer.rs`).
- **Correctness gate:** `./tests/coherence-gate-dflash.sh` after Leg A + each drafter
  (touches kernels/dispatch/spec path). Admission evidence in `hipfire-eval`
  batteries (acceptance rate τ, tokens/s vs AR baseline).

## Suggested sequencing (milestones)

1. **M0 — corpus:** download + tokenize the instruct mix (parallel to code work).
2. **M1 — Leg A:** gemma3 extract-layer capture + `SpecTarget for Gemma3Backend`;
   validate batched verify == per-token decode (coherence gate). *Unblocks both
   drafters and label-gen.*
3. **M2 — DSpark:** gemma3 label-gen → train (small corpus first) → `dspark_convert`
   → `Gemma3DsparkBody` → serving wiring → τ / tokens-s eval.
4. **M3 — DFlash:** resolve the training-path question, then labels → train → sidecar
   → `generate_dflash` wiring → eval.
5. **M4 — CASK:** sliding-window KV compat + compose with M2/M3; eval.
6. **M5 — scale:** larger corpus, tune block size / extract layers / markov rank;
   promote via `hipfire-eval`.

## Open questions / risks

- **DFlash training tooling existence** (Workstream 3) — the single biggest unknown;
  resolve in M1/M3 before committing effort.
- **Drafter body architecture** (all-global dense vs sliding-window) — fixes the
  sidecar layout; decide before M2 training.
- **262k vocab memory/perf** for markov + lm_head on the drafter (gfx1151, 4b).
- **Sliding-window verify correctness** — the main correctness surface for Leg A.
- Later: 27b-text SKU differs (`head_dim=128`, `query_pre_attn_scalar=168`) → a
  separate drafter shape; out of scope for this plan.
