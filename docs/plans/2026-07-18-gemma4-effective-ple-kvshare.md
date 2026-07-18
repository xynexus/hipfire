# Gemma4 "Effective" (E2B/E4B) bring-up: PLE + KV-sharing

Goal: make gemma-4-E2B / E4B (text) LOAD + SERVE so gemma4 KVarN (feat 8f2b2507b +
kvarn-512 kernel 170443011) can be validated end-to-end. Scope verdict (mapping agent +
E2B header): **PLE + KV-sharing only — NO AltUp/Laurel/MoE** (E2B: `enable_moe_block:false`,
0 altup/laurel tensors). Tractable. Strategy: implement the READABLE reference path first
(`run_reference_layer`/`attention_block`), validate via `HIPFIRE_GEMMA4_FORWARD_ORACLE=1`,
then mirror into the lowered superop program as a follow-up.

E2B facts: hidden 1536, 35 layers, head_dim 256 (local/SWA) / global_head_dim 512 (full),
num_key_value_heads 1, hidden_size_per_layer_input (ple_dim) 256, num_kv_shared_layers 20
(shared = layers 15..35), sliding_window 512, tie_word_embeddings, gelu_pytorch_tanh.

## KV-sharing — ~90% scaffolded, delta = forward branch
Already done: config resolves per-layer `KvProducer::SharedFrom{producer_layer}` (config.rs
276-332) picking the last non-shared layer of the SAME layer_type; `layered_kv_plan` emits
`.shared(producer)` (config.rs:374-377); runtime arena resolves shared→producer view
(layered_kv.rs resolved_binding/view) and panics on shared store. NEW: in `attention_block`
(forward.rs ~254-568) when `plan.kv_producer != KvProducer::Own` → SKIP wk/wv gemv + k_norm +
K-RoPE + the cache write (SWA ring-write AND Full kv_cache_write); still do Q (proj+q_norm+
q_rope+sqrt-scale) and attend READ-ONLY against the resolved producer view (attention_swa_gqa
or attention_f32 / kvarn). RoPE theta matches because producer is same layer_type. Loader may
still load wk/wv (present on all layers) — keep for now, skip-load later to save VRAM.

## PLE — new, self-contained. Confirmed math (transformers modeling_gemma4.py)
Tensors (HF names; quantizer copies verbatim; all present in gemma-4-E2B.oq8.hfq):
- model.language_model.embed_tokens_per_layer.weight  [vocab 262144, num_layers*ple_dim 8960]
- model.language_model.per_layer_model_projection.weight  [8960, hidden 1536]
- model.language_model.per_layer_projection_norm.weight  [ple_dim 256]
- per layer: per_layer_input_gate.weight [256,1536], per_layer_projection.weight [1536,256],
  post_per_layer_input_norm.weight [1536]

Embed-step precompute (once per token, store per_layer_inputs [35,256] in state):
1. per_layer_tokens = embed_tokens_per_layer[token] → [8960], reshape [35,256]. This is a
   ScaledWordEmbedding: ×sqrt(ple_dim)=sqrt(256)=16 (bf16-round like embed_token).
2. per_layer_proj = (per_layer_model_projection · x_embed) * hidden^-0.5 (=1536^-0.5) →
   [8960], reshape [35,256], then per_layer_projection_norm (RMSNorm over the 256 dim,
   batched ×35).  [x_embed = the scaled token embedding, i.e. state.x right after embed_token]
3. per_layer_inputs = (per_layer_proj + per_layer_tokens) * 2^-0.5.

Per-layer merge — AFTER the FFN residual add (end of run_reference_layer, after post_ffn_norm
+ add), gated identity residual:
  residual = h
  g = act_fn(per_layer_input_gate · h)          # [256], act_fn = gelu_pytorch_tanh
  g = g ⊙ per_layer_inputs[L]                    # [256]
  p = per_layer_projection · g                   # [1536]
  p = post_per_layer_input_norm(p)               # RMSNorm [1536]
  h = residual + p
NOTE: layer_scalar [1] (weights.rs:171, already loaded) is the EXISTING gemma residual scale,
NOT part of PLE. All primitives exist (weight_gemv, gelu_tanh_f32, mul_f32, rmsnorm_f32/
rmsnorm_batched, embedding_lookup_q8, scale_f32) — no new kernels.

## Loader delta (weights.rs load_dense_weights L134)
- Relax reject L139-148: drop the `hidden_size_per_layer_input!=0` and `kv_producer!=Own`
  clauses; KEEP the MoE (`FfnPlan::Dense`) reject.
- Add PleWeights { embed_per_layer: WeightTensor(Q8), model_projection: WeightTensor,
  projection_norm: GpuTensor } to Gemma4DenseWeights + free paths.
- Add {input_gate, projection, post_norm} to Gemma4DenseLayerWeights + free.

## State (Gemma4DenseState forward.rs:34)
Add per_layer_inputs GpuTensor [num_layers*ple_dim=8960], ple_gate [256], ple_proj [1536],
plmp_scratch [8960] scratch. ple_dim/num_layers from config.

## Checklist / status (2026-07-18)
- [x] weights: PLE structs (Gemma4PleWeights/Gemma4PleLayerWeights) + load 3 global +
      3 per-layer PLE tensors + free paths. Reject narrowed: PLE allowed; KV-share + MoE
      still rejected.
- [x] state: PLE scratch (per_layer_inputs/ple_plmp/ple_gate/ple_proj) in new_with_kv_mode.
- [x] forward embed: `ple_embed_precompute` (reference path, after embed_token).
- [x] forward reference layer: `ple_merge` after FFN, before layer_scalar (exact upstream
      order confirmed). All compiles + clippy clean + gemma4 unit tests pass. PLE is
      correct-by-construction (confirmed upstream math + exact op order); NOT yet
      model-validated because E2B also needs KV-share to load (below).
- [ ] **KV-share forward (THE REMAINING BLOCKER for E2B loading).** In `attention_block`,
      for `plan.kv_producer != Own`: skip wk/wv gemv + k_norm + K-RoPE + the cache write;
      do Q only + attend read-only against the producer view.
      - Full/global shared layers: CLEAN — producer's full cache already holds all
        positions (producer runs earlier in the same token); just `attention_f32` against
        `cache.k/cache.v`, no write.
      - **SWA/local shared layers: needs a ring-read primitive.** The producer's ring
        already holds the current token (producer stages→attends→ring_writes before the
        shared layer runs), but `swa_visibility_stage_batched` RE-INJECTS the current
        token's K (`kv_batch` for pos ≥ start_pos) — and the shared layer has no
        current-token K (wk skipped). Fix options: (a) a small gather kernel that reads
        the producer ring's current slot `[kv_head, :, position%window]` (strided by
        window) into `scratch.k` so normal staging re-injects the SAME value (E2B has
        kv_heads=1, so it's one head × head_dim strided copy); or (b) a `no_inject` flag
        on swa_visibility_stage that stages purely from the ring. (a) is least invasive.
      - Mirror the KV-share branch into the lowered superop attend (forward.rs run_attend).
      - Then flip the reject to also allow KV-share.
- [x] KV-share forward IMPLEMENTED (no new kernel): producer (Own) layers save their
      post-RoPE K/V (contiguous copy_d2d); shared layers restore it + skip wk/wv-write.
      SWA-shared re-inject into staging; Full-shared attend the producer cache. reject
      flipped to allow KV-share; `forward_step` forces the reference path for PLE/KV-share
      models; `use_double_wide_mlp` handled (shared layers 15-34 → intermediate 12288).
      kvarn-on-shared guarded (fp32 forced for KV-share models) — that's a follow-up.
- [~] **E2B LOADS + SERVES (fp32 KV) — but COHERENCE BUG (uncommitted, off-branch).**
      Loads fully (PLE + KV-share + double-wide all resolved). Produces fluent English
      and CORRECT short pattern completion (`1 2 3 4 5 6 7` → `8 9 10 11`). BUT semantic
      next-token prediction is WRONG — it copies a recent token instead of predicting
      (`opposite of hot is` → `hot`; `sun ... sets in the` → `east`; `roses are red,
      violets are` → `red`). Ruled OUT: embed_tokens_per_layer=Q8_0 (lookup correct),
      per_layer_model_projection=Oq8G256 (weight_gemv handles it). So the bug is in the
      PLE *math* or the KV-share *logic* (both touch the whole model / deep layers, which
      are the KV-shared 15-34 where semantic processing lives). NEXT: isolate — (a) run
      with PLE disabled vs KV-share disabled to bisect; (b) capture per-layer hidden
      states vs an HF-transformers reference (duat CUDA box) to find the first divergent
      op. The E2B-loading changes are UNCOMMITTED (broken forward kept off-branch); the
      committed branch (PLE at e118d49ec) still rejects KV-share so nothing serves garbage.
- [ ] then: HIPFIRE_KV_MODE=kvarn end-to-end (needs kvarn-on-shared-layer attend-only).

## Coherence-bug isolation (2026-07-18, in progress)
Symptom: E2B loads + fluent; correct short pattern completion (`1..7`→`8 9 10 11`) but
semantic next-token prediction COPIES a recent token (`opposite of hot is`→`hot`,
`sets in the`→`east`, `red, violets are`→`red`). Present at the FIRST generated token
(prefill), so it's fundamental, not decode-accumulation.
RULED OUT so far:
- Gross numerics: per-layer residual `|x|` trace (HIPFIRE_GEMMA4_DEBUG_NORMS=1) is smooth
  ~27→70, no NaN, NO discontinuity at the first shared layer (L15). So KV-share isn't
  exploding magnitudes.
- Attention scaling: upstream Gemma4TextAttention hardcodes `self.scaling = 1.0` (with
  q_norm) — hipfire's "score scale 1" (pre-scale q by √head_dim to cancel the kernel's
  1/√head_dim) MATCHES. Not the bug.
- V handling: upstream applies weightless `v_norm` (Gemma4RMSNorm with_scale=False), RoPE
  to q/k only — hipfire matches (rmsnorm_weightless_f32 on V; rope q/k). E2B is
  attention_k_eq_v=false (Separate v_proj), so the FromPreNormKey pre/post-norm subtlety
  doesn't apply here.
- PLE math + scales: match upstream get/project_per_layer_inputs + the decoder merge
  order (verified earlier). embed_tokens_per_layer=Q8 (lookup ok), model_projection=Oq8
  (weight_gemv ok).
- Quant: deprioritized (oq8 norms smooth; 8-bit rarely flips "cold"→"hot"). A bf16 E2B was
  built but blocked on a BF16 embed-table lookup (no bf16 embedding_lookup kernel) — not
  pursued since quant is an unlikely cause.
REMAINING SUSPECTS (need an HF per-layer reference to pin):
- KV-share LOGIC (my code): shared layers reuse the producer's K/V — subtle correctness.
- BASE gemma4 forward, UNVALIDATED on any real model (no gemma4 ever admitted): esp. the
  FULL/global layers' PROPORTIONAL partial RoPE (rotary_dim=128 of head_dim 512) and the
  head_dim-512 attention — global/semantic integration lives there; a bug would make the
  model fall back to local copying (matches the symptom).
NEXT (definitive): capture per-layer hidden states from HF transformers (gemma4 needs
transformers ~5.5-dev) on the duat CUDA box for one factual prompt, diff vs hipfire's
per-layer capture, find the first divergent op. `HIPFIRE_GEMMA4_DEBUG_NORMS=1` +
temporarily un-gating the loader reject reproduces locally.
- [ ] FOLLOW-UP: mirror PLE (+KV-share) into the lowered superop program
      (forward.rs lower_dense_forward ~935 / run_attend ~1035) for production perf; today
      lowered is the default so serving needs HIPFIRE_GEMMA4_FORWARD_ORACLE=1 until then.
