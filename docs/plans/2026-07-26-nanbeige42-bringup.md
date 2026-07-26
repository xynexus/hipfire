# Nanbeige4.2-3B bring-up plan (Looped Transformer, text-only)

Status: **draft** — started 2026-07-26.
Owner: (unassigned).

## Goal

Native hipfire support for the **Nanbeige4.2** family (`NanbeigeForCausalLM`),
with no GFX override and no Python in the hot path (project Rules 1/5). The
bring-up target is **`Nanbeige4.2-3B`** (agentic instruct SKU), staged in
`/srv/huggingface/models--nanbeige--Nanbeige4.2-3B`
(snapshot `f79d0a9e191d046628efee918646000adfdf3d0f`).

The model is **Llama-shaped per layer** (standard `q/k/v/o_proj`,
`gate/up/down_proj`, `input_layernorm`, `post_attention_layernorm`,
`embed_tokens`, `norm`, untied `lm_head`) — the entire novelty is a **Looped
Transformer**: the physical 22-layer stack is executed `num_loops = 2` times per
token, so the model presents **44 effective layers from 22 layers of weights**.
That means bring-up is mostly "clone the qwen35/gemma3 dense forward, wrap the
layer loop in an outer loop, and double the KV cache." No new kernels are
required.

`arch_id` allocation: **25 = `ARCH_ID_NANBEIGE`** (next free after gemma4 = 24;
20/21 are tooling-only, 22 free, 23 flux2, 24 gemma4). Crate:
`hipfire-arch-nanbeige`.

## Reference config (verified 2026-07-26 from the snapshot `config.json`)

| field | value | notes |
|---|---|---|
| architectures | `NanbeigeForCausalLM` | `model_type: "nanbeige"` |
| hidden_size | 3072 | |
| num_hidden_layers | **22** | *physical* layers on disk |
| **num_loops** | **2** | run the stack twice → 44 effective layers |
| num_attention_heads | 48 | |
| num_key_value_heads | 8 | GQA (6:1) |
| head_dim / kv_channels | **128** | **q/o attention dim = 48·128 = 6144 ≠ hidden 3072** |
| intermediate_size | 10752 | SiLU / SwiGLU |
| vocab_size | 166144 | **untied** lm_head (`tie_word_embeddings: false`) |
| max_position_embeddings | 262144 | |
| rms_norm_eps | 1e-5 | |
| rope_theta | 7e7 (`70000000`) | single global θ, no rope_scaling |
| attention_bias / mlp_bias | false | plain Llama projections |
| torch_dtype | bfloat16 | |
| bos / eos / pad | 166100 / 166101 / 0 | generation stops on 166101 |
| generation defaults | temp 0.6, top_p 0.95, top_k 20 | from `generation_config.json` |

Derived sizes: attention q_proj `3072→6144`, o_proj `6144→3072`, k/v_proj
`3072→1024`. On-disk footprint ≈ **8.34 GB** bf16 (of which ≈2 GB is the untied
166k-vocab embed + lm_head).

### Fields present in the code but **inactive** for this checkpoint

`modeling_nanbeige.py` ships forward-looking Nanbeige4.5 R&D that this SKU does
**not** enable (verified: config leaves them unset and the checkpoint carries
only standard Llama tensors — no ngram/hyperconnection weights):

- `enable_double_loop_split` / `loop_middle_layers` (**LoopSplit** — a different
  layer execution order; here `num_loops=2` is the plain "run the whole stack
  twice" mode).
- `loop_share_kv` (reuse pass-0 KV in later passes — **off**; each loop pass
  computes and caches its own KV).
- `enable_hyper_connection` + `enable_depth_attention` (**mHC** — off).
- `emb_neighbor_num` / n-gram embeddings + `NanbeigeNgramLayerFusion` (off).

**Scope decision:** implement only the enabled path (`num_loops` plain loop,
independent per-pass KV). Do **not** port LoopSplit / mHC / n-gram — leave clear
`unimplemented!`/validation errors if a future config turns them on, so a
Nanbeige4.5 artifact fails loudly instead of running a wrong forward.

## Architecture delta vs the qwen35/llama dense forward

The per-layer math is identical to a standard GQA Llama decoder (RMSNorm →
attn(RoPE, GQA) → residual → RMSNorm → SwiGLU → residual). The **only** deltas:

1. **Outer loop over the layer stack (the whole point).** The reference forward
   iterates `for layer_idx in 0..num_hidden_layers`; Nanbeige wraps that in
   `for loop_idx in 0..num_loops { for layer_idx in 0..num_hidden_layers { … } }`,
   threading `hidden_states` straight through — the output of pass 0 (after all
   22 layers, **without** an intermediate final-norm; `skip_loop_final_norm:
   false` applies the final `model.norm` only once, after the last pass) is the
   input to pass 1. The final `model.norm` + `lm_head` run once at the end.

2. **Loop-aware KV cache = doubled layer count.** The reference cache indexes
   `state.kv_cache.*[layer_idx]` with `num_hidden_layers` slots. Nanbeige must
   allocate **`num_hidden_layers * num_loops` = 44** KV slots and index them as
   `cache_layer_idx = layer_idx + loop_idx * num_hidden_layers`
   (mirrors `_get_loop_cache_layer_idx` in `modeling_nanbeige.py`). Every pass
   attends over its own causal KV history for that (loop_idx, layer_idx) pair.
   **KV memory is 2× a normal 22-layer model at the same context** — size
   accordingly.

3. **head_dim independent of hidden_size.** `48·128 = 6144 ≠ 3072`. The loader
   and attention setup must take `head_dim` from config, never derive it as
   `hidden/n_heads`. (qwen35/gemma3 already carry an explicit `head_dim`; reuse
   that path — do **not** assume square q/o projections.)

Everything else is stock: single-θ RoPE (7e7), no QK-norm, no sliding window, no
soft-capping, no attention/MLP bias, SwiGLU, untied lm_head, RMSNorm (plain `w`,
**not** the gemma `(1+w)` bake — do not add a norm offset).

## Compute cost note

Two loop passes ≈ **2× the decode FLOPs/latency** of a 22-layer 3B model (it is
a 44-layer model that happens to share weights). The win is VRAM/footprint (22
layers of weights), not speed. Weights stay resident across both passes — no
re-stream between loop 0 and loop 1. Factor the 2× compute and 2× KV into any
throughput/context planning.

## Implementation phases

### Phase 0 — registry + detection wiring (no forward yet)

Follows `docs/architecture-ids.md` §"Adding a new architecture":

1. `crates/hipfire-arch-api/src/lib.rs`: add
   `pub const ARCH_ID_NANBEIGE: u32 = 25;` to the `ARCH_ID_*` block.
2. `crates/hipfire-model/src/lib.rs`: re-export `ARCH_ID_NANBEIGE`; add the
   family name + `model_arch_family` arm.
3. `crates/hipfire-quantize/src/main.rs`: add `"nanbeige" => ARCH_ID_NANBEIGE`
   to the `model_type → arch_id` match (near the gemma4 arm), referencing the
   constant, not a literal. (No GGUF importer arm — no GGUF path for this model.)
4. `docs/model-support.toml`: add an `[[arch]] ids = [25] label = "nanbeige"`
   row; regenerate `model_support_generated.rs` via `hipfire gen-model-support`.
5. `docs/architecture-ids.md`: add the id-25 table row.

**Gate:** `./tests/no-gpu-ci.sh` green; `hipfire-quantize` recognizes the
`model_type: "nanbeige"` config and tags `arch_id 25` at ingest.

### Phase 1 — ingest / quantize path (produce a `.hfq`)

The tensor layout is vanilla Llama, so ingest reuses the existing dense mapping
(`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
`{input,post_attention}_layernorm`, `embed_tokens`, `norm`, `lm_head` — all
already handled in `hipfire-quantize/src/main.rs`).

- Verify the quantizer emits the untied `lm_head` (do **not** alias it to
  `embed_tokens`).
- Carry `num_loops`, `num_hidden_layers`, `head_dim`, `num_key_value_heads`,
  `rope_theta`, `rms_norm_eps`, `vocab_size`, `eos_token_id` into HFQ metadata
  (embed the full original `config.json` under `metadata.config`, per the
  gemma3/gemma4 convention).
- Add ingest-time validation that rejects the inactive features (LoopSplit /
  loop_share_kv / hyper-connection / n-gram) with a clear error if a future
  config turns them on.
- First artifact: **BF16** (`Nanbeige4.2-3B.bf16.hfq`) for a correctness
  baseline before any quant. Then **Q8** and **MQ4** once BF16 golden passes.

**Gate:** `hipfire-quantize` produces a BF16 `.hfq` whose metadata round-trips
`num_loops=2`, `num_hidden_layers=22`, `head_dim=128`.

### Phase 2 — `hipfire-arch-nanbeige` crate + looped forward

Scaffold from `hipfire-arch-template`; model the structure on
`hipfire-arch-qwen35` (closest dense GQA Llama, explicit head_dim). Files:

- `config.rs` — `NanbeigeConfig` parsed from HFQ metadata, incl. `num_loops`.
- `forward.rs` — `NanbeigeState` (KV cache sized `num_hidden_layers * num_loops`),
  `forward_step` / `forward_prefill_batch` / `embed_token`. Clone the qwen35
  dense forward, then:
  - wrap the `for layer_idx in 0..num_hidden_layers` body in
    `for loop_idx in 0..cfg.num_loops`;
  - compute `cache_layer_idx = layer_idx + loop_idx * num_hidden_layers` and use
    it for **every** KV read/write and for the RoPE position (positions are the
    token positions — unchanged across passes; only the cache slot changes);
  - apply `model.norm` + `lm_head` **once**, after the final pass.
- `arch.rs` — `Nanbeige: Architecture` (`arch_id() = 25`, `name() = "nanbeige"`,
  `config_from_hfq`, `load_weights`, `new_state`) and `NanbeigeBackend: SimpleAr`
  (`prefill` / `decode_step` / `logits` / `vocab_size`) delegating to
  `forward_step`, exactly like `Gemma3Backend`.
- `caps.rs` — advertise no fast-path caps initially (shared `run_simple_ar`
  loop); dense-AR only.
- Register the backend so the daemon can serve `arch_id 25`.

**KV allocation detail:** `new_state` allocates `k_gpu`/`v_gpu` (and any window
buffers, though none needed — no SWA) as `Vec` of length
`num_hidden_layers * num_loops`. Prefill writes both passes' KV in order; decode
appends one position per (loop_idx, layer_idx) slot per token.

**Gate:** load the BF16 `.hfq` on gfx1103, run `hipfire run` with a short prompt,
confirm coherent tokens and correct EOS (166101). Compare first-N-token logits
against a Python `transformers` reference (`trust_remote_code`) on the same
prompt — max-abs logit delta within bf16 tolerance. This is the correctness
tripwire for the loop + doubled-KV indexing (the two places most likely to be
wrong).

### Phase 3 — quant + eval admission

Once BF16 golden holds:

- Quantize Q8 and MQ4; re-run the logit-parity + short-generation smoke.
- Add a tiny golden fixture (per `tests/tiny-affected-gate.sh` conventions) that
  exercises `num_loops=2` so a regression in the loop wiring is caught by CI.
- Route model-quality admission through `hipfire-eval` (per project Verification
  rules) — the agentic SKU is best sanity-checked on a small reasoning/coding
  battery, not just perplexity.

**Gate:** `./tests/tiny-affected-gate.sh --require-coverage` green for the
nanbeige fixture; `hipfire-eval` fast tier recorded for BF16 and MQ4.

## Verification summary (per project rules)

- Workflow-only changes (Phase 0 wiring): `./tests/no-gpu-ci.sh`.
- Covered runtime/quant changes (Phases 1–3):
  `./tests/tiny-affected-gate.sh --require-coverage`.
- Model correctness: BF16 logit-parity vs `transformers` reference, then
  `hipfire-eval` batteries before any quant is promoted.
- GPU work coordinated via `hipfire lock {acquire,release,status}` (non-daemon
  binaries do not self-lock).

## Open questions / risks

1. **Loop numerics** — confirm the reference applies `model.norm` only after the
   final pass and feeds the *un-normed* pass-0 output into pass 1
   (`skip_loop_final_norm: false` semantics). The logit-parity gate settles this;
   read `modeling_nanbeige.py` around the `for loop_idx in range(num_loops)`
   body (lines ~2217 / ~2585) to confirm before coding.
2. **Prefill batching across loops** — a batched prefill must fill loop-0 KV for
   all positions across all 22 layers before starting loop-1 (loop-1 layer L
   attends over loop-1 KV history, which depends on loop-1's earlier positions).
   Order the prefill as loop-outer, position-inner to match decode.
3. **Fat attention shapes** — 6144-wide q/o with 3072 hidden; make sure the
   attention family and any fused gemv paths take the real head_dim/attn-dim and
   don't assume `hidden == n_heads·head_dim`.
4. **2× cost expectations** — set user/throughput expectations that this is a
   44-layer-equivalent model; the loop is a capacity/footprint trade, not free.
5. **Nanbeige4.5 forward-compat** — the shipped modeling file already contains
   LoopSplit / mHC / n-gram; if a 4.5 artifact appears, this crate should detect
   and reject those configs rather than silently mis-run them.
