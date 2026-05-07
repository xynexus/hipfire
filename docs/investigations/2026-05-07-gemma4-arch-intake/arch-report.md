# Gemma 4 Architecture Intake Report

**Date:** 2026-05-07
**Branch (code):** `gemma4-rebased-2026-05-07`
**Branch (docs):** `survey/moe-quant-cliff-2026-05-06`
**Original branch (preserved):** `origin/gemma4` (9 commits, last 2c4f9fd 2026-04-15)
**Rebase target:** `master` HEAD `262e5f6` (post-modular, post-PR #180 router-Q8 fix)

## TL;DR

Gemma 4 IS released as of 2026-05-07. The pre-existing `gemma4` branch
(authored 2026-04-14 → 2026-04-15) targeted the right architecture and
sized the right model (33B "31B" dense). It has been forward-ported as
a new `hipfire-arch-gemma4` crate on top of post-modular master. The
1077-LOC forward path compiles; sliding-window dispatch is stubbed.
Daemon/quantizer/tokenizer wiring and a real-weight smoke run remain.

## 1. Branch State

### 1.1. Released Gemma 4 model lineup (verified huggingface.co/google, 2026-05-07)

| HF model ID                          | Params | Class            | Status                 |
|--------------------------------------|--------|------------------|------------------------|
| `google/gemma-4-31B-it`              | 33B    | Image-Text→Text  | Active (8d ago, 8.4M↓) |
| `google/gemma-4-31B`                 | 33B    | Base             | Active (Apr 2)         |
| `google/gemma-4-26B-A4B-it`          | 27B    | Image-Text→Text  | Active (8d ago, 6.65M↓)|
| `google/gemma-4-26B-A4B`             | 27B    | Base             | Active (Apr 2)         |
| `google/gemma-4-E4B-it`              | 8B     | Any→Any (Instr.) | Active (8d ago, 5.46M↓)|
| `google/gemma-4-E2B-it`              | 5B     | Any→Any (Instr.) | Active (8d ago, 3.38M↓)|

The `26B-A4B` is an MoE variant (likely 4B activated). The `E*B` are
the Any-to-Any (audio/vision/text) variants — those use the audio /
video token IDs already wired into `Gemma4Config`. None are cached on
hiptrx (`~/.cache/huggingface/hub` has only Qwen3.5/3.6 dirs).

The Phase 1 scaffolding commit (`b1b4afa`) named the constants for
`gemma-4-31B`. That model exists; the size matches. The branch's
weights work targeted the dense 33B variant.

### 1.2. Original 9 commits (preserved on `origin/gemma4`)

```
b1b4afa Phase 1 scaffolding — arch_id=7 + Gemma4Config parser
dc7617c sliding-window via window_size uniform + Gemma 4 softcap kernel
f36b068 Phase 3a — forward pass skeleton + Gemma4Scratch
dbf0a3d Phase 3b — sliding + full layer decode bodies
3db5759 Phase 5 quantizer + engine loader + smoke test
440ca6c fix Codex review findings — KV sizing, head_dim=512, unload leaks
aca2154 decouple from deltanet feature gate
f867423 SPM-style BPE tokenizer + BOS prepend
2c4f9fd tokenizer guard SPM-BPE vs Unigram SentencePiece
```

The branch is **483 commits behind master** at start of intake. The
intervening master changes that conflict with gemma4:

- **Engine modularization** (PRs 1-12, masters `22648b5..f716488`):
  `crates/engine/` was renamed to `crates/hipfire-runtime/` and
  per-arch crates `hipfire-arch-{qwen35,qwen35-vl,llama,toy}` were
  split out. All 9 gemma4 commits live in the now-extinct
  `crates/engine/` tree.
- **gfx906 wave64 / dp4a / prefetch** (`d33b14a..ee1be8a`): touches
  the same `attention_flash_*.hip` kernels gemma4 extended with a
  sliding-window arg.
- **PR #180 router-Q8 fix** (`ee1be8a`): touches
  `is_q8_tensor` in `hipfire-quantize`.
- **MMQ auto-dispatch / WMMA / gfx12 ports** (`5c5b1df..52f0f56`):
  changes attention dispatch internals.

### 1.3. Rebase strategy decision

A direct `git rebase master gemma4` produces 9 cascading conflicts
because every gemma4 commit modifies `crates/engine/*` paths that no
longer exist on master. The first rebase step alone hit four
conflicts (modify/delete on `daemon.rs`, `lib.rs`; file-location
warnings on `gemma4.rs`, `gemma4_vision.rs`).

I aborted the per-commit rebase and instead did a **squash-port**: the
cumulative gemma4 work was forward-ported as a single conceptual unit
onto master's modular layout (a new `hipfire-arch-gemma4` crate plus
two net-new kernels and two dispatch helpers). This:

- Preserves `origin/gemma4` untouched as a rollback reference.
- Avoids replaying the same engine→arch-crate rename conflict 9×.
- Co-locates the gemma4 source in the canonical place per the post-
  modular contract (PR 11's llama crate is the structural template).

### 1.4. Rebase outcome

Single port commit on `gemma4-rebased-2026-05-07`:

```
7e9cc8a feat(arch-gemma4): port Phase 1-5 scaffolding onto modular master
```

Surface created:

| Path                                                      | Origin                                  | Notes                          |
|-----------------------------------------------------------|-----------------------------------------|--------------------------------|
| `crates/hipfire-arch-gemma4/Cargo.toml`                   | new                                     | non-deltanet, dense            |
| `crates/hipfire-arch-gemma4/src/lib.rs`                   | new                                     | module roots                   |
| `crates/hipfire-arch-gemma4/src/arch.rs`                  | new                                     | `Architecture` trait impl      |
| `crates/hipfire-arch-gemma4/src/gemma4.rs`                | gemma4 branch (1077 LOC)                | imports patched                |
| `crates/hipfire-arch-gemma4/src/gemma4_vision.rs`         | gemma4 branch (169 LOC)                 | imports patched                |
| `crates/hipfire-arch-gemma4/examples/gemma4_smoke_forward.rs` | gemma4 branch (176 LOC)             | gated off (`cfg(any())`)       |
| `kernels/src/logit_softcap.hip`                           | gemma4 branch                           | net-new                        |
| `kernels/src/rope_partial_halved.hip`                     | gemma4 branch                           | net-new                        |
| `crates/rdna-compute/src/kernels.rs`                      | additive (master)                       | 2 new SRC consts               |
| `crates/rdna-compute/src/dispatch.rs`                     | additive (master)                       | 2 new helpers                  |
| `Cargo.toml`                                              | additive (master)                       | new workspace member           |

### 1.5. Conflicts resolved (squash-port)

The squash-port mode eliminates per-commit rebase conflicts but
requires conscious choices about the merge surface. Decisions:

| Surface                        | Choice (master vs gemma4)       | Rationale                                                          |
|--------------------------------|----------------------------------|--------------------------------------------------------------------|
| Crate location                 | master (per-arch crate)          | Branch was pre-modular; modular layout is the new contract.        |
| daemon.rs additions            | DEFER                            | Branch added 126 LOC of arch_id==7 dispatch; replay on hipfire-runtime/examples/daemon.rs is a separate PR. |
| `is_q8_tensor` change          | master (PR #180)                 | Router-Q8 fix is correctness-critical and arch-agnostic.           |
| `attention_flash_*.hip` kernels| master + TODO                    | Branch added trailing `window_size` arg; master added wave64/dp4a. Both real; resolve by porting gemma4's sliding-window masking on top of master's wave64 path in a follow-up. Forward-path call sites are commented with `TODO(gemma4-sliding-window-kernel)`. |
| Tokenizer SPM-BPE detection    | DEFER                            | Master tokenizer added `eot_id` + `from_gguf_meta_json`; gemma4 added `is_spm_bpe`. The two diffs touch overlapping line ranges in different ways. Trivial conflict but must be re-applied additively on the post-modular tokenizer.rs. |
| Tokenizer guards (SPM-BPE vs Unigram) | DEFER                     | Same as above.                                                     |
| Quantizer arch_id=7 case       | DEFER                            | Branch added 27 LOC to hipfire-quantize/src/main.rs; master's PR #180 changed the quantizer surface. Conflicts at the architecture-match line.|
| `speed-baselines/*`            | master (kept)                    | gemma4 deleted some baselines that PRs since have re-added.        |

### 1.6. Build status

`cargo build --release` (full workspace, default features): **GREEN**.
`cargo build --release -p hipfire-arch-gemma4 --examples`: **GREEN**.
Warnings only — one unused-import in `gemma4.rs`, none new in master crates.

### 1.7. Smoke test status

**NOT RUN.** The example `gemma4_smoke_forward` is currently gated off
(`#[cfg(any())]`) because the forward body's four sliding-window
attention dispatches refer to a kernel signature that does not yet
exist on master. To run the smoke test, the gaps in §3 must be
addressed. No quantized Gemma 4 `.mq4` file exists in the project
caches (hiptrx HF cache lists only Qwen models). A fresh download
+ quantize would be required before any real-weight forward run.

## 2. Architecture Characterization

Source: `Gemma4Config` parser in `crates/hipfire-arch-gemma4/src/gemma4.rs`,
backed by HuggingFace `config.json` keys. Constants below are for
`gemma-4-31B`; other variants override per their `text_config`.

| Field                            | Value (31B)        | HF config key                       | Notes                                       |
|----------------------------------|--------------------|-------------------------------------|---------------------------------------------|
| hidden_size (`dim`)              | 5376               | `hidden_size`                       |                                             |
| n_layers                         | 60                 | `num_hidden_layers`                 |                                             |
| vocab_size                       | 262144             | `vocab_size`                        | 4× larger than Qwen3.5 (151936)             |
| n_heads                          | 32                 | `num_attention_heads`               | same count for sliding + full               |
| sliding_head_dim                 | 256                | `head_dim`                          | 5 of 6 layers                               |
| sliding_n_kv_heads               | 16                 | `num_key_value_heads`               | GQA ratio = 2                               |
| sliding_window                   | 1024               | `sliding_window`                    | per-layer cap (KV stride)                   |
| sliding_rope_theta               | 10000              | `rope_parameters.sliding_attention.rope_theta` | full rotation, default RoPE     |
| full_head_dim                    | 512                | `global_head_dim`                   | 1 of 6 layers; **2× sliding**               |
| full_n_kv_heads                  | 4                  | `num_global_key_value_heads`        | GQA ratio = 8                               |
| full_rope_theta                  | 1e6                | `rope_parameters.full_attention.rope_theta` | proportional partial RoPE         |
| full_partial_rotary_factor       | 0.25               | `rope_parameters.full_attention.partial_rotary_factor` | 64 of 256 pairs rotate         |
| attention_k_eq_v                 | true               | `attention_k_eq_v`                  | **V is pre-k_norm output of k_proj** — no `v_proj` weight on full layers |
| hidden_dim (FFN intermediate)    | 21504              | `intermediate_size`                 |                                             |
| layer_types pattern              | `[S,S,S,S,S,F] × 10` | `layer_types`                     | 5 sliding : 1 full                          |
| norm_eps                         | 1e-6               | `rms_norm_eps`                      |                                             |
| final_logit_softcapping          | 30.0               | `final_logit_softcapping`           | `tanh(logits/30)*30`                        |
| tie_word_embeddings              | true               | `tie_word_embeddings`               | lm_head aliases embed_tokens                |
| embed_scale                      | sqrt(5376)≈73.32   | derived                             | multiplied at every embed lookup            |
| Tokenizer family                 | SPM-style BPE      | `tokenizer.json` `model.type`+`▁`   | space=`▁` (U+2581), with merges (≠Unigram)  |
| BOS / EOS / PAD                  | 2 / 1 / 0          | `bos_token_id` etc                  | BOS prepended to encoded input              |

### Vision / multimodal token IDs (from top-level config, not text_config)

| Token            | ID      |
|------------------|---------|
| image            | 258880  |
| boi              | 255999  |
| eoi              | 258882  |
| audio (reserved) | 258881  |
| video (reserved) | 258884  |

Vision tower: `gemma4_vision.rs` ports the SigLIP-style image encoder
scaffold (169 LOC); not validated.

### Distinctives vs. Qwen3.5

1. **Hybrid sliding+full attention** (5:1 ratio); Qwen3.5 is dense FA.
2. **Heterogeneous head_dim within a model** (sliding=256, full=512).
3. **K=V on full layers** (`attention_k_eq_v` — saves a projection).
4. **Proportional partial RoPE** (HF rotate_half pairing, only first
   N pairs rotate); Qwen3.5 uses interleaved pairing.
5. **Sandwich RMSNorm** (4 norm sites per layer + per-layer scalar).
6. **Final logit softcap** (`tanh(x/30)*30`); not in Qwen3.5.
7. **Embed scale** (sqrt(dim) at lookup time).
8. **Tied LM head** (Qwen3.5 dense untied; A3B/A10B tied).
9. **Vocab 262144** (Qwen3.5 = 151936) — implies larger embed table
   memory pressure (5376 × 262144 × 2 bytes = 2.7 GB at fp16, hence
   the design choice of Q8F16 for the embed table).

## 3. Kernel-Fit Checklist

Coverage analysis for a single decode step on a Gemma 4 sliding or
full layer.

| Step                          | Kernel needed                       | Status                                                           |
|-------------------------------|-------------------------------------|------------------------------------------------------------------|
| Embed lookup + sqrt(dim) scale| `embed_lookup_q8f16` + `scale_f32`  | EXISTS (master via `weight_gemv` family + `scale_f32`)           |
| Input RMSNorm                 | `rmsnorm_f32`                        | EXISTS                                                           |
| Q/K/V projections (sliding)   | `weight_gemv` HFQ Q4/Q6/Q8           | EXISTS                                                           |
| Q/K projections (full, no V)  | `weight_gemv`                        | EXISTS — V alias to k_proj output is config-driven, not kernel    |
| Q-norm / K-norm RMSNorm       | `rmsnorm_f32`                        | EXISTS                                                           |
| V "no-scale RMSNorm" (full)   | `rmsnorm_f32` w/ ones-buffer         | EXISTS — uses shared ones tensor as scale, no learned weight     |
| Sliding RoPE (full rotation)  | `rope_f32`                           | EXISTS                                                           |
| Full proportional partial RoPE| `rope_partial_halved_f32`            | **NEW — added in this port** (kernel + dispatch helper)          |
| KV cache write                | `kv_cache_write_*` family            | EXISTS                                                           |
| Sliding-window FlashAttention | `attention_flash_*` + `window_size` arg | **MASTER LACKS** — see §4 gap 1                              |
| Full FlashAttention           | `attention_flash_*` (no window arg) | EXISTS                                                           |
| O-projection                  | `weight_gemv`                        | EXISTS                                                           |
| Post-attention RMSNorm        | `rmsnorm_f32`                        | EXISTS                                                           |
| Pre-FFN RMSNorm               | `rmsnorm_f32`                        | EXISTS                                                           |
| FFN gate + up (SwiGLU)        | `weight_gemv_swiglu_residual`        | EXISTS — but SwiGLU is `gelu_pytorch_tanh`, see §4 gap 2          |
| FFN down                      | `weight_gemv`                        | EXISTS                                                           |
| Post-FFN RMSNorm              | `rmsnorm_f32`                        | EXISTS                                                           |
| Per-layer scalar multiply     | `scale_f32` (host const)             | EXISTS                                                           |
| Final RMSNorm                 | `rmsnorm_f32`                        | EXISTS                                                           |
| LM head (tied)                | `weight_gemv` Q8F16                  | EXISTS                                                           |
| Final logit softcap           | `logit_softcap_f32`                  | **NEW — added in this port** (kernel + dispatch helper)          |

Net new kernels in this port: **2** (`logit_softcap`, `rope_partial_halved`).
Both are simple, pure-arithmetic (no MMA / wave-64 cleverness needed),
and unit-test-friendly. Both are now registered in
`crates/rdna-compute/src/kernels.rs` and have dispatch helpers on
the `Gpu` impl.

## 4. Remaining Gaps to a Coherence-Gate-Passing Forward

Listed in execution order; estimates assume one focused contributor.

### Gap 1: Sliding-window kernel diff forward-port (HARD BLOCKER)

The gemma4 branch commit `dc7617c` extended four kernels with a
`uint32_t window_size` parameter that masks keys at offsets
`<= cur_pos - window_size` to `-INFINITY`:

- `kernels/src/attention_flash_asym2_tile{,_batched}.hip`
- `kernels/src/attention_flash_asym3_tile{,_batched}.hip`
- `kernels/src/attention_flash_asym4_tile{,_batched}.hip`
- `kernels/src/attention_flash_q8_0_tile.hip`

Master's versions of these kernels have changed substantially (gfx906
wave64 prefetch, dp4a, MMQ tile redesign). The diff cannot be
applied verbatim; it must be reapplied on top of the post-modular
kernel bodies. Three-way merge per kernel (~50 LOC each, mostly
local). Dispatch helpers on the Rust side need a matching trailing
`window_size: u32` arg.

**Effort:** ~1-2 days for kernel port + dispatch update + a small
unit test that diff-checks sliding mask vs full mask on synthetic
shapes.

**Impact:** Blocks any sliding-layer forward call. Without this,
Gemma 4 cannot decode at all (sliding layers are 5/6 of the model).

### Gap 2: SwiGLU activation — `gelu_pytorch_tanh` ≠ Qwen's silu

Qwen3.5 SwiGLU uses `silu(gate) * up`. Gemma 4 uses `gelu_pytorch_tanh
(gate) * up`. The fused `weight_gemv_swiglu_residual` kernel hard-codes
the silu non-linearity. Either:

(a) Add a `weight_gemv_swiglu_gelutanh_residual` variant, OR
(b) Add a tensor activation kind enum + branch.

`gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
— pure scalar, no MMA. Best done as a kernel variant to avoid a hot-path branch.

**Effort:** ~half day.

### Gap 3: SPM-BPE tokenizer wiring on master

Branch commits `f867423` + `2c4f9fd` added `is_spm_bpe: bool` field to
`Tokenizer` and a Unigram-vs-BPE detection guard (require non-empty
merges to disambiguate ▁-prefix from plain SentencePiece). Master's
tokenizer added `eot_id: Option<u32>` and `from_gguf_meta_json` in
the same file. The two diffs are not on overlapping lines (one in
`from_hf_json`, the other in field decls and `is_gpt2_bpe` detection),
but both touched the constructor — clean three-way merge with manual
ordering. Per-encode branch in `encode_raw` to add: BOS prepend
+ ▁-replacement of leading space.

**Effort:** ~half day.

### Gap 4: Daemon arch dispatch (arch_id=7)

Branch commit `b1b4afa` added 126 LOC of `arch_id == 7` dispatch to
`crates/engine/examples/daemon.rs`. That file is now
`crates/hipfire-runtime/examples/daemon.rs` and has had multiple PRs
of structural change. Re-apply: load HFQ → check `arch_id == 7` →
`Gemma4::config_from_hfq` → `Gemma4::load_weights` → `Gemma4::new_state`
→ enter generation loop with `gemma4::forward_scratch`. Path well-
trodden by the qwen35 / llama dispatch arms.

**Effort:** ~half day.

### Gap 5: Quantizer arch_id=7 case

Branch commit `3db5759` added a 27-LOC arch-7 case to
`crates/hipfire-quantize/src/main.rs` (HFQ-out tensor naming + which
weights need Q8 floor for embed/lm_head). Master's PR #180 changed
the same area. Re-apply additively.

**Effort:** ~half day.

### Gap 6: Real-weight smoke run

No `gemma-4-*` `.mq4` exists in any project cache. To run the smoke
test:

1. Download `google/gemma-4-31B-it` (or `-E2B-it` for a faster first run)
   to hiptrx (`~/.cache/huggingface/hub`).
2. Quantize via `cargo run -p hipfire-quantize -- ... --arch gemma4`
   (after Gap 5).
3. Run `cargo run -p hipfire-arch-gemma4 --example gemma4_smoke_forward
   -- /path/to/gemma-4.mq4` (after Gaps 1-3).
4. Logits-finite check passes → manual eyeball of greedy-decode tokens
   against an HF reference impl.

**Effort:** ~1 day (download is large; first-time quantize on a
262144-vocab model is slow).

### Gap 7: Coherence-gate fixed prompts for Gemma 4

`scripts/coherence-gate.sh` has a fixed prompt matrix plus per-arch
expected behaviors. Add a `gemma4` profile (model resolve, prompt
shape — Gemma uses `<start_of_turn>user...<end_of_turn>\n
<start_of_turn>model\n` framing).

**Effort:** ~half day.

## Total effort estimate

**3-5 focused days** to land a coherence-gate-passing Gemma 4 forward
on hiptrx (4× R9700 gfx1201). Critical path is Gap 1 (sliding-window
kernel port) followed by Gap 6 (download + quantize). All other gaps
are parallelizable.

## Files

- `crates/hipfire-arch-gemma4/src/gemma4.rs` — primary forward path.
- `crates/hipfire-arch-gemma4/src/arch.rs` — trait impl.
- `crates/hipfire-arch-gemma4/src/gemma4_vision.rs` — vision tower.
- `crates/hipfire-arch-gemma4/examples/gemma4_smoke_forward.rs` — smoke test (gated off).
- `kernels/src/logit_softcap.hip` — Gemma final-logit softcap.
- `kernels/src/rope_partial_halved.hip` — proportional partial RoPE.
- `crates/rdna-compute/src/dispatch.rs` — `rope_partial_halved_f32` + `logit_softcap_f32` helpers (added at line ~12690).
- `crates/rdna-compute/src/kernels.rs` — `ROPE_PARTIAL_HALVED_SRC` + `LOGIT_SOFTCAP_SRC` (added at line ~847).

## References

- Original branch (preserved): `origin/gemma4`.
- Rebased branch: `origin/gemma4-rebased-2026-05-07`.
- Hugging Face Gemma 4 lineup: https://huggingface.co/google.
- Architecture-port skill / template: `crates/hipfire-arch-toy/`, `crates/hipfire-arch-llama/`.
- Architecture trait contract: `crates/hipfire-runtime/src/arch.rs`.
- Engine modularization PRD: `docs/plans/engine-modularization.prd`.
- Arch-intake pipeline (this report's methodology, generalized): `docs/plans/arch-intake-pipeline.md`.
