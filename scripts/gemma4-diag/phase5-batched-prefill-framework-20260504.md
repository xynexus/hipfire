# Gemma 4 Phase 5 — batched prefill framework (default OFF)

- Branch: `gemma4`
- Date: 2026-05-04
- Status: framework + gating in place; correctness bug pending

## What changed

`gemma4::forward_prefill_batch` is no longer a stub — it now contains a
full batched-prefill implementation modeled after `qwen35::forward_prefill_batch`
and `llama::prefill_forward`:

- Per-chunk batched scratch (max chunk 128, configurable via
  `HIPFIRE_GEMMA4_PREFILL_CHUNK`)
- `fused_rmsnorm_rotate_mq_batched` for input + pre-FFN norms (with rotation
  for MQ4 weights)
- `gemm_qkv_hfq4g256` (3 fused projections) + `gemm_gate_up_hfq4g256`
  for fused projections; per-projection `weight_gemm` fallback for non-MQ4
- `rmsnorm_batched` for q/k/v_norm and post-attn / post-ffn norms
- `rope_partial_halved_f32_batched` (new kernel — Gemma 4 uses halved
  pairing, qwen35 uses interleaved; both now have batched variants)
- `kv_cache_write_asym3_batched` for batched K (3-bit rotated) + V (Q8) writes
- Per-token attention loop using existing `attention_flash_asym3` (no
  batched FA for hd=512 yet, and looping keeps sliding/full uniform)
- `gelu_tanh_mul_batched_f32` (Phase 2.5 kernel) for SwiGLU
- Final norm + LM head + logit softcap on the LAST token only

## Daemon hook

`daemon::generate_gemma4` now calls `forward_prefill_batch` first and falls
back to the per-token `forward_scratch` loop on Err. The fast-path is
gated OFF by default (`HIPFIRE_GEMMA4_BATCHED_PREFILL=1` to enable).

## Why default OFF

Three model families end up in different fallback states:

| Model | Why batched returns Err |
|---|---|
| E2B / E4B | per-layer-embedding (n_embd_per_layer > 0) — needs batched per-layer-input lookup + inject (TODO) |
| 26B-A4B | MoE — needs batched router + indexed gate_up/down + GPU fold (TODO) |
| 31B / 31B-IT | `attention_k_eq_v=true` is gated off until the per-batch `K → V` path is debugged |

Forcing 31B through the batched path with `HIPFIRE_GEMMA4_BATCHED_KEQV=1`
runs without panic but produces incoherent output — there's a numerical
bug somewhere in the batched-projection / norm / RoPE / KV-write / attention
sequence that's been hard to isolate via code review. Pending follow-up
debugging session.

## What ships in this commit

- `kernels/src/rope_partial_halved_batched.hip` — new batched halved-pairing
  partial RoPE kernel (Gemma 4-specific)
- `crates/rdna-compute/src/{kernels,dispatch}.rs` — kernel registration +
  `rope_partial_halved_f32_batched` dispatch wrapper
- `crates/engine/src/gemma4.rs` — full batched-prefill implementation (gated)
- `crates/engine/examples/daemon.rs` — try-batched / fallback hook (gated off)

Coherence battery 6/6 strict pass — bit-identical to Phase 4 baseline
across all 6 tests, since the default-off behavior leaves the per-token
path in place.

## Cumulative phase 0 → 5 (no perf change yet — Phase 5 is gated off)

Same as Phase 4: 60.8 → 86.8 decode (+43%), 65.2 → 98.1 prefill (+50%),
752 → 499 ms TTFT (-34%) on 26B-A4B.

## Next steps

1. Debug the batched-path correctness bug on 31B (`HIPFIRE_GEMMA4_BATCHED_KEQV=1`):
   - Bisect: try `HIPFIRE_GEMMA4_FUSED_PROJ=0` to isolate projections
   - Compare per-layer outputs to single-token forward_scratch via dump
   - Likely culprits: q/k/v_norm batching, RoPE positions, KV slot ordering
2. Once 31B works, drop the `_BATCHED_KEQV` gate and the
   `attention_k_eq_v` refusal so 31B uses batched by default.
3. Add per-layer-embedding batched path for E-series.
4. Add batched MoE path for 26B-A4B (existing indexed-batched
   primitives in qwen35).
