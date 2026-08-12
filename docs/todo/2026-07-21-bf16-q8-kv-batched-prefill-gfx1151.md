# BF16 weights + Q8 KV batched-prefill → garbage on gfx1151

**STATUS: GUARDED/FIXED** (`is_batchable_la` now routes BF16 prefill per-token on
gfx1151 — bf16+q8 verified coherent, mq4/mq6 unaffected). The *root cause* (the
batched-arm BF16 q/k/v projection inflating `fa_q` ~9× on gfx1151) is still latent
— the guard avoids it rather than fixing the kernel/projection. Real fix is a
follow-up. Other archs (gfx1100/1201) not yet verified for the batched bf16 path.

**gfx1151 + BF16-weight model + Q8/asym KV cache = garbage output** (token
attractor, no crash). Needs ALL of: BF16 weights, Q8(/asym) KV, gfx1151, batched
prefill. Fine if any leg changes: bf16+fp32-KV ✅, f16-convert+q8-KV ✅,
bf16+q8-KV on gfx1103 ✅. Quantized-weight models (MQ4/MQ6) unaffected.

- **Mechanism:** Q8/asym KV sets `fa_batched_ok=true` → batched FullAttention
  prefill arm (`forward_prefill_chunk`, `crates/hipfire-arch-qwen35/src/qwen35.rs`).
  The **BF16 q/k/v projection in that batched arm inflates `fa_q` ~9×** on gfx1151
  (per-layer dump: residual bit-identical until layer 3 = first FullAttention,
  then output blows up 15×). fp32 KV → `fa_batched_ok=false` → per-token path,
  which is correct. DeltaNet/LinearAttention layers (`dn_*` projections) are fine.
  The `gemm_bf16_x_bf16_wmma` kernel is correct in isolation (parity passes all
  shapes) — bug is in how the batched FA arm *uses* it on gfx1151.
- **Ruled out:** the bf16 matmul kernel, the m128 path
  (`HIPFIRE_BF16_DENSE_M128=0`), graph capture (`HIPFIRE_GRAPH=0`), loader shape
  metadata, the fp32-GQA4 attention path — none fix it.
- **Trigger:** `a21dccf75` "stop forcing fp32 KV on bf16 models" + default
  `kv_cache=q8`. Latent bug it merely exposes (pre-existing in the batched arm).
  Good at `317de4fa`, garbage at `77154a110`/HEAD.
- **Workaround:** `HIPFIRE_KV_MODE=fp32` (or `kv_cache=fp32`) for bf16, OR
  `HIPFIRE_BF16_WEIGHTS=f16`.
- **Downstream:** bf16 reference *PPL* via `eval_hipfire` (which runs q8 KV by
  default) read ~4.9M garbage; `--kv-mode fp32` restores sane PPL (22.38 vs mq4
  25.69 / mq6 23.05, correctly ordered). NOTE: the *KLD* ~7.6 seen alongside is a
  SEPARATE, already-fixed issue — the standalone `eval_hipfire`/
  `build_kld_ref_hipfire` bins are DEPRECATED (`ee94e4aa3`) due to a known
  2.85-nat self-inconsistency from forward/env drift between the two binaries;
  use the daemon `kld_eval` op (≈0 self-consistency) instead. Also: rebuilding an
  fp32 kldref via the old bin panics `no implementation for KvWriteF32` (fp32-KV
  write gap in that prefill path).
- **Fix:** (1) guard — keep bf16 on fp32 KV / per-token on gfx1151; or (2) repair
  the bf16 q/k/v projection in the batched FA prefill arm.
