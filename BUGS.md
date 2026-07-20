# Bugs To Investigate

This is a lightweight reminder list. Add a short description, or record
revision + file + line number with a one-line explanation. Do not turn entries
into full investigations here.

## Stale DFlash gate guidance

The automatic DFlash gate was removed in `96afc6149`, but nested `AGENTS.md`
files, agent skills, README sections, and active plans still contain mandatory
or canonical `coherence-gate-dflash.sh` guidance, including obsolete `scripts/`
paths. Update normative references to the tiny affected-model gate while
retaining intentional manual-diagnostic and historical references.

## bfp16 GEMM: AMD `mm_bfp.cc` reference kernel fails peano legalization

**Toolchain gap, not ours.** The vendored AMD aie2p bfp16 GEMM
`mlir_aie/include/aie_kernels/aie2p/mm_bfp.cc` fails to compile with the installed
peano/llvm-aie backend: `error in backend: unable to legalize instruction:
<8 x s8> G_BUILD_VECTOR (in function: matmul_vectorized_bfp16)`. The failure is in
the kernel's shuffle/exponent **helper** (`scalarShuffleMatrixForBfp16ebs8` / the
block re-layout for `bfp16ebs8`), **not** the matmul: a minimal `mac_8x8_8x8T`
over `block_vector<bfp16ebs8,64>` compiles fine and emits one native `vmac.f`
(512 MACs/call, int8-rate). Workaround for a hipfire bfp16 GEMM — call the core
`mac_8x8_8x8T` directly and do our own block-layout packing, avoiding the helper.
Confirmed 2026-07-17 on halo (peano/llvm-aie, aie2p target).

## BF16 weights + Q8 KV batched-prefill → garbage on gfx1151 (detailed, by request)

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

- Qwen3 no-output-gate FullAttention faults in fused Q/K/V MQ4 projection;
  split projection should be used until the fused kernel is shape-audited.
- Rust/Axum `hipfire serve` still lacks legacy request cancellation:
  the legacy server sent daemon `{type:"abort", id}` on stream/non-stream client disconnect
  and `{type:"force_answer", id}` after the thinking watchdog, but the Rust
  daemon adapter currently owns stdin/stdout behind one mutable engine during
  generation and the daemon main loop is synchronous while generating. The
  shared protocol now has typed `abort`/`force_answer` messages, and Axum
  streaming drops the daemon when it detects a closed SSE channel after a
  daemon event. Effective mid-prefill cancellation and force-answer still need
  split write/read transport ownership plus generation-loop checkpoints.
- Qwen3.5-397B-A17B HFQM v2 paged grouped-MoE fresh-process prefill still fails
  for B=8 with 8 suffix tokens/session: cache16/cache64 report `hipMalloc: out
  of memory` while paging an expert module, and cache128 was SIGKILLed during
  the run. B=4 with 16 suffix tokens/session passes, so session fanout pressure
  needs a separate audit from total live-row scratch sizing.

## [High] Excessive use of .unwrap() leading to potential panics
- Category: Reliability / Maintainability
- Location: Project-wide (e.g., crates/hipfire-quantize/src/main.rs, crates/hipfire-arch-deepseek4/src/forward.rs)
- Summary: The codebase heavily relies on `.unwrap()` on Results and Options, which can cause the daemon or CLI to crash abruptly on unexpected inputs.
- Suggested fix: Replace `.unwrap()` with proper error handling using `Result` and `?`, or provide descriptive `expect()` messages.
- Scope: Cross-cutting
- Confidence: High

## [Medium] Excessive global state via OnceLock and thread_local!
- Category: Architecture / Maintainability
- Location: Project-wide (e.g., crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-rdna/src/dispatch/mod.rs)
- Summary: Global variables and thread-locals are used extensively for caching and environment configuration, making testing difficult and hiding dependencies.
- Suggested fix: Inject configuration and state through structs/context objects instead of relying on global statics.
- Scope: Architectural
- Confidence: High

## [High] Unchecked rotated-scratch aliasing in runtime weight dispatch
- Category: Reliability / Security
- Location: crates/hipfire-runtime/src/weights.rs
- Summary: Repeated usage of `unsafe` with `gpu.mq_x_rot.as_ref().unwrap().buf.alias()` combines panics and unsafe pointer aliasing.
- Suggested fix: Validate buffer initialization before attempting unsafe aliasing and provide safe abstractions for GPU memory management.
- Scope: Architectural
- Confidence: High

## Collated Findings from Gemini/Docs Review

- [Critical] Global state coupling is spreading across runtime and architecture crates:
  - `OnceLock` / `thread_local!` are used for environment-derived behavior in hot and shared code paths (`crates/hipfire-arch-deepseek4/src/forward.rs`, `crates/hipfire-rdna/src/dispatch/mod.rs`, `crates/hipfire-arch-qwen35/src/qwen35.rs`, `crates/hip-bridge/src/ffi.rs`).
  - This hides explicit configuration inputs and increases hidden coupling.
  - Suggested triage: list all env-backed globals and move them behind explicit config contexts when touching module boundaries.

- [High] Unchecked `unwrap()`/`as_ref().unwrap()` patterns are still concentrated in project-critical paths:
  - `crates/hipfire-runtime/src/weights.rs` around unsafe rotated-scratch aliasing.
  - Recommended: replace with explicit `Option`/`Result` handling and actionable error messages before crash.

- [High] Architectural correctness bug candidates remain explicitly referenced in comments:
  - `crates/hipfire-arch-deepseek4/src/spec_decode.rs` and `crates/hipfire-arch-deepseek4/src/forward.rs`: chunk/ring overwrite edge-case comments.
  - Suggested triage: validate each as still-reproducible and either close with explicit evidence or move to fixed list if already mitigated.
