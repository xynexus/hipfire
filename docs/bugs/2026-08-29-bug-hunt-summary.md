# Bug hunt 2026-08-29 — method, results, and what was NOT searched

Master `0c9e3d252`, nix1. Multi-agent hunt: subsystem finders, then every
candidate judged by three independent lenses (refute / reachability /
already-known-or-intentional), majority rule, no finding kept on one vote.

## Results

**13 confirmed defects**, plus 3 gate/CI defects found and fixed by hand.
**12 of the 13 are now fixed; 1 remains open** (status column below). The two
rows previously marked `OPEN — decision` have since been decided and fixed.

| # | severity | where | status | doc |
|---|---|---|---|---|
| 1 | critical | routed KVarN prefill attends to future tokens | FIXED | [`kvarn-routed-prefill-window-wrap`](2026-08-29-kvarn-routed-prefill-window-wrap.md) |
| 2 | critical | GGUF Q4_0 dequant nibble order | FIXED | [`gguf-dequant-q4_0-q5_k`](2026-08-29-gguf-dequant-q4_0-q5_k.md) |
| 3 | critical | GGUF Q5_K `qh` bit selection | FIXED | same |
| 4 | high | routed KVarN attention hardcoded to 4-bit | FIXED | [`kvarn-routed-attention-4bit-stride`](2026-08-29-kvarn-routed-attention-4bit-stride.md) |
| 5 | high | `compose_hfq` logical vs physical extent | FIXED | [`compose-hfq-logical-extent`](2026-08-29-compose-hfq-logical-extent.md) |
| 6 | high | `run_batch_cycle` leaks sessions on early exit | FIXED | [`serving-lifecycle-defects`](2026-08-29-serving-lifecycle-defects.md) |
| 7 | high | image workloads starve (`aging_ms = 0`) | FIXED (restart cap) | same |
| 8 | high | flow-match img2img uses additive noise | FIXED | [`diffusion-img2img-and-samplers`](2026-08-29-diffusion-img2img-and-samplers.md) |
| 9 | high | img2img reuses sigma-scaled latents as noise | FIXED | same |
| 10 | medium | executor v2 swaps two clients' answers | FIXED | [`serving-lifecycle-defects`](2026-08-29-serving-lifecycle-defects.md) |
| 11 | medium | "Euler a" runs deterministic Euler | FIXED (refused + un-advertised) | [`diffusion-img2img-and-samplers`](2026-08-29-diffusion-img2img-and-samplers.md) |
| 12 | low | `selected_prefill_requests` grows unbounded | FIXED | [`serving-lifecycle-defects`](2026-08-29-serving-lifecycle-defects.md) |
| 13 | low | DSpark `--resume` restarts the LR schedule | OPEN (example-only) | below |
| — | — | doc-freshness gates + 92 unrun tests | FIXED | [`doc-freshness-gate-one-way`](2026-08-29-doc-freshness-gate-one-way.md) |

Two of the three criticals are **silent wrong answers**: they produce plausible
output and no error. That is the class this codebase should be hunted for.

## ⚠️ Coverage: only 4 of 12 planned dimensions actually ran

The hunt was cut short by session limits three times. Finders completed for:

- KV cache and attention state
- Model container loading and layout
- Daemon, scheduler and server concurrency
- Diffusion and training subsystems

**Eight dimensions produced nothing at all** — no finder of theirs ever returned:

| not searched | why it matters |
|---|---|
| HIP kernels and dispatch | 864 kernel files; wave32/64, LDS, barriers, arch arms |
| Quantization math | the oq/mq encoders themselves |
| Per-architecture model implementations | `hipfire-arch-qwen35` alone is 69k LOC |
| Speculative decoding | draft/verify state rewind |
| Runtime model execution hot path | RoPE, sampling, chunked prefill |
| Unsafe Rust and FFI | `hip-bridge`, `hsa-bridge`, raw slices |
| The dropped-metadata class, repo-wide | the `awq_scale` pattern |
| Regressions in recently landed work | last ~45 commits |

**13 confirmed defects came from a third of the intended search.** Treat this
document as a floor, not a census. The two GGUF criticals came out of a single
dimension, and the two KVarN defects out of another — both landed on the first
pass of their area, which suggests the unsearched eight are not empty.

## One finding refuted — do not re-file

**TriAttention/CASK eviction ignores `quant_fwht`** (`crates/hipfire-runtime/src/triattn.rs:1715`).
Refuted 3/3, unanimously. The mode selector genuinely has no `quant_fwht` arm, and
`triattn_can_evict_kv_mode` (`load.rs:570-575`) genuinely whitelists fwht2/3/4 —
but the failure is unreachable:

- The only loader that pairs a shared `KvCache` with `eviction: Some(..)` is the
  qwen35 single-GPU path (`load.rs:2975`), and its KV match (`load.rs:2691-2760`)
  **has no fwht arm**. `--kv-mode fwht4` falls to `other =>` at `:2750`, warns
  *"unrecognized 'fwht4', defaulting to asym3"*, and builds a genuine Givens
  asym3 cache with `quant_fwht: false`. Eviction then scores correctly.
- The only loader that builds a real fwht cache is `load_model_pp`
  (`load.rs:3915/3924/3933`), which hard-sets `eviction: None` (`:3755`, `:4062`)
  and documents the refusal at `:3773-3775`.
- Those constructors leave `givens_cos/sin: None` (`kv.rs:2736-2739`), so a fwht
  cache reaching eviction would **panic** on the `.expect`, not corrupt silently.

The whitelist entry is still misleading and worth tidying, but it is not a bug.

## Confirmed but example-only: DSpark `--resume` LR restart

`crates/hipfire-train/src/dspark_train.rs:820` — `train_dspark_loop` discards the
checkpointed epoch, so a resumed run restarts warmup and the cosine decay from
epoch 0. `best_eval_loss` also resets (to `f32::INFINITY`, `:811`), and since
`ckpt_path` is the same `--out` just resumed from, `save_dspark_ckpt` (`:986`)
**overwrites the prior best checkpoint** with the resumed run's local bests — the
disk artifact regresses, not just the printed report.

Scoped down by verification: `train_dspark_loop` and `load_dspark_ckpt` have
exactly one caller, the cargo example `crates/hipfire-train/examples/dspark_train.rs`.
`hipfire-train` declares no `[[bin]]`, and `hipfire-daemon` uses the unrelated
`train_loop`/`ssm_drafter` path. So this is a developer-tool papercut, not
something a daemon/server/CLI user can hit.

The finder's sketched fix is also wrong: the DSCK epoch field holds the **best**
epoch, not epochs-completed (`dspark_train.rs:983-985`, `examples/dspark_train.rs:336`),
so `start_epoch = saved_epoch` would resume at the wrong point on the curve.

## Method note: verification earned its keep

Three lenses per finding, 33 verification agents for the second wave alone. The
pass killed one finding outright and **materially corrected six others** —
`selected_prefill_requests` named a route that cannot leak, the `aging_ms` finding
shipped a primary fix that would have made starvation worse, `compose_hfq`'s
silent-corruption sub-case turned out to be unreachable, and the img2img sigma
finding had the wrong scheduler branch in its arithmetic.

Every one of those would have gone into this document as fact on the finders' word
alone.
