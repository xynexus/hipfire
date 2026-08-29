# Bug hunt wave 2 — the eight dimensions the first pass never reached

Master `0c9e3d252`, nix1. Same method as
[wave 1](2026-08-29-bug-hunt-summary.md): subsystem finders, then every candidate
judged by three independent lenses, majority rule.

**21 candidates → 19 confirmed, 2 refuted. 16 fixed; 3 still open** (all three spec-decode, each with an executable plan — see the end of this doc). Wave 1 had searched only 4 of 12
planned dimensions; this pass covered the other eight (kernels, quant, arch-specs,
specdecode, hotpath, unsafe, silent-drop, recent).

Two dimensions independently rediscovered the same `attention_dflash_wmma_f32`
defect, and the `unsafe` dimension independently rediscovered the uncompilable
`attention_kvarn_routed_batched` that had already been found empirically — three
separate confirmations of the same class.

## Fixed in this pass

| severity | defect | verification |
|---|---|---|
| critical | `attention_dflash_wmma_f32` passed **10 args to an 11-arg kernel** (`dispatch/attention.rs`) — HIP reads the argument count from the code object, not the Vec, so `is_causal` came from adjacent heap bytes: a nondeterministic causal mask on a bidirectional attention | new `kernel_arity` test |
| low | `gemv_mq4g256` passed **7 args to a 5-arg kernel** (`dispatch/gemv.rs`) — `M` and `K` bound from the low halves of two sign POINTERS | new `kernel_arity` test |
| high | `attention_kvarn_routed_batched.hip` **could not compile** — see [its own doc](2026-08-29-kvarn-routed-attention-uncompilable.md) | GPU parity |
| high | `weight_gemv` hardcoded `row_stride: 0`, so every Q8HFQ GEMV output row dotted **weight row 0** | 3/3 verified; now derives from `dispatch_ref()` |
| high | AR decode destroyed every multi-byte codepoint BPE splits across two tokens (U+FFFD before EosFilter could buffer) | test asserts both directions |
| medium | DeepSeek V4 ignored its parsed `routed_scaling_factor`, hardcoding 2.2 — and the `OnceLock` made the first model's value stick for every later load | fixture pinned at 1.5 |
| medium | `hipfire serve` claimed `serve.pid` **before** binding, orphaning the live server from `stop`/`status` | write moved after `TcpListener` bind |
| high | AWQ sidecar dropped for every `FwhtG128` weight; `qt 52` mis-tagged as `Oq8G256`; Opus lm_head rotated with the wrong FWHT | all three now REFUSE (no G128 AWQ kernel, no `Oq8G128` dtype, no batched G128 rotation exists to wire) |
| high | oq4 ragged-K guard present at only 2 of NINE call sites | new invariant test, which found the last two itself |
| low | 33 `WeightRef` literals hardcoding `row_stride: 0` | guarded at the single consumer (`gemv_q8hfq`) instead of triaged |
| low | `gpu_slab_load` config key inert | threaded via the daemon's existing scoped-env seam |

### The prevention that came out of it

`crates/hipfire-rdna/src/kernel_arity.rs` — a `--lib` unit test (so
`no-gpu-ci.sh`'s `cargo test -p hipfire-rdna --lib` actually runs it) that parses
every `__global__` signature in `kernels/src/*.hip` and cross-checks it against
both launch styles: `kernargs![…]` and the raw `params: Vec<*mut c_void>`.

- **516 launch sites checked** of roughly 797.
- The rest are **skipped, not guessed at**: a kernel named by a variable
  (`tile_func_name`, resolved at runtime), a function naming two kernels, or one
  building two params vecs cannot be attributed statically. Guessing produced 8
  false positives in an early draft; requiring a string literal removed all 8.
- A `checked >= 500` floor assert guards against the coverage silently rotting to
  zero while the test keeps passing — the failure mode `no-gpu-ci.sh` documents.
- Validated in both directions: re-introducing either defect fails the test with
  the exact site, and it also catches an argument dropped from a `kernargs!` site
  (which is how a `bits`/`row_offset` regression slipped through during this very
  session's editing).

## Confirmed and still open

### critical

- ~~**AWQ sidecar silently dropped for every `FwhtG128` weight**~~ — **FIXED
  (fail-loud).** `RotationVariant::PlainG128` had a single arm that ignored
  `awq_scale` while `Plain` below it has all four (awq × batched) arms. No G128
  AWQ rotation kernel exists — `rotate_x_mq_awq` is the 256-point transform — so
  it cannot be wired. Since the quantizer stores AWQ weights PRE-SCALED (`W·s`),
  skipping the matching division of `x` is a silent wrong answer; the arm now
  returns an error naming the gap and the remedies. Wiring it properly needs a
  128-point AWQ kernel.
- ~~**`qt 52` expansion mis-tags 128-group data as `Oq8G256`**~~ — **FIXED
  (refuses).** Two things differ, either fatal: the expansion allocates
  `m*k + m*(k/group)*4`, so at group=128 the scale plane is TWICE what an
  `Oq8G256` consumer computes from `ng = k/256`; and the weights carry the
  128-point FWHT while every consumer rotates with the 256-point one. There is no
  correct expansion — merging two 128-groups needs requantization and still leaves
  the rotation wrong — so the load path now refuses instead of mislabelling. The
  existing test's comment claimed the opposite and was corrected; it now asserts
  the length is NOT mistakable for Oq8G256 and that the load refuses.
- **DDTree spec-decode replays a GDN tape the verify forward never wrote** —
  `crates/hipfire-arch-qwen35/src/speculative.rs:12527`. When the batched path is
  declined, the tape-less per-token loop leaves the tape untouched and the replay
  advances DeltaNet state that was never produced. `spec_step_dflash` already has
  the guard (`dflash_use_gdn_tape_replay`); the three DDTree steps do not.

### high

- ~~**`weight_gemv` discards `row_stride`**~~ — **FIXED.** `weights.rs:491`
  hardcoded `row_stride: 0`, so every Q8HFQ GEMV output row dotted **weight row 0**.
  The recovered reachability lens traced the whole chain: `--format q8hfq` is a
  documented production quant emitting padded strides (k=4096 → 4352), the loader
  maps qt 5 → `Q8HFQ`, and llama's live decode calls `weight_gemv` unconditionally
  on `wo` and `w_down` (plus the DSpark drafter). No upstream guard —
  `preflight_gemv_dtypes` passes Q8HFQ — and the kernels have no fallback.
  Fixed by deriving the `WeightRef` from `w.dispatch_ref()` with struct-update
  syntax and overriding only `rotation`/`awq_scale` (which stay `None` because the
  arms below rotate `x` themselves), so a field added later cannot be dropped here
  again.

  **Follow-up: RESOLVED, by guarding the consumer instead of the constructors.**
  33 other hand-rolled `WeightRef` literals still hardcode `row_stride: 0`, and
  triaging each for "can a stride-carrying dtype reach it" was the obvious next
  step. It is not needed: `WeightRef.row_stride` is read by exactly ONE dispatch
  arm, `K::GemvQ8HFQ` (`families/gemv.rs:493`), so `gemv_q8hfq` now rejects a
  stride that does not match the Q8HFQ layout. Any caller that loses the value
  fails loudly with the remedy named, whichever literal it came from. The layout
  became one definition (`hipfire_gpu_types::q8hfq_row_stride`) instead of a
  formula copy-pasted into both loaders that produce a Q8HFQ `WeightTensor`.

- **SWA scores LDS capped at 1024 columns** — `kernels/src/attention_swa_gqa_batched.hip:59`.
  **The reported mechanism was REFUTED by the recovered reachability lens, and a
  different real bug found in its place.** The LDS overflow cannot happen: every
  shipping path first launches `swa_visibility_stage_batched` with
  `block = [window, 1, 1]`, and HIP rejects any block over 1024 threads, so the
  attention kernel is never reached with `n_valid > 1024`. The kernel's own header
  documents that precondition.

  What IS real: cohere2 clamps its window only to `max_seq` (`config.rs:188`) while
  `arch.rs:159-171` yields a default `logical_max` of 2048, so a real on-disk
  artifact — `BLS-Mini-Code-1.0--bf16.hfq`, `sliding_window: 4096` — builds a plan
  window of 2048 and **every sliding layer fails its first staging launch** with
  `hipModuleLaunchKernel ... invalid argument`. That is a hard load/serve-time
  failure on a shipping model, not silent corruption. Fix: clamp at the plan
  boundary (`config.rs:188`, `.min(1024)`) plus an explicit `window <= 1024` check
  with a message naming the cause.
- ~~**DFlash/DDTree Opus lm_head rotates with the wrong FWHT**~~ — **FIXED
  (fail-loud).** It picked `group = 128` for `OqCompactG128` but called
  `ensure_mq_signs` / `rotate_x_mq_batched`, both 256-point. There is no BATCHED
  128-point rotation (`rotate_x_mq_128` is single-row), and this helper also backs
  `dflash_enqueue_verify_lm_head`, so the error is output-affecting rather than
  merely a worse draft. Now refuses, with the upgrade path named.
- ~~**oq4 ragged-K guard landed at 2 of its 4 call sites**~~ — **FIXED, and it was
  2 of NINE, not 2 of 4.** Guards added to qwen2, deepseek4, minimax,
  transformer_loader, lfm2moe, nemotron and zaya. The last two were found by the
  new invariant test, not by grep — my own earlier grep had been truncated by
  `head`. zaya needed the guard at its CALL SITE rather than inside `oq_repack`:
  that returns `Option`, and a `None` would fall through to `linear_dtype` and
  upload raw OQ4 blocks under the wrong dtype — silently wrong instead of loud.
  Pinned by `every_oq4_arch_load_call_site_pre_checks_ragged_k`, which allows an
  explicit `oq4-ragged-guarded-by-caller` marker so a delegating helper must SAY
  so rather than pass by accident.
- **`spec_step_dflash_mtp_tree` row/position desync** — `mtp_compose.rs:1276`
  scatters `accept_dflash + 1` hidden rows but advances position by
  `accept_dflash + accept_mtp + 1`, permanently desyncing the drafter context.
- ~~**AR decode destroys split multi-byte codepoints**~~ — **FIXED.**
  `arch.rs` called `tok.decode(&[token])` per token, so `from_utf8_lossy` replaced
  any codepoint whose bytes BPE split across two tokens with U+FFFD **before**
  EosFilter's hold-back could buffer them — routine mojibake on CJK, emoji and
  accented text. The loop now streams the DELTA of `decode_bytes(&committed)`,
  which is what that function documents itself for ("for incremental UTF-8
  streaming"); this loop was the one place not using it. `committed` is
  pushed/popped rather than reordered, because two `break` paths exit before the
  real push and moving it would change what `pick_next`'s penalties see.
  Pinned by `split_codepoint_survives_byte_delta_but_not_per_token_decode`, which
  asserts BOTH directions: that per-token decode really does emit U+FFFD, and
  that the delta recovers the character.

### medium / low

- ~~DeepSeek V4 ignores the parsed `routed_scaling_factor`~~ — **FIXED.** All
  three dispatch sites now default the `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` override to
  `cfg.routed_scaling_factor` instead of a literal 2.2. The `OnceLock` at
  `forward.rs:3749` had to go with it: caching made the FIRST model's factor stick
  for every model loaded afterwards in the same process. Pinned by asserting the
  test fixture's 1.5 — deliberately not 2.2, so a regression to a literal fails.
- Gemma3 spec-decode leaves rejected draft K/V in the SWA ring; post-wrap the
  stage kernel reads it back (`hipfire-arch-gemma3/src/spec_impl.rs:298`).
- `f5b32ea32`'s FP16 DeltaNet default silently disabled the fused **dense**
  multi-session prefill backend — the dense contract requires FP32 while its
  grouped-MoE sibling accepts both (`qwen35/prefill_batch.rs:1444`).
- ~~**Gemma3 spec-decode leaves rejected draft K/V in the SWA ring**~~ — still
  OPEN; needs the deepseek4 treatment (a per-local-layer staging buffer), which is
  GPU work.
- ~~`hipfire serve` writes `serve.pid` before binding~~ — **FIXED.** The write
  moved out of the CLI and into `hipfire_server::serve_loaded`, immediately after
  the `TcpListener` binds. Previously a serve that never became the live server —
  most obviously a second instance losing the race for the port — still claimed
  the record and orphaned the running one from `stop`/`status`. Both entry points
  (`hipfire serve` and the child `hipfire start` spawns) funnel through that bind,
  so the pid is now written exactly once, by the process that owns the port.
- ~~`gpu_slab_load` config key registered, documented and validated with no
  reader~~ — **FIXED.** It is now threaded through `ModelLoadParams` and installed
  as a scoped `HIPFIRE_GPU_SLAB_LOAD` guard for the duration of the load. No
  arch-loader signature change was needed after all: the daemon already uses
  exactly this seam for two qwen35 knobs (`qwen_residency_load_env`, renamed
  `model_load_env_guards`), because the arch loaders take `&Gpu` + `HfqFile` and
  deliberately do not receive `ModelLoadParams`. Set unconditionally, like both
  siblings — an "only if the env var is absent" gate would have inverted
  precedence against them for no gain, since the places pinning
  `HIPFIRE_GPU_SLAB_LOAD=0` set it on the standalone `tiny_quant_probe`, which
  never reaches these guards. `non_auto_value` maps the `auto` default to `None`,
  so a default config is byte-identical to before.

## Refuted — do not re-file

- **MoE bias-aware top-K sub-wave reduction** (`kernels/src/deepseek4_moe_topk_bias_aware.hip:79`).
- **MiniMax lowered decode gating on layer 0 only** (`hipfire-arch-minimax/src/forward.rs:230`).

## Still open after the fixing pass

Three from this wave, plus DSpark carried over from wave 1. Across BOTH waves:
**32 confirmed, 28 fixed, 4 open** — wave 1 is 12 of 13, wave 2 is 16 of 19.


Three need work I could not validate here, and one is a deliberate refusal:

- **DDTree replays a GDN tape the verify never wrote** (critical,
  `speculative.rs:12527` + two sibling steps). **Not patched deliberately.** I
  traced it far enough to say why, and to save the next person the same dead ends:
  - `GdnTape` has **no populated-marker** — it is preallocated GPU buffers plus
    `base_position`, and nothing records whether a given cycle wrote it.
  - `set_base_position` looks like a capture marker and is **not** one: it is
    called unconditionally at the top of `verify_dflash_block_*` for any tape
    passed in, before the batched path is chosen. Do not build a staleness check
    on it.
  - The non-tape fallback is not "skip the replay": DFlash rewinds from per-token
    DN **snapshots** written by the same batched path, so when the tape is absent
    the snapshots usually are too. Skipping the replay leaves the state
    un-advanced, which is also wrong.
  The correct fix is a `captured_rows` counter set at the real capture site and
  checked inside `replay_gdn` — one chokepoint covering DFlash, all three DDTree
  steps and MTP. That needs the write site instrumented on every capture path and
  a DDTree artifact to validate; a wrong guard here silently corrupts output,
  which is worse than the documented bug.

- **`spec_step_dflash_mtp_tree` row/position desync** (high) — needs
  gather-by-slot for the MTP-accepted hidden row; GPU work.
- **Gemma3 rejected draft K/V in the SWA ring** (medium) — needs a per-local-layer
  staging buffer, mirroring deepseek4.
- **DSpark `--resume`** (low, example-only) — a correct resume needs the DSCK
  format to carry epochs-completed AND `best_eval_loss`; it currently stores only
  the best epoch. `hipfire-train` has no `[[bin]]`, so nothing shipping can hit it.

All three spec-decode items above now have adversarially-checked, executable
plans in [`2026-08-29-remaining-three-plans.md`](2026-08-29-remaining-three-plans.md).
Read each plan's "why the obvious fix is wrong" section first — in ALL THREE the
approach this document originally suggested turns out to be wrong.
