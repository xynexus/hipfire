# Qwen3.8-Flash-Next (arch 26): continuous batching — plan

**Status:** scoped 2026-11-27, nothing built. Stacked on #495 (batched prefill)
and #496 (MTP); #497 (device repack) is independent. State box: halo gfx1151,
128 GB UMA, `Qwen3.8-Flash-Next-180B--oq4.hfq`, experts paged.

## Where it stands

- **Within one request: batched.** Prompt prefill runs layer-major in chunks of
  up to 512 (`prefill_gpu.rs`, #495): 512 tokens in 10.6 s vs ~88 s per-token.
- **Across requests: not at all.** The server's continuous-batching runner is on,
  but arch 26 fails both gates — it declares no `ContinuousBatching` capability
  (`hipfire-arch-qwen4exp-spec/src/lib.rs:775` registers only `Ingest, ToyModel`)
  and `batch_executor_for` (`serving-core/src/batch_executor.rs:278`) has no arm
  for it, so the daemon reports `batch_prefill_capable: false`. Every request
  takes the legacy path; concurrent requests run one at a time.

A prefill-only executor would NOT help: lfm2-moe has one and is never
batch-routed (no `ContinuousBatching` cap, and the runner has no "batched
prefill + legacy decode" mix). Batched **decode** is the deliverable.

## What the contract asks for

From `BatchExecutor` (`batch_executor.rs`) and the qwen35 / lfm2 implementations:

| method | must |
|---|---|
| `probe` | refuse every state `prefill`/`decode_step` would refuse (pp≠1, MTP head attached, …) — a refusal mid-cycle `fail_all`s the batch instead of falling back |
| `prefill` | per session: fresh / attach-checkpoint / continue-resident (`state_handle`); render the chat template in the daemon (`qwen35_materialize_batch_prefill_prompt` is reusable); leave each session's last-row logits in its own state; emit `…_session_error`* → `…_started` → `…_session_done` (with `state_handle`, `prefix_checkpoints[]`, `kv_capacity`) per session → `…_done` |
| `prefix_hash_preflight` | boundary candidates + `compute_qwen35_prefix_hash` (generic despite the name) |
| `decode_step` | sample each session's stored logits (greedy; think budget, stops, UTF-8 via `decode_marked`), run ONE forward over B rows, write each row's logits back; validate `logical_position` vs resident and vs KV capacity; emit `…_decode_step_session_done` per session + `…_decode_step_done` |

The server side (batch runner, prefix cache, checkpoints index, eviction policy)
is generic and needs nothing. Temperature / penalties / stop sequences / images
already route to the legacy path server-side; the batched path is greedy.

## Memory: why precision comes first

Per session, today (f32 everything):

| state | size |
|---|---|
| GDN recurrence + conv rings (36 layers) | 112 MiB |
| QSA KV (12 layers, 2 KV heads × 256, f32) + indexer keys | 216 MiB @ 4K · 1.7 GiB @ 32K · 6.9 GiB @ 128K |
| PLE dilated ring | 0.4 MiB |

Eight 32K sessions is ~15 GB of state on the same UMA pool the expert cache
pages into (the 8→16 GiB cache step alone was −24% decode). So:

- **KVarN becomes the default KV for this family** (operator decision,
  2026-11-27). 4-bit KVarN (`kv.rs`, `kvarn_bits: 4`, GROUP 128) cuts QSA KV
  ~7×: 1.7 GiB → ~0.25 GiB at 32K.
- **GDN state to f16 is an EXPERIMENT, gated on measurement** — 112 → 56 MiB.
  The f16 kernels exist (`gated_delta_net_f16_batch_seq`, and the routed variant
  below), but this tree has already been bitten by f16 narrowing CADENCE in
  DeltaNet state (the 2026-08-29 KVarN-batched-prefill residual), so it ships
  only if it passes the bar in P0b.

## Phases

### P0a — KVarN KV for the QSA layers (default)

QSA attends a SELECTED, non-contiguous slot set (`qsa_select_attend`:
indexer → top-k blocks → `qsa_gather_kv` → `attention_cold_slots`). Plan:

- store K/V as KVarN tiles per layer (write path: `kvarn_quantize_tile` as each
  128-token tile fills; the open tail tile stays f16 until it closes);
- gather: dequantize the SELECTED slots into the existing f32 `gath_k/gath_v`
  scratch (`kvarn_gather_k_tiles` / `kvarn_dequant_tile` are the building blocks),
  so `attention_cold_slots` is unchanged;
- indexer keys (`ix_keys`, 1×128 per token) — planned at source precision
  because they decide selection; narrowed to f16 after measuring it (P0a log,
  2026-11-28);
- known cost: a scattered pick drags its tile's per-channel scale/zero (the
  GLM-5.3 analysis priced this at ~4× read amplification for DSA). Below 2051
  visible tokens QSA is dense, so short contexts pay nothing.

**Bar:** held-out PPL at 4K and 32K context within 0.5% of f32 KV (the
`perplexity_qwen4exp` / batched scorer path, same corpus as #495); decode argmax
unchanged on the fixture; then the setting flips to default for arch 26.

### P0b — f16 GDN state (experiment)

Swap the per-layer `GdnState.recurrent` to f16 and the kernels to their f16
siblings, behind a schema setting (not an env var). **Bar:** long-context PPL
(32K) within 0.3% of f32, AND a 2K-token greedy decode whose argmax stream
matches f32 for ≥ 1K tokens on the 180B; tiny-state fixture rows recorded.
Fail ⇒ stay f32 and document it. Either way, record the result.

### P1 — per-session state, owned by the backend

Split `TrunkState` (one sequence today) into a `Qwen4ExpSession` = GDN states,
PLE ring + n-gram history, QSA caches (KVarN), cursor, last-row logits, prefix
hash. Ownership: **the backend owns its session map**, reached through a new
`ServingBackend` hook, instead of a `q4x_registry` field on `LoadedModel` — the
qwen35/lfm2 registries are exactly the per-arch `LoadedModel` fields the
model-family onboarding refactor wants gone. Needs:

- allocate (lazy), free (explicit — dropping frees nothing on this GPU stack),
  `fork` for checkpoints (copy GDN + PLE + QSA prefix + history), FIFO eviction
  through the generic `eviction_victims` (SES:2602) and the existing resident
  caps;
- arms for arch 26 in the arena backend mapping (`hipfire-state/src/lib.rs:90`)
  and `release_sessions` dispatch (SES:2831), both routed to the hook rather
  than another hard-coded family arm;
- the single-session serving path keeps working as "session 0".

### P2 — batch prefill (serial per session first)

Each session prefills through the existing single-sequence batched forward
(`prefill_chunks` into that session's state) — prefill is already batched per
session, so fusing ACROSS sessions only pays for short prompts; defer it.
Emits the events above; boundary checkpoints via `fork`. Reuses the generic
render, prefix hash and JSON builders.

### P3 — batched decode step (the payoff)

One layer-major forward over B rows, row b = session b at its own position:

| piece | status |
|---|---|
| dense linears, `lm_head` | `weight_gemm` over B rows — weights read once per step for all sessions |
| GDN recurrence | **exists**: `gated_delta_net_f{32,16}_routed_batch_seq` (per-session `state_ptrs` + `row_session_indices`), the kernels qwen35's fused decode uses; geometry matches (head_dim 128) |
| GDN conv ring | **exists**: `conv1d_silu_split_routed_f32_n` |
| QSA | `qsa_rows` already attends per row; generalise from one cache to a per-row cache |
| PLE | per-row: each session's n-gram history + its own ring through `ple_tail` |
| hyper-connections | per-row already |
| routed MoE | `moe_rows` already groups by expert ACROSS rows — each expert hit by any session is read once per step |
| token outcome | reuse the generic outcome; parameterise vocab/eos (it reads `q35_config` today) |

Shape of the change: give `prefill_chunk_into`'s row functions a per-row state
lookup (session, position, history) instead of one `TrunkState`. B=1 must stay
bit-identical to today's decode path.

### P4 — registration

`impl ContinuousBatching` (start at `max_batch_sessions = 8`, like qwen35) in
the spec crate's `register_arch!`; `batch_executor_for` arm; probe refuses an
attached MTP head (the batched path never speculates; MTP is opt-in legacy);
check the server's hard-coded `state_kinds: ["attention_kv","deltanet_recurrent"]`
(chat.rs:2456) is acceptable for this arch's state; arch 26 into
`KNOWN_RUNTIME_ARCH_IDS`.

### P5 — measurement and admission

- **Correctness:** fixture test in `qwen4exp-gate.sh` — B sessions decoded
  together vs each decoded alone, teacher-forced, logits within the batched
  forward's numeric bound, plus checkpoint fork/attach round-trip.
- **Throughput:** aggregate tok/s and per-session latency at B = 1, 2, 4, 8,
  same-subject and mixed prompts, 8 and 16 GiB expert cache; cold loads per
  step; resident memory.

## What to expect

Decode is page-in bound (~70 ms of a ~120 ms token at 16 GiB, after #497), and
the 2026-10 routing study (#491) found the per-routing miss rate roughly FLAT
in batch size. So batching mostly amortises the DENSE part (~45 ms/token GPU,
paid once per step instead of once per session) plus the overlap between
sessions' experts (same-subject unions 0.70–0.82× of independent). A rough
model gives **~1.6–2× aggregate throughput at B=8**, with per-session latency
somewhat worse than single-stream. That is an estimate to be replaced by P5's
numbers, not a promise. The lever that moves the paging floor itself stays
expert residency: the 2 MiB GTT granule (a 4.25 MB module costs ~6 MB) and a
decode-only expert form (~2× experts per budget) — both independent of this plan
and worth doing first or in parallel.

## Order and size

P0a → P0b (parallel-able) → P1 → P2 → P3 → P4, with P5's fixture test written
alongside P1. Rough size: P0a 1–2 sessions (new: KVarN write + selected-slot
gather), P0b 1, P1 1–2, P2 1, P3 2–3, P4 0.5, P5 1 — **~8–10 sessions**. P0 is
valuable on its own (single-stream memory) whether or not batching lands.

## Risks

- KVarN read amplification on scattered QSA picks above 2051 tokens — P0a's
  32K PPL and latency numbers decide.
- f16 GDN cadence drift — P0b may fail; f32 state then caps sessions by memory.
- Memory contention with the expert cache: more sessions = less cache = more
  page-ins per token. `max_batch_sessions` should be set from P5, not assumed.
- The batched path is greedy-only (a server-side rule today); sampled requests
  keep taking the legacy path.

## Log

- 2026-11-27 **P0a DONE — KVarN for the QSA layers, default KVarN-8.**
  `QsaCache` holds K as KVarN tiles (128 tokens; open block in an f32 window)
  and V as Q8_0; `qsa_gather_kvarn` dequantises only the SELECTED slots into the
  existing f32 gather scratch, so the attention kernel is unchanged. K is stored
  Hadamard-rotated (`rotate_x_mq`, orthonormal per 256 channels) with the query
  rotated to match. Gated by `parity_qsa_kvarn` (in `qwen4exp-gate.sh`) and the
  MTP fixture crossing a block boundary under f32 and KVarN-8.

  180B oq4, held-out wikitext2:

  | KV | 4K PPL | 32K PPL |
  |---|---|---|
  | f32 | 4.0550 | 4.4188 |
  | KVarN-4 un-rotated | 4.0880 (+0.81%, FAILS) | — |
  | KVarN-4 rotated | 4.0411 | 4.4341 (+0.35%, passes; error grows with ctx) |
  | KVarN-8 un-rotated | 4.0345 | 4.4135 (−0.12%) |
  | KVarN-8 rotated (default) | 4.0306 | not run |

  **Why KVarN-4 failed:** the paper's method is "a Hadamard rotation followed by
  dual-scaling variance normalization"; this QSA path had only the second half
  (`hipfire_kvquant::kvarn::hadamard_rows` already documents the gap). Adding
  the rotation turns +0.81% into −0.34%. The qwen3.5 / gemma3 / gemma4 callers
  of `kvarn_attend` already rotate at the call site (head_dim 256); zaya does
  not, and no site rotates head_dim 128.

  **Memory correction:** asymmetric KVarN (K quantised, V Q8_0, indexer keys
  f32) is ~2.9x smaller than f32 at 8-bit K, ~3.5x at 4-bit — not the ~7x
  estimated above. Per 32K session: f32 1.81 GB, KVarN-8 624 MB, KVarN-4
  523 MB. The f32 indexer keys are now ~1/3 of the cache: f16 indexer keys
  are the next cheap win (needs a selection-quality check).

  Default: `kv_cache=auto` -> KVarN-8 for this family; on halo a
  `model_overrides` entry sets `kv_cache: kvarn8` (the server's loader rewrites
  `auto` to the generic `kvarn` = 4-bit, and the global config is `kvarn4`).
  Rotated KVarN-4 measured at 32K (2026-11-28): 4.4341, +0.35% — inside the
  0.5% bar, but its error grows with context (−0.34% at 4K) while 8-bit stays
  flat, for ~100 MB/32K session. KVarN-8 stays the default; revisit 4-bit only
  with a 64K+ measurement.

- 2026-11-28 **f16 indexer keys.** `ix_keys` stored f16 (narrowed on store by
  `qsa_cache_write_f16`; `qsa_pool_norm_blocks_f16` pools and norms in f32).
  201 -> 101 MB per 32K session, so a KVarN-8 session is ~523 MB, not 624.
  `parity_indexer_gpu_vs_cpu` now also pools through the f16 path (max error
  within 4e-3 of the pooled key's magnitude).

  | 180B oq4, KVarN-8 | 4K PPL | 32K PPL |
  |---|---|---|
  | f32 KV (reference) | 4.0550 | 4.4188 |
  | f32 indexer keys | 4.0306 | — |
  | f16 indexer keys | 4.0446 (−0.26%) | 4.4075 (−0.26%) |

  Within the bar at both lengths. The 4K swing between the two key precisions
  (+0.35%) is bigger than f16 rounding "should" cause: the indexer's top-k
  block selection is discrete, so a perturbation flips near-tied picks and
  4K PPL wobbles by several tenths of a percent in EITHER direction (fp16 GDN
  state moved it −0.48% at 4K, +0.007% at 32K). Decide precision changes on
  32K, not 4K.
