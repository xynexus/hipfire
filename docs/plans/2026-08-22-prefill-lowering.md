# Plan: executor v2 §M2a — prefill lowering

Scoped 2026-08-22 against `master` at `46a0f7b9e` by four independent measurement
passes. Every claim carries a `file:line` that was opened; where a count is
possible there is a count. This subsystem has punished unmeasured estimates four
times in one session, so nothing here is asserted from intuition.

## 0. Read this before costing anything: the premise is false

The parent plan justifies M2a with "an unlowered prefill is by definition one
indivisible quantum, which defeats v2's claim of suspension *between* modules."

**Prefill is already suspendable at layer granularity, in production, today.**

`forward_prefill_chunk` takes `band: Option<&PrefillBandCtx<'_>>`
(`qwen35/prefill_chunk.rs:1885-1899`) carrying `layer_start` / `layer_end` /
`is_first_band` / `is_last_band`. It skips the embed when not first, skips
final-norm + lm_head when not last, and leaves the residual in caller-owned
`pbs.x_batch`. The EP driver **already calls it one layer at a time**
(`qwen35/ep.rs:315-351`, `layer_end: layer_idx + 1`; second site `ep.rs:2468`),
and the invariant is hard-asserted at `prefill_chunk.rs:2040-2046`.

**Two qualifiers, verified independently before this plan landed** — they do not
kill the band option, but they make "~1 day" optimistic:

* `PrefillBandCtx` is **`pub(crate)`** (`prefill_chunk.rs:1885`) with **zero**
  uses outside `hipfire-arch-qwen35`. An executor in the daemon cannot construct
  a band today; it has to be exposed first.
* `ep.rs:317` is inside **`forward_prefill_batch_ep`** (`ep.rs:215`) — the
  expert-parallel / multi-GPU path. So the per-layer drive is proven *there*, not
  on the single-GPU `pp == 1` default that nix1 runs. Driving the default path
  per-layer is additional work, not a switch.

What survives: the mechanism exists, is production code, and is not hypothetical.
What changes: costing it as "~1 day" assumes a band cursor the daemon can already
reach, and it cannot.

So the first thing this plan forces is a **decision, not an implementation**:

| buys | granularity | cost |
|---|---|---|
| band cursor over the existing `band` param | one layer | ~1 day, no numeric-path change |
| full prefill lowering | 4–7 ops per layer | see §5 |

**If a per-layer drain budget satisfies §1.1's realtime contract, M2a is not on
the critical path and should be reordered behind M3.** Decide that in writing
before any code. Otherwise this plan is recommending a rewrite to buy something a
cursor already provides.

And either way: the two heaviest prefill ops are single indivisible launches
whose cost scales with chunk length — the GDN recurrent kernel
(`prefill_chunk.rs:3040`) and batched attention (`prefill_chunk.rs:4937`). A
fully lowered prefill still has a per-quantum WCET proportional to chunk size.
**Lowering alone does not deliver the latency number; a chunk-size cap is
required regardless.**

## 1. There is no precedent, and the obvious reference is a mirage

**No arch in this tree lowers a prefill.** Not partially, not for one arm.

* gemma4 is the only crate holding a `LoweredForward` (`arch.rs:162`, built once
  at load), and its only consumer `forward_step_lowered` takes `token: u32`
  (`gemma4/forward.rs:1875`) — decode, n=1.
* gemma4 **has no batched prefill at all**: `SimpleAr::prefill`
  (`gemma4/arch.rs:184-199`) is `for &token in tokens` calling the decode
  function. Its prefill is "lowered" because its prefill *is* decode. Copying
  that shape onto qwen35 produces a per-token prefill — a large throughput
  regression, not a lowering.
* In-tree confirmation: `hipfire-dispatch/src/pipeline/superop.rs:41` — "only
  **decode** is lowered — prefill is still hand-written control flow."

**But the vocabulary already exists.** `qwen35/lowered.rs:98` emits four
`LayerProgram`s — DeltaNet (7 ops), FullAttn (5), DeltaNetMoe (6), FullAttnMoe
(4), 22 super-ops total — whose variant taxonomy is 1:1 with the arms of the
prefill layer loop, and it is default-ON for decode (`lowered.rs:643`). The
op-sequence design work is **done**. What is missing is a bindings impl that
executes those 22 ops over `n` rows instead of one.

**Lowering by itself buys zero suspension.** `run_layer_program`
(`superop.rs:534-544`) is an 11-line `for op in program { dispatch_super_op(..)? }`.
Zero hits for `Suspend|Yield|Poll::Pending|ControlFlow` across
`crates/hipfire-dispatch/`. The yield point is M3 work and a separate line item.

## 2. M2a owns three loops, not one file

| loop | file:line | span | inline `gpu.*` | reached by |
|---|---|---|---|---|
| `forward_prefill_chunk` | `prefill_chunk.rs:2415` | 5,478 | 222 | single-session `generate` |
| `forward_dense_session_batch_layers_full_precision` | `prefill_batch.rs:2576` | 557 | ~20 | fused multi-session prefill **and decode** |
| `forward_grouped_moe_session_batch_layers` | `prefill_batch.rs:3718` | 891 | ~39 | fused multi-session prefill **and decode** |

Consequences:

* **The parent plan's stated exit — a 4-session × 64-token batched prefill —
  exercises the two `prefill_batch.rs` loops, NOT `forward_prefill_chunk`**
  (`serving-core/qwen35_prefill.rs:916/1058/1355/1495`). Lowering chunk and then
  running that gate proves nothing about the code that changed. Name one owner
  per stage.
* **The session-batch loops are also the fused decode path**
  (`qwen35_decode.rs:1174`, `:1457`). Lowering them is not prefill-scoped and
  cannot be gated by a prefill-only oracle.
* **Direction is fixed: chunk first.** `prefill_chunk` is a private module
  (`mod.rs:86`, no `pub use`) with 4 intra-crate call sites. `prefill_batch` is
  `pub use prefill_batch::*` (`mod.rs:90`) with 67 pub fns and 13 external
  production call sites. Touching batch first makes the diff public API.
* **The two files are not near-duplicates.** Jaccard over called-identifier sets,
  corresponding arms: 0.08 / 0.08 / 0.24 / 0.12. *Within*-file dense-vs-MoE
  pairs: 0.57 and 0.50. **Stage by arm-pair inside one file, never by file.**

## 3. Staging

The work decomposes — but along the arm axis, not the file axis.

### M2a0 — verification first (prerequisite; see §4)

Nothing below is falsifiable without it. **Do not start M2a1 until it lands.**

*Exit:* a prefill probe that FAILS on a deliberately corrupted KV write at
position 0 of a 64-token prefill.

### M2a1 — proving ground: the FA per-token fallback arm *(net −650 lines)*

`prefill_chunk.rs:6042-6087` (46 lines) is a per-token loop over
`run_fa_layer_body` (`:7963-8623`, 661 lines). It runs at **batch=1 with a live
`pos`** — exactly the shape `Qwen35Bindings` already has.

Why this one: `run_fa_layer_body`'s own doc (`prefill_chunk.rs:7960`) says it is
"byte-exact with the FA branch of `forward_scratch_layers`" — the same hand arm
`lower_variant(Q35Variant::FullAttn)` already replaced and was validated against.
**The parity argument is pre-written in-tree.** Every `Qwen35Bindings` field
(`lowered.rs:139-153`) is already in scope at the call site. Zero new fields,
zero trait change.

*Load-bearing:* call `superop::run_layer_program(..)` directly. Do **not** call
`forward_scratch_layers_lowered` (`lowered.rs:683`) — it loops all layers and
appends final-norm + lm_head, which this arm must not do.

### M2a2 — the flag and gate shape *(no numeric change)*

Copy the four-predicate early-return shape from `decode_layers.rs:71-106`, and
emit the trace line **after** the decision, recording the decision rather than
one predicate (`decode_layers.rs:76-92` records why that matters).

**Use a new name, `HIPFIRE_PREFILL_LOWERED`.** `forward_lowered_enabled()` is a
`OnceLock` copy-pasted in five crates (`lowered.rs:645`, `deepseek4/forward.rs:2324`,
`qwen2/qwen2.rs:2063`, `lfm2moe/forward.rs:2047`, `minimax/forward.rs:1195`);
reusing it makes decode and prefill un-A/B-able and moves five arches at once.
Single gate point: `forward_prefill_batch_with_pbs_opts` (`prefill_batch.rs:6010`)
is the sole funnel for all single-sequence callers.

### M2a3 — batched bindings, dense FullAttn only *(the first real cost)*

Add `n_rows` and `pbs` to a **new** `Qwen35PrefillBindings` (do not widen the
decode one); implement the 5 ops of `lower_variant(FullAttn)` against the batched
kernels inlined at `prefill_chunk.rs:4080-6041` (1,962 lines, 86 `gpu.*`, ten
numbered sub-phases).

Trait cost is zero: all 15 `ForwardBindings` methods take exactly
`(&mut self, &mut Gpu, &DispatchCtx, &OpBinding)` (`superop.rs:335`). Shape
travels in the impl struct.

**Do not put the row count in `DispatchCtx`.** It has 3 fields, none per-call
(`context.rs:12-16`), its doc calls it "resolved once at `Gpu::init()`", and it
has 42 `::new()` sites across 13 crates. `ShapeInfo.batch_size` already exists,
is already documented "n tokens for prefill, 1 for decode", and already drives
`BatchGt`/`BatchEq` (`types.rs:431-435`). qwen35 super-ops carry `key: None`
(`lowered.rs:87`) — the kernel key resolves inside the handler, so **the same
four programs serve decode and prefill with no second lowering table.**

### M2a4 — DeltaNet, then the two MoE arms

DeltaNet last among dense arms, not first: 1,663 lines / 72 `gpu.*`, a
sequential per-token recurrent inner loop, in-place mutation of
`dn_state.s_matrices`, and interaction with `gdn_tape` capture — one of the four
hand-path escapes M2b must retire anyway. Sequencing it before M2b means
fighting both.

MoE arms share `prefill_moe_ffn_body_batched` (`prefill_chunk.rs:65-1872`)
behind one `Moe` super-op, so they cost less than their line count suggests.

**Known trap:** `(FullAttnMoe, FullAttention)` has no non-batched fallback — it
hits `_ => panic!` at `prefill_chunk.rs:7884`. A generic per-token binding turns
a loud panic into silently different numerics: the accept-and-miscompute class
M2a exists to prevent, arriving *through* M2a. Preserve the panic explicitly.

### M2a5 — session-batch loops *(separate milestone, not M2a)*

`prefill_batch.rs:2490-3133` and `:3633-4609` are already ~90% decomposed into
ten `pub fn *_layer(..)` op functions (`prefill_batch.rs:477+`), which is why
their inline `gpu.*` counts are ~20 and ~39 against chunk's 222. They are the
**easier** target — and they are also fused decode, so they need a
decode-inclusive gate and their own milestone. Rollback already ships:
`HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` accepts
`auto|serial|fused|fused_dense|fused_moe|grouped_moe`
(`hipfire-generate/src/lib.rs:1414`), and `auto` already falls back to
`SerialReference`.

## 4. Verification — the actual hole, and it is worse than "missing"

**Prefill produces KV and DeltaNet state, not tokens. Nothing in the automatic
tier reads either.**

* `tiny_harness.rs` contains the string "prefill" **zero** times in 1,397 lines.
  `run_ar_hash` feeds one token per call to `qwen35::forward_scratch` — the
  decode path.
* **False coverage, which is worse than none.** `tests/tiny-affected-gate.sh:232`
  maps any `crates/hipfire-arch-qwen35/*` edit onto the qwen3_5 families and runs
  tiny-quant, tiny-state and tiny-spec — all decode-only. So
  `--require-coverage` reports **COVERED for a prefill change it never
  executes**, and every stage in this plan would pass by default.
* `tests/smoke-generate-batch-prefill.sh:517` compares **one token per session**,
  i.e. only the final-position logit. A lowered prefill that writes correct KV at
  position n−1 and garbage at 0..n−2 passes it, then degrades on turn 2.
* **`HIPFIRE_FORWARD_ORACLE` does not exist.** The parent plan names it as M2a's
  exit and as the gate for every numeric-path collapse (plan L1466). It is two
  doc comments (`superop.rs:39`, `qwen35/mod.rs:1894`), the second of which says
  outright it is "advertised in `superop.rs` and implemented nowhere".
  Corroborated at `BUGS.md:961`. Budget it, or name a different gate.
* No primitive to build on: `state_hash|kv_hash|hash_kv|kv_checksum` over
  `crates/` returns **0** hits. Zero `#[test]` in `prefill_chunk.rs`,
  `prefill_batch.rs`, `lowered.rs`, `decode_layers.rs` — and `lower_variant` is
  documented "Pure → unit-testable" with no test.

**M2a0's minimum:**

1. A prefill probe in `tiny_harness.rs` that prefills N tokens and hashes **KV
   cache contents and `dn_state.s_matrices`**, not logits.
2. A reference to diff against — **already in-tree, no new oracle**:
   `qwen35_prefill_active_session` takes a per-token `forward_scratch` branch
   when `replay_as_generated_suffix || hier_enabled`
   (`qwen35_prefill.rs:150-166`), and `qwen35_prefill_owned_session_serial_segment`
   (`:217`) does it unconditionally. That path is *already lowered*
   (`decode_layers.rs:94`), so it validates batched bindings against a lowered
   per-token reference.
3. Copy the oracle shape that actually exists — `HIPFIRE_GEMMA4_FORWARD_ORACLE`
   — rather than the one that does not.

## 5. Cost, honestly

Not "~14.2k lines". The files are 8,623 (`prefill_chunk.rs`) and 6,354
(`prefill_batch.rs`) = **14,977**, but that is the wrong unit. What M2a actually
costs:

| stage | unit of work |
|---|---|
| M2a0 | a KV/DN state probe + wiring it into a gate that currently reports false coverage |
| M2a1 | ~10 lines added, 661 deleted, no new fields |
| M2a2 | a flag and an early return |
| M2a3 | 5 ops against 1,962 lines / 86 `gpu.*` of batched kernel calls |
| M2a4 | DeltaNet 1,663 lines / 72 `gpu.*`; MoE arms share one 1,808-line body |
| M2a5 | separate milestone; already ~90% decomposed |

The dominant risk is not volume. It is that **the gate lies**, so a wrong stage
looks green.

## 6. Stop-lines

* **The band cursor satisfies the latency contract.** Then stop: M2a is not on
  the critical path and this plan should be reordered behind M3 (§0).
* **M2a0 cannot be built** — i.e. no probe can distinguish a correct from a
  corrupted prefill state. Then every later stage is unfalsifiable; stop rather
  than proceed on green-by-default gates.
* **A stage needs `DispatchCtx` to grow a per-call field.** That is 42 call sites
  across 13 crates and means the shape belongs in the bindings impl instead.
* **A generic per-token binding would replace the `(FullAttnMoe, FullAttention)`
  panic** (`prefill_chunk.rs:7884`) with numerics. Preserve the panic; a silent
  miscompute is the outcome this milestone exists to prevent.
* **The work starts touching `prefill_batch.rs` before chunk is done.** That is
  67 pub fns and 13 external call sites — a public-API change wearing a refactor
  costume.
