# Bugs To Investigate

This is a lightweight reminder list of bugs to INVESTIGATE. Add a short
description, or record revision + file + line number with a one-line
explanation. Do not turn entries into full investigations here.

**When a bug is fixed, replace its entry with a one-line tombstone** in
"Fixed — tombstones" at the end of this file, and move any write-up worth
keeping to `docs/bugs/`. Entries that were REFUTED, RETRACTED, superseded or
closed-as-intentional stay in place in full: their value is stopping someone
re-filing or re-deriving them, which a tombstone would lose.

The 2026-08-29 multi-agent hunt is summarized in
`docs/bugs/2026-08-29-bug-hunt-summary.md` — including one finding it REFUTED and
the eight search dimensions that never ran.

## Hunt coverage gaps — what the method could NOT reach

**Written 2026-08-29 by the completeness critic (planned for wave 1, died with its
session limit, finally ran after both waves).** All 12 planned dimensions have now
been searched; this is about the classes no static lens can see.

Full ranking and the mechanical sweeps: `docs/bugs/2026-08-29-hunt-coverage-gaps.md`.

Highest value, all zero-to-cheap:

- **521 wave64/is_cdna branch sites and 349 gfx906 / 72 gfx1201 references have
  never executed** — every hunt ran on nix1 (gfx1103). The fleet already has halo
  (gfx1151) and medusa (gfx906 + gfx1201), and the gates already exist. Zero
  authoring cost.
- **20 files parse `b"HFQM"`; 12 have no `version >= 2` branch.** The confirmed
  `compose_hfq` bug and the recorded HFQ v2 embedded-offset bug are the same class,
  structurally replicated.
- **46 `X` / `X_batched` kernel sibling pairs in `kernels/src`** — "fixed one path,
  not its sibling" was the most common confirmed shape in both waves; this is the
  mechanical version of that hunt.
- **`HFQ_MAGIC`/`HFQ_VERSION` is redeclared in 8 files with the value disagreeing.**
  Every copied constant is a sibling that can drift.
- ~~**`/srv/hipfire/kldrefs` bf16 refs are known bad**~~ — **DONE 2026-08-29.**
  All refs selftested; 8 damaged files (18.4 GB) deleted from the real location
  (`/srv/Public/sadara/.hipfire/kldrefs/`, not `/srv/hipfire/kldrefs` which is
  empty). `qwen3.5-9b` proved HEALTHY and was kept; `qwen3.6-27b` is unverifiable
  (metadata missing `scoring_start`, selftest panics) and was kept. A fourth ref,
  `qwen3.6-35b-a3b`, was found damaged by a different defect (0% agreement on
  every chunk) and deleted.

**Correction to an earlier entry of mine:** I wrote that hipfire-cli's 92 unit
tests "never run anywhere". True for that crate (bin-only, so `--lib --workspace`
selects nothing), but it must not be generalized — `.github/workflows/ci.yml` does
run `cargo test --lib --workspace --locked` on every push, so the ~1900 tests in
library crates are NOT orphaned. Only bin-only crates fall through.

## Wave-2 hunt: 19 confirmed defects across the 8 unsearched subsystems

**Found 2026-08-29 on master `0c9e3d252`. 21 candidates, 19 confirmed 3-lens,
2 refuted. 16 FIXED; 3 still open** — all three spec-decode, each with an
adversarially-checked executable plan in
`docs/bugs/2026-08-29-remaining-three-plans.md`.

Full list, evidence and refutations: `docs/bugs/2026-08-29-bug-hunt-wave2.md`.

**Fixed:** `attention_dflash_wmma_f32` passed 10 args to an 11-arg kernel
(critical — HIP takes the count from the code object, so `is_causal` came from
adjacent heap bytes); `gemv_mq4g256` passed 7 to a 5-arg kernel (M and K bound
from sign-pointer bits); and the uncompilable `attention_kvarn_routed_batched`.
A new `--lib` test, `crates/hipfire-rdna/src/kernel_arity.rs`, now cross-checks
**516 launch sites** against the `.hip` signatures and catches this whole class
without a GPU.

**FIXED since:** AWQ-dropped-for-`FwhtG128` and the `qt 52` `Oq8G256` mis-tag
(both now REFUSE rather than compute silently wrong — at the time neither a G128
AWQ kernel nor an `Oq8G128` dtype existed to wire; **the dtype has since landed**
— qt 54, present in `hipfire-gpu-types` and the gemv dispatch table — see the
2026-08-30 update further down, which supersedes this parenthetical); the DFlash/DDTree Opus lm_head FWHT mismatch (also
refuses); `weight_gemv`'s `row_stride: 0`; AR-decode codepoint destruction;
DeepSeek's ignored `routed_scaling_factor`; `serve.pid`-before-bind; `"Euler a"`
silently running plain Euler (now refused AND un-advertised); the image-workload
starvation (restart cap); and the oq4 ragged-K guard — which was **2 of NINE** call
sites, not 2 of 4, the last two found by a new invariant test rather than by grep.

**Still open, highest severity first:**

- `crates/hipfire-arch-qwen35/src/speculative.rs:12527` — DDTree replays a GDN
  tape the verify forward never wrote when the batched path is declined.
  **Deliberately not patched.** `GdnTape` has no populated-marker;
  `set_base_position` looks like one and is NOT (it is called unconditionally at
  the top of `verify_dflash_block_*`); and the non-tape fallback is snapshot-based
  rewind, so skipping the replay is also wrong. Correct fix: a `captured_rows`
  counter set at the real capture site and checked inside `replay_gdn` — one
  chokepoint for DFlash, all three DDTree steps and MTP.
- `crates/hipfire-arch-qwen35/src/mtp_compose.rs:1276` — MTP scatters
  `accept_dflash + 1` hidden rows but advances position by
  `accept_dflash + accept_mtp + 1`. Needs gather-by-slot; GPU work.
- `crates/hipfire-arch-gemma3/src/spec_impl.rs:298` — rejected draft K/V left in
  the SWA ring; needs a per-local-layer staging buffer, mirroring deepseek4.
- **33 other hand-rolled `WeightRef` literals still hardcode `row_stride: 0`** —
  qwen35 GDN paths, deepseek4, dspark. Each is a bug only if a stride-carrying
  dtype can reach it. Untriaged.
FIXED since: `gpu_slab_load` is now honoured (threaded via the same scoped-env
seam the daemon already uses for two qwen35 knobs), and the `row_stride: 0`
literals no longer need triage — `gemv_q8hfq`, the ONLY consumer of
`WeightRef.row_stride`, now rejects a stride that does not match the Q8HFQ
layout, so any caller that loses it fails loudly instead of dotting weight row 0.

**The three spec-decode bugs above now have adversarially-checked, executable
plans: `docs/bugs/2026-08-29-remaining-three-plans.md`.** Read the "why the
obvious fix is wrong" section of each first — in ALL THREE the approach the
original report suggested is wrong:

- DDTree: mirroring `spec_step_dflash`'s eligibility predicate fails in both
  directions. Eligibility is n-dependent and each DDTree verify uses a different
  n; the tree verify needs a `kv_asym2_tree` term the DFlash predicate lacks; and
  `forward_prefill_batch_single_chunk_captured_opts` writes the tape WITHOUT
  consulting the predicate at all. The fix is a `captured_rows` counter — a record
  of what actually happened — plus the `FullPrefill` fallback `spec_step_dflash`
  already has. It also fixes two sibling bugs of the same class for free
  (`mtp_spec.rs:2727`, `mtp_compose.rs:503/1161`).
- MTP: `rows_to_keep = accept_dflash + accept_mtp + 1` on the contiguous scatter
  copies the token the target just REJECTED. The accepted MTP child lives at tree
  slot `b + (s-1)*mtp_k + k`, so the scatter must gather BY SLOT.
- Gemma3: copying deepseek4's `lo = accepted_len + 1` verbatim leaves exactly one
  rejected draft's K/V in the ring, because gemma3's `accept_len` excludes the
  bonus while deepseek4's includes it.

Recurring shape, again: **a fix applied to one path and not its sibling.**

## [GUARDED 2026-08-30] Four batched flash tiles are head_dim=256-only, unchecked

Second never-executed sweep from the coverage-gaps doc ("46 `X`/`X_batched`
sibling pairs"), ranked by commits touching exactly one side. `asym3`, `fwht3`,
`q8_0` and `f16k_q8v` batched tiles hardcode `d0 = tid * 8` (32 lanes x 8 dims =
head_dim 256) while their asym2/asym4/fwht2/fwht4 siblings got a
`n_halves = head_dim / 128` loop and the kvarn tile derives `dpt` at runtime. At
head_dim=128 the upper lanes read past the head and write past the partials
stride into the next tile's slot — silently wrong, no error. The reduce kernel
was fixed for exactly this and its comment claimed the tiles already agreed.

`head_dim=128` is real here — `qwen3.5-0.8b--oq4++.hfq` (32x128) and the
cohere2-moe `BLS-Mini-Code-1.0--bf16.hfq` both have it — but the KV mode narrows
it: three of the four kernels are deprecated modes and the fourth has no
production caller, so a default server does not reach them. Now REFUSED at the
four dispatch wrappers, with a source-scanning test that fails if the guarded
list and the kernels disagree.

Independently confirmed: `41d597e14`, on the DISCONNECTED pre-fork lineage (not
an ancestor of origin/master; edits a `crates/rdna-compute/` that does not exist
here), hit this on hardware — "threads 16..31 out of bounds -> HIP 700 illegal
memory access that wedged the stream (presented as a ~27-min hang)" — and fixed
it with exactly the `dpt = head_dim / 32` scoped below, noting it is
byte-identical at 256. Cited as evidence and a reference only; per AGENTS.md
upstream is not merged.

→ `docs/bugs/2026-08-30-batched-flash-tile-head-dim-256.md`

## [FIXED 2026-08-30] Three HFQM parsers ignored the container version

Found by running one of the never-executed sweeps from the coverage-gaps doc
("12 of 20 `b\"HFQM\"` parsers have no `version >= 2` branch"). Three were live:
`hipfire-train/src/hfq_patch.rs`, `hipfire-runtime/examples/hfq_split.rs` and
`hipfire-quantize/src/tools/draft_to_mq4.rs` read the version into a discarded
binding and walked a v1 index, so on a v2 artifact — which is what the
quantizer writes and what most of `/srv/hipfire/models` holds — every offset
after the first was wrong. `hfq_patch` also panicked on a truncated file from a
`Result`-returning function. All fixed and pinned by tests.

**The sibling half of that sweep is REFUTED and should not be re-filed:** the
`HFQ_VERSION` constant really does disagree across 6 files (2 vs 1), but the
five that say 1 are writers emitting a self-consistent v1 container. Importing
the canonical constant would MAKE a bug, not fix one.

→ `docs/bugs/2026-08-30-hfqm-v1-only-parsers.md`

## [High] cohere2 with `sliding_window > 1024` cannot serve — every sliding layer fails its first staging launch

**Found 2026-08-29 on master `0c9e3d252`, nix1. Confirmed by source trace plus an
empirical HIP check on gfx1103. PARTIALLY ADDRESSED — the constraint is now stated
once at the shared dispatch with an actionable message instead of an opaque HIP
error. NOT clamped: silently attending over 1024 keys when the model asked for
4096 changes output quality with no signal, which is precisely the bug class this
hunt kept finding. Serving cohere2 at its declared window still needs a chunked
staging kernel or an explicit quality decision.**

Surfaced while recovering a verification lens for a *different* reported bug (the
`attention_swa_gqa_batched` LDS bound, which was REFUTED — see below). This is the
real defect in that area.

- `crates/hipfire-arch-cohere2/src/config.rs:188` clamps the window only to
  `max_seq` (`window: self.sliding_window.min(max_seq)`), and
  `crates/hipfire-runtime/src/layered_kv.rs:151-155` validates only `1..=max_seq`.
  **No 1024 cap anywhere.** `crates/hipfire-arch-cohere2/src/arch.rs:159-171`
  yields a default `logical_max` of 2048.
- A real on-disk artifact hits it: `/srv/hipfire/models/BLS-Mini-Code-1.0--bf16.hfq`
  (HFQM v2, arch_id 25 = cohere2_moe) carries `"sliding_window": 4096`, so the plan
  window is 2048.
- `Gpu::swa_visibility_stage_batched` launches with `block = [swa_window, 1, 1]`
  (`dispatch/attention.rs:5587`), and HIP caps a block at 1024 threads. Verified
  empirically on gfx1103: `hipModuleLaunchKernel` returns `hipErrorInvalidValue (1)`
  **synchronously** for block=2048 and never runs the kernel. The kernel's own
  header states the precondition (`swa_visibility_stage_batched.hip:27`: "assumes
  swa_window <= 1024").
- Result: every sliding layer fails at `calibration_stream.rs:1062` with an opaque
  `hipModuleLaunchKernel ... invalid argument`. A hard load/serve-time failure, not
  silent corruption. Four callers route through the same staging launch (cohere2,
  gemma3 decode, gemma3 batched prefill, gemma4).
- **Why it is not a one-line fix:** clamping to 1024 at `config.rs:188` makes the
  model attend over a 1024-key window instead of the 4096 its config asks for. That
  is a quality change, not a guard. Someone has to decide whether cohere2 should be
  clamped, chunked, or refused with a clear message.
- Cheap improvement either way: state the constraint once in
  `Gpu::swa_visibility_stage_batched` (`window <= 1024`, with a message naming the
  cause) instead of letting all four callers discover it as an opaque HIP error.
- **The originally reported mechanism is REFUTED and should not be re-filed:** the
  LDS overflow in `attention_swa_gqa_batched.hip:59` cannot occur, because the
  staging launch above rejects the oversized window one kernel earlier, on every
  path. Full trace: `docs/bugs/2026-08-29-bug-hunt-wave2.md`.

## tiny-quant is RED on master: 3 `qwen3_5_moe` oq8 cells, all BETTER than baseline

**Found 2026-08-29 on master `0c9e3d252`, nix1 (`gfx1103`). Pre-existing —
reproduced at unmodified HEAD in a scratch worktree with byte-identical numbers,
so it is NOT caused by any in-flight work.**

| cell | measured | baseline | budget |
|---|---|---|---|
| `qwen3_5_moe/kld:oq8` | 0.002806 | 0.020287 | ±0.005072 |
| `qwen3_5_moe/kld:oq8+(calib)` | 0.002369 | 0.005677 | ±0.001419 |
| `qwen3_5_moe/kld:oq8++(calib)` | 0.002369 | 0.005677 | ±0.001419 |

- All three breach on the **GOOD** side — `oq8` measures **7x lower KLD** than its
  baseline. That is a quality improvement the 25% relative tolerance reads as a
  failure, so the gate is red for a good reason to celebrate.
- Scoped to oq8: `oq4`, `oq4+`, `oq4++`, `oq4.25++` and `mq3` on the same family
  all pass, and `qwen3_5_moe_indexed/kld:oq8` passes.
- Almost certainly a stale baseline rather than a live defect, but **what moved
  the number has not been identified** — do not re-record until it is. Recording
  a 7x improvement without knowing its cause would bake in whatever produced it.
- Separately, `tiny-state` is INCONCLUSIVE on this host for an unrelated reason:
  baselines are keyed by HIP version and were recorded at `hip=7.13.26154`, while
  nix1 now runs `hip=7.14.60850`, so 10 cells have no baseline to compare against.

## Spec-decode output is not byte-identical to plain AR decode

**Found 2026-08-27 on master, halo gfx1151. Qwen3.8-27B--oq4.25++ + dflash2.**

Greedy spec decode must be lossless, and is not: `--ar-baseline` reproduces
byte-for-byte, but every speculating config differs from it. The divergence
point is B-dependent (char 697 at B=6, 401 at B=8), so it tracks cycle
boundaries, not context length.

- **Root cause: verify runs the BATCHED forward, AR runs the PER-TOKEN forward,
  and the two are not numerically equivalent.** Documented and deliberately
  accepted at `is_batchable_la` (`qwen35/mod.rs`): |delta logit| ~6e-2 vs ~4e-6
  for pure reordering, 15% top-256 overlap, "anything that needs the two to
  agree bit-for-bit must pin the path explicitly". Spec decode is exactly that.
- **⚠️ `--kv-mode f32` looks IDENTICAL and that is an ARTIFACT** — f32 fails the
  batched-verify predicate, so verify silently falls back to per-token. Tell by
  the clock: f32 spec runs 6.79 tok/s vs AR's 15.56 (2.3x SLOWER), where kvarn8
  runs 21.35 vs 12.40. This also retires the older "diverges even at the f32
  oracle, therefore a verify bug" reading.
- **Two plausible causes are REFUTED, do not re-derive them:** a KVarN block
  sealed with later-rejected tokens (killed — `--kv-mode q8` diverges too, and
  q8 has no block tiling); and per-tile q8 KV scales in batched attention
  (killed — `kv_cache_write_q8_0_batched` scales per position per 32-elem block,
  identical granularity to per-token).
- Layer 0 already differs (1.29e-3, worst row = the last row of the 256-chunk),
  so it is not accumulated drift, and the MoE gate reads 0.000e0.
- Not a drafter problem: tau is measured against the verifier's own argmax.
- Full evidence: `docs/bugs/2026-08-27-spec-decode-ar-divergence.md`.
## zaya's two-stage lm_head fine pass uses plain bf16, not bf16-lut3 (medium)

`lmhead_twostage_serve` (`crates/hipfire-arch-zaya/src/gpu.rs`) coarse-scores all
V rows at q2/q4, host-selects the top-K, then **rescores those rows at plain
bf16**. It should rescore from `Bf16Lut3`.

Lut3 is not merely an on-disk codec — it is decoded IN KERNEL. `gemv_bf16l3.hip`,
`gemm_bf16l3_xf32.hip`, and `bf16_huff_decode.hip` exist, and qwen35 already does
exactly the right thing: `loading.rs:4619` hands the PACKED bytes to the GPU
rather than expanding them, with the measurement recorded right there
(`Bf16Lut3 packed, gemv_bf16l3_xf32  1.95 ms`). So a bf16 lm_head is strictly
worse than a bf16-lut3 one — same numerics out of the kernel, fewer bytes moved,
and lm_head is bandwidth-bound.

The storage side is already flexible: keep it lut3 on disk, or keep the smaller
`Bf16Huff` on disk and transcode to lut3 in memory (`bf16_huff_decode`), then
decode lut3 → bf16 inside the kernel.

Fix: rescore through the packed path, following qwen35's arm. The zaya two-stage
head is env-gated (`HIPFIRE_ZAYA_LMHEAD`, off by default), so this is latent
rather than shipping — but the same reasoning applies anywhere a bf16 lm_head is
materialized.

## [RESOLVED 2026-08-30] zaya's tiny-quant KLD cells measure NOTHING — the protected set was promoted to deprecated Q8

Found 2026-08-30 while recording the 128 missing gfx1151 tiny-quant baselines.

Every zaya KLD cell reports `mean_kld=0` — at **oq4**, a 4-bit weight quant:

    [pass] tiny:zaya:kld:oq4  mean_kld=0 max_kld=0.0001 n_scored=20
    [pass] tiny:zaya:kld:oq8  mean_kld=0 max_kld=0.0001 n_scored=20

The cells DO run — `n_scored=20`, `ldlq_attempts=22` on the ++ arms, and
`zaya/collect` passes with n_tensors=54 — so this is not a plumbing skip. The
model's output distribution simply does not move when its weights are quantized
to 4 bits. For contrast, on the same run:

    llama oq4  0.00829894      qwen3_5 oq4  0.25935006
    zaya  oq4  0.00002298      <- 45x below the next-smallest (nemotron_h 1.04e-3)

and zaya's `+` and `++` values are byte-identical to each other (0.00003716 for
both oq4 arms), where every healthy family separates them.

This is the failure mode #375 documents: a cell that PASSES without measuring
anything. So the seven zaya KLD baselines were NOT recorded — the gate keeps
reporting "no committed baseline" for them, which is honest, instead of a
recorded 0 that reads as coverage.

**ROOT CAUSE, found 2026-08-30 by inspecting the artifact — zaya's `oq4` cell is
not testing 4-bit attention.** Under `--format oq4` its whole attention stack
lands at Q8F16 (8-bit), not Oq4:

    zaya  oq4 artifact:  22 Oq4G256   12 Q8F16   44 F16   1 Bf16Lut3
    llama oq4 artifact:  15 Oq4G256    0 Q8F16    5 F16   1 Bf16Lut3

The 12 Q8F16 are q_proj, k_proj, v_proj_current, v_proj_delayed, o_proj and
router_mlp.out_proj on both layers. llama has ZERO tensors promoted that way.
Coverage is otherwise fine — of zaya's 44 F16, 42 are 1-D norms and the other two
are `[384, 2]` depthwise convs, both correctly left alone.

So zaya's "oq4" and "oq8" cells measure NEARLY THE SAME MODEL, which is exactly
why their KLDs are within 6x of each other (2.3e-5 vs 3.9e-6) where llama's differ
by 135x. The cell is not broken so much as mislabelled: it reports 4-bit coverage
for a model whose sensitive weights are 8-bit.

Not a zaya-spec policy: `hipfire-arch-zaya-spec` uses plain
`default_importance(self.role(tensor))` with no family override, so the promotion
comes from the generic importance/K-map path. The rule is the `q8_router` +
`high_precision_via_ingest` arm in `cli.rs`, which pinned the whole protected set
— routers, attention, gather tables — to `Q8F16` regardless of `--format`. That
predates the Q8 deprecation and is the actual defect: Q8 is deprecated, so the
protected set was being held at a format we no longer ship.

⚠️ Training the tiny zaya would NOT diagnose this, and would hide it: the promotion
is visible statically in the artifact and has nothing to do with the weights being
random.

**FIX (2026-08-30).** The protected set now takes an OQ-family home, chosen so it
is always strictly better than the bulk — which is what "protected" has to mean:

- bulk coarser than 8-bit (`oq4`/`oq3`/`oq2`/mixed) → `Oq8G256`.
- bulk already `Oq8G256` (`oq8`/`oq8+`/`oq8++`) → source precision (`BF16`).
  Promoting to `Oq8G256` here would set protected == bulk and delete the
  protected set outright. No finer-grouped Opus 8-bit exists to use instead:
  `Oq8Plain` is the same G256 unrotated, and both `OqPlus` variants are 4-bit.
- row-gathered tables (`TensorRole::Embed` / `LmHead`) stay `Q8F16`. A RUNTIME
  gap, not a policy: the gather has per-dtype entry points in
  `dispatch/embedding.rs` (f32 / q8 / q4k / hfq4g256 / bf16) and no `Oq8G256`
  one, so promoting `embed.weight [4096, 256]` produced a NON-FINITE KLD on the
  deepseek4 fixture. Add the gather kernel and that exclusion can go.

⚠️ **Deprecated Q8 is NOT fully gone.** A sweep of all 20 fixture families at
`oq4` and `oq8` leaves two distinct survivors, only the first of which is the
deliberate exclusion above:

    deepseek4{,_compressed}/oq4,oq8   embed.weight        <- gather table
    nemotron_h/oq4,oq8                lm_head.weight      <- gather table
    gemma4_moe/oq4                    17 tensors          <- K not 256-aligned
    qwen3_5_moe/oq4                    4 tensors          <- K not 256-aligned

The second group is `expert.*.down_proj` / `shared_expert.down_proj` / one
`o_proj`: protected by role (`ResidualWriter`), but with K not a multiple of 256,
so `Oq8G256` is not representable and they fall through to `Q8F16`. They are at
`Q8F16` today too — this is pre-existing, not a regression. Note they vanish from
the `oq8` column, because the bulk-is-8-bit arm sends them to `BF16` first.

Finishing the deprecation for that group means sending them to `BF16` as well,
which DOUBLES them — and for `gemma4_moe` they are 16 expert `down_proj`, a large
share of a real MoE artifact. That is a size/quality decision, not a cleanup, so
it is deliberately left open rather than folded into this fix. `OqPlusCompactG128`
exists precisely for K that is a multiple of 128 but not 256, but it is 4-bit
(W4A4), so it cannot serve as a protected home above an `oq4` bulk.

The gather test asks the ARCH for the tensor's role rather than matching names.
Name matching is wrong in both directions here: the shared
`is_embedding_table_name` misses deepseek4's `embed.weight`, which is how that
table reached this arm at all, while a widened `ends_with("embed.weight")` would
also swallow qwen35/qwen4exp's `pos_embed.weight`, a position table.

zaya now emits 0 `Q8F16` at both `oq4` and `oq8`, and its 7 gfx1151 baselines are
recorded for the first time. ⚠️ Its gfx1103 rows are now STALE — they were
recorded against the promoted artifact and must be re-recorded on that GPU.

**UPDATE 2026-08-30 — `Oq8G128` is the right home and is now wired, but is not
yet the default.** Measured reconstruction: Oq8G128 is 21.6% better than the
`Q8F16` it replaces at FEWER bits (8.125 vs 8.5), and 8.5% better than the
`Oq8G256` bulk. It serves correctly on gemma4 / nemotron / zaya. It is blocked as
a default by archs that fuse rmsnorm+rotate: qwen35 rotates the attention input
ONCE at FWHT-256 and shares it across every projection, so a 128-basis weight
there needs a G128 fused rmsnorm+rotate plus a split of that shared activation
(without it, KLD 0.83). The protected set therefore stays at BF16 for now. Full
detail: `docs/todo/2026-08-30-oq8g128-protected-set.md`.

⚠️ Do not read the resulting baseline moves as quality. The same change made
`qwen3_5_moe_indexed` `oq8` 6.9× better and `nemotron_h` `oq8` 2× worse; on a
random-init fixture KLD measures perturbation sensitivity, not quality. See
`docs/todo/2026-08-30-tiny-fixture-training-and-qat.md`.

## [RESOLVED 2026-08-29 — re-recorded on halo] 8 gfx1151 baselines are stale — `qwen3_5_moe_indexed` fixture changed shape

**Created 2026-08-27 by #379 (merged). CLEARED 2026-08-29** — recorded on halo,
GPU free. `tiny-quant-gate` for this family: PASS. Confirmed the drift was the
fixture and not the DeltaNet fp16 default that moved the other cells: with
`HIPFIRE_DN_STATE_FP16=0` the KLD is unchanged (oq4 0.067658 vs 0.067886).

#379 raised `moe_indexed_preset`'s `shared_inter` 128 → 256 so the shared
expert's `down_proj` stops being ragged for a 256-wide Opus group. That changes
the fixture's shape, which invalidates its recorded baselines.

The gfx1103 rows were re-recorded in that PR. **The gfx1151 rows could not be**
— recording needs the GPU, and halo's was held by an active `qat-opus-sweep`
(44+ min at the time, alongside a 1d2h `hipfire` and a 23h `hipfire-coexist`).
Contending with a QAT sweep to refresh baselines is the wrong trade, especially
if that sweep is measuring throughput.

| file | stale rows |
|---|---|
| `tests/tiny-quant-baselines.txt` | 7 (`gfx1151 qwen3_5_moe_indexed …`) |
| `tests/tiny-state-baselines.txt` | 1 (`gfx1151 qwen3_5_moe_indexed fp16`) |

**To clear, on halo, once the GPU is free:**

```sh
HIPFIRE_TINYQUANT_FAMILIES=qwen3_5_moe_indexed ./tests/tiny-quant-gate.sh --record
./tests/tiny-state-gate.sh --record   # scoped to the same family
```

Expect the recorded numbers to move substantially. That is **not** a quality
change — the fixture is a different model now, so the values measure a different
target and are not comparable to the pre-#379 ones. The gfx1103 side moved the
same way for the same reason.

Until then those cells report a mismatch or no-baseline. That is visible and
non-fatal — unlike the failure mode #375 documented, where a cell *passes*
without measuring anything — but it does mean gfx1151 has no live coverage of
this family, on the very fixture that just gained the uniform-Oq4 batched-prefill
cell (`docs/todo/2026-08-27-oq4-moe-prefill-coverage.md`).

## [RETRACTED — opt-in by design] HFQ requant and the K % 256 fallback

**Retracted 2026-08-27, same day it was filed.** The fallback is not missing, it
is opt-in. `HIPFIRE_OQ_RAGGED_Q8=1` keeps ragged-K tensors at Q8, and the code
says so plainly: *"Default stays padded-Opus (NPU loader), unchanged … padded
Opus ragged tensors only load on the NPU-native path."* Setting it produced a
loadable artifact (rc 101 -> no panic). I filed this without setting the
documented flag.

**The narrower residual issue is real:** the GPU load path `panic!`s
(`oq4_arch.rs:53`, "OQ4G256 requires K % 256 == 0") on an artifact that is
legitimately NPU-targeted, rather than erroring with what to do about it. Same
class as the MoE prefill panics fixed in #368/#369 — a `panic!` where a
descriptive error belongs, taking the process down with it.

Original entry, wrong as written, follows.

### HFQ requantization ignores the K % 256 fallback (RETRACTED)

**Found 2026-08-27, master `8b0ecc253`. Reproducible.**

The direct path (HF dir → `--format oq4`) correctly keeps tensors whose `K` is
not a multiple of 256 at `Q8_0`, because `Oq4G256` has a 256-wide group. The
**requantization** path (`.hfq` → `--format oq4`, with or without
`--tensor-format`) does not — it applies Opus to them anyway:

| tensor (K=128) | direct | requant |
|---|---|---|
| `linear_attn.in_proj_a.weight` | `qt=3` (Q8_0) | **`qt=34`** (Oq4G256) |
| `linear_attn.in_proj_b.weight` | `qt=3` (Q8_0) | **`qt=34`** (Oq4G256) |

Loading the resulting artifact aborts:

```
hipfire-runtime/src/oq4_arch.rs:53: assertion `left == right` failed:
OQ4G256 requires K %% 256 == 0 (got K=128)
```

- Repro: `hipfire-quantize --emit-fixture qwen3_5_moe_indexed --out hf`, then
  `--input hf --output base.hfq --format bf16`, then
  `--input base.hfq --output out.hfq --format oq4` → `out.hfq` panics on load.
- The quantizer exits 0 and reports "Done", so the artifact looks good until
  something tries to run it.
- Severity: it is a `panic!` in the load path, so it takes the process down —
  same class as the Oq4 MoE prefill panic fixed in #368/#369.
- Surfaced while building an Opus-attention fixture for
  `docs/plans/2026-08-27-is-batchable-la-opus-scope.md`; the shape guard is the
  thing standing between that scope and a testable fixture.

## Batched MoE prefill PANICS on an `Oq4G256` MoE artifact (daemon dies)

**Found 2026-08-27 on master `556249d8c`, nix1 gfx1103. Reproducible, rc=101.**

`Qwen3.6-35B-A3B--oq4.hfq` loaded with `kv_cache: kvarn`, then any prefill long
enough to admit batching:

```
thread 'main' panicked at crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:1282:30:
prefill_moe_ffn_body_batched: unsupported experts[0].gate_up dtype Oq4G256
  — admit predicate should have rejected this layer
```

The panic message is correct about the cause: the path-2 grouped arm handles
`F16 | BF16 | ParoQ4G128` only, and the admit predicate that gates it does not
check the routed experts' `gate_up` dtype for `Oq4G256`.

- **Not KLD-specific.** First hit via `kld_eval build_ref`, but a plain
  `generate` with a ~2.4 KB prompt panics identically. KLD merely triggers it
  every time because it always prefills `n_ctx` tokens.
- **Why it was not seen earlier:** with an f32 KV the batched path is *declined*
  (`forward_prefill_batch: -> per-token forward_scratch loop (batched prefill
  declined) [kv_f32=true]`), so the same model benches fine at pp512. Selecting
  `kvarn` — the non-deprecated family — admits batching and reaches the panic.
- **Severity: it kills the daemon**, so every co-resident model dies with it, not
  just the offending request. A dtype the fast path cannot handle should decline
  to the fallback, not `panic!`.
- Repro + logs: `docs/bugs/2026-08-27-oq4-moe-batched-prefill-panic.md`.

## [NOT A BUG — entry was wrong, corrected 2026-08-21] "GREEDY decode is not reproducible across requests"

**Retracted. Greedy decode is reproducible; the original entry mistook designed
multi-turn accumulation for nondeterminism.** Kept rather than deleted because
the measurement is real and the trap is easy to fall into twice.

- **What is actually happening.** A `generate` with no explicit `session_id`
  falls back to `loaded_model_default_session_id`, i.e. the single shared
  `QWEN35_LEGACY_SESSION_ID`. Consecutive session-less requests are therefore
  turns of ONE conversation, and the qwen35 arm says so in its own words:
  "multi-turn: prefill only the NEW turn tokens, continuing from
  `session.cursor.seq_pos` (KV cache + DeltaNet state are cumulative)"
  (`hipfire-serving-core/src/generate.rs`). Repeating a request is not repeating
  an input — the input grows each time. Different output is CORRECT.
- **The disproof.** The same prompt three times with DISTINCT `session_id`s
  (`s1`/`s2`/`s3`) gives `1764618420aa` three times, byte-identical to an
  isolated single-request run. Session isolation works. `{"type":"reset"}`
  between requests likewise restores it, because it clears the conversation.
- The `session.reset(gpu)` inside `generate` is only a **context-full overflow
  guard** (`seq_pos + prompt_est + max_tokens > max_seq`), not a
  new-conversation detector. That is deliberate.
- Original (wrong) reading, for the record: same request 3× → `1764618420aa`,
  `11845de02f29`, `9755690dde69`, read as "stale KV or reused scratch". The
  numbers are right; the interpretation was not.

**Two things that ARE real and came out of this:**

1. **A/B methodology.** Any harness issuing several session-less generates to
   one daemon is measuring a GROWING CONVERSATION, not a repeated request. Give
   each request a distinct `session_id` (or one generation per process). A
   multi-generation A/B is still valid if both sides run the identical sequence
   — the accumulation then cancels — which is why the executor-v2 M3a/M3b0
   parity runs were unaffected.
2. ~~**Arch-inconsistent semantics, unverified as intentional.**~~ **FIXED
   2026-08-21.** Under identical client behaviour (no `session_id`), qwen35
   accumulated a conversation while llama did not — the same protocol usage was
   stateful on one arch and stateless on another, decided by arch rather than
   deliberately.

   A `generate` naming no `session_id` is now a **one-shot on every arch**: the
   daemon resets the active session after activating it, so qwen35 no longer
   inherits the previous request's conversation. Multi-turn is unchanged and
   becomes explicit — send a `session_id` and state accumulates exactly as
   before (verified byte-identical: turn 2 of an explicit session hashes
   `e5a2bfac6e60` before and after).

   `prefill_already_done` is excluded from the reset: that contract says the
   caller already prefilled this session, so clearing it would discard the
   prefill the request depends on.

## [Low — re-record the baseline] oq4.25++ encoder changed at 8357081d3: +93% KLD on the random-init fixture, but −26% KLD on a REAL model
- **RESOLVED 2026-08-13, and it reverses the naive reading.** Measured on
  Qwen3.5-0.8B (real weights, one Hessian reused across both sides, both scored
  against one bf16-anchor reference):

  | | ppl | mean KLD vs bf16 |
  |---|---|---|
  | bf16 anchor | 15.105 | — (4.0e-10 self-check) |
  | **new selector (master)** | **15.740** | **0.030567** |
  | old selector (`b05f74a79`) | 16.126 | 0.041126 |

  KLD −25.7%, and the new selector recovers 38% of the quantization perplexity
  gap. Two independent metrics agree: `8357081d3` is a genuine improvement.
- **Action: re-record `gemma4_moe/kld:oq4.25++`. Do not revisit the selector.**
  The fixture is seeded random-init, where there is no outlier structure for a
  promotion-set search to find, so it moved the opposite way for a reason that
  says nothing about real models.
- Remaining gap: the model measured is dense (arch 5); the failing fixture is
  MoE. A real MoE is unmeasured — an assumption, not a measurement.
- Also established: `hipfire-quantize` IS deterministic (same binary + inputs →
  byte-identical payload); only HFQ front-metadata/tail key ordering varies.
- Method warning, because it produced a perfect-looking wrong answer first: an
  A/B script that runs `./target/release/hipfire-quantize` without rebuilding
  inherits whatever commit `git bisect` last left it at. Rebuild explicitly, and
  treat an impossibly clean agreement (two weight sets, KLD identical to 18
  digits) as a symptom rather than a result. Full write-up:
  `docs/tiny-quant-gate-8-failures.md`.

## [superseded, kept for the record] oq4.25++ encoder changed output at 8357081d3 — +93% KLD on the gemma4_moe fixture, real-model impact UNMEASURED
- Category: Quantization / encode-side
- Location: `crates/hipfire-quantize/src/codecs.rs` `mixed_clipsearch`,
  `crates/hipfire-quantize/src/ldlq.rs`; first bad commit `8357081d3`
  ("fix(opus): choose the mixed scale and promotion set jointly", 2026-08-06)
- `tests/tiny-quant-gate.sh` cell `gemma4_moe/kld:oq4.25++(calib)` went
  0.003077 -> 0.005952 (+93%, budget ±25%) and has been red since. Bisected
  over 549 commits on the measured VALUE, not pass/fail — the baseline file
  moves across that range. Bit-identical either side of the commit, so it is a
  deterministic step, not drift.
- **The commit predicted the opposite**: "At the shipped oq4.25++ default
  (N_out=3) this is a 0.6% SSE change; do not expect a visible KLD move there."
  It changed the encoder and did not re-record `tests/tiny-quant-baselines.txt`.
- The joint-argmin argument is sound; it minimizes group reconstruction SSE
  while the gate measures KLD. 0.6% SSE -> 93% KLD is the proxy/target gap.
- **Do not act on the fixture alone.** Tiny fixtures are seeded random-init over
  a synthetic token stream, so this shows the encoder's output changed
  materially — what the gate is for — not that real models quantize worse.
  The owed measurement is a real model quantized to oq4.25++ either side of
  `8357081d3`, compared by KLD; that decides re-record vs revisit-the-selector.
- Full write-up, provenance table and the two accompanying vacuous cells:
  `docs/tiny-quant-gate-8-failures.md` postscript.

## [CORRECTED] DeltaNet error attribution — the UPDATE/STORE dominates, NOT the KV dot

**The attribution below is WRONG and is kept only to show what produced it.** It
was measured with synthetic k/q that were NOT L2-normalised, while the real model
normalises them (`fused_qk_l2_norm_scale_f32`, visible in every DeltaNet launch
trace). Unnormalised k has ||k||^2 ~ 43, which inflates the dot product's dynamic
range and with it that term's rounding error. Re-measured with k/q L2-normalised,
beta a sigmoid in (0,1), and a decaying gate:

| term | with unnormalised k (WRONG) | with realistic inputs |
|---|---|---|
| only KV dot f32 | 2.530e-7 (81%) | **1.267e-8 (8.6%)** |
| only UPDATE f32 | 2.033e-7 (65%) | **1.477e-7 (100%)** |
| only TILE f32 | 1.380e-7 (44%) | 8.739e-8 (59%) |
| all f32 | 3.140e-7 | 1.481e-7 |

**The KV dot is ~8% of the error, not 81%.** The state update and its f32 tile
store are the whole term. Every recommendation derived from the old table is void:
compensated summation on the KV dot would target an 8% contributor, so neither
Kahan nor Dekker is worth writing.

**The error is also FLAT in context length, not compounding**, once the gate
decays as it does in a real model — 24 / 96 / 384 tokens gives 1.481e-7 /
1.602e-7 / 1.932e-7, a 30% rise for 16x the tokens. Old error decays out of the
state faster than new error accumulates.

**fp16/fp16 (half-precision ARITHMETIC, not just storage): ~1.5e-3**, four orders
of magnitude worse than f32 and roughly flat in length (1.44e-3 / 2.00e-3 /
1.57e-3). That is the answer to "how little precision does the recurrence need":
f32 arithmetic is required; f16 arithmetic is not viable, independently of what
the state is STORED as.

Three harness artifacts were chased before the inputs were realistic, each of
which produced NaN in the f64 REFERENCE and could have been reported as a
precision finding: a gate allowing alpha > 1, a signed beta inverting the delta
rule into positive feedback, and finally the missing qk normalisation. The
attribution swung by 10x between the first table and the last. **The lesson is
that this attribution is extremely sensitive to the input distribution, so any
future number from this harness needs the inputs checked against what the model
actually feeds the kernel.**

## [SUPERSEDED — see above] DeltaNet multi-step error attributed: the KV dot product dominates, storage is least

Measured 2026-08-12 with `deltanet_error_ablation`, now that both FP64 oracles
are validated. Each precision term is switched independently while everything
else runs in f64, so a configuration's error is attributable to what is left in
f32. Relative L2 error of the STATE against an all-f64 run:

| configuration | 24 tokens | 96 tokens |
|---|---|---|
| all f32 (models the kernel) | 3.140e-7 | 1.727e-6 |
| **only KV dot + reduction f32** | **2.530e-7** | **1.298e-6** |
| only UPDATE f32 (subsumes tile) | 2.033e-7 | 1.062e-6 |
| only TILE f32 (storage alone) | 1.380e-7 | 5.941e-7 |
| only OUT dot f32 | 0 | 0 |

The model is faithful: "all f32" lands at 3.140e-7 against the GPU f32 kernel's
measured 2.997e-7 on the same shape, ~5%. It runs on the CPU but reproduces the
GPU REDUCTION ORDER (4 values per lane, then a 5-level 32-lane halving tree),
which a serial sum would not.

**`kv = <S[r,:], k>` and its reduction tree is the largest single term** — 81% of
the total at 24 tokens, 75% at 96 — and the LDS tile's f32 storage is the
smallest of the three, at 44% / 34%. That inverts where the FP16-vs-FP32 debate
has been aimed: the argument has been about STORAGE width while the dominant loss
is a 128-term dot product summed in f32.

Two rows are structural and the table must not be read as a clean decomposition:
- `only OUT dot f32` is exactly 0 because `out_v` is written to the output and
  never fed back into S, so it cannot move the state at any token count. It does
  move the logits, which this experiment does not measure — a separate question.
- `all f32 EXCEPT tile` equals `all f32` because an f32 update already yields an
  f32-valued result, making the tile's rounding a no-op. **UPD subsumes TILE, so
  the terms are not orthogonal and do not sum** (81+65+44 > 100). The isolated
  storage cost is the `only TILE f32` row, where the update runs in f64 and only
  the store rounds.

Every term grows ~5.2-5.5x for 4x the tokens, i.e. slightly superlinear, matching
the compounding seen end to end.

**Compensated summation on the KV dot: modelled, and it is a real but PARTIAL
fix.** Neumaier compensation carried through the reduction tree (each lane keeps
`(sum, correction)` in f32; the tree combines both halves so the correction
survives each level):

| configuration | 24 tok | 96 tok |
|---|---|---|
| all f32 | 3.140e-7 | 1.727e-6 |
| **all f32 + Kahan KV** | **2.500e-7** (-20%) | **1.127e-6** (-35%) |
| only KV f32 | 2.530e-7 | 1.298e-6 |
| **only KV f32, Kahan** | **1.271e-7** (-50%) | **6.945e-7** (-46%) |

It roughly HALVES the KV term and takes 20-35% off the total, and the benefit
GROWS with sequence length — the right direction for a compounding error, and the
opposite of what dithering did.

It halves rather than eliminates because compensation fixes the ADDS, not the
MULTIPLIES: each `s[c]*k[c]` is rounded to f32 before it ever reaches the sum.
Removing that too needs two-product (Dekker/FMA) compensation, which is a larger
change. So the ceiling on this approach is roughly the halving shown, not the 12x
the f64 oracle achieves.

Cost, for the decision: one extra f32 register per lane and a second shuffle per
tree level — 5 extra shuffles per dot product, on a kernel that is latency- rather
than ALU-bound. Cheap, but not free, on the hot path.

**Two-product (Dekker/FMA) modelled too — it is strictly better in isolation and
buys almost nothing end to end, because it hits the UPDATE floor.** `e = fma(a, b,
-a*b)` recovers what each multiply discards, which compensation of the adds alone
cannot reach:

| configuration | 24 tok | 96 tok |
|---|---|---|
| only KV f32 | 2.530e-7 | 1.298e-6 |
| only KV, Kahan | 1.271e-7 | 6.945e-7 |
| **only KV, Dekker** | **9.582e-8** | **4.019e-7** |
| all f32 + Kahan KV | 2.500e-7 | 1.127e-6 |
| all f32 + Dekker KV | 2.009e-7 | 1.134e-6 |
| `only UPDATE f32` (the floor) | 2.033e-7 | 1.062e-6 |

Dekker cuts the isolated KV term by 62%/69% against plain f32 and beats Kahan by
~40%. But in the FULL configuration it lands on the update term almost exactly —
0.99x it at 24 tokens, 1.07x at 96 — and at 96 tokens it is even a hair WORSE
than Kahan (1.134e-6 vs 1.127e-6). That inversion is not noise to be explained
away: once the dot product is near-exact the remaining f32 errors dominate, the
trajectory shifts, and in a recurrence a smaller error in one term does not
monotonically reduce total divergence.

**So the actionable conclusion changes.** Compensating the KV dot — by either
method — takes the state error to the UPDATE floor and no further. The sequence
that would actually matter is:
1. compensate the KV dot (Kahan is enough; Dekker's extra accuracy is wasted
   below the floor), AND
2. compensate `s = alpha*s + k*delta` and its f32 tile store, which is what the
   floor is made of.

Doing (1) alone buys 20-35%. Neither alone approaches the f64 oracle's 12x.

All CPU-modelled, in the harness whose all-f32 row reproduces the GPU kernel to
~5%. No kernel change is written.

## [CLOSED — no action] DeltaNet f32 precision loss is real, measured, and IRRELEVANT to output

Measured 2026-08-12, and it ends the precision thread. Built a KLD reference with
the validated FP64 oracle, then scored the production f32 kernel against it on a
real corpus (`benchmarks/calib/calib-1m.txt` slice, n_ctx 512, teacher-forced):

    mean_kld(f32 kernel || fp64 oracle) = 3.744e-10

Against the oq4 quantization KLD already on record for this model, 0.039, that is
**~8 orders of magnitude smaller**. The model's own weight quantization swamps the
arithmetic precision by a factor of ~100 million.

So every lever this thread identified should NOT be pulled:
- **no Kahan on the KV dot** (modelled -20/-35% of a term worth 3.7e-10)
- **no Dekker/FMA** (already measured 0.6% WORSE end to end, and now moot)
- **no storage-precision change for accuracy reasons.** FP16 DeltaNet state
  remains purely a CAPACITY decision (19/64 -> 64/64 sessions at width 64); the
  accuracy argument on either side is noise at this scale.

What the thread was right about, and what it was wrong about:
- Right: the f32 kernel really does drift ~7x more from fp64 than FP16 storage
  drifts from f32 (3.5% vs 0.5% state divergence at 120 decode steps). The
  DIVERGENCE is real and reproducible.
- Wrong: the attribution of it. "The KV dot dominates, storage is smallest" was
  an artifact of unnormalised synthetic k/q; with realistic inputs the KV dot is
  ~8% and the UPDATE/STORE is the whole term. See the CORRECTED entry above.
- Wrong: that any of it reaches the output. State divergence is not KLD, and
  nobody had connected them. A recurrence amplifies perturbations, so the
  assumption cut both ways and needed measuring rather than arguing.

**Re-measured across context length, 2026-08-12, because a single 512-token point
cannot distinguish "small" from "small so far".** Same fp64-oracle reference, same
corpus, one chunk each:

**Read the "tokens scored" column carefully — it is `n_ctx/2 - 1`, not `n_ctx`.**
`scoring_start = n_ctx / 2` (`handlers/calibrate.rs:688`): the first half of every
chunk is unscored context warm-up, so a chunk of 512 scores 255 tokens at depths
256..511. An earlier revision of this table asserted `n_ctx - 1` for the first two
rows; that was inferred from the third row's shape rather than read from the runs,
and it was wrong.

| chunk length | tokens scored | context depth | mean_kld(f32 \|\| fp64 oracle) |
|---|---|---|---|
| 512 | 255 (obs) | 256..511 | 3.744e-10 |
| 2048 | 1023 (derived) | 1024..2047 | 1.477e-10 |
| 8192 | 4095 (obs) | 4096..8191 | 2.016e-10 |
| 16384 | 8191 (obs) | 8192..16383 | 7.834e-11 |
| 32768 | 16383 (obs) | 16384..32767 | 1.360e-10 |

`(obs)` = read from the run's `total_scored`; `(derived)` = computed from
`n_ctx/2 - 1`, because that run's output was not captured to a file and only its
KLD survives. The 512 row was re-run to confirm both the count and the KLD
(3.7444e-10, bit-reproducible), which is what established the formula.

**64x of context depth, and the KLD does not move.** Every point sits within ~5x
of every other, the ordering is non-monotonic, and the DEEPEST point (16383
tokens) is 2.8x LOWER than the shallowest. There is no growth term to
extrapolate and no trend to fit — this is scatter around ~1.5e-10, not a curve.
The mechanism is the one the corrected ablation identified: a decaying gate
drains old error out of the recurrent state faster than new error accumulates.

The state is advanced token-by-token across all 16383 tokens within the chunk,
so recurrent accumulation over a long context IS exercised. What is still not
exercised is autoregressive feedback — teacher forcing means a perturbed logit
never changes the next input token. That is the one remaining scope caveat, and
against ~8 orders of margin it does not put the conclusion in doubt.

Measurement note on the last row, because the conditions differ: the 32768 score
ran with `HIPFIRE_QWEN35_PAGED_EXPERTS=0` (all experts resident) while its
reference was built WITH paging, and every other row used paging on both sides.
Paging is a residency mechanism, not a numeric one, so this should not matter —
and empirically it did not: the row lands mid-range among the others rather than
as an outlier, which is itself a small piece of evidence that expert paging is
numerically neutral. The switch was forced, not chosen; see below.

**Flat, and non-monotonic, over an 8x span of context** — the 8192 point is
LOWER than the 512 one. There is no growth term to extrapolate. This is the
end-to-end confirmation of the mechanism the corrected ablation found: a decaying
gate drains old error from the state faster than new error accumulates, so the
divergence does not compound with length. Note the state genuinely is advanced
token-by-token across all 4095 tokens within the chunk, so recurrent accumulation
IS exercised here; what is not exercised is autoregressive feedback.

That last point is the remaining scope caveat, stated so nobody over-reads this:
the eval is teacher-forced, so a perturbed logit never changes the next input
token. The 3.5% figure came from 120 autoregressive DECODE steps. With ~8 orders
of margin and a measured-flat length response, the conclusion is not in doubt.

**CLOSED. Do not re-open on new precision modelling alone.** The question was
"how much precision does the DeltaNet recurrence need, including near max
context", and it is answered: f32 arithmetic is required and sufficient; f16
ARITHMETIC is not viable (~1.5e-3, four orders worse, see the CORRECTED entry);
the f32 error does not compound with context over a 64x sweep; and the whole term
is ~8 orders below the model's own quantization. Re-opening needs a NEW REGIME —
autoregressive long generation is the only one identified — not another
estimate of a term already bounded at 1e-10.

### Side finding: the expert pager leaks, and it is what blocked 16383 at first

Both paged attempts at the 32768 chunk died with
`paged expert residency ... hipMalloc(1.55 MiB), free=15.1 MiB of
total=43008.0 MiB`. The diagnostic part is HOW they died:

| expert cache budget | survived |
|---|---|
| 14336 MB | ~1 hour |
| 6144 MB | ~1 minute |
| pager off (all resident) | ran to completion |

**A smaller cache died faster.** That is backwards for a budget — a smaller cache
should bound residency harder, not exhaust memory sooner — and it is exactly the
signature of the `upload_raw`/`GpuPool` asymmetry already filed as a Tier-1
prerequisite: `Gpu::upload_raw` mallocs directly while `free_tensor` frees into
`GpuPool`, so every eviction parks VRAM in a free-list the next cold load cannot
reach. Smaller cache -> more evictions -> more paging traffic -> faster leak.
Disabling the pager removes every eviction, and the same run then completed.

So this is not "expert-cache pressure" or a capacity limit of the box, which is
how the earlier note in this file described it. It is a leak proportional to
paging traffic, and the three-point cache sweep above is the cheapest reproducer
found so far. Worth attaching to that filing rather than leaving it here.


**FIXED 2026-08-15.** `WeightPager::ensure_expert_module_resident` had two
allocation branches; the `module_requires_host_repack` one used the unpooled
`gpu.upload_raw` while its sibling used the pooled path, and BOTH free through
`gpu.free_tensor` into `GpuPool`. `module_requires_host_repack` is true for
exactly `Oq4G256` / `Oq8G256` / `OqPlusCompact` — every Opus artifact — so a
paging Opus MoE took the leaking branch on every cold load. Switched to
`upload_raw_pooled`.

Verified end to end on the reproducer above: the 32768-chunk score at a 6144 MB
cache, which died in ~1 minute, now runs 1h49m to completion and returns
`mean_kld = 1.360282858575701e-10` — **bit-identical to the pager-off run**, so
the fix removes the failure without moving numerics.

Quantified by `cargo run --release -p hipfire-rdna --example
pool_churn_upload_raw` (the M1a exit measurement):

    pooled    4000 cycles:  total_new += 0,  total_reused += 4000,  VRAM +0 B
    unpooled   200 cycles:  VRAM +419,430,400 B  = 400 MiB stranded

200 unpooled cycles strand 400 MiB; 4000 pooled cycles strand nothing.

Note the tiny gates cannot cover this — `tiny-affected-gate.sh
--require-coverage` reports "no tiny coverage selected for changed paths",
because the tiny fixtures do not page experts. The evidence above is the
coverage.
## [High] The FP32 DeltaNet reference drifts ~7x MORE than FP16 drifts from it

Measured 2026-08-12 with a new FP64-accumulate oracle
(`kernels/src/gated_delta_net_f64acc{,_routed_batch_seq}.hip`,
`HIPFIRE_DN_STATE_F64_ORACLE=1`). FP32 storage, `double` tile and arithmetic,
identical routing and lane mapping — so it isolates the error the f32 kernel
accrues inside its own tile from any storage round-trip.

L2 relative divergence of the DeltaNet state, 35B-A3B, 120 decode steps (pos 144):

| comparison | divergence |
|---|---|
| FP16 storage vs the FP32 kernel | 5.05e-03 |
| **the FP32 kernel vs the FP64 oracle** | **3.51e-02** |

**The reference is ~7x further from fp64 than FP16 is from the reference.** Every
FP16-vs-FP32 KLD figure quoted for this subsystem — including the 2.57e-03 that
kept FP16 opt-in — measures divergence from an accumulator that is itself drifting
harder than the thing being measured.

Why: `gated_delta_net.hip` is float throughout. The per-token update does a
`HD`-term dot product, a cross-lane `__shfl_down` tree, and a multiply-accumulate
into the state, all in f32, and the result is fed back in. Storage format is a
side issue next to that.

What this reframes:
- **FP16 state storage is a second-order concern.** Arguing about 10 vs 24
  mantissa bits of STORAGE while the ACCUMULATION loses more than that is the
  wrong axis.
- Compensated (Kahan/Neumaier) summation in f32 is the obvious lever and costs no
  fp64 rate penalty — the dot product and the `__shfl_down` reduction are both
  ordinary summations. Worth trying before any further storage-format work.
- The oracle is not a serving path: fp64 on consumer RDNA3 runs at a small
  fraction of fp32. It is a correctness reference, measured offline.

Caveat on reading these numbers: a recurrence amplifies any perturbation, so
neither figure is "the error" in an absolute sense — both are trajectory
divergence after 144 steps. The comparison is still apples-to-apples: swapping
f32->f64 accumulation moves the state 7x more than swapping f16->f32 storage does.

**Status 2026-08-12: the ORACLE ITSELF IS NOW VALIDATED, but only the PLAIN
kernel — the 3.5% figure came from the ROUTED one and stays provisional.**

`parity_gated_delta_net_f64acc` checks both GPU kernels against an independent f64
CPU implementation of the recurrence:

| kernel | rel L2 err vs f64 CPU reference |
|---|---|
| `gated_delta_net_f32` | 2.997e-7 |
| `gated_delta_net_f64acc` | **2.497e-8** |

The oracle sits at the FP32 STORAGE floor (~6e-8), which is its design point — it
accumulates in double but still stores f32, so one narrowing at the end is
unavoidable. It is 12x closer to truth than the f32 kernel, and that gap is the
term it exists to isolate.

Two things this caught, both mine:
- The first oracle used `TILE_ROWS 8` where the kernel defines **4**, inferred
  from a stale comment ("TILE_ROWS x 128 floats = 4KB"; 4x128x4 is 2KB) instead
  of read from the `#define`. The dispatcher launches `128/TILE_ROWS` blocks, so
  the blocks overran the row range: **relative error ~1.0, i.e. output unrelated
  to the reference** — while still producing plausible aggregate numbers in a
  serving run.
- The acceptance bound was first set to 1e-15, which failed a CORRECT kernel.
  The bound was wrong, not the kernel.

**The ROUTED oracle is now validated too, so the 3.5% is ESTABLISHED.**
`parity_gated_delta_net_f64acc_routed` drives the routed kernels through
session-major pointer tables against an independent f64 CPU reference, with the
three sessions' rows INTERLEAVED in the batch — a reference that processed each
session's rows contiguously would agree with a kernel that ignored routing
altogether, so the interleaving is what makes the check mean something:

| routed kernel | rel L2 err vs f64 CPU reference |
|---|---|
| `gated_delta_net_f32_routed_batch_seq` | 1.570e-7 |
| `gated_delta_net_f64acc_routed_batch_seq` | **2.585e-8** |

Same shape as the plain pair: the oracle sits at the FP32 storage floor and the
f32 kernel is ~6x worse. Both oracles are now checked against an independent
implementation, so the comparison at the top of this entry rests on measured
kernels rather than on assumption.

## [Medium] FP16 DeltaNet state error COMPOUNDS with sequence length — no bug, but the framing understates it

Investigated 2026-08-12 on the suspicion that a 45x KLD gap between the 2B and
the 35B (5.65e-05 vs 2.57e-03, from `pr/deltanet-fp16-state`) was too dramatic for
a storage-precision change. It is not a bug. Three candidate bugs were ruled out
by measurement:

- **Overflow.** `gated_delta_net_f16.hip:115` is a bare `(_Float16)` cast with no
  scale and no clamp, so FP16's 65504 ceiling applies directly. Measured max|S| is
  **16.2** on the 35B and **13.8** on the 0.8B — an order of magnitude of headroom.
  `over_fp16_max=0`, `nonfinite=0` on both.
- **Arithmetic silently done in FP16.** It is not: the kernel keeps
  `__shared__ float S_tile`, widens on load (`(float)S_global[i]`), does every
  update in f32, and narrows once on store. The "storage only, arithmetic stays
  FP32" claim holds.
- **A dtype/plumbing error.** None found.

What IS true, and what the "storage only" framing understates: **the state is a
recurrent accumulator that gets re-rounded to FP16 on every kernel invocation**,
with round-to-nearest and no error feedback. That bias compounds. Measured on the
35B, FP16-vs-FP32 relative divergence of the state's L2 norm:

| decode steps | seq pos | L2 relative divergence |
|---|---|---|
| 2 | 26 | 2.49e-06 |
| 40 | 64 | **3.22e-05** |

**13x more error for 2.5x more tokens** — superlinear, not a fixed storage cost.

**Update 2026-08-12 — the KLD half of that inference was tested and does NOT
hold.** "A KLD figure measured at one context length understates longer ones" was
a prediction, not a measurement. Scoring the f32 kernel against the fp64 oracle at
chunk lengths 512 / 2048 / 8192 gives 3.744e-10 / 1.477e-10 / 2.016e-10 — flat and
non-monotonic over 8x (see the CLOSED entry above). Two things keep this from
being a contradiction, and both matter: that sweep measures the f32-vs-oracle pair
under teacher forcing, whereas the table above measures f16-vs-f32 state under
autoregressive decode. So the superlinear STATE compounding stands as measured;
what is now known false is the assumption that state divergence carries into KLD
proportionally. The f16-vs-f32 KLD-vs-length sweep remains unmeasured.

So a KLD figure measured at one context length was assumed to understate longer
ones, and a model with more recurrent layers to accumulate more of it. That is
the mechanism
behind "worse on the bigger model", together with the 35B carrying 2.6x more of
its state in FP16's low-precision region (31.3% of elements subnormal in FP16 vs
12.2% on the 0.8B; min |S| 3.1e-14 vs 3.9e-12, and FP16 flushes everything below
~6e-8 — the FP16 runs bottom out at exactly 2.98e-8).

**Dithered (stochastic) rounding was tried and is WORSE — negative result, do not
retry.** `HIPFIRE_DN_STATE_FP16_DITHER=1` narrows with a dither hashed from the
value's own bits and the element index (a pure function of the input, so
spec-decode snapshots still restore exactly what they saved — the property the Q8
path's stochastic rounding broke). It runs in both the single-session and routed
f16 state kernels. Measured, FP16-vs-FP32 L2 divergence of the state:

| decode steps | seq pos | round-to-nearest | dithered |
|---|---|---|---|
| 2 | 26 | 2.49e-06 | 1.83e-05 |
| 40 | 64 | 3.22e-05 | 5.34e-05 |
| **120** | **144** | **5.05e-03** | **2.69e-02** |

Worse at every measured length, and the gap widens with context. The two short
points alone suggested the opposite (growth 12.9x vs 2.9x, fitting to N^2.8 vs
N^1.2) — that fit was an artifact of extrapolating from two points, and the
120-step run falsified it. The lesson is the measurement, not the model.

So the compounding is NOT principally a rounding-BIAS artifact: the dither
removes bias but injects up to 1 ULP of per-step noise, and the recurrence
amplifies that noise faster than it amplifies the bias. The residual error is the
mantissa itself — 10 bits is not enough for this accumulator over hundreds of
steps, whatever the rounding mode.

The flag is left in, defaulting OFF, with these numbers on it: the branch is
uniform and free, and keeping it makes the negative result cheap to re-verify
instead of cheap to re-attempt. What is still untried is error feedback (carry an
FP32 residual, ~1.5x the FP16 size), which cancels rather than randomises the
error — a different mechanism, and the only remaining lever short of keeping the
state in FP32.

## [High] Session release LEAKS its KV and DeltaNet GPU buffers — and blocks sound CoW

Found 2026-08-12 while implementing copy-on-write checkpoints. Sessions are
released with `m.q35_registry.sessions.remove(session_id)`
(`serving-core/src/session.rs:1621`, `:1848`), which drops the
`Qwen35RequestSessionState` — but nothing frees its device memory:

- no `Drop` impl on `KvCache`, `DeltaNetState`, or the session state;
- `GpuTensor` has no `Drop` either (v2-plan risk #1 states this outright);
- the only `free_tensor` in `session.rs` is for `logits` (`:301`). Nothing frees
  `k_gpu`, `v_gpu`, `k_window`, `s_matrices`, `conv_states`.

An `OwnedTensor` RAII wrapper DOES exist (`hipfire-rdna/src/dispatch/mod.rs:377`)
and is simply not used for session state.

Per released session that is ~30 MiB of DeltaNet state (FP16) plus ~6.5 MiB of KV
on a 35B-A3B — and double that for any session that also has a retained Final
checkpoint. It is invisible in the usual way: on 42 GiB of GTT it reads as "the
model got slower" long before it reads as OOM.

**This is the prerequisite for copy-on-write checkpoints, not a detail beside
them.** CoW needs to know when a shared buffer's last referent goes away so the
survivor can free it. On a base where release frees nothing, "sharing" a buffer is
indistinguishable from leaking it twice: the implementation would appear to work —
tests would pass, memory would look fine relative to today — precisely because the
system already never frees. That is a fake CoW, and the failure mode when
ownership is later added is a use-after-free on a buffer some other session is
still reading.

Order of work: give session state real ownership first (`OwnedTensor` or an
explicit release path on the registry remove), prove it with the VRAM slope
sampled in a long multi-session run, and then build CoW on top. The acceptance
test for the CoW step already exists — `HIPFIRE_KVARN_DUMP` compares two sessions'
K state numerically, so a session reading a buffer another session wrote shows up
as a diff rather than as plausible text.

## [Medium] Batch-64 collapse is GTT exhaustion from per-session DeltaNet state, not batching

Profiled 2026-08-12 at widths 16/32/64 on `Qwen3.6-35B-A3B--oq4` (kvarn KV,
paged experts, 8 GiB expert cache, max_seq 512):

| width | sessions ok | tok/s |
|---|---|---|
| 16 | 16/16 | 5.66 |
| 32 | 32/32 | 5.54 |
| 64 | **19/64** | 2.55 |

The full error — truncated in the earlier sweep, which is why this looked like a
batching problem — names the cause exactly:

```
clone qwen35 checkpoint dn.s_matrices[14] alloc:
  hipMalloc(2097152 bytes = 2.00 MiB), free=10.6 MiB of total=43008.0 MiB
```

Ten MiB free of 42 GiB. It is an out-of-memory, and the allocation that fails is a
**checkpoint clone of DeltaNet state**, not anything KV or expert related.

**Per-session cost, from `text_config` (`linear_attention: 30, full_attention: 10`):**

| item | size |
|---|---|
| DeltaNet state — 30 layers x `[524288] F32` | **60 MiB** |
| KVarN KV — 10 layers (records + f32 window + Q8 V) | ~6.5 MiB |
| checkpoint clone of the DN state | **+60 MiB** |

**The recurrent state is ~9x the KV cost per session, and the checkpoint doubles
it.** Concurrency on a hybrid model is therefore bounded by DeltaNet state, not by
the KV cache — which is the opposite of where capacity planning usually looks, and
the opposite of where this plan's own 0.4 analysis looks (expert bytes and KV).

**FIXED for the collapse by `HIPFIRE_DN_STATE_FP16=1`, measured 2026-08-12.**
FP16 state halves the 60 MiB to 30 MiB and that is enough to make 64 sessions fit:

| width | FP32 state | FP16 state |
|---|---|---|
| 16 | 16/16, 5.66 tok/s | 16/16, 5.96 |
| 32 | 32/32, 5.54 | 32/32, 6.25 |
| **64** | **19/64, 2.55** | **64/64, 6.31** |

Zero allocation failures at any width, and achieved decode widths now reach 44
(against ~20 before). Throughput also becomes monotonic in width instead of
collapsing.

FP16 state is opt-in (`hipfire_env::DN_STATE_FP16.flag()`); it was briefly made
default on 2026-08-09 and reverted the same day because surviving Q8 dispatch arms
faulted on half-size state. Those kernels and callers have since been deleted, so
that blocker is gone; what still holds the default is that the supporting evidence
is one prompt on one model. This measurement is a second, independent reason to
want it (capacity, not just accuracy) and should be weighed in that decision.

**Checkpoint lifecycle, traced 2026-08-12 — and CoW is NOT the lever it looks
like.** The allocation that fails is a `Qwen35PrefillCheckpointKind::Final`
checkpoint, created for EVERY session after batch prefill
(`qwen35_prefill.rs:1924`), via
`sequence_state_arena_checkpoint_session_state(source -> dest)`, which keeps BOTH
the live session and the snapshot. The default eviction policy is
`SequenceStateEvictionPolicy::ManualReleaseOnly`, so those snapshots are never
reclaimed automatically. Per session that is ~67 MiB live + ~67 MiB retained.

Note `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS=0` does NOT suppress it — that gates
SemanticBoundary checkpoints only, and the Final one is unconditional. Verified:
the collapse reproduces identically with boundary checkpoints off.

Copy-on-write would help far less than it appears, because the two halves behave
differently once the snapshot exists:

| state | live session's writes | CoW value |
|---|---|---|
| KV (~6.5 MiB) | append-only PAST the checkpoint cursor; `[0, cursor)` is never rewritten | shareable permanently — a real saving, but only ~10% of the session |
| DeltaNet (60 MiB FP32 / 30 FP16) | the recurrent matrix is OVERWRITTEN every step | first decode step materializes the copy — CoW DEFERS it, and since every session decodes, peak memory is unchanged |

So CoW buys the KV tenth and defers the DeltaNet nine-tenths. It does not reduce
the peak that OOMs. The levers that actually move it, in order:
1. **FP16 DeltaNet state** — measured above, takes 64 sessions from 19/64 to 64/64.
2. **Release the Final checkpoints.** They are retained under ManualReleaseOnly
   for the process lifetime; nothing reclaims them.
3. **Do not take a Final checkpoint per session** when no prefix reuse will
   consume it — it is a snapshot for resume, and a batch of one-shot completions
   never reads it back.
- `rocm-smi --showmeminfo vram` is useless here: it reported 80-93 MiB of 256 MiB
  across the whole run, because that is the dedicated carve-out, not the 42 GiB
  GTT pool the allocator actually draws from. The allocator's own
  `free=X of total=Y` message is the only truthful source on this box.

Not yet checked: `PrefillBatchScratch` is sized from `pbs.max_batch`, so per-round
scratch (activations, fa_q/k/v, logits) also grows with width and may be a
co-factor at 64. The DN clone is what actually failed, but it failed with 10 MiB
left, so whatever else grew is complicit.

## [Superseded by the above] KVarN dense arm diverges from serial far more than the Q8 control

Found 2026-08-11 once the fused dense path was reachable. Same model
(`Qwen3.5-0.8B-Base--oq8`), same fused dense backend, 48-token greedy, fused vs
serial:

| KV | prompts matching serial |
|---|---|
| q8 (control) | 3/4 |
| **kvarn (ported arm)** | **0/4** |

One kvarn divergence starts at the FIRST token — serial answers "The capital of
France is **Paris**." while fused emits a completely different reasoning-style
response. That is not the late near-tie signature the grouped-MoE arm showed,
where kvarn and Q8 diverged at byte-identical positions.

So the dense KVarN arm is NOT yet at parity with its baseline, unlike the
grouped-MoE arm which is. Do not enable dense KVarN for production on this
evidence.

**Narrowed 2026-08-11 — the PREFILL arm is implicated, and it is not the flush.**

| config | prompts diverging from serial |
|---|---|
| kvarn, fused prefill + fused decode | 4/4, one from the FIRST token |
| kvarn, **serial** prefill + fused decode | 3/4 — the token-0 case is FIXED |
| q8, fused prefill + fused decode (control) | 1/4 |

Three things this rules out:
- **Not cross-session contamination.** The independence test passes: the probe's
  output is byte-identical (755 chars) whether batched with one set of companions
  or a completely different set. State does not leak across rows.
- **Not the block flush.** These are 48-token generations from short prompts, so
  no session reaches position 127 and gather+quantize never fires.
- **Not the shared kernels.** The routed window write, routed attention and flush
  executor are the same code the grouped-MoE arm uses, and that arm matches its
  Q8 baseline at byte-identical divergence positions.

**Ruled out so far (each by measurement, none by argument):**

| hypothesis | how it died |
|---|---|
| cross-session contamination | independence test PASSES — probe byte-identical (755 ch) across different companions |
| the block flush | 48-token generations never reach position 127; gather+quantize never fires |
| the shared kernels | routed write / attention / flush are the SAME code the grouped-MoE arm uses, and that arm matches its Q8 baseline |
| a dense-vs-MoE difference in the layer body | the two `if let Some(kvarn)` arms diff by COMMENTS ONLY — functionally identical |
| a second, unported KV write site in the dense body | both layer functions have exactly one write + one attention per KV mode |
| the KVarN FWHT rotation | `prefill_chunk.rs` rotates K/Q at `head_dim == 256` and `prefill_batch.rs` does not — but disabling it on BOTH sides (`HIPFIRE_KVARN_ROTATE=0`) leaves the divergence at 4/4 |

The rotation asymmetry is real and worth fixing on its own — the routed batch path
has no KVarN rotation while the chunked path does, and both test models are
head_dim 256 (record size 17664 B). It is simply not what causes this.

**Root cause NOT found.** Text-level A/B has run out of resolution: every
remaining hypothesis needs to see the actual K values. The next step is numerical,
not another parity run — dump a session's KVarN window and records after a fused
prefill and after a serial prefill of the same tokens and diff them. That
localizes it to the write, the records, or the read in one experiment instead of
one guess per run.

## [Superseded] No available Qwen3.5 artifact can exercise the fused DENSE batch path — all VL-wrap

Found 2026-08-11 while trying to verify the KVarN dense port. Three non-AWQ dense
artifacts were quantized specifically for this and none reaches the fused dense
backend:

| artifact | source | result |
|---|---|---|
| `Qwen3.5-0.8B--oq4.hfq` | HF snapshot | serial (4.00x launches at width 4) |
| `Qwen3.5-0.8B--oq8.hfq` | HF snapshot | serial |
| `Qwen3.5-0.8B-Base--oq8.hfq` | `.hfa`, quantizer reports `Architecture: qwen3_5 (id=5)` | serial |

Every one logs `qwen3.5-vl text wrapper: mrope_interleaved=true` at load, so the
runtime wraps it as VL regardless of the arch the QUANTIZER reports, and
`is_qwen35_dense_arch_id(m.arch_id)` is false — the first term of the fused dense
decode selection. Note the quantizer and the runtime disagree about the
architecture of the same file, which is worth a look on its own.

**Not a KVarN problem.** The Q8 control on the same model is also 4.00x serial,
i.e. the path is refused in the mode it was built for. Two earlier hypotheses were
tested and killed the same way: AWQ pre-scaling (removed, still serial) and an
unaccepted weight dtype (`oq8` maps to the accepted `Oq8G256`, still serial).

Consequences:
- the KVarN **dense** arm is code-complete but unexercised; the grouped-MoE arm is
  the verified one.
- `docs/plans/2026-08-09-v2-daemon-module-major-multistream.md` names
  `qwen3.5-0.8b--oq4++.hfq` as the first-demonstration interactive model and calls
  it "arch 5 (dense)". It is not — it VL-wraps too. M3's demonstration needs a
  different model, and "dense is correct here: it isolates the scheduling claim
  from the MoE claim" does not hold with this artifact.

To verify either, a genuinely non-VL dense model is needed. Every Qwen3.5 variant
on hand (`0.8B`, `0.8B-Base`) carries the mrope/VL metadata that triggers the
wrapper.

## [Low] Fused grouped-MoE batch diverges from serial decode — systematic, NOT contamination

Found 2026-08-11 while validating the KVarN port, using the shipped Q8 path as a
control. Same model, same KV mode, same greedy decode, same prompts; the only
variable is serial (batch 1) vs fused (batch 4). Longest common prefix of the
outputs, 200-token generations on `Qwen3.6-35B-A3B--oq4`:

| prompt | kvarn | q8 (shipped) |
|---|---|---|
| bicycle derailleur | 580 chars | **580** |
| water cycle | 17 | **17** |
| printing press | 959 | 1143 (exact) |
| refrigerator | 31 | **31** |

**Three of four diverge at the byte-identical position under two different K
formats** (4-bit var-norm records vs Q8). That rules out KV quantization as the
cause and localizes it to the shared fused machinery — the routed batched
attention/MoE kernels or their reduction order — not to either KV path.

Two prompts diverge after ~17 and ~31 characters, i.e. the fused and serial
outputs are essentially different texts. Greedy decoding does amplify a near-tie
into a different continuation, and every output stays coherent, so this is not
obviously corruption. But a 1.7% common prefix is a large effect to attribute to
rounding, and it is worth deciding which it is before leaning harder on the fused
path for throughput.

Not caused by the KVarN port: the control run is on the shipped Q8 path with the
port gated off. Filed separately so it is not mistaken for KVarN fallout.

Next step if pursued: a logit-level comparison rather than text. Note
`HIPFIRE_FORWARD_ORACLE`, which `superop.rs:39` advertises for exactly this
("available for dual-run diffing"), is **not implemented** — the name appears in
that doc comment and nowhere else in the tree.

**DOWNGRADED 2026-08-11 (Medium -> Low): sessions are independent.** The test that
separates "different but valid" from "rows contaminate each other" is whether a
session's output depends on WHO ELSE is in the batch. Same probe prompt, greedy,
one daemon lifetime, batch 4 both times, only the other three rows changed:

| KV | probe alone | probe + companions X | probe + companions Y | X vs Y |
|---|---|---|---|---|
| kvarn | 776 chars | 762 | 762 | **byte-identical (762)** |
| q8 | 784 chars | 769 | 769 | **byte-identical (769)** |

Under both KV modes the probe's output is unchanged by its batch companions. So
the fused path does not leak state across rows, and it is deterministic: the
serial-vs-fused difference is the same 17-character divergence point regardless
of KV format AND regardless of batch content.

That makes this a systematic difference between two implementations of the same
math — reduction order and precision in the routed batched kernels versus the
per-session ones — rather than corruption or nondeterminism. Greedy decoding
turns one near-tie into a different continuation, which is why a benign numeric
difference presents as 1.7% common prefix.

Still worth a logit-level check if the fused path is ever leaned on for
throughput, but it is not a correctness blocker and it is not new.

## [RESOLVED 2026-08-30 — re-traced, kept for the record] `kv_cache = "auto"` bypasses the KV deprecation gate and is the shipping default

**The coupling this entry demanded was honoured and the fix landed.** The entry
said "the fix is coupled to the kvarn port and should not be applied alone";
`load.rs:854` now normalises `auto` (and the empty string) to **kvarn** before
anything else reads it, with the reasoning inline. Re-traced 2026-08-30 while
fixing issue #386:

- `q35_kv_mode` has exactly ONE producer, `load.rs:2938`, and it is assigned the
  already-normalised `kv_mode`.
- `qwen35_allocate_session_state` reads the mode only from `m.q35_kv_mode`, so
  the literal `"auto"` cannot reach a KV constructor.

Two leftovers, neither reachable but both worth knowing about before someone
adds a caller: `session.rs:1773`/`:1782` still carry `"auto" | ""` arms that
build **asym3** (at head_dim 256) or **q8**, and `load.rs:3028` still matches
`"q8" | "int8" | "auto" | ""`. They predate the normalisation and now disagree
with it — a new path that hands a raw config value straight to session
allocation would get a deprecated mode, silently, which is exactly what this
entry was about. They are commented as such rather than deleted, since removing
KV construction arms wants a GPU gate.

The original entry follows, unchanged, because its *reasoning* is what sequenced
the fix correctly.

## [Original, superseded] `kv_cache = "auto"` bypasses the KV deprecation gate and is the shipping default

Found 2026-08-11 while auditing the KV deprecation. The deprecation added in this
branch refuses `q8`/`asym3`/etc **by name**, but the default path never presents a
name it recognises:

- `default_kv_cache()` returns `"auto"` (`hipfire-config/src/lib.rs:104`), and it
  is the `#[serde(default)]` for the config field (`:310`). `routes/chat.rs:3250`,
  `:3297` and `hipfire-cli/src/commands/chat.rs:333` also send it literally.
- `"auto"` is not in `DEPRECATED_KV_MODES`, so `reject_deprecated_kv_mode`
  (`serving-core/src/load.rs:537`) passes it.
- It then matches `"q8" | "int8" | "auto" | ""` (`load.rs:2745`) and builds a **Q8**
  cache — or `new_gpu_asym3_capped` at `head_dim == 256` (`load.rs:2491`,
  `:3634`, `session.rs:1717`). Five construction sites resolve `auto` to a
  deprecated mode.

So an operator who sets nothing gets exactly the mode the gate exists to refuse,
with no warning. An operator who names it explicitly is refused. Note the empty
string is NOT affected: `load.rs:773` normalises `""` to `fp32` *before* the gate,
so this is specific to the literal `"auto"` — which is the default.

**The fix is coupled to the kvarn port and should not be applied alone.** Pointing
`auto` at kvarn is the obvious correction and matches the stated intent (kvarn is
the default; asym/q8 deprecated), but the fused grouped-MoE prefill and decode
paths **hard-require Q8** (see the entry below). Today `auto -> q8` is the only
reason batched MoE decode is reachable at all; switching `auto` to kvarn would
silently disable fusion for every default deployment — trading a naming
inconsistency for a ~2x throughput regression at width 16. Sequence it after the
kvarn port, or land both together.

## [High] Grouped-MoE fused prefill-session batch requires Q8 KV — blocks batching on kvarn
- Category: Capacity / KV-mode coupling — the first real dependent the KV
  deprecation surfaced
- Surfaced 2026-08-10 once the KV sizing fixes (`21aca50bb`, `f2d59b442`) made
  batched prefill stop being memory-bound. The error is explicit:
  ```
  qwen35 grouped-MoE fused prefill-session batch backend failed:
    "grouped MoE session fused prefix row 0 must use Q8 KV state for the MQ4
     control path"
  ; use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial
  ```
- **This is the deprecation working as intended.** Q8 is now gated at load
  (`6a4e32b68`), and this path hard-requires it — so the fused grouped-MoE batch
  backend is a concrete port target for the kvarn migration, not an unrelated
  bug. It is exactly the "break it and the breakage names what needs fixing"
  outcome that was the point of gating rather than deleting.
- **The suggested fallback does not work.** Setting
  `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto` on the server process leaves the
  error unchanged at batch 4 and batch 16. Either `auto` still selects the fused
  backend, or the env does not reach the spawned daemon — untested which.
  `serial` is untried.
- Consequence: batch >= 2 on the 35B remains blocked, but the blocker has moved
  from memory exhaustion to a single KV-mode coupling with a named owner.
- **`serial` IS a viable interim, and it produced the first multi-stream
  generation on this model (2026-08-10).**
  `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial` with kvarn:

  | batch | ok | aggregate tok/s | per-stream tok/s |
  |---|---|---|---|
  | 2 | **2/2** | 7.86 | 3.93 |
  | 3 | **3/3** | 7.19 | 2.40 |
  | 4 | 0/4 | — | fails |

  (`auto` does NOT work — unchanged error. Only `serial` does.)
- **The Q8 coupling is in TWO paths, not one, and the second has a batch
  threshold.** Under `serial` the prefill error disappears and the SAME
  requirement reappears at decode:
  ```
  batch decode: qwen35 fused grouped-MoE native decode advance:
    "grouped MoE session fused prefix row 0 must use Q8 KV state ..."
  ```
  It is clean at batch 2-3 and fires at batch 4, so the fused grouped-MoE decode
  advance appears to engage at batch >= 4 and hard-requires Q8. The port target is
  therefore both the fused prefill AND the fused decode path.
- **Throughput result worth noting on its own: batching buys nothing here yet.**
  Aggregate is FLAT from batch 2 to 3 (7.86 -> 7.19 tok/s) while per-stream falls
  3.93 -> 2.40. ~~At these widths the sessions are serialising rather than sharing a
  pass over weights~~ — **RETRACTED 2026-08-11, see below: at batch 1-3 the fused
  path is not selected at all, so this measured the serial fallback, not fusion.**
- **Port target localized 2026-08-10 — ONE validator, not two paths.** Correcting
  the previous line: the fused decode does not carry its own copy of the check, it
  reuses the prefill contract, which is why the DECODE failure said "prefix" and
  matched the prefill message verbatim.
  - `validate_grouped_moe_prefill_session_batch_state_contract`
    (`qwen35/prefill_batch.rs:1061`) is the single enforcement site:
    ```rust
    if !signature.kv_quantized || !signature.kv_quant_q8 {
        return Err("... must use Q8 KV state for the MQ4 control path")
    }
    ```
    kvarn sets `quant_kvarn`, never `quant_q8`, so it fails here. Note kvarn is
    NOT in the adjacent asym/fwht rejection list — it is excluded only by the
    positive Q8 test.
  - The fused entry point is named
    `forward_prefill_grouped_moe_session_batch_prefix_q8_kv`
    (`prefill_batch.rs:2907`), and a sibling error reads "first MoE target is
    plain Q8 KV". Q8-only was a deliberate first-target scope with a named
    extension point, not an accident — so the port is generalising a contract
    that anticipated this, not undoing a mistake.
  - Every other caller of the validator is a test in `qwen35/mod.rs` (~4856-4951),
    including one asserting the fp32 rejection, so those pin the current contract
    and will need updating with it.
- **Measured 2026-08-11 — the flat curve below batch 4 was POLICY, and fusion
  itself gives 1.11x.** Three separate corrections, all from direct measurement:
  - `qwen35_grouped_moe_decode_auto_latency_gate_passed` is `session_count >= 4`
    (`hipfire-generate/src/lib.rs:1522`). Below that, `auto` selects
    `SerialReference` deliberately. `HIPFIRE_LAUNCH_TRACE=1` confirms it
    structurally: width 3 issues **exactly 3.00x the launches of width 1 with every
    grid dimension unchanged** (1322 launches per row, three times). So every
    batch 1/2/3 throughput number ever quoted in this entry measured the serial
    fallback. The flatness was designed, not broken.
  - **Where fusion does run, 4 rows buy 11%.** Under Q8 KV via
    `HIPFIRE_KV_ALLOW_DEPRECATED=1`, one daemon lifetime, `decode_step rows=4`
    confirmed on 32 steps: batch 1 = 7.92 tok/s aggregate, batch 4 = 8.80
    (**1.11x**), per-stream 7.92 -> 2.20. That is roughly what the amortization
    curve predicts at 4 slots — its knee is near `n_exp/k` = 512/8 = **64** — so it
    is NOT evidence the fused kernel is broken. It means no reachable batch width
    is anywhere near the knee.
  - **Batch 8+ is currently unmeasurable for an unrelated reason:** HTTP 429 from
    the `requests_per_minute` bucket in `hipfire-server/src/api_auth.rs`, not from
    anything in the batching path. Raise it in the server config before quoting any
    N >= 8 number.
- **[Independent, same repro] The refusal fires at EXECUTION, not selection, and
  takes the request down instead of falling back.** At batch >= 4 with kvarn the
  auto path sets `FusedGroupedMoeLayerChunked` — the selection-time capability
  validator does not test KV mode — and the Q8 requirement is then asserted deep in
  the decode advance, returning an error to the client rather than degrading to
  `SerialReference` as batch 1-3 does. Two defects in one:
  - the capability predicate is not wired into *backend selection*, which is the
    exact anti-pattern the v2 plan lists as a Tier-1 prerequisite;
  - the error is delivered as **HTTP 200 with an `{"error": ...}` body**, so any
    client checking status codes sees success. That alone is worth fixing
    independently of the port — it is how this failure hid inside a sweep harness
    that counted 200s as successes.
- **RESOLVED 2026-08-11 — batching DOES scale (2.08x at width 16); the flat curve
  was four caps, not one defect.** Measured with Q8 KV, loopback bind, raised
  `BATCH_MAX`, `auto` prefill; prefill and decode separated by differencing
  `max_tokens=1` against `max_tokens=64` in one daemon lifetime:

  | width | prefill | decode step | decode tok/s | vs w1 |
  |---|---|---|---|---|
  | 1 | 0.93 s | 11.3 ms | 88.6 | 1.00x |
  | 8 | 3.45 s | 52.2 ms | 153.3 | 1.73x |
  | 16 | 7.05 s | 87.3 ms | 183.3 | **2.07x** |

  End-to-end at width 16 is 16.5 tok/s vs 7.9 at width 1. The caps, each of which
  flattens the curve on its own:
  - `max_in_flight_text = 4` in `RatePolicy::default()` — a CONCURRENCY cap, and
    the actual source of the HTTP 429 above (not the per-minute bucket). Binding
    `--host 127.0.0.1` selects `loopback_default()` where it is 0 = unlimited.
  - `BATCH_MAX_DEFAULT = 8` (`hipfire-server/src/batch_runner.rs:421`) — the
    envelope never exceeds 8 rows however many sessions are waiting.
  - the `n >= 4` decode latency gate (above).
  - `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial`, which was a harness carryover:
    16 sequential prefills are 73% of wall time at width 16. **Under Q8, `auto`
    prefill works** (14.80 s -> 7.05 s at width 16), so the earlier note in this
    entry that "`auto` does NOT work — only `serial` does" holds for kvarn only.
- **Single-stream MoE decode is launch-bound on gfx1103.** A width-1 decode step
  issues **1322 launches in 11.3 ms** — ~8.5 us each, essentially ROCm's launch
  overhead. At width 16 the same 1322 launches take 87.3 ms (66 us each), i.e. it
  has crossed into real work. This is why batching amortizes ~2x when the
  expert-byte curve predicts only ~1.14x at that width, and it is an argument for
  grouping modules into FEWER launches.
- **Width 64 is not reachable on nix1.** Raising `BATCH_MAX` to 64 yields an
  achieved width near 18; batch 64 collapses to 2.22 tok/s with 20/64 sessions and
  `generate_batch_prefill ... failed to create checkpoint`. Since the MoE
  amortization knee is at `n_exp/k` = 512/8 = 64, the capacity argument cannot be
  evaluated on this box at 35B-A3B.
- **RESOLVED 2026-08-11 by the KVarN port; the fallback defect now matters MORE.**
  `qwen35_kvarn_fused_batch_enabled()` defaults ON as of this date, so KVarN — the
  supported mode — reaches the fused path and batched MoE decode is live again.
  But the refuse-at-execution defect above is now the kill switch's sharp edge:
  verified that `HIPFIRE_QWEN35_KVARN_FUSED_BATCH=0` with KVarN KV makes batch >= 4
  requests FAIL rather than fall back to `SerialReference`. Anyone reaching for the
  kill switch during an incident would trade a suspected numerical issue for hard
  request failures.
- **FIXED 2026-08-11 — the refusal now happens at SELECTION, on both paths.** Two
  separate holes, and the second was the interesting one:
  - prefill: `validate_qwen35_fused_grouped_moe_prefill_model_capability`
    (`serving-core/src/session.rs`) never looked at the KV mode, so selection
    could not see the incompatibility. Added a narrow check that rejects exactly
    KVarN-with-the-gate-off and leaves every other mode's routing untouched.
  - decode: `validate_qwen35_grouped_moe_decode_model_capability` built a
    SYNTHETIC probe signature with `kv_quant_q8: true` hardcoded, which made its
    KV test vacuous — it passed for every mode, including modes the body would
    then reject. The probe now derives its flags from `m.q35_kv_mode`. A
    capability probe that asserts the capability it is meant to test is worse than
    no probe: it reports "supported" for a configuration that fails on the next
    call.

  Verified end to end. With `HIPFIRE_QWEN35_KVARN_FUSED_BATCH=0` and KVarN KV,
  batch 4 now SUCCEEDS and its output is byte-identical to serial on 4/4 prompts
  (i.e. it really did route to `SerialReference`); with the gate at its default it
  diverges on 2/4, the fused signature. The divergence profile doubles as a
  backend detector, which is how selection was confirmed rather than assumed.
- **Consequence worth stating plainly: batched MoE decode is presently
  unreachable.** q8 is on `DEPRECATED_KV_MODES` (`serving-core/src/load.rs:533`)
  and the fused path requires q8, so on every supported KV mode there is no batch
  size that fuses — below 4 it is policy-serial, at 4+ it errors. The kvarn port is
  therefore not an optimization; it is a prerequisite for measuring the v2 plan's
  central amortization claim at all.
- **Port RE-SIZED 2026-08-11 — the 2026-08-10 sizing below was materially wrong.**
  It said the port was "(1) add a kvarn dispatch arm using the existing
  `attention_kvarn_routed_batched`, (2) widen the validator". Step 1 is real but
  incomplete, and it omits the entire write half:
  - **`attention_kvarn_routed_batched` is READ-ONLY.** Its doc is explicit — "K
    dequant is in place", "Mirrors `attention_q8_0_routed_batched`". It does not
    write KV (`hipfire-rdna/src/dispatch/attention.rs:1169`).
  - **It needs THREE pointer tables, not two:** `rec_ptrs` (4-bit K block
    records), `win_ptrs` (f32 recent window), `v_ptrs` (Q8_0 V). The f32/q8 arms
    take `kv_k_ptrs`/`kv_v_ptrs`. So
    `DensePrefillSessionBatch{Host,Device}PointerTables` and
    `...PointerTableShape` each need a third table plumbed through
    `validate_shape` and every construction site (19 refs, 2 files — contained).
  - **There is no routed kvarn write, at all.** In kvarn the write is fused into
    `kvarn_attend` (`hipfire-rdna/src/dispatch/kv.rs:1609`), which is
    single-session by construction: it takes `records`/`window`/`v_cache` as bare
    tensors with one scalar `start_pos`, and appends K via a HOST-side loop of
    `memcpy_dtod_at_auto` at 128-token block boundaries. Routed rows have per-row
    sessions and per-row positions, so none of that transfers. `kernels/src/`
    has `kv_cache_write_{f32,q8_0}_routed_batched.hip` and no kvarn equivalent.

  **Two ways to close the write gap, and they differ by ~an order of magnitude:**

  | option | cost | captures |
  |---|---|---|
  | A: new routed kvarn K-write kernel | a new HIP kernel (quantize + append routed by `row_session_indices`/`row_positions`) | everything |
  | B: keep the write per-session (loop the existing single-session append), route only ATTENTION | plumbing + a loop | nearly everything — see below |

  **Recommend B first.** The measurement that decode is *launch-bound* (a width-1
  step is 1322 launches in 11.3 ms, ~8.5 us each) also bounds what B costs: at
  width 16 a per-session write loop is 16 sessions x 10 attention layers = ~160
  copy ops per step against an 87 ms step — order 1.5%. Attention is the part that
  actually amortizes, and B routes it. A is the right end state but should be
  justified by a measurement of B's residual, not assumed.

  Ordering is unchanged and still load-bearing: **widen
  `validate_grouped_moe_prefill_session_batch_state_contract` LAST.** Widening it
  before a kvarn read/write path exists routes kvarn KV into the Q8 kernel, which
  is silent corruption rather than an error.
- **Port sized 2026-08-10 — and relaxing the validator ALONE would be a
  correctness bug.** `prefill_batch.rs` dispatches exactly two attention kernels:
  ```
  gpu.attention_f32_routed_batched
  gpu.attention_q8_0_routed_batched
  ```
  There is no kvarn arm and no asym arm, across 37 KV-mode branches in the file.
  So admitting kvarn at the contract without adding a dispatch arm would route
  kvarn-quantized KV into the Q8 (or fp32) kernel — silent wrong output, not an
  error. This is the same accept-and-miscompute class as the indexed-OQ null table
  earlier in this branch, and it is why the validator must not simply be widened.
- **The kernel needed already exists:** `attention_kvarn_routed_batched.hip`
  (alongside `attention_flash_kvarn_tile_batched.hip`). So the port is two
  coordinated edits, not new kernel work:
  1. add a kvarn arm dispatching `attention_kvarn_routed_batched` beside the
     existing f32/q8 arms, and
  2. extend `validate_grouped_moe_prefill_session_batch_state_contract` to accept
     `kv_quant_kvarn` — in that order, so the contract never admits a mode the
     dispatch cannot serve.
  The tests in `qwen35/mod.rs` (~4856-4951) pin the current contract and move with
  step 2.
- Until then `serial` + kvarn caps usable concurrency at 3.
- Scope: blocks multi-stream measurement on the target model
- Confidence: High (explicit runtime error naming the requirement)

## [High] Batched prefill OOMs on the 35B at batch=4 — blocks all multi-stream measurement
- Category: Correctness / Capacity (batched prefill)
- Measured 2026-08-10 on nix1 (gfx1103, 45.1 GB GTT) via `hipfire serve` +
  concurrent `/v1/chat/completions`, `--max-seq 512`, 32 max_tokens.
- Error, identical in every failing config:
  ```
  batch prefill: daemon generate_batch_prefill error: HipError(2): hipMalloc: out of memory
  ```
- **It is the model, not residency.** Discriminated three ways:

  | model | residency | batch 4 |
  |---|---|---|
  | `qwen3.5-0.8b--oq4++` | resident | **OK** — 22.3 aggregate tok/s, 5.6/stream |
  | `Qwen3.6-35B-A3B--oq4` | paged, 2 GiB budget | **OOM** |
  | `Qwen3.6-35B-A3B--oq4` | resident (paging off) | **OOM** |

  So the batch runner itself works; the 35B specifically cannot batch-prefill.
  Single-stream on the same 35B artifact is fine (13.9 tok/s warm), so this is a
  batched-path allocation, not model capacity.
- **Why it matters beyond the immediate ask:** every performance number in this
  investigation is `batch=1`, which is the worst case for the MoE amortization
  curve. The whole capacity argument for module-major execution
  (`docs/plans/2026-08-09-...` 0.4) rests on behaviour at N=16..128 streams, and
  right now that regime **cannot be measured on the target model at all**.
- Suspicion, unverified: `rocm-smi` reports VRAM total = 256 MB on this APU (the
  dedicated carve-out; the 45.1 GB is GTT). An allocation that must land in real
  VRAM rather than GTT would OOM almost immediately and would scale with batch.
  Worth checking which allocation in the batched prefill path is not GTT-backed
  before assuming the sizes are simply too large.
- **CHASED 2026-08-10. Not a MoE bug at all — it is KV allocation, and there are
  two distinct walls.** `HipRuntime::malloc` now names its size, and
  `HIPFIRE_MALLOC_BACKTRACE=1` names the caller.
  - **Wall 1 — fp32 KV allocates 258 MiB in a single chunk.**
    ```
    hipMalloc(270532608 bytes = 258.00 MiB): out of memory
      0 hip_bridge::ffi::HipRuntime::malloc
      1 hipfire_rdna::pool::GpuPool::alloc
      3 hipfire_rdna::dispatch::Gpu::zeros
      4 hipfire_runtime::kv::KvCache::alloc_k_v_filtered
      7 hipfire_serving_core::session::qwen35_allocate_session_state
      9 run_generate_batch_prefill_serial_qwen35
     11 hipfire_daemon::handlers::batch::prefill
    ```
    The 19 GB model loads fine because it is many per-tensor allocations; this is
    the first single buffer to cross the line. `rocm-smi` reports the dedicated
    VRAM pool as exactly 256 MiB on this APU, so a 258 MiB request cannot be
    served from it.
    **`--kv-cache q8` clears this wall** (confirmed: log shows `KV cache: Q8`, the
    258 MiB OOM disappears). Note `AGENTS`-adjacent prior art: fp32 KV also forces
    per-token prefill, so it was never the right mode for batching anyway.
  - **Wall 2 — the batch path CLONES the KV cache per session.** With Q8 it gets
    further and then fails at:
    ```
    failed to create checkpoint qwen35-checkpoint:batch-...:
      clone qwen35 checkpoint kv.k_gpu[3] alloc:
      hipMalloc(71860224 bytes = 68.53 MiB): out of memory
    ```
    `kv.k_gpu[3]` is one layer's K tensor, so a checkpoint clone costs roughly a
    second full KV cache per session (10 KV-carrying layers x k and v). That is
    the real capacity model of batched prefill on this arch and it is not
    documented anywhere.
- **Answered 2026-08-10: the clone is UNCONDITIONAL, and it is not for batching
  correctness.** `run_generate_batch_prefill_serial_qwen35`
  (`qwen35_prefill.rs:1925`) ends with a bare loop:
  ```rust
  for session in &result.sessions {
      ...
      emit_qwen35_prefill_checkpoint(m, gpu, arena_backend, hook)?;  // no guard
  }
  ```
  and `emit_qwen35_prefill_checkpoint`'s own doc says what it is for: emitting a
  boundary "so clients can resume from a cached prefix". That is **prefix
  caching** — a feature — and `clone_gpu_tensor` implements it as a deep
  device-to-device copy "to snapshot session state without aliasing the live
  buffers". So peak batch-prefill memory is ~2x KV per session, paid whether or
  not any client ever resumes.
- **Recommended fix, in order of increasing scope:**
  1. Make the checkpoint opt-in per request (clients that will not resume should
     not pay for the snapshot). Smallest change, unblocks the batch sweep.
  2. Make it lazy / copy-on-write — snapshot only when the live KV is first
     mutated past the boundary.
  3. Leave it and document 2x KV as the batch capacity model, which caps batch
     width at roughly half what the KV budget suggests.
  This is a **semantics** change, not an allocation fix: prefix-cache resume is
  observable behaviour clients may depend on, so it wants a decision rather than
  a patch.
- Also noted in passing: the function is named `..._serial_qwen35` and prefills
  the batch's sessions in a loop. Whether "batched prefill" is actually fused
  across sessions on this path, or serial-with-a-batch-envelope, was not
  established and is worth checking before any throughput conclusion is drawn
  from it.
- Instrumentation landed with this: `HipRuntime::malloc` reports the requested
  size on failure, and `HIPFIRE_MALLOC_BACKTRACE=1` captures the allocating
  stack. The bare "hipMalloc: out of memory" this started from could not
  distinguish a sizing bug from pool placement from genuine pressure.
- Scope: Capacity — blocks the multi-stream half of the v2 thesis
- Confidence: High (three-way discrimination, identical error)

## [Medium] tiny-quant: three `oq4.25++(calib)` cells breach budget — two on the GOOD side
- Category: Correctness / Quant (mixed Opus) + gate tolerance design
- Location: `tests/tiny-quant-gate.sh`; baselines in `tests/tiny-quant-baselines.txt`
- Summary: on `origin/master` at `33d9dcbd2` the gate is **188 pass / 3 fail**, and
  all three failures are the same cell, `kld:oq4.25++(calib)`:

  | family | measured | baseline | budget | direction |
  |---|---|---|---|---|
  | qwen3_legacy | 0.004369 | 0.005979 | ±0.001495 | **better** by 0.00161 |
  | gemma4_moe | 0.005952 | 0.003077 | ±0.000769 | **worse**, ~1.9x |
  | zaya | 0.000023 | 0.000036 | ±0.000010 | **better** by 0.000013 |

- **"3 failures" overstates it.** Only `gemma4_moe` is a real degradation. The
  other two are *improvements* that trip a **symmetric** relative tolerance —
  the gate flags movement, not loss. `zaya` is the clearest case: at absolute
  magnitudes of 2e-5, a 25% budget is ±1e-5, so almost any change trips it.
- Pre-existing, NOT from the v2 branch: verified by running the identical three
  families in a detached worktree at pristine `origin/master` — the failures
  reproduce with byte-identical measured values, baselines and budgets.
- **`zaya` is FLAKY, not failing (observed 2026-08-10).** A later full-gate run on
  the same commit reported **2** failing cells, not 3: `zaya` passed. Nothing in
  that path changed between runs. This is the predicted consequence of scoring a
  2e-5 cell against a ±25% relative budget (±1e-5) — the cell flips on ordinary
  run-to-run variation. Treat `zaya/oq4.25++` as a tolerance defect, and do not
  read a single green run as having fixed it.
- **Do not re-record baselines from one run.** `--record` would bake in whichever
  side of the flake that run landed on, and would also silently absorb the real
  `gemma4_moe` regression.
- Also note the gate's `findings: N` counts **skips as well as failures** — a run
  showing 9 findings here is 2 fails plus 7 explicitly blocked `deepseek4_mtp`
  cells. Read the `fail` lines, not the findings count.
- Two separable actions: (a) investigate `gemma4_moe` oq4.25++ as a genuine ~1.9x
  mixed-Opus regression — it is the one cell that is reproducibly worse; (b) give
  the KLD budget an absolute floor (and consider making it one-sided) so near-zero
  cells stop flipping. Re-recording the baselines would hide (a) — do (a) first.
- Scope: Correctness (mixed Opus) + gate design
- Confidence: High (byte-identical reproduction at pristine origin/master)

## [Low] Opportunistic .unwrap() → error-handling cleanup (convention, not a tracked bug)
- Category: Reliability / Maintainability
- Location: Project-wide (~6.8k non-test `.unwrap()` sites; most guard true
  invariants, not user input)
- Summary: Prefer `?`/descriptive `expect()` over bare `.unwrap()` on paths
  that can fail on user input or external files. This is a fix-as-you-touch
  convention, not a specific reproducible crash — a blanket sweep is neither
  feasible (6.8k sites) nor desirable (many unwraps encode real invariants).
- Named exemplars — both resolved (2026-07-21/22):
  - `hipfire-runtime/src/weights.rs`: 14 raw
    `unsafe { …as_ref().unwrap().buf.alias() }` rotated-scratch sites → one
    documented `Gpu::mq_x_rot_f32()` accessor (SAFETY comment + actionable
    `expect()`).
  - `hipfire-quantize/src/main.rs` `SafetensorsFile::open`: the model-load
    header parse (`from_utf8`/`from_str`/`from_value`/8-byte length) now returns
    clean `io::Error(InvalidData)` messages instead of panicking on a
    truncated/malformed `.safetensors` file.
- Confidence: Low (convention; no open crash tracked)

## [Closed] "Excessive" global state via OnceLock — intentional, not a defect
- Category: Architecture / Maintainability
- Location: crates/hipfire-arch-deepseek4/src/forward.rs (`mod env_cache`),
  crates/hipfire-rdna/src/dispatch/mod.rs, crates/hip-bridge/src/ffi.rs
- Resolution (2026-07-22): Investigated. The flagged `OnceLock`/`thread_local!`
  statics are a deliberate, documented hot-path optimization: they cache
  `HIPFIRE_*` env-derived debug/tuning knobs read once, because an uncached
  `std::env::var` per lookup cost ~200μs/token (43 layers × ~5 lookups × ~1μs
  syscall). They are set-once, read-only, and idiomatic. Converting them to
  injected config context would re-add that per-token cost (or require threading
  a config struct through the entire hot path) for near-zero benefit — these are
  debug/tuning knobs, not core mutable state. Not a bug.
- Residual guidance (minor): do not introduce globals for *core mutable state*
  or *user-facing config*; those belong in explicit context objects. Env
  debug/tuning knobs behind `OnceLock` remain the accepted pattern.

## [High] Stale SWA ring-buffer slots after speculative reject (post-wrap corruption)
- Category: Reliability / Correctness
- Location: crates/hipfire-arch-deepseek4/src/spec_decode.rs:224-233,401-428;
  read side kernels/src/deepseek4_attn_swa.hip; config `sliding_window=128`.
- Mechanism (code-confirmed 2026-07-22, no empirical run — see blocker):
  1. The draft/verify loop increments `state.n_tokens` per step so SWA K/V
     writes land IN THE REAL per-layer ring at draft positions N+1..N+K
     (spec_decode.rs:224-230). Slot index = `n_tokens % sliding_window`.
  2. On partial accept only `state.n_tokens` is restored (line 428); the ring
     DATA at the K−n_accept uncommitted slots is never invalidated.
  3. The decode SWA kernel reads slots `[0, n_valid)` LINEARLY with no
     per-slot position mask (deepseek4_attn_swa.hip) — it trusts n_valid.
  Result: PRE-wrap (total seq < 128) the stale slots sit at indices ≥ n_valid
  and are excluded → safe. POST-wrap (seq ≥ sliding_window=128) the ring is
  full; uncommitted draft writes evict positions still inside the next
  forward's 128-wide window, so the linear read consumes rejected-token K/V →
  silent attention corruption.
  Refined boundary (2026-07-22, from the verify/accept indexing): verify feeds
  `[last_token, draft[0..k-2]]` at base `last_position+1`, and the NEXT decode
  overwrites exactly ONE stale slot (the corrected token's, verify column
  `accepted_len`). So the still-stuck stale columns are `[accepted_len+1, k)`,
  nonempty only when **k ≥ n_accept+3** (never k=2; k=3 only at n_accept=0) AND
  post-wrap. Real but narrower than "any partial accept". Only the modular SWA
  ring aliases; `full_k_cache` is absolute-indexed + causally safe, and the MTP
  ring only affects draft acceptance (verify still guarantees correct output).
- Fix: IMPLEMENTED, gated OFF pending GPU validation. `spec_decode::swa_rewind`
  (behind `HIPFIRE_DEEPSEEK4_SPEC_KV_REWIND=1`) snapshots the K soon-to-be-
  evicted main-layer SWA slots before the verify (strided per-slot copy into
  per-layer `swa_k_snap`/`swa_v_snap`) and restores the uncommitted columns
  `[accepted_len+1, k)` after the accept, wrap-aware. Pure slot arithmetic is
  unit-tested (`cargo test -p hipfire-arch-deepseek4 swa_rewind`, 4/4). Enable-
  by-default is blocked on an AR-vs-spec losslessness A/B on a runnable model:
  a compressor-F16 `deepseek4-q8-mtp` re-quant is in progress on halo (the mq4
  artifact is unloadable — see below). Validation: pre-fix expect divergence
  post-128 with k=3; post-fix expect token-identical.
- Empirical status (halo, gfx1151): BLOCKED. The only deepseek4 artifact on
  halo (`deepseek-v4-flash--mq4.hfq`) will not run on the current daemon build:
  its MQ4 `compressor.wkv` is rejected by the F16-native compressor path
  (`HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=1` default), and `=0` routes it to an
  unsupported `gemv.unknown`. Black-box AR-vs-spec-decode A/B needs a
  re-quantized compressor-F16 model first.
- Scope: Architectural
- Confidence: High on mechanism (code-confirmed); reproduction pending a
  runnable model.
- Note: The sibling `forward.rs` chunk/ring path is NOT affected — its
  non-aligned-with-compress-events case returns an explicit `Err`.

## [Medium] tiny-quant `++` cells: gemma4_moe expert gate_up gets no Hessian — a hand-rolled capture map names it SPLIT while the artifact FUSES it
- Category: Test coverage / calibration name resolution
- Location: `crates/hipfire-serving-core/src/tiny_harness.rs` `capture_names()`
  Gemma4 arm (L1067); `crates/hipfire-quantize/src/main.rs`
  `calibration_tensor_name_candidates` (L6425)
- **This entry has been wrong twice. The mechanism below is established by
  name-level comparison, not inference.** First filed [High] with the right
  symptom and a guessed mechanism; then retracted on a stale code comment
  ("routed experts ... we don't name them") that does not describe what the
  Gemma4 arm actually does. It names them — under the wrong names.
- The capture map registers, per expert:

      model.language_model.layers.{L}.experts.{E}.gate_proj    <- SPLIT
      model.language_model.layers.{L}.experts.{E}.up_proj      <- SPLIT
      model.language_model.layers.{L}.experts.{E}.down_proj

  while the quantized fixture FUSES gate and up, so the quantizer resolves
  `...experts.{E}.gate_up_proj.weight`. `down_proj` matches; `gate_up_proj`
  matches nothing. Hence exactly 16 missing (2 layers x 8 experts) with
  `pooled=0`, and 16 expert `down_proj` among the successes. The arithmetic
  closes: 22 dense + 16 down + 16 gate_up = 54 attempts, 38 success.
- `qwen3_5_moe` is the control and the fix in miniature: its arm calls the arch's
  REAL walker (`qwen35::build_capture_names`), the names agree, `missing=0`.
- **Root cause is the divergence, not the names.** The harness hand-rolls capture
  maps for 7 families (Qwen2, DotsOcr, Deepseek4, Gemma3, Gemma3Vl, MiniMax,
  Gemma4) while 4 use the arch's real walker. Gemma3 and MiniMax hand-roll even
  though `hipfire-arch-gemma3/src/calibration.rs:38` and
  `hipfire-arch-minimax/src/calibration.rs:37` exist. Gemma4 has no walker at
  all. The harness is meant to reuse real hipfire; every hand-rolled map is a
  second source of truth that can drift from the artifact layout, and this is
  that drift.
- Note the genuine subtlety before "just fix the names": at runtime the Gemma4
  model holds gate and up as SEPARATE tensors (captured by pointer), while the
  artifact fuses them. So the capture cannot simply emit the fused name — one
  name, two pointers. Either `calibration_tensor_name_candidates` learns the
  fused<->split correspondence, or the collector combines the two captures.
- Fix order: (1) give gemma4 a real `build_capture_names` in its arch crate and
  switch Gemma3/MiniMax to theirs, so there is one source of truth; (2) resolve
  fused<->split in the candidate list; (3) make `ldlq_report_and_validate`
  strict about `missing > 0` for a `++` format — a partial application currently
  emits an artifact named `++` regardless, which is the mislabelled-artifact
  failure the quantizer refuses `qtip3++` over.
- Shipped-artifact impact still UNMEASURED: real models calibrate through the
  real engine (`calibration/expert_capture.rs:777`), not this map. A real dense
  model returned `missing=0, 186/186`; a real MoE has not been run.
- Visible per calibrated cell in `results.jsonl` as `ldlq_{success,attempts,
  missing,k_mismatch,pack_failed,pooled}` and `ldlq_skipped`.

## [Medium] Calibration coverage: three open questions, all reducing to one missing measurement
Examined 2026-08-20. Recorded together because they turn out to share a root.

### 1. The collectors disagree about `lm_head`, and minimax is the outlier
Surveyed all eight `build_capture_names` walkers: **minimax captures `lm_head`;
the other seven do not.** llama's walker states the convention — "The lm-head
(`output`) is not captured for a Hessian — like every other arch collector it is
KLDREF-only" — and it is right about the other seven.

The defect is not which side is correct, it is that the quantizer's LDLQ
ELIGIBILITY set and the collectors' CAPTURE set disagree: the quantizer attempts
`lm_head`, so a family following the convention reports a permanent `missing=1`
and its `++` artifact is, strictly, not fully covered. **This is what blocks
`HIPFIRE_LDLQ_STRICT` from becoming the default** — a strict pass fails seven
families out of eight.

Natural experiment, from switching the harness to minimax's real walker
(PR #254) with baselines that had been recorded while `lm_head` was uncovered:

    oq4+     0.00129443 -> 0.00129443   unchanged
    oq4++    0.00129443 -> 0.00129443   unchanged
    oq4.25++ 0.00089918 -> 0.00089271   -0.7%
    oq8+     0.00000675 -> 0.00000675   unchanged
    oq8++    0.00000675 -> 0.00000675   unchanged

So covering `lm_head` with LDLQ has **no effect this fixture can resolve** — four
of five cells identical, one moving 0.7%, all inside the ±25% budget. Read that
as "below the fixture's resolution", not as "no effect": the values are printed
at six significant figures.

### 2. gemma4 still has no capture walker
Confirmed post-merge: no `build_capture_names` in `hipfire-arch-gemma4`, and
`tiny_harness.rs:1060` still hand-rolls the map. This is the family whose
hand-rolled map named expert projections SPLIT while the artifact FUSES them,
which PR #252 patched at the quantizer (fused -> split Hessian fallback). The
durable fix is a walker in the arch crate; the harness arm then follows the
other four.

### 3. Real-MoE oq4.25++ is still unmeasured, and is NOT tractable on this box
**Attempted 2026-08-20; the "tractable" claim in the first version of this entry
was wrong.** I sized LFM2.5-8B-A1B (10.5 GB) and never checked whether its arch
can be calibrated at all. It cannot:

    InvalidSourcePlan("no native calibration adapter is registered for architecture 11")

**Only five arches register a calibration adapter** — qwen35, gemma3, zaya,
gemma4, cohere2 (`register_calibration_adapter!`). lfm2moe is not one, so no
Hessian can be produced for it through the production path, and `oq4.25++`
without a Hessian is not the format under test.

The smallest real MoE on a supported arch is `Qwen3.5-35B-A3B` at ~44 GB. The
measurement needs an HF restore (~44 GB, because `calibrate` accepts ONLY an HF
snapshot — not `.hfa`, not `.hfq`; use `hipfire-coexistence repack` to restore)
plus a bf16 anchor (~44 GB), a Hessian, and two `oq4.25++` artifacts (~19 GB
each): **~170 GB of working space against 65 GB free.** Not feasible here.
It needs a bigger box, or a small real MoE on one of those five arches.

### 3b. The tiny harness calibrates families production cannot
Falling out of the above, and pointing the same way as the capture-map drift:
the tiny gate runs five calibrated cells for `lfm2_moe` (`oq4+`, `oq4++`,
`oq4.25++`, `oq8+`, `oq8++`) while the production `calibrate` CLI refuses arch 11
outright. It can do that because the harness uses its own `CalibCollector` +
`capture_names()` rather than the `layer_stream` adapter registry.

So a family can look **calibration-covered in the gate while being
uncalibratable in production** — the harness/production divergence again, but
inverted: here the harness does MORE than the real path, which is the direction
that manufactures false confidence rather than false alarms.

### The shared root
(1) and (3) are the same question — *does a calibration/encoder change help real
weights?* — and neither can be answered on the tiny fixtures, because seeded
random-init weights have no outlier or correlation structure for AWQ scaling or
Hessian feedback to exploit. That is the same limitation that made the gemma4
fixture move OPPOSITE to the real model on oq4.25++
(`docs/tiny-quant-gate-8-failures.md`). One real-MoE run on LFM2.5-8B-A1B would
inform both: quantize either side of `8357081d3`, and separately with `lm_head`
capture on and off.

## 5. Two `qwen3_5_moe` cells drifted ~10–15% worse and the tolerance hid it

`tests/tiny-quant-baselines.txt` last recorded `qwen3_5_moe` on **2026-07-22**
(`753df2b27`). 83 quant commits landed since. Re-measuring on master
`2121401b3` (gfx1103):

| cell | baseline (Jul 22) | now | Δ |
|---|---|---|---|
| `oq8+` / `oq8++` (calib) | 0.008147 | 0.005677 | **−30%** (better; re-recorded) |
| `oq4` | 0.175872 | 0.194060 | **+10.3%** (worse) |
| `oq4.25++` (calib) | 0.165681 | 0.190065 | **+14.7%** (worse) |

Only the oq8 pair FAILED, because the per-cell tolerance is **0.25 relative** —
wide enough to swallow a 15% regression silently. The two oq8 lines are
re-recorded; the oq4 lines are deliberately **left stale so the drift stays
visible**, which is why the gate still reports those numbers against July.

**This is not the gfx1103 first-run position effect (~8.6%).** Two independent
runs gave 0.19406/0.19407 for `oq4` and 0.190065/0.19007 for `oq4.25++` —
identical to four digits. It is deterministic.

The pointed part: `oq4.25++` is the premier format in the quant priority
hierarchy, and it has lost its margin. In July it beat `oq4+` clearly
(0.1657 vs 0.1951); now they are within noise of each other (0.1901 vs 0.1947).
Whatever moved it took away the reason to prefer it on this fixture.

Not bisected — that is 83 commits of GPU re-quantization. Candidates that touch
exactly this path: `72cd1c10b` (MoE routers stay lossless BF16),
`80c498b37` (undercovered routed experts go to W8). Both would plausibly explain
the oq8 improvement; neither obviously explains the oq4 regression.

Caveat that applies to all of it, per §3: these are seeded random-init tiny
fixtures with no outlier structure, so an AWQ/Hessian format can move opposite
to the real model here. The finding is "the gate's numbers moved and nobody
noticed", not yet "oq4.25++ regressed on real weights".

## [High] Qwen3.5-122B-A10B serves INCOHERENT text

Both artifacts emit byte-identical garbage; loading and memory are fine (68.99 GB
resident, the old 3.5x GTT blowup gone). `Qwen3.6-35B-A3B` on the same arch and
kernels is coherent at 61.5 tok/s. Ruled out by measurement: lm_head, the OQ8
router path, per-expert missing AWQ, and — the leading suspect, now wrong —
mixed compact+Oq8 expert layers, which reproduce at 13 MB via
`tests/tiny-moe-mixed-gate.sh` and move KLD by under 1%. Cause still open.

→ `docs/bugs/2026-08-26-122b-incoherence.md`

## [High] 122B prefill capped at 29 tok/s — mixed layers cannot use grouped prefill

Flat at 29.0–29.1 tok/s for n=61, 92 and 658 alike (69.6 for the 35B-A3B, 306 for
the 27B). Two coupled causes, either alone a no-op: the admission gate has no
variant for a two-QUANT expert mix so it maps to `Invalid` while
`routed_oq_mixed_compact` sits computed alongside and unconsulted, AND
`gemm_oq_compact_moe_grouped_wmma` takes `block_stride` as a launch-wide scalar
where the decode GEMV takes a per-expert table. Patching only the gate gives
`verdict=true` and 29.0 → 29.0.

→ `docs/bugs/2026-08-26-122b-prefill-ceiling.md`

## [Medium — PARTLY FIXED 2026-08-30] `--mixed-bpw` is silently ignored unless the input is an `.hfq`

Threaded only into `run_hfq_source_pipeline`, so safetensors and `.hfa` input
drop it with no warning and produce uniform experts. (`.hfa` input itself is
fine — consumed in place, headers verbatim, confirmed on the 122B's 180 GB
archive.) Consequence: a mixed artifact can only be built via `.hfq`
re-quantization, which selects a larger tensor set and picked up a K=128 tensor
the runtime refuses outright. Guarded for `OqPlusCompact` as of `224acb1cb`;
`Oq4`/`Oq8` still lack it.

→ `docs/bugs/2026-08-27-mixed-bpw-ignored-off-hfq.md`

## `*_k8_indexed*` MoE kernels are named for a `k` they do not hardcode

**Found 2026-08-29 on `feat/buffer-origin-tag-and-route`, halo. Low priority —
naming/doc only, no wrong output. Cost a scoping pass an XL estimate.**

Every indexed routed-expert GEMV takes `K_TOP` as a **runtime kernel argument**
and uses it only as a grid dimension and a stride:

- `gemv_oq4g256_moe_down_k8_indexed_batched_expanded.hip:28` — `int M, int K, int K_TOP`
- `gemv_oq8g256_moe_down_k8_indexed_batched_expanded.hip:21` — same
- `gemv_oq4g256_moe_gate_up_indexed_batched.hip:25` — same
- `moe_down_combine_k8_batched.hip:26` — `int M, int K_TOP`

Nothing in those bodies requires `K_TOP == 8`. The `k8` is inherited from the
kernel they were ported from. The only genuine compile-time `k` is in the
top-k SELECTION kernels, which are a different and much smaller surface:

- `moe_softmax_topk_k8.hip:27` — `#define K_TOP 8`
- `moe_topk_renorm_k8.hip:21` — `#define K_TOP 8`

Consequence: reading the file names suggests supporting `k != 8` means writing a
new expert-GEMV family (XL). It does not — the expert compute path is already
k-generic, and the work is the two selection kernels plus threading `k` through
`INDEXED_MOE_K_TOP` / `oq_indexed_admissible` / `use_gpu_topk` / the loader
repack so they keep agreeing. Rename to `_indexed_` (dropping `k8`), or add a
one-line header note on each, so the next reader does not re-derive this.

## `HfqFile::from_safetensors` cannot open the shipped Qwen3.8-Flash-Next; the quantizer can (medium)

**Found 2026-08-30 on halo, against the restored 336 GB checkpoint.**

Two readers of the same on-disk format disagree about whether this model is
readable:

    hipfire-quantize --input <dir>      -> "Found 1658 tensors", quantizes fine
    HfqFile::from_safetensors(<dir>)    -> InvalidData:
        tensor model.language_model.layers.1.ple.ple_embedding.layer_multipliers
        has unsupported dtype "I64" (from_safetensors handles bf16/f16/f32 source only)

The three n-gram derived tables — `layer_multipliers`, `ngram_heads_offsets`,
`ngram_heads_vocab_sizes` — are **I64** in the shipped checkpoint. They are index
metadata, not weights, and are reproducible from config (the loader's own weight
plan marks them `Expect::derived`), so nothing needs them to be quantised. But
`from_safetensors` refuses the WHOLE DIRECTORY on the first one it meets.

Consequence: any path that opens a raw HF directory as an `HfqFile` is closed for
this family. That includes `examples/manifest_real`, which is what found this — it
was written to validate the tensor manifest against the real checkpoint rather
than against the committed shapes list, and cannot.

Fix is probably to pass integer tensors through as opaque bytes (they are already
`QuantType::OpaqueBytes`-shaped: index data with no numeric interpretation the GPU
needs) rather than mapping them to a float dtype. Refusing the file is the wrong
default for a tensor nothing will compute on.

## [FIXED 2026-08-30 for oq4; oq8 now WARNS] `K % 256 != 0` splits one MoE layer across THREE quant families, and drops calibration

**Found 2026-08-29 on `feat/buffer-origin-tag-and-route`, halo. Reproduces on a tiny fixture
in ~2 s, no GPU. Verified at the byte level; the calibration half is the serious part.**

`--format oq4`, exit 0, no warning. Two fixtures differing only in `moe_intermediate_size`:

```
qwen3_5_moe_indexed   (mi 512)   shared.down [256,256] Oq4G256   gate_up Oq4G256   down [256,512] Oq4G256
qwen3_5_moe_mi640_k10 (mi 640)   shared.down [512,640] Q8F16     gate_up Oq4G256   down [512,640] HFQ4G128
```

**FIX (2026-08-30).** The admission gate asked `inner_k % 256 == 0` and, failing
that, dropped the tensor out of Opus entirely. It now asks which Opus group the
tensor's K ADMITS — the rotate is an FWHT, so powers of two only: 256, else 128,
else none — and routes a 128-but-not-256 K to `OqPlusCompactG128` (qt 52), which
exists for exactly this case and, unlike the HFQ4G128 it used to land on, carries
calibration.

Scope of the fix, precisely:

- `--format oq4`: FIXED. `gemma4_moe` and `qwen3_5_moe` (both mi=128, both in the
  tiny-quant gate) now emit zero HFQ4G128 routed experts; `down_proj [256, 128]`
  stays `OQ4-EXP`. tiny-quant-gate PASSES with the change.
- `--format oq8` / `oq8+`: NOT fixed — there is no G128 8-bit *input format* wired
  into `quantize_hfq_source_tensor` yet (the `Oq8G128` quant type exists; the
  producer path does not). These still fall out of Opus, but now print
  `⚠️ down_proj: K=128 admits no Opus group for this format ... which also drops
  calibration for this tensor` instead of exiting 0 in silence. The silence was
  the actual complaint in this entry.

This matters beyond the fixtures: Qwen3.8-Flash-Next ships `moe_intermediate_size
= 640`, so every routed `down_proj` in that model was on the broken path.

At mi=640 **one layer holds three families**, and two tensors of *identical shape* `[512,640]`
get different answers, because there are two independent fallback policies:

- stacked routed experts fall to `quantize_hfq4g128` — `cli.rs:11183`, the terminal `else`;
- the generic 2D path falls to `Q8F16` — the `k % 256 != 0` arms around `cli.rs:4547/4566/4626`.

**The serious consequence is loss of calibration, not the wrong kernel family.**
`supports_g256 = inner_k % 256 == 0` (`cli.rs:10841`) gates the Opus arm at `cli.rs:11081`
(`stacked_oq_format.filter(|_| supports_g256)`). That arm is the only route into
`quantize_hfq_source_tensor`, which its own comment says "applies AWQ/LDLQ and contributes to
strict Hessian validation". At mi=640 every routed `down_proj` skips it and lands on plain RTN
— and unlike the MQ4 branch just above it, the terminal `else` does **not** call
`awq_pre_scale_weights`. So `--format oq4++ --hessian` silently ships uncalibrated,
un-AWQ'd routed down-weights and exits 0. A random-init fixture can never surface this.

**Second consequence: batched prefill is excluded permanently.** The mixed gate_up/down pair
forces `profile=Invalid` (`qwen35/mod.rs:1655`), and `HFQ4G128` is not in the
`moe_prefill_quant_family` ladder at all (`qwen35/mod.rs:2597-2628`, instrumented `_` arm).
The `Q8F16` shared down is an *independent* second decline, so fixing only the routed pair
leaves it standing. At 512 experts that turns prefill throughput into decode throughput, and
under daemon-default KVarN it trips the runtime's own "Output from this point is not
trustworthy" warning.

Repro:

```sh
hipfire-quantize --emit-fixture qwen3_5_moe_mi640_k10 --out /tmp/fx
hipfire-quantize --input /tmp/fx --output /tmp/fx.hfq --format oq4
hipfire inspect /tmp/fx.hfq --tensors | grep 'layers.0.mlp'
```

Fixture preset: `Qwen35Tiny::moe_mi640_k10_preset`
(`crates/hipfire-arch-qwen35-spec/src/lib.rs`), the probe for the Qwen3.8-Flash-Next geometry
— see `docs/plans/2026-08-29-qwen4exp-flash-next-scope.md` §3.1.

**A warning is NOT the fix.** An earlier draft of this entry said "the fallback should at
minimum be loud"; that is wrong, because an `eprintln!` restores neither calibration coverage
nor batched-prefill admission. The fix is to make the two fallback policies agree and to keep
the calibrated path reachable at `K % 256 != 0` — which needs a group size that divides K.
Note `640 = 2^7 * 5`, so the largest power of two dividing it is 128, and since the Opus rotate
is an FWHT the group must be a power of two — 128 is the only option, not "128 or 160".

Unrelated observation from the same run, do not chase it as part of this: the summary prints
`Mean quant error: 0.00000000` for **both** presets, so that metric reads 0 on the known-good
path too and is not evidence about this bug.

## Load admission under-estimates routed-expert GTT by 26 GB — estimator rounds per TENSOR, pager allocates per MODULE

**Found 2026-08-29 on `feat/buffer-origin-tag-and-route`, halo. Arithmetic on the two
functions, cross-checked against a real artifact's module table. Not yet reproduced as a
failed load — the model that would show it is not converted yet.**

The two sides of expert residency use the same `gtt_alloc_cost` at different granularity:

- `estimated_module_resident_bytes` (`weight_pager.rs:929`) — `for tensor in &module.tensors
  { resident += gtt_alloc_cost(len) }`, i.e. **per tensor**. This is what feeds
  `check_load_headroom`.
- `ensure_expert_module_resident` (`weight_pager.rs:1573`) — `gtt_alloc_cost(
  module_resident_len(&module))`, i.e. **once per module**, and its comment says so
  deliberately ("One allocation per module (gate_up + down contiguous), so the GTT rounding
  applies once to the whole module rather than per tensor").

Both are individually correct. They disagree because they land in different regimes of
`gtt_alloc_cost`, and for a mid-sized expert the per-tensor split rounds *better* than the
real per-module allocation:

```
Qwen3.8-Flash-Next, oq4, hidden 2560 / mi 640, 512 experts x 49 layers:
  gate_up 1,689,600 B -> 2,097,152   (512 KiB..2 MiB regime: next power of two)
  down      870,400 B -> 1,048,576   (same regime)
  per-tensor sum                       3,145,728  = 3 MiB
  module  2,560,000 B -> 4,194,304   (>2 MiB regime: next 2 MiB multiple)
  per-module                           4,194,304  = 4 MiB

  raw                    64.2 GB
  estimator  (admission) 78.9 GB   1.229x
  pager      (actual)   105.2 GB   1.638x
  SHORTFALL              26.3 GB
```

So admission can pass on 78.9 GB and the pager then needs 105.2 GB. The failure would land
mid-load, and per the GTT notes it surfaces as a `page allocation failure`, not an OOM kill,
with RSS showing nothing.

**Why no existing fixture catches it:** on the tiny MoE fixtures every expert tensor is under
512 KiB, where `gtt_alloc_cost` is page-granular and the two formulas agree exactly (measured:
1.006x both, on `qwen3_5_moe_mi640_k10`). The divergence needs tensors in the
512 KiB..2 MiB band with a module over 2 MiB — a shape no fixture currently has.

Fix direction: make the estimator ask `module_resident_len` and round once, matching the
pager. Note that grouped/slab expert allocation (see
`docs/plans/2026-08-29-qwen4exp-flash-next-scope.md` §3.1a) would shrink BOTH numbers and
largely close the gap, but it does not remove the need for the two to agree.


## `gtt_alloc_cost`'s 512 KiB..2 MiB regime over-estimates by ~1.6%

**Found 2026-08-29 on `feat/buffer-origin-tag-and-route`, halo gfx1151. Measured with the
function's own harness. Low priority — it errs on the safe side.**

`gtt_alloc_cost` (`hipfire-runtime/src/weight_pager.rs:842-880`) models three regimes, and its
doc says they are measured. The >2 MiB regime reproduces exactly; the middle one does not.

`cargo run --release -p hipfire-rdna --example gtt_granularity -- <bytes> 64`:

```
requested   predicted (next pow2)   actual      delta
1,689,600   2,097,152               2,064,384   -32,768
  870,400   1,048,576               1,015,808   -32,768
```

Both land 32 KiB under the next power of two, so the real granule in that band is finer than
`next_power_of_two()`. The >2 MiB cases are exact (2,560,000 -> 4,194,304; 5,120,000 ->
6,291,456; 10,240,000 -> 10,485,760), as is the model's central claim that the granule there
is a flat 2 MiB rather than a power of two.

Consequence is small and in the safe direction — the estimator reserves slightly more than the
driver takes. It does NOT rescue the estimator-vs-pager divergence filed above: recomputing
that with measured values gives 77.3 GB of admission estimate against 105.2 GB of actual pager
allocation, so the shortfall is ~28 GB rather than ~26 GB.

Worth correcting the doc comment either way, since it presents all three regimes as measured
and one of them no longer is.

## `__shfl_xor` at offset 32 silently doubles a reduction on gfx1151 (wave32)

**Found 2026-08-29 writing `qsa_block_score.hip`. Not a bug in shipped code — the new
kernel was wrong and a negative control caught it — but the failure mode is worth recording
because it is silent and the arithmetic looks right.**

A 64-thread block with the usual `for (o = LANES/2; o > 0; o >>= 1) dot += __shfl_xor(dot, o)`
reduction is correct on wave64 and WRONG on wave32. gfx1151 is wave32, so the `o = 32` step
crosses a wave boundary, where `__shfl_xor` returns the caller's own value — the step becomes
`dot += dot` and the result is exactly **2x** too large.

It does not error, does not produce NaN, and the doubling is uniform, so a tolerance-based
check with a loose bound passes. It was caught only because the control asserted an exact
expected value: `got 0.88388, want 0.44194`.

Existing kernels are fine — `gated_norm_f32` and the other wave-reduction kernels launch 32
threads and reduce from offset 16, which is the shape to copy. The trap is writing a NEW
kernel with a 64-lane block out of habit.

Worth a one-line note near the reduction idiom in `kernels/README` or in a shared header, so
the next kernel author does not re-derive it from a wrong answer.

**DONE 2026-08-30.** There is no `kernels/README`, so the note went in
`kernels/AGENTS.md` beside the other kernel rules, with the correct shape spelled
out (`offset = 16`, explicit width `32`). A note alone would not stop the next
author, so `wave_reduction_offsets_are_wave32_safe` in
`crates/hipfire-rdna/src/kernel_arity.rs` also enforces it: a `__shfl` reduction
starting at offset 32 fails unless the kernel is genuinely wave64 (under
`kernels/src/gfx906/` AND `wave64` in the filename). Swept the tree first — the
only offset-32 reductions today are in `attention_flash_q8_0_dp4a_wave64.gfx906.hip`,
which is correct; every other kernel already reduces from 16 with an explicit
width. Verified the test fails on an injected `offset = 32` in `attention.hip`.

## qsa_block_score cannot express the reference's block-key pipeline (low)

`kernels/src/qsa_block_score.hip` mean-pools each block's keys and scores them in
the same launch. The reference does two more things between those steps:

    pooled = mean(raw_keys[block])
    pooled = k_layernorm(pooled)                       # not expressible
    blockk = rope(pooled, at the block's FIRST position)  # nor this
    score  = sum_h relu(q_h . blockk) / sqrt(head_dim)

so a QSA indexer built on that kernel scores un-normalised, un-rotated keys and
selects the wrong blocks. Not a wrong-numbers bug in the kernel itself — it
computes what it says — but it is unusable for the indexer as written.

Fixed by splitting: `qsa_pool_norm_blocks` + `qsa_score_prepared`
(`kernels/src/qsa_block_prepare.hip`), with the rotation applied between them via
the normal batched RoPE path. `qsa_block_score` is kept for the pool-and-score
case it was written for. Verified by
`hipfire-arch-qwen4exp/examples/parity_indexer_gpu_vs_cpu`, which differences the
selected token set against the CPU indexer that `reference_oracle.rs` pins to
upstream — exact, at five sequence lengths straddling the dense-below boundary.

Found 2026-08-29 while wiring the GPU QSA block.

## [FIXED 2026-08-29] tiny-prefill's hidden-state probe only runs if a stale binary is lying around

`tests/tiny-prefill-gate.sh:184` gates its hidden-state divergence check on

    HID="./target/release/examples/compare_prefill_hidden_paths"

existing. Nothing builds it — the gate builds `hipfire-eval` and the fixtures,
not that example. So on a CLEAN checkout the probe silently does not run, and
the gate reports `fail=0`; on a developer tree that happens to have built it in
some earlier session it runs and reports `fail=1`.

Measured 2026-08-29 on gfx1151, same commit (`origin/master` 0c9e3d252), same
`HIPFIRE_TINYQUANT_FAMILIES=qwen3_5,qwen3_5_moe,qwen3_5_moe_indexed`:

    without the binary built:  ran=4 fail=0 skip=2   (no `hidden ...` lines at all)
    with it built:             ran=4 fail=1 skip=2
                               FAIL hidden-state divergence kvarn = 1.06e-1 (ceiling 5e-2)

So the check most likely to catch a drafter-visible regression is exactly the
one CI never runs. Same shape as the `$HOME/.hipfire/bin/hipfire` staleness in
`tests/qwen4exp-gate.sh` (fixed there by building and using the in-tree binary):
a gate reading through whatever happens to be on disk rather than building what
it means to test.

**FIXED.** The gate now builds `-p hipfire-runtime --example
compare_prefill_hidden_paths` alongside its other targets, and a missing binary
is an infra error (exit 2) rather than a skip.

A SECOND silent-skip in the same block was found while fixing it. The probe
emits one of two mutually exclusive summary lines:

    FIRST DIVERGING LAYER: 0   (worst overall 1.06e-1)
    IDENTICAL across all layers (worst 3.31e-4)

The parser matched only `worst overall`, so the GOOD case produced an empty
string and was dropped by `[ -n "$wq" ] || continue`. A measurement that did not
happen was indistinguishable from one that passed — including for the fp32
INVARIANT arm, the positive control that is supposed to prove the probe can see
anything at all. Both shapes are parsed now, and an unparseable summary FAILS.

What that made visible immediately, on cells that previously printed nothing:

    hidden fp32:  0.00e0  (invariant, now actually asserted — 3 families)
    qwen3_5_moe   q8 0.00e0   kvarn 0.00e0

The ceiling was NOT raised. See the entry below for the one cell that fails.

## [RESOLVED 2026-08-29] The 4 drifted gfx1151 tiny-state cells: root-caused, both causes benign

**Investigated 2026-08-29 on halo. All 10 failing cells are intended changes
whose gfx1151 baselines were never re-recorded. Neither is a quality regression.
Re-recording is the correct action.**

The 10 split into two populations with DIFFERENT causes:

**(a) 3 cells — `qwen3_5`, `qwen3_5_vl`, `qwen3_5_moe` at fp16.** Caused by
`f5b32ea32 feat(deltanet): default the DeltaNet state to fp16, via config not
env`, which flipped `deltanet_state_precision` fp32 → fp16 for every DeltaNet
model. That commit validated with `no-gpu-ci` plus a Qwen3.8-27B generation diff
and did not run or re-record the tiny GPU gates.

PROOF, not inference — the commit kept `HIPFIRE_DN_STATE_FP16` as an override
that wins in either direction, so the old default is reachable:

    HIPFIRE_DN_STATE_FP16=0 ./tests/tiny-state-gate.sh

    qwen3_5/fp16      OK  0xdb7f521096082900/0x1187a5029a78ab92
    qwen3_5_vl/fp16   OK  0x896bcb3ca08b4529/0x7d4becc77df5201e
    qwen3_5_moe/fp16  OK  0xe74fc5626832105f/0x25975b98e79e8b12

All three match their recorded baselines BYTE-EXACTLY, logit hash included.
Pinning the old default reproduces the old numbers, so nothing else moved.

⚠️ `qwen3_5`'s drifted token hash `0xbccf1d6a241a4482` is the PRE-FP-contraction-pin
gfx1103 value, which invites the reading that `cf8ec5c12`'s pin was lost. It was
not: the pragma is present in all six GDN kernels, and the on-disk kernel cache
(`~/.hipfire/kernels/gfx1151/*.hip`) is byte-identical to the sources. The
LOGIT hash is a third value, not the pre-pin one — the token hash coinciding is
argmax luck, the same coincidence `d36f2081f` called out in the other direction.

**(b) 1 state cell + 6 quant cells — `qwen3_5_moe_indexed`.** The #379 fixture
shape change, already documented in the entry below. Confirmed independent of
(a): with `HIPFIRE_DN_STATE_FP16=0` the quant drift is unchanged (oq4 reads
0.067658 vs 0.067886), and the state cell moves to a THIRD value rather than
matching.

**CLEARED 2026-08-29** — both populations recorded on halo in one pass:

```sh
./tests/tiny-state-gate.sh --record
HIPFIRE_TINYQUANT_FAMILIES=qwen3_5_moe_indexed ./tests/tiny-quant-gate.sh --record
```

`tiny-state-gate: PASS (18 cells)` and `tiny-quant-gate: PASS`. The recorded
diff is exactly the 4 state rows and the 7 `qwen3_5_moe_indexed` quant rows —
the other 14 state rows re-recorded byte-identical, which is itself evidence
nothing else moved.

⚠️ gfx1103 was NOT re-recorded (different host). Its rows for these cells were
last touched by `d36f2081f`, BEFORE `f5b32ea32` flipped the DeltaNet default, so
they are stale in the same way and will drift on the next gfx1103 run.

## [FIXED 2026-08-29] The fp16 DeltaNet state narrows once per LAUNCH, so per-token prefill rounds n times

Two beliefs were wrong and both are now measured, not argued.

**1. The KVarN per-token WRITE path is NOT broken. It was fixed.**

With an FP32 DeltaNet state — i.e. both prefill paths fed identical inputs —
batched and per-token agree under KVarN at 3.31e-4, FLAT:

    n =  32  127  128  129  200   ->  3.31e-4 every time

including straight through the 128-token flush boundary that used to step
(`n=127 2.29e-2, n=128 3.90e-2`, recorded when the bug was live). The
segment-then-flush ordering in `kvarn_attend` is what fixed it.

The runtime warning in `prefill_batch.rs` still said "the per-token fallback is
MEASURED to emit a different token stream ... Use --kv-mode q8 (or f32) until the
KVarN per-token write path is fixed", which steered users off the DEFAULT KV mode
for a bug that no longer exists. Corrected. The `fa_kv_ok` comment in
`tests/tiny-prefill-gate.sh` was stale the same way: it claimed KVarN fails that
test and takes the per-token fallback, but `fa_kv_ok` gained
`|| kv_cache.quant_kvarn` as a perf fix (54 -> 301 tok/s), so KVarN takes the
BATCHED arm exactly as Q8 does. Corrected.

**2. There is no "f16 math" losing precision. The kernel already computes in f32.**

`gated_delta_net_f16.hip` keeps `S_tile` as an FP32 working copy in LDS for the
whole launch. The ONLY f16 is S's storage BETWEEN launches, narrowed once at the
end of each launch. So:

    batched prefill   1 launch for n tokens   -> 1 narrowing
    per-token prefill n launches              -> n narrowings

That is the entire mechanism. Measured at the DeltaNet layer (layer 0, no KV
involved, identical under q8 and KVarN):

    n =    8     16     32     64    128
        1.15e-3 1.15e-3 1.63e-3 1.63e-3 2.36e-3

Growing sublinearly with n, consistent with accumulated rounding — 1.63e-3 at
n=32 is close to the random-walk estimate sqrt(32) * 2^-11 = 2.8e-3.

KVarN is the AMPLIFIER, not the source: the same 1.63e-3 layer-0 split reaches
1.04e-2 by the next layer under q8 and 1.06e-1 under KVarN.

**3. The dither makes path agreement WORSE, while helping long-run drift.**

    dither on   L0 1.63e-3   L1 1.04e-2
    dither off  L0 1.27e-3   L1 6.61e-3

`f32_to_f16_dither` keys on the VALUE'S OWN BITS, so two paths whose values
already differ slightly make DIFFERENT rounding decisions; round-to-nearest is
correlated across paths and tracks better. The dither is still right for its
stated purpose (breaking correlated bias on a recurrent accumulator, where drift
was measured superlinear), but it trades that against batched-vs-per-token
agreement. Previously undocumented.

**FIXED by (a): an FP32 shadow across the per-token fallback.**

`DnFp32Shadow` (prefill_batch.rs) holds S in f32 for the duration of the
per-token prefill loop and rounds only where the BATCHED path rounds — every
`PREFILL_MAX_BATCH` tokens, plus once at the end. Measured at the DeltaNet layer:

    n        before     after      fp32-state floor
    8        1.15e-3    2.23e-7    2.23e-7
    32       1.63e-3    2.23e-7    2.23e-7
    128      2.36e-3    2.44e-7    2.44e-7

i.e. FP16 now reproduces the FP32 trajectory exactly, ~4 orders of magnitude
better. Downstream, `tiny-prefill-gate` goes from fail=1 to fail=0: the
qwen3_5_moe_indexed kvarn cell drops 1.06e-1 -> 3.31e-4 and q8 1.04e-2 -> 1.06e-3,
both the fp32-state figures. `tiny-state-gate` still PASSes 18/18.

Four things that had to be right, each of which would have left the cadence
fixed and the numbers still wrong:

  * The narrow must reproduce `f32_to_f16_dither` INCLUDING its index derivation
    — the element's offset within its HEAD xor `head << 19`, not a flat global
    index. A flat index dithers every head but head 0 differently.
  * The shadow must be seeded by WIDENING, never zeroing: prefill runs
    mid-sequence (DFlash FullPrefill rollback replays a committed prefix), so a
    zeroed shadow destroys live state on exactly the path this helps.
  * It must round at the batched path's chunk boundaries, not just once at the
    end — batched narrows per `PREFILL_MAX_BATCH` chunk, so "narrow once" would
    have been MORE accurate than batched and still different from it. The
    boundary does narrow-then-widen, injecting exactly the batched rounding.
  * Restore on the error path. Every call in the loop uses `?`; leaving
    `s_matrices` on f32 buffers under a FP16 tag makes the next decode read f32
    bytes as `_Float16*`, and `free_gpu` frees the wrong buffers.

SCOPE, deliberately narrow: only the per-token fallback, which is the one
un-chunked token loop with a single exit. There is NO general prefill begin/end
hook — prefill is chunked at three nested levels and the march executor parks the
session between bands — so a shadow spanning "a prefill" would have nowhere to
hang and would have to survive `suspend`. The persistent state stays FP16, so the
~149 MB -> ~75 MB per-session win (19/64 vs 64/64 concurrent sessions at 27B) is
untouched outside the fallback.

⚠️ **RETRACTED 2026-08-30: neither "still open" item was real.** I wrote that the
fused multi-session prefill and the `use_gdn_per_token` loop still narrow per
token. Checked both:

* **Fused multi-session prefill already narrows once per LAUNCH.**
  `gated_delta_net_f16_routed_batch_seq.hip` widens S once at :112, loops
  `for (int b = 0; b < batch_rows; b++)` filtering by session at :115, and
  narrows once at :176. Its caller
  (`grouped_moe_prefill_session_batch_gated_delta_net_f16_layer`,
  prefill_batch.rs:1019) is invoked once per DeltaNet layer with `row_count`, NOT
  inside a token loop, and the entry point bounds rows by `pbs.max_batch` —
  `build_dense_prefill_session_batch_execution_plan(&inputs, pbs.max_batch)` and
  an explicit error if `multi_state_prefix_rows > pbs.max_batch`. So a fused
  launch covers at most PREFILL_MAX_BATCH rows: the SAME cadence as batched
  prefill and as the fixed fallback. Nothing to change.
  (The mapping note this came from actually said that path "keeps the per-launch
  narrowing" — I misread it as a defect.)

* **`prefill_lowered.rs`'s `use_gdn_per_token` loop is deliberate and opt-in.**
  Gated on `force_q8_gdn_per_token || (gdn_tape.is_some() &&
  q8_gdn_verify_per_token_enabled())`, and the latter reads
  `HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN`, default OFF (prefill_batch.rs:5491). Its own
  comment gives the reason: for f16, per-token and batched differ, and a verify
  arm must reproduce what DECODE will do — which is per token. Making it batched
  would defeat its purpose.

So all three live prefill paths now narrow once per <=PREFILL_MAX_BATCH unit and
agree with each other.

**The alternative that was NOT taken:**

  a) Keep S in FP32 for the duration of a multi-token prefill and narrow once at
     the end. The PERSISTENT state stays fp16, so the memory win the fp16 default
     bought (~149 MB -> ~75 MB on a 27B) is kept; only a transient prefill-time
     shadow is fp32. Makes per-token match batched exactly, because both then
     narrow once.

  b) Store S as fp16 PLUS an int8 residual of the discarded mantissa bits: 3
     bytes/element against fp32's 4, recovering ~8 mantissa bits, so per-narrowing
     error drops from ~2^-11 to ~2^-19 (~256x). Cheaper than fp32 but gives back a
     third of the memory win, and the residual has to join the spec-decode
     snapshot or the kernel's documented restore-exactly property breaks.

(a) is the better trade unless the state is being narrowed often outside prefill.

Do NOT "fix" this by raising the tiny-prefill ceiling.

## [tiny-state now GREEN; tiny-prefill still red — re-confirmed 2026-08-31] Three tiny-gate cells are already failing on origin/master

**Re-run 2026-08-31 on nix2 (gfx1103) at `e4025250f`, A/B against a working
branch on the same host and build cache. Two of the three populations have
diverged in outcome:**

- **`tiny-state` PASSES, 18/18 cells.** The 4 drifted cells now match, because
  the baselines were re-recorded to what this file lists as the *observed*
  values — `qwen3_5/fp16` is now baselined at
  `0xed48922801655d8b/0xbccf1d6a241a4482`, which is exactly the hash this entry
  records as drifting against `0xdb7f521096082900/0x1187a5029a78ab92`. So the
  "confirm the cause before re-recording" caution below was overtaken; the
  re-record happened. Worth knowing if anyone still expects those cells red.
- **`tiny-prefill` is unchanged and still red**, and it is NOT branch-local:
  clean `origin/master` and the branch both give `ran=4 fail=6`, the same six
  `hidden-state probe emitted no parseable summary` failures (fp32/q8/kvarn on
  `qwen3_5` and `qwen3_5_moe`) and the same two `qwen3_5_moe` SKIPs
  (`batched prefill did not execute for this fixture`). The probe DOES work —
  it prints `hidden fp32: 0.00e0 / q8 0.00e0 / kvarn 0.00e0` for the family that
  parses — so this is the parser or the probe's output shape on the other two
  families, not a quality regression.

So the tripwire consequence below is now specific to `tiny-prefill`:
`tiny-affected-gate` still fails for every commit touching a covered path, but
`tiny-state` is once again a discriminating signal.

Original 2026-08-29 investigation follows.

## [RESOLVED 2026-08-29 for state+quant; prefill open — see the entry above] Three tiny-gate cells are already failing on origin/master

Confirmed 2026-08-29 by running each gate in a clean `git worktree` at
`origin/master` (0c9e3d252) and comparing to a working branch — the observed
hashes and KLD numbers are byte-identical, so none of it is branch-local:

* `tiny-state`: 4 cells drift — `qwen3_5/fp16`, `qwen3_5_vl/fp16`,
  `qwen3_5_moe/fp16`, `qwen3_5_moe_indexed/fp16`. e.g. qwen3_5/fp16 observes
  `0xed48922801655d8b/0xbccf1d6a241a4482` against a baseline of
  `0xdb7f521096082900/0x1187a5029a78ab92`.
* `tiny-quant`: 6 `qwen3_5_moe_indexed` KLD cells drift (oq4, oq8, oq4+, oq4++,
  oq8+, oq8++); oq4 reads 0.067886 against a 0.052557 baseline.
* `tiny-prefill`: the divergence above, when its probe is actually built.

Likely the same class as "8 gfx1151 baselines stale after the #379 fixture
shape change" (e47b6c7aa) — a fixture change that moved the numbers without the
baselines being re-recorded. Worth confirming that before re-recording, because
if it is NOT a fixture change then these are real quality regressions and
re-recording would bury them.

Consequence today: `tiny-affected-gate` fails for every commit that touches a
covered path, escalates to the coherence battery, and that battery only fails on
hard errors — so the tripwire is permanently tripped and no longer discriminates.
## [High] `hipfire stop` / `restart` ORPHANS the daemon child — every loaded model stays resident

**Found 2026-08-30 on master `886436954`, nix1 (gfx1103, 62 GiB unified).**

`stop()` (`crates/hipfire-cli/src/commands/daemon.rs:191`) signals exactly one
pid — the one in `serve.pid` — and `restart()` (`:246`) is just `stop` + `start`.
Nothing reaps the `hipfire daemon --listen` child, so it is reparented to init and
keeps every model it has loaded. `StdioTransport::spawn_with` does set
`kill_on_drop(true)` (`crates/hipfire-daemon-adapter/src/lib.rs:208`), but that
runs on Drop, and `stop` escalates to `SIGKILL` (`daemon.rs:210` for `--force`,
`:233` after the 5s graceful deadline) where no Drop ever runs.

Measured: one orphan up 57 minutes held **44 GiB**. Killing that single pid took
the machine from `used=44G avail=17G` to `used=2G avail=60G`. No process shows it
in RSS — the daemon's own RSS was 0.5 GB — because GPU allocations on unified
memory do not appear there, so the usual `ps` sweep says the machine is idle while
three quarters of RAM is gone.

Consequence, and how it presents: a benchmark loop that called `hipfire restart`
between models OOM'd four of six models with
`hipMalloc(2228224 bytes = 2.12 MiB), free=10.1 MiB of total=43008.0 MiB` and
`refusing to load: this model needs about 32.0 GiB resident and only 17.3 GiB is
available`. The admission check is doing its job; the memory it is refusing to
overcommit belongs to a server that was stopped. On unified memory this is worse
than a normal leak, because the loader's own error text warns that overcommitting
"invokes the OOM killer on whatever else is running".

Workaround: `hipfire stop && kill $(pgrep -f 'hipfire daemon --listen')`.

Note this is NOT the same as the resolved "hipfire-daemon inference worker killed
on client disconnect" tombstone — that was the child dying too eagerly; this is the
child not dying at all.

## [Medium] `finish_reason` is `"stop"` on `max_tokens` truncation — on BOTH decode paths

**Found 2026-08-30 on master `886436954`.** Minimal repro, `Qwen3.5--0.8b-oq4++`
with `max_tokens: 32` on a prompt that cannot finish in 32 tokens ("Count slowly
from one to five hundred in words"):

    {"finish":"stop","tokens":32}

Exactly at the cap, reported as a natural stop. Reproduced identically at 512 and
1536. It is **not** the continuous-batching runner: re-running with
`HIPFIRE_SERVER_PREFILL_BATCH=0` (confirmed absent from `serve.log`) gives the same
`{"finish":"stop","tokens":32}`, so the legacy path has it too.

What makes this non-obvious: three separate mappings all look correct —
`crates/hipfire-runtime/src/arch.rs:1350`
(`StopReason::MaxTokens if generated >= ctx.max_tokens => "length"`),
`crates/hipfire-serving-core/src/generate_arch.rs:1217`
(`generated_count >= max_tokens`), and
`crates/hipfire-server/src/batch_runner.rs:1107`. The comment at
`generate_arch.rs:1200-1206` says this was already fixed once, for exactly this
symptom ("Without this the CLI fell back to 'stop' for every non-tool-call turn,
hiding `max_tokens` truncation").

Two leads, neither confirmed — do not assume the first:
- **The comparand, not the comparison.** `config.json` carries `max_tokens: 131072`.
  If the engine's `ctx.max_tokens` / `max_tokens` is the CONFIG ceiling rather than
  the request's, then `generated >= max_tokens` is false for every real request and
  all three sites are dead code. This would explain both paths failing at once,
  which a per-path bug does not.
- **A fourth site that hardcodes it.** `batch_runner.rs:1085` sends
  `{"finish_reason": "stop"}` unconditionally when a step yields no per-session
  done event ("end the request cleanly"), with no budget check.

Also note `chat.rs:1232` only ever upgrades to `"length"` via
`detect_tool_call_truncation`, which returns `None` unless there is an unclosed
`<tool_call>` — so the OpenAI layer cannot rescue the plain-text case.

Impact: strict clients use `finish_reason: "length"` to decide whether to retry
with a larger budget, and every eval or benchmark that trusts it will score
truncated output as complete. It did exactly that here — a model that never
terminates was recorded as finishing cleanly until the generated code was
compiled.

## [Medium] Three served models emit unusable Python; ZAYA1 is the one that looks like OUR bug

**Found 2026-08-30 on master `886436954`, nix1.** One prompt (write
`merge_intervals(intervals)`), `temperature: 0`, `max_tokens: 512`, each model
loaded into an empty machine, generated function executed against six cases
(empty / single / overlapping / touching / unsorted / nested):

| model | decode t/s | gen tok | code |
|---|---|---|---|
| `Qwen3.5--0.8b-oq4++` | 67.3 | 512 (cap) | ✗ never emits code |
| `MiniCPM5--1B.oq4.25++` | 52.5 | 362 | ✗ wrong |
| `ZAYA1--8b.oq4++` | 28.1 | 180 | ✗ syntactically broken |
| `Qwen3.6--35B-A3B.oq4.25++` | 23.6 | 428 | ✓ |
| `Qwen3.5-9B--oq4.25++` | 10.5 | 353 | ✓ |
| `Qwen3.8-27B--oq4.25++` | 4.7 | 310 | ✓ |

Three passing models on the same server, harness and prompt rule out a harness
artifact. The two tiny models are plausibly just capability limits and are recorded
as a baseline, not a claim: MiniCPM5-1B returns complete, syntactically valid,
WRONG code (`[[1,3]] -> []`); Qwen3.5-0.8b degenerates into a repetition loop
("Let me write the function. But note: ... We'll write the function.") and produced
**zero** complete code blocks in 1536 tokens.

**ZAYA1-8b is the one worth investigating.** It stopped after 180 tokens
mid-expression — `if not intervals(` — then emitted its closing fence, with
`finish_reason: "stop"` and well under the cap. An 8.8B model truncating inside an
expression and then closing the block reads like a decode/tokenizer defect rather
than weak coding ability. It is also the model whose quant is UNVERIFIED: see
"zaya's tiny-quant KLD cells measure NOTHING" above — those cells were deliberately
left unrecorded, so nothing would have caught a bad ZAYA1 quant. Cheapest next
step is a KLD cell against a reference, not more prompting.

Incidental, worth its own look: `Qwen3.6--35B-A3B` (34.7B, E3.4B) decodes **5x**
faster than `Qwen3.8-27B` (27.8B, E26.9B) — 23.6 vs 4.7 tok/s — and 2.2x faster
than the dense 9B, while being the largest artifact on disk. Consistent with active
params, but 4.7 tok/s for the 27B is low enough to be worth confirming it is not
the same mixed-layer prefill/decode ceiling recorded for the 122B above.

## Fixed — tombstones

One line per fixed bug, newest work last. Full write-ups live in `docs/bugs/`;
anything that had no doc of its own was moved to the archive rather than deleted,
so nothing is recoverable only from `git log`.

- **Routed KVarN prefill: rows in a wrapped block attend to FUTURE tokens** — FIXED 2026-08-29. [`2026-08-29-kvarn-routed-prefill-window-wrap`](docs/bugs/2026-08-29-kvarn-routed-prefill-window-wrap.md)
- **`hipfire model compose` is broken for every default bf16-codec artifact** — FIXED 2026-08-29. [`2026-08-29-compose-hfq-logical-extent`](docs/bugs/2026-08-29-compose-hfq-logical-extent.md)
- **GGUF import silently scrambles Q4_0 and mis-decodes Q5_K** — FIXED 2026-08-29. [`2026-08-29-gguf-dequant-q4_0-q5_k`](docs/bugs/2026-08-29-gguf-dequant-q4_0-q5_k.md)
- **Routed KVarN attention is hardcoded to 4-bit — `kvarn8` / `kvarn2` read garbage** — FIXED 2026-08-29. [`2026-08-29-kvarn-routed-attention-4bit-stride`](docs/bugs/2026-08-29-kvarn-routed-attention-4bit-stride.md)
- **down_proj gets no Hessian/imatrix on bf16 models — `gemv_bf16_xf32` never tapped** — FIXED. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Fused DENSE batch path was unreachable — a LUT3 lm_head, not the VL wrapper** — RESOLVED 2026-08-11. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Routed KVarN prefill wrote K in the WRONG BASIS — both arms** — FIXED 2026-08-11. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **max_seq was inflated by the generation budget, sizing KV for a 132K context** — FIXED. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Paged Opus MoE on the 35B wedged the GPU (MES hang → driver reset)** — FIXED 2026-08-10. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Routed OQ experts repacked for kernels that never ran — non-finite KLD** — FIXED. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **tiny-quant was RED for Opus across four MoE families** — RESOLVED by rebase. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Quantized-from-HFQ artifacts lose config/tokenizer (dangling v2 tail pointer)** — RESOLVED. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **hipfire-daemon inference worker killed on client disconnect (was theorized as "GPU fault under model-swap churn")** — RESOLVED. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Batched prefill garbage for bf16/f16 llama models (was: "attention_q8_0_kv_batched masked prefill garbage for decoupled head_dim")** — RESOLVED. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **bf16 KLD reference artifacts contain chunk 0 replicated 1175×** — RESOLVED 2026-08-29 — damaged artifacts deleted. [archive](docs/bugs/2026-08-29-fixed-archive.md)
- **Four serving-path lifecycle defects** — FIXED 2026-08-29. [`2026-08-29-serving-lifecycle-defects`](docs/bugs/2026-08-29-serving-lifecycle-defects.md)
- **Diffusion img2img noise is wrong on BOTH scheduler families** — FIXED 2026-08-29. [`2026-08-29-diffusion-img2img-and-samplers`](docs/bugs/2026-08-29-diffusion-img2img-and-samplers.md)
- **Drafter-free n-gram spec decode never speculated — the spine was discarded every step (8% slower than plain AR)** — FIXED 2026-09-01. [`2026-09-01-ngram-spine-discarded-by-block-fallback`](docs/bugs/2026-09-01-ngram-spine-discarded-by-block-fallback.md)
- **`BlockController` could never return a block above 8; `dflash_adaptive_b` applied to nothing; renamed keys' ENV spellings evaporated** — FIXED 2026-09-01. [`2026-09-01-spec-block-controller-and-naming`](docs/bugs/2026-09-01-spec-block-controller-and-naming.md)
- **Speculative decode does not reproduce AR output — the BATCHED verify forward (b>=2) disagrees with the single-token forward at slot 0; FP16 DeltaNet state is the main contributor but not the only one; invalidates cross-width benchmarks** — OPEN, diagnosed 2026-09-01. [`2026-09-01-spec-decode-not-output-equivalent-to-ar`](docs/bugs/2026-09-01-spec-decode-not-output-equivalent-to-ar.md)
- **`tiny-prefill-gate` ran its hidden-state probe at `--n 32`, below the 256 prefill chunk — every per-KV-mode reading was structurally 0, so the 5e-2 ceiling could never fire** — FIXED 2026-09-01. [`2026-09-01-spec-decode-not-output-equivalent-to-ar`](docs/bugs/2026-09-01-spec-decode-not-output-equivalent-to-ar.md)
- **DeltaNet low-redundancy FP32 guard had no production caller — `default_state_quant` opened with `let _ = config;`, so small models silently got FP16 state with no warning** — FIXED 2026-09-02. [`2026-09-02-deltanet-redundancy-gate-had-no-caller`](docs/bugs/2026-09-02-deltanet-redundancy-gate-had-no-caller.md)
- **`kv_cache` never reaches the loader — it reads a raw `HIPFIRE_KV_MODE` env var instead, so every model runs KVarN 4-bit and `kv_cache=fp32` is silently ignored** — OPEN, diagnosed 2026-09-02. [`2026-09-02-kv-cache-setting-never-reaches-the-loader`](docs/bugs/2026-09-02-kv-cache-setting-never-reaches-the-loader.md)
- **A batched read of a quantised KV cache != the same reads one row at a time — one root under both the prefill batched/per-token divergence and spec-decode's non-equivalence to AR; fp32 KV is clean, bit width is irrelevant** — LOCALISED 2026-09-02. [`2026-09-02-kv-batched-vs-single-row-read-divergence`](docs/bugs/2026-09-02-kv-batched-vs-single-row-read-divergence.md)
- **n-gram speculation + FP16 DeltaNet state drives the model into echoing 87% of its prompt — and the resulting "3.4x speedup" IS that corruption; neither ingredient degenerates alone** — OPEN, measured 2026-09-02. [`2026-09-02-ngram-speculation-drives-prompt-echo`](docs/bugs/2026-09-02-ngram-speculation-drives-prompt-echo.md)
- **KVarN write AND read paths proven batch-invariant (bit-exact, guarded against vacuous passes) — the batched/per-token divergence first appears at layer 0, which has NO KV, so it originates in the DeltaNet recurrence** — measured 2026-09-02. [`2026-09-02-kvarn-write-path-is-batch-invariant`](docs/bugs/2026-09-02-kvarn-write-path-is-batch-invariant.md)
- **FP16 DeltaNet recurrence was not chunk-invariant — FIXED (dither index reconciled across all three f16 GDN kernels, then narrow per token); now gated by tiny-deltanet-gate — one 64-token launch differs from 64 single-token launches in every state element (worst |rel| 1.989); FP32 is bit-exact. Origin of the prefill and spec/AR divergences** — OPEN, measured 2026-09-02. [`2026-09-02-fp16-deltanet-recurrence-is-not-chunk-invariant`](docs/bugs/2026-09-02-fp16-deltanet-recurrence-is-not-chunk-invariant.md)
- **8-bit calibration (`oq8+`) makes KLD up to 1250x WORSE on qwen2/dots_ocr and 1.8-2.9x worse on four more families, reproducibly on two GPU arches — and the tiny-quant baselines record the broken values as expected** — OPEN, measured 2026-09-02. [`2026-09-02-oq8-calibration-makes-kld-catastrophically-worse`](docs/bugs/2026-09-02-oq8-calibration-makes-kld-catastrophically-worse.md)
- **tiny-prefill-gate reported 5 hidden-state comparisons that never ran as passes (batched arm declines, both arms go per-token, `0.00e0` reads as "invariant satisfied")** — FIXED 2026-09-02, now reported as NOT-MEASURED. [`2026-09-02-prefill-hidden-probe-vacuous-passes`](docs/bugs/2026-09-02-prefill-hidden-probe-vacuous-passes.md)
