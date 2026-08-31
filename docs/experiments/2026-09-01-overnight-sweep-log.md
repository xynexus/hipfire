# Overnight performance sweep — run log (2026-09-01)

Goal: `docs/plans/2026-09-01-overnight-performance-sweep-goal.md`
Branch: `perf/overnight-2026-09-01` (branched from `origin/master` @ 7f739be21)

Entries are appended as work happens, failures included.

---

## Step 0 — land in-flight work · DONE

Tree was dirty on `master`; branched first.

- `563ff6f02` n-gram hoist off `DflashState` onto `LoadedModel`, plus
  `NgramState::take_live_for` and its scope-leak test.
- `d3d1f24f0` corrected `docs/perf/ddtree-vs-chain-opus.md` + `serve_real`
  step-count flag.

Corrections folded into the doc (both were errors in yesterday's draft):
- called Qwen3.6-27B "dense" — it has **48 linear-attention layers of 64**, so
  the hybrid-recurrent path *was* exercised;
- recommended budget 8 / topk 2. Real optimum is **budget 12, topk 1** =
  32.89 tok/s (2.16x AR, 3.21x chain), and topk=1 is a linear spine, not a tree.

Gate: pending below.

---

## W1 — `is_batchable_la` and the G128 Opus dtypes · IN PROGRESS

**Finding that changed the item's shape.** The goal file framed this as "add two
arms". It is not. `OqCompactG128` and `Oq8G128` appear in the GEMV tables and
**nowhere in any GEMM route** — so the decline was load-bearing, not an
oversight. Admitting them blindly would route to an unhandled batched GEMM.

Then the good news: `gemm_oq8_grouped_wmma` **already takes `group` at runtime**
(`n_groups = K/group`, `tiles_per_group = group/16`; both integral at 128).
Nothing about the kernel is 256-specific. Three Rust call sites above it were:

| site | was |
|---|---|
| `ensure_oq8_scratch_batched` (mod.rs:2744) | `need_xs = n * (k / 256)` |
| `quantize_act_oq8_batched` (quant.rs:2344) | `ng = k / 256` |
| `gemm_oq8_grouped_prequant` (quant.rs:2403) | `ng = k / 256` |

**The first is a latent buffer overrun.** The activation-scale plane is one f32
per group per row, so it DOUBLES at group 128. Admitting a G128 dtype without
fixing it writes past `oq8_xs_batch`.

Landed (commit below): each gains a `_g` variant taking `group`; the originals
delegate with 256, so existing callers are byte-identical. Sizing extracted to
`oq8_batched_xs_len(n, k, group)` with a GPU-free test asserting
`g128 == 2 * g256` — the ratio, not just the absolute, so reintroducing the
constant fails.

Gates: `cargo test -p hipfire-rdna` 63/63, clippy clean, `no-gpu-ci` exit 0.

**Next for W1** (not yet done):
1. Route `Oq8G128` / `OqCompactG128` through the `_g` variants at the prefill
   call sites in `prefill_lowered.rs` (4 sites) — needs the dtype at the site.
2. ⚠️ Confirm the rotation basis: G128 needs FWHT-128 (seeds 43/1043), not
   FWHT-256 (42/1042). A mismatch does not cancel and yields plausible wrong
   logits — check what `RotationPlan::FwhtG128` actually drives on this path.
   **Do not admit the dtypes until this is verified.**
3. Then add the `is_batchable_la` arms in BOTH `qwen35/mod.rs` and
   `runtime/dispatch.rs` (matching-pair comment says keep in sync).
4. Need a G128 artifact to test on — none exists locally; quantize a small model
   to oq8 at group 128.

### W1 step 2 — the rotation basis · CORRECTION, and W1 is DEPRIORITISED

⚠️ **My first write-up of this step was wrong in three ways.** Kept visible
rather than rewritten, because the error is instructive: I re-derived a known
result, mis-stated the prior art, and proposed a fix that does not work.

**`docs/todo/2026-08-30-oq8g128-protected-set.md` already had this**, and its
title says so: *"needs a G128 fused rmsnorm+rotate"*. Specifically:

1. I wrote that the KLD 0.83 "was attributed to the format". It was not. That doc
   argues Oq8G128 is **21.6% better than the Q8F16 it replaces using FEWER bits**
   (RMSE 4.110e-3 vs 5.243e-3, 8.125 vs 8.5 b/w). Nobody blamed the format.
2. I presented the fused-256 root cause as a finding. It was already documented,
   along with the note that `gemm_oq8_grouped_wmma` "needed nothing (it already
   took `group`)" and that `mq_rotate_x_128` was only missing a batched dispatch.
3. **My proposed fix was wrong.** I said "unfuse it: rmsnorm →
   `rotate_x_mq_128_batched` → `gemm_..._g(128)`". That misses the actual
   difficulty, which the doc states plainly: qwen35's lowered executor fuses
   rmsnorm+rotate into one kernel and **every attention projection consumes that
   single FWHT-256-rotated activation**. Giving one weight the 128 basis needs a
   G128 fused rmsnorm+rotate *and* a way to split the shared activation, because
   the layer then needs both a 256- and a 128-rotated copy of the same input.
   Unfusing one call site does not get there.

**What survives from this branch's work:** commit `d7ba9d27a` is still real. The
batched-path scratch allocator sized the activation-scale plane `n * (k / 256)`,
which doubles at group 128 — a latent overrun the prior doc does not cover,
because it was about the GEMV path and the batched path was blocked anyway. That
groundwork stands and is behaviour-neutral (18 tiny-state hashes bit-identical).

**W1 is deprioritised, and this changes the ranked list.** The goal file ranked
it #2 on the premise "add two dispatch arms". That premise is dead: it needs a
new fused G128 kernel family (`RmsnormRotateMqG128` plus AWQ/batched siblings —
none of the `RmsnormRotateMq*` keys have a G128 form) and an activation-splitting
scheme in the lowered executor. That is a multi-day kernel job, not an overnight
dispatch fix.

Two further constraints from the same doc, worth not rediscovering:
- there is **no AWQ sidecar support at G128** (`rotate_x_mq_128_for` takes the
  non-AWQ path) — check before enabling calibrated paths;
- **LDLQ has no G128 form** (`cli.rs` refuses `--ldlq` at group 128, because
  `oqplus_compact_ldlq_pack` emits 256-element blocks).

And a measurement warning: the tiny-quant KLD numbers **cannot adjudicate this**
— the same change scored 6.9x better on one family and 2x worse on another, on
random-init fixtures where KLD measures perturbation sensitivity, not quality.
Use the reconstruction test.

**Next: skip to W2** (the 3.2x chain-vs-spine gap), which is measured, local, and
has no kernel dependency. W1 resumes only if a G128 fused rotate is worth a
dedicated session.

**Process lesson for the rest of this run:** grep `docs/todo/` and `docs/plans/`
for the subsystem BEFORE analysing it. Two hours of this tick re-derived a
document that already existed.

---

## W2 — chain vs tree · THE HEADLINE WAS WRONG, and the cause is a general trap

**Retraction first.** I reported chain DFlash at 10.26 tok/s, "loses to AR by
0.67x", and a 3.2x chain-vs-tree gap I called "bug-shaped". All of that is
withdrawn. Warm, same session, repeated twice:

| arm | pass 1 | pass 2 | tau | vs AR |
|---|---|---|---|---|
| AR | 15.25 | 15.25 | 1.00 | 1.00x |
| chain DFlash B=16 | 22.31 | 21.98 | 2.49 | **1.45x** |
| DDTree budget 12, topk 1 | 32.69 | 32.76 | 5.79 | **2.14x** |

**Both speculative paths beat AR.** No defect. The real gap is 1.47x.

### The cause, proven not assumed

Moved `~/.hipfire/kernels/gfx1151` aside (pure JIT cache, regenerates on demand,
restored afterwards to its original 1846 entries) and re-ran the same command:

| kernel cache | chain B=16 | tau |
|---|---|---|
| cold | **6.41 tok/s** | 2.4865 |
| warm | **22.13 tok/s** | 2.4865 |

**3.45x, tau bit-identical.** The computation is the same; only wall-clock
differs, because kernels JIT-compile INSIDE the timed window. The identical tau
is what makes it dangerous — every acceptance metric looks healthy while
throughput reads a third of real.

This is now measurement trap #3 in the goal file. It invalidates any first-run
benchmark in this repo, and it is worth more than the W2 item it destroyed.

### What actually survives as W2

Block size is NOT the confound — chain returns tau **2.4865 at both B=12 and
B=16**, accepting the same 92 tokens, with accept_rate falling 0.226 -> 0.166 as
the extra width is wasted. So chain's draft saturates at ~2.5 accepted no matter
how much room it gets, while the same drafter reaches 5.79 through the tree
path's confidence-pruned spine (`build_ddtree_tree_with_cutoff`) vs chain's
unconditional per-position argmax.

Do NOT retry the DFlash2 candidate selector: already implemented, gated behind
`HIPFIRE_DFLASH2_SELECTOR=1`, measured WORSE (tau 2.421 -> 2.25, decode 6.14 ->
5.92).

### Housekeeping

- `~/.hipfire/models/Qwen3.6-27B--dflash.bf16.hfq` (3.46 GB, arch 20) promoted
  out of /tmp. Reproduce: `dflash_convert --input
  /srv/huggingface/models--z-lab--Qwen3.6-27B-DFlash/snapshots/*/ --output <path>`.
- Corrected: `docs/perf/ddtree-vs-chain-opus.md`, the goal file's baseline table
  and W2, and the published artifact.

---

## W3 — KVarN × batched prefill · CLOSED, already fixed before this run started

**Does not reproduce.** `compare_prefill_hidden_paths` on
`qwen3.5-2b--bf16.hfq`, warm cache, against the fp32-KV reference:

| KV mode | batched | per-token | ratio |
|---|---|---|---|
| q8 | 2.057e-2 | 1.580e-2 | 1.30x |
| **kvarn** | **1.994e-2** | **1.607e-2** | **1.24x** |

Against the recorded defect of batched **9.334e-1** vs per-token 1.633e-2 — a
57x gap. It is now 1.24x, i.e. gone.

**Vacuity check, because the prior write-up warns about exactly this:** on a
compact target both arms silently fall back to per-token and report *identical*
numbers (the 27B read batched == per-token == 1.203e-2, which proves nothing).
Here the two arms differ (1.994e-2 vs 1.607e-2), so the batched path really ran.
Also confirmed `--n` is honoured (`rows = ....min(n)`, `tokens = (0..n)`); the
metric is invariant n=8..96 only because it is a MAX and the max sits in the
first 8 rows.

**Closed by `96d53741c` (2026-08-29), before this goal file was written.** That
commit overturned both premises the item rested on:

1. The KVarN per-token write path was never broken — `kvarn_attend`'s
   segment-then-flush ordering fixed it. Batched and per-token agree at 3.31e-4
   given identical inputs, flat across n=32/127/128/129/200, straight through the
   128-token flush boundary that used to step.
2. The residual divergence is **fp16 narrowing cadence in the DeltaNet state**,
   not KV at all: batched prefill is one launch narrowing S once, per-token is n
   launches narrowing n times. KVarN is the AMPLIFIER (1.04e-2 under q8 vs
   1.06e-1 under KVarN at the next layer), not the source.

The two real fixes (fp32 S for the duration of a prefill, or fp16 + an int8
mantissa residual at 3 B/element) are already costed in BUGS.md and are design
calls, not cleanups.

### ⚠️ Pattern — three for three

W1, W2 and W3 have each had a **stale premise**:

- W1 framed as "add two dispatch arms"; the real blocker was documented in
  `docs/todo/2026-08-30-oq8g128-protected-set.md` a day earlier.
- W2 rested on a cold-cache measurement artifact of my own making.
- W3 was fixed on 2026-08-29, two days before the goal file listed it as open.

**The ranked list was built from memory entries rather than from the repo's own
docs, and the repo is ahead of the memory.** For the remaining items, the first
action must be `git log --grep` plus a `docs/todo` + `docs/plans` + `BUGS.md`
sweep for the subsystem, BEFORE any analysis or measurement. Two of these three
would have been closed in minutes that way.

---

## Bulk pre-screen of the remaining items (applying the lesson)

Screened every remaining item with `git log --grep` + a `docs/todo`,
`docs/plans`, `docs/bugs` sweep BEFORE analysing any of them. Results:

| item | repo status |
|---|---|
| W5a calib seq-len 2048 | no prior work — **live** |
| W5b LQER low-rank residual | no prior work — **live** |
| W5c OQ8 router default | no prior work — **live** |
| W5d embed 16-bit rule | no prior work — **live** |
| W6a GTT 2 MiB rounding | `docs/plans/2026-08-23-gtt-2mib-rounding-moe-memory.md` — **half closed, half now actionable (below)** |
| W6b repacker pre-split | only this branch's own commits — **live** |
| W6c oq8 GEMM ceiling | no prior work — **live** |

Two minutes of screening, and it kept the remaining seven honest.

## ⭐ NEW FINDING — 20 GiB of MoE memory from one allocation shape

The GTT doc attributes the 122B's failure to two stacked ~1.9x amplifications:

| cause | factor | status |
|---|---|---|
| compact -> Oq8 expansion on load | 1.80x | **CLOSED** — compact-resident is default-on now (`compact_resident_enabled`, only `0/off/false/no` disables) |
| GTT allocation rounded to 2 MiB | 1.88x | still live |

But the surviving half is not a driver problem to work around — it is an
allocation SHAPE problem, and the fix already exists in this tree.

`HIPFIRE_ALLOC_REPORT` on `Qwen3.6-35B-A3B--oq4.25++` shows the resident loader
allocating each expert's two projections SEPARATELY:

    20.31 GiB   10240 x 2129920 B   <- expert gate_up
    10.16 GiB   10240 x 1064960 B   <- expert down

Both land just over a rounding boundary. Through `gtt_alloc_cost`:

| shape | GTT bytes/expert | vs raw |
|---|---|---|
| two separate allocations | 6 291 456 | 1.969x |
| **one allocation** | **4 194 304** | **1.313x** |

**33.3% saved — 2.00 MiB per expert, 60.0 GiB -> 40.0 GiB across 10240 experts.**

And `weight_pager` ALREADY does this: a module holds gate_up and down back to
back in one buffer, which is why `ResidentExpertViews` hands out byte offsets
rather than two tensors. Its own comment says "the GTT rounding is paid once per
module rather than per projection". The resident loader simply never adopted the
shape.

**So the item is not "work around 2 MiB rounding" — it is "give the resident MoE
loader the same one-allocation-per-expert shape the pager already uses."** That
reframing makes it tractable, and 20 GiB is the difference between the 122B
loading and not.

Promoted to the top of the remaining queue: highest measured payoff, existing
pattern to copy, no new kernel.

### ⚠️ CORRECTION to the GTT finding — it is model-shaped, and it does NOT unlock the 122B

I wrote "20 GiB is the difference between the 122B loading and not." **Wrong.**
Computed from the real artifacts' routed-expert tensor sizes through
`gtt_alloc_cost` (no GPU needed — the rounding rule is pure arithmetic):

| model | expert pairs | raw | separate | one/expert | saving |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B oq4.25++ | 10240 | 16.5 GiB | 30.0 (1.814x) | 20.0 (1.209x) | **10.0 GiB (33.3%)** |
| Qwen3.5-122B-A10B oq4.25++ | 12288 | 63.0 GiB | 75.8 (1.204x) | 74.5 (1.184x) | **1.3 GiB (1.7%)** |

**The benefit is entirely a function of where per-expert size sits on the 2 MiB
grid.** The 35B's pairs are ~1.6 MB and land just over a boundary, so rounding
costs 1.81x and merging recovers a third. The 122B's are ~5.25 MB, already well
above the grid, so rounding only costs 1.20x and merging recovers almost nothing.

So: worth doing for the 35B class, **not** a 122B unlock. The 122B's problem was
the compact->Oq8 expansion (1.80x), and that is already closed by
compact-resident being default. My earlier 60->40 GiB figure also came from the
doc's POST-EXPANSION alloc report; against today's compact-resident reality the
same model is 30.0 -> 20.0 GiB.

Two lessons, and the second is the general one:

1. Do not carry a memory-resident number (the doc's alloc report) into a present
   claim without re-deriving it against current defaults. Compact-resident landed
   between that doc and now and moved every figure in it.
2. **A ratio measured on one model is not a property of the code.** This one
   varies 33.3% -> 1.7% across two models of the same family and quant. Check the
   shape before generalising.

Also confirmed while checking: the GPU pool RECYCLES WHOLE BUFFERS, it does not
sub-allocate from slabs (`pool.rs::alloc` pops a free-list entry or `hipMalloc`s
at the requested size). So routing expert loads through the pool is NOT an
alternative fix — the per-tensor rounding applies either way. The allocation
shape is the only lever.

**Revised status:** still the best remaining item by measured payoff on the 35B
class, but it is a 10 GiB win on one model shape, not a 20 GiB structural unlock.
The refactor it needs (`load_moe_expert` returning bytes so the caller can pack
gate_up||down into one buffer, with the existing pointer tables retargeted at
base and base+len) touches the loading path for every MoE model, so it wants a
dedicated session with real-model validation — not an unattended 3 a.m. edit.

---

## Implementations landed this tick

**1. `feat(calib)` — calibration sequence length defaults to 2048** (`548a10711`).
`calib_sequences` split only when `HIPFIRE_CALIB_SEQ_LEN` was set; unset gave
`usize::MAX`, so the O(n²) shape was the default and the cost grew with the very
budget that makes calibration worth doing. 2048 matches the n_ctx KLD references
are built at. Evidence was already in the function's own doc comment: zaya1-8b at
32768 tokens, 10746 s -> 1065 s, quality unchanged.

⚠️ The first version of the test sized its input as `DEFAULT_CALIB_SEQ_LEN * 3`,
so raising the constant grew the input to match — it could not fail. Input is a
literal now, and the negative control (revert to `usize::MAX`) genuinely fails.

**2. `fix(docs)` — six ASCII tables rustdoc was compiling** (`db02784e6`).
`cargo test --doc` failed workspace-wide; indented blocks in doc comments are
treated as Rust. Fenced as ```text. Workspace doctests are clean now — they were
unusable as a signal before, because each crate stopped at its first failure.

**3. `fix(gate)` — tiny-spec-gate demanded coverage it had made unreachable.**
The checkpoint-rollback stage fails on `origin/master` too (verified in a scratch
worktree at 7f739be21), so every spec-touching commit was tripping it and
escalating to the full coherence battery.

Root cause is in a log line the gate never read:

    DFlash checkpoint rollback: DISABLED — it needs FP32 DeltaNet state and this
    session resolved FP16.

The stage set `HIPFIRE_DFLASH_CHECKPOINT_ROLLBACK=1` but not
`HIPFIRE_DN_STATE_FP16=0`, so `replay_checkpoint` was pinned at 0 and the
assertion could not pass whatever the code did. Everything else worked as
designed — 31 cycles, 0 accepted (random-init drafter, deliberately), rollback
going through `replay_full_prefill=31`.

One env var. Gate now reports **"spec loop OK (checkpoint engaged 31x, output ==
AR)"**, which is exactly the second-model coverage `825d3ccfc` said was missing.
Non-vacuous both ways: fails without the fix, and the 31 firings still produce
AR-identical output.

---

## ⭐ FULL AUDIT OF THE 20-ITEM LIST — six items were already done or void

Verified each against the repo rather than against memory. This is the headline
result of the night, and it is about the list, not the code.

| # | item | verified status |
|---|---|---|
| 1 | chain vs tree 3.2x gap | **RETRACTED** — my own cold-cache artifact; chain beats AR 1.45x |
| 2 | `is_batchable_la` G128 arms | **premise wrong** — needs a G128 fused rotate + activation splitting (multi-day); groundwork landed |
| 3 | KVarN x batched prefill 57x | **CLOSED** — fixed by `96d53741c`, 2026-08-29 |
| 6 | oq8 GEMM at 54% of ceiling | unverified; M-slab work not where I expected |
| 9 | unify accept loops onto `Speculator` | **LIVE** — still zero implementors |
| 10 | gemma3 windowed attention kernel | **DONE** — `attention_swa_gqa_batched` + `swa_ring_write` already wired in `gemma3/forward.rs` |
| 11 | lm_head two-stage | **DONE** — `lmhead_twostage` shipped; extending to the body is a research item |
| 12 | LQER low-rank residual to config | **LIVE** — `HIPFIRE_LOWRANK_R` still Developer-env only |
| 13 | GuidedQuant / end-loss gradients | **LIVE** — no implementation in tree |
| 14 | calib seq-len 2048 default | **DONE TONIGHT** (`548a10711`) |
| 16 | embed stays 16-bit | **DONE** — `--embed-precision` defaults to `source`; the override runs before the Q8 arm |
| 17 | OQ8 MoE router default | **VOID — would be a DOWNGRADE.** Routers already default to lossless BF16 (`--q8-router` is opt-in). Promoting to OQ8 makes them worse. The "lying counter" is also already gone. |
| 18 | GTT 2 MiB rounding | half closed (compact-resident); remainder is 33% on the 35B, **1.7% on the 122B** — not the unlock I claimed |
| 20 | qwen4_exp batched forward | **LIVE** — still per-token by construction |

**Six of the twenty were already done, void, or retracted.** One (#17) would have
made things worse if implemented: routers are lossless today, and the item asked
to quantize them to 8-bit.

### Why — and it is not that the repo moved fast

The list was built from **memory entries**, and every one of those entries was
true when written. The repo simply moved past them:

- `--embed-precision source` landed 2026-07-22
- gemma3 SWA + ring-write landed before this branch
- the router policy went to lossless BF16
- KVarN batched was fixed 2026-08-29
- the G128 blocker was documented 2026-08-30

The list was written 2026-09-01 and did not check any of it. **Memory is a
cache, and nothing was invalidating it.**

### What this changes

1. **Stop treating the ranked list as a work order.** It is a set of hypotheses
   whose evidence is up to six weeks stale.
2. The genuinely live items are fewer and different: **#9 (Speculator seam, zero
   implementors), #12 (LQER still env-only), #13 (GuidedQuant, the real quality
   ceiling), #20 (qwen4_exp batched forward), and #18 for the 35B class.**
3. Anything sourced from memory needs `git log --grep` + a `docs/todo` sweep
   before it earns a rank. Six items, ~2 minutes each, would have caught all of
   this before the list was published.
