# Overnight performance sweep — run log (2026-09-01)

Goal: `docs/plans/2026-09-01-overnight-performance-sweep-goal.md`
Branch: `perf/overnight-2026-09-01` (branched from `origin/master` @ 7f739be21)

Entries are appended as work happens, failures included.

---

# MORNING SUMMARY

**Headline: the ranked list was the problem, not the code.** Of twenty items,
**seven did not survive contact with the repo** — already done, already fixed,
or wrong about what the thing is. One of them (#17, "make the OQ8 router the
default") would have made quality *worse*: routers already default to lossless
BF16. Another (#12, LQER) would have shipped a knob that emits bf16-sized
artifacts with simulated quality.

The list was built from memory entries. Every one was true when written; nothing
invalidates a memory when the code moves. `--embed-precision source` landed
2026-07-22, gemma3's windowed attention was already wired, KVarN batched was
fixed 2026-08-29, the G128 blocker was documented 2026-08-30 — and the list was
written 2026-09-01 without checking any of it.

## What shipped (21 commits, all gates green)

| commit | what |
|---|---|
| `563ff6f02` | n-gram state hoisted off `DflashState` onto the model, + a scope-leak test |
| `d7ba9d27a` | batched oq8 path made group-parametric — **fixes a latent buffer overrun** at group 128 |
| `548a10711` | calibration sequence length defaults to 2048 (was `usize::MAX`, i.e. O(n²)) |
| `db02784e6` | six ASCII tables rustdoc was compiling; workspace doctests clean |
| `f212ae076` | **`tiny-spec-gate` fixed** — it had been failing on `master` since it landed |
| `217e5c909` | **cold-cache guard** — the trap that produced this run's own wrong headline |
| `322324721` | **#4 landed: paged decode 0.25 -> 0.18 s/tok (1.39x)** on the 180B |

## What I got wrong, and corrected

1. **"Chain DFlash loses to AR by 0.67x, a 3.2x bug-shaped gap."** A cold JIT
   kernel cache. Warm: chain 22.1 tok/s, **1.45x AR**. Proven by moving the
   cache aside — cold 6.41 vs warm 22.13, **3.45x, with tau bit-identical**.
   Corrected in the doc, the goal file, the artifact and memory, and the guard
   in `217e5c909` now catches it.
2. **"20 GiB is the difference between the 122B loading and not."** It is 10.0
   GiB on the 35B and **1.3 GiB on the 122B**. The benefit depends on where
   per-expert size sits on the 2 MiB grid.
3. **W1's root cause.** I re-derived a document that already existed and then
   proposed a fix that cannot work.

## Where to pick up

Live and worth doing, verified against the tree:

- ~~#4 paged-expert per-access overhead~~ — **LANDED** (`322324721`), 1.39x.
- **#18 MoE allocation shape** — designed in
  `docs/todo/2026-09-01-moe-expert-pair-allocation.md`, 10 GiB on the 35B class.
  Blocked on one ownership decision (`sub_offset` yields a `NonOwning` buffer and
  `dispose` debug-asserts on one by design), not on effort. **Closest to ready.**
- **#9 Speculator seam** — the trait exists, `SpecTarget` has three implementors,
  `Speculator` has zero. Five hand-rolled accept loops could collapse onto it.
- **#13 GuidedQuant** — the real quality ceiling; `hipfire-train` is already fp32
  GPU autograd, so the keystone exists.
- **#20 qwen4_exp batched forward** — gates everything speculative on the 180B.

Everything else on the list is closed, void, or needs re-deriving from the repo
rather than from memory.

## Process changes earned tonight

1. **Grep before ranking.** `git log --grep` + `docs/todo` + `docs/plans` +
   `BUGS.md`, per item, before it earns a place in a plan. Two minutes each would
   have caught all seven.
2. **Discard the first run after any rebuild.** Now enforced by a warning.
3. **Check a gate can fail.** `tiny-spec-gate` asserted a path it had itself
   disabled; the guard in `serve_fixture` and the ratio assertion in
   `oq8_batched_xs_len` exist for the same reason.

---


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

### #12 LQER — MISCHARACTERISED (seventh)

"Promote the low-rank residual to a first-class config setting with a default
rank per bit-width." It cannot be promoted: it is not a format.

`HIPFIRE_LOWRANK_R` lives inside the **`use_qtip_sim` simulation path**
(`cli.rs:12816`), operates only on `QuantType::BF16` tensors, and ends with

    t.data = f32_slice_to_bf16_bytes(&deq);

It writes BF16 back. It is a **quality probe**: it bakes the quality of a
hypothetical W4 + UᵥVᵥ correction into a bf16 artifact so KLD can be measured
without building the format. Its own comment says so — "sims the quality of W4 +
a 2-WMMA UᵥVᵥ correction".

Promoting it to a config default would hand operators a knob that produces
**bf16-sized artifacts with simulated quality**. The measurement behind the item
(-13% at 2 bit; qtip4+LDLQ+lr32 beating oq4++) is real and says the format is
worth BUILDING — storage layout plus a 2-WMMA runtime correction. That is a
project, not a config flip.

**Seven of twenty now dead, done, or mischaracterised.** And the pattern in the
last three is sharper than "stale": #17 would have made quality worse, and #12
would have shipped a misleading knob. A memory-sourced rank is not just
out-of-date, it can be actively wrong about what the thing IS.

### Remaining live items are all large

#9 (Speculator seam), #13 (GuidedQuant), #18 (MoE allocation shape), #20
(qwen4_exp batched forward). None is an overnight win; each wants a dedicated
session with real-model validation. **The cheap wins are exhausted** — which is
itself the answer to "how many of the twenty survive contact with measurement".

---

## Cold-cache guard landed (`217e5c909`)

The trap that produced this run's wrong headline now announces itself.
`hipfire_rdna::jit_compiles()` counts kernels hipcc actually compiled;
`dflash_spec_demo` prints a warning before BOTH `BENCH METRICS` blocks when it
is non-zero.

Verified by moving the cache aside:

    cold: "COLD KERNEL CACHE: 32 kernel(s) were compiled during this run"  6.38 tok/s
    warm: silent                                                          26.32 tok/s

Two mistakes worth recording, because both are the same shape as the bugs this
run kept finding:

1. The first attempt put the guard before the AR-baseline metrics block only.
   The test drove the spec arm, so it printed nothing and looked like the counter
   was broken. **A guard in one of two arms covers nothing** — same failure as
   tiny-spec-gate asserting a path it had disabled.
2. The helper was then inserted between `#[cfg(not(feature = "deltanet"))]` and
   its stub `main`, so the attribute applied to the helper and the file had two
   unconditional `main`s. Anchoring text insertion on a string that appears more
   than once is how both of these happened.

---

## #4 LANDED — paged decode 1.39x

`gate_up.expert()` and `down.expert()` each locked, hashed and ensured **the same
module** — one module holds both projections. `ExpertStack::expert_pair` does it
once.

Qwen3.8-Flash-Next-180B, 8 GiB budget, 64 decode steps:

| | before | after |
|---|---|---|
| decode | 0.25 s/tok | **0.18 s/tok (1.39x)** |
| pager hits | 48245 | **15605** |
| cold loads | 17035 | 17035 |
| evictions | 15670 | 15670 |
| argmax | 1892 (13.9764) | unchanged |

The hit delta is exactly 64 x 48 x 10 plus prefill, so the mechanism is confirmed
rather than inferred. This is the fix the earlier measurement pointed at — a
48 GiB budget with zero evictions being no faster than an 8 GiB one thrashing
15670 said the cost was per-access work, not fetching.

**Trap #3 caught me again, in a new form.** The first measurement after the edit
read 0.24 s/tok with the pager counters IDENTICAL to baseline — because
`cargo build --workspace` had not rebuilt the example, so I was timing the OLD
binary. The counters were the tell: if ensures had halved, hits could not be
unchanged. **When a change shows no effect, check you measured the change** — the
same discipline as discarding a cold first run.

I had also written this item off as "a dedicated session". That was wrong: it was
one contained method in code I wrote earlier tonight, and it is the largest
throughput win of the run.

---

## Three corrections/findings from re-measuring #4

### 1. The 1.39x claim was a best-single-run number — it is ~1.3x

Repeated the 8 GiB post-fix config three times, clean:

    0.21 · 0.19 · 0.19 s/tok   (median 0.19)

Pre-fix samples were 0.25, 0.25, 0.24. So the honest figure is
**~0.245 -> ~0.19 s/tok, about 1.3x**, not the 1.39x taken from a single 0.18
reading. The mechanism is unchanged and still confirmed by the hit-count delta
(48245 -> 15605); only the magnitude was over-stated. Commit `322324721`'s
message says 1.39x and should be read as the best observed run.

### 2. NEW TRAP — a run is contaminated by the previous run's residency

One 8 GiB run read **0.27 s/tok with counters identical to a 0.19 run**. It had
been preceded by a 48 GiB run that left 45.2 GiB resident; `free` showed 22 GiB
still in buff/cache. On a UMA box the page cache and GPU memory are one pool, so
the previous run's footprint is still charged to the next one.

**Successive benchmark runs are not independent when they differ in residency.**
Interleave arms, or re-run each arm until it settles, and treat the first run
after a large-footprint arm the same way as a cold-cache run: discard it. This is
the second time tonight the same discipline was needed and the second distinct
mechanism behind it (the first was JIT compilation).

### 3. #19 (repacker pre-split) is KILLED by measurement

The item's premise is that page-in cost matters, so pre-splitting compact planes
at repack time — making `prepare_expert_module` a memcpy instead of a transform —
would pay. Post-fix, comparing budgets:

| budget | cold loads | evictions | decode |
|---|---|---|---|
| 48 GiB | 7711 | **0** | 0.30 s/tok |
| 8 GiB | 17035 | 15670 | **0.19 s/tok** |

**2.2x MORE cold loads is 1.6x FASTER.** Cold-load count does not drive decode
time here, so making each cold load cheaper attacks the wrong term. #19 is not
worth a session on this evidence.

And the counter-intuitive corollary, which is the useful part: **a bigger paging
budget is SLOWER**. 45.2 GiB resident on a 128 GB UMA box pressures the shared
pool enough to cost more than the page-ins it avoids. The budget should be tuned
DOWN, not up — the opposite of how a cache budget usually behaves, and worth
knowing before anyone "optimises" by raising it.

---

## #7 MoE decode roofline · VERIFIED — premise stale, and a dead end found

### The item's numbers were stale

It said "AR 57.3 tok/s = 36% of a 157 ceiling; `grouped_v3` = 41.5% of decode".
Two perf commits have landed since that was written — `dfc8141d7` (dwordx4 weight
loads in indexed MoE gate_up, 57.3 -> 60.1) and `08dd9eea9` (split-K for small-M
compact GEMV, 60.1 -> 63.4).

Measured now, `hipfire-eval --battery speed` on `Qwen3.6-35B-A3B--oq4.25++`:

    pp32   decode 62.8 tok/s   prefill 224.6 tok/s
    pp128  decode 60.1 tok/s   prefill 360.1 tok/s

Roofline: 3B active x 0.53 B/param (oq4.25) = 1.59 GB/token; 248.5 / 1.59 =
**~156 tok/s**, so decode is at **40%**, not 36%.

### Where decode time actually goes

`rocprofv3 --kernel-trace` via the daemon stdin protocol, 64 greedy tokens,
queried from `rocpd_kernel_dispatch` (NOT `top_kernels`, which mis-scales):

| kernel | calls | ms | % |
|---|---|---|---|
| `gemv_oq_compact_grouped_v3_splitk` | 7040 | 237.5 | 22.8 |
| `gemv_oq_compact_moe_gate_up_k8_indexed_batched_spl` | 2600 | 154.9 | 14.9 |
| `gemv_oq_compact_grouped_v3` | 10305 | 125.7 | 12.1 |
| `gemv_oq_compact_moe_down_k8_indexed_batched_expand` | 2600 | 112.8 | 10.8 |
| **`sample_top_p`** | **65** | **78.5** | **7.5** |
| **`__amd_rocclr_copyBuffer`** | **27311** | **77.0** | **7.4** |
| `gated_delta_net_f16` | 1950 | 37.4 | 3.6 |

**The MoE experts are NOT the dominant term.** gate_up + down = 25.7%; the DENSE
compact GEMVs (v3_splitk + v3 + v2) are 36.2%. An item aimed at "MoE decode"
would be optimising the smaller half.

### The sampler: 7.5% of GPU time, 0% of wall clock — a dead end

`sample_top_p` is ONE 256-thread block (grid `[1,1,1]`) that insertion-sorts a
per-thread top-K over the whole vocabulary: each thread scans vocab/256 ~= 970
entries doing up to TOP_K compares each, on 1 of 40 CUs. **1.21 ms per token**
for a selection that reads ~1 MB (~4 us of bandwidth).

At temperature 0 none of it is needed, so I added a greedy fast path routing to
`argmax_f32` (guarded on the penalties being neutral; `blocked_tokens` needs no
guard since step 2 already writes `-INF` into the logits).

Result, same daemon command, warm, control re-run in the same session:

| | kernel cost/call | tok/s | output sha |
|---|---|---|---|
| `sample_top_p` | 1.21 ms | 52.0 | `a3f1b5c88b4da66b` |
| `argmax_f32` | **0.32 ms (3.8x less)** | **52.1** | `a3f1b5c88b4da66b` |

**Bit-identical output, 3.8x less kernel time, and ZERO wall-clock change.**

Both paths pay one 8-byte D2H sync per token, and that latency dominates the
kernel duration — so the kernel time was never on the critical path. **REVERTED**:
shipping it would add a branch and a per-call `hipMalloc` for no user-visible
gain.

**The lever for sampling is removing the per-token D2H sync, not making the
kernel faster.** `argmax_f32` is also grid `[1,1,1]` and mallocs per call, so
parallelising it would not have helped either. Anyone optimising sampling should
start there and can skip the experiment above.

### Still open from this profile

`__amd_rocclr_copyBuffer` is **7.4% of GPU time across 27311 calls** — ~427 per
token. Not investigated. That is a lot of small copies for a decode loop and is
the most suspicious remaining line.

---

## #6 oq8 GEMM ceiling · VERIFIED — number stale, and a real cliff found (but not where I guessed)

`examples/bench_oq8_gemm_small_n`, warm, gfx1151. GB/s counts WEIGHT bytes only.

| projection | weights | B=1 | B=16 | B=17 | B=32 |
|---|---|---|---|---|---|
| gate/up `[17408, 5120]` | 86.3 MiB | 150.7 | 152.1 | **78.7 (1.92x time)** | 66.9 |
| down `[5120, 17408]` | 86.3 MiB | 173.9 | 171.2 | 158.7 (1.10x) | 149.6 |
| o_proj `[5120, 5120]` | 25.4 MiB | 311.8 | 315.9 | 266.5 (1.17x) | 255.7 |

**The "54% of ceiling" figure is stale.** DRAM-bound shapes run 150–174 GB/s =
**60–70%** of the 248.5 GB/s ceiling. `o_proj` at 25.4 MiB runs 311 GB/s, ABOVE
the DRAM ceiling, because it fits the 32 MB MALL — a different regime, not a
better kernel.

### The real finding: a ~1.9x cliff at B=17, on gate/up only

`gate/up` is flat from B=1 to B=16 (~150 GB/s, 0.99–1.02x the B=1 call) and then
takes **1.92x the time at B=17** — nearly double for one extra row. `down` and
`o_proj` cross the same boundary gently (1.10x, 1.17x). So it is shape-specific,
not a general tiling limit, and it means **an oq8 batch of 17 costs about what 32
costs**.

### The connection I guessed, and it is WRONG

DDTree budget 16 linearizes to 1 seed + 16 = 17 verify rows, so I predicted the
cliff explained why budget 12 (32.89 tok/s) beats budget 16 (30.51). Tested
directly on the 27B:

| budget | verify rows | tok/s | tau |
|---|---|---|---|
| 13 | 14 | 31.72 | 5.84 |
| 14 | 15 | 29.93 | 5.60 |
| 15 | **16** | 30.48 | 5.95 |
| 16 | **17** | 30.50 | 5.95 |
| 17 | 18 | 30.42 | 5.95 |

**No cliff at the 16→17 boundary** — 15/16/17 are within noise of each other.
The hypothesis is refuted, and the reason is simple: the bench measures **oq8
W8A8**, while `Qwen3.6-27B--oq4.25++` dispatches the **compact** kernel family
(`gemv_oq_compact_*`). Different kernels; the cliff does not apply.

The plateau has a duller explanation visible in the same table: **tau saturates
at 5.9474 for budgets 15/16/17** — the tree stops growing, so extra budget buys
nothing and costs a little. That is the "tune on wall-clock, not tau" rule again,
not a hardware cliff.

**Actionable:** the B=17 cliff is real and worth fixing for oq8 models (it caps
useful batched-verify width at 16 on gate/up), but it is NOT what limits DDTree
on a compact target. Anyone chasing it should confirm which kernel family their
model actually dispatches first — that check is 30 seconds and would have saved
this detour.

---

## #8 W4A4 beyond prefill · VERIFIED — cannot help decode, by arithmetic

The item asks to extend 4-bit activations past prefill. It cannot pay at decode,
and the reason is not difficulty — it is that A4 buys COMPUTE throughput
(measured int4 = 2.0x int8 at the gfx1151 ISA) while decode is
weight-bandwidth-bound at B=1.

Qwen3.6-35B-A3B, per decode token:

| | bytes | share |
|---|---|---|
| weights (3B active x 4.25 bits) | 1.594 GB | 99.4% |
| activations (62 layers x ~8 hidden-sized f32 vectors) | 10.16 MB | **0.637%** |

Taking activations f32 -> int4 removes **0.558% of per-token traffic**:

    roofline, weights alone       155.9 tok/s
    roofline, weights + f32 acts  154.9 tok/s
    roofline, weights + int4 acts 155.8 tok/s

A **0.6% ceiling improvement** — against a measured decode of 62.8 tok/s, which
is 40% of that ceiling. The realisable gain is nil.

**W4 is already there at decode**: the profile in #7 shows every decode GEMV is
`gemv_oq_compact_*`, i.e. 4.25-bit weights. It is the A4 half that has nowhere to
go, because there are no activation bytes worth saving at batch 1.

So #8 is **prefill-only by nature**, and prefill already has the iu4 path with a
goal file of its own (`docs/plans/2026-08-23-iu4-gemm-close-the-half-goal.md`,
measuring ~50% of the 110.9 TOPS iu4 ceiling). Nothing to extend; the item was
asking for a batch-1 win from a batch-N mechanism.

---

## OPEN, well-characterised: `__amd_rocclr_copyBuffer` is 7.4% of decode GPU time

Not on the twenty; it fell out of #7's profile and is the most suspicious
remaining line.

`rocprofv3`, Qwen3.6-35B-A3B, greedy decode:

| shape | calls | ms |
|---|---|---|
| grid 512 / wg 512 | **25771** | **74.5** |
| everything else | 52 | 0.3 |

**~537 copies per token, ~2.9 us each, one workgroup — launch-latency-bound, not
bandwidth-bound.** 62 layers means roughly 8–9 per layer per token.

### Candidates eliminated

- **`lowered.rs:496-497`** (`dn_q_raw -> dn_q`, `dn_k_raw -> dn_k`): fires only
  when `linear_num_key_heads == linear_num_value_heads`. This model is 16 vs 32,
  so it takes the `repeat_interleave_qk_f32` branch instead. **Not the source.**
- **`decode_layers.rs`** (ten `memcpy_dtod_at_auto` sites): that hand path is not
  live — `HIPFIRE_FORWARD_LOWERED` defaults on. **Not the source.**

### Still in play

- `moe_decode.rs:401,403` — two `memcpy_htod` per MoE layer (top-k indices and
  weights). Every layer is MoE here, so ~124/token.
- `lowered.rs:423,439` — `pos_buf` htod, 4 bytes, per layer.

Those account for maybe a third. **The rest is unattributed** — there are copy
sites I have not found, likely in the KV or expert-dispatch paths.

### How to attribute it

The kernel trace gives dispatches, not call sites. Either instrument
`memcpy_dtod_auto` / `memcpy_htod_auto` with a counter keyed by
`std::panic::Location::caller()`, or run under `HIP_LAUNCH_BLOCKING=1` with a
sampling profiler to get host backtraces.

### Why it is worth doing

At 2.9 us and ~537 per token this is ~1.56 ms/token against a ~19 ms token —
the 7.4% the profile reports. Unlike the sampler (see #7), these are not hidden
behind a sync that would eat the saving: they are serialised kernel launches in
the decode path. Halving them is worth ~4%, which is more than anything left on
the twenty-item list except the large structural work.

### ⚠️ CORRECTION — the copyBuffer lead is model LOAD, not decode. It is dead.

Built the attribution rather than guessing further: `HIPFIRE_COPY_REPORT=1`
records `Location::caller()` at the HIP boundary (`hip_bridge::copy_census`),
with `#[track_caller]` on the `*_auto` wrappers so attribution passes through to
the real origin. One run answered it:

    === copy census (42716 copies) ===
      41662 calls   18385.36 MB   97.5%  dispatch/mod.rs:2284   <- upload_raw
        490 calls       1.29 MB    1.1%  dispatch/kv.rs:1800
        221 calls       4.42 MB    0.5%  dispatch/mod.rs:2175
         48 calls       0.00 MB    0.1%  qwen35/mod.rs:1077
         48 calls       0.00 MB    0.1%  runtime/sampler.rs:239

`mod.rs:2284` is `upload_raw`'s `memcpy_htod` — the **weight loader**. 18.4 GB
across 41662 calls is the 19.26 GB model going to the GPU. **That is load, not
decode.**

Decode-time copies are `kv.rs:1800` (490), `qwen35/mod.rs:1077` (48) and
`sampler.rs:239` (48) — about **12 per token totalling 1.3 MB**. Negligible.

**My "~537 copies per token" was wrong**: I divided a whole-run profile
(load + generate + unload) by decode tokens. The 7.4% `copyBuffer` share is
dominated by the one-time load, which at 6.7 s for 19.26 GB is ~2.9 GB/s and
unremarkable.

**Third time tonight the same error shape appeared** — attributing a whole-run
measurement to one phase. The first was the cold kernel cache, the second was
timing a stale binary. **Profile the phase you mean, or partition the numbers
before dividing.**

The instrumentation is kept (`HIPFIRE_COPY_REPORT=1`, off by default, one relaxed
atomic load per copy, verified 52.0 tok/s and byte-identical output with it
compiled in). It paid for itself immediately by killing a lead I had called "the
most promising unexamined thing I've seen tonight".

---

## #20 qwen4_exp batched forward · PAYOFF QUANTIFIED — it is a usability bug, not a spec-decode enabler

I had justified #20 as "gates everything speculative on the 180B". That
undersells it. Measured prefill scaling (`serve_real <model> <steps> <prompt_len>`,
8 GiB expert budget, warm):

| prompt tokens | prefill time | tok/s |
|---|---|---|
| 8 | 2.35 s | 3.4 |
| 16 | 4.64 s | 3.4 |
| 32 | 6.49 s | 4.9 |
| 64 | 10.95 s | **5.8** |

Linear, as a per-token forward must be. The slight rise with length is the expert
cache warming, not batching.

**Extrapolated: a 2048-token prompt takes ~6 minutes before the first output
token.** A 512-token prompt takes ~88 s.

For contrast, `Qwen3.6-35B-A3B--oq4.25++` — which HAS batched prefill — measures
**224.6 tok/s at pp32 and 360.1 at pp128** on the same box. That is a **40-60x
gap**, and it is entirely the batched/per-token distinction, not model size:
the 180B's decode is 0.18 s/tok (5.6 tok/s), so prefill and decode run at
comparable rates, which is the signature of a prefill that never batches.

**So #20 is not "unlock speculation" — it is "make the model usable on a real
prompt".** Speculation is a second-order benefit of the same work. That is a much
stronger case for the dedicated session, and it reprioritises #20 above the
remaining kernel items.

The blocker is unchanged and structural: `decode_step_into` advances exactly one
position, and both recurrent halves (Gated DeltaNet state, PLE conv ring) are
sequential by construction. A batched forward needs the chunked-scan treatment
that `gated_delta_net_f16` already has on the qwen35 side — that is the reference
implementation to copy, and it is the reason this is a session rather than a
patch.
