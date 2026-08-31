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
