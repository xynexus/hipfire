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
