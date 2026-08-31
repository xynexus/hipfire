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

### W1 step 2 — the rotation basis · ROOT-CAUSED

The blocker is a **kernel** gap, not a dispatch one, and it explains a previously
unexplained result.

`dtype_rotation_plan` is correct — `Oq8G128` and `OqCompactG128` both map to
`RotationPlan::FwhtG128` (`hipfire-dispatch/src/types.rs:121-122`). The GEMV path
honours it via `ensure_mq_signs_128` (seeds 43/1043).

**But batched prefill rotates with the FUSED rmsnorm+rotate kernels, and all
three are FWHT-256 only:**

| kernel | evidence |
|---|---|
| `fused_rmsnorm_mq_rotate.hip` | "FWHT rotation per 256-element group" |
| `fused_rmsnorm_mq_rotate_awq.hip` | `const int groups_total = K / 256;` (line 92) |
| `fused_rmsnorm_mq_rotate_plain.hip` | "group-of-256 butterflies" |

So a G128 weight met FWHT-256 activations. The rotations do not cancel, which is
**exactly** the previously-recorded "qwen35 KLD 0.83 with Oq8G128" — a number
that was attributed to the format and is in fact this kernel mismatch. The emit
was then reverted and the infra kept, which is why `quantize_oq8g128` exists in
`codecs.rs:1016` with **no CLI format token to reach it** ("Both keep the
Oq8G256 W8A8 runtime format").

**The missing piece already exists.** `rotate_x_mq_128_batched`
(`dispatch/rope.rs:112`) is a batched FWHT-128 rotation, and its own doc says:
*"Kernel-side this needed nothing: `mq_rotate_x_128` already offsets by
`blockIdx.y * K` ... it was simply never launched with a grid.y > 1."*

So every component is present; nothing is fused for G128, that is all.

**W1 is therefore a 4-step chain, in this order:**

1. Wire an **unfused** G128 branch into batched prefill:
   rmsnorm → `rotate_x_mq_128_batched` → `gemm_oq8_grouped_act_batched_g(.., 128)`.
   Two kernels instead of one — slower than the fused G256 path, far faster than
   the per-token fallback it replaces.
2. Add the `is_batchable_la` arms in **both** `qwen35/mod.rs` and
   `runtime/dispatch.rs` (matching-pair comment: keep in sync).
3. Re-enable the `oq8g128` emit token in `hipfire-quantize` — currently there is
   no way to produce a test artifact at all.
4. Measure. **Falsifiable prediction: KLD 0.83 collapses to ~oq8 levels.** If it
   does not, this root cause is wrong and the format is genuinely at fault.

Step 3 depends on 1+2 landing, or the artifact is unservable again and the emit
gets reverted a second time.

**Bug filed by this analysis:** the deprecation note "Oq8G128 is bad" is wrong —
it was never a format problem. Anything asserting that should be corrected.
