# prefill_lowered's dtype chain should be table-driven

**Status:** proposed, not started. Raised while wiring Qtip3G256 into batched prefill.

`qwen35/prefill_lowered.rs` selects a kernel for the QKV projections with a 21-branch
`if / else if` chain over weight dtype, inside a 4386-line function. The line count is
not the problem. The problem is that the chain **re-implements selection logic the
dispatch registry already models**, and that a dtype missing from it fails *silently*.

## HARD REQUIREMENT: the selector must REJECT an unknown dtype, never fall through

This is the point of the whole change. Whatever replaces the chain — registry lookup,
match, table — **must fail loudly on a dtype it does not know**. Not fall through to a
default key. Not pick "closest". Not silently skip the rotation. Fail.

Concretely:

- The dtype -> kernel lookup returns `Option`/`Result`, and a miss is an **error**, not a
  fallback. `run_plain_gemm_key`'s existing `OqCompactG256` refusal is the right
  behaviour; it is just hand-written per dtype instead of structural.
- Prefer an **exhaustive `match` on `DType` with no `_` wildcard**, so a newly added
  format breaks the build at every site that must handle it. A wildcard arm reintroduces
  exactly the silent fallthrough this is meant to delete.
- The rotation flag must come from the **same row** as the kernel key, so "wired for
  compute but not admitted for rotation" is unrepresentable.
- A startup/registry self-check should assert that every `DType` reachable by the loader
  has a row, so the failure surfaces at load rather than mid-forward.

### Why this is not hypothetical

Three instances, all in one session:

1. `run_plain_gemm_key` already has to refuse `OqCompactG256` by hand — a missing arm
   would decode the weight as another format on an unrotated activation.
2. `Qtip3G256` had the identical latent exposure and no guard (fixed `577403444`).
3. Wiring `Qtip3G256` into the `prefill_lowered` chain **correctly** (rotation admission
   + arm together, KLD unchanged at 0.189103) still did not make it run: prefill stayed
   at 234 tok/s, because a SEPARATE upstream gate routes the dtype away from the lowered
   path entirely. A correct arm in one list, dead because of another list.

(3) is the disease in one sentence: **the admission decision is spread across multiple
independent lists, and being absent from any of them is silent.** A dtype that is wired,
correct, and never called looks exactly like a dtype that is fast.

## The branches are three shapes, not 21 cases

| shape | branches | body |
|---|---|---|
| fused-QKV key | ~5 (mq, mq3, mq3_lloyd, fp4, q8+wmma) | `run_fused_qkv_key(KEY, …)` |
| plain-GEMM key | ~2 (q8, qtip3) | `run_plain_gemm_key(KEY, …)` per projection |
| prequant-then-loop | 2 (oq8, oqCompact) | `quantize_act_oq8_batched_interleaved` once, then a `*_prequant` GEMM per projection |

Plus **oq4**, which has a genuine 3-way choice (`force_a4` / `n >= 64` / f16-WMMA) and
writes the q/k/v triple out inline three times.

So ~9 branches are pure `dtype -> KernelKey` rows. And the conditions they hand-code
already exist as registry data:

- `q8_wmma_arch` / `has_wmma()`  ->  `ArchPredicate`
- oq4's `n >= 64`                ->  `ShapePredicate::BatchGe(64)`

## Why it is worth doing: it removes a silent-corruption class

Two instances found in one session:

- `run_plain_gemm_key` must explicitly refuse `OqCompactG256`, because a dtype with no
  arm gets decoded by the fallthrough key as a different format AND misses that call
  site's rotation-admission list — "silent corruption on both counts".
- `Qtip3G256` had the identical latent exposure and no guard (fixed 577403444).

Both exist because "dtype absent from an if/else chain" is invisible. As table data it
becomes an exhaustive match or a startup assertion.

The sharper version is the **rotation-admission list** (`qkv_is_mq`): it is a SECOND
list that must stay in sync with the branch chain, with nothing enforcing it. Adding a
rotated dtype to one and not the other is exactly the failure the guard describes. A
table should carry `needs_fwht_rotation` as a column so the two cannot diverge.

## Suggested scope

1. Table-drive the ~9 mapping branches (`dtype -> KernelKey`, `needs_rotation`).
2. Unify the two prequant branches into one shape with a kernel selector.
3. Leave oq4's genuine choice as a branch initially; move `n >= 64` to `shape_gate` after.
4. Derive the rotation-admission list from the same table.

## Do NOT do it casually

Hot path across 8+ formats. Behaviour preservation needs a KLD gate PER FORMAT, not a
spot check — the failure mode is a plausible-looking model that scores wrong, and the
oq8 comment in that file records a real instance (unrotated activation -> PPL 3.5e6).
