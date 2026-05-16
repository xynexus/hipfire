# F16 attention path — complete the F16 cascade

**Device:** R9700 #1 on hiptrx (`ROCR_VISIBLE_DEVICES=1`)
**Branch:** `feat/f16-attention` (off `perf/dflash-phase1-target-hidden-collapse`)
**Target saves:** **0.33 GB at ctx=64K, 0.66 GB at ctx=128K**
**Quality risk:** LOW-MEDIUM (F16 attention with F32 accumulate is standard; verify with coherence gate)

## What's currently F32 in the attention path

After B2 (F16 k/v_ctx_cached), the F32 holdouts in `draft_forward`'s
attention block are:

- `k_cat`, `v_cat`: `[L+B, kv_dim]` F32 — populated each cycle by
  widening the F16 cache (B2 added a convert_f16_to_f32 step at concat).
- `k_noise`, `v_noise`: `[B, kv_dim]` F32 — wk/wv noise projection
  for the current block.
- `rope_batched_f32` runs on F32 k_cat.
- `rmsnorm_batched` on K-noise slot of k_cat runs F32.
- `attention_dflash_f32` reads F32 k_cat / v_cat / q.

Each is small (k_cat at ctx=64K is `(L+B) * kvd * 4` = 320 MB) but
collectively they're 320 + 320 + 5 (noise) = ~645 MB at ctx=64K
that's still F32. F16 halves this to ~322 MB net — 0.33 GB savings.

More importantly: this lever removes the convert_f16_to_f32 step at
concat (B2 added it because k_cat is F32 but cache is F16). If
everything in the attention path is F16, the concat becomes a plain
memcpy_dtod and the convert kernel is unused per-cycle.

## Plan

1. **k_cat / v_cat F16 alloc**: change `DflashScratch.k_cat /
   v_cat` from F32 to F16. Storage halves; downstream consumers
   must be F16-aware.
2. **k_noise / v_noise F16**: same change; wk/wv noise output writes
   F16 directly (route through B2's `kv_f32_chunk` + narrow, or
   add an F16-output GEMM variant).
3. **Concat path** (`dflash.rs:1020+`): replace
   `convert_f16_to_f32_to(cache → k_cat)` with a plain
   `memcpy_dtod_at(cache → k_cat)` since both are F16 now.
   Same for V.
4. **F16 RoPE kernel**: write a sibling of `rope_batched_f32` that
   takes F16 input/output (or F16 in, F16 out, but accumulate in
   F32 internally — the math is just a 2-element rotation per pair).
   File: `kernels/src/rope_batched_f16.hip` (new).
5. **F16 rmsnorm on K-noise**: `rmsnorm_batched_f16` sibling.
   K-noise slot is now F16; per-head normalize must support F16
   input/output (with F32 accumulation internally for stability).
   File: `kernels/src/rmsnorm_f16.hip` (new) OR an additional symbol
   in `kernels/src/rmsnorm.hip` if that file's structure permits.
6. **F16 attention kernel**: write `attention_dflash_f16` —
   sibling of `attention_dflash_f32`. Same math (softmax with FP32
   accumulator, F16 K/V/Q inputs, F32 score buffer internal, F16
   final output). File: `kernels/src/attention_dflash_f16.hip` (new).
   This is the biggest kernel addition in the lever.

## Implementation order (recommended)

Step 1 (`f16-cat-storage-prep`): change k_cat/v_cat alloc to F16,
update concat to memcpy, keep RoPE/rmsnorm/attention as F32 BUT
widen the F16 k_cat/v_cat into temporary F32 scratches before they
run. Use `convert_f16_to_f32_to` for these. Cost: same as current
B2 concat widen. Sanity-check: bench output byte-exact.

Step 2 (`f16-rope`): port RoPE to F16. Now k_cat stays F16 across
the rotation — no widen-back.

Step 3 (`f16-rmsnorm-knoise`): port K-noise rmsnorm to F16. K-noise
slot is now F16 throughout.

Step 4 (`f16-attention`): port the attention kernel to F16. The
ATTENTION is the big one — needs careful softmax handling. Read
the existing `attention_dflash_f32.hip` first; mirror it with F16
loads / stores but F32 internal score accumulation.

Each step COMMITS independently and must pass the canonical bench
+ coherence gate before moving to the next.

## Watch out for

- **Softmax precision**: F16 scores can underflow / overflow in
  long-context attention. F32 accumulator + F32 max-subtract is
  standard practice. Don't use F16 throughout the softmax — that
  WILL break long-context.
- **RoPE on F16**: the rotation matrix entries are computed from
  positions and theta; these are F32 in the existing kernel. Keep
  the trig in F32, narrow on store.
- **Attention output dtype**: existing attention kernel writes F32
  `attn_out` (consumed by wo GEMM). Decide: keep `attn_out` F32 and
  narrow inside the attention kernel before storing, OR change
  `attn_out` to F16. If F16, wo GEMM input chain needs updating
  (same pattern as B1's wk/wv input widening).
- **gemv vs attention kernel**: the existing path uses
  `attention_dflash_f32`. Don't accidentally route through a
  different attention kernel (gqa_softmax / flash_attn paths from
  the target's Qwen35 code) — DFlash has its own bidirectional
  attention with the K = concat(ctx, noise) layout.

## Validation matrix

| Test | Command | Pass condition |
|---|---|---|
| Build | `cargo build --release --features deltanet --example dflash_spec_demo` | clean |
| Canonical bench (each step) | `dflash_spec_demo ... --max 256 --kv-mode asym3 --no-chatml` | τ=13.2727 ±0.5 % each step |
| Coherence gate asym3 | `HIPFIRE_FORCE_SPEC_GATE=1 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Coherence gate q8 | `HIPFIRE_FORCE_SPEC_GATE=1 HIPFIRE_GATE_KV_MODE=q8 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Long-context (16K) | run with `--ctx 16384 --max 100` | tokens coherent, no τ collapse |
| Ctx bisect | as in coordination doc | ceiling lifts ≥3K |

## Done criteria

- [ ] All 4 sub-steps land as separate commits, each gate-clean
- [ ] τ on canonical drops by < 2 % cumulative
- [ ] Long-context bench (16K) shows no degradation vs F32 baseline
- [ ] Net VRAM saved at ctx=64K on hiptrx ≥ 0.30 GB measured
- [ ] Commit messages document per-step gate output + VRAM delta

## Handoff prompt

```
Implement the F16 attention path — port k_cat/v_cat, k_noise/v_noise,
RoPE, K-noise rmsnorm, and attention_dflash to F16 storage. Detailed
plan at `docs/plans/c-track/f16-attention.md`.

Your branch: `feat/f16-attention` (off `perf/dflash-phase1-target-hidden-collapse`
HEAD 2533a1b4). Your worktree on hiptrx:
`~/hipfire/.worktrees/c-track-f16-attn/`. GPU: device 1
(`export ROCR_VISIBLE_DEVICES=1`).

This is a 4-substep lever — each substep is its own commit with its
own gate validation:
  Step 1 (storage prep): k_cat/v_cat F16 storage, concat → memcpy.
    Widen to F32 just before RoPE/rmsnorm/attention. Bench byte-exact.
  Step 2 (F16 RoPE): port rope_batched_f32 to F16. Now k_cat stays F16
    across rotation.
  Step 3 (F16 K-noise rmsnorm): port rmsnorm_batched to F16 for the
    K-noise slot.
  Step 4 (F16 attention): port attention_dflash_f32 to F16. Softmax
    accumulator stays F32; only loads/stores narrow.

Each kernel addition is a new .hip file in `kernels/src/`. Add the
src constant in `crates/rdna-compute/src/kernels.rs` and the dispatch
wrapper in `crates/rdna-compute/src/dispatch.rs` next to the F32
sibling.

Hard rules:
  - Softmax accumulator MUST stay F32 (long-context underflow risk).
  - RoPE trig MUST stay F32 (precision in long ctx).
  - Each substep gate-clean before next. τ drift < 0.5 % per step,
    cumulative < 2 %.

Validation: canonical merge_sort bench + coherence-gate-dflash.sh on
asym3 AND q8 AT EACH STEP. Plus a long-context bench at ctx=16384,
max=100 to catch attention precision drift the canonical bench would
miss.

Don't push to origin without coordinator approval.

Begin by reading `docs/plans/c-track/f16-attention.md` and
`docs/plans/c-track-parallel-coordination.md`.
```
