# C1 — Eliminate target_hidden_proj as a full-L buffer

**Device:** R9700 #0 on hiptrx (`ROCR_VISIBLE_DEVICES=0`)
**Branch:** `feat/c1-fused-fc-wk-wv` (off `perf/dflash-phase1-target-hidden-collapse`)
**Target saves:** **0.66 GB at ctx=64K, 1.31 GB at ctx=128K**
**Quality risk:** LOW (no math change; same projections, just no L-sized persistence)

## What's currently bloating

After B1, `target_hidden_proj` lives as `[L, hidden]` F16 (1.31 GB
F32 → 0.66 GB F16 at 64K). But the L-sized buffer is **only ever
read at the chunk just written by the current cycle's FC** — the
historical `[0..cached_rows)` slice is dead data, because per-layer
wk/wv already consumed it and wrote into `k_ctx_cached / v_ctx_cached`
during prior cycles.

## Plan

1. Resize `DflashScratch.target_hidden_proj` from `[L, hidden]` to
   `[MQ_X_ROT_CHUNK_ROWS, hidden]` F16 (~20 MB at hidden=5120,
   versus 0.66 GB at ctx=64K).
2. Inside `draft_forward`'s FC chunk-loop (currently at
   `dflash.rs:828-882` on the perf branch — verify line numbers
   on your branch), reuse the same chunk slot every iteration; the
   downstream per-layer wk/wv block reads from `target_hidden_proj[0..n*h]`
   for each chunk's delta rows. No need to map chunk offsets to
   absolute `(cached_rows + row)` positions because the buffer
   is now exactly the chunk being processed.
3. Make sure the per-layer wk/wv path reads `target_hidden_proj` at
   offset 0 (not `(cached_rows + row) * h`), and that this matches
   the chunk that the FC just wrote.
4. Update `target_hidden_proj_f32_chunk` retention as-is (unchanged).
   It's already chunk-sized.
5. Remove the `cached_rows` bookkeeping for `target_hidden_proj`
   specifically — but **keep `draft_ctx_cached_rows` for k_ctx_cached
   / v_ctx_cached invalidation semantics**. Those are still full-L
   caches.

## Watch out for

- The chunk loop currently iterates `delta / chunk_rows` chunks per
  call. After C1, each iteration must complete BEFORE the next
  overwrites `target_hidden_proj[0..]`. Inline the per-layer
  wk/wv calls inside the chunk loop body, OR run all FC chunks first
  followed by all wk/wv chunks (but then the wk/wv block needs the
  L-sized proj — defeats the win). **Inline-per-chunk is required.**
- The 5 draft layers' wk/wv reads all share the same chunk content.
  Sequence per chunk:
  ```
  for chunk in delta:
      widen target_hidden[chunk] F16 → F32 (target_hidden_delta_f32)
      fc → target_hidden_proj_f32_chunk (F32 chunk)
      rmsnorm → in-place
      narrow → target_hidden_proj (F16 chunk, no offset)
      for li in 0..n_layers:
          widen target_hidden_proj F16 → F32 (reuse target_hidden_delta_f32 sub-region)
          wk → kv_f32_chunk
          k_norm → in-place
          narrow → k_ctx_cached[li][cached_rows+chunk_row..]
          wv → kv_f32_chunk
          narrow → v_ctx_cached[li][cached_rows+chunk_row..]
  ```
- This **changes the structure of `draft_forward`** — the per-layer
  loop currently runs OUTSIDE the FC chunk loop. You're inverting
  the loop order: chunk-outer, layer-inner.
- Existing per-layer attention (Q/K/V noise, K/V concat with
  prefix, RoPE, attention kernel, output projection, FFN) all
  still run in the EXISTING outer per-layer pass. Don't fuse those
  into the chunk loop — they need the FULL k_ctx_cached/v_ctx_cached
  prefix, not just the delta.

## Validation matrix

After build clean, run on R9700 #0:

| Test | Command (prepend `ROCR_VISIBLE_DEVICES=0`) | Pass condition |
|---|---|---|
| Build | `cargo build --release --features deltanet --example dflash_spec_demo` | clean, no new warnings |
| Canonical bench | `dflash_spec_demo ... --max 256 --kv-mode asym3 --no-chatml` | τ=13.2727 exact, tokens byte-exact baseline |
| Coherence gate (asym3) | `HIPFIRE_FORCE_SPEC_GATE=1 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Coherence gate (q8) | `HIPFIRE_FORCE_SPEC_GATE=1 HIPFIRE_GATE_KV_MODE=q8 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Ctx bisect | `for ctx in 65536 98304 131072; do ...` | ceiling rises by ~6K (0.66 GB) |

## Sub-tasks (1-day chunks)

1. Resize `target_hidden_proj` alloc, update `free_gpu`, update all
   accessors that use `target_hidden_proj.sub_offset(cached_rows * h, ...)`
   to use offset 0 instead. Build clean.
2. Invert the loop in `draft_forward`: chunk-outer, layer-inner.
   Move per-layer wk/wv into the chunk-loop body.
3. Smoke test: τ=13.2727 byte-exact on merge_sort. If not, the
   semantics drifted somewhere.
4. Coherence gate (both asym3 + q8). Ctx-bisect. Commit.

## Done criteria

- [ ] τ=13.2727 byte-exact tokens on canonical merge_sort bench
- [ ] coherence gate "no hard errors" on asym3 AND q8
- [ ] ctx ceiling lifts ≥4K on hiptrx (R9700, 32 GB) — proves the L-sized buffer is gone
- [ ] commit message reports: peak VRAM at ctx=128K hiptrx, peak VRAM at ctx=65K (matches reference k9lin), gate output paths

## Handoff prompt (copy into the agent's session)

```
Implement C1 — eliminate target_hidden_proj as a full-L buffer in the
DFlash draft scratch. Detailed plan at `docs/plans/c-track/c1-fused-fc-wk-wv.md`
on the `perf/dflash-phase1-target-hidden-collapse` branch — read that
file end-to-end before touching code.

Your branch: `feat/c1-fused-fc-wk-wv` (already created off the perf
branch's HEAD `2533a1b4`). Your worktree on hiptrx:
`~/hipfire/.worktrees/c-track-c1-fused/`. Your assigned GPU: device 0.
Pin `export ROCR_VISIBLE_DEVICES=0` in your shell before any cargo or
benchmark invocation.

The lever: target_hidden_proj is currently F16 storage but sized to
the full L max-ctx (`L × hidden × 2` bytes, 0.66 GB at ctx=64K). The
historical [0..cached_rows] portion is dead data — only the current
cycle's delta chunk is ever read by the per-layer wk/wv. Shrink
target_hidden_proj to a chunk-sized buffer
(`MQ_X_ROT_CHUNK_ROWS × hidden`, ~20 MB) and invert the
draft_forward loop structure (chunk-outer, layer-inner) so per-layer
wk/wv reads from offset 0 within the freshly-written chunk.

Don't touch the per-layer attention block (Q/K/V noise, concat,
RoPE, attention kernel, FFN) — those still need the full
k_ctx_cached / v_ctx_cached prefix.

Validation: τ=13.2727 byte-exact tokens on canonical merge_sort
bench AND coherence-gate-dflash.sh "no hard errors" on both asym3
and q8 KV modes (q8 via `HIPFIRE_GATE_KV_MODE=q8`). Then ctx-bisect
to confirm ceiling lifts.

Commit message format per the coordination doc:
`docs/plans/c-track-parallel-coordination.md` § "Commit messages".

Don't push to origin without coordinator approval — `git push hiptrx
<branch>` for local backup only.

Begin by reading `docs/plans/c-track/c1-fused-fc-wk-wv.md` and
`docs/plans/c-track-parallel-coordination.md` end-to-end.
```
