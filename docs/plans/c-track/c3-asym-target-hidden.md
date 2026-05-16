# C3 — Asym3 quantize target_hidden

**Device:** R9700 #3 on hiptrx (`ROCR_VISIBLE_DEVICES=3`)
**Branch:** `feat/c3-asym-target-hidden` (off `perf/dflash-phase1-target-hidden-collapse`)
**Target saves:** **~1.6-2.0 GB at ctx=64K, ~3.3-4.0 GB at ctx=128K**
**Quality risk:** HIGH — drafter cross-attention reads target_hidden directly

## Why this is the biggest single remaining lever

After B3, `target_hidden` is F16 at `[L, ne, hidden]` (3.28 GB at
64K). It's the single largest per-ctx buffer in DflashScratch.
Going from F16 to asym3 (3-bit + scale/zero):

- F16: `L × ne × hidden × 2` bytes = 3.28 GB at 64K
- asym3 (3-bit + 4 bytes overhead per group of 256, K-style):
  approximately `L × ne × hidden × 0.5` bytes (4 bits effective due
  to scale overhead at hidden=5120 → 20 groups/row × 4 bytes scale =
  80 bytes overhead per (pos, extract) on top of 3-bit data) ≈
  0.82 GB at 64K. **Saves 2.46 GB at 64K.**

This is the lever that, combined with the others, could plausibly
crack 128K on 24 GB.

## Why it's HIGH risk

`target_hidden` is the input to the FC GEMM in the drafter, which
feeds wk/wv for every layer's K/V context. Quantizing it to 3 bits
means the drafter's cross-attention reads from a heavily-lossy
representation of the target's hidden states.

This is analogous to (and possibly correlates with) the asym3 KV
cache regression on the target side that broke DFlash drafter prose
τ. The mechanism: drafter is trained on F32/F16 hidden states; 3-bit
asym hidden states are OOD for it.

**Mitigation:** validate with coherence-gate-dflash + a multi-prompt
prose battery + 16K-context bench. Be prepared to drop back to F16
if prose τ collapses.

## Plan

1. **Storage**: change `DflashScratch.target_hidden` from F16
   `[tot * ne * h]` to asym3-quantized. Quantization layout decision:
   match the K-asym3 layout from `llama.rs:new_gpu_asym3_*` (3-bit
   payload + per-group scale/zero), where each "group" is a row of
   `hidden=5120` elements partitioned into groups of 256 (so
   `5120 / 256 = 20 groups/row`).
2. **Per-row layout**: each `(pos, extract)` becomes one logical
   "row" with the K-asym3 byte layout. Per-pos-per-extract:
   `4 + hidden/2 = 4 + 2560 = 2564` bytes (vs 5120 × 2 = 10240 F16).
   Saves ~75 % per row.
3. **Verify writes (commit_staging_to_ring)**: existing path narrows
   F32 staging → F16 then memcpys F16 row to target_hidden. Replace
   with F32 staging → asym3-quantize-on-write directly into
   target_hidden at the right offset.
4. **FC GEMM read (`draft_forward` chunk loop)**: existing path
   widens F16 target_hidden chunk → F32 then runs FC GEMM. Replace
   F16 → F32 with an asym3 → F32 dequant kernel. Use existing
   asym3 K dequant helpers if shape-compatible; otherwise write a
   sibling.
5. **AR `write_at_head` / `write_rows_at_head`**: same F32 → asym3
   quantize-on-write pattern as commit_staging_to_ring.
6. **download_hidden_block**: existing path widens F16 → F32 in
   download_scratch_f32 then D2H. Replace F16 → F32 with asym3 →
   F32 dequant in the same scratch.
7. **scatter_hidden_block_to_interleaved fallback path**: dtod copy
   of asym3 slot bytes (smaller than F16 — but same shape, just
   different bytes/slot). Update slot_bytes computation.

## Per-buffer alloc accounting (post-C3 at 64K)

| Buffer | Pre-C3 (F16) | Post-C3 (asym3) | Save |
|---|---:|---:|---:|
| target_hidden | 3.28 GB | 0.82 GB | **2.46 GB** |
| target_hidden_delta_f32 (scratch) | 100 MB | 100 MB | 0 |
| target_hidden_proj | 0.66 GB (F16) | 0.66 GB | 0 (separate buffer) |

## Watch out for

- **drafter retrain MAY be needed**: if 3-bit hidden state is OOD
  for the drafter cross-attention, τ tanks no matter how careful
  the dequant is. Plan B: 4-bit asym (asym4) saves less (1.64 GB)
  but is closer to F16 precision. Consider falling back to asym4
  if asym3 fails coherence.
- **FC GEMM input precision**: F16 widened to F32 had 11 mantissa
  bits. asym3 widened to F32 has effectively 3-4 mantissa bits.
  The FC GEMM accumulates in F32, so within-GEMM precision is OK
  but the INPUT itself is heavily lossy.
- **Long-context degradation**: precision loss compounds over many
  cycles. Test at ctx=16K, 32K, 64K with the same prompt; τ should
  stay stable.
- **target_hidden_abs_positions** + eviction compaction: the
  per-row asym3 metadata (scale, zero) gets re-shuffled during
  eviction. Update `apply_eviction_retain_to_draft` to copy
  full asym3 row bytes (not F16 bytes).

## Implementation order

1. (`c3-asym3-storage-only`): change alloc to asym3-shape, but
   keep the writers/readers F16-compatible by routing through a
   F16 intermediate scratch. Validates the storage shape without
   changing math. Bench byte-exact.
2. (`c3-asym3-write-side`): replace commit_staging_to_ring's narrow
   to F16 with quantize-on-write to asym3 directly. F32 staging →
   asym3 target_hidden. Coherence gate.
3. (`c3-asym3-read-side`): replace FC GEMM's F16→F32 widen with
   asym3→F32 dequant. Coherence gate + long-prose battery.
4. (`c3-eviction-mirror`): update apply_eviction_retain_to_draft.
   Test with eviction-triggering long-context bench.
5. **If coherence fails on prose**: fall back to asym4 (still gives
   ~1.64 GB savings).

## Validation matrix

| Test | Command | Pass condition |
|---|---|---|
| Build | `cargo build --release --features deltanet --example dflash_spec_demo` | clean |
| Canonical bench | `dflash_spec_demo ... --max 256 --kv-mode asym3 --no-chatml` | τ=13.2727 ±10 % (asym3 introduces measurable drift) |
| Coherence gate asym3 (canonical) | `HIPFIRE_FORCE_SPEC_GATE=1 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Coherence gate q8 | `HIPFIRE_FORCE_SPEC_GATE=1 HIPFIRE_GATE_KV_MODE=q8 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Long-prose 1 (Roman empire, 800 tok) | asym3 | τ > 0.5 |
| Long-prose 2 (Roman empire, 800 tok) | q8 | τ > 4.0 |
| Long-prose 3 (different prose prompt) | q8 | τ > 4.0 |
| Long-context (32K prefix) | --ctx 32768 --max 100 asym3 | τ > 5, no attractor |
| Ctx bisect | as in coordination doc | ceiling lifts ≥18K on 24 GB |

## Done criteria

- [ ] Each substep gate-clean
- [ ] Coherence gate "no hard errors" on asym3 + q8 + 3 long-prose battery
- [ ] τ on canonical drops by < 10 % cumulative
- [ ] Prose τ on q8 stays > 4.0 (pre-C3 baseline ~4.71)
- [ ] Net VRAM saved at ctx=64K ≥ 2.0 GB measured
- [ ] If asym3 fails coherence: ship asym4 fallback with ~1.5 GB saved

## Handoff prompt

```
Implement C3 — quantize target_hidden from F16 to asym3 (3-bit) in
the DFlash draft scratch. Detailed plan at
`docs/plans/c-track/c3-asym-target-hidden.md`.

Your branch: `feat/c3-asym-target-hidden` (off
`perf/dflash-phase1-target-hidden-collapse` HEAD 2533a1b4). Your
worktree on hiptrx: `~/hipfire/.worktrees/c-track-c3-asym-th/`.
GPU: device 3 (`export ROCR_VISIBLE_DEVICES=3`).

This is the BIGGEST remaining VRAM lever (~2.46 GB at ctx=64K) AND
the HIGHEST quality risk. target_hidden feeds the drafter's
cross-attention FC GEMM — quantizing it to 3 bits is analogous to
the target-side asym3 KV regression that broke DFlash prose τ. You
MUST validate with the long-prose battery (3 prompts, both asym3
and q8 KV-mode) before claiming green.

Substep order:
  1. (`c3-asym3-storage-only`): change alloc to asym3-shape, route
     writers/readers through F16 intermediate to validate storage.
     Bench byte-exact.
  2. (`c3-asym3-write-side`): quantize-on-write in
     commit_staging_to_ring + write_at_head + write_rows_at_head.
  3. (`c3-asym3-read-side`): dequant-on-read in the FC GEMM chunk
     loop. Replace convert_f16_to_f32 widening with asym3 → F32
     dequant.
  4. (`c3-eviction-mirror`): update apply_eviction_retain_to_draft.

Reuse asym3 K helpers from llama.rs:new_gpu_asym3_* where possible.
Per-row asym3 layout matches K-asym3: `4 bytes scale/zero per group
of 256 + 3-bit packed payload`.

If coherence gate / long-prose τ collapses on asym3: fall back to
asym4 (4-bit, 1.64 GB saved instead of 2.46 GB). Commit BOTH the
attempted asym3 and the asym4 fallback as separate branches if you
have to pivot.

Validation: canonical + coherence-gate-dflash on asym3 + q8 + 3
long-prose battery (Roman empire + 2 other prompts, 800 tok each,
max=300) on q8. Plus a 32K-context bench. If prose τ on q8 drops
below 4.0 — STOP and report.

Don't push to origin without coordinator approval — fall-back
branches stay local until reviewed.

Begin by reading `docs/plans/c-track/c3-asym-target-hidden.md` and
`docs/plans/c-track-parallel-coordination.md`.
```
