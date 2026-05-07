# PR-A progress: hetero PP+DFlash on hipx (gfx1151 + gfx1010)

**Date:** 2026-05-07
**Branch:** `feat/hetero-pp-dflash` (pushed to origin)
**PRD:** `docs/plans/hetero-pflash-dflash.prd` (v1.2)

## What's shipped

### Commit 1 — `feat(hetero-dflash): drafter device pinning + cross-card helpers (PR-A foundation)` (`f47cdfd`)

Load-side foundation. Default codepath byte-identical to master.

- `HIPFIRE_DFLASH_DRAFTER_DEVICE=N` env opens a dedicated `Gpu` for the drafter.
- `LoadedModel.dflash_drafter_gpu: Option<Gpu>` carries it through load/generate/unload.
- `load_dflash_state(target_gpu, drafter_gpu: Option<&mut Gpu>)` allocates `draft_weights` + `draft_scratch` on the drafter handle; `hidden_rb` / `target_snap` / `gdn_tape` / `verify_scratch` / `ddtree` on the target gpu.
- `unload_model` frees draft buffers against the drafter handle and drops the dedicated Gpu cleanly.
- `multi_gpu::cross_card_copy` + `cross_card_wait`: free-function form for two free-standing `Gpu` refs (no shared `Gpus`).
- `generate_dflash` refuses with a clear error when `dflash_drafter_gpu.is_some()` (spec-step refactor not shipped).

### Commit 2 — `feat(hetero-dflash): offset-aware peer copy primitives + cross_card_copy_at`

- `hip-bridge::ffi::HipRuntime::memcpy_peer_at` + `memcpy_peer_at_async`: byte-offset-aware peer copies for sub-slicing without intermediate `DeviceBuffer`s.
- `multi_gpu::cross_card_copy_at`: free-function form of offset-aware peer copy with the same async-stream-vs-sync-host-staged decision tree as `cross_card_copy`.

These are the primitives the spec-step body refactor (next commit) will use to ship sub-slices of `draft_scratch.x`, the embedding lookup output, and `hidden_rb` slots between target and drafter cards.

## What remains (Step 2: spec_step_dflash body refactor)

The substantive engineering. ~150-200 LOC across `crates/hipfire-arch-qwen35/src/speculative.rs` + `crates/hipfire-runtime/examples/daemon.rs` + helpers.

### Surface

1. **`DflashState` (or `DflashScratch`) extension**: lazy-allocate two staging buffers on target_gpu when `drafter_gpu.is_some()`:
   - `embd_staging_target: GpuTensor` — `[B × hidden]` floats. Phase 2 writes embeddings here, then ships cross-card to `draft_scratch.x` on drafter_gpu.
   - `draft_hidden_staging_target: GpuTensor` — `[(B-1) × hidden]` floats. Phase 5 cross-card-ships drafter hidden output here, then runs target's lm_head GEMM against it.

2. **`spec_step_dflash` signature** (`speculative.rs:2384`):
   ```rust
   pub fn spec_step_dflash(
       target_gpu: &mut Gpu,
       drafter_gpu: &mut Gpu,        // NEW — same as target_gpu when same-device-id
       target: &mut ModelSlot,
       draft_weights: &DflashWeights,
       // ... same as before
   ) -> HipResult<SpecStepResult>
   ```
   Internal: `let same_device = target_gpu.device_id == drafter_gpu.device_id;` drives the fast-path-vs-staging-path bifurcation per phase.

3. **Phase 2 (embedding lookup, lines ~2505-2526)**: when hetero, write embeddings to `embd_staging_target` on target_gpu, then `cross_card_copy_at` to `draft_scratch.x` on drafter_gpu.

4. **Phase 4 (`dflash::draft_forward`, line 2592)**: pass `drafter_gpu` instead of `gpu`. The function body is purely drafter-side; just rename the parameter.

5. **Phase 5 (draft lm_head, lines ~2638-2742)**: when hetero, `cross_card_copy_at` `draft_scratch.x[h..]` from drafter_gpu → `draft_hidden_staging_target` on target_gpu. Run lm_head GEMM on target_gpu against the staging buffer.

6. **Phase 6 (`verify_dflash_block`, line ~2823)**: target-side helper, just rename `gpu` → `target_gpu` parameter inside `verify_dflash_block_inner` (lines 1854-2193).

7. **Phase 9 (`scatter_hidden_block_to_interleaved`, line 3060)**: when hetero, the existing function (which does N×ne D2D copies on a single Gpu) needs a cross-card variant that does the same copies via `cross_card_copy_at` from `hidden_rb.layer_bufs[ext]` (target_gpu) → `draft_scratch.target_hidden` (drafter_gpu).

8. **Phase 10 (target_snap restore + tape replay)**: target-side, rename only.

9. **`generate_dflash` (`daemon.rs:1998`)**: drop the refusal added in commit 1; pass `m.dflash_drafter_gpu.as_mut().unwrap_or(/* reborrow gpu */)` as the `drafter_gpu` argument.

### Cross-card op count per cycle

| Phase | Direction | Size | Frequency |
|---|---|---|---|
| 2 embedding | target→drafter | `B × hidden × 4` (~327 KB at B=16, h=5120) | once/cycle |
| 5 draft lm_head | drafter→target | `(B-1) × hidden × 4` (~307 KB) | once/cycle |
| 9 hidden scatter | target→drafter | `(τ+1) × ne × hidden × 4` (~820 KB at τ=8, ne=5, h=5120) | once/cycle |

Total ~1.5 MB cross-card per cycle. USB4 v2 sustained ~10 GB/s effective → ~0.15 ms transfer overhead per cycle. At 30 ms/cycle target, <1% overhead. Easy to clear the ≥1.25× anchor (≥33.75 tok/s vs 27.0 solo).

### Test path (after step 2 lands)

```bash
ssh hipx
cd ClaudeCode/autorocm/hipfire
git fetch origin && git checkout feat/hetero-pp-dflash
cargo build --release -p hipfire-runtime --example daemon

# Smoke
HIPFIRE_DFLASH_DRAFTER_DEVICE=1 \   # gfx1010 = HIP device 1 on hipx
  ./target/release/examples/daemon \
  --model ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen3.5-2b.mq4 \
  --kv-mode asym3 --no-chatml --max 120

# Coherence
HIPFIRE_DFLASH_DRAFTER_DEVICE=1 ./scripts/coherence-gate-dflash.sh
HIPFIRE_DFLASH_DRAFTER_DEVICE=1 ./scripts/coherence-gate.sh

# Perf bench
HIPFIRE_DFLASH_DRAFTER_DEVICE=1 ./scripts/bench_dflash_27b.sh \
  --prompt benchmarks/prompts/lru_1100tok.txt --runs 3
# Expect: ≥33.75 tok/s decode (1.25× Exp #10 anchor 27.0)

# Solo baseline
unset HIPFIRE_DFLASH_DRAFTER_DEVICE
./scripts/bench_dflash_27b.sh --prompt benchmarks/prompts/lru_1100tok.txt --runs 3
# Expect: ~27.0 tok/s
```

## Continuation options

- **Same-session push**: continue the body refactor here. Costs ~150-200 LOC of careful surgery in `speculative.rs`; cross-card buffer placement at three phase boundaries.
- **Fresh session**: pick up with this doc + the pushed branch. Full context budget for the surgery.
- **Codex rescue**: hand the body refactor to Codex with this doc + `09-per-card-prefill-rates.md` + `10-gfx1151-solo-dflash-27b.md` as context. Two-phase plan (refactor → smoke on hipx) is well-bounded for delegation.

## References

- PRD: `docs/plans/hetero-pflash-dflash.prd`
- Empirical anchors: `docs/investigations/2026-05-07-rdna1-perf-research/09-per-card-prefill-rates.md` (5.07× WMMA prefill), `10-gfx1151-solo-dflash-27b.md` (1.84× solo, 27.0 tok/s)
- Smoke status (PR1 of PRD): `docs/investigations/2026-05-07-rdna1-perf-research/11-hetero-dflash-smoke-status.md`
- Foundation commit: `f47cdfd` `feat(hetero-dflash): drafter device pinning + cross-card helpers (PR-A foundation)`
- Plan: `~/.claude/plans/read-the-prd-v1-2-immutable-minsky.md`
