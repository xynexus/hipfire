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

### Commit 2 — `feat(hetero-dflash): offset-aware peer copy primitives + cross_card_copy_at` (`e86c105`)

- `hip-bridge::ffi::HipRuntime::memcpy_peer_at` + `memcpy_peer_at_async`: byte-offset-aware peer copies for sub-slicing without intermediate `DeviceBuffer`s.
- `multi_gpu::cross_card_copy_at`: free-function form of offset-aware peer copy with the same async-stream-vs-sync-host-staged decision tree as `cross_card_copy`.

These are the primitives the spec-step body refactor (next commit) will use to ship sub-slices of `draft_scratch.x`, the embedding lookup output, and `hidden_rb` slots between target and drafter cards.

### Commit 3 — `feat(hetero-dflash): spec_step_dflash dual-Gpu body refactor (PR-A step 2)`

Generation-side surgery. Default codepath (single-Gpu DFlash, env unset) byte-identical to master; hetero path opt-in via `HIPFIRE_DFLASH_DRAFTER_DEVICE=N`.

- `spec_step_dflash` signature gains `drafter_gpu_opt: Option<&mut Gpu>` as the second parameter (immediately after the target `gpu`).
- `VerifyScratch` gains two cross-card staging buffers on `target_gpu`: `embd_staging` ([max_n × dim] f32) and `draft_hidden_staging` ([max_n × dim] f32) — ~800 KB at 27B max_n=20, dim=5120. Allocated unconditionally; unused on the homogeneous path.
- Phase 2 hetero branch: write B target embeddings into `embd_staging` on `gpu`, single `cross_card_copy_at` of B × hidden f32 (~327 KB) into `draft_scratch.x` on the drafter, drafter-side `cross_card_wait`. Homogeneous branch unchanged (D2D into draft_scratch.x).
- Phase 4 routes `dflash::draft_forward` to the drafter Gpu via match on `drafter_gpu_opt.as_deref_mut()`. Same body, same arguments — no signature change in `draft_forward`.
- Phase 5 hetero branch: cross-card-ship `draft_scratch.x[h..h+(B-1)*h]` (drafter) into `draft_hidden_staging` (target), then run target's lm_head GEMM (HFQ4G256 / MQ4G256 / MQ3G256 / Q8_0) against the staging buffer. Per-row `weight_gemv` fallback path now refuses cleanly under hetero (would dispatch wrong-device buffers; production target dtypes hit the batched path).
- Phase 9 dispatches the new `scatter_hidden_block_to_interleaved_cross_card` helper when hetero — same semantics as the existing per-(row, ext) D2D scatter, but each copy goes through `cross_card_copy_at` + `cross_card_wait`.
- `daemon.rs::generate_dflash` drops the load-time refusal added in commit 1 and threads `m.dflash_drafter_gpu.as_mut()` into `spec_step_dflash`. Disjoint-field borrows let `df = m.dflash.as_mut().unwrap()` and `m.dflash_drafter_gpu.as_mut()` coexist on the same call.
- `dflash_spec_demo.rs` (single-Gpu example) passes `None` and is byte-identical to master at runtime.

Workspace cargo check clean. Diff: `crates/hipfire-arch-qwen35/src/speculative.rs` +259, `daemon.rs` +14/-12, `dflash_spec_demo.rs` +1.

## What remains (Step 3: smoke + bench on hipx)

The plumbing is in place. Step 3 is empirical: the cross-card path now needs to be exercised on the real `hipx` rig (gfx1151 target + gfx1010 drafter) to confirm correctness, coherence, and the ≥1.25× perf anchor.

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

## Continuation options for Step 3

- **hipx run**: ssh to hipx, `git fetch && git checkout feat/hetero-pp-dflash`, follow the test path above. First verify `HIPFIRE_DFLASH_DRAFTER_DEVICE` unset reproduces the Exp #10 anchor (~27.0 tok/s solo), then enable it and look for ≥33.75 tok/s + clean coherence.
- **Codex rescue**: if the smoke trips up on a wrong-device-buffer panic or stream-ordering bug, hand the trace to Codex with this doc as context. The cross-card phases (2 / 5 / 9) are the most likely failure surface; the rest is byte-identical to master.

## References

- PRD: `docs/plans/hetero-pflash-dflash.prd`
- Empirical anchors: `docs/investigations/2026-05-07-rdna1-perf-research/09-per-card-prefill-rates.md` (5.07× WMMA prefill), `10-gfx1151-solo-dflash-27b.md` (1.84× solo, 27.0 tok/s)
- Smoke status (PR1 of PRD): `docs/investigations/2026-05-07-rdna1-perf-research/11-hetero-dflash-smoke-status.md`
- Foundation commit: `f47cdfd` `feat(hetero-dflash): drafter device pinning + cross-card helpers (PR-A foundation)`
- Step-1.5 commit: `e86c105` `feat(hetero-dflash): offset-aware peer copy primitives + progress doc`
- Plan: `~/.claude/plans/read-the-prd-v1-2-immutable-minsky.md`
