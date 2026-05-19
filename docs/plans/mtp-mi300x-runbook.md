# MTP on MI300x — runbook + scope

**Branch:** `feat/mtp-mi300x` (seeded off `feat/mtp` HEAD `f63da81a`)
**Hardware:** AMD MI300x, gfx942 (CDNA3, wave64, 192GB HBM3)
**Created:** 2026-05-19

## Why this branch exists

Prior MTP work on `feat/mtp` reached the limits of gfx1100 (7900 XTX) —
the deferred memo `project_mtp_native_head_deferred_2026_05_15` recorded
standalone MTP at 39.7 tok/s on 27B-3.5 (below AR's 45) with the trunk
`lm_head` dominating per-step cost. The master plan
(`docs/plans/mtp-dflash-composition-master-plan.md`) projects 250-350+
tok/s if composition with DFlash works, but no prototype exists yet.

MI300x changes three things relative to gfx1100:

1. **VRAM headroom (192GB)** — can hold full-precision MTP training
   state, larger MTP heads, longer sidecar context windows without
   OOM-managing each allocation.
2. **Compute headroom (~1.3 PFLOPS FP16 matrix)** — lm_head cost
   (which dominated MTP solo on gfx1100) shrinks proportionally;
   composition's "single trunk verify" pattern should be even cheaper
   relative to verify+drafter on this arch.
3. **CDNA3 wave64 ISA** — different kernel constraints. WMMA builtins
   don't apply; MFMA equivalents do. Existing gfx942 wave64 kernels
   already cover production dense + MoE paths; MTP head reuses them.

This is **not** "port MTP to MI300x" — the MTP code is arch-agnostic at
the Rust level. It is "use the MI300x to validate composition claims
the gfx1100 couldn't."

## Existing artifacts to reuse

### Already shipped (on feat/mtp)
- `crates/hipfire-arch-qwen35/src/qwen35.rs::Qwen35MtpHead` — single-block
  + lm_head forward path (commit `5e5c8e56`).
- `crates/hipfire-runtime/examples/mtp_compose.rs` — DFlash + MTP
  linear-chain composition demo (commit `cfabee79`, falsified on
  gfx1100 but reusable as a starting harness).
- `crates/hipfire-quantize/src/mtp_extract.rs` — safetensors MTP head
  extractor (arch_id=21, commit `426be400`).
- Bundled `.mq4-mtp` format (commit `d344af5b`).

### Rental infrastructure (on master, validated)
- `scripts/amd_quickdeploy.sh` — DO MI300x bootstrap.
- `scripts/mi300_after_3m.sh` — settle-then-bench harness.
- `scripts/mi300_chain_runner.sh` — multi-config sweep.
- Per `project_mi300x_rental_2026_05_18_delivery`: prior ~$10 / 3.5h
  rental; recipe is solid.

### Planning docs
- `docs/plans/mtp-dflash-composition-master-plan.md` — **read first.**
- `docs/plans/dflash-mtp-composition-orthogonal.md` — composition
  algebra.
- `docs/plans/mtp-cycle-anatomy.md` — per-cycle timing.
- `docs/plans/mtp-unsloth-target-2026-05-15.md` — external calibration.

## Phase ordering

Master plan recommends "probe composition FIRST with current weak MTP."
Same here, with MI300x-specific Phase 0:

### Phase 0 — rocprof current state on gfx942 (1 session, ~$5)

Spin up a droplet, deploy `feat/mtp`, run existing benches under
`rocprofv3 --kernel-trace --stats`. Output:

- Per-kernel time for solo MTP (`mtp_only_demo`).
- Per-kernel time for composition (`mtp_compose` / `dflash_mtp_demo`).
- Coverage audit via `scripts/coverage-audit.py` (new same-day
  tooling — mirrors the Rust-side `HIPFIRE_ROCPROF_CSV` integration).
- Atlas row with `arch=gfx942`, all metrics + rocprof coverage.

**Decision gate:** if MTP solo hits ≥60 tok/s without further work, the
bottleneck WAS gfx1100. Skip Phase 1, go to Phase 2 composition. If
solo is still <50, lm_head bottleneck is fundamental; Phase 1 needed.

### Phase 1 — solo MTP improvements (conditional on Phase 0)

Per deferred memo: bottleneck was lm_head BW. On gfx942's HBM3 (~5.3
TB/s) this should be a smaller fraction than on gfx1100's 960 GB/s. If
still the bottleneck:

1. **Compressed vocab head** — cvs4k/cvs8k/cvs16k variants already in
   `~/.hipfire/models/`. Pick the tightest that doesn't tank PPL.
2. **Native MTP head retrain** (EAGLE-style, deferred 2026-05-15):
   was projected at 1-2 weeks gfx1100; MI300x makes it tractable in
   hours.

### Phase 2 — MTP-extended verify composition

Implement the master plan's "single trunk verify" pattern:

- Trunk verify ONE batched forward over
  `[last_committed, draft_0..draft_{K1-1}, mtp_0..mtp_{K2-1}]`.
- Accept longest matching prefix: DFlash drafts first, then MTP chain
  off DFlash's last accepted draft.

Most work is in `crates/hipfire-arch-qwen35/src/speculative.rs` — extend
the existing DFlash verify path to take an MTP draft sequence and
report combined acceptance count. The MTP forward itself stays as-is.

Risk is correctness, not perf: MTP must accept only on exact-match
against trunk argmax at each position (no coherence gate skip).

## What MI300x is NOT for in this branch

- **Not for porting kernels to gfx942 MFMA.** Existing gfx942 wave64
  kernels cover production dense + MoE paths.
- **Not for a new quant format.** Stick to MQ4 / `.mq4-mtp` bundled.
- **Not a long-running rental.** Each session ≤4h, ≤$15.

## First-session checklist

```sh
# 1. Spin up droplet
./scripts/amd_quickdeploy.sh

# 2. SCP model bundles (rsync, ~30 GB)
rsync -avh ~/.hipfire/models/qwen3.5-27b.mq4-mtp \
           ~/.hipfire/models/qwen3.5-27b-dflash.mq4 \
   <droplet>:/root/.hipfire/models/

# 3. On droplet: pull this branch, build
ssh <droplet> 'cd hipfire && git fetch && git checkout feat/mtp-mi300x \
  && cargo build --release --features deltanet \
       --example bench_qwen35_mq4 \
       --example mtp_only_demo \
       --example mtp_compose'

# 4. Phase 0 rocprof — solo MTP
ssh <droplet> 'cd hipfire && mkdir -p /tmp/cov-mtp-solo && \
  ./scripts/rocprof-wrap.sh /tmp/cov-mtp-solo -- \
    HIPFIRE_PROFILE=1 \
    ./target/release/examples/mtp_only_demo \
      ~/.hipfire/models/qwen3.5-27b.mq4-mtp \
      --emit-atlas /tmp/cov-mtp-solo/atlas.jsonl'

ssh <droplet> './scripts/coverage-audit.py \
  --internal /tmp/cov-mtp-solo/atlas.jsonl \
  --rocprof  /tmp/cov-mtp-solo/trace_kernel_stats.csv'

# 5. Phase 0 rocprof — composition (same pattern, mtp_compose target)
# 6. Tear down. Record in docs/research/mtp-mi300x-phase0-results.md
```

## Cross-refs

- Master plan: `docs/plans/mtp-dflash-composition-master-plan.md`
- Deferred memo: `project_mtp_native_head_deferred_2026_05_15`
- gfx1100 falsified levers: `project_mtp_qualcomm_probe_v1_aborted_2026_05_15`
- Rental delivery: `project_mi300x_rental_2026_05_18_delivery`
- rocprof methodology: `docs/methodology/rocprof-coverage.md` (added
  same-session; cross-arch tooling for Phase 0)

## Status board

- [ ] Phase 0: rocprof solo + composition on gfx942
- [ ] Phase 0 decision gate: does solo MTP hit ≥60 tok/s on MI300x?
- [ ] Phase 1 (conditional): lm_head reduction lever
- [ ] Phase 2: MTP-extended verify prototype
- [ ] Phase 2 measurement: composition tok/s on canonical 27B-3.5 bench
- [ ] Land or document deferral
