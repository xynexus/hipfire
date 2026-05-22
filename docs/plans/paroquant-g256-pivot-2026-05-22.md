# PARO4G256 Pivot — PRD (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` HEAD `9359fd9b`.
> Hardware target: hiptrx R9700 / gfx1201 / RDNA4.
> Trigger: m2 falsification on G128 (both i8 MMQ and FP16 WMMA) empirically
> validates Phase 1's deferred conclusion that **G256 is structurally optimal
> for prefill kernel efficiency on gfx12**.

## The trigger (why this PRD exists now)

Phase 1 (2026-05-21) ran a CPU-side quality probe of G256 vs G128 PARO
and decided to skip the G256 native runtime: "G256 grid alone is too small
a BW win (-1.8% bytes) to justify a parallel format implementation." That
decision was BW-only. Today's m2 falsification on G128 reveals the
**compute-amortization** side that Phase 1 didn't measure:

```
                       VGPR   Waves/SIMD   K-tiles/group   Per-WG work
MQ4 G256 k2 (gfx12)    60     25           16              1.0×
MQ4 G256 m2 (gfx12)    ~80    19           16              2.0×    ← +20-40% (production)
PARO G128 k2 (gfx12)   69     22           8               0.5×
PARO G128 m2 (gfx12)   90     17           8               1.0×    ← -25% wall (falsified)
```

m2 only wins when per-WG work growth outpaces occupancy loss. G256 gets
2× compute for 25% occupancy loss = net win. G128 gets 2× compute for
23% occupancy loss but starts from half the K-amortization = falsified.

Quoting Phase 1's own doc (`docs/investigations/paro-g256-perfmax/phase-1-g256-quality-probe.md:96`):

> **"PARO4G256_MQ (Path B) is the only path to MQ4-class perf"** but
> requires re-rotating weights against MQ4 G256 quant grid — a ~5–6 day
> re-quantization+kernel project.

The deferred work is now the gating dependency for closing the prefill
gap. This PRD scopes that work.

## Goal

PARO A3B prefill ≥ 1500 tok/s wall (vs current 730) on shisa A3B-PARO
@ --prefill 256, gfx1201. Stretch goal: 2500+ tok/s wall, matching MQ4
G256's claimed ceiling.

Quality goal: NRMSE ≤ 0.10 vs current G128 production output (Phase 1
gate: ≤ 0.10 per-module).

## Two paths to G256

### Path A (PARO4G256_AWQ — "regroup only")

Take existing PARO4G128 weights and **regroup** their per-group scales/zeros
into G256-shaped storage. One G256 group covers two consecutive G128 groups;
we pick ONE scale+zero per 256-element block (losing precision vs the
original two G128 scales).

- **Quality**: NRMSE 0.084-0.085 (Phase 1 probe on 0.8B + 9B, AWQ storage)
- **Cost**: ~1-2 days
  - Regrouping script (one-time per checkpoint, runs CPU-side)
  - DType::ParoQ4G256 enum + storage layout
  - Port of MQ4 G256 gfx12 kernels (FP16 WMMA + m2 + i8 MMQ + WMMA k2 cross-arch fallback) to PARO4G256
  - Dispatch routing
  - Loader (regroup at load OR pre-converted weights file)
- **Quality risk**: 0.084 NRMSE is within the GOAL.md gate but is a clear
  quality regression from G128 (no NRMSE since native). The output text
  may shift; coherence-gate must validate before default-on.

### Path B (PARO4G256_MQ — "re-rotate against G256 grid")

Re-run AWQ calibration at G256 granularity, producing fresh Givens rotation
pairs/thetas/channel_scales aligned to G256 group boundaries. The output
weight file is fundamentally different from the existing G128 PARO.

- **Quality**: NRMSE 0.092-0.096 (Phase 1 probe, MQ storage)
- **Cost**: 5-6 days
  - Re-quantization pipeline (re-run AWQ at G256; runs offline on calibration data, hours of compute)
  - New PARO4G256_MQ storage layout (rotation metadata aligned to G256)
  - All the same kernel + dispatch + loader work as Path A
  - Plus regenerating the shisa A3B-PARO weights file from scratch (large download + compute)
- **Quality**: actually SLIGHTLY WORSE than Path A on the probe (0.092 vs
  0.084 NRMSE), so the precision gain isn't there. Path B's value is
  enabling MQ-style row-major storage that the existing MQ4 kernels read
  natively — but our PARO4G256 kernels can be designed to read either
  layout, so MQ-storage doesn't add value.

### Verdict: Path A is the practical answer.

Phase 1 documented Path B as "the only path to MQ4-class perf" because
it assumed reusing MQ4 G256 kernels verbatim. We don't need that — we
write PARO4G256 kernels that handle the per-row Givens rotation as an
X-side pre-pass (already how G128 works). The kernel layout (G256 group
size, 16 K-tiles/group, m2-amenable VGPR profile) is what matters, not
whether the storage format is byte-identical to MQ4G256.

**Recommended: Path A.** Lower cost, equivalent quality, same kernel
benefit.

## Implementation plan (Path A)

### Phase 0 — kernel scaffold (1 day, can start independently)

Port the gfx12 G256 kernels to PARO4G256 naming/dispatch with a hardcoded
group stride and synthetic data validation. This is reusable work — it
runs even before the conversion tool exists.

Files to create:
- `kernels/src/gemm_paro_q4g256_moe_grouped_wmma.gfx12.hip` (mirror of
  the FP16 WMMA G128 kernel I just shipped, but with K/256 + 136 B/group
  stride + 16 K-tiles per group)
- `kernels/src/gemm_paro_q4g256_moe_grouped_wmma_m2.gfx12.hip` (mirror of
  MQ4 G256 m2 — should win on G256 layout)
- `kernels/src/gemv_paro_q4g256_moe_gate_up_indexed.hip` (decode-path)
- Possibly: i8 MMQ G256 variant for opt-in research

Files to modify:
- `crates/rdna-compute/src/kernels.rs` — SRC registrations
- `crates/rdna-compute/src/dispatch.rs` — dispatch methods
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — routing under
  `HIPFIRE_MOE_PARO_G256=1` opt-in
- `crates/hipfire-arch-qwen35/src/...` (DType definition) — `DType::ParoQ4G256`

Synthetic validation: write a small Rust test that feeds the kernel
hand-crafted G256-shaped PARO weights + known X and asserts output is
within 1e-4 of a CPU reference.

### Phase 1 — conversion tool (1 day, depends on phase 0)

Build a Python conversion script that takes existing PARO4G128 safetensors
(e.g. shisa A3B-PARO) and emits PARO4G256 storage. Per-pair of G128 groups:

```python
# Regrouping per Phase 1 Path A:
g256_scale  = (g128_scale[g]  + g128_scale[g+1])  / 2  # or max() — TBD by quality
g256_zero   = (g128_zero[g]   + g128_zero[g+1])   / 2
g256_qweights = concat(g128_qweights[g], g128_qweights[g+1])  # 64 B + 64 B = 128 B payload
# Plus rotation metadata stays per-row (unchanged), since Givens rotation
# is applied to X before the kernel (not per-group).
```

Output a `.paroquant-g256` safetensors file or a binary `.hfq` checkpoint.

Phase 1 probe script (`scripts/paroquant_g256_probe.py`) already exists
and validated this conversion math; reuse it for the tool's NRMSE
self-check during conversion.

### Phase 2 — end-to-end integration + bench (1 day, depends on phase 1)

- Loader: accept PARO4G256 dtypes for shisa A3B-PARO weights
- Bench: shisa A3B-PARO via PARO4G256, FP16 WMMA + m2 default-on
- Coherence-gate validation (mandatory per CLAUDE.md)
- A/B vs G128 baseline (730 tok/s):
  - Expected: G256 k2 ≥ 800 tok/s (per-group amortization win)
  - Expected: G256 m2 ≥ 1200 tok/s (the lever G128 couldn't use)
  - Stretch: G256 m2 + perm nibble unpack ≥ 1500 tok/s

### Phase 3 — A3B coherence + ship (0.5-1 day)

- Run coherence-gate on shisa A3B-PARO + z-lab A3B-PARO with G256
  default-on
- Verify text output sensible (no attractor / repetition)
- If clean: flip default-on for gfx12 PARO. Document quality delta
  (NRMSE 0.085) in CLAUDE.md.

## Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| NRMSE 0.085 causes attractor/loop in A3B prose | Medium | Coherence-gate must pass. If fail, fall back to Path B (re-rotation) or keep G128 default. |
| G256 m2 doesn't actually win on PARO (kernel-correctness or pipelining issue) | Low | Phase 0 validates kernel correctness with synthetic data before phase 2. |
| Regrouping math (avg vs max for merged scale) hurts quality | Medium | Try both, pick whichever passes coherence-gate. |
| Loader complexity (dual DType paths) bloats code | Low | Mirror the existing MQ4G256 / ParoQ4G128 split. |
| Bench cold-run overhead obscures the win | Low | Use `--prefill-runs 4` with JIT warmup (per the methodology established today). |

## Total cost estimate

| Phase | Duration | Can be parallelized? |
|---|---|---|
| Phase 0 (kernel scaffold) | 1 day | Yes — kernel work doesn't depend on conversion tool |
| Phase 1 (conversion tool) | 1 day | Yes — Python tooling parallel to kernels |
| Phase 2 (e2e integration) | 1 day | Sequential — depends on phase 0 + 1 |
| Phase 3 (coherence + ship) | 0.5-1 day | Sequential |
| **Total** | **3-4 days** | (Phase 1 estimated 5-6 days for Path B; Path A is faster) |

## Decision point

Phase 0 (kernel scaffold) is the cheapest commit — pure Rust + HIP code,
no model conversion, validated against synthetic data. It produces
reusable artifacts even if the rest of the pivot stalls. Recommendation:
start with Phase 0 as a sub-1-day proof-of-concept; if the synthetic
benchmark confirms G256 m2 wins on PARO-layout weights, commit to Phases
1-3.

## Alternative: Don't pivot

If the pivot scope is unacceptable, the remaining G128 optimization
ceiling is:
- Givens-rotation kernel profiling/fusion (PARO-specific overhead MQ4
  doesn't have) — likely 5-10% headroom
- Decode m4-without-LDS variant — uncertain, possibly 5% headroom
- hipGraph capture for routed-MoE decode — engineering-heavy, 10-20%
  headroom on decode but not prefill

These keep the G128 default at ~730 tok/s prefill with maybe 50-100
tok/s of further refinement, but the m2 lever stays locked.
