# RotorQuant KV-cache quantization for hipfire

> **Branch:** `feat/rtq` — seed commit only. No implementation yet.
> **Status:** OPEN. Pick this up cold; everything you need is in this doc.
> **Reference impl:** `DrBearJew/llama.cpp@tbq4-rdna3-experiment` (MIT) +
> `COLLAB_ROTORQUANT.md` in that repo.

## What it is

**RotorQuant** is a family of KV-cache rotation formats — drop-in replacements
for TBQ4's signed-FWHT rotation step. The reference ships two variants:

- **PlanarQuant** (2D-Givens): pairs of dimensions are rotated by a small
  bank of fixed 2D Givens (sin/cos) angles. Cheaper than FWHT — O(d) muls
  per block vs O(d log d). Designed for narrow head-dims where FWHT's
  log-factor dominates.
- **IsoQuant** (4D-quaternion): groups of 4 dimensions are rotated by a
  fixed 4D quaternion (8 muls + 4 adds per quad). Provides denser mixing
  than 2D Givens at slightly higher cost.

Both share the rest of the TBQ4 storage layout (per-block L2 + 4-bit packed
indices, ~4.25 bpv, block 128). The rotation choice is the only knob.

The hypothesis: on RDNA3, the FWHT log-factor is dwarfed by KV-cache memory
BW, so PlanarQuant should not win on perf — but **iso-quant 4D-quaternion**
may preserve outliers better than FWHT on certain head-dim shapes (e.g.,
head_dim=256 = 64 quads cleanly), and Givens may simplify the de-rotation
in scoring kernels.

## Why hipfire wants it

Same reason as TBQ4 (`feat/tbq`):
- KV cache compression at long context
- Better outlier preservation than naive int4
- Drop-in via `--kv-mode` flag selection

Difference: RotorQuant's rotation kernels are **simpler** (no FWHT shared
memory pattern) → easier first land than TBQ4. If RotorQuant clears the
coherence-gate, it validates the rotation-then-quantize family on hipfire
KV-cache scope; TBQ4 then becomes a perf-comparison ablation.

## Where it lives in hipfire

Same files as TBQ4 (see `docs/plans/tbq4-kv-cache-plan.md` for the full
list). The two branches are independent and bench-comparable; either one
can land first. The format enum (`Mode::Asym3`, etc. in `qwen35.rs` /
`dispatch.rs`) gets two new variants:

- `Mode::PlanarQuant` (2D Givens)
- `Mode::IsoQuant` (4D quaternion)

Or a single `Mode::RotorQuant { rotation: PlanarOrIso }` variant — pick
whichever is more idiomatic with the existing enum shape.

## First task (smoke gate, ~half-day each variant)

For each variant (start with PlanarQuant — simpler):

1. **Implement K-write kernel** at `kernels/planar_write.hip`
   (or `iso_write.hip`):
   - Input: pre-RoPE K vector `[head_dim]` F32
   - Apply rotation:
     - **Planar:** for each pair `(i, i+1)` in [0..head_dim/2), rotate by
       fixed angle `theta_i` (load from device-constant table). One sin
       and one cos per pair, two muls.
     - **Iso:** for each quad `(i, i+1, i+2, i+3)` in [0..head_dim/4),
       apply fixed quaternion product (8 muls + 4 adds per quad). The
       quaternion bank can be a single global quaternion or per-quad.
   - Then standard 4-bit centroid quant per 128-element block (same as TBQ4)
2. **Implement scoring kernel.** Two design choices:
   - **Option A:** dequantize K → un-rotate → standard Q·K dot. Simpler;
     adds a per-position un-rotation cost.
   - **Option B:** rotate Q at scoring time so attention happens in the
     rotated domain (no un-rotation of K needed). Cheaper at runtime
     since Q is one vector vs N positions of K, but requires the rotation
     matrix to be its own inverse (unitary) — true for Givens (transpose
     = inverse) and quaternion (conjugate = inverse), so both work.
3. **Wire into `Mode::PlanarQuant` enum variant** + `--kv-mode planar`
   CLI flag.
4. **Smoke test:** canonical 27B-3.5 LRU with `--kv-mode planar
   --max-n 1 --temp 0.0`. Coherent code = ship.

## Constraints

(Same as TBQ4 plan — see `docs/plans/tbq4-kv-cache-plan.md`)

- No Python in inference path
- Coherence-gate required before commit
- TriAttention sidecar compatibility (`crates/rdna-compute/src/triattn.rs`)
- Reuse existing FWHT machinery if helpful (sister `feat/tbq` branch)

## Why two branches not one

Independent quant kernels with different math + per-format scoring paths.
If they collide on enum names (`Mode::*`) the merge is trivial. Splitting
gates each one's empirical falsification independently — neither blocks
the other.

## Reference

- DrBearJew COLLAB_ROTORQUANT.md:
  <https://raw.githubusercontent.com/DrBearJew/llama.cpp/tbq4-rdna3-experiment/COLLAB_ROTORQUANT.md>
- Theory background: planar Givens rotations are the building block of
  Jacobi eigenvalue methods; quaternion rotations are SU(2) → SO(3)
  rotations. Both are unitary by construction (good for KV invertibility).

## Bench gate

Same as TBQ4:
- match or beat asym3 on canonical 27B-3.5 LRU bench (decode tok/s + coherence)
- longctx PPL on 16k-32k prompts (where rotated KV should pull ahead)
- coherence-gate full battery

If RotorQuant doesn't clear, document the regression and close as
falsified-experiment per the MEMORY.md falsified-format pattern.

## Sister branch

`feat/tbq` — TBQ4 (signed-FWHT rotation, same KV-cache scope). RotorQuant
is the cheaper-rotation cousin; if FWHT is BW-bound on RDNA3, RotorQuant
should match or beat it on perf with similar quality.
