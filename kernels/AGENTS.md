# AGENTS.md - HIP kernels

This subtree owns HIP kernel sources. Kernel changes are high risk because small
layout or launch-shape edits can silently change model behavior.

For a tree-wide inventory (family × arch), the JIT/selector dispatch model, and
the current dead-source / LDS-hazard / quant-contract findings, see
[`docs/kernels/kernel-audit.md`](../docs/kernels/kernel-audit.md).

## Kernel Rules

- Keep kernels HIP/ROCm-direct. Do not introduce Vulkan, wgpu, or cross-vendor
  compute code here.
- Consider RDNA2, RDNA3, and RDNA4 before accepting an optimization. If a path is
  arch-specific, guard and document it explicitly.
- For WMMA/MFMA/lane-layout work, use the AMD matrix calculator skill instead of
  relying on memory.
- After kernel edits, run the narrow relevant kernel/unit check if one exists,
  then `./tests/coherence-gate-dflash.sh` for behavior-facing changes.

## Wave-reduction idiom: reduce from offset 16, and pass the width

RDNA is **wave32**. A cross-lane reduction written the wave64 way —
`for (o = LANES/2; o > 0; o >>= 1) v += __shfl_xor(v, o)` with `LANES = 64` —
has an `o = 32` step that crosses a wave boundary, where `__shfl_xor` returns the
caller's own value. The step becomes `v += v` and the result is exactly **2x**
too large.

It does not error, does not produce NaN, and the doubling is uniform, so a
tolerance check with a loose bound passes. It was found by a control that
asserted an exact value (`got 0.88388, want 0.44194`) while writing
`qsa_block_score.hip`.

Copy the shape every attention kernel here already uses:

```c
for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, 32);
```

The trailing `32` is the width and is not optional — state it rather than
inheriting the default. The only kernels that may reduce from offset 32 are
wave64 ones, which live under `kernels/src/gfx906/` and say `wave64` in the
filename; `wave_reduction_offsets_are_wave32_safe` in
`crates/hipfire-rdna/src/kernel_arity.rs` enforces exactly that.

## Quant activation-basis contract (Opus Quant / Magnum)

- OQ (Opus Quant: `Oq8G256`, `Oq4G256`, …) and MQ (Magnum) weights are quantized
  in a **FWHT-rotated basis**. A GEMV/GEMM against them consumes activations that
  have been Hadamard-rotated first: the caller must run `rotate_x_mq` (or the
  AWQ-scaled / batched variants) on `x` before the kernel, and the reference host
  path does exactly this (see `oq8_gemv_into` in
  `crates/hipfire-dispatch/src/pipeline/steps.rs`). Feeding a raw (unrotated)
  activation to an OQ/MQ gemv compiles and runs but produces garbage. When adding
  a new OQ/MQ gemv kernel or an indexed-MoE expert path, rotate the input in the
  dispatch/arch caller, never assume raw `x`.
