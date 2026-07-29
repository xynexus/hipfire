# AGENTS.md - HIP kernels

This subtree owns HIP kernel sources. Kernel changes are high risk because small
layout or launch-shape edits can silently change model behavior.

## Kernel Rules

- Keep kernels HIP/ROCm-direct. Do not introduce Vulkan, wgpu, or cross-vendor
  compute code here.
- Consider RDNA2, RDNA3, and RDNA4 before accepting an optimization. If a path is
  arch-specific, guard and document it explicitly.
- For WMMA/MFMA/lane-layout work, use the AMD matrix calculator skill instead of
  relying on memory.
- After kernel edits, run the narrow relevant kernel/unit check if one exists,
  then `./tests/coherence-gate-dflash.sh` for behavior-facing changes.

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
