# Upstream PR — mlir-aie: Trace `memref.view` in `dma_bd` buffer lowering

Ready-to-submit fix for **Xilinx/mlir-aie**. Branch `memref-view-dma-bd-fix`
(one commit) in the local `/home/sadara/build/mlir-aie` checkout; patch at
`mlir-aie-memref-view-fullelf-fix.patch` (git format-patch, applies to `main`).

## Title
`[AIEX] Trace memref.view in dma_bd buffer lowering (fixes --generate-full-elf)`

## Problem
`AIEDMATasksToNPU` (`--aie-dma-tasks-to-npu`) resolves a `dma_bd`'s buffer to a
runtime-sequence block argument via `AIEX::traceSubviewToBlockArgument`
(`lib/Dialect/AIEX/Utils/AIEUtils.cpp`), which follows `memref.cast`,
`memref.reinterpret_cast`, and `memref.subview`. It does **not** follow
`memref.view`, so a BD whose buffer is a `memref.view` slice of a runtime input
falls through to:

```
'aie.dma_bd' op Buffer argument must be a constant aie.buffer, a runtime
 sequence input argument, or a (chain of) subview(s) or cast(s) of a block
 argument with constant offsets and strides equal to one.
```

This is exactly the pattern the **fused whole-model `--generate-full-elf`**
lowering emits: the fused `main` device's DMA orchestration materializes every
buffer as a typed `memref.view` slice of one flat byte-arena runtime input, e.g.

```mlir
%v = memref.view %arg0[%byte_off][] : memref<Nxi8> to memref<2048xbf16>
"aie.dma_bd"(%v) <{... offset = 0, len = 2048, dimensions = ...}>
```

so `--generate-full-elf` fails NPU lowering for any non-trivial fused model.

## Fix
Add a `memref::ViewOp` case to `traceSubviewToBlockArgument`. A `memref.view`
result has an identity (contiguous) layout, so tracing only needs to accumulate
the constant byte offset and follow the source (bailing on dynamic sizes or a
non-constant byte-shift). ~18 lines; mirrors the existing `subview`/`cast` cases.

## Test
`test/bd-chains-and-dma-tasks/dma-tasks-to-npu/good_view.mlir` — a `dma_bd` on a
`memref.view` of a runtime input lowers to
`aiex.npu.address_patch {arg_idx = 0, arg_plus = <byte_off>}`. Mirrors the
existing `good_subview.mlir`. Fails without the fix (the pass errors out, so
there is no output for FileCheck to match); passes with it.

## Impact
Unblocks whole-model fused-ELF compilation. Downstream: took the IRON
Llama-3.2-1B fused decode from non-compiling to a working 8 tok/s single-
dispatch-per-token decode (vs ~1 tok/s for the non-fused re-prefill path).

## To submit
1. Fork `github.com/Xilinx/mlir-aie`, add the fork as a remote.
2. `git push <fork> memref-view-dma-bd-fix`
3. Open a PR against `main` with the title/body above.

(Not auto-submitted: no fork/credentials configured here.)
