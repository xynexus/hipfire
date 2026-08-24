# NPU draft offload for DFlash2: what exists, and why SERIAL offload cannot reach 55

Assessed after GPU-side work took Qwen3.8-27B spec decode from 5.75 to 38.97
tok/s against a 55.3 roofline, with `draft 39ms + verify 80ms` at B=6.

## What is already there

- `DflashScratch::npu_draft_forward` and `enable_npu_draft` exist and are real:
  `hipfire_xdna::DflashNpuBody` is a full W4A8 multicore body with flash
  attention, fc projection and per-layer weights.
- It declines losslessly when `b != body_b` or `position < l_ctx`, falling back
  to the GPU draft, so it is safe to leave wired.

## What is missing

1. **DFlash2 is unsupported.** `dflash_body.rs` contains ZERO occurrences of
   `conv` — the per-layer grouped dynamic causal convs that define DFlash2 have
   no NPU kernel. Confirmed by grep, not inferred.
2. **Never wired into serving.** `enable_npu_draft` has exactly one caller,
   `examples/dflash_spec_demo.rs`, behind `HIPFIRE_DFLASH_NPU_DRAFT=1`.
3. **No runnable artifacts on this box.** The body wants a weights dir, a flash
   manifest and an r14 dir; the defaults (`/tmp/dflash_w`,
   `/tmp/dflash_manifest_flash.json`, `~/.hipfire/npu/r14_1x2x128_nb128`) are all
   absent. `~/.hipfire/npu` holds 39 MB of `aie.mlir` + `.o` SOURCES, with no
   `final.xclbin`/`insts.bin` anywhere — `npu_decode_bench` panics on the
   missing xclbin. Building them needs the mlir-aie toolchain, which this tree
   already records as environment-blocked (placer skew).

## The arithmetic: serial offload is the wrong shape

Draft is BANDWIDTH-BOUND on the GPU: 6.5 ms/token for a 1.18 GiB drafter is
181 GB/s, 72% of the ~250 GB/s this part sustains. The NPU shares the same
LPDDR5X, so moving that work does not add bandwidth — a serial NPU draft moves
the same bytes through the same memory.

Worse, the measured NPU dispatch floor on this box is ~4 ms/dispatch. A 5-layer
drafter over 6 positions is ~30 dispatches ~= 120ms, against the GPU's 39ms. A
serial NPU draft would be ~3x SLOWER, not faster.

## The shape that would work: overlap, not substitution

The value of a second engine here is CONCURRENCY. Draft(N) depends on
verify(N-1), so nothing overlaps inside a cycle — but block N+1 can be drafted
OPTIMISTICALLY during verify of block N, assuming the last drafted token is
accepted, and discarded when it is not. At tau 5.333/B=8 full-accept cycles are
common (accept=7 is frequent in the phase trace).

Hiding the draft entirely leaves cycle = verify:

    80ms verify -> 5.333 / 0.080 = 66 tok/s

which clears 55. That is the only NPU story that reaches the target.

It requires, in order:

1. a DFlash2 grouped-dynamic-conv NPU kernel (none exists),
2. the mlir-aie toolchain unblocked to build xclbins,
3. an artifact path from the `.hfq` drafter to the body's weights/manifest/r14
   format (today only the demo harness produces these),
4. a pipelined-speculation restructure so draft N+1 issues before verify N
   retires,
5. **and the dispatch floor to fit 6-8 sequential draft steps inside an 80ms
   verify window** — at ~4 ms/dispatch it does NOT (30 dispatches ~= 120ms), so
   the NPU draft would become the new critical path even fully overlapped.

Point 5 is the one to settle first and it is cheap relative to the rest: build
ONE drafter-shaped NPU GEMM and time a dispatch. If the floor stays at ~4ms the
whole approach is dead regardless of how good the kernels are, because 30
dispatches cannot hide inside 80ms. If a drafter-shaped dispatch is ~1ms, the
budget closes and the rest is worth building.

## Recommendation

Do not start the DFlash2 NPU kernel yet. Measure the dispatch floor at the
drafter's shape first — it is a single number that decides whether any of the
remaining four items pay off.
