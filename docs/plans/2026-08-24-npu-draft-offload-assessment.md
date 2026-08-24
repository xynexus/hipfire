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

## MEASURED: the NPU streams weights at ~3 GB/s, and that settles it

Compiled artifacts DO exist (`~/.hipfire/npu/embgemma_aie2p_*` carry
`final.xclbin` + `insts.bin`), so `npu_decode_bench` runs. Across every geometry
that tiles:

    K=2048 N=8192  8 dispatches   2.730 ms/token-linear  ->  3.1 GB/s W stream
    K=2048 N=8192  8 dispatches   2.793 ms                   3.0 GB/s
    K=2048 N=8192  8 dispatches   9.757 ms                   0.9 GB/s
    K=2048 N=2048 16 dispatches   5.028 ms                   0.4 GB/s
    (weight upload itself: 8 MB in 15.2 ms = 0.5 GB/s)

The dispatch floor is FINE — 8 dispatches in 2.73 ms is 0.34 ms each, far better
than the ~4 ms this tree had recorded. That was not the problem. The problem is
streaming bandwidth: ~3 GB/s against the GPU's measured 250.

    drafter sweep, 1.18 GiB
      GPU    6.5 ms/token   (181 GB/s, 72% of peak)
      NPU    390 ms/token   (at 3.1 GB/s)
      ratio  60x slower

A B=6 draft on the NPU is ~2.3 SECONDS per cycle against an 80 ms verify window.
Fully overlapped it would still be the critical path by a factor of 29. The NPU
would need 81x more streaming bandwidth just to MATCH the GPU, which is not a
tuning gap.

Caveat, stated plainly: these runs report MISMATCHES because the configs are
guesses against harness kernels built for embedding-gemma, not for a drafter. The
NUMERICS are wrong. But the timing is the throughput of dispatch + weight
streaming on this hardware path, it is ~3 GB/s on every variant that ran, and no
amount of kernel tuning closes 81x.

Why this is consistent with "NPU ~55 TOPS ~= GPU ~56 TOPS": that is COMPUTE
parity. Drafting at B=1 is bandwidth-bound, not compute-bound, and it is the
memory path that differs by ~80x here. The NPU remains interesting for
compute-dense batched work; sequential single-token drafting is the worst
possible shape for it.

## Recommendation

Do NOT build the DFlash2 NPU kernel. The dispatch floor was never the blocker and
the bandwidth gap is not closeable by kernel work. The remaining GPU-side path to
55 — verify 80ms against its 58ms floor, draft 39ms against 29ms — is a far
better use of the same effort.
