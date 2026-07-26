# R64 — exact-R63 device trace

R64 injects declarative AIE trace operations into the parsed, canonical R63
raw MLIR. `build_r64.sh` first requires the untraced graph to be byte-identical
to `r15_gen.py w4 3 6 8`; trace configuration is then added without regenerating
or changing any existing core, ObjectFIFO, runtime DMA, or immutable weight
layout operation.

Core-tile packet-flow trace routing is not viable on this fully occupied graph:
both eight-flow and four-flow builds remained CPU-bound for more than five
minutes after address assignment, and even one core flow exceeded four minutes.
Those failures are retained as tooling limits, not treated as kernel results.

The admitted path traces one shim at a time. Shim trace lowering builds in
seconds and observes the native DMA task start/finish/starvation events without
routing a packet through the saturated compute network. Columns 0–3 carry both
activation and weight input streams and retain their terminal output event.
Columns 4–7 lose the final S2MM finish event at trace stop even though the full
output oracle passes, so their spans are not fabricated or admitted.

- input DMA channel 0 running/stalled;
- input DMA channel 1 running/stalled;
- output DMA channel 0 running/stalled;
- vector instructions and stream stalls.

The exact device arguments are 589,824 activation bytes, 2,359,296 compact
QKV `.rdna2.hfp` bytes, and 1,769,472 physical output bytes. At the measured
56.173 GB/s feed roof, their serial byte floor is about 84 microseconds. The
padded 679.477M operations have an optimistic 59-microsecond compute floor at
R58 stage 2's 11.497 TOPS. These are isolated floors, not additive predictions.

Run only under `hipfire lock`:

```bash
export PYTHONPATH=/opt/xilinx/xrt/python
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/opt/rocm/lib
export R63_QKV_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp"
R64_TRACE_TILE=shim R64_TRACE_START=0 R64_TRACE_COLS=1 \
R64_CACHE_DIR="$HOME/.hipfire/npu/embgemma_r64_trace_shim0" ./build_r64.sh
R61_CACHE_DIR="$HOME/.hipfire/npu/embgemma_r64_trace_shim0" \
R64_TRACE_SIZE=16777216 R64_TRACE_TILE=shim R64_TRACE_START=0 R64_TRACE_COLS=1 \
  "$HOME/.venv/bin/python" ../r61/r61_raw_run.py
```

Acceptance requires the full 327,680-output oracle in each run, all four
activation-bearing shim columns, non-overflowed and fully paired required
intervals, and a reported device input-to-output span. Trace timings are
compared with the warmed production wrapper; cold Python command timing remains
diagnostic only.

## Result

Twelve locked fresh-process traces cover shim columns 0–3, three trials each.
Every run passes all 327,680 real outputs with `max_abs=3.8147e-6`; no trace
overflows or unmatched required events occur.

| metric | median | range |
|---|---:|---:|
| device input-to-output span | 241.248 us | 240.189–243.356 us |
| aggregate effective traffic | 19.559 GB/s | 19.390–19.645 GB/s |
| output-starvation time | 198.240 us | 197.037–200.795 us |
| final drain after both inputs finish | 28.286 us | 28.176–28.388 us |
| padded compute rate | 2.817 TOPS | — |
| useful compute rate | 2.086 TOPS | — |

The warmed production wrapper is 1.0292 ms, so the traced device window is
23.4% of wrapper time and roughly 788 us remains in activation preparation,
submission/synchronization, and physical-output deblocking. The output stream
is starved for most of the device window, confirming compute backpressure, but
the device projection itself is not the dominant wrapper cost.

The next ratchet is R65: keep the current compact W4 `.rdna2.hfp` and projection
math, but emit the mutable result directly into the already-verified R28/R27
BF16 Q and K/V attention layouts. This removes host deblocking and enables a
shared-BO projection-to-attention chain; it does not reorder immutable tensor
blocks in the kernel.

Durable rows: `../results/r64-full-qkv-shim-trace-20260713.csv`.
