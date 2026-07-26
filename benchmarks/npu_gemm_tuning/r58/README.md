# R58 — packed Opus HFP and in-core nibble decode

R58 is the first compute-bearing step after R57's feed-only measurements. The
input is the real loader-produced
`EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp` artifact:

- tensor blocks are already reordered once, offline, into the production
  `NpuGemmWholeScaled` stream;
- every 16 KiB physical tile retains 12 KiB of packed signed W4 data and a
  padded f32 scale tail;
- the AIE kernel still performs the required low/high-nibble decode/swizzle;
- no kernel performs a global tensor-block conversion.

`r58_decode_guard` decodes every weight nibble with AIE2P's native signed
`int4 -> int8` unpack, returns 64 lane sums, and exposes one decoded vector for
a byte-exact low/high/sign oracle. The performance harness
must compare trace-timed input-port bandwidth against the R57 packed feed-only
baseline; host wall time is reported only as context.

## Measured result

Three locked AIE2P trials per mode used the same layer-0 QKV artifact, eight
columns, four consumer rows, 18 blocks per column, 16-KiB tiles, 12-KiB packed
data regions, 1.8-GHz H clock, XRT 2.25.0, amdxdna 2.25.0, and firmware 1.1.2.65:

| mode | median wire GB/s | median packed-data GB/s | decode retention |
|---|---:|---:|---:|
| `PACKED_FEED_ONLY` | 56.173 | 42.130 | reference |
| `NIBBLE_DECODE` | 56.061 | 42.046 | 99.80% |
| `COMPUTE_STAGE1` | 55.933 | 41.949 | 99.77% vs decode |
| `COMPUTE_STAGE2` | 39.919 | 29.939 | 71.37% vs stage 1 |

All three decode trials passed the whole-stream lane-sum oracle and the
byte-exact real-vector oracle. The four-way logical decoded-byte median is about
336.4 GB/s. All stage-1 compute trials passed an exact int32 MMUL oracle and
reached 2.685 TOPS median for one native int8-by-int4 operation per 128 packed
bytes. Variations around 100% are measurement noise; the supported conclusion
is that both nibble decode and the first MMUL preserve the external feed roof
without traced receive stalls.

Stage 2 applies the complete 6x16 per-core MMUL schedule for one K=256 group.
It passes exact parity at 11.497 TOPS median. Wire delivery drops to 39.919 GB/s
and receive stalls rise to 57.45%, directly identifying compute backpressure.
The feed reduction is therefore explained and paired with a 4.28x useful-TOPS
gain. Raw rows are in
`../results/r58-nibble-decode-20260712.csv`.

## Scale and full-output correctness

An attempted checksum-only scale stage was rejected. Horizontal float
reductions interleaved with the virtual int8-by-int4 MMUL changed later integer
results under Peano. Per-N-block scalar sentinels and a full 64-lane integer
sentinel both caught the corruption; restoring rounding/saturation state and
reloading activation vectors did not fix it. The broken mode was removed rather
than timed or admitted.

The accepted scale/output gate uses the production `NpuGemmWholeScaled` vector
store schedule through the new `npu_opus_hfp_verify` example. On the real
layer-0 concatenated Q/K/V tensors and the real packed QKV HFP artifact, all
327,680 M256xN1280 outputs matched `OpusPackedMatrix::reference_f32` with
`max_abs=0.0000002`. Three wrapper iterations averaged 0.8635 ms (0.5829
logical TOPS). That wrapper number includes CPU activation preparation,
dispatch, synchronization, and output deblocking; it is a single-projection
measurement, not model tok/s or a trace-derived kernel ceiling.

## Generic HFP format matrix

The versioned HFP contract now also covers the existing full-K slab schedule,
so compact mixed `qt=36` matrices no longer redo global slab ordering at every
model load. `FullKV1` records W4, W8, or mixed W4+dense-W8 entries, direct/scaled
output flags, physical columns, group/slab geometry, source SHA-256, and payload
SHA-256. Mixed W4 base entries remain nibble-packed; representation-local
decode/swizzle remains an AIE kernel operation.

A locked real-artifact AIE2P matrix checked 26 cases with zero mismatches. It
covers W4/W8 plain, `+`, and `++`; mixed OQ4.125, OQ4.25 plain/`+`/`++`, OQ4.5,
and OQ6.5; overlay counts 1, 3, 7, and 39; individual N=256/768/1152 and
combined N=1280/2304 roles. Maximum absolute error was 2e-7 for W4/W8 and zero
for the direct mixed full-K integer/scaling oracle. A repeated OQ6.5 load kept
the HFP mtime, size, and complete-file SHA unchanged, proving cache reuse.

The one-iteration wrapper times in the result file came from a debug build and
are diagnostics only. They are not kernel ceilings, resident-layer timings, or
end-to-end throughput. Durable rows:
`../results/r58-opus-hfp-format-matrix-20260713.csv`.
