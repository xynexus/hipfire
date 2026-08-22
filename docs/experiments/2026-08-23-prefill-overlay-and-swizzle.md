# Prefill: the overlay was the real target, not the GEMM

State box: halo, Strix Halo gfx1151, 40 CU / 20 WGP, ~2.9 GHz, 128 GB UMA,
248.5 GB/s DRAM, 32 MB MALL, ROCm 7.14. Qwen3.8-27B oq4.25++ +CASK, 2059-token
prompt, kvarn, MAX_BATCH=512.

## Step 0 first: what fraction of prefill is even the GEMM?

Two sessions of GEMM tuning had never established this. `rocprofv3
--kernel-trace` on `bench_qwen35_speed` (it does its GPU work in-process, so the
daemon stdin route in AGENTS.md is not needed), segmented on the GEMM cluster to
isolate the measured prefill from model load:

| category | ms | % |
|---|---|---|
| GEMM (iu4x2 W4A8) | 3842.7 | 46.0% |
| **OVERLAY correction** | **2295.1** | **27.5%** |
| OVERLAY (k-major transpose) | 223.7 | 2.7% |
| attention | 1040.4 | 12.4% |
| gated delta net | 391.3 | 4.7% |
| everything else | 292.1 | 3.5% |
| KV | 147.3 | 1.8% |
| act quantize+interleave | 124.4 | 1.5% |

Window 8.457 s, **GPU busy 98.8%** -- so the earlier 99.2% inference-only figure
holds. A naive read of the same trace gives 47%, because the span then includes
model load; the large idle gaps all precede `__amd_rocclr_copyBuffer`.

**The sparse overlay was 30.2% of prefill** -- to correct 3 outliers per 256
weights, i.e. 1.17% of the weight positions and 1.17% of the MACs. Per corrected
weight it was ~50x less efficient than the dense GEMM it corrects. Two sessions
of effort had gone into the 46% and none into the 30%.

## Fix 1: the accumulators were not in registers (+3.8%)

The `q` loops were bounded by the runtime `nblk`, so `acc[q][t]` / `isum[q][t]`
could not be assigned static registers and LLVM fell back to M0-relative indexed
register moves. The hot loop was **144 instructions carrying 7 MAC-ish ops --
20.6 instructions per MAC**: 24 `v_movrels_b32`/`v_movreld_b32`, 33 `v_mov`, 77
SALU of pure index bookkeeping, around 3 loads.

Bounding every q loop by the compile-time `MAXQ` and predicating the tail keeps
q constant per unrolled copy. `v_movrel` 24 -> 0, 59 VGPRs, 16 waves/SIMD.

Cost: at B < 512 the predicated tail does wasted work (B=128 regressed 2%).
Prefill runs at ubatch 512 where `nblk == 4`, so the hot path does not pay it.

## Fix 2: the Y store was 67% of the kernel (+9.3%)

Ablation, gate/up M=17408 K=5120 B=512 (corr_T_ms):

| variant | ms | cost |
|---------|----|------|
| baseline | 1.657 | -- |
| gather replaced by a constant | 1.174 | gather = 0.48 (29%) |
| **Y store predicated off** | **0.555** | **store = 1.10 (67%)** |
| neither | 0.324 | |

Y is `[B, M]` f32, so it is contiguous in ROWS -- but a wave owned ONE row and
128 b, so its 128 stores hit 128 distinct cache lines using 4 bytes of each.
**32x write amplification on a read-modify-write.**

Coalescing wants consecutive rows in consecutive lanes; the gather wants
consecutive b in consecutive lanes, which is the entire reason XT is k-major.
Both at once: keep the gather, accumulate into LDS, transpose there before the
store. A workgroup takes 32 consecutive rows so one store covers a full 128 B
line.

The LDS tile must be `[32][128+1]`. At stride 128 dwords (0 mod 64) every lane
of the transposed readback lands on ONE bank -- a 32-way conflict that hands the
win straight back.

Bonus: the b-block becomes a grid dimension rather than a register array, so the
`nblk`-overrun GPU fault that forced 512-wide launches cannot occur.

## Fix 3: grid swizzle on the GEMM (+2.7%)

The launch was `[M/BM, B/BN, 1]` with `blockIdx.x` fastest, so the whole 47 MB
weight set was swept once per N-block -- 4 sweeps at B=512 against a 32 MB MALL
that cannot hold one. 190 MB of DRAM where 47 MB would do.

Making N the fast axis puts an M-block's N-blocks back to back; the resident
working set becomes ~174 kB of weights plus the whole 2.6 MB X tensor. The old
order was not arbitrary (it kept X resident) -- the swizzle keeps both, which
stops holding once X is large: gate/up at B=2048 regresses 0.91x, where X is
10.5 MB.

## Fix 4: the transpose ran per projection (+1.8%)

4463 transposes against 2303 quantizes: it depends only on the activation, so
gate/up and q/k/v each redid it. Same bug class as the fragment interleave.
Hoisted beside the quantize, guarded by a generation counter (the activation
scratch is reused across layers, so a pointer check would be wrong) keyed with
ng and n. Trace confirms 4463 -> 2304.

## Result

| | start | end |
|---|---|---|
| prefill | ~234 tok/s | **281.2 tok/s** |
| prefill window | 8.457 s | 7.381 s |
| GEMM | 3842.7 ms (46.0%) | 3667.5 ms (50.3%) |
| overlay | 2295.1 ms (27.5%) | 1476.6 ms (20.3%) |
| transpose | 223.7 ms (2.7%) | 107.2 ms (1.5%) |
| attention | 1040.4 ms (12.4%) | 1030.0 ms (14.1%) |

Every change is bit-identical (`max|diff| = 0.00e0` against the b-major
reference) except the swizzle, which is a pure launch-order change and passes
parity.

## What is left, in order

1. **GEMM, 50.3%**, still at ~54% of the measured 110.9 TOPS iu4 ceiling. Its
   21% staging and 20% LDS-read costs are latency-shaped and resisted every
   geometry change (see 2026-08-23-iu4-gemm-ceiling-attribution.md). Manual
   fragment pipelining is now also ruled out: LLVM already hoists 14 `ds_load`s
   to the top of the body and consumes them under progressively relaxed
   `lgkmcnt(12)...(5)` waits, i.e. it is already software-pipelined.
2. **Overlay, 20.3%.** The store is fixed, so the gather now dominates. It reads
   ~535 MB per gate/up call to consume a 2.6 MB activation, because each wave
   gathers for one output row and rows do not share idx values.
   **LDS staging of the group tile does not pay** at achievable row-block sizes:
   staging a 256x128 tile costs 32 kB against R*384 B gathered, so break-even is
   R = 85 rows per workgroup, and the accumulator registers cap R well below
   that. Documented so it is not re-derived.
3. **Attention, 14.1%**, never examined.
