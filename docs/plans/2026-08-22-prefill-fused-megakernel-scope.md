# Scope: fused megakernel for the KVarN + iu4x2 batched prefill path

> **STATUS 2026-08-22: EXECUTED AND KILLED.** Phase 0 (operand mapping) and a
> second fusion attempt were carried out. The kill criterion below was met: the
> fused kernel is 6-12% WORSE than the split pair, not 15% better. More
> importantly, **the DRAM premise in this document is wrong** — see
> "CORRECTION" below. Kept as the record of why, not as a plan.

State box: halo, Strix Halo gfx1151, 128 GB UMA, 8000 MT/s (248.5 GB/s measured
pure-read, 64 KB LDS/workgroup, 1536 VGPRs/SIMD). Qwen3.8-27B oq4.25++
(`OqPlusCompact` qt 36, G=256, n_out=3), KVarN KV, `HIPFIRE_KVARN_BATCHED_PREFILL=1`,
2059-token prompt. Measured at 217 tok/s prefill after the iu4x2 routing.

## Finding that reframes the whole question: prefill is NOT launch-bound

A first read of the kernel trace said GPU busy was 52.2% of wall — which would
have made launch-gap elimination worth up to 1.9x and a megakernel the obvious
move. **That reading was wrong.** The daemon loads the model inside the traced
process, and 47.1% of the span is idle following `__amd_rocclr_copyBuffer`
during *weight streaming*, before the first GEMM ever runs. Restricted to the
inference region:

| | |
|---|---|
| GPU busy / span | **99.2%** |
| launches | 25,813 |
| copyBuffer gap during inference | **0.0%** |

So **a megakernel built to eliminate launch overhead can recover at most 0.8%.**
That is the same conclusion the ZAYA decode megakernel reached — coop kernel,
four phases byte-identical, tok/s flat, because decode was GPU-exec-bound. Do
not spend the effort again for that reason.

The case for fusion here is entirely about **DRAM traffic**, and it is strong.

## The real target: the overlay correction is 23.9% of prefill

Inference-region kernel shares, both compact GEMM wrappers routed to iu4x2:

| kernel | calls | % of prefill busy |
|---|---|---|
| `gemm_oq_compact_iu4x2_wmma` | 4464 | **52.3%** |
| `oq_compact_overlay_correct_t` | 4464 | **23.9%** |
| `attention_flash_kvarn_tile_batched` | 368 | 11.1% |
| `gated_delta_net_f32` | 480 | 3.7% |
| `oq_compact_x8_transpose` | 4464 | 2.0% |
| everything else (32 kernels) | — | ~7% |

The correction applies **3 entries per 256 weights — 1.2% of the arithmetic —
for 23.9% of the time.** A ~20x inefficiency, and it is not mysterious. Its
traffic model, per gate/up call (M=17408, K=5120, ng=20, B=256, n_out=3):

| term | bytes |
|---|---|
| activation gathers, `M x ng x n_out x B` | 267.4 MB |
| `y` read-modify-write, `2 x M x B x 4` | 35.7 MB |
| **total** | **303.0 MB** -> 1.22 ms at 248.5 GB/s |
| the GEMM's own weight bytes, `M x ng x 136` | 47.3 MB -> 0.19 ms |

**The correction moves 6.4x the GEMM's weight traffic.** Measured 0.98 ms sits
below the 1.22 ms model, so L2 reuse across rows helps, but the kernel is
traffic-bound, not compute-bound. Each row's 3 overlay indices are its own — the
positions are per-row order statistics, not shared — so every row re-reads its
own three activation rows.

## CORRECTION: the DRAM premise below is WRONG

This document argued that the correction moves 303 MB against the GEMM's 47 MB
of weights, so fusing it into a kernel that already has the activation in LDS
deletes ~267 MB of DRAM traffic. **That is not what happens.**

The arithmetic `M x ng x n_out x B` counts *gather operations*, not distinct
bytes. The distinct working set is only `256 idx rows x B x ng` = **1.31 MB**,
which fits in the 2 MB L2 (never mind the 32 MB MALL). Proof by physics: 267 MB
in the measured 0.978 ms implies **273 GB/s, above the 248.5 GB/s DRAM
ceiling** — impossible unless the reads are cache hits.

So the correction is **gather-throughput-bound, not DRAM-bound**, and fusion has
no DRAM saving to bank. It only trades L2-hot gathers for LDS gathers while
adding register pressure. That is precisely what both attempts measured.

## Phase 0 result: the WMMA operand layout forbids the proposed fix

`matrix_calculator.py --architecture gfx1151 --instruction
v_wmma_i32_16x16x16_iu4 --register-layout -B -w 32` gives, for `B[K][N]`:
**lane `n` owns column `n`, with all 16 K values packed as nibbles across 2
VGPRs.** The WMMA therefore *requires* b-major staging (column contiguous over
K). A K-major tile cannot feed it, so the "K-major LDS view" proposed below is
not available as a replacement, only as a +16 KB addition that would drop
resident workgroups per CU from 3 to 1.

A single int8 b-major tile (64 x 256 = 16,384 B, actually *smaller* than
xh+xl's 16,896 B) would halve the overlay's LDS reads, but costs ~600 extra VALU
ops per group per thread to re-split nibbles inside the WMMA loop — the operand
loads read the same 4 dwords either way. Not taken.

## Attempt #2 result (j-outer ordering)

Attempt #1 read the 6-byte overlay record inside the `nb` loop, so it was
fetched 4x redundantly. Attempt #2 reorders to j-outer/nb-inner, reading it once
per row and holding only the current row's 3 entries — 166 VGPRs, **9 waves, 0
spills**, occupancy fully preserved, parity PASS on all 7 shapes.

It bought ~2%. Medians of 3, ms:

| shape | fused #2 | fused #1 | split pair |
|---|---|---|---|
| gate/up | 3.332 | 3.415 | **3.137** |
| down | 2.995 | 2.980 | **2.674** |
| qkv | 1.136 | 1.143 | **0.996** |
| wo | **0.672** | 0.674 | 0.722 |
| gate/up B=128 | 1.865 | 1.833 | **1.698** |
| gate/up B=512 | 6.612 | 6.757 | **6.340** |

So the record redundancy was never the bottleneck — the 192 LDS byte-gathers
are, and they do not get cheaper by reordering. Reverted; the tree keeps the
separate k-major pass.

## Why fusion looked like the right lever

`gemm_oq_compact_iu4x2_wmma` **already stages the entire 256-element activation
group in LDS** before its two WMMA passes. An overlay lookup inside that kernel
therefore costs **zero DRAM traffic** — the value is already resident. Fusing
deletes all 267 MB of gathers and the 35.7 MB `y` round-trip, and takes the
2.0% transpose with it.

Upper bound: if the correction (23.9%) and transpose (2.0%) vanish and the GEMM
grows by X% of its own 52.3%:

| GEMM cost increase | prefill time | tok/s from 217 |
|---|---|---|
| 0% | 0.741x | 293 |
| +10% | 0.793x | 274 |
| +20% | 0.845x | 257 |
| +46% | 1.0x | 217 (break-even) |

So the fusion has ~46% of the GEMM's runtime as headroom before it stops paying.

## Why attempt #1 lost, and what must change

The epilogue fusion was implemented, is **numerically exact** (parity PASS, 7
shapes, max|rel| 1.55e-7) and holds occupancy (167 VGPRs, 9 waves, 0 spills) —
and it measured **9% SLOWER** than keeping the correction separate. See
`docs/experiments/2026-08-22-overlay-gemm-epilogue-fusion.md`. It did not lose on
DRAM. It lost on two things:

1. **LDS lane mapping.** The WMMA layout forces one b-column per lane
   (`b_col = nb*16 + lane`), and the LDS tile is staged **b-major**
   (`xh + c*lds_stride + d*4`, where `c` is the b index). So an overlay lookup is
   a *byte* gather, and each thread does `4 nb x 8 rows x 3 entries x 2 planes =
   192 LDS byte-gathers per group`. The standalone k-major kernel is free to
   choose its mapping, gives each lane 4 consecutive b, and issues the same
   gathers as **dwords** — 4x fewer ops.
2. **Register pressure.** `iacc_hi[4][8] + iacc_lo[4][8] + facc[4][8]` already
   own ~96 VGPRs. All three attempts to cache the 6-byte overlay record per row
   spilled: MAXOV=4 register cache 152 spills, MAXOV=3 86 spills, compile-time
   unrolled inline read 277 spills. Only the dynamic-trip inline read fits.

**The proposal is therefore not "fuse harder" — it is to fix the LDS layout.**

## Candidate: dual-view LDS staging

Stage a **K-major int8 view** of the group alongside (or instead of) the two
b-major nibble planes, so that the 16 b-columns a lane group needs for `x[idx]`
are **contiguous in LDS**. That turns 192 byte-gathers into a handful of
dword-wide reads — the exact advantage the standalone k-major kernel already
demonstrates at 1.5-3.0x over the b-major correction.

LDS budget, current kernel (`OQC4X2_COLS=64`, `lds_stride=132`):

| buffer | bytes |
|---|---|
| `xh` | 8,448 |
| `xl` | 8,448 |
| **current total** | **16,896 (16.5 KB)** |
| proposed K-major int8 view, `256 x 64` | +16,384 (16 KB) |
| **proposed total** | **~32.9 KB of 64 KB** |

Open question this scope does not answer: whether the two WMMA passes can read
their operands from a single K-major int8 tile (deriving nibbles on the fly),
which would make the extra buffer a *replacement* rather than an addition. That
depends on the `V_WMMA_I32_16X16X16_IU4` operand register mapping and should be
settled with the AMD matrix instruction calculator before any code is written.

## Phases and exit criteria

| phase | work | exit criterion |
|---|---|---|
| 0 | Settle the IU4 operand mapping with the matrix calculator; decide dual-view vs K-major-only | a written lane->element map; no code |
| 1 | Add the K-major LDS view; keep the WMMA passes untouched; fuse the overlay against it | `parity_gemm_oq_compact_iu4x2` PASS with the overlay-inclusive oracle |
| 2 | Bench vs `unfused GEMM + separate k-major correction` on all 6 shapes | fused total beats 3.137 / 2.674 / 0.996 / 0.722 / 1.698 / 6.340 ms |
| 3 | Resource check | <= 9 waves/SIMD lost at most 1 step, 0 spills |
| 4 | End-to-end | prefill > 240 tok/s at 2059 tokens, text identical, KLD vs iu8 ref within the 2.96e-4 already accepted |

**Kill criteria.** Abandon and keep the separate pass if, at the end of phase 2,
the fused kernel is not at least 15% better than the split pair on gate/up and
down — the two shapes that dominate. A marginal win does not justify a second
maintained kernel path, and 9%-worse is where attempt #1 already sits.

## Dependency the scope must name

**All of this pays off only inside `HIPFIRE_KVARN_BATCHED_PREFILL=1`.** In the
shipped default configuration batched prefill does not run at all: fp32 KV fails
`fa_kv_ok` and stays per-token, and under KVarN the batched path is gated off
because `forward_prefill_batch` runs its own batched attention and never
populates the KVarN window/records. Default prefill is 14.4 tok/s (kvarn) /
15.0 (fp32).

So the ordering question is real: **fixing KVarN batched prefill is worth ~14x
on the default path; this megakernel is worth up to ~1.35x on a path that is
currently opt-in.** Scope the megakernel, but do not schedule it ahead of the
KVarN batched-prefill faithfulness bug.

## Already measured dead — do not re-run

| approach | result |
|---|---|
| epilogue fusion, b-major LDS (attempt #1) | -9% vs separate pass |
| shared overlay positions across rows | 5.4% capture (13.5% ceiling unrotated) |
| low-rank residual `A·B^T` | -1.5% vs the overlay's -41.8% |
| G=64 grouping, no overlay | 58% worse |
| column concentration of outliers | top-16 columns hold ~8% vs 6.25% uniform |
| bitplanes | no int1/int2 WMMA on gfx1151 |
| QAT toward bounded support | FWHT Gaussianizes by CLT (kurtosis 2.97, max/sigma 3.04 vs 2.89 expected); and plain oq4 at 4.0 bits is under the 4.25 floor |
| megakernel for launch overhead | 99.2% GPU busy — at most 0.8% recoverable |
