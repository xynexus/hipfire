# R6 — where is the XDNA1 cascade GEMM actually bound? (compute-per-sync + INNER)

R5 shipped a working streaming K-cascade W4A8 GEMM on XDNA1 (gfx1103) but it topped
out at ~0.56 TOPS, and R5's discriminators *read* it as per-objectfifo-sync bound.
R6 tests that directly and **corrects it**: the wall is neither sync nor feed — it is
the **mmul MAC chain itself, running ~15× off peak (un-pipelined)**. That is a kernel
bug, not a hardware ceiling, so there is real headroom.

Reuses the R5 rig unchanged (`../r5/`): the kernel is `-DKSLICE`/`-DINNER`
parametrized and the generator derives buffer sizes from KSLICE. `r6_intensity.sh`
sweeps KSLICE; the INNER probe is two hand-built points.

## 1. Compute-per-sync sweep (`r6_intensity.sh`, 4×4, N_BTILES=1024)

KSLICE = mmuls each core contracts per output tile. Raising it scales BOTH compute
and feed per objectfifo iteration while the fixed per-iteration sync stays constant.

| KSLICE | per-dispatch | µs/tile | TOPS |
|---|---|---|---|
| 16  | 645 µs  | 0.630 | 0.42 |
| 32  | 1256 µs | 1.227 | 0.43 |
| 64  | 2128 µs | 2.078 | 0.50 |
| 128 | 4074 µs | 3.979 | 0.53 |

TOPS barely moves (0.42→0.53) and per-tile time is ~linear in KSLICE:
**µs/tile ≈ 0.152 + 0.030·KSLICE**. The fixed sync (0.152 µs) is only ~24% even at
KSLICE=16 — so R5 was **not** sync-bound. The dominant term is **~30 ns per
mmul-step**, and it scales with the work. (L1 caps the sweep at KSLICE≤128 on aie2:
double-buffered A+W = 256·KSLICE B/core → 32 KB at 128.)

## 2. INNER probe — feed-bound or core-bound? (KSLICE=16, N_BTILES=1024, 4×4)

INNER recomputes the K-slice INNER times over the **same resident L1 tiles** — MACs
scale, feed does not. If feed-bound: time flat, TOPS ×INNER. If core-bound: time ×INNER.

| INNER | per-dispatch | MACs/disp | TOPS | c0 |
|---|---|---|---|---|
| 1 | 630 µs  | 1.34e8 | 0.43 | 1024 |
| 8 | 2956 µs | 1.07e9 | 0.73 | 8192 |

8× the compute over fixed feed → time grew **4.7×**, not flat. So it is **core-bound**,
not feed-bound. The marginal cost of the extra (feed-free) passes is
(2956−630)/7/1024 = **0.324 µs/tile = ~20 ns/mmul of pure compute**. Combined with the
KSLICE slope (~30 ns/mmul incl. feed) the decomposition is **~20 ns compute + ~10 ns
feed per mmul-step**, plus ~0.15 µs/tile fixed sync that amortizes.

## Verdict — un-pipelined MAC chain, ~15× off peak (fixable)

~20 ns/mmul @ 1.8 GHz = **~36 cycles per `mmul<4,16,8>`**. Per-core INT8 peak is
~2–3 cycles/mmul, so the core runs **~15× slow**. The kernel accumulates a whole
K-slice into a single `MMUL c` — a serial dependency chain that never reaches II≈1,
so the mmul latency is fully exposed. Feed is a real but secondary term (~1/3).

This overturns the R5 "sync-bound, K-cascade is the wrong axis" read: the cascade and
the array geometry are fine; the **microkernel** is the wall. Corollaries:
- R5's cascade C-residency and R6's bigger K-tiles can't help while each mmul is 15×
  slow — the compute term dominates everything.
- **R6's guess:** ~10–15× of headroom if the MAC chain pipelines. **(R7 disproves this.)**

## R7 — multi-accumulator: hypothesis DISPROVEN, ~1.35× consolation win

R6 hypothesized the ~36 cyc/mmul was a single-accumulator dependency stall. R7 split
the K contraction across NACC independent accumulators (`-DNACC`, tree-summed;
`../r5/r5_cascade.cc`) and swept it. First confirmed the shape is **native, not
emulated**: aie2 int8×int4 `<4,16,8>` lowers to `::mac_4x16_16x8_conf` (a real
intrinsic; `emulated_mmul_intrinsics.hpp` is unused for it).

Throughput (4×4, feed + MACs held fixed; NACC = internal pipelining only):

| NACC | KSLICE=16 TOPS | KSLICE=128 TOPS |
|---|---|---|
| 1 | 0.42 | 0.46 |
| 2 | 0.44 | 0.53 |
| 4 | **0.49** | **0.62** |
| 8 | 0.56* | 0.40 (spills) |

The decisive test is the **feed-free marginal rate** (INNER probe, isolates pure
compute): NACC=1 → **20.3 ns/mmul**, NACC=8 → **18.0 ns/mmul**. Essentially unchanged.
So the mmul is **not** accumulator-latency-stalled — `mac_4x16_16x8_conf` simply runs
**~32 cycles/mmul on AIE-ML gen1 (Phoenix)**, and that op throughput *is* the wall.
Multi-accumulator helps only by overlapping the loads/sync around the MACs (~1.35×),
and NACC=8 spills the accumulator register file (KSLICE=128 regresses to 0.40).

*The KSLICE=16 NACC=8 "0.56" is inflated by fill/drain at a tiny K; the compute-heavy
KSLICE=128 column is the honest read, where NACC=4 (0.62) is the peak and 8 spills.

**Verdict.** NACC=4 is the new default — a real, correctness-preserving ~1.35× (best
~0.62 TOPS). But the ~0.6-TOPS ceiling is the int8×int4 mmul-op throughput on Phoenix,
**not** a dataflow, sync, feed, or pipelining bug — consistent with the ~12%-of-peak
XDNA1 int8 GEMM seen before. Closing the rest is not an mlir-aie kernel-structure
problem: it needs a faster op (does int8×int8 `<4,16,8>`/`<8,16,8>` clock fewer cycles?
does aie2p/Strix run the same op faster?), which is the only remaining probe worth
running before declaring XDNA1 W4A8 GEMM a hard ~0.6-TOPS floor.

Repro: `R5_ARCH=aie2 ./r6_intensity.sh 4 4 1024 300 16 32 64 128`; NACC/INNER via
`R5_NACC=4 R5_INNER=1 ../r5/r5_build.sh <mlir> <wd> <KSLICE>`.

> **R8 correction:** the "op-throughput floor" verdict above is WRONG. The INNER probe
> reloads `ldA`+`ldW` every mac, so it measured ~2 L1 loads/mmul, not the op. A resident
> microbench (`../r8/`) clocks the same int4 `<4,16,8>` at **~1 ns/mmul (~489 GMAC/s/core,
> ~15 TOPS array)** — ~25× above the streaming result. XDNA1 GEMM is **load-bound**
> (no data reuse), not op-bound. NACC's weak ~1.35× is because it can't cut loads/mac.
> The real fix is load-reuse register tiling (R9), not anything in R5–R7.

## AIE2P EmbeddingGemma full-K submission (2026-07-10)

`r6_gen_mp_fullk.py` and `r6_gen_mp_fullk_mixed.py` stream every K=256
group and N slab through one AIE runtime sequence. `r6_fullk_cache.sh` builds
W4, W8, or mixed caches under `~/.hipfire/npu`; the EmbeddingGemma inventory
wrapper builds padded K=768/1280/3072 and N=256/768/1152/3072 shapes.

Correctness on halo/gfx1151:

- W4, arbitrary mixed W4+dense-W8 residual, and W8 return exact int32 group
  partials for K-group counts 3, 5, and 12.
- The Opus executor matches its CPU integer/scaling oracle exactly, including
  AWQ sidecars and variable mixed overlay counts.
- K=1152 uses five groups after zero-padding the pre-rotation activation to
  K=1280; W4, mixed, and W8 all passed exact hardware parity.
- Groups 3 and 5 use direct group-major output DMA. The shim task queue cannot
  hold twelve per-group output tasks, so K=3072 records `physical` in
  `output-layout.txt` and uses the correctness-preserving host transpose.

Mixed residual experiments narrowed two dead ends:

- A scalar sparse-overlay AIE kernel was exact but took 5.76 ms for
  M=256/K=768/N=768. Densifying the arbitrary residual into W8 and accumulating
  it into the W4 int32 tile on AIE reduced submission time to about 0.53 ms.
- Fusing BF16 group scaling into the AIE core produced corrupt/NaN output. DMA
  and constant BF16 stores were correct; the failure isolated to the current
  Peano AIE2P acc32/int32-to-float lane conversion path. Group partials remain
  int32 and scaling stays in the caller until that conversion is proven.

The production seam includes activation packing, one dispatch, and output
layout reconstruction, but excludes FWHT/quantization, group scaling,
attention, norms, pooling, and Dense activation. At M=256 its weighted
EmbeddingGemma projection inventory was:

| mode | weighted hybrid projection ms | aggregate logical TOPS |
|---|---:|---:|
| W4 | 73.083 | 0.744 |
| mixed | 131.275 | 0.414 |
| W8 | 108.765 | 0.500 |

These are hybrid projection measurements, not model throughput. The local
OQ4++ hybrid model control and full-K path produced identical embeddings
(minimum GPU cosine 0.98601860), so full-K preserved the legacy NPU result, but
the existing 0.999 GPU-parity gate still failed. The measured full-K hybrid was
130.9 input tok/s at 23.19 W package power, or 5.6 package tok/J. The remaining
10k tok/s objective requires projection fusion plus resident attention/norm/
residual/pooling/Dense execution; it is not satisfied by this slice.
