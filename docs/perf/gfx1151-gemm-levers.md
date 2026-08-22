# gfx1151 prefill levers — a living registry

Every lever tried on the Qwen3.8-27B oq4.25++ prefill path, with its measured
effect and what would overturn it. **Several entries here reversed once measured
properly** — treat "confidence" as load-bearing, and revise rather than delete.

State box: halo, Strix Halo gfx1151, 40 CU / 20 WGP, ~2.9 GHz, 1536 VGPRs/SIMD
(granule 24 wave32 / 12 wave64), LDS 64 banks x 4 B, 64 kB visible per workgroup,
248.5 GB/s measured DRAM, 32 MB MALL. iu4 WMMA = 16 cyc, iu8/bf16/f16 = 32 cyc.

Session arc: prefill **15.4 -> 234 tok/s**. Uncorrected ceiling 324. Target 350.

---

## A. Shipped wins

| lever | effect | confidence | revise if |
|---|---|---|---|
| KVarN attention re-parallelization | 288.4 -> 66.9 us (4.3x) | high, repeated | — |
| lm_head skip for non-final prompt tokens | prefill +7% | high | — |
| K-major overlay correction (dword gather, table once per row/group) | 1.5-3.0x over b-major, **bit-identical** | high | — |
| iu4x2 consumes int8 directly, splits digits in-kernel | free (same bytes: K/2+K/2==K) | high, ISA-checked | — |
| Route compact batched GEMM -> iu4x2 | +14.6% | high | — |
| ...and the `grouped_prequant` wrapper too | +17.5% total | high | — |
| Wave64 two-pass compact GEMM | 1.26x kernel, +8.6% e2e | high | — |
| ubatch 512 (vs 256 default) | +6% | high | wider is worse: 1024 -> 188, 2048 -> 164 |
| TriAttention declined for non-evictable KV | fixes a GPU **panic**, unblocks long ctx | high | — |
| Tile b-range in the overlay correction | fixes a GPU **memory fault** above B=512 | high | — |

## B. Neutral — measured, not worth carrying

| lever | effect | why |
|---|---|---|
| Fuse overlay into the GEMM epilogue (nb-outer) | **-9%** | WMMA lane mapping forces 192 LDS *byte* gathers/thread/group |
| Same, j-outer (record read once per row) | -6..14% | record redundancy was never the bottleneck; the gathers are |
| `dwordx4` staging in the probe | <2% | staging cost is the barrier, not load width |
| rung 5 `TWNt` 2->4 | 0% | instr/WMMA 22.2->19.1 but occupancy 12->8; exactly cancels |
| Hoisting LDS base pointers | 0% | compiler already did it |
| Larger prefill ubatch (1024, 2048) | -12%, -22% | — |

## C. Dead — refuted by measurement

| lever | verdict | evidence |
|---|---|---|
| Coarse scale (`kSegments=1`, Kairic's contract) | 1.62x H-weighted error | and does NOT compose with pooling: even P=128 recovers only 55% |
| oq8 (8-bit, no overlay) | **2.4x SLOWER** prefill | 700 vs 1656 tok/s on the 2B; wider formats need in-loop unpack |
| Dense bf16 (dequant-once, llama.cpp's large-batch path) | slower than compact | 1529 vs 1665 prefill, 4.6x worse decode |
| Row-shared tile promotion | recovers only 22% of the overlay | WMMA A-tile is [16 rows][16 K]: widening must be row-uniform |
| Megakernel to cut launch overhead | <=0.8% available | inference-region GPU busy is **99.2%** |
| Transplant: shrink workgroup tile (WMt 2->1) for occupancy | **1.2x slower** | halves A-fragment reuse; occupancy bought by discarding reuse loses |

## D. Open / promising

| lever | expected | status |
|---|---|---|
| ~~Finish the rung-5 tiled kernel~~ | **DONE, and it did NOT transfer** — see §E correction | correct (parity PASS) but 5-14% SLOWER than the shipping wave64 |
| Pooled post-WMMA correction, P=64 | matches overlay quality, removes the gather | quality proven; sharing width is the cost knob, kernel unwritten |
| Skip LDS entirely, fragments direct from global | removes the barrier | DRAM is only 4% of staged cost, so reuse is affordable |
| KVarN batched-prefill faithfulness | unlocks batched prefill by default (**~14x** on the default path) | residual is generic batched-vs-per-token, not KVarN |

---

## E. The first-principles ladder (the model everything else is judged against)

gfx1151, wave32, 8 chains, each rung adding ONE ingredient:

| rung | TOPS | that rung costs |
|---|---|---|
| pure issue (operands in registers) | 103.2 | — |
| + both operands from LDS (2.00/WMMA, ISA-verified) | 96.8 | 1.07x |
| + per-group fold (i32->f32, scale, reset) | 89.2 | 1.08x |
| + global staging, naive (2 barriers) | 73.4 | 1.22x |
| + global staging, double-buffered (1 barrier) | 88.0 | **1.01x** |
| + real GEMM tiling (wave/subtile/k-step addressing) | 84.5 | 1.04x |
| **shipping wave64 compact GEMM** | **48.2** | **1.75x** |

Every ingredient is cheap; the shipping kernel is 1.75x below their sum. The
difference is how the tile is split: same BM=64 x BN=128, but 8 wave32 waves at
114 VGPRs / 12 waves-per-SIMD versus 4 wave64 waves at 234 VGPRs / 3.

### ⚠ CORRECTION: the 84.5 TOPS ceiling did NOT transfer

The rung-5 probe was built into a real kernel (`gemm_oq_compact_iu4x2_tiled`,
wave32, BM=64 x BN=128, 8 waves as 2x4, 179 VGPRs, **8 waves/SIMD**, 20480 B
LDS). It is correct — parity PASS on all 5 shapes at max|rel| 1.36e-7, first try
— and it is **5-14% SLOWER** than the shipping wave64 kernel:

| shape | tiled TOPS | shipping w64 TOPS | probe |
|---|---|---|---|
| gate/up | 52.3 | 54.9 | 84.5 |
| down | 44.1 | 51.0 | 84.5 |
| qkv | 50.0 | 52.3 | 84.5 |
| wo | **40.0** | 37.0 | 84.5 |
| B=512 | 49.9 | 53.0 | 84.5 |

So the "1.75x available" claim was wrong, and so was the rule of thumb derived
from it. The probe reached 84.5 TOPS because it did **not** read a real weight
layout, load per-row f16 scales, write M x B outputs, or guard boundaries. Real
kernels — both of them, at 179 and 234 VGPRs, at 8 and 3 waves/SIMD — land at
~50 TOPS regardless. **The ladder is sound as a per-ingredient diagnostic and
unsound as an absolute ceiling.**

What survives: occupancy is NOT the dominant variable it appeared to be. Two
kernels differing 179 vs 234 VGPRs and 8 vs 3 waves/SIMD perform within 10% of
each other, and every attempt to trade registers for instructions or vice versa
(TWNt 2->4, base hoisting, scale-pointer hoisting, WMt 2->1) came back neutral or
worse. Something outside the register/occupancy/instruction-count axis is setting
the ~50 TOPS plateau, and it is not yet identified.

**Wave64's real property**, from the matrix calculator: `v_wmma_i32_16x16x16_iu4`
needs 4 GPRs for C/D in wave64 against 8 in wave32 (the 16x16 tile spreads over
64 lanes, not 32). Physical cost per tile is identical; the PER-LANE VGPR count
halves, and that is what hits the 256-VGPR wall. So wave64 buys 2x the tile
before spilling — but on this workload that headroom is worth more spent on wave
count than on blocking.

## F. Hardware facts that close off levers

- **No direct global->LDS DMA on gfx1151.** `__builtin_amdgcn_global_load_lds`
  errors with `needs target feature vmem-to-lds-load-insts`; compiles on gfx942.
  Zero `*_lds` load opcodes exist for RDNA. Staging must go global -> VGPR -> LDS.
- **SDMA cannot reach LDS.** SDMA does not appear in the RDNA3.5 shader ISA at
  all; it is a fabric copy engine, and LDS is not in that address space.
- **DS offsets**: single-address form has a 16-bit byte offset (64 kB reach);
  the 2addr form has two 8-bit scaled offsets, <=2040 B each (`rdna35:7094`).
- **iu4 is 2x iu8**, and bf16/f16 are the same 32 cyc as iu8. Two iu4 passes cost
  the same WMMA cycles as one iu8 pass — which is why a tile promoted to int8
  weights is cycle-neutral in a W4A8 two-pass kernel.
- **DPP cannot be dual-issued under VOPD** and costs an extra cycle
  (`rdna35:4048, 4102`); `ds_bpermute` uses the LDS pipe. Neither is a free
  substitute for LDS operand delivery — which is moot anyway, since supply is ~1%.

## G. Measurement traps that produced wrong answers here

Each of these caused a wrong conclusion in this session before being caught.

| trap | signature | guard |
|---|---|---|
| `hipfire eval` caches, key excludes the environment | identical A/B numbers, same `started_utc` | `--force` |
| `include_str!` bakes kernel source into the binary | ablation changes nothing | always `cargo build`; empty-kernel control |
| Hot-path `hipMalloc` | microbench and e2e disagree in *direction* | persistent scratch |
| Edit silently didn't apply (rustfmt had collapsed the line) | flat A/B against a strong microbench | confirm the new kernel in a **kernel trace** |
| Probe operands hoisted into registers | ISA shows 0 LDS reads | launder the offset through inline asm; count `ds_load` |
| Wrong ISA mnemonic (`ds_read` vs `ds_load`) | "zero LDS accesses" on gfx11 | grep `ds_load`/`ds_store` |
| `r'\\b'` in a Python f-string is a literal backslash | all instruction counts zero | single backslash in a raw string |
| Degenerate control (fp32 KV never batches) | control reads exactly 0.00e0 | verify the control arm actually exercises the path |
| Mean over a heavy tail | "10% regression" that is 1.0012x at p99 | compare distributions |
| Trace includes model load | GPU busy 52% "launch-bound" | restrict to the inference region |
