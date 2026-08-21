# What is blocking each performance number — measured, ranked

Halo, gfx1151, Qwen3.8-27B oq4.25++. Every figure below was measured in this
session; inferences are marked as such.

## Ceilings, so "blocked" means something

    DRAM pure-read (measured)      248.5 GB/s
    int8 WMMA peak (probe)          52.9 TOPS
    int4 WMMA peak (probe)          99.2 TOPS
    27B weights at 4.25 bits         14.3 GB   -> 17.4 tok/s decode at the DRAM ceiling
    27B forward                      54 GFLOP/token -> ~980 tok/s prefill at int8 peak

---

## 1. DECODE — 15.0 tok/s @ 232 GB/s = 93% of the DRAM ceiling. NOT SOFTWARE-BLOCKED.

There is no software blocker left. Decode reads 14.3 GB of weights per token and
achieves 93% of what the memory system can deliver. FETCH_SIZE showed 1.05x
overfetch, so it is not reading anything it does not need.

The only levers are outside software:

* **BLOCKER: DRAM clock.** `dmidecode` reports memory configured at **8000 MT/s
  against modules rated 8532** — ~6.6% of headroom sitting in firmware.
* **BLOCKER: the 4.25-bit floor.** Decode time is bytes/bandwidth, so the only
  large lever is fewer bits, and 4.25 is a hard constraint on this work.

Anything that claims a large decode win without changing one of those two is
measuring something else.

---

## 2. PREFILL — 186 tok/s vs a ~980 ceiling = 19%. THE BIGGEST ADDRESSABLE GAP.

Prefill is **85.1% one kernel**: `gemm_oq_compact_grouped_wmma`, the **iu8**
compact GEMM, measured at **13.0-13.8 TOPS = 25% of the int8 peak**.

Three nested blockers, in order of size:

* **BLOCKER 2a: prefill runs the iu8 kernel, not the iu4 one.** The compact W4A4
  kernel now measures **53.4 TOPS** on the same shape — **4x** the iu8 path. By
  Amdahl on an 85% share, 4x on that kernel is ~2.8x overall: **186 -> ~510
  tok/s** (inferred, not measured end-to-end).
* **BLOCKER 2b: the iu8 kernel never received the optimisations the iu4 one did.**
  It has no LDS staging, a runtime staging division, no G=256 specialisation and
  no double buffering. Those took the iu4 kernel 20.6 -> 33 TOPS (1.6x). It also
  carries **15.89 G VALU, 7.5x the iu4 kernel** — it must widen int4 to int8,
  which iu4 does not. Estimated 13 -> ~20 TOPS, i.e. prefill ~186 -> ~280.
  This is INDEPENDENT of W4A4 and helps today.
* **BLOCKER 2c: the W4A4 path cannot be switched on yet.** See section 4.

---

## 3. SPEC-DECODE — 5-12 tok/s by task, still under plain decode's 15.1.

* **BLOCKER 3a: the sparse overlay correction.** Required for correctness — the
  overlay IS the `++` in oq4.25++ — and it costs **90-203% of the GEMM it
  corrects**, erasing the entire W4A4 win (gate/up nets 0.80x). THREE layouts
  were built and measured; the best is the least bad:

      lanes over b-columns (shipped)             90-203%   net 0.80-1.45x
      lanes over rows, b split across grid.y    450-930%   net 0.22-0.57x
      lanes over b + activation staged in LDS   340-460%   net 0.38-0.60x

  At ~1.2% of the GEMM's MACs this should be nearly free; all three are ~100x off
  that. It is a memory-layout problem and it is UNSOLVED.
* **BLOCKER 3b: acceptance is task-dependent, 2.19-5.00.** On structured output
  the drafter is already AT spec (translate 5.00, json 4.88 vs the paper's 4.80
  mean); prose is 2.19. Not a defect — the drafter has less to predict.

---

## 4. THE W4A4 KERNEL — 53.4 TOPS = 54% of peak, and NOT WIRED IN.

* **BLOCKER 4a: the per-group rescale.** Against the pure integer core at the
  same shapes (62.8 / 49.9 / 69.0) the compact kernel reaches **85% / 85% / 75%**,
  so the rescale costs **15-25%**. Largest single item inside the kernel.
* **BLOCKER 4b: no int4 activation quantizer exists.** The kernel takes packed
  int4 activations; nothing produces them on the serving path.
* **BLOCKER 4c: activation precision.** Per-(token, group) crest is 4.86 (bulk) /
  5.89 (down_proj) / 8.35 (out_proj), so int4 leaves **0.84-1.4 rms levels** —
  marginal EVERYWHERE, not locally. The 2-pass 8-bit path answers this (1.03-1.45x
  the 1-pass, still 1.55-2.58x over iu8) but is likewise unwired.
* **BLOCKER 4d: wave64 portability.** AGENTS.md requires RDNA2/3/4. The kernel is
  opted in per-file by a compiler-flag comment; it needs a capability predicate
  before it can be a default.
* **BLOCKER 4e: shape routing.** `wo` INVERTS — wave32 is 1.2x faster there, since
  BN=256 leaves one N-tile at B=256 and BM=64 over M=5120 is 80 workgroups. Both
  kernels must be kept and selected on shape.

---

## 5. WHAT IS *NOT* A BLOCKER — ruled out with evidence, do not re-spend time

* **LDS bank conflicts.** `SQC_LDS_BANK_CONFLICT` = **0**. The 132-byte padding
  works.
* **VALU instruction count, in the iu4 kernel.** Cutting it 5% (1.47G -> 1.39G,
  ratio 3.7 -> 3.4) moved throughput **0%**. It stopped being the constraint.
* **Packed FP32 in the epilogue.** Vector-expression rewrite: VALU 1.47 -> 1.49G,
  throughput unchanged. The compiler already emits it.
* **Batching the LDS loads.** Unchanged — the unrolled loop was already scheduled
  that way.
* **The scattered overlay gather in the decode GEMV.** Ablating it measured 59.6
  vs 59.3 — free, Xq stays in L1.
* **Routing small-batch compact around `multicol`.** WMMA at B=8 measured 6.72 vs
  multicol's 11.84.
* **NPU offload for the drafter.** NPU and GPU deliver the SAME ~220 GMAC/s at
  drafter shapes; it is lateral, and the r14 kernels are not built for aie2p.
* **MTP instead of DFlash2.** DFlash2 wins on this model per byte (4.80 vs 4.28
  accepted, single-pass vs n sequential).

---

## Ranked by expected value

1. **Wire iu4/W4A4 into prefill** (4b, 4c, 4d, 4e then 2a) — inferred ~2.8x on
   prefill. Largest by a wide margin, and gated on integration, not kernels.
2. **Give the iu8 kernel the iu4 treatment** (2b) — ~1.5x on prefill, independent,
   helps today, no format or precision risk.
3. **Solve the overlay correction** (3a) — unblocks spec-decode entirely and is
   also required by 4a's path to serving.
4. **The per-group rescale** (4a) — 15-25% inside an already-fast kernel.
5. **DRAM clock** (1) — ~6.6% on decode, firmware, no code.
