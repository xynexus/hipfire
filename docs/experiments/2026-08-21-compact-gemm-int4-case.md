# The case for a compact iu4 GEMM — and why tiling comes first

**Box:** halo, gfx1151. All numbers measured 2026-08-21.

## Hardware premise: CONFIRMED

`examples/probe_gfx1151_iu4_wmma`:

    IU4 median 0.1731 ms   99,237 GOPS
    IU8 median 0.3248 ms   52,900 GOPS
    ratio 1.876x

So int8 peak is **52.9 TOPS** and int4 peak is **99.2 TOPS** on this part. The
`reference_gfx1151_int4_isa_rate` note (2.0x) holds to within measurement.

## Where the compact prefill GEMM actually is

`gemm_oq_compact_grouped_wmma`, gate/up M=17408 K=5120 B=256
(`examples/bench_oq_compact_gemm`):

| variant | TOPS | % of 52.9 |
|---|---|---|
| current (overlay cap 16) | 10.70 | 20% |
| − nibble decode | 10.29* | +6% |
| − overlay scan | ~10.9 | +12% |
| − activation loads | 16.56 | **+70%** |
| − all three | **32.31** | **61%** |

(*measured from the cap-32 baseline of 9.72.)

Two conclusions:

1. **The activation loads are the dominant single term.** The cause is a tiling
   imbalance: one workgroup covers 16 rows x 128 columns, so it reads `128*K`
   bytes of X against `8*K` bytes of weights — 16:1. With M/16 = 1088 row-tiles,
   X is re-read ~1088 times: ~1.4 GB of X traffic against 45 MB of weights.
   LDS staging does NOT fix this (each X element is used once per workgroup);
   more ROWS per workgroup does.
2. **The kernel's structural ceiling is ~32 TOPS**, not 52.9. Even with all data
   movement free, the WMMA issue rate plus weight loads and output writes cap it
   at 61% of int8 peak.

## What int4 buys, honestly

It is NOT primarily the MAC rate. We use 20% of the int8 MAC peak, so doubling
that peak changes nothing on its own. int4 helps for two OTHER reasons:

1. **int4 activations HALVE the activation traffic** — which is the measured
   bottleneck. This is the real win and it attacks the right term.
2. **Compact bulk weights are ALREADY signed int4 nibbles.** Feeding them to
   `wmma_i32_16x16x16_iu4_w32` raw removes the decode entirely (worth ~6%) and
   costs nothing to produce — the format already stores exactly what iu4 wants.

Only once tiling is fixed and the kernel approaches its ~32 TOPS ceiling does the
iu4 MAC rate start to matter, at which point it raises that ceiling toward ~60.

## The outlier structure falls out symmetrically

Compact already stores `bulk int4 + sparse int8 overlay` for WEIGHTS, and the
loader zeroes the bulk nibble under each overlay, so a weight correction is
exactly `val * x[idx]` — a sparse rank-few update, already implemented this way
in the decode GEMV.

Doing the same on the ACTIVATION side gives the symmetric design: bulk
activations in int4, plus a sparse int8 pass for the few outlier channels. The
FWHT rotation compact already applies is what makes this viable — suppressing
activation outliers is precisely what that rotation is for.

## PROOF that tiling dominates the MAC rate

`gemm_oq4_grouped_wmma` is an EXISTING iu4 x iu4 W4A4 GEMM. Benched at the same
shapes (`examples/bench_w4a4_vs_compact`):

    gate/up  1.37 TOPS      down 2.08      qkv 4.56      wo 3.85

That is **7.8x SLOWER** than the iu8 compact GEMM's 10.70, despite iu4 being
1.876x faster at the ISA. Its grid is `[M/16, B/16]` with no N-blocking, so every
b-tile re-reads the whole weight matrix — the same redundancy `OQC_NB` was added
to fix. Tiling beats the MAC rate by ~8x here.

**So do not start from this kernel**, and do not expect iu4 alone to double
anything.

## Recommended order

1. **Tiling**: more rows per workgroup, to amortize the X reads. Worth up to 3x
   (10.70 -> 32.31) and helps every dtype.
2. **int4 activations**: halves the dominant traffic term, and lets the bulk
   nibbles feed iu4 raw with no decode.
3. **Sparse outlier passes** on both sides, mirroring the existing weight overlay.
4. Only then does raising the MAC ceiling from 52.9 to 99.2 become the binding
   constraint.

---

# Follow-up: where the iu4 kernel's remaining headroom is

Built and measured (`gemm_oq_compact_iu4_wmma`, B=256). Ablations, TOPS:

| variant | gate/up | down | qkv | wo |
|---|---|---|---|---|
| as shipped | 20.58 | 18.04 | 21.83 | 25.68 |
| activation loads removed | 39.73 | 56.81 | 50.46 | 50.47 |
| activation + weight loads removed | 69.95 | 82.07 | 57.32 | 61.90 |

Read against the **iu4** peak of 99.2 TOPS (not the 52.9 int8 peak — that one no
longer applies once the MACs are int4):

- The WMMA issue path reaches **70-83% of peak**, so the instruction side is
  healthy and is NOT what to work on.
- The kernel is **still activation-bound**: removing the activation loads is
  worth 2-3x. Same lever as the iu8 twin, just at a higher absolute level.
- Weight loads cost a further ~1.5x.

**The tiling parameter is already at its optimum.** OQC4_MW swept 8/16/32:

    MW=8    19.85  12.69  19.82   5.00
    MW=16   20.88  18.43  21.86  25.57
    MW=32   15.13  15.57  19.28   5.95

So there is no cheap parameter win left. The remaining 2-3x needs a STRUCTURAL
change to how activations are fed — LDS staging of the activation tile is the
obvious candidate, since with OQC4_MW row-tiles per workgroup the activation
columns are genuinely reused across waves.

## Sequencing note

That structural work is best done AFTER the activation-outlier design is settled,
not before: the outlier scheme decides whether the dense int4 plane stays the
only activation input, and it is the one part of W4A4 with real quality risk. A
kernel tuned against the wrong activation layout gets tuned twice.

---

# Why performance is "still bad" — measured with hardware counters

First, scope it. Decode is NOT the problem: 15.0 tok/s at 232 GB/s is **93% of the
248.5 GB/s DRAM ceiling** — hardware-limited, nothing left. Prefill at 186 tok/s
against a ~980 compute ceiling is **19% of peak**, and prefill is 85% one kernel.
So "performance is bad" means "the compact GEMM is at ~20-30% of peak".

Ablations answered *where* the time goes but not *why*. `rocprofv3 --pmc` does.
gfx1151 exposes a thin counter set; the useful ones here were SQ_INSTS_VALU,
SQ_INSTS_LDS, SQC_LDS_BANK_CONFLICT and SQC_LDS_IDX_ACTIVE.

| kernel | VALU | LDS | bank conflicts |
|---|---|---|---|
| iu8 | 15.89 G | 0 | — |
| iu4 (start) | 2.12 G | 0.33 G | **0** |
| iu4x2 | 3.13 G | 0.64 G | **0** |

Three findings:

1. **Zero LDS bank conflicts.** The 132-byte column padding works — confirmed,
   not assumed. Do not spend time there.
2. **iu8 issues 7.5x the VALU of iu4** — the nibble decode and overlay scan it
   pays and iu4 does not. That, not memory, is why the iu8 path sits at ~25%.
3. **The iu4 kernel was issuing 5.8 non-WMMA VALU per WMMA.** Against 312M WMMA
   instructions in the bench, the *index arithmetic was the workload* and the
   matrix core was idle waiting for it.

## Two fixes, both from the counter, both measured

**Integer division in the staging loop.** `i / dwords_per_col` with a runtime
divisor expands to ~20 VALU on AMD, executed once per staged dword.
`dwords_per_col` is `group/8`, always 16 or 32 — a power of two — so it is a
shift and a mask. VALU 2.12G -> 1.82G, ratio 5.8 -> 4.8.

Worth noting the shape-dependence, because it is the tell: `down` (K=17408, 68
groups) gained **+16%** while gate/up (20 groups) barely moved. The staging loop
runs once per group, so the shape with 3.4x the groups shows 3.4x the benefit.

**Runtime `tiles_per_group` and `lds_stride`.** Specialising the G=256 case makes
both compile-time, so the k-loop fully unrolls and every LDS offset becomes an
immediate rather than an add per (k-tile, b-tile). VALU 1.82G -> 1.47G, ratio
4.8 -> **3.7**.

Measured, medians of 3 (single runs mislead here — one `down` sample read -8%
when the median was +8%):

    gate/up  29.60 -> 31.66 TOPS      down  27.11 -> 29.31
    qkv      30.89 -> 33.59           wo    30.04 -> 33.43

## Where the compact iu4 GEMM now stands

    20.58 TOPS  1-pass, no LDS staging
    28.76       + LDS-staged activations
    29.60       + staging division removed
    31.7-33.6   + G=256 specialisation        <- ~33% of the 99.2 TOPS iu4 peak

against ~13 TOPS for the iu8 path it replaces, i.e. **~2.4x**.

## What is left, and it is still VALU

3.7 non-WMMA VALU per WMMA remain, and the pure-WMMA ablation ceiling was ~70
TOPS, so roughly 2x is still on the table. The residue is the per-group rescale
epilogue: 8 b-tiles x 8 accumulator rows x (mul, mul, add) per group, which is
~220 VALU against 128 WMMA and does not shrink with any tiling choice, because
both terms scale together.

The one structural escape is doing MORE WMMA per epilogue. That is exactly what
the 2-pass kernel does — its ratio is **3.6 against the 1-pass kernel's 4.8 at
the same point**, because the epilogue amortises over twice the matrix work. So
the 8-bit path is not merely affordable, it is *more efficient per instruction*
than the 4-bit one.

---

# Why the compact W4A4 kernel is still ~2x short: it is the wrong SHAPE

After the counter-driven fixes the compact iu4 GEMM sits at 32.9 / 27.2 / 34.2 /
35.0 TOPS (gate-up / down / qkv / wo), i.e. ~33% of the 99.2 TOPS peak. Three
successive instruction-level attempts then returned NOTHING:

  * rewriting the rescale epilogue as vector expressions to coax packed FP32:
    VALU 1.47G -> 1.49G, throughput unchanged. The compiler already emits it.
  * hoisting the 64-bit scale-pointer multiplies out of the group loop: VALU
    1.47G -> 1.39G (ratio 3.7 -> 3.4) and throughput unchanged — a 5% instruction
    cut buying 0%, which is what says VALU issue is no longer binding.
  * batching all NB LDS b-operand loads ahead of the WMMAs: unchanged. The
    unrolled loop was already scheduled that way.

Three nulls in a row is the signal to stop tuning and check the SHAPE.

**This repo already contains a tuned iu4 GEMM that is ~2x faster at these exact
shapes.** `gemm_iu4_i32_wmma_lds`, measured here today:

| shape | compact iu4 (mine) | gemm_iu4_i32_wmma_lds | ratio |
|---|---|---|---|
| gate/up 17408x5120 | 32.9 | **62.8** (63% of peak) | 1.9x |
| down 5120x17408 | 27.2 | **49.9** (50%) | 1.8x |
| qkv 6144x5120 | 34.2 | **69.0** (70%) | 2.0x |
| wo 5120x4096 | 35.0 | 30.8 | **0.88x** |

Its recorded tuning arc explains the gap exactly, and my kernel lands precisely
where that arc says a wave32 double-buffered design lands:

    single-chain                        ~3.5k GOP/s
    wave32 LDS + double-buffer 2x8      22.6k     <- the design I have
    + wave64                            30.6k     (+35%)
    + ds_load_b128 fragment reads       31.5k
    + BK=64 K-strip, N-heavy BM64/BN256 49.7k

So the missing levers are **wave64** (via the `// HIPFIRE_COMPILER_FLAGS:
-mwavefrontsize64` source magic comment and `v_wmma_i32_16x16x16_iu4_w64`),
**BK=64**, **N-heavy tiling** (BM=64, BN=256 — the opposite of my BM=256/BN=128),
and **b128 fragment reads**. None is a tweak; each changes the lane mapping or the
tile.

Note `wo` INVERTS: mine is 14% faster there. BN=256 gives only one N-tile at
B=256, and BM=64 over M=5120 is just 80 workgroups — the N-heavy shape starves on
small M. So this is not "replace one with the other", it is a shape-dependent
routing decision.

## What the port actually involves

`gemm_iu4_i32_wmma_lds` is the pure integer core: dense int4 A, dense int4 X,
i32 out, no scales. Compact's BULK NIBBLES ARE dense int4 — the same bytes. So
the port is narrow rather than a rewrite:

1. address A from the compact split-plane nibble region instead of a dense plane;
2. add the per-group f16 weight scale and per-(token, group) activation scale to
   its epilogue;
3. leave the sparse overlay where it is — a separate pass either way.

That is the work worth doing, and it is worth ~2x on the three shapes that carry
the FFN. Hill-climbing the wave32 design further is not.
