# Fusing the sparse overlay into the GEMM epilogue — implemented, measured, reverted

State box: halo, Strix Halo gfx1151, 128 GB UMA, RAM configured 8000 MT/s
(256 GB/s peak, 248.5 GB/s measured pure-read). Qwen3.8-27B, `OqPlusCompact`
(qt 36), G=256, n_out=3, stride 136 = exactly 4.25 bits/weight.

## What was built

`gemm_oq_compact_iu4x2_wmma.hip` — the 2-pass exact-W4A8 kernel — already stages
the entire 256-element activation group in LDS as two digit planes (`xh`, `xl`)
before running its WMMA passes. That makes it the natural fusion point: the
`x[g*G + idx]` an overlay entry needs is already resident, and the overlay shares
the group's `sw[j] * sx` scaling, so `val * x[idx]` folds straight into the i32
accumulator before the float rescale. The loader zeroes the bulk nibble under each
overlay position, so the WMMA contributes 0 there and the fused term is *exact*,
not a correction on top of an approximation.

Implemented, and it is correct: `parity_gemm_oq_compact_iu4x2` PASSes all 7 shapes
(max|rel| 1.55e-7) against a CPU oracle extended to include the overlay term.
Occupancy is preserved: **167 VGPRs, 9 waves/SIMD, 0 spills** (baseline 147 / 9 / 0).

## Result: 9% slower than keeping the pass separate. Reverted.

Medians of 3 runs, ms. "unfused" is the same kernel with `n_ov = 0` so the compiler
dead-codes the overlay loop; "corr" is the k-major standalone correction
(`oq_compact_overlay_correct_t`), including its activation transpose.

| shape | M | K | B | fused | unfused | + corr | total | fused vs |
|---|---|---|---|---|---|---|---|---|
| gate/up | 17408 | 5120 | 256 | 3.415 | 2.159 | 0.978 | 3.137 | **+8.9%** |
| down | 5120 | 17408 | 256 | 2.980 | 1.954 | 0.720 | 2.674 | **+11.4%** |
| qkv | 6144 | 5120 | 256 | 1.143 | 0.639 | 0.357 | 0.996 | **+14.8%** |
| wo | 5120 | 4096 | 256 | 0.674 | 0.437 | 0.285 | 0.722 | −6.6% |
| gate/up | 17408 | 5120 | 128 | 1.833 | 1.189 | 0.509 | 1.698 | **+8.0%** |
| gate/up | 17408 | 5120 | 512 | 6.757 | 4.141 | 2.199 | 6.340 | **+6.6%** |

Run-to-run spread inside each config is 3-6%, so a consistent +9% median across
5 of 6 shapes is outside the noise band. Both remain far ahead of the iu8 baseline
(iu8 + corr is 13.30 ms over the four B=256 shapes vs 7.53 ms for unfused + corr).

## Why fusion loses, structurally

The overlay is 3 entries per (row, group) — 1.2% of the arithmetic — but costs
40-60% of the GEMM's runtime. It is not redundant work, it is a **lane-mapping
mismatch**:

- In the WMMA kernel the matrix-core layout dictates the mapping: one lane owns
  one b-column (`b_col = nb*16 + lane`). An overlay lookup is therefore a *byte*
  gather from LDS, and each thread does 4 `nb` blocks x 8 rows x 3 entries x 2
  planes = **192 LDS byte-gathers per group**.
- The standalone k-major kernel is free to choose its mapping, and it picks one
  where each lane owns 4 consecutive b — so the same total gathers issue as
  **dword** loads, 4x fewer memory ops, plus it reads the 6-byte overlay record
  once per (row, group) into registers.

The fused kernel cannot do that second trick. Two attempts to cache the record
per-row both blew the register file, because `iacc_hi[4][8] + iacc_lo[4][8] +
facc[4][8]` already own ~96 VGPRs:

| variant | VGPRs | waves/SIMD | spills |
|---|---|---|---|
| baseline (no overlay) | 147 | 9 | 0 |
| inline read, dynamic trip count | **167** | **9** | **0** |
| register cache, MAXOV=4 | 256 | 5 | 152 |
| register cache, MAXOV=3 | 256 | 5 | 86 |
| inline read, compile-time unrolled trip count | 256 | 5 | 277 |

Unrolling the gather across the already-unrolled nb/j loops is the worst of all —
the dynamic loop is doing useful work by serializing and keeping pressure low.

`wo` is the one shape fusion wins, and the reason is consistent with the above:
K=4096 is the smallest weight volume here, so the separate pass's *fixed* costs
(second kernel launch, activation transpose, and the M x B f32 read-modify-write
of `y`) are largest relative to the GEMM. 6.6% on the smallest projection does not
justify maintaining a second kernel path.

## Why training toward bounded support is not the escape hatch

The tempting alternative is to make the overlay unnecessary: QAT the model so no
weight group has values a single int4 scale cannot hold, then ship plain oq4.
Four reasons that does not work here, in descending order of decisiveness.

**1. The rotation Gaussianizes by construction, so post-rotation support is not
a trainable property.** The quantizer applies a Hadamard/FWHT rotation before
grouping. Each rotated coordinate is a signed sum of 256 weighted inputs, so by
the CLT its distribution is Gaussian almost regardless of what the pre-rotation
weights look like. Whatever bounded distribution you train, the rotation maps it
back to ~Gaussian. Measured on the actual 27B rotated groups: **kurtosis 2.97**
(Gaussian = 3.00) and **max|w|/sigma = 3.04** (expected max of 256 Gaussian draws
= 2.89). The groups are Gaussian and very slightly *platykurtic* — flatter-tailed
than normal. There is no heavy tail to train away. To keep bounded support you
would have to drop the rotation, and unrotated is measurably worse (the
shared-position study put the unrotated ceiling at 13.5% capture).

**2. "Outliers" here are order statistics, not a pathology.** The top-3 of 256
Gaussian samples are large because 256 samples of a Gaussian have a top-3.
Suppressing them is not removing a defect; it is demanding a specific non-Gaussian
shape. For a *uniform* distribution — the one int4-with-one-scale is matched to —
max/sigma = sqrt(3) = 1.73. Getting from 3.04 to 1.73 is a 1.76x dynamic-range
compression at fixed variance, imposed on 23.8 B parameters. That is a
distributional constraint, not a regularizer nudge.

**3. The gradient is the wrong shape.** A bounded-support penalty is a max over
each group, so its gradient touches exactly one weight per group per step, while
the task loss it is fighting touches all 256. Softened variants trade that for a
weaker constraint. Either way the penalty is fighting the loss at a ~256x
disadvantage per step across ~93 M groups.

**4. It does not buy anything the goal is allowed to spend.** The hard floor is
4.25 bits/weight. Plain oq4 is 4.0, so "train out the outliers and drop the
overlay" does not clear the floor — it lands under it with a worse model. The
honest framing is the reverse: **the overlay is the best measured way to spend the
0.25 bits above plain int4**, and every alternative use of those bits has now been
measured worse — shared positions capture 5.4%, low-rank residual −1.5% vs the
overlay's −41.8%, G=64 is 58% worse, column concentration puts the top-16 columns
at ~8% against 6.25% uniform, and bitplanes have no int1/int2 WMMA to run on.

Even granting all of the above, the payoff would be ~0.25 bits = 5.9% of weight
bytes = ~0.9 GB. Decode is already at 90% of the 248.5 GB/s ceiling, so that is
~6% tok/s — for a full QAT run over 23.8 B parameters on one APU, against kernel
work that is already measured and in hand.

## Standing conclusion

Keep the overlay correction as a separate k-major pass. It is bit-identical
(max|diff| 0.00e0), 1.5-3.0x faster than the shipped b-major version, and 9%
faster than fusing it. The fusion is implemented and correct and is preserved in
this note's history if the lane mapping ever changes; nothing in the tree depends
on it.
