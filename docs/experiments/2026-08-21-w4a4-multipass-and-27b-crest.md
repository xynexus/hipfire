# Multi-pass iu4 activations, and the 27B activation shape

Two measurements, both on halo (gfx1151), 2026-08-21.

## 1. 8-bit activations as TWO iu4 passes — measured, and better than modelled

The contraction is linear, so an activation carries as radix-16 digits:
`x = 16*x_hi + x_lo`, `x_hi` signed int4, `x_lo` unsigned int4, together spanning
int8 exactly. `wmma_i32_16x16x16_iu4_w32` takes per-operand signedness flags, so
both digits feed the same instruction, and `16*iacc_hi + iacc_lo` recombines in
i32 before any f32 scale — exact, not approximate.

**Correctness:** `parity_gemm_oq_compact_iu4x2` checks against an oracle computed
on the ORIGINAL int8 activations (not merely self-consistency). 7 shapes, G=256
and G=128, max 1.36e-7.

**Speed**, against the iu8 compact GEMM at B=256:

| shape | iu8 | 1-pass iu4 | 2-pass (8-bit) | 2p vs iu8 | 2p vs 1p |
|---|---|---|---|---|---|
| gate/up | 3.339 ms | 1.581 | 2.118 | **1.58x** | 1.34x |
| down | 5.159 | 1.942 | 1.999 | **2.58x** | **1.03x** |
| gate/up B=128 | 1.832 | 0.833 | 1.146 | 1.60x | 1.38x |
| gate/up B=512 | 6.828 | 3.029 | 4.406 | 1.55x | 1.45x |

**Two iu4 passes beat the iu8 kernel by 1.55-2.58x at identical 8-bit
precision.** An earlier estimate here said 2 passes would be ~1.07x of iu8, i.e.
a LOSS — that was wrong twice over. It priced N passes as N full passes, when the
weight tile is loaded once and consumed by both; and it ignored that iu8 also
pays the nibble decode and the slower WMMA rate. A later model predicted 1.24x;
the measurement is better than that too.

`down` is the striking one: 8-bit activations cost **3% over 4-bit** there. That
shape is data-movement-bound enough that the second WMMA pass is nearly free.

Cost: two live i32 accumulator sets, so this variant halves NB (8 -> 4). Note
that makes the table above not a pure pass-count comparison — the 2-pass arm also
has half the b-tiles per workgroup, hence more weight traffic per output. It wins
anyway, so the confound only understates it.

## 2. Qwen3.8-27B activation crest, by tensor class

Full 64-layer calibration, `calib-multi-8m` corpus, 32 sequences x 2048 context,
imatrix-only. 496 captured tensors.

| class | mean crest | max | n |
|---|---|---|---|
| bulk (in_proj_*, gate, up, qkv) | 27.0 | 36.2 | 368 |
| out_proj / o_proj | 166.4 | 223.6 | 64 |
| down_proj | 182.3 | 225.7 | 64 |

A **6-7x gap**, with 128 of 496 tensors (26%) in the hard classes. The 0.8B showed
the same split at 6-7 vs 50-63, so this is structural, not model-specific — and
it is markedly worse on the larger model.

## ⚠️ The ABSOLUTE crest values overstate the difficulty. The RATIO is the signal.

Taken literally, crest 226 with int8's 127 levels leaves rms at 0.56 of a level,
which should be catastrophic — and yet **this exact model serves coherently today
at W4A8**, 15.1 tok/s. The statistic and the working system disagree, and the
statistic is what is wrong.

The reason: this crest is `max|x| over the WHOLE CORPUS / rms over the whole
corpus`, per channel. The real quantizer scale is per (token, group of 256) — a
far narrower window than corpus-wide. A channel whose magnitude drifts slowly
across the corpus has a huge corpus-wide crest and a perfectly tame per-token
one.

So use these numbers COMPARATIVELY (down_proj/out_proj are ~7x harder than the
bulk — trustworthy) and not as an absolute bit-width verdict. Deciding actual
bit-widths needs a per-(token, group) statistic: absmax within a group for one
token, against that group's rms. The current reduction sums over tokens and
cannot produce it — that is the next capture to add, not a re-derivation from
what is already stored.

---

# CORRECTION: measured per-(token, group), the outlier story mostly disappears

The crest factors above are corpus-wide: `max|x| over the whole corpus / rms over
the whole corpus`, per channel. An Opus scale is chosen per (token, group of
256). `calib_group_crest_reduce_f32` now captures the latter directly, emitted as
`<tensor>.groupcrest` `[2, K/256]` (row 0 sums per-token crest → mean, row 1
maxes).

Qwen3.5-0.8B, 24 sequences x 1024, same corpus:

| class | corpus | grp-mean | grp-max | overstated by | n |
|---|---|---|---|---|---|
| bulk | 9.6 | **4.86** | 12.37 | 2.0x | 138 |
| down_proj | 84.9 | **5.89** | 15.83 | **14.4x** | 24 |
| out_proj | 101.6 | **8.35** | 15.93 | **12.2x** | 24 |

Two things follow, and the second invalidates a design direction.

**1. The overstatement is NON-UNIFORM.** 2x on the bulk, 12-14x on the
post-nonlinearity classes. So the corpus metric inflated not just the magnitudes
but the GAP between classes.

**2. There is no small set of outlier tensors to protect.** The real spread is
bulk 4.86, down_proj 5.89, out_proj 8.35 — **1.2-1.7x**, not the 6-7x the corpus
metric showed. `grp-max` is 12.4-15.9 across every class. The earlier reading —
"5 of 7 projections are comfortable, only down_proj/out_proj need help" — was an
artifact of the wrong statistic, and the per-channel outlier scheme it pointed to
would have been wasted work.

What the numbers say instead: signed int4 has 7 positive levels, so rms lands at
7/crest — 1.4 levels on the bulk, 0.84 on out_proj. int4 activations are
MARGINAL EVERYWHERE, not locally. int8's 127 levels give 8-26 levels at the same
crests, comfortable everywhere. That is exactly why W4A8 is what ships.

## The design this leaves

Use the **2-pass 8-bit activation path everywhere**. It costs 1.03-1.45x the
1-pass and still beats the iu8 kernel it replaces by 1.55-2.58x. No channel
permutation, no outlier tile, no sparse activation correction — the machinery all
of that would have required is unnecessary.

The weight-side sparse overlay remains a separate and still-unsolved problem
(see the correction pass notes); this changes only the ACTIVATION side.
