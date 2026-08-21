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
