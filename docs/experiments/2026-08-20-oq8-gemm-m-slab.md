# The Opus W8A8 GEMM collapses on tall-thin shapes — and the fix is a launch loop

**Status:** fixed and measured 2026-08-20, gfx1151 (Strix Halo).
**Result:** `gemm_oq8_grouped_wmma` on `[17408, 5120]` at B=9 goes **18.4 → 135.2
GB/s (7.3x)**. DFlash2 verify on Qwen3.8-27B drops **916 ms → 221 ms**, and
end-to-end decode goes **2.04 → 4.27 tok/s** against a 7.50 baseline.
No kernel math changed — only which launch computes which rows.

## Finding

`gemm_oq8_grouped_wmma` is 69 % of GPU time in a DFlash verify
(`2026-08-20-dflash-verify-profile.md`). Benched on the three dense-27B
projection shapes at B=9, on COLD weights:

| shape | weights | one launch | M-slabbed |
|---|---|---|---|
| gate/up `[17408, 5120]` | 86.3 MiB | **18.4 GB/s** | **135.2 GB/s** |
| down `[5120, 17408]` | 86.3 MiB | 169.1 | 168.0 |
| o_proj `[5120, 5120]` | 25.4 MiB | 145.1 | 140.1 |

The first two shapes have the **identical byte count** and identical per-block
work. The only difference is M — and the tall one runs 9x slower. Sweeping M at
fixed K=5120 shows a smooth `~M^2` collapse (283 GB/s at M=5120 down to 14 GB/s at
M=20480) with no cliff, so it is not a stride or alignment effect.

The fix does not touch the math: split the launch into row slabs, advance `W`/`Ws`
per slab, and have the kernel write `Y[out_col * m_all + m_base + out_row]`. Each
output word is produced by exactly the same block arithmetic as before —
`parity_oq8_gemm` checks all 52,224 words of a placement fixture land exactly.

    rows/launch   1024   2176   4352   5120   8704   17408 (one launch)
    GB/s cold      101    133    132    136    108    18

Broad plateau, so the constant is not knife-edge. `HIPFIRE_OQ8_GEMM_SLAB_ROWS`
overrides it, `HIPFIRE_OQ8_GEMM_SLAB=0` disables the split.

**Root cause is not established.** The split was found by ablation, not by
explaining the collapse. `~M^2` at constant total work points at something shared
between blocks — the scattered `Y` writes (the B output columns sit M floats
apart, so they spread as M grows) are the leading suspect, but that is a
hypothesis, not a measurement. Worth closing properly, because whatever it is
probably affects the other batched GEMMs the same way.

## ⚠️ Measure cold, or measure the cache

gfx1151 is LPDDR5X-8000 on a 256-bit bus — **~256 GB/s theoretical** — behind a
**32 MB MALL**. Timing a loop over one weight buffer smaller than 32 MiB reports
cache bandwidth:

    o_proj [5120, 5120], 25.4 MiB:   288 GB/s warm     140 GB/s cold

288 GB/s is *above the DRAM peak*, which is the tell — any figure near or above
~256 GB/s on this part is a cache hit. `bench_oq8_gemm_small_n --cold`
round-robins over enough distinct buffers to evict the MALL between touches.

This mattered: the slab constant was originally justified by the warm 283 GB/s
figure at M=5120, i.e. by a cache artifact. The value survived re-tuning against
cold weights; the reasoning did not. The two 86 MiB shapes were never affected
either way, which is why the headline 7.3x is unchanged.

Also: run more than 2-3 timed iterations. The first cold run did `copies`
iterations only and produced a 12 % swing that looked like a slab regression on
o_proj; it was noise.

## Effect on DFlash

Per-cycle phases at B=8 (`HIPFIRE_SPEC_PHASES=1`), q8 KV:

    before:  draft 74 ms   verify 916 ms   replay 131 ms x accepted
    after:   draft 74 ms   verify 221 ms   replay 131 ms x accepted

The ceiling, at perfect acceptance (accept=7, no replay): `(74 + 221) / 8` =
37 ms/token = **27 tok/s against the 7.50 baseline, a 3.6x ceiling** — up from the
1.0x that the same arithmetic gave before this change. Speculation on this model
is now worth pursuing, and two things follow:

1. **`replay` is now the dominant term** at realistic acceptance — 131 ms per
   accepted token, which at tau=1.7 is 43 % of the cycle. It needs per-position
   DeltaNet checkpoints during verify so rollback is a restore, not a re-run.
2. **Acceptance finally pays.** At tau=1.7 we measure 4.27 tok/s; the unapplied
   DFlash2 candidate selector is now worth implementing, which it was not when the
   ceiling was 1.0x.

## Reproduce

    cargo run --release -p hipfire-rdna --features deltanet \
      --example bench_oq8_gemm_small_n -- --cold
    HIPFIRE_OQ8_GEMM_SLAB=0 <same>          # single-launch comparison
    HIPFIRE_OQ8_GEMM_SLAB_ROWS=2176 <same>  # re-tune the target
