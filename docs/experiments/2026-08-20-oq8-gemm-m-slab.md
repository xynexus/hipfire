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

## Follow-up: how much is actually left, and why LDS did not get it

After the memory BIOS change (peak 240 -> 256 GB/s) the slabbed kernel reads
**139.9 GB/s** cold, against a pure-read stream's **250**. That 56 % looks like
~1.8x sitting on the table. It is not — most of the gap is the access SHAPE, and
the shipped kernel is already near the ceiling that shape allows.

Measured with the compute stripped out (`bench_read_bw --gemm-pattern`, same
85 MiB, gate/up `[17408, 5120]`):

| weight read | slabs | GB/s |
|---|---|---|
| scattered — 16 lanes on 16 rows, K bytes apart (what the kernel does) | 1 | 19.6 |
| **scattered, 4 slabs** — what the SHIPPED kernel does | 4 | **151.3** |
| scattered + multiple waves per block | 4 | 160.7 |
| cooperative (all lanes sweep one row), block=32 | 4 | 193.2 |
| **cooperative, block=512, unslabbed** | 1 | **231.3** |

So:

* The shipped GEMM's **139.9 is 92 % of the 151.3** its own access pattern can
  reach. Nothing that leaves the pattern alone can find more than ~8 %.
* Slabbing is confirmed in isolation as the whole of the earlier 7.3x: the same
  scattered pattern goes 19.6 -> 151.3 purely by splitting the launch.
* Widening the block (several waves, each on its own 16-row tile) is worth
  **+8 %** and is a small change — the cheapest remaining win.
* The real **1.5x** (151 -> 231) needs COALESCED reads, which the WMMA operand
  layout forbids directly: `v_wmma_i32_16x16x16_iu8_w32` requires lane L to hold
  row L, so the rows must arrive via LDS.

**The obvious LDS fix was measured and LOST.** Staging the group's 16x256 B tile
through LDS with cooperative loads gave **101 GB/s** against the unstaged 139.9.
The first cut had a 16-way bank conflict — a 256 B row stride is 64 dwords, so
every row starts on bank 0 — and padding the stride to 272 B only reached 108,
still well short. At block=32 the barrier and LDS round-trip cost more than the
coalescing saves. Both variants were byte-exact, so this is purely a perf result.

What a real attempt needs, from the table above: the 231 GB/s row is
**cooperative loads AND a wide block AND no slabbing**, i.e. a different tiling —
many waves per block, each owning a 16-row tile, block-cooperative staging, and
an LDS budget that keeps enough waves resident (16 waves x 16 rows x 256 B is
64 KB, so it has to stage in K-chunks). That is a kernel rewrite, not a tweak,
and it is worth ~1.5x on 69 % of DFlash verify's GPU time.

## Multi-wave blocks: the cheap +12 %

The scoping above put ~8 % within reach without changing the access shape, by
running several waves per block so there are several independent WMMA chains to
hide the scattered read's latency. Implemented (`gemm_oq8_grouped_wmma_mw`, no
LDS, block width from `blockDim.x`) and measured cold on gate/up `[17408, 5120]`
at B=9:

| waves/block | 1 | 2 | 4 | **8** | 16 |
|---|---|---|---|---|---|
| GB/s | 141.0 | 148.8 | 150.3 | **158.4** | 154.5 |

**+12 %** on the tall-thin shape, and `down` / `o_proj` are flat within noise, so
8 waves is now the default (`HIPFIRE_OQ8_GEMM_MW`, 0 selects the original
one-wave kernel). Byte-exact — same reads, same math, only the block's wave count
changes.

End-to-end on Qwen3.8-27B oq4.25++ + CASK with DFlash2, q8 KV, at the post-BIOS
memory clock (256 GB/s peak):

| | verify/cycle | decode |
|---|---|---|
| 1 wave | 213 ms | 4.91 tok/s |
| **8 waves** | **197 ms** | **5.04 tok/s** |
| plain decode, no draft | — | 8.00 tok/s |

Identical output text, same tau. Verify is now **197 ms against the 978 ms this
started at** — a 5.0x reduction across the M-slab and multi-wave changes together.

DFlash is still a net loss (5.04 against 8.00) because acceptance is the binding
constraint now, not verify: at tau=2.05 the cycle commits ~3 tokens for
`draft 74 + verify 197 + replay 3x131` ms. The ceiling at perfect acceptance is
`(74 + 197) / 8` = 34 ms/token = **29 tok/s against 8.00, a 3.6x** — so the
remaining work is the drafter (the unapplied DFlash2 candidate selector) and
`replay`, not the GEMM.

## Latent: the compact GEMM has the same shape

`gemm_oq_compact_grouped_wmma` launches `grid_m = m.div_ceil(16)` — a single
launch over all of M, exactly the shape that collapsed here. It did not appear in
the Qwen3.8-27B profile (that model's body is Oq8G256, and compact is only its
lm_head), so it is latent rather than hot, but any compact-resident Opus model
will hit it on a tall projection. The M-slab transformation is shape-only and
provably bit-identical, so porting it there is cheap; it just needs the same
`m_all` / `m_base` kernel parameters and a placement parity case.
