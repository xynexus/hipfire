# TODO: a batched Opus-compact GEMM designed for decode widths (17..128 rows)

Status: TODO (scoped, not started). Measured state and dead ends below are from
branch `feat/swarm-batching-paged-kv` as of `38a87544f`, Qwen3.8-27B--oq4.25++,
gfx1151.
Date: 2026-10-01

## Why

Continuous batching put the decode step in a regime the Opus-compact kernels
were never designed for. A step feeds every projection with one row per session
plus speculative draft rows — 16..96 rows — where the existing kernels were
tuned either for B=1 (`gemv_oq_compact_grouped_v3`, B<=8 multicol) or for prefill
(`gemm_oq_compact_iu4x2_w64`, tuned at B=128..512).

A decode-only kernel trace of 64 sessions (`bench.py ... 64 long 200`, run under
`rocprofv3`, see "How to measure") splits the step:

| kernel | share of decode |
|---|---|
| `gemm_oq_compact_iu4x2_w64_n64` (the GEMM) | **48%** (383 us avg) |
| `gated_delta_net_f16_routed_batch_seq` | 15% |
| `oq_compact_overlay_correct_trs` (sparse overlay) | **12%** (99 us avg) |
| `attention_kvarn_routed_batched_gqa6` | 9% |
| x8 transpose + quantize + rotate + interleave | ~5% |

So ~65% of a 64-session decode step is the Opus matmul pipeline, and the GEMM
itself runs ~2.7x below the weight-bandwidth bound.

## Where it stands (don't re-derive)

Serving route `gemm_oq_compact_act_batched`, per projection, ms (incl.
quantize/transpose/correction), from `bench_oq_compact_route`:

| rows | gate/up (17408x5120) | down (5120x17408) | qkv (6144x5120) | path |
|---|---|---|---|---|
| 8 | 0.22 | 0.27 | 0.07 | wide multicol (bandwidth-bound, ~210 GB/s) |
| 16 | 0.33 | 0.34 | 0.13 | wide multicol (~8 TOPS, compute-bound) |
| 24 | 0.52 | 0.55 | 0.21 | wide multicol |
| 33 | 0.60 | 0.58 | 0.18 | BN=64 tile + `_trs` correction |
| 64 | 0.67 | 0.70 | 0.22 | BN=64 tile + `_trs` correction |
| 128 | 1.08 | 1.28 | 0.47 | default BM=64 x BN=128 tile + `_tr` |

Floors at gfx1151's ~210 GB/s achievable weight bandwidth: gate/up and down
~0.21 ms, qkv ~0.075 ms. At 64 rows the GEMM alone is ~0.51 ms (gate/up) —
~22 int8 TOPS, ~120 GB/s of weights.

Landed already: BN=64 tile for B<=64 (`395fe0a66`), wide multicol for batched
serving up to 24 rows (`395fe0a66`), small-B overlay correction `_trs`
(`395fe0a66`), tiled x8 transpose + 512-row chunking for prefill widths
(`a521a5588`).

## Tried, and why it did not work

- **Tile reshapes of the w64 kernel** (WARPS_M/N, WMt/WNt via `#define`
  prefixes, swept at B=33/64/128): best alternative moved one shape ~5% and lost
  on another (e.g. `4,1,1,4` gate/up 0.506 -> 0.472 ms at B=33, down 0.471 ->
  0.506). BN=64 (`2,1,2,4`) was the only clear win and is in.
- **Fusing the overlay into the GEMM's group fold** (`OQ_FUSED_OVERLAY`, gathers
  from K-major XT inside the fold): gate/up B=33 **0.59 -> 1.10 ms**. 96 byte
  gathers per thread per group stall the WMMA pipeline far worse than a separate
  pass costs. Reverted; do not retry in that form.
- **Fusing the overlay from LDS, not global** (2026-10-02, `OQ_FUSE_OVERLAY`
  per strip, before the fold): entries loaded once per group into registers
  (`OV_SLOTS`=3), x[idx] rebuilt from the interleaved strip already in LDS,
  `val*x` added to the group's i32 `accl`. Correct (fused vs separate pass
  max_rel 1.6e-4..3.8e-4, vs an overlay term ~1e3 rel) and same occupancy (3
  waves/SIMD on `_n64`, no spills), but **40-50% slower on every shape**:
  gate/up B=64 0.65 -> 0.93 ms, down 0.66 -> 0.83, qkv B=512 1.34 -> 2.01. Why:
  on RDNA3.5 WMMA executes on the SIMD's own VALU, so every VALU op the overlay
  adds sits in the same issue stream as the matmul -- and this version also did
  its control work (table decode, strip match, divergent branch) per lane on
  the VALU, which the scalar unit could have taken (next entry). (Not because the GEMM is
  WMMA-bound -- it runs ~37% of WMMA peak at B=64; see the memory finding
  below.) The output-stationary lane
  map makes it worse (a wave's 4 row-groups diverge: ~64 body runs per group
  each with 8 LDS byte loads + ~24 VALU, for 96 real MACs). A row-uniform remap
  (scalar-loaded entries, lane = column, a side f32 accumulator merged through
  LDS at the end) cuts that ~5x on paper, which still estimates at 20-40% of
  the GEMM vs ~24% for the separate pass -- break-even at best.
- **Overlay driven from the scalar unit** (2026-10-02, the row-uniform remap
  above, built): the SALU is separate silicon from the VALU the WMMA uses, so
  the loop walked one weight row at a time across the wave (lane = column) --
  side entry as a wave-uniform `s_load_b64`, idx/val decode, strip match and
  branch all on the SALU (`s_cvt_f32_f16`, `s_mul_f32` on gfx1151), only the
  per-column multiply-add on the VALU, a per-row f32 accumulator merged into
  facc through LDS. Correct (vs separate pass <=4.3e-4 rel), occupancy kept at
  3 with `amdgpu_waves_per_eu(3)` (240 VGPRs, no spills), row loop unrolled
  (32 batched s_loads, no M0 indexing). **2.3x SLOWER** (cold, B=64: gate/up
  0.59 -> 1.37 ms, down 0.61 -> 1.35, qkv 0.22 -> 0.46). Ablation: with the LDS
  reads and nibble math removed -- scalar loop only -- still 1.21 ms. Why: a
  wave issues in order, so the ~96 entry checks + 32 s_loads it walks per strip
  are cycles it is not issuing WMMAs, and every wave on the SIMD carries the
  same load; the scalar side is the bottleneck, not the VALU. And the VALU/LDS
  remainder (1.37 - 1.21 = ~0.16 ms) alone is about what the whole separate
  pass costs (~0.15), so even a free scalar side breaks even. The only form
  left would pre-bucket the overlay by strip at weight-load time (a new side
  plane) to cut the scalar walk ~4x -- and its VALU part still only breaks
  even. **The overlay stays a separate pass.**
- **`_trs` correction past 64 rows**: 2-6x worse than `_tr` (side-plane re-reads
  per 16-column block). Keep the B<=64 threshold.
- **Wide multicol at 4 lanes per group, 64 weights per lane** (2026-10-07,
  9..16 rows). The idea: past 8 rows the kernel is VALU-bound, and each
  (row, column) pays cvt + scale mul + fmac plus an overlay LDS gather for only 8
  dot4. At 4 lanes per group the same epilogue covers 16 dot4, halving it per MAC
  (ISA at B=16: 768 dot4 against the same 48 cvt/fma). A half-empty last round
  (ng % 8 == 4: 20 and 68 groups) loaded a real group with its scale zeroed.
  Parity passed, and it was slower on 3 of 4 shapes. Cold ms, master -> LPG=4
  with RW=3:

  | B | gate/up | down | qkv | wo |
  |---|---|---|---|---|
  | 9 | 0.233 -> 0.297 | 0.258 -> 0.282 | 0.095 -> 0.133 | 0.115 -> 0.110 |
  | 12 | 0.265 -> 0.364 | 0.278 -> 0.333 | 0.114 -> 0.147 | 0.130 -> 0.107 |
  | 16 | 0.321 -> 0.471 | 0.313 -> 0.412 | 0.144 -> 0.197 | 0.144 -> 0.129 |

  - The activation tile is 8 groups x BC x 256 B: 32 KB at B=16, so 2
    workgroups per WGP (4 waves/SIMD, against 8). VGPRs rose 191 -> 232.
  - RW=2 cut VGPRs to 174 but was worse again (gate/up B=16 0.589): it loses
    row reuse of the tile.
  - Only `wo` (K=4096, no tail round) gained, about 1% of a step. That isn't
    worth a second kernel set.
  - What's left is the tile itself. The epilogue saving is real but smaller
    than the occupancy it costs, so a version that pays has to stage fewer
    bytes per MAC.

## Landed: the B<=64 tile was bandwidth-bound on over-fetch (2026-10-02)

`GL2C_EA_RDREQ_*` counters (rocprofv3 `--pmc`) on the BN=64 tile at B=64
showed it pulling **~2.1x its weight bytes** from memory: gate/up 99.5 MB for
47.7 MB of nibbles+scales, i.e. ~200 GB/s -- at the DRAM limit, on waste. Two
sources: (1) each strip read 32 B per weight row and relied on L2 holding the
128-byte line for the next three strips; (2) the weight stream evicted the
side table between group folds (ablating the scale reads dropped 18 MB: a
6.5x over-fetch of a 2.8 MB plane). Fix (`A_GROUP_LINE`, `_n64` only): read
each row's whole group line once, non-temporally, into registers and slice it
into LDS per strip. EA traffic now 48.8 / 49.0 / 17.1 MB (gate/up / down /
qkv, ~ideal). Bit-exact. Route, `HIPFIRE_BENCH_COLD=1` (weights not MALL-warm,
as in serving), B=25..64: gate/up -12..16%, down -9..12%, qkv -5..7%.
End to end, 27B at 64 sessions (committed / decode_ms, same binary with and
without the define): **143.9 -> 154.0 / 155.3 tok/s (+7-8%)**.

Not for multi-N-block grids (B > 64): other workgroups reuse those lines, and
non-temporal reads evict them (m128 at B=512: 97 -> 230 MB, slower). Also
tried and neutral: padding the A tile in LDS (halved the 38% bank-conflict
rate, no time change -- LDS is not the limiter), double-buffering the group
line (no gain). Bench note: without `HIPFIRE_BENCH_COLD=1`, shapes that fit
the 32 MB MALL (qkv, wo) are timed warm and mislead -- wo is 0.155 ms warm,
0.33 cold at B<=64 (~32 GB/s; its K<5120 wave32 path is the next suspect).

## Landed: a BN=32 tile for 17..32 rows (2026-10-03)

The BN=64 tile does 64 columns of WMMA work for any B. A BN=32 variant
(WARPS 2x1, WMt 2, WNt 2, group-line staging; still one N-block) wins from 17
rows. Cold route, at 24 rows: gate/up 0.488 -> 0.407 ms, down 0.479 -> 0.384,
qkv 0.190 -> 0.147. Serving routing is now: wide multicol <= 16, BN=32 tile
17..32, BN=64 tile 33..64, default tile above. End to end, 27B: 20 sessions
101.1 -> 108.5 tok/s, 24 -> 107.3 -> 122.0, 32 -> 125.4 -> 141.6. (Swept
BM=32 and BM=64/WARPS_M=1 variants of BN=32 too: within a few % either way.)

Remaining gap for Corrode-sized swarms (3-15 sessions): the wide multicol is
bandwidth-bound to 8 rows but turns VALU-bound past that -- cold, gate/up
0.226 ms at 8 rows, 0.271 at 12, 0.326 at 16, against a ~0.21 ms floor.
A BN=16 w64 tile does not close it (quiet host, cold, route; multicol vs
the best BN=16 tile, WARPS 1x1): gate/up 0.267 vs 0.321 ms at 12 rows, 0.325
vs 0.319 at 16; down 0.281 vs 0.304 / 0.316 vs 0.323; qkv 0.112 vs 0.122 /
0.144 vs 0.117 (WMt=2). Only qkv-sized M (6144) prefers a tile at 14-16 rows,
worth ~1% of a 16-session step -- not routed. The tile is not WMMA-bound
here either (BN=16 runs 1088 tiny workgroups); closing the 9..16 gap needs a
different decode-width kernel, not another tile shape.

## Tried: split-K for the B<=64 tile (2026-10-02, reverted)

Wave-scheduling rounds are real on this kernel (cold, GEMM only, B=64, K=5120:
120 workgroups 0.184 ms, 240 -> 0.334, 256 -> 0.427, 360 -> 0.477), and down
(M=5120 -> 80 workgroups, K=17408) under-fills: 102 GB/s vs 114-131 for fuller
grids. Built split-K with a deterministic last-arriving-slice reduction (each
K-slice writes its partial; the tile's last slice sums them in slice order and
resets a per-tile counter -- bit-identical run to run, <=2.1e-4 rel vs unsplit).
Swept: down best at 3 slices (GEMM B=33 0.502 -> 0.393 ms, B=64 0.529 ->
0.462); qkv (20 groups) and gate/up lose at any split. Route, down only: -8% at
B=25, -2.5% at B=64. **End to end at 64 sessions: 154.5 / 153.9 vs 154.6 tok/s
off -- nothing**, since down at B=64 is a quarter of the GEMM time and gained
2.5%. Corrode's own swarms (3-15 sessions) decode at B<=24, the multicol path,
so it does not reach them either. Not worth a counter buffer, a partial plane
and a second epilogue; revisit only if B=25..48 becomes the serving regime.

## Scope of the redesign

Goal: B in 17..128 at >=60% of weight bandwidth with the overlay included —
i.e. gate/up at 64 rows <= ~0.35 ms (from 0.67), with no regression at B<=16 or
B>=256.

Candidates, roughly in order of expected payoff:

1. **Overlay applied at weight-decode time, not as a gather.** The overlay is a
   per-weight property (`idx`, `val` per 256-weight block, n_out ~3). Patch the
   int4 weight fragment in LDS/registers with the overlay values before the WMMA
   — e.g. split each overlay into a bulk-nibble correction plus a residual that
   fits the int4 range across the hi/lo digit passes, or stage the overlay
   weights as a tiny dense int8 side-GEMM over only the touched K columns of
   the tile. Either way the activation is read once, as part of the GEMM, and
   both the correction kernel and the x8 transpose that feeds it disappear
   (~17% of decode at 64 sessions).
2. **Occupancy.** The BN=64 tile is 2 waves/128 threads with ~14 kB LDS double
   buffered; profile with `rocprofv3 --pmc` (VALU/LDS/mem stall counters) to
   confirm latency-bound, then try deeper K pipelining (3-stage), BK=128, or a
   split-K over a second grid axis for K=17408 (down) where 272 M-blocks is the
   whole grid.
3. **Activation quantize/interleave fused into the previous op** (rmsnorm/rotate
   already touch x): saves the ~5% of separate tiny launches per projection.
4. **A multicol variant on WMMA for 9..32 rows**: wide multicol decodes weights
   at bandwidth but does the B-column dot on VALU (~8 TOPS). Same weight decode,
   iu8 WMMA for the columns.

Out of scope here: prefill widths (B>=256; at ~21-32 TOPS after chunking, a
separate exercise), the MoE expert GEMMs (`prefill_moe_ffn_body_batched`, own
kernels), and non-Opus formats.

## Acceptance

- `parity_oq_compact_route` passes: tile paths bit-exact vs the default tile
  where the arithmetic order is unchanged, otherwise a stated tolerance
  (multicol today: <=1.4e-4 rel) and a note in this doc saying why.
- `bench_oq_compact_route` (serving route, B=1..512) shows the target with no
  regression outside 17..128.
- End to end (`toks.sh`-style: committed tokens / decode_ms from the batch
  runner's `decode cycle done` log), 27B at 64 sessions spec off, currently
  165 tok/s: target >=200.
- `./tests/tiny-prefill-gate.sh` and the commit hook's gates pass.

## How to measure

- Route microbench: `cargo build --release -p hipfire-rdna --examples &&
  target/release/examples/bench_oq_compact_route 16,24,33,48,64,96,128`
  (`HIPFIRE_BENCH_SERVING=0` for the single-request routing;
  `HIPFIRE_OQ_COMPACT_NO_CORRECT=1` times the GEMM without the overlay — wrong
  output, timing only). NB: `cargo build --release` alone does NOT rebuild
  examples.
- Parity: `target/release/examples/parity_oq_compact_route`.
- Kernel trace of a live decode: run `rocprofv3 --kernel-trace --stats -d DIR
  -o run -f csv -- hipfire serve --model M` in the foreground, drive load, then
  `kill -TERM` the **daemon** PID first and the server second — the trace only
  flushes on a clean exit (SIGINT to the server does nothing). Take the window
  after the last prefill-sized GEMM for decode-only numbers.
