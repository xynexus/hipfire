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
- **`_trs` correction past 64 rows**: 2-6x worse than `_tr` (side-plane re-reads
  per 16-column block). Keep the B<=64 threshold.

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
