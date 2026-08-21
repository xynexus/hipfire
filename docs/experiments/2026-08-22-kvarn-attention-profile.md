# KVarN is slower than the f32 oracle — and it is not the attention math

Follow-up to `2026-08-22-kvarn-profile.md`, which asked only whether KV
quantization could move decode end-to-end. This one profiles the attention
KERNEL, which is the claim that actually matters: KVarN reads ~5x fewer KV bytes,
so its attention kernel should be several times faster than the f32 oracle.
**It is 1.56x SLOWER, and the reason is not arithmetic.**

Measured on halo (gfx1151), Qwen3.8-27B--oq4.25++, ~3000-token prompt, greedy
decode, `HIPFIRE_CASK_OFF=1` (eviction cap removed), identical workload both
sides, identical call counts (33952).

| | GPU time | per dispatch |
|---|---|---|
| `attention_f32` (oracle) | 6.276 ms | 185 ns |
| `attention_flash_kvarn_tile_batched` | **9.793 ms** | **288 ns** |

## The kernel body is NOT what is being timed

Six independent experiments on the KVarN kernel, each a real change to the
inner work, measured with the same query and units:

| experiment | attention (ms) |
|---|---|
| baseline | 9.793 |
| runtime integer div/mod -> shift/mask (cpb is a power of two) | 9.794 |
| fold dequant into Q (4 float ops/elem -> 2) | 9.789 |
| **remove the per-token cross-lane reduction entirely** | 9.795 |
| **make every record load hit one cache line** | 9.794 |
| **remove the three prologue integer divides** | 9.794 |
| **gut Phase D (the Q8 V accumulation) completely** | 9.795 |

**Every one is within 0.06% of baseline, including deleting whole phases that
produce wrong answers.** A kernel whose runtime is invariant to its own body is
not compute-bound, memory-bound, or reduction-bound. It is at the per-dispatch
floor.

> Correction worth recording: I briefly read the Phase D result as 9.789 -> 0.010
> and called it a 1000x win. That was a units error — I had switched the query
> divisor from `/1e6` to `/1e9` between the two runs. Same number, different
> scale. Always re-tabulate every variant with one query.

## Why: the run is 99.9% dispatch overhead

Across the whole profiled run:

```
total GPU kernel time   148.6 ms
total dispatches      2,999,434
wall time                ~180 s      ->  GPU busy 0.08%
```

Three million dispatches for a 3000-token prompt plus six decode tokens is ~1000
kernel launches per token, each doing ~50 ns of GPU work. At that granularity the
attention kernels are simply at the hardware's minimum dispatch duration and the
185-vs-288 ns gap is launch shape, not work. (Profiling forced
`HIPFIRE_GRAPH=0` — see below — which removes graph amortisation and so
exaggerates the absolute dispatch count, but the f32/KVarN comparison is
apples-to-apples.)

## The actual cause of the end-to-end regression: KVarN ADDS DISPATCHES

Same run, dispatch counts by kernel:

| kernel | kvarn | f32 | delta |
|---|---|---|---|
| `mq_rotate_x` | 70,026 | **7** | **+70,019** |
| `__amd_rocclr_copyBuffer` | 38,389 | 4,437 | **+33,952** |
| `attention_flash_asym_reduce_batched` | 33,952 | — | +33,952 |
| `kvarn_gather_k_tiles` / `kvarn_quantize_tile` | 256 / 256 | — | +512 |
| `kv_cache_write*` | 33,952 | 67,904 | −33,952 |
| **total** | **2,999,434** | **2,892,772** | **+106,662 (+3.7%)** |

**+3.7% dispatches against a measured ~4% end-to-end slowdown (14.5 -> 13.9
tok/s).** That is the whole regression. Two entries deserve attention:

- **`__amd_rocclr_copyBuffer` +33,952 — exactly one buffer copy per attention
  call.** A runtime memcpy in the per-attention path is not something the KVarN
  design calls for; this looks like a staging copy that should not exist.
- **`mq_rotate_x` 7 -> 70,026**, i.e. two rotations per attention call, where the
  f32 path rotates 7 times in the entire run.

Neither is the attention kernel. Both are launch-count problems.

## What this means for the 4-10x expectation

It is not refuted — it is **untestable in this regime**. Attention cannot show a
bandwidth win while each dispatch does ~50 ns of work and the GPU is 0.08% busy.
Two things have to change before the comparison means anything:

1. **Batched prefill must engage** (see `2026-08-22-qwen38-27b-end-to-end-speed.md`)
   so a dispatch covers many tokens instead of one.
2. **The extra per-call dispatches above must go**, since in a dispatch-bound
   regime they are the entire measured difference.

Only then does the 5.3x byte advantage have room to appear.

## Kernel changes kept

Three edits landed. **None produced a measurable change here** — that is the
point of the table above — and they are kept because they are strictly less work
and will matter once the regime is GPU-bound:

- `attention_flash_kvarn_tile_batched`, `kvarn_dequant_tile`,
  `kvarn_quantize_tile`: `bits` is a runtime parameter, so `idx / cpb` and
  `idx % cpb` were a runtime integer divide. `bits` is always in {2,4,8}, so cpb
  is always a power of two and shift/mask is exact.
- `attention_flash_kvarn_tile_batched` Phase A: fold the dequant into Q
  (`partial = sc * (sum qd*A[i] + C)` with `A[i] = mq[i]*sa[i]`,
  `C = sum mq[i]*za[i]` hoisted per tile), 4 float ops per element -> 2.
- `attention_flash_kvarn_tile_batched` Phase D: the Q8 block scale and block
  pointer were reloaded for every dim. A thread owns dims `d0..d0+dpt-1` with
  `d0 = tid*dpt`, a contiguous run starting at a multiple of dpt, so it can never
  straddle a 32-dim Q8 block — `bi` is loop-invariant. Hoisting drops the f16
  scale loads 8x (1024 -> 128 per wave at head_dim 256).

All seven `parity_kvarn_*` suites PASS, with the fused-flash error unchanged at
1.13e-5 against the f64 host reference.

## Tooling

`rocprofv3` aborts on the KVarN path (`CHECK failure: retired dangling
correlation IDs`) — **corrected from the previous note, which blamed KVarN: the
trigger is hipGraph.** `HIPFIRE_GRAPH=0` makes it profile cleanly. That also
means any rocprofv3 profile of this daemon is taken with graph amortisation
disabled, and its absolute dispatch overhead is therefore pessimistic.
