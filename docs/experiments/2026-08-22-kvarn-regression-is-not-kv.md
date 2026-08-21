# The KVarN decode regression is NOT in the KV path — it is the FFN GEMV

**Supersedes the conclusion in `2026-08-22-kvarn-attention-profile.md`**, which
said the ~4% KVarN decode regression was "spread thinly across the extra
dispatches, no single smoking gun". That was wrong. It was reached by comparing
whole-kernel totals and dispatch COUNTS without ever breaking GPU time down per
kernel SHAPE. Doing so localises almost all of it to one place.

## The A/B is real and reproducible

Back-to-back, same binary, repeated (`--battery speed`, CASK off):

| | rep1 | rep2 |
|---|---|---|
| f32 KV (oracle) | 14.5 | 14.5 |
| kvarn | 14.0 | 14.0 |
| kvarn, `HIPFIRE_KVARN_ROTATE=0` | 14.0 | 14.1 |

~3.4%, reproducible. **The two per-attention rotations cost nothing** — disabling
both changes nothing.

## Where the GPU time actually goes

Total GPU kernel time: **f32 138.6 ms -> kvarn 148.6 ms, +10.0 ms (+7.2%)**.

| kernel | f32 | kvarn | delta |
|---|---|---|---|
| attention (f32 -> kvarn tile + asym_reduce) | 6.28 | 10.02 | **+3.74 ms** |
| **`gemv_oq_compact_grouped_v3`** | 122.56 | 128.69 | **+6.13 ms** |
| everything else combined | 9.8 | 9.9 | +0.1 ms |

**The weight GEMV — not attention — is the largest term.** Same kernel, same
weights, essentially the same call count, 116.4 -> 122.0 ns/call.

## It is ONE shape, and it is the FFN

Breaking the GEMV down by grid size:

| grid_x | f32 ns | kvarn ns | ratio |
|---|---|---|---|
| 1536 | 3.9 | 3.9 | 1.002 |
| 32768 | 14.2 | 14.2 | 1.000 |
| 163840 | 144.8 | 144.8 | 1.000 |
| 196608 | 73.5 | 73.5 | 1.000 |
| 327680 | 119.8 | 119.8 | 1.000 |
| 393216 | 143.2 | 143.1 | 0.999 |
| **524288** | **209.5** | **230.3** | **1.099** |

Six of seven shapes are bit-identical. **The entire +6.13 ms is grid_x=524288**
(271,623 x 209.5 ns = 56.9 ms vs 273,738 x 230.3 ns = 63.0 ms).

From the dispatch sequence, that shape is the pair of GEMVs issued right after
`fused_rmsnorm_mq_rotate_awq` and before `silu_mul_f32` — **the FFN gate and up
projections**. They have nothing to do with the KV cache.

(Also unexplained and separate: kvarn issues **+2115 calls** of that shape,
~1 extra per token.)

## Everything plausible has been ruled out

The gate/up slowdown survives every ablation:

| ablation | gate/up ns | vs f32 |
|---|---|---|
| f32 oracle | 209.5 | 1.000 |
| kvarn baseline | 230.3 | 1.099 |
| kvarn, attention cross-lane reduction gutted | 231.1 | 1.103 |
| kvarn, attention record loads made coalesced | 230.4 | 1.100 |
| **kvarn, attention Phase D gutted entirely** | 230.4 | 1.100 |
| **kvarn, the per-attention sync memcpy removed** | 229.9 | = |
| kvarn, `HIPFIRE_KVARN_ROTATE=0` | (end-to-end unchanged) | |

and it is invariant to context and to buffer size:

- **Near-zero context** (5-token prompt): 117.7 -> 121.8 ns/call overall, and the
  524288 shape 214.5 -> 229.8 (1.072). So it is **not** KV traffic — it is there
  when the KV cache is essentially empty.
- **max_seq sweep** 1024 / 2048 / 4096 / 16384: ratio **1.072, 1.079, 1.070,
  1.073** — dead flat. So it is **not** KVarN's buffer footprint.
- **Not a global clock/DVFS effect**: of the kernels present in both modes,
  `gated_delta_net_f32` is 49.6 ns in BOTH, `rope_partial_halfsplit_f32` 5.0 in
  both, `conv1d_silu_split_f32` 3.2 in both. Only the FFN GEMV moves.

So: selecting KVarN makes the single most expensive kernel in the model ~7%
slower, deterministically, with no KV data involved, and none of KVarN's actual
work is responsible.

## Why this matters before touching batched prefill

Two reasons to stop and understand this first:

1. **Wiring batched KVarN prefill would not fix it.** The regression is in decode,
   in the FFN, and is independent of the prefill path.
2. **It is 7% of the kernel that is 86-91% of decode.** That is worth more than
   everything else measured in this session's KVarN work combined, and it is
   currently unexplained.

Remaining hypothesis is memory placement — KVarN allocates a different set of
buffers at load (records + a constant ~512 kB window + a lazily-allocated ~512 kB
`kvarn_tiles`, instead of the f32 `k_gpu`), which could shift where the gate/up
weight tensors land and change their DRAM channel/bank distribution. That fits
"deterministic, shape-specific, size-independent" but is NOT confirmed, and the
flat max_seq sweep is mild evidence against it (different buffer sizes, identical
ratio). Confirming it needs allocation-address logging
(`HIPFIRE_MALLOC_BACKTRACE` exists as a starting point), not more kernel ablation
— six ablations have already come back null.
