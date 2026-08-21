# KVarN profiled: 5.3x on KV MEMORY, 1.01x on decode speed — and why

Profiled Qwen3.8-27B--oq4.25++ decode on halo (gfx1151) with `rocprofv3
--kernel-trace --stats`, driving the daemon over its stdin JSON protocol so it is
the profiler's own child. ~3000-token prompt, greedy decode.

## The f32 decode profile

| kernel | calls | % of decode |
|---|---|---|
| `gemv_oq_compact_grouped_v3` | 317441 | **91.05%** |
| `gated_delta_net_f32` | 30720 | 3.76% |
| **`attention_f32`** | 10240 | **1.68%** |
| `fused_rmsnorm_mq_rotate_awq` | 81920 | 1.27% |
| everything else | — | < 0.5% each |

**Attention — the only thing that touches the KV cache — is 1.68% of decode.**
The weight GEMV is 91%.

## Why KVarN cannot be 4-10x faster here

It IS 4-10x, on the axis it actually addresses. Measured resident KV, at
`physical_cap=896` with 16 of 64 layers carrying KV:

| | KV resident | share of per-token traffic |
|---|---|---|
| f32 | 117.4 MB | 0.75% |
| kvarn (K 4b var-norm + V Q8) | 22.0 MB | 0.14% |

**5.3x less KV memory.** But the per-token traffic that decode is bound by is
**15.46 GB of weights** against 117 MB of KV. Amdahl caps the end-to-end win:

> making ALL KV traffic **free** would be a **1.017x** decode speedup.

So there is no configuration of KV quantization that yields 4-10x decode on this
model at this context length. The measured end-to-end (14.5 f32 -> 13.9 kvarn)
is consistent with "no gain, plus a little extra unpack work", and decode is
already at 90% of its DRAM bound — that bound is set by weights.

## Where the speed benefit does arrive

KV grows with context, weights do not. KV bytes per context token (f32, 16 full-
attention layers, 4 KV heads x 256) = 131072 B, so:

| context | KV | share of traffic | KVarN end-to-end |
|---|---|---|---|
| 896 (today's cap) | 0.12 GB | 0.8% | **1.01x** |
| 8,192 | 1.07 GB | 6.5% | 1.06x |
| 32,768 | 4.29 GB | 21.7% | 1.21x |
| 131,072 | 17.18 GB | 52.6% | 1.75x |
| 262,144 | 34.36 GB | 69.0% | 2.27x |

**Crossover — KV traffic equals weight traffic — is at ~118k tokens.** Below
~30k, KVarN is a memory-capacity feature (it is what lets the long context fit at
all); above ~100k it becomes a throughput feature, asymptotically approaching the
5.3x byte ratio.

The current run never gets there: TriAttention eviction pins `physical_cap=896`,
so KV is bounded at 117 MB no matter how long the prompt is. Any KVarN speed
measurement taken under that cap is measuring 0.8% of the traffic.

## Tooling defect found

**`rocprofv3` ABORTS on the KVarN path** — reproducible 3/3, `--stats` and
`--output-format csv` alike:

```
F0822 correlation_id.cpp:248] retired dangling correlation IDs: 1340
*** Check failure stack trace: ***
```

f32 survives to write `run_results.db` (then SIGSEGVs in teardown *after* output
generation completes, so the db is intact and usable). KVarN produces no output
at all. Both crashes are in the profiler's shutdown, not the workload — the
generate request completes normally in both. So the KVarN kernel-level breakdown
in this document is derived from the f32 profile plus byte arithmetic, not
measured directly, and that is why.
