# Qwen3.8-27B oq4.25++ end-to-end: decode is done, prefill is 70x off

Measured on halo (gfx1151, 128 GB UMA) via `hipfire eval --battery speed`, which
drives the daemon and self-locks. Daemon rebuilt first — it links `hipfire-rdna`,
so a stale binary silently measures the old kernels.

| KV mode | prefill tok/s | decode tok/s | ttft |
|---|---|---|---|
| fp32 (eval default) | **15.5** | **14.5** | 15.45 s |
| kvarn | 14.3 | 13.9 | 16.74 s |

Model geometry from the HF config (`text_config`): 64 layers, hidden 5120,
intermediate 17408, 16 full-attention + 48 linear-attention layers, vocab 248320,
untied head. **GEMM body = 23.82 B params**; artifact streams 14.40 GiB.

## Decode is essentially finished

Weights are 15.46 GB/token against a measured 248.5 GB/s pure-read ceiling, so
the bandwidth bound is **16.07 tok/s**. Measured 14.5 = **90.2% of it**.

There is no meaningful decode headroom left at this quant. Getting past it needs
fewer weight bytes (a lower bit-width, which the 4.25-bit floor forbids) or
speculative decode, not a faster kernel.

Note the ceiling itself is optimistic: it was measured with an idle CPU, and on
an APU the CPU shares the memory controller.

## Prefill is running PER-TOKEN

**prefill/decode = 1.07.** A batched prefill should be compute-bound and roughly
two orders of magnitude faster than decode; a ratio of 1.0 means every prompt
token is paying its own full weight sweep.

| basis | tok/s | measured 15.5 is |
|---|---|---|
| iu4 no-op issue ceiling (105 TOPS) | 2204 | 0.7% |
| best measured W4A4 GEMM (53.4 TOPS) | 1121 | **1.4%** |

240 prompt tokens x 69 ms/token (the decode cost) = 16.5 s, which is the observed
15.45 s ttft. That arithmetic alone confirms it.

## Cause, for the fp32 case — confirmed

`prefill_batch.rs` has an explicit guard:

```rust
// F4 guard: reject batched prefill when KV tier has no batched keys.
// F32 KV has only BatchEq(1) -> MissingImpl at resolve.
let kv_f32 = !kv_cache.quantized && !kv_cache.quant_q8 && !kv_cache.quant_hfq4;
let eligible = eligible && !kv_f32 && !kv_asym2_tree;
```

`hipfire eval` defaults to **fp32 KV**, so the battery disables batched prefill
by construction. Everything upstream passes — with `HIPFIRE_KERNEL_TRACE=1` the
eligibility trace reports `force_fallback=false n=240 dn_quant=FP32
all_layers_dense_la=true`, i.e. the model qualifies and only the KV tier rejects
it.

**So the 15.5 tok/s prefill is partly an artifact of the eval harness's KV
default, not of serving configuration.** Any prefill number taken from
`--battery speed` without an explicit `--kv-mode` is measuring the per-token path.

## But KVarN does NOT fix it — and that part is NOT yet explained

Switching to `--kv-mode kvarn` passes the `kv_f32` guard, and prefill got
*slightly worse* (14.3). More telling: with kvarn **neither**
`HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1` nor the `HIPFIRE_KERNEL_TRACE=1`
`pbs_eligible` line prints at all, while both print under fp32. The batched entry
`forward_prefill_batch_with_pbs_opts` is therefore never reached under kvarn —
something routes around it further upstream. Not identified; the two diagnostics
above are the thread to pull.

This is consistent with the recorded KVarN batched-prefill defect (batched + KVarN
is far less faithful than per-token), so a deliberate exclusion somewhere is
plausible — but it has not been located and should not be assumed.

## What this reorders

Every kernel result this session moves decode, which is already at 90% of its
bound, or moves prefill GEMM throughput, which prefill is not currently using at
all. **Prefill at 1.4% of its realistic ceiling is worth more than the entire
remaining W4A4 GEMM headroom** (53.4 -> 105 TOPS would be ~2x on a path that is
not running). Making batched prefill actually engage is the highest-value work
outstanding on this model, ahead of any further kernel tuning.
