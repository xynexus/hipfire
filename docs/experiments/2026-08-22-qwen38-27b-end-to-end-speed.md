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

## RESOLVED: KVarN prefills per-token BY DESIGN

*(This section originally read "NOT yet explained". It is now located.)*

`hipfire-serving-core/src/generate.rs:2923` declines batched prefill for KVarN
outright, with the reason in place:

```rust
if kv.quant_kvarn {
    // KVarN (and the deferred-hierarchical two-tier cache built on it)
    // require the per-token attention dispatch (kv_cache_attention_dispatch):
    // the batched forward_prefill_batch runs its own batched attention and
    // never populates the KVarN window/records (nor the hier hot ring), so
    // the prompt KV is wrong and decode degenerates. Prefill per-token via
    // forward_scratch ... Slower prefill, but kvarn is a KV-memory mode, not
    // a throughput one.
    for &tok in &new_tokens { qwen35::forward_scratch(...) }
```

So `forward_prefill_batch_with_pbs_opts` is never reached under KVarN — which is
exactly what the missing `pbs_eligible` / `prefill-eligible` traces showed. Not a
bug; a deliberate correctness fallback.

**Both KV modes therefore prefill per-token, for different reasons:**

| KV mode | why prefill is per-token |
|---|---|
| f32 (eval default, oracle only) | `kv_f32` guard — f32 KV has only `BatchEq(1)`, so batched prefill would hit `MissingImpl` at resolve |
| **kvarn (production)** | **explicitly declined — batched prefill never populates the KVarN window/records, so the prompt KV would be wrong** |

That is why prefill measured ~15 tok/s in *every* configuration tried, with and
without the CASK eviction cap.

## The fix is well-defined, and the pieces already exist

Batched prefill does not have to be incompatible with KVarN — the batched KVarN
write path is already written and in use elsewhere in the same file:

- `prefill_batch.rs:702` — `kv_cache_write_kvarn_window_routed_batched`
- `prefill_batch.rs:864` — `kvarn_gather_k_tiles` + `kvarn_quantize_tile`

What is missing is wiring those into the batched forward so it populates the
window/records the way the per-token path does, at which point the
`generate.rs:2923` fallback can be lifted. **That is the single highest-value
piece of work outstanding on this model**: prefill is at 1.4% of its realistic
ceiling, ~70x, against a decode that is already at 90% of its DRAM bound.

