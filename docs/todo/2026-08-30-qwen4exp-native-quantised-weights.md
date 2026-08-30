# TODO: keep qwen4_exp's weights quantised on the GPU (the last fit blocker)

**Status:** open (proposed 2026-08-30). This is the ONLY remaining blocker to
serving the shipped Qwen3.8-Flash-Next checkpoint.

## Where the model stands

Three pieces landed and are gate-checked on a fixture:

- a registered `ServingFactory` — arch 26 resolves by data lookup, prefill and
  decode produce finite moving logits, and `reset_session` is proven total;
- quantised artifacts LOAD — q8f16/q4k/oq4/oq8/OQ+ compact are dequantised at
  load, with the gate requiring the argmax to match the bf16 control;
- the 102 GB n-gram table STREAMS, row by row, bit-identically to resident.

What remains is memory, not correctness: weights are dequantised to **f32** at
load, so an oq4 artifact costs what an f32 one does.

## The arithmetic that makes this mandatory

The shipped model is ~360 GB at bf16. Subtracting the 102 GB n-gram table (which
now streams and never lands in memory) leaves a ~129 GB bf16 trunk — about 64.5 B
parameters. Dequantised to f32 that is **~258 GB** resident, on a 128 GB machine.
At native oq4 the same trunk is **~32 GB**. There is no version of this that fits
without the weights staying quantised.

## Why this is not a field swap

`hipfire_runtime::weights::WeightTensor` + `weight_gemv` already exist, every
format is handled by `TransformerLoader::load_weight`, and the 24 `gemv_f32` call
sites in this crate would mostly convert mechanically.

**The MoE experts are the exception, and they are the bulk of the bytes.**
`moe_gpu::view2d` takes a sub-view into ONE stacked f32 tensor per projection:

```rust
let gu = view2d(&w.gate_up, e * gu_sz, 2 * mi, hidden);
gpu.gemv_f32(&gu, x, &s.gu)?;
```

A byte offset into a stacked tensor is only a valid weight when the encoding is
flat. A quantised expert carries its own scales (and, for OQ, its own FWHT
grouping), so per-expert addressing has to go through the INDEXED routed-expert
kernels — `gemv_oq4g256_moe_gate_up_indexed`, `..._moe_down_k8_indexed_batched`
and their oq8 siblings — the same machinery qwen3.5 uses. That is the real work
here, not the field type.

Three things about that path are already known and should not be re-derived:

1. **`K_TOP` is a runtime kernel argument**, not a compile-time constant, in every
   indexed expert GEMV. The `k8` in those filenames is inherited naming. Only the
   top-k SELECTION kernels (`moe_softmax_topk_k8`, `moe_topk_renorm_k8`) hardcode
   `#define K_TOP 8`, and they touch no weights — so this model's top-10 routing
   is a much smaller change than the file names suggest. (BUGS.md carries this.)
2. **`moe_intermediate_size` 640 is not 256-aligned**, and at K=640 the quantizer
   silently drops routed `down_proj` out of Opus into HFQ4G128 — losing
   calibration entirely, exit 0, no warning. That bug is in BUGS.md and must be
   fixed before any quantised expert path is trusted on this geometry. 640 = 2^7·5,
   so 128 is the only usable Opus group (the rotate is an FWHT: powers of two only).
3. **The experts fit resident once grouped** — the GTT 2 MiB rounding measurement
   put the 512-expert set at ~66 GB with N=4 grouping versus 105 GB per-tensor, so
   this does not additionally require the pager.

## Where the bytes actually are

Measured against the shipped geometry (hidden 2560, mi 640, 48 layers, 512
experts, vocab 248320):

```
  routed experts          120.80 B params   97.3%
  shared experts            0.24 B params    0.2%
  attn + GDN (approx)       2.52 B params    2.0%
  lm_head                   0.64 B params    0.5%
  TOTAL trunk             124.18 B params
```

**The routed experts are 97.3% of the trunk.** Converting every other linear in
the crate saves 2.7% and changes nothing about whether the model fits. That is
worth stating plainly because the natural instinct — do the easy mechanical ones
first, then the hard one — produces a large diff, a satisfying green gate, and no
progress against the actual constraint.

## Suggested order

1. **Fix the `mi=640` Opus admission bug.** DONE 2026-08-30 for `oq4` (see
   BUGS.md): a 128-aligned K now routes to `OqPlusCompactG128` instead of falling
   out of Opus and losing calibration. Still open for `oq8`/`oq8+`, which have no
   G128 producer path and now warn loudly instead of failing silently. Until this
   held, any oq4 number on this geometry was measuring an uncalibrated HFQ4G128
   fallback, not a quantisation result.
2. **Routed experts onto the indexed kernels.** This is the whole job. Thread `k`
   (10 here) through `INDEXED_MOE_K_TOP` / `oq_indexed_admissible` / the loader
   repack so they keep agreeing, and replace `moe_gpu::view2d` + `gemv_f32` with
   the indexed GEMVs.
3. The other linears, whenever. They are 2.7% and can follow at leisure — or never,
   if `weight_gemv` on an f32 `WeightTensor` is not worth the churn.

## Do not

Do not convert experts by dequantising per-expert into scratch and reusing
`gemv_f32`. It would work and it would be pointless: the whole reason for this
work is to stop materialising f32 weights.
