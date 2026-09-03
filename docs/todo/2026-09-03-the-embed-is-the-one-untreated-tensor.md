# The embedding table is the only tensor that gets no quantization treatment

Status: **OPEN**, 2026-09-03. Two concrete follow-ups, both from the 0.8B–9B
scale study (`docs/reports/2026-09-03-qwen35-scale-study.md`).

## What the code does today

`cli.rs:12072` — ask for **any** Opus body (`oq4`, `oq8`, `oq8+`, mixed) and the
embedding table is routed to plain `Q8F16`:

```rust
} else if (use_oq4 || use_opus_mixed || use_oq8 || use_oq8_plus) && is_embed {
    // Embedding lookup has its own loader path. It supports Q8
    // directly, while OQ4/OQ8 are GEMV/GEMM weight formats.
    let q = quantize_q8f16(&f32_data);
```

`--embed-precision` accepts only `source | q8 | bf16 | f16 | hfq4`. Neither
`quantize_q8f16` nor `quantize_hfq4g256` applies FWHT, AWQ, or Hessian/LDLQ — both
are plain affine. And the gather kernels are exactly
`embedding_{bf16,bf16_batched,f16,f16_batched,f32_batched,hfq4g128,hfq4g256,hfq4g256_batched,q4k,q8,q8_batched}`
— no Opus member.

So the embed is the **one tensor in the model that receives none of the
quantization machinery**, and it is also the tensor whose error is most costly:
it seeds the residual stream *unnormalized*.

## 1. The gather-friendly fallback should be Bf16Lut3, not Q8F16

`Q8F16` is **lossy**. `Bf16Lut3` is a **lossless recoding** of bf16 and is already
plumbed for exactly this tensor: `expand_bf16_index` keeps a LUT3 *head* packed by
default because it is "the only large tensor that is a pure GEMV consumer", and the
comment there already notes that on a tied model "this same entry also backs the
embedding gather... the gather decodes it explicitly at `token_embd` load."

So the machinery to carry a LUT3 embed exists. Routing Opus bodies to `Q8F16`
spends real KLD for a size win that a lossless coding may match. **Measure
`Bf16Lut3` against `Q8F16` for the embed on the same body** — if the sizes are
close, the lossy fallback has no reason to exist.

## 2. Build a conditioned embedding-gather quant

The measured cost of dropping the embed from 8-bit to 4-bit, same `oq4.25` body,
identical recipe across scale:

| model | embed share | Δkld (evalA) | relative to the body's own KLD |
|---|---|---|---|
| 0.8B | 33.8% | +0.0284 | +27% |
| 2B | 27.0% | +0.0308 | +42% |
| 4B | 15.1% | +0.0169 | +23% |
| 9B | 11.4% | +0.0044 | +4.8% |

Both arms are **untreated**. The embed's error is outlier-driven — which is exactly
what an FWHT rotation fixes, and exactly what every *body* tensor already gets. A
rotated 4-bit gather format should therefore be materially cheaper than `HFQ4G256`,
and the whole 27–42% penalty at 0.8B–2B is the headroom.

The work is a gather kernel, not a codec: rotation is per-256-group along `k`, so a
gathered row can be inverse-rotated in-kernel the way `dequant_oq4g256` does on the
host. The payoff is concentrated at small scale, where the embed is a third of the
parameters — and small models are where a 4-bit embed is currently unaffordable.
