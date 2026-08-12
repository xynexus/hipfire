# BF16 codecs: finish the in-memory half

MoE routers are now stored lossless BF16 (`is_moe_like` path in
`hipfire-quantize`, `--q8-router` restores the old Q8F16). This note is about the
part that is **not** done, and why it gets more valuable as models grow.

## What already works

Writing plain BF16 gets on-disk compression for free. The container's BF16
storage codec runs transparently on write — measured on the arch-6 toy MoE:

```
model.layers.0.mlp.gate.weight   BF16  [8, 256]  4.10 KB  [stored Bf16Huff 2.69 KB, 1.524x]
```

So "Bf16Huff on disk" is already the behaviour. Nothing needs to name the codec.

## What is missing: BF16 codecs are on-disk-only

`QuantType::logical()` maps `Bf16Lut3 | Bf16Huff -> BF16`, and the index entry is
rewritten to the logical type before any consumer sees it. Neither has a GPU
`DType`. So both decode to full-width BF16 at load, and the compression buys
**file size only** — nothing downstream ever sees compressed bytes.

That is why `Bf16Lut3`'s own doc comment ("Blocks are independently decodable so
a kernel can consume this **compressed in VRAM** — the ratio applies to weight
bandwidth, not just file size") describes an intent, not current behaviour.

## The upgrade: Huff on disk, Lut3 in VRAM

The two codecs are not competitors, they are a pipeline:

| | ratio | random access | role |
|---|---|---|---|
| `Bf16Huff` | ~1.50x | no — variable-length symbols need sequential decode | **on disk / over the wire**, where only total bytes matter |
| `Bf16Lut3` | ~1.38x | **yes** — fixed-size independently decodable blocks | **resident in VRAM**, decoded per block inside the kernel |

Target: store Huff, recode to Lut3 during load, keep Lut3 resident, and teach the
consuming kernels to decode a block on the fly. The recode seam already exists —
`hipfire_quant_format::storage` owns the lossless-recoding rule and
`is_lossless_recoding()`; today it only ever recodes *to* BF16.

## Why bother, given routers are 0.02% of a 35B

They are not the reason. At this size the whole question is rounding error, and
the honest framing is that routers were done for **correctness** (lossless
top-k selection), not for bytes.

The codec work pays off on a different axis, and it scales the wrong way to
ignore:

- **Disk and wire bandwidth.** A terabyte-class artifact at 1.5x is ~330 GB less
  to move. Fetch, verify, and repair over the lossy link this repo already
  fights (`hipfire hub fetch`'s progress-based retry exists because of it) all
  scale with bytes.
- **Memory bandwidth, which is the real prize.** Decode is weight-bandwidth
  bound. A Lut3-resident tensor cuts the bytes the kernel reads per token by
  ~1.38x for *lossless* BF16 — not a quality trade, a free reduction in the
  quantity that actually gates tok/s.
- **Streaming.** With paged/streamed experts the model does not fit resident, so
  weight traffic is continuous rather than one-time. Compression multiplies
  against every token, and the working set that fits in VRAM grows by the same
  ratio.

Applies well beyond routers: any BF16 tensor — VL towers, undercovered experts
kept at source precision by `--expert-coverage-policy preserve-undercovered`,
embed — is a candidate.

## Watch out

- **Ratios are per-tensor-class, not universal.** The 1.38x/1.50x figures come
  from whole-model BF16 and a MedGemma-27B VL tower. The same codecs manage only
  **1.075x** on a `.calib.hfq` bf16 Hessian, because `XᵀX` spans far more
  octaves than weights. `bf16_lut3::encode_if_smaller` is the existing guard —
  measure before assuming a ratio.
- The escape-plane length is data-dependent, so `block_bytes()` is `None` for
  these types and byte length does not follow from the shape.
- Readers must take an owned copy (`tensor_data_vec`), never the zero-copy
  `tensor_data` mmap slice, unless they decode blockwise on the GPU — which is
  precisely the capability this note is asking for.
