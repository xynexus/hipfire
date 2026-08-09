# Plan: make `ModelSource` able to express HFQ

Scoped 2026-08-10, after wiring `.hfa` into `hipfire-quantize` (PR #242) hit the
same wall from the other side.

## The problem, precisely

`hipfire_model::ModelSource` is the source abstraction. `impl ModelSource for
HfqFile` exists — and is a **stub**:

```rust
fn tensor_data(&self, _name: &str) -> Option<(&TensorInfo, &[u8])> {
    // ... this method is not directly usable for HFQ. The HFQ path
    // continues to use HfqFile's native methods; ModelSource is
    // primarily for safetensors.
    None
}
fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> { None }
```

Two independent reasons it cannot be filled in as written:

1. **`&TensorInfo` is the wrong type.** It is safetensors-shaped; HFQ carries its
   own `HfqTensorInfo` with a per-tensor `quant_type`. You cannot return a
   borrowed reference to a type the source does not store.
2. **`&[u8]` cannot express a decoded payload.** HFQ tensors may be losslessly
   recoded (`Bf16Lut3` = 49, `Bf16Huff` = 50, the latter being the DEFAULT) or
   quantized. Their stored bytes are not the logical tensor. Decoding produces
   an owned buffer, and a borrowed-slice contract has nowhere to put it.

### What that costs today

- **Streamed calibration cannot read `.hfq` or `.hfa`.**
  `calibration/layer_stream.rs` binds the CONCRETE `SafetensorsSource` (4 sites)
  rather than the trait, and derives provenance from
  `identity_kind: "safetensors_header_hash"`, which the read-ledger and
  `--resume` depend on. So `hipfire-coexistence calibrate --model` takes a
  safetensors dir or cache root, and nothing else.
- **bf16 recode decoding is copy-pasted per arch.** `16cb54d56` had to patch
  "decode lossless bf16 recodings" into FIVE arch loaders separately
  (embeddinggemma, lfm2moe, minimax, nemotron, qwen35-vl) because there is no
  shared layer that could own it.
- **Every large-model conversion still pays a full restore** for the calibrate
  leg: 244 GB for Qwen3.5-122B-A10B, ~730 GB for the 397B.

## The precedent — this is already solved twice, locally

Both existing HFQ-shaped readers reached for the same answer independently:

| where | signature |
|---|---|
| `HfqInputFile::tensor_data` (hipfire-quantize) | `-> Cow<'_, [u8]>` |
| `SafetensorsFile::tensor_data` (PR #242) | `-> Cow<'_, [u8]>` |

`Cow` is the shape: borrowed for an mmap'd verbatim payload, owned for a decoded
one. The refactor is to lift that from two local fixes into the shared trait.

## Proposed shape

```rust
pub struct TensorView<'a> {
    pub desc: TensorDesc,        // owned, source-neutral
    pub data: Cow<'a, [u8]>,     // borrowed for mmap, owned when decoded
}

pub trait ModelSource {
    fn tensor(&self, name: &str) -> Option<TensorView<'_>>;
    fn tensor_desc(&self, name: &str) -> Option<TensorDesc>;
    fn source_identity(&self) -> SourceIdentity;   // replaces the
                                                   // safetensors_header_hash
                                                   // assumption
    // ... metadata_json / arch_id / tensor_names unchanged
}
```

`TensorDesc` is owned (not borrowed) so a source can synthesise it — that is
what unblocks HFQ, whose descriptor is computed from `HfqTensorInfo` rather than
stored in safetensors form. It must carry enough to describe both: logical
shape, logical dtype, and the source's own quant/codec tag.

**Decode-on-read**: the source returns LOGICAL bytes. A consumer never sees
codec bytes, which is what lets the five arch-loader decode copies collapse into
one.

## Sequencing

Each step lands independently and leaves the tree green.

- **P0 — introduce `TensorView` / `TensorDesc`; migrate `SafetensorsSource`
  only.** Everything else keeps compiling. No behaviour change.
  *Verify*: byte-identical artifacts before/after on a small model.
  *Watch*: the safetensors path MUST stay `Cow::Borrowed`. If it silently starts
  copying, a 244 GB model copies 244 GB. Assert borrowed-ness in a test.

- **P1 — implement it for real on `HfqFile`, replacing the stub.** Decode-on-read
  for the lossless bf16 recodings.
  *Verify*: for a model that exists as both `.hfq` and a source dir, every
  tensor must compare equal through the trait.

- **P2 — move bf16 recode decoding out of the five arch loaders** into the
  source layer.
  *Verify*: existing arch loads unchanged; the tiny-fixture gate covers all five.

- **P3 — make `layer_stream` generic over the source**, and make provenance
  source-defined (`source_identity`) instead of assuming safetensors headers.
  This is the step with real risk: the read-ledger, `--resume` and the
  boundary-checkpoint machinery all key on that identity.
  *Verify*: there is a ready-made oracle —
  `hipfire-coexistence artifact compare-calibration --reference <a> --candidate <b>`.
  A streamed calibrate from a dir and from the same model's `.hfa` must produce
  matching artifacts.

- **P4 — accept `.hfa` / `.hfq` as calibrate sources.** Removes the last restore.

## Also worth knowing before starting

- The memory planner in `layer_stream` reports `max_layer_source_bytes` (1.69 GB
  on Qwen3.5-35B-A3B, against a 34 GB host reserve). For a compressed source the
  STORED size is not the decoded size, so the planner must ask for the logical
  size or it will under-reserve.
- `release_tensor_pages` / `release_tensor_range_pages` are `posix_fadvise`
  hints meaningful only for mmap. For owned buffers the equivalent is dropping
  them, which the caller already does — the trait should say so rather than
  leaving implementors guessing.
- Trait implementors to update: `SafetensorsSource`, `HfqFile` (stub today),
  and two `FakeSource` test doubles (`arch-qwen35/src/calibration_stream.rs`,
  `runtime/src/calibration/source.rs`).

## Why it is worth doing

It is the same blocker three times over: the 122B/397B restore cost, the
per-arch decode duplication, and `.hfq` not being a calibration source. One
owned-or-borrowed view type retires all three.
