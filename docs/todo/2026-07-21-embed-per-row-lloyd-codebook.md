# TODO: embedding table → per-row Lloyd codebook

**Status:** open (proposed 2026-07-21). Pairs with the landed lm_head→QTIP
trellis (`HIPFIRE_QTIP_LM_HEAD`, commit 93db157f6): trellis the untied lm_head
(a matmul), Lloyd-codebook the embedding table (a gather).

## Motivation

The embedding table (`model.embed_tokens.weight`, `[vocab × dim]`) is the single
largest tensor and is **gather**-accessed (row lookup per token) — the trellis
can't random-access it, so today the mq4/qtip paths force it to `Q8F16`
(`main.rs` ~L4305, the `is_embed_table` arm added with the lm_head work). A
per-row **Lloyd codebook** is gather-friendly (each row decodes from its own
small codebook) and ~2.5× smaller than Q8 (3-bit index + codebook vs 1 B/w),
recovering the head/embed size that Q8F16 leaves on the table.

## What already exists (reuse, don't rebuild)

- **Codec:** `QuantType::MQ3G256Lloyd = 20`
  (`crates/hipfire-quant-format/src/lib.rs:48`) — MagnumQuant 3-bit + per-256-
  group Lloyd-Max 8-entry fp16 codebook, 112 B/group. Applied to a `[vocab, dim]`
  tensor this is already effectively a per-row codebook at 256-element
  granularity (dim/256 groups per row, each self-contained).
- **Quantizer:** `quantize_mq3g256_lloyd` (`main.rs` ~L8611/8921) already emits
  this format for matmul weights; `dequantize_mq2g256_lloyd_to_f32` exists.
- **Embedding-lookup pattern:** `EmbeddingFormat` enum
  (`crates/hipfire-runtime/src/weights.rs:100`, variants F32/Q4K/HFQ4G256/
  HFQ4G128/Q8_0); dispatch in `crates/hipfire-runtime/src/llama.rs` (L151-163
  single, L968-999 batched) → `gpu.embedding_lookup_{hfq4g256,q8}_batched`;
  kernels `kernels/src/embedding_{hfq4g256,q8,q4k}_batched.hip`. The loader maps
  the embed tensor's on-disk quant type → `EmbeddingFormat` around
  `crates/hipfire-runtime/src/hfq.rs:1987-2000`.

## What's missing (the actual work)

1. **`EmbeddingFormat::MQ3G256Lloyd`** variant in `weights.rs`.
2. **Lookup kernel** `kernels/src/embedding_mq3g256_lloyd_batched.hip`: for each
   token id, offset to its row, decode each 256-group (3-bit index → fp16
   codebook entry → f32) into the output embedding. Model it on
   `embedding_hfq4g256_batched.hip`. Add the `gpu.embedding_lookup_mq3g256_lloyd_batched`
   wrapper + register it.
3. **Loader mapping:** `hfq.rs` — when `embed_tokens.weight` on-disk quant type
   is `MQ3G256Lloyd`, select `EmbeddingFormat::MQ3G256Lloyd`. Add the arm to the
   `llama.rs` dispatch (single + batched, incl. the `matches!(...)` fast-path at
   L968).
4. **Quantizer opt-in:** route the embedding table through `quantize_mq3g256_lloyd`
   instead of `quantize_q8f16`. Mirror the lm_head flag — e.g.
   `HIPFIRE_EMBED_LLOYD=1` (or a `.embed-mq3l` modifier). The force-Q8 sites to
   branch: the qtip path `is_embed_table` arm (`main.rs` ~L4305) and the mq4/OQ
   embed handling (the `is_embed` decision ~L3633).

## Validation

- Emission: quantize an untied fixture; confirm `embed_tokens.weight` →
  `MQ3G256Lloyd` (ON) vs `Q8F16` (OFF), lm_head unaffected. (Same untied-fixture
  trick used for lm_head: `tiny_quant_probe`/hand-added `lm_head.weight`.)
- Runtime: load + forward — the embedding lookup must decode correct rows.
  Best signal is a **real** model coherence check (embed quant is token-identity
  sensitive; 3-bit may degrade more than Q8). A tiny_quant KLD cell would give a
  regression tripwire (see the mixed-precision tiny-test todo).
- Size/perf: confirm the ~2.5× embed shrink and measure decode-token overhead of
  the codebook indirection on the hot path vs the Q8 lookup.

## Open questions / design

1. **Bit width.** MQ3G256Lloyd is 3-bit. Embed is quality-sensitive — consider
   also supporting a 4-bit Lloyd (`mq4l`) variant, or make the width selectable,
   and compare KLD/coherence vs Q8F16 before defaulting anything.
2. **Tied models.** A tied embed also serves as lm_head (a matmul); a
   gather-only Lloyd embed can't be the lm_head. So embed→Lloyd pairs with an
   **untied** head (use `--rotate` to synthesize + `HIPFIRE_QTIP_LM_HEAD` to
   trellis it). Document this as the intended combo; a Lloyd-codebook *matmul*
   kernel would be a separate, larger effort if tied Lloyd is ever wanted.
3. **Row/group layout.** Confirm MQ3G256Lloyd blocks are row-major (a row's
   dim/256 groups contiguous) so a gather is one base offset + sequential group
   decode; if the on-disk layout interleaves, the kernel needs a stride map.
4. **Scope of loaders.** `llama.rs` is the shared embedding path (llama/qwen2/
   gemma3/qwen3.5 route through it); confirm each arch that would use Lloyd embed
   goes through this dispatch and not an arch-private lookup.

## Future kernel work: bf16 codebooks + RDNA3+ intrinsics

Applies to the Lloyd lookup kernel above (and is a broader theme for the decode
kernel family, not embed-specific):

- **bf16-codebook variants.** The MQ3G256Lloyd codebook is 8-entry **fp16**
  today. A parallel set of kernels that store/decode the codebook (and other
  dequant paths) as **bf16** would match the model's native dtype end-to-end,
  avoid fp16 range clipping on outlier centroids, and feed the RDNA3+ bf16 path
  directly. This is a *second* set of kernels alongside the fp16 ones (both kept
  — pick per source/target dtype), plus a `QuantType`/`EmbeddingFormat` distinction
  (e.g. a bf16-codebook Lloyd tag) so the loader routes to the right decode.
- **RDNA3+ conversion/packed intrinsics.** The decode path currently leans on
  generic casts. On gfx11+ (RDNA3 / RDNA3.5 / RDNA4) use the hardware conversion
  + packed intrinsics (native bf16 `v_cvt`, packed f16/bf16 ops) in the decode
  loop instead. Because these gather/decode kernels are per-element (not WMMA),
  the win is in conversion throughput and packed loads, not matrix accumulate —
  distinct from the WMMA-accumulate story (RDNA3 f16-accumulate is no faster than
  f32-accumulate; see `reference_rdna3_wmma_accumulate`).
- **Portability guardrail (AGENTS.md).** RDNA3+ intrinsics need an RDNA2/gfx10
  fallback (generic cast path) so the kernels still build+run on RDNA2. Gate the
  intrinsic path by arch at compile/dispatch time; never make gfx11+ the only
  path.
