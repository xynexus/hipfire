# Qwen3 and EmbeddingGemma NPU embedding support

Date: 2026-07-17

## Summary

Add full-resident XDNA encoding for the three Qwen3 embedding models and extend
EmbeddingGemma from its fixed 256-token path to bucketed sequences up to 2048
tokens. Deliver Qwen3-Embedding-0.6B first, then admit 4B and 8B through the same
geometry-driven implementation.

Tokenization remains host-side; transformer layers, pooling, and normalization
execute on the NPU. OQ8+ is the initial production format.

## Implementation changes

- Preserve the underlying architecture identity: Qwen3 remains arch 1 and
  EmbeddingGemma remains arch 19. Add explicit HFQ embedding metadata for
  workload type, pooling, prompts, output dimensions, NPU storage layout, and
  supported sequence geometry. Runtime admission must use this metadata, never
  filenames.
- Extend safetensors preparation and quantization to ingest SentenceTransformers
  modules and prompts. Produce canonical artifacts such as
  `Qwen3-Embedding-0.6B--npu.oq8+.gfx1151.hfq`.
- Add a Qwen embedding state that loads only encoder-required tensors and avoids
  autoregressive output projection, generation scratch, and persistent KV
  allocation.
- Implement geometry-driven AIE2P Qwen encoder graphs covering causal attention,
  Q/K normalization, RoPE, SwiGLU FFN, final RMSNorm, last-real-token pooling,
  and L2 normalization.
- Add compiled sequence buckets of 128, 256, 512, 1024, and 2048 tokens. Group
  requests by bucket, use segmented attention to prevent cross-document leakage,
  and chunk each group to at most 4096 padded rows per dispatch.
- Reject inputs exceeding 2048 tokens with a clear API error; do not silently
  truncate. Padding must never affect causal attention or last-token selection.
- Generalize the resident EmbeddingGemma scheduler to the same bucket contract
  while retaining its bidirectional/sliding attention, mean pooling, Dense
  heads, and Matryoshka behavior.
- Extend `/v1/embeddings` with optional `input_type: "query" | "document"`,
  defaulting to `document`. Apply the artifact's corresponding prompt.
  `/v1/rerank` continues to apply query and document prompts automatically.
- Cache NPU images by NPU architecture, model geometry, quant format, sequence
  bucket, and dispatch batch. Missing or incompatible images fail closed for
  NPU-only artifacts rather than silently claiming NPU execution.
- Land in stages: shared metadata and scheduler, 0.6B vertical slice,
  EmbeddingGemma 2048-token generalization, then 4B and 8B geometry admission.
  Preserve unrelated dirty work and commit each stage separately.

## Test and admission plan

- Unit-test SentenceTransformers metadata ingestion, architecture/workload
  routing, prompt selection, bucket assignment, overflow rejection, padding
  masks, and last-token pooling.
- Compare individual NPU components with PyTorch/Hugging Face oracles before
  full-model admission: QKV, Q/K norm and RoPE, segmented causal attention,
  output projection, FFN, final norm, pooling, and normalization.
- Run full-model parity at lengths 1, 127, 128, 255, 256, 257, 511, 512, 1024,
  and 2048, including mixed-length batches and duplicate-document isolation
  checks.
- Require same-artifact GPU/NPU cosine of at least 0.999 and OQ8+-versus-BF16
  cosine of at least 0.995. On a fixed `mteb/scifact` subset, require nDCG@10
  degradation no greater than 1% relative to the BF16 Hugging Face reference.
- Exercise the real HTTP API for query and document embeddings, dimensions,
  ordering, mixed batches, invalid roles, oversized inputs, and reranking.
- Benchmark batch sizes 1, 2, 4, 8, 16, and 32 across all sequence buckets,
  reporting actual tokens/second, documents/second, latency percentiles, energy,
  padding ratio, and confirmed backend. Do not classify hybrid or fallback
  measurements as NPU results.
- Run the NPU hardware gates under `hipfire lock`, the relevant embedding
  evaluation battery, `./tests/coherence-gate-dflash.sh` for shared quant/runtime
  changes, and `./tests/no-gpu-ci.sh`.
- Admit 4B and 8B only after the 0.6B implementation passes the full correctness,
  quality, API, and hardware matrix without model-size-specific execution
  branches.

## Assumptions

- Initial support targets AIE2P/XDNA2 on the current gfx1151 host;
  portability-sensitive interfaces remain geometry-driven so another NPU
  generation can supply separate compiled images.
- OQ4 and mixed-precision NPU admission are deferred until dedicated int4
  kernels and quality evidence exist.
- The 2048-token cap applies equally to all four models for this release,
  regardless of the larger context advertised by Qwen.
- "Full encoder NPU" excludes host tokenization and initial embedding-table
  lookup, but includes every transformer layer and the final
  pooling/normalization pipeline.
