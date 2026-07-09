// SPDX-License-Identifier: Apache-2.0
// hipfire — embeddinggemma text-embedding encoder arch crate. See LICENSE / NOTICE.

//! embeddinggemma (text embeddings) — `arch_id = 19`.
//!
//! A **bidirectional** Gemma-3 encoder that emits pooled sentence embeddings rather
//! than autoregressive logits. It shares the Gemma-3 backbone (see
//! `hipfire-arch-gemma3`) with three deltas:
//!
//! 1. **Bidirectional (non-causal) attention** — routes to the masked attention
//!    kernel (`hipfire_rdna::…attention_f32_batched_masked`) with an all-visible
//!    additive mask instead of the causal batched kernel.
//! 2. **Mean pooling** over the final-layer hidden states (attention-mask weighted).
//! 3. **Matryoshka Dense projection head(s)** on the pooled vector, then L2 norm.
//!
//! ## Bring-up status
//!
//! Phase 1 (this commit): [`config`] — `EmbeddingGemmaConfig` + the HFQ-metadata
//! parser (Gemma-3 shape + the sentence-transformers pooling/Dense/Matryoshka
//! block), unit-tested with no GPU dependency.
//!
//! Phase 2 (GPU, follow-on): `weights` (Gemma-3 layer weights + Dense heads),
//! `forward` (bidirectional prefill + pool + project), and the pooling serving seam
//! wired from `hipfire-serving-core`. The offline `Ingest` quant-policy lives in the
//! lean `hipfire-arch-embeddinggemma-spec` crate on the same `arch_id`.

pub mod config;
pub mod forward;
pub mod weights;

pub use config::{config_from_metadata_json, DenseHead, EmbeddingGemmaConfig, PoolingMode};
pub use forward::embed_forward;
pub use weights::{DenseHeadHost, EmbeddingGemmaWeights};
