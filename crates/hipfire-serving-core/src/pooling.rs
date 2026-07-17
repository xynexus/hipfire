// SPDX-License-Identifier: Apache-2.0
// hipfire — embedding + reranking pooling seam. See LICENSE / NOTICE.

//! The non-autoregressive serving path: turn already-tokenized text into pooled
//! sentence embeddings (for `/v1/embeddings`) or relevance scores (for
//! `/v1/rerank`), without an lm_head decode loop.
//!
//! This module owns the arch-agnostic orchestration and scoring math; the daemon
//! supplies the loaded model, the tokenizer, and the task-prompt prefixing, then
//! maps the results onto the wire protocol. Two model shapes are supported:
//!
//! * **Encoder embeddings** (embeddinggemma, arch_id 19) — a bidirectional prefill
//!   + mean-pool + Dense heads, via [`hipfire_arch_embeddinggemma::embed_forward`].
//! * **Decoder pooling / yes-no rerank** (Qwen3, Gemma3-text, …) — last-token or
//!   mean pooling of the final hidden state, or the softmax of a yes/no token pair
//!   at the final position (the [`rerank_yes_no`] helper).
//!
//! Matryoshka truncation ([`truncate_and_renormalize`]) and cosine reranking
//! ([`rank_by_cosine`]) are pure and unit-tested here.

use hipfire_arch_embeddinggemma::embed_forward;
use hipfire_rdna::Gpu;

use crate::model::EmbeddingGemmaState;

#[cfg(target_os = "linux")]
fn embed_planned_embeddinggemma(
    gpu: &mut Gpu,
    state: &EmbeddingGemmaState,
    tokenized: &[Vec<u32>],
    projector: &mut hipfire_arch_embeddinggemma::NpuOpusProjector,
) -> Result<Vec<Vec<f32>>, String> {
    let Some(metadata) = state.embedding_metadata.as_ref() else {
        return hipfire_arch_embeddinggemma::embed_batch_forward_with_projector(
            gpu,
            &state.weights,
            &state.config,
            tokenized,
            projector,
        );
    };
    let lengths = tokenized.iter().map(Vec::len).collect::<Vec<_>>();
    let dispatches =
        crate::embedding_batch::plan_embedding_dispatches(&lengths, &metadata.sequence)?;
    let mut output = vec![Vec::new(); tokenized.len()];
    for dispatch in dispatches {
        let documents = dispatch
            .request_indices
            .iter()
            .map(|&index| tokenized[index].clone())
            .collect::<Vec<_>>();
        let embeddings = hipfire_arch_embeddinggemma::embed_batch_forward_with_projector(
            gpu,
            &state.weights,
            &state.config,
            &documents,
            projector,
        )?;
        if embeddings.len() != dispatch.request_indices.len() {
            return Err(format!(
                "embeddinggemma bucket {} returned {} embeddings for {} documents",
                dispatch.bucket,
                embeddings.len(),
                dispatch.request_indices.len()
            ));
        }
        for (embedding, request_index) in embeddings
            .into_iter()
            .zip(dispatch.request_indices.into_iter())
        {
            output[request_index] = embedding;
        }
    }
    Ok(output)
}

/// Encode a batch of already-tokenized texts with the embeddinggemma encoder,
/// returning one L2-normalized embedding per text, truncated+renormalized to
/// `dims` (Matryoshka). `dims` is the *resolved* output length — the caller should
/// pass `cfg.resolve_dims(requested)`.
pub fn embed_batch_embeddinggemma(
    gpu: &mut Gpu,
    state: &EmbeddingGemmaState,
    tokenized: &[Vec<u32>],
    dims: usize,
) -> Result<Vec<Vec<f32>>, String> {
    #[cfg(target_os = "linux")]
    if let Some(projector) = state.npu_projector.as_ref() {
        match projector.lock() {
            Ok(mut projector) => {
                match embed_planned_embeddinggemma(gpu, state, tokenized, &mut *projector) {
                    Ok(mut embeddings) => {
                        embeddings
                            .iter_mut()
                            .for_each(|embedding| truncate_and_renormalize(embedding, dims));
                        return Ok(embeddings);
                    }
                    Err(error) if state.weights.resident_only() => {
                        return Err(format!(
                            "embeddinggemma resident NPU encode failed ({error}); no GPU fallback is loaded"
                        ));
                    }
                    Err(error) => eprintln!(
                        "embeddinggemma resident NPU encode failed ({error}); retrying on GPU"
                    ),
                }
            }
            Err(_) if state.weights.resident_only() => {
                return Err(
                    "embeddinggemma resident NPU state is poisoned; no GPU fallback is loaded"
                        .to_string(),
                );
            }
            Err(_) => {
                eprintln!("embeddinggemma resident NPU state is poisoned; retrying encode on GPU")
            }
        }
    }
    let mut out = Vec::with_capacity(tokenized.len());
    for tokens in tokenized {
        let mut v = embed_forward(gpu, &state.weights, &state.config, tokens)?;
        truncate_and_renormalize(&mut v, dims);
        out.push(v);
    }
    Ok(out)
}

/// Matryoshka truncation: keep the first `dims` components and re-normalize to unit
/// length. `dims == 0` or `dims >= v.len()` leaves the (already-normalized) vector
/// untouched.
pub fn truncate_and_renormalize(v: &mut Vec<f32>, dims: usize) {
    if dims == 0 || dims >= v.len() {
        return;
    }
    v.truncate(dims);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Cosine similarity of two equal-length vectors. Returns 0 if either is degenerate.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Rerank `doc_embeddings` against `query_embedding` by cosine similarity. Returns
/// `(original_index, score)` sorted by descending score. Used when the reranker is
/// an embedding model (no dedicated yes/no head).
pub fn rank_by_cosine(query_embedding: &[f32], doc_embeddings: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, d)| (i, cosine_similarity(query_embedding, d)))
        .collect();
    sort_desc(&mut scored);
    scored
}

/// Cross-encoder reranker score from a yes/no token pair at the final position:
/// `softmax([logit[yes], logit[no]])[0]` — the probability the document is relevant
/// (Qwen3-Reranker uses true=9693 / false=2152). Numerically stable.
pub fn rerank_yes_no(logits: &[f32], yes_token_id: usize, no_token_id: usize) -> f32 {
    let y = logits[yes_token_id];
    let n = logits[no_token_id];
    let m = y.max(n);
    let ey = (y - m).exp();
    let en = (n - m).exp();
    ey / (ey + en)
}

/// Sort `(index, score)` pairs by descending score, stable on ties (lower index
/// first) so equal-scoring documents keep input order.
pub fn sort_desc(scored: &mut [(usize, f32)]) {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_renormalizes_to_unit_length() {
        // A normalized 4-vec; truncate to 2 dims and re-normalize.
        let mut v = vec![0.5, 0.5, 0.5, 0.5];
        truncate_and_renormalize(&mut v, 2);
        assert_eq!(v.len(), 2);
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.70710677).abs() < 1e-5);
    }

    #[test]
    fn truncate_noop_when_dims_ge_len_or_zero() {
        let mut v = vec![0.6, 0.8];
        truncate_and_renormalize(&mut v, 0);
        assert_eq!(v, vec![0.6, 0.8]);
        truncate_and_renormalize(&mut v, 5);
        assert_eq!(v, vec![0.6, 0.8]);
    }

    #[test]
    fn cosine_of_identical_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn rank_orders_relevant_first() {
        let q = vec![1.0, 0.0];
        let docs = vec![
            vec![0.0, 1.0], // orthogonal → 0
            vec![1.0, 0.0], // identical → 1
            vec![0.7, 0.7], // ~0.707
        ];
        let ranked = rank_by_cosine(&q, &docs);
        assert_eq!(ranked[0].0, 1); // best is doc 1
        assert_eq!(ranked[1].0, 2); // then doc 2
        assert_eq!(ranked[2].0, 0); // orthogonal last
    }

    #[test]
    fn yes_no_softmax_prefers_yes_when_higher() {
        // logits: index 3 = yes, index 1 = no.
        let logits = vec![0.0, -2.0, 0.0, 3.0];
        let p = rerank_yes_no(&logits, 3, 1);
        assert!(p > 0.99, "strong yes → p≈1, got {p}");
        // Symmetric case → 0.5.
        let eq = vec![0.0, 1.0, 0.0, 1.0];
        assert!((rerank_yes_no(&eq, 3, 1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sort_desc_stable_on_ties() {
        let mut s = vec![(0, 0.5f32), (1, 0.9), (2, 0.5), (3, 0.9)];
        sort_desc(&mut s);
        assert_eq!(s.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3, 0, 2]);
    }
}
