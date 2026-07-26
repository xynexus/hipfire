// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE / NOTICE.

//! Geometry-only batching for compiled embedding graphs.
//!
//! Requests are grouped by sequence bucket, then split so a dispatch never
//! exceeds the artifact's padded-row ceiling. The plan carries both padding
//! masks and last-real-token rows; executors must consume those rather than
//! treating padding as semantic input.

use std::collections::BTreeMap;

use hipfire_model::embedding::{EmbeddingAttentionMode, EmbeddingSequenceMetadata};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingDispatchChunk {
    pub bucket: usize,
    /// Compiled dispatch geometry. This is the next power of two at or above
    /// the real request count and may therefore include trailing dummy slots.
    pub dispatch_batch: usize,
    /// Original request indices, in stable input order within this bucket.
    pub request_indices: Vec<usize>,
    pub token_lengths: Vec<usize>,
    pub padded_rows: usize,
    /// One byte per padded row: 1 for a real token, 0 for padding.
    pub padding_mask: Vec<u8>,
    /// Row indices in the padded dispatch matrix used by last-token pooling.
    pub last_real_token_rows: Vec<usize>,
    /// Fixed-size document boundaries for block-diagonal/segmented attention.
    pub segment_offsets: Vec<usize>,
}

impl EmbeddingDispatchChunk {
    /// Reference visibility rule for the segmented-attention ABI. Compiled
    /// graphs consume the same bucket/length descriptors; this method keeps a
    /// small, exact oracle for unit/parity tests without allocating an O(rows²)
    /// dense mask.
    pub fn attention_visible(
        &self,
        query_row: usize,
        key_row: usize,
        mode: EmbeddingAttentionMode,
        sliding_window: Option<usize>,
    ) -> bool {
        if query_row >= self.padded_rows || key_row >= self.padded_rows {
            return false;
        }
        let query_segment = query_row / self.bucket;
        let key_segment = key_row / self.bucket;
        if query_segment != key_segment || query_segment >= self.token_lengths.len() {
            return false;
        }
        let query = query_row % self.bucket;
        let key = key_row % self.bucket;
        let real_tokens = self.token_lengths[query_segment];
        if query >= real_tokens || key >= real_tokens {
            return false;
        }
        match mode {
            EmbeddingAttentionMode::Causal => key <= query,
            EmbeddingAttentionMode::Bidirectional => true,
            EmbeddingAttentionMode::BidirectionalSliding => {
                sliding_window.is_some_and(|window| window > 0 && query.abs_diff(key) < window)
            }
        }
    }
}

pub fn plan_embedding_dispatches(
    token_lengths: &[usize],
    geometry: &EmbeddingSequenceMetadata,
) -> Result<Vec<EmbeddingDispatchChunk>, String> {
    let mut grouped = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    for (request_index, &tokens) in token_lengths.iter().enumerate() {
        let bucket = geometry
            .bucket_for_len(tokens)
            .map_err(|error| format!("embedding input {request_index}: {error}"))?;
        grouped
            .entry(bucket)
            .or_default()
            .push((request_index, tokens));
    }

    let mut dispatches = Vec::new();
    for (bucket, requests) in grouped {
        let per_dispatch = geometry.max_padded_rows_per_dispatch / bucket;
        if per_dispatch == 0 {
            return Err(format!(
                "embedding bucket {bucket} exceeds the {}-row dispatch ceiling",
                geometry.max_padded_rows_per_dispatch
            ));
        }
        for requests in requests.chunks(per_dispatch) {
            let dispatch_batch = requests.len().next_power_of_two();
            let padded_rows = dispatch_batch * bucket;
            let mut padding_mask = vec![0u8; padded_rows];
            let mut last_real_token_rows = Vec::with_capacity(requests.len());
            let mut segment_offsets = Vec::with_capacity(dispatch_batch + 1);
            segment_offsets.push(0);
            for (document, (_, tokens)) in requests.iter().enumerate() {
                let start = document * bucket;
                padding_mask[start..start + tokens].fill(1);
                last_real_token_rows.push(start + tokens - 1);
                segment_offsets.push(start + bucket);
            }
            for document in requests.len()..dispatch_batch {
                segment_offsets.push((document + 1) * bucket);
            }
            dispatches.push(EmbeddingDispatchChunk {
                bucket,
                dispatch_batch,
                request_indices: requests.iter().map(|(index, _)| *index).collect(),
                token_lengths: requests.iter().map(|(_, tokens)| *tokens).collect(),
                padded_rows,
                padding_mask,
                last_real_token_rows,
                segment_offsets,
            });
        }
    }
    Ok(dispatches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> EmbeddingSequenceMetadata {
        EmbeddingSequenceMetadata::npu_default()
    }

    #[test]
    fn assigns_boundary_lengths_to_compiled_buckets() {
        let plan = plan_embedding_dispatches(
            &[1, 127, 128, 255, 256, 257, 511, 512, 1024, 2048],
            &geometry(),
        )
        .unwrap();
        assert_eq!(
            plan.iter().map(|chunk| chunk.bucket).collect::<Vec<_>>(),
            vec![128, 256, 512, 1024, 2048]
        );
        assert_eq!(plan[0].request_indices, vec![0, 1, 2]);
        assert_eq!(plan[1].request_indices, vec![3, 4]);
        assert_eq!(plan[2].request_indices, vec![5, 6, 7]);
    }

    #[test]
    fn overflow_is_rejected_without_truncation() {
        let error = plan_embedding_dispatches(&[2049], &geometry()).unwrap_err();
        assert!(error.contains("input 0"));
        assert!(error.contains("maximum supported length is 2048"));
    }

    #[test]
    fn padding_mask_and_last_token_rows_are_document_local() {
        let plan = plan_embedding_dispatches(&[1, 127], &geometry()).unwrap();
        let chunk = &plan[0];
        assert_eq!(chunk.padded_rows, 256);
        assert_eq!(chunk.segment_offsets, vec![0, 128, 256]);
        assert_eq!(chunk.last_real_token_rows, vec![0, 128 + 126]);
        assert_eq!(chunk.padding_mask[0], 1);
        assert!(chunk.padding_mask[1..128].iter().all(|value| *value == 0));
        assert!(chunk.padding_mask[128..128 + 127]
            .iter()
            .all(|value| *value == 1));
        assert_eq!(chunk.padding_mask[255], 0);
    }

    #[test]
    fn dispatches_are_chunked_at_4096_padded_rows() {
        let lengths = vec![128; 33];
        let plan = plan_embedding_dispatches(&lengths, &geometry()).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].request_indices.len(), 32);
        assert_eq!(plan[0].padded_rows, 4096);
        assert_eq!(plan[1].request_indices, vec![32]);
        assert_eq!(plan[1].padded_rows, 128);
    }

    #[test]
    fn non_power_of_two_request_counts_use_the_next_compiled_batch() {
        let plan = plan_embedding_dispatches(&[128, 128, 128], &geometry()).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].dispatch_batch, 4);
        assert_eq!(plan[0].padded_rows, 512);
        assert_eq!(plan[0].request_indices, vec![0, 1, 2]);
        assert_eq!(plan[0].segment_offsets, vec![0, 128, 256, 384, 512]);
        assert!(plan[0].padding_mask[384..].iter().all(|value| *value == 0));
    }

    #[test]
    fn segmented_attention_blocks_padding_future_tokens_and_other_documents() {
        let plan = plan_embedding_dispatches(&[3, 2], &geometry()).unwrap();
        let chunk = &plan[0];
        assert!(chunk.attention_visible(2, 0, EmbeddingAttentionMode::Causal, None));
        assert!(!chunk.attention_visible(0, 2, EmbeddingAttentionMode::Causal, None));
        assert!(!chunk.attention_visible(2, 3, EmbeddingAttentionMode::Causal, None));
        assert!(!chunk.attention_visible(2, 128, EmbeddingAttentionMode::Causal, None));
        assert!(chunk.attention_visible(129, 128, EmbeddingAttentionMode::Causal, None));
    }

    #[test]
    fn bidirectional_sliding_attention_is_segment_local() {
        let plan = plan_embedding_dispatches(&[4, 4], &geometry()).unwrap();
        let chunk = &plan[0];
        assert!(chunk.attention_visible(0, 3, EmbeddingAttentionMode::Bidirectional, None));
        assert!(!chunk.attention_visible(
            0,
            3,
            EmbeddingAttentionMode::BidirectionalSliding,
            Some(3)
        ));
        assert!(chunk.attention_visible(
            1,
            3,
            EmbeddingAttentionMode::BidirectionalSliding,
            Some(3)
        ));
        assert!(!chunk.attention_visible(3, 128, EmbeddingAttentionMode::Bidirectional, None));
    }
}
