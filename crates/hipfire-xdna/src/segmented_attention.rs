// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Geometry-driven physical buffer contract for segmented Qwen3 attention.
//!
//! The AIE2P graph maps one KV head to each of the eight columns. The two Q
//! heads sharing that KV head stay on the same column; token groups are spread
//! over the four core rows. Every document owns disjoint Q, K/V, and output
//! regions, so real-length and causal masks never address another document.

#[cfg(target_os = "linux")]
use crate::{DeviceBuffer, NpuKernel, XdnaError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedAttentionGeometry {
    pub sequence_bucket: usize,
    pub dispatch_batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl SegmentedAttentionGeometry {
    pub const CORE_ROWS: usize = 4;
    pub const CORE_COLS: usize = 8;
    pub const QUERIES_PER_CORE: usize = 4;
    pub const BLOCK_KEYS: usize = 16;
    pub const MMUL_K: usize = 8;
    pub const MMUL_N: usize = 8;
    pub const BF16_BYTES: usize = 2;
    pub const LENGTH_TRAILER_BYTES: usize = 512;
    pub const COMPILED_BUCKETS: [usize; 5] = [128, 256, 512, 1024, 2048];

    pub fn validate(self) -> Result<Self, String> {
        if !Self::COMPILED_BUCKETS.contains(&self.sequence_bucket) {
            return Err(format!(
                "segmented attention bucket {} is not one of {:?}",
                self.sequence_bucket,
                Self::COMPILED_BUCKETS
            ));
        }
        if self.dispatch_batch == 0
            || self
                .sequence_bucket
                .checked_mul(self.dispatch_batch)
                .is_none_or(|rows| rows > 4096)
        {
            return Err("segmented attention dispatch must contain 1..=4096 padded rows".into());
        }
        let q_heads_per_kv = self.query_heads.checked_div(self.kv_heads);
        if self.kv_heads != Self::CORE_COLS
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !matches!(q_heads_per_kv, Some(2 | 4))
        {
            return Err(format!(
                "segmented Qwen3 attention requires 8 KV heads and 2 or 4 query heads per KV head, got {}/{}",
                self.query_heads, self.kv_heads
            ));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(Self::MMUL_K) {
            return Err(format!(
                "segmented attention head_dim {} must be a positive multiple of {}",
                self.head_dim,
                Self::MMUL_K
            ));
        }
        Ok(self)
    }

    pub fn query_groups_per_document(self) -> usize {
        self.sequence_bucket / (Self::CORE_ROWS * Self::QUERIES_PER_CORE)
            * (self.query_heads / self.kv_heads)
    }

    pub fn key_blocks(self) -> usize {
        self.sequence_bucket / Self::BLOCK_KEYS
    }

    pub fn q_tile_bytes(self) -> usize {
        Self::QUERIES_PER_CORE * self.head_dim * Self::BF16_BYTES
    }

    pub fn q_pair_bytes(self) -> usize {
        2 * self.q_tile_bytes() + Self::LENGTH_TRAILER_BYTES
    }

    pub fn q_join_bytes(self) -> usize {
        Self::CORE_COLS / 2 * self.q_pair_bytes()
    }

    pub fn q_document_bytes(self) -> usize {
        Self::CORE_ROWS * self.query_groups_per_document() * self.q_join_bytes()
    }

    pub fn q_bytes(self) -> usize {
        self.dispatch_batch * self.q_document_bytes()
    }

    pub fn kv_tile_bytes(self) -> usize {
        2 * Self::BLOCK_KEYS * self.head_dim * Self::BF16_BYTES
    }

    pub fn kv_head_bytes(self) -> usize {
        self.key_blocks() * self.kv_tile_bytes()
    }

    pub fn kv_document_bytes(self) -> usize {
        self.kv_heads * self.kv_head_bytes()
    }

    pub fn kv_bytes(self) -> usize {
        self.dispatch_batch * self.kv_document_bytes()
    }

    pub fn output_document_bytes(self) -> usize {
        self.sequence_bucket * self.query_heads * self.head_dim * Self::BF16_BYTES
    }

    pub fn output_bytes(self) -> usize {
        self.dispatch_batch * self.output_document_bytes()
    }

    fn query_location(self, head: usize, token: usize) -> (usize, usize, usize, usize) {
        let q_per_kv = self.query_heads / self.kv_heads;
        let col = head / q_per_kv;
        let q_local = head % q_per_kv;
        let token_chunk = token / (Self::CORE_ROWS * Self::QUERIES_PER_CORE);
        let row = token % (Self::CORE_ROWS * Self::QUERIES_PER_CORE) / Self::QUERIES_PER_CORE;
        let lane = token % Self::QUERIES_PER_CORE;
        let token_chunks_per_row =
            self.sequence_bucket / (Self::CORE_ROWS * Self::QUERIES_PER_CORE);
        let group = q_local * token_chunks_per_row + token_chunk;
        (row, col, group, lane)
    }

    pub fn q_offset(self, document: usize, head: usize, token: usize, dim: usize) -> Option<usize> {
        if !self.valid_q_coordinate(document, head, token, dim) {
            return None;
        }
        let (row, col, group, lane) = self.query_location(head, token);
        let tile = document * self.q_document_bytes()
            + (row * self.query_groups_per_document() + group) * self.q_join_bytes()
            + col / 2 * self.q_pair_bytes()
            + col % 2 * self.q_tile_bytes();
        Some(
            tile + (dim / Self::MMUL_K * Self::QUERIES_PER_CORE * Self::MMUL_K
                + lane * Self::MMUL_K
                + dim % Self::MMUL_K)
                * Self::BF16_BYTES,
        )
    }

    pub fn k_offset(self, document: usize, head: usize, token: usize, dim: usize) -> Option<usize> {
        self.kv_offset(document, head, token, dim, false)
    }

    pub fn v_offset(self, document: usize, head: usize, token: usize, dim: usize) -> Option<usize> {
        self.kv_offset(document, head, token, dim, true)
    }

    fn kv_offset(
        self,
        document: usize,
        head: usize,
        token: usize,
        dim: usize,
        value: bool,
    ) -> Option<usize> {
        if document >= self.dispatch_batch
            || head >= self.kv_heads
            || token >= self.sequence_bucket
            || dim >= self.head_dim
        {
            return None;
        }
        let block = token / Self::BLOCK_KEYS;
        let key_lane = token % Self::BLOCK_KEYS;
        let key_tile = key_lane / Self::MMUL_N;
        let dim_tiles = self.head_dim / Self::MMUL_K;
        let key_tiles = Self::BLOCK_KEYS / Self::MMUL_N;
        let block_base = document * self.kv_document_bytes()
            + head * self.kv_head_bytes()
            + block * self.kv_tile_bytes();
        let packed = if value {
            Self::BLOCK_KEYS * self.head_dim
                + (dim / Self::MMUL_N * key_tiles + key_tile) * Self::MMUL_K * Self::MMUL_N
                + key_lane % Self::MMUL_K * Self::MMUL_N
                + dim % Self::MMUL_N
        } else {
            (key_tile * dim_tiles + dim / Self::MMUL_K) * Self::MMUL_K * Self::MMUL_N
                + dim % Self::MMUL_K * Self::MMUL_N
                + key_lane % Self::MMUL_N
        };
        Some(block_base + packed * Self::BF16_BYTES)
    }

    pub fn output_offset(
        self,
        document: usize,
        head: usize,
        token: usize,
        dim: usize,
    ) -> Option<usize> {
        if !self.valid_q_coordinate(document, head, token, dim) {
            return None;
        }
        let (row, col, group, lane) = self.query_location(head, token);
        let output_join_bytes = Self::CORE_ROWS * self.q_tile_bytes();
        Some(
            document * self.output_document_bytes()
                + (col * self.query_groups_per_document() + group) * output_join_bytes
                + row * self.q_tile_bytes()
                + (lane * self.head_dim + dim) * Self::BF16_BYTES,
        )
    }

    fn valid_q_coordinate(self, document: usize, head: usize, token: usize, dim: usize) -> bool {
        document < self.dispatch_batch
            && head < self.query_heads
            && token < self.sequence_bucket
            && dim < self.head_dim
    }

    pub fn pack_q_bf16(self, values: &[u16]) -> Result<Vec<u8>, String> {
        self.pack_q_bf16_with_lengths(
            values,
            &vec![self.sequence_bucket as u32; self.dispatch_batch],
        )
    }

    pub fn pack_q_bf16_with_lengths(
        self,
        values: &[u16],
        real_lengths: &[u32],
    ) -> Result<Vec<u8>, String> {
        let expected =
            self.dispatch_batch * self.query_heads * self.sequence_bucket * self.head_dim;
        if values.len() != expected {
            return Err(format!(
                "segmented attention Q has {} values; expected {expected}",
                values.len()
            ));
        }
        if real_lengths.len() != self.dispatch_batch {
            return Err(format!(
                "segmented attention has {} lengths; expected {}",
                real_lengths.len(),
                self.dispatch_batch
            ));
        }
        for (document, &length) in real_lengths.iter().enumerate() {
            if length == 0 || length as usize > self.sequence_bucket {
                return Err(format!(
                    "segmented attention length[{document}]={length} is outside 1..={}",
                    self.sequence_bucket
                ));
            }
        }
        let mut packed = vec![0u8; self.q_bytes()];
        for document in 0..self.dispatch_batch {
            for head in 0..self.query_heads {
                for token in 0..self.sequence_bucket {
                    for dim in 0..self.head_dim {
                        let source =
                            (((document * self.query_heads + head) * self.sequence_bucket + token)
                                * self.head_dim)
                                + dim;
                        write_bf16(
                            &mut packed,
                            self.q_offset(document, head, token, dim).unwrap(),
                            values[source],
                        );
                    }
                }
            }
        }
        for (document, &length) in real_lengths.iter().enumerate() {
            for row in 0..Self::CORE_ROWS {
                for group in 0..self.query_groups_per_document() {
                    for pair in 0..Self::CORE_COLS / 2 {
                        let offset = document * self.q_document_bytes()
                            + (row * self.query_groups_per_document() + group)
                                * self.q_join_bytes()
                            + pair * self.q_pair_bytes()
                            + 2 * self.q_tile_bytes();
                        packed[offset..offset + size_of::<u32>()]
                            .copy_from_slice(&length.to_le_bytes());
                    }
                }
            }
        }
        Ok(packed)
    }

    pub fn pack_kv_bf16(self, keys: &[u16], values: &[u16]) -> Result<Vec<u8>, String> {
        let expected = self.dispatch_batch * self.kv_heads * self.sequence_bucket * self.head_dim;
        if keys.len() != expected || values.len() != expected {
            return Err(format!(
                "segmented attention K/V have {}/{} values; expected {expected} each",
                keys.len(),
                values.len()
            ));
        }
        let mut packed = vec![0u8; self.kv_bytes()];
        for document in 0..self.dispatch_batch {
            for head in 0..self.kv_heads {
                for token in 0..self.sequence_bucket {
                    for dim in 0..self.head_dim {
                        let source = (((document * self.kv_heads + head) * self.sequence_bucket
                            + token)
                            * self.head_dim)
                            + dim;
                        write_bf16(
                            &mut packed,
                            self.k_offset(document, head, token, dim).unwrap(),
                            keys[source],
                        );
                        write_bf16(
                            &mut packed,
                            self.v_offset(document, head, token, dim).unwrap(),
                            values[source],
                        );
                    }
                }
            }
        }
        Ok(packed)
    }

    pub fn unpack_output_bf16(self, bytes: &[u8]) -> Result<Vec<u16>, String> {
        if bytes.len() != self.output_bytes() {
            return Err(format!(
                "segmented attention output has {} bytes; expected {}",
                bytes.len(),
                self.output_bytes()
            ));
        }
        let mut output =
            vec![
                0u16;
                self.dispatch_batch * self.query_heads * self.sequence_bucket * self.head_dim
            ];
        for document in 0..self.dispatch_batch {
            for head in 0..self.query_heads {
                for token in 0..self.sequence_bucket {
                    for dim in 0..self.head_dim {
                        let destination =
                            (((document * self.query_heads + head) * self.sequence_bucket + token)
                                * self.head_dim)
                                + dim;
                        let offset = self.output_offset(document, head, token, dim).unwrap();
                        output[destination] =
                            u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn pack_output_bf16(self, values: &[u16]) -> Result<Vec<u8>, String> {
        let expected =
            self.dispatch_batch * self.query_heads * self.sequence_bucket * self.head_dim;
        if values.len() != expected {
            return Err(format!(
                "segmented attention canonical output has {} values; expected {expected}",
                values.len()
            ));
        }
        let mut packed = vec![0u8; self.output_bytes()];
        for document in 0..self.dispatch_batch {
            for head in 0..self.query_heads {
                for token in 0..self.sequence_bucket {
                    for dim in 0..self.head_dim {
                        let source =
                            (((document * self.query_heads + head) * self.sequence_bucket + token)
                                * self.head_dim)
                                + dim;
                        let offset = self.output_offset(document, head, token, dim).unwrap();
                        write_bf16(&mut packed, offset, values[source]);
                    }
                }
            }
        }
        Ok(packed)
    }
}

fn write_bf16(destination: &mut [u8], offset: usize, value: u16) {
    destination[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Loaded segmented-attention image and its reusable argument buffers.
///
/// The image is geometry-specific. Callers must load the `final.xclbin` and
/// `insts.bin` produced for the same bucket and batch as `geometry`.
#[cfg(target_os = "linux")]
pub struct NpuSegmentedAttention {
    kernel: NpuKernel,
    geometry: SegmentedAttentionGeometry,
    queries: DeviceBuffer,
    key_values: DeviceBuffer,
    output: DeviceBuffer,
}

#[cfg(target_os = "linux")]
impl NpuSegmentedAttention {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        geometry: SegmentedAttentionGeometry,
    ) -> Result<Self, XdnaError> {
        let geometry = geometry.validate().map_err(invalid)?;
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let queries = kernel.alloc_arg(geometry.q_bytes())?;
        let key_values = kernel.alloc_arg(geometry.kv_bytes())?;
        let output = kernel.alloc_arg(geometry.output_bytes())?;
        Ok(Self {
            kernel,
            geometry,
            queries,
            key_values,
            output,
        })
    }

    pub fn geometry(&self) -> SegmentedAttentionGeometry {
        self.geometry
    }

    /// Run causal grouped-query attention over one padded document batch.
    ///
    /// Q is canonical `[B, 16, S, D]`; K/V are canonical `[B, 8, S, D]`.
    /// Padded query rows are returned as zero and keys beyond each real length
    /// are never observed by the image.
    pub fn run(
        &mut self,
        queries: &[u16],
        keys: &[u16],
        values: &[u16],
        real_lengths: &[u32],
    ) -> Result<Vec<u16>, XdnaError> {
        let packed_q = self
            .geometry
            .pack_q_bf16_with_lengths(queries, real_lengths)
            .map_err(invalid)?;
        let packed_kv = self.geometry.pack_kv_bf16(keys, values).map_err(invalid)?;
        let packed_output = self.run_packed(&packed_q, &packed_kv)?;
        self.geometry
            .unpack_output_bf16(&packed_output)
            .map_err(invalid)
    }

    /// Run the compiled attention image at its physical packed boundary.
    /// This is the zero-layout-work seam used by NPU Q/K/V packers and the
    /// NPU token-major output unpacker.
    pub fn run_packed(
        &mut self,
        packed_queries: &[u8],
        packed_key_values: &[u8],
    ) -> Result<Vec<u8>, XdnaError> {
        if packed_queries.len() != self.geometry.q_bytes()
            || packed_key_values.len() != self.geometry.kv_bytes()
        {
            return Err(invalid(format!(
                "segmented attention packed inputs have {}/{} bytes; expected {}/{}",
                packed_queries.len(),
                packed_key_values.len(),
                self.geometry.q_bytes(),
                self.geometry.kv_bytes()
            )));
        }
        self.queries.as_mut_slice().copy_from_slice(packed_queries);
        self.key_values
            .as_mut_slice()
            .copy_from_slice(packed_key_values);
        self.kernel.recreate_hwctx()?;
        self.kernel.dispatch_synced(
            &[&self.queries, &self.key_values, &self.output],
            &[true, true, true],
        )?;
        self.kernel.sync_output(&self.output)?;
        Ok(self.output.as_slice().to_vec())
    }
}

#[cfg(target_os = "linux")]
fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> SegmentedAttentionGeometry {
        SegmentedAttentionGeometry {
            sequence_bucket: 128,
            dispatch_batch: 2,
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn qwen_geometry_covers_compiled_buckets_and_row_ceiling() {
        for sequence_bucket in SegmentedAttentionGeometry::COMPILED_BUCKETS {
            let max_batch = 4096 / sequence_bucket;
            assert!(SegmentedAttentionGeometry {
                sequence_bucket,
                dispatch_batch: max_batch,
                ..geometry()
            }
            .validate()
            .is_ok());
            assert!(SegmentedAttentionGeometry {
                sequence_bucket,
                dispatch_batch: max_batch + 1,
                ..geometry()
            }
            .validate()
            .is_err());
        }
    }

    #[test]
    fn geometry_accepts_both_qwen3_grouped_query_topologies() {
        for query_heads in [16, 32] {
            let value = SegmentedAttentionGeometry {
                query_heads,
                ..geometry()
            };
            assert_eq!(value.validate().unwrap(), value);
            assert_eq!(
                value.query_groups_per_document(),
                value.sequence_bucket / 16 * (query_heads / value.kv_heads)
            );
        }
    }

    #[test]
    fn physical_sizes_match_canonical_qwen_tensors() {
        let geometry = geometry();
        assert_eq!(geometry.query_groups_per_document(), 16);
        assert_eq!(geometry.key_blocks(), 8);
        assert_eq!(geometry.q_bytes(), 1_310_720);
        assert_eq!(geometry.kv_bytes(), 1_048_576);
        assert_eq!(geometry.output_bytes(), 1_048_576);
    }

    #[test]
    fn q_k_v_and_output_mappings_are_bijective() {
        let geometry = SegmentedAttentionGeometry {
            dispatch_batch: 1,
            ..geometry()
        };
        let mut q = vec![false; geometry.q_bytes() / 2];
        let mut kv = vec![false; geometry.kv_bytes() / 2];
        let mut output = vec![false; geometry.output_bytes() / 2];
        for head in 0..geometry.query_heads {
            for token in 0..geometry.sequence_bucket {
                for dim in 0..geometry.head_dim {
                    let offset = geometry.q_offset(0, head, token, dim).unwrap() / 2;
                    assert!(!std::mem::replace(&mut q[offset], true));
                    let offset = geometry.output_offset(0, head, token, dim).unwrap() / 2;
                    assert!(!std::mem::replace(&mut output[offset], true));
                }
            }
        }
        for head in 0..geometry.kv_heads {
            for token in 0..geometry.sequence_bucket {
                for dim in 0..geometry.head_dim {
                    for offset in [
                        geometry.k_offset(0, head, token, dim).unwrap() / 2,
                        geometry.v_offset(0, head, token, dim).unwrap() / 2,
                    ] {
                        assert!(!std::mem::replace(&mut kv[offset], true));
                    }
                }
            }
        }
        let trailer_words = trailer_words(geometry);
        assert_eq!(q.into_iter().filter(|value| !value).count(), trailer_words);
        assert!(kv.into_iter().all(|value| value));
        assert!(output.into_iter().all(|value| value));
    }

    #[test]
    fn output_pack_and_unpack_round_trip() {
        let geometry = geometry();
        let values = (0..geometry.output_bytes() / 2)
            .map(|index| index.wrapping_mul(251) as u16)
            .collect::<Vec<_>>();
        let packed = geometry.pack_output_bf16(&values).unwrap();
        assert_eq!(geometry.unpack_output_bf16(&packed).unwrap(), values);
    }

    #[test]
    fn q_pack_and_output_unpack_preserve_canonical_order() {
        let geometry = SegmentedAttentionGeometry {
            dispatch_batch: 1,
            ..geometry()
        };
        let values = (0..geometry.dispatch_batch
            * geometry.query_heads
            * geometry.sequence_bucket
            * geometry.head_dim)
            .map(|index| index as u16)
            .collect::<Vec<_>>();
        let packed = geometry.pack_q_bf16(&values).unwrap();
        let mut physical_output = vec![0u8; geometry.output_bytes()];
        for head in 0..geometry.query_heads {
            for token in 0..geometry.sequence_bucket {
                for dim in 0..geometry.head_dim {
                    let source =
                        (head * geometry.sequence_bucket + token) * geometry.head_dim + dim;
                    let q_offset = geometry.q_offset(0, head, token, dim).unwrap();
                    assert_eq!(
                        u16::from_le_bytes([packed[q_offset], packed[q_offset + 1]]),
                        values[source]
                    );
                    write_bf16(
                        &mut physical_output,
                        geometry.output_offset(0, head, token, dim).unwrap(),
                        values[source],
                    );
                }
            }
        }
        assert_eq!(
            geometry.unpack_output_bf16(&physical_output).unwrap(),
            values
        );
        for row in 0..SegmentedAttentionGeometry::CORE_ROWS {
            for group in 0..geometry.query_groups_per_document() {
                for pair in 0..SegmentedAttentionGeometry::CORE_COLS / 2 {
                    let offset = (row * geometry.query_groups_per_document() + group)
                        * geometry.q_join_bytes()
                        + pair * geometry.q_pair_bytes()
                        + 2 * geometry.q_tile_bytes();
                    assert_eq!(
                        u32::from_le_bytes(packed[offset..offset + 4].try_into().unwrap()),
                        geometry.sequence_bucket as u32
                    );
                }
            }
        }
    }

    #[test]
    fn unsupported_head_topology_and_coordinates_are_rejected() {
        assert!(SegmentedAttentionGeometry {
            query_heads: 8,
            kv_heads: 8,
            ..geometry()
        }
        .validate()
        .is_err());
        let geometry = geometry();
        assert_eq!(geometry.q_offset(2, 0, 0, 0), None);
        assert_eq!(geometry.k_offset(0, 8, 0, 0), None);
        assert_eq!(geometry.v_offset(0, 0, 128, 0), None);
        assert_eq!(geometry.output_offset(0, 0, 0, 128), None);
    }

    fn trailer_words(geometry: SegmentedAttentionGeometry) -> usize {
        geometry.dispatch_batch
            * SegmentedAttentionGeometry::CORE_ROWS
            * geometry.query_groups_per_document()
            * (SegmentedAttentionGeometry::CORE_COLS / 2)
            * (SegmentedAttentionGeometry::LENGTH_TRAILER_BYTES / size_of::<u16>())
    }
}
