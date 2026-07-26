// SPDX-License-Identifier: Apache-2.0

//! Physical AIE2P buffer contract for EmbeddingGemma M256 attention.

/// Fixed M256/GQA layout shared by the fused QKV producer and R27 attention.
///
/// Q uses 4x8 BF16 MMUL tiles. K uses transposed dimension-by-key 8x8 tiles,
/// while V uses key-by-dimension 8x8 tiles. One packed K/V sequence is replayed
/// internally for all six Q groups by the R27 DMA schedule.
pub struct EmbeddingGemmaAttentionLayout;

impl EmbeddingGemmaAttentionLayout {
    pub const TOKENS: usize = 256;
    pub const QUERY_HEADS: usize = 3;
    pub const KV_HEADS: usize = 1;
    pub const HEAD_DIM: usize = 256;
    pub const CORE_ROWS: usize = 4;
    pub const CORE_COLS: usize = 8;
    pub const CORES: usize = Self::CORE_ROWS * Self::CORE_COLS;
    pub const QUERIES_PER_CORE: usize = 4;
    pub const QUERY_GROUPS: usize =
        Self::QUERY_HEADS * Self::TOKENS / (Self::CORES * Self::QUERIES_PER_CORE);
    pub const BLOCK_KEYS: usize = 16;
    pub const KEY_BLOCKS: usize = Self::TOKENS / Self::BLOCK_KEYS;
    pub const MMUL_K: usize = 8;
    pub const MMUL_N: usize = 8;
    pub const DIM_TILES: usize = Self::HEAD_DIM / Self::MMUL_K;
    pub const KEY_TILES: usize = Self::BLOCK_KEYS / Self::MMUL_N;
    pub const BF16_BYTES: usize = 2;

    pub const Q_TILE_BYTES: usize = Self::QUERIES_PER_CORE * Self::HEAD_DIM * Self::BF16_BYTES;
    pub const Q_JOIN_BYTES: usize = Self::CORE_COLS * Self::Q_TILE_BYTES;
    pub const Q_BYTES: usize = Self::CORE_ROWS * Self::QUERY_GROUPS * Self::Q_JOIN_BYTES;
    pub const KV_TILE_BYTES: usize = 2 * Self::BLOCK_KEYS * Self::HEAD_DIM * Self::BF16_BYTES;
    pub const KV_BYTES: usize = Self::KEY_BLOCKS * Self::KV_TILE_BYTES;
    pub const OUTPUT_JOIN_BYTES: usize = Self::CORE_ROWS * Self::Q_TILE_BYTES;
    pub const OUTPUT_BYTES: usize = Self::CORE_COLS * Self::QUERY_GROUPS * Self::OUTPUT_JOIN_BYTES;

    pub const fn q_offset(head: usize, token: usize, dim: usize) -> Option<usize> {
        if head >= Self::QUERY_HEADS || token >= Self::TOKENS || dim >= Self::HEAD_DIM {
            return None;
        }
        let linear = head * Self::TOKENS + token;
        let group = linear / (Self::CORES * Self::QUERIES_PER_CORE);
        let remainder = linear % (Self::CORES * Self::QUERIES_PER_CORE);
        let core = remainder / Self::QUERIES_PER_CORE;
        let lane = remainder % Self::QUERIES_PER_CORE;
        let row = core / Self::CORE_COLS;
        let col = core % Self::CORE_COLS;
        let tile =
            (row * Self::QUERY_GROUPS + group) * Self::Q_JOIN_BYTES + col * Self::Q_TILE_BYTES;
        Some(
            tile + (dim / Self::MMUL_K * Self::QUERIES_PER_CORE * Self::MMUL_K
                + lane * Self::MMUL_K
                + dim % Self::MMUL_K)
                * Self::BF16_BYTES,
        )
    }

    pub const fn k_offset(token: usize, dim: usize) -> Option<usize> {
        if token >= Self::TOKENS || dim >= Self::HEAD_DIM {
            return None;
        }
        let block = token / Self::BLOCK_KEYS;
        let key_lane = token % Self::BLOCK_KEYS;
        let key_tile = key_lane / Self::MMUL_N;
        let packed =
            ((key_tile * Self::DIM_TILES + dim / Self::MMUL_K) * Self::MMUL_K * Self::MMUL_N)
                + dim % Self::MMUL_K * Self::MMUL_N
                + key_lane % Self::MMUL_N;
        Some(block * Self::KV_TILE_BYTES + packed * Self::BF16_BYTES)
    }

    pub const fn v_offset(token: usize, dim: usize) -> Option<usize> {
        if token >= Self::TOKENS || dim >= Self::HEAD_DIM {
            return None;
        }
        let block = token / Self::BLOCK_KEYS;
        let key_lane = token % Self::BLOCK_KEYS;
        let key_tile = key_lane / Self::MMUL_K;
        let values =
            block * Self::KV_TILE_BYTES + Self::BLOCK_KEYS * Self::HEAD_DIM * Self::BF16_BYTES;
        let packed =
            ((dim / Self::MMUL_N * Self::KEY_TILES + key_tile) * Self::MMUL_K * Self::MMUL_N)
                + key_lane % Self::MMUL_K * Self::MMUL_N
                + dim % Self::MMUL_N;
        Some(values + packed * Self::BF16_BYTES)
    }

    pub const fn output_offset(head: usize, token: usize, dim: usize) -> Option<usize> {
        if head >= Self::QUERY_HEADS || token >= Self::TOKENS || dim >= Self::HEAD_DIM {
            return None;
        }
        let linear = head * Self::TOKENS + token;
        let group = linear / (Self::CORES * Self::QUERIES_PER_CORE);
        let remainder = linear % (Self::CORES * Self::QUERIES_PER_CORE);
        let core = remainder / Self::QUERIES_PER_CORE;
        let lane = remainder % Self::QUERIES_PER_CORE;
        let core_row = core / Self::CORE_COLS;
        let col = core % Self::CORE_COLS;
        Some(
            (col * Self::QUERY_GROUPS + group) * Self::OUTPUT_JOIN_BYTES
                + core_row * Self::Q_TILE_BYTES
                + lane * Self::HEAD_DIM * Self::BF16_BYTES
                + dim * Self::BF16_BYTES,
        )
    }

    pub fn pack_q_bf16(values: &[u16]) -> Option<Vec<u8>> {
        if values.len() != Self::QUERY_HEADS * Self::TOKENS * Self::HEAD_DIM {
            return None;
        }
        let mut packed = vec![0u8; Self::Q_BYTES];
        for head in 0..Self::QUERY_HEADS {
            for token in 0..Self::TOKENS {
                for dim in 0..Self::HEAD_DIM {
                    let source = (head * Self::TOKENS + token) * Self::HEAD_DIM + dim;
                    write_bf16(
                        &mut packed,
                        Self::q_offset(head, token, dim)?,
                        values[source],
                    );
                }
            }
        }
        Some(packed)
    }

    pub fn pack_kv_bf16(keys: &[u16], values: &[u16]) -> Option<Vec<u8>> {
        if keys.len() != Self::TOKENS * Self::HEAD_DIM || values.len() != keys.len() {
            return None;
        }
        let mut packed = vec![0u8; Self::KV_BYTES];
        for token in 0..Self::TOKENS {
            for dim in 0..Self::HEAD_DIM {
                let source = token * Self::HEAD_DIM + dim;
                write_bf16(&mut packed, Self::k_offset(token, dim)?, keys[source]);
                write_bf16(&mut packed, Self::v_offset(token, dim)?, values[source]);
            }
        }
        Some(packed)
    }

    pub fn unpack_output_bf16(bytes: &[u8]) -> Option<Vec<u16>> {
        if bytes.len() != Self::OUTPUT_BYTES {
            return None;
        }
        let mut output = vec![0u16; Self::QUERY_HEADS * Self::TOKENS * Self::HEAD_DIM];
        for head in 0..Self::QUERY_HEADS {
            for token in 0..Self::TOKENS {
                for dim in 0..Self::HEAD_DIM {
                    let offset = Self::output_offset(head, token, dim)?;
                    output[(head * Self::TOKENS + token) * Self::HEAD_DIM + dim] =
                        u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                }
            }
        }
        Some(output)
    }
}

fn write_bf16(destination: &mut [u8], offset: usize, value: u16) {
    destination[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::EmbeddingGemmaAttentionLayout as Layout;

    #[test]
    fn physical_sizes_match_r27_contract() {
        assert_eq!(Layout::Q_BYTES, 393_216);
        assert_eq!(Layout::KV_BYTES, 262_144);
        assert_eq!(Layout::OUTPUT_BYTES, 393_216);
    }

    #[test]
    fn q_and_output_mappings_are_bijective() {
        let mut q_seen = vec![false; Layout::Q_BYTES / 2];
        let mut output_seen = vec![false; Layout::OUTPUT_BYTES / 2];
        for head in 0..Layout::QUERY_HEADS {
            for token in 0..Layout::TOKENS {
                for dim in 0..Layout::HEAD_DIM {
                    let q = Layout::q_offset(head, token, dim).unwrap() / 2;
                    let output = Layout::output_offset(head, token, dim).unwrap() / 2;
                    assert!(!std::mem::replace(&mut q_seen[q], true));
                    assert!(!std::mem::replace(&mut output_seen[output], true));
                }
            }
        }
        assert!(q_seen.into_iter().all(|seen| seen));
        assert!(output_seen.into_iter().all(|seen| seen));
    }

    #[test]
    fn single_replay_kv_mapping_is_bijective() {
        let mut seen = vec![false; Layout::KV_BYTES / 2];
        for token in 0..Layout::TOKENS {
            for dim in 0..Layout::HEAD_DIM {
                let k = Layout::k_offset(token, dim).unwrap() / 2;
                let v = Layout::v_offset(token, dim).unwrap() / 2;
                assert!(!std::mem::replace(&mut seen[k], true));
                assert!(!std::mem::replace(&mut seen[v], true));
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn out_of_range_coordinates_are_rejected() {
        assert_eq!(Layout::q_offset(Layout::QUERY_HEADS, 0, 0), None);
        assert_eq!(Layout::k_offset(Layout::TOKENS, 0), None);
        assert_eq!(Layout::v_offset(0, Layout::HEAD_DIM), None);
        assert_eq!(Layout::output_offset(0, Layout::TOKENS, 0), None);
    }
}
