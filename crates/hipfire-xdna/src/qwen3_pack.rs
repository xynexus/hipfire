// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! NPU-side token-major to segmented-attention packing for Qwen3 encoders.

use crate::{DeviceBuffer, NpuKernel, SegmentedAttentionGeometry, XdnaError};

pub struct NpuQwen3QueryPack {
    kernel: NpuKernel,
    geometry: SegmentedAttentionGeometry,
    queries: DeviceBuffer,
    lengths: DeviceBuffer,
    packed: DeviceBuffer,
}

pub struct NpuQwen3KvPack {
    kernel: NpuKernel,
    geometry: SegmentedAttentionGeometry,
    keys: DeviceBuffer,
    values: DeviceBuffer,
    packed: DeviceBuffer,
}

pub struct NpuQwen3AttentionUnpack {
    kernel: NpuKernel,
    geometry: SegmentedAttentionGeometry,
    segmented: DeviceBuffer,
    token_major: DeviceBuffer,
}

impl NpuQwen3AttentionUnpack {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        geometry: SegmentedAttentionGeometry,
    ) -> Result<Self, XdnaError> {
        let geometry = geometry.validate().map_err(invalid)?;
        let kernel = NpuKernel::load(xclbin, instructions)?;
        Ok(Self {
            segmented: kernel.alloc_arg(geometry.output_bytes())?,
            token_major: kernel.alloc_arg(geometry.output_bytes())?,
            kernel,
            geometry,
        })
    }

    pub fn geometry(&self) -> SegmentedAttentionGeometry {
        self.geometry
    }

    /// Convert the segmented-attention physical output into token-major
    /// `[B,S,QH,D]` BF16 values for the output projection.
    pub fn run(&mut self, segmented: &[u8]) -> Result<Vec<u16>, XdnaError> {
        if segmented.len() != self.geometry.output_bytes() {
            return Err(invalid(format!(
                "Qwen3 attention unpack has {} bytes; expected {}",
                segmented.len(),
                self.geometry.output_bytes()
            )));
        }
        self.segmented.as_mut_slice().copy_from_slice(segmented);
        self.token_major.as_mut_slice().fill(0);
        self.kernel.recreate_hwctx()?;
        self.kernel
            .dispatch_synced(&[&self.segmented, &self.token_major], &[true, true])?;
        self.kernel.sync_output(&self.token_major)?;
        Ok(self
            .token_major
            .as_slice()
            .chunks_exact(size_of::<u16>())
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect())
    }
}

impl NpuQwen3KvPack {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        geometry: SegmentedAttentionGeometry,
    ) -> Result<Self, XdnaError> {
        let geometry = geometry.validate().map_err(invalid)?;
        let elements = geometry.dispatch_batch
            * geometry.sequence_bucket
            * geometry.kv_heads
            * geometry.head_dim;
        let kernel = NpuKernel::load(xclbin, instructions)?;
        Ok(Self {
            keys: kernel.alloc_arg(elements * size_of::<u16>())?,
            values: kernel.alloc_arg(elements * size_of::<u16>())?,
            packed: kernel.alloc_arg(geometry.kv_bytes())?,
            kernel,
            geometry,
        })
    }

    pub fn geometry(&self) -> SegmentedAttentionGeometry {
        self.geometry
    }

    /// Pack token-major `[B,S,Hkv,D]` BF16 keys and values into the
    /// segmented-attention block layout.
    pub fn run(
        &mut self,
        token_major_keys: &[u16],
        token_major_values: &[u16],
    ) -> Result<Vec<u8>, XdnaError> {
        let expected = self.geometry.dispatch_batch
            * self.geometry.sequence_bucket
            * self.geometry.kv_heads
            * self.geometry.head_dim;
        if token_major_keys.len() != expected || token_major_values.len() != expected {
            return Err(invalid(format!(
                "Qwen3 K/V pack has {}/{} values; expected {expected} each",
                token_major_keys.len(),
                token_major_values.len()
            )));
        }
        write_bf16(&mut self.keys, token_major_keys);
        write_bf16(&mut self.values, token_major_values);
        self.packed.as_mut_slice().fill(0);
        self.kernel.recreate_hwctx()?;
        self.kernel.dispatch_synced(
            &[&self.keys, &self.values, &self.packed],
            &[true, true, true],
        )?;
        self.kernel.sync_output(&self.packed)?;
        Ok(self.packed.as_slice().to_vec())
    }
}

impl NpuQwen3QueryPack {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        geometry: SegmentedAttentionGeometry,
    ) -> Result<Self, XdnaError> {
        let geometry = geometry.validate().map_err(invalid)?;
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let query_elements = geometry.dispatch_batch
            * geometry.sequence_bucket
            * geometry.query_heads
            * geometry.head_dim;
        Ok(Self {
            queries: kernel.alloc_arg(query_elements * size_of::<u16>())?,
            lengths: kernel.alloc_arg(geometry.dispatch_batch * size_of::<u32>())?,
            packed: kernel.alloc_arg(geometry.q_bytes())?,
            kernel,
            geometry,
        })
    }

    pub fn geometry(&self) -> SegmentedAttentionGeometry {
        self.geometry
    }

    /// Pack token-major `[B,S,QH,D]` BF16 queries into the segmented-attention
    /// physical layout, including one real-length trailer per Q core pair.
    pub fn run(
        &mut self,
        token_major_queries: &[u16],
        real_lengths: &[u32],
    ) -> Result<Vec<u8>, XdnaError> {
        let expected = self.geometry.dispatch_batch
            * self.geometry.sequence_bucket
            * self.geometry.query_heads
            * self.geometry.head_dim;
        if token_major_queries.len() != expected {
            return Err(invalid(format!(
                "Qwen3 query pack has {} values; expected {expected}",
                token_major_queries.len()
            )));
        }
        validate_lengths(self.geometry, real_lengths)?;
        write_bf16(&mut self.queries, token_major_queries);
        for (bytes, &value) in self
            .lengths
            .as_mut_slice()
            .chunks_exact_mut(size_of::<u32>())
            .zip(real_lengths)
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        self.packed.as_mut_slice().fill(0);
        self.kernel.recreate_hwctx()?;
        self.kernel.dispatch_synced(
            &[&self.queries, &self.lengths, &self.packed],
            &[true, true, true],
        )?;
        self.kernel.sync_output(&self.packed)?;
        Ok(self.packed.as_slice().to_vec())
    }
}

fn write_bf16(buffer: &mut DeviceBuffer, values: &[u16]) {
    for (bytes, &value) in buffer
        .as_mut_slice()
        .chunks_exact_mut(size_of::<u16>())
        .zip(values)
    {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn validate_lengths(
    geometry: SegmentedAttentionGeometry,
    lengths: &[u32],
) -> Result<(), XdnaError> {
    if lengths.len() != geometry.dispatch_batch {
        return Err(invalid(format!(
            "Qwen3 pack has {} lengths; expected {}",
            lengths.len(),
            geometry.dispatch_batch
        )));
    }
    for (document, &length) in lengths.iter().enumerate() {
        if length == 0 || length as usize > geometry.sequence_bucket {
            return Err(invalid(format!(
                "Qwen3 pack length[{document}]={length} is outside 1..={}",
                geometry.sequence_bucket
            )));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}
