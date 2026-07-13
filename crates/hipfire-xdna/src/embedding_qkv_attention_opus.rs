// SPDX-License-Identifier: Apache-2.0

//! Resident compact-Opus QKV projection, head packing, and attention.
//!
//! Immutable tensor-block order comes directly from a validated `.rdna2.hfp`
//! payload. The destination hardware context owns the uploaded weight BO for
//! its lifetime; repeated dispatches only refresh mutable activation/state
//! arguments. Local nibble decode and lane swizzle remain kernel work.

use std::path::Path;

use crate::opus_hfp::{self, OpusHfpEncoding, OpusHfpLayout};
use crate::{DeviceBuffer, NpuKernel, XdnaError};

const M: usize = 256;
const K: usize = 768;
const N: usize = 1280;
const ACTIVATION_BYTES: usize = 737_280;
const WEIGHT_BYTES: usize = 2_359_296;
const STAGE_BYTES: usize = 2_457_600;
const ATTENTION_BYTES: usize = M * K * size_of::<u16>();
const RESULT_BYTES: usize = STAGE_BYTES + ATTENTION_BYTES;
const Q_BYTES: usize = 3 * ATTENTION_BYTES;
const KV_BYTES: usize = 2 * ATTENTION_BYTES;
const MAX_CONTEXT_COMMANDS: usize = 1_000;

pub struct NpuEmbeddingQkvAttentionOpusWeights {
    weights: DeviceBuffer,
}

pub struct NpuEmbeddingQkvAttentionOpusOutput {
    pub result: Vec<u8>,
    pub queries: Vec<u8>,
    pub key_values: Vec<u8>,
}

/// One R76-compatible AIE2P hardware context with a destination-owned compact
/// Opus weight BO.
pub struct NpuEmbeddingQkvAttentionOpus {
    kernel: NpuKernel,
    activations: DeviceBuffer,
    result: DeviceBuffer,
    queries: DeviceBuffer,
    key_values: DeviceBuffer,
    primed: bool,
    context_commands: usize,
}

impl NpuEmbeddingQkvAttentionOpus {
    pub fn load_cached(cache: &str) -> Result<Self, XdnaError> {
        let manifest =
            std::fs::read_to_string(format!("{cache}/shape.txt")).map_err(XdnaError::Open)?;
        for field in [
            "op=resident-opus-qkv-projection-headnorm-rope-pack-attention",
            "mode=w4-scaled",
            "m=256",
            "k=768",
            "n=1280",
            "contexts=1",
            "attention_enqueue_window=3",
            "weights=qkv-compact-rdna2-hfp",
        ] {
            if !manifest.lines().any(|line| line == field) {
                return Err(invalid(format!("resident Opus QKV cache missing {field}")));
            }
        }
        let kernel = NpuKernel::load(
            &std::fs::read(format!("{cache}/final.xclbin")).map_err(XdnaError::Open)?,
            &std::fs::read(format!("{cache}/insts.bin")).map_err(XdnaError::Open)?,
        )?;
        Ok(Self {
            activations: kernel.alloc_arg(ACTIVATION_BYTES)?,
            result: kernel.alloc_arg(RESULT_BYTES)?,
            queries: kernel.alloc_arg(Q_BYTES)?,
            key_values: kernel.alloc_arg(KV_BYTES)?,
            kernel,
            primed: false,
            context_commands: 0,
        })
    }

    pub const fn activation_bytes() -> usize {
        ACTIVATION_BYTES
    }

    pub const fn stage_bytes() -> usize {
        STAGE_BYTES
    }

    pub const fn result_bytes() -> usize {
        RESULT_BYTES
    }

    /// Validate and upload one already-converted QKV HFP payload. The returned
    /// BO is allocated by this executor's kernel context and remains resident
    /// until the layer weights are dropped.
    pub fn upload_weights_prepacked(
        &self,
        path: &Path,
    ) -> Result<NpuEmbeddingQkvAttentionOpusWeights, XdnaError> {
        let (descriptor, payload) = opus_hfp::read_existing(path).map_err(invalid)?;
        let valid = descriptor.encoding == OpusHfpEncoding::W4
            && descriptor.layout == OpusHfpLayout::WholeScaledV1
            && descriptor.m == M as u32
            && descriptor.k == K as u32
            && descriptor.n == N as u32
            && descriptor.columns == 8
            && descriptor.groups == 3
            && descriptor.m_macros == 3
            && descriptor.n_macros == 2
            && descriptor.outblocks == 6
            && descriptor.tile_bytes == 16_384
            && descriptor.payload_bytes == WEIGHT_BYTES as u64
            && payload.len() == WEIGHT_BYTES;
        if !valid {
            return Err(invalid(
                "resident Opus QKV HFP descriptor does not match the R76 schedule",
            ));
        }
        let mut weights = self.kernel.alloc_arg(WEIGHT_BYTES)?;
        weights.as_mut_slice().copy_from_slice(&payload);
        self.kernel.sync_to_device(&weights)?;
        Ok(NpuEmbeddingQkvAttentionOpusWeights { weights })
    }

    /// Replace mutable input/state bytes without touching the resident weight
    /// BO. The state seed contains only mutable result storage and the
    /// loader-prepared norm/RoPE tails used by the fused graph.
    pub fn set_input(&mut self, activations: &[u8], stage_seed: &[u8]) -> Result<(), XdnaError> {
        if activations.len() != ACTIVATION_BYTES || stage_seed.len() != STAGE_BYTES {
            return Err(invalid("resident Opus QKV input geometry mismatch"));
        }
        self.activations.as_mut_slice().copy_from_slice(activations);
        self.result.as_mut_slice()[..STAGE_BYTES].copy_from_slice(stage_seed);
        self.result.as_mut_slice()[STAGE_BYTES..].fill(0);
        self.queries.as_mut_slice().fill(0);
        self.key_values.as_mut_slice().fill(0);
        self.kernel.sync_to_device(&self.activations)?;
        self.kernel.sync_to_device(&self.result)?;
        self.kernel.sync_to_device(&self.queries)?;
        self.kernel.sync_to_device(&self.key_values)?;
        self.primed = false;
        Ok(())
    }

    pub fn run(&mut self, weights: &NpuEmbeddingQkvAttentionOpusWeights) -> Result<(), XdnaError> {
        if weights.weights.len() != WEIGHT_BYTES {
            return Err(invalid("resident Opus QKV weight geometry mismatch"));
        }
        if self.context_commands >= MAX_CONTEXT_COMMANDS {
            self.kernel.recreate_hwctx()?;
            self.primed = false;
            self.context_commands = 0;
        }
        if !self.primed {
            self.dispatch(weights, true)?;
            self.context_commands += 1;
            self.queries.as_mut_slice().fill(0);
            self.key_values.as_mut_slice().fill(0);
            self.result.as_mut_slice()[STAGE_BYTES..].fill(0);
            self.kernel.sync_to_device(&self.queries)?;
            self.kernel.sync_to_device(&self.key_values)?;
            self.kernel.sync_to_device(&self.result)?;
            self.primed = true;
        }
        self.dispatch(weights, false)?;
        self.context_commands += 1;
        Ok(())
    }

    pub fn read_output(&self) -> Result<NpuEmbeddingQkvAttentionOpusOutput, XdnaError> {
        self.kernel.sync_output(&self.result)?;
        self.kernel.sync_output(&self.queries)?;
        self.kernel.sync_output(&self.key_values)?;
        Ok(NpuEmbeddingQkvAttentionOpusOutput {
            result: self.result.as_slice().to_vec(),
            queries: self.queries.as_slice().to_vec(),
            key_values: self.key_values.as_slice().to_vec(),
        })
    }

    fn dispatch(
        &self,
        weights: &NpuEmbeddingQkvAttentionOpusWeights,
        sync: bool,
    ) -> Result<(), XdnaError> {
        self.kernel.dispatch_synced(
            &[
                &self.activations,
                &weights.weights,
                &self.result,
                &self.queries,
                &self.key_values,
            ],
            &[sync, sync, sync, sync, sync],
        )
    }
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}
