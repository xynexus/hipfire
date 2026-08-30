// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The `Architecture` impl — what the daemon loads a Qwen3.8-Flash-Next artifact
//! through.
//!
//! Weights come off the HFQ one tensor at a time via [`HfqTensorReader`]. Nothing
//! here holds the full weight set: the shipped model is ~360 GB and its n-gram
//! table alone is 102 GB.

use crate::config::Qwen4ExpConfig;
use crate::trunk_gpu::{TensorReader, TrunkScratch, TrunkState, TrunkWeights};
use hipfire_arch_api::ARCH_ID_QWEN4EXP;
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;

/// Reads HFQ tensors as f32.
///
/// Float encodings only, for now. A quantised routed-expert tensor returns a
/// NAMED error rather than being silently skipped or zero-filled — a model that
/// loads with holes in it is far worse to debug than one that refuses to load.
pub struct HfqTensorReader<'a> {
    pub hfq: &'a HfqFile,
}

impl TensorReader for HfqTensorReader<'_> {
    fn source_path(&self) -> Option<std::path::PathBuf> {
        Some(self.hfq.path().to_path_buf())
    }

    fn read_raw(&self, name: &str) -> Option<(u8, Vec<u8>)> {
        self.hfq.tensor_data_logical(name).ok()
    }

    fn read(&self, name: &str) -> Result<Vec<f32>, String> {
        let (qt, bytes) = self
            .hfq
            .tensor_data_logical(name)
            .map_err(|e| format!("qwen4_exp: reading `{name}`: {e:?}"))?;

        // Element count from the artifact's own index, not inferred from the
        // byte length — a packed format's bytes do not determine `n`.
        let info = self
            .hfq
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("qwen4_exp: `{name}` has no index entry"))?;
        let dims: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        let n: usize = dims.iter().product();

        // Float encodings pass through; quantised ones are DEQUANTISED to f32.
        //
        // ⚠️ This makes a quantised artifact LOADABLE, not memory-cheap: the
        // weights land on the GPU as f32, so an oq4 model costs the same resident
        // bytes as an f32 one. That is the right trade for a fixture and for
        // quality evidence (it is what lets arch 26 into the tiny-quant battery),
        // and the WRONG one for the shipped 360 GB checkpoint, which needs the
        // weights to STAY quantised behind `hipfire_runtime::weights::WeightTensor`
        // so the iu4/iu8 kernels read them directly. Both are tracked in
        // docs/model-support.toml.
        //
        // A format with no arm returns a NAMED error. A model that loads with
        // holes in it, or with a tensor silently read as the wrong layout, is far
        // worse to debug than one that refuses.
        use hipfire_runtime::quant as dq;
        match qt {
            // F32 / F16 / BF16.
            2 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()),
            1 => Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect()),
            16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect()),
            // Lossless BF16 recodings (Bf16Lut3 / Bf16Huff). These are pure
            // storage: decoding yields the identical BF16 bytes.
            49 | 50 => {
                let logical = hipfire_runtime::hfq::decode_bf16_packed(qt, &bytes, n)
                    .ok_or_else(|| format!("qwen4_exp: `{name}` bf16 recoding is corrupt"))?;
                Ok(logical
                    .chunks_exact(2)
                    .map(|c| {
                        f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)
                    })
                    .collect())
            }
            3 => Ok(dq::dequant_q8f16(&bytes, n)),
            4 => Ok(dq::dequant_q4k(&bytes, n)),
            // Opus W4/W8. `OqPlusG256` (33) is W4 stored in Oq4G256 blocks — the
            // loader ordinarily nibble-expands it to int8 for the W8A8 path; here
            // it dequantises through the same blocks.
            33 | 34 => Ok(dq::dequant_oq4g256(&bytes, n)),
            35 => Ok(dq::dequant_oq8g256(&bytes, n)),
            // Compact Opus carries a sparse outlier overlay, so it needs the 2-D
            // shape rather than just the element count.
            36 | 52 => {
                if dims.len() != 2 {
                    return Err(format!(
                        "qwen4_exp: `{name}` is compact Opus (qt {qt}) but its shape is {dims:?}; \
                         the overlay decode needs a 2-D [rows, cols] tensor"
                    ));
                }
                Ok(dq::dequant_oqplus_compact(&bytes, dims[0], dims[1]))
            }
            other => Err(format!(
                "qwen4_exp: `{name}` is quant type {other}, which this loader does not \
                 dequantise. Serve a float artifact (bf16/f16/f32) or one of the supported \
                 quantised formats (q8f16, q4k, oq4, oq8, oq+ compact)."
            )),
        }
    }
}

/// Public alias so the streamed n-gram reader shares this exact conversion
/// rather than growing a second, subtly different one.
pub fn f16_to_f32_pub(h: u16) -> f32 {
    f16_to_f32(h)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign << 31,
        // Subnormal: value is `man * 2^-24`. Renormalise by finding the top set
        // bit `e`, so `man = 2^e * (1 + f)` and the f32 exponent is `e + 103`
        // (= e - 24 + 127).
        0 => {
            let e = 31 - man.leading_zeros();
            (sign << 31) | ((e + 103) << 23) | ((man - (1 << e)) << (23 - e))
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

pub struct Qwen4Exp;

impl Architecture for Qwen4Exp {
    type Weights = TrunkWeights;
    type State = TrunkState;
    type Config = Qwen4ExpConfig;

    fn arch_id() -> u32 {
        ARCH_ID_QWEN4EXP
    }

    fn name() -> &'static str {
        "qwen4exp"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Qwen4ExpConfig::from_metadata_json(&hfq.metadata_json)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        let reader = HfqTensorReader { hfq };
        TrunkWeights::upload(gpu, cfg, &reader).map_err(|e| format!("qwen4_exp: {e:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        TrunkState::new(gpu, cfg, cfg.max_seq_hint())
            .map_err(|e| format!("qwen4_exp: state: {e:?}"))
    }
}

/// Assemble the PLE n-gram table from its checkpoint shards.
///
/// `split_ngram_parts` shards the table on disk (128 parts in the shipped model);
/// loading concatenates them, which is what the reference does too.
///
/// ⚠️ **Fixture scale only.** The shipped table is 102 GB — 41% of the model — and
/// must never be materialised like this. The serving path reads single rows out of
/// the sharded file through [`crate::ngram_store`], using
/// [`crate::ngram::NgramHasher::locate`] to map a flat row to its shard; this
/// helper exists so small artifacts can be loaded without that machinery.
pub fn load_ngram_table(r: &dyn TensorReader, cfg: &Qwen4ExpConfig) -> Result<Vec<f32>, String> {
    let n = cfg
        .ngram
        .as_ref()
        .ok_or_else(|| "qwen4_exp: no PLE layer in this config".to_string())?;
    let base = format!(
        "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding",
        n.layer_idx
    );
    // The unsharded name is what a safetensors source produces.
    if let Ok(v) = r.read(&format!("{base}.weight")) {
        return Ok(v);
    }
    let (_, _, padded) =
        crate::ngram_head_layout_at(n.vocab_size_base, n.heads(), n.divisible_by, n.ple_index);
    let want = padded as usize * n.head_dim();
    let mut out = Vec::with_capacity(want);
    for i in 0..n.shards {
        out.extend_from_slice(&r.read(&format!("{base}.shard_{i}.weight"))?);
    }
    if out.len() != want {
        return Err(format!(
            "qwen4_exp: n-gram shards total {} elements, expected {want} \
             ({} shards x {} rows x {})",
            out.len(),
            n.shards,
            padded as usize / n.shards,
            n.head_dim()
        ));
    }
    Ok(out)
}

/// Scratch is allocated alongside state; kept out of the trait, which does not
/// model it.
pub fn new_scratch(gpu: &mut Gpu, cfg: &Qwen4ExpConfig) -> Result<TrunkScratch, String> {
    TrunkScratch::new(gpu, cfg, cfg.max_seq_hint())
        .map_err(|e| format!("qwen4_exp: scratch: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_representative_values() {
        // 0, ±1, ±0.5, a subnormal, and the largest finite half.
        for (h, want) in [
            (0x0000u16, 0.0f32),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x3800, 0.5),
            (0x0001, 5.960_464_5e-8), // smallest subnormal, 2^-24
            (0x03ff, 6.097_555_e-5),  // LARGEST subnormal — catches a wrong renormalise
            (0x0400, 6.103_515_6e-5), // smallest normal
            (0x7bff, 65504.0),
        ] {
            let got = f16_to_f32(h);
            assert!(
                (got - want).abs() <= want.abs() * 1e-6 + 1e-12,
                "f16 {h:#06x} -> {got}, want {want}"
            );
        }
    }
}
