// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Where the PLE n-gram embedding rows come from.
//!
//! The table is the single reason this model cannot be loaded like any other: it
//! is **102 GB**, 41% of the parameters, and it is GATHERED — one token touches
//! exactly `heads_per_ngram` rows of it. Holding it resident to read 8 rows per
//! token is the wrong shape by three orders of magnitude.
//!
//! So the trunk asks for rows through this trait instead of indexing a slice.
//! Two providers exist and they must agree exactly:
//!
//! * [`ResidentRows`] — a host slice, which is what a fixture uses and what the
//!   parity examples were written against.
//! * [`HfqShardRows`] — reads each row out of the artifact's shard tensors with a
//!   ranged pread, touching `heads * head_dim * 4` bytes per token rather than
//!   102 GB.
//!
//! `examples/serve_fixture` runs both over the same artifact and requires
//! IDENTICAL logits, which is the only way to know the streaming path's row
//! addressing (shard split, row-in-shard, element offset) is right — a wrong row
//! is still a perfectly finite embedding.

use crate::ngram::NgramHasher;
use hipfire_runtime::hfq::HfqFile;

/// Supplies `head_dim`-wide embedding rows by flat row index.
pub trait NgramRows {
    /// Gather `rows`, concatenated in order, as `rows.len() * head_dim` f32.
    fn gather(&self, rows: &[u64], head_dim: usize) -> Result<Vec<f32>, String>;
}

/// The whole table, resident on the host.
pub struct ResidentRows<'a> {
    pub table: &'a [f32],
}

impl NgramRows for ResidentRows<'_> {
    fn gather(&self, rows: &[u64], head_dim: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(rows.len() * head_dim);
        for &r in rows {
            let start = r as usize * head_dim;
            let end = start + head_dim;
            let slice = self.table.get(start..end).ok_or_else(|| {
                format!(
                    "ngram row {r} is past the resident table ({} rows of {head_dim})",
                    self.table.len() / head_dim.max(1)
                )
            })?;
            out.extend_from_slice(slice);
        }
        Ok(out)
    }
}

/// Rows read straight out of the artifact's `ngram_embedding.shard_*` tensors.
///
/// Holds no table — one gather is `rows.len()` ranged preads. The shipped model
/// splits the table into `split_ngram_parts` shards, so a flat row maps to
/// `(shard, row_in_shard)` through [`NgramHasher::locate`]; getting that mapping
/// wrong yields a valid-looking embedding from the wrong row, which is why the
/// two providers are differenced rather than each being checked alone.
pub struct HfqShardRows<'a> {
    hfq: &'a HfqFile,
    hasher: &'a NgramHasher,
    /// `layers.<n>.ple.ple_embedding.ngram_embedding.shard_` — `<index>.weight` is
    /// appended per lookup.
    prefix: String,
    /// On-disk encoding of the shard tensors, checked once at construction so a
    /// per-row read does not re-derive it.
    quant_type: u8,
}

impl<'a> HfqShardRows<'a> {
    /// `layer` is the ZERO-BASED layer the PLE block rides (`ple_layer_ids` in the
    /// file is one-based — see `config.rs`).
    pub fn new(hfq: &'a HfqFile, hasher: &'a NgramHasher, layer: usize) -> Result<Self, String> {
        let prefix =
            format!("model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_");
        let first = format!("{prefix}0.weight");
        let info = hfq
            .find_tensor_info(&first)
            .ok_or_else(|| format!("qwen4_exp: no n-gram shard tensor `{first}`"))?;
        // Only fixed-width float encodings have a computable per-row offset. A
        // quantised or variable-length-coded shard would need its block structure
        // decoded, so refuse by name rather than read the wrong bytes.
        if !matches!(info.quant_type, 1 | 2 | 16) {
            return Err(format!(
                "qwen4_exp: n-gram shards are quant type {} — streamed row reads need a \
                 fixed-width float encoding (f32/f16/bf16). Load this model with a resident \
                 table, or store the n-gram table unquantised.",
                info.quant_type
            ));
        }
        Ok(Self {
            hfq,
            hasher,
            prefix,
            quant_type: info.quant_type,
        })
    }

    fn elem_bytes(&self) -> usize {
        match self.quant_type {
            2 => 4,
            _ => 2,
        }
    }

    fn decode(&self, bytes: &[u8], out: &mut Vec<f32>) {
        match self.quant_type {
            2 => out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            ),
            16 => {
                out.extend(bytes.chunks_exact(2).map(|c| {
                    f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)
                }))
            }
            _ => {
                out.extend(bytes.chunks_exact(2).map(|c| {
                    crate::arch::f16_to_f32_pub(u16::from_le_bytes(c.try_into().unwrap()))
                }))
            }
        }
    }
}

impl NgramRows for HfqShardRows<'_> {
    fn gather(&self, rows: &[u64], head_dim: usize) -> Result<Vec<f32>, String> {
        let eb = self.elem_bytes();
        let row_bytes = head_dim * eb;
        let mut out = Vec::with_capacity(rows.len() * head_dim);
        for &r in rows {
            let loc = self.hasher.locate(r);
            let name = format!("{}{}.weight", self.prefix, loc.shard);
            let offset = loc.row_in_shard as usize * row_bytes;
            let bytes = self
                .hfq
                .tensor_byte_range(&name, offset, row_bytes)
                .ok_or_else(|| {
                    format!(
                        "qwen4_exp: n-gram row {r} -> `{name}` bytes [{offset}, {}) is out of \
                         range or unreadable",
                        offset + row_bytes
                    )
                })?;
            self.decode(&bytes, &mut out);
        }
        Ok(out)
    }
}
