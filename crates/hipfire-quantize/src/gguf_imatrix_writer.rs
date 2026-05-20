//! GGUF v3 imatrix writer.
//!
//! Emits a GGUF file byte-near-identical to llama.cpp's `llama-imatrix
//! --output-format gguf` output, so the existing reader at
//! [`crate::gguf_input`] consumes it without changes and downstream consumers
//! (notably `hipfire-quantize::main::load_imatrix`) work transparently.
//!
//! ## Reference layout (verified against
//! `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf`)
//!
//! Header (24 bytes):
//! - magic = "GGUF"
//! - u32 version = 3
//! - u64 tensor_count
//! - u64 metadata_kv_count
//!
//! Metadata key-value pairs (we emit 1 or 2 depending on `dataset_name`):
//! - `general.type` (string, vtype=8) = "imatrix"
//! - `imatrix.datasets` (array<string>, vtype=9 over vtype=8) = \[dataset\]
//!   (only when `dataset_name` is `Some`)
//!
//! Real llama-imatrix output also stores `imatrix.chunk_count` (u32) and
//! `imatrix.chunk_size` (u32). The downstream consumer
//! (`hipfire-quantize::main::load_imatrix`) ignores them, so we omit them
//! here — keeping the API minimal. Add them via a future field on a config
//! struct if a future consumer needs them.
//!
//! Tensor section: TWO tensors per [`ImatrixEntry`]:
//! - `<name>.in_sum2`  F32  shape `[k]`     — Σx² per channel
//! - `<name>.counts`   F32  shape `[1]`     — token count contributing
//!
//! All tensor data is packed at the tensor-data section start (32-byte
//! aligned), with per-tensor offsets relative to that base, in the same
//! order the tensor records appear in the header. Inter-tensor padding to
//! 32 bytes mirrors llama.cpp's default `general.alignment = 32`.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

/// GGUF magic: `"GGUF"` little-endian = 0x46554747.
const GGUF_MAGIC: u32 = 0x46554747;
/// Format version we emit.
const GGUF_VERSION: u32 = 3;

/// GGUF metadata value type tags (subset; matches the reader's
/// `read_typed_value`). Numbers are stable across the format spec.
mod vtype {
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
}

/// GGML tensor dtypes. Matches `GgmlType::F32` in `gguf_input.rs`.
const GGML_TYPE_F32: u32 = 0;

/// Default GGUF tensor-data alignment (matches llama.cpp's
/// `general.alignment` default — the reader uses this constant when the
/// metadata key is absent, so omitting it from the metadata block keeps the
/// header smaller without changing semantics).
const TENSOR_DATA_ALIGNMENT: usize = 32;

/// One per-linear-layer activation accumulator. The pair (`in_sum2`,
/// `counts`) becomes two GGUF tensors named `<name>.in_sum2` and
/// `<name>.counts` per llama-imatrix convention.
pub struct ImatrixEntry {
    /// Base tensor name in the GGUF imatrix convention — already includes
    /// the trailing `.weight` so the produced GGUF tensor names look like
    /// `blk.0.attn_q.weight.in_sum2` (see the qwen3.5-0.8b ref imatrix at
    /// `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf`).
    pub name: String,
    /// Per-channel Σx² accumulator. Length `K` defines the tensor shape.
    pub in_sum2: Vec<f32>,
    /// Total token count contributing to `in_sum2`. Stored as a 1-element
    /// F32 tensor (this is what llama-imatrix emits, even though logically
    /// it could be a metadata u64).
    pub counts: f32,
}

/// Write a GGUF v3 imatrix file containing `entries`. `dataset_name`, if
/// supplied, is stored as `imatrix.datasets[0]` (string-array metadata).
///
/// The output is consumable by:
/// - [`crate::gguf_input::GgufFile::open`] (this crate's reader)
/// - `hipfire-quantize`'s `load_imatrix` lookup helper
/// - llama.cpp's own imatrix-aware tooling (matching `general.type =
///   "imatrix"` + the two-tensor-per-layer convention)
pub fn write_gguf_imatrix(
    path: &Path,
    entries: &[ImatrixEntry],
    dataset_name: Option<&str>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // ─── Header ────────────────────────────────────────────────────────
    w.write_all(&GGUF_MAGIC.to_le_bytes())?;
    w.write_all(&GGUF_VERSION.to_le_bytes())?;

    let tensor_count = (entries.len() as u64) * 2; // in_sum2 + counts per entry
    w.write_all(&tensor_count.to_le_bytes())?;

    let mut metadata_count: u64 = 1; // general.type
    if dataset_name.is_some() {
        metadata_count += 1; // imatrix.datasets
    }
    w.write_all(&metadata_count.to_le_bytes())?;

    // ─── Metadata ──────────────────────────────────────────────────────
    write_string_kv(&mut w, "general.type", "imatrix")?;
    if let Some(ds) = dataset_name {
        write_string_array_kv(&mut w, "imatrix.datasets", &[ds])?;
    }

    // ─── Tensor info (offset relative to tensor-data start) ────────────
    //
    // We compute each tensor's offset up front so we can write the info
    // section (which sits in the header section) before we know absolute
    // file positions for the data blob.
    //
    // Layout per tensor in this writer:
    //   <name>.in_sum2  F32  shape=[k]   → k * 4 bytes
    //   <name>.counts   F32  shape=[1]   → 4 bytes (still padded to 32 B)
    //
    // Padding policy: pad each tensor's footprint up to a 32-byte boundary
    // (TENSOR_DATA_ALIGNMENT). The reference reference 0.8B imatrix shows
    // the same pattern — `in_sum2` is 1024×4 = 4096 (already aligned),
    // `counts` is 4 bytes but the next tensor starts 32 bytes later.
    let mut tensor_offsets: Vec<u64> = Vec::with_capacity(entries.len() * 2);
    let mut cur_off: u64 = 0;
    for e in entries {
        // in_sum2
        tensor_offsets.push(cur_off);
        let in_sum2_bytes = (e.in_sum2.len() as u64) * 4;
        cur_off += align_up(in_sum2_bytes, TENSOR_DATA_ALIGNMENT as u64);
        // counts (1 F32 = 4 bytes, padded to 32)
        tensor_offsets.push(cur_off);
        cur_off += align_up(4, TENSOR_DATA_ALIGNMENT as u64);
    }
    let total_tensor_bytes = cur_off;

    for (i, e) in entries.iter().enumerate() {
        // in_sum2 tensor info: name, n_dims=1, shape=[k], dtype=F32, offset.
        let in_sum2_name = format!("{}.in_sum2", e.name);
        write_string(&mut w, &in_sum2_name)?;
        w.write_all(&1u32.to_le_bytes())?; // n_dims
        let k = e.in_sum2.len() as u64;
        w.write_all(&k.to_le_bytes())?; // shape[0] = K
        w.write_all(&GGML_TYPE_F32.to_le_bytes())?;
        w.write_all(&tensor_offsets[i * 2].to_le_bytes())?;

        // counts tensor info: name, n_dims=1, shape=[1], dtype=F32, offset.
        let counts_name = format!("{}.counts", e.name);
        write_string(&mut w, &counts_name)?;
        w.write_all(&1u32.to_le_bytes())?; // n_dims
        w.write_all(&1u64.to_le_bytes())?; // shape[0] = 1
        w.write_all(&GGML_TYPE_F32.to_le_bytes())?;
        w.write_all(&tensor_offsets[i * 2 + 1].to_le_bytes())?;
    }

    // ─── Pad to tensor-data alignment ──────────────────────────────────
    //
    // `BufWriter` doesn't track the absolute file position without flushing
    // and `Seek`ing. Flush, then query the inner file's position so the pad
    // computation is exact even though we wrote variable-width metadata.
    w.flush()?;
    let info_end = w.get_mut().stream_position()?;
    let data_start = align_up(info_end, TENSOR_DATA_ALIGNMENT as u64);
    let info_pad = data_start - info_end;
    if info_pad > 0 {
        write_zero_pad(&mut w, info_pad as usize)?;
    }

    // ─── Tensor data (in the order we wrote tensor info) ──────────────
    for e in entries {
        // in_sum2: K * F32, then pad to 32-byte boundary
        let in_sum2_bytes = (e.in_sum2.len() as u64) * 4;
        for &v in &e.in_sum2 {
            w.write_all(&v.to_le_bytes())?;
        }
        let in_sum2_pad = align_up(in_sum2_bytes, TENSOR_DATA_ALIGNMENT as u64) - in_sum2_bytes;
        if in_sum2_pad > 0 {
            write_zero_pad(&mut w, in_sum2_pad as usize)?;
        }
        // counts: 1 F32 = 4 bytes, pad to 32
        w.write_all(&e.counts.to_le_bytes())?;
        write_zero_pad(&mut w, (TENSOR_DATA_ALIGNMENT as usize) - 4)?;
    }

    // Sanity: total bytes written in the data section equals `total_tensor_bytes`.
    //
    // `BufWriter::stream_position` reports the underlying file position
    // (not the logical write cursor), so we must flush before the comparison.
    w.flush()?;
    debug_assert_eq!(
        w.get_mut().stream_position()?.saturating_sub(data_start),
        total_tensor_bytes,
        "data-section size mismatch (writer bug)"
    );

    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) / align * align
}

fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    let n = bytes.len() as u64;
    w.write_all(&n.to_le_bytes())?;
    w.write_all(bytes)
}

fn write_string_kv<W: Write>(w: &mut W, key: &str, value: &str) -> std::io::Result<()> {
    write_string(w, key)?;
    w.write_all(&vtype::STRING.to_le_bytes())?;
    write_string(w, value)
}

fn write_string_array_kv<W: Write>(
    w: &mut W,
    key: &str,
    values: &[&str],
) -> std::io::Result<()> {
    write_string(w, key)?;
    w.write_all(&vtype::ARRAY.to_le_bytes())?;
    // array header: element vtype + count (u64)
    w.write_all(&vtype::STRING.to_le_bytes())?;
    w.write_all(&(values.len() as u64).to_le_bytes())?;
    for v in values {
        write_string(w, v)?;
    }
    Ok(())
}

fn write_zero_pad<W: Write + Seek>(w: &mut W, n: usize) -> std::io::Result<()> {
    if n == 0 {
        return Ok(());
    }
    // Up to 64 bytes — well within typical pad sizes (≤ TENSOR_DATA_ALIGNMENT
    // − 1 = 31 in practice).
    const Z: [u8; 64] = [0u8; 64];
    let mut left = n;
    while left > 0 {
        let chunk = left.min(Z.len());
        w.write_all(&Z[..chunk])?;
        left -= chunk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_input::{GgmlType, GgufFile};
    use tempfile::NamedTempFile;

    fn entry(name: &str, in_sum2: Vec<f32>, counts: f32) -> ImatrixEntry {
        ImatrixEntry { name: name.to_string(), in_sum2, counts }
    }

    #[test]
    fn roundtrip_two_tensors_via_reader() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![
            entry("blk.0.attn_q.weight", vec![1.0_f32, 2.0, 3.0, 4.0], 128.0),
            entry(
                "blk.0.ffn_down.weight",
                (0..16).map(|i| (i as f32) * 0.5).collect(),
                256.0,
            ),
        ];

        write_gguf_imatrix(tf.path(), &entries, Some("test-corpus.txt"))
            .expect("write_gguf_imatrix");

        let gguf = GgufFile::open(tf.path()).expect("reader open");
        assert_eq!(gguf.version, 3);
        // 2 tensors per entry × 2 entries = 4 tensors
        assert_eq!(gguf.tensors.len(), 4);
        assert_eq!(gguf.meta_str("general.type"), Some("imatrix"));

        // Locate the in_sum2 and counts pairs by name.
        let find_tensor = |name: &str| {
            gguf.tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("tensor {name} missing"))
        };

        // First entry
        let a_in_sum2 = find_tensor("blk.0.attn_q.weight.in_sum2");
        assert_eq!(a_in_sum2.dtype, GgmlType::F32);
        assert_eq!(a_in_sum2.shape, vec![4]);
        let a_data = gguf.tensor_data(a_in_sum2);
        let a_floats: Vec<f32> = a_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(a_floats, vec![1.0_f32, 2.0, 3.0, 4.0]);

        let a_counts = find_tensor("blk.0.attn_q.weight.counts");
        assert_eq!(a_counts.dtype, GgmlType::F32);
        assert_eq!(a_counts.shape, vec![1]);
        let c_bytes = gguf.tensor_data(a_counts);
        let c_val = f32::from_le_bytes([c_bytes[0], c_bytes[1], c_bytes[2], c_bytes[3]]);
        assert_eq!(c_val, 128.0);

        // Second entry — verify all 16 values roundtrip
        let b = find_tensor("blk.0.ffn_down.weight.in_sum2");
        assert_eq!(b.shape, vec![16]);
        let b_data = gguf.tensor_data(b);
        let b_floats: Vec<f32> = b_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for i in 0..16 {
            assert_eq!(b_floats[i], (i as f32) * 0.5);
        }
        let b_counts = find_tensor("blk.0.ffn_down.weight.counts");
        let cb = gguf.tensor_data(b_counts);
        let cb_val = f32::from_le_bytes([cb[0], cb[1], cb[2], cb[3]]);
        assert_eq!(cb_val, 256.0);
    }

    #[test]
    fn omitting_dataset_emits_only_general_type_metadata() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![entry("blk.0.attn_q.weight", vec![1.0, 2.0], 1.0)];
        write_gguf_imatrix(tf.path(), &entries, None).unwrap();

        let gguf = GgufFile::open(tf.path()).unwrap();
        assert_eq!(gguf.meta_str("general.type"), Some("imatrix"));
        assert!(gguf.meta("imatrix.datasets").is_none());
        // No tensors should be missing; metadata-only optionality is the
        // semantic, the tensor section is unaffected.
        assert_eq!(gguf.tensors.len(), 2);
    }

    #[test]
    fn header_magic_and_version_match_spec() {
        let tf = NamedTempFile::new().unwrap();
        write_gguf_imatrix(tf.path(), &[], None).unwrap();
        let bytes = std::fs::read(tf.path()).unwrap();
        assert_eq!(&bytes[0..4], b"GGUF");
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            GGUF_VERSION
        );
        // tensor_count = 0
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
        // metadata_count = 1 (general.type only)
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 1);
    }

    #[test]
    fn tensor_data_is_32byte_aligned_and_offsets_correct() {
        // The reference imatrix shows in_sum2 tensors start with K*4 bytes,
        // then a padded gap before the next tensor. Replicate that pattern
        // with a 1-element in_sum2 (4 bytes, pad to 32) so offsets are easy
        // to predict.
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![entry("x", vec![1.5_f32], 7.0)];
        write_gguf_imatrix(tf.path(), &entries, None).unwrap();

        let gguf = GgufFile::open(tf.path()).unwrap();
        let in_sum2 = gguf.tensors.iter().find(|t| t.name == "x.in_sum2").unwrap();
        let counts = gguf.tensors.iter().find(|t| t.name == "x.counts").unwrap();
        // First tensor offset within the data section must be 0.
        assert_eq!(in_sum2.offset, 0);
        // Second tensor must start 32 bytes after the first (pad up from 4).
        assert_eq!(counts.offset, TENSOR_DATA_ALIGNMENT);

        // tensor_data_offset itself must also be 32-aligned (default
        // `general.alignment` is 32).
        assert_eq!(gguf.tensor_data_offset % TENSOR_DATA_ALIGNMENT, 0);
    }

    #[test]
    fn empty_entries_writes_no_tensors() {
        let tf = NamedTempFile::new().unwrap();
        write_gguf_imatrix(tf.path(), &[], Some("ds")).unwrap();

        let gguf = GgufFile::open(tf.path()).unwrap();
        assert_eq!(gguf.tensors.len(), 0);
        assert_eq!(gguf.meta_str("general.type"), Some("imatrix"));
    }
}
