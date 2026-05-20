//! `.imq` (**im**atrix + **m**agnum-**q**uant) writer — hipfire-native
//! binary imatrix format.
//!
//! Companion to [`crate::gguf_imatrix_writer`]. Same input — a slice of
//! [`ImatrixEntry`] — but writes a compact little-endian binary container
//! instead of a GGUF v3 file.
//!
//! The format trades GGUF compatibility for: a fixed 24-byte header
//! (vs GGUF's variable-length kv-metadata block), no per-tensor name
//! / shape / offset metadata table (vs GGUF's two-tensor records and
//! 32-byte-aligned padding around each tensor), and a u64 `n_tokens`
//! scalar (vs GGUF's `.counts` companion tensor + alignment padding).
//! On a Qwen3.5-0.8B-scale model the savings are modest (~1-2%); the
//! real value is owning the format so we can extend it (v2 metadata,
//! MoE per-expert layout, …) without staying byte-compatible with
//! llama.cpp.
//!
//! ## Wire format (v1)
//!
//! ```text
//! Header (24 B):
//!   4 B  magic     = "IMQ\0"
//!   4 B  version   = u32 LE (= 1)
//!   8 B  n_tensors = u64 LE
//!   8 B  reserved  = u64 LE (must be 0; future v2 may carry a global
//!                  metadata block pointer here)
//!
//! Per-tensor record (n_tensors entries, packed contiguously):
//!   4 B   name_len  = u32 LE (UTF-8 length in bytes; MAX 4096)
//!   N B   name      = UTF-8 (canonical HF safetensors key, e.g.
//!                     "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
//!   4 B   k         = u32 LE (number of input channels)
//!   8 B   n_tokens  = u64 LE (token count — single scalar; GGUF stores
//!                     this as a separate `.counts` F32 tensor)
//!   K×4 B in_sum2   = F32 LE row (per-channel Σ x²)
//! ```
//!
//! ## Design notes
//!
//! - **u64 n_tokens scalar.** This crate's GGUF imatrix writer keeps
//!   `counts` as a 1-element F32 tensor (the value is logically a token
//!   count); MoE-extended variants in the wild can grow the tensor to
//!   `[1, n_mat]`. `.imq` simplifies this to a single u64 per record
//!   and saves a tensor info entry + 32-byte payload-aligned slot per
//!   linear layer.
//! - **HF safetensors names verbatim.** Producer emits names like
//!   `model.layers.0.self_attn.q_proj.weight`. The reader returns them
//!   unchanged; the existing safetensors→ggml-style translation lives
//!   at the consumer site (see [`crate::main::safetensors_to_ggml_name`]).
//! - **Endianness fixed LE.** Matches the sibling HFHS Hessian sidecar
//!   format ([`crate::hessian_io`]); no byte-order magic needed since the
//!   format is hipfire-internal.
//! - **Tensor order is emitter-defined.** The consumer indexes by name
//!   (via `HashMap`), not by file position, so writers can sort/shuffle
//!   freely.
//! - **No global metadata block in v1.** The reserved u64 in the header
//!   gives room for a future v2 to store a dataset name / chunk-count /
//!   tokenizer hash without breaking layout.

use crate::gguf_imatrix_writer::ImatrixEntry;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// `.imq` magic bytes: `b"IMQ\x00"`.
pub const IMQ_MAGIC: &[u8; 4] = b"IMQ\x00";
/// Format version this writer emits.
pub const IMQ_VERSION: u32 = 1;
/// Header size in bytes (magic + version + n_tensors + reserved).
pub const IMQ_HEADER_SIZE: usize = 24;
/// Maximum permitted tensor-name length in bytes.
pub const IMQ_MAX_NAME_LEN: usize = 4096;

/// Write a hipfire-native `.imq` imatrix file containing `entries`.
///
/// The `_dataset_name` argument is accepted for API parity with
/// [`crate::gguf_imatrix_writer::write_gguf_imatrix`] but is **not stored
/// in `.imq` v1** — there is no global metadata block. A future v2 may
/// repurpose the reserved u64 in the header to point to a metadata
/// section. Until then the argument is silently ignored, matching the
/// GGUF writer's flexible signature so callers can switch formats with a
/// single line change.
///
/// # Errors
///
/// - I/O errors from creating the parent directory or writing the file.
/// - Any single tensor name longer than [`IMQ_MAX_NAME_LEN`] bytes — the
///   writer returns an `InvalidInput` error rather than silently writing
///   a record the reader would reject.
pub fn write_imq(
    path: &Path,
    entries: &[ImatrixEntry],
    _dataset_name: Option<&str>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // ── Header ──────────────────────────────────────────────────────────
    w.write_all(IMQ_MAGIC)?;
    w.write_all(&IMQ_VERSION.to_le_bytes())?;
    w.write_all(&(entries.len() as u64).to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // reserved — must be 0 in v1

    // ── Per-tensor records ─────────────────────────────────────────────
    for e in entries {
        let name_bytes = e.name.as_bytes();
        if name_bytes.len() > IMQ_MAX_NAME_LEN {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "tensor name length {} exceeds IMQ_MAX_NAME_LEN {} ({:?})",
                    name_bytes.len(),
                    IMQ_MAX_NAME_LEN,
                    e.name,
                ),
            ));
        }
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&(e.in_sum2.len() as u32).to_le_bytes())?;
        // n_tokens — promote the F32 counts scalar to u64. Calibration
        // collectors store this as `counts: f32` because GGUF's wire layout
        // requires a 1-element F32 tensor; the value is logically an
        // integer token count, so `as u64` is a faithful narrowing.
        let n_tokens = e.counts as u64;
        w.write_all(&n_tokens.to_le_bytes())?;
        for &v in &e.in_sum2 {
            w.write_all(&v.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imq_reader::load_imq;
    use std::io::Read;
    use tempfile::NamedTempFile;

    fn entry(name: &str, in_sum2: Vec<f32>, counts: f32) -> ImatrixEntry {
        ImatrixEntry { name: name.to_string(), in_sum2, counts }
    }

    #[test]
    fn test_write_then_read_roundtrip() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![
            entry(
                "model.layers.0.self_attn.q_proj.weight",
                vec![1.0_f32, 2.0, 3.0, 4.0],
                128.0,
            ),
            entry(
                "model.layers.0.mlp.down_proj.weight",
                (0..16).map(|i| (i as f32) * 0.5).collect(),
                256.0,
            ),
        ];

        write_imq(tf.path(), &entries, Some("calibration")).expect("write_imq");

        let map = load_imq(tf.path()).expect("load_imq");
        assert_eq!(map.len(), 2);

        let q = map
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj missing");
        // F32-identical (we wrote LE bytes and read them back; no
        // arithmetic in between).
        assert_eq!(q.as_slice(), &[1.0_f32, 2.0, 3.0, 4.0]);

        let d = map
            .get("model.layers.0.mlp.down_proj.weight")
            .expect("down_proj missing");
        assert_eq!(d.len(), 16);
        for i in 0..16 {
            assert_eq!(d[i], (i as f32) * 0.5);
        }
    }

    #[test]
    fn test_roundtrip_preserves_n_tokens() {
        // Separate roundtrip via the low-level header parse — `load_imq`
        // doesn't expose n_tokens, but a future v2 metadata reader will.
        // Verify the wire-level field is written correctly.
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![entry("t", vec![1.5_f32], 7.0)];
        write_imq(tf.path(), &entries, None).unwrap();

        let bytes = std::fs::read(tf.path()).unwrap();
        // After 24 B header + 4 B name_len + 1 B "t" + 4 B k = pos 33,
        // the next 8 bytes are the u64 n_tokens little-endian.
        let n_tokens =
            u64::from_le_bytes(bytes[33..41].try_into().unwrap());
        assert_eq!(n_tokens, 7);
    }

    #[test]
    fn test_empty_file() {
        let tf = NamedTempFile::new().unwrap();
        write_imq(tf.path(), &[], None).unwrap();

        let mut bytes = Vec::new();
        let mut f = std::fs::File::open(tf.path()).unwrap();
        f.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len(), IMQ_HEADER_SIZE);
        assert_eq!(&bytes[0..4], IMQ_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            IMQ_VERSION
        );
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);

        // Empty file roundtrips to an empty map.
        let map = load_imq(tf.path()).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_rejects_overly_long_name() {
        let tf = NamedTempFile::new().unwrap();
        let long_name = "x".repeat(IMQ_MAX_NAME_LEN + 1);
        let entries = vec![entry(&long_name, vec![1.0_f32], 1.0)];
        let err = write_imq(tf.path(), &entries, None).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    /// Size comparison vs the GGUF-equivalent imatrix writer. Uses a
    /// shape close to Qwen3.5-0.8B (64 layers × 7 slots × K ∈ {1024, 4096}).
    /// Confirms IMQ is smaller (the headline claim) and reports the
    /// actual ratio to the test log — useful when reasoning about
    /// per-entry overhead choices in a future v2.
    #[test]
    fn imq_is_smaller_than_gguf() {
        use crate::gguf_imatrix_writer::write_gguf_imatrix;
        let mut entries: Vec<ImatrixEntry> = Vec::new();
        for layer in 0..64u32 {
            for (slot, k) in [
                ("self_attn.q_proj", 1024usize),
                ("self_attn.k_proj", 1024),
                ("self_attn.v_proj", 1024),
                ("self_attn.o_proj", 1024),
                ("mlp.gate_proj", 4096),
                ("mlp.up_proj", 4096),
                ("mlp.down_proj", 4096),
            ] {
                entries.push(ImatrixEntry {
                    name: format!("model.layers.{layer}.{slot}.weight"),
                    in_sum2: vec![1.0_f32; k],
                    counts: 2048.0,
                });
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let imq_path = tmp.path().join("probe.imq");
        let gguf_path = tmp.path().join("probe.gguf");
        write_imq(&imq_path, &entries, Some("calibration")).unwrap();
        write_gguf_imatrix(&gguf_path, &entries, Some("calibration")).unwrap();
        let imq_sz = std::fs::metadata(&imq_path).unwrap().len();
        let gguf_sz = std::fs::metadata(&gguf_path).unwrap().len();
        eprintln!(
            "imq-vs-gguf: entries={} imq={} B gguf={} B ratio={:.4} reduction={:.1}%",
            entries.len(),
            imq_sz,
            gguf_sz,
            (imq_sz as f64) / (gguf_sz as f64),
            (1.0 - (imq_sz as f64) / (gguf_sz as f64)) * 100.0
        );
        // IMQ MUST be smaller than GGUF — that's the design promise.
        // The actual savings are modest for this writer (the existing
        // GGUF path stores `counts` as 1×F32, not K×F32) but always
        // positive thanks to dropping per-tensor metadata + alignment
        // padding.
        assert!(
            imq_sz < gguf_sz,
            "expected imq < gguf, got imq={imq_sz} gguf={gguf_sz}"
        );
    }

    #[test]
    fn test_header_layout_matches_spec() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![entry("t", vec![0.25_f32], 42.0)];
        write_imq(tf.path(), &entries, Some("ignored")).unwrap();
        let bytes = std::fs::read(tf.path()).unwrap();
        assert_eq!(&bytes[0..4], b"IMQ\x00");
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            1
        );
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);
        // name_len = 1, "t", k = 1
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            1
        );
        assert_eq!(&bytes[28..29], b"t");
        assert_eq!(
            u32::from_le_bytes(bytes[29..33].try_into().unwrap()),
            1
        );
        // n_tokens = 42, in_sum2 = [0.25]
        assert_eq!(
            u64::from_le_bytes(bytes[33..41].try_into().unwrap()),
            42
        );
        assert_eq!(
            f32::from_le_bytes(bytes[41..45].try_into().unwrap()),
            0.25
        );
        // Total file size: header + name_len + name + k + n_tokens + K*4
        assert_eq!(bytes.len(), 24 + 4 + 1 + 4 + 8 + 4);
    }
}
