//! `.imq` (**im**atrix + **m**agnum-**q**uant) reader — counterpart to
//! [`crate::imq_writer`].
//!
//! Returns a `HashMap<canonical_HF_name, Vec<f32>>` matching the
//! shape that [`crate::main::load_imatrix`] produces for the GGUF path,
//! except keys here are **HF-safetensors canonical** (e.g.
//! `model.layers.0.self_attn.q_proj.weight`) rather than ggml-style.
//! The existing safetensors→ggml translation lives at the consumer site
//! ([`crate::main::safetensors_to_ggml_name`]); the reader does **not**
//! translate names.
//!
//! Wire format is documented exhaustively in [`crate::imq_writer`].

use crate::imq_writer::{IMQ_HEADER_SIZE, IMQ_MAGIC, IMQ_MAX_NAME_LEN, IMQ_VERSION};

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Parse a `.imq` file and return a per-tensor `in_sum2` lookup keyed by
/// canonical HF safetensors tensor name.
///
/// # Errors
///
/// - I/O errors on open / read.
/// - `InvalidData` on bad magic, unsupported version, or any structural
///   inconsistency (truncated record, name_len overflow, non-UTF8 name).
pub fn load_imq(path: &Path) -> io::Result<HashMap<String, Vec<f32>>> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    parse_imq_bytes(&bytes)
}

/// Buffer-driven parse path. Kept private — the `Path`-based [`load_imq`]
/// is the public entry, mirroring the GGUF-side `load_imatrix`. Exposing
/// the byte-slice form would commit us to an API the GGUF path can't
/// cheaply offer (it mmaps internally).
fn parse_imq_bytes(bytes: &[u8]) -> io::Result<HashMap<String, Vec<f32>>> {
    if bytes.len() < IMQ_HEADER_SIZE {
        return Err(invalid_data(format!(
            "file too small for IMQ header: {} < {}",
            bytes.len(),
            IMQ_HEADER_SIZE
        )));
    }
    if &bytes[0..4] != IMQ_MAGIC {
        return Err(invalid_data(format!(
            "bad IMQ magic: got {:?}, expected {:?}",
            &bytes[0..4],
            IMQ_MAGIC
        )));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != IMQ_VERSION {
        return Err(invalid_data(format!(
            "unsupported IMQ version {version}; this build understands v{IMQ_VERSION}"
        )));
    }
    let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let reserved = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if reserved != 0 {
        // Reserved must be 0 in v1; a non-zero value indicates either a
        // forward-compatible v2 file mislabelled with version=1 or a
        // corrupted header. Reject rather than silently ignore.
        return Err(invalid_data(format!(
            "IMQ v1 reserved field must be 0; got {reserved:#x}"
        )));
    }

    let mut map: HashMap<String, Vec<f32>> = HashMap::with_capacity(n_tensors);
    let mut pos = IMQ_HEADER_SIZE;

    for tensor_idx in 0..n_tensors {
        // ── name_len ───────────────────────────────────────────────────
        ensure_room(bytes, pos, 4, tensor_idx, "name_len")?;
        let name_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if name_len > IMQ_MAX_NAME_LEN {
            return Err(invalid_data(format!(
                "tensor {tensor_idx}: name_len {name_len} exceeds IMQ_MAX_NAME_LEN {IMQ_MAX_NAME_LEN}"
            )));
        }
        // ── name (UTF-8) ───────────────────────────────────────────────
        ensure_room(bytes, pos, name_len, tensor_idx, "name")?;
        let name = std::str::from_utf8(&bytes[pos..pos + name_len])
            .map_err(|e| {
                invalid_data(format!(
                    "tensor {tensor_idx}: name is not valid UTF-8: {e}"
                ))
            })?
            .to_string();
        pos += name_len;
        // ── k (u32) ────────────────────────────────────────────────────
        ensure_room(bytes, pos, 4, tensor_idx, "k")?;
        let k = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        // ── n_tokens (u64) — read past it; v1 doesn't expose this to
        // the consumer. A future v2 reader can return a richer struct.
        ensure_room(bytes, pos, 8, tensor_idx, "n_tokens")?;
        pos += 8;
        // ── in_sum2 (K × F32) ──────────────────────────────────────────
        let payload_bytes = k.checked_mul(4).ok_or_else(|| {
            invalid_data(format!(
                "tensor {tensor_idx}: K={k} overflows usize × 4"
            ))
        })?;
        ensure_room(bytes, pos, payload_bytes, tensor_idx, "in_sum2")?;
        let mut values = Vec::with_capacity(k);
        for i in 0..k {
            let off = pos + i * 4;
            values.push(f32::from_le_bytes(
                bytes[off..off + 4].try_into().unwrap(),
            ));
        }
        pos += payload_bytes;

        map.insert(name, values);
    }

    if pos != bytes.len() {
        // Trailing junk after the last record — reject so corrupted
        // files (e.g. an interrupted append) fail loudly rather than
        // partially load.
        return Err(invalid_data(format!(
            "IMQ has {} trailing bytes after last tensor record",
            bytes.len() - pos
        )));
    }
    Ok(map)
}

fn ensure_room(
    bytes: &[u8],
    pos: usize,
    need: usize,
    tensor_idx: usize,
    field: &str,
) -> io::Result<()> {
    if pos + need > bytes.len() {
        return Err(invalid_data(format!(
            "IMQ truncated at tensor {tensor_idx} {field}: need {need} bytes at offset {pos}, file is {}",
            bytes.len()
        )));
    }
    Ok(())
}

fn invalid_data<S: Into<String>>(msg: S) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_imatrix_writer::ImatrixEntry;
    use crate::imq_writer::write_imq;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn entry(name: &str, in_sum2: Vec<f32>, counts: f32) -> ImatrixEntry {
        ImatrixEntry { name: name.to_string(), in_sum2, counts }
    }

    #[test]
    fn loads_back_writer_output() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![
            entry("a.weight", vec![1.0, 2.0, 3.0], 100.0),
            entry("b.weight", vec![-0.5, 0.5], 50.0),
        ];
        write_imq(tf.path(), &entries, None).unwrap();
        let map = load_imq(tf.path()).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a.weight").unwrap().as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(map.get("b.weight").unwrap().as_slice(), &[-0.5, 0.5]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(b"XXX\x00").unwrap();
        tf.write_all(&[0u8; 20]).unwrap();
        tf.flush().unwrap();
        let err = load_imq(tf.path()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_future_version() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(IMQ_MAGIC).unwrap();
        tf.write_all(&99u32.to_le_bytes()).unwrap();
        tf.write_all(&0u64.to_le_bytes()).unwrap();
        tf.write_all(&0u64.to_le_bytes()).unwrap();
        tf.flush().unwrap();
        let err = load_imq(tf.path()).unwrap_err();
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn rejects_nonzero_reserved() {
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(IMQ_MAGIC).unwrap();
        tf.write_all(&IMQ_VERSION.to_le_bytes()).unwrap();
        tf.write_all(&0u64.to_le_bytes()).unwrap();
        tf.write_all(&0xdeadbeefu64.to_le_bytes()).unwrap();
        tf.flush().unwrap();
        let err = load_imq(tf.path()).unwrap_err();
        assert!(err.to_string().contains("reserved"));
    }

    #[test]
    fn rejects_truncated_record() {
        // Header claims 1 tensor, but body is missing.
        let mut tf = NamedTempFile::new().unwrap();
        tf.write_all(IMQ_MAGIC).unwrap();
        tf.write_all(&IMQ_VERSION.to_le_bytes()).unwrap();
        tf.write_all(&1u64.to_le_bytes()).unwrap();
        tf.write_all(&0u64.to_le_bytes()).unwrap();
        tf.flush().unwrap();
        let err = load_imq(tf.path()).unwrap_err();
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn rejects_trailing_junk() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![entry("t", vec![1.0], 1.0)];
        write_imq(tf.path(), &entries, None).unwrap();
        // Append an extra byte.
        let mut bytes = std::fs::read(tf.path()).unwrap();
        bytes.push(0xaa);
        std::fs::write(tf.path(), &bytes).unwrap();
        let err = load_imq(tf.path()).unwrap_err();
        assert!(err.to_string().contains("trailing"));
    }

    #[test]
    fn empty_file_returns_empty_map() {
        let tf = NamedTempFile::new().unwrap();
        write_imq(tf.path(), &[], None).unwrap();
        let map = load_imq(tf.path()).unwrap();
        assert!(map.is_empty());
    }
}
