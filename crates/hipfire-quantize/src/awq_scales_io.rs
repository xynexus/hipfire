//! HFSC (Hipfire Scales) v1 — AWQ-scale sidecar for iterative AWQ+GPTQ rounds.
//!
//! Round-N orchestration of the AWQ-aware-GPTQ v3 algorithm
//! (`scripts/mq4_masked_calib.py::run_iterative_awq_gptq`) needs a way to
//! hand the round-N-1 damped scale vector to the round-N call into the
//! quantizer. The Python prototype writes an HFQ "base" file with
//! `.awq_scale.weight` F16 sidecar tensors; that path is heavy (it copies
//! a full quantized model) so the Rust orchestrator uses this lightweight
//! binary format instead.
//!
//! Format (v1) — mirrors HFHS layout (`hfhs_writer.rs`) for consistency
//! but stores one 1D F32 vector per tensor instead of a K×K matrix.
//!
//! ```text
//!   Header (24 bytes):
//!     [0..4)    magic     u8[4]  = b"HFSC"
//!     [4..8)    version   u32_le = 1
//!     [8..16)   n_tensors u64_le
//!     [16..24)  reserved  u64_le = 0
//!
//!   Per-tensor record (variable length):
//!     [0..4)              name_len    u32_le
//!     [4..4+name_len)     name        utf8 bytes
//!                                     (HFQ tensor name WITHOUT the trailing
//!                                      `.weight` — same convention as HFHS)
//!     [...4)              expert_idx  u32_le   (0 for dense, expert idx for MoE)
//!     [...4)              k           u32_le   (length of the scale vector)
//!     [...k*4)            payload     row-major F32 (LE)
//! ```
//!
//! Consumer integration: `hipfire-quantize` accepts `--awq-scales <path>`
//! to override its imatrix-derived AWQ scales on a per-tensor basis. The
//! lookup convention matches `HessianSidecar::get`: query by name (stripped
//! of `.weight`) and `expert_idx`.

use byteorder::{ByteOrder, LittleEndian};
use memmap2::{Advice, Mmap};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const HFSC_MAGIC: &[u8; 4] = b"HFSC";
const HFSC_VERSION: u32 = 1;
const HEADER_SIZE: usize = 24;

#[derive(Debug)]
pub enum AwqScalesError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    TruncatedFile { needed: usize, have: usize },
    InvalidUtf8,
}

impl std::fmt::Display for AwqScalesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AwqScalesError::Io(e) => write!(f, "I/O error: {e}"),
            AwqScalesError::InvalidMagic(m) => {
                write!(f, "invalid HFSC magic: got {m:?}, expected {:?}", HFSC_MAGIC)
            }
            AwqScalesError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported HFSC version {v}, this build understands v{HFSC_VERSION}"
                )
            }
            AwqScalesError::TruncatedFile { needed, have } => {
                write!(f, "HFSC truncated: needed {needed} bytes, file is {have}")
            }
            AwqScalesError::InvalidUtf8 => write!(f, "HFSC tensor name is not valid UTF-8"),
        }
    }
}

impl std::error::Error for AwqScalesError {}

impl From<std::io::Error> for AwqScalesError {
    fn from(e: std::io::Error) -> Self {
        AwqScalesError::Io(e)
    }
}

/// One per-tensor AWQ-scale vector to be written.
#[derive(Debug, Clone)]
pub struct AwqScaleEntry {
    /// Tensor name — typically the HFQ tensor name with the trailing
    /// `.weight` stripped (matches the HFHS convention so the same lookup
    /// key can be re-used).
    pub name: String,
    /// 0 for dense tensors; per-expert index for MoE.
    pub expert_idx: u32,
    /// Per-input-channel scale vector (length K).
    pub scales: Vec<f32>,
}

/// Write a v1 HFSC file containing the supplied AWQ scales.
pub fn write_awq_scales(path: &Path, entries: &[AwqScaleEntry]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header
    w.write_all(HFSC_MAGIC)?;
    w.write_all(&HFSC_VERSION.to_le_bytes())?;
    let n_tensors = entries.len() as u64;
    w.write_all(&n_tensors.to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // reserved

    // Records
    for e in entries {
        let name_bytes = e.name.as_bytes();
        let name_len = name_bytes.len() as u32;
        let k = e.scales.len() as u32;
        w.write_all(&name_len.to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&e.expert_idx.to_le_bytes())?;
        w.write_all(&k.to_le_bytes())?;
        for &v in &e.scales {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// mmap-backed reader for HFSC files. Builds a `(name, expert_idx) ->
/// scales` index at open time, the same way `HessianSidecar` does.
pub struct AwqScalesSidecar {
    mmap: Mmap,
    _file: File,
    index: HashMap<(String, u32), TensorEntry>,
}

struct TensorEntry {
    payload_offset: usize,
    k: usize,
}

impl AwqScalesSidecar {
    /// Open the file and build the index. Errors out on bad magic /
    /// unsupported version / truncated records.
    pub fn open(path: &Path) -> Result<Self, AwqScalesError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        {
            mmap.advise(Advice::Sequential).ok();
        }
        let _ = Advice::Sequential;

        if mmap.len() < HEADER_SIZE {
            return Err(AwqScalesError::TruncatedFile {
                needed: HEADER_SIZE,
                have: mmap.len(),
            });
        }
        let magic: [u8; 4] = mmap[0..4].try_into().unwrap();
        if &magic != HFSC_MAGIC {
            return Err(AwqScalesError::InvalidMagic(magic));
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != HFSC_VERSION {
            return Err(AwqScalesError::UnsupportedVersion(version));
        }
        let n_tensors = LittleEndian::read_u64(&mmap[8..16]) as usize;
        let _reserved = LittleEndian::read_u64(&mmap[16..24]);

        let mut index = HashMap::with_capacity(n_tensors);
        let mut pos = HEADER_SIZE;
        for _ in 0..n_tensors {
            if pos + 4 > mmap.len() {
                return Err(AwqScalesError::TruncatedFile {
                    needed: pos + 4,
                    have: mmap.len(),
                });
            }
            let name_len = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            if pos + name_len + 8 > mmap.len() {
                return Err(AwqScalesError::TruncatedFile {
                    needed: pos + name_len + 8,
                    have: mmap.len(),
                });
            }
            let name = std::str::from_utf8(&mmap[pos..pos + name_len])
                .map_err(|_| AwqScalesError::InvalidUtf8)?
                .to_string();
            pos += name_len;
            let expert_idx = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let k = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            let payload_bytes = k * 4;
            if pos + payload_bytes > mmap.len() {
                return Err(AwqScalesError::TruncatedFile {
                    needed: pos + payload_bytes,
                    have: mmap.len(),
                });
            }
            index.insert(
                (name, expert_idx),
                TensorEntry {
                    payload_offset: pos,
                    k,
                },
            );
            pos += payload_bytes;
        }

        Ok(Self {
            mmap,
            _file: file,
            index,
        })
    }

    /// Look up the F32 scales for a tensor. Returns `None` if the tensor
    /// is not present — the caller treats this as "no override, use the
    /// imatrix-derived scales for this tensor".
    pub fn get(&self, name: &str, expert_idx: u32) -> Option<Vec<f32>> {
        let entry = self.index.get(&(name.to_string(), expert_idx))?;
        let bytes = &self.mmap[entry.payload_offset..entry.payload_offset + entry.k * 4];
        let mut out = Vec::with_capacity(entry.k);
        for i in 0..entry.k {
            out.push(LittleEndian::read_f32(&bytes[i * 4..i * 4 + 4]));
        }
        Some(out)
    }

    /// Number of tensors in the file.
    pub fn n_tensors(&self) -> usize {
        self.index.len()
    }

    /// All `(name, expert_idx)` keys in deterministic order.
    pub fn keys(&self) -> Vec<(String, u32)> {
        let mut keys: Vec<_> = self.index.keys().cloned().collect();
        keys.sort();
        keys
    }
}

impl std::fmt::Debug for AwqScalesSidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwqScalesSidecar")
            .field("mmap_len", &self.mmap.len())
            .field("n_tensors", &self.index.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn ent(name: &str, expert_idx: u32, scales: Vec<f32>) -> AwqScaleEntry {
        AwqScaleEntry { name: name.to_string(), expert_idx, scales }
    }

    #[test]
    fn roundtrip_two_tensors() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![
            ent("model.layers.0.self_attn.q_proj", 0, vec![1.0_f32, 2.0, 3.0, 4.0]),
            ent("model.layers.5.mlp.experts.gate_up_proj", 7, vec![0.5_f32, 1.5]),
        ];
        write_awq_scales(tf.path(), &entries).expect("write");

        let r = AwqScalesSidecar::open(tf.path()).expect("read");
        assert_eq!(r.n_tensors(), 2);
        let a = r.get("model.layers.0.self_attn.q_proj", 0).unwrap();
        assert_eq!(a, vec![1.0_f32, 2.0, 3.0, 4.0]);
        let b = r.get("model.layers.5.mlp.experts.gate_up_proj", 7).unwrap();
        assert_eq!(b, vec![0.5_f32, 1.5]);
        assert!(r
            .get("model.layers.5.mlp.experts.gate_up_proj", 0)
            .is_none());
    }

    #[test]
    fn empty_file_writes_valid_header() {
        let tf = NamedTempFile::new().unwrap();
        write_awq_scales(tf.path(), &[]).expect("empty write");

        let bytes = std::fs::read(tf.path()).unwrap();
        assert_eq!(&bytes[..4], b"HFSC");
        assert_eq!(bytes.len(), 24);

        let r = AwqScalesSidecar::open(tf.path()).unwrap();
        assert_eq!(r.n_tensors(), 0);
        assert!(r.get("anything", 0).is_none());
    }

    #[test]
    fn bad_magic_rejected() {
        let tf = NamedTempFile::new().unwrap();
        std::fs::write(tf.path(), b"NOPE_______________     ").unwrap();
        let result = AwqScalesSidecar::open(tf.path());
        assert!(matches!(result, Err(AwqScalesError::InvalidMagic(_))));
    }
}
