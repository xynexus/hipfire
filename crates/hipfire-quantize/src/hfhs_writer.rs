//! HFHS (Hipfire Hessian Sidecar) v1 writer.
//!
//! Produces the binary file consumed by [`crate::hessian_io::HessianSidecar`].
//! This is the Rust-native counterpart to `scripts/collect_hessian.py`'s
//! `write_hessian_file()` and replaces it on the Tier 1 BF16 calibration
//! pipeline.
//!
//! Format (v1) — also documented in `scripts/collect_hessian.py:25-43`
//! and `crates/hipfire-quantize/src/hessian_io.rs:28-32`. Mirrored exactly
//! so the existing mmap reader handles the output without changes.
//!
//! ```text
//!   Header (24 bytes):
//!     [0..4)    magic     u8[4]  = b"HFHS"
//!     [4..8)    version   u32_le = 1
//!     [8..16)   n_tensors u64_le
//!     [16..24)  reserved  u64_le = 0
//!
//!   Per-tensor record (variable length):
//!     [0..4)              name_len    u32_le
//!     [4..4+name_len)     name        utf8 bytes
//!     [...4)              expert_idx  u32_le   (default 0; per-expert MoE)
//!     [...4)              K           u32_le
//!     [...4)              dtype_flag  u32_le   (1=F32, 2=F64)
//!     [...K*K*4)          payload     row-major F32 payload (LE)
//! ```
//!
//! Only F32 is emitted today — the Tier 1 collector accumulates in F32 and
//! the format reader promotes to F64 at Cholesky time. F64 storage is left
//! to a future writer if the collector ever exposes a high-precision mode.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// HFHS magic bytes.
const HFHS_MAGIC: &[u8; 4] = b"HFHS";
/// Format version emitted by this writer (matches the reader's
/// `HFHS_VERSION_SUPPORTED`).
const HFHS_VERSION: u32 = 1;
/// dtype flag for F32 payloads (mirrors `DTYPE_F32` in `hessian_io.rs`).
const DTYPE_F32: u32 = 1;

/// One per-tensor Hessian to be written.
pub struct HessianEntry {
    /// Tensor name — typically the `.hfq` tensor name with `.weight` stripped
    /// (see `hessian_io.rs::HessianSidecar::get` for the lookup convention).
    pub name: String,
    /// 0 for dense tensors; per-expert index for MoE.
    pub expert_idx: u32,
    /// Input dimension. Payload length must equal `k * k`.
    pub k: u32,
    /// Row-major Hessian payload (length `k * k`).
    pub h: Vec<f32>,
}

/// Write a v1 HFHS file containing the supplied Hessians. Entries are written
/// in the order supplied — the reader builds a HashMap index keyed by
/// `(name, expert_idx)`, so order does not affect lookup semantics but does
/// influence sequential mmap access patterns.
///
/// Returns `io::Error` on truncated writes, missing parent dir, or malformed
/// entries (payload size mismatch panics via `debug_assert!`; release builds
/// emit whatever the caller supplied — caller is expected to validate
/// upstream).
pub fn write_hfhs(path: &Path, entries: &[HessianEntry]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header (24 bytes)
    w.write_all(HFHS_MAGIC)?;
    w.write_all(&HFHS_VERSION.to_le_bytes())?;
    let n_tensors = entries.len() as u64;
    w.write_all(&n_tensors.to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // reserved

    // Per-tensor records
    for e in entries {
        let name_bytes = e.name.as_bytes();
        let name_len = name_bytes.len() as u32;
        let expected = (e.k as usize) * (e.k as usize);
        debug_assert_eq!(
            e.h.len(),
            expected,
            "HessianEntry {:?}: payload length {} != K*K = {}",
            e.name,
            e.h.len(),
            expected
        );

        w.write_all(&name_len.to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&e.expert_idx.to_le_bytes())?;
        w.write_all(&e.k.to_le_bytes())?;
        w.write_all(&DTYPE_F32.to_le_bytes())?;

        // Payload (K*K * 4 bytes, row-major, LE).
        //
        // SAFETY-style note: `f32::to_le_bytes` per element keeps endian
        // explicit. We could `unsafe { std::slice::from_raw_parts(..., len *
        // 4) }` and write the slab in one shot on LE hosts, but the
        // per-element loop costs roughly nothing through `BufWriter` (4 KB
        // buffer batches writes) and is byte-identical on big-endian hosts.
        // Match `collect_hessian.py`'s `H_final.tobytes(order="C")` semantics:
        // numpy on LE hosts writes raw IEEE-754 LE.
        for &v in &e.h {
            w.write_all(&v.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hessian_io::{HessianDtype, HessianSidecar};
    use tempfile::NamedTempFile;

    fn ent(name: &str, expert: u32, k: u32, h: Vec<f32>) -> HessianEntry {
        HessianEntry { name: name.to_string(), expert_idx: expert, k, h }
    }

    #[test]
    fn roundtrip_two_tensors_matches_reader() {
        let tf = NamedTempFile::new().unwrap();
        let entries = vec![
            // tA: symmetric PSD-ish, K=2, expert=0
            ent("tA", 0, 2, vec![1.0_f32, 0.5, 0.5, 2.0]),
            // tB: K=3, expert=7 (MoE-style)
            ent(
                "tB",
                7,
                3,
                vec![
                    3.0_f32, 1.0, -1.0,
                    1.0, 4.0, 2.0,
                    -1.0, 2.0, 5.0,
                ],
            ),
        ];

        write_hfhs(tf.path(), &entries).expect("write_hfhs");

        let sc = HessianSidecar::open(tf.path()).expect("reader open");
        assert_eq!(sc.n_tensors(), 2);

        let a = sc.get("tA", 0).expect("tA missing");
        assert_eq!(a.k, 2);
        assert_eq!(a.dtype, HessianDtype::F32);
        assert_eq!(a.at(0, 0), 1.0);
        assert_eq!(a.at(0, 1), 0.5);
        assert_eq!(a.at(1, 0), 0.5);
        assert_eq!(a.at(1, 1), 2.0);

        let b = sc.get("tB", 7).expect("tB missing");
        assert_eq!(b.k, 3);
        assert_eq!(b.dtype, HessianDtype::F32);
        assert_eq!(b.at(0, 0), 3.0);
        assert_eq!(b.at(0, 2), -1.0);
        assert_eq!(b.at(1, 1), 4.0);
        assert_eq!(b.at(2, 2), 5.0);

        // Lookups with the wrong expert_idx must miss.
        assert!(sc.get("tB", 0).is_none());
        assert!(sc.get("tA", 7).is_none());
        assert!(sc.get("missing", 0).is_none());
    }

    #[test]
    fn empty_file_writes_valid_header() {
        let tf = NamedTempFile::new().unwrap();
        write_hfhs(tf.path(), &[]).expect("empty write");

        let bytes = std::fs::read(tf.path()).expect("re-read");
        assert_eq!(&bytes[..4], b"HFHS");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);
        assert_eq!(bytes.len(), 24);

        let sc = HessianSidecar::open(tf.path()).expect("empty reader open");
        assert_eq!(sc.n_tensors(), 0);
    }

    #[test]
    fn byte_layout_matches_python_writer() {
        // Mirror of `scripts/collect_hessian.py::write_hessian_file`'s on-disk
        // layout for a single tiny tensor — every offset / size must match
        // byte-for-byte so the existing reader (and any external tool that
        // grew a dependency on it) doesn't regress.
        let tf = NamedTempFile::new().unwrap();
        let e = ent("xyz", 0, 2, vec![1.0_f32, 2.0, 3.0, 4.0]);
        write_hfhs(tf.path(), &[e]).unwrap();
        let bytes = std::fs::read(tf.path()).unwrap();

        // Header
        assert_eq!(&bytes[0..4], b"HFHS");
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(&bytes[8..16], &1u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &0u64.to_le_bytes());
        // Per-tensor record starts at 24
        let mut off = 24;
        assert_eq!(&bytes[off..off + 4], &3u32.to_le_bytes()); // name_len
        off += 4;
        assert_eq!(&bytes[off..off + 3], b"xyz");
        off += 3;
        assert_eq!(&bytes[off..off + 4], &0u32.to_le_bytes()); // expert_idx
        off += 4;
        assert_eq!(&bytes[off..off + 4], &2u32.to_le_bytes()); // K
        off += 4;
        assert_eq!(&bytes[off..off + 4], &1u32.to_le_bytes()); // dtype = F32
        off += 4;
        // Payload: 4 F32s = 16 bytes
        for (i, &v) in [1.0_f32, 2.0, 3.0, 4.0].iter().enumerate() {
            let here = off + i * 4;
            assert_eq!(&bytes[here..here + 4], &v.to_le_bytes());
        }
        off += 16;
        assert_eq!(off, bytes.len(), "trailing bytes");
    }

    #[test]
    fn preserves_payload_order_row_major() {
        // Row-major means H[i, j] is at offset (i * K + j). Verify by reading
        // back through `HessianRef::at`.
        let tf = NamedTempFile::new().unwrap();
        // 3x3 with unique values per cell
        let mut h = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                h.push((i * 10 + j) as f32);
            }
        }
        write_hfhs(tf.path(), &[ent("rm", 0, 3, h)]).unwrap();
        let sc = HessianSidecar::open(tf.path()).unwrap();
        let r = sc.get("rm", 0).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(r.at(i, j), (i * 10 + j) as f64, "H[{i},{j}]");
            }
        }
    }
}
