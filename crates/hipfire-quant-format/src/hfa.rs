// SPDX-License-Identifier: Apache-2.0
//! Random-access reader for `.hfa` (HFAR) source archives.
//!
//! An `.hfa` holds a HuggingFace snapshot directory — `config.json`, the
//! tokenizer files, and the `*.safetensors` shards — with BF16 tensor payloads
//! losslessly recoded. Until now the only way to consume one was
//! `hipfire-coexistence repack --output <dir>`, a full restore: for
//! Qwen3.5-397B-A17B that is a 550 GB archive expanding to ~730 GB on disk
//! before a single tensor can be read.
//!
//! That restore is pure overhead for a quantizer that walks tensors once, and
//! it is fatal for layer-streamed work, which needs to index the source per
//! layer rather than materialise all of it.
//!
//! So this reads tensors straight out of the archive. Each shard is stored as a
//! VERBATIM safetensors header followed by pieces in blob order, each piece
//! independently decodable and carrying its logical length. Cumulative logical
//! lengths therefore give a byte-exact map from a tensor's `data_offsets` to
//! the pieces covering it — so a tensor read decodes only its own pieces, not
//! the shard.
//!
//! The format is defined by `hipfire-coexistence`'s `repack.rs`; this is a
//! reader for it. It deliberately lives here rather than there because
//! `hipfire-coexistence` already depends on `hipfire-quantize`, so the
//! dependency cannot run the other way.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"HFAR0002";
const MAGIC_V1: &[u8; 8] = b"HFAR0001";

/// One independently decodable run of bytes inside a shard's blob.
#[derive(Debug, Clone)]
struct Piece {
    stored_off: u64,
    stored_len: u64,
    /// Logical (decoded) length. The piece occupies exactly this many bytes of
    /// the reconstructed blob, which is what makes the offset map exact.
    len: u64,
    codec: String,
}

#[derive(Debug, Clone)]
enum Entry {
    /// Stored verbatim — small files (config.json, tokenizer) and any shard the
    /// packer could not improve on.
    Raw { off: u64, len: u64 },
    Shard {
        header_off: u64,
        header_len: u64,
        pieces: Vec<Piece>,
        /// Prefix sums of `pieces[i].len`, one longer than `pieces`.
        starts: Vec<u64>,
    },
}

pub struct HfaArchive {
    path: PathBuf,
    entries: Vec<(String, Entry)>,
}

/// Cheap sniff: does this path name an HFAR archive? Used to decide whether
/// `--input` is an archive or a directory without committing to a parse.
pub fn is_hfa(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).is_ok() && (&magic == MAGIC || &magic == MAGIC_V1)
}

/// Map a logical byte range onto the pieces covering it.
///
/// Returns `(piece_index, offset_within_piece, length)` triples. Split out as a
/// pure function because this is where the arithmetic can go wrong: a tensor
/// rarely starts on a piece boundary, and `blob_plan` splits large tensors into
/// fixed-size units, so a single tensor routinely spans many pieces.
fn map_range(starts: &[u64], start: u64, end: u64) -> Vec<(usize, u64, u64)> {
    let mut out = Vec::new();
    if end <= start {
        return out;
    }
    // First piece whose end is past `start`.
    let mut i = match starts.binary_search(&start) {
        Ok(i) => i,
        Err(i) => i - 1,
    };
    while i + 1 < starts.len() && starts[i] < end {
        let p_start = starts[i];
        let p_end = starts[i + 1];
        let lo = start.max(p_start);
        let hi = end.min(p_end);
        if hi > lo {
            out.push((i, lo - p_start, hi - lo));
        }
        i += 1;
    }
    out
}

fn decode_piece(codec: &str, stored: Vec<u8>, logical_len: u64) -> Option<Vec<u8>> {
    let out = match codec {
        // Parallel decode: serial `decode` measured ~140 MB/s on this box, which
        // is ~1.4 h for a 730 GB model and would dominate the read it is meant
        // to avoid. Small pieces stay serial — the split costs more than it
        // saves below a few MB.
        "bf16h" => {
            let n = (logical_len / 2) as usize;
            if logical_len >= 4 << 20 {
                let threads = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1);
                hipfire_primitives::bf16_huff::decode_par(&stored, n, threads)?
            } else {
                hipfire_primitives::bf16_huff::decode(&stored, n)?
            }
        }
        _ => stored,
    };
    (out.len() as u64 == logical_len).then_some(out)
}

impl HfaArchive {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("hfa: open {}: {e}", path.display()))?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)
            .map_err(|e| format!("hfa: read magic: {e}"))?;
        if &magic != MAGIC && &magic != MAGIC_V1 {
            return Err(format!(
                "{} is not a hipfire repack archive",
                path.display()
            ));
        }
        let total = f.metadata().map_err(|e| format!("hfa: stat: {e}"))?.len();
        if total < 24 {
            return Err("hfa: archive is too short to hold an index".into());
        }
        f.seek(SeekFrom::Start(total - 16))
            .map_err(|e| format!("hfa: seek trailer: {e}"))?;
        let mut b = [0u8; 16];
        f.read_exact(&mut b)
            .map_err(|e| format!("hfa: read trailer: {e}"))?;
        let index_off = u64::from_le_bytes(b[..8].try_into().unwrap());
        let index_len = u64::from_le_bytes(b[8..].try_into().unwrap());
        if index_off >= total || index_len > total || index_off + index_len > total {
            return Err("hfa: index offset/length out of range".into());
        }
        f.seek(SeekFrom::Start(index_off))
            .map_err(|e| format!("hfa: seek index: {e}"))?;
        let mut ib = vec![0u8; index_len as usize];
        f.read_exact(&mut ib)
            .map_err(|e| format!("hfa: read index: {e}"))?;
        let index: serde_json::Value =
            serde_json::from_slice(&ib).map_err(|e| format!("hfa: parse index: {e}"))?;
        let files = index
            .get("files")
            .and_then(|v| v.as_array())
            .ok_or("hfa: archive index has no files")?;

        let mut entries = Vec::with_capacity(files.len());
        for fe in files {
            let rel = fe
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("hfa: index entry has no path")?
                .to_string();
            let entry = match fe.get("kind").and_then(|v| v.as_str()) {
                Some("raw") => Entry::Raw {
                    off: fe
                        .get("off")
                        .and_then(|v| v.as_u64())
                        .ok_or("hfa: raw entry has no off")?,
                    len: fe
                        .get("len")
                        .and_then(|v| v.as_u64())
                        .ok_or("hfa: raw entry has no len")?,
                },
                Some("safetensors") => {
                    let header_off = fe
                        .get("header_off")
                        .and_then(|v| v.as_u64())
                        .ok_or("hfa: no header_off")?;
                    let header_len = fe
                        .get("header_len")
                        .and_then(|v| v.as_u64())
                        .ok_or("hfa: no header_len")?;
                    let tensors = fe
                        .get("tensors")
                        .and_then(|v| v.as_array())
                        .ok_or("hfa: no tensors")?;
                    let mut pieces = Vec::with_capacity(tensors.len());
                    let mut starts = Vec::with_capacity(tensors.len() + 1);
                    let mut acc = 0u64;
                    for t in tensors {
                        starts.push(acc);
                        let len = t.get("len").and_then(|v| v.as_u64()).ok_or("hfa: no len")?;
                        pieces.push(Piece {
                            stored_off: t
                                .get("stored_off")
                                .and_then(|v| v.as_u64())
                                .ok_or("hfa: no stored_off")?,
                            stored_len: t
                                .get("stored_len")
                                .and_then(|v| v.as_u64())
                                .ok_or("hfa: no stored_len")?,
                            len,
                            codec: t
                                .get("codec")
                                .and_then(|v| v.as_str())
                                .unwrap_or("raw")
                                .to_string(),
                        });
                        acc += len;
                    }
                    starts.push(acc);
                    Entry::Shard {
                        header_off,
                        header_len,
                        pieces,
                        starts,
                    }
                }
                other => return Err(format!("hfa: unknown archive entry kind {other:?}")),
            };
            entries.push((rel, entry));
        }
        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    pub fn file_names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(p, _)| p.as_str())
    }

    /// Shard paths, in index order.
    pub fn safetensors_names(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|(p, e)| matches!(e, Entry::Shard { .. }) || p.ends_with(".safetensors"))
            .map(|(p, _)| p.clone())
            .collect()
    }

    fn entry(&self, rel: &str) -> Result<&Entry, String> {
        self.entries
            .iter()
            .find(|(p, _)| p == rel)
            .map(|(_, e)| e)
            .ok_or_else(|| format!("hfa: {rel} is not in {}", self.path.display()))
    }

    /// Fully reconstruct a small file — `config.json`, tokenizer files. Not for
    /// shards: it would materialise the whole blob, which is the cost this
    /// reader exists to avoid.
    pub fn read_small_file(&self, rel: &str) -> Result<Vec<u8>, String> {
        let mut f = File::open(&self.path).map_err(|e| format!("hfa: open: {e}"))?;
        match self.entry(rel)? {
            Entry::Raw { off, len } => {
                f.seek(SeekFrom::Start(*off))
                    .map_err(|e| format!("hfa: seek {rel}: {e}"))?;
                let mut buf = vec![0u8; *len as usize];
                f.read_exact(&mut buf)
                    .map_err(|e| format!("hfa: read {rel}: {e}"))?;
                Ok(buf)
            }
            Entry::Shard { .. } => Err(format!(
                "hfa: {rel} is a safetensors shard; use tensor_bytes/shard_header"
            )),
        }
    }

    /// The shard's safetensors header JSON (without the 8-byte length prefix).
    pub fn shard_header(&self, rel: &str) -> Result<serde_json::Value, String> {
        let Entry::Shard {
            header_off,
            header_len,
            ..
        } = self.entry(rel)?
        else {
            return Err(format!("hfa: {rel} is not a safetensors shard"));
        };
        let mut f = File::open(&self.path).map_err(|e| format!("hfa: open: {e}"))?;
        f.seek(SeekFrom::Start(*header_off))
            .map_err(|e| format!("hfa: seek header: {e}"))?;
        let mut hdr = vec![0u8; *header_len as usize];
        f.read_exact(&mut hdr)
            .map_err(|e| format!("hfa: read header: {e}"))?;
        serde_json::from_slice(&hdr).map_err(|e| format!("hfa: parse {rel} header: {e}"))
    }

    /// Decode exactly the blob range `[start, end)` of a shard.
    pub fn read_blob_range(&self, rel: &str, start: u64, end: u64) -> Result<Vec<u8>, String> {
        let Entry::Shard { pieces, starts, .. } = self.entry(rel)? else {
            return Err(format!("hfa: {rel} is not a safetensors shard"));
        };
        let blob_len = *starts.last().unwrap_or(&0);
        if end > blob_len {
            return Err(format!(
                "hfa: {rel} range {start}..{end} runs past the {blob_len}-byte blob"
            ));
        }
        let mut f = File::open(&self.path).map_err(|e| format!("hfa: open: {e}"))?;
        let mut out = Vec::with_capacity((end - start) as usize);
        for (idx, off_in_piece, len) in map_range(starts, start, end) {
            let p = &pieces[idx];
            f.seek(SeekFrom::Start(p.stored_off))
                .map_err(|e| format!("hfa: seek piece: {e}"))?;
            let mut sb = vec![0u8; p.stored_len as usize];
            f.read_exact(&mut sb)
                .map_err(|e| format!("hfa: read piece: {e}"))?;
            let logical = decode_piece(&p.codec, sb, p.len)
                .ok_or_else(|| format!("hfa: corrupt {} piece in {rel}", p.codec))?;
            let lo = off_in_piece as usize;
            out.extend_from_slice(&logical[lo..lo + len as usize]);
        }
        Ok(out)
    }

    /// Tensor payload by name, plus its dtype and shape from the header.
    pub fn tensor_bytes(
        &self,
        rel: &str,
        name: &str,
    ) -> Result<(Vec<u8>, String, Vec<usize>), String> {
        let hdr = self.shard_header(rel)?;
        let meta = hdr
            .get(name)
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("hfa: {rel} has no tensor {name}"))?;
        let offs = meta
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or("hfa: tensor has no data_offsets")?;
        let start = offs
            .first()
            .and_then(|v| v.as_u64())
            .ok_or("hfa: bad offs")?;
        let end = offs
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or("hfa: bad offs")?;
        let dtype = meta
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let shape = meta
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        Ok((self.read_blob_range(rel, start, end)?, dtype, shape))
    }

    /// Every tensor in the archive, mapped to the shard holding it — the
    /// index a caller needs to walk a model without restoring it.
    pub fn tensor_index(&self) -> Result<HashMap<String, String>, String> {
        let mut map = HashMap::new();
        for rel in self.safetensors_names() {
            let hdr = self.shard_header(&rel)?;
            if let Some(obj) = hdr.as_object() {
                for k in obj.keys() {
                    if k != "__metadata__" {
                        map.insert(k.clone(), rel.clone());
                    }
                }
            }
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offset arithmetic is the whole reader. A tensor almost never starts
    /// on a piece boundary and `blob_plan` splits big tensors into fixed units,
    /// so spanning several pieces with partial ends is the normal case, not an
    /// edge case.
    #[test]
    fn map_range_covers_partial_and_spanning_reads() {
        // Pieces of 10, 10, 5 bytes -> boundaries 0,10,20,25.
        let starts = [0u64, 10, 20, 25];

        // Wholly inside one piece.
        assert_eq!(map_range(&starts, 2, 8), vec![(0, 2, 6)]);
        // Exactly one whole piece.
        assert_eq!(map_range(&starts, 10, 20), vec![(1, 0, 10)]);
        // Spans all three, partial at both ends.
        assert_eq!(
            map_range(&starts, 5, 23),
            vec![(0, 5, 5), (1, 0, 10), (2, 0, 3)]
        );
        // Touching the very end.
        assert_eq!(map_range(&starts, 24, 25), vec![(2, 4, 1)]);
        // Empty and inverted ranges yield nothing rather than panicking.
        assert!(map_range(&starts, 7, 7).is_empty());
        assert!(map_range(&starts, 9, 3).is_empty());

        // Every single-byte read must land in exactly one piece, and the
        // reassembled length must equal the request for every range.
        for start in 0..25u64 {
            assert_eq!(map_range(&starts, start, start + 1).len(), 1);
            for end in start..=25 {
                let got: u64 = map_range(&starts, start, end)
                    .iter()
                    .map(|(_, _, l)| l)
                    .sum();
                assert_eq!(got, end - start, "range {start}..{end}");
            }
        }
    }
}
