// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Architecture-agnostic lossless repacking of a model directory.
//!
//! The quantizer refuses a model it cannot recognise — `derive_arch_id` rejects
//! an unknown `model_type` at open time, before a single tensor byte is read —
//! and where it does convert, it may transform: gemma3 bakes `+1.0` into its
//! RMSNorm weights, and some loaders drop a vision tower. Every one of those is
//! a *runtime* concern. Compression has none: `bf16_huff` takes `&[u8]` and
//! returns `Option<Vec<u8>>`, and everything it needs — name, dtype, shape,
//! bytes — is already in the safetensors header, which is self-describing.
//!
//! So this reads no `config.json`, resolves no architecture, and interprets no
//! tensor name. It compresses BF16 payloads, stores everything else verbatim,
//! and reproduces the source **byte for byte**.
//!
//! # Why the header is stored verbatim
//!
//! A safetensors file is `[u64 header_len][header JSON][data blob]`. Rebuilding
//! the header from parsed values would have to reproduce its exact key order and
//! spacing to be byte-identical, which the format does not guarantee. Keeping
//! the original bytes sidesteps that entirely: unpacking writes back the same
//! length, the same header, and a blob reassembled from the same offsets.
//!
//! If a file's tensors do not exactly tile its blob — a gap, an overlap, padding
//! this code does not know about — the whole file is stored raw rather than
//! guessed at. Losing the compression on one shard beats losing its bytes.
//!
//! # Archive layout
//!
//! ```text
//! [8]  magic "HFAR0001"
//! [..] payloads, in index order
//! [..] index JSON
//! [8]  u64 index offset
//! [8]  u64 index length
//! ```
//!
//! The index is written last because a payload's stored length is not known
//! until it has been compressed, and compressing twice to learn it would double
//! the work.

use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"HFAR0001";
/// Copy granularity for verbatim payloads. Bounds memory on a multi-GB shard.
const CH: usize = 8 << 20;

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            other => return Err(format!("repack: unexpected argument {other:?}").into()),
        }
    }
    let input = input.ok_or("repack requires --input")?;
    let output = output.ok_or("repack requires --output")?;
    if input.is_dir() {
        pack(&input, &output)
    } else {
        unpack(&input, &output)
    }
}

/// One tensor inside a safetensors shard.
struct Entry {
    name: String,
    /// Offset within the shard's data blob.
    off: u64,
    len: u64,
    /// "bf16h" when the payload is Huffman-recoded, "raw" otherwise.
    codec: &'static str,
    stored_off: u64,
    stored_len: u64,
}

fn read_st_header(path: &Path) -> Option<(u64, Vec<u8>, serde_json::Value, u64)> {
    let mut f = File::open(path).ok()?;
    let total = f.metadata().ok()?.len();
    let mut n = [0u8; 8];
    f.read_exact(&mut n).ok()?;
    let hlen = u64::from_le_bytes(n);
    if hlen > 512 << 20 || 8 + hlen > total {
        return None;
    }
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr).ok()?;
    let parsed: serde_json::Value = serde_json::from_slice(&hdr).ok()?;
    Some((hlen, hdr, parsed, total - 8 - hlen))
}

/// Tensors sorted by offset, or `None` when they do not exactly tile the blob.
fn tiling(parsed: &serde_json::Value, blob_len: u64) -> Option<Vec<(String, u64, u64, String)>> {
    let obj = parsed.as_object()?;
    let mut v: Vec<(String, u64, u64, String)> = Vec::new();
    for (k, meta) in obj {
        if k == "__metadata__" {
            continue;
        }
        let m = meta.as_object()?;
        let o = m.get("data_offsets")?.as_array()?;
        let (s, e) = (o.first()?.as_u64()?, o.get(1)?.as_u64()?);
        let dtype = m.get("dtype")?.as_str()?.to_string();
        if e < s {
            return None;
        }
        v.push((k.clone(), s, e - s, dtype));
    }
    v.sort_by_key(|t| t.1);
    let mut cursor = 0u64;
    for (_, s, l, _) in &v {
        if *s != cursor {
            return None; // gap or overlap
        }
        cursor += l;
    }
    if cursor != blob_len {
        return None; // trailing bytes this code does not model
    }
    Some(v)
}

fn pack(dir: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    collect(dir, dir, 0, &mut paths);
    paths.sort();

    let mut w = std::io::BufWriter::new(File::create(out)?);
    w.write_all(MAGIC)?;
    let mut pos = MAGIC.len() as u64;
    let mut files_json: Vec<serde_json::Value> = Vec::new();
    let (mut n_bf16, mut n_raw, mut before, mut after) = (0usize, 0usize, 0u64, 0u64);

    for p in &paths {
        let rel = p.strip_prefix(dir)?.to_string_lossy().replace('\\', "/");
        let is_st = p.extension().map(|e| e == "safetensors").unwrap_or(false);
        let parsed = if is_st { read_st_header(p) } else { None };

        // Non-safetensors, or a shard whose tensors do not tile: store verbatim.
        let Some((hlen, hdr, parsed, blob_len)) = parsed else {
            let (o, l) = copy_raw(p, 0, None, &mut w, &mut pos)?;
            files_json.push(serde_json::json!({
                "path": rel, "kind": "raw", "off": o, "len": l
            }));
            n_raw += 1;
            continue;
        };
        let Some(tensors) = tiling(&parsed, blob_len) else {
            let (o, l) = copy_raw(p, 0, None, &mut w, &mut pos)?;
            files_json.push(serde_json::json!({
                "path": rel, "kind": "raw", "off": o, "len": l
            }));
            eprintln!("repack: {rel} tensors do not tile its blob — stored verbatim");
            n_raw += 1;
            continue;
        };

        let base = 8 + hlen;
        let mut f = File::open(p)?;
        let mut entries: Vec<Entry> = Vec::new();
        for (name, off, len, dtype) in tensors {
            f.seek(SeekFrom::Start(base + off))?;
            let mut buf = vec![0u8; len as usize];
            f.read_exact(&mut buf)?;
            before += len;
            let (codec, bytes) = if dtype == "BF16" {
                match hipfire_primitives::bf16_huff::encode_if_smaller(&buf) {
                    Some(packed) => ("bf16h", packed),
                    None => ("raw", buf),
                }
            } else {
                ("raw", buf)
            };
            if codec == "bf16h" {
                n_bf16 += 1;
            }
            let stored_off = pos;
            w.write_all(&bytes)?;
            pos += bytes.len() as u64;
            after += bytes.len() as u64;
            entries.push(Entry {
                name,
                off,
                len,
                codec,
                stored_off,
                stored_len: bytes.len() as u64,
            });
        }
        // Header bytes verbatim — the only way to guarantee a byte-identical file.
        let hdr_off = pos;
        w.write_all(&hdr)?;
        pos += hdr.len() as u64;

        files_json.push(serde_json::json!({
            "path": rel, "kind": "safetensors",
            "header_off": hdr_off, "header_len": hlen, "blob_len": blob_len,
            "tensors": entries.iter().map(|e| serde_json::json!({
                "name": e.name, "off": e.off, "len": e.len,
                "codec": e.codec, "stored_off": e.stored_off, "stored_len": e.stored_len
            })).collect::<Vec<_>>()
        }));
    }

    let index = serde_json::to_vec(&serde_json::json!({ "files": files_json }))?;
    let index_off = pos;
    w.write_all(&index)?;
    w.write_all(&index_off.to_le_bytes())?;
    w.write_all(&(index.len() as u64).to_le_bytes())?;
    w.flush()?;

    let total = std::fs::metadata(out)?.len();
    let src: u64 = paths.iter().filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum();
    eprintln!(
        "repack: {} files ({} shards, {} verbatim), {} BF16 tensors recoded \
         {:.2} GB -> {:.2} GB; archive {:.2} GB vs source {:.2} GB ({:.4}x)",
        paths.len(),
        files_json.len() - n_raw,
        n_raw,
        n_bf16,
        before as f64 / 1e9,
        after as f64 / 1e9,
        total as f64 / 1e9,
        src as f64 / 1e9,
        src as f64 / total.max(1) as f64
    );
    Ok(())
}

fn collect(root: &Path, dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Dot-entries are included: this is an archive, so omitting a file is a
        // correctness bug, not tidiness. `.eval_results/` in a real checkpoint
        // was silently dropped when this skipped them. `.git` is the one
        // exclusion -- it is VCS state, often larger than the weights, and not
        // part of the model.
        if name == ".git" {
            continue;
        }
        if p.is_dir() {
            collect(root, &p, depth + 1, out);
        } else if p.is_file() {
            out.push(p);
        }
    }
}

fn copy_raw(
    src: &Path,
    from: u64,
    len: Option<u64>,
    w: &mut impl Write,
    pos: &mut u64,
) -> Result<(u64, u64), Box<dyn Error>> {
    let mut f = File::open(src)?;
    f.seek(SeekFrom::Start(from))?;
    let want = len.unwrap_or_else(|| f.metadata().map(|m| m.len() - from).unwrap_or(0));
    let start = *pos;
    let mut left = want;
    let mut buf = vec![0u8; CH.min(want.max(1) as usize)];
    while left > 0 {
        let n = (buf.len() as u64).min(left) as usize;
        f.read_exact(&mut buf[..n])?;
        w.write_all(&buf[..n])?;
        left -= n as u64;
        *pos += n as u64;
    }
    Ok((start, want))
}

fn unpack(archive: &Path, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut f = File::open(archive)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err("not a hipfire repack archive".into());
    }
    let total = f.metadata()?.len();
    f.seek(SeekFrom::Start(total - 16))?;
    let mut b = [0u8; 16];
    f.read_exact(&mut b)?;
    let index_off = u64::from_le_bytes(b[..8].try_into()?);
    let index_len = u64::from_le_bytes(b[8..].try_into()?);
    f.seek(SeekFrom::Start(index_off))?;
    let mut ib = vec![0u8; index_len as usize];
    f.read_exact(&mut ib)?;
    let index: serde_json::Value = serde_json::from_slice(&ib)?;

    let files = index.get("files").and_then(|v| v.as_array()).ok_or("archive index has no files")?;
    let mut n = 0usize;
    for fe in files {
        let rel = fe.get("path").and_then(|v| v.as_str()).ok_or("index entry has no path")?;
        let dest = out_dir.join(rel);
        if !dest.starts_with(out_dir) {
            eprintln!("repack: skipping {rel} — escapes the output directory");
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut o = std::io::BufWriter::new(File::create(&dest)?);

        match fe.get("kind").and_then(|v| v.as_str()) {
            Some("raw") => {
                let off = fe.get("off").and_then(|v| v.as_u64()).ok_or("raw entry has no off")?;
                let len = fe.get("len").and_then(|v| v.as_u64()).ok_or("raw entry has no len")?;
                f.seek(SeekFrom::Start(off))?;
                let mut left = len;
                let mut buf = vec![0u8; CH.min(len.max(1) as usize)];
                while left > 0 {
                    let k = (buf.len() as u64).min(left) as usize;
                    f.read_exact(&mut buf[..k])?;
                    o.write_all(&buf[..k])?;
                    left -= k as u64;
                }
            }
            Some("safetensors") => {
                let hoff = fe.get("header_off").and_then(|v| v.as_u64()).ok_or("no header_off")?;
                let hlen = fe.get("header_len").and_then(|v| v.as_u64()).ok_or("no header_len")?;
                f.seek(SeekFrom::Start(hoff))?;
                let mut hdr = vec![0u8; hlen as usize];
                f.read_exact(&mut hdr)?;
                o.write_all(&hlen.to_le_bytes())?;
                o.write_all(&hdr)?;
                // Tensors were written in offset order, so writing them back in
                // index order reproduces the blob exactly.
                let tensors = fe.get("tensors").and_then(|v| v.as_array()).ok_or("no tensors")?;
                for t in tensors {
                    let so = t.get("stored_off").and_then(|v| v.as_u64()).ok_or("no stored_off")?;
                    let sl = t.get("stored_len").and_then(|v| v.as_u64()).ok_or("no stored_len")?;
                    let ln = t.get("len").and_then(|v| v.as_u64()).ok_or("no len")?;
                    let codec = t.get("codec").and_then(|v| v.as_str()).unwrap_or("raw");
                    f.seek(SeekFrom::Start(so))?;
                    let mut sb = vec![0u8; sl as usize];
                    f.read_exact(&mut sb)?;
                    if codec == "bf16h" {
                        let d = hipfire_primitives::bf16_huff::decode(&sb, (ln / 2) as usize)
                            .ok_or_else(|| {
                                format!("corrupt bf16h payload for {:?}", t.get("name"))
                            })?;
                        if d.len() as u64 != ln {
                            return Err(format!("bf16h expanded to {} bytes, expected {ln}", d.len()).into());
                        }
                        o.write_all(&d)?;
                    } else {
                        o.write_all(&sb)?;
                    }
                }
            }
            other => return Err(format!("unknown archive entry kind {other:?}").into()),
        }
        o.flush()?;
        n += 1;
    }
    eprintln!("repack: restored {n} files to {}", out_dir.display());
    Ok(())
}
