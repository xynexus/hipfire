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

/// v2 records an XXH3-64 over every stored payload. v1 archives predate that
/// and stay readable -- they hold the only copy of models whose sources are
/// gone, so refusing them to gain a checksum would be the wrong trade.
const MAGIC: &[u8; 8] = b"HFAR0002";
const MAGIC_V1: &[u8; 8] = b"HFAR0001";

/// XXH3-64 of a stored payload. Detects the corruption a read sweep cannot:
/// bytes that come back without an I/O error but are no longer what was
/// written. On a no-redundancy array that is the failure mode with no other
/// backstop.
fn xxh3(bytes: &[u8]) -> u64 {
    twox_hash::XxHash3_64::oneshot(bytes)
}
/// Copy granularity for verbatim payloads. Bounds memory on a multi-GB shard.
const CH: usize = 8 << 20;
/// Encode unit. Caps peak memory at roughly 1.7x this (the piece plus its
/// packed copy) however large a tensor is, and keeps every unit far below the
/// u32 element count `bf16_huff` can describe.
const UNIT: u64 = 1 << 30;

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut verify: Option<PathBuf> = None;
    let mut check_only = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--verify" => verify = it.next().map(PathBuf::from),
            "--check" => check_only = true,
            other => return Err(format!("repack: unexpected argument {other:?}").into()),
        }
    }
    let input = input.ok_or("repack requires --input")?;
    // Integrity without the source: the archives whose sources were deleted have
    // nothing left to compare against, so the stored checksums are the only
    // remaining evidence they are still what was written.
    if check_only {
        return check(&input);
    }
    // Verify compares the archive against the source without materialising it.
    // A restore-then-diff needs temp space equal to the whole model, which is
    // exactly what is unavailable when the model is large enough to matter.
    if let Some(src) = verify {
        return restore(&input, None, Some(&src));
    }
    let output = output.ok_or("repack requires --output")?;
    if input.is_dir() {
        pack(&input, &output)
    } else {
        restore(&input, Some(&output), None)
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
    /// XXH3-64 of the payload as stored. Detects an archive that rotted at rest.
    xxh3: u64,
    /// XXH3-64 of the ORIGINAL source bytes, before any recoding.
    ///
    /// The stored hash cannot catch a payload that is intact but decodes wrong
    /// -- which is precisely what the u32 chunk-offset overflow produced: a
    /// well-formed bitstream whose chunks decoded from wrapped positions, with
    /// `Max quant error: 0.00000000` reported alongside. Hashing the source
    /// content makes a restore verifiable end to end, codec included.
    src_xxh3: u64,
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
            let (o, l, x) = copy_raw(p, 0, None, &mut w, &mut pos)?;
            files_json.push(serde_json::json!({
                "path": rel, "kind": "raw", "off": o, "len": l, "xxh3": x
            }));
            n_raw += 1;
            continue;
        };
        let Some(tensors) = tiling(&parsed, blob_len) else {
            let (o, l, x) = copy_raw(p, 0, None, &mut w, &mut pos)?;
            files_json.push(serde_json::json!({
                "path": rel, "kind": "raw", "off": o, "len": l, "xxh3": x
            }));
            eprintln!("repack: {rel} tensors do not tile its blob — stored verbatim");
            n_raw += 1;
            continue;
        };

        let base = 8 + hlen;
        let mut f = File::open(p)?;
        let mut entries: Vec<Entry> = Vec::new();
        for (name, off, len, dtype) in tensors {
            // Split into UNIT-sized pieces, each encoded independently.
            //
            // Two problems fall out of treating a tensor as one unit. A 21.47 GB
            // stacked-expert tensor is a 21.47 GB anonymous allocation, which
            // thrashes a 31 GB host; and `encode_if_smaller` declines anything
            // past u32::MAX elements (~8.6 GB of BF16), so the largest tensors
            // were stored raw. Llama-4-Maverick is 100% BF16 and still came out
            // at 1.0126x for exactly that reason.
            //
            // Pieces are contiguous and recorded in offset order, and restore
            // already writes entries back in that order, so a split tensor
            // reassembles with no reader change.
            let mut done = 0u64;
            let mut part = 0usize;
            while done < len {
                let take = UNIT.min(len - done);
                f.seek(SeekFrom::Start(base + off + done))?;
                let mut buf = vec![0u8; take as usize];
                f.read_exact(&mut buf)?;
                before += take;
                let src_hash = xxh3(&buf);
                let (codec, bytes) = if dtype == "BF16" && take % 2 == 0 {
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
                    name: if part == 0 {
                        name.clone()
                    } else {
                        format!("{name}#{part}")
                    },
                    off: off + done,
                    len: take,
                    codec,
                    stored_off,
                    stored_len: bytes.len() as u64,
                    xxh3: xxh3(&bytes),
                    src_xxh3: src_hash,
                });
                done += take;
                part += 1;
            }
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
                "codec": e.codec, "stored_off": e.stored_off, "stored_len": e.stored_len,
                "xxh3": e.xxh3, "src_xxh3": e.src_xxh3
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
    let src: u64 = paths
        .iter()
        .filter_map(|p| p.metadata().ok())
        .map(|m| m.len())
        .sum();
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
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
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
) -> Result<(u64, u64, u64), Box<dyn Error>> {
    let mut f = File::open(src)?;
    f.seek(SeekFrom::Start(from))?;
    let want = len.unwrap_or_else(|| f.metadata().map(|m| m.len() - from).unwrap_or(0));
    let start = *pos;
    let mut left = want;
    let mut buf = vec![0u8; CH.min(want.max(1) as usize)];
    // Hashed streaming so a multi-GB verbatim shard is never held whole.
    let mut h = twox_hash::xxhash3_64::Hasher::new();
    while left > 0 {
        let n = (buf.len() as u64).min(left) as usize;
        f.read_exact(&mut buf[..n])?;
        std::hash::Hasher::write(&mut h, &buf[..n]);
        w.write_all(&buf[..n])?;
        left -= n as u64;
        *pos += n as u64;
    }
    Ok((start, want, std::hash::Hasher::finish(&h)))
}

/// Reconstruct the archive, either writing it to `out_dir` or comparing it
/// against `check` byte for byte. One code path, so verification exercises
/// exactly the bytes a restore would produce.
fn restore(
    archive: &Path,
    out_dir: Option<&Path>,
    check: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let mut f = File::open(archive)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC && &magic != MAGIC_V1 {
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

    let files = index
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or("archive index has no files")?;
    let mut n = 0usize;
    for fe in files {
        let rel = fe
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("index entry has no path")?;
        let mut o: Sink = match (out_dir, check) {
            (Some(d), _) => {
                let dest = d.join(rel);
                if !dest.starts_with(d) {
                    eprintln!("repack: skipping {rel} — escapes the output directory");
                    continue;
                }
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                Sink::Write(std::io::BufWriter::new(File::create(&dest)?))
            }
            (None, Some(src)) => Sink::Compare {
                f: File::open(src.join(rel))
                    .map_err(|e| format!("verify: cannot open source {rel}: {e}"))?,
                path: rel.to_string(),
                ok: true,
                read: 0,
            },
            (None, None) => return Err("restore needs an output or a verify target".into()),
        };

        match fe.get("kind").and_then(|v| v.as_str()) {
            Some("raw") => {
                let off = fe
                    .get("off")
                    .and_then(|v| v.as_u64())
                    .ok_or("raw entry has no off")?;
                let len = fe
                    .get("len")
                    .and_then(|v| v.as_u64())
                    .ok_or("raw entry has no len")?;
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
                let hoff = fe
                    .get("header_off")
                    .and_then(|v| v.as_u64())
                    .ok_or("no header_off")?;
                let hlen = fe
                    .get("header_len")
                    .and_then(|v| v.as_u64())
                    .ok_or("no header_len")?;
                f.seek(SeekFrom::Start(hoff))?;
                let mut hdr = vec![0u8; hlen as usize];
                f.read_exact(&mut hdr)?;
                o.write_all(&hlen.to_le_bytes())?;
                o.write_all(&hdr)?;
                // Tensors were written in offset order, so writing them back in
                // index order reproduces the blob exactly.
                let tensors = fe
                    .get("tensors")
                    .and_then(|v| v.as_array())
                    .ok_or("no tensors")?;
                for t in tensors {
                    let so = t
                        .get("stored_off")
                        .and_then(|v| v.as_u64())
                        .ok_or("no stored_off")?;
                    let sl = t
                        .get("stored_len")
                        .and_then(|v| v.as_u64())
                        .ok_or("no stored_len")?;
                    let ln = t.get("len").and_then(|v| v.as_u64()).ok_or("no len")?;
                    let codec = t.get("codec").and_then(|v| v.as_str()).unwrap_or("raw");
                    f.seek(SeekFrom::Start(so))?;
                    let mut sb = vec![0u8; sl as usize];
                    f.read_exact(&mut sb)?;
                    let logical: Vec<u8> = if codec == "bf16h" {
                        let d = hipfire_primitives::bf16_huff::decode(&sb, (ln / 2) as usize)
                            .ok_or_else(|| {
                                format!("corrupt bf16h payload for {:?}", t.get("name"))
                            })?;
                        if d.len() as u64 != ln {
                            return Err(format!(
                                "bf16h expanded to {} bytes, expected {ln}",
                                d.len()
                            )
                            .into());
                        }
                        d
                    } else {
                        sb
                    };
                    // The decoded content must match what was packed. A payload
                    // can be byte-perfect and still decode wrong; this is the
                    // only check that would have caught that.
                    if let Some(want) = t.get("src_xxh3").and_then(|v| v.as_u64()) {
                        let got = xxh3(&logical);
                        if got != want {
                            return Err(format!(
                                "restore: {:?} decoded to xxh3 {got:016x}, expected {want:016x}",
                                t.get("name").and_then(|v| v.as_str()).unwrap_or("?")
                            )
                            .into());
                        }
                    }
                    o.write_all(&logical)?;
                }
            }
            other => return Err(format!("unknown archive entry kind {other:?}").into()),
        }
        o.flush()?;
        if let Sink::Compare { path, ok, f, read } = &o {
            // Also catches a source that is longer than the reconstruction:
            // every byte must be consumed, not merely match while it lasted.
            let src_len = f.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
            if !*ok {
                return Err(format!("verify: {path} content differs").into());
            }
            if src_len != *read {
                return Err(format!(
                    "verify: {path} is {src_len} bytes, archive reconstructs {read}"
                )
                .into());
            }
        }
        n += 1;
    }
    match (out_dir, check) {
        (Some(d), _) => eprintln!("repack: restored {n} files to {}", d.display()),
        (_, Some(src)) => {
            // Checking only that every archived file matches the source proves
            // nothing about files the archive does not contain. Ornith-1.0-35B
            // reported "62 files byte-identical" while the source held 105 --
            // 45.8 GB of `.git/lfs` objects that `collect` skips. Deleting on
            // that verdict would have destroyed them, so enumerate the source
            // too and fail on anything the archive is missing.
            let archived: std::collections::HashSet<String> = files
                .iter()
                .filter_map(|f| f.get("path").and_then(|v| v.as_str()).map(String::from))
                .collect();
            let mut extra: Vec<(u64, String)> = Vec::new();
            let mut walk = vec![src.to_path_buf()];
            while let Some(d) = walk.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else {
                    continue;
                };
                for e in rd.flatten() {
                    let path = e.path();
                    if path.is_dir() {
                        walk.push(path);
                    } else if path.is_file() {
                        let rel = path
                            .strip_prefix(src)
                            .map(|r| r.to_string_lossy().replace('\\', "/"))
                            .unwrap_or_default();
                        if !archived.contains(&rel) {
                            extra.push((path.metadata().map(|m| m.len()).unwrap_or(0), rel));
                        }
                    }
                }
            }
            if !extra.is_empty() {
                extra.sort_by(|a, b| b.0.cmp(&a.0));
                let bytes: u64 = extra.iter().map(|e| e.0).sum();
                let sample: Vec<&str> = extra.iter().take(3).map(|e| e.1.as_str()).collect();
                return Err(format!(
                    "verify: {} source file(s) are NOT in the archive ({:.2} GB). Deleting the source would lose them. e.g. {:?}",
                    extra.len(),
                    bytes as f64 / 1e9,
                    sample
                )
                .into());
            }
            eprintln!("repack: verified {n} files byte-identical, source fully covered");
        }
        _ => {}
    }
    Ok(())
}

/// Either writes the reconstruction or checks it against the original.
enum Sink {
    Write(std::io::BufWriter<File>),
    Compare {
        f: File,
        path: String,
        ok: bool,
        read: u64,
    },
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Sink::Write(w) => w.write(buf),
            Sink::Compare { f, ok, read, .. } => {
                let mut cmp = vec![0u8; buf.len()];
                match f.read_exact(&mut cmp) {
                    Ok(()) if cmp == buf => *read += buf.len() as u64,
                    _ => *ok = false,
                }
                Ok(buf.len())
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Sink::Write(w) => w.flush(),
            Sink::Compare { .. } => Ok(()),
        }
    }
}

/// XXH3-64 over a byte range of the archive, streamed.
fn hash_range(f: &mut File, off: u64, len: u64) -> std::io::Result<u64> {
    f.seek(SeekFrom::Start(off))?;
    let mut h = twox_hash::xxhash3_64::Hasher::new();
    let mut left = len;
    let mut buf = vec![0u8; CH.min(len.max(1) as usize)];
    while left > 0 {
        let n = (buf.len() as u64).min(left) as usize;
        f.read_exact(&mut buf[..n])?;
        std::hash::Hasher::write(&mut h, &buf[..n]);
        left -= n as u64;
    }
    Ok(std::hash::Hasher::finish(&h))
}

/// Validate every stored payload against its recorded XXH3-64.
///
/// A read sweep proves the bytes are *readable*; this proves they are the same
/// bytes. On an array with no redundancy those are different questions, and
/// only the second one catches silent corruption.
fn check(archive: &Path) -> Result<(), Box<dyn Error>> {
    let mut f = File::open(archive)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC && &magic != MAGIC_V1 {
        return Err("not a hipfire repack archive".into());
    }
    let v1 = &magic == MAGIC_V1;
    let total = f.metadata()?.len();
    f.seek(SeekFrom::Start(total - 16))?;
    let mut b = [0u8; 16];
    f.read_exact(&mut b)?;
    let io = u64::from_le_bytes(b[..8].try_into()?);
    let il = u64::from_le_bytes(b[8..].try_into()?);
    f.seek(SeekFrom::Start(io))?;
    let mut ib = vec![0u8; il as usize];
    f.read_exact(&mut ib)?;
    let index: serde_json::Value = serde_json::from_slice(&ib)?;
    let files = index
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or("no files")?;

    let (mut checked, mut unchecked, mut bad, mut bytes) = (0usize, 0usize, 0usize, 0u64);
    let mut decoded = 0usize;

    for fe in files {
        let path = fe.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let mut tally = |res: std::io::Result<u64>,
                         want: Option<u64>,
                         what: &str,
                         checked: &mut usize,
                         unchecked: &mut usize,
                         bad: &mut usize,
                         bytes: &mut u64,
                         len: u64| match res {
            Err(e) => {
                println!("  READ-ERROR {path} {what}: {e}");
                *bad += 1;
            }
            Ok(got) => match want {
                None => *unchecked += 1,
                Some(w) if w == got => {
                    *checked += 1;
                    *bytes += len;
                }
                Some(w) => {
                    println!("  CORRUPT    {path} {what}: xxh3 {got:016x} != {w:016x}");
                    *bad += 1;
                }
            },
        };
        match fe.get("kind").and_then(|v| v.as_str()) {
            Some("raw") => {
                let off = fe.get("off").and_then(|v| v.as_u64()).unwrap_or(0);
                let len = fe.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
                let r = hash_range(&mut f, off, len);
                tally(
                    r,
                    fe.get("xxh3").and_then(|v| v.as_u64()),
                    "",
                    &mut checked,
                    &mut unchecked,
                    &mut bad,
                    &mut bytes,
                    len,
                );
            }
            Some("safetensors") => {
                if let Some(ts) = fe.get("tensors").and_then(|v| v.as_array()) {
                    for t in ts {
                        let off = t.get("stored_off").and_then(|v| v.as_u64()).unwrap_or(0);
                        let len = t.get("stored_len").and_then(|v| v.as_u64()).unwrap_or(0);
                        let nm = t.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                        let r = hash_range(&mut f, off, len);
                        tally(
                            r,
                            t.get("xxh3").and_then(|v| v.as_u64()),
                            nm,
                            &mut checked,
                            &mut unchecked,
                            &mut bad,
                            &mut bytes,
                            len,
                        );
                        // Decode and check the content hash: proves the payload
                        // still reproduces the original bytes, not merely that
                        // it is unchanged since it was written.
                        if t.get("codec").and_then(|v| v.as_str()) == Some("bf16h") {
                            if let Some(want) = t.get("src_xxh3").and_then(|v| v.as_u64()) {
                                let ln = t.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
                                if f.seek(SeekFrom::Start(off)).is_ok() {
                                    let mut sb = vec![0u8; len as usize];
                                    if f.read_exact(&mut sb).is_ok() {
                                        match hipfire_primitives::bf16_huff::decode(
                                            &sb,
                                            (ln / 2) as usize,
                                        ) {
                                            Some(d) if xxh3(&d) == want => decoded += 1,
                                            Some(_) => {
                                                println!("  DECODE-BAD {path} {nm}: content hash mismatch");
                                                bad += 1;
                                            }
                                            None => {
                                                println!("  DECODE-FAIL {path} {nm}");
                                                bad += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    println!(
        "repack: {checked} payload(s) match ({:.2} GB), {decoded} decode-verified, \
{bad} bad, {unchecked} unchecked{}",
        bytes as f64 / 1e9,
        if v1 {
            " — v1 archive, written before checksums existed"
        } else {
            ""
        }
    );
    if bad > 0 {
        return Err(format!("{bad} payload(s) failed integrity check").into());
    }
    Ok(())
}
