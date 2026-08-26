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
//! Bytes inside a blob that no tensor claims — alignment padding, a gap between
//! entries — are stored verbatim in place, so a shard that does not tile exactly
//! still compresses the tensors it does describe. Only a genuine ambiguity makes
//! the whole file fall back to raw: overlapping tensors, or one running past the
//! end of the blob. See [`blob_plan`].
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
//!
//! # Chunk tables, and the move off XXH3
//!
//! Each file entry also carries an optional `src_len` and `chunks`: BLAKE3-256
//! over fixed [`CHUNK`] windows of the file's ORIGINAL bytes. Source-keyed, so
//! a mismatched window maps straight onto a `Range` request against the origin
//! — see [`ChunkHasher`] for why the window is keyed to the source rather than
//! to stored payloads, and why the hash is cryptographic where the per-payload
//! [`xxh3`] is not.
//!
//! The field is additive and optional, deliberately: no magic bump, so an
//! archive written today still restores under a binary that predates the table,
//! which matters when an archive is the only surviving copy of its model. The
//! per-payload `xxh3`/`src_xxh3` are still written and still checked.
//!
//! The intended end state, once every live archive has been re-packed with a
//! table, is to stop writing `xxh3`/`src_xxh3` and keep reading them so older
//! archives still verify — the same treatment [`MAGIC_V1`] already gets.

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
///
/// Note this rarely bites: measured across ZAYA1-8B's 2483 tensors, exactly one
/// (`model.embed_tokens.weight`, 1074 MB) exceeds it, and the second largest is
/// 16.8 MB. The effective repair unit is therefore the tensor -- 8.4 MB median
/// -- not this constant.
pub(crate) const UNIT: u64 = 1 << 30;

/// Source-keyed chunk tables live in `hipfire-hub`, which owns fetching and
/// verifying files and is the crate that acts on a mismatched window. The
/// archive is one producer of a table; `download` is the other, and both must
/// agree byte for byte, so there is exactly one implementation.
///
/// See [`hipfire_hub::chunks`] for why the window is keyed to the source file
/// and why the hash is cryptographic where [`xxh3`] above is not, and for the
/// plan to retire `xxh3`/`src_xxh3` once every live archive carries a table.
pub(crate) use hipfire_hub::{ChunkHasher, ChunkTable};

/// Encode one piece of a blob.
///
/// Every BF16 payload in the model goes through `bf16_huff` — 1.50x measured on
/// Qwen3-1.7B against an order-0 entropy floor of 1.52x, so there is very little
/// left on the table — and there is deliberately no per-tensor exception.
///
/// The LM head is the tensor that invites one, because `bf16_lut3`'s fixed-width
/// code can be decoded inside a GEMV and huffman's cannot. But that is a
/// *residency* decision, not a storage one: a loader that wants weights
/// compressed in VRAM can transcode `bf16h` to `bf16_lut3` on the way there.
/// Paying 1.38x on disk to pre-bake a choice the loader is free to make itself
/// is strictly worse.
///
/// `encode_if_smaller` declines when the packed form would not actually be
/// smaller, so a payload is never stored larger than raw — which is what makes
/// it safe to offer it every piece, gaps and non-BF16 dtypes included, rather
/// than deciding up front which are worth trying.
pub(crate) fn encode_piece(dtype: &str, buf: Vec<u8>) -> (&'static str, Vec<u8>) {
    if dtype != "BF16" || buf.len() % 2 != 0 {
        return ("raw", buf);
    }
    match hipfire_primitives::bf16_huff::encode_if_smaller(&buf) {
        Some(packed) => ("bf16h", packed),
        None => ("raw", buf),
    }
}

/// Expand a stored payload back to the exact source bytes.
///
/// `logical_len` is the original byte length, which the codec needs because it
/// stores no trailing element count a decoder could infer padding from.
pub(crate) fn decode_piece(codec: &str, stored: Vec<u8>, logical_len: u64) -> Option<Vec<u8>> {
    let out = match codec {
        "bf16h" => hipfire_primitives::bf16_huff::decode(&stored, (logical_len / 2) as usize)?,
        _ => stored,
    };
    (out.len() as u64 == logical_len).then_some(out)
}

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut verify: Option<PathBuf> = None;
    let mut check_only = false;
    let mut upgrade = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--verify" => verify = it.next().map(PathBuf::from),
            "--check" => check_only = true,
            "--upgrade" => upgrade = true,
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
    if upgrade {
        let output = output.ok_or("repack --upgrade requires --output")?;
        return upgrade_archive(&input, &output);
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

/// What a completed archive cost, for the one-line summary each producer prints.
#[derive(Default)]
pub(crate) struct Stats {
    pub n_bf16h: usize,
    pub n_verbatim_files: usize,
    /// Logical bytes fed to the codecs, and what they became.
    pub before: u64,
    pub after: u64,
}

/// Incremental writer for the archive layout.
///
/// Both producers go through this: [`pack`], which reads whole files off disk,
/// and the hub's streaming packer, which never holds a whole file. Sharing one
/// writer is what keeps the two byte-identical — a property
/// `stream_produces_the_same_archive_as_pack` asserts directly, and the only
/// practical defence against two emitters of one format drifting apart.
pub(crate) struct ArchiveWriter {
    w: std::io::BufWriter<File>,
    pos: u64,
    files: Vec<serde_json::Value>,
    /// Start offset and running hash of the payload currently being appended.
    cur: Option<(u64, twox_hash::xxhash3_64::Hasher)>,
    pub stats: Stats,
}

impl ArchiveWriter {
    pub fn create(out: &Path) -> std::io::Result<Self> {
        let mut w = std::io::BufWriter::new(File::create(out)?);
        w.write_all(MAGIC)?;
        Ok(Self {
            w,
            pos: MAGIC.len() as u64,
            files: Vec::new(),
            cur: None,
            stats: Stats::default(),
        })
    }

    /// Open a payload. Bytes appended until [`Self::end_payload`] form one blob.
    pub fn begin_payload(&mut self) {
        debug_assert!(self.cur.is_none(), "a payload is already open");
        self.cur = Some((self.pos, twox_hash::xxhash3_64::Hasher::new()));
    }

    pub fn append(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let (_, h) = self.cur.as_mut().expect("append outside a payload");
        std::hash::Hasher::write(h, bytes);
        self.w.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// Close the payload, yielding `(stored_off, stored_len, xxh3)`.
    pub fn end_payload(&mut self) -> (u64, u64, u64) {
        let (start, h) = self.cur.take().expect("end_payload outside a payload");
        (start, self.pos - start, std::hash::Hasher::finish(&h))
    }

    /// Append a payload that is already in memory.
    pub fn store(&mut self, bytes: &[u8]) -> std::io::Result<(u64, u64, u64)> {
        self.begin_payload();
        self.append(bytes)?;
        Ok(self.end_payload())
    }

    /// Append a byte range of a file, streamed so a multi-GB shard is never held
    /// whole.
    /// `chunks`, when given, is fed the same bytes in the same order. A verbatim
    /// payload is stored unchanged, so its source bytes and its stored bytes are
    /// the same stream and one pass serves both hashes.
    pub fn store_file_range(
        &mut self,
        src: &Path,
        from: u64,
        len: Option<u64>,
        mut chunks: Option<&mut ChunkHasher>,
    ) -> std::io::Result<(u64, u64, u64)> {
        let mut f = File::open(src)?;
        f.seek(SeekFrom::Start(from))?;
        let want = len.unwrap_or_else(|| f.metadata().map(|m| m.len() - from).unwrap_or(0));
        self.begin_payload();
        let mut left = want;
        let mut buf = vec![0u8; CH.min(want.max(1) as usize)];
        while left > 0 {
            let n = (buf.len() as u64).min(left) as usize;
            f.read_exact(&mut buf[..n])?;
            if let Some(c) = chunks.as_deref_mut() {
                c.update(&buf[..n]);
            }
            self.append(&buf[..n])?;
            left -= n as u64;
        }
        Ok(self.end_payload())
    }

    pub fn push_file(&mut self, entry: serde_json::Value) {
        self.files.push(entry);
    }

    pub fn n_files(&self) -> usize {
        self.files.len()
    }

    /// A position the archive can be returned to by [`Self::rollback`].
    pub fn mark(&self) -> (u64, usize) {
        (self.pos, self.files.len())
    }

    /// Discard everything written since `mark`.
    ///
    /// The streaming packer needs this because it commits a file's payloads to
    /// the archive *before* that file's SHA-256 can be checked — the digest is
    /// only known once the last byte has arrived. Rolling back leaves the
    /// archive exactly as it was, so a rejected transfer costs the bytes on the
    /// wire and nothing else.
    pub fn rollback(&mut self, mark: (u64, usize)) -> std::io::Result<()> {
        self.cur = None;
        self.w.flush()?;
        self.w.get_ref().set_len(mark.0)?;
        self.w.seek(SeekFrom::Start(mark.0))?;
        self.pos = mark.0;
        self.files.truncate(mark.1);
        Ok(())
    }

    /// Write the index and trailer. Returns the archive's final size.
    pub fn finish(mut self) -> std::io::Result<(u64, Stats)> {
        let index = serde_json::to_vec(&serde_json::json!({ "files": self.files }))
            .map_err(std::io::Error::other)?;
        let index_off = self.pos;
        self.w.write_all(&index)?;
        self.w.write_all(&index_off.to_le_bytes())?;
        self.w.write_all(&(index.len() as u64).to_le_bytes())?;
        self.pos += index.len() as u64 + 16;
        self.w.flush()?;
        // A rollback can have left the file physically longer than what is now
        // live. The trailer is read from `len - 16`, so a stale tail would be
        // read as the index offset and the archive would not open at all.
        self.w.get_ref().set_len(self.pos)?;
        self.w.get_ref().sync_all()?;
        Ok((self.pos, self.stats))
    }
}

/// One tensor inside a safetensors shard.
pub(crate) struct Entry {
    pub name: String,
    /// Offset within the shard's data blob.
    pub off: u64,
    pub len: u64,
    /// The codec the payload was stored under: `bf16h`, `bf16l3`, or `raw`.
    pub codec: &'static str,
    pub stored_off: u64,
    pub stored_len: u64,
    /// XXH3-64 of the payload as stored. Detects an archive that rotted at rest.
    pub xxh3: u64,
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

/// The name given to a run of blob bytes no tensor claims.
///
/// It carries the offset so a diagnostic can say *where* the unclaimed bytes
/// were. Nothing reads it back — [`restore`] walks pieces in index order and
/// dispatches on `codec`, never on `name` — so it cannot collide with a real
/// tensor in any way that matters.
fn gap_name(off: u64) -> String {
    format!("__unclaimed__@{off}")
}

/// One piece of a blob, already split to [`UNIT`].
pub(crate) struct Piece {
    pub name: String,
    pub part: usize,
    /// Offset within the shard's data blob.
    pub off: u64,
    pub len: u64,
    /// The safetensors dtype, or `""` for an unclaimed run — which is simply a
    /// dtype no codec claims, so gaps fall out as raw with no special case.
    pub dtype: String,
}

/// Segment a shard's data blob into everything it contains, in blob order.
///
/// The earlier version of this demanded that tensors *exactly* tile the blob and
/// gave up on the whole file otherwise, which meant one unmodelled padding byte
/// cost the compression on every BF16 tensor in the shard. Unclaimed runs are
/// not actually a hazard: they only have to be written back where they were, and
/// storing them verbatim in offset order does exactly that.
///
/// So this returns `None` only for the two cases that are genuinely ambiguous
/// rather than merely unmodelled:
///
/// - **overlap** — two tensors claiming the same byte have no defined write-back
///   order, and picking one would silently corrupt the other;
/// - **a tensor running past the blob**, which means the header does not
///   describe this file.
///
/// A well-formed shard produces no gap pieces at all, so archives of the models
/// that already packed cleanly are unchanged, byte for byte.
pub(crate) fn blob_plan(parsed: &serde_json::Value, blob_len: u64) -> Option<Vec<Piece>> {
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

    let mut segments: Vec<(String, u64, u64, String)> = Vec::new();
    let mut cursor = 0u64;
    for (name, s, l, dtype) in v {
        if s < cursor {
            return None; // overlap
        }
        if s > cursor {
            segments.push((gap_name(cursor), cursor, s - cursor, String::new()));
        }
        cursor = s.checked_add(l)?;
        if cursor > blob_len {
            return None; // runs past the end of the blob
        }
        // A zero-length tensor owns no bytes; it is fully described by the
        // header, which is stored verbatim, so it needs no piece.
        if l > 0 {
            segments.push((name, s, l, dtype));
        }
    }
    if cursor < blob_len {
        segments.push((gap_name(cursor), cursor, blob_len - cursor, String::new()));
    }

    // Split every segment to UNIT.
    //
    // Two problems fall out of treating a tensor as one unit. A 21.47 GB
    // stacked-expert tensor is a 21.47 GB anonymous allocation, which thrashes a
    // 31 GB host; and `encode_if_smaller` declines anything past u32::MAX
    // elements (~8.6 GB of BF16), so the largest tensors were stored raw.
    // Llama-4-Maverick is 100% BF16 and still came out at 1.0126x for exactly
    // that reason.
    //
    // Pieces are contiguous and recorded in offset order, and restore writes
    // them back in that order, so a split segment reassembles with no reader
    // change. Gaps are split too, which the streaming packer needs: it buffers a
    // piece at a time, so an unbounded gap would be an unbounded allocation.
    let mut out = Vec::new();
    for (name, off, len, dtype) in segments {
        let mut done = 0u64;
        let mut part = 0usize;
        while done < len {
            let take = UNIT.min(len - done);
            out.push(Piece {
                name: name.clone(),
                part,
                off: off + done,
                len: take,
                dtype: dtype.clone(),
            });
            done += take;
            part += 1;
        }
    }
    Some(out)
}

fn pack(dir: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    collect(dir, dir, 0, &mut paths);
    paths.sort();

    let mut aw = ArchiveWriter::create(out)?;

    for p in &paths {
        let rel = p.strip_prefix(dir)?.to_string_lossy().replace('\\', "/");
        let is_st = p.extension().map(|e| e == "safetensors").unwrap_or(false);
        let parsed = if is_st { read_st_header(p) } else { None };

        // Non-safetensors, or a shard whose blob cannot be segmented: verbatim.
        let Some((hlen, hdr, parsed, blob_len)) = parsed else {
            let mut ch = ChunkHasher::new();
            let (o, l, x) = aw.store_file_range(p, 0, None, Some(&mut ch))?;
            aw.push_file(raw_file_entry(&rel, o, l, x, Some(&ch.finish())));
            aw.stats.n_verbatim_files += 1;
            continue;
        };
        let Some(plan) = blob_plan(&parsed, blob_len) else {
            let mut ch = ChunkHasher::new();
            let (o, l, x) = aw.store_file_range(p, 0, None, Some(&mut ch))?;
            aw.push_file(raw_file_entry(&rel, o, l, x, Some(&ch.finish())));
            eprintln!("repack: {rel} tensors overlap or overrun its blob — stored verbatim");
            aw.stats.n_verbatim_files += 1;
            continue;
        };

        let base = 8 + hlen;
        let mut f = File::open(p)?;
        let mut entries: Vec<Entry> = Vec::new();
        // The table is keyed to the SOURCE file, so it is fed the file's bytes
        // in file order: the 8-byte header length, the header, then the blob.
        // `blob_plan` yields pieces in blob order and covers unclaimed runs too,
        // so replaying it reproduces the blob exactly.
        let mut ch = ChunkHasher::new();
        ch.update(&hlen.to_le_bytes());
        ch.update(&hdr);
        for piece in &plan {
            f.seek(SeekFrom::Start(base + piece.off))?;
            let mut buf = vec![0u8; piece.len as usize];
            f.read_exact(&mut buf)?;
            ch.update(&buf);
            entries.push(encode_and_store(&mut aw, piece, buf)?);
        }
        let table = ch.finish();
        // A plan that failed to tile its blob would yield a table describing
        // fewer bytes than the file holds, and every chunk after the gap would
        // be hashed at the wrong offset. Cheap to check, silent corruption if
        // not: the table would look valid and localize damage to the wrong
        // ranges forever after.
        let src_len = base + blob_len;
        if table.src_len() != src_len {
            return Err(format!(
                "repack: {rel} chunk table covers {} bytes, file is {src_len}",
                table.src_len()
            )
            .into());
        }
        // Header bytes verbatim — the only way to guarantee a byte-identical file.
        let (hdr_off, _, _) = aw.store(&hdr)?;
        aw.push_file(safetensors_file_entry(
            &rel,
            hdr_off,
            hlen,
            blob_len,
            &entries,
            Some(&table),
        ));
    }

    let n_files = aw.n_files();
    let (total, stats) = aw.finish()?;
    let src: u64 = paths
        .iter()
        .filter_map(|p| p.metadata().ok())
        .map(|m| m.len())
        .sum();
    eprintln!("{}", summary(n_files, total, src, &stats));
    Ok(())
}

/// Encode one piece, append it, and describe it.
///
/// Shared by [`pack`] and the hub's streaming packer so both produce the same
/// bytes and the same index entry for the same input. The two differ only in
/// where `buf` came from — a seek on disk, or the socket.
pub(crate) fn encode_and_store(
    aw: &mut ArchiveWriter,
    piece: &Piece,
    buf: Vec<u8>,
) -> std::io::Result<Entry> {
    debug_assert_eq!(
        buf.len() as u64,
        piece.len,
        "piece filled to the wrong size"
    );
    aw.stats.before += piece.len;
    let src_hash = xxh3(&buf);
    let (codec, bytes) = encode_piece(&piece.dtype, buf);
    if codec == "bf16h" {
        aw.stats.n_bf16h += 1;
    }
    let (stored_off, stored_len, x) = aw.store(&bytes)?;
    aw.stats.after += stored_len;
    Ok(Entry {
        name: if piece.part == 0 {
            piece.name.clone()
        } else {
            format!("{}#{}", piece.name, piece.part)
        },
        off: piece.off,
        len: piece.len,
        codec,
        stored_off,
        stored_len,
        xxh3: x,
        src_xxh3: src_hash,
    })
}

pub(crate) fn raw_file_entry(
    rel: &str,
    off: u64,
    len: u64,
    xxh3: u64,
    chunks: Option<&ChunkTable>,
) -> serde_json::Value {
    let mut e =
        serde_json::json!({ "path": rel, "kind": "raw", "off": off, "len": len, "xxh3": xxh3 });
    if let (Some(t), Some(map)) = (chunks, e.as_object_mut()) {
        map.insert("src_len".into(), t.src_len().into());
        map.insert("chunks".into(), t.to_json());
    }
    e
}

pub(crate) fn safetensors_file_entry(
    rel: &str,
    header_off: u64,
    header_len: u64,
    blob_len: u64,
    entries: &[Entry],
    chunks: Option<&ChunkTable>,
) -> serde_json::Value {
    let mut e = serde_json::json!({
        "path": rel, "kind": "safetensors",
        "header_off": header_off, "header_len": header_len, "blob_len": blob_len,
        "tensors": entries.iter().map(|e| serde_json::json!({
            "name": e.name, "off": e.off, "len": e.len,
            "codec": e.codec, "stored_off": e.stored_off, "stored_len": e.stored_len,
            "xxh3": e.xxh3, "src_xxh3": e.src_xxh3
        })).collect::<Vec<_>>()
    });
    if let (Some(t), Some(map)) = (chunks, e.as_object_mut()) {
        map.insert("src_len".into(), t.src_len().into());
        map.insert("chunks".into(), t.to_json());
    }
    e
}

pub(crate) fn summary(n_files: usize, archive: u64, src: u64, s: &Stats) -> String {
    format!(
        "repack: {n_files} files ({} shards, {} verbatim), {} BF16 payloads recoded \
         {:.2} GB -> {:.2} GB; archive {:.2} GB vs source {:.2} GB ({:.4}x)",
        n_files - s.n_verbatim_files,
        s.n_verbatim_files,
        s.n_bf16h,
        s.before as f64 / 1e9,
        s.after as f64 / 1e9,
        archive as f64 / 1e9,
        src as f64 / 1e9,
        src as f64 / archive.max(1) as f64
    )
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
        // Parsed up front so the hasher is built only when there is a table to
        // check it against, and at the window that table was written with.
        let want_chunks = fe.get("chunks").and_then(ChunkTable::from_json);
        let sink: Sink = match (out_dir, check) {
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
        let mut o = Teed {
            sink,
            chunks: want_chunks
                .as_ref()
                .map(|t| ChunkHasher::with_size(t.size())),
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
                    let codec = t
                        .get("codec")
                        .and_then(|v| v.as_str())
                        .unwrap_or("raw")
                        .to_string();
                    f.seek(SeekFrom::Start(so))?;
                    let mut sb = vec![0u8; sl as usize];
                    f.read_exact(&mut sb)?;
                    let logical: Vec<u8> = decode_piece(&codec, sb, ln).ok_or_else(|| {
                        format!(
                            "corrupt {codec} payload for {:?} (expected {ln} bytes)",
                            t.get("name")
                        )
                    })?;
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
        // Chunk check before the whole-file compare: when both would fail, the
        // chunk indices say WHERE, which is the difference between "this file is
        // wrong" and a Range request that fixes it.
        if let (Some(want), Some(got)) = (&want_chunks, o.chunks.take()) {
            let got = got.finish();
            if want.hashes().len() != got.hashes().len() {
                return Err(format!(
                    "verify: {rel} reconstructs {} chunks, index records {}",
                    got.hashes().len(),
                    want.hashes().len()
                )
                .into());
            }
            let bad: Vec<usize> = want
                .hashes()
                .iter()
                .zip(got.hashes())
                .enumerate()
                .filter(|(_, (w, g))| w != g)
                .map(|(i, _)| i)
                .collect();
            if !bad.is_empty() {
                let ranges: Vec<String> = bad
                    .iter()
                    .take(8)
                    .map(|i| {
                        let (at, len) = want.range(*i);
                        format!("{at}-{}", at + len - 1)
                    })
                    .collect();
                return Err(format!(
                    "verify: {rel} differs in {} of {} chunks; refetch bytes {}{}",
                    bad.len(),
                    want.hashes().len(),
                    ranges.join(", "),
                    if bad.len() > 8 { ", ..." } else { "" }
                )
                .into());
            }
        }
        if let Sink::Compare { path, ok, f, read } = &o.sink {
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

/// A [`Sink`] that also feeds the reconstructed stream to a chunk hasher.
///
/// Every byte a restore emits already passes through one `write_all`, which is
/// the same property that lets `restore` serve write-out and compare from one
/// code path. Hashing here means the table is checked against exactly the bytes
/// a restore would produce, decode included -- not against the archive's stored
/// form, which is a different byte sequence entirely.
struct Teed {
    sink: Sink,
    chunks: Option<ChunkHasher>,
}

impl Teed {
    /// Inherent rather than via [`Write`]: `Write::write` may report a short
    /// write and `write_all` would then re-offer the tail, which would hash
    /// those bytes twice and corrupt every window after them.
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        if let Some(c) = &mut self.chunks {
            c.update(buf);
        }
        self.sink.write_all(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.sink.flush()
    }
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
pub fn check(archive: &Path) -> Result<(), Box<dyn Error>> {
    let mut f = File::open(archive)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC && &magic != MAGIC_V1 {
        return Err("not a hipfire repack archive".into());
    }
    let v1 = &magic == MAGIC_V1;
    let total = f.metadata()?.len();
    // The index footer is the last thing written, so a plausible one is what
    // separates "archive" from "interrupted write". Validate it before seeking
    // by it: garbage offsets otherwise surface as bare OS errors (EINVAL),
    // which is exactly what a user sent here to judge a suspect file gets.
    let truncated = || {
        format!(
            "{}: no valid index footer — the archive is truncated, most likely \
             an interrupted download or pack; re-fetch or re-pack it",
            archive.display()
        )
    };
    if total < 24 {
        return Err(truncated().into());
    }
    f.seek(SeekFrom::Start(total - 16))?;
    let mut b = [0u8; 16];
    f.read_exact(&mut b)?;
    let io = u64::from_le_bytes(b[..8].try_into()?);
    let il = u64::from_le_bytes(b[8..].try_into()?);
    if io < 8 || il == 0 || io.checked_add(il).map_or(true, |end| end + 16 != total) {
        return Err(truncated().into());
    }
    f.seek(SeekFrom::Start(io))?;
    let mut ib = vec![0u8; il as usize];
    f.read_exact(&mut ib)?;
    let index: serde_json::Value = serde_json::from_slice(&ib)
        .map_err(|e| format!("{}: index footer is unreadable: {e}", archive.display()))?;
    let files = index
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or("no files")?;

    let (mut checked, mut unchecked, mut bad, mut bytes) = (0usize, 0usize, 0usize, 0u64);
    let mut decoded = 0usize;

    for fe in files {
        let path = fe.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let tally = |res: std::io::Result<u64>,
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
                        let codec = t.get("codec").and_then(|v| v.as_str()).unwrap_or("raw");
                        if codec == "bf16h" || codec == "bf16l3" {
                            if let Some(want) = t.get("src_xxh3").and_then(|v| v.as_u64()) {
                                let ln = t.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
                                if f.seek(SeekFrom::Start(off)).is_ok() {
                                    let mut sb = vec![0u8; len as usize];
                                    if f.read_exact(&mut sb).is_ok() {
                                        match decode_piece(codec, sb, ln) {
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

/// Rewrite a v1 archive as v2, adding the checksums it was written without.
///
/// Payloads are copied verbatim rather than re-encoded: the bytes are already
/// correct, and re-encoding would burn CPU to produce the same thing while
/// widening the window in which something can go wrong. What is added is the
/// evidence — an XXH3 over each stored payload, and one over the content it
/// decodes to.
///
/// This writes a new file rather than rewriting the index in place. In-place
/// would be far cheaper, since only the tail changes, but it leaves a window
/// where the trailer is half-written and the archive reads as corrupt. These
/// archives are the only copy of their models on an array with no redundancy,
/// so the cheap path is the wrong one.
fn upgrade_archive(src: &Path, dst: &Path) -> Result<(), Box<dyn Error>> {
    let mut f = File::open(src)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic == MAGIC {
        return Err(format!("{} is already v2", src.display()).into());
    }
    if &magic != MAGIC_V1 {
        return Err("not a hipfire repack archive".into());
    }
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

    let mut w = std::io::BufWriter::new(File::create(dst)?);
    w.write_all(MAGIC)?;
    let mut pos = MAGIC.len() as u64;
    let mut out_files: Vec<serde_json::Value> = Vec::new();
    let (mut n_pay, mut n_dec) = (0usize, 0usize);

    // Copy one payload across, hashing the stored bytes on the way.
    let carry = |f: &mut File,
                 w: &mut dyn Write,
                 off: u64,
                 len: u64,
                 pos: &mut u64|
     -> std::io::Result<(u64, u64, Vec<u8>)> {
        f.seek(SeekFrom::Start(off))?;
        let start = *pos;
        let mut h = twox_hash::xxhash3_64::Hasher::new();
        let mut left = len;
        let mut buf = vec![0u8; CH.min(len.max(1) as usize)];
        // Only retained when the caller needs to decode it for the content hash.
        let mut keep = Vec::new();
        let small = len as usize <= (1 << 30);
        while left > 0 {
            let n = (buf.len() as u64).min(left) as usize;
            f.read_exact(&mut buf[..n])?;
            std::hash::Hasher::write(&mut h, &buf[..n]);
            if small {
                keep.extend_from_slice(&buf[..n]);
            }
            w.write_all(&buf[..n])?;
            left -= n as u64;
            *pos += n as u64;
        }
        Ok((start, std::hash::Hasher::finish(&h), keep))
    };

    for fe in files {
        let mut e = fe.clone();
        match fe.get("kind").and_then(|v| v.as_str()) {
            Some("raw") => {
                let off = fe.get("off").and_then(|v| v.as_u64()).unwrap_or(0);
                let len = fe.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
                let (start, x, _) = carry(&mut f, &mut w, off, len, &mut pos)?;
                e["off"] = start.into();
                e["xxh3"] = x.into();
                n_pay += 1;
            }
            Some("safetensors") => {
                let hoff = fe.get("header_off").and_then(|v| v.as_u64()).unwrap_or(0);
                let hlen = fe.get("header_len").and_then(|v| v.as_u64()).unwrap_or(0);
                let mut tensors = fe
                    .get("tensors")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                for t in tensors.iter_mut() {
                    let off = t.get("stored_off").and_then(|v| v.as_u64()).unwrap_or(0);
                    let len = t.get("stored_len").and_then(|v| v.as_u64()).unwrap_or(0);
                    let ln = t.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
                    // Owned: the JSON value is mutated below, so no borrow of it
                    // may still be live.
                    let codec = t
                        .get("codec")
                        .and_then(|v| v.as_str())
                        .unwrap_or("raw")
                        .to_string();
                    let (start, x, bytes) = carry(&mut f, &mut w, off, len, &mut pos)?;
                    t["stored_off"] = start.into();
                    t["xxh3"] = x.into();
                    // The content hash needs the decoded bytes. For a raw
                    // payload the stored bytes are the content.
                    if !bytes.is_empty() {
                        if let Some(d) = decode_piece(&codec, bytes, ln) {
                            t["src_xxh3"] = xxh3(&d).into();
                            n_dec += 1;
                        }
                    }
                    n_pay += 1;
                }
                // Header bytes are part of the payload region; carry them too.
                let (hstart, _, _) = carry(&mut f, &mut w, hoff, hlen, &mut pos)?;
                e["header_off"] = hstart.into();
                e["tensors"] = serde_json::Value::Array(tensors);
            }
            _ => {}
        }
        out_files.push(e);
    }

    let new_index = serde_json::to_vec(&serde_json::json!({ "files": out_files }))?;
    let index_off = pos;
    w.write_all(&new_index)?;
    w.write_all(&index_off.to_le_bytes())?;
    w.write_all(&(new_index.len() as u64).to_le_bytes())?;
    w.flush()?;

    eprintln!(
        "repack: upgraded {} -> v2, {n_pay} payload(s) checksummed, {n_dec} content-hashed",
        src.file_name().unwrap_or_default().to_string_lossy()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub_archive::StreamPacker;
    use hipfire_hub::{ByteSink, RepoFile};

    /// A scratch directory that cleans up after itself.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(tag: &str) -> Tmp {
            // Counter as well as pid: two tests in one process would otherwise
            // collide, and they run concurrently by default.
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let p = std::env::temp_dir().join(format!(
                "hipfire-repack-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create scratch dir");
            Tmp(p)
        }
        fn join(&self, s: &str) -> PathBuf {
            self.0.join(s)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Deterministic BF16 bytes with the narrow exponent spread a trained weight
    /// tensor actually has.
    ///
    /// Uniform random bytes would be the wrong fixture: `encode_if_smaller`
    /// would correctly decline them, and every assertion about compression would
    /// pass vacuously against a `raw` payload.
    fn bf16_weights(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(n * 2);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
            let v = (((s >> 16) & 0xff_ffff) as f64 / (1u64 << 24) as f64) - 0.5;
            let x = ((u + v) * 0.05) as f32;
            // BF16 is the top 16 bits of the f32, which is why the truncation is
            // exact rather than a rounding choice.
            out.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
        }
        out
    }

    /// `[u64 header_len][header JSON][blob]`. `trailing` appends bytes past the
    /// last tensor that no entry claims.
    fn synth_safetensors(tensors: &[(&str, &str, Vec<u8>)], trailing: usize) -> Vec<u8> {
        let mut hdr = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (name, dtype, data) in tensors {
            let start = blob.len();
            blob.extend_from_slice(data);
            hdr.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": [data.len() / 2],
                    "data_offsets": [start, blob.len()],
                }),
            );
        }
        blob.extend(std::iter::repeat(0xA5u8).take(trailing));
        let hb = serde_json::to_vec(&serde_json::Value::Object(hdr)).expect("header json");
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hb);
        out.extend_from_slice(&blob);
        out
    }

    /// A small model directory: a config, a tokenizer, and two BF16 shards.
    fn synth_model(trailing: usize) -> Vec<(String, Vec<u8>)> {
        vec![
            ("config.json".into(), br#"{"model_type":"test"}"#.to_vec()),
            (
                "model-00001.safetensors".into(),
                synth_safetensors(
                    &[
                        ("model.embed_tokens.weight", "BF16", bf16_weights(4096, 1)),
                        (
                            "model.layers.0.mlp.up.weight",
                            "BF16",
                            bf16_weights(8192, 2),
                        ),
                    ],
                    trailing,
                ),
            ),
            (
                "model-00002.safetensors".into(),
                synth_safetensors(
                    &[
                        ("lm_head.weight", "BF16", bf16_weights(4096, 3)),
                        // F32 has no BF16 codec and must survive as raw.
                        ("model.norm.weight", "F32", vec![0x11; 512]),
                    ],
                    0,
                ),
            ),
            ("tokenizer.json".into(), vec![b'{'; 4096]),
        ]
    }

    fn write_model(dir: &Path, files: &[(String, Vec<u8>)]) {
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes).expect("write fixture");
        }
    }

    fn read_index(archive: &Path) -> serde_json::Value {
        let mut f = File::open(archive).expect("open archive");
        let total = f.metadata().expect("archive metadata").len();
        f.seek(SeekFrom::Start(total - 16)).expect("seek trailer");
        let mut b = [0u8; 16];
        f.read_exact(&mut b).expect("read trailer");
        let off = u64::from_le_bytes(b[..8].try_into().unwrap());
        let len = u64::from_le_bytes(b[8..].try_into().unwrap());
        f.seek(SeekFrom::Start(off)).expect("seek index");
        let mut ib = vec![0u8; len as usize];
        f.read_exact(&mut ib).expect("read index");
        serde_json::from_slice(&ib).expect("parse index")
    }

    /// Every codec that appears on a tensor payload, across all shards.
    fn codecs(index: &serde_json::Value) -> Vec<String> {
        index["files"]
            .as_array()
            .expect("files array")
            .iter()
            .filter_map(|f| f.get("tensors").and_then(|t| t.as_array()))
            .flatten()
            .filter_map(|t| t.get("codec").and_then(|c| c.as_str()))
            .map(String::from)
            .collect()
    }

    /// Drive the streaming packer over a model, in chunks that land on no
    /// convenient boundary.
    fn stream_model(out: &Path, files: &[(String, Vec<u8>)], chunk: usize) {
        let total: u64 = files.iter().map(|(_, b)| b.len() as u64).sum();
        let mut packer = StreamPacker::create(out, total).expect("create stream archive");
        let mut src = 0u64;
        for (name, bytes) in files {
            let rf = RepoFile {
                path: name.clone(),
                size: bytes.len() as u64,
                sha256: None,
                git_oid: None,
            };
            src += bytes.len() as u64;
            packer.begin_file(&rf).expect("begin file");
            for c in bytes.chunks(chunk) {
                packer.chunk(c).expect("feed chunk");
            }
            packer.finish_file().expect("finish file");
        }
        packer.finish(src).expect("finish archive");
    }

    /// Files large enough to span several chunk windows.
    ///
    /// The ordinary fixture is a few KB, so every file would be a single chunk
    /// and every assertion about window boundaries would pass vacuously.
    fn synth_chunked_model() -> Vec<(String, Vec<u8>)> {
        vec![
            ("config.json".into(), br#"{"model_type":"test"}"#.to_vec()),
            (
                "model-00001.safetensors".into(),
                // 4 MiB per tensor: the shard spans two full windows plus a
                // short third once the header is counted.
                synth_safetensors(
                    &[
                        ("a.weight", "BF16", bf16_weights(1 << 21, 7)),
                        ("b.weight", "BF16", bf16_weights(1 << 21, 9)),
                    ],
                    0,
                ),
            ),
        ]
    }

    /// The chunk table must describe the SOURCE file, byte for byte.
    ///
    /// Recomputed here straight from the original on disk rather than from
    /// anything the packer emitted — a table derived from the packer's own view
    /// would agree with itself however wrong it was.
    #[test]
    fn chunk_table_describes_the_source_file() {
        let t = Tmp::new("chunktable");
        let src = t.join("model");
        std::fs::create_dir_all(&src).expect("model dir");
        let files = synth_chunked_model();
        write_model(&src, &files);
        let archive = t.join("a.hfa");
        pack(&src, &archive).expect("pack");

        let index = read_index(&archive);
        let (mut checked, mut multi) = (0usize, false);
        for fe in index["files"].as_array().expect("files") {
            let rel = fe["path"].as_str().expect("path");
            let table = fe
                .get("chunks")
                .and_then(ChunkTable::from_json)
                .unwrap_or_else(|| panic!("{rel} carries no chunk table"));
            assert_eq!(table.size(), hipfire_hub::CHUNK, "{rel}: unexpected window");
            let bytes = std::fs::read(src.join(rel)).expect("read source");
            let got: Vec<[u8; 32]> = bytes
                .chunks(table.size())
                .map(|c| *blake3::hash(c).as_bytes())
                .collect();
            assert_eq!(
                table.hashes(),
                got,
                "{rel}: table does not describe the source"
            );
            assert_eq!(table.src_len(), bytes.len() as u64, "{rel}: wrong length");
            multi |= got.len() > 1;
            checked += 1;
        }
        assert!(checked >= 2, "fixture covered too few files");
        assert!(
            multi,
            "no file spanned more than one window, so boundaries went untested"
        );
    }

    /// The property that keeps two hand-written emitters of one format honest.
    ///
    /// `pack` seeks a file on disk and the streaming packer fills from a socket,
    /// but they share `blob_plan`, `encode_and_store` and `ArchiveWriter`, so
    /// there is exactly one answer for what the archive should contain. If this
    /// ever fails, the two have diverged and one of them is writing a format the
    /// other cannot.
    #[test]
    fn stream_produces_the_same_archive_as_pack() {
        let t = Tmp::new("identity");
        let src = t.join("model");
        std::fs::create_dir_all(&src).expect("model dir");
        let files = synth_model(0);
        write_model(&src, &files);

        let packed = t.join("packed.hfa");
        pack(&src, &packed).expect("pack");

        // 7 bytes at a time splits the length prefix, the header, and every
        // tensor boundary — the state machine never gets an aligned chunk.
        let streamed = t.join("streamed.hfa");
        stream_model(&streamed, &files, 7);

        let a = std::fs::read(&packed).expect("read packed");
        let b = std::fs::read(&streamed).expect("read streamed");
        assert_eq!(
            a.len(),
            b.len(),
            "archives differ in size: packed {} vs streamed {}",
            a.len(),
            b.len()
        );
        assert!(a == b, "packed and streamed archives differ byte for byte");

        // Vacuous otherwise: two archives of entirely raw payloads would also
        // match.
        assert!(
            codecs(&read_index(&packed)).iter().any(|c| c == "bf16h"),
            "fixture compressed nothing, so the comparison proved nothing"
        );
    }

    /// The head is stored as huffman like any other weight — the lut3 decision
    /// belongs to the loader, which can transcode on the way to VRAM.
    #[test]
    fn lm_head_is_stored_as_huffman_like_every_other_weight() {
        let t = Tmp::new("head");
        let src = t.join("model");
        std::fs::create_dir_all(&src).expect("model dir");
        write_model(&src, &synth_model(0));
        let archive = t.join("a.hfa");
        pack(&src, &archive).expect("pack");

        let index = read_index(&archive);
        let head = index["files"]
            .as_array()
            .expect("files")
            .iter()
            .filter_map(|f| f.get("tensors").and_then(|t| t.as_array()))
            .flatten()
            .find(|t| t["name"] == "lm_head.weight")
            .expect("lm_head entry present");
        assert_eq!(head["codec"], "bf16h", "lm_head should be huffman-coded");
        assert!(
            !codecs(&index).iter().any(|c| c == "bf16l3"),
            "no payload should be stored as lut3"
        );
    }

    /// A shard with unclaimed trailing bytes used to fall back to whole-file
    /// verbatim, losing the compression on every tensor in it.
    #[test]
    fn unclaimed_bytes_no_longer_cost_the_whole_shard() {
        let t = Tmp::new("gap");
        let src = t.join("model");
        std::fs::create_dir_all(&src).expect("model dir");
        let files = synth_model(64); // 64 bytes past the last tensor
        write_model(&src, &files);

        let archive = t.join("a.hfa");
        pack(&src, &archive).expect("pack");
        let index = read_index(&archive);

        let padded = index["files"]
            .as_array()
            .expect("files")
            .iter()
            .find(|f| f["path"] == "model-00001.safetensors")
            .expect("padded shard present");
        assert_eq!(
            padded["kind"], "safetensors",
            "a shard with trailing padding must still be segmented, not stored raw"
        );
        let names: Vec<&str> = padded["tensors"]
            .as_array()
            .expect("tensors")
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(
            names.iter().any(|n| n.starts_with("__unclaimed__")),
            "the trailing bytes should appear as an unclaimed piece, got {names:?}"
        );
        assert!(
            codecs(&index).iter().any(|c| c == "bf16h"),
            "the padded shard's tensors should still be compressed"
        );

        // And it still reproduces the source exactly, padding included.
        restore(&archive, None, Some(&src)).expect("verify against source");

        // The streaming path has to agree on the segmentation too, or a fetch
        // and a repack of the same checkpoint would disagree.
        let streamed = t.join("streamed.hfa");
        stream_model(&streamed, &files, 13);
        assert!(
            std::fs::read(&archive).expect("packed") == std::fs::read(&streamed).expect("streamed"),
            "streamed archive differs once the blob has a gap"
        );
    }

    /// Round-trip: the archive must reproduce the directory byte for byte.
    #[test]
    fn restore_reproduces_the_source_exactly() {
        let t = Tmp::new("roundtrip");
        let src = t.join("model");
        std::fs::create_dir_all(&src).expect("model dir");
        let files = synth_model(0);
        write_model(&src, &files);

        let archive = t.join("a.hfa");
        pack(&src, &archive).expect("pack");
        check(&archive).expect("stored checksums verify");

        let out = t.join("restored");
        restore(&archive, Some(&out), None).expect("restore");
        for (name, want) in &files {
            let got = std::fs::read(out.join(name)).expect("restored file");
            assert!(&got == want, "{name} did not round-trip");
        }
    }

    /// Overlap is the one case that genuinely cannot be segmented: two tensors
    /// claiming a byte have no defined write-back order.
    #[test]
    fn overlapping_tensors_are_refused_and_stored_verbatim() {
        let hdr = serde_json::json!({
            "a": { "dtype": "BF16", "shape": [4], "data_offsets": [0, 8] },
            "b": { "dtype": "BF16", "shape": [4], "data_offsets": [4, 12] },
        });
        assert!(
            blob_plan(&hdr, 12).is_none(),
            "overlapping tensors must not produce a plan"
        );

        let past_end = serde_json::json!({
            "a": { "dtype": "BF16", "shape": [8], "data_offsets": [0, 16] },
        });
        assert!(
            blob_plan(&past_end, 8).is_none(),
            "a tensor running past the blob must not produce a plan"
        );
    }

    /// Gaps anywhere — leading, interior, trailing — are covered, and the plan
    /// tiles the blob exactly so restore can concatenate it back.
    #[test]
    fn blob_plan_covers_every_byte_in_order() {
        let hdr = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "b": { "dtype": "BF16", "shape": [4], "data_offsets": [16, 24] },
            "a": { "dtype": "BF16", "shape": [4], "data_offsets": [4, 12] },
            "empty": { "dtype": "BF16", "shape": [0], "data_offsets": [12, 12] },
        });
        let plan = blob_plan(&hdr, 32).expect("plan");
        let mut cursor = 0u64;
        for p in &plan {
            assert_eq!(p.off, cursor, "plan is not contiguous at {}", p.off);
            cursor += p.len;
        }
        assert_eq!(cursor, 32, "plan does not cover the whole blob");

        let named: Vec<&str> = plan.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            named,
            vec![
                "__unclaimed__@0",
                "a",
                "__unclaimed__@12",
                "b",
                "__unclaimed__@24"
            ],
            "expected leading, interior and trailing gaps around the tensors"
        );
        // A zero-length tensor owns no bytes and needs no piece; the header it is
        // described by is stored verbatim.
        assert!(!named.contains(&"empty"));
    }
}

#[cfg(test)]
mod check_footer_tests {
    #[test]
    fn truncated_archive_reports_truncation_not_an_os_error() {
        let dir = std::env::temp_dir().join(format!("hfa-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("partial.hfa");
        // Valid magic followed by payload bytes cut off mid-write: no footer.
        std::fs::write(&p, [&super::MAGIC[..], &[0u8; 4096]].concat()).unwrap();
        let msg = super::check(&p).unwrap_err().to_string();
        assert!(msg.contains("truncated"), "got: {msg}");
        // Shorter than even a footer.
        std::fs::write(&p, &super::MAGIC[..]).unwrap();
        let msg = super::check(&p).unwrap_err().to_string();
        assert!(msg.contains("truncated"), "got: {msg}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
