// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Fetch a HuggingFace repo straight into an `.hfa` archive.
//!
//! `hub fetch` followed by `repack` produces the same archive, but it needs the
//! raw checkpoint on disk first — so converting a 400 GB model costs 400 GB of
//! scratch you may not have, and the machine most likely to want the compressed
//! form is the one least able to stage the uncompressed one. This removes that
//! peak: bytes are encoded as they come off the socket and the raw tree is never
//! written at all.
//!
//! # How a stream becomes an archive
//!
//! A safetensors file is `[u64 header_len][header JSON][data blob]`, which is
//! exactly the order it arrives in. So the header is readable before any tensor
//! byte shows up, and by the time the blob starts, the tiling is already known:
//! which tensor owns which byte range, and what dtype it is. Bytes then land in
//! the current tensor piece, and a piece is encoded and appended the moment it
//! fills. Nothing larger than one [`repack::UNIT`] is ever held.
//!
//! The header itself is written *after* the tensors, matching what
//! [`repack::pack`] does — it is small, so buffering it is free, and matching
//! keeps the two producers byte-identical.
//!
//! # What is given up
//!
//! Verification order. A blob fetch renames a file into place only once its
//! digest matches, so nothing downstream ever sees unverified bytes. Here the
//! bytes are already in the archive by the time the digest is known, so a
//! mismatch is undone rather than prevented — [`repack::ArchiveWriter::rollback`]
//! returns the archive to exactly where the file began. The end state is the
//! same; the window in between is the cost of not staging.
//!
//! Cross-process resume, for the same reason a `.part` no longer exists. A
//! dropped connection still resumes byte-for-byte within the run; a run that
//! dies restarts the file it was on.

use std::error::Error;
use std::io;
use std::path::Path;

use hipfire_hub::{ByteSink, RepoFile, StreamProgress};

use crate::repack::{
    self, blob_plan, encode_and_store, raw_file_entry, safetensors_file_entry, ArchiveWriter,
    Entry, Piece,
};

/// A safetensors header this refuses to treat as one. Mirrors the cap in
/// `repack::read_st_header` so the streamed and on-disk readers agree on what
/// counts as plausible.
const MAX_HEADER: u64 = 512 << 20;

/// What the packer is doing with the bytes of the file currently in flight.
enum Stage {
    /// Still reading `[u64 header_len][header JSON]`.
    Header { buf: Vec<u8> },
    /// Header parsed and the tiling known; filling `plan[idx]`.
    Tensors {
        hdr: Vec<u8>,
        blob_len: u64,
        plan: Vec<Piece>,
        idx: usize,
        buf: Vec<u8>,
        entries: Vec<Entry>,
    },
    /// Not a safetensors shard this can model — everything goes in verbatim.
    /// A payload is open for the duration.
    Verbatim,
}

struct CurFile {
    rel: String,
    /// Total size the hub declared, needed to derive `blob_len` from the header.
    size: u64,
    /// Archive position to return to if this file has to be undone.
    mark: (u64, usize),
    stage: Stage,
}

/// Packs a sequence of streamed files into one archive.
pub struct StreamPacker {
    aw: ArchiveWriter,
    cur: Option<CurFile>,
}

impl StreamPacker {
    pub fn create(out: &Path) -> io::Result<Self> {
        Ok(StreamPacker {
            aw: ArchiveWriter::create(out)?,
            cur: None,
        })
    }

    /// Begin a file. Every subsequent [`ByteSink::chunk`] belongs to it until
    /// [`Self::finish_file`] or [`Self::abort_file`].
    pub fn begin_file(&mut self, file: &RepoFile) -> io::Result<()> {
        let mark = self.aw.mark();
        let looks_st = file.path.ends_with(".safetensors");
        let stage = if looks_st && file.size > 8 {
            Stage::Header { buf: Vec::new() }
        } else {
            self.aw.begin_payload();
            Stage::Verbatim
        };
        self.cur = Some(CurFile {
            rel: file.path.clone(),
            size: file.size,
            mark,
            stage,
        });
        Ok(())
    }

    /// Commit the file: emit its index entry.
    pub fn finish_file(&mut self) -> Result<(), Box<dyn Error>> {
        let cur = self.cur.take().ok_or("finish_file outside a file")?;
        match cur.stage {
            Stage::Verbatim => {
                let (off, len, x) = self.aw.end_payload();
                self.aw.push_file(raw_file_entry(&cur.rel, off, len, x));
                self.aw.stats.n_verbatim_files += 1;
            }
            // A file that ended mid-header never reached its tensors. It cannot
            // be stored verbatim either, because its bytes were buffered rather
            // than written, so the only honest outcome is to refuse it.
            Stage::Header { buf } => {
                return Err(format!(
                    "{}: stream ended after {} bytes, inside the safetensors header",
                    cur.rel,
                    buf.len()
                )
                .into());
            }
            Stage::Tensors {
                hdr,
                blob_len,
                plan,
                idx,
                buf,
                entries,
            } => {
                if idx < plan.len() || !buf.is_empty() {
                    return Err(format!(
                        "{}: stream ended with {} of {} tensor pieces filled",
                        cur.rel,
                        idx,
                        plan.len()
                    )
                    .into());
                }
                // Tensors first, then the header — the order `pack` writes in.
                let hlen = hdr.len() as u64;
                let (hdr_off, _, _) = self.aw.store(&hdr)?;
                self.aw.push_file(safetensors_file_entry(
                    &cur.rel, hdr_off, hlen, blob_len, &entries,
                ));
            }
        }
        Ok(())
    }

    /// Undo everything the in-flight file wrote.
    pub fn abort_file(&mut self) -> io::Result<()> {
        if let Some(cur) = self.cur.take() {
            self.aw.rollback(cur.mark)?;
        }
        Ok(())
    }

    /// Write the index and trailer. `src_bytes` is the summed size of the
    /// source files, for the ratio line.
    pub fn finish(self, src_bytes: u64) -> io::Result<u64> {
        let n_files = self.aw.n_files();
        let (total, stats) = self.aw.finish()?;
        eprintln!("{}", repack::summary(n_files, total, src_bytes, &stats));
        Ok(total)
    }

    /// Drive the state machine over one arriving chunk.
    fn feed(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        let Self { aw, cur } = self;
        let cur = cur
            .as_mut()
            .ok_or_else(|| io::Error::other("chunk arrived outside a file"))?;

        while !bytes.is_empty() {
            match &mut cur.stage {
                Stage::Verbatim => {
                    aw.append(bytes)?;
                    return Ok(());
                }
                Stage::Header { buf } => {
                    // Take enough to learn the header length, then the header.
                    let want = if buf.len() < 8 {
                        8
                    } else {
                        let hlen =
                            u64::from_le_bytes(buf[..8].try_into().expect("8 bytes checked above"));
                        // Implausible, or a header that cannot fit in the file:
                        // not a safetensors shard this can model. Zero is in
                        // that set and has to be — a zero-length header can
                        // never parse, and treating it as "keep reading" would
                        // spin, since there is nothing left to read.
                        if hlen == 0 || hlen > MAX_HEADER || 8 + hlen > cur.size {
                            let prefix = std::mem::take(buf);
                            aw.begin_payload();
                            aw.append(&prefix)?;
                            cur.stage = Stage::Verbatim;
                            continue;
                        }
                        (8 + hlen) as usize
                    };
                    let take = want.saturating_sub(buf.len()).min(bytes.len());
                    buf.extend_from_slice(&bytes[..take]);
                    bytes = &bytes[take..];
                    if buf.len() < want {
                        return Ok(()); // need more before deciding
                    }
                    if buf.len() == 8 {
                        continue; // learned hlen; loop to read the header
                    }

                    let hdr = buf[8..].to_vec();
                    let blob_len = cur.size - buf.len() as u64;
                    let plan = serde_json::from_slice::<serde_json::Value>(&hdr)
                        .ok()
                        .and_then(|p| blob_plan(&p, blob_len));
                    match plan {
                        Some(plan) => {
                            cur.stage = Stage::Tensors {
                                hdr,
                                blob_len,
                                plan,
                                idx: 0,
                                buf: Vec::new(),
                                entries: Vec::new(),
                            };
                        }
                        None => {
                            eprintln!(
                                "hub: {} tensors overlap or overrun its blob — stored verbatim",
                                cur.rel
                            );
                            let prefix = std::mem::take(buf);
                            aw.begin_payload();
                            aw.append(&prefix)?;
                            cur.stage = Stage::Verbatim;
                        }
                    }
                }
                Stage::Tensors {
                    plan,
                    idx,
                    buf,
                    entries,
                    ..
                } => {
                    let Some(piece) = plan.get(*idx) else {
                        // The tiling accounted for every blob byte, so anything
                        // past it means the file is not the size the hub said.
                        // The digest check would catch it too; failing here says
                        // why.
                        return Err(io::Error::other(format!(
                            "{}: {} bytes past the end of its tensor plan",
                            cur.rel,
                            bytes.len()
                        )));
                    };
                    let need = piece.len as usize - buf.len();
                    let take = need.min(bytes.len());
                    buf.extend_from_slice(&bytes[..take]);
                    bytes = &bytes[take..];
                    if buf.len() as u64 == piece.len {
                        let full = std::mem::take(buf);
                        entries.push(encode_and_store(aw, piece, full)?);
                        *idx += 1;
                    }
                }
            }
        }
        Ok(())
    }
}

impl ByteSink for StreamPacker {
    fn chunk(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.feed(bytes)
    }

    /// Restart the in-flight file from byte zero.
    fn reset(&mut self) -> io::Result<()> {
        let Some(cur) = self.cur.as_ref() else {
            return Ok(());
        };
        let (rel, size, mark) = (cur.rel.clone(), cur.size, cur.mark);
        self.aw.rollback(mark)?;
        let looks_st = rel.ends_with(".safetensors");
        let stage = if looks_st && size > 8 {
            Stage::Header { buf: Vec::new() }
        } else {
            self.aw.begin_payload();
            Stage::Verbatim
        };
        self.cur = Some(CurFile {
            rel,
            size,
            mark,
            stage,
        });
        Ok(())
    }
}

/// Fetch `repo@revision` directly into `out`, one file at a time.
///
/// Files are taken in the listing's sorted order, which is the order
/// [`repack::pack`] walks a directory in, so the archive this produces is the
/// archive packing the same checkpoint would produce.
pub async fn fetch_to_archive(
    out: &Path,
    repo: &str,
    revision: &str,
    include: Option<&str>,
    files: Vec<RepoFile>,
) -> Result<u64, Box<dyn Error>> {
    let mut files = files;
    if let Some(pat) = include {
        files.retain(|f| glob_match(pat, &f.path));
    }
    if files.is_empty() {
        return Err(format!("{repo}: no files matched").into());
    }

    let src_bytes: u64 = files.iter().map(|f| f.size).sum();
    eprintln!(
        "hub: {} file(s), {:.2} GB to fetch → {}",
        files.len(),
        src_bytes as f64 / 1e9,
        out.display()
    );

    let mut packer = StreamPacker::create(out)?;
    for f in &files {
        let mut st = StreamProgress::new(f);
        packer.begin_file(f)?;
        match hipfire_hub::run::fetch_file_streamed_with_retry(
            repo, revision, f, &mut st, &mut packer,
        )
        .await
        {
            Ok(()) => packer.finish_file()?,
            Err(e) => {
                // The file's payloads are already in the archive; take them back
                // out so a failed run leaves no half-file behind.
                packer.abort_file()?;
                return Err(format!("{}: {e}", f.path).into());
            }
        }
    }
    Ok(packer.finish(src_bytes)?)
}

/// Minimal `*` glob, matching `hipfire_hub::run`'s so `--include` behaves the
/// same on both paths.
fn glob_match(pat: &str, s: &str) -> bool {
    let mut parts = pat.split('*');
    let Some(first) = parts.next() else {
        return true;
    };
    if !s.starts_with(first) {
        return false;
    }
    let mut idx = first.len();
    let mut last = "";
    for p in parts {
        last = p;
        if p.is_empty() {
            continue;
        }
        match s[idx..].find(p) {
            Some(i) => idx += i + p.len(),
            None => return false,
        }
    }
    pat.ends_with('*') || s.ends_with(last)
}
