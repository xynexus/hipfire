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
    ChunkHasher, Entry, Piece,
};

/// A safetensors header this refuses to treat as one. Mirrors the cap in
/// `repack::read_st_header` so the streamed and on-disk readers agree on what
/// counts as plausible.
const MAX_HEADER: u64 = 512 << 20;

/// How often to emit a progress *line*. Long enough not to clutter a log, short
/// enough that a stalled transfer becomes obvious quickly.
const LINE_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

/// How often to refresh the bar's caption. A bar is watched live, so a figure
/// five seconds stale reads as frozen; the bar's own position updates on every
/// chunk regardless, throttled by indicatif's draw target.
const BAR_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

/// How long without a single new byte before the transfer is called stalled.
///
/// This is the state the whole feature exists to surface: a real fetch sat at
/// ~90 MB for over three minutes mid-shard. Saying so outright beats leaving the
/// reader to notice that a number has not changed.
const STALL_AFTER: std::time::Duration = std::time::Duration::from_secs(10);

/// Below these, a rate estimate is dominated by connection setup rather than
/// throughput, and the ETA it implies is worse than none: the first sample of a
/// real fetch read `0.0 MB/s — eta 25h53m` about ten seconds before settling at
/// 1.3 MB/s and 3m. An alarming number that is simply wrong costs more trust
/// than a `--` does.
const ETA_MIN_BYTES: u64 = 4 << 20;
const ETA_MIN_ELAPSED: std::time::Duration = std::time::Duration::from_secs(10);

/// Progress for a transfer that can run for hours, in whichever form the
/// terminal can carry.
///
/// Without it the run is silent between the opening line and the summary. On a
/// 272 MB model over a slow link that was 11 minutes of nothing; on a
/// multi-hundred-GB one it is indistinguishable from a hang, which is exactly
/// how it was misread during bring-up.
///
/// # Why both a bar and lines
///
/// A bar is the better thing to watch, but indicatif hides itself when stderr
/// is not a terminal — and a multi-hour fetch is usually run detached, under
/// `nohup` or CI or ssh, which is precisely where the output matters most.
/// Rendering only a bar would hand those runs *nothing*, strictly worse than
/// the lines they have today. So the bar is the interactive form and the
/// periodic line is the recorded one, and both read the same counters.
///
/// # Why the rate is measured over the whole run
///
/// A link that drops and resumes then reports the throughput actually being
/// achieved, instead of a figure that swings between zero and line speed every
/// time a connection stalls. Stalling is reported explicitly instead, which is
/// the honest signal — and the reason there is deliberately no `steady_tick`
/// here: a spinner that keeps animating through a stall makes a dead transfer
/// look alive, defeating the point of showing progress at all.
struct Progress {
    /// Total source bytes across every file in the fetch.
    total: u64,
    /// Source bytes in files already committed.
    done: u64,
    /// Source bytes accepted for the file in flight.
    cur: u64,
    started: std::time::Instant,
    /// Last time a line was printed or the bar's caption refreshed.
    last: std::time::Instant,
    /// The high-water mark, and when it was last raised — the basis for calling
    /// a transfer stalled.
    watermark: u64,
    moved_at: std::time::Instant,
    /// `None` when stderr is not a terminal, which selects line mode.
    bar: Option<indicatif::ProgressBar>,
}

impl Progress {
    fn new(total: u64) -> Self {
        let now = std::time::Instant::now();
        // 4 Hz matches the draw rate hipfire-quantize settles on.
        let bar = indicatif::ProgressBar::with_draw_target(
            Some(total),
            indicatif::ProgressDrawTarget::stderr_with_hz(4),
        );
        let bar = if bar.is_hidden() {
            // Detached: leave a log behind rather than draw to nobody.
            None
        } else {
            bar.set_style(
                // `decimal_*`, not `{bytes}`: indicatif's default is binary, and
                // a bar reading `259.82 MiB` beside a log line reading `272 MB`
                // for the same fetch invites the reader to hunt for a
                // discrepancy that is not there. The line mode's units are
                // decimal, so these are too.
                indicatif::ProgressStyle::with_template(
                    "hub: [{bar:32}] {decimal_bytes}/{decimal_total_bytes} {msg}",
                )
                .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                .progress_chars("=> "),
            );
            Some(bar)
        };
        Progress {
            total,
            done: 0,
            cur: 0,
            started: now,
            last: now,
            watermark: 0,
            moved_at: now,
            bar,
        }
    }

    fn advance(&mut self, n: usize, label: &str) {
        self.cur += n as u64;
        let got = self.done + self.cur;
        if got > self.watermark {
            self.watermark = got;
            self.moved_at = std::time::Instant::now();
        }
        if let Some(b) = &self.bar {
            // Cheap and atomic, so the bar keeps moving between caption
            // refreshes; the draw target throttles actual redraws.
            b.set_position(got);
        }
        let every = if self.bar.is_some() {
            BAR_EVERY
        } else {
            LINE_EVERY
        };
        // The first report is due one interval in, so a fetch that finishes
        // quickly — every unit test, most small repos — emits nothing extra.
        if self.last.elapsed() < every {
            return;
        }
        self.last = std::time::Instant::now();
        let tail = self.tail(label, got);
        match &self.bar {
            Some(b) => b.set_message(tail),
            None => eprintln!(
                "hub: {} ({:.0}%) — {tail}",
                fmt_pair(got, self.total),
                if self.total > 0 {
                    got as f64 / self.total as f64 * 100.0
                } else {
                    0.0
                },
            ),
        }
    }

    /// `<file> — <rate> — eta <t>`, or a stall notice in place of the ETA.
    /// Shared by both modes so they can never disagree.
    fn tail(&self, label: &str, got: u64) -> String {
        let stalled = self.moved_at.elapsed();
        if stalled >= STALL_AFTER {
            return format!("{label} — stalled {}", fmt_dur(stalled.as_secs()));
        }
        let elapsed = self.started.elapsed();
        let rate = got as f64 / elapsed.as_secs_f64().max(1e-3);
        let eta =
            if got >= ETA_MIN_BYTES && elapsed >= ETA_MIN_ELAPSED && rate > 0.0 && self.total > got
            {
                fmt_dur(((self.total - got) as f64 / rate) as u64)
            } else {
                "--".to_string()
            };
        format!("{label} — {:.1} MB/s — eta {eta}", rate / 1e6)
    }

    /// Print around the bar rather than through it. An unsynchronised write to
    /// stderr would corrupt a live bar mid-redraw.
    fn note(&self, msg: &str) {
        match &self.bar {
            Some(b) => b.suspend(|| eprintln!("{msg}")),
            None => eprintln!("{msg}"),
        }
    }

    fn finish_file(&mut self) {
        self.done += self.cur;
        self.cur = 0;
    }

    /// The in-flight file restarted, so its bytes no longer count toward the
    /// total transferred. Without this an ignored `Range` would inflate the
    /// figure past 100%.
    ///
    /// The watermark is left alone deliberately: it tracks whether bytes are
    /// still arriving at all, and a restart is not a stall.
    fn restart_file(&mut self) {
        self.cur = 0;
        self.moved_at = std::time::Instant::now();
    }

    /// Clear the bar so the summary that follows lands on a clean line.
    fn finish(&self) {
        if let Some(b) = &self.bar {
            b.finish_and_clear();
        }
    }
}

/// Both figures share one unit, chosen from the total, so they stay directly
/// comparable. GB is too coarse below a gigabyte — a 272 MB fetch spends most of
/// its life reading `0.00/0.27 GB`, which shows no movement at all.
fn fmt_pair(got: u64, total: u64) -> String {
    if total >= 1_000_000_000 {
        format!("{:.2}/{:.2} GB", got as f64 / 1e9, total as f64 / 1e9)
    } else {
        format!("{:.0}/{:.0} MB", got as f64 / 1e6, total as f64 / 1e6)
    }
}

fn fmt_dur(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m{:02}s", s / 60, s % 60),
        s => format!("{}h{:02}m", s / 3600, (s % 3600) / 60),
    }
}

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
    /// Source-keyed chunk table, built from the bytes as they arrive. The wire
    /// delivers the file in order, so this is the same stream `pack` hashes off
    /// disk and the two produce identical tables for identical files.
    chunks: ChunkHasher,
}

/// Packs a sequence of streamed files into one archive.
pub struct StreamPacker {
    aw: ArchiveWriter,
    cur: Option<CurFile>,
    progress: Progress,
}

impl StreamPacker {
    /// `total_bytes` is the summed source size of everything to be fetched,
    /// used only to give progress a denominator.
    pub fn create(out: &Path, total_bytes: u64) -> io::Result<Self> {
        Ok(StreamPacker {
            aw: ArchiveWriter::create(out)?,
            cur: None,
            progress: Progress::new(total_bytes),
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
            chunks: ChunkHasher::new(),
        });
        Ok(())
    }

    /// Commit the file: emit its index entry.
    pub fn finish_file(&mut self) -> Result<(), Box<dyn Error>> {
        let cur = self.cur.take().ok_or("finish_file outside a file")?;
        self.progress.finish_file();
        let CurFile {
            rel, stage, chunks, ..
        } = cur;
        let table = chunks.finish();
        match stage {
            Stage::Verbatim => {
                let (off, len, x) = self.aw.end_payload();
                self.aw
                    .push_file(raw_file_entry(&rel, off, len, x, Some(&table)));
                self.aw.stats.n_verbatim_files += 1;
            }
            // A file that ended mid-header never reached its tensors. It cannot
            // be stored verbatim either, because its bytes were buffered rather
            // than written, so the only honest outcome is to refuse it.
            Stage::Header { buf } => {
                return Err(format!(
                    "{}: stream ended after {} bytes, inside the safetensors header",
                    rel,
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
                        rel,
                        idx,
                        plan.len()
                    )
                    .into());
                }
                // Tensors first, then the header — the order `pack` writes in.
                let hlen = hdr.len() as u64;
                let (hdr_off, _, _) = self.aw.store(&hdr)?;
                self.aw.push_file(safetensors_file_entry(
                    &rel,
                    hdr_off,
                    hlen,
                    blob_len,
                    &entries,
                    Some(&table),
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
        self.progress.finish();
        let n_files = self.aw.n_files();
        let (total, stats) = self.aw.finish()?;
        eprintln!("{}", repack::summary(n_files, total, src_bytes, &stats));
        Ok(total)
    }

    /// Drive the state machine over one arriving chunk.
    fn feed(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        let Self { aw, cur, .. } = self;
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
        // Hashed before `feed` consumes them: the table is keyed to the source
        // file, so it wants the bytes as they arrive on the wire, not whatever
        // the state machine buffers or reorders on the way into the archive.
        if let Some(cur) = self.cur.as_mut() {
            cur.chunks.update(bytes);
        }
        self.feed(bytes)?;
        // Disjoint field borrows: naming the file costs no allocation per chunk.
        let Self { cur, progress, .. } = self;
        let label = cur.as_ref().map(|c| c.rel.as_str()).unwrap_or("?");
        progress.advance(bytes.len(), label);
        Ok(())
    }

    fn note(&self, msg: &str) {
        self.progress.note(msg);
    }

    /// Restart the in-flight file from byte zero.
    fn reset(&mut self) -> io::Result<()> {
        self.progress.restart_file();
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
            // A restart replays the file from byte zero, so the partial table
            // has to go with it. Keeping it would hash the retried prefix twice
            // and shift every window after the resume point.
            chunks: ChunkHasher::new(),
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
    jobs: usize,
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

    let mut packer = StreamPacker::create(out, src_bytes)?;
    for f in &files {
        let mut st = StreamProgress::new(f);
        packer.begin_file(f)?;
        match hipfire_hub::run::fetch_file_streamed_with_retry(
            repo,
            revision,
            f,
            &mut st,
            &mut packer,
            jobs,
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
