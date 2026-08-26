// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Reliable HuggingFace fetch, verify, and repair.
//!
//! The model store this feeds is RAID0 with no redundancy — deliberate, since
//! models are re-downloadable — which makes this the recovery mechanism rather
//! than a convenience. The bar is not "usually works": a download that silently
//! produces a wrong file is worse than one that fails loudly.
//!
//! So every byte is hashed on the way in and a file becomes visible at its
//! final path *only* after its digest matches what the hub declared. There is
//! no path through this module that writes an unverified file where a reader
//! would find it.
//!
//! Per AGENTS.md this is external-ecosystem interop: offline tooling, never
//! linked into the daemon, server, or runtime hot path.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

pub mod cache;
pub mod chunks;
pub mod run;

pub use chunks::{ChunkHasher, ChunkMiss, ChunkTable, CHUNK};

const HUB: &str = "https://huggingface.co";
const UA: &str = concat!("hipfire-hub/", env!("CARGO_PKG_VERSION"));
/// Leave this much free after a fetch. A previous bulk job filled the
/// filesystem to 100% and stalled the machine; refusing early is cheap.
pub const SPACE_MARGIN: u64 = 20 * 1024 * 1024 * 1024;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Retrying will not help: bad credentials, missing repo, gated model.
    Fatal(String),
    /// Worth another attempt: 5xx, timeout, connection reset.
    Retryable(String),
    Io(std::io::Error),
    /// The bytes arrived but are not what the hub said they would be.
    Digest {
        path: String,
        want: String,
        got: String,
    },
    /// Refused before writing anything.
    Space {
        need: u64,
        have: u64,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Fatal(m) => write!(f, "{m}"),
            Error::Retryable(m) => write!(f, "{m} (retryable)"),
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Digest { path, want, got } => {
                write!(f, "{path}: digest mismatch — expected {want}, got {got}")
            }
            Error::Space { need, have } => write!(
                f,
                "refusing to start: needs {:.1} GB plus margin, {:.1} GB free",
                *need as f64 / 1e9,
                *have as f64 / 1e9
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        // A timeout or a dropped connection is worth another attempt; a
        // decoding or builder fault is not.
        if e.is_timeout() || e.is_connect() || e.is_request() {
            Error::Retryable(e.to_string())
        } else {
            Error::Fatal(e.to_string())
        }
    }
}

/// One file in a repo revision, with whatever integrity anchor the hub offers.
#[derive(Debug, Clone)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// LFS oid — a SHA-256 of the content. `None` for a small non-LFS file.
    pub sha256: Option<String>,
    /// Git blob SHA-1, present for every file the tree API lists.
    ///
    /// For a non-LFS file this is the only content hash the hub offers, and it
    /// is a real one: `sha1("blob <len>\0" + content)`. Verified against all 13
    /// non-LFS files of amd/chatglm3-6b-onnx-ryzenai-npu, 13 matches, 0
    /// mismatches. Treating those files as merely length-checkable left
    /// integrity on the table.
    pub git_oid: Option<String>,
}

impl RepoFile {
    /// Whether the file can be proven correct at all.
    ///
    /// Callers must report *which* hash applied rather than implying a stronger
    /// guarantee than was made: SHA-256 over the content, or the weaker git
    /// blob SHA-1.
    pub fn is_content_addressed(&self) -> bool {
        self.sha256.is_some() || self.git_oid.is_some()
    }
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(UA)
        // Fail a stalled connection rather than hanging on it. Without this a
        // dead transfer occupies a slot indefinitely.
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| Error::Fatal(format!("http client: {e}")))
}

fn auth(req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match std::env::var("HF_TOKEN") {
        Ok(t) if !t.is_empty() => req.bearer_auth(t),
        _ => req,
    }
}

/// Pull the `rel="next"` URL out of a `Link` header, if any.
///
/// The hub paginates the tree API (RFC 8288 style): `<url>; rel="next"`.
/// Ignoring it silently truncates large repos, and a truncated listing means a
/// fetch that reports success with files missing — so every page is followed.
fn link_next(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let link = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    for part in link.split(',') {
        let (url, params) = part.split_once(';')?;
        if params.contains("rel=\"next\"") {
            return Some(
                url.trim()
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .to_string(),
            );
        }
    }
    None
}

/// List a revision's files via the tree API, following pagination.
///
/// `lfs.oid` is the SHA-256 this module verifies against; the top-level `oid`
/// is a git blob sha1 and is *not* a content hash of the file.
pub async fn list_files(repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
    let client = client()?;
    let mut url = format!("{HUB}/api/models/{repo}/tree/{revision}?recursive=1");
    let mut out = Vec::new();
    loop {
        let resp = auth(client.get(&url)).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(classify(status, &format!("listing {repo}@{revision}")));
        }
        let next = link_next(resp.headers());
        let body: serde_json::Value = resp.json().await?;
        let arr = body
            .as_array()
            .ok_or_else(|| Error::Fatal(format!("{repo}: tree API did not return a list")))?;
        push_tree_page(arr, &mut out);
        match next {
            Some(n) => url = n,
            None => break,
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn push_tree_page(arr: &[serde_json::Value], out: &mut Vec<RepoFile>) {
    for e in arr {
        if e.get("type").and_then(|v| v.as_str()) != Some("file") {
            continue;
        }
        let Some(path) = e.get("path").and_then(|v| v.as_str()) else {
            continue;
        };
        let size = e.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let sha256 = e
            .get("lfs")
            .and_then(|l| l.get("oid"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let git_oid = e
            .get("oid")
            .and_then(|v| v.as_str())
            .filter(|s| s.len() == 40)
            .map(|s| s.to_string());
        out.push(RepoFile {
            path: path.to_string(),
            size,
            sha256,
            git_oid,
        });
    }
}

fn classify(status: reqwest::StatusCode, what: &str) -> Error {
    if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        Error::Retryable(format!("{what}: {status}"))
    } else if status == reqwest::StatusCode::UNAUTHORIZED
        || status == reqwest::StatusCode::FORBIDDEN
    {
        Error::Fatal(format!(
            "{what}: {status} — set HF_TOKEN for a gated or private repo"
        ))
    } else {
        Error::Fatal(format!("{what}: {status}"))
    }
}

/// Free bytes on the filesystem holding `path`.
pub fn free_space(path: &Path) -> std::io::Result<u64> {
    let probe = if path.exists() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(Path::new("/")).to_path_buf()
    };
    let c = std::ffi::CString::new(probe.as_os_str().as_encoded_bytes())
        .map_err(|e| std::io::Error::new(ErrorKind::InvalidInput, e))?;
    // SAFETY: `c` is a valid NUL-terminated path; statvfs only writes `s`.
    unsafe {
        let mut s: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut s) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(s.f_bavail as u64 * s.f_frsize as u64)
    }
}

/// Refuse before writing anything if the fetch cannot fit.
pub fn preflight(files: &[RepoFile], dest: &Path, already_have: u64) -> Result<u64> {
    let total: u64 = files.iter().map(|f| f.size).sum();
    let need = total.saturating_sub(already_have);
    let have = free_space(dest)?;
    if have < need.saturating_add(SPACE_MARGIN) {
        return Err(Error::Space { need, have });
    }
    Ok(need)
}

/// Hash a file that is already on disk.
pub async fn sha256_file(path: &Path) -> Result<String> {
    let mut f = tokio::fs::File::open(path).await?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = tokio::io::AsyncReadExt::read(&mut f, &mut buf).await?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Git blob hash of a file already on disk: `sha1("blob <len>\0" + content)`.
pub async fn git_blob_sha1_file(path: &Path) -> Result<String> {
    use sha1::Digest as _;
    let len = tokio::fs::metadata(path).await?.len();
    let mut h = sha1::Sha1::new();
    h.update(format!("blob {len}\0").as_bytes());
    let mut f = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = tokio::io::AsyncReadExt::read(&mut f, &mut buf).await?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Where a streamed transfer's bytes go instead of to a blob file.
///
/// This is a trait rather than a closure because the consumer carries state
/// across chunks — the safetensors header it has not finished reading, the
/// tensor piece it is filling — and has to be told when to throw that state
/// away.
pub trait ByteSink {
    /// The next bytes, in order, from the start of the file.
    fn chunk(&mut self, bytes: &[u8]) -> std::io::Result<()>;
    /// Discard everything accepted for this file; the transfer restarts at 0.
    fn reset(&mut self) -> std::io::Result<()>;
    /// Report a transfer event — a resume, a retry — to whatever is rendering
    /// this fetch.
    ///
    /// The default writes to stderr, which is what this module used to do
    /// directly. A sink drawing a progress bar overrides it so the message can
    /// be printed *around* the bar: an unsynchronised write to stderr corrupts
    /// a live bar, and the hub has no way to suspend one it does not own.
    fn note(&self, msg: &str) {
        eprintln!("{msg}");
    }
}

/// Resume state for a streamed transfer, carried across retry attempts.
///
/// A store-backed fetch resumes from the `.part` on disk and re-hashes its
/// prefix. A streamed one has no `.part` — the bytes were consumed as they
/// passed — so the digest state has to live here instead, which means resume
/// works only within the process that started it.
///
/// That is the trade the single-copy archive buys. A fetch interrupted by a
/// dropped connection still resumes byte-for-byte, which is the case that
/// actually happens and the one a fixed attempt budget cannot survive; a fetch
/// interrupted by the *process* dying restarts that file rather than the repo.
pub struct StreamProgress {
    sha: Sha256,
    /// Retained only when the hub offers no SHA-256, so the weaker git blob
    /// SHA-1 can still be checked. Those files are the small non-LFS ones —
    /// configs and tokenizers — so holding them is bounded in practice.
    side: Option<Vec<u8>>,
    consumed: u64,
}

impl StreamProgress {
    pub fn new(file: &RepoFile) -> Self {
        StreamProgress {
            sha: Sha256::new(),
            side: file.sha256.is_none().then(Vec::new),
            consumed: 0,
        }
    }

    /// Source bytes accepted so far — the progress measure the retry loop
    /// judges attempts by.
    pub fn consumed(&self) -> u64 {
        self.consumed
    }

    fn accept(&mut self, chunk: &[u8]) {
        self.sha.update(chunk);
        if let Some(buf) = &mut self.side {
            buf.extend_from_slice(chunk);
        }
        self.consumed += chunk.len() as u64;
    }

    fn restart(&mut self) {
        self.sha = Sha256::new();
        if let Some(buf) = &mut self.side {
            buf.clear();
        }
        self.consumed = 0;
    }
}

/// Fetch one file straight into `sink`, verified, without ever staging it on
/// disk.
///
/// This is the same transfer as [`fetch_file_from`] with the blob file removed:
/// the digest is still computed over every byte on the wire and still checked
/// before the caller is told the file is good. What changes is *when* the caller
/// learns that. A blob is renamed into place only after it verifies, so nothing
/// downstream ever sees unverified bytes; a sink has already consumed them by
/// then, so the sink — not this function — owns undoing that. `Error::Digest`
/// is the signal to do so.
pub async fn fetch_file_streamed(
    base: &str,
    repo: &str,
    revision: &str,
    file: &RepoFile,
    st: &mut StreamProgress,
    sink: &mut dyn ByteSink,
) -> Result<()> {
    let url = format!("{base}/{repo}/resolve/{revision}/{}", file.path);

    let mut req = auth(client()?.get(&url));
    if st.consumed > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={}-", st.consumed));
    }
    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(classify(status, &file.path));
    }
    // A server that ignores Range answers 200 with the whole body. There is no
    // way to skip the prefix without re-deriving the sink's internal state, so
    // take the honest option and start the file over.
    let resuming = st.consumed > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if st.consumed > 0 && !resuming {
        st.restart();
        sink.reset()?;
    } else if resuming {
        sink.note(&format!(
            "hub: resuming {} at {:.2} GB",
            file.path,
            st.consumed as f64 / 1e9
        ));
    }

    let want = resp
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string())
        .filter(|s| s.len() == 64)
        .or_else(|| file.sha256.clone());

    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            // Keep the progress: the next attempt ranges from here rather than
            // starting over, which is what makes a multi-GB shard finish at all
            // on a link that drops.
            Err(e) => return Err(Error::Retryable(format!("{}: {e}", file.path))),
        };
        sink.chunk(&chunk)?;
        st.accept(&chunk);
    }

    let got = hex(&st.sha.clone().finalize());
    if let Some(want) = &want {
        if &got != want {
            return Err(Error::Digest {
                path: file.path.clone(),
                want: want.clone(),
                got,
            });
        }
    } else if let Some(want) = &file.git_oid {
        use sha1::Digest as _;
        let body = st.side.as_deref().unwrap_or(&[]);
        let mut g = sha1::Sha1::new();
        g.update(format!("blob {}\0", st.consumed).as_bytes());
        g.update(body);
        let got_git = hex(&g.finalize());
        if &got_git != want {
            return Err(Error::Digest {
                path: file.path.clone(),
                want: want.clone(),
                got: got_git,
            });
        }
    } else if file.size != 0 && st.consumed != file.size {
        return Err(Error::Fatal(format!(
            "{}: got {} bytes, expected {}",
            file.path, st.consumed, file.size
        )));
    }
    Ok(())
}

/// Fetch one byte range of a file.
///
/// The building block for chunk repair: a damaged window costs its own bytes
/// rather than the whole shard. A 4 GB shard with one bad 4 MiB window is a
/// 4 MiB transfer here against 4 GB for a refetch, which on this link is the
/// difference between seconds and half an hour.
///
/// Refuses a `200`. A server that ignores `Range` answers with the entire body,
/// and quietly accepting that would turn "repair one window" into "download the
/// file, twice" — the caller is expected to fall back to a whole-file fetch
/// rather than have that hidden from it.
pub async fn fetch_range(
    base: &str,
    repo: &str,
    revision: &str,
    path: &str,
    from: u64,
    len: u64,
) -> Result<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let url = format!("{base}/{repo}/resolve/{revision}/{path}");
    let last = from + len - 1;
    let resp = auth(client()?.get(&url))
        .header(reqwest::header::RANGE, format!("bytes={from}-{last}"))
        .send()
        .await
        .map_err(|e| Error::Retryable(format!("{path}: range request: {e}")))?;

    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(Error::Fatal(format!(
            "{path}: range {from}-{last} answered {}, not 206",
            resp.status()
        )));
    }
    let body = resp
        .bytes()
        .await
        .map_err(|e| Error::Retryable(format!("{path}: range body: {e}")))?;
    if body.len() as u64 != len {
        return Err(Error::Retryable(format!(
            "{path}: range {from}-{last} returned {} bytes, wanted {len}",
            body.len()
        )));
    }
    Ok(body.to_vec())
}

/// Download one file into the blob store, verified, and commit it atomically.
///
/// The digest is computed while streaming, so a wrong file is rejected without
/// a second pass over it. The bytes land in a PID-scoped `.part` — scoped so
/// concurrent runs in the same store cannot adopt each other's partial writes —
/// and are renamed into place only after the digest matches. A failure at any
/// point leaves nothing at the final path.
pub async fn fetch_file(
    repo: &str,
    revision: &str,
    file: &RepoFile,
    store: &cache::Store,
) -> Result<PathBuf> {
    fetch_file_from(HUB, repo, revision, file, store).await
}

/// [`fetch_file`] against an explicit origin, so the fault-injection suite can
/// serve truncated bodies, wrong digests and dropped connections from a local
/// socket. Production always passes [`HUB`].
pub async fn fetch_file_from(
    base: &str,
    repo: &str,
    revision: &str,
    file: &RepoFile,
    store: &cache::Store,
) -> Result<PathBuf> {
    // A content-addressed blob we already hold is already proven; there is
    // nothing to gain by fetching it again.
    if let Some(name) = file.sha256.as_deref().or(file.git_oid.as_deref()) {
        let blob = store.blob_path(name);
        if blob.exists() {
            // Backfill a table for a blob that predates them — anything fetched
            // by another client, or by this one before chunk tables existed.
            // Without this a store can never acquire tables at all: the blob is
            // already correct, so nothing ever re-downloads it, so `repair`
            // would be stuck on the whole-file path forever. One sequential
            // read, and only the first time.
            if store.read_chunks(name).is_none() {
                if let Ok(t) = crate::ChunkTable::of_file(&blob, crate::CHUNK) {
                    let _ = store.write_chunks(name, &t);
                }
            }
            return Ok(blob);
        }
    }

    let url = format!("{base}/{repo}/resolve/{revision}/{}", file.path);
    let part = store.part_path(&file.path);

    // Resume a partial transfer rather than restarting it. This is not an
    // optimisation: on a connection that drops mid-stream, a 4 GB single-shot
    // download never completes, however many times it is retried. Measured on
    // this host, a repair attempt reached 0.34 GB of 4.00 GB four times over.
    //
    // The existing prefix is re-hashed rather than trusting a persisted hash
    // state: it costs one sequential read and removes a whole class of
    // resume-corruption bug.
    let mut h = Sha256::new();
    // Built in the same passes as the digest, so recording a chunk table costs
    // no extra read. It follows `h` exactly -- same feeds, same reset -- because
    // both describe the same byte stream and disagreeing about the prefix is the
    // one way this could produce a table that looks valid and is not.
    let mut ch = crate::ChunkHasher::new();
    let mut written = 0u64;
    if let Ok(md) = tokio::fs::metadata(&part).await {
        let have = md.len();
        if have > 0 && (file.size == 0 || have < file.size) {
            let mut f = tokio::fs::File::open(&part).await?;
            let mut buf = vec![0u8; 8 << 20];
            loop {
                let n = tokio::io::AsyncReadExt::read(&mut f, &mut buf).await?;
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
                ch.update(&buf[..n]);
            }
            written = have;
        } else {
            // Complete or overlong: start clean rather than guess.
            let _ = tokio::fs::remove_file(&part).await;
        }
    }

    let mut req = auth(client()?.get(&url));
    if written > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={written}-"));
    }
    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(classify(status, &file.path));
    }
    // A server that ignores Range answers 200 with the whole body; honour that
    // by discarding the prefix instead of appending to it.
    let resuming = written > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if written > 0 && !resuming {
        h = Sha256::new();
        ch = crate::ChunkHasher::new();
        written = 0;
        let _ = tokio::fs::remove_file(&part).await;
    }
    if resuming {
        eprintln!(
            "hub: resuming {} at {:.2} GB",
            file.path,
            written as f64 / 1e9
        );
    }

    // Prefer the digest the hub states on the response; fall back to the one
    // from the tree listing.
    let want = resp
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string())
        .filter(|s| s.len() == 64)
        .or_else(|| file.sha256.clone());

    if let Some(p) = part.parent() {
        tokio::fs::create_dir_all(p).await?;
    }
    let mut out = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part)
        .await?;

    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                // Flush what arrived and keep the .part: the next attempt
                // resumes from here instead of starting over.
                let _ = out.flush().await;
                let _ = out.sync_all().await;
                return Err(Error::Retryable(format!("{}: {e}", file.path)));
            }
        };
        h.update(&chunk);
        ch.update(&chunk);
        out.write_all(&chunk).await?;
        written += chunk.len() as u64;
    }
    out.flush().await?;
    // Durable before we claim it is good.
    out.sync_all().await?;
    drop(out);

    let got = hex(&h.finalize());

    // Length is the only check available for non-LFS files; say so rather than
    // implying the stronger one.
    if let Some(want) = &want {
        if &got != want {
            let _ = tokio::fs::remove_file(&part).await;
            return Err(Error::Digest {
                path: file.path.clone(),
                want: want.clone(),
                got,
            });
        }
    } else if let Some(want) = &file.git_oid {
        // No LFS digest, but the tree API's `oid` is a git blob hash and is
        // verifiable. Length alone would accept a same-size wrong file.
        use sha1::Digest as _;
        let mut g = sha1::Sha1::new();
        g.update(format!("blob {written}\0").as_bytes());
        let mut rd = tokio::fs::File::open(&part).await?;
        let mut buf = vec![0u8; 8 << 20];
        loop {
            let n = tokio::io::AsyncReadExt::read(&mut rd, &mut buf).await?;
            if n == 0 {
                break;
            }
            g.update(&buf[..n]);
        }
        drop(rd);
        let got_git = hex(&g.finalize());
        if &got_git != want {
            let _ = tokio::fs::remove_file(&part).await;
            return Err(Error::Digest {
                path: file.path.clone(),
                want: want.clone(),
                got: got_git,
            });
        }
    } else if file.size != 0 && written != file.size {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(Error::Fatal(format!(
            "{}: got {written} bytes, expected {}",
            file.path, file.size
        )));
    }

    // Name the blob the way the HF cache does: the LFS sha256 when there is
    // one, otherwise the git blob sha1 that serves as the etag. Naming a
    // non-LFS blob by its content sha256 instead makes it invisible to every
    // lookup that starts from the tree API, which is how this was found.
    let blob_name = want
        .as_deref()
        .or(file.git_oid.as_deref())
        .unwrap_or(&got)
        .to_string();
    let final_path = store.blob_path(&blob_name);
    if let Some(p) = final_path.parent() {
        tokio::fs::create_dir_all(p).await?;
    }
    tokio::fs::rename(&part, &final_path).await?;
    // Only after the blob is committed, so a table never describes a transfer
    // that was rejected. Best-effort: a store that cannot hold the sidecar still
    // holds a correct blob, and verify falls back to the whole-file digest.
    let _ = store.write_chunks(&blob_name, &ch.finish());
    Ok(final_path)
}

#[cfg(test)]
mod link_tests {
    use super::link_next;
    use reqwest::header::{HeaderMap, HeaderValue, LINK};

    fn map(v: &str) -> HeaderMap {
        let mut m = HeaderMap::new();
        m.insert(LINK, HeaderValue::from_str(v).unwrap());
        m
    }

    #[test]
    fn follows_next_and_ignores_other_rels() {
        let m = map("<https://h.co/api/models/x/tree/main?cursor=abc>; rel=\"next\"");
        assert_eq!(
            link_next(&m).as_deref(),
            Some("https://h.co/api/models/x/tree/main?cursor=abc")
        );
        let m = map("<https://h.co/p1>; rel=\"prev\", <https://h.co/p3>; rel=\"next\"");
        assert_eq!(link_next(&m).as_deref(), Some("https://h.co/p3"));
        assert_eq!(link_next(&map("<https://h.co/p1>; rel=\"prev\"")), None);
        assert_eq!(link_next(&HeaderMap::new()), None);
    }
}
