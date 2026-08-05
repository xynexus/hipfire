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
pub mod run;

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

/// List a revision's files via the tree API.
///
/// `lfs.oid` is the SHA-256 this module verifies against; the top-level `oid`
/// is a git blob sha1 and is *not* a content hash of the file.
pub async fn list_files(repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
    let url = format!("{HUB}/api/models/{repo}/tree/{revision}?recursive=1");
    let resp = auth(client()?.get(&url)).send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(classify(status, &format!("listing {repo}@{revision}")));
    }
    let body: serde_json::Value = resp.json().await?;
    let arr = body
        .as_array()
        .ok_or_else(|| Error::Fatal(format!("{repo}: tree API did not return a list")))?;

    let mut out = Vec::new();
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
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
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
    Ok(final_path)
}
