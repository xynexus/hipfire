// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Fault injection for the fetch path.
//!
//! These assert failure, not success. Every integrity bug found while building
//! the archive tooling hid behind a clean-looking result — a `u32` overflow
//! corrupting tensors while the converter reported `Max quant error:
//! 0.00000000`, and a one-directional verify reporting "62 files
//! byte-identical" while 43 were missing from the archive. A suite that only
//! checks the happy path reproduces exactly that blind spot.
//!
//! The invariant under test is the one the whole design rests on: **a file that
//! was not proven correct never appears at its final path.**

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use hipfire_hub::{cache::Store, fetch_file_from, Error, RepoFile};

/// How the injected origin should misbehave.
#[derive(Clone, Copy)]
enum Fault {
    /// Serve the whole body correctly.
    None,
    /// Declare the full length, send less, then close.
    Truncate(usize),
    /// Serve a complete body that is not what the digest promises.
    WrongBytes,
    /// Drop the first attempt partway; serve the remainder on the retry, so
    /// resume is exercised rather than a fresh download.
    DropThenResume(usize),
    /// Answer 200 with the whole body even when a Range was requested.
    IgnoreRange,
}

struct Origin {
    base: String,
    hits: Arc<AtomicUsize>,
}

/// Minimal HTTP/1.1 origin. Hand-rolled rather than pulling a test server
/// dependency into a workspace that keeps its surface deliberately lean.
fn serve(body: Vec<u8>, fault: Fault) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_bg = hits.clone();

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { break };
            let n = hits_bg.fetch_add(1, Ordering::SeqCst);
            let range_from = read_request_range(&mut s);
            let _ = handle(&mut s, &body, fault, n, range_from);
        }
    });

    Origin {
        base: format!("http://{addr}"),
        hits,
    }
}

/// Parse `Range: bytes=N-`, returning N.
fn read_request_range(s: &mut TcpStream) -> Option<u64> {
    let mut r = BufReader::new(s.try_clone().ok()?);
    let mut from = None;
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).ok()? == 0 || line == "\r\n" {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
            from = v.trim().trim_end_matches('-').parse::<u64>().ok();
        }
    }
    from
}

fn handle(
    s: &mut TcpStream,
    body: &[u8],
    fault: Fault,
    hit: usize,
    range_from: Option<u64>,
) -> std::io::Result<()> {
    let start = range_from.unwrap_or(0) as usize;
    let partial = range_from.is_some() && !matches!(fault, Fault::IgnoreRange);
    let slice = if partial && start <= body.len() {
        &body[start..]
    } else {
        body
    };

    let (code, reason) = if partial {
        (206, "Partial Content")
    } else {
        (200, "OK")
    };
    write!(
        s,
        "HTTP/1.1 {code} {reason}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
        slice.len()
    )?;

    match fault {
        // Declare the full length but send less, then hang up mid-body.
        Fault::Truncate(n) => {
            s.write_all(&slice[..n.min(slice.len())])?;
        }
        // Fail the first attempt partway, serve the rest afterwards.
        Fault::DropThenResume(n) if hit == 0 => {
            s.write_all(&slice[..n.min(slice.len())])?;
        }
        _ => {
            s.write_all(slice)?;
        }
    }
    s.flush()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().fold(String::new(), |mut a, b| {
        use std::fmt::Write;
        let _ = write!(a, "{b:02x}");
        a
    })
}

struct Fixture {
    _dir: tempdir::TempDir,
    store: Store,
    root: PathBuf,
}

/// A throwaway store. Kept local so a failing test cannot touch a real cache.
mod tempdir {
    use std::path::{Path, PathBuf};
    pub struct TempDir(pub PathBuf);
    impl TempDir {
        pub fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "hipfire-hub-test-{}-{tag}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn fixture(tag: &str) -> Fixture {
    let dir = tempdir::TempDir::new(tag);
    let root = dir.path().to_path_buf();
    let store = Store::new(&root, "org/model");
    std::fs::create_dir_all(store.dir().join("blobs")).unwrap();
    Fixture {
        _dir: dir,
        store,
        root,
    }
}

fn payload(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8).collect()
}

/// Nothing may remain at a committed path, and no stray blob may exist.
fn assert_nothing_committed(fx: &Fixture, sha: &str) {
    let blob = fx.store.blob_path(sha);
    assert!(
        !blob.exists(),
        "a file that was never proven correct appeared at its final path: {}",
        blob.display()
    );
    let blobs = fx.store.dir().join("blobs");
    let committed: Vec<_> = std::fs::read_dir(&blobs)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                // `.part` files are working state, not commitments.
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        committed.is_empty(),
        "unexpected committed blob(s): {committed:?}"
    );
    let _ = &fx.root;
}

// ── the faults ──────────────────────────────────────────────────────────────

/// A body that does not match the promised digest must be refused outright.
/// This is the single most important assertion in the suite: it is the check
/// that a silently-wrong download cannot pass.
#[tokio::test]
async fn wrong_bytes_are_rejected_and_nothing_is_committed() {
    let fx = fixture("wrongbytes");
    let real = payload(64 * 1024);
    let promised = sha256_hex(&payload(32 * 1024)); // digest of different content
    let origin = serve(real, Fault::WrongBytes);

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: 64 * 1024,
        sha256: Some(promised.clone()),
        git_oid: None,
    };
    let err = fetch_file_from(&origin.base, "org/model", "main", &file, &fx.store)
        .await
        .expect_err("a digest mismatch must not succeed");

    match err {
        Error::Digest { want, got, .. } => {
            assert_eq!(want, promised);
            assert_ne!(got, promised);
        }
        other => panic!("expected a digest mismatch, got {other:?}"),
    }
    assert_nothing_committed(&fx, &promised);
}

/// A body cut short must fail, and must not be committed as if complete.
#[tokio::test]
async fn truncated_body_fails_and_is_not_committed() {
    let fx = fixture("truncated");
    let body = payload(256 * 1024);
    let sha = sha256_hex(&body);
    let origin = serve(body, Fault::Truncate(4096));

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: 256 * 1024,
        sha256: Some(sha.clone()),
        git_oid: None,
    };
    let err = fetch_file_from(&origin.base, "org/model", "main", &file, &fx.store)
        .await
        .expect_err("a truncated body must not succeed");
    assert!(
        matches!(err, Error::Retryable(_) | Error::Digest { .. }),
        "expected retryable or digest failure, got {err:?}"
    );
    assert_nothing_committed(&fx, &sha);
}

/// A connection dropped mid-transfer must resume, not restart, and the result
/// must still be verified. This is the behaviour a 4 GB shard on a lossy link
/// depends on entirely.
#[tokio::test]
async fn dropped_connection_resumes_and_verifies() {
    let fx = fixture("resume");
    let body = payload(512 * 1024);
    let sha = sha256_hex(&body);
    let origin = serve(body.clone(), Fault::DropThenResume(100 * 1024));

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: body.len() as u64,
        sha256: Some(sha.clone()),
        git_oid: None,
    };

    // First attempt drops partway and must keep the partial for resume.
    let first = fetch_file_from(&origin.base, "org/model", "main", &file, &fx.store).await;
    assert!(
        first.is_err(),
        "the dropped attempt should not have succeeded"
    );
    let part = fx.store.part_path(&file.path);
    let carried = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    assert!(
        carried > 0,
        "the partial transfer was discarded, so the retry would restart from zero"
    );

    // Second attempt resumes and completes.
    let blob = fetch_file_from(&origin.base, "org/model", "main", &file, &fx.store)
        .await
        .expect("resume should complete");
    assert_eq!(std::fs::read(&blob).unwrap(), body, "resumed bytes differ");
    assert!(origin.hits.load(Ordering::SeqCst) >= 2);
}

/// A `.part` holding the wrong bytes must not be spliced into a "valid" file.
/// Resuming on top of garbage is the subtle way to manufacture corruption.
#[tokio::test]
async fn corrupt_partial_is_not_spliced_into_a_valid_file() {
    let fx = fixture("badpart");
    let body = payload(256 * 1024);
    let sha = sha256_hex(&body);

    // Plant a partial whose contents are not a prefix of the real body.
    let part = fx.store.part_path("model.safetensors");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, vec![0xAAu8; 50 * 1024]).unwrap();

    let origin = serve(body, Fault::None);
    let file = RepoFile {
        path: "model.safetensors".into(),
        size: 256 * 1024,
        sha256: Some(sha.clone()),
        git_oid: None,
    };
    let err = fetch_file_from(&origin.base, "org/model", "main", &file, &fx.store)
        .await
        .expect_err("resuming onto a corrupt prefix must not produce a 'valid' file");
    assert!(
        matches!(err, Error::Digest { .. }),
        "expected a digest mismatch, got {err:?}"
    );
    assert_nothing_committed(&fx, &sha);
}

/// An origin that ignores Range answers 200 with the whole body. Appending that
/// to an existing prefix would corrupt the file, so the prefix must be dropped.
#[tokio::test]
async fn origin_ignoring_range_restarts_instead_of_appending() {
    let fx = fixture("norange");
    let body = payload(128 * 1024);
    let sha = sha256_hex(&body);

    let part = fx.store.part_path("model.safetensors");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, &body[..20 * 1024]).unwrap();

    let origin = serve(body.clone(), Fault::IgnoreRange);
    let file = RepoFile {
        path: "model.safetensors".into(),
        size: body.len() as u64,
        sha256: Some(sha.clone()),
        git_oid: None,
    };
    let blob = fetch_file_from(&origin.base, "org/model", "main", &file, &fx.store)
        .await
        .expect("a 200 response should restart cleanly");
    assert_eq!(
        std::fs::read(&blob).unwrap(),
        body,
        "the prefix was appended to instead of discarded"
    );
}

/// Refuse before writing when the revision cannot fit, rather than filling the
/// filesystem and discovering it partway.
#[test]
fn preflight_refuses_when_the_fetch_cannot_fit() {
    let fx = fixture("space");
    let huge = RepoFile {
        path: "model.safetensors".into(),
        size: u64::MAX / 4,
        sha256: None,
        git_oid: None,
    };
    let err = hipfire_hub::preflight(&[huge], fx.store.dir(), 0)
        .expect_err("an impossible fetch must be refused up front");
    assert!(matches!(err, Error::Space { .. }), "got {err:?}");
}

/// A `.part` from a process that no longer exists is inert, and sweeping it
/// must not disturb committed blobs.
#[test]
fn stale_parts_are_swept_without_touching_blobs() {
    let fx = fixture("stale");
    let blobs = fx.store.dir().join("blobs");
    std::fs::write(blobs.join("deadbeef"), b"committed").unwrap();
    // PID 0 is never a live user process here.
    std::fs::write(blobs.join(".model.safetensors.0.part"), vec![0u8; 4096]).unwrap();

    let freed = fx.store.sweep_stale_parts().expect("sweep");
    assert_eq!(freed, 4096, "the stale partial was not reclaimed");
    assert!(
        blobs.join("deadbeef").exists(),
        "sweeping removed a committed blob"
    );
    assert!(!blobs.join(".model.safetensors.0.part").exists());
}

/// A partial left by a process that has died must be picked up, not discarded.
///
/// This is the gap the suite had: every other case here exercises a transport
/// fault *within one run*, so a bug in what survives across runs went
/// unnoticed. PID-scoping the partial gave concurrency safety and silently
/// removed the ability to resume after an interruption — a killed 7 GB fetch
/// left 0.18 GB that the next run could neither see nor use, and would have
/// swept as litter.
#[test]
fn a_dead_runs_partial_is_adopted_rather_than_discarded() {
    let fx = fixture("orphan");
    let blobs = fx.store.dir().join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();

    // A partial tagged with a pid that is not running. Named the way the
    // previous run would have named it.
    let orphan = blobs.join(".model.safetensors.999999.part");
    std::fs::write(&orphan, vec![7u8; 8192]).unwrap();

    let adopted = fx.store.adopt_orphan_parts().expect("adopt");
    assert_eq!(
        adopted, 8192,
        "the interrupted run's progress was not recovered"
    );

    let mine = fx.store.part_path("model.safetensors");
    assert!(mine.exists(), "the partial was not claimed by this run");
    assert_eq!(std::fs::read(&mine).unwrap().len(), 8192);
    assert!(
        !orphan.exists(),
        "the orphan was left behind as well as copied"
    );
}

/// A partial belonging to a *live* process must be left strictly alone, or the
/// concurrency safety that pid-scoping exists for is gone.
#[test]
fn a_live_runs_partial_is_never_stolen() {
    let fx = fixture("livepart");
    let blobs = fx.store.dir().join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();

    // This process is alive by definition, and is not us only in the sense
    // that the name is built by hand — using our own pid proves the guard
    // covers the self case too.
    let live = blobs.join(format!(".other.safetensors.{}.part", std::process::id()));
    std::fs::write(&live, vec![3u8; 4096]).unwrap();

    let adopted = fx.store.adopt_orphan_parts().expect("adopt");
    assert_eq!(adopted, 0, "a live process's partial was taken");
    assert!(live.exists(), "a live process's partial was removed");

    let freed = fx.store.sweep_stale_parts().expect("sweep");
    assert_eq!(freed, 0, "a live process's partial was swept");
    assert!(live.exists());
}

/// Adoption must not lose ground: if this run already got further than the
/// orphan did, keep the better one.
#[test]
fn adoption_keeps_whichever_partial_got_further() {
    let fx = fixture("further");
    let blobs = fx.store.dir().join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();

    let mine = fx.store.part_path("model.safetensors");
    std::fs::write(&mine, vec![1u8; 9000]).unwrap();
    std::fs::write(
        blobs.join(".model.safetensors.999999.part"),
        vec![2u8; 1000],
    )
    .unwrap();

    fx.store.adopt_orphan_parts().expect("adopt");
    assert_eq!(
        std::fs::metadata(&mine).unwrap().len(),
        9000,
        "a shorter orphan overwrote this run's longer partial"
    );
}

// ── the streamed path ───────────────────────────────────────────────────────
//
// `fetch_file_streamed` gives up the property the assertions above rest on: a
// sink has already consumed the bytes by the time the digest is known, so
// "nothing is committed" cannot be checked at a final path. What replaces it is
// narrower but is the thing that can actually go wrong here — the sink must end
// up holding the body **exactly once**. Resume that re-sends a prefix would
// duplicate; a reset that fails to fire would splice two attempts together.

#[derive(Default)]
struct Recorder {
    got: Vec<u8>,
    resets: usize,
}

impl hipfire_hub::ByteSink for Recorder {
    fn chunk(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.got.extend_from_slice(b);
        Ok(())
    }
    fn reset(&mut self) -> std::io::Result<()> {
        self.got.clear();
        self.resets += 1;
        Ok(())
    }
}

/// A dropped connection must resume from where it stopped, not re-send a prefix
/// the sink has already encoded.
#[tokio::test]
async fn streamed_dropped_connection_resumes_without_duplicating() {
    let body = payload(512 * 1024);
    let sha = sha256_hex(&body);
    let origin = serve(body.clone(), Fault::DropThenResume(100 * 1024));

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: body.len() as u64,
        sha256: Some(sha),
        git_oid: None,
    };

    let mut st = hipfire_hub::StreamProgress::new(&file);
    let mut sink = Recorder::default();

    let first = hipfire_hub::fetch_file_streamed(
        &origin.base,
        "org/model",
        "main",
        &file,
        &mut st,
        &mut sink,
    )
    .await;
    assert!(
        first.is_err(),
        "the dropped attempt should not have succeeded"
    );
    let carried = st.consumed();
    assert!(
        carried > 0,
        "progress was discarded, so the retry would restart from zero"
    );

    hipfire_hub::fetch_file_streamed(&origin.base, "org/model", "main", &file, &mut st, &mut sink)
        .await
        .expect("resume should complete and verify");

    assert_eq!(sink.resets, 0, "a clean resume must not reset the sink");
    assert_eq!(
        sink.got.len(),
        body.len(),
        "sink holds {} bytes for a {} byte body — the prefix was duplicated or lost",
        sink.got.len(),
        body.len()
    );
    assert!(sink.got == body, "resumed bytes differ from the source");
    assert!(origin.hits.load(Ordering::SeqCst) >= 2);
}

/// An origin that answers 200 to a Range request must make the sink start over.
/// Appending the full body onto a partial is how a resume manufactures a file
/// that is longer than the original and matches nothing.
#[tokio::test]
async fn streamed_origin_ignoring_range_restarts_the_sink() {
    let body = payload(256 * 1024);
    let sha = sha256_hex(&body);
    let origin = serve(body.clone(), Fault::IgnoreRange);

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: body.len() as u64,
        sha256: Some(sha),
        git_oid: None,
    };

    let mut st = hipfire_hub::StreamProgress::new(&file);
    let mut sink = Recorder::default();

    // Prime a partial so the retry carries a Range header the origin will ignore.
    let dropper = serve(body.clone(), Fault::Truncate(64 * 1024));
    let _ = hipfire_hub::fetch_file_streamed(
        &dropper.base,
        "org/model",
        "main",
        &file,
        &mut st,
        &mut sink,
    )
    .await;
    assert!(st.consumed() > 0, "no partial to resume from");

    hipfire_hub::fetch_file_streamed(&origin.base, "org/model", "main", &file, &mut st, &mut sink)
        .await
        .expect("a full 200 body should still verify");

    assert_eq!(
        sink.resets, 1,
        "the sink should have been reset exactly once"
    );
    assert!(
        sink.got == body,
        "sink holds {} bytes for a {} byte body — the ignored Range spliced two attempts",
        sink.got.len(),
        body.len()
    );
}

/// The digest check must still fire on the streamed path. It cannot prevent the
/// sink seeing bad bytes, so it has to at least tell the caller to undo them.
#[tokio::test]
async fn streamed_wrong_bytes_are_reported_as_a_digest_error() {
    let real = payload(64 * 1024);
    let promised = sha256_hex(&payload(32 * 1024)); // digest of different content
    let origin = serve(real, Fault::WrongBytes);

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: 64 * 1024,
        sha256: Some(promised),
        git_oid: None,
    };

    let mut st = hipfire_hub::StreamProgress::new(&file);
    let mut sink = Recorder::default();
    let err = hipfire_hub::fetch_file_streamed(
        &origin.base,
        "org/model",
        "main",
        &file,
        &mut st,
        &mut sink,
    )
    .await
    .expect_err("a body that does not match its digest must be refused");

    assert!(
        matches!(err, Error::Digest { .. }),
        "expected a digest mismatch, got {err:?} — the caller keys its rollback on this"
    );
}

/// A truncated body must not verify, however well-formed the prefix looked.
#[tokio::test]
async fn streamed_truncated_body_does_not_verify() {
    let body = payload(256 * 1024);
    let sha = sha256_hex(&body);
    let origin = serve(body.clone(), Fault::Truncate(90 * 1024));

    let file = RepoFile {
        path: "model.safetensors".into(),
        size: body.len() as u64,
        sha256: Some(sha),
        git_oid: None,
    };

    let mut st = hipfire_hub::StreamProgress::new(&file);
    let mut sink = Recorder::default();
    let r = hipfire_hub::fetch_file_streamed(
        &origin.base,
        "org/model",
        "main",
        &file,
        &mut st,
        &mut sink,
    )
    .await;
    assert!(r.is_err(), "a short body must not be accepted as complete");
    assert!(
        sink.got.len() < body.len(),
        "the sink should hold only what arrived"
    );
}
