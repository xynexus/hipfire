//! Where inbound request frames come from, and where their replies go.
//!
//! The daemon used to read and execute in one statement —
//! `for line in stdin.lock().lines()` with the whole request `match` inline — so
//! parsing and GPU execution were the same thread doing the same thing in
//! lockstep. Splitting them is what the rest of the merge needs:
//!
//! - **A queue exists.** Once frames arrive on a channel rather than straight off
//!   the reader, there is somewhere for a scheduler to sit and reorder them. Today
//!   the daemon never *chooses* anything, which is why it needs no scheduler.
//! - **A control frame can overtake GPU work.** `abort` currently replies that it
//!   "is handled on the control channel, not the request channel" — and there is
//!   no control channel, so an abort can only be read *after* the generation it
//!   wanted to cancel has finished. A separate reader is the first half of fixing
//!   that.
//! - **More than one client.** Each accepted connection is another producer on the
//!   same channel, carrying its own reply sink.
//!
//! Execution deliberately stays on the main thread. `hipfire_rdna::Gpu` is
//! initialised there and HIP contexts are thread-affine, so moving the executor
//! to a spawned thread would mean moving GPU init with it. Moving the *readers*
//! instead gets the same decoupling with none of that risk — and keeps a single
//! writer per connection, so frames cannot interleave mid-line.
//!
//! Malformed input is forwarded as a frame rather than reported by a reader, so
//! every response is written by the executor and stays ordered against the
//! requests around it.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use hipfire_runtime::cancel::{self, StopKind};

/// A cloneable handle to one connection's reply stream.
///
/// Cloneable because every frame from a connection has to be able to answer on
/// it, and `Write` needs `&mut`. The mutex is effectively uncontended — only the
/// executor writes, and only one request is in flight at a time — but it is what
/// lets the handle be shared without handing out `&mut` to the same stream twice.
///
/// **Only the executor may write.** The lock is taken per `write` call, and
/// `writeln!` lowers to several of them, so two threads writing one frame could
/// interleave mid-line and corrupt the stream. That is why reader threads handle
/// control frames silently instead of acknowledging them — which also matches
/// `abort` being fire-and-forget by protocol.
#[derive(Clone)]
pub(crate) struct ReplySink(Arc<Mutex<Box<dyn Write + Send>>>);

impl ReplySink {
    pub fn new(writer: impl Write + Send + 'static) -> Self {
        Self(Arc::new(Mutex::new(Box::new(writer))))
    }
}

impl Write for ReplySink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // A poisoned lock means a previous writer panicked mid-frame. The stream
        // may be mid-line, but dropping every later reply is worse than a possibly
        // ragged one, so recover rather than propagate.
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .flush()
    }
}

/// What a reader produces.
pub(crate) enum Payload {
    /// A frame that parsed as JSON. Not yet validated as a `DaemonRequest`.
    Request(serde_json::Value),
    /// A line that was not valid JSON, carrying the parse error.
    Malformed(String),
}

/// One thing for the executor to deal with, and where its answer goes.
pub(crate) struct Inbound {
    pub payload: Payload,
    pub reply: ReplySink,
    /// Which connection this arrived on.
    ///
    /// The scheduler needs it because a connection's frames are a DEPENDENCY
    /// CHAIN, not independent work: `reserve_session_state` → `generate_batch_prefill`
    /// → `generate_batch_decode_step` → `release_sessions`, or plain
    /// `load` → `generate` → `unload`. Reordering within a connection would run a
    /// request before the one it depends on. Across connections there is no such
    /// relationship, which is the only place reordering is safe.
    pub conn: u64,
    /// Global arrival order, used to break priority ties so equal-priority work
    /// stays first-come-first-served.
    pub seq: u64,
    /// Lower is sooner, matching `hipfire_scheduler`'s numbering (0 realtime,
    /// 64 default, 255 opportunistic). Absent on the wire means default.
    pub priority: u8,
}

/// How many frames may sit between the readers and the executor.
///
/// This was 0 (rendezvous) until the daemon had a scheduler: with nothing to
/// choose between, buffering bought nothing and the OS pipe was a fine queue.
/// A scheduler needs several pending requests IN HAND to reorder, so a reader
/// must be able to deposit a frame while the executor is busy — at capacity 0 a
/// long generation blocks every reader, and nothing can queue behind it.
///
/// Bounded rather than unbounded so a client cannot make the daemon buffer without
/// limit; past this the readers block again and backpressure returns to the pipe.
const INBOUND_CAPACITY: usize = 256;

static NEXT_CONN: AtomicU64 = AtomicU64::new(0);
static NEXT_SEQ: AtomicU64 = AtomicU64::new(0);

/// Read a request's scheduling priority, clamped to the same scale the server
/// uses so both ends agree on what "high" means.
fn frame_priority(value: &serde_json::Value) -> u8 {
    value
        .get("priority")
        .and_then(|v| v.as_i64())
        .map(hipfire_scheduler::clamp_scheduler_priority)
        .unwrap_or(hipfire_scheduler::SCHED_PRIORITY_DEFAULT)
}

/// Default socket path, defined once in the adapter so the listening end and the
/// connecting end cannot disagree about where the door is.
pub(crate) fn default_socket_path() -> PathBuf {
    hipfire_daemon_adapter::default_socket_path()
}

/// Spawn the stdin reader and return the executor's end of the channel.
///
/// The channel closes when stdin reaches EOF or errors, which is how the executor
/// learns to shut down — the same condition the old `Err(_) => break` handled.
pub(crate) fn spawn_stdin_reader() -> Receiver<Inbound> {
    let (tx, rx) = sync_channel(INBOUND_CAPACITY);
    let reply = ReplySink::new(std::io::stdout());
    let conn = NEXT_CONN.fetch_add(1, Ordering::Relaxed);
    std::thread::Builder::new()
        .name("hipfire-daemon-stdin".to_string())
        .spawn(move || read_frames(std::io::stdin().lock(), &tx, &reply, conn))
        .expect("spawn stdin reader thread");
    rx
}

/// Bind the listening socket and spawn the accept loop.
///
/// Unlike the stdio path this channel never closes on its own: the listener
/// outlives any one client, so the daemon keeps serving after a disconnect and
/// only exits on a signal.
pub(crate) fn spawn_socket_listener(path: &Path) -> std::io::Result<Receiver<Inbound>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Any socket file here is stale by definition: we already hold the exclusive
    // flock on daemon.pid, so no other daemon is live to own it. Bind would
    // otherwise fail with EADDRINUSE against a file nobody is listening on.
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let listener = UnixListener::bind(path)?;
    // Owner-only: a unix socket in ~/.hipfire is same-uid by intent, matching the
    // trust model `admin.secret` already uses. This is not an authentication
    // boundary and is not meant to become one.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;

    let (tx, rx) = sync_channel(INBOUND_CAPACITY);
    std::thread::Builder::new()
        .name("hipfire-daemon-accept".to_string())
        .spawn(move || accept_loop(listener, tx))
        .expect("spawn accept thread");
    Ok(rx)
}

/// Accept connections forever, giving each its own reader thread.
fn accept_loop(listener: UnixListener, tx: SyncSender<Inbound>) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        // Separate fds for the two directions, so the reader thread can block on
        // reads while the executor writes replies. That is what makes a single
        // writer per connection possible without sharing one handle mutably.
        let Ok(write_half) = stream.try_clone() else {
            continue;
        };
        let tx = tx.clone();
        let _ = std::thread::Builder::new()
            .name("hipfire-daemon-conn".to_string())
            .spawn(move || serve_connection(stream, write_half, &tx));
    }
}

fn serve_connection(read_half: UnixStream, write_half: UnixStream, tx: &SyncSender<Inbound>) {
    let reply = ReplySink::new(write_half);
    let conn = NEXT_CONN.fetch_add(1, Ordering::Relaxed);
    read_frames(BufReader::new(read_half), tx, &reply, conn);
}

/// Frames a reader answers itself, without involving the executor.
///
/// This is the control channel. The executor may be mid-generation and not
/// reading the queue at all, so a frame that has to reach a *running* request
/// cannot travel through it — that is exactly why `abort` was previously
/// unimplementable and replied that it "is handled on the control channel, not the
/// request channel".
///
/// Returns true when the frame was consumed here. A control frame naming no
/// request is *not* consumed: it goes to the executor so the caller gets told
/// their request was malformed, rather than being silently dropped.
fn handle_control_frame(value: &serde_json::Value) -> bool {
    let kind = match value.get("type").and_then(|v| v.as_str()) {
        Some("abort") => StopKind::Abort,
        Some("force_answer") => StopKind::ForceAnswer,
        _ => return false,
    };
    let target = value.get("id").and_then(|v| v.as_str()).unwrap_or_default();
    cancel::request(target, kind)
}

/// Read newline-delimited frames from `source` and forward them until EOF, a read
/// error, or the executor hanging up.
fn read_frames(source: impl BufRead, tx: &SyncSender<Inbound>, reply: &ReplySink, conn: u64) {
    for line in source.lines() {
        let Ok(line) = line else {
            // A read error is EOF as far as the protocol is concerned; the old
            // loop broke here too.
            return;
        };
        if line.trim().is_empty() {
            continue;
        }
        let mut priority = hipfire_scheduler::SCHED_PRIORITY_DEFAULT;
        let payload = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(value) => {
                // Control frames never enter the queue: with a rendezvous channel
                // and a busy executor, enqueuing an abort would block until the
                // generation it wants to stop had already finished.
                if handle_control_frame(&value) {
                    continue;
                }
                priority = frame_priority(&value);
                Payload::Request(value)
            }
            Err(error) => Payload::Malformed(error.to_string()),
        };
        // A send error means the executor is gone, so there is nobody left to
        // read anything for.
        if tx
            .send(Inbound {
                payload,
                reply: reply.clone(),
                conn,
                seq: NEXT_SEQ.fetch_add(1, Ordering::Relaxed),
                priority,
            })
            .is_err()
        {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(input: &str) -> Vec<Payload> {
        let (tx, rx) = sync_channel(64);
        let reply = ReplySink::new(Vec::new());
        read_frames(std::io::Cursor::new(input.to_string()), &tx, &reply, 0);
        drop(tx);
        rx.into_iter().map(|frame| frame.payload).collect()
    }

    #[test]
    fn reader_forwards_frames_skips_blanks_and_defers_parse_errors() {
        let frames = drain("{\"type\":\"ping\"}\n\n   \n{\"type\":\"pong\"}\n");
        assert_eq!(
            frames.len(),
            2,
            "blank and whitespace-only lines are skipped"
        );
        assert!(matches!(&frames[0], Payload::Request(v) if v["type"] == "ping"));
        assert!(matches!(&frames[1], Payload::Request(v) if v["type"] == "pong"));

        // A malformed line is forwarded rather than reported by the reader, so the
        // executor stays the only writer and the error keeps its place in the
        // stream relative to the requests around it.
        let mixed = drain("{\"type\":\"ping\"}\nnot json\n{\"type\":\"pong\"}\n");
        assert_eq!(mixed.len(), 3);
        assert!(matches!(&mixed[0], Payload::Request(_)));
        assert!(matches!(&mixed[1], Payload::Malformed(_)));
        assert!(matches!(&mixed[2], Payload::Request(_)));
    }

    #[test]
    fn reader_stops_when_the_executor_hangs_up() {
        // Capacity 0 plus a dropped receiver: the first send fails and the reader
        // returns instead of spinning on a channel nobody is reading.
        let (tx, rx) = sync_channel(0);
        drop(rx);
        let reply = ReplySink::new(Vec::new());
        read_frames(
            std::io::Cursor::new("{\"type\":\"ping\"}\n".to_string()),
            &tx,
            &reply,
            0,
        );
    }

    #[test]
    fn every_frame_from_a_connection_answers_on_that_connection() {
        // Two readers with distinct sinks: whichever frame the executor picks up,
        // its reply has to go back to the sink that produced it. This is the
        // property that makes more than one client possible at all.
        let (tx, rx) = sync_channel(64);
        let first = ReplySink::new(Vec::new());
        let second = ReplySink::new(Vec::new());
        read_frames(
            std::io::Cursor::new("{\"n\":1}\n".to_string()),
            &tx,
            &first,
            1,
        );
        read_frames(
            std::io::Cursor::new("{\"n\":2}\n".to_string()),
            &tx,
            &second,
            2,
        );
        drop(tx);

        let frames: Vec<_> = rx.into_iter().collect();
        assert_eq!(frames.len(), 2);
        // Same underlying stream for a frame and the sink it arrived on.
        assert!(Arc::ptr_eq(&frames[0].reply.0, &first.0));
        assert!(Arc::ptr_eq(&frames[1].reply.0, &second.0));
        assert!(!Arc::ptr_eq(&frames[0].reply.0, &frames[1].reply.0));
    }

    #[test]
    fn a_reply_sink_clone_writes_to_the_same_stream() {
        // `Responder` swaps in a clone per frame, so a clone must not be a
        // separate buffer.
        let sink = ReplySink::new(Vec::new());
        let mut a = sink.clone();
        let mut b = sink.clone();
        write!(a, "one ").unwrap();
        write!(b, "two").unwrap();
        let guard = sink.0.lock().unwrap();
        // Can't read a Box<dyn Write> back, so assert via the shared identity
        // instead: both clones point at one allocation.
        drop(guard);
        assert!(Arc::ptr_eq(&a.0, &b.0));
    }
}
