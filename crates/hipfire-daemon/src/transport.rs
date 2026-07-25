//! Where inbound request frames come from.
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
//! - **More than one client becomes conceivable.** A socket listener is another
//!   producer on this channel.
//!
//! Execution deliberately stays on the main thread. `hipfire_rdna::Gpu` is
//! initialised there and HIP contexts are thread-affine, so moving the executor
//! to a spawned thread would mean moving GPU init with it. Moving the *reader*
//! instead gets the same decoupling with none of that risk.
//!
//! All writing also stays on the executor thread — malformed input is reported as
//! an [`Inbound::Malformed`] frame rather than by the reader — so responses keep a
//! single writer and stay ordered against the requests that produced them.

use std::io::BufRead;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

/// One thing the executor has to deal with, in arrival order.
pub(crate) enum Inbound {
    /// A frame that parsed as JSON. Not yet validated as a `DaemonRequest`.
    Request(serde_json::Value),
    /// A line that was not valid JSON, carrying the parse error.
    ///
    /// The reader does not report this itself: the error frame has to be ordered
    /// against real responses and written by the same thread that writes them.
    Malformed(String),
}

/// Rendezvous, so a reader cannot run ahead of the executor and buffer the whole
/// backlog in memory.
///
/// This preserves today's backpressure exactly — the OS pipe stays the queue, as
/// it is now. M4 is what deliberately raises this, because a scheduler needs
/// several pending requests in hand before it has anything to choose between.
const INBOUND_CAPACITY: usize = 0;

/// Spawn the stdin reader and return the executor's end of the channel.
///
/// The channel closes when stdin reaches EOF or errors, which is how the executor
/// learns to shut down — the same condition the old `Err(_) => break` handled.
pub(crate) fn spawn_stdin_reader() -> Receiver<Inbound> {
    let (tx, rx) = sync_channel(INBOUND_CAPACITY);
    std::thread::Builder::new()
        .name("hipfire-daemon-stdin".to_string())
        .spawn(move || read_frames(std::io::stdin().lock(), &tx))
        .expect("spawn stdin reader thread");
    rx
}

/// Read newline-delimited frames from `source` and forward them until EOF, a read
/// error, or the executor hanging up.
fn read_frames(source: impl BufRead, tx: &SyncSender<Inbound>) {
    for line in source.lines() {
        let Ok(line) = line else {
            // A read error is EOF as far as the protocol is concerned; the old
            // loop broke here too.
            return;
        };
        if line.trim().is_empty() {
            continue;
        }
        let frame = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(value) => Inbound::Request(value),
            Err(error) => Inbound::Malformed(error.to_string()),
        };
        // A send error means the executor is gone, so there is nobody left to
        // read anything for.
        if tx.send(frame).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(input: &str) -> Vec<Inbound> {
        let (tx, rx) = sync_channel(64);
        read_frames(std::io::Cursor::new(input.to_string()), &tx);
        drop(tx);
        rx.into_iter().collect()
    }

    #[test]
    fn reader_forwards_frames_skips_blanks_and_defers_parse_errors() {
        let frames = drain("{\"type\":\"ping\"}\n\n   \n{\"type\":\"pong\"}\n");
        assert_eq!(
            frames.len(),
            2,
            "blank and whitespace-only lines are skipped"
        );
        assert!(matches!(&frames[0], Inbound::Request(v) if v["type"] == "ping"));
        assert!(matches!(&frames[1], Inbound::Request(v) if v["type"] == "pong"));

        // A malformed line is forwarded rather than reported by the reader, so the
        // executor stays the only writer and the error keeps its place in the
        // stream relative to the requests around it.
        let mixed = drain("{\"type\":\"ping\"}\nnot json\n{\"type\":\"pong\"}\n");
        assert_eq!(mixed.len(), 3);
        assert!(matches!(&mixed[0], Inbound::Request(_)));
        assert!(matches!(&mixed[1], Inbound::Malformed(_)));
        assert!(matches!(&mixed[2], Inbound::Request(_)));
    }

    #[test]
    fn reader_stops_when_the_executor_hangs_up() {
        // Capacity 0 plus a dropped receiver: the first send fails and the reader
        // returns instead of spinning on a channel nobody is reading.
        let (tx, rx) = sync_channel(0);
        drop(rx);
        read_frames(
            std::io::Cursor::new("{\"type\":\"ping\"}\n".to_string()),
            &tx,
        );
    }
}
