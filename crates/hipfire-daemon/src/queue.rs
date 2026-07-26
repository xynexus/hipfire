//! What the daemon runs next.
//!
//! Until this existed the daemon never *chose* anything: frames were executed in
//! arrival order, one at a time, which is why it needed no scheduler. Choosing
//! requires two things it now has — several pending frames in hand (the inbound
//! channel is buffered rather than a rendezvous) and a priority on the wire.
//!
//! # The invariant that makes reordering safe
//!
//! **A connection's frames are a dependency chain; different connections are
//! independent.** `reserve_session_state` → `generate_batch_prefill` →
//! `generate_batch_decode_step` → `release_sessions` only works in that order, as
//! does `load` → `generate` → `unload`. Reordering within a connection would run a
//! request before the one it depends on, and nothing in the protocol declares
//! those dependencies — the order *is* the declaration.
//!
//! So this queue never reorders within a connection. It holds one FIFO per
//! connection and chooses only among their HEADS. Priority therefore decides which
//! client goes next, never which of a client's own requests goes next.
//!
//! A consequence worth stating: with a single connection (all of stdio, and any
//! one socket client) selection is a no-op and behaviour is exactly FIFO as
//! before. Reordering only becomes observable with concurrent clients, which is
//! also the only situation where it is safe.
//!
//! Priorities follow `hipfire_scheduler`'s scale — lower is sooner, 0 realtime,
//! 64 default, 255 opportunistic — so the daemon and the server mean the same
//! thing by a number.

use std::collections::BTreeMap;
use std::collections::VecDeque;

use crate::transport::Inbound;

/// Pending work, grouped into one first-come-first-served queue per connection.
#[derive(Default)]
pub(crate) struct PendingQueue {
    /// `BTreeMap` rather than `HashMap` so that when priority and arrival order
    /// both tie, selection is still deterministic instead of hash-order.
    per_connection: BTreeMap<u64, VecDeque<Inbound>>,
}

impl PendingQueue {
    pub fn push(&mut self, frame: Inbound) {
        self.per_connection
            .entry(frame.conn)
            .or_default()
            .push_back(frame);
    }

    pub fn is_empty(&self) -> bool {
        self.per_connection.values().all(VecDeque::is_empty)
    }

    /// Total frames waiting across all connections. Surfaced in the scheduling
    /// trace today; M4c's `/health` counters are the next consumer.
    pub fn len(&self) -> usize {
        self.per_connection.values().map(VecDeque::len).sum()
    }

    /// Take the frame that should run next.
    ///
    /// Considers only the head of each connection — taking a frame from behind a
    /// head would reorder that connection against itself. Among the heads: lowest
    /// priority number wins; ties go to the earliest arrival, so equal-priority
    /// work stays first-come-first-served rather than favouring whichever
    /// connection happens to sort first.
    pub fn pop_next(&mut self) -> Option<Inbound> {
        let chosen = self
            .per_connection
            .iter()
            .filter_map(|(conn, queue)| queue.front().map(|head| (*conn, head.priority, head.seq)))
            .min_by_key(|(_, priority, seq)| (*priority, *seq))
            .map(|(conn, _, _)| conn)?;

        let queue = self.per_connection.get_mut(&chosen)?;
        let frame = queue.pop_front();
        // Drop the connection's slot once drained so a long-lived daemon does not
        // accumulate an entry per connection it has ever served.
        if queue.is_empty() {
            self.per_connection.remove(&chosen);
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{Payload, ReplySink};

    fn frame(conn: u64, seq: u64, priority: u8, tag: &str) -> Inbound {
        Inbound {
            payload: Payload::Request(serde_json::json!({ "tag": tag })),
            reply: ReplySink::new(Vec::new()),
            conn,
            seq,
            priority,
        }
    }

    fn tag_of(frame: &Inbound) -> String {
        match &frame.payload {
            Payload::Request(v) => v["tag"].as_str().unwrap_or_default().to_string(),
            Payload::Malformed(_) => "<malformed>".to_string(),
        }
    }

    fn drain(queue: &mut PendingQueue) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(f) = queue.pop_next() {
            out.push(tag_of(&f));
        }
        out
    }

    #[test]
    fn one_connection_is_never_reordered_however_the_priorities_fall() {
        // The safety property. A single connection's frames are a dependency chain
        // (reserve -> prefill -> decode -> release), so even a screaming-priority
        // frame behind a low-priority one must wait its turn.
        let mut q = PendingQueue::default();
        q.push(frame(1, 0, 255, "reserve"));
        q.push(frame(1, 1, 0, "prefill-urgent"));
        q.push(frame(1, 2, 200, "decode"));
        assert_eq!(drain(&mut q), ["reserve", "prefill-urgent", "decode"]);
    }

    #[test]
    fn priority_decides_between_connections() {
        let mut q = PendingQueue::default();
        q.push(frame(1, 0, 200, "bulk-a"));
        q.push(frame(2, 1, 0, "realtime"));
        q.push(frame(1, 2, 200, "bulk-b"));
        // The urgent frame arrived second but runs first; connection 1 keeps its
        // own order behind it.
        assert_eq!(drain(&mut q), ["realtime", "bulk-a", "bulk-b"]);
    }

    #[test]
    fn equal_priority_stays_first_come_first_served() {
        let mut q = PendingQueue::default();
        q.push(frame(7, 3, 64, "third"));
        q.push(frame(2, 1, 64, "first"));
        q.push(frame(5, 2, 64, "second"));
        // Selection must follow arrival, not connection id — otherwise the
        // lowest-numbered connection would quietly starve the others.
        assert_eq!(drain(&mut q), ["first", "second", "third"]);
    }

    #[test]
    fn a_head_blocks_only_its_own_connection() {
        let mut q = PendingQueue::default();
        q.push(frame(1, 0, 255, "opportunistic-head"));
        q.push(frame(1, 1, 0, "urgent-but-stuck-behind"));
        q.push(frame(2, 2, 64, "other-client"));
        // The other client overtakes the opportunistic head, but the urgent frame
        // behind it does not — it is still second in its own chain.
        assert_eq!(
            drain(&mut q),
            [
                "other-client",
                "opportunistic-head",
                "urgent-but-stuck-behind"
            ]
        );
    }

    #[test]
    fn drained_connections_are_forgotten() {
        let mut q = PendingQueue::default();
        q.push(frame(1, 0, 64, "only"));
        assert_eq!(q.len(), 1);
        assert!(q.pop_next().is_some());
        assert!(q.is_empty());
        assert!(q.pop_next().is_none());
        assert!(
            q.per_connection.is_empty(),
            "a long-lived daemon must not keep a slot per connection it ever served"
        );
    }
}
