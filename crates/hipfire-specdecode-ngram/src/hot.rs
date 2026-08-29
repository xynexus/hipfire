// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RAM staging tier: the live table, and the admission gate in front of disk.
//!
//! Every observed gram lands here first. Grams earn their way to the cold store
//! by being seen `promote_count` times; the rest are evicted and forgotten
//! without ever touching disk. That keeps the write path a trickle rather than
//! a flood — measured on 1M tokens of Rust with the full order ladder, **71% of
//! distinct grams never serve a single draft**, so most of what a naive design
//! would persist is dead weight.
//!
//! Drafting reads this tier at `min_count = 1`. The measurement is explicit
//! that a count threshold *hurts* drafting (1.80 accepted/step at min_count=1
//! vs 0.75 at 9) while precision is flat across count buckets — so the
//! threshold gates persistence only, never proposal.
//!
//! Eviction is amortized: rather than maintain an exact MRU list, the table
//! runs one scoring pass when it grows past capacity and drops the weakest
//! quarter. Qualifying evictees go to `pending`, which the caller drains a few
//! at a time between tokens so no single decode step eats a large write.

use std::collections::{HashMap, VecDeque};

use crate::cold::{Staged, NO_TOKEN};

#[derive(Debug, Clone, Copy)]
struct HotEntry {
    /// Last token of the context — the cold store's block key.
    key: u32,
    next: u32,
    next2: u32,
    count: u16,
    order: u8,
    /// Stream position when last observed; the age clock.
    last_seen: u32,
    /// Already handed to the cold tier at least once.
    persisted: bool,
}

pub struct HotTable {
    map: HashMap<u64, HotEntry>,
    capacity: usize,
    /// Count at which a gram is worth a disk write.
    promote_count: u16,
    pos: u32,
    pending: VecDeque<Staged>,
    /// Cap on `pending` so a pathological burst cannot grow it without bound.
    pending_cap: usize,
    pub evicted: u64,
    pub promoted: u64,
}

impl HotTable {
    pub fn new(capacity: usize, promote_count: u16) -> Self {
        Self {
            map: HashMap::with_capacity(capacity.min(1 << 20)),
            capacity,
            promote_count: promote_count.max(1),
            pos: 0,
            pending: VecDeque::new(),
            pending_cap: 1 << 16,
            evicted: 0,
            promoted: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Advance the age clock. Call once per committed token.
    #[inline]
    pub fn tick(&mut self) {
        self.pos = self.pos.wrapping_add(1);
    }

    /// Returns `(next, next2, count)` if this context is known.
    #[inline]
    pub fn lookup(&self, fp: u64) -> Option<(u32, u32, u16)> {
        self.map.get(&fp).map(|e| (e.next, e.next2, e.count))
    }

    /// Record `context -> next` (and, for high orders, the token after it).
    ///
    /// A context whose follower changes is *replaced*, not histogrammed: the
    /// most recent continuation is the one worth drafting, and keeping a full
    /// per-context histogram costs memory for a signal the measurement says is
    /// flat (precision does not rise with count).
    pub fn observe(&mut self, key: u32, fp: u64, order: u8, next: u32, next2: u32) {
        let pos = self.pos;
        let promote = self.promote_count;
        let mut newly_qualified = None;

        match self.map.get_mut(&fp) {
            Some(e) => {
                if e.next == next {
                    e.count = e.count.saturating_add(1);
                } else {
                    // Follower changed: this context is ambiguous. Decay rather
                    // than reset so a stable gram survives one-off noise.
                    e.count = e.count.saturating_sub(1).max(1);
                    e.next = next;
                }
                if next2 != NO_TOKEN {
                    e.next2 = next2;
                }
                e.last_seen = pos;
                if !e.persisted && e.count >= promote {
                    e.persisted = true;
                    newly_qualified = Some(Staged {
                        key: e.key,
                        fp,
                        next: e.next,
                        next2: e.next2,
                        count: e.count,
                        order: e.order,
                    });
                }
            }
            None => {
                self.map.insert(
                    fp,
                    HotEntry {
                        key,
                        next,
                        next2,
                        count: 1,
                        order,
                        last_seen: pos,
                        persisted: false,
                    },
                );
            }
        }

        if let Some(s) = newly_qualified {
            self.push_pending(s);
        }
        if self.map.len() > self.capacity {
            self.compact();
        }
    }

    fn push_pending(&mut self, s: Staged) {
        if self.pending.len() < self.pending_cap {
            self.pending.push_back(s);
            self.promoted += 1;
        }
    }

    /// Drop the weakest quarter. Amortized: one O(n) pass per capacity/4
    /// observations, so the per-token cost is O(1).
    fn compact(&mut self) {
        let pos = self.pos;
        let promote = self.promote_count;
        let keep = self.capacity * 3 / 4;

        let mut scored: Vec<(i64, u64)> = self
            .map
            .iter()
            .map(|(fp, e)| {
                let age = pos.wrapping_sub(e.last_seen) as i64;
                // ponytail: count minus log2-ish age. Popular beats recent, but
                // a stale gram loses ground steadily. Revisit if the measured
                // age distribution says otherwise — it is non-monotonic on
                // prose (very old grams are structural and score well).
                (e.count as i64 * 8 - age.min(1 << 20), *fp)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let mut drain: Vec<Staged> = Vec::new();
        for (_, fp) in scored.into_iter().skip(keep) {
            if let Some(e) = self.map.remove(&fp) {
                self.evicted += 1;
                if !e.persisted && e.count >= promote {
                    drain.push(Staged {
                        key: e.key,
                        fp,
                        next: e.next,
                        next2: e.next2,
                        count: e.count,
                        order: e.order,
                    });
                }
            }
        }
        for s in drain {
            self.push_pending(s);
        }
    }

    /// Take up to `n` grams queued for the cold store. Call between tokens so
    /// the write path stays a trickle.
    pub fn drain_pending(&mut self, n: usize) -> Vec<Staged> {
        let take = n.min(self.pending.len());
        self.pending.drain(..take).collect()
    }

    /// Everything still queued — for shutdown, before a merge.
    pub fn take_all_pending(&mut self) -> Vec<Staged> {
        self.pending.drain(..).collect()
    }

    /// Forget the session-local table. The cold store is untouched.
    pub fn reset(&mut self) {
        self.map.clear();
        self.pos = 0;
    }
}
