// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixed-size, block-structured n-gram store on disk.
//!
//! ## Why blocks, and why keyed on the *last* context token
//!
//! 4 KiB is the disk read quantum, so the unit of layout is the block, and the
//! goal is to put the grams that get read together into the same block.
//!
//! A draft step at stream position `i` probes every order `k` in the ladder:
//! contexts `tokens[i-k..i]`. Those contexts have *different first tokens*
//! (`tokens[i-8]` vs `tokens[i-2]`) but they all end at the same token,
//! `tokens[i-1]`. Keying blocks on the **last** context token therefore lands
//! the entire order ladder in one block: one 4 KiB read serves all 7 probes.
//! Keying on the first token would scatter them across up to 7 blocks.
//!
//! ## Skew, and why this is a directory rather than a tree
//!
//! Token frequency is Zipfian, so blocks do not fill evenly — measured on 1M
//! tokens of Rust, the hottest key holds 76,305 grams against a median of 28,
//! and one-block-per-key fits only 24% of grams (see
//! `benchmarks/results/devlog_20260829_ngram_specdecode_replay_sweep.md`).
//!
//! The skew is absorbed by giving a hot key many blocks:
//!
//! ```text
//! directory[last_token] -> (first_block, n_blocks)   // dense array, in RAM
//! block = first_block + (fp % n_blocks)              // exactly 1 block read
//! within block: records sorted by fp, binary search  // inside the loaded page
//! ```
//!
//! The second-level hash is what keeps a hot key cheap: a key owning 449 blocks
//! still costs one read, where binary-searching across its blocks would cost
//! log2(449) ≈ 9. A B-tree would handle the same skew by splitting, but it
//! exists to search *sparse ordered* keys and to *grow*; here the top-level key
//! is a dense bounded integer (a token id), which wants an array, and the
//! budget is fixed, where the answer to a full block is eviction, not a split.
//!
//! ## Two write paths
//!
//! - [`ColdStore::insert_in_place`] — the between-tokens trickle. One block
//!   read-modify-write, one dirty page, no directory change. Fails (returns
//!   `false`) when the key owns no blocks yet; that gram waits for a merge.
//! - [`ColdStore::merge`] — the periodic rewrite. Recomputes the directory from
//!   scratch, so it is also the rebalance. Because a merge rewrites the file
//!   anyway, there is no need to pay for online rebalancing.

use std::fs::OpenOptions;
use std::io;
use std::path::Path;

use memmap2::{Mmap, MmapMut};

pub const BLOCK_BYTES: usize = 4096;
pub const RECORD_BYTES: usize = 24;
pub const BLOCK_HEADER_BYTES: usize = 16;
/// 4096 - 16 header = 4080, / 24 = 170 exactly.
pub const RECORDS_PER_BLOCK: usize = (BLOCK_BYTES - BLOCK_HEADER_BYTES) / RECORD_BYTES;

const MAGIC: u32 = 0x484e_4731; // "HNG1"
const DIR_ENTRY_BYTES: usize = 8;
/// Sentinel in `next2` meaning "no second token stored".
pub const NO_TOKEN: u32 = u32::MAX;

/// One stored n-gram.
///
/// `next2` carries a second continuation token for high orders only. Chaining
/// re-probes at full order and drifts less, so it is the better predictor —
/// but each chain step lands on a *different* block (the last token changed),
/// so from the cold tier a chained token costs another 4 KiB read. Storing the
/// second token inline halves that I/O on the orders where the measurement
/// says a second token is actually reliable (order >= 5 accepts ~3 tokens;
/// order 2 accepts 0.75, where `next2` would be noise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub fp: u64,
    pub next: u32,
    pub next2: u32,
    pub count: u16,
    /// Merge epoch when this gram was last observed. Age = current - this.
    pub epoch: u16,
    pub order: u8,
    pub flags: u8,
}

impl Record {
    fn read(b: &[u8]) -> Self {
        Self {
            fp: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            next: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            next2: u32::from_le_bytes(b[12..16].try_into().unwrap()),
            count: u16::from_le_bytes(b[16..18].try_into().unwrap()),
            epoch: u16::from_le_bytes(b[18..20].try_into().unwrap()),
            order: b[20],
            flags: b[21],
        }
    }

    fn write(&self, b: &mut [u8]) {
        b[0..8].copy_from_slice(&self.fp.to_le_bytes());
        b[8..12].copy_from_slice(&self.next.to_le_bytes());
        b[12..16].copy_from_slice(&self.next2.to_le_bytes());
        b[16..18].copy_from_slice(&self.count.to_le_bytes());
        b[18..20].copy_from_slice(&self.epoch.to_le_bytes());
        b[20] = self.order;
        b[21] = self.flags;
        b[22] = 0;
        b[23] = 0;
    }

    /// Eviction score: popularity discounted by staleness. A gram that has not
    /// been seen for many merge epochs loses to a fresher one of equal count.
    ///
    /// ponytail: linear decay, one epoch = one point. Swap for an exponential
    /// decay only if the measured age distribution says the tail matters.
    fn score(&self, now: u16) -> i32 {
        let age = now.saturating_sub(self.epoch) as i32;
        self.count as i32 - age
    }
}

/// A gram waiting to be written, produced by the hot tier's admission gate.
#[derive(Debug, Clone, Copy)]
pub struct Staged {
    /// Last token of the context — the block key.
    pub key: u32,
    pub fp: u64,
    pub next: u32,
    pub next2: u32,
    pub count: u16,
    pub order: u8,
}

/// How the file is mapped.
///
/// A read-only store maps shared and immutable, so every process that opens the
/// same base file shares one copy of its pages — which matters on a UMA part
/// where that page cache is competing with model weights. A writable store
/// costs its own copy, which is another reason to keep the big shared table
/// read-only and the per-user tables small.
enum Backing {
    Ro(Mmap),
    Rw(MmapMut),
}

impl Backing {
    #[inline]
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Ro(m) => m,
            Backing::Rw(m) => m,
        }
    }
    #[inline]
    fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Backing::Rw(m) => m,
            Backing::Ro(_) => unreachable!("write path guarded by is_read_only()"),
        }
    }
}

/// Fixed-size block store. The file is allocated in full at creation and never
/// grows; the budget *is* the file size.
pub struct ColdStore {
    map: Backing,
    vocab: usize,
    n_blocks: usize,
    /// Byte offset of block 0.
    data_off: usize,
    /// Directory mirrored in RAM: `(first_block, n_blocks)` per token.
    dir: Vec<(u32, u32)>,
    epoch: u16,
    /// Bump allocator over blocks never yet handed to a key.
    ///
    /// Without this a fresh store cannot bootstrap: every key owns zero blocks,
    /// so every `insert_in_place` fails and nothing is ever written until a
    /// merge — and a full-file merge is far too slow to run per request. Keys
    /// are sparse in practice (measured: ~14.6k distinct keys over 1M tokens of
    /// Rust, against a 150k+ vocab), so handing out one block on first write
    /// covers the common case with a single dirty page and no rebalance.
    next_free: u32,
}

fn dir_bytes(vocab: usize) -> usize {
    (vocab * DIR_ENTRY_BYTES).div_ceil(BLOCK_BYTES) * BLOCK_BYTES
}

impl ColdStore {
    /// Create (or truncate) a store of `n_blocks` 4 KiB blocks.
    pub fn create(path: &Path, vocab: usize, n_blocks: usize) -> io::Result<Self> {
        let total = BLOCK_BYTES + dir_bytes(vocab) + n_blocks * BLOCK_BYTES;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total as u64)?;
        let mut mm = unsafe { MmapMut::map_mut(&file)? };
        mm[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        mm[4..8].copy_from_slice(&(vocab as u32).to_le_bytes());
        mm[8..12].copy_from_slice(&(n_blocks as u32).to_le_bytes());
        mm[12..14].copy_from_slice(&0u16.to_le_bytes()); // epoch
        mm[16..20].copy_from_slice(&0u32.to_le_bytes()); // next_free
        Ok(Self {
            map: Backing::Rw(mm),
            vocab,
            n_blocks,
            data_off: BLOCK_BYTES + dir_bytes(vocab),
            dir: vec![(0, 0); vocab],
            epoch: 0,
            next_free: 0,
        })
    }

    /// Open writable. Use for a store private to its scope (a user's own).
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Self::from_backing(Backing::Rw(unsafe { MmapMut::map_mut(&file)? }))
    }

    /// Open read-only.
    ///
    /// **Any store reachable by more than one user must be opened this way.**
    /// `next`/`next2` are stored in plaintext and `fingerprint` is a fixed
    /// public function, so a writable shared store is a continuation oracle:
    /// guess a context, hash it, and read what followed it. Read-only keeps a
    /// shared table to content that was deliberately published into it.
    pub fn open_read_only(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        Self::from_backing(Backing::Ro(unsafe { Mmap::map(&file)? }))
    }

    fn from_backing(map: Backing) -> io::Result<Self> {
        let bytes = map.bytes();
        if bytes.len() < BLOCK_BYTES || u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != MAGIC
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an hng1 store",
            ));
        }
        let vocab = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let n_blocks = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let epoch = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
        let next_free = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let data_off = BLOCK_BYTES + dir_bytes(vocab);
        let expect = data_off + n_blocks * BLOCK_BYTES;
        if bytes.len() < expect {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store truncated",
            ));
        }
        // Pull the directory into RAM: vocab * 8B, ~1.2 MB at a 151k vocab.
        let mut dir = Vec::with_capacity(vocab);
        for t in 0..vocab {
            let o = BLOCK_BYTES + t * DIR_ENTRY_BYTES;
            dir.push((
                u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap()),
            ));
        }
        Ok(Self {
            map,
            vocab,
            n_blocks,
            data_off,
            dir,
            epoch,
            next_free,
        })
    }

    #[inline]
    pub fn is_read_only(&self) -> bool {
        matches!(self.map, Backing::Ro(_))
    }

    pub fn epoch(&self) -> u16 {
        self.epoch
    }
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    #[inline]
    fn block_of(&self, key: u32, fp: u64) -> Option<usize> {
        let (first, n) = *self.dir.get(key as usize)?;
        if n == 0 {
            return None;
        }
        Some(first as usize + (fp % n as u64) as usize)
    }

    #[inline]
    fn block_slice(&self, b: usize) -> &[u8] {
        let o = self.data_off + b * BLOCK_BYTES;
        &self.map.bytes()[o..o + BLOCK_BYTES]
    }

    #[inline]
    fn block_slice_mut(&mut self, b: usize) -> &mut [u8] {
        let o = self.data_off + b * BLOCK_BYTES;
        &mut self.map.bytes_mut()[o..o + BLOCK_BYTES]
    }

    #[inline]
    fn block_len(blk: &[u8]) -> usize {
        (u16::from_le_bytes(blk[4..6].try_into().unwrap()) as usize).min(RECORDS_PER_BLOCK)
    }

    #[inline]
    fn rec_at(blk: &[u8], i: usize) -> Record {
        let o = BLOCK_HEADER_BYTES + i * RECORD_BYTES;
        Record::read(&blk[o..o + RECORD_BYTES])
    }

    /// Point lookup. One block read, then a binary search inside that block.
    pub fn lookup(&self, key: u32, fp: u64) -> Option<Record> {
        let b = self.block_of(key, fp)?;
        let blk = self.block_slice(b);
        let n = Self::block_len(blk);
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let r = Self::rec_at(blk, mid);
            match r.fp.cmp(&fp) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(r),
            }
        }
        None
    }

    /// Write one gram, touching a single block. Returns false when the key owns
    /// no blocks (nothing to write into until the next merge).
    ///
    /// If the gram is already present its count is bumped in place. If the
    /// block is full, the lowest-scoring resident is evicted — but only if the
    /// newcomer actually outranks it, so a busy block cannot be churned by a
    /// stream of one-off grams.
    pub fn insert_in_place(&mut self, s: &Staged) -> bool {
        if self.is_read_only() {
            return false;
        }
        let b = match self.block_of(s.key, s.fp) {
            Some(b) => b,
            // Key owns nothing yet: hand it one free block if any remain.
            // When the file is full the gram waits for a merge, which is the
            // only thing that can take space from another key.
            None => match self.alloc_block(s.key) {
                Some(b) => b,
                None => return false,
            },
        };
        let epoch = self.epoch;
        let blk = self.block_slice_mut(b);
        let n = Self::block_len(blk);

        // Sorted position for fp.
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if Self::rec_at(blk, mid).fp < s.fp {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Already present: bump count, refresh epoch, keep the better next2.
        if lo < n {
            let mut r = Self::rec_at(blk, lo);
            if r.fp == s.fp {
                r.count = r.count.saturating_add(s.count.max(1));
                r.epoch = epoch;
                if r.next2 == NO_TOKEN {
                    r.next2 = s.next2;
                }
                let o = BLOCK_HEADER_BYTES + lo * RECORD_BYTES;
                r.write(&mut blk[o..o + RECORD_BYTES]);
                return true;
            }
        }

        let new = Record {
            fp: s.fp,
            next: s.next,
            next2: s.next2,
            count: s.count.max(1),
            epoch,
            order: s.order,
            flags: 0,
        };

        if n < RECORDS_PER_BLOCK {
            // Shift the tail right by one record and drop it in.
            let from = BLOCK_HEADER_BYTES + lo * RECORD_BYTES;
            let to = from + RECORD_BYTES;
            let end = BLOCK_HEADER_BYTES + n * RECORD_BYTES;
            blk.copy_within(from..end, to);
            new.write(&mut blk[from..from + RECORD_BYTES]);
            blk[4..6].copy_from_slice(&((n + 1) as u16).to_le_bytes());
            return true;
        }

        // Full: evict the weakest resident, but only if the newcomer wins.
        let mut worst = 0usize;
        let mut worst_score = i32::MAX;
        for i in 0..n {
            let sc = Self::rec_at(blk, i).score(epoch);
            if sc < worst_score {
                worst_score = sc;
                worst = i;
            }
        }
        if new.score(epoch) <= worst_score {
            return false;
        }
        // Remove `worst`, then insert at the sorted position.
        let hdr = BLOCK_HEADER_BYTES;
        let wf = hdr + worst * RECORD_BYTES;
        let end = hdr + n * RECORD_BYTES;
        blk.copy_within(wf + RECORD_BYTES..end, wf);
        let ins = if worst < lo { lo - 1 } else { lo };
        let f = hdr + ins * RECORD_BYTES;
        blk.copy_within(f..end - RECORD_BYTES, f + RECORD_BYTES);
        new.write(&mut blk[f..f + RECORD_BYTES]);
        true
    }

    /// Give `key` its first block, if the store has one spare.
    fn alloc_block(&mut self, key: u32) -> Option<usize> {
        if (key as usize) >= self.vocab || (self.next_free as usize) >= self.n_blocks {
            return None;
        }
        let b = self.next_free;
        self.next_free += 1;
        self.dir[key as usize] = (b, 1);
        // Persist directory entry + allocator cursor so a reopen sees them.
        let o = BLOCK_BYTES + key as usize * DIR_ENTRY_BYTES;
        let nf = self.next_free;
        let m = self.map.bytes_mut();
        m[o..o + 4].copy_from_slice(&b.to_le_bytes());
        m[o + 4..o + 8].copy_from_slice(&1u32.to_le_bytes());
        m[16..20].copy_from_slice(&nf.to_le_bytes());
        Some(b as usize)
    }

    /// Blocks not yet handed to any key.
    pub fn free_blocks(&self) -> usize {
        self.n_blocks.saturating_sub(self.next_free as usize)
    }

    /// Rewrite the whole file from its current contents plus `staged`, and
    /// recompute the directory. This is the rebalance: block counts are
    /// reassigned in proportion to each key's population, so a key that has
    /// grown hot since the last merge gets the blocks it now needs.
    ///
    /// When the store cannot hold everything, the globally lowest-scoring grams
    /// are dropped — a fixed budget discards rather than grows.
    pub fn merge(&mut self, staged: &[Staged]) {
        if self.is_read_only() {
            return;
        }
        let epoch = self.epoch.wrapping_add(1);

        // Gather everything as (key, record).
        let mut all: Vec<(u32, Record)> = Vec::new();
        for t in 0..self.vocab {
            let (first, n) = self.dir[t];
            for b in first as usize..first as usize + n as usize {
                let blk = self.block_slice(b);
                let len = Self::block_len(blk);
                for i in 0..len {
                    all.push((t as u32, Self::rec_at(blk, i)));
                }
            }
        }
        for s in staged {
            all.push((
                s.key,
                Record {
                    fp: s.fp,
                    next: s.next,
                    next2: s.next2,
                    count: s.count.max(1),
                    epoch,
                    order: s.order,
                    flags: 0,
                },
            ));
        }

        // Fold duplicates (a staged gram that already lives on disk).
        all.sort_unstable_by(|a, b| (a.0, a.1.fp).cmp(&(b.0, b.1.fp)));
        let mut folded: Vec<(u32, Record)> = Vec::with_capacity(all.len());
        for (k, r) in all {
            match folded.last_mut() {
                Some((pk, pr)) if *pk == k && pr.fp == r.fp => {
                    pr.count = pr.count.saturating_add(r.count);
                    pr.epoch = pr.epoch.max(r.epoch);
                    if pr.next2 == NO_TOKEN {
                        pr.next2 = r.next2;
                    }
                }
                _ => folded.push((k, r)),
            }
        }

        // Budget: keep a little slack per block so in-place inserts between
        // merges have somewhere to land.
        let target_fill = RECORDS_PER_BLOCK * 3 / 4;
        let capacity = self.n_blocks * target_fill;
        if folded.len() > capacity {
            folded.sort_unstable_by(|a, b| b.1.score(epoch).cmp(&a.1.score(epoch)));
            folded.truncate(capacity);
            folded.sort_unstable_by(|a, b| (a.0, a.1.fp).cmp(&(b.0, b.1.fp)));
        }

        // Blocks per key, proportional to population.
        let mut per_key: Vec<(u32, usize, usize)> = Vec::new(); // (key, start_idx, len)
        {
            let mut i = 0usize;
            while i < folded.len() {
                let k = folded[i].0;
                let start = i;
                while i < folded.len() && folded[i].0 == k {
                    i += 1;
                }
                per_key.push((k, start, i - start));
            }
        }
        let mut want: usize = per_key
            .iter()
            .map(|(_, _, n)| n.div_ceil(target_fill))
            .sum();
        // Should not happen after the truncate above, but stay safe.
        while want > self.n_blocks {
            per_key.pop();
            want = per_key
                .iter()
                .map(|(_, _, n)| n.div_ceil(target_fill))
                .sum();
        }

        // Zero the data region, then lay keys out contiguously.
        let data_off = self.data_off;
        let data_len = self.n_blocks * BLOCK_BYTES;
        self.map.bytes_mut()[data_off..data_off + data_len].fill(0);
        self.dir.clear();
        self.dir.resize(self.vocab, (0, 0));

        let mut next_block = 0usize;
        for (k, start, len) in &per_key {
            let nb = len.div_ceil(target_fill).max(1);
            let first = next_block;
            next_block += nb;
            self.dir[*k as usize] = (first as u32, nb as u32);

            // Distribute by the same second-level hash lookups will use, so a
            // record always lands in the block a reader will probe.
            let mut buckets: Vec<Vec<Record>> = vec![Vec::new(); nb];
            for (_, r) in &folded[*start..*start + *len] {
                buckets[(r.fp % nb as u64) as usize].push(*r);
            }
            for (bi, mut recs) in buckets.into_iter().enumerate() {
                recs.sort_unstable_by_key(|r| r.fp);
                // A hash bucket can still overflow a block; keep the strongest.
                if recs.len() > RECORDS_PER_BLOCK {
                    recs.sort_unstable_by(|a, b| b.score(epoch).cmp(&a.score(epoch)));
                    recs.truncate(RECORDS_PER_BLOCK);
                    recs.sort_unstable_by_key(|r| r.fp);
                }
                let b = first + bi;
                let blk = self.block_slice_mut(b);
                blk[4..6].copy_from_slice(&(recs.len() as u16).to_le_bytes());
                for (i, r) in recs.iter().enumerate() {
                    let o = BLOCK_HEADER_BYTES + i * RECORD_BYTES;
                    r.write(&mut blk[o..o + RECORD_BYTES]);
                }
            }
        }

        // Persist the directory and the epoch.
        for t in 0..self.vocab {
            let (f, n) = self.dir[t];
            let o = BLOCK_BYTES + t * DIR_ENTRY_BYTES;
            self.map.bytes_mut()[o..o + 4].copy_from_slice(&f.to_le_bytes());
            self.map.bytes_mut()[o + 4..o + 8].copy_from_slice(&n.to_le_bytes());
        }
        self.epoch = epoch;
        self.next_free = next_block as u32;
        let nf = self.next_free;
        self.map.bytes_mut()[12..14].copy_from_slice(&epoch.to_le_bytes());
        self.map.bytes_mut()[16..20].copy_from_slice(&nf.to_le_bytes());
    }

    pub fn flush(&self) -> io::Result<()> {
        match &self.map {
            Backing::Rw(m) => m.flush(),
            Backing::Ro(_) => Ok(()),
        }
    }

    /// (records, blocks in use) — for the occupancy telemetry.
    pub fn occupancy(&self) -> (usize, usize) {
        let mut recs = 0usize;
        let mut blocks = 0usize;
        for t in 0..self.vocab {
            let (first, n) = self.dir[t];
            for b in first as usize..first as usize + n as usize {
                blocks += 1;
                recs += Self::block_len(self.block_slice(b));
            }
        }
        (recs, blocks)
    }
}
