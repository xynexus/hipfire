// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Drafter-free n-gram speculative decode.
//!
//! Produces a draft spine from token statistics alone — no drafter model, no
//! sidecar weights, no GPU. The intended wiring is **n-gram first, drafter on
//! miss**: [`NgramSpec::draft`] returns a spine to hand to the existing
//! `pld_spine` seam in `spec_step_dflash`, and `None` falls through to the
//! DFlash path unchanged.
//!
//! Unlike the sibling strategy crates (dflash, ddtree, mtp, dspark) this one is
//! **not** generic over `SpecDecodeTarget` and pulls in no GPU dependencies: an
//! n-gram drafter never runs a model, so it needs no view onto one.
//!
//! ## Tiers
//!
//! Probed most-specific first, because a gram observed in this session beats a
//! stale one from a shared corpus:
//!
//! | tier | scope | writes | source |
//! |---|---|---|---|
//! | [`Tier::Hot`] | session | RAM | this request |
//! | [`Tier::User`] | one user | serving | that user's traffic |
//! | [`Tier::Topic`] | one subject | serving *or* read-only | a session type, e.g. `python-coding` |
//! | [`Tier::Base`] | one tokenizer | never | offline, published corpus |
//!
//! ## Scope is keyed on the tokenizer, not the model
//!
//! Records are token ids, which only mean anything inside one tokenizer. Every
//! quant variant of one base shares a table; Qwen and Llama never can.
//!
//! ## Sharing a writable table across users leaks their text
//!
//! `next`/`next2` are stored in plaintext and [`fingerprint`] is a fixed public
//! function, so a table reachable by two users is a continuation oracle: guess
//! a context, hash it, read what followed, append, repeat. That is a text- and
//! credential-extraction path, so anything shared is opened **read-only**
//! ([`ColdStore::open_read_only`]) and only ever holds content deliberately
//! published into it. A per-user table stays writable because it is private.
//!
//! ## Defaults come from measurement
//!
//! The defaults in [`NgramConfig`] are the operating point from
//! `benchmarks/results/devlog_20260829_ngram_specdecode_replay_sweep.md`
//! (1M tokens of Rust, CPU acceptance oracle): orders 2..8, no count threshold
//! on drafting, `max_spine = 16`, and a chain floor of 8.
//!
//! The chain floor is the load-bearing knob. Without it the chain never
//! terminates — it pads to `max_spine` and burns verify width:
//!
//! | chain floor | accepted/step | drafted/step | verify efficiency |
//! |---|---|---|---|
//! | 0 | 2.81 | 16.00 | 20.6% |
//! | 6 | 2.48 | 9.03 | 32.3% |
//! | 8 | 2.23 | 6.94 | 37.9% |
//!
//! Lower it toward 0 if verify width is cheap on your part; raise it if the
//! target forward is bandwidth-bound and wasted draft slots cost real time.

pub mod cold;
pub mod hot;

use std::io;
use std::path::{Path, PathBuf};

use cold::{ColdStore, Staged, NO_TOKEN};
use hot::HotTable;

/// Which tier served a drafted token.
///
/// Recorded per position so acceptance can be attributed back. Tiers are
/// probed in this order and the first hit wins, so a hit at one tier is by
/// construction a gram no more-specific tier had — which makes the per-tier
/// acceptance counts a direct measure of each tier's marginal value on live
/// traffic, with no second experiment needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Hot = 0,
    User = 1,
    Topic = 2,
    Base = 3,
}

impl Tier {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    pub const ALL: [Tier; 4] = [Tier::Hot, Tier::User, Tier::Topic, Tier::Base];
    pub fn name(self) -> &'static str {
        match self {
            Tier::Hot => "hot",
            Tier::User => "user",
            Tier::Topic => "topic",
            Tier::Base => "base",
        }
    }
}

/// Which store the write path feeds. Only a private store may be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteTarget {
    /// The per-user store (the default; it is private by construction).
    User,
    /// The topic store — only legitimate when that store is private to this
    /// user. A shared topic store is opened read-only and will refuse writes.
    Topic,
    /// Learn nothing durable; hot tier only.
    None,
}

#[derive(Debug, Clone)]
pub struct NgramConfig {
    /// Probe orders, longest first.
    pub orders: Vec<u8>,
    /// Minimum count to draft from. 1 is measured-optimal; the threshold that
    /// matters is `promote_count`, which gates disk writes.
    pub min_count: u16,
    /// Count at which a gram is worth persisting.
    pub promote_count: u16,
    pub max_spine: usize,
    /// After the first drafted token, only keep extending while the winning
    /// order is at least this. 0 disables the gate.
    pub chain_floor: u8,
    /// Hot-tier entry budget.
    pub hot_capacity: usize,
    /// Grams flushed to the writable store per committed token.
    pub flush_per_token: usize,
    /// Orders at or above this store a second continuation token inline,
    /// halving cold-tier I/O for a chained pair. Below it the second token is
    /// noise (order 2 accepts 0.75 tokens on average).
    pub next2_min_order: u8,
    pub write_target: WriteTarget,
    /// Suppress drafting from a gram whose MEASURED acceptance is confidently
    /// poor. `0.0` disables the gate entirely (the default) — it exists because
    /// observation count is a weak proxy for acceptance and, at the top end, an
    /// actively misleading one.
    ///
    /// Replay sweep, 400k tokens of this repo's Rust (`ngram_replay_sweep`):
    ///
    /// | gram count at hit | proposals | precision |
    /// |---|---|---|
    /// | 2 | 105555 | 57.0% |
    /// | 9-16 | 94700 | **62.9%** |
    /// | 17+ | 293012 | **56.7%** |
    ///
    /// Precision peaks at 9-16 and FALLS at 17+, below even count-2 — and that
    /// bucket is 293k proposals, the largest of the five. A gram seen 500 times
    /// with many distinct continuations is frequent and unpredictable, and no
    /// count threshold can tell it from a frequent, reliable one. Outcome can.
    ///
    /// Compared against a lower confidence bound, not the raw rate, so a gram
    /// that went 0-for-1 is not condemned on one sample. See
    /// [`acceptance_lower_bound`].
    ///
    /// ## Measured (`ngram_replay_sweep`, 400k tokens of this repo's Rust)
    ///
    /// `--orders 8..2 --max-spine 16 --chain-floor 8 --min-count 1`, gate over
    /// `>= 8` proposals. The control arm reproduces the documented operating
    /// point (2.10 accepted/step vs the 1M-token table's 2.23):
    ///
    /// | min_acceptance | accepted/step | drafted/step | verify eff |
    /// |---|---|---|---|
    /// | off | 2.10 | 6.31 | 33.3% |
    /// | 0.05 | 1.93 | 4.08 | 47.3% |
    /// | 0.10 | 1.89 | 3.92 | 48.1% |
    /// | 0.15 | 1.82 | 3.78 | 48.3% |
    /// | **0.20** | **1.81** | **3.74** | **48.5%** |
    /// | 0.30 | 1.74 | 3.58 | 48.5% |
    /// | 0.50 | 1.58 | 3.32 | 47.5% |
    ///
    /// The trade is **-14% accepted for -41% drafted**, and verify efficiency
    /// rises 33.3% -> 48.5%. Note it reaches operating points `chain_floor`
    /// cannot: the floor is already at the maximum order and bottoms out at
    /// 6.31 drafted/step, so this opens a region of the curve that was not
    /// previously reachable.
    ///
    /// 0.20 is the knee — 0.30 matches its efficiency while accepting less, so
    /// it is dominated, and everything above 0.30 gives efficiency back.
    ///
    /// ## The SHIPPED implementation, measured by `replay_real`
    ///
    /// The table above is the simplified sweep. `replay_real` drives THIS crate
    /// over the same corpus, and its control reproduces the published 1M-token
    /// operating point exactly (2.23 accepted/decode step):
    ///
    /// | min_acceptance | accepted/decode step | drafted/proposing | verify eff |
    /// |---|---|---|---|
    /// | off | 2.23 | 10.57 | 26.7% |
    /// | 0.05 | 1.84 | 6.33 | 38.5% |
    /// | 0.10 | 1.68 | 5.91 | 38.6% |
    /// | 0.15 | 1.63 | 5.73 | 39.3% |
    /// | 0.20 | 1.57 | 5.53 | 39.4% |
    ///
    /// **Trust these over the sweep's.** The trade is steeper here — **-30%
    /// accepted for -48% drafted** at 0.20, against the sweep's -14%/-41% —
    /// because the sweep models neither the inline `next2` pair nor the tier
    /// ladder, so it under-counts both what a gram offers and what suppressing
    /// it costs. The direction and the rough magnitude agree; the exact
    /// exchange rate does not, and the shipped path is the one that matters.
    ///
    /// The knee is flatter than the sweep suggested: 0.05 already captures most
    /// of the efficiency (38.5% of the eventual 39.4%) while giving up far less
    /// acceptance (1.84 vs 1.57). If this is ever defaulted on, **start at 0.05,
    /// not 0.20.**
    ///
    /// **It is still not defaulted on, and no replay can decide that.**
    /// Whether -14% accepted is worth -41% drafted depends on what a wasted
    /// draft slot costs on the part, which is exactly the axis `chain_floor`'s
    /// own note describes ("lower it if verify width is cheap; raise it if the
    /// target forward is bandwidth-bound"). That needs a GPU tok/s comparison
    /// on a real model, not a CPU acceptance oracle.
    pub min_acceptance: f32,
    /// Proposals a gram needs before `min_acceptance` may suppress it. Below
    /// this the bound is too wide to act on and the gram drafts normally.
    pub min_acceptance_proposals: u32,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            orders: vec![8, 7, 6, 5, 4, 3, 2],
            min_count: 1,
            promote_count: 3,
            max_spine: 16,
            chain_floor: 8,
            hot_capacity: 1 << 20,
            flush_per_token: 4,
            next2_min_order: 5,
            write_target: WriteTarget::User,
            // Off by default: this changes what gets drafted, so it ships
            // measurable-but-inert until a sweep says what the bar should be.
            min_acceptance: 0.0,
            min_acceptance_proposals: 8,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct NgramStats {
    pub steps: u64,
    pub steps_proposed: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub drafted_by_tier: [u64; 4],
    pub accepted_by_tier: [u64; 4],
    pub lookups_by_tier: [u64; 4],
    pub hits_by_tier: [u64; 4],
    /// Drafted / accepted counts bucketed by POSITION in the spine.
    ///
    /// This is the observable that decides both `min_acceptance` and
    /// `chain_floor`, and it is pure counts — no clock, so it replays
    /// identically. See [`NgramStats::marginal_acceptance`].
    pub drafted_by_depth: [u64; MAX_TRACKED_DEPTH],
    pub accepted_by_depth: [u64; MAX_TRACKED_DEPTH],
}

/// Spine depths tracked for the marginal-acceptance curve. Beyond this the
/// counts fold into the last bucket; `max_spine` defaults to 16.
pub const MAX_TRACKED_DEPTH: usize = 16;

impl NgramStats {
    pub fn coverage(&self) -> f64 {
        self.steps_proposed as f64 / self.steps.max(1) as f64
    }
    pub fn accepted_per_step(&self) -> f64 {
        self.accepted as f64 / self.steps.max(1) as f64
    }
    /// Accepted / drafted. Every point below 100% is verify width spent on a
    /// token the target rejected.
    pub fn verify_efficiency(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
    pub fn accepted_in(&self, t: Tier) -> u64 {
        self.accepted_by_tier[t.idx()]
    }
    pub fn drafted_in(&self, t: Tier) -> u64 {
        self.drafted_by_tier[t.idx()]
    }
    /// Share of accepted tokens only this tier could supply — its marginal
    /// value, since a more-specific tier would have won the probe otherwise.
    /// P(accepted | drafted) at spine position `depth`.
    ///
    /// ## Why this is the number that matters
    ///
    /// Drafting one more slot costs the marginal compute of a wider verify,
    /// `c1`; an accepted token saves a whole AR step, `c0 + c1` (the weight read
    /// dominates `c0`). So the slot is worth drafting exactly when
    ///
    /// ```text
    /// P(accept) * (c0 + c1) > c1        i.e.   P(accept) > c1 / (c0 + c1)
    /// ```
    ///
    /// The right-hand side is dimensionless and is a property of the part and
    /// model — the marginal cost of one verify slot as a fraction of a full AR
    /// step — not of the moment. Call it `r`. Then:
    ///
    /// - `min_acceptance` **is** `r`. That is the whole derivation.
    /// - the optimal `chain_floor` is the largest depth whose marginal
    ///   acceptance still exceeds `r`.
    ///
    /// Both knobs are approximations of one rule: stop drafting when the
    /// marginal acceptance falls below the marginal slot cost.
    ///
    /// **Choosing the width is not this type's job.**
    /// `hipfire_specdecode_dspark::dspark_block_controller::BlockController`
    /// already does it, better, and is explicitly drafter-agnostic — it argmaxes
    /// `tau(N) / (t_ar + (N-1)*dt)`, i.e. committed tokens per window wall-time,
    /// which is tok/s directly rather than the greedy marginal rule this curve
    /// would support. The n-gram drafter feeds it from `generate.rs` like DFlash
    /// does. This curve stays as a DIAGNOSTIC: it is what showed depth 1 beating
    /// depth 0, which is a property of the store (the inline `next2` pair) that
    /// a width controller has no reason to surface.
    ///
    /// ## Measured curve (`replay_real`, 400k tokens of this repo's Rust)
    ///
    /// | depth | drafted | P(accept) |
    /// |---|---|---|
    /// | 0 | 316909 | 60.1% |
    /// | 1 | 160432 | **60.8%** |
    /// | 2 | 132558 | 54.6% |
    /// | 4 | 132487 | 40.6% |
    /// | 8 | 131770 | 24.8% |
    /// | 14 | 130207 | 14.3% |
    ///
    /// Two readings, and the second matters more than the gate this was built
    /// for.
    ///
    /// **Depth 1 beats depth 0.** The second token of a stored `next2` pair is a
    /// BETTER prediction than a fresh probe, which is a direct validation of the
    /// inline-pair design — it is not merely free, it is more accurate.
    ///
    /// **The curve decays slowly, which argues for a wide draft.** Marginal
    /// acceptance is still 24.8% at depth 8 and 14.3% at depth 14, so on a
    /// bandwidth-bound part — where an extra verify slot is nearly free, which
    /// this module's header asserts is the usual case — the throughput-optimal
    /// move is to draft MORE, not less. That is the opposite direction to
    /// `min_acceptance`, which is only the right tool where a slot is genuinely
    /// dear. `BlockController` decides that from live window timing; this curve
    /// only explains why its answer is usually "wide".
    ///
    /// Deliberately NOT derived from a clock. A wall-clock term would make the
    /// bar depend on thermals and on which other requests share the batch, and
    /// spec-decode output is not yet provably identical to AR decode here (open
    /// entry in BUGS.md) — so a timing-dependent bar risks non-reproducible
    /// TEXT, not merely non-reproducible speed. Counts replay exactly.
    pub fn marginal_acceptance(&self, depth: usize) -> f64 {
        let d = depth.min(MAX_TRACKED_DEPTH - 1);
        let drafted = self.drafted_by_depth[d];
        if drafted == 0 {
            return 0.0;
        }
        self.accepted_by_depth[d] as f64 / drafted as f64
    }

    pub fn marginal_share(&self, t: Tier) -> f64 {
        self.accepted_by_tier[t.idx()] as f64 / self.accepted.max(1) as f64
    }
}

/// splitmix64 over `(order, context)`.
///
/// Only the fingerprint is stored — never the context tokens. A collision
/// yields a wrong draft token, which the target rejects, so correctness never
/// depends on it and the store needs no key comparison or collision chain.
#[inline]
pub fn fingerprint(order: u8, toks: &[u32]) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ (order as u64).wrapping_mul(0xff51_afd7_ed55_8ccd);
    for &t in toks {
        h ^= t as u64;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 31;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 29;
    }
    h
}

/// Reduce a caller-supplied label (a user id, a session type like
/// `"cold-fusion research"`) to a filesystem-safe slug.
///
/// These arrive from request data and are used to build paths, so this is a
/// trust boundary: everything outside `[a-z0-9-]` is replaced, so `..` and `/`
/// cannot survive and no input can escape its scope directory. Returns `None`
/// when nothing usable is left, which callers must treat as "no such scope"
/// rather than falling back to a shared one.
pub fn slug(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len().min(64));
    let mut last_dash = true; // suppresses a leading dash
    for c in s.chars() {
        let c = c.to_ascii_lowercase();
        let ok = c.is_ascii_lowercase() || c.is_ascii_digit();
        if ok {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
        if out.len() >= 64 {
            break;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// The identity of a decoding session, for picking tables.
///
/// `session_type` is a free-form subject label ("python-coding",
/// "cold-fusion-research"). It selects a topic table here, and is deliberately
/// a plain named scope so the same key can later select other per-subject
/// artifacts.
#[derive(Debug, Clone, Default)]
pub struct SessionScope {
    /// Identifies the tokenizer, not the model — every quant variant of one
    /// base shares a table.
    pub tokenizer_id: String,
    pub user_id: Option<String>,
    pub session_type: Option<String>,
}

/// Where the per-scope tables live on disk.
pub struct ScopeLayout {
    root: PathBuf,
}

impl ScopeLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Shared, read-only, per tokenizer.
    pub fn base(&self, tokenizer_id: &str) -> Option<PathBuf> {
        let t = slug(tokenizer_id)?;
        Some(self.root.join("base").join(format!("{t}.ngram")))
    }

    /// Per subject. Private when it lives under a user, shared when it does
    /// not — and a shared one must be opened read-only.
    pub fn topic(&self, tokenizer_id: &str, topic: &str, user_id: Option<&str>) -> Option<PathBuf> {
        let t = slug(tokenizer_id)?;
        let s = slug(topic)?;
        Some(match user_id.and_then(slug) {
            Some(u) => self
                .root
                .join("user")
                .join(u)
                .join(&t)
                .join(format!("{s}.ngram")),
            None => self.root.join("topic").join(&t).join(format!("{s}.ngram")),
        })
    }

    /// Private, writable, per user.
    pub fn user(&self, tokenizer_id: &str, user_id: &str) -> Option<PathBuf> {
        let t = slug(tokenizer_id)?;
        let u = slug(user_id)?;
        Some(self.root.join("user").join(u).join(format!("{t}.ngram")))
    }
}

pub struct NgramSpec {
    cfg: NgramConfig,
    hot: HotTable,
    user: Option<ColdStore>,
    topic: Option<ColdStore>,
    base: Option<ColdStore>,
    scope: SessionScope,
    /// Rolling committed-token history. Only the last `max_order + 2` matter.
    hist: Vec<u32>,
    /// Reusable draft buffer, and the tier that served each position.
    spine: Vec<u32>,
    tiers: Vec<Tier>,
    /// The gram that produced each spine position, parallel to `tiers`.
    ///
    /// A gram carrying an inline `next2` pushes TWO positions, so the same
    /// `(key, fp)` appears twice here — deliberately. Those are two independent
    /// chances for the target to accept, and collapsing them would under-count
    /// the proposals of exactly the grams that offer the most.
    spine_grams: Vec<(u32, u64)>,
    /// Measured outcome per gram. Bounded by `outcomes_cap`; see `note_outcome`.
    outcomes: std::collections::HashMap<(u32, u64), GramOutcome>,
    outcomes_cap: usize,
    /// Hard cap on drafted TOKENS when a throughput controller is driving.
    /// See [`set_spine_token_cap`](NgramSpec::set_spine_token_cap).
    spine_token_cap: Option<usize>,
    stats: NgramStats,
    max_order: usize,
    /// Grams the trickle path could not place — a key owns no blocks until a
    /// merge assigns it some, so on a cold (or newly hot) key the write has
    /// nowhere to go. These wait here for the next merge rather than being
    /// dropped, which is what bootstraps an empty store.
    merge_backlog: Vec<Staged>,
    merge_backlog_cap: usize,
}

/// Measured outcome for one gram: how often drafting it was proposed, and how
/// often the target kept the token.
///
/// `u32` rather than `u64` deliberately — this is per-gram and bounded by the
/// hot table's capacity, so the memory is `capacity * 8` bytes and a counter
/// that could reach 4 billion proposals for a single gram is not a real shape.
/// Saturating adds, so a pathological session cannot wrap one into a low count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GramOutcome {
    pub proposed: u32,
    pub accepted: u32,
}

impl GramOutcome {
    #[inline]
    fn observe(&mut self, accepted: bool) {
        self.proposed = self.proposed.saturating_add(1);
        if accepted {
            self.accepted = self.accepted.saturating_add(1);
        }
    }
}

/// Lower bound of a Wilson score interval on the acceptance rate.
///
/// The point estimate `accepted / proposed` is unusable as a gate: it ranks a
/// gram that went 1-for-1 above one that went 40-for-50, which is the same flaw
/// `promote_count`'s raw threshold has one level up. The lower bound of the
/// interval instead asks "what acceptance can we be confident this gram at
/// least has", so evidence must accumulate before a gram is judged either way.
///
/// Wilson rather than a Beta posterior quantile because it is a closed form —
/// this runs in the draft path, which is latency-critical.
///
/// `z = 1.96` (95%). `proposed == 0` returns 0.0: no evidence is not evidence of
/// badness, and the caller gates on `min_acceptance_proposals` before acting.
pub fn acceptance_lower_bound(o: GramOutcome) -> f32 {
    if o.proposed == 0 {
        return 0.0;
    }
    let n = o.proposed as f32;
    let p = (o.accepted as f32) / n;
    const Z: f32 = 1.96;
    let z2 = Z * Z;
    let denom = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let margin = Z * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();
    ((centre - margin) / denom).max(0.0)
}

impl NgramSpec {
    pub fn new(cfg: NgramConfig) -> Self {
        let mut cfg = cfg;
        cfg.orders.sort_by(|a, b| b.cmp(a)); // longest first
        let max_order = *cfg.orders.iter().max().unwrap_or(&2) as usize;
        let hot = HotTable::new(cfg.hot_capacity, cfg.promote_count);
        let cfg_hot_capacity = cfg.hot_capacity;
        Self {
            cfg,
            hot,
            user: None,
            topic: None,
            base: None,
            scope: SessionScope::default(),
            hist: Vec::with_capacity(64),
            spine: Vec::new(),
            tiers: Vec::new(),
            spine_grams: Vec::new(),
            outcomes: std::collections::HashMap::new(),
            // Same order as the hot table: the ledger is only useful for grams
            // recent enough to draft again, so it is pointless for it to
            // outlive them.
            outcomes_cap: cfg_hot_capacity,
            spine_token_cap: None,
            stats: NgramStats::default(),
            max_order,
            merge_backlog: Vec::new(),
            merge_backlog_cap: 1 << 21,
        }
    }

    /// Cap the drafted spine in TOKENS, so an external throughput controller can
    /// drive the width.
    ///
    /// `BlockController` owns that decision (see `marginal_acceptance`); this is
    /// the seam it drives through.
    ///
    /// **This is not `cfg.max_spine`, and the difference is load-bearing.**
    /// `max_spine` bounds probe STEPS, and a step carrying an inline `next2`
    /// pushes two tokens — so a `max_spine` of 16 drafts up to 24. A controller
    /// setting `max_spine` from `block()` would therefore be steering a quantity
    /// roughly double what it believes it is steering, and its cost model would
    /// be fitted against the wrong width. This cap counts tokens.
    ///
    /// `None` (the default) leaves drafting exactly as it was, so the measured
    /// baselines still hold; only a driven session narrows.
    ///
    /// Clamped to at least 1 — a zero cap would silently stop drafting rather
    /// than narrow it, which is invisible from outside.
    pub fn set_spine_token_cap(&mut self, n: Option<usize>) {
        self.spine_token_cap = n.map(|v| v.max(1));
    }

    /// The active token cap, if a controller has set one.
    pub fn spine_token_cap(&self) -> Option<usize> {
        self.spine_token_cap
    }

    /// `cfg.max_spine` — a bound on probe STEPS, not on drafted tokens.
    pub fn max_spine_steps(&self) -> usize {
        self.cfg.max_spine
    }

    pub fn set_scope(&mut self, scope: SessionScope) {
        self.scope = scope;
    }
    pub fn scope(&self) -> &SessionScope {
        &self.scope
    }

    /// Attach the private, writable per-user store, creating it if absent.
    pub fn attach_user(&mut self, path: &Path, vocab: usize, n_blocks: usize) -> io::Result<()> {
        self.user = Some(open_or_create(path, vocab, n_blocks)?);
        Ok(())
    }

    /// Attach a topic store. `writable` is only legitimate when the path is
    /// private to this user; a shared topic table must be read-only, or it
    /// becomes a cross-user continuation oracle.
    pub fn attach_topic(
        &mut self,
        path: &Path,
        vocab: usize,
        n_blocks: usize,
        writable: bool,
    ) -> io::Result<()> {
        self.topic = Some(if writable {
            open_or_create(path, vocab, n_blocks)?
        } else {
            ColdStore::open_read_only(path)?
        });
        Ok(())
    }

    /// Attach the shared base table. Always read-only — it is reachable by
    /// every user, so nothing derived from one user's traffic may enter it.
    pub fn attach_base(&mut self, path: &Path) -> io::Result<()> {
        self.base = Some(ColdStore::open_read_only(path)?);
        Ok(())
    }

    pub fn stats(&self) -> &NgramStats {
        &self.stats
    }
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }
    pub fn store(&self, t: Tier) -> Option<&ColdStore> {
        match t {
            Tier::User => self.user.as_ref(),
            Tier::Topic => self.topic.as_ref(),
            Tier::Base => self.base.as_ref(),
            Tier::Hot => None,
        }
    }
    pub fn merge_backlog_len(&self) -> usize {
        self.merge_backlog.len()
    }

    /// The store the write path feeds, if any is both configured and writable.
    fn write_store(&mut self) -> Option<&mut ColdStore> {
        let s = match self.cfg.write_target {
            WriteTarget::User => self.user.as_mut(),
            WriteTarget::Topic => self.topic.as_mut(),
            WriteTarget::None => None,
        }?;
        if s.is_read_only() {
            return None;
        }
        Some(s)
    }

    /// Start a new sequence: drop the rolling token history so the next draft
    /// does not continue the previous prompt.
    ///
    /// The hot table is deliberately **kept**. It carries ~95% of the measured
    /// marginal value, and grams only reach disk after `promote_count`
    /// observations — clearing it per request resets every counter, so almost
    /// nothing is ever promoted and the cold tiers stay empty. Cross-user
    /// isolation does not rely on this: a request with a different scope gets a
    /// different `NgramSpec` entirely, so one user's hot state is never visible
    /// to another.
    pub fn reset_sequence(&mut self) {
        self.hist.clear();
    }

    /// Forget everything session-local, hot table included. The on-disk stores
    /// are untouched.
    pub fn reset(&mut self) {
        self.hist.clear();
        self.hot.reset();
    }

    /// Feed committed tokens — prompt at prefill, then each accepted token.
    ///
    /// Observation runs one token behind so the second continuation token is
    /// known when a gram is recorded.
    pub fn observe(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.hist.push(t);
            self.hot.tick();

            let len = self.hist.len();
            for &ord in &self.cfg.orders {
                let o = ord as usize;
                if len < o + 2 {
                    continue;
                }
                let ctx = &self.hist[len - 2 - o..len - 2];
                let key = ctx[o - 1];
                let next = self.hist[len - 2];
                let next2 = if ord >= self.cfg.next2_min_order {
                    self.hist[len - 1]
                } else {
                    NO_TOKEN
                };
                self.hot
                    .observe(key, fingerprint(ord, ctx), ord, next, next2);
            }

            // Trickle the admitted tail to disk, a few grams per token.
            let per_token = self.cfg.flush_per_token;
            let cap = self.merge_backlog_cap;
            let batch = if self.write_store().is_some() {
                self.hot.drain_pending(per_token)
            } else {
                Vec::new()
            };
            if !batch.is_empty() {
                let mut spill = Vec::new();
                if let Some(store) = self.write_store() {
                    for s in &batch {
                        // A key that owns no blocks yet cannot take an in-place
                        // write, and neither can a full block whose residents
                        // all outrank this gram. Either way it waits for a merge.
                        if !store.insert_in_place(s) {
                            spill.push(*s);
                        }
                    }
                }
                for s in spill {
                    if self.merge_backlog.len() < cap {
                        self.merge_backlog.push(s);
                    }
                }
            }

            // Keep only what probing and observation need.
            let keep = self.max_order + 2;
            if self.hist.len() > 4 * keep {
                self.hist.drain(..self.hist.len() - keep);
            }
        }
    }

    /// Look one context up across the tiers, most specific first. The first hit
    /// wins, so a hit at tier N means no tier below N had this gram.
    fn probe(&mut self, key: u32, fp: u64) -> Option<(u32, u32, Tier)> {
        // Count the lookup whether or not it hits, so hit-rate per tier is
        // comparable across tiers.
        self.stats.lookups_by_tier[Tier::Hot.idx()] += 1;
        if let Some((next, next2, count)) = self.hot.lookup(fp) {
            if count >= self.cfg.min_count {
                self.stats.hits_by_tier[Tier::Hot.idx()] += 1;
                return Some((next, next2, Tier::Hot));
            }
        }
        let min = self.cfg.min_count;
        for tier in [Tier::User, Tier::Topic, Tier::Base] {
            let store = match tier {
                Tier::User => self.user.as_ref(),
                Tier::Topic => self.topic.as_ref(),
                Tier::Base => self.base.as_ref(),
                Tier::Hot => None,
            };
            let Some(store) = store else { continue };
            self.stats.lookups_by_tier[tier.idx()] += 1;
            if let Some(r) = store.lookup(key, fp) {
                if r.count >= min {
                    self.stats.hits_by_tier[tier.idx()] += 1;
                    return Some((r.next, r.next2, tier));
                }
            }
        }
        None
    }

    /// Draft a spine for the current context, or `None` on a miss (caller falls
    /// back to the drafter model).
    ///
    /// Probing is longest-order-first. After the first token, continuation is
    /// gated on `chain_floor`: a chain that has fallen back to a low order is
    /// drifting, and every further token costs verify width for little return.
    pub fn draft(&mut self) -> Option<&[u32]> {
        self.stats.steps += 1;
        self.spine.clear();
        self.tiers.clear();
        self.spine_grams.clear();
        if self.hist.len() < 2 {
            return None;
        }

        let start = self.hist.len().saturating_sub(self.max_order);
        let mut ctx: Vec<u32> = self.hist[start..].to_vec();

        let orders = self.cfg.orders.clone();
        for step in 0..self.cfg.max_spine {
            // Token cap, when a controller is driving. Checked at the top of the
            // step because a single step can emit two tokens.
            if self
                .spine_token_cap
                .is_some_and(|cap| self.spine.len() >= cap)
            {
                break;
            }
            let mut hit = None;
            for &ord in &orders {
                let o = ord as usize;
                if ctx.len() < o {
                    continue;
                }
                if step > 0 && ord < self.cfg.chain_floor {
                    continue;
                }
                let sfx = &ctx[ctx.len() - o..];
                let key = sfx[o - 1];
                let fp = fingerprint(ord, sfx);
                // Outcome gate. Skipping CONTINUES the order loop rather than
                // breaking it, so a gram with a bad record falls through to a
                // shorter order instead of ending the spine — the shorter gram
                // is a worse prediction than a good long one, but a better one
                // than a long gram we have measured as unreliable.
                if self.suppressed(key, fp) {
                    continue;
                }
                if let Some(h) = self.probe(key, fp) {
                    hit = Some((h, key, fp));
                    break; // orders is longest-first
                }
            }
            let ((next, next2, tier), key, fp) = match hit {
                Some(h) => h,
                None => break,
            };
            self.spine.push(next);
            self.tiers.push(tier);
            self.spine_grams.push((key, fp));
            ctx.push(next);

            // A stored second token is free — it needs no further block read.
            let token_room = self
                .spine_token_cap
                .map(|cap| self.spine.len() < cap)
                .unwrap_or(true);
            if next2 != NO_TOKEN && token_room && self.spine.len() < self.cfg.max_spine {
                self.spine.push(next2);
                self.tiers.push(tier);
                self.spine_grams.push((key, fp));
                ctx.push(next2);
            }
        }

        if self.spine.is_empty() {
            return None;
        }
        self.stats.steps_proposed += 1;
        self.stats.drafted += self.spine.len() as u64;
        for t in &self.tiers {
            self.stats.drafted_by_tier[t.idx()] += 1;
        }
        Some(&self.spine)
    }

    /// Report how many leading tokens of the last [`draft`](Self::draft) the
    /// target accepted, so acceptance can be attributed per tier.
    pub fn record_acceptance(&mut self, accepted: usize) {
        let n = accepted.min(self.tiers.len());
        self.stats.accepted += n as u64;
        for t in self.tiers.iter().take(n) {
            self.stats.accepted_by_tier[t.idx()] += 1;
        }
        // Per-gram outcome. Both halves matter: crediting alone would make every
        // gram look perfect, and the rejected tail is the signal that identifies
        // a frequent-but-unpredictable context.
        for i in 0..self.spine_grams.len() {
            let g = self.spine_grams[i];
            self.note_outcome(g, i < n);
        }
        // Marginal-acceptance curve, by spine position. Counts only.
        for i in 0..self.tiers.len() {
            let d = i.min(MAX_TRACKED_DEPTH - 1);
            self.stats.drafted_by_depth[d] += 1;
            if i < n {
                self.stats.accepted_by_depth[d] += 1;
            }
        }
    }

    /// Record one proposal outcome, evicting wholesale if the ledger is full.
    ///
    /// Clearing rather than evicting one entry is deliberate: an LRU here would
    /// cost a second index and this map exists to be cheap. Losing the ledger
    /// costs nothing permanent — the gate simply stops suppressing until the
    /// counts rebuild, which is the safe direction to fail.
    fn note_outcome(&mut self, gram: (u32, u64), accepted: bool) {
        if self.outcomes.len() >= self.outcomes_cap && !self.outcomes.contains_key(&gram) {
            self.outcomes.clear();
        }
        self.outcomes.entry(gram).or_default().observe(accepted);
    }

    /// Has this gram measurably earned being skipped?
    ///
    /// False whenever the gate is off, the evidence is thin, or the gram is
    /// unknown — the gate only ever acts on grams it has actually watched fail.
    fn suppressed(&self, key: u32, fp: u64) -> bool {
        if self.cfg.min_acceptance <= 0.0 {
            return false;
        }
        match self.outcomes.get(&(key, fp)) {
            Some(&o) if o.proposed >= self.cfg.min_acceptance_proposals => {
                acceptance_lower_bound(o) < self.cfg.min_acceptance
            }
            _ => false,
        }
    }

    /// Measured outcome for one gram, for tests and offline analysis.
    pub fn outcome(&self, key: u32, fp: u64) -> Option<GramOutcome> {
        self.outcomes.get(&(key, fp)).copied()
    }

    /// Compact and rebalance the writable store, folding in everything queued.
    /// Expensive (a full file rewrite) — call between requests, not per token.
    pub fn merge(&mut self) -> io::Result<()> {
        let mut batch = std::mem::take(&mut self.merge_backlog);
        batch.extend(self.hot.take_all_pending());
        if let Some(store) = self.write_store() {
            store.merge(&batch);
            store.flush()?;
        }
        Ok(())
    }
}

fn open_or_create(path: &Path, vocab: usize, n_blocks: usize) -> io::Result<ColdStore> {
    match ColdStore::open(path) {
        Ok(s) => Ok(s),
        Err(_) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            ColdStore::create(path, vocab, n_blocks)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NgramConfig {
        NgramConfig {
            promote_count: 2,
            ..Default::default()
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hng-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A perfectly periodic stream is the case where the answer is known by
    /// construction: after two cycles every context has one unambiguous
    /// follower, so the drafter must produce a full spine and the target must
    /// accept all of it.
    #[test]
    fn periodic_stream_drafts_perfectly() {
        let mut ng = NgramSpec::new(cfg());
        let period = 32u32;
        let stream: Vec<u32> = (0..period * 8).map(|i| i % period).collect();
        ng.observe(&stream);

        let spine = ng.draft().expect("periodic stream must draft").to_vec();
        assert!(!spine.is_empty());
        let last = *stream.last().unwrap();
        for (k, &tok) in spine.iter().enumerate() {
            assert_eq!(
                tok,
                (last + 1 + k as u32) % period,
                "spine[{k}] wrong: {spine:?}"
            );
        }
    }

    #[test]
    fn random_stream_does_not_draft() {
        let mut ng = NgramSpec::new(cfg());
        let mut r = 0x1234_5678u64;
        let stream: Vec<u32> = (0..20000)
            .map(|_| {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                (r % 100_000) as u32
            })
            .collect();
        ng.observe(&stream);
        assert!(ng.draft().is_none(), "random stream produced a draft");
    }

    #[test]
    fn chain_floor_bounds_the_spine() {
        let mut ng = NgramSpec::new(NgramConfig {
            chain_floor: 9,
            next2_min_order: 99,
            ..cfg()
        });
        let stream: Vec<u32> = (0..32 * 8).map(|i| i % 32).collect();
        ng.observe(&stream);
        assert_eq!(
            ng.draft().expect("must draft").len(),
            1,
            "floor must stop the chain"
        );
    }

    #[test]
    fn cold_store_roundtrips_and_rebalances() {
        let dir = tmpdir("roundtrip");
        let path = dir.join("t.hng");
        let mut store = ColdStore::create(&path, 4096, 64).unwrap();

        // A fresh store must take writes without a merge first — otherwise it
        // can never bootstrap, because a merge is far too slow to run per
        // request. The key owns nothing yet, so this exercises the on-demand
        // block allocator.
        let s = Staged {
            key: 7,
            fp: 0xdead_beef,
            next: 11,
            next2: NO_TOKEN,
            count: 3,
            order: 4,
        };
        assert!(store.insert_in_place(&s), "fresh store must accept a write");
        assert_eq!(store.lookup(7, 0xdead_beef).unwrap().next, 11);

        let staged: Vec<Staged> = (0..500u64)
            .map(|i| Staged {
                key: (i % 16) as u32,
                fp: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                next: (i as u32) + 1,
                next2: NO_TOKEN,
                count: 3,
                order: 5,
            })
            .collect();
        store.merge(&staged);
        for s in &staged {
            assert_eq!(
                store.lookup(s.key, s.fp).expect("survives merge").next,
                s.next
            );
        }

        let extra = Staged {
            key: 3,
            fp: 0x1234_5678_9abc_def0,
            next: 99,
            next2: 100,
            count: 2,
            order: 6,
        };
        assert!(store.insert_in_place(&extra));
        store.flush().unwrap();
        drop(store);

        let re = ColdStore::open(&path).unwrap();
        assert_eq!(re.lookup(extra.key, extra.fp).unwrap().next, 99);
        assert_eq!(
            re.lookup(staged[0].key, staged[0].fp).unwrap().next,
            staged[0].next
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The serving path only ever calls `insert_in_place` (a merge rewrites the
    /// whole file and is far too slow per request), so a store that cannot
    /// bootstrap itself would silently never persist anything. This is the
    /// regression that shipped and was caught only by an end-to-end run.
    #[test]
    fn fresh_store_persists_without_any_merge() {
        let dir = tmpdir("bootstrap");
        let path = dir.join("b.hng");
        {
            let mut store = ColdStore::create(&path, 4096, 128).unwrap();
            assert_eq!(store.free_blocks(), 128);
            for i in 0..200u64 {
                let s = Staged {
                    key: (i % 24) as u32,
                    fp: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    next: i as u32 + 7,
                    next2: NO_TOKEN,
                    count: 3,
                    order: 5,
                };
                assert!(
                    store.insert_in_place(&s),
                    "write {i} rejected by a fresh store"
                );
            }
            assert!(store.free_blocks() < 128, "no blocks were allocated");
            assert!(store.occupancy().0 > 0, "nothing landed");
            store.flush().unwrap();
        }
        // Reopen: the on-demand directory entries and the allocator cursor must
        // both have been persisted, or the data is unreachable after restart.
        let re = ColdStore::open(&path).unwrap();
        assert!(
            re.free_blocks() < 128,
            "allocator cursor did not survive reopen"
        );
        for i in 0..200u64 {
            let got = re
                .lookup((i % 24) as u32, i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .unwrap_or_else(|| panic!("gram {i} lost across reopen"));
            assert_eq!(got.next, i as u32 + 7);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cold_store_evicts_rather_than_grows() {
        let dir = tmpdir("eviction");
        let path = dir.join("e.hng");
        let mut store = ColdStore::create(&path, 256, 2).unwrap();
        let staged: Vec<Staged> = (0..10_000u64)
            .map(|i| Staged {
                key: 1,
                fp: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                next: i as u32,
                next2: NO_TOKEN,
                count: (i % 500) as u16 + 1,
                order: 5,
            })
            .collect();
        store.merge(&staged);
        let (recs, blocks) = store.occupancy();
        assert!(
            blocks <= 2,
            "must not allocate past the budget, got {blocks}"
        );
        assert!(
            recs <= 2 * cold::RECORDS_PER_BLOCK,
            "must not exceed the file, got {recs}"
        );
        assert!(recs > 0, "eviction must not empty the store");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hot_promotes_only_past_the_threshold() {
        let mut ng = NgramSpec::new(NgramConfig {
            promote_count: 4,
            ..Default::default()
        });
        let stream: Vec<u32> = (0..2000).map(|i| (i * 7919) as u32 % 50_000).collect();
        ng.observe(&stream);
        assert_eq!(
            ng.hot.pending_len(),
            0,
            "one-off grams must not be staged for disk"
        );

        let mut ng2 = NgramSpec::new(NgramConfig {
            promote_count: 4,
            ..Default::default()
        });
        let cyc: Vec<u32> = (0..16 * 20).map(|i| i % 16).collect();
        ng2.observe(&cyc);
        assert!(
            ng2.hot.pending_len() > 0,
            "repeated grams must reach the write queue"
        );
    }

    /// A read-only store must refuse every write path, so a shared table can
    /// never absorb one user's text and hand it to another.
    #[test]
    fn read_only_store_refuses_writes() {
        let dir = tmpdir("readonly");
        let path = dir.join("b.hng");
        let mut store = ColdStore::create(&path, 1024, 16).unwrap();
        let staged: Vec<Staged> = (0..100u64)
            .map(|i| Staged {
                key: (i % 8) as u32,
                fp: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                next: i as u32,
                next2: NO_TOKEN,
                count: 5,
                order: 5,
            })
            .collect();
        store.merge(&staged);
        store.flush().unwrap();
        let (before, _) = store.occupancy();
        drop(store);

        let mut ro = ColdStore::open_read_only(&path).unwrap();
        assert!(ro.is_read_only());
        // Reads still work.
        assert_eq!(
            ro.lookup(staged[0].key, staged[0].fp).unwrap().next,
            staged[0].next
        );
        // Writes are refused, and change nothing.
        let intruder = Staged {
            key: 1,
            fp: 0xabcd_ef01_2345_6789,
            next: 4242,
            next2: NO_TOKEN,
            count: 99,
            order: 6,
        };
        assert!(
            !ro.insert_in_place(&intruder),
            "read-only store accepted an in-place write"
        );
        ro.merge(&[intruder]);
        assert!(
            ro.lookup(intruder.key, intruder.fp).is_none(),
            "read-only store absorbed a merge"
        );
        assert_eq!(ro.occupancy().0, before, "read-only store mutated");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The base tier must serve grams the session has never seen, and be
    /// attributed to `Tier::Base` so its value is measurable separately.
    #[test]
    fn base_tier_serves_and_is_attributed() {
        let dir = tmpdir("base");
        let path = dir.join("base.hng");

        // Build a base table from a stream, offline.
        let period = 24u32;
        let corpus: Vec<u32> = (0..period * 40).map(|i| i % period).collect();
        {
            let mut builder = NgramSpec::new(NgramConfig {
                promote_count: 2,
                ..Default::default()
            });
            builder.attach_user(&path, 1024, 256).unwrap();
            builder.observe(&corpus);
            builder.merge().unwrap();
        }

        // A fresh session with only the base attached: hot knows nothing yet,
        // so anything drafted must have come from base.
        let mut ng = NgramSpec::new(NgramConfig {
            write_target: WriteTarget::None,
            ..Default::default()
        });
        ng.attach_base(&path).unwrap();
        ng.observe(&corpus[..16]);
        let spine = ng.draft().expect("base tier must serve a draft").to_vec();
        assert!(!spine.is_empty());

        ng.record_acceptance(spine.len());
        let st = ng.stats();
        assert!(
            st.accepted_in(Tier::Base) > 0,
            "base draft not attributed to Tier::Base"
        );
        assert_eq!(st.accepted_in(Tier::User), 0);
        assert!(st.marginal_share(Tier::Base) > 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Scope labels come from request data, so they must never escape their
    /// directory or collide across users.
    #[test]
    fn slugs_cannot_escape_their_scope() {
        assert_eq!(slug("python-coding").unwrap(), "python-coding");
        assert_eq!(
            slug("cold-fusion research").unwrap(),
            "cold-fusion-research"
        );
        assert_eq!(slug("../../etc/passwd").unwrap(), "etc-passwd");
        assert_eq!(slug("a/../../b").unwrap(), "a-b");
        assert!(slug("../..").is_none());
        assert!(slug("///").is_none());
        assert!(slug("").is_none());
        assert!(slug(&"x".repeat(500)).unwrap().len() <= 64);

        let layout = ScopeLayout::new("/tmp/ng");
        let p = layout
            .topic("qwen3.5", "../../escape", Some("bob"))
            .unwrap();
        assert!(
            p.starts_with("/tmp/ng/user/bob"),
            "topic escaped its scope: {p:?}"
        );
        assert!(!p.to_string_lossy().contains(".."));

        // A user table and a shared topic table never resolve to one path.
        let u = layout.user("qwen3.5", "bob").unwrap();
        let t = layout.topic("qwen3.5", "python", None).unwrap();
        assert_ne!(u, t);
    }

    /// The property the whole scoping design exists to guarantee: what one
    /// user's table learns must be unreachable from another user's session.
    /// `next`/`next2` are plaintext, so a leak here is a text-extraction path,
    /// not a quality regression.
    #[test]
    fn one_users_table_is_unreachable_from_another() {
        let dir = tmpdir("tenancy");
        let layout = ScopeLayout::new(&dir);

        // A distinctive sequence only user "alice" ever sees.
        let secret: Vec<u32> = (0..400).map(|i| 90_000 + (i % 37)).collect();
        {
            let mut alice = NgramSpec::new(NgramConfig {
                promote_count: 2,
                ..Default::default()
            });
            let p = layout.user("qwen3-5", "alice").unwrap();
            alice.attach_user(&p, 200_000, 512).unwrap();
            alice.observe(&secret);
            alice.merge().unwrap();
            // Alice can draft her own material.
            assert!(
                alice.draft().is_some(),
                "alice should draft her own sequence"
            );
        }

        // Bob, same root, same tokenizer scope, different user.
        let mut bob = NgramSpec::new(NgramConfig {
            promote_count: 2,
            ..Default::default()
        });
        let pb = layout.user("qwen3-5", "bob").unwrap();
        bob.attach_user(&pb, 200_000, 512).unwrap();
        // Feed Bob the *prefix* of Alice's sequence — the exact probe an
        // attacker would use to walk the continuation out of a shared table.
        bob.observe(&secret[..12]);
        let leaked = bob.draft();
        assert!(
            leaked.is_none(),
            "bob drafted from alice's table: {leaked:?} — tables are not isolated"
        );
        assert_eq!(bob.stats().accepted_in(Tier::User), 0);

        // And the two tables really are separate files.
        let pa = layout.user("qwen3-5", "alice").unwrap();
        assert_ne!(pa, pb);
        assert!(pa.exists() && pb.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A session type gets its own table, under the user, so topic material
    /// never lands in a table another user can reach.
    #[test]
    fn session_type_gets_a_private_table_under_the_user() {
        let dir = tmpdir("topic-scope");
        let layout = ScopeLayout::new(&dir);

        let user_tbl = layout.user("qwen3-5", "alice").unwrap();
        let topic_tbl = layout
            .topic("qwen3-5", "python coding", Some("alice"))
            .unwrap();
        let shared_tbl = layout.topic("qwen3-5", "python coding", None).unwrap();

        // Distinct from the plain user table, and nested under that user.
        assert_ne!(user_tbl, topic_tbl);
        assert!(topic_tbl.starts_with(dir.join("user").join("alice")));
        // A shared topic table is a different path entirely — it must never
        // collide with a user's private one.
        assert_ne!(topic_tbl, shared_tbl);
        assert!(!shared_tbl.starts_with(dir.join("user")));

        // The label is slugged on the way in.
        assert!(topic_tbl.to_string_lossy().contains("python-coding"));

        // A writable topic table under the user round-trips.
        let mut ng = NgramSpec::new(NgramConfig {
            promote_count: 2,
            write_target: WriteTarget::Topic,
            ..Default::default()
        });
        ng.attach_topic(&topic_tbl, 200_000, 256, true).unwrap();
        let stream: Vec<u32> = (0..300).map(|i| 70_000 + (i % 29)).collect();
        ng.observe(&stream);
        ng.merge().unwrap();
        assert!(
            ng.store(Tier::Topic).unwrap().occupancy().0 > 0,
            "topic table stayed empty"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn acceptance_is_attributed_per_tier() {
        let mut ng = NgramSpec::new(cfg());
        let stream: Vec<u32> = (0..32 * 8).map(|i| i % 32).collect();
        ng.observe(&stream);
        let n = ng.draft().unwrap().len();
        ng.record_acceptance(n);
        let st = ng.stats();
        assert_eq!(st.accepted, n as u64);
        assert_eq!(st.accepted_in(Tier::Hot), n as u64);
        assert_eq!(st.accepted_in(Tier::Base), 0);
        assert!(st.verify_efficiency() > 0.0);
    }
}

#[cfg(test)]
mod acceptance_tests {
    use super::*;

    #[test]
    fn the_bound_ranks_sustained_performance_above_a_lucky_streak() {
        // The whole reason for a bound rather than accepted/proposed: the point
        // estimate ranks 1-for-1 (100%) above 40-for-50 (80%), which is the same
        // flaw `promote_count`'s raw threshold has one level up.
        let lucky = GramOutcome {
            proposed: 1,
            accepted: 1,
        };
        let proven = GramOutcome {
            proposed: 50,
            accepted: 40,
        };
        assert!(
            acceptance_lower_bound(proven) > acceptance_lower_bound(lucky),
            "40/50 ({:.3}) must outrank 1/1 ({:.3})",
            acceptance_lower_bound(proven),
            acceptance_lower_bound(lucky)
        );
    }

    #[test]
    fn the_bound_is_conservative_and_tightens_with_evidence() {
        // Same 80% rate, more evidence -> a higher (tighter) lower bound, but
        // never above the rate itself.
        let small = GramOutcome {
            proposed: 10,
            accepted: 8,
        };
        let large = GramOutcome {
            proposed: 1000,
            accepted: 800,
        };
        let (bs, bl) = (acceptance_lower_bound(small), acceptance_lower_bound(large));
        assert!(bs < bl, "more evidence must tighten: {bs:.3} vs {bl:.3}");
        assert!(
            bl < 0.8,
            "the bound must stay below the point estimate: {bl:.3}"
        );
        assert!(bl > 0.75, "1000 samples should be tight: {bl:.3}");
    }

    #[test]
    fn no_evidence_is_not_evidence_of_badness() {
        assert_eq!(acceptance_lower_bound(GramOutcome::default()), 0.0);
        // ... and a gram with no record is never suppressed, whatever the bar.
        let mut cfg = NgramConfig::default();
        cfg.min_acceptance = 0.9;
        let spec = NgramSpec::new(cfg);
        assert!(!spec.suppressed(7, 0xdead_beef));
    }

    #[test]
    fn the_gate_is_inert_until_both_the_bar_and_the_evidence_are_set() {
        let mut cfg = NgramConfig::default();
        cfg.min_acceptance_proposals = 4;
        // Default bar is 0.0 = off, which is the shipped state.
        assert_eq!(NgramConfig::default().min_acceptance, 0.0);

        cfg.min_acceptance = 0.5;
        let mut spec = NgramSpec::new(cfg);
        let gram = (3u32, 99u64);
        // Three rejections: below the evidence floor, so still not suppressed.
        for _ in 0..3 {
            spec.note_outcome(gram, false);
        }
        assert!(
            !spec.suppressed(gram.0, gram.1),
            "3 < min_acceptance_proposals"
        );
        // The fourth crosses the floor, and 0-for-4 is confidently below 0.5.
        spec.note_outcome(gram, false);
        assert!(
            spec.suppressed(gram.0, gram.1),
            "0/4 must suppress at a 0.5 bar"
        );

        // A gram that actually performs is never suppressed.
        let good = (4u32, 100u64);
        for _ in 0..20 {
            spec.note_outcome(good, true);
        }
        assert!(!spec.suppressed(good.0, good.1));
    }

    #[test]
    fn a_full_ledger_fails_toward_drafting_not_away_from_it() {
        // Clearing loses counts, so the gate stops suppressing until they
        // rebuild. That is the safe direction: it drafts more, never less.
        let mut cfg = NgramConfig::default();
        cfg.min_acceptance = 0.5;
        cfg.min_acceptance_proposals = 1;
        cfg.hot_capacity = 4;
        let mut spec = NgramSpec::new(cfg);
        let victim = (1u32, 1u64);
        spec.note_outcome(victim, false);
        assert!(spec.suppressed(victim.0, victim.1));
        for i in 2..12u64 {
            spec.note_outcome((i as u32, i), false);
        }
        assert!(
            !spec.suppressed(victim.0, victim.1),
            "an evicted gram must draft again, not stay condemned"
        );
    }
}

#[cfg(test)]
mod marginal_curve_tests {
    use super::*;

    fn stats_with(drafted: &[u64], accepted: &[u64]) -> NgramStats {
        let mut s = NgramStats::default();
        for (i, (&d, &a)) in drafted.iter().zip(accepted).enumerate() {
            s.drafted_by_depth[i] = d;
            s.accepted_by_depth[i] = a;
        }
        s
    }

    #[test]
    fn the_curve_reads_back_what_was_recorded() {
        let s = stats_with(&[100, 100, 100], &[90, 50, 10]);
        assert!((s.marginal_acceptance(0) - 0.90).abs() < 1e-9);
        assert!((s.marginal_acceptance(1) - 0.50).abs() < 1e-9);
        assert!((s.marginal_acceptance(2) - 0.10).abs() < 1e-9);
        // An unseen depth is 0.0, not a divide by zero.
        assert_eq!(s.marginal_acceptance(9), 0.0);
    }

    #[test]
    fn the_curve_is_a_pure_function_of_counts() {
        // The point of deriving from counts rather than a clock: identical
        // inputs give an identical answer, so a replay is bit-identical.
        let a = stats_with(&[100, 100], &[90, 40]);
        let b = stats_with(&[100, 100], &[90, 40]);
        for d in 0..MAX_TRACKED_DEPTH {
            assert_eq!(a.marginal_acceptance(d), b.marginal_acceptance(d));
        }
    }
}

#[cfg(test)]
mod width_seam_tests {
    use super::*;

    /// The seam an external throughput controller drives.
    #[test]
    fn the_spine_cap_is_settable_and_never_zero() {
        let mut spec = NgramSpec::new(NgramConfig::default());
        assert_eq!(spec.spine_token_cap(), None, "undriven by default");
        spec.set_spine_token_cap(Some(4));
        assert_eq!(spec.spine_token_cap(), Some(4));
        // A zero cap would silently stop drafting rather than narrow it —
        // "off by configuration", which is invisible from outside.
        spec.set_spine_token_cap(Some(0));
        assert_eq!(
            spec.spine_token_cap(),
            Some(1),
            "zero must clamp, not disable"
        );
    }

    /// Narrowing the cap must actually shorten the draft, or the controller is
    /// steering something that does not move.
    #[test]
    fn a_narrower_cap_shortens_the_draft() {
        let mut cfg = NgramConfig::default();
        cfg.chain_floor = 0; // let the chain run to the cap
        let mut spec = NgramSpec::new(cfg);
        // Teach it a long, perfectly repeating sequence so the spine can fill.
        let cycle: Vec<u32> = (0..8).collect();
        for _ in 0..64 {
            spec.observe(&cycle);
        }
        let undriven = spec.draft().map(|s| s.len()).unwrap_or(0);
        spec.set_spine_token_cap(Some(2));
        let narrow = spec.draft().map(|s| s.len()).unwrap_or(0);
        assert!(undriven > 2, "the fixture should draft a long spine");
        assert!(
            narrow <= 2,
            "the cap counts TOKENS: undriven={undriven} narrow={narrow}"
        );

        // The documented hazard: cfg.max_spine bounds STEPS, and a step with an
        // inline next2 emits two tokens — so it is not a token cap and must not
        // be used as one by a controller.
        spec.set_spine_token_cap(None);
        let steps_cap = spec.max_spine_steps();
        assert!(
            undriven > steps_cap,
            "max_spine ({steps_cap}) bounds steps, not tokens; drafted {undriven}"
        );
    }
}
