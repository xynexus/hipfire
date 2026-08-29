# n-gram spec-decode: replay sweep (2026-08-29)

Baseline evidence for a drafter-free n-gram speculative-decode path (n-gram
first, DFlash on miss). Measured with
`crates/hipfire-arch-qwen35/examples/ngram_replay_sweep.rs`.

## Method

CPU-only acceptance oracle. Replay a recorded token stream through a
multi-order n-gram table; at each position ask what the table *would* draft.
The stream is its own ground truth, so a correct prediction is exactly a token
the target would have accepted — acceptance without a verify pass, a drafter,
or a GPU.

Validated by `--self-check`: a perfectly periodic stream must accept every
chained token (1523/1523 chains × 8 tokens), and a random stream over a 100k
vocab must produce ~no 5-gram proposals (0 in 20000 tokens, which also bounds
fingerprint collisions).

Tokenizer: `Qwen3-4B--bf16.hfq` (index-only open). Arms:
- **CODE** — this repo's `crates/**/*.rs`, 1M tokens
- **PROSE** — `benchmarks/calib/calib-multi-labelled.jsonl`, 1M tokens

**Caveat:** corpus text is a proxy for a decode stream. It captures prompt-echo
repetition but not the model's own output-side repetition, so these are a lower
bound on a real serving stream. Numbers below are not a throughput claim — they
are accept-rate structure used to choose a data structure.

## Per-order precision (independent arms, min_count=2)

| order | CODE cover | CODE precision | PROSE cover | PROSE precision |
|------:|-----------:|---------------:|------------:|----------------:|
| 5 | 30.8% | 76.0% | 7.3% | 65.7% |
| 4 | 39.7% | 70.0% | 12.8% | 55.6% |
| 3 | 53.8% | 60.4% | 24.0% | 44.3% |
| 2 | 73.7% | 47.8% | 46.6% | 32.1% |

Order is the dominant quality signal. Mean accepted tokens when order *n* wins
the first draft (CODE): n=5 → 3.00, n=4 → 1.68, n=3 → 1.19, n=2 → 0.75. High
order does predict multiple tokens; a bigram essentially predicts one.

## Order ladder (min_count=1, max_spine=8)

| orders | CODE acc/step | PROSE acc/step |
|---|---:|---:|
| 5,4,3,2 | 1.80 | 0.51 |
| 6,5,4,3,2 | 1.95 | 0.53 |
| 8,7,6,5,4,3,2 | 2.11 | 0.54 |
| 12,10,8,6,5,4,3,2 | 2.22 | 0.54 |

Going past quad keeps paying on code, flat on prose.

## Admission threshold — for *drafting*, min_count hurts

CODE, orders 5,4,3,2:

| min_count | coverage | acc/step |
|---:|---:|---:|
| 1 | 85.0% | **1.80** |
| 2 | 73.7% | 1.39 |
| 3 | 66.9% | 1.18 |
| 5 | 58.8% | 0.96 |
| 9 | 49.8% | 0.75 |

Precision bucketed by count-at-hit-time is **flat** (58.5% → 60.9% across
counts 2..17+ on CODE; 39% → 44% on PROSE). Count does not predict correctness
once a gram exists.

**Consequence:** the count threshold belongs on the *disk-write* gate, not the
draft gate. Draft from everything in RAM; only persist what has proven itself.

## Chain stop rule — the biggest single win

With min_count=1 the chain never terminates early: it pads to `max_spine` and
burns verify width. Gating *continuation* on the winning order fixes this
(CODE, orders 8..2, min_count=1, max_spine=16):

| chain floor | acc/step | drafted/step | verify efficiency |
|---:|---:|---:|---:|
| 0 (none) | 2.81 | 16.00 | 20.6% |
| 4 | 2.73 | 13.03 | 24.6% |
| 6 | 2.48 | 9.03 | 32.3% |
| 8 | 2.23 | 6.94 | **37.9%** |

floor=8 cuts drafted tokens 57% for a 21% accept loss. It dominates a plain
`max_spine=8` cap (2.11 acc/step at 8.00 drafted, 31.1% efficiency) on both
axes.

## Cold table vs hot table

Score the last 50% of the stream; prime the table on the first 50%.

| | CODE acc/step | PROSE acc/step |
|---|---:|---:|
| cold only (primed, frozen) | 1.32 | 0.31 |
| hot only (warms from empty, full stream) | 2.23 | 0.45 |
| cold + hot | **2.79** | 0.54 |

**The hot table is the dominant term.** A frozen cold table alone is worth
about half of a self-warming hot table. Cold adds a real but secondary ~25% on
top of hot.

## Budget

Share of hits retained keeping only the top-K grams by count (CODE, 1M tokens):

| orders | distinct grams | top-100k keeps | note |
|---|---:|---:|---|
| 5,4,3,2 | 1.46M | 81.8% | 100k grams ≈ 2.3 MB @ 24 B |
| 8,7,6,5,4,3,2 | 3.44M | 39.5% | 29% of grams serve 100% of hits |

At four orders the useful table is **megabytes**. The full order ladder is what
inflates it — high orders give the multi-token wins *and* generate the long
tail of near-unique contexts. **71% of grams at the full ladder never serve a
single hit**, which is the strongest argument for the write-admission gate.

## PLD baseline (existing `PldMatcher`, 8k context window)

| arm | coverage | acc/proposing step | acc/decode step |
|---|---:|---:|---:|
| CODE | 48.6% | 3.20 | 1.55 |
| PROSE | 15.5% | 1.59 | 0.25 |

PLD has higher per-proposal quality but much lower coverage, and costs an
O(context) rescan per step. The table beats it on acc/decode-step in both arms
at O(1) per probe.

## Recommendation

- Orders 2..8, `min_count=1` for drafting, `max_spine=16`, chain floor 8.
- Count threshold gates persistence, not drafting.
- Build the hot tier and the three-tier fallback first. The GB cold table is
  the weaker half of the win and should be sized from a real serving trace, not
  from this corpus proxy.

## Not yet measured

- Real decode streams (model output, not corpus text). Blocked on GPU: the only
  local Qwen3.5/3.6 target+drafter pair is `Qwen3.6-35B-A3B--oq4.25++.hfq`
  (19 GB) on nix2's 32 GB UMA, and the lock was held during this run.
- Interaction with DFlash on the miss path (does n-gram steal the easy tokens
  and leave DFlash the hard ones, lowering *its* accept rate?).

## Block occupancy (added: fixed-file block store)

Modelling a fixed 1 GB file of 4 KB disk-aligned blocks, 24 B records →
262,144 blocks × 170 records. `--block-report`.

| | CODE | PROSE |
|---|---:|---:|
| distinct keys in use | 14,637 | 49,564 |
| grams per key: max | 76,305 | 162,819 |
| grams per key: p99 | 3,258 | 1,082 |
| grams per key: p50 | 28 | 14 |
| **1 block per key** | **24.4% of grams fit** | **33.6% fit** |
| multi-block: blocks needed | 31,230 (11.9% of file) | 72,460 (27.6%) |
| multi-block: mean fill | 64.8% | 43.7% |

Fill skew is ~2,700:1 between the hottest key and the median. **One block per
key is unworkable** — 76% of grams spill on CODE. Multi-block assignment fits
comfortably (12% of a 1 GB file) at ~65% fill.

First-token and last-token keying give *identical* occupancy — both are just
the token unigram distribution. They differ in read amplification, not fill:
every order-*k* context at position *i* ends at `tokens[i-1]`, so last-token
keying puts the whole order ladder in one block (1 read/chain step), while
first-token keying scatters orders 2..8 across up to 7 blocks.

### Resulting structure — directory, not a tree

The top-level key is a **dense bounded integer** (a token id, ~151k vocab), so
mapping it to a block needs an array, not a search structure:

- `directory[last_token] -> (first_block, n_blocks)` — 151k × 8 B = 1.2 MB, RAM.
- `block = first_block + (fingerprint % n_blocks)` — second-level hash keeps it
  to exactly **1 block read** even for a key owning 449 blocks.
- within the block: records sorted by fingerprint, binary search over 170
  entries, entirely inside the already-loaded 4 KB page.

Skew is absorbed by `n_blocks` per key, recomputed during the offline merge that
rewrites the file anyway — so there is no online rebalancing to pay for. A
B-tree would solve the same skew by splitting, but it exists to search sparse
ordered keys and to grow; here the key space is dense and the budget is fixed,
where the answer to a full block is eviction, not a split.

## Implementation (`hipfire-specdecode-ngram`)

Built to the measured operating point. The crate is **GPU-free and dependency-
free** (`memmap2` only): unlike dflash/ddtree/mtp/dspark it is not generic over
`SpecDecodeTarget`, because an n-gram drafter never runs a model.

- `hot.rs` — RAM staging + admission gate. Drafts read at `min_count=1`; grams
  reach disk only at `promote_count`. Amortized eviction (one scoring pass per
  capacity/4 observations), qualifying evictees queued for the write trickle.
- `cold.rs` — fixed block file. `directory[last_token] -> (first_block,
  n_blocks)` in RAM, `block = first_block + (fp % n_blocks)`, records sorted by
  fingerprint inside the block. Two write paths: `insert_in_place` (one dirty
  page, between tokens) and `merge` (full rewrite, recomputes the directory —
  this is the rebalance).
- `lib.rs` — tiering, chain policy, per-tier acceptance attribution.

### Verified end-to-end against the same oracle

`cargo run --release -p hipfire-specdecode-ngram --example replay_real`,
1M tokens of `crates/**/*.rs`, chain_floor=8:

| | coverage | acc/step | verify eff |
|---|---:|---:|---:|
| hot only (cap 1M) | 82.3% | 2.82 | 31.0% |
| hot cap 64k, cold empty (session 1) | 74.6% | 2.39 | 33.4% |
| hot cap 64k, cold warm (session 2) | 79.5% | **2.65** | 35.0% |

Session 2 reuses the store session 1 wrote. **Cold marginal share rises 6.2% →
15.8%** — the cold tier's contribution is cross-session, which is exactly the
case a single-session measurement cannot see. (Upper bound: session 2 replays
the same corpus, so gram overlap is total. Real traffic will transfer less.)

Throughput 76k tok/s CPU with the cold store attached, 866k without — both far
above decode rates, so the drafter is not on the critical path. A 1 GB merge
costs ~7 s, which confirms merge must stay periodic and off the token path.

### Bug found by the end-to-end run

The first run wrote **zero** records to disk. `insert_in_place` correctly
refuses a key that owns no blocks yet, but the caller dropped the gram instead
of holding it — so an empty store could never bootstrap its own directory.
Failed in-place writes now land in a merge backlog. Unit tests missed this
because they merge before inserting; only the cold-start path was broken.

### Telemetry that answers the open question

`NgramStats` records `accepted_hot` / `accepted_cold`, where a cold hit is by
construction a gram hot did not have (hot is probed first and wins ties). So
`cold_marginal_share()` is the cold tier's value measured directly on live
traffic — no second experiment needed to decide whether it pays for its bytes.

## Tiers, scoping, and the serving path

### Four tiers, probed most-specific first

| tier | scope | writes | source |
|---|---|---|---|
| `Hot` | session | RAM | this request |
| `User` | one user | serving | that user's traffic |
| `Topic` | one subject | serving *or* read-only | a session type, e.g. `python-coding` |
| `Base` | one tokenizer | never | offline, published corpus |

First hit wins, so a hit at tier N means no more-specific tier had that gram —
which makes `NgramStats::marginal_share(tier)` each tier's value measured
directly on live traffic.

Verified cross-crate: a base table built from `hipfire-runtime` (147,448
records), then decoded against `hipfire-arch-qwen35`:

| | coverage | acc/step | base marginal |
|---|---:|---:|---:|
| user table only | 66.0% | 2.83 | — |
| + base attached read-only | 79.5% | 2.93 | **4.45%** |

### Scope is the tokenizer, not the model

Records are token ids, so every quant variant of one base may share a table and
two tokenizers never can. `ngram_scope` defaults to the model filename, which
never wrongly shares; an operator opts two models into one table explicitly.

### Sharing a writable table across users leaks their text

`next`/`next2` are stored in plaintext and `fingerprint` is a fixed public
function, so a table two users can reach is a **continuation oracle**: guess a
context, hash it, read what followed, append, repeat. That extracts text and
credentials, so it is closed structurally, not by convention —
`ColdStore::open_read_only` refuses `insert_in_place` and `merge`, and the base
tier has no other constructor. Read-only also maps shared/immutable, so all
tenants share one copy of the base pages instead of one each.

Scope labels (`user_id`, `session_type`) arrive from request data and are used
to build paths, so `slug()` reduces them to `[a-z0-9-]`: `../../etc/passwd`
becomes `etc-passwd` and cannot escape its scope directory.

### Serving wiring

`generate_dflash` builds an `NgramSpec` when `ngram_spec` is on, seeds it with
the prompt (prompt-echo is where a training-free drafter earns most of its
acceptance), drafts before each `spec_step_dflash`, and passes the spine to the
existing `pld_spine` parameter — the seam already shrinks the verify batch to
`1 + spine.len()` and skips the drafter forward. On a miss the parameter stays
`None` and the DFlash path is byte-for-byte unchanged, so the feature is
strictly additive.

Acceptance is attributed **only** when the n-gram tier actually supplied the
draft; otherwise `step.accepted` belongs to the DFlash drafter.

Knobs (`ngram_spec`, `ngram_store_root`, `ngram_scope`) follow the
`dflash_adaptive_b` path: config default, per-load param override, off by
default. Empty `ngram_store_root` = RAM only, nothing written to disk.

### Per-request scoping (done)

`NgramRequestScope { user_id, session_type }` is threaded
`daemon/handlers/generate.rs -> generate_start -> generate_dflash`, in the style
`raw_override` and `sampler_seed` established — deliberately not a global,
because a global that leaked here would write one user's text into another's
table. Requests carry optional `user_id` and `session_type` fields; absent
`user_id` means daemon-local, which is correct single-tenant.

A `session_type` resolves to a table **under that user**
(`<root>/user/<user>/<tokenizer>/<topic>.ngram`), so topic material stays
private and may be written. A topic table shared across users would have to be
read-only, exactly like the base tier.

Isolation is pinned by test: `one_users_table_is_unreachable_from_another`
feeds user B the *prefix* of user A's private sequence — the probe an attacker
would use to walk a continuation out of a shared table — and asserts B drafts
nothing.

### Four tiers verified together

Base built from `hipfire-runtime`, topic from `hipfire-quantize`, decoded
against `hipfire-arch-qwen35` (400k tokens, hot cap 32k):

| tier | lookups | hits | accepted | marginal |
|---|---:|---:|---:|---:|
| hot | 2,064,746 | 2,064,746 | 1,118,584 | 95.24% |
| user | 1,858,439 | 0 | 0 | 0.00% |
| topic | 1,858,439 | 75,546 | 31,846 | 2.71% |
| base | 1,782,893 | 51,441 | 24,070 | 2.05% |

Coverage 76.5%, 2.94 accepted/step, 36.1% verify efficiency. `base` sees fewer
lookups than `topic` because `topic` is probed first and already served some —
the cascade working as designed. `user` is empty because it is created at the
start of this session.

### GPU: end-to-end, validated

Ran on nix2 (gfx1103) against **`Qwen3.5-9B--mq4.hfq`** (5.30 GB, arch 5,
`lm_head` = `MQ4G256`) + **`Qwen3.5-9B--dflash.mq4.hfq`**, both copied from
`/srv/hipfire` into `~/.hipfire/{models,drafts}` under canonical names.

Finding the pair took three eliminations, all environmental:
- `Qwen3.6-35B-A3B--oq4.25++` needs 32.0 GiB against 28.7 GiB `MemAvailable`;
  `max_seq` does not move it (MoE expert weights, not KV).
- Both `oq4.25++` targets have an `OqPlusCompact` (`quant_type=36`) `lm_head`,
  unsupported by `speculative.rs`'s batched GEMM on gfx1103.
- The Qwen3.5-9B / 3.6-27B DFlash sidecars had no local `.hfq` target until the
  mq4 one was found under `/srv/hipfire/drafts/native-oq-evidence/`.

**`kv_mode` matters.** The default F32 KV cache has *no batched write* — a
documented gap (`kv_tier.rs`: "F32 → no batched keys exist") — so spec-decode
fails with `no implementation for KvWriteF32`. This is independent of n-gram:
ngram-on and ngram-off failed identically. `kvarn` (non-deprecated) works, as do
q8/asym4 behind `HIPFIRE_KV_ALLOW_DEPRECATED=1`.

Baseline vs n-gram, same prompt, `max_tokens=250`, temperature 0, `kv_mode=kvarn`:

| | tok/s |
|---|---|
| ngram off, 6 requests | 13.0 12.6 13.3 13.3 13.1 13.3 — **mean 13.10** |
| ngram on, 6 requests | 15.7 14.6 17.3 19.8 17.2 17.0 |

Learning across the session (same six requests):

| req | tok/s | accept_rate | coverage | acc/step | hot entries | user records |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 15.7 | 0.16 | 11.0% | 0.27 | 1,689 | 2 |
| r3 | 17.3 | 0.20 | 28.6% | 0.40 | 4,388 | 208 |
| r6 | 17.0 | 0.20 | 47.3% | 0.85 | 6,865 | 761 |

**Cross-restart value.** A *fresh daemon process* against the store the previous
run wrote:

| req | tok/s | coverage | user records | user hits | user accepted | user marginal |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 17.2 | 24.6% | 761 | 24 | 8 | **44.4%** |
| s2 | 14.8 | 36.4% | 771 | 58 | 42 | **55.3%** |

Within one load the user tier reports 0 hits — correct, not a bug: it is written
*from* hot, so hot always wins the probe. Its value appears only across a
restart, which is exactly what the per-tier attribution isolates.

Tables landed at `user/alice/qwen3-5.ngram` and
`user/alice/qwen3-5/rust-coding.ngram` — topic nested under the user, private.

### Three bugs the GPU run caught that unit tests did not

1. **A fresh cold store could never bootstrap.** `insert_in_place` refused any
   key owning no blocks, and only `merge` assigned blocks — but a merge rewrites
   the whole file (~7 s/GB) and cannot run per request. Fixed with an on-demand
   block allocator (bump cursor persisted in the superblock); keys are sparse
   (~14.6k of a 150k vocab), so one block on first write covers it with a single
   dirty page. Pinned by `fresh_store_persists_without_any_merge`.
2. **`NgramSpec` was rebuilt per request**, resetting the hot table and every
   promotion counter, so almost nothing reached disk. It now lives on
   `DflashState` keyed by scope; a different user/session swaps it out rather
   than inheriting.
3. **`reset()` wiped the hot table** on every request, defeating (2) even after
   it was fixed. Split into `reset_sequence()` (rolling history only, used by
   serving) and `reset()` (full wipe). Coverage went 11%→47% across six requests
   once this was right; before the fix it was flat.

Telemetry is emitted in the `done` event as `ext.ngram`, with per-tier lookups /
hits / drafted / accepted / marginal share and store occupancy.

### Still open

- Shared (cross-user, read-only) topic tables have API support but no config
  surface; only per-user topic tables are reachable today.
- The write path feeds one store (`write_target`, default `User`), so a topic
  table only fills when it is the write target — it stays empty otherwise.
- `merge` runs only on a scope change; a long-lived load never rebalances.

## Configuration surface

Nine fields, all `LoadTime` / `GLOBAL_MODEL_RUNTIME`, each overridable per load
via the daemon's `params` (the `dflash_adaptive_b` pattern), and off by default.

| field | default | what it does |
|---|---|---|
| `ngram_spec` | `false` | master switch |
| `ngram_store_root` | `""` | table root; empty = RAM only, nothing persisted |
| `ngram_scope` | `""` | tokenizer scope name; empty = derive from model filename |
| `ngram_store_mb` | `256` | **the budget** — file is allocated in full and never grows |
| `ngram_orders` | `8,7,6,5,4,3,2` | probe ladder, longest first |
| `ngram_chain_floor` | `8` | min winning order to keep extending; 0 disables |
| `ngram_max_spine` | `16` | max draft length |
| `ngram_promote_count` | `3` | observations before a gram is persisted |
| `ngram_write_target` | `user` | which store the write path feeds (`user`/`topic`/`none`) |

Per-request (generate): `user_id`, `session_type`.

Defaults are the measured operating point. Malformed `ngram_orders` falls back
to the default ladder with a warning rather than leaving an empty ladder;
`ngram_write_target` falls back to `user`, which is private by construction.

### Knobs verified to take effect on real GPU decode

Qwen3.5-9B mq4 + DFlash, 3 requests each:

| setting | drafted/step | verify eff | acc/step | blocks free |
|---|---:|---:|---:|---:|
| `ngram_chain_floor=0` | 22.15 | 7.7% | 0.36 | 65504 |
| `ngram_chain_floor=8` | 6.13 | 27.1% | 0.49 | 65502 |
| `ngram_store_mb=16` | 6.13 | 27.1% | 0.49 | 4062 |
| `ngram_orders="2"`, floor 0 | 15.12 | 11.8% | 0.42 | 65499 |

The chain-floor result reproduces the CPU sweep on live decode, and here floor 8
dominates outright — fewer drafted tokens *and* higher accepted/step. Budget is
honoured: the 16 MiB store allocates 18,767,872 bytes against 270,426,112 for
the 256 MiB default.

Note the directory is `vocab * 8 B` (~1.6 MB at a 200k vocab) regardless of
budget, so it is a large fraction of a very small store — 16 MiB of blocks costs
~18 MB on disk.

### Not wired

- No CLI flags; the knobs are reachable via config file and daemon load params
  only.
- `hot_capacity`, `flush_per_token` and `next2_min_order` stay at their defaults
  — no measurement yet says they need operator control.

## Handoff state (2026-08-29, end of session)

**Everything is uncommitted.** 14 modified files + 3 untracked
(`crates/hipfire-specdecode-ngram/`, the qwen35 sweep example, this devlog).
`./tests/no-gpu-ci.sh` exits 0; 13 crate tests + 2 `NgramSetup` tests pass;
clippy clean on every touched crate.

Artifacts pulled from `/srv/hipfire` for the GPU test, still local (5.5 GB):
- `~/.hipfire/models/Qwen3.5-9B--mq4.hfq`
- `~/.hipfire/drafts/Qwen3.5-9B--dflash.mq4.hfq`

These are the only local target+drafter pair that can run DFlash on gfx1103
(`MQ4G256` lm_head). Deleting them means the GPU path cannot be re-tested here.

### Deliberate shortcuts (`ponytail:` markers)

- `cold.rs:116` — eviction score is linear age decay, one epoch = one point.
- `hot.rs:171` — hot eviction is `count * 8 - age`. Measured age precision is
  non-monotonic on prose (very old grams score *well*, being structural), so a
  smarter curve may pay; nothing measured says so yet.

### Open, in rough priority order

1. **`merge` only runs on a scope change.** A long-lived single-user load never
   rebalances its store, so block assignment drifts from the real key
   distribution as it fills. Needs a cadence (idle, or every N promotions).
2. **A topic table only fills when it is the `write_target`.** The write path
   feeds exactly one store, so with the default `user` a topic table is
   read-only in practice. Fine today, wrong once topic tables are the point.
3. **Shared (cross-user, read-only) topic tables** have API support
   (`attach_topic(.., writable=false)`) but no config surface.
4. **No CLI flags** — config file, daemon load params, or `PATCH
   /admin/config/editor` only. See `docs/plans/tui-webui-config-editing.md`.
5. `hot_capacity`, `flush_per_token`, `next2_min_order` are not configurable;
   no measurement yet says they need to be.

### Not carried over from the CPU sweep

The corpus-replay numbers used a *fresh* table per arm. The serving path now
carries hot state across requests, so the two are not directly comparable —
serving coverage climbs 11% → 47% over six requests where the sweep held steady.
Re-run the sweep with carry-over before comparing them again.
