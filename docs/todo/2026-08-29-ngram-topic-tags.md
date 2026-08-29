# TODO — n-gram topic tables → topic **tags** (multiple topics per session)

Status: **not started** (design). Supersedes the single-`session_type` topic
tier shipped in the n-gram spec-decode work.

## Why

Today a session carries at most one topic. `NgramRequestScope` has
`session_type: Option<&str>`, which resolves to exactly one table at
`<root>/user/<user>/<tokenizer>/<topic>.ngram`
(`ScopeLayout::topic`), attached as the single `Tier::Topic`.

That is wrong for how sessions actually look. A session is "rust + hipfire +
kernel work", not one label — and the interesting reuse is *across* the
intersection: someone writing HIP kernels benefits from both the `rust` and the
`gpu-kernels` tables. One label per session forces a false choice and fragments
the very tables that should be shared.

The plan is to move `session_type` to a **tag set**.

## What has to change

### 1. Scope and layout

`session_type: Option<&str>` becomes an ordered tag list. `ScopeLayout::topic`
already slugs its input (`slug()` reduces to `[a-z0-9-]`, so `../../etc/passwd`
becomes `etc-passwd`); each tag needs the same treatment individually, not as a
joined string, or `a/b` silently becomes the single tag `a-b`.

Path stays one file per tag — `<root>/user/<user>/<tokenizer>/<tag>.ngram` — so
tables are shared across sessions that share a tag, which is the point.

### 2. The tier model has to stop being one slot per tier

`NgramSpec` currently holds `user`, `topic`, `base` as three `Option<ColdStore>`
and `Tier` is a 4-value enum indexing fixed-width stat arrays
(`drafted_by_tier: [u64; 4]`). N tags means N topic stores.

Two options:

- **Collapse to a probe list.** `Vec<(TierLabel, ColdStore)>` probed in order,
  with stats keyed by label. More invasive, but it removes the arity limit for
  good and makes the "probed most-specific first, first hit wins" rule the
  single thing that defines a tier.
- **Keep the enum, fan out inside Topic.** Less churn, but per-tag attribution
  is lost — and per-tier attribution is the whole reason the telemetry exists
  (`marginal_share` answers "does this table earn its bytes?"). Losing it per
  tag defeats the purpose, because the decision we actually want is *which tags
  are worth keeping*.

Prefer the probe list. Cap the number of attached tag tables (each is an mmap
and an fd) and document the cap.

### 3. Probe order across tags

Tiers are ordered by specificity and the first hit wins — that is what makes
`accepted_in(tier)` a marginal value rather than a raw count. Tags have no
natural order, so pick one and justify it:

- request order (caller states most-specific first) — simple, puts the choice
  on the caller;
- by table size or hit rate — self-tuning, but makes attribution drift as the
  tables grow, which muddies exactly the number we are trying to read.

Request order is probably right. Whatever is chosen, it must be **stable within
a session**, or the per-tag marginal numbers mean nothing.

### 4. The write path

`write_target` currently names one store (`user` / `topic` / `none`). With N
tags, "write to topic" is ambiguous. Decide:

- write to every attached tag table (N× the write amplification, and a gram
  learned in a rust+gpu session pollutes both single-topic tables), or
- write to the first tag only, or
- keep writing to the user table and let tag tables be built offline from
  curated corpora.

The third keeps the write path single, keeps tag tables clean, and fits the
"generate training data" goal that motivated topics in the first place. It also
sidesteps the sharing hazard below.

### 5. Privacy — do not lose this

Tag tables that live under `<root>/user/<user>/` are private and may be written.
The moment a tag table is shared *between* users it must be opened read-only
(`ColdStore::open_read_only`), because `next`/`next2` are stored in plaintext
and `fingerprint` is a fixed public function: a writable shared table is a
continuation oracle — guess a context, hash it, read what followed, append,
repeat. That is a text- and credential-extraction path.

Shared tags are the obvious next request after this change ("everyone doing rust
shares the rust table"). It is only safe with a read-only table populated from
deliberately published content, never from user traffic.

`one_users_table_is_unreachable_from_another` in
`crates/hipfire-specdecode-ngram/src/lib.rs` pins the current guarantee; extend
it to tags rather than replacing it.

### 6. Config and wire format

- `ngram_write_target` gains whatever (4) decides.
- Request field `session_type` → `session_tags` (array). Keep accepting a bare
  string as a one-element list for a release so existing callers do not break.
- `docs/config-schema.*` regenerate from the schema; do not hand-edit.

## Reference

- Implementation and measurements:
  `benchmarks/results/devlog_20260829_ngram_specdecode_replay_sweep.md`
- Crate: `crates/hipfire-specdecode-ngram/`
