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

### 4. The write path — SUPERSEDED 2026-09-01 by the mixture design below

The question in this section ("with N tags, where does a gram get written?") has
no good answer because it is the wrong question. Every option below makes a HARD
assignment at write time, and a hard assignment is what creates the
unrecoverable-misclassification failure. §4b removes the assignment instead.

Kept in full: the options and their costs are what make it clear why the mixture
is worth the lookup cost.

### 4a. The original write-path options

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

**Owner's position, 2026-09-01 — a fourth option: classify, then direct.** None
of the three above is satisfying. The shape that seems workable is a
**treesitter or classification model detecting code/topic**, with the n-gram
update routed to the matching store on that signal.

What that changes, relative to the three:

- It restores a **single write destination per gram**, which is the property that
  made option three attractive — without giving up online learning, which option
  three does give up by sourcing tag tables from curated corpora only.
- It moves the cost from write amplification to **classification on the write
  path**, and adds a dependency the n-gram store does not have today.
- It introduces a failure mode the others do not have: a **misclassified gram
  lands in the wrong table and is indistinguishable from a correct one
  afterwards**. Offline curation cannot misclassify, because the corpus is chosen
  deliberately. Worth deciding up front whether a wrong tag is recoverable —
  today nothing records why a gram went where it did.

**Narrowing that looks right:** if the split that actually matters is **code vs
prose**, treesitter alone is likely enough, and is the version to build first. A
parse either succeeds or it does not — a far stronger and cheaper signal than a
topic classifier, deterministic, no model dependency, and no inference on the
write path. A general topic classifier is a much larger commitment for a signal
that is fuzzier at exactly the boundaries that matter.

Still open. This is recorded as the direction, not a settled design; the write
path stays as it is until it is.

### 4b. The mixture design — no routing decision at all

**The per-store n-gram likelihood IS the language detector.** Tokens like `fn `,
`) {` or a `a..z*()` identifier shape have high probability under a Rust store
and low probability under a prose store. So there is nothing to hand-write: ask
each attached store "how well would you have predicted the last ~50 tokens?" and
weight it by the answer. A keyword list, or a treesitter parse, is a hand-rolled
and strictly worse approximation of a number the stores already compute.

Treesitter was considered and rejected for the topic half specifically: it is a
parser generator over formal grammars, so it can be extended to new *languages*
(markdown, LaTeX, and any code grammar) but not to new *subjects* — prose about
GPUs and prose about cooking have identical syntax. It would still be a fine
code-vs-prose and which-language signal, but the mixture gets that for free.

**What this is called.** A dynamic mixture-of-experts LM; in the older
literature, adaptive LM interpolation or a topic-mixture LM (Iyer & Ostendorf,
"Modeling long-distance dependence: topic mixtures vs. dynamic cache models",
compares exactly these two). The recency half is a cache language model (Kuhn &
De Mori 1990). The weighting has two equivalent readings — a Bayesian posterior
over a latent domain with exponential forgetting, or prediction with expert
advice (multiplicative weights / Hedge), which is the same update and comes with
regret bounds.

**The update.** Per store `k`, a log-weight with forgetting:

    s_k <- lambda * s_k + log p_k(t | context)   # lambda ~ 0.98 => ~50-token memory
    pi  <- softmax(s)                            # posterior over stores
    p(t|c) = sum_k pi_k * p_k(t|c)               # linear mixture, never a hard pick

Three details that decide whether this works:

- **Floor the weights**: `pi <- (1-eps)*pi + eps/K`. Pure multiplicative weights
  drive a store to zero and it never recovers when the context switches back —
  a real failure at a rust -> prose -> rust boundary.
- **Back off before mixing.** A store with no data for a context contributes its
  backoff/unigram, not zero, or an empty store poisons the mixture. Kneser-Ney or
  Witten-Bell per store; the mixture sits ABOVE the smoothing.
- **Linear, not log-linear.** Product-of-experts lets one confident store veto.
  Linear degrades gracefully and is what the literature uses for backoff LMs.

**Scoring and promotion.** `ngram_spec_promote_count` is a raw count threshold,
which has the classic flaw that a 1-for-1 gram outranks a 40-for-50 one. For a
speculative-decode drafter the quality metric is ACCEPTANCE, not frequency:
track `(proposals, accepted)` per gram and promote on a lower confidence bound —
`Beta(1 + accepted, 1 + rejected)`, promote when the 5th percentile clears the
bar (Wilson score is the cheaper closed form). Evidence then has to accumulate
before promotion, and a lucky single hit cannot outrank sustained performance.

The two compose into one ranking number:

    expected accepted tokens  ~=  pi_k * P_accept(gram)

**Costs, and they land in the hot path.** This is K store lookups per proposal
instead of 1, in latency-critical spec decode. Mitigations in order: cap the
attached stores (§2 already requires a cap), recompute `pi` every N tokens rather
than every token since it is smooth, and prune to the top-2 stores for the actual
proposal.

**Determinism is the open question.** A mixture whose weights depend on recent
history makes drafting non-deterministic across sessions unless `pi` is seeded
from a fixed state. Much of this repo's testing rests on byte-identical replay
(`tiny-state-gate`, the AR-hash controls), so decide up front whether the drafter
is required to be reproducible, and if so seed and record `pi`.

**What survives from §4a:** writes still need a destination, but it can be
provenance-keyed and exact — file extension and path, whether the text is a chat
turn or a file body — because provenance is known at write time and cannot be
misclassified. The mixture handles the read side, so a coarse or even wrong
write partition degrades quality rather than corrupting a table.

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
