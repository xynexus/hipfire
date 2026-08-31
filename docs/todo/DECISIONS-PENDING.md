# Decisions pending — todo items blocked on an owner call

Raised 2026-08-31 while working `docs/todo` unattended. Each item below is
blocked on a judgement that is not mine to make: a product semantic, a
compatibility break, or a scope/sequencing choice. Everything else in
`docs/todo` was either startable or closed.

The distinction used: an "open question" answerable by running an experiment is
NOT listed here — those are work, not decisions.

---

## 1. SpinQuant R1/R2 — which path to a servable artifact

`2026-07-01-spinquant-r1r2-future-work.md`

The bake math is proven and unit-tested; nothing writes a servable `.hfq` yet.

- **(a) `--rotate <M.bin>` in `hipfire-quantize`** — productionizable, but
  re-implements `apply_r1`/`apply_r2` host math inside the quantizer, because
  `hipfire-quantize` cannot depend on `hipfire-train`. ~1 new module wired into
  the per-tensor loop.
- **(b) rotated-safetensors round-trip** — no quantizer edits at all, heavier
  I/O (full-model round-trip), gets a first end-to-end artifact sooner.

The doc calls (a) the productionizable form and (b) good for a first artifact.
Both are implementable; this is sequencing, so it wants an owner.

## 2. n-gram topic tags — the write path

`2026-08-29-ngram-topic-tags.md`

With N attached tags, "write to topic" is ambiguous:

- write to **every** attached tag table — N-fold write amplification, and a gram
  learned in a `rust+gpu` session pollutes both single-topic tables;
- write to the **first** tag only;
- keep writing to the **user** table and build tag tables offline from curated
  corpora.

This decides what the stored data MEANS, so it is not a default I should pick.
The doc leans toward the third (keeps the write path single, keeps tag tables
clean, fits the "generate training data" motivation).

Secondary, same doc: probe order across tags (§3). Tags have no natural order;
the doc leans request order. Whatever is chosen must be stable within a session
or the per-tag marginal numbers are meaningless.

## 3. Coexistence flag spelling — a deliberate compatibility break

`2026-08-30-coexistence-subcommand-promotion.md`

`import gguf` spells its paths `--in`/`--out`; every other promoted command uses
`--input`/`--output`. Options: unify (breaks scripts that use the old spelling),
add a clap `alias` (accepts both, at the cost of two documented names for one
flag), or leave the split. The doc says explicitly this "wants its own
decision".

Note the promotion already made one behaviour change on purpose: `ArgBag`
silently ignored unknown flags, so `two-pass --typo 1` used to be accepted and
dropped; under clap it errors. Any script passing a flag the tool never read
will now fail loudly.

## 4. STEEL long-context NPU — productionize an IRON bridge?

`2026-07-15-steel-large-context-npu.md`

Step 5 is "decide whether a hipfire<->IRON bridge (author in IRON, compile
offline, dispatch via amdxdna) is worth productionizing for the long-context
path". That is a strategic commitment to a second authoring toolchain, not a
technical unknown.

## 5. n-gram cold store — merge on unload?

`2026-08-29-ngram-cold-store-merge-cadence.md`

Should unload trigger a merge? It would flush the backlog and leave the file
tidy for the next load; today `merge_backlog` is simply dropped with the
`NgramSpec`. Cheap either way, but it is a policy call about when work happens.

The crash-safety item raised beside it is NOT a decision — `merge` zeroes the
data region before rewriting, so a crash mid-merge leaves a partly zeroed body;
tmp-file + rename makes it atomic and is the pattern `hipfire-vision-cache`
already uses for its manifest. That one is just work.

---

## Go / no-go, rather than a design answer

Five documents are unapproved by their own status line, so starting them is a
scope decision rather than a technical one:

| doc | status |
|---|---|
| `bug-reporting.md` | "idea, not yet approved for build" |
| `npu-gpu-heterogeneous-prefill.md` | "research idea, not yet approved for build"; also depends on prereqs that do not exist |
| `kvarn-hot-bitwidth.md` | "idea. Owner: KV" |
| `merge-sidecar.md` | "idea. Owner: KV" |
| `2026-07-22-embedding-quant-improvements.md` | "open / research" |
