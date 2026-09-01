# Decisions pending — todo items blocked on an owner call

Raised 2026-08-31 while working `docs/todo` unattended. Each item below is
blocked on a judgement that is not mine to make: a product semantic, a
compatibility break, or a scope/sequencing choice. Everything else in
`docs/todo` was either startable or closed.

The distinction used: an "open question" answerable by running an experiment is
NOT listed here — those are work, not decisions.

---

## 1. ~~SpinQuant R1/R2 — which path to a servable artifact~~ — WITHDRAWN 2026-09-01

**This was my error, not a real decision.** I grepped
`2026-07-01-spinquant-r1r2-future-work.md` for "Two clean paths (pick one)",
found it, and registered a fork. The surrounding line reads
`<details><summary>Original two-path plan (path (a) was chosen)</summary>` — the
choice was made long ago, and the section is collapsed history.

Explored 2026-09-01 and confirmed against the tree. **Path (a) is built and
serving:**

- `hipfire-train/examples/learn_r1_dump.rs` learns R1 and writes `HFR1` + `h:u32`
  + `h*h` f32, validating fp-invariance before it writes (max|Δlogit| 5.6e-5).
- `hipfire-quantize/src/rotate.rs` + `--rotate <M.r1>` consumes it — same magic,
  `h` checked against `hidden_size` — folds each RMSNorm into its readers and
  rotates residual readers/writers/embedding, installing the result as a global
  override so **every** codec branch quantizes rotated weights with no per-branch
  edit. Unit tests `r1_plan_is_orthonormal` and `bake_cancels_codec_fwht_leaving_m`
  (the bake proof) both live in `rotate.rs`.
- The doc records serving as DONE end-to-end, and a real Supra-50M -> 73 MB oq4
  artifact with R1 orthonormality 1.1e-6.

Path (b), the safetensors round-trip, was never needed and is not a live option.

**The genuine open item is R2 DEPLOYMENT, and it is a constraint rather than a
choice.** R2 *learning* is already DONE and measured (§1 of that doc: per-head
ctx-learned R2 beats the fixed rotation, +1.35). What blocks deploying it is
that the codec's 256-wide FWHT crosses head boundaries, so a per-head
`apply_r2` cannot be un-baked — the net transform would be
`F₂₅₆ ∘ blockdiag(R2)`, not `R2`. That needs a per-head codec variant, which is
separate work with no decision attached.

Worth knowing before anyone funds it: the same doc measured that on gfx1151,
**W4A4 ≈ W4A8 (~1.04–1.06x)** with equally-tuned kernels, and concludes W4A8
with no rotation is about as fast for far less risk. The SpinQuant throughput
payoff is marginal *on this APU*; revisit on CDNA/gfx12, which are core-bound.

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

**Owner's position, 2026-09-01:** no good method yet. The one shape that seems
workable is a **treesitter or classification model detecting code/topic**, and
routing n-gram updates to the respective stores on that signal — i.e. neither
"write to all N tags" nor "first tag only", but *classify, then direct*.

That reframes the question rather than answering it, and it is worth saying how:

- It makes the write path **single per gram** again (one classified destination),
  which is what made option three attractive — without giving up online learning
  the way "build tag tables offline from curated corpora" does.
- It moves the cost from write amplification to **classification latency on the
  write path**, and adds a dependency the n-gram store does not currently have.
  Treesitter is cheap and deterministic for the code/not-code split; a
  classification model is neither, and would want to be async or batched.
- It introduces a **new failure mode**: a misclassified gram lands in the wrong
  table and is indistinguishable from a correct one afterwards. Option three's
  offline curation has no equivalent, because the corpus is chosen deliberately.
- Treesitter alone may cover the case that motivated topics. If the split that
  matters is code vs prose, a parse either succeeds or it does not, and that is a
  far stronger signal than a topic classifier — and cheap enough for the write
  path.

**Still open.** Recorded as the owner's current thinking, not a decision. If it
proceeds, the narrow version — treesitter, code vs not-code, one destination per
gram — is the one that keeps the write path single and adds no model dependency.

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

## 6. Quant benchmark queue — restart the 35B pipeline, or drop it?

`2026-08-08-quant-benchmark-queue-handoff.md`

Re-checked 2026-08-31: the two "in flight" jobs are dead, their scratchpad and
scripts are gone, and every expensive Qwen3.5-35B-A3B output (bf16 43 GB, calib
1.8 GB, `oq4.25++` 17.9 GB) is absent from `/srv`. The source `.hfa` (47.7 GB)
survives, so it is restartable from §"Recipe" — but it is a restart, not a
resume, and the doc's "three stages banked" is no longer true.

That is hours of GPU on a box whose daemon is serving. Whether the benchmark
queue is still worth it is a priority call. Not restarted unattended.

The one durable result needs no rerun and is recorded in place: the per-expert
LDLQ factorization is irreducible for calibrated recipes (AWQ rebases the
Hessian per tensor, damping is applied after the rebase).

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
