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

## 2. ~~n-gram topic tags — the write path~~ — DISSOLVED 2026-09-01

Not decided: **removed**. Every option on the table (write to all N tags, write
to the first, curate offline) made a HARD assignment at write time, and the hard
assignment was the thing generating the unrecoverable-misclassification problem.

The design that replaces it is a **dynamic mixture-of-experts** over the attached
stores, written up as §4b of `2026-08-29-ngram-topic-tags.md`. Each store is
weighted by how well it would have predicted the recent context
(`s_k <- lambda*s_k + log p_k(t|c)`, softmax to a posterior), and prediction is
the weighted mixture. Nothing is ever hard-assigned, so nothing can be
misassigned.

The realisation that collapses it: **the per-store likelihood is already the
detector.** `fn `, `) {` and an identifier shape are high-probability under a
Rust store and low under a prose store, so a keyword list — or a treesitter parse
— is a hand-rolled, strictly worse approximation of a number the stores compute
anyway. Treesitter was considered explicitly and is the wrong tool for the topic
half: it extends to new *languages*, not new *subjects*.

**Two real questions remain, and both are engineering rather than product:**

1. **Hot-path cost.** K lookups per proposal instead of 1, in latency-critical
   spec decode. Cap the stores, recompute the posterior every N tokens rather
   than per token, prune to top-2 for the proposal.
2. **Determinism.** History-dependent weights make drafting non-reproducible
   across sessions unless the posterior is seeded from fixed state. Much of this
   repo's testing rests on byte-identical replay, so decide whether the drafter
   must be reproducible.

Promotion is also specified there: replace the raw `ngram_spec_promote_count`
threshold with a Beta-Bernoulli lower confidence bound on measured ACCEPTANCE,
so a 1-for-1 gram stops outranking a 40-for-50 one.

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

## 7. ~~MoE routed experts — one allocation per expert, and who owns it?~~ — DECIDED 2026-09-01

**Per-layer arena.** Owner's call, made after re-measuring the premise rather
than accepting it: *does ROCm actually have to round to 2 MiB?*

It does — and that answer chose the shape. Measured live with
`gtt_granularity`: `down_proj` at 1 064 960 B consumes 2 092 958 (**1.965x**),
but a 128-expert arena at 408 944 640 B consumes exactly 408 944 640 (**1.000x**).
The granule is charged **per allocation**, so a large enough arena amortises it
away entirely.

That makes the pair shape the rejected option: on Qwen3.6-35B-A3B it saves
24.0 GiB where the arena saves **35.4 GiB** — 11.4 GiB more, landing on the raw
byte count exactly.

Design and remaining work: `2026-09-01-moe-expert-pair-allocation.md`.
Queued: `.agents/sessions/moe-expert-allocation/`.

**Still to solve in implementation, not a decision:** the indexed MoE kernels
read `expert_gate_up_ptrs` / `expert_down_ptrs` as independent addresses; under
an arena those become base + offset and must be measured against those kernels.
The `NonOwning`/`dispose()` guard still applies, but an arena is a simpler
lifetime — one owned allocation per layer that outlives every view by
construction.

---

## 8. ~~The `*-DFlash--bf16.hfq` drafters on `/srv` are unusable~~ — WITHDRAWN 2026-09-01

**Registered as a decision, then answered the same day. It is work, not a call.**

I framed it as delete-vs-re-cut, weighted by "six of seven have no matching local
HF source, so deleting may be destructive in a way no re-cut can undo". That
framing assumed the only route back was a reconvert from upstream.

It is not. The artifacts are **mislabelled, not structurally different** — 58
identical tensor names (including `fc.weight` and `hidden_norm.weight`), identical
geometry, and `config.dflash_config` already carrying `mask_token_id` and
`target_layer_ids`. Everything `DflashConfig::from_hfq` needs is already inside
each file, under different keys and behind an `arch 1` tag.

Two commands repair one, no upstream checkpoint required:

```sh
hfq rearch   src.hfq step1.hfq --arch-id 20
hfq meta-set step1.hfq out.hfq --key dflash --value-file dflash.json --json
```

Verified by output, not by loading: the repaired 27B drafter produced a
**token-for-token identical** draft stream and identical `decode_tau` (10.6667)
against one freshly built by `dflash_convert` from the same checkpoint.

So nothing needs deleting and nothing is at risk. Recipe and field mapping:
`docs/help/hfq-container-surgery.md`. The remaining six are queued as
`.agents/sessions/repair-dflash-drafters/`.

**What made this a false decision:** I registered it while the cheap route was
unexplored. A decision entry should assert that the alternatives were *tried*,
not merely that one looked expensive — otherwise the register accumulates
questions that a few minutes of work would answer, which is the same failure mode
as the memory-sourced ranked list this session spent the night correcting.

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
