# Goal: unattended overnight performance sweep (quality + throughput)

**Written 2026-09-01 for an unattended run.** Assume the operator is asleep. Every
decision this file does not make for you, make conservatively and write down why.

## Objective

Land as many of the twenty ranked improvements as survive contact with
measurement, in the order below. Improve one or both of:

- **quality** — PPL / KLD against a held-out reference
- **throughput** — TG (decode tok/s) and PP (prefill tok/s)

Fix any bug found along the way, including ones not on the list. A bug found and
fixed with a test is worth more than a planned item half-done.

Full ranked list with evidence:
`https://claude.ai/code/artifact/689b3e26-5672-489e-8d99-339806ac8514`

---

## Standing authorisation (granted by the operator 2026-09-01)

- **Run tests, gates, benchmarks and experiments freely**, including long GPU runs.
- **Full management of `~/.hipfire/`** — create, move, overwrite and **delete**
  files there, models included, without asking.
- **Commit freely** on a topic branch.

### Hard limits — these are NOT authorised

1. **Never delete anything under `/srv`.** It is a shared NFS mount at 89% full
   (835 G free) holding calibration packages and HF snapshots that are often the
   only copy and are expensive to rebuild. Read-only, always.
2. **Never `git push`, open a PR, or merge.** Commit locally; the operator
   integrates. No force-push, no history rewrite, no `git rebase` of anything
   already pushed.
3. **Do not work on `master`.** Branch first (see Step 0). The tree is currently
   dirty ON master — that is the first thing to fix.
4. **Do not delete a local model whose source you cannot name.** Before deleting
   `~/.hipfire/models/X.hfq`, confirm either (a) an HF snapshot or `.hfa` source
   exists on `/srv`, or (b) a `hipfire-quantize` command reproduces it, and
   record that command in the run log. If neither, keep it.
5. **Do not `pkill -f`.** It matches your own shell's command line and surfaces
   as a bare `exit 144`. Kill recorded pids, or use `scripts/pkill-safe.sh` **by
   path** (a Claude Code shell function shadows the PATH symlink).

### Disk

`/home` has 1.2 T free; `~/.hipfire/models` is 1.2 T. Space is not tight, so
delete for *hygiene*, not urgency. Safe to clear without ceremony:
`~/.hipfire/scratch/`, stale run dirs under `~/.hipfire/experiments/`, and
`/tmp/claude-*` scratch. Keep `~/.hipfire/calib/` — those are expensive and some
exist only locally.

---

## Measurement discipline — read before running anything

Three traps have each cost a previous session real time:

1. **`hipfire eval` caches on model + prompt + binary hash, NOT on environment.**
   An env-only A/B silently replays arm 1. **Identical numbers mean a cache hit,
   not a null result.** Pass `--force`.
2. **`rocprofv3`'s `top_kernels` view mis-scales durations.** Query
   `rocpd_kernel_dispatch` in `run_results.db` instead.
3. **Discard the first run after any rebuild, kernel edit, or new arm.** Kernels
   are JIT-compiled into `~/.hipfire/kernels/<arch>/` and the compile happens
   INSIDE the timed window. Measured: chain DFlash reads **6.41 tok/s** cold and
   **22.13** warm — **3.45x** — with tau bit-identical both times. That identity
   is the trap: acceptance metrics look healthy while throughput is a third of
   real. A control must be re-run in the same warm session as its treatment,
   never quoted from an earlier one. This one already cost this run a wrong
   headline.
4. **Never calibrate and evaluate on the same corpus.** A previous "−13.6%" result
   was train-on-test and had to be retracted. Keep the KLD reference disjoint.

**Locking.** Non-daemon GPU binaries do not self-lock — wrap examples and benches
in `hipfire lock acquire <label>` / `release`. But **never** wrap `hipfire-eval`,
`tests/tiny-*.sh`, or `coexistence calibrate`: they acquire it themselves and
will deadlock naming your own label as the blocker.

**Check the repo before analysing — this has now cost three items.** W1, W2 and
W3 each rested on a stale premise, and two of them were already answered in the
tree: W1's blocker in `docs/todo/2026-08-30-oq8g128-protected-set.md`, W3's fix
in commit `96d53741c`. This list was built from memory entries, and **the repo is
ahead of the memory**. First action on any remaining item: `git log --grep=<topic>`
plus a sweep of `docs/todo`, `docs/plans` and `BUGS.md`. Only then measure.

**Every performance claim needs a control.** Same binary, same prompt, same
artifact, both arms in the same session. A number without its baseline is not a
result.

---

## Step 0 — land what is already in flight (do this first, ~30 min)

The tree is dirty on `master`. Nothing else starts until this is clean.

```sh
git checkout -b perf/overnight-2026-09-01
```

1. **Commit the n-gram hoist** — `NgramState` moved off `DflashState` onto
   `LoadedModel`, plus `take_live_for` and its scope-leak test. Five files:
   `serving-core/src/{model,generate,load}.rs`,
   `daemon/src/handlers/lifecycle.rs`. Already builds, clippy-clean, 156 tests
   pass, `no-gpu-ci` exit 0.
2. **Commit `examples/serve_real.rs`** — optional step-count argument. Four
   decode steps cannot measure a paged model; the flag is what made the 64-step
   numbers possible.
3. **Correct and commit `docs/perf/ddtree-vs-chain-opus.md`.** It is wrong twice:
   - it calls Qwen3.6-27B "dense" — it has **48 linear-attention layers of 64**,
     so the hybrid-recurrent path *was* exercised;
   - it recommends budget 8 / topk 2. The real optimum is **budget 12, topk 1**
     (32.89 tok/s, τ 5.79, zero slow-path cycles), and topk=1 is a linear spine —
     no tree at all. Fold in the topk sweep and the
     `HIPFIRE_DDTREE_TAPE_DUMP=1` fast/slow counts.

Gate before moving on: `./tests/no-gpu-ci.sh` must exit 0.

---

## Baseline to establish once, then never re-derive

Record these in the run log before touching anything. Everything later is
measured against them.

⚠️ **All 27B figures below are WARM-CACHE, re-measured 2026-09-01 and repeated.**
Any earlier quote of chain at 10.26 tok/s is withdrawn (cold-cache artifact).

| what | command | value |
|---|---|---|
| 27B AR decode | `dflash_spec_demo --ar-baseline` | **15.25** tok/s (both passes) |
| 27B chain DFlash B=16 | `--block-size 16` | **22.1** tok/s, tau 2.49 |
| 27B best spec | `--ddtree-batched --ddtree-budget 12 --ddtree-topk 1` | **32.7** tok/s, tau 5.79 |
| 27B chain, COLD cache | (do not use — trap reference) | 6.41 tok/s |
| 180B paged decode | `serve_real <model> 64` | 0.25 s/tok @ 8 GiB |
| 180B eager decode | `HIPFIRE_QWEN4EXP_PAGED_EXPERTS=0` | 0.08 s/tok |

Artifacts: `~/.hipfire/models/Qwen3.6-27B--oq4.25++.hfq`,
`~/.hipfire/models/Qwen3.8-Flash-Next-180B--oq4.hfq`.

**Drafter gotcha, already paid for:** every `*-DFlash--bf16.hfq` on this box is
**arch 1, not arch 20**, so `DflashConfig::from_hfq` rejects all seven and no
DFlash path can run from them. Rebuild from HF source:

```sh
./target/release/dflash_convert \
  --input /srv/huggingface/models--z-lab--Qwen3.6-27B-DFlash/snapshots/*/ \
  --output ~/.hipfire/models/Qwen3.6-27B--dflash.bf16.hfq
```

**DONE 2026-09-01** — `~/.hipfire/models/Qwen3.6-27B--dflash.bf16.hfq` (3.46 GB,
arch 20) exists and is what the corrected numbers above were measured with.

---

## Work items, in execution order

Ordering rule: **highest confidence × lowest blast radius first**, so that a run
which stalls at 3 a.m. has still landed something. Each item ends with a commit
or an explicit "abandoned, because —" note in the run log.

### W1 · `is_batchable_la` is missing the G128 Opus dtypes  *(list #2)* — DEPRIORITISED

⚠️ **Groundwork landed (`d7ba9d27a`); the rest is NOT an overnight item.** The
premise below — "add two arms" — is wrong. `docs/todo/2026-08-30-oq8g128-
protected-set.md` already documents the real blocker: qwen35's lowered executor
fuses rmsnorm+rotate and shares ONE FWHT-256-rotated activation across every
attention projection, so a G128 weight needs a new G128 fused rotate kernel
family AND a way to split that shared activation. Multi-day kernel work.

What DID land is worth keeping: the batched scratch allocator sized the
activation-scale plane `n * (k / 256)`, which doubles at group 128 — a latent
overrun that doc does not cover. Now group-parametric, behaviour-neutral.

Also do not rediscover: no AWQ sidecar at G128, no LDLQ at G128, and tiny-quant
KLD **cannot adjudicate** G128 (6.9x better on one family, 2x worse on another,
on random-init fixtures). Use the reconstruction test.

Original framing, retained for context:

Batchable today: `Oq4G256`, `Oq8G256`, `OqCompactG256`. **No arm for
`OqCompactG128` or `Oq8G128`** — verified by reading the arms in
`crates/hipfire-arch-qwen35/src/qwen35/mod.rs`. The function's own comment: a
false here "sends a whole layer class down a per-token path with no outward sign
but a GEMV-heavy histogram."

- Add the two arms plus their grouped-WMMA routes, mirroring the `Oq8G256` arm.
- The mirrored copy in `crates/hipfire-runtime/src/dispatch.rs` has a
  matching-pair comment — **keep both in sync**.
- `kernel_trace::record_fallback` already fires on decline: use it to prove the
  fallback stops.

**Done when:** a G128 artifact's prefill tok/s rises with no KLD change, and the
decline trace is silent. **If no G128 artifact exists locally, make one** —
quantize a small model to `oq8` at group 128.

### W2 · Why does the tree path get tau 5.79 where chain gets 2.49?  *(list #1, REVISED)*

⚠️ **This item was void as first written.** It said "chain is 3.2x slower and
loses to AR (10.26 vs 15.23)". That chain number was a cold-kernel-cache
artifact. Corrected, warm, repeated:

| arm | tok/s | tau |
|---|---|---|
| AR | 15.25 | 1.00 |
| chain DFlash B=16 | 22.1 | 2.49 |
| DDTree budget 12, topk 1 | 32.7 | 5.79 |

**Both speculative paths beat AR.** There is no defect. The remaining gap is
1.47x and it is entirely acceptance.

The real question, and it is a good one: **block size is not the confound.**
Chain gives tau 2.4865 at BOTH B=12 and B=16 — identical accepted tokens (92),
with accept_rate falling 0.226 -> 0.166 as the extra proposals are wasted. So
chain's draft saturates at ~2.5 accepted regardless of how much width it is
given, while the tree path's spine reaches 5.79 from the same drafter.

- Find where the tree path's spine construction differs from chain's
  per-position argmax — `build_ddtree_tree_with_cutoff` prunes on confidence,
  chain proposes B-1 unconditionally.
- Do NOT re-try the DFlash2 candidate selector: already implemented, gated behind
  `HIPFIRE_DFLASH2_SELECTOR=1`, and MEASURED WORSE (tau 2.421 -> 2.25, decode
  6.14 -> 5.92).

**Done when:** the tau difference is explained. If chain can be made to truncate
on confidence too, that is the win; if not, the finding is that chain's fixed
block size is the wrong shape and DDTree-topk1 should be the default spine.

### W3 · KVarN × batched prefill  *(list #3)* — CLOSED, does not reproduce

⚠️ **Already fixed by `96d53741c` (2026-08-29), two days before this file listed
it as open.** Re-measured on `qwen3.5-2b--bf16.hfq`, warm:

| KV mode | batched | per-token | ratio |
|---|---|---|---|
| q8 | 2.057e-2 | 1.580e-2 | 1.30x |
| kvarn | 1.994e-2 | 1.607e-2 | **1.24x** (was 57x) |

That commit overturned both premises: the KVarN write path was never broken
(`kvarn_attend`'s segment-then-flush ordering fixed it; 3.31e-4 flat through the
128-token flush boundary), and the residual divergence is **fp16 narrowing
cadence in the DeltaNet state** — batched narrows S once, per-token narrows n
times. KVarN amplifies it, it does not cause it.

The two real fixes are costed in BUGS.md and are design calls (they touch the
spec-decode snapshot contract), not overnight cleanups.

**Vacuity note for whoever re-checks:** on a compact target BOTH arms fall back
to per-token and report identical numbers, which proves nothing. Only trust this
comparison when the two arms differ.

### W4 · Paged-expert per-access overhead  *(list #4)*

48 GiB budget with **zero** evictions (0.27 s/tok) is no faster than 8 GiB
thrashing 15670 evictions (0.25). So the 3× paged-vs-eager gap is fixed
per-access cost — ~480 mutex+hash+LRU ops per token (48 layers × top-10).

- Resolve the resident view **once per layer**, take the pager lock **once per
  layer**, not once per expert.
- `ExpertStack::expert()` in `crates/hipfire-arch-qwen4exp/src/moe_gpu.rs` is the
  site; it currently locks per call.
- Guard rail: the paged and resident paths must stay bit-identical. The
  `qwen4exp-gate` paged arm asserts this, including that cold-loads and evictions
  are non-zero — **do not weaken that assertion to make an arm pass.**

**Done when:** 180B paged decode improves measurably at the same argmax, and
`./tests/qwen4exp-gate.sh` passes.

### W5 · Cheap default flips, each with a KLD check  *(list #14, #12, #17, #16)*

These are results that already exist but sit behind an env var or a memory. Each
is a small diff; the work is the confirmation run, so do them together.

1. **`HIPFIRE_CALIB_SEQ_LEN=2048` as the default** — 10.1× faster calibration at
   the same quality (the capture is O(seq²)).
2. **Low-rank residual (LQER) into the config schema** — −13% at 2 bit; today
   only reachable via `HIPFIRE_LOWRANK_R`, so no production artifact gets it.
   Needs a default rank per bit-width.
3. **OQ8 MoE router as default** — currently `HIPFIRE_OQ8_ROUTER`. While there,
   **delete** the `"Q8: routers"` log counter: it is categorical and reports the
   category rather than what happened.
4. **Embedding stays 16-bit, as an enforced protected-set rule** — it rides the
   residual unnormalised, so an 8-bit embed costs ~40% of the whole oq4 KLD
   budget to save ~500 MB. Put the measurement in the comment.

Settings go in `crates/hipfire-config/src/schema.rs` and **all three of**
`docs/config-schema.{json,toml,md}` must be regenerated or CI fails. `HIPFIRE_*`
env is debug-only; a user-facing option is a schema field.

**Done when:** each flip has a before/after KLD on a held-out corpus in the log.

### W6 · Opportunistic, only if time remains

In preference order: **#18** GTT 2 MiB rounding on OQ blocks (they land 1.6% over
and pay 97%); **#19** pre-split compact planes in the repacker so page-in is a
memcpy; **#6** the oq8 GEMM at 54% of the 248.5 GB/s ceiling — note LDS staging
was already tried and lost (101–108 vs 139.9), so do not repeat it.

---

## Bugs found along the way

Fix them. Rules:

- A bug fix gets its **own commit**, separate from the feature work it interrupted.
- Every non-trivial fix leaves **one runnable check** that fails without it —
  verify that it fails, by reverting the fix and re-running. A test that passes
  both ways is worse than no test.
- Root-cause, do not symptom-patch: grep the other callers of the function before
  editing, and fix it where they all route through.
- If a fix is too large for the night, write it up in `docs/bugs/` with the repro
  and move on. **Do not leave a half-applied fix in the tree.**

Two already known, not yet filed:

- `examples/qwen35_hfq_modules repack` writes an artifact that fails to open —
  `HFQM v2 tail hash mismatch`. Not needed for paging, but it is the tool someone
  reaches for first.
- Every local `*-DFlash--bf16.hfq` is arch 1, not 20 (above). Decide whether to
  re-cut them under canonical names and delete the unusable ones — this is
  exactly what the deletion authorisation is for.

---

## Run log — required

Append to `docs/experiments/2026-09-01-overnight-sweep-log.md` as you go, not at
the end. One entry per attempt, **including failures** — a failed approach that
narrows the search space is a result worth keeping (repo invariant).

Each entry: what was tried · the command · the number · the control it is
compared against · what happened next.

Record every deletion under `~/.hipfire/` with the reproduce command.

## Stop conditions

Stop and leave the tree clean if any of these hit:

- `./tests/no-gpu-ci.sh` fails and the cause is not obvious within two attempts;
- a GPU run wedges twice in a row (check `hipfire lock status`; a stale holder
  line names a dead pid — `hipfire lock kill` clears it);
- free space on `/home` drops below 100 G;
- you find yourself re-running the same failing command a third time.

Leaving a clean tree with a written blocker beats leaving a broken one with a
guess.

## Morning handoff

Finish by writing a summary at the top of the run log: what landed, what moved
and by how much, what is still open, and — most usefully — **what you learned
that changes the ranked list.** The list is a hypothesis, not a work order; if
the measurements say item 6 is dead and something unranked matters more, say so.
