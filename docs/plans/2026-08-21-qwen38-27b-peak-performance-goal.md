# Goal: Qwen3.8-27B decode to near theoretical peak

Launch with: `/goal` + `/loop`, pointing at this file.

Target: **Qwen3.8-27B, oq4.25++ +CASK, gfx1151 / Strix Halo, 128 GB UMA.**
Raise decode throughput toward theoretical peak. Spend as much time as needed.

## Phase 1 — no speculative decoding

Get plain autoregressive decode to the memory bandwidth limit BEFORE touching
spec-decode. Mixing the two hides which one moved.

State as of 2026-08-21:

| | |
|---|---|
| dense decode now | **11.50 tok/s** (8.00 at session start) |
| MoE decode now | 52.10 tok/s |
| roofline @ compact residency | ~18.1 tok/s |
| measured pure-read ceiling | 248–250 GB/s |
| configured memory | 256 GB/s (8000 MT/s; modules rated 8532 = 273) |

### The known next lever — measured, not yet built

The **sparse overlay** in the compact decode GEMV is worth **25%**, not the 6%
an N_out sweep alone suggests. At the shipped N_out=3 on `down [5120, 17408]`:

    baseline                     178.3 GB/s
    activation loads ablated     174.5        (no gain — masked)
    overlay ablated              222.1        (+25%)
    both ablated                 228.4        (+28%)
    pure-read ceiling            248

6% is the MARGINAL cost of entries 2 and 3. Every configuration including
N_out=1 pays the loop's fixed cost — setup, broadcast byte loads, the divergent
owning-lane test, and the dependent scattered `X[idx]` load. Comparing N_out=3
against N_out=1 subtracts exactly that out. See commit `868568e0d` and
`crates/hipfire-rdna/examples/bench_oq_compact_nout.rs`.

Two more findings from the same ablations:

- **The N_out cliff is the loop, not the bytes.** With the overlay ablated,
  throughput is flat 222–228 GB/s across N_out 1→8.
- **The costs overlap.** Ablating activation loads alone gains nothing: the
  overlay's dependent chain is the critical path and X latency hides behind it.

### Designed fix

1. **Zero the bulk nibble at load time** in `normalize_compact_overlays`, so a
   correction becomes `val * X[idx]` instead of `(val - bulk) * X[idx]`. This
   severs the dependency on the owning lane's registers. Duplicate handling
   changes from "superseded val := bulk" to "loser's val := 0".
2. **Parallelize the entry scan** — lane `e` handles entry `e`. The n_ov serial
   dependent iterations collapse to one parallel step; the existing wave
   reduction sums the corrections, so they need not land in the owning lane.

Blast radius: both the compact GEMV and the compact GEMM read these blocks, so
both kernels, the normalizer, and their parity fixtures move together.

### After the overlay

Re-test the levers that previously LOST — **multi-row** and **re-alignment**.
They were measured against a kernel whose critical path was the overlay, so they
attacked a non-binding cost. Activation-load pressure is worth ~3% once the
overlay is gone. The residual 228 → 248 is decode ALU and weight-load shape.

## Phase 2 — speculative decoding

Only once Phase 1 is at or near the bandwidth limit. Layer on **DFlash,
DFlash2, DDTree, DSpark, and/or NGram / prompt-lookup decoding**. Measure each
independently before combining.

### RESULT (2026-08-21): the premise does not hold for this model

Phase 2 was executed. Spec-decode was made to RUN on compact Opus for the first
time and improved 90% (≈4.5 → 8.55 tok/s), but it does **not** beat plain decode
at 15.1, and the available fixes do not close the gap:

| step | spec-decode |
|---|---|
| starting point (parked, unmeasurable) | ~4.5 |
| batched verify | 5.56 |
| + multi-column compact GEMM (B ≤ 16) | 6.52 |
| + GDN-tape rollback replay | **8.55** |
| + restoring acceptance from the KV fork (projected) | **~14.4** |

**Even the complete fix lands below plain decode.** Restoring τ from 2.000 to
3.375 scales 8.55 to ~14.4 < 15.1. Beating autoregressive decode additionally
needs a drafter with higher τ or a much cheaper one — a training/architecture
task, not a kernel task.

**Why, structurally:** Phase 1 left decode at ~91% of the DRAM read ceiling with
1.05× overfetch and bytes already at the 4.25-bit floor. Spec-decode pays a
drafter (~64 ms/cycle here, 18% of the profile) to save weight sweeps the target
is already reading near-optimally. At 4.25 bits there is too little headroom for
that trade to pay. **Phase 1 succeeding is what makes Phase 2 fail.**

That is the answer to this phase, not an open task. The supporting work — a 2.8×
prefill, a bit-identical small-batch GEMM, and a correctness bug in the default
KV configuration — is in
`docs/plans/2026-08-21-compact-batched-prefill-blocker.md`.

Re-open Phase 2 if any of these change: a materially cheaper or higher-τ drafter,
prompt-lookup drafting (not implemented in this tree), or a lower-bit target that
restores headroom — which the 4.25-bit floor currently forbids.

## Latitude

You may change kernels, dispatch, runtime, the quantization itself, and you may
requantize the model.

**Hard floor: never below 4.25 bits per weight.**

## Measurement discipline

Hard-won. Violating these produces confident wrong answers — each of these cost
a wrong conclusion tonight.

- Kernel `.hip` sources are `include_str!`'d into the Rust binary. Editing a
  kernel **without a cargo rebuild measures the OLD kernel.** Two ablations
  reported "no effect" because nothing had changed.
- **An ablation that does not break parity did not run.** Verify the negative.
- Report **GB/s, not ms**, whenever a change alters bytes moved.
- Within-session A/B noise is ±10–15%. A single-run "+8%" is noise. Use
  `scripts/probe_commits.sh <baseline> <candidate>` for cross-process A/B, and
  interleave A/B runs.
- Never measure against your last bench run. Measure against the **committed
  baseline binary**.
- **Profile before attributing.** "Sequential DeltaNet is the bottleneck" was
  inferred from a code comment and was actually 0.7% of runtime.
- Use the `hipfire-kernel-tuning` skill and read its notes before reaching for a
  lever.

## GPU coordination

`hipfire lock {acquire,release}` around GPU benches and examples.

Do **NOT** wrap self-locking things — `hipfire-eval`, the `tiny-*` gates, and
`coexistence calibrate` all self-lock and will deadlock naming your own label as
the blocker.

## Verification

- `./tests/no-gpu-ci.sh` for workflow-only changes.
- `./tests/tiny-affected-gate.sh --require-coverage --base <ref>` as the GPU
  correctness front tier for runtime/quant changes.
- Parity examples for kernel changes.
- Keep RDNA2/3/4 portability: no unreachable branches, predicate on capability
  not arch id.

## Reporting

Document failures as well as wins — a lever that should have worked and didn't
narrows the search space, and belongs in the commit message with the hypothesis
for why. Commit incrementally on `perf/dram-read-bandwidth` with real measured
numbers in the messages.

**Correct your own earlier claims in place when measurement overturns them.**
That happened three times tonight and each correction was worth more than the
original claim.

## Subagents, workflows, teammates

Default for this run: **single-threaded, no subagents, no workflows.**

This is deliberate, not a formality. GPU work here is serialized by
`hipfire lock` — only one process holds the GPU at a time — so parallel agents
running benches would contend on the lock rather than finish sooner, and
concurrent GPU consumers make the ±10–15% noise band worse at exactly the moment
measurement discipline matters most.

If you want to override, say so explicitly in the launching prompt:

- **Read-only exploration subagents** are the safe subset — codebase search,
  reading kernels, drafting variants. They touch no GPU. Enable with a line like
  "you may use read-only Explore subagents for codebase search."
- **Workflows** need explicit opt-in regardless (the keyword `ultracode`, or
  asking for one in your own words). Not recommended here for the same
  lock-contention reason.
- **Never** run two GPU-touching agents concurrently.

## Final verification (post-merge with origin/master)

Re-taken after merging origin's `fix(qwen35): dense DeltaNet arm never applied
ffn_norm`, which lands in the hand decode path this branch also touches.

| measurement | result |
|---|---|
| dense decode, `--prefill 16` | **15.1 tok/s** (217.5 GiB/s), 2 reps, zero variance |
| dense decode, `--prefill 512` | **14.0 tok/s** (202.2 GiB/s), 2 reps, zero variance |
| `parity_gemv_oq_compact_multicol` | PASS (max 1.12e-7) |
| `parity_rmsnorm_rotate_awq_width` | PASS (0.00e0, exact) |

**State the context with the number.** The headline 15.1 is at prefill=16; at
prefill=512 the same binary does 14.0, because KV attention grows with context
while the weight stream does not. Comparisons to other runtimes are only
meaningful at equal context.

### tiny-affected-gate: 8 failing cells, NONE of them from this branch

The gate reports FAIL on this branch (qwen2/hfq4, gemma3/q8f16, gemma3/hfq4,
minimax/mq4, qwen3_5/q8f16, qwen3_5_moe/{q8f16,mq6,mq4}). All eight reproduce
**identically — to the last printed digit —** on clean `origin/master` in a
scratch worktree. `tests/tiny-quant-baselines.txt` is stale on master; this
branch adds no drift.

Two methodology notes, both mistakes made while checking this:

- `tiny-affected-gate.sh` SELECTS cells from the diff against `--base`. Running
  the branch with `--base origin/master` and master with `--base origin/master~1`
  compares **different cell sets**, and the resulting "8 vs 4" looked like four
  new failures that did not exist. To compare like with like, drive the
  underlying gate directly with `HIPFIRE_TINYQUANT_FAMILIES=<families>`.
- The first explanation reached for — that the `fused_rmsnorm_mq_rotate_awq`
  width change (256 -> 1024) shifted float reduction order — was refuted by the
  comment written in that same diff: the fixtures are hidden=256 and never reach
  the `k >= 2048` branch, so they give it zero coverage. Read your own notes
  before theorising.
