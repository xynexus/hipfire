# Compact-resident Opus was excluded from batched prefill — and that is why
# spec-decode has never won on this family
#
# STATUS: FIXED for the dense DeltaNet + FullAttention path (commit
# `feat(qwen35): batched prefill for compact Opus`). Prefill 15.1 -> 24.5 tok/s
# at 128 tokens, decode unchanged, generated text character-identical to the
# per-token path. The scope estimate below turned out to be wrong in the useful
# direction — read the RESOLUTION section at the end before the estimate.

Found while opening Phase 2 of
`docs/plans/2026-08-21-qwen38-27b-peak-performance-goal.md`. It is a Phase 2
blocker, not a Phase 1 one, and it is structural rather than a tuning miss.

## The observation

`prefill_tok_s` on Qwen3.8-27B oq4.25++ is FLAT in the prompt length:

    prefill = 1     14.5 tok/s
    prefill = 8     15.5
    prefill = 32    15.5
    prefill = 128   15.4
    prefill = 512   15.3

Processing 512 tokens costs 512x processing one. Decode on the same build is
15.1 tok/s, so prefill and decode run at the SAME per-token rate — a batched
prefill is doing nothing a loop would not.

rocprofv3 on a 128-token prefill says exactly what is happening:

    gemv_oq_compact_grouped_v3    63986 calls     (= 496/pass x 129 passes)
    gemm_oq_compact_grouped_wmma      0 calls

The batch=1 decode GEMV runs the whole prefill, token by token. The batched
compact GEMM is never dispatched — even though it exists and
`parity_gemm_oq_compact` passes on it.

This is not a bench artifact: `bench_qwen35_speed` calls
`qwen35::forward_prefill_batch`, the real entry point.

## Why it matters far beyond prefill

**Speculative decoding verifies K draft tokens on the batched prefill path.**
The entire premise of spec-decode on a bandwidth-bound decode is that verifying
K tokens reads the weights ONCE, so K accepted tokens cost about one weight
sweep. If verify runs per-token, K tokens cost K sweeps and spec-decode cannot
win by construction — it can only add drafter overhead.

That is the explanation for artifacts already sitting in `~/.hipfire/drafts`:

    Qwen3.8-27B--dflash.oq4+.hfq.parked-slower-than-plain-decode
    Qwen3.8-27B--dflash2.oq4+.hfq.parked-slower-than-plain-decode

and for the recorded DFlash2 result of 4.27 tok/s against 7.50 plain decode.
Those were read as drafter-quality problems. They are not. The verify path was
never amortizing, so no drafter of any quality could have won.

Phase 1 makes this WORSE, not better: the target decode is now 31% faster, so
the bar spec-decode must clear went up while verify stayed per-token.

## Root cause

`qwen35::prefill_batch_pbs_eligible` -> `is_batchable_la` (`qwen35/mod.rs`)
admits, for every layer projection:

    MQ4G256, HFQ4G256, MQ6G256, HFQ6G256, Q8_0, ParoQ4G128,
    F32, F16, BF16, and MQ3G256 on WMMA archs

The **entire Opus family is absent** — `OqCompactG256`, `OqCompactG128`,
`Oq8G256`. One ineligible projection dtype drops the whole model to the
per-token fallback in `forward_prefill_batch_with_pbs_opts`.

Confirmed directly with the runtime's own diagnostic:

    HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1
    [prefill-eligible] final=false base=false kv_f32=false kv_asym2_tree=false
                       dn_quant=FP32 n=32 kv(q8=true ...)

`base=false` with KV and DeltaNet state both acceptable — the dtype list is the
only failing term. KV mode is NOT the gate: asym3 / q8 / kvarn all measure
15.2-15.5 tok/s prefill.

## What wiring it actually costs

Admitting the dtype alone is a two-line change and it makes eligibility pass —
but the forward then fails LOUDLY and correctly:

    compact-resident Opus (OqCompactG256) reached a KernelKey GEMM fallthrough
    with no compact arm; it would be decoded as another format on an unrotated
    activation.

That guard (`run_plain_gemm_key`, `qwen35/mod.rs`) is doing its job: the
fallthrough key is an HFQ4 one, and the same missing arm means the rotation
admission list never rotated the activation either. Silent corruption on both
counts, refused.

Scope of the real fix:

- **38** `run_plain_gemm_key` call sites and **32** `run_residual_gemm_key`
  call sites, each a dtype match chain needing a compact arm.
- The **fused QKVZA** and **fused gate+up** kernels are not plain GEMMs and have
  no compact equivalent at all — they need new kernels and new dispatch table
  entries, not just an arm.
- Compact weights need the FWHT rotation applied to the batched activation;
  `dense_session_prefill_gemm_full_precision` already shows the recipe
  (`rotate_x_mq_batched_for` then `gemm_oq_compact_act_batched`), so the
  per-call-site pattern is known.

**That scope estimate was wrong.** It counted every call site in the file,
including the MoE chains a dense model never reaches, and it assumed the compact
GEMM entry points did not exist. They did:
`gemm_oq_compact_residual_act_batched` and `gemm_oq_compact_act_batched` were
already implemented and unused. The dense path needed SEVEN arms, each a
one-for-one mirror of the Oq8 arm beside it, plus one new helper
(`gemm_oq_compact_grouped_prequant`) so gate+up and q/k/v share a single
activation quantize. Walking the loud guard one site at a time made it safe to
do incrementally.

## Expected payoff

- Prefill: currently ~15.5 tok/s at any length. A 2048-token prompt takes over
  two minutes. The batched path on other dtypes reaches 1646 tok/s on
  qwen3.5-0.8b bf16 (recorded in `is_batchable_la`'s own comment), so the
  headroom here is one to two orders of magnitude, not percent.
- Spec-decode: becomes possible at all. Only once verify amortizes is it worth
  evaluating JetSpec / DFlare / DART / SSD, all of which assume K-token verify
  costs about one weight read.

## RESOLUTION (same day)

DONE for the dense DeltaNet + FullAttention path:

    prefill 128    15.1 -> 24.5 tok/s   (+62%)
    prefill 512    14.8 -> 23.6         (+59%)
    decode         70.67 -> 70.74 ms/token — unchanged

Greedy generation through the daemon is CHARACTER-IDENTICAL between the batched
and per-token paths; parity_gemv_oq_compact and parity_gemm_oq_compact pass; MoE
compact is untouched (its FFN admission list keeps it on its own path).

Prefill plateaus near ~24 tok/s rather than scaling with N because the DeltaNet
recurrence is still sequential per token. Batching amortizes every NON-DeltaNet
kernel; a chunked/parallel-scan DeltaNet is what would lift the plateau.

KNOWN INTERACTION, measured, unexplained: with the opt-in two-stage lm_head ALSO
enabled, running the batched prefill costs decode 6.4% (65.86 -> 70.08
ms/token). It is allocation ORDER, not the prefill work — with two-stage off the
two paths measure 67.35 vs 67.42. Padding the 318 MB coarse tier to a 2 MiB
boundary does NOT move it, so it is physical placement on this UMA APU rather
than base alignment.

## Still open

1. MoE routed/shared-expert chains still have no compact arms — MoE compact
   models keep the per-token prefill.
2. Fused compact QKVZA and gate+up kernels still do not exist; the dense path
   uses the unfused arms, which is why prefill gains 60% and not more.
3. The two-stage-lm_head allocation interaction above.
4. Spec-decode is now worth re-measuring for the first time: re-run
   DFlash/DFlash2 and consider unparking the drafts. Note the bar moved — dense
   decode is 15.1 tok/s now, not the 7.50 those numbers were taken against.


## PHASE 2 MEASURED (2026-08-21, after the prefill wiring)

Spec-decode now RUNS on this target and family for the first time
(`HIPFIRE_DFLASH_ALLOW_OPUS=1`, DFlash2 drafter, KVarN):

    accept_rate 0.482   tau 3.38   decode 4.45 tok/s

against plain decode at **15.1 tok/s** — so spec-decode is still 3.4x SLOWER,
and the parked label on the drafter remains accurate. But the drafter is NOT the
problem: tau 3.38 means it is genuinely predicting three-plus tokens per cycle.
`HIPFIRE_SPEC_PHASES=1` says where the time goes, per cycle at B=8:

    draft    52,000 us
    ngram     1,700
    verify   613,000        <-- 76.6 us per token x 8: EIGHT SEQUENTIAL DECODES
    replay   68,000-473,000 (7 of 8 rollbacks took `replay_full_prefill`)
    total   670,000-1,146,000

**Verify costs one full weight sweep PER DRAFT TOKEN.** That is the whole
result: 613 ms / 8 = 76.6 ms is exactly a decode step, so the verify is not
batching at all. The same is true of rollback — `replay_gdn_tape=0` and
`replay_full_prefill=7`, so every rejection re-runs a full prefill instead of
replaying the cheap tape.

Both are the SAME missing piece as the prefill blocker, one level down: the
compact chain has GEMM arms for an ordinary forward, and no writer for the
per-position state that verify and tape-replay consume. The prefill fix
deliberately makes compact DECLINE whenever the forward must export that state
(see `allow_compact`), which is what keeps spec-decode correct today — at the
cost of leaving it per-token.

### What fixing it is worth

With verify batched, 8 draft tokens cost about ONE sweep (~70 ms) instead of 613:

    cycle = 52 (draft) + 70 (verify) + small  ~= 125 ms for tau 3.4 tokens
          ~= 27 tok/s, against 15.1 plain -> ~1.8x

Add cheap tape replay (removing the 270-473 ms full-prefill rollback) and the
draft's own 52 ms becomes the next target. This is the first time the ceiling
has been an arithmetic estimate off measured phases rather than a guess.

### Order of work, revised

1. Compact writer for the GDN tape + hidden ring buffer, so verify and rollback
   can batch. This is the single highest-value item in the tree right now.
2. Then re-measure DFlash2 and unpark the drafters.
3. Only then evaluate JetSpec / DFlare / DART / SSD — every one of them assumes
   K-token verify costs about one weight read, which is precisely what item 1
   buys.

## Why spec-decode cannot win with per-token verify — a proof, not an estimate

Let B be the draft width and tau the accepted tokens per cycle. With verify
running one full weight sweep PER DRAFT TOKEN, a cycle costs B sweeps (plus the
draft) and yields tau tokens, so throughput is `tau / B` tokens per sweep against
plain decode's `1`. Since `tau <= B` always, spec-decode is at best equal and in
practice far worse. Measured here: tau 3.38 at B=8 gives 0.42 tokens/sweep, i.e.
4.45 tok/s against 15.1.

No choice of B, drafter, or acceptance rate escapes this. Batching the verify is
not an optimisation of the current design — it is the precondition for the
design to make sense at all. That is why the four candidate methods (JetSpec,
DFlare, DART, SSD) cannot be evaluated on this family yet.

## Where the remaining bug actually is

Admitting compact to the hidden-EXPORTING forward reproducibly breaks the
drafter (accept_rate 0.468 -> 0.000, random vocab ids with a repeating cycle).
The obvious suspects were ruled out by inspection and are NOT the cause:

- all five `hidden_rb` per-layer captures read `pbs.x_batch` AFTER the layer
  completes and sit outside every dtype arm;
- `per_token_hidden_out` is a plain `rmsnorm_batched` over `pbs.x_batch`,
  likewise dtype-independent;
- both ring-commit sites (`prefill_batch.rs` chunk loop, `speculative.rs` graph
  path) handle their own `n`;
- batched-vs-per-token prefill produces CHARACTER-IDENTICAL text, which means
  `pbs.x_batch` is correct at every position, not just the last — each
  position's output feeds the KV the next one reads.

So the defect is subtler than a missing arm: something about chunking, staging
size, or ring-head alignment on the seed path.

THE OBVIOUS DIAGNOSTIC DOES NOT WORK, and the trap is worth naming. Dumping with
`HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1` and diffing batched against per-token appears
to show divergence from LAYER 0 at ~5.0 relative error. That reading is FALSE:
the two paths dump under DIFFERENT TAGS — `{prefix}.batched.L{i}` versus
`{prefix}.pertoken.L{i}` — which are different call sites capturing different
quantities, so the comparison is apples-to-oranges. The giveaway is that it
contradicts a stronger measurement: batched and per-token prefill generate
CHARACTER-IDENTICAL text, which cannot be true if layer 0 diverged by 500%.

Likewise `{prefix}.fnorm` is emitted once per forward (the last token) unless
per-token hidden output is requested, and in the batched path it reads a scratch
buffer that path never fills — it dumps 5120/5120 NaN. That NaN is a DUMP
artifact, not the model state; the same run's text is correct.

A valid comparison has to capture the SAME buffer at the SAME point in both
paths — most directly `per_token_hidden_out` itself, which is what the drafter
actually consumes. That instrumentation does not exist yet and is the real next
step.

Until then compact declines the exporting forward (`allow_compact`), which keeps
spec-decode correct at its historical numbers and costs only the verify batching
that was never working anyway.

## ROOT CAUSE OF THE DRAFTER BREAKAGE — it is not a compact bug

Built the valid comparator the previous note called for
(`hipfire-runtime` example `compare_prefill_hidden_paths`): both paths, ONE
process, the SAME `HiddenStateRingBuffer` the drafter actually consumes, diffed
layer by layer and position by position.

    Qwen3.8-27B oq4.25++ (compact, probe-forced to batch)
        diverges from LAYER 0, worst |rel| 6.40

    CONTROL — qwen3.5-2b bf16, a dtype that has ALWAYS batched,
    with no compact code involved anywhere
        diverges from LAYER 0, worst |rel| 0.93

**The control diverges too.** The batched and per-token prefill paths export
different hidden states for EVERY dtype. This is not something compact broke; it
is a pre-existing property of the two paths, and it is already documented in this
tree — `is_batchable_la`'s own comment records it for BF16:

> the batched path is not numerically identical to per-token. Typical
> |delta logit| is ~6e-2 (max 2.4e-1) against ~4e-6 for pure reordering, and only
> 15% of positions keep the same top-256 set … most likely q8 KV scales taken
> per-tile in the batched attention versus per-token in the fallback.

So the DFlash breakage is fully explained: compact models NEVER batched prefill,
so their drafters have only ever seen the PER-TOKEN capture. Switching compact to
batched handed them the other one, and acceptance collapsed to zero — not because
either forward is wrong (generation is character-identical) but because the
drafter is sensitive to which capture it receives.

That makes the shipped gate exactly right rather than merely conservative:
compact declines the hidden-EXPORTING forward, so drafters keep the capture they
were built against, while ordinary prefill still batches.

CAVEAT on the two numbers: 6.40 vs 0.93 are different models (27B vs 2B) at
different widths, so they do NOT establish that compact diverges more than bf16.
The load-bearing claim is only that the control diverges AT ALL.

### What this changes about the roadmap

Batching the verify is still the precondition for spec-decode (the tau/B proof
above is unaffected). But the work is no longer "make compact export hidden
correctly" — it is:

1. Decide which capture is canonical, and make the two paths agree, OR
2. Retrain/validate the drafters against the batched capture.

(1) is the better target because it also removes a documented,
silently-accepted numerical fork between the two prefill paths that affects
every dtype — not just this family.

## The divergence is ENTIRELY KV quantization — and KVarN is the worst case

Same comparator, same model (qwen3.5-2b bf16), 48 positions, only the KV tier
changed:

    KV = fp32     IDENTICAL across all layers      worst 0.00e0
    KV = q8       diverges from layer 0            worst 1.72e-2
    KV = kvarn    diverges from layer 0            worst 9.34e-1

With UNQUANTIZED KV the batched and per-token prefill paths agree EXACTLY. So
the fork is not in the GEMM arms, the chunking, the ring, or anything dtype-
specific — it is the KV tier, precisely as `is_batchable_la`'s comment guessed
("per-tile vs per-token q8 KV scales"). That guess is now a measurement.

TWO CONSEQUENCES WORTH ACTING ON.

**KVarN is ~54x worse than q8 here** (9.3e-1 vs 1.7e-2). KVarN is the sanctioned
default family and q8 is deprecated, so the default KV choice moved the
prefill-path fork from "small" to "large". Any DFlash drafter validated under q8
KV is being handed a substantially different capture under KVarN. That is a
testable hypothesis — re-measure drafter acceptance across KV tiers — and it may
already be costing acceptance on models that DO batch prefill, independently of
anything compact.

**The fix for spec-decode is now specific**: make the batched KV write/read use
the same quantization granularity as the per-token fallback. Then the paths
agree, the batched verify exports the capture the drafter expects, and the tau/B
proof above stops binding.

NOT a usable workaround: running fp32 KV to dodge it. The F4 guard in
`forward_prefill_batch_with_pbs_opts` forces per-token fallback for f32 KV
("F32 KV has only BatchEq(1) -> MissingImpl at resolve"), so the one tier where
the paths agree is also the one tier that cannot batch.

## Batched verify MEASURED — it is necessary but not sufficient

Forced the small-B verify to batch (`HIPFIRE_PROBE_COMPACT_HIDDEN=1`) on
Qwen3.8-27B oq4.25++ / DFlash2, 32 tokens:

    verify per-token, q8 KV    4.49 tok/s   tau 3.375   accept 0.482
    verify BATCHED,   q8 KV    4.84         tau 2.000   accept 0.286
    verify BATCHED,   kvarn    4.81         tau 2.000   accept 0.286

TWO results, both needed to plan the real fix.

**Batching the verify buys only +7.8%, not the ~8x the arithmetic predicts.**
Compact's batched path amortizes about 1.6x (prefill 15.2 -> 24.5) where bf16's
amortizes about 5x (2B: 20.8 -> 104.7). The compact arms added here are UNFUSED —
separate GEMMs plus residual adds, mirroring the Oq8 arms — while the admitted
dtypes have fused QKVZA and gate+up kernels. So the verify still costs ~5 sweeps
rather than ~1.

**Acceptance drops 0.482 -> 0.286 (tau 3.375 -> 2.000)**, which is the KV-tier
prefill-path fork above, now priced end-to-end. Note it costs the same under q8
as under kvarn even though the hidden divergence differs 54x — so acceptance is
not simply proportional to that norm.

Net: spec-decode remains ~3x slower than plain decode either way, and the earlier
tau/B proof is not the whole story. THREE things must land together:

1. Fused compact QKVZA + gate+up prefill kernels, so the batched path amortizes
   like bf16's rather than 1.6x.
2. Match batched KV write/read granularity to the per-token fallback, so
   acceptance survives batching.
3. Then re-measure. With verify at bf16-like amortization (613 -> ~123 ms) and
   tau preserved at 3.375, a cycle is ~175 ms => ~19 tok/s against 15.1 plain,
   and the draft's own 52 ms becomes the next target.

Doing (1) or (2) alone is measurably not enough, which is the point of recording
both numbers.

## FINAL Phase 2 accounting — why spec-decode still loses, precisely

After the N-blocking fix (compact GEMM 42x -> ~5x redundant, prefill 15.2 -> 42.4
tok/s), spec-decode was re-measured with the verify forced to batch:

    verify per-token   4.52 tok/s   tau 3.375   accept 0.482
    verify BATCHED     5.56         tau 2.000   accept 0.286   (+23%, was +7.8%)

Still 2.7x below plain decode's 15.1. The phase breakdown says why:

    B=8 cycle: draft 64,000us  verify 511,000us  replay 203,000us

**verify = 511 ms for 8 tokens = 64 us per draft token, against plain decode's
66 us.** Verify is barely cheaper PER TOKEN than simply decoding, even batched
and even with the faster GEMM.

The reason is structural and is the real answer to Phase 2: the WMMA GEMM tiles
B by 16, so at spec-decode's verify width (B = K+1, typically 8) it computes a
16-wide tile for 8 useful columns AND has no N to amortize the int4 decode +
overlay scan over. **Spec-decode's batch size is exactly where this kernel is
worst.** N-blocking helps prefill (B in the hundreds) enormously and the verify
barely at all, which is why prefill went 2.8x and spec-decode went 23%.

### The fix, and an attempt at it that FAILED

The right kernel for B <= 16 is not the WMMA GEMM but a MULTI-COLUMN GEMV: read
each weight row once, exactly like the decode GEMV, and accumulate B columns
against it. Weight traffic becomes ONE sweep for all B tokens instead of one per
token. With verify at ~one sweep (~66 ms) and tau 3.375:

    cycle ~= 64 (draft) + 66 (verify) = 130 ms for 3.375 tokens ~= 26 tok/s

which would finally beat plain decode's 15.1.

I wrote that kernel (`gemv_oq_compact_multicol`, modelled on the v3 decode GEMV,
consuming the same int8 activation plane as the WMMA GEMM so it drops in at
small B) and it FAILED parity on every G=256 shape — including ones with no
ragged tail and B=1, so a fundamental error rather than an edge case. Two real
bugs were found and fixed along the way and are worth knowing for the next
attempt:

  * `ng` need not be a multiple of 4 (K=256 gives ng=1), so lanes with gir>0
    read past the end;
  * with 8 lanes per group, only overlay entry `lig` is applied — N_out > 8
    needs the striding residual loop the v3 kernel has.

Neither was the root cause. The kernel was REVERTED rather than shipped broken;
the tree is green (parity_gemm_oq_compact and parity_gemv_oq_compact PASS). The
design above is sound and the payoff is quantified — it needs someone to find
the remaining defect with a numerical single-shape dump rather than by reading.

## Where the spec-decode time ACTUALLY is, after the multicol kernel

`gemv_oq_compact_multicol` (bit-identical, B <= 16) took the verify from 613 ->
407 ms and spec-decode from 4.49 -> 6.87 tok/s. Profiling the run then shows the
verify GEMM is no longer the problem:

    gemm_oq_compact_grouped_wmma   304 calls   419.6 ms   33%   <- SEED prefill
    gemv_oq_compact_grouped_v3    2523        320.1      25%   <- ROLLBACK re-prefill
    gemm_qkvza_hfq4g256_wmma        96        225.4      18%   <- the DRAFT model
    gemv_oq_compact_multicol       193        193.4      15%   <- verify, ~48 ms/cycle

**The verify GEMM is now 48 ms per cycle.** The remaining costs are the DRAFT
(~64 ms/cycle) and ROLLBACK, which re-runs a full target prefill per rejection —
2523 GEMV calls is roughly five whole target forwards. `replay_gdn_tape=0`
throughout: the cheap tape replay never engages.

### The arithmetic now favours spec-decode, if replay is fixed

    cycle = 64 (draft) + 48 (verify) + ~0 (tape replay) = 112 ms
    at tau 2.00 (today, KV-fork-degraded)  ->  17.9 tok/s
    at tau 3.375 (KV fork fixed)           ->  30.1 tok/s

against plain decode's 15.1. So for the first time the cycle budget CLEARS the
bar — the blocker is no longer the GEMM or the tau/B arithmetic, it is that
every rejection pays a full re-prefill.

### What is still gating the tape

`dflash_use_gdn_tape_replay(caller_supplied_tape, verify_populates_tape)`. The
demo DOES allocate a `GdnTape`, and forcing `verify_populates_tape` true (probe)
still leaves `replay_gdn_tape=0`, so a further term gates it —
`kv_batched_capable` or one of the eight conditions inside
`prefill_batch_pbs_eligible`. That is the next thing to isolate, and it is a
one-predicate question rather than a kernel project.

Probes used for these measurements were reverted; `HIPFIRE_PROBE_COMPACT_HIDDEN`
remains (documented, measurement-only, default off).

## The last term: rollback replay is disabled REPO-WIDE, not by compact

`replay_gdn_tape=0` is not a compact problem and not something this branch
introduced. `dflash_force_serial_rollback_replay` defaults ON for every model:

> Conservative default while proving rollback parity: replay committed tokens
> through the same serial target path as AR. Fast GDN-tape replay remains
> diagnostic-only because one-step production replay and multi-step fast tape
> rows still lack parity.

`HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY=0` opts out. Measured with it off (plus
the compact tape/hidden probes so compact can populate the tape at all):

    serial replay (default)   replay_full_prefill=1  8.55 -> n/a   6.89 tok/s
    GDN-tape replay           replay_gdn_tape=1                    8.55 tok/s   +24%

That default was NOT changed here. It is a documented correctness-pending
decision about rollback parity, and flipping it is the tape-parity project's
call, not a perf change.

## Session progression, and what is still missing

    spec-decode   4.49 -> 5.56 -> 6.52 -> 6.87 -> 8.55 tok/s   (+90%)
                  batched   multicol   tape-probe   tape replay
                  verify    verify

against plain decode's 15.1. Every remaining term is now identified and priced:

1. **Acceptance is halved** — 0.482 -> 0.286, tau 3.375 -> 2.000 — by the
   KV-tier prefill-path fork. Restoring it alone scales 8.55 by 3.375/2.000 to
   ~14.4, which still does not clear 15.1 but is within noise of it.
2. **The draft costs ~64 ms/cycle** (`gemm_qkvza_hfq4g256_wmma`, 18% of the
   profile). At tau 3.375 that is ~19 ms per accepted token against the target's
   66 — the drafter is no longer negligible once the target side is fixed.
3. Verify is ~48 ms/cycle and no longer the binding term.

So spec-decode on this family needs BOTH the KV-granularity fix and a cheaper
drafter to beat autoregressive decode. That is a materially different conclusion
from where this started ("the drafters are bad" / "verify cannot amortize"), and
every number above is measured rather than modelled.

## Two measurement artifacts RULED OUT, and the honest verify accounting

Before concluding that spec-decode loses, two things that would have made the
number unfairly pessimistic were checked and neither applies:

- **Seed prefill amortization.** The seed is 419 ms and 41% of a short run's
  kernel time. But `decode_tok_s` is decode-only: --max 32 and --max 128 both
  report 8.54/8.53, so the seed is already excluded. The number is fair.
- **Demo-only overhead.** `dflash_spec_demo` runs a rollback PARITY CHECK
  (`rollback_parity: checked=...`), which shows up as ~1032 extra
  `gemv_oq_compact_grouped_v3` calls, roughly two whole target forwards. That is
  diagnostic cost this harness adds and production would not pay, so the true
  production figure is somewhat BETTER than 8.55 — but not by the 1.8x needed.

With tape replay on, the cycle is:

    draft 64,500us   verify 370,683us   replay 24,470us   total 463,203us

Replay is solved (203 -> 24 ms). Verify is 370 ms while `gemv_oq_compact_multicol`
accounts for ~48 ms/cycle, so the remainder is the rest of the verify forward plus
the demo's parity check, not the compact GEMM.

## FINAL STATE

    dense decode    11.50 -> 15.1 tok/s   (Phase 1, at the bandwidth limit)
    dense prefill   15.2  -> 42.5         (2.8x)
    MoE decode      52.10 -> 54.8
    spec-decode     ~4.5  -> 8.55         (+90%, still 1.8x BELOW plain decode)

Spec-decode does not beat autoregressive decode on this family. The remaining
terms are measured, not modelled: acceptance halved by the KV-tier prefill fork
(restoring tau 3.375 scales 8.55 to ~14.4), and the drafter at ~64 ms/cycle.
Both must land; neither alone clears 15.1.

## SEPARATE BUG FOUND: batched prefill + KVarN is badly wrong

Not a compact issue and not a spec-decode issue — this affects every model that
batches prefill with the sanctioned default KV family.

Unquantized KV is the reference (both prefill paths agree EXACTLY there), so
`compare_prefill_hidden_paths` now measures each quantized arm against it.
qwen3.5-2b bf16, 48 positions:

    KV = q8      batched 1.615e-2   per-token 2.392e-2
    KV = kvarn   batched 9.334e-1   per-token 1.633e-2

Under q8 both paths are faithful and the batched one is slightly BETTER. Under
KVarN the batched path is **57x worse than per-token**, while per-token KVarN is
as accurate as q8. That is not a granularity difference — per-token KVarN proves
the tier itself is fine — it is a defect in the BATCHED KVarN attention.

Reproducible across sizes: n=16 -> 9.328e-1, n=48 -> 9.334e-1, n=64 -> 1.029e0,
against a per-token arm pinned at 1.633e-2 throughout.

CAVEAT, so the table is not over-read: running the same comparison on
Qwen3.8-27B oq4.25++ reports batched == per-token == 1.203e-2, which is NOT
evidence that compact is fine. Compact declines the hidden-exporting forward by
design (`allow_compact`), so BOTH arms ran per-token there — the test does not
exercise the batched path on that model. The defect is demonstrated on a dtype
that actually batches.

### Why this matters beyond this branch

- KVarN is the sanctioned default and q8 is deprecated, so the default
  configuration is the broken one.
- It plausibly degrades any DFlash drafter on a model that DOES batch prefill —
  the drafter is handed a capture that is ~1.0 relative error from the truth.
- It is upstream of the whole spec-decode question here: fixing it is what would
  let compact batch a hidden-exporting forward at all.

### Not the whole acceptance story

Worth stating so the next person does not over-attribute: the spec-decode
acceptance loss measured here (0.482 -> 0.286) was under **q8** KV, where the
batched path is faithful (1.6e-2). So acceptance drops even when the batched
capture is accurate — the drafter is sensitive to WHICH capture it gets, not only
to how accurate it is. The KVarN bug and the acceptance loss are two separate
problems that happened to surface together.

## One more negative, so it is not re-tried

Enabling the two-stage lm_head during spec-decode (`HIPFIRE_LMHEAD_TWOSTAGE=q2`,
which is worth 2.1x on the lm_head in ordinary decode) moved spec-decode 8.53 ->
8.55 tok/s, i.e. nothing. The verify's per-position lm_head is not where its
370 ms goes. Combined with the kernel profile — `gemv_oq_compact_multicol` is
only ~48 ms/cycle of that 370 — the remainder is the rest of the verify forward
plus this harness's rollback parity check, and it is NOT any single kernel that
has been tried.
