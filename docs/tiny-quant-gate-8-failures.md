# The 8 tiny-quant-gate failures, classified

`tests/tiny-quant-gate.sh` has failed 8 cells on every run this session,
identically, including on commits proven not to touch the paths involved
(reverting a change and re-running reproduces every number to six decimals).
They are not one problem, and only one of them is a regression.

## Reading the message

    KLD drift 0.001790 vs baseline 0.002662 (budget ±0.000665)

"drift" is misleading: the first number is the **current mean KLD**, not a
delta (`executor_tinyquant.rs:495` formats `cell.mean_kld` then `b`). Lower is
better, so several of these "failures" are the fixture scoring *better* than its
recorded baseline.

## Classification

| cell | current | baseline | what it is |
|---|---|---|---|
| `qwen2/kld:hfq4` | 0.001790 | 0.002662 | better — stale baseline |
| `gemma3/kld:q8f16` | 0.000868 | 0.001592 | better — stale baseline, **deprecated format** |
| `gemma3/kld:hfq4` | 0.094058 | 0.158772 | better — stale baseline |
| `qwen3_5/kld:q8f16` | 0.000538 | 0.000843 | better — stale baseline, **deprecated format** |
| `minimax/kld:mq4` | **0.000000** | 0.001042 | **vacuous** |
| `qwen3_5_moe/kld:mq6` | 0.215099 | 0.154634 | **vacuous** (see below) |
| `qwen3_5_moe/kld:mq4` | 0.215099 | 0.154634 | **vacuous** (see below) |
| `qwen3_5_moe/kld:q8f16` | 0.179210 | 0.141306 | worse, but **Q8 is deprecated** |

**Four are stale baselines in the good direction.** Re-recording clears them and
loses nothing; they are noise that trains the reader to ignore the gate.

**`minimax/kld:mq4` scores exactly 0.000000.** A quantised model with zero KLD
against its own reference is not a pass, it is a cell measuring nothing. Already
on the open-decisions list.

**`qwen3_5_moe/kld:mq6` and `kld:mq4` are the same cell twice.** They report
0.215099 against 0.154634 — identical to six decimals in BOTH the current run
and the committed baseline, across a 6-bit and a 4-bit format. Two different bit
widths cannot produce bit-identical KLD; these are not measuring their nominal
formats. The identity held when the baselines were recorded, so this is
long-standing rather than new. Only `mq6` was on the known-vacuous list — `mq4`
belongs there too, making it three vacuous cells, not two.

**`qwen3_5_moe/kld:q8f16` is on a deprecated format, so it is not worth chasing.**
An earlier revision of this document called it "the only real signal" — 0.179210
against a 0.141306 baseline, +27% and well outside the ±0.035 budget, which on a
nearly-lossless format would be alarming. **Q8 weights are deprecated** per the
2026-07-18 directive (`docs/plans/2026-07-18-blocked-feature-coverage-plans.md`
:169: "Q8 (weight and KV) is being deprecated"). A regression in a format on its
way out does not earn investigation; the cell earns deletion.

That takes the three `q8f16` cells here — `gemma3`, `qwen3_5`, and
`qwen3_5_moe` — out of scope entirely, whichever direction they moved.

**So none of the 8 requires a fix.** Every one is a stale baseline, a vacuous
cell, or a deprecated format. The whole set clears with re-recording and
deletion, no debugging.

## Why this matters beyond the cleanup

A gate that fails 8 cells on every run cannot report a 9th. Every runtime change
this session had to be cleared by reverting it and re-running to compare
numbers, because "8 failures" carries no information.

And the failure set is *entirely* clearable: re-record the stale baselines, drop
the three vacuous cells, drop the deprecated-format cells. Nothing here needs
debugging. That is the strongest argument for doing it — the standing noise is
not protecting against anything, it is only hiding whatever comes next.

## Postscript, 2026-08-13: the 9th arrived, and it was hidden for six days

The section above predicted that standing noise "is only hiding whatever comes
next." That happened, and this is the record of it.

Running the gate on merged master (`37eb5c464`) now reports **3** failures, not
8 — the clearable set was largely re-recorded in the meantime. One of the three
is new, real, and was never in the original eight:

| cell | current | baseline | what it is |
|---|---|---|---|
| `gemma4_moe/kld:oq4.25++(calib)` | 0.005952 | 0.003077 | **regression, +93%, bisected** |
| `qwen3_5_moe/kld:oq8+(calib)` | 0.005677 | 0.008147 | better — and **vacuous**, see below |
| `qwen3_5_moe/kld:oq8++(calib)` | 0.005677 | 0.008147 | better — and **vacuous**, see below |

### The regression: `gemma4_moe/kld:oq4.25++`

Bisected over 549 commits (10 steps) to:

    8357081d3  fix(opus): choose the mixed scale and promotion set jointly  (2026-08-06)

The bisect tested the measured **value**, not the gate's pass/fail — the
baseline file itself moves across that range, so a pass/fail bisect would have
found "when the baseline was edited" instead of "when the number changed." The
value is bit-identical 0.003077 at every commit before and 0.005952 at every
commit after, so this is a deterministic step, not drift.

**That commit predicted the opposite of what happened.** Its own message says:

> At the shipped oq4.25++ default (N_out=3) this is a 0.6% SSE change; do
> not expect a visible KLD move there.

The move is +93% of baseline, on exactly that shipped default, against a ±25%
budget. The commit changed `codecs.rs` and `ldlq.rs` and did **not** re-record
`tests/tiny-quant-baselines.txt`, so the cell has been red ever since — absorbed
into the standing-failure noise this document was written about.

The commit's mathematical argument is not in question: for a fixed scale the
top-N_out gain sort is the optimal promotion set, so sweeping the grid and
recomputing the set inside the loop really is the joint argmin. The gap is that
the argmin is over **group reconstruction SSE**, and the gate measures **KLD**.
A 0.6% SSE change producing a 93% KLD change is that proxy/target gap stated
numerically — a better weight-space optimum is not automatically a better
output-space one.

**What this does NOT establish.** The tiny fixtures are seeded random-init models
over a synthetic token stream (`executor_tinyquant.rs:6-10`). On random weights
there is no correct answer to move toward, so this is evidence that the encoder's
output changed materially and deterministically — which is what the gate is for —
and NOT evidence that real models quantize worse. **The real-model impact is
unmeasured and is the question that decides what to do here**: re-record the
baseline (if real models are unaffected or improve) or revisit the selector (if
they regress). Deciding from the fixture alone would be reading a
regression-detector as a quality metric.

### The other two are one vacuous cell twice

`qwen3_5_moe/kld:oq8+` and `kld:oq8++` report **0.005677 in the current run and
0.008147 in the committed baseline — identical to six decimals in both**, across
two formats that differ by whether Hessian/LDLQ error feedback runs. That is the
same signature this document used to classify `mq6`/`mq4` as vacuous.

The cross-check that makes it a property of this cell rather than of `++`
generally: on `gfx1151`, `qwen3_5_moe_indexed` records **different** values for
the two (`oq8+` 0.00307532, `oq8++` 0.00290585), so the second `+` is not
inherently a no-op. Both moved in the better direction anyway.

### Provenance of these numbers

Reproduced at three commits with bit-identical results, so the merge under test
introduced none of it:

| commit | `gemma4_moe/oq4.25++` | `qwen3_5_moe/oq8+`, `oq8++` |
|---|---|---|
| `37eb5c464` (merge of PR #248) | 0.005952 | 0.005677 |
| `1cc6868cb` (pre-merge master) | 0.005952 | 0.005677 |
| `c958348c3` (before the routers-BF16 change) | 0.005952 | not run |
| `753df2b27` (where the baseline was recorded) | **0.0031 — PASS** | not run |

The `72cd1c10b` routers-lossless-BF16 change was the first suspect, since it is
already documented above as having moved `qwen3_5_moe` values. It is not the
cause: the failure reproduces at its parent.

## Resolved 2026-08-13: on a REAL model the change is a 26% IMPROVEMENT — re-record, do not revert

The section above said the fixture alone could not decide this and that a real
model had to be measured. It was, and it reverses the naive reading.

Protocol: Qwen3.5-0.8B from safetensors, quantized to `oq4.25++` on both sides of
the bisect boundary with **one Hessian generated once and reused**, so the
encoder is the only variable, and both artifacts scored against **one KLD
reference built from the bf16 anchor of the same weights**.

| | perplexity | mean KLD vs bf16 | ppl gap to bf16 |
|---|---|---|---|
| bf16 anchor | 15.105 | — (4.0e-10 self-check) | — |
| **`8357081d3`+ (new selector)** | **15.740** | **0.030567** | **0.635** |
| `b05f74a79` (old selector) | 16.126 | 0.041126 | 1.022 |

**KLD −25.7%, and the new selector recovers 38% of the quantization perplexity
gap.** Two independent metrics agree. The commit is a real improvement on real
weights; the fixture's +93% is an artifact of seeded random-init weights, where
there is no outlier structure for a promotion-set search to find and the choice
is near-degenerate.

**So the action is to re-record `gemma4_moe/kld:oq4.25++`, not to revisit the
selector.** The cell is measuring a genuine encoder change, in the wrong
direction for the wrong reason.

**Remaining gap, stated rather than papered over:** the model measured is dense
(arch 5) and the failing fixture is MoE. A real MoE has not been measured. The
burden has shifted — the change demonstrably helps a real dense model — but
"MoE behaves the same" is an assumption here, not a measurement.

### The trap this run fell into first, recorded because it produced a perfect-looking wrong answer

The first A/B reported the two sides as **identical to 18 significant digits**
(0.041125837713479996 both), with byte-identical quantized payloads. That was
wrong, and it looked clean.

Cause: the driver script invoked `./target/release/hipfire-quantize` for the
master side **without rebuilding it first**, and an earlier `git bisect run` had
left that binary built at `b05f74a79` — the GOOD side. So "master vs
`b05f74a79`" was really "`b05f74a79` vs `b05f74a79`", and of course it matched.

What exposed it: two different weight sets cannot produce a bit-identical mean
KLD over 1023 tokens. The confirming test was re-quantizing at the same commit
and comparing payloads — which also establishes, as a by-product, that
`hipfire-quantize` **is** deterministic: same binary and inputs give a
byte-identical payload, and only the HFQ front-metadata and tail key ordering
varies between runs (~13 MB of a 647 MB artifact, no weight bytes).

Two standing lessons: **rebuild explicitly before an A/B that depends on which
commit built the binary** — `git bisect` leaves the tree's binary at an arbitrary
commit — and **treat an impossibly clean agreement as a symptom, not a result.**

## Correction 2026-08-13: the `+`/`++` "no-op" is NOT established

An earlier note here (and PR #252's description) read the identical `+`/`++`
baseline rows as a silent no-op — "ask for `++`, get `+` output." **That claim
was overstated and the evidence does not support it.**

**What the direct test shows.** Qwen3.5-0.8B (real weights, dense, arch 5)
quantized to `oq8+` and `oq8++` from the same source with the same Hessian:

    oq8+   quantization_hash = 3ae1a4908e56c478
    oq8++  quantization_hash = 3b1ad88133127973
    oq8++  LDLQ tensors: success=186 attempts=186 missing=0 k_mismatch=0 pack_failed=0

LDLQ is applied to every eligible tensor and the quantized payload genuinely
differs. **`++` is not a no-op.**

**Why the rows looked identical.** `tests/tiny-quant-baselines.txt` stores 8
decimal places and `results.jsonl` rounds to ~6 significant figures, so any
difference below that is invisible. "Identical in the baseline file" therefore
cannot distinguish *no change* from *a change smaller than the recorded
precision*, and I read it as the former.

**What remains true, and is the more useful conclusion.** On the tiny fixtures
`++` produces no *resolvable* KLD change, while `gfx1151`
`qwen3_5_moe_indexed` records `oq8+` 0.00307532 against `oq8++` 0.00290585 — a
~5% gap, clearly visible. The unified explanation also covers the `oq4.25++`
inversion documented above: **seeded random-init fixtures cannot resolve
calibrated-format quality differences.** Random weights have no outlier or
correlation structure for AWQ scaling or Hessian error feedback to exploit, so
the calibrated cells measure almost nothing on them — which is exactly why the
fixture moved *opposite* to the real model on `oq4.25++`.

So these `+`/`++` cells are weak tests, not evidence of a bug. Treat the whole
calibrated block on random-init fixtures as low-information.

**Genuinely open, stated narrowly:** whether LDLQ reaches *routed expert*
tensors on the MoE path. `ldlq_report_and_validate` only enforces
`success > 0`, so a build where LDLQ covered the dense tensors and skipped every
routed expert would pass silently. The counter that answers it
(`LDLQ_MISSING`) is printed to the quantizer's stderr, which the tiny harness
discards. Answering it needs either that stderr surfaced or one real MoE
quantized to `oq8+` and `oq8++` with payload hashes compared — the same
technique used above.
