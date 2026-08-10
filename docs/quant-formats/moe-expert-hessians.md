# Routed-expert Hessians — what they would cost, and whether they would pay

Status 2026-08-07. Two studies, both reproducible:
`moe_expert_saliency_study` (hipfire-runtime, zero GPU) and
`hessian_generalisation_study` (hipfire-quantize, CPU only).

## The problem

Routed MoE experts are captured **imatrix-only**. `calibration_stream.rs:311`
pins them to `CapturePolicy::ImatrixOnly`, `expert_capture.rs:338` asserts it,
and the resident collector blocks them with a `vec![".experts."]` substring
list (`arch-zaya/src/calibration.rs:53`). `--ldlq` does not fail on a missing
Hessian — it logs `ldlq: skip <t>` and falls back to RTN. So **an `oq4++` on a
MoE model is `++` on the dense path and plain RTN on every routed expert**,
while reporting success. Verify on any artefact with
`hipfire inspect <x>.calib.hfq`.

Storing the missing Hessians is infeasible: for qwen3.6-35b-a3b, 9778 captured
expert-tensors at the compact `4K + K(K−1)` size (gate_up K=2048 → 4.20 MB,
down K=512 → 0.26 MB) is **43.6 GB**, against 1.8 GB for the whole current
calib.

## Q1 — experts do not share an input basis, and the two projections differ

Measured from the per-expert imatrix already in every MoE calib. The imatrix is
`diag(XᵀX)`, the diagonal of the Hessian in question, so cosine between
L1-normalised per-expert profiles bounds what a layer-POOLED Hessian would
lose. (L1-normalise first, or a hot and a cold expert differ purely by row
count, which is not specialisation.)

Mean pairwise cosine between experts, per layer:

| projection | zaya1-8b (16 experts) | qwen3.6-35b-a3b (256 experts) |
|---|---|---|
| `gate_up_proj` | 0.82 – 0.90 | 0.62 – 0.77 |
| `down_proj` | 0.04 – 0.23 | 0.09 – 0.35 (min ≈ 0.0000) |

Mechanically obvious in hindsight, which is a good sign. `gate_up` reads the
**shared residual stream** — every expert sees the same distribution, filtered
by routing. `down_proj` reads that expert's **own SwiGLU intermediate**,
produced by its own gate/up weights, so each expert's input lives in a private
basis. Near-orthogonality is exactly what that predicts.

**Do not build one mechanism for both.**

## Q2 — at a routed expert's sample ratio, does `XᵀX` still help?

A per-expert Hessian sees only the rows routed to that expert: `n/K ≈ 2` for a
top-1 16-expert layer at a realistic corpus. Measured on the DENSE path
(qwen3.5-0.8b layer 12, where K spans 3.5× so one capture sweeps the ratio),
oq4 LDLQ vs RTN, fitted on one corpus half and evaluated on a **disjoint**
half. Metric is the H-weighted proxy loss LDLQ minimises, normalised by the
same quantity for the unquantized weights.

| tensor | K | n | n/K | in-sample | held-out | vs RTN |
|---|---|---|---|---|---|---|
| gate_proj | 1024 | 512 | 0.50 | 0.001714 | 0.003180 | 0.806× |
| gate_proj | 1024 | 2048 | 2.00 | 0.001958 | 0.002820 | **0.715×** |
| gate_proj | 1024 | 8192 | 8.00 | 0.002344 | 0.002787 | 0.706× |
| out_proj | 2048 | 512 | 0.25 | 0.018815 | 0.027755 | 0.962× |
| out_proj | 2048 | 2048 | 1.00 | 0.021616 | 0.026677 | 0.925× |
| out_proj | 2048 | 8192 | 4.00 | 0.022242 | 0.024684 | 0.855× |
| down_proj | 3584 | 512 | 0.14 | 0.002321 | 0.011324 | 0.840× |
| down_proj | 3584 | 2048 | 0.57 | 0.004124 | 0.010230 | 0.759× |
| down_proj | 3584 | 8192 | 2.29 | 0.006061 | 0.009510 | **0.706×** |

Three things, and the first one surprised us:

1. **LDLQ beats RTN on held-out data at every ratio tested**, down to n/K =
   0.14, where the Hessian has rank 512 for a 3584-wide problem. The damping
   (`0.01 × mean diag(H)`, matching production) absorbs the rank deficiency.
   The sample-starvation objection to per-expert Hessians is therefore **not**
   fatal.
2. **The benefit saturates early.** gate_proj: 0.806 → 0.715 → 0.706 across
   n/K 0.5 → 2 → 8. Going from n/K 2 to 8 buys ~1%. A routed expert at n/K ≈ 2
   already captures nearly all of the available gain.
3. **Overfitting is real but shrinking**, and visible in the right signature:
   the in-sample/held-out gap is 4.9× at n=512 and 1.57× at n=8192 for
   down_proj, and in-sample error *rises* with n (0.0023 → 0.0061) because a
   better-sampled Hessian is a harder, more honest target.

### The caveat that governs the decision

This measures the **proxy loss LDLQ itself optimises**, so it is biased toward
showing benefit. `opus-quant.md` §7 already records that **on held-out data,
plain `XᵀX` LDLQ ≈ no calibration** in end-loss terms, and
`opus_outlier_budget_study` records weight-space SSE and KLD disagreeing in
this codebase before. Both can be true at once: LDLQ reliably reduces
H-weighted *weight* error out of sample (this table) while that reduction fails
to become *KLD*. Nothing here contradicts that; this table ranks a mechanism,
it does not certify a format.

## Q3 — the KLD measurement: pooled `gate_up` WORKS

The proxy-loss table above was explicitly not enough to justify building
anything, so the pooled `gate_up` Hessian was implemented and measured
end-to-end on **zaya1-8b** (16 experts, top-1, 40 MoE layers).

No new capture was needed. zaya's `mlp.gate.down_proj` already accumulates a
full Hessian over `normed` — the *same* tensor the experts consume
(`arch-zaya/src/gpu.rs:1333` vs `:1390`), same K=2048, over every token. Under
top-1 routing every token is routed, so that tensor **is** the layer-pooled
`gate_up` Hessian, already sitting in every zaya calib. The change is a
quantizer-side lookup: `HIPFIRE_POOLED_EXPERT_HESSIAN=1` lets a routed expert's
gate/up borrow a layer-pooled donor, guarded by an exact K match and restricted
to `POOLABLE_EXPERT_LEAVES` (never `down_proj`/`w2`).

Both arms: `--format oq4++ --hessian <8k-token calib> --ldlq`, scored against a
16-chunk bf16 reference. `mode: score` takes its tokens from the reference, so
the arms are paired by construction.

| arm | LDLQ success | missing | mean_kld |
|---|---|---|---|
| baseline (experts RTN) | 360 | 1280 | 0.573656 |
| pooled `gate_up` | 1000 (+640) | 640 | **0.555906** |

**Paired over 16 chunks: −3.09%, 95% CI [−5.60%, −0.59%], t = −2.42.** The CI
excludes zero, so this is a resolved improvement, not noise. Inter-arm
correlation is 0.977, which is why 16 chunks suffice.

+640 is exactly 40 layers × 16 experts of `gate_up`, and `missing` falls from
1280 to 640 — the remaining 640 are `down_proj`, correctly still skipped.

Caveats: one model, one corpus, one format, n=16. The effect is modest and the
CI is wide (0.6–5.6%). Note also how much headroom remains — a baseline
mean_kld of 0.574 is poor, because every routed expert was RTN; pooling
`gate_up` recovers 3% of that, and `down_proj` (still RTN) is plausibly the
larger remaining share.

**This clears the gate set below.** The proxy-loss result did become a KLD
result, so the per-expert `down_proj` pass is now justified rather than
speculative.

## Q4 — per-expert `down_proj`: NO measurable gain (negative result)

Q3 ended by predicting that the per-expert `down_proj` pass was "justified
rather than speculative". **Measurement falsified that.**

Two things collapsed the cost first. The 43.6 GB storage figure is dominated by
`gate_up` at K=hidden; **`down_proj`'s K is the MoE INTERMEDIATE width, which
shrinks as experts multiply**, so storing every per-expert `down_proj` Hessian
costs only ~2.6 GB on *both* zaya1-8b (640 × K=2048) and qwen3.6-35b-a3b
(9778 × K=512). So the fused compute-then-discard pass was **never needed** —
the expensive projection is the one that pools, and the un-poolable one is
cheap. `HIPFIRE_CALIB_IMATRIX_ONLY=".gate_up_proj"` captures it with the
existing collector: +607 Hessians, calib 865 MB → 3.2 GB, and capture time
unchanged (746.8 s vs 744 s).

Arm C = pooled `gate_up` + per-expert `down_proj`, same corpus, same 8192-token
budget, same 16-chunk reference as A and B.

| arm | LDLQ success | missing | mean_kld |
|---|---|---|---|
| A baseline (experts RTN) | 360 | 1280 | 0.573656 |
| B pooled `gate_up` | 1000 | 640 | 0.555906 |
| C B + per-expert `down_proj` | 1607 | 33 | 0.554680 |

C's counters are exactly right: 360 dense + 640 pooled `gate_up` + 607
per-expert `down_proj`, and the 33 missing are experts that never activated.

**B → C: −0.22%, 95% CI [−2.92%, +2.47%], t = −0.16.** Nothing. The whole
measurable gain (−3.09%, A → B) came from pooled `gate_up`; adding per-expert
`down_proj` on top adds no resolvable improvement. A → C is −3.31% but with a
CI spanning zero (r drops to 0.931), i.e. it is B's gain plus noise.

Why, plausibly — none of these is tested:

- **Sample starvation.** 8192 tokens over 16 experts is ~512 rows for K=2048,
  n/K ≈ 0.25, and the per-expert row counts are wildly imbalanced (1 to 4262).
  Many experts get a near-degenerate Hessian that damping collapses toward RTN.
  Q2 measured 0.84× at n/K = 0.14 on a DENSE tensor, where rows are well mixed;
  a starved expert is a harder case than that.
- **Parameter share.** In zaya `gate_up` is [4096, 2048] and `down_proj` is
  [2048, 2048], so `gate_up` is 2/3 of expert weight. B already treated the
  larger share.
- **The SwiGLU intermediate may simply be near-isotropic**, making `XᵀX`
  ≈ scaled identity and LDLQ ≈ RTN regardless of sample count.

The honest limit: at n=16 chunks this cannot resolve effects below ~3%. It
does not show `down_proj` Hessians are worthless — it shows they are not worth
+2.4 GB of calib and ~200 s of quantize at this token budget on this model.
The cheap follow-up is to re-run Arm C at a much larger token budget so the
starved experts fill; if the effect is real it should grow.

## Q5 — 4x the calibration budget: the budget is the real lever, and per-expert `down_proj` turns UNSTABLE

Q4 blamed sample starvation (median n/K ≈ 0.22). Tested by re-running at 32768
tokens, with both arms built from ONE calib
(`HIPFIRE_LDLQ_SKIP_EXPERT_LEAVES=down_proj` produces the control), so they
differ only in the thing under test.

**Was it budget or corpus diversity?** Budget, for the bulk. Row counts per
expert scale almost exactly with the token budget, and the concentration is
invariant across a 16x range:

| tokens | p25 | p50 | p75 | p90 | rows<256 | top-10% share of rows |
|---|---|---|---|---|---|---|
| 2k | 45 | 119 | 194 | 273 | 88% | 28.5% |
| 8k | 237 | 456 | 725 | 1019 | 27% | 28.3% |
| 32k | 994 | 1874 | 2795 | 3875 | 11% | **28.4%** |

A diversity-bound corpus would show the skew WORSENING as tokens are added.
It does not — the top decile holds the same 28.4% of rows at every budget, so
the corpus reaches experts proportionally. Diversity binds only the floor:
`min` stays at 1 row and ~4% of experts remain under 32 rows at any English
budget (consistent with the router profiler's CJK finding). Median n/K went
0.22 → **0.89**.

**Cost note: the capture is O(n²).** 4x the tokens cost 14.3x the time
(747 s → 10659 s), so 32k is the practical ceiling on this path; the ~73k
needed for median n/K ≈ 2 would be ~40 h.

### The budget appears to dominate — but this measurement is CONFOUNDED (see Q7)

Same method, same code, only 4x the calibration tokens:

**pooled `gate_up` @8k → @32k: −12.92% KLD** (0.555906 → 0.484070),
95% CI [−17.89%, −7.95%], t = −5.10.

**⚠ RETRACTED as a budget effect.** Q7 found that the calibration corpus and
the KLD reference are the SAME FILE read from offset 0, and the reference is
tokens [0, 32768). So the 8k arm saw 25% of the evaluation set and the 32k arm
saw exactly 100% of it and nothing else. The measured gain is at least partly
train-on-test, not a budget effect. Do not cite this number.

Attribution caveat that still applies if the effect is ever re-measured
cleanly: at 32k the 360 dense Hessians improve too, so it would be the
whole-model benefit of more calib tokens, not something expert-specific.

### Per-expert `down_proj` at 32k: neutral on most content, catastrophic on some

| arm (32k calib) | LDLQ success | missing | mean_kld |
|---|---|---|---|
| B32 pooled `gate_up` only | 1000 | 640 | **0.484070** |
| C32 + per-expert `down_proj` | 1617 | 23 | 0.624441 |

Aggregate reads +29%, but the inter-arm correlation **collapses to 0.17** (it
was 0.92–0.98 everywhere else), which means instability, not a shift. Per
chunk:

- **14 of 16 chunks: ratio 0.94–1.03** (median 0.987, i.e. slightly better)
- **2 chunks blow up 3.5x**: chunk 1 0.5366 → 1.8878, chunk 2 0.3827 → 1.4019

Excluding those two: −1.82%, 95% CI [−3.75%, +0.11%]. So per-expert `down_proj`
is worth ~1.8% *when it works* and occasionally destroys a chunk.

**The blowup is NOT starvation.** 8k had MORE starved experts than 32k (41 vs
25 under 32 rows) and produced no blowups at all. More data made it worse. That
points at the opposite mechanism: a better-sampled Hessian makes LDLQ commit
harder to the calibration distribution, so out-of-distribution content fails
worse — which is precisely `opus-quant.md` §7's "plain `XᵀX` LDLQ ≈ no
calibration on held-out data", showing up as variance rather than as a mean
shift. A minimum-rows guard would therefore probably NOT fix it.

Two follow-ups would discriminate, neither run: (a) a min-rows guard — if
blowups persist, starvation is exonerated for good; (b) much heavier damping on
expert `down_proj` — if blowups vanish, over-commitment is confirmed.

## Q6 — the O(n²) capture was self-inflicted: split the sequence, get 10x

Q5 found the budget to be the largest lever but the capture superlinear, which
capped it. The cause was structural, not fundamental: the resident calibration
ran the whole budget as ONE sequence, and attention is O(seq²).

A Hessian is a sum of per-row outer products — it does not care whether the
rows came from one context or many. And KLD references are built at
`n_ctx=2048`, so shorter calibration sequences match the evaluation
distribution rather than diverge from it. `HIPFIRE_CALIB_SEQ_LEN=2048` splits
the stream into independent sequences.

Same 32768 tokens, same blocklist, same everything else:

| capture | wall | hessians | imatrix | experts covered |
|---|---|---|---|---|
| one 32768 sequence | **10746 s** | 978 | 1595 | 617 |
| 16 x 2048 | **1065 s** | 985 | 1609 | **624** |

**10.1x faster**, and it captures slightly MORE — 16 independent contexts route
more diversely than one continuous document. Row distributions are
statistically identical (p50 1874 vs 1880, p25 994 vs 972, top-decile share
28.4% vs 28.6%).

Quality is equivalent, and the budget win survives intact:

- one-32k-sequence vs 16x2048: **−0.77%, 95% CI [−3.40%, +1.87%]** — no
  resolved difference, which is the desired outcome.
- pooled @8k vs pooled @32k-split: **−13.59%, 95% CI [−18.74%, −8.44%]**,
  t = −5.17.

So the −13.6% quality gain is available for 1065 s of capture instead of
10746 s. This removes the ceiling Q5 hit: 128k tokens becomes ~70 min rather
than ~40 h.

### Hoisted to the shared seam, and validated on a second arch

The problem was never zaya-specific. `collect()` now takes
`sequences: &[&[u32]]` and hands the closure the split view, so the policy is
central and no arch can silently calibrate under one unbounded context. Taking
sequences rather than a flat token list also unifies the two notions already in
the tree — qwen35 and gemma3 drive calibration from a `SampleSet`, and those
samples are now re-split by the same policy.

The other arches had the same defect wearing different clothes: nemotron,
minimax and lfm2moe run a per-token decode loop with `pos` running to the end
and state reset ONCE before the loop, so KV/SSM state grows to the whole
budget. For nemotron (Mamba-2) that meant the recurrent state carried the
entire corpus's history — a semantics bug, not just a cost one.

Validated on **nemotron** (Nano-4B-BF16, 8192 tokens, dense):

| arm | wall | hessians | tokens | md5 |
|---|---|---|---|---|
| single sequence | 737 s | 92 | 8192 | `e3be7c82…` |
| 4 x 2048 split | 706 s | 92 | 8192 | `f3606057…` |

Both `diag(H)-vs-Σx²` CONSISTENT. Identical tensor and token counts (nothing
double-tapped or dropped) with DIFFERENT payload hashes — which is the point:
the per-sequence `model.reset` genuinely re-scoped the statistics. Timing is
flat (1.04x), as expected: nemotron's cost is the linear per-token decode loop,
and only its hybrid attention layers pay the quadratic term. **For a per-token
arch this change buys correctness, not speed.** The 10.1x is specific to arches
whose forward is sequence-shaped, like zaya.

minimax and lfm2moe are wired identically but not GPU-tested here.

## Q7 — the budget curve inverts, because the budget experiment was train-on-test

Pushing to 131072 tokens (4x again, now affordable thanks to Q6) did not
continue the trend. It reversed it:

| calib tokens | mean_kld (pooled `gate_up`) | vs previous |
|---|---|---|
| 8k | 0.555906 | — |
| 32k | 0.480360 | −13.59% |
| 128k | 0.499066 | **+3.89%** |

32k → 128k: **+3.89% WORSE**, 95% CI [+0.42%, +7.37%], t = +2.19 — the CI
excludes zero, so the reversal is real, not noise.

An inverted U in calibration size is a red flag, and the cause is a flaw in the
experiment, not a property of calibration. **The calibration corpus and the KLD
reference are the same file, both read from offset 0**, and the reference is
`n_ctx=2048 x max_chunks=16` = tokens [0, 32768):

| calib | tokens read | overlap with the evaluation set |
|---|---|---|
| 8k | [0, 8192) | 25% |
| 32k | [0, 32768) | **exactly 100%, and nothing else** |
| 128k | [0, 131072) | 100% plus 98k tokens of dilution |

That reproduces the observed curve exactly: eval-set coverage climbs 25% → 100%
(the apparent −13.6%), then extra data pulls the Hessian toward the wider
corpus and away from the evaluated tokens (+3.9% back). It is textbook
train-on-test, and it is the same generalisation failure `opus-quant.md` §7
already records for `XᵀX` LDLQ — here amplified by an experiment that handed
the calibration the answer sheet.

**What this does and does not invalidate:**

- **Invalid: the budget claim.** −12.92% / −13.59% must not be cited. Whether
  calibration budget helps at all is now UNMEASURED on this model.
- **Still valid: pooled `gate_up` (−3.09%) and the per-expert `down_proj`
  results.** Both arms of each of those comparisons used the SAME calibration
  artifact, so they consumed identical data. The absolute KLD is optimistic for
  every arm equally; the DIFFERENCE between arms is unaffected.
- **Still valid: Q6's 10.1x sequence-split speedup.** It is a timing result
  with no statistical claim attached, and its quality check was a
  no-difference test between two arms on the same tokens.

The clean re-test is cheap now: calibrate from a corpus region DISJOINT from
the reference's [0, 32768) — e.g. skip the first ~500k characters — and re-run
the 8k/32k/128k sweep. Until that runs, "more calibration tokens" is not a
supported recommendation.

## What to build, if anything

- **`gate_up`: layer-pooled Hessian.** Q1 says pooling costs little
  (expert-vs-pooled cosine 0.78–0.95) and it multiplies the sample count by E,
  moving n/K from ~2 to ~500. One K×K per layer ≈ **168 MB for a whole model**,
  storable in the existing artefact, no fused pass, no new pipeline.
- **`down_proj`: per-expert or nothing.** Pooling near-orthogonal bases is
  meaningless. Per-expert is affordable if it is never stored: hold one
  **layer's** experts and discard after quantizing that layer — ~1.1 GB peak
  for qwen3.6-35b-a3b, against 43.6 GB stored. Finalising each expert as it
  hits its `target_rows` quota drops the peak to one expert (~4.5 MB).
- Compute is not the constraint: ~357 TFLOP to accumulate every expert Hessian
  (2·K²·4096 per expert) plus ~28 TFLOP of LDL factorisation — minutes on
  gfx1151, against the 43.6 GB of storage that was the actual blocker.

The plumbing largely exists. `expert_capture.rs` already gathers rows per
expert from the routing permutation, quota-capped and tiled, "carrying partial
reduction tiles across model microbatches", and `CapturePolicy` already has a
`HessianAndImatrix` variant. The gate is the assertion at
`expert_capture.rs:338`, not missing machinery.

Settled by measurement:

- **Always calibrate on tokens DISJOINT from the evaluation set.** Q7 shows the
  default of pointing both at the same corpus from offset 0 produces a large
  fake improvement and an inverted-U budget curve. This is the most important
  thing on this page.
- **Use `HIPFIRE_CALIB_SEQ_LEN=2048` regardless.** It is a pure 10.1x capture
  speedup at equal quality (Q6), independent of every statistical question here.
- **Calibration token budget: UNMEASURED.** The −13.59% was train-on-test and
  is retracted. Re-run the sweep on a disjoint corpus region before believing
  any budget recommendation.
- **Ship pooled `gate_up`** (`HIPFIRE_POOLED_EXPERT_HESSIAN=1`). −3.09% KLD,
  CI excludes zero, costs nothing to store and needs no new capture.
- **Do NOT enable per-expert `down_proj`.** At 8k it bought −0.22% (CI spans
  zero); at 32k it is ~1.8% better on 14/16 chunks and 3.5x WORSE on 2, which
  is a net loss and a stability risk. Both knobs stay opt-in and off.
- **The fused compute-then-discard pass is not needed and should not be
  built.** Its whole motivation was 43.6 GB of storage, and that number was
  `gate_up`'s; `gate_up` pools, and `down_proj` stores in ~2.6 GB.

Open, in rough order of value:

1. **Re-run the 8k/32k/128k budget sweep on a corpus region disjoint from the
   reference** (Q7). Everything currently believed about calibration budget
   rests on a train-on-test measurement. Cheap now: with Q6 a 128k capture is
   ~65 min, and the whole sweep is a few hours.
   Also port `HIPFIRE_CALIB_SEQ_LEN` to the other arches' resident calibration
   — they all share the single-sequence shape.
2. **Generalise the pooled donor beyond zaya.** The principled donor is the
   ROUTER (`mlp.gate`), which by construction consumes the FFN input — already
   first in `POOLED_HESSIAN_DONORS`, but unverified on a qwen-style MoE.
   Confirm K matches there and re-measure.
3. Discriminate the C32 blowup mechanism (min-rows guard vs heavier damping)
   only if per-expert `down_proj` is ever wanted again.
4. More chunks: n=16 resolves the pooled effect's sign but not its magnitude
   (0.6–5.6%), and cannot resolve anything below ~3%.

## Reproduce

```sh
# Q1 — zero GPU, uses any existing MoE calib
cargo run --release -p hipfire-runtime --example moe_expert_saliency_study -- \
  --calib ~/.hipfire/calib/zaya1-8b-resident-2ktok.calib.hfq

# Q2 — needs Hessians on two disjoint corpus halves (~10 min of capture)
cargo run --release -p hipfire-quantize --example hessian_generalisation_study -- \
  --fits fit.512.calib.hfq,fit.2048.calib.hfq,fit.8192.calib.hfq \
  --test test.8192.calib.hfq
```

Capture note: `collect_artifacts` sizes the sequence to `--max-tokens`, and
16384 fails on qwen3.5-0.8b with `hipModuleLaunchKernel: invalid argument`.
8192 is the working ceiling for that path today.
