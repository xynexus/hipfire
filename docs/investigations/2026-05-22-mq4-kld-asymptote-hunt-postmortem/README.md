# Post-mortem: MQ4G256 KLD asymptote hunt — 2026-05-21 → 2026-05-22

> "I just hope this lever works." — kaden, immediately before this post-mortem
> was commissioned. The next lever is the residual learnable butterfly. This
> document is about what came before it.

## Headline

We spent ~36 hours of autonomous compute and three falsified-lever experiments
trying to push the closed-form MQ4G256 KLD on Qwen3.5-0.8B below the prior
session's 0.1257 baseline. We ended at **0.1327**, *worse* than where we
started. The empirical floor for this format + pipeline is now characterized.
The hunt itself was not wasted — the falsifications narrowed the search space
materially — but the journey was longer than it should have been, and the
pattern of mistakes was repeatable rather than one-off.

The takeaway is structural: **per-row weight optimization (AWQ + GPTQ + STE
scale learning + PARO K=0 stage-2 weight fine-tune) cannot close the gap on
MQ4G256 because the bottleneck is rotation-layer degrees of freedom, not
scale/weight precision.** Three independent attempts hit the same wall from
different angles. The wall is real.

## Initial goals (where we wanted to land)

Two goals, set at session start (~2026-05-21 evening):

1. **Goal 1 — KLD asymptote across Qwen trunk family**: find the closed-form
   MQ4G256 quality floor on 0.8B, 9B, 27B, A3B with target KLD ≤ 0.08, or
   plateau-exit when no remaining lever moves the needle.
2. **Goal 2 — Beat ParoQuant's A3B numbers**: 0.071 overall / 0.089 wikitext
   at 4.677 BPW.

Operating constraints (user-stated, verbatim):
- "rely on your best judgement while the goal is active... don't suspend
  operations awaiting user input until the goal is cleared"
- "mi300x is paid hourly so keep busy"
- "branches OK, no master push"

## Where we actually landed

**Goal 1 outcome:**
- 0.8B: empirical floor 0.1327 (mix-v1, ctx=4096, α=0.55, paper formula, F1).
  Prior session reportedly hit 0.1257; we regressed 5% and could not
  reproduce. Target 0.08 unreachable from within MQ4G256 calibration.
- 9B: 0.180 on mix-v1. Best we measured this session.
- 27B: 0.427 measured but contaminated by Qwen3.5-vs-Qwen3.6 kldref mismatch;
  un-rebuilt. PPL 9.11 suggests model itself is fine.
- A3B: imatrix collection crashed on first attempt, not retried (pivoted to
  ParoQuant path instead).

**Goal 2 outcome:** structurally unreachable with AWQ+GPTQ. Björn Bösel's
PR #316 measurement of A3B-MQ4+AWQ at KLD 0.946 vs A3B-PARO at 0.0933
documents the floor: MoE per-row weight optimizers can't capture top-k
routing-conditioned activation variance regardless of calibration corpus,
α, or iteration depth. The 10× gap is rotation-DOF, not parameter tuning.

## Trajectory (compressed timeline)

| Phase | Lever | Result | Time spent |
|---|---|---|---|
| 1 | v3 iterative AWQ-aware-GPTQ (8 cells α×δ) | No-op proven; quant-Hessian ≈ BF16-Hessian | ~6 hr |
| 2 | False-plateau "0.133" declaration | User caught: prior baseline was 0.1257, not 0.133 | ~30 min |
| 3 | autoawq formula direct test | +159% KLD regression vs paper formula | ~1 hr |
| 4 | α=0.85 wikitext full eval | Slight regression vs α=0.75 | ~1 hr |
| 5 | Find lost calibration-mix-v1 corpus | Located on agent worktree, md5 verified | ~30 min |
| 6 | Mix-v1 6-cell α×ctx_len sweep | Best 0.1327 ctx=4096 α=0.55; still 5% short of 0.1257 | ~3 hr |
| 7 | autoawq via iterate (sidecar-bypassed) | No-op (sidecar override flag) | ~30 min |
| 8 | autoawq direct (no sidecar) | +159% regression — falsified | ~1 hr |
| 9 | PyTorch grad-scale-learning (BRECQ+STE) | +20% regression vs paper | ~4 hr (subagent) |
| 10 | "No GPTQ" ablation (paper vs grad-scale) | **Cell A: paper-no-GPTQ = paper-GPTQ** (GPTQ noise-floor finding) | ~2 hr |
| 11 | "No GPTQ" cell B (grad-scale-no-GPTQ) | **+20% regression PERSISTS** — root cause is NOT GPTQ alignment | ~30 min |
| 12 | PARO K=0 stage-1+2 (no rotations) | **+360% KLD regression** — catastrophically falsified | ~5 hr (subagent + my retry) |
| 13 | Literature pivot to ButterflyQuant + docs/plans writeup | Validates helix intuition as published method | ~2 hr |

Total wall clock: ~28 hr session active + ~8 hr subagent compute.
mi300 droplet billing: ~$10/hr × 36 hr ≈ **$360** of compute for a clean
empirical floor + three falsifications + a pivot.

## What we learned (the real wins)

1. **GPTQ is noise-floor on MQ4G256 + AWQ at this bit budget.** Paper-formula
   AWQ scales with and without GPTQ produced identical KLD to 4 decimal places
   (0.132679 vs 0.132733). This was the cleanest single finding of the
   session. It saves 3 minutes of Hessian-build per quantization run forever.

2. **Simple gradient-based proxy losses don't transfer.** Two independent
   methods (BRECQ+STE simple scales, PARO K=0 stage-2 weight FT) drove their
   proxy losses down 6.4% and 43% respectively, while production KLD went UP
   20% and 360% respectively. The proxy objective measured per-Linear MSE.
   The deployment objective is downstream KLD vs BF16 reference. They are
   not the same function and do not co-monotone in this pipeline.

3. **Why proxy fails on MQ4G256, specifically**: `(x/s)·(W·s) = x·W` is
   invariant in continuous math, but hipfire's per-256-group min-max RTN
   quantization breaks the invariance once `s` varies non-uniformly within
   a group. The PARO K=0 stage-1 drove channel_scales 3-4× away from
   geomean=1.0; that magnification widened per-group dynamic range,
   coarsened MQ4 RTN bins, amplified quant error. The "math is invariant"
   intuition is true only outside the quantization step.

4. **Structural floor on MoE quality with AWQ+GPTQ**: Björn's PR #316
   characterized the ceiling at ~0.95 KLD for A3B-MoE under any AWQ+GPTQ
   recipe. Per-row weight optimizers structurally cannot capture
   routing-conditioned activation variance — needs rotation-layer DOF.
   This is why ParoQuant works on MoE where AWQ doesn't.

5. **The path forward is rotation-DOF, not scale tuning.** The published
   methods that beat AWQ on long-generation reasoning tasks (SpinQuant,
   ButterflyQuant, ParoQuant) all add rotation degrees of freedom. The
   methods that try to learn better scales without adding rotation DOF
   (this session's BRECQ+STE, PARO K=0) repeatedly fail to transfer
   proxy gains to production KLD.

## What I (the orchestrator) did wrong

Honest critique. The user said "my pride was clouding my judgement"; mine
clouded mine too, in a few specific patterns:

### 1. Over-commitment to "the next lever will probably close the gap"

Before each experiment I predicted a meaningful win:
- v3 iteration: "should close gap meaningfully" → no-op
- autoawq: "could be the formula difference" → +159% regression
- grad-scale-learning: "literature says 5-15% improvement, very likely" → +20% regression
- PARO K=0: "should capture most of PARO's benefit at MQ4 perf" → +360% regression

In each case the language was hedged but the SEQUENCING was not — I kept
escalating to full GPU runs without checkpointing on "is the prior pattern
of proxy-doesn't-transfer holding?" The same proxy-to-production failure
mode appeared three times before I treated it as a class of problem rather
than three independent surprises.

### 2. The no-GPTQ control should have been the FIRST experiment, not the 10th

After the v3 iteration falsification ("quant-Hessian ≈ BF16-Hessian, no
update needed") it should have been obvious GPTQ was noise-floor. Running
the no-GPTQ control would have cost 30 minutes and saved ~3 hr of
Hessian-building across subsequent cells.

I deferred it because each individual experiment "needed GPTQ for fair
comparison". That's locally true but globally wrong — the comparison being
fair to the baseline matters less than knowing whether the baseline has
load-bearing components.

### 3. Missed prior art

ButterflyQuant (arXiv:2509.09679, Xu et al., Feb 2026 revision) directly
implements learnable orthogonal butterfly transforms for LLM quantization.
The user's "FWHT bonded to a helix" intuition wasn't speculative — it was
published method. I read the ParoQuant paper carefully and missed the
adjacent search neighborhood for the closely-related butterfly work. A
30-minute literature pass at the start would have surfaced this.

The doc the user wrote/found this morning surfaces it properly, with the
right framing: residual butterfly form `B_residual(0)=I` initialization,
+3-5% realistic runtime overhead, NOT zero. My characterization was
optimistic.

### 4. Sub-experiment scoping was too coarse

Every grad-scale and PARO experiment went straight to "wrap all 138 AWQ-F1
tensors, 128 sequences, 5 epochs, full slice eval." The subagent's smoke
phase (1 tensor, 4 sequences, 1 epoch) was just sanity-checking gradient
flow, not the actual hypothesis. A "wrap 5 tensors, 8 sequences, 2 epochs,
short-slice eval" intermediate would have shown the proxy-to-production
mismatch in 30 minutes instead of 4 hours.

I made the same mistake on PARO K=0 — the subagent's full 128-seq × 5-epoch
× 138-tensor run was 35 minutes of training before discovering the +360%
regression. A 30-minute hypothesis-check would have caught the geomean drift
at the channel-scale stage of stage 1.

### 5. Stop conditions weren't defined upfront

The doc the user wrote this morning has explicit stop-conditions in its
Tier 3 plan:

```
- If optimizer cannot beat current MQ4+AWQ on 0.8B KLD, do not port to HIP yet.
- If HIP overhead exceeds +10%, treat as paper/research only.
- If sidecar/routing complexity creates silent-wrong-output risks, require
  a new quant type rather than optional sidecar semantics.
```

I should have framed each session-experiment with stop conditions like that.
"If grad-scale doesn't beat 0.125 on a 5-tensor 8-seq smoke, do not full-run"
would have terminated the experiment at ~30 minutes instead of ~4 hours.

### 6. False-plateau declaration

At one point I declared "0.8B asymptote plateaued at 0.133" without checking
that 0.133 was *worse* than the prior session's 0.1257. The user caught this
immediately. That mistake is a smaller version of the larger orchestration
pattern: anchoring on the experiment's local result without comparing to
the wider context.

## Process critiques (the structural ones)

Beyond my individual mistakes:

### Subagent fragility

The PARO K=0 subagent died with an API 500 mid-run after building the
script and launching training. The script + training + commits survived
the crash (good), but no automated post-train chain existed (bad). I had
to manually script the quantize+eval handoff after the crash. For autonomous
multi-hour subagent work, the pattern needs to be: subagent writes ALL
chain steps as standalone scripts, commits them, then KICKS OFF the chain.
That way crash-mid-run still produces complete artifacts.

### Notifier-grep brittleness

Multiple notifiers false-fired on early-in-log matches (`writing.*safetensors`
during stage 1 was matched by the post-train chain meant to fire only after
final export). Notifier patterns need explicit DONE markers, not heuristic
matching. The patterns I used (`==.*DONE`) were generally right; the bug was
in chains I let the subagent write.

### Mi300 droplet cost vs information value

~$360 of mi300 time for a clean empirical floor + three falsifications is
not catastrophic — but most of the cost was repeating the same proxy-fails
pattern. If we'd terminated grad-scale at the smoke stage when scale
movement (32.27 → 32.40 in 1 layer × 1 epoch) was already suspicious for
its smallness, we'd have saved ~$80.

## What we'd do differently (for the next lever)

The doc's Tier 3 plan already encodes most of this, but for the record:

1. **Literature first.** Spend the first 1 hour on prior-art search before
   any GPU spend. Use ToolSearch, fetch papers, build a quick table of "what
   methods exist that touch this design point."

2. **Residual form initialization.** Always start `theta=0` so the experiment
   reduces to current behavior. Bisectability + production-fallback safety.

3. **Sub-tensor smoke before full-tensor run.** Wrap 5 tensors, 8 seqs,
   2 epochs, run a SHORT-SLICE eval (e.g., --max-chunks 16, 4 min). If the
   smoke doesn't show plausible proxy-vs-production directional agreement,
   don't escalate. This is the doc's "Stop conditions" applied at experiment
   sub-step granularity.

4. **No-GPTQ control as the SECOND step.** First step = "reproduce baseline".
   Second step = "ablate the most-suspicious-of-being-noise component."

5. **Define stop conditions upfront.** For the butterfly experiment, the
   doc already specifies: "If optimizer cannot beat current MQ4+AWQ on 0.8B
   KLD, do not port to HIP yet." Apply the same rigor to every future lever.

6. **Subagent autonomy needs full-chain scripting.** Subagents should write
   complete scripts including post-train, eval, comparison, and reporter
   logic. Then kick off the chain as a single nohup'd shell script. That way
   API crashes don't leave half-finished work.

## Where we go from here

Per the doc (`docs/plans/quant-strategy-research-recommendations.md`):

**Tier 1 — production:** Stabilize AWQ/raw-sumsq sidecar correctness and
measurement. Sweep AWQ formula × α × scope systematically. This is concrete
production work, not research.

**Tier 2 — production:** Validate MQ3-Lloyd as the near-term sub-4-bit
quality lever. The infrastructure is already wired (quantizer + runtime +
kernels exist); needs empirical validation across 0.8B/4B/9B/27B.

**Tier 3 — research:** Residual learnable butterfly prototype. CPU/Python
reference first. Offline optimizer with stop-condition gate. Only port to
HIP if Python beats MQ4+AWQ on 0.8B KLD. This is the lever the user is
hoping works.

**Parallel:** Native ParoQuant via `feat/paro-g256-perfmax` branch
(handoff agent prompt prepared, gated to hiptrx gfx12). External baseline
+ quality-lane shipping option.

## The honest read on the butterfly hope

The user said "I just hope this lever works." Reality check:

**Plausibly works (~50%):** Residual butterfly with `B(0)=I` initialization
is a well-conditioned optimization landscape. ButterflyQuant validates the
broader approach on published benchmarks. The math is clean — it's a
strictly more expressive transform than current fixed-FWHT, so it CAN'T do
worse than current MQ4+AWQ (with B(0)=I init, equality is the floor).

**Plausibly underperforms native ParoQuant:** Butterfly has fewer DOF than
PARO's 8 nested Givens rotations × continuous angles × pair-index search.
The doc's projection of ~0.10-0.12 KLD for butterfly on 0.8B vs ~0.08-0.10
for native PARO is realistic.

**The bear case:** even residual butterfly might suffer the same proxy-fails
mode if the reconstruction loss doesn't co-monotone with KLD on MQ4G256.
The doc anticipates this: stop conditions at smoke stage, don't port to
HIP until Python evidence justifies.

**What it would mean if it works:** hipfire ships a quality lane that
approaches PARO quality at MQ4 perf, without a new GPU kernel architecture.
That's the closest thing to "have your cake and eat it too" available given
the constraints we've now characterized.

**What it would mean if it doesn't:** we accept the 0.1327 floor for the
MQ4 perf lane and ship native ParoQuant for the quality lane via Björn's
PR stack + paro-g256-perfmax. Two-lane shipping, clear separation, no
illusion of getting both at one cost. That's a worse outcome than "butterfly
works" but not a catastrophic one — the path is documented.

## Reference index

Branches:
- `feat/grad-scale-learn` (d458539b) — falsified, kept as research artifact
- `feat/paro-stage2-mq4-prototype` (0ea96ed1) — falsified, kept as research artifact
- `feat/paro-g256-perfmax` (f833925f) — handoff to gfx12 agent, ready
- `worktree-awq-raw-sumsq-converter` (in flight, AWQ stabilization base)

Memory entries (cross-session persistent):
- `project_mq4_falsified_levers_2026_05_22` — v3 / F2 / autoawq / α / GPTQ-noise
- `project_grad_scale_learn_falsified_2026_05_22` — BRECQ+STE +20%
- `project_paro_k0_falsified_2026_05_22` — stage-2-only +360%
- `project_paro_g256_perfmax_branch_2026_05_22` — handoff branch
- `reference_mq4_canonical_recipe` — 0.1327 closed-form best
- `reference_bjorn_paroquant_prs` — #316/#317/#318

Plans:
- `docs/plans/quant-strategy-research-recommendations.md` — the master plan
  going forward
- `docs/plans/imatrix-tier1-hipfire-native.md` — Tier 1 calibration plan

Papers:
- ParoQuant: arXiv 2511.10645 (Liang et al., ICLR 2026)
- ButterflyQuant: arXiv 2509.09679 (Xu et al., Feb 2026 revision)
- AWQ: arXiv 2306.00978 (Lin et al., MLSys 2024)
- GPTQ: arXiv 2210.17323 (Frantar et al., ICLR 2023)
- SpinQuant: arXiv 2405.16406 (Liu et al., ICLR 2025)

---

*"The hunt itself was not wasted — the falsifications narrowed the search
space materially — but the journey was longer than it should have been,
and the pattern of mistakes was repeatable rather than one-off."*

Three closed doors. One opening one (butterfly). One ready alternative
(ship native PARO). That's actually a survivable strategic position. The
session worked.
