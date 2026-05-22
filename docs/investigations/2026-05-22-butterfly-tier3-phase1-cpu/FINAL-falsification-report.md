# Butterfly Residual on MQ4G256 — FINAL Falsification Report (2026-05-22)

After three methodology variants on Qwen3.5-0.8B, the **residual learnable
butterfly transform does not meaningfully reduce model KLD on the hipfire
MQ4G256+AWQ pipeline.** The lever is decisively falsified.

## Test matrix

All three trained the same 138 AWQ-F1 target Linears on the same model
(Qwen3.5-0.8B BF16, mix-v1 corpus, paper-formula AWQ at α=0.55). Only
the optimization methodology differs.

| Phase | Method | LR | Steps | Δ KLD vs baseline | Verdict |
|---|---|---:|---:|---:|---|
| 4 | Joint per-Linear MSE | 1e-3 | 512 (4 × 128) | **+0.31%** | FAIL — worse |
| 4b | Sequential per-Linear MSE | 1e-2 | 138 × 100 | **+0.71%** | FAIL — worse |
| 4c | Joint direct KLD loss | 1e-3 | 128 (4 × 32) | **−0.47%** | FAIL — within noise, absolute 0.146 > 0.10 |

**Sample-size noise band**: same theta=0 model evaluates to KLD 0.108986
at 64 seqs and KLD 0.146909 at 32 seqs (35% difference). The Phase 4c
"improvement" of -0.47% is far below this noise threshold.

## Why each variant failed

### Phase 4 (joint MSE)
- Per-Linear MSE proxy dropped 8% (5.6e-2 → 3.22e-2 mean across 4 epochs)
- Model KLD went +0.31% (worse)
- **Classic proxy-doesn't-transfer**: BRECQ-style per-Linear reconstruction
  loss decreases, but the per-tensor optimizations interact destructively
  when composed across all 138 Linears at the model logit level.
- Same failure mode as
  [[project_grad_scale_learn_falsified_2026_05_22]] and
  [[project_paro_k0_falsified_2026_05_22]].

### Phase 4b (sequential MSE — paper's actual method)
- Hypothesis: joint SGD's cross-tensor gradient interactions cancel the
  per-tensor benefits. ButterflyQuant paper uses sequential layer-by-layer
  reconstruction.
- Eliminated cross-tensor coupling by training one tensor at a time with
  ORACLE inputs cached, then freezing.
- Result: trained KLD **0.109757**, even WORSE than Phase 4's 0.109326.
- Some tensors improved per-Linear MSE (layers 5, 10, 13: −1% to −7%) but
  more regressed at lr=1e-2 (layers 16, 18, 20, 23: +2% to +6%).
- **Falsifies the joint-cancellation hypothesis**: removing joint gradients
  did NOT fix the problem; it made it worse.

### Phase 4c (joint KLD loss — most expensive, decisive)
- Hypothesis: per-Linear MSE is the WRONG proxy. Use the production
  metric directly: KL-divergence between BF16 oracle logits and student
  logits.
- Eliminated proxy mismatch entirely. The "training loss" IS what we
  measure at evaluation.
- Result: trained KLD **0.146217** vs baseline **0.146909**. Δ = −0.47%.
- Within the 35% noise floor of 32-seq KLD measurement.
- **Falsifies the proxy-mismatch hypothesis**: even when proxy =
  production metric, butterfly residual cannot find a meaningful descent
  direction. Absolute KLD still 0.146, well above the 0.10 gate.

## What is now decisively known

1. **The MQ4G256 KLD floor at ~0.13 (production) / ~0.109 (Python 64-seq)
   is real.** It survives all attempted within-MQ4-perf-lane calibration
   levers tried this session and prior:
   - Closed-form AWQ (canonical, 0.1327)
   - v3 / F2 / autoawq / alpha sweep
   - grad-scale-learn (BRECQ + STE)
   - PARO K=0 (stage-1+stage-2 fine-tune)
   - **Butterfly residual (joint MSE / sequential MSE / direct KLD)**

2. **The proxy-doesn't-transfer failure mode applies even when the proxy
   IS the production metric.** The butterfly residual form, parameterized
   as 1024 SO(2) angles per Linear, lacks the expressive degrees of freedom
   to meaningfully reshape the MQ4G256 + AWQ + min-max-RTN loss landscape
   in a useful direction.

3. **The DOF gap is structural, not methodological.** Three completely
   independent optimization methodologies fail. The issue is the form
   itself, not how we train it.

## What might still work (NOT tested, await user direction)

1. **Native (non-residual) butterfly replacing FWHT.** Breaks the
   bisectable property (no theta=0 fallback to current pipeline) but
   gives more DOF — the rotation isn't constrained to be a small
   perturbation around fixed FWHT. Risk: any optimizer mistake makes the
   model unusable. Compute estimate: ~$50.

2. **Higher-order butterflies** (16x16 blocks or 64x64) instead of 2x2.
   More DOF per layer but larger compute footprint. Compute estimate: ~$30.

3. **Learnable D1/D2 sign tables** (instead of FWHT seeds 42/1042).
   Discrete optimization. Different DOF flavor. ~$10.

4. **Different group size** (G128 instead of G256). Halves the dynamic
   range per group. Breaks MQ4G256 storage format. Compute: ~$5 to test.

## What is the right next move (per master plan)

Per [[project_butterfly_pivot_queued_2026_05_22]]:
> ~50%: butterfly hits proxy-doesn't-transfer mode like the others → accept
> 0.1327 floor + ship native PARO via paro-g256-perfmax

This is the realized scenario. The pre-queued fallback is:

**Accept the 0.1327 KLD floor as the MQ4G256 perf-lane quality ceiling.
Ship native ParoQuant via `feat/paro-g256-perfmax` for the quality lane
(separate branch, parallel-path agent).**

PARO pays ~10% runtime perf cost for ~30% PPL reduction. That's the
quality vs perf trade users will get for the upgrade.

## Compute spend

| Phase | Wall-clock | $ (mi300 @ ~$2.5/hr) |
|---|---:|---:|
| 1 (Python CPU) | <1 min | $0.00 |
| 2 (Python self-test) | <1 min | $0.00 |
| 3 (smoke gate) | 0.5 min | $0.02 |
| 4 (full joint MSE) | 25 min | $1.04 |
| 4b (sequential MSE) | 4 min | $0.17 |
| 4c (direct KLD loss) | 6 min | $0.25 |
| **Total butterfly investigation** | **~36 min** | **~$1.48** |

Plan budget: ~$700 for full 16-phase plan. **Spent 0.2% of budget** to
falsify the lever.

## Artifacts kept

All on mi300 (read-only, for forensic value):
- `/workspace/butterfly-phase3-smoke/` (smoke gate PASS data)
- `/workspace/butterfly-phase4-0.8b/` (joint MSE FAIL data)
- `/workspace/butterfly-phase4b-sequential/` (sequential MSE FAIL data)
- `/workspace/butterfly-phase4c-kldloss/` (direct KLD FAIL data)

On the branch:
- `scripts/butterfly_core.py` (numpy reference)
- `scripts/verify_butterfly256.py` (Phase 1 math verifier)
- `scripts/learn_butterfly_mq.py` (full trainer with joint/sequential/KLD modes)
- `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/` (reports)

## Final recommendation

**HALT.** Butterfly residual is exhaustively falsified across joint MSE,
sequential MSE, and joint KLD loss methodologies. No further within-form
methodology variant is expected to break the proxy-doesn't-transfer wall.

Pivot per master plan: accept MQ4G256 floor + ship native ParoQuant via
`feat/paro-g256-perfmax`. User has been notified.

`failed:` butterfly residual on MQ4G256+AWQ falsified across all
methodology variants — joint MSE +0.31%, sequential MSE +0.71%, direct
KLD −0.47% (within noise). Trained KLD ≥ 0.10 in all variants. Locked
HARD gate failed. Pivot to native ParoQuant per pre-queued fallback.
