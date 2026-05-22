# Phase 16 Report — Butterfly Residual MQ4G256 Investigation

> Per IMPLEMENTATION_PLAN.md, this is the formal Phase 16 deliverable.
> Phases 5-15 were NOT executed: Phase 4 hit the locked HARD stop
> condition (`KLD > 0.10 on 0.8B → STOP, no Rust port`). Three
> methodology variants tested before halting; the lever is decisively
> falsified.

## Per-model × per-arch results table

Only Qwen3.5-0.8B was tested (Phase 4-4c). Phases 5-7 (9B, 27B, A3B-MoE)
not run — gating the run at the smallest model's HARD failure preserved
~$700 of plan budget.

| Model | Arch | KLD (butterfly) | KLD (MQ4+AWQ baseline) | Δ | decode tok/s | speed gate | Status |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | gfx942 (Python sim) — Phase 4 joint MSE | 0.109326 | 0.108986 (Python 64-seq) | +0.31% | n/a | n/a — no Rust port | **FAIL** |
| Qwen3.5-0.8B | gfx942 (Python sim) — Phase 4b sequential MSE | 0.109757 | 0.108986 (Python 64-seq) | +0.71% | n/a | n/a — no Rust port | **FAIL** |
| Qwen3.5-0.8B | gfx942 (Python sim) — Phase 4c direct KLD | 0.146217 | 0.146909 (Python 32-seq) | −0.47% (within noise) | n/a | n/a — no Rust port | **FAIL** |
| Qwen3.5-9B | NOT RUN | — | — | — | — | — | gated by 0.8B fail |
| Qwen3.6-27B | NOT RUN | — | — | — | — | — | gated by 0.8B fail |
| Qwen3.6-35B-A3B | NOT RUN | — | — | — | — | — | gated by 0.8B fail |

Note: production-pipeline KLD (via `eval_hipfire` + kldref binary, the
canonical hipfire eval) was not measured for any variant because the
Rust port (Phase 8) requires Phase 4 to pass. The Python-internal KLD
correlates with the production scale (Python in-memory 0.109 ≈ production
0.1327 from prior session memory) but is not identical.

## Coherence + speed gate pass status

**Not applicable.** The lever was falsified at the Python validation
stage. No HIP kernel ports, no Rust integration, no runtime dispatch.
Coherence and speed gates (Phases 14) are downstream of the Rust port
and were not exercised.

## Cost summary

| Phase | Wall-clock | Compute spend (mi300 @ ~$2.5/hr) |
|---|---:|---:|
| 1 — Python CPU `butterfly256` + verify | <1 min (local) | $0.00 |
| 2 — Python offline optimizer | <1 min (local self-test) | $0.00 |
| 3 — Smoke gate (PASS, -2.5%) | 0.5 min mi300 | $0.02 |
| 4 — Full Python on 0.8B, joint MSE | 25 min mi300 | $1.04 |
| 4b — Per-tensor sequential MSE | 4 min mi300 | $0.17 |
| 4c — Direct KLD loss | 6 min mi300 | $0.25 |
| 5-15 — NOT RUN (gated by 4 fail) | 0 | $0.00 |
| **Total** | **~36 min mi300** | **~$1.48** |

Plan budget: ~$700 for full 16-phase. **Spent 0.2% of budget** to
exhaustively falsify the lever across three methodology variants.

## Recommendation

**SHELF** the residual learnable butterfly form for MQ4G256+AWQ.

The lever is decisively falsified by three independent optimization
methodologies on the smallest dense trunk model. The 1024 SO(2) angles
per Linear in the residual form lack the structural DOF to reshape the
MQ4G256 + AWQ + min-max-RTN loss landscape toward usable KLD descent.

This is the fourth proxy-doesn't-transfer failure on the MQ4G256 perf
lane this session ([[project_mq4_falsified_levers_2026_05_22]],
[[project_grad_scale_learn_falsified_2026_05_22]],
[[project_paro_k0_falsified_2026_05_22]],
[[project_butterfly_residual_falsified_2026_05_22]]). The 0.1327
production KLD floor is now strongly empirically established.

### Pivot per pre-queued master plan

Per [[project_butterfly_pivot_queued_2026_05_22]] (user-acknowledged
50% odds at lever start): **accept the floor + ship native ParoQuant
via `feat/paro-g256-perfmax`** (parallel branch, separate gfx12 agent).
PARO pays ~10% runtime perf for ~30% PPL improvement — that's the
quality vs perf trade for the upgrade.

### Open future research directions (NOT pursued without user direction)

These were enumerated but not tested — they break locked params
(form, storage, perf ceiling), requiring user reauthorization:

1. **Native (non-residual) butterfly replacing FWHT** — breaks
   bisectable property; more DOF. Compute: ~$50.
2. **Higher-order butterflies** (16x16 / 64x64 blocks) — more DOF per
   layer, larger compute footprint. Compute: ~$30.
3. **Learnable D1/D2 FWHT sign tables** — discrete optimization (vs
   continuous theta). ~$10.
4. **G128 instead of G256** — halves dynamic range per group; breaks
   MQ4G256 storage format. ~$5 to test.

## Master push action

**No master push.** The Rust port (Phase 8) was not executed. No code
on `feat/learnable-fwht` outside the branch has any runtime effect on
hipfire. The branch carries only:
- Python scripts (`scripts/butterfly_core.py`, `scripts/verify_butterfly256.py`,
  `scripts/learn_butterfly_mq.py`) — research-only, no production wiring
- Investigation docs (`docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/`)
- IMPLEMENTATION_PLAN.md (the historical plan record)

These are useful for reproducibility if a future agent wants to revisit.
No master/main consequence either way.

**Awaiting user direction** on whether to:
- (a) Merge `feat/learnable-fwht` to master as a research-artifact branch
  (low-risk — Python scripts only, no production code touched)
- (b) Shelf the branch as historical record (keep on origin, no merge)
- (c) Delete the branch
- (d) Pursue one of the open future research directions before deciding (a/b/c)

Defaulting to **HALT** (no master push, no PR) per the locked
operational rule.

## Artifacts kept (forensic value)

mi300 (read-only):
- `/workspace/butterfly-phase3-smoke/` — smoke gate PASS data
- `/workspace/butterfly-phase4-0.8b/` — joint MSE FAIL data
- `/workspace/butterfly-phase4b-sequential/` — sequential MSE FAIL data
- `/workspace/butterfly-phase4c-kldloss/` — direct KLD FAIL data

Branch `feat/learnable-fwht` on `origin`, HEAD will be at `~75300bc5`
after this Phase 16 report commits:
- `IMPLEMENTATION_PLAN.md` (the 16-phase plan)
- `scripts/butterfly_core.py`, `scripts/verify_butterfly256.py`,
  `scripts/learn_butterfly_mq.py`
- `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/` (all reports
  including this Phase 16 final)

Memory: `project_butterfly_residual_falsified_2026_05_22.md` captures
the lever close-out for future-session discoverability.

## Final status

`failed:` butterfly residual MQ4G256 falsified across joint MSE,
sequential MSE, and direct KLD loss methodologies. Locked HARD gate
not satisfied. Master push not requested (lever shelved). Pre-queued
fallback path (native PARO via `feat/paro-g256-perfmax`) is available
on a separate branch with its own agent. User direction required to
either close this branch or pursue out-of-scope future research.
