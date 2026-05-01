# Overnight Plan 2026-05-01 (UTC)

Branch: `overnight/2026-05-01`
Start: 2026-05-01T08:20Z
Principal: Kaden, asleep ~8h
Agent: Claude Code (Opus 4.7)

## Pre-flight (already on master, pushed)

- `4e1a7e2` feat(cli): tri-state mmq_screen toggle (off/on/auto) with legacy migration.
- `fa93b13` fix(install): Windows kernel-compile path parity with Linux (#112). Master has the fix; #112 is closeable from a code POV; reply queued in Phase 1.

## Triage results

Classified live issue list (`gh issue list --state open --limit 100`).

### Tier 1: correctness (do first)

| # | Reporter | Title | Notes |
|---|---|---|---|
| 111 | Fluorax | MQ4 tool-call JSON malformation, Qwen3.6:27b | Full repro vs llama.cpp baseline. Root cause hypothesis: MQ4 distribution shift on structured tokens (`{`, `"`, `:`, `}`); same class as #87. Plan: replicate the malformation pattern locally on 7900 XTX, characterize, ship a parseToolCalls repair fallback (defensive, ships value tonight) plus log the calibration ask in MANUAL_REVIEW for offline retrain. |
| 19  | self     | hipGraph MoE divergence ~40 tokens     | Known; escalate, not closing tonight. |
| 30  | self     | qkvza_hfq4g256 multi-block divergence  | Known; escalate. |
| 50  | thelittlefireman | gfx1152 segfault + incoherent | No Strix Halo hardware; escalate with concrete debug ask. |

### Tier 2: deployment

| # | Reporter | Title | Notes |
|---|---|---|---|
| 110 | Fluorax | DFlash draft path uses `process.cwd()`, fails in Docker | Fix in `cli/index.ts` ~line 390 candidate-path block. Use `dirname(resolveModelPath(target))` first; keep cwd + ~/.hipfire as fallbacks. Reply with workaround confirmation + commit. |
| 112 | djismgaming | compile-kernels.ps1 missing | ALREADY FIXED on master `fa93b13`. Comment + reply. |
| 87  | self     | auto-MMQ regression on tool-call (gfx1151) | ADDRESSED by #104 + tonight's tri-state toggle (`4e1a7e2`). Reply summary, leave open for Kaden to close. |
| 82  | dero     | Windows hipcc space-in-path           | Diagnosed already. Fix: `cfg(windows)` GetShortPathNameW conversion of `-I<HIP_PATH>/include` in `compiler.rs:260`. Implement on Linux, escalate verification (Windows-only). |

### Tier 3: docs / perf

| # | Reporter | Title | Notes |
|---|---|---|---|
| 107 | JonhJonhD | preserve-thinking + chat template | Docs ask. Plan: add a section to `docs/MODELS.md` covering thinking modes, max_think_tokens, chat-template handling per arch family, and link from `GETTING_STARTED.md`. |

### Defer to morning (DEFERRED.md)

- #105 CPU+GPU split (feature; llama.cpp parity ask).
- #92 DFlash drafts for MoE variants (already in flight, depends on Path C training).
- #58, #63, #76, #77, #78 (features / design proposals).
- #38, #39, #40, #41, #42, #43, #45, #60, #61, #70, #89, #113, #114, #115, #116 (roadmap / research / known help-wanted).

### Already-closed-by-other-work (note in reply)

- #112 by `fa93b13` (compile-kernels.ps1 + daemon precompile parity).
- #87 by PR #104 + `4e1a7e2` (default OFF + tri-state toggle).

## Phase 2: Megakernel (after issue queue)

Goals from contract:
1. MQ3 megakernel performant + enhanced. Tok/s win on gfx1100, zero quality regression. VGPR before/after.
2. MQ4 megakernel if time.

Pre-Phase-2 reading:
- `kernels/src/gemv_mq3g256.hip` (current GEMV path, baseline).
- `kernels/src/gemv_mq3g256_residual*.hip` (recent fusion, 2026-04-27 PR #71).
- `kernels/src/gemm_qkv_hfq4g256_wmma.hip` and `gemm_gate_up_*` (HFQ4 fused projection pattern; Lloyd-MQ3 K4-unroll proposal lives here).
- Memory footprint per layer: norm + gate + up + silu*mul + down + residual. Six launches per FFN block currently; megakernel target = one launch.

Highest-leverage candidate: FFN megakernel for the dense path. Norm + gate + up + silu*mul + down + residual fused. ~10-20% decode gain projected. Quality risk: numerical ordering of fused fp16 mul-add chain vs current per-stage path (RMSNorm scale in particular). Test against hipfire perplexity harness on wikitext2-test.

## Termination

Stop when:
- Kaden returns and engages.
- All Phase 1 issues are closed/escalated AND Phase 2 yields three consecutive regressions on its current target.
- Contract clock runs out.

Final state: clean tree on `overnight/2026-05-01` (or merged back to master where appropriate); `OVERNIGHT_LOG.md` summary block at top; `MANUAL_REVIEW.md` sorted by unblock impact; `bench/overnight-*.txt` for each kernel rev.
