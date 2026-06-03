# MQ3 / MQ3-Lloyd gfx1151 Size-Gate Audit

- date: 2026-06-03
- arch: gfx1151
- commit: fab9d2bc88d2de7b5febfa9ec8afed80b6700557
- branch: qwen35-native-mtp
- control format: MQ4
- scope: current local plain MQ3 and MQ3-Lloyd readiness evidence

This audit keeps the size-gated MQ3 decision explicit. A hard-error-free
coherence smoke is treated as dispatch/runtime evidence only when the decoded
text hits a token cap before a clean final answer.

## Current inventory

Plain MQ3 canonical symlinks and SHA-256 provenance are recorded in
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq3-artifact-provenance.json`.
The current local inventory covers:

| fixture | canonical candidate | current role |
|---|---|---|
| Qwen3.5 4B dense | `qwen3.5-4b-mq3.hfq` | sub-9B boundary check |
| Qwen3.5 9B dense | `qwen3.5-9b-mq3.hfq` | dense quality boundary |
| Qwen3.5 27B dense | `qwen3.5-27b-mq3.hfq` | larger dense candidate |
| Qwen3.5 35B-A3B MoE | `qwen3.5-35b-a3b-mq3.hfq` | routed-expert candidate |
| Qwen3.6 35B-A3B MoE | `qwen3.6-35b-a3b-mq3.hfq` | refresh routed-expert candidate |

MQ3-Lloyd now has one current canonical 9B artifact in the
`/home/sadara/Models/hipfire-candidates/gfx1151-readiness` inventory:
`qwen3.5-9b-lloyd-mq3.hfq` (`4568261632` bytes, md5
`d540c7b7c66d0420c8e6ae5e88a3f0ec`, sha256
`b9c0f3970af064f12b078b2c2e2c4ef34e2275f7ad592eccbae223bbe3ce9bdc`).
It is exposed to the loader as `qwen3.5-9b.mq3-lloyd`. Current
MQ3-Lloyd 4B, 27B, and A3B artifacts are still missing.

## Size decisions

| fixture | current decision | evidence |
|---|---|---|
| Qwen3.5 4B dense | reject / boundary weakness | `2026-06-03-mq3-boundary-coherence.md` completed without hard errors, but the 4B Paris row hit the 80-token cap while thinking and did not emit the requested final sentence. `2026-06-03-mq3-4b-kld.json` then regressed versus MQ4 on the pinned 4B BF16 ref: MQ3 mean KLD `0.5174443790`, PPL `13.2078443`; MQ4 mean KLD `0.1309652724`, PPL `9.99321`. |
| Qwen3.5 9B dense | reject / quality-risky | `2026-06-03-mq3-coherence.md` and `2026-06-03-mq3-boundary-coherence.md` both hit the 300-token cap on the sheep prompt before a clean final answer. Bounded PPL regressed versus MQ4 (`17.6010` vs `12.0679`) and c20 BF16-referenced KLD regressed versus MQ4 (`0.8255300002` vs `0.2363852917`). |
| Qwen3.5 27B dense | candidate, incomplete | Paris coherence completed cleanly, bounded PPL was favorable to MQ3 (`12.2539` vs MQ4 `12.7570`), and 3-run AR/DFlash perf medians now exist. The dense gate still lacks a comparable qwen3.5-27B BF16/Q8 KLD reference; see `2026-06-03-mq3-27b-kld-reference-audit.md`. |
| Qwen3.5 35B-A3B MoE | research candidate | `2026-06-03-mq3-a3b-coherence.md` completed the sheep prompt with a clean final answer of `9`. `2026-06-03-mq3-a3b-ppl.json` improved versus MQ4 on the bounded slice (`8.2158` vs `8.5826`). `2026-06-03-mq3-a3b-broader-coherence.json` added trains, HumanEval, and long-LRU prompt coverage with no hard errors; HumanEval completed cleanly, trains capped for both MQ4/MQ3, and long-LRU capped for MQ3 while MQ4 completed. A3B KLD is blocked on a missing matching HFKLDR reference; see `2026-06-03-mq3-a3b-kld-reference-audit.md`. DFlash/spec is blocked on a missing paired A3B draft sidecar; see `2026-06-03-mq3-a3b-dflash-fixture-audit.md`. |
| Qwen3.6 35B-A3B MoE | research candidate | `2026-06-03-mq3-a3b-coherence.md` completed the sheep prompt with a clean final answer of `9`. `2026-06-03-mq3-a3b-ppl.json` regressed versus MQ4 on the bounded slice (`6.5041` vs `6.3211`). `2026-06-03-mq3-a3b-broader-coherence.json` added trains, HumanEval, and long-LRU prompt coverage with no hard errors; HumanEval completed cleanly, trains capped for both MQ4/MQ3, and long-LRU capped for both MQ4/MQ3. A3B KLD is blocked on a missing matching HFKLDR reference; see `2026-06-03-mq3-a3b-kld-reference-audit.md`. DFlash/spec is blocked on a missing paired A3B draft sidecar; see `2026-06-03-mq3-a3b-dflash-fixture-audit.md`. |
| MQ3-Lloyd 9B | research-gated / dense-promotion rejected | `2026-06-03-coherence-full-after-mq3-lloyd.md` completed the sheep and long-prefill LRU rows without hard errors. The sheep row ended with final number `9`; the long-prefill row gave a coherent O(1) get/put LRU explanation. `2026-06-03-mq3-lloyd-9b-kld.md` then rejected dense promotion for this artifact on the c20 BF16-referenced gate: MQ3-Lloyd mean KLD `0.553297`, PPL `9.64255`; MQ4 control mean KLD `0.236385`, PPL `9.04742`. Historical gfx1100 9B KLD improved over plain MQ3 (`1.6913` vs `2.6221`) but lagged MQ4 (`0.8762`) and MQ6 (`0.6254`), so it remains prior evidence only. |

## Required next gates

- Do not promote Qwen3.5 4B or 9B plain MQ3 on the current evidence.
- Generate or locate and manifest-pin a comparable qwen3.5-27B BF16/Q8 KLD
  reference, then run 27B KLD before any dense MQ3 promotion claim.
- Generate or locate and manifest-pin matching A3B HFKLDR references, then run
  the prepared A3B MQ4/MQ3 KLD cases before any MoE MQ3 promotion claim.
- Generate or locate and manifest-pin paired A3B DFlash draft sidecars before
  A3B DFlash/spec rows can count toward any MoE MQ3 promotion claim.
- Follow up capped A3B trains and long-LRU rows with tighter max-token or
  no-think prompt shapes if MQ3 A3B remains a promotion target.
- Broaden the 27B AR and DFlash/spec prompt envelope before a release-facing
  perf claim.
- Do not spend 9B MQ3-Lloyd promotion perf time on the current artifact hash;
  it failed the current BF16-referenced KLD/PPL gate versus MQ4.
- Only rerun 9B MQ3-Lloyd quality gates after a producer or calibration change
  creates a new artifact hash.
- Generate or locate current MQ3-Lloyd 4B/27B/A3B artifacts before expanding
  the lane beyond the single 9B artifact.
