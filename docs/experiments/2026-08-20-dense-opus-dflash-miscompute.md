# Dense + Opus + DFlash miscomputes (MoE is fine)

**Status:** OPEN. Root cause unresolved after eight measured eliminations.
**Impact:** blocks all DFlash speculative-decode work on dense Opus targets,
and therefore any DFlash/DFlash2 performance measurement on Qwen3.8-27B.
**Guard:** Opus lm_head quant types stay refused at load; `HIPFIRE_DFLASH_ALLOW_OPUS=1`
opts in for debugging only. Dense drafters on disk are renamed
`*.parked-dense-opus-miscompute` so sibling discovery cannot pick them up.

## Symptom

A dense Qwen3.5-family target quantized to Opus, with a DFlash1 drafter attached,
emits garbage from the FIRST token and runs ~20x slower than plain decode.

    Qwen3.8-27B--oq4.25++  plain          7.5  tok/s   '<think>\nThe user wants...'
    Qwen3.8-27B--oq4.25++  + DFlash1      0.40 tok/s   '嘟 plain'
    Qwen3.6-27B--oq4.25++  + DFlash1      0.41 tok/s   'extr'

The first committed token is already wrong, which rules out "drafts are poor";
spec-decode is lossless by construction (the target verifies every token), so a
bad drafter can only cost acceptance rate, never correctness.

## Control that WORKS

    Qwen3.5-35B-A3B--oq4.25++ (MoE) + DFlash1   8.17 tok/s, coherent
    same baseline                                35.7 tok/s

Same quant family, same lm_head quant type (qt=36 OqPlusCompact), same Opus
DeltaNet projections carrying AWQ sidecars, same CASK sidecar. The ONLY salient
difference is dense vs MoE.

## Reproduce

    # drafter must be un-parked first
    mv ~/.hipfire/drafts/Qwen3.8-27B--dflash.oq4+.hfq.parked-* \
       ~/.hipfire/drafts/Qwen3.8-27B--dflash.oq4+.hfq
    HIPFIRE_DFLASH_ALLOW_OPUS=1 ./target/release/hipfire-daemon <<'JSON'
    {"type":"load","model":"/home/sadara/.hipfire/models/Qwen3.8-27B--oq4.25++.hfq","params":{"max_seq":4096}}
    {"type":"generate","id":"p0","prompt":"Explain in two sentences why speculative decoding speeds up inference.","temperature":0.0,"max_tokens":48,"thinking":false}
    {"type":"unload"}
    JSON

Useful switches: `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1`, `HIPFIRE_SPEC_PHASES=1`,
`HIPFIRE_VERIFY_GRAPH=0`, `HIPFIRE_DFLASH_NO_BATCHED_LMHEAD=1`,
`HIPFIRE_DFLASH_ROLLBACK_COMPARE=1`.

## Eliminated — each by measurement, do not re-test

| # | Hypothesis | How it was ruled out |
|---|---|---|
| 1 | Opus batched lm_head verify arms | `HIPFIRE_DFLASH_NO_BATCHED_LMHEAD=1` forces the non-batched route: byte-identical garbage |
| 2 | CASK / TriAttention | identical garbage with the sidecar parked |
| 3 | Verify graph capture | identical with `HIPFIRE_VERIFY_GRAPH=0` |
| 4 | Opus batched GEMM kernels | `parity_oq8_gemm` 45.15 dB SQNR; `parity_gemm_oq_compact` bit-identical |
| 5 | Drafter lineage / mismatch | z-lab's own MATCHED Qwen3.6-27B drafter garbles too, on a second dense model |
| 6 | Stale GDN tape replay | real bug, fixed in `2ba31acd3`; output unchanged |
| 7 | `prefill_batch.rs` batched DeltaNet arms | instrumented one-shot prints: NEITHER arm fires, for either model, draft or not — dead code here |
| 8 | KV eviction / short KV ring | `max_seq=896` so `physical_cap == max_seq` still garbles; eviction is CASK-driven, and (2) already cleared CASK |

## What IS established

* An rocprofv3 draft-vs-plain kernel diff shows the draft run adds ONLY drafter
  kernels (`gemm_dflash_oq4_plain_dp4a_staged_8w`, `quantize_dflash_act_g256`,
  `attention_dflash_f32`, `rope_batched_f32`) plus the lm_head arm
  (`gemm_oq8_grouped_wmma` + `quantize_act_oq8`). The target's verify body runs
  the SAME kernels as plain decode — there is no separate dtype-specific body.
* `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1` shows BOTH models fall through to the
  per-token fallback in `forward_prefill_batch_with_pbs_opts`:

      dense 3.8-27B : final=false base=true  kv_f32=true   dn_quant=FP32
      MoE   35B-A3B : final=false base=false kv_f32=true   dn_quant=FP32

  The dense `base=true` / `final=false` disagreement is what motivated fix (6).

## THE OPEN QUESTION

Both models take the **same per-token fallback**, yet only the dense one
miscomputes. Same code path, opposite outcomes. Start there.

Concretely: the fallback loops `forward_scratch*` per token and is described as
"byte-identical to decode" — and plain decode on this exact model is correct.
So either the fallback is not in fact byte-identical to decode when driven by
verify (position/state setup across the block), or something downstream of it
consumes its output differently for dense than for MoE. Suggested first probe:
dump the per-position `final_hidden` rows the fallback produces during verify
and diff them against the same positions produced by serial decode on identical
input; the divergence point localizes it.

## Why this matters beyond the bug

DFlash on the working MoE control is a **4.4x regression** (8.17 vs 35.7 tok/s).
`HIPFIRE_SPEC_PHASES=1` per B=16 cycle: `verify=540ms draft=47ms replay=0-402ms`,
against a 28ms/token baseline — verifying 17 positions costs ~17 serial decodes,
because on an A3B every position picks its own top-8 of 128 experts, so batched
verify reads ~17x the expert bytes one decode does and amortizes NOTHING.

Speculation should pay on DENSE targets, where block weights are shared and
verify genuinely amortizes — which is exactly the path this bug breaks. That is
why fixing this outranks implementing better drafters (DFlash2 included):
a better draft cannot fix a verify that costs as much as decoding outright.

Related: `replay` scales WITH acceptance at ~28ms per accepted token
(accept=0 -> 27ms, accept=14 -> 402ms, accept=15 -> 0ms), i.e. accepted tokens
are paid for twice. Independent defect, worth fixing on its own.

See also `docs/todo/2026-08-20-handover-dflash2-qwen38-27b.md`.
