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

