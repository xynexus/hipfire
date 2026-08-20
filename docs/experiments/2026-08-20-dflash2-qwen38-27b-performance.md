# DFlash2 on Qwen3.8-27B oq4.25++ + CASK — measured

**Status:** measured 2026-08-20 on halo (gfx1151, 128 GB UMA), after the dense
hand-decode-path fix (`b7b7a9ae5`) made dense spec-decode correct at all.
**Verdict:** DFlash2 is **2.3× better than DFlash1** and still a **3.7×
regression against plain decode**. The ceiling is structural, not drafter
quality.

> ⚠️ **The MECHANISM below is retracted; the numbers stand.** A rocprofv3 trace
> (`2026-08-20-dflash-verify-profile.md`) shows DeltaNet is **0.7 %** of GPU time,
> not the bottleneck this document blames. 69 % is `gemm_oq8_grouped_wmma`, the
> batched Opus GEMM, running at 47 GB/s where the decode GEMV gets 215 GB/s on
> the same weights. The true ceiling is **1.0×**, not the 1.22× computed here.
> Read the profile doc for the corrected analysis; everything below the "Numbers"
> table is superseded.

All three configurations emit **byte-identical text** to the no-draft baseline,
which is the losslessness spec-decode is supposed to have and is the first time
it has held on a dense target here.

## Numbers

Prompt: "Explain in two sentences why speculative decoding speeds up inference."
128 new tokens, greedy, `HIPFIRE_DFLASH_ALLOW_OPUS=1`.

| config | KV | decode tok/s | τ | accept rate |
|---|---|---|---|---|
| baseline, no draft | f32 | **7.50** | — | — |
| DFlash1 drafter, B=16 | f32 | 0.79 | 1.098 | 0.069 |
| DFlash2 drafter, B=8 | f32 | 1.79 | 1.804 | 0.226 |
| DFlash2 drafter, B=8 | q8 | **2.01** | 1.778 | 0.222 |

DFlash2's block is 8 where DFlash1's is 16, and it drafts in 74 ms/cycle against
DFlash1's 197 ms — half the block for half the cost, at 2.6× the acceptance rate.
Note the selector is not applied yet (see the handover), so 1.78 is a floor for
what this checkpoint can do, not its number.

## Why it still loses — the arithmetic

`HIPFIRE_SPEC_PHASES=1`, q8 KV, steady state:

    draft=74ms  ngram=1.8ms  verify=910ms  restore=1.8ms  replay=131ms × accepted

Baseline decode is 133 ms/token. So **verify costs ~6.8 serial decodes to check 9
positions** — it barely amortizes. Best case, at 100 % acceptance (accept=8, no
replay):

    (74 + 910) ms / 9 tokens = 109 ms/token  ->  9.2 tok/s  vs  7.5 baseline

**1.22× is the absolute ceiling** for DFlash on this target as it stands, and only
if the drafter were perfect. No drafter improvement — DFlash2, the selector, a
larger block — can get past it.

## Why verify does not amortize

Qwen3.8-27B is a HYBRID stack: 48 of 64 layers are LinearAttention (DeltaNet),
16 are FullAttention. Batched prefill's own comment says it plainly — "the inner
`gated_delta_net` batch_seq loop is still sequential per token, so the per-chunk
DeltaNet cost is linear in N either way; raising the batch just amortizes the
NON-DeltaNet kernels". Three quarters of the stack is a recurrence that a wider
batch does not help.

The batched path is worth engaging anyway, but only just: f32 KV forces the
per-token fallback (`kv_f32` in `forward_prefill_batch_with_pbs_opts`), and
switching to q8 KV flips `final=false` -> `final=true` for a verify of
1130 ms -> 910 ms, i.e. **20 %**. That is the whole amortization available today.

This generalizes the earlier MoE result rather than contradicting it. On
Qwen3.5-35B-A3B the blocker was expert bytes (every position picks its own top-8
of 128 experts, so batched verify reads ~17× the bytes one decode does); here it
is the DeltaNet recurrence. Both are cases where the batch dimension buys nothing.

## What would actually move it

1. **Batch the DeltaNet recurrence across the verify block.** This is the only
   change that lifts the 1.22× ceiling. The chunked-SSD prefill work on Nemotron
   Mamba-2 is the shape of the answer — a block-scan formulation instead of a
   per-token loop.
2. **Kill `replay`.** 131 ms per accepted token, a full serial decode each, purely
   to re-advance DeltaNet state over tokens verify already ran. At accept=4 that is
   40 % of the cycle.

   `HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE=1` measured as an exact no-op here (τ and
   tok/s identical to the digit) for a reason that has nothing to do with the tape
   being populated: the arm carries `&& !gpu.arch_caps.is_rdna3p5()`, and halo is
   gfx1151, so on this machine the flag can never take effect. Do not read that
   no-op as evidence about the tape.

   It would not have helped anyway. The serial-tape arm loops
   `forward_scratch_capture_gdn_tape` over the accepted prefix — one full serial
   forward per accepted token, the same cost as the replay it replaces, just
   capturing a tape while it goes. `PrefixVerify` runs a whole extra verify. Every
   implemented rollback mechanism pays a forward per accepted token.

   So killing replay is not a matter of picking a different existing arm. It needs
   DeltaNet state to be *checkpointed per position during verify* (verify already
   walks those positions) so rollback is a restore instead of a re-run. That is the
   same batched-recurrence work as (1), which is a reason to do (1) first.
3. **The candidate selector**, for τ. Cheap relative to the above and it is the
   drafter this checkpoint actually describes — but per the ceiling arithmetic it
   improves the numerator of a fraction whose denominator is the problem.

Order matters: (1) is the only one that changes the verdict. Doing (3) first
produces a better drafter for a mechanism that still cannot pay.

## Reproduce

    HIPFIRE_KV_MODE=q8 HIPFIRE_KV_ALLOW_DEPRECATED=1 HIPFIRE_DFLASH_ALLOW_OPUS=1 \
    HIPFIRE_SPEC_PHASES=1 ./target/release/hipfire-daemon <<'JSON'
    {"type":"load","model":"/home/sadara/.hipfire/models/Qwen3.8-27B--oq4.25++.hfq","params":{"max_seq":4096,"draft":"/home/sadara/.hipfire/drafts/Qwen3.8-27B--dflash2.oq4+.hfq"}}
    {"type":"generate","id":"p0","prompt":"Explain in two sentences why speculative decoding speeds up inference.","temperature":0.0,"max_tokens":128,"thinking":false}
    {"type":"unload"}
    JSON

Note q8 KV is a DEPRECATED tier kept alive for this measurement; kvarn8 is the
supported quantized KV but does not satisfy the batched-verify key predicate
(`kv_cache.quant_q8 || quant_hfq4`), so DFlash silently falls back to plain AR
under it. Wiring kvarn into that predicate is a prerequisite for shipping any of
this.
