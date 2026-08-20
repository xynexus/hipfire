# Dense DFlash miscomputes — ROOT CAUSE: the hand decode path is broken on dense

**Status:** ROOT-CAUSED 2026-08-20. Not "dense + Opus + DFlash" at all.
**Cause:** the hand-written decode arms in `crates/hipfire-arch-qwen35/src/qwen35/decode_layers.rs`
miscompute on DENSE qwen3.5-family models. DFlash verify is the only caller left
that forces them, so the bug looked like a DFlash bug.
**Reproduce in ~10 seconds**, no drafter and no Opus quant needed:

    HIPFIRE_FORWARD_LOWERED=0 ./target/release/hipfire-daemon <<'JSON'
    {"type":"load","model":"/home/sadara/.hipfire/models/qwen3.5-2b--bf16.hfq","params":{"max_seq":4096}}
    {"type":"generate","id":"p0","prompt":"Explain in two sentences why speculative decoding speeds up inference.","temperature":0.0,"max_tokens":48,"thinking":false}
    {"type":"unload"}
    JSON

    LOWERED=1 (default)  '<think>\nThinking Process:\n\n1.  **Analyze the Request:** ...'
    LOWERED=0 (hand)     '...\n0...  ,0...  $ $0...$0...$0...$0...$0...$0...$'

## Why DFlash was the only thing that noticed

`forward_scratch_layers` (`decode_layers.rs:11`) routes to the lowered super-op
executor by default and skips it in exactly three cases:

    if forward_lowered_enabled()
        && hidden_rb.is_none()          // <-- DFlash verify passes Some
        && gdn_tape_capture.is_none()   // <-- DFlash verify passes Some
        && !rq_hand_optin
        && !hipfire_steer::is_active()

Spec-decode verify needs per-position hidden extraction (`hidden_rb`) to feed the
drafter, and optionally a GDN tape. Both force the hand arms. Production decode
never does — so the hand path rotted unobserved. The comment above that branch
already recorded the breakage ("bf16 self-KLD 13.89 vs lowered 0.000"); what was
missing was that DFlash verify is a live caller of it.

## Evidence

| config | forward path | result |
|---|---|---|
| Qwen3.6-27B bf16, plain decode | lowered (default) | coherent |
| Qwen3.6-27B bf16, plain decode, `HIPFIRE_FORWARD_LOWERED=0` | hand | **1-2 tokens, empty** |
| Qwen3.6-27B bf16 + DFlash1 | hand (forced by `hidden_rb`) | **garbage, accept=0** |
| Qwen3.5-35B-A3B bf16, plain decode, `HIPFIRE_FORWARD_LOWERED=0` | hand | coherent, τ 4.33 w/ DFlash |
| qwen3.5-2b bf16, plain decode, `HIPFIRE_FORWARD_LOWERED=0` | hand | **garbage** |

So the fault is in the hand path's DENSE arms (`LayerWeights::DeltaNet` /
`FullAttention`). The MoE arms (`DeltaNetMoe` / `FullAttnMoe`) are fine, which is
the entire dense-vs-MoE asymmetry that made this look like a DFlash bug.

Two consequences for the old elimination list: every one of the eight
hypotheses was inside DFlash machinery, which is why none of them landed. And
**Opus is eliminated** — a plain bf16 dense target fails identically.

The direct probe (`HIPFIRE_DFLASH_VERIFY_DEBUG=1`, added with this writeup)
localizes it to verify slot 0, which is not speculative at all — slot 0 is the
already-committed seed token, so its argmax is just what plain decode emits:

    [verify-dbg] start_pos=20 b=16 mode=direct
                 in=[96570, 239, 11, 198, ...] argmax=[248046, 10733, ...]

`in[0]=96570` is `<think>`, the token plain decode had just committed; plain
decode's next token is `198` (`\n`) but verify's slot 0 says `248046`. One
non-speculative forward, wrong. That is the whole bug.

## Getting a bf16 target to load with a drafter

`dflash_batched_lm_head_supported` / the load-time whitelist now admit BF16 (16)
and the losslessly recoded pair Bf16Lut3 (49) / Bf16Huff (50), with matching arms
in `dflash_enqueue_verify_lm_head`. That is what makes the unquantized control
above runnable; it is also a real feature (bf16 targets can attach a drafter).

## What to do next

Two options, in preference order:

1. **Teach the lowered executor to populate `hidden_rb` and the GDN tape**, then
   delete the hand arms. Converge-and-delete: one forward path, and spec-decode
   stops being the only user of a path nothing else exercises.
2. Fix the dense hand arms. Cheaper short-term, but keeps two forwards alive and
   the next rot is only a matter of time.

Either way bisect on `qwen3.5-2b--bf16.hfq` with `HIPFIRE_FORWARD_LOWERED=0` —
seconds per iteration, and no drafter in the loop.

## Still true, still separate (measured on the MoE control)

DFlash on Qwen3.5-35B-A3B oq4.25++ is a **4.4x regression** (8.17 vs 35.7 tok/s).
`HIPFIRE_SPEC_PHASES=1` per B=16 cycle: `verify=540ms draft=47ms replay=0-402ms`
against a 28ms/token baseline — verifying 17 positions costs ~17 serial decodes,
because on an A3B every position picks its own top-8 of 128 experts, so batched
verify reads ~17x the expert bytes one decode does and amortizes NOTHING.
Speculation should pay on DENSE targets, where block weights are shared — which
is exactly the path this bug broke.

Also independent: `replay` scales WITH acceptance at ~28ms per accepted token
(accept=0 -> 27ms, accept=14 -> 402ms, accept=15 -> 0ms), i.e. accepted tokens
are paid for twice.

See also `docs/todo/2026-08-20-handover-dflash2-qwen38-27b.md`.
