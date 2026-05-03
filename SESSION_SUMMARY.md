# Session summary — Gemma 4 movie-night contract

## Headline

Both immediate priorities shipped clean. Coherence battery 6/6 strict
at session end with the new fused-default code path; outputs
bit-identical to the pre-fused baseline. No regressions, no MANUAL_REVIEW
escalations.

## Priority 1: E4B quality investigation

**Outcome: ROOT CAUSE FOUND, FIXED, COMMITTED.**

The "E4B multilingual gibberish" surfaced in the original gate report
(`[User [] } <b>TheLocal_ES_G2015_IDENTITY_N</b>...`) was not a
quantization issue, not RoPE, not layer drift. It was a missing chat
template wiring.

Hypothesis ladder:
- (a) Tokenizer mismatch — REJECTED. Engine encoded the gate prompt
  to `[2, 3689, 563, 506, 5279, 529, 7001, 236881, 25685, 528, 886,
  2822, 13315, 236761]`, byte-identical to HF reference tokenizer
  output.
- (b) Quant precision — NOT TESTED, skipped after disambiguator below.
- (c) RoPE — NOT TESTED, skipped after disambiguator below.
- (d) Layer norm / final projection — NOT TESTED, skipped after
  disambiguator below.

The disambiguator: quantized E4B-IT (the instruction-tuned variant)
and ran the same prompt through the smoke harness with a manual chat
template wrapper:

  raw prompt:        "What is the capital of France?..."
                     -> "\nWhat is the capital of France?<turn|>omet"  (parrot)
  templated prompt:  "<start_of_turn>user\n{prompt}<end_of_turn>
                      \n<start_of_turn>model\n"
                     -> "The capital of France is Paris.\n<end_of_turn>"

Same model, same engine, same MG4G256 quant. The difference is
solely the chat template. Engine and quantization are correct;
the daemon was running IT models in raw-prompt mode.

Three coupled fixes landed in commit `60ce66f`:

1. **Chat template** — `generate_gemma4` now wraps prompt + system
   in the literal Gemma 4 template (from `chat_template.jinja` in
   the HF E4B-it snapshot) when the model path contains `-it` OR a
   system_prompt is provided OR `HIPFIRE_GEMMA4_CHAT=on`.
2. **Clean stop** — buffers trailing bytes that could be a prefix of
   the multi-token `<end_of_turn>` literal so the marker doesn't
   leak to the client. Also stops on the compact end-of-turn token
   id 106 when chat template is active.
3. **Multi-turn KV consistency** — restructured the decode loop to
   ALWAYS forward the accepted token through the KV writer before
   breaking on stop conditions. Mirrors LLaMA's pattern. The
   previous structure left m.seq_pos and m.conversation_tokens
   advanced past an unwritten KV slot, breaking multi-turn.

Verified end-to-end through the daemon:
  r1: "What is the capital of France?" -> "Paris is the capital of France."
  r2: "And what about Germany?"        -> "Berlin is the capital of Germany."
Both turns coherent, no boundary garbage, 88 tok/s decode on E4B-IT.

Base-model coherence battery: 6/6 strict pass, byte-identical output
to the pre-chat-template baseline (chat-template path is dormant on
non-IT models).

**No MANUAL_REVIEW escalation needed.**

## Priority 2: 26B-A4B MoE indexed-GEMV

**Outcome: PHASE 1 SHIPPED, DEFAULT-ON.**

The per-expert serialized loop (`apply_moe_branch` in gemma4.rs) was
launching 5 kernels × 8 experts = 40 launches per MoE layer, ~1,200
launches per token at 30 layers. The Qwen3.5-MoE A3B path already
had `gemv_hfq4g256_moe_gate_up_k8_indexed` for exactly this fused
case; integrating it required only Rust glue (no kernel authoring
in this session).

Phase 1 (commits `1d86adb` + `7b90b4c`) ships:

1. **Per-layer expert pointer table** — `load_moe_layer_extras`
   builds a `[2 * n_exp]` F32 tensor of u64 device addresses,
   one per expert's `gate_up_proj.buf`. Mirrors qwen35.
2. **Three new scratch buffers** on `Gemma4Scratch`: `moe_pre2_rot`
   (FWHT-rotated input), `moe_expert_gate_batch` and
   `moe_expert_up_batch` ([8 × mi] each, the kernel's split outputs).
3. **Fused dispatch** — replaces 8 `weight_gemv` calls on
   gate_up_proj with one `gemv_hfq4g256_moe_gate_up_k8_indexed`
   launch. Per-expert SwiGLU + Q8 down + scaled-add still
   serialized (no indexed-Q8 down kernel exists today; 26B-A4B's
   down_proj is k=704 which forces the Q8F16 fallback).

Critical bug found mid-integration: kernel signature uses M = full
output row count (= 2 × mi for [2*mi, dim] weight), splitting rows
[0..mi) into y_gate and [mi..2*mi) into y_up. Initial pass of `mi`
ran only the gate half and left y_up uninitialized; top-5 logits
dropped from [25.09 ...] to [20.36 ...]. Fixed to pass `2 * mi`.

Bench:
  smoke   26B-A4B legacy: 66.3 tok/s decode, top-5 [25.09, 24.63, 24.55, 24.51, 24.41]
  smoke   26B-A4B fused:  73.3 tok/s decode, top-5 [25.09, 24.63, 24.55, 24.51, 24.41]
                          BIT-IDENTICAL logits, +10.5% decode

  daemon  r1 80t legacy:  62.4 tok/s decode, BIT-IDENTICAL output
  daemon  r1 80t fused:   67.9 tok/s decode, +8.8%
  daemon  r2 120t legacy: 57.4 tok/s decode, BIT-IDENTICAL output
  daemon  r2 120t fused:  61.9 tok/s decode, +7.8%

Coherence battery with fused enabled: 6/6 strict, output bit-identical
to baseline across all six tests. Default flipped to ON in `7b90b4c`;
`HIPFIRE_GEMMA4_MOE_FUSED=0` is the safety hatch.

Phase 2 (next session): author `gemv_q8_0_moe_down_indexed_k8` to
fuse the down side too. Combined with Phase 1, expected 5-7 tok/s
-> 30-40 tok/s on 26B-A4B (still launch-bound but ~6× fewer launches).
Spec at `docs/plans/gemma4-moe-indexed-gemv.md`.

**No MANUAL_REVIEW escalation needed.**

## Priority 3: Eviction wiring

**Outcome: NOT ENTERED.**

P1 + P2 consumed the available time after accounting for repeated
GPU-lock contention (other agents running `pflash-niah-bench`
batteries throughout the session). Eviction work explicitly gated on
">= 60 minutes remaining after P1 + P2 commit" per the contract;
that threshold not met.

Existing comment in daemon.rs:1112-1118 captures the gap; the next
session should pick up from there with the spec doc as authored
in the contract.

## Coherence battery at session end

**6/6 strict pass.** Snapshot at
`scripts/gemma4-diag/coherence-baseline-fused-default-20260502.md`.
All outputs bit-identical to the pre-session baseline at
`scripts/gemma4-diag/coherence-baseline-20260502.md` for the base
models (E2B, E4B, 31B, 26B-A4B in raw-prompt mode), now with the
fused-default MoE path.

## Defense-in-depth

Both KV-cache bounds checks (prefill loop pos < max_seq, decode
loop pos < max_seq) preserved through the chat-template + fused-MoE
work. The single_turn_floor refuse and BOS-aware seq_pos guard
also preserved.

## Commits this session

  60ce66f feat(daemon/gemma4): chat template + clean stop + multi-turn KV consistency
  997236f test(gemma4): coherence-gate slice for Gemma 4 family    (pre-session baseline)
  1d86adb feat(gemma4/moe): fused indexed-GEMV gate_up (Phase 1, +10% decode)
  7b90b4c feat(gemma4/moe): flip fused indexed-GEMV default to ON

## Outstanding items

- Phase 2 indexed-Q8 down kernel (full MoE perf path)
- Eviction wiring (Priority 3 from contract)
- Chat template auto-detection from .hfq metadata (current path
  heuristic works but isn't first-class)
