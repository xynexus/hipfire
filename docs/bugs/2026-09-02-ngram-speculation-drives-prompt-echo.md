# n-gram speculation can drive the model into echoing its prompt — and the "speedup" is that corruption

Status: **OPEN, mechanism measured.** Found 2026-09-02 while auditing why an
earlier throughput figure would not reproduce. It is the practical consequence of
`2026-09-01-spec-decode-not-output-equivalent-to-ar.md`, and it upgrades that bug
from "output differs" to "output degrades".

## What happens

`qwen3.5-0.8b--oq4++.hfq`, greedy, prompt = 6151 chars of a Rust source file,
two 400-token generations per run, three runs per configuration. `echo_frac` is
the fraction of 40-char windows of the OUTPUT that appear verbatim in the prompt.

| configuration | echoed | mean coverage | mean tok/s |
|---|---|---|---|
| n-gram ON, FP16 DeltaNet state | **3 of 6** (every g2 at 87.2%) | 0.323 | 161.4 |
| n-gram ON, FP32 DeltaNet state | 0 of 6 | 0.388 | 76.5 |
| n-gram OFF, FP16 DeltaNet state | 0 of 2 | — | 71.9 |

Byte-reproducible: all three reps of each cell agree to the digit.

Two facts do the work:

- **Neither ingredient degenerates alone.** FP16 state without speculation never
  echoes. Speculation with FP32 state never echoes. Only the combination does.
- **The echoing runs are the fast ones.** The 87.2%-echo generations run at
  263 tok/s; the non-echo FP16 generations run at 59. The throughput is bought
  with the corruption.

## Mechanism

1. Speculative decode is not bit-exact on this stack — the batched verify and the
   per-token AR path disagree under quantised KV / narrow state (see the
   companion doc; fp32 KV is byte-identical to AR, everything else is not).
2. The n-gram tier drafts from what it has observed, and it observes the PROMPT
   (`ng.observe(&prompt_tokens)` at request start).
3. A verify that is not exact accepts some drafted tokens the target would not
   have chosen on its own.
4. Each accepted prompt-derived token pulls the continuation further toward the
   prompt.
5. The more the output resembles the prompt, the better the n-gram predicts it —
   coverage rises toward 1.0, draft lengths grow, and throughput climbs.

It is a feedback loop, and the verify is exactly the component that is supposed
to break it. It cannot, because it does not reproduce the target's own argmax.

## Why this matters more than the losslessness bug reads on its own

"Speculative output differs from AR output" sounds like a numerical footnote.
This is what it cashes out as: a 400-token answer that is 87% copied from the
question, produced deterministically, while the throughput counter reports a
3.4x speedup. Both the quality metric and the performance metric mislead in the
same direction at the same time.

It also invalidates a benchmark. The 328/397 tok/s recorded in
`2026-09-01-ngram-spine-discarded-by-block-fallback.md` was measured under FP16
state with n-gram speculation on — the exact configuration above. That figure is
marked indicative in its own document; this is why.

## Mitigation in place, and its limit

`2026-09-02-deltanet-redundancy-gate-had-no-caller.md` restores the guard that
forces FP32 state below a redundancy floor of 3000. The 0.8B measures 2048, so it
now gets FP32 and does not degenerate (0 of 6 above).

**That is a mitigation, not a fix, and it does not cover every model.**
`Qwen3.6-35B-A3B` measures 4096, above the floor, so it still runs FP16 state. It
was not tested here — the loop requires a model, a prompt long enough to seed the
n-gram, and enough tokens to fall in. Whether models above the floor exhibit the
same loop is **open and worth checking before trusting n-gram speculation on
them**.

The real fix is upstream: make the verify reproduce AR. While it does not, any
draft source that can propose prompt text can bias the output toward it.

## Reproduction

    # degenerates (g2 echoes 87%)
    HIPFIRE_NGRAM_SPEC=1 HIPFIRE_DN_STATE_FP32_BELOW=0 hipfire daemon < echo.jsonl
    # does not
    HIPFIRE_NGRAM_SPEC=1 hipfire daemon < echo.jsonl
    # does not (speculation off)
    HIPFIRE_NGRAM_SPEC=0 HIPFIRE_DN_STATE_FP32_BELOW=0 hipfire daemon < echo.jsonl

`echo.jsonl` loads the model and issues two identical 400-token greedy generates
whose prompt is `"Here is a Rust module:\n\n"` plus 6151 chars of
`crates/hipfire-specdecode-ngram/src/hot.rs`. The first generation never echoes;
the second always does. Something carried between requests is required to enter
the loop — identifying it is the next step.
