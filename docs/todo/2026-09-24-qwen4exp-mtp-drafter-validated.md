# Qwen3.8-Flash-Next MTP: the head DRAFTS — 56% acceptance

**Status:** composition VALIDATED 2026-09-24 on the shipped 180B checkpoint.
**The GPU port and spec-decode wiring are justified.**

> ⚠️ **This file previously concluded the opposite** ("does not beat doing
> nothing", "do NOT port"). That was wrong, and the cause is worth keeping: the
> probe fed the trunk a SYNTHETIC prompt (`9707 + i%977`), which drove it into
> degenerate repetition — `198, 271, 487, 220` cycling. That inflated the
> do-nothing baseline to 31.8% (when consecutive tokens repeat, reusing the last
> state "predicts" the next one) and starved the head of a distribution it was
> trained on. Both the cosine and the token metric flipped sign once the trunk
> generated its own continuation. **Measure drafters on a self-consistent
> distribution, never on synthetic ids.**

## The result

Trunk generates autoregressively from one seed token; the head drafts against
its own output. 16 scored positions, `lm_head` applied for real tokens:

    MTP acceptance        = 9/16  (56.2%)
    do-nothing acceptance = 0/16  (0.0%)

56% sits inside the 60-80% band the DeepSeek-V3 paper reports for MTP, on a head
quantised to `oq8` rather than training precision. Matches include rare tokens
(30463, 1193), so it is not chance.

The hidden-state metric agrees once the distribution is right:

    do-nothing: cos(collapsed_t, collapsed_t+1) = +0.4179
    best MTP:                                     +0.4945   (+0.0766)

## The convention, now pinned by measurement

Both free variables the tensor shapes leave open are settled, and `Fusion` is an
enum in `mtp.rs` rather than a comment:

    fusion            emb   target   cos      vs-baseline
    broadcast-all     t+1   t+1     +0.4945    +0.0766   <- WINNER
    broadcast-all     t+0   t+1     +0.3277    -0.0902
    stream0-only      t+1   t+1     +0.3032    -0.1147
    stream0-only      t+0   t+1     +0.2959    -0.1219

`broadcast-all` beats `stream0-only` by ~0.19, and `emb=t+1 -> target=t+1` is the
documented MTP convention (`h_t` + `emb(x_{t+1})` -> `x_{t+2}`). The module's
"natural reading" was right on both counts.

## ⚠️ Two traps this probe exists to hold

1. **Residual inheritance.** `mtp::forward` ends with the SAME mixer collapse the
   trunk uses, over a `wide` built from the trunk's own `wide(t)`, so its output
   inherits the trunk's residual. Raw cosine at `target=t+0` reads +0.63 for a
   head that predicts nothing. An unrelated-position control does NOT catch this
   (an unrelated position does not share the residual). Only the do-nothing
   baseline does. Keep it.
2. **Cosine is a proxy; tokens are the metric.** In 2560 dimensions two states at
   modest cosine can argmax to the same row out of 248320. The cosine gain here
   is a slim +0.0766 while acceptance is 56% — the proxy badly understates it.
   Score `argmax(lm_head . draft)` before believing any verdict.

## Next

1. **`mtp_gpu.rs`** — mirror the CPU forward with the existing `moe_gpu` /
   `hc_gpu` / `attn_gpu` primitives. The head is one sparse-attention layer plus
   a mixer; every kernel it needs already exists.
2. **Spec-decode wiring** — `deepseek4/src/spec_decode.rs` (688 lines) is the
   reference, including the `mtp_last_hidden` carry.
3. **Measure tok/s** against the 5.9 tok/s AR baseline. Note
   `project_dflash2_qwen38_perf`: compact Opus is excluded from batched prefill,
   so a K-token verify can cost K weight sweeps — acceptance alone does not
   guarantee a win.

## Producer/consumer seams found on the way

Nothing had ever consumed an MTP head end to end, and three were broken:

* `--include-prefix mtp.` built an EMPTY artifact and exited 0 — fixed `0e9feb7b3`.
* `QuantType::Oq8G128` (54) missing from `from_code`, so the pager rejected its
  own artifacts — fixed `4f5174f2c`.
* The quantizer SPLITS stacked MoE experts while the CPU forward wants them
  stacked; the probe restacks, and any GPU wiring must too.
