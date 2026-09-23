# Qwen3.8-Flash-Next MTP: the head does not beat doing nothing

**Status:** open. Measured 2026-09-24 on the shipped 180B checkpoint.
**Bottom line: do NOT port `mtp.rs` to GPU or wire spec-decode yet.** The head as
composed adds no predictive value, and a drafter built on it would look healthy
while never earning its weights.

## What was measured

`examples/mtp_probe`, two phases so the 9.7 GiB head and the ~20 GiB paged trunk
never coexist (holding both OOM-killed it twice on this 124 GB UMA box):

    mtp_probe dump  <base.hfq> states.bin 16     # trunk on GPU -> 961 KB of states
    mtp_probe score <mtp.hfq>  states.bin        # head on CPU, scored

Per position `t` the head is given the trunk's WIDE residual `h_t` plus an
embedding, and its collapsed output is compared by cosine against the trunk's own
collapsed state. Both feed the same `lm_head`, so alignment is the question.

## The result

    do-nothing baseline: cos(collapsed_t, collapsed_t+1) = +0.6284

    fusion            emb   target   signal   control   vs-baseline
    broadcast-all     t+1   t+1     +0.6058  +0.4369      -0.0226   <- best
    broadcast-all     t+0   t+1     +0.5506  +0.4243      -0.0778
    stream0-only      t+1   t+1     +0.5035  +0.3572      -0.1249
    stream0-only      t+0   t+1     +0.5032  +0.3582      -0.1252

**Reusing the trunk's existing hidden state predicts the next position BETTER
than running a 2.6B-parameter head over it.** Every one of the 12 combinations
scored negative against the baseline.

## Two things this DID pin

The module docs list two choices the tensor shapes do not constrain. Both are now
measured rather than assumed, and `Fusion` is an enum in `mtp.rs` instead of a
paragraph:

* `broadcast-all` beats `stream0-only` by ~0.10 — the "natural reading" was right.
* `emb=t+1, target=t+1` is the best index convention, matching the documented
  MTP convention (`h_t` + `emb(x_{t+1})` -> `x_{t+2}`).

So the free variables are resolved, and neither was the defect.

## ⚠️ The trap that nearly passed this

Raw cosine says the head looks GOOD: 0.7593 at `target=t+0`, against an
unrelated-position control of 0.4243. That is an artifact. `mtp::forward` ends
with `w.mixer.read(&wide[t])` — the SAME collapse the trunk applies — over a
`wide` built from the trunk's own `wide(t)`, so the output inherits the trunk's
residual. A high cosine there measures "the residual survived the layer".

An unrelated-position control does NOT catch this, because an unrelated position
does not share the residual. Only the do-nothing baseline does. Any future
measurement of this head must keep it.

## What to try next, cheapest first

1. **Are the weights trained at all?** Upstream sets
   `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and DISCARDS these tensors
   on load. A head that was never trained to convergence — or was reset before
   release — would behave exactly like this under every composition. Compare the
   MTP tensors' statistics (std, tail weight, outlier channels) against the
   equivalent trunk-layer tensors; fresh-init weights look visibly different.
   This is cheap and would explain the result completely.
2. **Context.** The probe runs the head over its own 15-token window with no
   trunk KV. If the head is meant to attend over the trunk's cache, it is being
   starved. Feeding it real KV is more work but testable.
3. **A trace.** Failing both, the composition cannot be settled by inspection —
   upstream drops the weights, so there is no reference forward. A trace from an
   implementation that does run it is the only ground truth.

## Producer/consumer seams found on the way

Nothing had ever consumed an MTP head end to end, and three seams were broken:

* `--include-prefix mtp.` built an EMPTY artifact and exited 0 (the MTP skip was
  keyed on the deepseek4 formats). Fixed `0e9feb7b3`.
* `QuantType::Oq8G128` (54) was missing from `from_code`, so the pager rejected
  its own artifacts. Fixed `4f5174f2c`.
* The quantizer SPLITS stacked MoE experts (`experts.<e>.<proj>.weight`) while
  the CPU forward wants the source's stacked `experts.gate_up_proj`. Any future
  wiring must restack; the probe does.
