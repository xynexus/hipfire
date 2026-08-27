# Spec-decode output is not byte-identical to plain AR decode

**Model:** Qwen3.8-27B--oq4.25++.hfq + `dflash2.oq4+` draft, halo (gfx1151),
greedy, `dflash_spec_demo`.

Under greedy decode, speculative decoding is supposed to be *lossless*: a draft
token is committed only when it equals the target's own argmax, so the emitted
text must match plain autoregressive decode exactly. On this model it does not.

## What was measured

`--ar-baseline` reproduces byte-for-byte across runs, so the reference is
deterministic. Against it, at `--max 256` on a Python prompt:

| config | vs AR |
|---|---|
| `--no-speculate` | IDENTICAL |
| `--no-adaptive-b --block-size 6` | DIFFERS |
| `--no-adaptive-b --block-size 8` | DIFFERS |
| adaptive (default) | DIFFERS |

The divergence point is **B-dependent** — char 697 at B=6, char 401 at B=8 —
so it tracks where cycle boundaries fall, not a fixed context length. This is
why a short run can look clean: at `--max 128` with B=6 the run ends before the
first flip.

## Two hypotheses, both refuted by measurement

**1. "A KVarN block sealed during verify keeps rejected tokens."** KVarN
quantizes K in 128-token blocks with a joint Sinkhorn variance normalization, so
a block sealed mid-speculation could in principle bake in tokens that are later
rejected — and the spec path never re-flushes (`kvarn` appears 3 times in
`speculative.rs`, all construction). The position evidence fit: same prompt and
B=8, `--max 80` (never reaches position 128) is IDENTICAL, `--max 200` DIFFERS.

Refuted: `--kv-mode q8` also diverges, and Q8 has no block tiling, no Sinkhorn
and no records at all. The 128-boundary correlation was coincidence — enough
tokens had simply accumulated to flip an argmax.

**2. "The batched attention takes q8 KV scales per-tile."** This is the
explanation `is_batchable_la`'s own comment offers, hedged as "most likely".

Refuted by reading the kernel: `kv_cache_write_q8_0_batched` derives its scale
from `positions[bid]` over a 32-element block within one head of one token —
exactly the granularity the per-token write uses. Scale granularity is identical
on both paths.

## Actual root cause

**Verify runs the batched forward; AR decode runs the per-token forward; the two
are not numerically equivalent.** This is documented and *deliberately accepted*
in `qwen35/mod.rs` at `is_batchable_la`:

> CAVEAT, deliberately accepted: the batched path is not numerically identical
> to per-token. Typical |delta logit| is ~6e-2 (max 2.4e-1) against ~4e-6 for
> pure reordering, and only 15% of positions keep the same top-256 set. […]
> Anything that needs the two to agree bit-for-bit must pin the path explicitly.

Speculative decoding is precisely something that needs them to agree bit-for-bit.

⚠️ **`--kv-mode f32` looks IDENTICAL, and that result is an artifact.** f32 KV
does not satisfy the batched-verify predicate, so verify silently falls back to
per-token and is trivially equal to AR. The timings give it away: under f32,
spec decode runs **6.79 tok/s against AR's 15.56** — 2.3x *slower*, at tau 3.9.
Under kvarn8 it is 21.35 vs 12.40, genuinely batched, and it diverges. Do not
read the f32 row as evidence that quantized KV causes this.

This also corrects the older note that DFlash "diverges at every KV tier
including the f32 oracle, therefore a verify forward bug" — the f32 tier was
never running the verify path being blamed.

## Where the difference enters

`compare_prefill_hidden_paths --model <27B> --n 512 --kv-mode q8` runs both
forwards in one process against the same ring buffer:

    layer   worst|rel|   at row
        0     1.29e-3      254
        3     5.87e-3       13
       21     8.14e-3      421

    against the fp32-KV reference: batched 4.760e-2, per-token 1.254e-2

Layer 0 already differs, so this is not accumulated drift. The worst layer-0 row
is **254 — the last row of the 256-token chunk**, the row with the most in-chunk
neighbours, and the delta then compounds up the stack. That is the signature of
in-chunk K/V being read one way by the batched path and another by per-token:
within a chunk the batched path attends over its neighbours directly, where the
per-token path reads those same neighbours back through the quantized cache.

The MoE grouped-vs-indexed gate in the same tool reports `worst |rel| 0.000e0`,
so the FFN is not involved.

## What a fix requires

Numerical parity between the batched and per-token forwards, or an explicit
pin so verify and AR provably take the same path. Not a one-line change, and not
a KV-tier change: both KV hypotheses above are dead, layer 0 is already
divergent, and the accumulators are f32 on both paths.

Note the scope is wider than spec decode — the same caveat means batched prefill
and per-token prefill disagree for *every* dtype, bf16 included.

## Not affected

Draft quality: tau is measured against the verifier's own argmax, so it remains
a valid drafter metric. The +71% tau from driving the draft at its trained block,
and the adaptive-B controller fix, are independent of this and stand.
