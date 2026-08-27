# TODO: cover `Oq4G256` routed experts in tiny-prefill-gate

**Status:** OPEN. Blocks deleting `HIPFIRE_MOE_OQ4_UNIFORM_PATH1`, the opt-in
added by #368/#369 because path-1 parity for uniform Oq4 experts is unverified.

## First: a correction

While scoping this I claimed, in a bug doc, a merged PR body and a code comment,
that:

> *"tiny-prefill-gate SKIPS qwen3_5_moe, so batched MoE prefill has NO parity
> coverage at all."*

**That is wrong twice over**, and the gate says so in its own header:

- `qwen3_5_moe` SKIPs **deliberately**. Its preset is top-2-of-8 while
  `moe_prefill_topk_shape_supported` requires `k_top ∈ {8,10}`, so the refusal is
  `moe_topk_ok=false (K=2, E=8)` — *a fixture shape with no arch term in it*. It
  is "deliberately kept that way as the regression cover for the ADMISSION
  guard".
- **`qwen3_5_moe_indexed` is top-8-of-16, reaches grouped path-2, and PASSES.**
  Observed: `max_kld=0.00000541 argmax=4/4`, with the corrupt-prefix self-check
  firing.

So batched MoE prefill **is** covered. I read a SKIP as a hole without reading
the comment directly above the code that produces it.

## What is actually uncovered

Every gate cell quantizes with `--format fp16` (`tiny-prefill-gate.sh:159`), so
the covered MoE cell exercises **F16 experts through path 2**. Nothing exercises
**`Oq4G256` routed experts**, which is the path #368 had to gate behind a flag.

## Why an `oq4` cell cannot just be added

Measured 2026-08-27 on the `qwen3_5_moe_indexed` fixture, with
`HIPFIRE_MOE_OQ4_UNIFORM_PATH1=1` set:

| format | quantize | probe |
|---|---|---|
| `fp16` | OK | **rc=0**, `max_kld 0.00000291` |
| `oq4` | OK | rc=3 — *batched prefill did not execute* |
| `oq4.25` | OK | rc=3 — *batched prefill did not execute* |
| `oq8` | OK | rc=3 — *batched prefill did not execute* |

The decline is **not** in the MoE admission path. `is_batchable_la` — applied to
the *attention* projections (`wqkv`, `wz`, `w_beta`, `w_alpha`, `wo`, …) —
accepts only:

```
MQ4G256 HFQ4G256 MQ6G256 HFQ6G256 Q8_0 ParoQ4G128 F32 F16
```

No Opus dtype is in that set. An all-Opus artifact therefore declines batching at
the attention layer, before the MoE FFN is considered at all.

Note the real `Qwen3.6-35B-A3B--oq4.hfq` **does** batch — that is how the panic
was found — so production oq4 artifacts evidently pair a batchable attention
dtype with `Oq4G256` experts. That mixed shape is what a fixture has to
reproduce.

## What the work actually is

Not a gate change — a **fixture** change. Options, roughly in order of effort:

1. **Emit a mixed-precision tiny fixture**: batchable attention (e.g. `Q8_0` or
   `MQ4G256`) plus `Oq4G256` routed experts. Closest to what real artifacts look
   like, and directly exercises the path the flag guards. Needs the quantizer to
   accept per-tensor-class formats for `--emit-fixture` output, which it does not
   today.
2. **Widen `is_batchable_la`** to accept Opus attention dtypes, if the batched
   attention arms in fact support them. That is a real behaviour change with its
   own parity question, and should not be done just to make a gate cell run.
3. **Gate against a real artifact** rather than a tiny fixture — slow, needs a
   35B on the box, and is the sort of thing tiny-prefill-gate exists to avoid.

(1) is the right one. (2) is worth *investigating* separately, because if the
attention arms do handle Opus dtypes then real oq4 artifacts are being pushed to
the per-token path unnecessarily — which would be a significant prefill
regression hiding in plain sight.

## Do not add a cell that always SKIPs

The gate's own philosophy rules this out: a SKIP that "looks configured but gates
nothing" is precisely what PR #370 called out on gfx1151, and what the
path-check in this gate exists to prevent. A cell must be able to fail.

## Exit

`HIPFIRE_MOE_OQ4_UNIFORM_PATH1` can be deleted when a cell exercises uniform
`Oq4G256` routed experts through path 1 and passes the KLD comparison against the
per-token reference — with the corrupt-prefix self-check firing, as every other
cell requires.
