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

(1) is the right one. **(2) was investigated 2026-08-27 — see below. The feared
regression does not exist, and the exclusion is not the sole barrier.**

## The `is_batchable_la` Opus question — investigated

**Q: is excluding Opus dtypes there an oversight that costs real oq4 artifacts
the batched path?**

**A: no measurable regression, and the stated justification is stale — but
lifting the exclusion is not sufficient either.**

Three findings:

**1. The justification in the code is out of date.** `prefill_chunk.rs` argues
Opus must be excluded because "Opus (Oq4/Oq8/OqCompact) weights are FWHT(+AWQ)-
rotated OFFLINE, so the activation must be rotated to match. Leaving them out of
this predicate sends an UNROTATED x into an Opus GEMM — the dense path records
that outcome as 'garbage: PPL 3.5e6'." That describes the **old hand-rolled LA
body**, which the very same comment says was replaced ("This used to be ~790
lines of hand-rolled dtype dispatch"). The shared lowered super-ops that replaced
it **do** carry Opus arms and rotation machinery:

| dtype | arms in the lowered LA matcher |
|---|---|
| `Oq4G256` | 18 |
| `Oq8G256` | 18 |
| `OqCompactG256` | 17 |

plus 31 `FWHT` / 14 `fwht` / 5 `rotate_x` references in that file.

**2. Lifting the exclusion does not make an all-Opus fixture batch.** Measured
behind a temporary flag that made `is_batchable_la` accept all three Opus dtypes:
`oq4` and `oq8` both still returned **rc=3, "batched prefill did not execute"**,
with `[pbs-gate] verdict=false` while every *named* term printed true. The
remaining decline is inside `moe_ffn_batched_admissible` on a path that records
no fallback line. So `is_batchable_la` is **not the sole barrier**, and the flag
was reverted rather than shipped — it demonstrated nothing.

**3. Production oq4 artifacts are unaffected anyway.** They do not have Opus
attention. The engine quantizer "keeps q/k/v/o at Q8 alongside the Q8 router +
shared_expert_gate", and the tiny fixture's own dtype dump shows the same mixed
shape a real artifact has:

```
router=BF16 shared_gate=Oq4G256 shared_up=Oq4G256 shared_down=Q8_0
expert_gate_up=Oq4G256 expert_down=Oq4G256 gu_uniform=true down_uniform=true
```

Q8_0 **is** in `is_batchable_la`'s accepted set, which is why the real 35B
batches (and why it reached the panic). **No prefill regression is hiding here.**

### Are OqCompact and Oq8 supported?

They differ by path, and the distinction matters:

| dtype | lowered LA | MoE grouped path-2 | notes |
|---|---|---|---|
| `Oq8G256` | ✅ 18 arms | ✅ real arms (`gemm_oq8g256_moe_grouped_wmma`) | declared legitimately |
| `OqCompactG256` | ✅ 17 arms | ✅ arms (`gemm_oq_compact_moe_grouped_f32`) | **opt-in**: path 2 is BIT-EXACT vs the decode GEMV and 1.4–3.3× faster, but the WMMA sibling was unverified |
| `Oq4G256` | ✅ 18 arms | ❌ **none** | the gap that caused the panic (#368) |

So of the three, only `Oq4G256` lacked a path-2 arm — which is exactly why it,
and not its siblings, hit `other => panic!`.

## Do not add a cell that always SKIPs

The gate's own philosophy rules this out: a SKIP that "looks configured but gates
nothing" is precisely what PR #370 called out on gfx1151, and what the
path-check in this gate exists to prevent. A cell must be able to fail.

## Exit

`HIPFIRE_MOE_OQ4_UNIFORM_PATH1` can be deleted when a cell exercises uniform
`Oq4G256` routed experts through path 1 and passes the KLD comparison against the
per-token reference — with the corrupt-prefix self-check firing, as every other
cell requires.


---

# ⚠️ BLOCKER: removing Q8 from the models breaks batched prefill

**Raised 2026-08-27 after a challenge that "there should be no Q8 tensors left in
the models". There are — and they are load-bearing.**

## The census

`hfq list` on a fixture quantized minutes earlier with the current binary
(`--format oq4`, arch `qwen3_5_moe_indexed`):

| qt | dtype | count |
|---|---|---|
| 34 | Oq4G256 | 68 |
| 50 | Bf16Huff | 13 |
| **3** | **Q8_0** | **11** |
| 16 | BF16 | 2 |
| 49 | Bf16Lut3 | 1 |

Real `Qwen3.6-35B-A3B--oq4.hfq`: **271 Q8_0 tensors** of 21,094.

The Q8 tensors are not incidental — they are the attention projection set:

```
layer.N.linear_attn.in_proj_qkv / in_proj_a / in_proj_b / in_proj_z / out_proj
layer.N.self_attn.k_proj / o_proj
layer.N.mlp.shared_expert.down_proj          (fixture)
layer.N.mlp.shared_expert_gate               (35B)
lm_head.weight                               (35B)
```

## Why they are load-bearing

`is_batchable_la` — which gates whether a layer may use batched prefill at all —
accepts `MQ4G256 HFQ4G256 MQ6G256 HFQ6G256 Q8_0 ParoQ4G128 F32 F16`. **No Opus
dtype is in that set.**

So `Q8_0` attention is the *only* reason an oq4 MoE artifact reaches batched
prefill. Replace it with Opus and every such artifact silently drops to the
per-token path — the regime measured at **4.2 tok/s** in
`2026-08-27-induction-quantum-wcet-nix1.md`, versus batched.

This is not hypothetical. An all-Opus tiny fixture does exactly that today:
`rc=3, "batched prefill did not execute"` for `oq4`, `oq4.25` and `oq8`.

## The ordering constraint

**`is_batchable_la` must gain working Opus support BEFORE Q8 attention is
removed**, or the removal takes batched prefill with it — silently, since a
decline is not an error.

Two things make that harder than adding dtypes to a `matches!`:

1. The exclusion's stated justification is stale (see above), but the
   `moe_prefill_quant_family` ladder's own comment says the rotation exists:
   Opus routed experts are admitted on RDNA because "both attention arms rotate
   for it (the `is_mq`/`qkv_is_mq` predicates)". Those two statements contradict
   each other and one of them is out of date.
2. Widening the predicate is **not sufficient** on its own — measured. See the
   next section.

## The silent decline — partially chased

With `is_batchable_la` temporarily accepting all three Opus dtypes, an all-oq4
fixture still declined. Narrowed to `ffn_admissible=false`, i.e.
`moe_ffn_batched_admissible`. Not resolved further, and here is the honest state:

- **every decline path in that function does call `decline()`**, so nothing is
  silent in the code;
- but `decline()` → `record_fallback()` only fills a map. **`kernel_trace::report()`
  is what prints it, and the tree's only caller is `hipfire-arch-llama`
  (`arch.rs:180`).** qwen35 never calls it, which is why the 4B (llama family)
  printed a `SLOW PATHS TAKEN` summary and every qwen35 MoE run printed nothing;
- adding a `report()` call at the end of `forward_prefill_batch_with_pbs_opts`
  does **not** fix it — when the gate declines, that function is never entered.
  Adding it to the `if !eligible` branch at `prefill_batch.rs:6421` did not fire
  either, so the tiny fixture's decline is taken on a path upstream of both.
  That attempt was reverted rather than shipped: instrumentation that cannot be
  shown to fire is worth nothing.

**Next step for whoever picks this up:** find where the qwen35 forward decides
against batching for this fixture (upstream of `prefill_batch.rs:6421`) and flush
the trace there. The `[pbs-gate]` diagnostic recomputes proxy terms for display
and prints `per_layer_dtypes~=true` while the real verdict is false — so it
actively misleads. Its own comment demands "Name every term"; it does not.
