# Qwen3.8-27B oq4.25++ — what plain decode can reach on this box

**Measured 2026-08-20**, gfx1151 after the memory BIOS change (peak 256 GB/s,
pure-read achievable 250.3). No spec-decode: plain AR.

## The correction that matters

`oq4.25++` is 4.25 bits/weight **on disk**. It is not what sits in memory.
`oq8_arch_load` (`crates/hipfire-runtime/src/oq8_arch.rs:301`) **expands
OqPlusCompact to Oq8G256 — one int8 per weight — at load**, by default; compact
residency is opt-in behind `HIPFIRE_OQ_COMPACT_RESIDENT` "while the end-to-end
path is validated".

Confirmed by watching RSS during a load: **2 -> 30 GiB**, not the 15.5 GB the
file suggests.

So the bytes a decode step actually streams are:

| | expanded (today) | compact |
|---|---|---|
| body + lm_head, 25.6 B params | 25.6 GB int8 | 13.6 GB @ 4.25 bit |
| per-group f32 scales | 0.40 GB | (in-block) |
| DeltaNet state, 48 layers, R+W | 0.20 GB | 0.20 GB |
| KV (q8, 16/64 layers, short ctx) | ~0.01 GB | ~0.01 GB |
| `embed_tokens` — **row lookup, not streamed** | — | — |
| **total per token** | **~26.2 GB** | **~13.8 GB** |

Counting the embedding (1.84 GB packed) would be wrong: decode touches one 10 KB
row of it.

## The roofline

    achievable 250.3 GB/s / 26.2 GB = 9.5 tok/s   <- today
    achievable 250.3 GB/s / 13.8 GB = 18.1 tok/s  <- compact residency

**Measured today: 8.00 tok/s = 84 % of the 9.5 roofline.** Plain decode is
already close to the memory wall *for the residency it is using*. There is no 2x
hiding in the kernels — the earlier estimate of an 18.1 ceiling was wrong because
it used the on-disk size.

**A >16 tok/s target is unreachable at int8 residency**: it needs
16 x 26.2 = **419 GB/s**, against a 256 GB/s hardware peak. The only way to the
target is to move fewer bytes.

## Compact residency: the lever, and the one thing blocking it

| config | tok/s | text |
|---|---|---|
| expanded int8 (default) | 8.00 | correct |
| **compact, all tensors** | 9.10 | **BROKEN — 1 token, empty** |
| compact except `down_proj` | **9.10** | **correct** |

Bisected with the `HIPFIRE_OQ_COMPACT_RESIDENT_ONLY_K` / `_ONLY_M` hooks the code
ships for exactly this, in three runs:

* `ONLY_M=248320` (lm_head only) -> correct
* `ONLY_M=17408` (gate/up only) -> correct
* `ONLY_K=17408` (**down_proj only**) -> 1 token, empty

**`down_proj` is the sole broken class.** It is not the block layout: `down_proj`
`[5120, 17408]` and `gate_proj` `[17408, 5120]` are both 47.35 MB over 348,160
blocks, a uniform 136 B stride (N_out=3), so the dispatcher's
`block_stride = byte_size / (m*ng)` derivation is exact for both. The difference
is orientation — everything that works has K in {5120, 6144}; the broken one is
the only projection whose K is the FFN intermediate. Root cause NOT established.

## Why the win is only +14 %, not +60 %

Compact-except-`down` moves ~16.4 GB/token against the expanded 26.2, a 1.6x
reduction in bytes, and delivers only 8.00 -> 9.10 tok/s (+13.8 %). Its roofline
is 15.3 tok/s, so it runs at **59 % of roofline where the expanded path runs at
84 %**. The compact GEMV pays for the bytes it saves in nibble decode and sparse
overlay work.

That sets the real requirement for 16 tok/s. It needs BOTH:

1. **`down_proj` fixed**, so full compact residency is usable — 13.8 GB/token,
   roofline 18.1; and
2. **the compact GEMV brought from ~59 % to ~88 % of roofline** — about 1.5x on
   `gemv_oq_compact_grouped_v2`.

Either alone is not enough: full compact at today's 59 % efficiency lands near
10.7 tok/s.

## Where decode's time goes (rocprofv3, expanded path, 64 tokens)

    fused_gate_up_oq8_gemv    5120 calls  3563 ms  43.3%
    gemv_oq8_grouped_v2       9281 calls  2745 ms  33.4%
    fused_qkvza_oq8_gemv      3072 calls  1162 ms  14.1%
    gated_delta_net_f32       3120 calls   158 ms   1.9%

The two fused GEMVs are 57 % of decode and are **coalesced** — `boff = g*group +
lane*8`, so 32 lanes cover 256 contiguous bytes. They do NOT have the scattered
per-lane row read that cost the batched GEMM 7.3x. What they do have is the **v1
narrow-load pattern**: two 4-byte `int32` weight loads and eight scalar f32
activation loads per lane per group. `gemv_oq8_grouped_v2`'s own header says it
replaced exactly that with 16-byte `dwordx4` weight loads and `float4` activation
loads for "4x fewer memory instructions". Porting v2's load strategy into
`fused_gate_up_oq8_gemv` and `fused_qkvza_oq8_gemv` is the obvious next
experiment on the expanded path, and the same lever likely applies to the compact
GEMV's 59 %.

## Root-causing `down_proj` — the kernel is innocent

`parity_gemv_oq_compact` (new; the decode GEMV had **no** parity coverage at all —
`parity_gemm_oq_compact` covers the batched GEMM prefill uses) runs
`gemv_oq_compact_grouped_auto` against an f32 reference over the same expansion
`oq8_arch_load` performs, on the real Qwen3.8-27B projection classes:

    gate/up   [17408, 5120]  ng=20   rel=4.37e-7  PASS
    qkv       [10240, 5120]  ng=20   rel=4.37e-7  PASS
    lm_head  [248320, 5120]  ng=20   rel=4.37e-7  PASS
    attn_out  [5120,  6144]  ng=24   rel=4.34e-7  PASS
    down      [5120, 17408]  ng=68   rel=4.42e-7  PASS
    down x2   [5120, 34816]  ng=136  rel=4.39e-7  PASS

**The compact GEMV is correct at `down_proj`'s exact geometry**, and at double its
K. So the fault is in how `down_proj` is DRIVEN, not in the kernel.

Narrowing from there, all confirmed by reading the dispatch:

* rotation is not it — `dtype_post_rotation_variant` puts **both** `Oq8G256` and
  `OqCompactG256` in `GemvVariant::Prerotated`, identical treatment;
* the block layout is not it — `down_proj` and `gate_proj` are both 47.35 MB over
  348,160 blocks at a uniform 136 B stride, so the dispatcher's
  `block_stride = byte_size / (m*ng)` is exact for both;
* the auto-selector is not it — `ng` is even for both (68 and 20), so both take
  the v2 kernel.

What IS specific to `down_proj`: it is the only projection reached through
`weight_gemv_swiglu_residual` (`weights.rs:1285`, called from the live lowered
decode path at `lowered.rs:308`). `dense_swiglu_residual_route` classifies only
`MQ6G256`; every Opus dtype falls to `Unclassified` and shares a generic
fallback, and there is **no `GemvOqCompact*Residual` kernel key** where MQ4, MQ3,
HFQ4, HFQ6, Qtip and the Lloyd variants all have one.

**Next step: instrument `weight_gemv_swiglu_residual` for `OqCompactG256` vs
`Oq8G256` on the same input.** The failure mode to look for first is a double
rotation — `weight_gemv`'s own tail warns that routing a prerotated dtype through
`run_auto` "would re-rotate ... double-applying the involutory FWHT and feeding
effectively-unrotated activations to the prerotated kernel (garbage logits)",
which matches the symptom exactly: finite, wrong, and only on the one projection
whose activation the caller has already rotated.

## Double rotation: REFUTED. And both compact kernels are innocent.

`HIPFIRE_DOWN_TRACE=1` instruments `weight_gemv_swiglu_residual`'s `_` arm — the
one `down_proj` takes — and prints what the GEMV is actually handed. Same prompt,
same seed, first three decode calls:

    expanded  dtype=Oq8G256       m=5120 k=17408 awq_scale=true mq_x_rot_elems=32768
    compact   dtype=OqCompactG256 m=5120 k=17408 awq_scale=true mq_x_rot_elems=32768

So, against the hypothesis:

* **the AWQ sidecar IS attached** in both — `supports_awq_sidecar` already lists
  `OqCompactG256`/`G128`, and its comment records that omitting them was a
  previous incarnation of exactly this bug;
* **the shared rotation scratch is big enough** — 32768 elems against K=17408, no
  overrun;
* **both dtypes take the SAME code path** — neither `weight_gemv_residual` nor
  `weight_gemv_swiglu_residual` has an Opus arm, so both fall to the identical
  `_` fallback. There is no second rotation to double-apply.

**Double rotation is refuted.** And both compact kernels are now proven correct at
`down_proj`'s geometry:

* `parity_gemv_oq_compact` (new): the decode GEMV, ~4.4e-7 rel at
  `[5120, 17408]` and at double the K;
* `parity_gemm_oq_compact` (extended): the batched GEMM, **bit-identical** at
  `[512, 17408]` for B=1 and B=9 across N_out 1/3/7/16 and both group sizes. Its
  previous shapes topped out at **K=3584**, so the entire large-K regime was
  untested — down_proj is K=17408.

Also eliminated by reading: `oqplus_compact_to_oq8_combined` performs exactly the
expansion the parity reference does (sign-extended nibbles + sparse overlay, no
AWQ folding, same `block_bytes = len / n_groups` derivation, which is an exact
136 for both `down_proj` and `gate_proj`); and `weight_gemm` has no compact arm
but its fallback is a per-token `weight_gemv` loop, i.e. correct-but-slow.

## Still open

The trace shows `down_proj`'s INPUT already differs at the first decode call —
absmax 4.62e-1 expanded against 3.58e0 compact — and `gate`/`up` are expanded in
BOTH runs under `ONLY_K=17408`. So the divergence is inherited from prefill,
which ran 64 layers with a compact `down_proj` before decode began. That is
consistent, but it does NOT localize the fault, because the batched kernel
prefill uses is bit-identical at this shape.

What that leaves: something in the prefill/batched CALL SITE that differs by
dtype without either kernel being wrong — the activation-rotation or
activation-quantize stage feeding the compact path, or a `block_stride` /
routing decision made somewhere other than the two dispatchers already checked.
Next probe should dump per-layer hidden state during PREFILL (not decode) for the
two dtypes and find the first layer that diverges; the `--gemm-pattern`-style
isolation is exhausted, since both kernels pass.
