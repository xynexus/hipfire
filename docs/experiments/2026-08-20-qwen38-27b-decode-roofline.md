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
