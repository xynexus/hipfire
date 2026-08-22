# The unfused MoE OOM is payload, not allocator overhead

Status: measured 2026-08-22, nix1 / gfx1103. Corrects `2026-08-22-m4-decided.md`
and the premise of `moe-expert-residency-unification.md`.

## What was claimed

`2026-08-22-m4-decided.md` (§M4) measured that `HIPFIRE_QWEN35_MOE_OQ_INDEXED=0`
cannot load `Qwen3.6-35B-A3B--oq4` at all — it dies at layer 25 of 40 with
11.56 GiB of payload placed and 15.9 MiB free of 43 008 MiB — and deliberately
declined to name the mechanism, saying only that allocator overhead per buffer
object and a larger unpacked layout would both present that way.

`moe-expert-residency-unification.md` (§M5) leans the other way: its "Why"
section is built on **buffer-object count**, citing 20 480 BOs and an upstream
measurement of 4.35 GB of pure allocator overhead at that count.

## What was measured

A census on `Gpu::upload_raw` — the bare, unpooled `hipMalloc` every per-tensor
load goes through — counting calls and bytes
(`HIPFIRE_UPLOAD_RAW_CENSUS=1`, off by default):

| | `upload_raw` calls | bytes uploaded | outcome |
|---|---|---|---|
| fused (default) | **20 480** | 17.09 GiB | loads; 17.77 GiB placed |
| unfused (`=0`) | 12 288 (and counting) | **19.56 GiB** | OOM |

The unfused path makes **fewer** allocations and moves **more** bytes. At 12 288
calls — roughly 60% of the way — it had already uploaded more than the fused
path's entire 17.09 GiB.

Per tensor: **0.834 MiB fused vs 1.63 MiB unfused, ≈1.96×.**

## Exactly why: the arch layout stores every expert twice

The ratio is not approximately 2×, it is 2.000× by construction.
`oq4_arch_combined_len` (`hipfire-runtime/src/oq4_arch.rs:34`) is

```
m*(k/2)  +  m*ng*4  +  m*ng*132          ng = k/256
```

and its own doc names the three regions:

- `[split nibbles m*(k/2)]` — prefill MMQ/f16, at `sub_offset 0`
- `[split f32 scales m*ng]` — prefill weight-scale region
- `[interleaved m*ng*132]` — decode GEMVs, `[f32 scale][128 nibbles]` per group

The **interleaved region alone is the compact MoE-block layout** the indexed path
uses. So the arch layout is the compact one *plus* a second, differently-shaped
copy of the same weights for prefill:

| m | k | arch combined | MoE blocks | ratio |
|---|---|---|---|---|
| 2048 | 4096 | 8.25 MiB | 4.12 MiB | **2.000×** |
| 4096 | 2048 | 8.25 MiB | 4.12 MiB | **2.000×** |
| 1024 | 2048 | 2.06 MiB | 1.03 MiB | **2.000×** |

Measured was 1.96× (1.63 vs 0.834 MiB/tensor); the shortfall is the non-expert
weights, which are in both modes and use neither layout.

**And the indexed path proves the second copy is not necessary.** It serves both
prefill and decode from the interleaved form alone — batched
`moe_topk_renorm_k8_batched` plus the indexed kernels for prefill, indexed GEMVs
for decode. The dual-region layout is the general path that predates them, not a
requirement.

## What that settles

**It is payload expansion, not allocator overhead.** The buffer-object count is
the same order in both modes, and the mode with *more* buffer objects is the one
that loads. What differs is what each buffer holds: `load_moe_expert`'s own
comment says the non-indexed arm falls through to `load_weight_tensor`, which
produces the dense `oq4_arch_load` / `oq8_combined` layouts, while the indexed
arm repacks into the compact MoE-block layout (132 B / 260 B). The dense form is
about twice the size, and 2 × 17.8 GiB does not fit in 43 GiB.

Two consequences worth stating plainly:

1. **§M4's hedge was the right call and is now resolved** — against the reading
   the M5 plan would have suggested.
2. **Consolidating allocations would NOT fix this.** §M5 Phase 2's treatment
   (one owning allocation per expert, tensors aliasing in) is real and worth
   having — it halves buffer objects for lfm2moe — but it moves the same bytes.
   The qwen35 unfused OOM is a *format* problem: the non-indexed path needs a
   compact routed-expert layout, or it needs the indexed kernels. That is a
   quant/kernel-contract question, not a residency one, and it does not belong
   to M5.

## What it does not say

The upstream 4.35 GB figure the M5 plan cites is not refuted — it was measured
on a different box, a different artifact, and a 7900 XTX with discrete VRAM
rather than UMA. Allocator overhead at 20 480 BOs may well be real there. What
is established here is only that **on this box, for this artifact, it is not the
term that decides the OOM** — and therefore not the thing to fix first.
