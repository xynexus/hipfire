# MoE routed prefill: why it is at 8%, and the two fixes in order

halo/gfx1151, Qwen3.6-35B-A3B--oq4.25++ (E=256, k_top=8, shared=OqCompact,
routed=Oq8), kvarn, 2026-08-23.

## The measurement

| model | active | tok/s | TOPS |
|---|---|---|---|
| Qwen3.8-27B dense | 27B | 300 | **16.2** |
| Qwen3.6-35B-A3B MoE | 3B | 215 | **1.29** |

**8% of the dense path's arithmetic efficiency with 1/9th the active
parameters.** A kernel trace puts 47.8% of MoE prefill in two kernels, and their
grid explains it:

    const int krank = blockIdx.y;   // which of k_top
    const int bid   = blockIdx.z;   // which TOKEN
    const char* w = expert_ptrs[topk_indices[bid * K_TOP + krank]];

Each (token, expert) pair re-reads that expert's weights. At n=512 / k_top=8 /
E=256 an expert is picked by ~16 tokens, so its weights are read **~16x**.

## Fix 1 (the unlock): get resident experts onto the grouped path

The grouped path already exists — "path 2", SGLang-style scatter into per-expert
buckets + grouped WMMA + unscatter, default-on, with **+114% (gfx1100)** and
**+192% (gfx1201)** recorded in-code for MQ4 A3B.

It is not reachable for this model, and the reason is structural rather than
dtype-related:

    let bucketed_routed_experts = ffn.experts.is_empty()          // paged
                               || dtypes.routed_profile.is_mixed(); // mixed

Path 2 lives inside `if routed_expert_buckets.is_some()` (prefill_chunk.rs
838-1860). A **resident, uniform-dtype** model fails both conditions, so it falls
to section 6 at line 1861, which delegates to `MoeFamily::run_prefill` — a
different dispatcher whose routed arms are the per-token indexed GEMVs.

Note the path-2 CORE is bucket-independent: `moe_scatter_fused_k8` →
grouped GEMM → `moe_gate_up_unscatter_k8` are GPU-side and derive everything from
`topk_indices`. The CPU bucket download exists for PAGING. So this is a
dispatcher-integration problem, not a kernel one.

Already landed toward it: `gemm_oq8g256_moe_grouped_wmma` (the Oq4 sibling with a
260 B `[f32 scale | 256 int8]` block; `OQ8_BLK` confirms 260, and resident
experts point straight at interleaved blocks so `weight_byte_offset` is 0) plus
`DType::Oq8G256` arms on both path-2 dispatches. **Currently unreachable** for
exactly the reason above — which the `[features]` report now states in one line
instead of costing four kernel traces.

## Fix 2 (second-order): module-resident, sweeping its tiles

The grouped kernel is **token-tile-major with expert-tagged tiles**: grid is
(row_tiles, slot_tiles), `expert_tile_ids[blockIdx.y]` selects the expert, and a
weight tile is re-read for **each token tile**.

Weight-stationary inverts that: a block owns one weight tile and sweeps all of
that expert's token tiles, so the weight is read once.

**It is size-dependent, and that matters for sequencing.** At n=512 an expert
holds ~16 tokens ≈ one 16-row tile, so module-major and tile-major coincide and
the win is ~0. At n=2048 an expert holds ~64 tokens = 4 tiles, and the current
shape re-reads its weights 4x where weight-stationary reads them once. The gain
scales with tokens-per-expert = n·k_top/E.

**CORRECTION.** An earlier draft claimed a register wall: that holding a weight
tile while sweeping an expert's token tiles needs accumulators for every tile at
once, so a block could only take P≈4 tiles. That is wrong, and the error was loop
order. It is only true with K outermost. Weight-stationary puts TOKEN TILES
outermost and K innermost, so exactly ONE accumulator is live — the full K
reduction for (row_tile, token_tile) completes before moving to the next tile.

The real question is whether the weight ROW-TILE (16 rows x full K) fits in LDS,
and at this model's dimensions (dim=2048, moe_intermediate=512) it comfortably
does, as OQ8 260 B blocks:

| projection | m | k | 16-row tile | LDS headroom left |
|---|---|---|---|---|
| gate_up | 1024 | 2048 | **32.5 KB** | 31.5 KB |
| down | 2048 | 512 | **8.1 KB** | 55.9 KB |

So the weight is read **once per (expert, row-tile)** and reuse is bounded only by
tokens-per-expert, not by registers and not by a P parameter:

    tokens/expert T = n*k_top/E,  weight reads today = ceil(T/16),  module-resident = 1

    n=512    T=16    1 tile   -> 1x today, 1x resident  (no gain)
    n=1024   T=32    2 tiles  -> 2x today, 1x resident
    n=2048   T=64    4 tiles  -> 4x today, 1x resident
    n=4096   T=128   8 tiles  -> 8x today, 1x resident

The size-dependence stands — at the current ubatch of 512 it buys nothing — but
the ceiling is higher than the P-limited sketch implied, and it grows linearly
with batch. Note the headroom also leaves room to double-buffer the activation
tiles against the weight load, which is what keeps the sweep from stalling.

## Order

Fix 1 first, and not only because it is larger. Fix 2 is a kernel optimisation on
a path this model cannot currently execute — building it first would repeat the
mistake already made once here, where the Oq8 grouped kernel and its arms landed
correct, verified, and unreachable.
