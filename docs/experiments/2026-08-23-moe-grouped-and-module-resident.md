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

Sketch, with the register constraint that shapes it: holding A for all of an
expert's token tiles needs accumulators for every tile at once (16 rows x T
tokens), which does not fit for large T. So a block owns one output row tile and
**P token tiles**, keeps the weight K-slice in LDS across the inner loop over
those P, and carries 16x16xP/64 accumulators per lane — P=4 is 16 per lane in
wave64, comfortable. That gives P-fold weight reuse rather than unbounded, and
degrades to today's kernel at P=1.

## Order

Fix 1 first, and not only because it is larger. Fix 2 is a kernel optimisation on
a path this model cannot currently execute — building it first would repeat the
mistake already made once here, where the Oq8 grouped kernel and its arms landed
correct, verified, and unreachable.
