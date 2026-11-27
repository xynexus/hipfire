# TODO: one derived layout that serves BOTH qwen4_exp consumers

> **Update 2026-11-27 — the page-in transform now runs ON THE DEVICE**
> (`kernels/src/pager_repack.hip`, `weight_pager::device_repack`). For
> `PerExpertNative` modules whose tensors are canonical `Oq4G256` and/or kept-
> compact `OqPlusCompact*`, the pager uploads the DISK bytes and kernels write
> the per-expert form in GPU memory — byte-identical to the host transform
> (`examples/pager_repack_parity`: 512 modules of the shipped 180B, 2.18 GB,
> 0 bytes differ; also a `tests/qwen4exp-gate.sh` stage). Anything else still
> takes the host path; `HIPFIRE_PAGER_CPU_REPACK=1` forces it for A/B.
>
> Measured, 180B oq4, 160 decode steps, halo gfx1151:
>
> | expert cache | host repack | device repack | cold load |
> |---|---|---|---|
> | 8 GiB (default) | 0.17 s/tok | **0.13 s/tok** (−24%) | 533 → 330 us |
> | 16 GiB | 0.14 s/tok | **0.12 s/tok** (−14%) | 569 → 382 us |
>
> Cold-load counts are identical between arms (same routing, same tokens).
> This makes the on-disk repack this note describes much less valuable: the
> host transform is gone without changing the format. What remains:
>
> * a resident module is **4.25 MB but costs ~6 MB** of the cache budget
>   (8 GiB holds 1365 modules) — the 2 MiB GTT granule; grouping modules per
>   allocation is the lever (see the grouped-allocation measurements);
> * the device form still carries BOTH the split planes (batched GEMM) and the
>   interleaved decode copy of `Oq4G256` gate_up (~2x its bytes); a
>   decode-only form would roughly double the experts a budget holds.

> **Update 2026-08-31 — this is an OPTIMISATION, not a prerequisite.** Paged
> qwen4_exp shipped without it: `hipfire-quantize` already emits an HFQM v2
> module table, so artifacts are pageable as written. What the repacker would buy
> is a cheaper page-in — pre-split compact planes so `prepare_expert_module` is a
> memcpy rather than a transform. See
> `2026-08-31-qwen4exp-paged-experts-via-weight-pager.md`.

**Status:** open (2026-08-31). Design note, written after `--arch-experts` was
prototyped and BACKED OUT for foreclosing paging.

## The constraint

qwen4_exp is about to grow paged expert support, so anything the repacker emits
has to satisfy two consumers at once:

* the **per-expert** path — `moe_forward` does CPU top-k and loops one expert at a
  time through `weight_gemv`, so it wants a `WeightTensor` it can upload verbatim;
* the **paged** path — `WeightPager` fetches a routed-expert module as a byte
  range and hands it to the indexed kernels, so it wants per-expert addressing and
  no CPU transform per page-in.

## Why the obvious answer is wrong

`Oq4G256ArchPacked` (qt 37) is the arch *combined* layout, and `oq4_arch_load`
returns it as `Cow::Borrowed` — the per-expert path uploads it with ZERO repack.
That is tempting and it is a dead end: the combined layout is plane-split with an
interleaved decode tail whose relative offset `hfqm_modules.v1` cannot express, so
`WeightPager::register_expert_module` refuses qt 37 outright. An `--arch-experts`
mode was written, measured to build, and reverted for exactly this reason.

## The layouts, laid side by side

```
qt 34  Oq4G256            [f16 scale][128 nibbles]  130 B/group   block-interleaved
qt 53  Oq4G256MoeBlocks   [f32 scale][128 nibbles]  132 B/group   block-interleaved
qt 37  Oq4G256ArchPacked  nibbles | scales | tail   plane-split   NOT pageable
qt 36  OqPlusCompact      [f16][128 nib][overlay]   G256, variable stride
qt 52  OqPlusCompactG128  [f16][64 nib][overlay]    G128, variable stride
```

## CORRECTION: the answer is compact split-plane, not qt 53

An earlier draft of this note named qt 53 as the one layout that could serve both
consumers. That was wrong, and the thing that shows it is the **iu4x2 kernel
family**.

`gemm_oq_compact_iu4x2_{wmma,tiled,w64}` take, verbatim from their own header:

    W : compact SPLIT-PLANE blocks (bulk nibbles fed to the core RAW)

They feed the nibbles to the matrix core with no dequantisation, doing W4A8 as two
`wmma_i32_16x16x16_iu4` passes over the SAME weight tile (`x = 16*x_hi + x_lo`).
The second pass adds WMMA issue and one more int4 activation plane and **nothing**
to weight traffic. So compact is not a storage compromise that kernels tolerate —
it is what the fastest W4 path wants to read.

And compact already has the COMPLETE kernel set, for both consumers:

```
decode          gemv_oq_compact_grouped{,_v2,_v3,_mw}
indexed decode  gemv_oq_compact_moe_indexed
prefill/batched gemm_oq_compact_iu4x2_{wmma,tiled,w64}
MoE batched     gemm_oq_compact_moe_grouped_{wmma,f32}
```

That removes the premise the qt 53 argument rested on. qt 53 would need a new
`weight_gemv` arm and gains nothing over a form that is already 4.25 b/w and
already has tuned kernels at both batch sizes.

## So the real defect is the pager's expansion

`weight_pager.rs` says it plainly, at `estimated_module_resident_bytes`:

> `module_tensor_resident_len` answers for the PAGER, which repacks OqPlusCompact
> to Oq8 blocks. The resident loader no longer does: it keeps the compact planes,
> and `split_compact_planes` preserves the byte count exactly. Asking the pager's
> question here would over-estimate a compact artifact by 1.80x and refuse a load
> that now fits — which is precisely the 122B.

The RESIDENT loader was modernised to stop expanding. The PAGER was not:
`module_resident_len` still maps `OqPlusCompact -> oq8_moe_packed_len`, 4.25 b/w
becoming 8.125 (1.80x), and `module_requires_host_repack` still lists it. That
expansion is what OOM-killed a 170 GB load here at `free=7085.9 MiB`, and it would
double every paged `down_proj`.

**Neither path recognises G128.** `module_resident_len` does not list qt 52 at all
(it falls through to `tensor.data_size`), and the compact-resident predicate in
`estimated_module_resident_bytes` accepts only qt 36 and qt 35 — so qwen4_exp,
whose `down_proj` is qt 52, is excluded from the non-expanding path on both counts
and mis-priced by admission.

## The work, in dependency order

1. **Pager stops expanding compact.** Page `OqPlusCompact` verbatim as split
   planes (`split_compact_planes` is byte-count-preserving, so the resident length
   is the disk length) and drop it from `module_requires_host_repack`. This is the
   change the resident loader already made.
2. **Teach both paths G128 (qt 52).** `module_resident_len` and the compact
   predicate. The stride is per-artifact — the overlay count varies, which is why
   `block_bytes()` is `None` — so derive it from the buffer, as `weight_gemv`'s
   compact arm does.
3. **Repacker pre-splits the planes**, so page-in is a verbatim fetch rather than
   a per-page-in CPU transform — the same idea as qt 34 -> 53, applied to the
   layout that actually has the kernels. Today `--moe-blocks` rewrites only qt 34,
   leaving all 24576 `down_proj` tensors untouched.
4. Only then revisit whether qt 53 and qt 37 need to exist for routed experts at
   all. qt 37 remains right for DENSE tensors, which nobody pages.

## Already done

`optimize` no longer needs RAM equal to the derived model — it streams (115 GiB
RSS and an OOM kill on the 170 GB artifact, versus 0.5 GiB and ~2 min after). Any
of the above can now actually be run on a model large enough to need it.
