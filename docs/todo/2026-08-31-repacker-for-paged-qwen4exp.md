# TODO: one derived layout that serves BOTH qwen4_exp consumers

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
qt 37  Oq4G256ArchPacked  nibbles | scales | tail   plane-split
qt 52  OqPlusCompactG128  [f16][64 nib][overlay]    VARIABLE stride, G128
```

**qt 53 is the one that can serve both.** It is per-expert addressable and the
pager already fetches it verbatim (`module_requires_host_repack` deliberately
omits it). It is also structurally what a block-interleaved GEMV reads — the same
shape `gemv_oq_compact_grouped_auto(.., group, block_stride)` already consumes,
which `weight_gemv` gained an arm for on 2026-08-30. qt 53 is simply canonical
with f32 scales and no overlay.

## What is missing

1. **`weight_gemv` cannot read qt 53's layout.** `DType::Oq4G256` means the
   plane-split combined form, so a MoE-block tensor needs either its own dtype or
   a block-interleaved Oq4 arm alongside the compact one. This is the only piece
   standing between qt 53 and a zero-repack per-expert load.
2. **`down_proj` is qt 52 and the pager does not know it.** `module_resident_len`
   lists Oq4G256 / Oq4G256MoeBlocks / Oq8G256 / OqPlusCompact; qt 52 falls through
   to `_ => tensor.data_size`, which is the wrong resident length. Note qt 52's
   `block_bytes()` is `None` — the overlay count is per-artifact, so its stride
   must be derived from the buffer, not assumed. Either teach the pager that, or
   have the repacker convert `down_proj` to a fixed-stride form.
3. **The repacker only rewrites qt 34.** `--moe-blocks` converts routed OQ4 and
   copies everything else, so on this model it leaves 24576 `down_proj` tensors at
   qt 52 untouched — half the expert set, and the half neither consumer can page.

## Suggested shape of the work

- Give qt 53 a `weight_gemv` arm (or dtype), so ONE derived artifact loads verbatim
  per-expert and pages verbatim. That deletes the reason qt 37 exists for this
  family.
- Decide `down_proj`: either the pager learns compact-G128 (deriving stride from
  the buffer, as `weight_gemv`'s compact arm does), or the repacker emits a
  fixed-stride G128 block form for it.
- Only then consider retiring qt 37 for routed experts. It remains the right answer
  for DENSE tensors, which no one pages.

## Already done

`optimize` no longer needs RAM equal to the derived model — it streams (115 GiB
RSS and an OOM kill on the 170 GB artifact, versus 0.5 GiB and ~2 min after). Any
of the above can now actually be run on a model large enough to need it.
