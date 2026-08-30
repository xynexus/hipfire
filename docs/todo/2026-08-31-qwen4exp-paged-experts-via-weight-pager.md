# TODO: page qwen4_exp's routed experts through `weight_pager` (not a second cache)

**Status:** open (2026-08-31). A bespoke lazy-expert cache was written, measured,
and then REVERTED because it duplicated residency and eviction logic that
`hipfire_runtime::weight_pager` already owns.

⚠️ History note for anyone bisecting: the feature landed in `02efe522b` and the
revert was swept into `2369b4ef5` ("fix(optimize): stream the derived package"),
whose message says nothing about removing it — staged changes from the revert were
picked up by an unrelated commit. `git log -S LazyExperts` finds both. The measurements
are kept below because they are the reason to do this properly.

## What the throwaway version proved

On the shipped 180B artifact, fetching routed experts on demand instead of
uploading all of them, 8 tokens in:

```
eager        56.2 GiB resident   prefill 0.34s   0.08 s/tok
lazy          8.5 GiB resident   prefill 2.29s   0.37 s/tok
```

Identical output (argmax 1892, 13.9764). So the idea is sound and the payoff is
large: 6.6x less resident for ~4.6x slower decode. It is a FOOTPRINT tool, not a
speed one.

It also settled a design question by measurement. Grouping experts per fetch to
amortise gfx1151's 2 MiB GTT rounding — one `down_proj` is ~870 KB, a 2.4x tax —
measured strictly WORSE:

```
group=4:  29.6 GiB after 8 tokens, 1.07 s/tok
group=1:   8.5 GiB after 8 tokens, 0.37 s/tok
```

Grouping multiplies COVERAGE: each routed expert drags in 3 that may never be
used, and with top-10 routing over 48 layers the union grows fast enough that the
extra bytes swamp the saved allocations.

## Why it could not simply call the pager

The artifact is ready — it carries **24576 `RoutedExpert` module records** (512
experts x 48 layers), so `register_expert_modules` has its input. The mismatch is
in what the pager's expert-module path is FOR:

1. **It targets indexed-kernel consumers.** Resident modules are handed out only
   through `patch_expert_module_ptr_table`, which pushes device pointers into the
   per-layer table the `*_indexed*` MoE kernels read. qwen4_exp has no such table:
   its MoE does CPU top-k and loops one expert at a time through `weight_gemv`, so
   it needs a `GpuTensor` per expert, not a pointer table.
2. **Its length contract is G256.** `module_resident_len` maps `OqPlusCompact` to
   `oq8_moe_packed_len`, i.e. compact blocks are EXPANDED to int8 MoE blocks on
   page-in. This model's `down_proj` is `OqPlusCompactG128` (qt 52), which the
   table does not list at all — it falls through to `tensor.data_size`, which is
   the wrong length.
3. **That expansion is what OOM'd the eager load** in the first place: compact
   4.25 b/w -> 8 b/w added ~20 GB. Under the pager it would be bounded by the
   eviction budget rather than fatal, but it is still 2x the bytes per resident
   expert.

## What integrating actually needs

In `weight_pager`:

- `OqPlusCompactG128` in `module_resident_len` and `module_requires_host_repack`,
  with a G128 packed length (`oq_moe.rs` has only 256-group helpers today).
- A public accessor returning a resident module's buffer and its two relative
  offsets — the information `patch_expert_module_ptr_table` already computes
  internally — so a non-indexed consumer can build a `WeightTensor` view.
- Ideally, a residency mode that keeps compact blocks COMPACT rather than
  expanding to MoE blocks, for consumers that decode compact natively
  (`GemvOqCompactG128Prerotated` exists).

In qwen4_exp: hold one `WeightPager` per model, register the modules from
`hfq.modules()`, and have `ExpertStack` ask it for residency instead of owning a
cache. That deletes the eviction question here entirely — which is the point.

## Do NOT

Do not reintroduce a per-arch expert cache. Residency, LRU, GTT-aware accounting
and the transport are the pager's job, and a second implementation gets the easy
half (a HashMap) while silently missing the hard half (eviction under pressure,
`gtt_alloc_cost`, and the shared budget across everything else resident).
