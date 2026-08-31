# TODO: page qwen4_exp's routed experts through `weight_pager` (not a second cache)

**Status:** DONE (2026-08-31) — qwen4_exp routes its routed experts through
`hipfire_runtime::weight_pager`. See "How it landed" at the bottom for what the
plan below got wrong. A bespoke lazy-expert cache was written, measured, and then
REVERTED because it duplicated residency and eviction logic the pager already
owns.

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

## How it landed

Both halves are in. The plan above was right about the pager work and wrong about
one thing that mattered.

**In `weight_pager`:** a second residency layout,
`ExpertResidentLayout::PerExpertNative`, alongside the existing
`IndexedMoeBlocks`. It keeps compact blocks COMPACT (so the 2x expansion that
OOM'd the eager load never happens) and hands out `resident_expert_views(key)` —
the buffer plus the two relative offsets — so a consumer that has no pointer
table can build a `WeightTensor` per expert.

**In qwen4_exp:** `ExpertStack` became an enum, `Resident { .. } | Paged(..)`,
decided PER LAYER at load. `ExpertStack::expert()` now takes `&mut Gpu` and pages
on demand. Nothing else in the MoE forward changed.

### The plan was wrong about the repacker

It assumed an artifact had to be repacked before it could be paged. It does not:
**`hipfire-quantize` already writes an HFQM v2 module table**, so every artifact
this arch loads is pageable as emitted. The `docs/todo/2026-08-31-repacker-for-
paged-qwen4exp.md` work is a page-in *cost* optimisation (pre-split planes so the
transform is not redone on every page-in), not a prerequisite. Nothing was
blocked on it.

Presence of the module table is therefore the switch — there is no flag, because
there is nothing for a flag to choose between. `HIPFIRE_QWEN4EXP_PAGED_EXPERTS=0`
exists only so the resident path stays reachable for A/B, which is what the gate
uses.

### Two divergence bugs, and the structural fix

`PerExpertNative` originally re-derived the device transform in its own `match`,
parallel to the one in `trunk_gpu::stack_experts`. They diverged twice:

* the compact arms dropped `normalize_compact_overlays`, which the resident
  loader does before splitting planes — a paged compact expert decoded junk
  corrections out of its unused overlay slots;
* `Oq8G256` was missing entirely from `module_requires_host_repack`, so an oq8
  module took the verbatim-fetch branch and the GEMV was handed raw on-disk
  bytes. Symptom: **non-finite prefill logits**, with no error from the pager.

Neither is catchable by review of two similar-looking matches, so the fix is
structural: one `per_expert_native_form()`, called by the pager AND by
`stack_experts`. The only thing that still restates the layout is
`per_expert_native_len()` — needed because the pager budgets before it reads any
bytes — and `prepare_expert_module` now errors if the two disagree.

### Evidence

`tests/qwen4exp-gate.sh` runs paged and resident over the SAME oq4 artifact with
a 512 KiB budget, far below the fixture's expert bytes:

```
resident                             argmax 3443
paged  160 cold loads, 159 evictions argmax 3443
```

One module resident at a time, churning the whole way, bit-identical output. The
cold-load and eviction counts are asserted, not just printed: the first version
of this check compared the two arms *before* any forward pass, saw a match, and
passed while paging nothing. "Paged agrees with resident" is evidence only if the
paged arm paged.

On the shipped 180B artifact (`examples/serve_real`, 8 GiB expert budget):

```
                 load    prefill(4)  decode      resident experts   argmax
eager (reverted) 65.5s   0.34s       0.08 s/tok  56.2 GiB           1892 (13.9764)
lazy  (reverted)  ~65s   2.29s       0.37 s/tok   8.5 GiB           1892 (13.9764)
paged (shipped)   6.9s   2.48s       0.32 s/tok   8.0 GiB           1892 (13.9764)
```

24576 modules registered, 2163 cold loads, 5517 hits, 798 evictions — the budget
is enforced, not merely declared. Load is ~9.5x faster than eager because nothing
uploads 56 GiB of experts before the first token; decode pays for the fetches
instead. It remains a FOOTPRINT tool: 7x less resident for ~4x slower decode, at
identical output.

### Known-broken neighbour, not fixed here

`cargo run -p hipfire-runtime --example qwen35_hfq_modules repack` writes an
artifact that fails to open: `HFQM v2 tail hash mismatch`. It is not needed for
paging (see above), so it was left alone — but it is broken, and it is the tool
someone would reach for first.
