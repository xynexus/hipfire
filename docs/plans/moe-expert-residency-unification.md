# MoE Expert Residency — One Unit, Two Policies

Status: in progress (2026-08-08) — Phases 0, 1 and 2 landed; Phase 3 unblocked
and not started.

## Why

Four arch crates load routed MoE experts four different ways, and the
difference is not architectural — it is who happened to write the loader:

| Arch | Expert residency | Owning device allocations |
|---|---|---|
| `hipfire-arch-deepseek4` (`src/arch.rs:298-357`) | host-side `combined` concat across owned experts, one upload, ptr table = `base + local*stride` | ~2 per layer |
| `hipfire-arch-minimax` (`src/minimax.rs:686-760`) | same compact-blob pattern, EP-shard aware | ~2 per layer |
| `hipfire-arch-qwen35` (`src/qwen35/loading.rs:1191-1221`) | slab-aliased, **or** per-tensor, **or** pager-paged, **or** stacked `[E,M,K]` aliases | 1 per 512 MiB bank … or 2 × n_exp × n_layers |
| `hipfire-arch-lfm2moe` (`src/lfm2moe.rs:838-885`) | per-expert `upload_wt_raw` for gate_up and down | 2 × n_exp × n_layers |

`Gpu::upload_raw` is a bare `hipMalloc` with no pooling
(`crates/hipfire-rdna/src/dispatch/mod.rs:2018`), so the unpacked shape means
one buffer object per expert per projection. On a 256-expert / 40-layer
artifact that is 20,480 BOs. Upstream `warpfront/hipfire` measured that exact
count at **4.35 GB of pure allocator overhead** on a 7900 XTX with zero change
in requested payload bytes (PR #534 discussion, fixed there by packing to 80
owners). Two of our four loaders already avoid it; two do not.

Meanwhile eviction — the other answer to the same problem — is reachable by
exactly one arch, and only because that arch is the one that called the pager.

## The assumption to retire

Packing and paging are not alternatives, and the pager is not qwen-specific.

`WeightPager` and `hfq_modules` both live in `hipfire-runtime` and are generic
in `WeightId::Expert { layer, expert, role }`, `ExpertRole`, `ExpertShape`, and
`HfqModuleKind`. Qwen appears only in doc comments naming the v0.1 target
(`weight_pager.rs:8`, `:70`, `:917`).

More importantly, the pager's **module** path already *is* packing:
`register_expert_module` (`weight_pager.rs:1025`) requires a record whose
`gate_up_proj` and `down_proj` are both locatable by
`find_module_tensor_rel_ptr`, and `ensure_expert_module_resident` (`:1191`)
sizes **one** allocation via `module_resident_len` (`:651`) and tracks it in a
module LRU. One owning buffer, several logical tensors at relative offsets,
plus eviction. Consolidated ownership and eviction already coexist there.

The real constraint is only that **the pack unit and the page unit must be the
same unit**. Everything else was wiring.

## What already exists

The container work is done, and it is already arch-neutral:

- **Every MoE `.hfq` ships a routed-expert module table.** The quantizer emits
  `hfqm_modules` (`hipfire-quantize/src/hfq_out.rs:451-530`, format
  `hipfire.hfqm.modules.v1`) into the tail metadata, with per-module
  `data_offset` / `data_size` and per-tensor `rel_offset` / `quant_type` /
  `shape` / `group_size`.
- **Grouping is family-independent already.** `expert_key`
  (`hfq_out.rs:406-413`) finds a `layers.<N>` segment and an `experts.<E>`
  segment anywhere in the dotted name, so it matches
  `layers.{L}.ffn.experts.{e}.w1` (deepseek4),
  `…mlp.experts.{e}.gate_up_proj` (qwen35),
  `…block_sparse_moe.experts.{e}.w1` (minimax) and
  `…feed_forward.experts.{e}.w1` (lfm2moe) without change. The comment at
  `:428-430` states the intent outright: "every expert a contiguous page-in
  unit while remaining independent of the model-family name."
- **`canonical_tensor_order` already lays each expert out contiguously**
  (`hfq_out.rs:431`), which is what makes a module a single-range page-in.
- **The format already carries a policy field**: `placement_policy`, currently
  hardcoded to `"lazy_lru"` (`hfq_out.rs:509`).
- **The consumer ABI is already common.** All four archs hand kernels the same
  thing: a device `u64` pointer table per projection role, `[2 * n_exp]` F32
  slots (`qwen35/layout.rs:96-100`, `lfm2moe.rs:546-547`,
  `deepseek4.rs:347-355`). `patch_expert_ptr_table` (`weight_pager.rs:1270`)
  already writes into it arch-agnostically.
- `hipfire_runtime::oq_moe` already holds the shared OQ4 / OQ+compact →
  MoE-block repack helpers all four loaders call.

So this is not a new subsystem. It is connecting a finished container feature
to three arch crates that never called it, and adding the one policy the
runtime side is missing.

## Architecture

### 1. Residency policy on the pager

`PagerConfig` (`weight_pager.rs:850`) currently has `vram_budget_bytes` and
`trace`; eviction is implicitly always-on and disabled only by setting the
budget to `u64::MAX`. Make the intent explicit:

```rust
pub enum ResidencyPolicy {
    /// Every registered module stays resident for the model's lifetime.
    /// Equivalent to today's hand-rolled packing: one owning allocation per
    /// module, no eviction, no per-token admission.
    PinAll,
    /// Admit on demand, evict LRU against a byte budget. Today's behavior.
    LazyLru { vram_budget_bytes: u64 },
}
```

`PinAll` is not a new mechanism — it is `ensure_expert_module_resident` with
eviction switched off and admission hoisted to load time. Packing becomes the
degenerate case of paging at 100% residency, which is why one code path can
serve both.

Honor `placement_policy` from the module record as the default, so an artifact
can express its own intent, with the runtime free to override on VRAM
pressure.

### 2. Finish the role-rank table

`expert_role_rank` (`hfq_out.rs:415-426`) maps
`gate_up_proj` / `gate_proj` → 0, `up_proj` → 1, `down_proj` → 2, everything
else → 3. deepseek4, minimax, and lfm2moe name their projections `w1` / `w3` /
`w2`, so all three land in the unordered bucket and get no deterministic role
ordering within a module. Extend the table (`w1` → 0, `w3` → 1, `w2` → 2).

This changes canonical tensor order for those families, so it **moves their
artifact hashes** — see Risks.

### 3. Arch adoption seam

An arch opts in by supplying what is genuinely arch-specific and nothing else:

- **Fusion**: whether `gate_up` arrives pre-fused (qwen35) or as `w1`‖`w3` to
  byte-concat (the other three).
- **Repack**: which `oq_moe` helper applies, if any.
- **EP ownership predicate**: `|expert| -> bool`, plus the non-owned pointer
  policy. deepseek4's rule is genuinely its own — non-owned `gate_up` points at
  a shared *zeroed* dummy so SwiGLU(0,0)=0 contributes nothing, and non-owned
  `down` reuses the base because its input is zero regardless
  (`arch.rs:174-185`). That reasoning stays in deepseek4.
- **AWQ sidecar convention**: lfm2moe attaches one per-layer scale to expert 0
  (`lfm2moe.rs:894-912`); qwen35 loads per-tensor.
- **Per-expert dtype bookkeeping**: qwen35 tracks
  `expert_gate_up_dtypes: Vec<DType>` because mixed-precision induction can
  leave one expert at BF16 (`layout.rs:115-121`).

Everything else — module registration, allocation, pointer-table construction
and patching, residency accounting — moves behind the shared unit.

### 4. What deliberately does not change

- **qwen35's slab path stays.** It reaches the same end state by a different
  mechanism (alias into 512 MiB banks) and is the default on UMA
  (`HIPFIRE_GPU_SLAB_LOAD=auto` → `gpu.integrated`, `loading.rs:2435-2445`).
  It is not a loader to unify; it is a second residency backend that already
  works.
- **Forward paths stay untouched.** Every arch keeps reading the same pointer
  table with the same layout. Only the addresses inside it change, from N
  independent BOs to `base + offset`.

## Phases

Each phase lands independently and is gated before the next starts.

**Phase 0 — parity harness. DONE (2026-08-08).** The gate was red on 14 cells
across exactly the families this plan touches, so it could not serve as a parity
oracle. Triaged by bisecting against a worktree at the baseline-record commit
`5dc01e4b0`:

| cells | verdict |
|---|---|
| deepseek4 `oq8`/`oq8+`/`oq8++`, deepseek4_compressed ×3 (6) | **real regression** — green at `0060481ee`, red at `8b9ee5392` |
| lfm2_moe `oq8`, `oq8+`, `quantize:oq8++` (3) | **real regression** — same boundary |
| minimax 4 `oq4` cells + `quantize:oq8++` (5) | **pre-existing**, fails identically at `5dc01e4b0` |

Root cause of the 9 regressed cells, all one commit: `8b9ee5392` inserted an
**unguarded `_` wildcard arm above the `oq8` literals** in
`HfqInputFormat::from_flag` (`hipfire-quantize/src/main.rs:3919`), replacing a
guarded `_ if parse_opus_mixed_format(flag).is_some()`. That made
`"oq8"` / `"oq8+"` / `"oq8++"` unreachable, so `from_flag` returned `None` for
every OQ8 flag and each call site degraded differently — silently skipped
tensors (lfm2_moe scoring KLD exactly 0.000000), a wrong-format fallback
(deepseek4 at 0.0387), or the "no LDLQ-eligible tensors were attempted" hard
error. `oq4` was unaffected because its arm sits above the wildcard. rustc
reported the whole thing as `unreachable_patterns`; the warning was the
diagnosis.

Fixed by ordering: `oq8` arms moved above the wildcard, with a comment pinning
why the wildcard must stay last. Result — deepseek4, deepseek4_compressed and
lfm2_moe fully green; whole-gate failures 14 → 7.

**The fix also unmasked minimax**, whose `oq8`/`oq8+` cells had been passing only
because the quantizer was not producing OQ8 at all. With OQ8 restored it showed
its true 7 failing Opus cells — the numbers recorded as an inherited breakage in
`2026-08-05-opus-across-model-families.md:82-93`.

**That is also fixed now.** Bisecting minimax separately (its baselines date from
`753df2b27`, 421 commits back) named `1fa0f04dd`, which defaulted LUT3 heads to
stay resident and taught six loaders to decode packed tensors — minimax not among
them. It uploaded packed head bytes tagged as a logical dtype, which only a
batch-1 GEMV can read, so it was wrong at prefill, and KLD scores through
prefill. Confirmed by A/B: `HIPFIRE_BF16L3_RESIDENT=0` made all 7 pass on
unmodified master. Both minimax head sites now decode, mirroring
`transformer_loader`.

**Phase 0 exit state.** `tiny-quant` passes across all 17 selected families.
`tiny-state` is 8 failures, verified present on pristine `origin/master` with
identical observed hashes and all in `fp16` cells (gemma3, gemma3_vl,
gemma4_dense, gemma4_moe, gemma4_ple, qwen3_5, qwen3_5_moe, qwen3_5_vl) — the
un-re-recorded drift `1fa0f04dd` already called "the pre-existing baseline 8".
Every family this plan uses as an oracle is green in both gates, minimax
included.

**Carry-forward risk for this plan.** deepseek4, lfm2moe, llama, nemotron and
embeddinggemma read tensors themselves and have no packed-decode arm either.
They pass only because the recoding applies to plain-BF16 gather-shaped tensors
that actually get smaller, so their fixtures never hand them a `qt=49`. Phase 2
touches lfm2moe and Phase 3 touches deepseek4 — if either starts failing on a
head tensor, check this before suspecting the residency work.

**Phase 1 — `ResidencyPolicy::PinAll`. DONE (`fc55d47e4`).** `PagerConfig.policy`
with `LazyLru` (default, unchanged behavior) and `PinAll`. `PinAll` fails closed
via `may_evict` → `PinnedBudgetExceeded`, checked inside `evict_lru_until` so no
call site can forget it. 4 new pager unit tests; qwen3_5_moe 12/12 green.

**Phase 2 — lfm2moe onto the shared unit. DONE (`4c068d084`, `a67a4e983`,
`7a60228a4`).** Split into a pager half and an arch half:

- `ModuleRole` + role-ordered layout, so `w1`/`w3`/`w2` name projections just as
  `gate_up_proj`/`down_proj` do, and a split gate_up is fused contiguously.
- `ExpertModulePtrs` / `expert_module_ptrs`, so an adopter can build non-owning
  views over pager-owned storage.
- `HIPFIRE_LFM2_EXPERT_RESIDENCY=pin`, with explicit ownership
  (`MoeFfn::experts_are_views`, `Lfm2MoeWeights::expert_pager`).

Validated as an A/B against the same baselines: 8/8 lfm2_moe quant cells pass on
both arms, and the pinned path is confirmed exercised rather than skipped.
Routed-expert buffer objects go 16 → 8 on the tiny fixture (one per module
rather than one per projection).

Two things the plan got wrong, found by building it:

- **Eviction is not "available by config" for lfm2moe.** `PinAll` is; `LazyLru`
  needs the forward to admit its routed experts per token, and the lfm2 forward
  has no such hook. The flag refuses `lru` explicitly rather than pinning
  everything while claiming to page.
- **The real-world artifact orders its tensors `w1, w2, w3`** — down sits
  *between* the gate_up halves. Ordering by `rel_offset` would have fused w1
  with w2. This is why the layout is role-driven, and it was a live case rather
  than a hypothetical.

**Phase 3 — deepseek4 and minimax onto the shared unit, at per-expert
granularity. No coarser unit.**

An earlier revision of this section claimed migrating these two would be a 128×
allocation regression and made a coarse residency unit a precondition. That was
wrong, and worth recording because the error is easy to repeat: it compared
deepseek4's per-layer packing against the shared unit **under `PinAll`** — a
configuration deepseek4 would never select. The reason to give it the pager is
eviction, and under `LazyLru` the allocation count is the resident working set,
not the registered count. The section even stated that and then concluded the
opposite.

**Per-expert granularity is required, not merely tolerable.** Concurrent
microbatches route to different experts, so admission has to happen per expert —
which is what the existing machinery already does:
`patch_expert_ptr_table(layer, top_indices, …)` patches exactly the experts a
step routed to. A per-layer unit would force whole layers resident and destroy
the routing-driven working set that makes paging worth anything. The scheduler
already speaks this vocabulary: `ResidencyMode::{Auto, Full, QwenMoeModules}`
(`hipfire-scheduler/src/lib.rs:67`), where the module-granularity mode is named
after the only arch that currently has it.

So deepseek4's per-layer `combined` blob (`arch.rs:323`) is not a baseline to
preserve — it is the obstacle. One layer-wide allocation cannot admit or evict a
single expert, which is precisely what microbatching needs.

Scope:

1. Register deepseek4's routed-expert modules and drive residency through the
   pager, keeping its EP-ownership predicate and non-owned pointer policy as
   inputs (that reasoning stays in deepseek4 — a wrong pointer there is silent
   numerical corruption).
2. Same for minimax.
3. Keep the per-layer blob path available for the single-stream,
   everything-fits case, selected by policy rather than by arch. `PinAll` at
   per-expert granularity is the configuration to avoid for a 256-expert model,
   and nothing should select it by default.

Validate: `deepseek4`, `deepseek4_compressed`, `minimax` rows unchanged, plus a
multi-GPU EP smoke on medusa.

**The role-rank half of this phase is dropped.** It existed to give `w1`/`w3`/
`w2` deterministic ordering *within* a module so the fused pair would be
adjacent. Phase 2a solved that at the pager, by role, independent of on-disk
order — so changing `expert_role_rank` now buys nothing and would move artifact
hashes for three families for no functional gain. Contiguity of an expert's
tensors, which is what makes a module a single page-in range, already holds
without it.

**Phase 4 — enable eviction where it pays.** deepseek4 is the arch that wants
it most: the largest expert footprint, and today its only answer to not fitting
is EP sharding across N GPUs, which does nothing for a single-card user. Gate
on a measured decode-latency delta, not on the mechanism working.

## Risks

- **Device addresses and allocation counts change.** This is exactly where
  upstream's equivalent packing fix hit a gfx12 HipGraph regression and had to
  ship gfx11-only. Any HipGraph-captured MoE path needs an explicit gfx11 vs
  gfx12 A/B before the default flips.
- ~~**Canonical tensor order is artifact-visible.**~~ Retired: the role-rank
  change that would have moved those hashes is dropped, because Phase 2a made it
  unnecessary. No phase now changes artifact bytes.
- **`PinAll` must not silently regress into paging.** If a `PinAll` model does
  not fit, failing closed with a named error beats thrashing an LRU nobody
  asked for.
- **Non-owned EP pointer policy is load-bearing correctness**, not an
  optimization. Preserve deepseek4's zeroed-dummy semantics verbatim when it
  moves behind the shared unit; a wrong pointer there is silent numerical
  corruption, not a crash.

## Open questions

- ~~Should the page/pack unit ever be coarser than one expert?~~ **Answered: no.**
  Microbatching requires per-expert admission, so a coarser unit would defeat the
  routing-driven working set that makes paging worth anything. The question only
  looked open because an earlier revision compared against `PinAll`, which no
  large MoE should select.
- ~~`hipfire-arch-zaya`, `-cohere2`, `-nemotron` device-side residency not
  traced.~~ **Traced (2026-08-08):**
  - **nemotron — unpacked.** `model.rs:334` loops routed experts calling
    `load_linear_hfq(hfq, gpu, …)` per projection, so one device allocation per
    projection per expert per layer.
  - **zaya — unpacked.** `gpu.rs:363` `GpuExpert { gate_up, down }`, one record
    per expert with two owning `LinearWeight`s, freed individually at `:437`.
  - **cohere2 — no device-side routed-expert path** in `src/*.rs`; its per-expert
    loops are calibration-stream only.

  So the per-expert-unpacked shape was never unique to lfm2moe: **three arches
  had it** (lfm2moe, now converted, plus zaya and nemotron). That raises the
  value of the shared unit above what this plan assumed when it picked lfm2moe as
  the only adopter, and it is an argument for resolving the Phase 3 fork in
  favour of a real shared implementation rather than leaving each arch its own.
