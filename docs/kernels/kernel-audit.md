# HIP kernel audit — `kernels/src`

_Audit date: 2026-08-30. Scope: inventory/catalog + health/red-flags of the
`kernels/src` HIP tree. No GPU runs; static analysis over sources and the
`hipfire-rdna` dispatch layer. Regenerate with the queries in
[Methodology](#methodology)._

## Summary

| Metric | Count |
| --- | --- |
| `.hip` source files | 876 |
| `__global__` entry points | 973 (919 distinct names) |
| Files with ≥1 entry point | 800 (the rest are shared headers/helpers) |
| Kernel-source `include_str!` consts in `kernels.rs` | 875 |
| Arch-selector functions (`*_for_arch`, quant GEMM/GEMV overlays) | 33 — **all reachable from dispatch** |
| Distinct kernel names referenced in Rust | 910 / 919 |
| **Orphan sources (loaded by nothing)** | **8** — see [F1](#f1-orphan-kernel-sources) |

Headline: the tree is overwhelmingly live. 910 of 919 kernel names and every
one of the 33 arch-selector functions are reachable. Three findings follow —
one concrete (8 dead sources), two accuracy/coverage notes.

## How dispatch reaches a kernel

Kernels are **JIT-compiled at first use**, not prebuilt. Each `.hip` is pulled
into `crates/hipfire-rdna/src/kernels.rs` as an `include_str!` string const
(`GEMM_..._SRC`). A launch calls `ensure_kernel(module_name, source, func_name)`
(`dispatch/mod.rs:1726`), which compiles the source to an `.hsaco` via
`self.compiler.compile(...)`, caches the resolved `hipFunction` by `func_name`,
and reuses it thereafter. `ensure_kernel` is the **universal chokepoint**: every
launch resolves here, and `kernel_trace::record(func_name)` fires on each one.

Arch selection takes two forms. The 33 quantized GEMM/GEMV overlay families
(MQ-Lloyd, HFQ, HFP) use a **selector function** that maps arch → source; other
families (attention, norm, elementwise, …) dispatch by a direct
`ensure_kernel("name", NAME_SRC, "name")` with an arch-tagged const chosen inline
at the call site. A selector example (`kernels.rs`):

```rust
pub fn gemm_qkv_mq4g256_lloyd_wmma_for_arch(arch: &str) -> (&str, &str) {
    match arch {
        "gfx1201" | "gfx1200" => (..._GFX12_SRC,   "..._gfx12"),
        "gfx1151"             => (..._GFX1151_SRC, "..._k4_gfx1151"),
        "gfx1100" | ...       => (..._SRC,         "..._rdna3"),
        // ...
    }
}
```

The selector returns the `(source, func_name)` pair for the live GPU; dispatch
feeds it straight into `ensure_kernel`. This is why a kernel source can be
"referenced" only inside `kernels.rs` yet still be live — the selector that
names it is the reachable code. **Reachability in this audit means: named in
dispatch, or named by a selector that dispatch calls.** All 33 selectors are
called, so the chain holds.

Reachability here is static. For empirical confirmation, `kernel_trace`
(`crates/hipfire-rdna/src/kernel_trace.rs`, off by default) records every
`func_name` that actually launches — run a representative model set with it on
to get the runtime-launched set.

## Inventory — family × arch

Files per family and arch target (`portable` = lives in `kernels/src/` root and
selected for RDNA generically or by a filename-tagged variant):

```
FAMILY            portable  gfx906  gfx942 gfx1030 gfx1100 gfx1103  gfx11 gfx1151  gfx12 gfx1201  TOTAL
gemm                   170      18       5      28       2       ·      ·      30      43       ·    296
gemv                   122       ·       3       5      12       ·      1       2       1       2    148
attention               81       2       ·       ·       ·      13      ·       ·       3       ·     99
fused_qkv/gate          32       ·       3       ·       6       ·      ·       1       ·       ·     42
kv/cache                39       ·       ·       ·       ·       ·      ·       ·       ·       ·     39
elementwise             33       ·       ·       ·       ·       ·      ·       ·       ·       ·     33
conv/ssm                25       ·       ·       ·       ·       ·      ·       3       ·       ·     28
norm                    19       ·       1       ·       ·       3      ·       2       ·       ·     25
moe/router              21       ·       ·       ·       ·       ·      ·       1       ·       ·     22
rope                    16       ·       ·       ·       ·       ·      ·       ·       ·       ·     16
softmax/sample          13       ·       ·       ·       ·       2      ·       ·       ·       ·     15
fwht/rotate             12       ·       1       ·       ·       ·      ·       ·       1       ·     14
embed                   13       ·       ·       ·       ·       ·      ·       ·       ·       ·     13
dequant/pack             5       ·       ·       ·       ·       ·      ·       ·       ·       ·      5
train                    2       ·       ·       ·       ·       ·      ·       ·       ·       ·      2
other                   76       ·       ·       ·       ·       1      ·       1       1       ·     79
TOTAL                  679      20      13      33      20      19      1      40      49       2    876
```

(Family is a filename heuristic; a file with mixed ops lands in its dominant
family. `other` is mostly probes, test scaffolds, and multi-op fused bodies.)

Coverage reading:

- **`portable` carries the tree (679/876).** Most families exist only as a
  portable source; the arch columns are hand-optimized overlays for the hot
  GEMM/GEMV/attention paths, not full re-implementations.
- **gfx1100 and gfx1103 have almost no bespoke GEMM/GEMV** — they run the
  portable RDNA3 sources. gfx1103's own column is attention + norm + softmax
  (the LDS reductions in [F2](#f2-gfx1103gfx11-lds-barrier-surface)).
- **gfx12 (43) and gfx1151 (30) dominate the bespoke GEMM overlays** — the WMMA
  prefill and Lloyd-MQ paths. gfx906/gfx942/gfx1030 have older dot-product GEMM
  overlays.
- **gfx1201 (2) and gfx11 (1) are nearly empty** — they inherit gfx12 / RDNA3
  portable paths via the selectors rather than carrying their own sources.

## Findings

### F1 — Orphan kernel sources

Eight `.hip` sources are compiled into the binary as `include_str!` string
consts but reached by **no dispatch path and no called selector**. They add to
binary size and to the maintenance/confusion surface (a reader can't tell they
are dead), and none is covered by a gate. All date to a June-2026 gfx12/RDNA4
and DeepSeek-4 push; the live paths that replaced them use differently-named
sources.

| Source (`kernels/src/`) | Const | Last commit | Status |
| --- | --- | --- | --- |
| `attention_dflash_wmma.gfx12.hip` | `ATTENTION_DFLASH_WMMA_GFX12_SRC` | 2026-06-07 | gfx12 DFlash-F32 WMMA lowering; dispatch loads the portable `ATTENTION_DFLASH_WMMA_SRC` instead |
| `attention_dflash_wmma_m32.gfx12.hip` | `ATTENTION_DFLASH_WMMA_M32_GFX12_SRC` | 2026-06-07 | as above, batch≥32 variant |
| `gemm_hfq4g256_residual_mmq.gfx12.hip` | `GEMM_HFQ4G256_RESIDUAL_MMQ_GFX12_SRC` | 2026-06-06 | gfx12 i8-WMMA MMQ residual body (6 globals); superseded |
| `gemm_gate_up_hfq4g256_wmma_gfx12_bt.hip` | `GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC` | 2026-06-08 | batch-tiled prefill; live path uses non-`_bt` `gemm_gate_up_hfq4g256_wmma_gfx12` (`overlays/gfx12.rs`) |
| `gemm_hfq4g256_residual_wmma_gfx12_bt.hip` | `GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC` | 2026-06-08 | as above, residual |
| `gemm_qkvza_hfq4g256_wmma_gfx12_bt.hip` | `GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC` | 2026-06-08 | as above, QKVZA |
| `deepseek4_attn_swa_topk_batched_wmma.hip` | `V4F_ATTN_SWA_TOPK_BATCHED_WMMA_SRC` | 2026-06-06 | DSA gathered-attention WMMA; only named in a `bench_dsa_direct_wmma.rs` doc comment, not loaded |
| `deepseek4_attn_swa_topk_direct_wmma.hip` | `V4F_ATTN_SWA_TOPK_DIRECT_WMMA_SRC` | 2026-06-06 | as above, direct variant |

Note the three `*_gfx12_bt` sources trace to a commit titled _"batch-tiled
prefill GEMMs … +26-33% AR prefill, default-on"_. The **feature is live** — but
via the non-`_bt` gfx12 overlays; these `_bt` sources are the superseded
first cut, not the shipped path. Nothing here suggests a regression.

**Recommendation:** confirm with a `kernel_trace` run (none of the eight should
appear), then either delete the source + its const, or, if kept as a staged
next-gen variant, wire a selector arm behind a feature flag and add a coherence
gate — a source with neither a caller nor a gate rots silently. This is the
recurring hazard the config area names elsewhere: something that compiles,
applies to nothing, and says nothing.

### F2 — gfx1103/gfx11 LDS + barrier surface

35% of the tree touches LDS: **300 files use `__shared__`, 284 use
`__syncthreads`.** That is fine on most arches, but gfx1103 (Phoenix) and gfx11
are the CWSR / HIP-719 hazard targets (see `AGENTS.local.md` and
`docs/plans/gfx1103-lds-hip719-investigation.md`) where multi-wave barrier / LDS
paths can wedge the GPU unless the host boots `amdgpu.cwsr_enable=0`.

Seven kernels in the hazard arches use `__syncthreads`:

```
gfx1103/rmsnorm.gfx1103.hip            gfx1103/layernorm.gfx1103.hip
gfx1103/softmax.gfx1103.hip            gfx1103/argmax.gfx1103.hip
gfx1103/max_prob.gfx1103.hip           gfx1103/rmsnorm_at_slot_buf.gfx1103.hip
gfx11/gemv_hfp4g32_dot2.gfx11.hip
```

**Accuracy correction worth carrying back to `AGENTS.local.md`:**
`AGENTS.local.md` names `rmsnorm_f32_at_slot_buf_gfx1103` among "register-tiled /
wave-reduced **no-LDS** kernels … preferred PRODUCTION default." The intra-wave
reduction is indeed register/DPP-based (`__shfl_down`, no LDS), but the kernel
still keeps a minimal `extern __shared__ float wave_sums[]` + two
`__syncthreads()` for the **cross-wave** reduction (up to 16 wave32 waves):

```c
// kernels/src/gfx1103/rmsnorm_at_slot_buf.gfx1103.hip
for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32); // no LDS
extern __shared__ float wave_sums[];   // <-- LDS scratch
__syncthreads();                       // <-- barrier, twice
```

So it is **wave-reduced, not fully LDS-free**. On nix2 this is safe (CWSR off);
on any Phoenix host *without* the workaround it still touches the hazard surface.
The "no-LDS" wording overstates it — either tighten the note to "wave-reduced,
minimal cross-wave LDS" or, if a truly LDS-free cross-wave path is wanted, that
is a real kernel task, not a doc fix. Not a correctness bug; a precision issue in
the guidance that could mislead someone porting to an unpatched host.

### F3 — Quant activation-basis (FWHT-rotation) contract: documentation coverage

`kernels/AGENTS.md` states the load-bearing contract: OQ/MQ/HFQ weights are
quantized in a **FWHT-rotated basis**, so the caller must Hadamard-rotate `x`
(`rotate_x_mq`) before the GEMV/GEMM; feeding a raw activation "compiles and runs
but produces garbage." Correctness lives in the dispatch/arch caller, not the
kernel — so this is **not a bug**, but the kernels are where a new author reads
for the invariant, and coverage is thin:

- **371** OQ/MQ/HFQ/HFP `gemv`/`gemm` kernel files.
- **54** (≈15%) mention `rotate`/`fwht`/`hadamard` in a comment.

The other ~85% carry no in-file marker that their `x` input is pre-rotated. A
one-line banner comment on each OQ/MQ GEMV/GEMM entry ("input `x` must be
FWHT-rotated — see `kernels/AGENTS.md`") would make the contract local to where
someone edits, and is cheap next time each family is touched.

## Methodology

Reproducible from repo root; intermediate TSVs were built with these:

```sh
# entry points: file<TAB>__global__ name
grep -rlE '__global__' kernels/src --include=*.hip   # 800 files

# source-const → file map and reachability lives in kernels.rs:
#   1. every `pub const *_SRC = include_str!("...hip")`  (875)
#   2. a const is LIVE if referenced by dispatch, or by a `*_for_arch`
#      selector that dispatch itself calls (33 selectors, all called)
#   3. consts live by neither = orphans (8)  → F1

# LDS / barrier hazard surface
grep -rl '__shared__'     kernels/src --include=*.hip | wc -l   # 300
grep -rl '__syncthreads'  kernels/src --include=*.hip | wc -l   # 284

# runtime confirmation of the live set
#   set the kernel_trace env flag and run a model sweep; compare launched
#   func_names against the 8 orphans (expect none).
```

Static reachability can miss a source pulled in by a macro or `build.rs`
codegen; none of the 8 orphans is (checked: no `build.rs`/codegen reference,
example refs are doc-comments only). Confirm with `kernel_trace` before any
deletion.
