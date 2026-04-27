# hipfire arch-port playbook

When you (or an agent helping you) want to add a new GPU arch to
hipfire, read this end-to-end before touching code. Most of the
mistakes other arch ports have hit are documented here so you don't
spend a week chasing them.

## When to use this skill

- A user reports a HIP codegen / kernel-select failure on a new arch
  (e.g. issue #54: `Cannot select: intrinsic %llvm.amdgcn.wmma...`
  on gfx1201).
- You're adding `gfx1200` / `gfx1201` / `gfx1151` / `gfx1152` /
  `gfx94x` / `gfx950` to the supported list.
- You see the `(EXPERIMENTAL — opt-in only)` flag on a feature and
  want to mainstream it on a new arch.
- You're refactoring `dispatch.rs`'s arch-conditional branches.

## What's in this skill

| File | Purpose |
|---|---|
| `playbook.md` (this) | Top-level workflow, when-to-use, contributor pointer |
| `wmma-matrix.md` | WMMA operand-shape × builtin × lane-layout table per arch |
| `validation.md` | The three gates every port must pass before merge |
| `contributor-onboarding.md` | If you have hardware and want to help — start here |

## The arch-port workflow (the load-bearing 6 steps)

### 1. Read `wmma-matrix.md` first

Most arch ports involve at least one matrix kernel (GEMM/WMMA/MFMA).
The matrix doc lists the operand shapes, builtin names, and lane
layouts for every arch hipfire currently knows about. **The single
biggest pitfall is assuming an `#ifdef` macro swap of the builtin
name is enough — it isn't, because operand vector lengths and
per-lane K-packing differ between archs.**

### 2. Check `dispatch.rs` for the existing arch-conditional sites

Every arch-aware GEMM dispatch in `crates/rdna-compute/src/dispatch.rs`
has the shape:

```rust
if has_<feature>(&self.arch) {
    return self.<kernel>_<feature>(...);
}
if has_<fallback_feature>(&self.arch) {
    return self.<kernel>_<fallback_feature>(...);
}
return self.<kernel>_baseline(...);
```

Don't `||` your new arch onto the existing predicate — author a new
predicate function (e.g. `has_wmma_f16_gfx12`) and add a NEW dispatch
branch above the fallback. Keep gfx11 untouched. Empirical reason
in the next section.

### 3. The "predicate-vs-inline" gfx11 perf trap

Earlier this session (commit `a048544`, reverted in `1f3bad3`)
replacing six inline

```rust
if self.arch.starts_with("gfx11") || self.arch.starts_with("gfx12") {
```

calls with a single `has_wmma_f16(&self.arch)` predicate (returning
`arch.starts_with("gfx11")` for gfx11) caused a measured 50% prefill
regression on gfx1100 — even though the predicate evaluates to the
same `true` on gfx1100. Mechanism is **not yet root-caused**; could
be inlining / register-allocator interaction with the dispatch
function's hot loop.

Until that's diagnosed: **do not factor existing inline arch checks
into helper functions**. Add new arch branches *above* the existing
inline check, e.g.:

```rust
if self.arch.starts_with("gfx12") {
    return self.gemm_<x>_wmma_gfx12(...);
}
if self.arch.starts_with("gfx11") {
    return self.gemm_<x>_wmma(...);  // existing gfx11 path, unchanged
}
```

This is verbose but byte-identical to the old codegen for the gfx11
host. Once we figure out why predicate-vs-inline matters, we can
refactor.

### 4. Author the new arch's kernel(s) as separate `.hip` files

Naming convention: `<existing_kernel_name>_gfx12.hip` (or
`_gfx1201.hip` if the variant is sub-arch-specific). Examples in
the existing tree:

- `kernels/src/gemv_hfq4g256.gfx1030.v1.hip` — gfx1030 RDNA2 variant
- `kernels/src/gemm_hfq4g256_residual_wmma_k4.hip` — gfx11 K4 WMMA

Single-file `#ifdef __gfx12__` is fine *only* when:
- The operand types are identical across archs (rare for WMMA/MFMA)
- The lane layout is identical (rare)
- The tuning constants are identical (rare)

For WMMA in particular, **the gfx11 → gfx12 port is NOT a single-file
ifdef**; operand vector lengths differ (`<16 x fp16>` vs `<8 x fp16>`)
and per-lane K-packing differs. Use a separate file.

### 5. Wire the include + dispatch

In `crates/rdna-compute/src/kernels.rs`:

```rust
pub const GEMM_X_WMMA_GFX12_SRC: &str = include_str!(
    "../../../kernels/src/gemm_x_wmma_gfx12.hip"
);
```

In `crates/rdna-compute/src/dispatch.rs`, add the dispatch branch
ABOVE the existing gfx11 inline check (per step 3).

### 6. Validate against all three gates (see `validation.md`)

A new arch port is merge-ready ONLY when:

1. **Channel-test passes** on real hardware (the contributor's
   target arch). This is correctness — `cargo run --release -p
   engine --example test_kernels` (or the QA variant) emits "OK"
   for every dispatched kernel on the new arch.
2. **Coherence-gate passes** (`./scripts/coherence-gate.sh`). No
   panics, no zero-tokens, no timeouts on the canonical
   small-prompt battery.
3. **Speed-gate passes** on the regression-baseline arch
   (`./scripts/speed-gate.sh --fast`). The new code path **cannot
   regress gfx1100** (or whichever arch the baseline lives on).

If you don't have hardware for the target arch, you cannot merge
— flag it in the PR and find a contributor with hardware (see
`contributor-onboarding.md` and #45 watchers).

## Quick reference

- WMMA matrix → `wmma-matrix.md`
- Validation procedure → `validation.md`
- Contributing without privileged repo access → `contributor-onboarding.md`
- Memory pin on this topic: `memory/project_wmma_correctness_fix.md`
  (the gfx11 `acc[j] = C[2*j + (tid>>4)][tid & 15]` mapping was
  silently wrong for 6 weeks before being caught — assume any
  per-lane mapping for a new arch is wrong until proven by channel-
  test on hardware).

## Known traps

| Trap | Symptom | Memory |
|---|---|---|
| WMMA C-mapping wrong | All-WMMA models emit garbage / fail correctness | `project_wmma_correctness_fix.md` |
| Removing "dead" WMMA kernels | Per-cycle GEMM cost ~2× on dispatch path that secretly uses it | `project_27b_dflash_perf_analysis_2026_04_22.md` (PR #32 → 9a2c667 recovery) |
| Bypassing speed-gate | Local-env regression masked by `--no-verify` lands on master | feedback (this session, commit `a048544` → reverted in `1f3bad3`) |
| Predicate-vs-inline arch check | 50% prefill regression on gfx11 from a "no-op" refactor | This session, root cause not yet found |
| Greedy degenerate decode | "Engine bug" smoke tests halt; turns out `--temp 0` + `<think>` exhaust max_tokens | `feedback_quality_gate_baselines_degenerate.md` |
| Firmware shadowing | `/lib/firmware/updates/amdgpu` overrides kernel firmware → SMU IF mismatch → 50% prefill drop, looks like code regression | `feedback_firmware_shadowing_perf_trap.md` |

## Skill discoverability

This skill lives at `.skills/hipfire-arch-port/`. Triggers in
`skill.json` cover the obvious phrases. Future agents asking
"how do I support gfx1XYZ?" should land here directly.
