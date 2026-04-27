# Contributor onboarding for arch ports

Hipfire is a small team (effectively one maintainer + occasional
help) with a fixed set of test hardware (currently gfx1100 / gfx1030
on the local bench, gfx942 / MI300X via remote rentals, gfx1010 on
shelf). New arch support **only happens** when someone with the
hardware contributes the port — there's no emulator path, no
maintainer with a 9070 XT in their drawer.

If you're reading this because you have hardware that hipfire
doesn't yet support, this is the doc that gets you from "I have a
GPU" to "I landed a port".

## What you need

- The target GPU, accessible from a Linux box with ROCm 7.0+
  installed (`rocminfo` should list your arch — `gfx1201`, `gfx1200`,
  `gfx1151`, etc.).
- Time to run a few hour-long benches. The validation gates aren't
  fast.
- A GitHub account and a hipfire fork. Comfort with `git`,
  `cargo`, and `bash`.
- Patience. The first WMMA port took ~2 weeks of work plus a
  6-week silent-corruption bug to discover and fix.

## What you do NOT need

- Privileged repo access (no commit bit required — this is fork +
  PR).
- A high-bandwidth conversation with the maintainer. The skill in
  `.skills/hipfire-arch-port/` and the codebase patterns are
  enough to do a port end-to-end without supervision.
- An LLM agent — but **agent assistance is genuinely helpful**.
  Most of this skill is written assuming you're collaborating with
  one (Claude Code, Cursor, etc.). See "Working with an agent"
  below.

## The 7-step contributor workflow

### 1. Reproduce the issue (or pick a port target)

If responding to a bug report (e.g. issue #54):

```bash
git clone https://github.com/Kaden-Schutt/hipfire
cd hipfire
cargo build --release --features deltanet -p engine --example daemon
./target/release/examples/daemon
# In another shell:
hipfire run qwen3.5:0.8b "What is the capital of France?"
```

Confirm the failure mode the reporter described. If you can't
reproduce, file a comment on the issue with what you saw — the
debug surface narrows fast.

If picking a port target proactively (e.g. you have a 9070 XT and
want full perf, not just the safe fallback):

```bash
rocminfo | grep gfx
# If your arch isn't in dispatch.rs's WMMA / dot2 dispatch tree,
# this skill's `playbook.md` is the starting point.
```

### 2. Read the playbook + matrix

`.skills/hipfire-arch-port/playbook.md` end-to-end. Then
`.skills/hipfire-arch-port/wmma-matrix.md` for the operand-shape
table.

The most common arch-port mistake: assuming a single-file
`#ifdef __gfx12__` macro swap of the WMMA builtin name is enough.
It isn't — operand vector lengths differ between gfx11 and gfx12.

### 3. Pick a small kernel to port first

Don't start with `gemm_qkvza_hfq4g256_wmma` — that's a six-output
kernel with complex LDS staging. Start with something like
`gemm_qkv_hfq4g256_wmma` or even smaller. Get the lane-mapping
right on a kernel you can debug on your own hardware before
scaling.

### 4. Author the kernel(s) as separate `.hip` files

`kernels/src/<existing_name>_<arch>.hip`. Naming convention is
documented in `playbook.md`. Reference the gfx11 version side-by-
side and walk through:

- LDS staging (does the K-tile striding need to change? for gfx11→
  gfx12 — yes, K-packing per lane goes 16→8).
- WMMA call (swap builtin name AND adjust operand vector types).
- Output writeback (does the C-mapping change? assume yes, verify
  empirically).

### 5. Wire dispatch + add the channel-test case

`crates/rdna-compute/src/kernels.rs`: add `include_str!` for the
new kernel.

`crates/rdna-compute/src/dispatch.rs`: add an arch-conditional
branch ABOVE the existing inline check for the new arch. **Do
not** factor the existing inline check into a helper function —
that triggered a 50% regression on gfx1100 in this session for
mechanism-not-yet-known reasons. Just add a new `if` above:

```rust
// New arch port (gfx12, RDNA4):
if self.arch.starts_with("gfx12") {
    return self.gemm_<x>_wmma_gfx12(...);
}
// Existing gfx11 path (RDNA3) — unchanged byte-for-byte:
if self.arch.starts_with("gfx11") || self.arch.starts_with("gfx12") {
    return self.gemm_<x>_wmma(...);
}
```

(Yes the existing gfx11 inline check still mentions gfx12 — leaving
it alone is the safest path until the predicate-vs-inline mystery
is solved. The new branch above takes precedence.)

`crates/engine/examples/test_kernels.rs`: add a test case that
exercises your new kernel on the new arch.

### 6. Run all three gates locally

```bash
./scripts/coherence-gate.sh
./scripts/speed-gate.sh --fast
cargo run --release -p engine --example test_kernels
```

All three must pass. If channel-test fails, you've got a per-lane
mapping wrong — go back to step 4 and instrument with `eprintln!`s
of `(tid, output_index)` to derive the correct mapping.

### 7. Open a PR

PR template (recommended structure for arch ports):

```markdown
## What this is
Arch port: gfx<XYZ> (e.g. gfx1201 / 9070 XT).

## What it adds
- New kernels: `kernels/src/gemm_*_<arch>.hip` (list)
- Dispatch branches in `dispatch.rs` (line ranges)
- `kernels.rs` includes
- `test_kernels.rs` cases

## Hardware tested
- <Your GPU model> on <distro / kernel / ROCm version>
- channel-test: PASS/FAIL on each kernel
- coherence-gate: clean (attach report path)
- speed-gate --fast: passes on gfx<your-baseline-arch>

## What's NOT in this PR
- Things you deliberately scoped out (e.g. fp8 mixed-precision
  variants, MoE kernels). List them with rationale.

## Known limitations
- ...
```

The maintainer will review. Likely 1-2 rounds of feedback before
merge — most arch-port PRs end up needing per-lane mapping
adjustments based on the maintainer's reading of the lane layout.

## Working with an agent (Claude Code / Cursor / Codex)

This skill (the one you're reading) is **designed for agent
collaboration**. If you're working with Claude Code:

```
> Read .skills/hipfire-arch-port/ first. I want to port gfx1201
> WMMA. I have a 9070 XT to test on.
```

The agent will read the playbook + matrix + validation docs,
locate the existing gfx11 kernels, and propose a port plan. You
review the plan before any code is written, then iterate.

**Useful agent prompts:**

- "Walk me through the gfx11 → gfx12 LDS staging changes for
  `gemm_qkv_hfq4g256_wmma.hip`. Don't write code yet."
- "What's the C-mapping for gfx12 WMMA? Cite the ROCm header
  source."
- "I'm seeing channel-test fail with `expected 1.0, got 4.0` on
  output index 3. Derive the lane mapping that would produce
  this." (then paste the failure output)

**Important agent guardrails for this codebase:**

- The agent **cannot bypass the speed-gate with `--no-verify`**.
  This is enforced by repo policy. If the agent suggests
  `--no-verify`, push back — it's almost always wrong.
- The agent **must run `./scripts/coherence-gate.sh` after kernel
  changes**, not just claim "should be fine". The gate is fast
  (~2-4 min) and the false negatives from skipping it are
  silent corruption.
- The agent **must check git status before each commit** to make
  sure no stray test files / debug prints land.

## Communication

- Issue tracker: https://github.com/Kaden-Schutt/hipfire/issues
- Tag the maintainer (@Kaden-Schutt) when:
  - You've reproduced an issue and want a sanity check on direction
    before sinking time into a port.
  - The port is done and the PR is ready for review.
  - You've hit a wall (channel-test fails, gate breaks) and want
    a second set of eyes.

- Don't tag for:
  - Routine progress updates (post in the issue thread instead).
  - Pre-port reading questions (the skill should answer most of
    them; if not, that's a skill-improvement PR opportunity).

## What landed this session

- Issue #54 (9070 XT crash) was reported 2026-04-27 by an external
  user.
- A naive dispatch fallback was attempted (commit `a048544`) and
  reverted (`1f3bad3`) because:
  - It bypassed the speed-gate inappropriately.
  - The "no-op" predicate refactor regressed gfx1100 prefill 50%.
- This skill was authored to capture the lessons before the next
  port attempt.
- kmbandy (GitHub) volunteered to do the gfx1201 port with R9700
  hardware — see issue #45 comment for context.

## You're contributing into a small, opinionated codebase

That's not a complaint, that's an invitation. The maintainer cares
deeply about arch coverage and will work with you to get a port
landed. The flip side: code style / commit hygiene / test
discipline matter, and PRs that skip the gates or hand-wave the
correctness story don't get merged. The gates are not bureaucracy
— they exist because every one of them caught a real bug at some
point.

Welcome aboard.
