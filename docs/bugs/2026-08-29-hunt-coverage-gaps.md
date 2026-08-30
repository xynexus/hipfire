# What the bug hunts did NOT cover — ranked

Master `0c9e3d252`. Written by the completeness critic that was planned for wave 1,
died with that run's session limit, and finally ran after both waves finished. All
12 planned dimensions have now been searched; this is about what the *method*
cannot reach, not what was skipped.

## High value

**1. Cross-arch execution — the fleet has the hardware and nobody ran on it.**
Dispatch carries 349 `gfx906`, 128 `gfx942`, 72 `gfx1201`, 62 `gfx1200`, 46
`gfx1150` arch-conditional references and **521 wave64/is_cdna branch sites**. On
nix1 (`gfx1103`) not one of them ever executes. The fix costs no authoring at all:
run the gates that already exist — `tiny-affected-gate.sh`,
`fixture-golden-gate.sh`, `coherence-gate.sh`, `opus-compact-gate.sh` — on halo
(gfx1151) and medusa (gfx906 MI50 + gfx1201 W7800).

**2. HFQ container versioning — the same class, structurally replicated.**
`compose_hfq` mixing packed offsets with expanded lengths and the recorded "HFQ v2
embedded-offset" bug are the *same* defect shape. **20 files parse `b"HFQM"`; 12
have no `version >= 2` branch**, and 6 writers still emit v1. Candidates:
`hipfire-serving-core/src/load.rs`, `hipfire-lora-hfq`, `hipfire-evidence`,
`hipfire-train/src/hfq_patch.rs`, `hipfire-coexistence/src/induction/orchestrate.rs`,
and several `hipfire-quantize/src/tools/*`.

**3. Kernel sibling divergence — 46 mechanically identifiable pairs.**
The most common confirmed shape across both waves was *fixed one path, not its
sibling*, and `kernels/src` holds 46 `X` / `X_batched` pairs. Diff only the
load-bearing lines: the causal/window mask expression, `n_valid`/`n_kv` bounds, LDS
array dimensions **and the constant sizing them**, and the group/bit-width constant.
Both KVarN bugs and the SWA LDS question were exactly this.

**4. Launch-argument TYPE and ORDER — count is now covered, types are not.**
`crates/hipfire-rdna/src/kernel_arity.rs` catches the arg-*count* class that bit
twice, but by its own doc it checks count only and skips ambiguous sites. A site
passing the right *number* of arguments in the wrong order or the wrong width is
still invisible. Extend it to match declared `__global__` parameter names against
the Rust identifiers, and point it at the launch sites it does not scan
(hipfire-runtime 10, hsa-bridge 15, hip-bridge 9, zaya 2, deepseek4 1).

**5. Large subsystems that produced zero findings.** By LOC the un-hunted mass is
`hipfire-xdna` (39.5k), `hipfire-train` (32k), `redline` (6k), plus
`hipfire-scheduler`, `-steer`, `-state`, `-monitor`, `-kvquant`, `-vision-cache`,
`-hneurons`.

## Medium

- **30 `unsafe impl Send`/`Sync`** (hip-bridge alone has 10) — each an unproven
  claim that a raw HIP handle is safe to move across threads. A list of 30
  justifications is bounded; a race hunt without it is not.
- **Security boundary** — `hipfire-priv-helper` itself is small and defensive; the
  risk is *upstream*, in how `doctor.rs:1488` resolves the helper binary and
  whether `unsafe_to_elevate()` rejects a writable or non-root-owned path on every
  branch. Plus constant-time comparison and token revocation in `hipfire-auth`.
- **Evidence integrity — the oracles were never audited.** *(KLD refs: DONE
  2026-08-29 — all selftested, 8 damaged files / 18.4 GB deleted, one healthy ref
  kept, one unverifiable ref kept and flagged. See BUGS.md.)* The audit found the
  original report was wrong about both the path and the scope, so the remaining
  advice stands with more force: **run the instrument's self-test before trusting
  any oracle**, and do it per-artifact rather than per-label. Fixture baselines
  under `tests/tiny-*-baselines.txt` have NOT had the equivalent audit — re-derive
  the stale ones, do not re-record them.
- **GPU allocation lifetime** — 21 `hipFree` sites against 10 `impl Drop`. An
  allocation with a `?` between it and its free is a leak the compiler cannot flag.
  One Drop guard per type beats one free per exit; the N-th exit is the one missed.
- **Numerical drift needing a real model.** Every confirmed quant finding so far is
  discrete and unit-testable. Small systematic drift — a wrong-but-plausible scale,
  an accumulation order — is invisible to both static lenses and tiny fixtures.
  Needs real-model KLD sweeps on halo against a *validated* ref.

## Correction to an earlier claim of mine

I reported that `hipfire-cli`'s 92 unit tests "have never run anywhere". The
narrow claim is right — the crate is bin-only, so `cargo test --lib --workspace`
selects zero targets from it — but it should not be generalized. **`.github/workflows/ci.yml`
does run `cargo test --lib --workspace --locked` on every push and PR**, so the
~1900 unit tests across 40+ *library* crates are not orphaned;
`tests/no-gpu-ci.sh`'s 13 named crates are a local subset, not the whole story.
Only bin-only crates fall through that gap. The critic's advice: flip the CI test
job to `--all-targets` and do one clippy cleanup so the advisory lint gate can be
promoted — but do not spend a hunt wave on "tests nobody runs", that lens is spent.

## Mechanical sweeps that would beat more LLM finders

1. Diff the 46 `X`/`X_batched` kernel pairs on the four load-bearing line classes.
2. Grep the 20 `b"HFQM"` parsers for a `version >= 2` branch; the 12 without one
   are the list. Same for the other self-defined containers with their own
   MAGIC+VERSION (`hipfire-kld/src/{archive,hfkseq}.rs`, `train/src/checkpoint.rs`).
3. **Duplicated-constant and duplicated-helper sweep** — `HFQ_MAGIC`/`HFQ_VERSION`
   is redeclared in 8 files *with the value disagreeing*; `BUNDLE_TRAILER_MAGIC` in
   2; `ELF_MAGIC` in 2. Every copied constant is a sibling that can drift. Collapse
   into `hipfire-quant-format` and let the compiler enforce agreement.
   The same applies to copied *helpers*: `pid_alive` is implemented twice in one
   crate — `hipfire-cli/src/commands/lock.rs:145` and
   `hipfire-cli/src/commands/daemon.rs:575` — which is the identical shape, found
   incidentally while fixing the `serve.pid` bug. Grep for duplicate `fn` bodies
   across sibling modules, not just duplicate constants.
4. Extend `kernel_arity.rs` from count to argument names, and to the other crates.
5. Run the existing gates on halo and medusa — zero authoring cost, executes 521
   wave64 branch sites that have never run.
6. **Arch-branch orphan sweep** — only 3 `.gfx12.hip` files exist against 134
   `gfx1200`/`gfx1201` dispatch references. Either the generic body is correct
   there (unverified) or a branch selects a kernel that does not exist for the arch.
7. Allocate-without-Drop sweep (21 frees vs 10 Drops).
8. Audit the 30 `unsafe impl Send`/`Sync` against actual cross-thread use.
9. `kldref_selftest` over `/srv/hipfire/kldrefs` — do this first.
