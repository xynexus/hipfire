# rocprofv3 Coverage Auditing

## The Hidden Lever Problem

In the 2026-05-19 A3B prefill investigation, rocprofv3 showed `gemm_q8_0_batched`
consuming 65% of prefill GPU time (~348ms, 640 calls) while the internal
`HIPFIRE_PROFILE` summed to only ~85ms. The kernel had no `profile::begin_timer`
wrapper. The internal profile was missing two-thirds of the actual work.

**Pattern recurred on 2026-05-22**: rocprofv3 showed `gemm_hfq4g128` at 70% of
PARO prefill GPU time on shisa-A3B-PARO/gfx1201, completely invisible to the
internal profile. Half a session was burned analyzing a 45% kernel that was
actually 11% of GPU time, missing the actual 70% bottleneck — which once found
yielded a +250% prefill win in under an hour.

## How to Run a Coverage Audit (automatic — preferred)

**As of 2026-05-22, `HIPFIRE_PROFILE=1` auto-execs under rocprofv3.** No more
manual `scripts/rocprof-wrap.sh` invocation required. The bench binary
detects the env var, self-execs under `rocprofv3 --kernel-trace --stats`,
sets `HIPFIRE_ROCPROF_CSV` for the rerun, and prints both the internal
profile AND the rocprof cross-check report.

```bash
# This is now the canonical way to profile:
HIPFIRE_PROFILE=1 \
  ./target/release/examples/bench_qwen35_mq4 model.hfq \
    --prefill 256 --gen 1
```

Output now includes:
1. `[HIPFIRE_PROFILE] forced rocprofv3 attribution → /tmp/hipfire-profile-PID-TS/trace_kernel_stats.csv`
2. Internal `=== PROFILE ===` block (serialized per-kernel timing)
3. `=== ROCPROF COVERAGE (XX%) ===` cross-check block (HSA-queue-level truth)
4. **Blindspot lines** for kernels rocprof saw but internal missed — fix
   these by adding `profile::begin_timer` wrappers at their dispatch sites.

**Opt-out** (e.g. rocprofv3 unavailable in CI / broken):
```bash
HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_NO_ROCPROF=1 \
  ./target/release/examples/bench_qwen35_mq4 ...
```
Prints a loud warning that hidden lever kernels may be invisible.

**Recursion guard**: `HIPFIRE_ROCPROF_CHILD=1` is set on the self-exec child;
the bench detects this and runs normally without re-execing.

**Implementation**: `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:
maybe_self_exec_under_rocprof()`. Other benches inherit this pattern by
calling the same helper (TODO: refactor into shared module under
`hipfire-runtime` so any bench picks it up automatically).

## How to Run a Coverage Audit (manual — legacy / non-bench binaries)

For binaries that don't yet have the auto-exec guard (most non-bench
examples), or when you need a separate output directory:

```bash
mkdir -p /tmp/cov-run
scripts/rocprof-wrap.sh /tmp/cov-run -- \
    env HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_NO_ROCPROF=1 \
    ./target/release/examples/<bench> model.hfq \
        --prefill 256 --gen 1 \
        --emit-atlas /tmp/cov-run/atlas.jsonl

scripts/coverage-audit.py \
    --internal /tmp/cov-run/atlas.jsonl \
    --rocprof  /tmp/cov-run/trace_kernel_stats.csv
```

Set `HIPFIRE_PROFILE_NO_ROCPROF=1` to disable the auto-exec since
`rocprof-wrap.sh` is doing the work explicitly. `coverage-audit.py` runs
the same cross-check logic independently as a redundant Python-side audit.

## How to Read the Report

**Coverage %**: fraction of rocprofv3 GPU time that the internal profile
accounts for. A kernel is "covered" if its internal alias (e.g.
`"gemm_q8_0_batched"`) appears as a substring of the rocprof mangled symbol.

**Top kernels**: ranked by rocprof duration -- the ground-truth hot path.

**Blindspots**: kernels rocprof saw but the internal profile missed entirely.
Each blindspot needs a `profile::begin_timer` wrapper at its dispatch site.

## When to Escalate

| Coverage | Action |
|----------|--------|
| >= 90% | Internal profile is reliable. No action needed. |
| 75-90% | At least one significant kernel is un-tracked. Investigate blindspots. |
| < 75% | Almost certainly a new kernel was added without a timer. Fix immediately. |

After any kernel dispatch addition (new kernel launch site, new quant format,
new architecture port), run a quick coverage audit to confirm the timer
wrapper was included.

## Implementation References

- `crates/rdna-compute/src/profile_rocprof.rs` -- Rust: CSV parser, coverage
  computation, `stop_with_rocprof`, inline `#[cfg(test)]` tests
- `crates/hipfire-atlas/src/profile_report.rs` -- Atlas: `AtlasProfileReport`
  type + `AtlasRow::set_profile_report` serializer
- `scripts/rocprof-wrap.sh` -- shell wrapper that invokes rocprofv3 and sets
  `HIPFIRE_ROCPROF_CSV` for the child bench process
- `scripts/coverage-audit.py` -- standalone Python audit (redundant with Rust,
  usable without rebuilding the bench binary)
- `scripts/kernel_atlas.py` -- `parse_rocprof_coverage_section` +
  `annotate_rocprof_coverage` helpers; collect-ar surfaces the coverage
  metrics in the atlas rows and tags BLINDSPOT rows for downstream filtering.
