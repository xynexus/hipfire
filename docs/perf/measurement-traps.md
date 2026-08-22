# Measurement traps on this repo

Things that have silently produced wrong perf conclusions here. Each entry is a
trap that was actually hit, with the tell and the fix.

## `hipfire eval` caches, and the cache key does NOT include the environment

`hipfire eval` reuses cached rows across invocations and replays them verbatim —
including the original `started_utc` and `elapsed_ms`. The key covers the model,
the prompt, and `binary_hash`, so a **rebuild** correctly misses. **An env var
does not.**

That makes eval unsound by default for the most common perf A/B there is: same
binary, same prompt, one `HIPFIRE_*` flag flipped. Both arms return the first
arm's numbers, with no warning.

Observed 2026-08-22 while measuring the iu4x2 prefill route: four invocations
over several minutes all reported `prefill 15.5, ttft 15499.0` and the same
`started_utc`; a timed repeat finished in **7 s wall while reporting
`elapsed_ms: 59647`**. It nearly produced a "no effect" verdict on a change
worth +14%.

- **Tell:** identical numbers across an A/B, especially identical to the last
  decimal, and an unchanged `started_utc` between result files.
- **Fix:** `--force` (ignore cache hits) or `--regenerate` (delete and replace
  the matching rows) for any env-only comparison. With `--force`, the same A/B
  read 194.9/199.4 -> 220.3/227.6 tok/s.
- **Corollary:** identical numbers across an A/B are evidence of a cache hit,
  not of a null result. Check `started_utc` before believing a no-op.

## `include_str!` bakes kernel source into the Rust binary

Kernel `.hip` sources are `include_str!`'d into the dispatch crate. Deleting the
cached `.hsaco` under `~/.hipfire/kernels/<arch>/` only forces a re-JIT **of the
string already compiled into the binary**, so editing the `.hip` file and
re-running without `cargo build` measures the OLD kernel.

- **Tell:** an ablation that should have changed something measures identical.
- **Fix:** always `cargo build` the consuming binary after touching a `.hip`.
  Sanity-check a null ablation with an empty-kernel control — it should collapse
  to launch overhead (~0.7 us), not stay at the real kernel's time.

## Allocating in the GPU hot path reads as a broken kernel

A per-call `alloc_tensor`/`free_tensor` inside a GEMM wrapper is ~900
`hipMalloc`/`hipFree` per prefill chunk on a 64-layer model. It cost prefill
195.0 -> 132.2 tok/s while every microbench still reported the kernel 1.43x
faster than the path it replaced.

- **Tell:** microbenchmarks and end-to-end disagree in *direction*.
- **Fix:** persistent scratch (`ensure_*_scratch_batched`), sized on demand.

## `pkill -f hipfire-daemon` kills the shell running it

`-f` matches the full command line, which includes the script's own invocation.
Surfaces as a bare `exit 144` with no output. Use `pkill -x hipfire-daemon`, or
kill by explicit pid.

## Mean over a heavy tail

A kernel reported 10% slower by mean was 1.0012x by bottom-99% ratio — the mean
was carried by a handful of outlier launches. Compare distributions, or a
trimmed statistic, before calling a regression.
