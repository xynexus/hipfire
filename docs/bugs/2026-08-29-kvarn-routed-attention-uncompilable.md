# The routed KVarN attention kernel could not compile at all

Status: found and **FIXED** 2026-08-29, master `0c9e3d252`, nix1 (`gfx1103`).
Introduced by `e586262b8` on 2026-08-25 — live on master for three days.

## Symptom

Any code path reaching `attention_kvarn_routed_batched` dies at runtime with a
`hipcc` failure, not a wrong answer:

```
hipcc compilation failed for attention_kvarn_routed_batched:
  attention_kvarn_routed_batched.hip:46:5: error: expected ')'
     46 |     const int h = blockIdx.x;
  attention_kvarn_routed_batched.hip:27:58: note: to match this '('
     27 | extern "C" __global__ void attention_kvarn_routed_batched(
  ... 20 errors generated when compiling for gfx1103.
```

Found by running `cargo run --release -p hipfire-rdna --example parity_kvarn_routed`,
which panicked before measuring anything.

## Cause

`e586262b8` ("feat(kvarn): kv_window_precision config field + convert the last
window consumers") appended a `window_f16` parameter and put the closing `) {` of
the parameter list **inside the new line comment**:

```c
    int group
,
    int window_f16                                  // 1 = window holds f16, 0 = f32) {
    const int h = blockIdx.x;
```

`git blame` puts lines 44–45 on that commit and everything around them on the
original `e2df1bbfa`. The parameter list is never closed, so the translation unit
is invalid and the kernel cannot be built for any architecture.

## Fix

```c
    int group,
    // 1 = the window holds f16, 0 = f32.
    int window_f16
) {
```

After it, the kernel compiles and its own parity example passes:

```
parity_kvarn_routed on gfx1103: routed-vs-single-session max-abs-err=3.73e-8
  (rows->sessions [2, 0, 1]) -> PASS
```

## Why nothing caught it

**No gate compiles the kernels.** `tests/no-gpu-ci.sh` never invokes `hipcc`, and
kernels are compiled lazily at runtime on first use, so a syntax error surfaces
only when something actually dispatches that kernel — on a machine whose
`~/.hipfire/kernels/<arch>` cache does not already hold a stale `.hsaco`.

`scripts/compile-kernels.sh` exists and does exactly the check that would have
caught this, but nothing runs it. A sweep of all 653 kernels for `gfx1103` found
**this as the only genuine syntax error**; the other 10 failures are legitimate
arch-gating (`*_gfx12_bt` WMMA, `*_w64` wave64 kernels, and `gemv_mq8g256`, which
needs `dot1-insts` that gfx1103 lacks).

Adding a per-arch compile sweep to CI would be cheap next to what it protects —
though it needs an allowlist for the arch-gated kernels, or it reports 10 false
failures on every host.

## Note on the two other bugs in this kernel

This defect **masked** the two correctness bugs filed against the same kernel the
same day — they were unreachable while it could not build:

- [`kvarn-routed-attention-4bit-stride`](2026-08-29-kvarn-routed-attention-4bit-stride.md)
  — records assumed 4-bit regardless of the cache's `kvarn_bits`.
- [`kvarn-routed-prefill-window-wrap`](2026-08-29-kvarn-routed-prefill-window-wrap.md)
  — rows in a wrapped block attended to future tokens.

Both are now fixed as well, and the parity examples pass unchanged at
`max-abs-err=3.73e-8` after all three changes.
