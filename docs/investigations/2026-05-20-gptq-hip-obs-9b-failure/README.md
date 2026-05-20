# GPTQ-HIP OBS Phase 2 fails on every K>=4096 tensor (9B mix-corpus run)

**Date:** 2026-05-20
**Branch:** `feat/tier1-phase3-moe-integration` (worktree `tier1-phase3-integration`)
**Run:** 9B Tier 1 mix-corpus calibration on MI300x droplet
**Bugs:** two independent failures in the Tier 1 GPTQ-HIP pipeline diagnosed
empirically off the same end-to-end run.

The run setup:

* Model: Qwen3.5-9B (HF BF16) on MI300x droplet
* Calibration corpus: `calibration-mix-v1` (chat/code/prose/tool-call mix)
* Imatrix file: `/workspace/qwen3.5-9b.tier1.mix.imatrix.gguf` (498 tensors, all
  HF-canonical names — `model.language_model.layers.N.<slot>.weight.in_sum2`)
* Hessian file: `/workspace/qwen3.5-9b.tier1.mix.hessian.bin` (33.8 GB, 248
  entries F32, HFHS format)
* Quantize flags: `HIPFIRE_GPTQ_HIP_OBS=1` (Phase 2 GPU OBS path enabled)
* Logs: `/workspace/9b.mix.{imatrix,hessian,quant,eval}.log`

Both bugs are silently destructive:

* Bug 1 turns 248 GPTQ-target tensors into plain MQ4 (no GPTQ at all) — the
  most expensive lever in the Tier 1 pipeline becomes a no-op.
* Bug 2 turns AWQ into an identity-scale no-op for every HF-named imatrix
  file, including every hipfire-native GGUF + IMQ emit path.

Together they explain why the 9B mix-corpus calibration produced no
measurable downstream KLD improvement over plain MQ4 baseline.


## Bug 1 — GPU OBS Phase 2 fails on 100% of K>=4096 tensors with misleading `diag mean=0`

### Empirical evidence

All 248 GPTQ-target tensors fell back to plain MQ4 with the same warning:

```
warning: GPTQ failed for <name>:
  Cholesky of K=4096 Hessian failed even at damp=1.175805e-2 (diag mean=0.000000e0);
  skip GPTQ for this tensor; falling back to plain MQ4
```

The `diag mean=0.000000e0` is misleading. The actual Hessian for
`model.language_model.layers.23.self_attn.q_proj` (K=4096) is healthy:

* 4096 / 4096 diagonal entries non-zero
* mean = 1.1758
* sum = 4816
* max = 653.71

Confirming the Hessian was loaded correctly: the reported
`damp=1.175805e-2` matches `initial_damp(0.01) * actual_diag_mean(1.1758)`
exactly — i.e. the formula `let mut damp = initial_damp * diag_mean;` at
`gptq_hip.rs:243` ran with the real diag_mean, and only the final error
construction overwrites `diag_mean` with the hardcoded `0.0` sentinel.

### Root cause

The error printout's `diag_mean=0.0` comes from two hardcoded `diag_mean: 0.0`
fields in error returns inside `gptq_column_sequential_hip`
(`crates/hipfire-quantize/src/gptq_hip.rs`):

* Line 1175 — the K-time D2H diagonal-copy loop returns
  `CholeskyError::SingularEvenWithMaxDamp { max_damp: effective_damp, k, diag_mean: 0.0 }`.
* Line 1189 — the `to_chol_err` closure used by every subsequent device-
  allocation / H2D failure in the same function also hardcodes `diag_mean: 0.0`.

The Cholesky function `compute_damped_inv_cholesky_upper_hip_keep`
(`gptq_hip.rs:363`) DOES carry the real diag_mean in every error path
(line 387, 400, 435, 440, 445, 454, 462, 467, 475, 481, 486, 491). Those
errors bubble up unchanged via `?` at line 1151. So if the failure were
inside `_keep`, the warning would print the real diag_mean.

Since the warning prints `diag_mean=0.0`, the failure cannot be in `_keep`.
By construction it must be in `gptq_column_sequential_hip` AFTER the
`?` at line 1151 — i.e. in steps 3-6 of the function.

The most likely suspect is the K-time single-double `hipMemcpy`
diagonal-copy loop at gptq_hip.rs:1163-1180. It issues K = 4096 (and up
to K = 12288 for mlp.down_proj) synchronous 8-byte `hipMemcpy` calls. At
~5 µs/call latency that's ~20 ms — 60 ms for K=12288 — and any one of
them returning non-SUCCESS aborts the tensor. Even if the failure is
intermittent (e.g. backpressure on a queue), the K-fold amplification
makes the per-call failure probability O(1) at K=4096.

This is a known v1 shortcut. The block comment at gptq_hip.rs:1158-1160
acknowledges it:

> At ~5µs/copy that's 60ms — acceptable for Phase 2 v1; future work can
> fuse to one chunked copy.

### Why Phase D (GPU Cholesky only, no OBS) succeeded but Phase 2 OBS fails

Phase D was validated on 9B per task #138. It does NOT run the K-time
D2H diagonal loop — it copies the full U back to host with ONE
`hipMemcpy` of `K * K * 8` bytes. The block-loop OBS is what's new in
Phase 2, and the K-time D2H loop is the v1 shortcut introduced inside
the block-loop orchestrator.

Phase D parity test (gptq_hip.rs:`parity_gate_k4096`) does NOT exercise
the OBS column path — it only validates the upper-Cholesky math. So
the K-time D2H loop has zero unit-test coverage at K=4096, and it
broke the moment it was wired into the Phase 2 column-sequential
orchestrator.

### Fix

Replace the K-time D2H diagonal copy with ONE bulk D2H of the full
K x K F64 buffer, then extract the diagonal host-side. For K=4096 the
buffer is 128 MB at ~50 GB/s PCIe = 2.5 ms (vs ~20 ms K-time). For
K=12288 it's 1.2 GB at ~50 GB/s = 24 ms (vs ~60 ms K-time). Faster AND
issues one syscall per tensor instead of K.

Also replace EVERY hardcoded `diag_mean: 0.0` in error returns of
`gptq_column_sequential_hip` with a real diag_mean computed host-side
from `h_target` BEFORE the function returns. Compute once at function
entry; reuse in `to_chol_err` and the bulk-D2H error path.

```rust
let diag_mean_for_err: f64 =
    (0..k_dim).map(|i| h_target[(i, i)]).sum::<f64>() / k_dim as f64;
```


## Bug 2 — `imatrix_weights_for()` silently misses HF-named imatrix files

### Empirical evidence

The 9B mix imatrix file at `/workspace/qwen3.5-9b.tier1.mix.imatrix.gguf`
contains 498 tensors. All use HF-canonical names (e.g.
`model.language_model.layers.0.linear_attn.in_proj_a.weight.in_sum2`).
Zero use the legacy `blk.*` ggml-style names.

`imatrix_weights_for(safetensors_name)` at
`crates/hipfire-quantize/src/main.rs:2887-2891` does:

```rust
fn imatrix_weights_for(safetensors_name: &str) -> Option<&'static [f32]> {
    let im = IMATRIX.get()?;
    let ggml_name = safetensors_to_ggml_name(safetensors_name)?;
    im.get(&ggml_name).map(|v| v.as_slice())
}
```

The lookup ALWAYS converts safetensors → ggml-style names. But the
IMATRIX HashMap is populated by `load_imatrix` (main.rs:2709-2879) from
whatever names the FILE uses. The supported sources are:

* IMQ format (hipfire-native): keys = HF-canonical names
* GGUF imatrix (legacy llama.cpp): keys = ggml-style names
* GGUF imatrix (hipfire-native emit, e.g. `collect_imatrix`): keys =
  HF-canonical names

For HF-named files (case 1 and case 3), the HashMap key is the HF name,
the lookup builds the ggml-converted key, the lookup misses, and AWQ
silently no-ops with identity scales.

### Root cause

`imatrix_weights_for` was written for the GGUF/legacy llama.cpp case
only. The HF-canonical key path was added later (IMQ format,
hipfire-native GGUF emit) but the lookup function was never updated to
match. The `load_imatrix` dispatch correctly handles both file formats
on the writer side; the reader-side lookup was the missing half.

### Fix

Try the HF-canonical name as-is first (the IMQ + hipfire-native GGUF
case), then fall back to the ggml-converted name (legacy llama.cpp
case):

```rust
fn imatrix_weights_for(safetensors_name: &str) -> Option<&'static [f32]> {
    let im = IMATRIX.get()?;
    // Try HF-canonical first (hipfire-native GGUF + IMQ format).
    if let Some(v) = im.get(safetensors_name) {
        return Some(v.as_slice());
    }
    // Fall back to ggml-style (legacy llama-imatrix output).
    let ggml_name = safetensors_to_ggml_name(safetensors_name)?;
    im.get(&ggml_name).map(|v| v.as_slice())
}
```


## Suggested follow-ups

* **CPU vs GPU OBS parity test at K=4096.** The existing `parity_gate_k4096`
  test in gptq_hip.rs only covers the Cholesky math, not the full column-
  sequential OBS path. Add a parity test that runs CPU `gptq_column_sequential`
  and GPU `gptq_column_sequential_hip` against the same Hessian + weights
  at K=4096 and checks element-wise max abs diff. Without that, regressions
  in the bulk-D2H + block-loop orchestrator stay invisible until a 9B-scale
  end-to-end run surfaces them.

* **Re-run 9B mix-corpus calibration with both fixes shipped.** The 248
  tensors that fell back to plain MQ4 in this run need to be re-quantized
  through the GPU OBS path; only then will downstream KLD measurements
  reflect the Tier 1 pipeline's actual capability.

* **Add a `[gptq-imatrix-coverage]` log line.** Emit `(hit / total)` count
  alongside the existing `[gptq-clamp]` diagnostic so the next run with a
  silent imatrix-lookup mismatch fails loud at the first quantize call,
  not after the eval log shows no KLD movement.

* **Audit other hardcoded `diag_mean: 0.0` sites.** Search the wider codebase
  for places that construct `CholeskyError::SingularEvenWithMaxDamp` with a
  sentinel diag_mean — every one will produce the same misleading warning
  the next time it fires.
