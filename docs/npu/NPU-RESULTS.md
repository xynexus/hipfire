# NPU Kernel Results

## Strix Halo NPU (aie2p) — the ceiling is FEEDING, not compute

> **halo** box (RYZEN AI MAX+ 395, NPU Strix Halo **aie2p/npu2**, 4 compute rows
> × 8 cols = 32 cores). Investigation: 2026-07-04. Full detail + reproducible
> harness in `benchmarks/npu_gemm_tuning/` (`findings.md`, `tune.sh`).

**The Strix Halo NPU is a real, un-throttled 58 TOPS int8 — but the hard part is
feeding the cores, not the cores.** Evidence:

- **Hardware peak = 58 TOPS** (`hipfire-xdna` resource_info: `npu_tops_max=58`,
  `npu_clk_max=1800`). First-principles: 32 cores × 512 int8 MAC/mmul (8×8×8) ×
  1.8 GHz ≈ 59 TMAC. `crates/hipfire-xdna/examples/npu_info.rs` dumps it live.
- **NOT power/clock-throttled.** Under GEMM load, `default` pmode already boosts
  to the full 58-TOPS budget with the AIE compute clock maxed at 1800 MHz.
  `xrt-smi configure --pmode turbo` is a **confirmed no-op** (15.2 vs 15.7 TOPS).
- **Real GEMM caps at ~12–27% of peak — it's feed/overhead-bound.** A tuned int8
  matmul (mlir-aie `whole_array`) tops out at **15.7 TOPS (27%)**; every knob
  explored (output-tile size is the only lever and it's L1-capped at 64 KB/core;
  columns maxed; k/fifo_depth/OPT_PERF/pre-tiled-weights all no-op or marginal;
  the `mm.cc` microkernel is AMD-optimal). Throughput scales with **output-tile
  size** because it amortizes per-tile feed/sync overhead (DMA setup, C
  accumulator load/store, objectFIFO acquire/release, software-pipeline
  fill/drain) — the cores are **starved**, not saturated.
- **AMD's own shipped kernel confirms it.** Built DynamicDispatch from source and
  ran the production `mladf` int4 gemm on the NPU: flat **~7 TOPS** across LLM
  shapes (a memory-bound weight-quant *decode* kernel) — *below* our int8
  reference. No measured real GEMM — reference or production — approaches 58.

**Takeaway:** treat 58 TOPS as a theoretical ceiling; budget **~15–16 TOPS int8**
for a real Strix Halo GEMM. The lever for more is a better *dataflow* that keeps
the cores fed (larger effective tiles / less per-tile overhead), not compute,
power, columns, or datatype. Bottleneck = feeding the AIE array.

## Platform

- **Machine**: nix1
- **NPU**: NPU1 (XDNA / Ryzen AI, Phoenix silicon) — AIE2 / AIE-ML, 16 TOPS
- **Tile grid**: 4 compute columns × 4 core rows (column_width=4)
- **Driver**: amdxdna-dkms 7.0.0-rc1+git20260310.6b13cb8f4
- **XRT**: 2.25.0 (2026-06-01)
- **Date**: 2026-06-13

## SwiGLU (silu_mul_bf16) — Qwen3.5 dense FFN

Kernel: `tools/npu/silu_mul_bf16.cc`, AIE2 LUT-based tanh.
Computes `out[i] = silu(gate[i]) * up[i]` in BF16 across all 4 NPU columns.
Warmup: 20 iterations. Timed: 200 iterations.

| hidden_size | model  | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|-------------|--------|-----------|---------------|---------|---------|----------------|-----------|
| 8960        | 1.5B   | 52.5      | 201           | 198     | 302     | 247            | 0.27      |
| 18944       | 7B     | 111.0     | 216           | 209     | 359     | 262            | 0.53      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × hidden_size × 2 bytes / npu_mean.

### Observations

The ~190 µs floor is fixed dispatch/DMA overhead. The 7B size does 2.1× the
data in only 7% more time, confirming compute is not the bottleneck. The NPU
path is intended for pipelined use where this overhead is hidden by concurrent
GPU work, not for synchronous calls.

## Correctness (oracle test)

Reference: `silu(gate) * up` computed in float32, cast to bfloat16.  
Tolerance: atol=0.02, rtol=0.02. Max abs error ~0.047 (≈3 bfloat16 ULPs at 1.0,
consistent with LUT-based tanh rounding).

| hidden_size | max_abs_err | mean_abs_err | max_rel_err | result |
|-------------|-------------|--------------|-------------|--------|
| 8960        | 0.04688     | 0.00291      | 0.0234      | PASS   |
| 18944       | 0.04688     | 0.00290      | 0.0235      | PASS   |

## RMSNorm (rms_norm_weighted_bf16) — Qwen3.5 hidden norm

Kernel: `tools/npu/rms_norm_weighted_bf16.cc`, AIE2, single-tile full-row design.
Computes `out[i] = (x[i] / rms(x)) * weight[i]` in BF16.
Single-tile design: tile_size=hidden_size so the entire row lands in one AIE core for
the reduction pass. Uses `aie::invsqrt(vector<float, N>)` (hardware RSQRT).
Warmup: 20 iterations. Timed: 200 iterations.

| hidden_size | model | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|-------------|-------|-----------|---------------|---------|---------|----------------|-----------|
| 1536        | 1.5B  | 9.0       | 187           | 178     | 290     | 243            | 0.05      |
| 3584        | 7B    | 21.0      | 167           | 161     | 265     | 204            | 0.13      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × hidden_size × 2 bytes / npu_mean.

### Observations

The dispatch floor is ~160–190 µs, same as SwiGLU. BW is lower than SwiGLU because the
data footprint (3 × hidden_size ≈ 9–21 KB) is smaller than the FFN sizes. The single-tile
design uses only 1 of 4 NPU columns — this is unavoidable for a reduction operation where
all elements must be visible for the sum(x²) pass. Pipelined with GPU compute, the dispatch
overhead is hidden.

### Correctness (oracle test)

Reference: `(x / sqrt(mean(x²) + 1e-5)) * weight` in float32, cast to bfloat16.  
Tolerance: atol=0.02, rtol=0.02. Max abs error ~0.031 (≈2 bfloat16 ULPs at 1.0,
from bfloat16 broadcast of float inv_rms).

| hidden_size | max_abs_err | mean_abs_err | max_rel_err | result |
|-------------|-------------|--------------|-------------|--------|
| 1536        | 0.03125     | 0.00512      | 0.0193      | PASS   |
| 3584        | 0.03125     | 0.00471      | 0.0155      | PASS   |

## RoPE (rope_rotate_bf16) — Qwen3.5 rotary position embedding

Kernel: `tools/npu/rope_rotate_bf16.cc`, AIE2, single-tile half-split design.
Applies `x_rot = x*cos - y*sin, y_rot = y*cos + x*sin` in BF16 for dims [0, n_rot),
pass-through for dims [n_rot, head_dim). Half-split layout: x at [0, n_rot/2),
y at [n_rot/2, n_rot). Separate Q and K xclbins (tile_size=head_dim).
Warmup: 20 iterations. Timed: 200 iterations. Config: n_heads=8, n_kv_heads=2,
head_dim=256, n_rot=64 (Qwen3.5-1.5B dense).

| tensor | n_heads | total_elem | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|--------|---------|-----------|-----------|---------------|---------|---------|----------------|-----------|
| Q      | 8       | 2048      | 4.0       | 185           | 171     | 304     | 246            | 0.05      |
| K      | 2       | 512       | 1.0       | 166           | 161     | 255     | 211            | 0.01      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × total_elem × 2 bytes / wall_mean.

### Observations

Same ~160–190 µs dispatch floor as SwiGLU and RMSNorm. The Q tensor (8 heads × 256 = 2048 elements, 4 KiB) and K tensor (2 heads × 256 = 512 elements, 1 KiB) are both small; BW is dispatch-floor-dominated. The cs param (64 elements, 128 B) is acquired once per dispatch and reused for all head iterations, avoiding 8× or 2× redundant transfers. Single-tile design (one AIE column) since all heads are processed serially.

### Correctness (oracle test)

Reference: half-split float32 RoPE with `freq_base=500000` (Qwen3.5 theta), `pos=1`.
Tolerance: atol=0.02, rtol=0.02. max_abs ≤ 1 bfloat16 ULP (0.03125 at magnitude 1–2).

| tensor | max_abs_err | mean_abs_err | max_rel_err | result |
|--------|-------------|--------------|-------------|--------|
| Q      | 0.03125     | 0.00064      | 0.1034      | PASS   |
| K      | 0.01562     | 0.00057      | 0.0667      | PASS   |

## QK Head Norm (rms_norm_head_bf16) — per-head RMSNorm on Q and K

Kernel: `tools/npu/rms_norm_head_bf16.cc`, AIE2, single-tile design.
Applies `out[h][i] = (x[h][i] / rms(x[h])) * weight[i]` per head in BF16.
Shared weight `[head_dim]` is a tensor param acquired once for all head iterations.
Mirrors `gpu.rmsnorm_batched()` in the Qwen3.5 forward pass (runs after QKV projection, before RoPE).
Warmup: 20 iterations. Timed: 200 iterations. Config: n_heads=8, n_kv_heads=2, head_dim=256.

| tensor | n_heads | total_elem | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|--------|---------|-----------|-----------|---------------|---------|---------|----------------|-----------|
| Q      | 8       | 2048      | 4.0       | 187           | 177     | 291     | 235            | 0.05      |
| K      | 2       | 512       | 1.0       | 175           | 169     | 259     | 218            | 0.01      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 tensors × total_elem × 2 bytes / wall_mean.

### Observations

Same ~170–190 µs dispatch floor. The weight tensor param pattern (acquired once, reused across 8 or 2 head iterations) amortizes the weight DMA cost rather than paying it per head. Single-tile design — each head requires the full vector for the reduction pass.

### Correctness (oracle test)

Reference: per-head float32 RMSNorm, `eps=1e-5`, random weight in [0.5, 1.5].  
Tolerance: atol=0.02, rtol=0.02.

| tensor | max_abs_err | mean_abs_err | max_rel_err | result |
|--------|-------------|--------------|-------------|--------|
| Q      | 0.04688     | 0.00448      | 0.0199      | PASS   |
| K      | 0.06250     | 0.00449      | 0.0148      | PASS   |

## Attn Output Gate (sigmoid_mul_bf16) — Qwen3.5 attention gating

Kernel: `tools/npu/sigmoid_mul_bf16.cc`, AIE2 LUT-based tanh.
Computes `out[i] = sigmoid(gate[i]) * x[i]` across all 4 NPU columns.
Replaces `gpu.sigmoid_f32 + gpu.mul_f32` when `config.attn_output_gate=true`.
Warmup: 20 iterations. Timed: 200 iterations. Config: n_heads=8, head_dim=256, q_dim=2048.

| q_dim | n_heads | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|-------|---------|-----------|---------------|---------|---------|----------------|-----------|
| 2048  | 8       | 4.0       | 183           | 176     | 268     | 227            | 0.05      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).  
**wall time**: end-to-end per-call latency measured on the host.  
**BW**: effective memory bandwidth = 3 × q_dim × 2 bytes / wall_mean.

### Observations

Same ~183 µs dispatch floor. Uses 4 NPU columns (parallel), same as SwiGLU. Data footprint (3 × 4 KiB = 12 KiB) is small; BW-dominated by dispatch. One kernel per decode step (only present when `attn_output_gate=true`).

### Correctness (oracle test)

Reference: `sigmoid(gate) * x` in float32 where `sigmoid(x) = 1/(1+exp(-x))`.  
Tolerance: atol=0.02, rtol=0.02. max_abs=0.016 (≈1 ULP at magnitude ≤1, from LUT tanh rounding).

| q_dim | max_abs_err | mean_abs_err | max_rel_err | result |
|-------|-------------|--------------|-------------|--------|
| 2048  | 0.01562     | 0.00211      | 0.0216      | PASS   |

## Kernel 7: Softmax (`softmax_bf16`)

- **Date**: 2026-06-14
- **Status**: PASS (all ctx_len variants)
- **Config**: n_heads=8, Qwen3.5-1.5B dense, NPU1 (Phoenix/AIE2)
- **Algorithm**: 3-pass scalar poly-exp (from mlir-aie bf16_softmax.cc), + max-subtraction pre-pass
- **Exp method**: range-reduction `2^trunc(x*log2e) * 2^frac(x*log2e)`, degree-2 poly for fractional; clamped for underflow (`ix < -127 → 0`). Handles masked -inf positions correctly (produce exactly 0).
- **Note on LUT exp**: `getExpBf16` from lut_based_ops.h only handles positive inputs (LUT covers [0, 7.97]); negative inputs (all values after max-subtraction) give `exp(7.97)` via truncation — unusable here.

| ctx_len | max_abs  | npu_mean | wall_mean |
|---------|----------|----------|-----------|
| 64      | 0.00391  | 608 µs   | 672 µs    |
| 128     | 0.00269  | 1343 µs  | 1486 µs   |
| 256     | 0.00122  | 1881 µs  | 1953 µs   |
| 512     | 0.00073  | 3860 µs  | 4134 µs   |

**Dispatch floor: ~170 µs; compute scales ~7.4 µs/element (scalar loop).** Softmax is compute-heavy (exp + sum + normalize) and the scalar implementation dominates. Not NPU-competitive with GPU DFlash for typical decode contexts (DFlash fuses QK^T + softmax + AV in one kernel). Useful only for the non-DFlash fallback paths (Q8 FA, gqa4) at short context lengths (≤128) where standalone GPU softmax dispatch overhead dominates.

## Kernel 8: Fused HeadNorm + RoPE (`headnorm_rope_bf16`)

- **Date**: 2026-06-14 (built; awaiting hardware run)
- **Status**: pending first run
- **Config**: n_heads=8, n_kv_heads=2, head_dim=256, Qwen3.5-1.5B dense, NPU1 (Phoenix/AIE2)
- **Algorithm**: 3-pass vectorized (VEC=16):
  - Pass 1: `sum_sq = Σ x_i²`, `inv_rms = 1/sqrt(mean_sq + 1e-5)` via `aie::invsqrt`
  - Pass 2 (rotation region [0, n_rot)): normalize (x * inv_rms * weight) then rope-rotate (half-split layout)
  - Pass 3 (passthrough [n_rot, head_dim)): normalize only
- **Tensor param**: single packed buffer `[weight (head_dim), cs (n_rot)]` — avoids a new FFI signature
- **Dispatch savings**: replaces 4 separate dispatches (headnorm_q + rope_q + headnorm_k + rope_k)
  with 2 dispatches (Q and K), saving 2 × ~170 µs × 28 layers ≈ **9.5 ms/step**
- **Artifact**: `qwen35-headnorm-rope-{q,k}-{n_heads}h{head_dim}d.{xclbin,instr.bin}`

| tensor | n_heads | total_elem | data (KiB) | npu mean (µs) | npu p50 | npu p99 | wall mean (µs) | BW (GB/s) |
|--------|---------|-----------|-----------|---------------|---------|---------|----------------|-----------|
| Q      | 8       | 2048      | 4.0       | 189           | 180     | 287     | 236            | 0.03      |
| K      | 2       | 512       | 1.0       | 176           | 161     | 297     | 227            | 0.01      |

**npu time**: hardware cycle counter from `XRTKernelResult.npu_time` (excludes host dispatch).
**wall time**: end-to-end per-call latency measured on the host.
**BW**: effective memory bandwidth = 2 tensors × total_elem × 2 bytes / wall_mean.

### Correctness (oracle test)

Reference: per-head float32 RMSNorm + half-split RoPE applied to normalized output.
Tolerance: atol=0.02, rtol=0.02.

| tensor | max_abs_err | mean_abs_err | max_rel_err | result |
|--------|-------------|--------------|-------------|--------|
| Q      | 0.03125     | 0.00544      | 3.3603*     | PASS   |
| K      | 0.03125     | 0.00574      | 0.0243      | PASS   |

*max_rel=3.36 on Q is a near-zero element (numerator ~0.03, denominator ~0.009) — passes absolute tolerance; consistent with prior headnorm and rope results individually.

### Strategic note

This kernel is Stage 1 of the MLIR-AIE migration roadmap approved 2026-06-14.
The remaining 2-dispatch reduction (Q and K in a single dispatch) requires raw
MLIR-AIE to assign different weight tensors to separate tile columns; that is
Stage 2 (single pre-attention dispatch per layer).

## Inference Integration — Qwen3.5-0.8B SwiGLU NPU Bench

- **Date**: 2026-06-14
- **Model**: `qwen3.5-0.8b.mq4.hfq` (MQ4 quantized, 0.50 GiB HFQ payload)
- **xclbin**: `qwen35-swiglu-3584.xclbin` (hidden_size=3584, intermediate FFN dim)
- **Activation**: `HIPFIRE_QWEN35_FFN_BF16=xdna1 HIPFIRE_QWEN35_FFN_BF16_LAYER=all`
- **Path**: `forward_scratch_layers` → `weight_gemv_swiglu_residual_bf16_probe` → xdna1

### Changes needed to wire xdna1 on MQ4 models

1. **Load bypass**: `load_bf16_down_shadow_for` returned error for non-BF16 tensors when `FFN_BF16=xdna1`.
   Fixed: early-return `Ok(None)` for xdna1 mode — the down GEMV uses the original MQ4 tensor on GPU, the shadow w_down data is never needed.

2. **Forward-pass bypass**: `weight_gemv_swiglu_residual_bf16_probe` gated on shadow presence.
   Fixed: dispatch to xdna1 path before the shadow guard, using `w_down.k` for hidden_size.

3. **XRT session limit**: creating one XRT handle per layer_idx (24 total) hits the NPU context limit on NPU1, crashing with `free(): invalid pointer` after the 2nd handle.
   Fixed: all layers with the same hidden_size share one handle (cache key = hidden_size).

### Results

| mode                         | decode tok/s | ms/tok | wall tok/s |
|------------------------------|-------------|--------|------------|
| GPU only (baseline)          | 60.8        | 16.45  | 59.9       |
| SwiGLU NPU, layer=0 only     | 60.6        | 16.50  | 59.7       |
| SwiGLU NPU, all 24 layers    | 59.3        | 16.85  | 54.3       |

**The NPU SwiGLU path is ~2.4% slower than GPU-only on the 0.8B model.**

### Analysis

The 24 × ~180 µs dispatch floor = ~4.3 ms extra per token (serial host→NPU→GPU→NPU...).
The GPU SwiGLU for 24 layers takes ~0.3 ms total (tiny elementwise op on RDNA).
Net: +4.0 ms/token = -1.5 tok/s.

NPU SwiGLU only makes sense when:
- GPU is fully compute-saturated by GEMVs and the elementwise SwiGLU contends for waves
- Dispatches overlap with concurrent GPU work (not currently the case — dispatch is synchronous)
- On larger models where GEMV latency makes the ~4 ms dispatch overhead small relative to step time

For the 0.8B model (16 ms/tok baseline), the dispatch overhead is 25% of step time — not viable.
For a 7B model (~100 ms/tok), the same 4 ms would be ~4% — marginal.
Real NPU benefit requires async NPU dispatch or hardware-level DMA overlap.

---

## Performance Tuning — Qwen3.5-0.8B Decode (2026-06-14)

Systematic A/B benchmarks to identify easy decode gains. Baseline: FP32 state (previous session, run.rs).

Bench tool: `bench_qwen35_speed`, MQ4, gfx1103, `--gen 80 --warmup 5 --prefill 64`.

### DeltaNet State Quantization

| State quant | tok/s (gen) | vs Q8 |
|-------------|-------------|-------|
| Q8 (daemon default) | 62.2 | — |
| FP32 | 58.9 | -5.4% |
| Q4 | 58.7 | -5.6% |

**Q8 is optimal.** FP32 wastes 18 × 1MB = 18MB/step of bandwidth (read+write per DeltaNet layer).
Q4 is slower than Q8 — requant overhead outweighs bandwidth savings at this scale.

### KV Cache Mode

| KV mode | tok/s (gen) |
|---------|-------------|
| q8 | 62.0 |
| asym3 | 62.2 |
| asym4 | 61.1 |

Negligible on 0.8B — only 6 FullAttention layers, 2 KV heads, tiny KV footprint.

### Weight Format

| Format | tok/s |
|--------|-------|
| MQ4 | 62.2 |
| MQ6 | 51.2 |

MQ4 already correct choice. MQ6 = -18% (higher dequant cost per weight).

### hipGraph

Hardcoded `let use_graph = false;` at qwen35.rs:8331 (disabled 2026-05-15, token-0 attractor on gfx11+ROCm 7.x). `HIPFIRE_GRAPH=1` is a no-op on this branch. Would be +0.6–0.7% per prior measurements if re-enabled.

### Takeaway

The daemon already uses Q8 state by default. The 60.8 tok/s in the previous session was from `run.rs` which forced FP32 state. Real decode ceiling with current code: **~62 tok/s on gfx1103 MQ4**. Further gains require re-enabling hipGraph (needs bug investigation) or kernel-level optimization.

## Branch integration history — `NpuKernel` API union (2026-07-06)

When the local NPU line (R5–R15 + async dispatch) was rebased onto `chaingun`,
it met a parallel NPU effort already upstream. Both had independently extended
`NpuKernel` from the same base:

- **upstream**: blocking `submit(-> u64)` / `wait(seq)`, `submit_synced`
  (selective per-arg flush), `sync_output` (pipelined read-back cache reconcile),
  `import_dmabuf`, multi-slot command-BO cache.
- **local (kept)**: the async `NpuInFlight` owning-handle path for GPU∥NPU
  overlap with scheduler correlation tags.

Both were kept (union, nothing dropped). The only clash was the `submit`/`wait`
names, resolved by renaming the async pair to **`submit_inflight` /
`wait_inflight`** (`submit_tagged` / `poll` / `NpuInFlight` unchanged). The sole
async caller is `examples/async_smoke.rs`.

**Bisect note:** the reconciliation landed as a separate tip commit (`merge-fix(npu):
unify local async NpuKernel API …`), so the three commits that introduce/inherit
the async API before it — `feat(npu): async NPU dispatch split …` through
`refactor(rdna): single-source kernarg lists …` — do **not** individually compile
(duplicate `submit`/`wait` in `hipfire-xdna`). This is inherent to the divergent
rebase; the branch tip is green. `git bisect skip` that span when bisecting a
`hipfire-xdna` build across it.

---

## XDNA2 memory hierarchy and MALL characterization (2026-07-12)

The full 297-chapter local UG1079 manual was audited and the R1 feed probe was
extended across working-set sizes and CPU/GPU contention controls.

Headline results on Strix Halo:

- 14.4 GB/s active per receive stream;
- about 56.5 GB/s aggregate across eight columns;
- no throughput knee from 64 KiB through 64 MiB, including the 2 MiB and 32 MiB
  capacities reported by `rocminfo` for the DSP agent;
- 56.35 GB/s for one region shared by eight columns versus 55.65 GB/s for eight
  distinct regions;
- 43.04 GB/s under CPU DRAM pressure, 18.21 GB/s under a 512 MiB GPU stream, but
  no degradation under GPU hot-copy sets totaling 16 or 32 MiB.

The current amdxdna SHMEM path has no observed usable access to GPU MALL. The
resident EmbeddingGemma attention and FFN phases also achieve only about 0.9
GB/s against this 56.5-GB/s feed roof, so their current 9-14 ms phase costs are
not explained by global memory saturation.

See
[`npu-memory-bandwidth-cache-characterization.md`](npu-memory-bandwidth-cache-characterization.md)
for methods, limitations, manual references, roofline calculations, raw-result
links, and the qualified MALL conclusion.

## EmbeddingGemma bandwidth-first packed-W4 ladder (2026-07-12)

The projection-weight path now converts tensor-block order once in the loader
and persists a versioned, source/payload-SHA-validated `.rdna2.hfp` artifact.
W4 stays nibble-packed through external and memory-tile DMA; the AIE core owns
only representation-local nibble/lane handling. The real layer-0 combined QKV
payload is 2,359,296 bytes across eight columns.

Trace-timed medians from three locked trials per accumulated mode:

| mode | wire GB/s | useful TOPS | key evidence |
|---|---:|---:|---|
| packed feed only | 56.173 | — | same real HFP/topology control |
| signed nibble decode | 56.061 | — | lane-sum + byte-exact vector parity |
| first MMUL per W tile | 55.933 | 2.685 | exact int32 MMUL parity |
| production 6x16 K-group compute | 39.919 | 11.497 | exact parity; 57.45% receive stalls |

Decode retains 99.80% of the feed-only roof, and the first MMUL retains 99.77%
of decode. The 6x16 stage deliberately increases arithmetic work 6x: feed
retention falls to 71.37%, but useful TOPS rises 4.28x and receive stalls
identify compute backpressure. This is an explained roofline transition, not a
memory-system regression. Scaling, distinct output placement, end-to-end
projection integration, tok/s, and tok/J remain open.

Production scale/output correctness subsequently passed on the real packed
layer-0 QKV artifact: 327,680 outputs, zero mismatches, `max_abs=2e-7`. The
0.8635-ms wrapper timing includes CPU activation packing, NPU dispatch/sync, and
output deblocking, so it is not comparable to the trace-derived table and does
not establish model tok/s.

On 2026-07-13 the same checksummed offline-layout contract was extended to the
full-K slab schedule used by compact mixed Opus. A 26-case locked hardware
matrix passed W4, W8, and mixed `qt=36` across plain/`+`/`++`, overlay counts
1/3/7/39, and N=256/768/1152/1280/2304 with zero mismatches. W4/W8 maximum
absolute error was 2e-7; direct mixed full-K matched exactly. An OQ6.5 cache-hit
run preserved mtime, size, and SHA. This proves generic HFP projection
correctness and offline layout reuse, not resident-layer integration or model
tok/s. Rows:
`benchmarks/npu_gemm_tuning/results/r58-opus-hfp-format-matrix-20260713.csv`.

R59 then validated the packed-resident argument boundary. Because the DPU
regmap has five data-argument slots, final R34/R35 kernels require one offline
HFP bundle per destination context rather than separate projection and
parameter BOs. A generic `ResidentContextBundleV1` now preserves each role's
block order and records segment lengths plus source/payload hashes.

Locked three-trial medians were 56.121/56.053 GB/s for the R34/R35 separate-BO
controls and 55.884/56.018 GB/s for their production-shaped bundles. Both
bundles retain at least 99.48% of R58 feed-only with zero receive stalls; all role and
4-KiB parameter guards pass. This proves the weight ABI/DMA seam only, not R34
compute, resident-layer correctness, or model tok/s. Rows:
`benchmarks/npu_gemm_tuning/results/r59-resident-weight-abi-20260713.csv`.

Implementation and raw rows:

- `benchmarks/npu_gemm_tuning/r58/`
- `benchmarks/npu_gemm_tuning/results/r58-nibble-decode-20260712.csv`

R60 moves the same method into the actual R34 destination-context ABI. It
consumes the unchanged shared activation argument and the bundled QKV/O/params
HFP together. Locked three-trial medians are 53.838 GB/s for one exact MMUL,
54.252 GB/s for a complete K=256 group, and 50.912 GB/s for all three groups
plus real activation/weight scaling. These retain 96.34%, 97.08%, and 91.10%
of R59's 55.884-GB/s bundled baseline. All eight-column output checks pass;
the scaled stage covers the first 4x16 QKV output tile and shows about 9.9%
receive stalls. It remains a partial projection result, not model tok/s.

- `benchmarks/npu_gemm_tuning/r60/`
- `benchmarks/npu_gemm_tuning/results/r60-first-shared-input-mmul-20260713.csv`

R61 completes the corresponding QKV projection and direct row-major output.
All 327,680 M256/N1280 values and padded cells pass at `max_abs=3.8147e-6`.
The result is not performance-admitted: final median NPU time is 4.114 ms
(0.122 useful TOPS), about 4.8x slower than the 0.8635-ms whole-scaled control.
Native vector interleave reduced a 6.870-ms scalar version to 4.238 ms, but a
tile-local 6-KiB transformed activation cache did not improve the median. This
initially implicated legacy R34 activation compatibility, but R62/R63 later
falsified that conclusion: the apparent gap was dominated by comparing a cold
Python raw-runtime command with a warmed production wrapper.

- `benchmarks/npu_gemm_tuning/r61/`
- `benchmarks/npu_gemm_tuning/results/r61-full-qkv-rowmajor-20260713.csv`

R62 supplies producer-native W4 activations and preserves complete parity, but
only moves the cold raw-runtime median to 3.850 ms (row-major) or 3.856 ms
(physical output and canonical R15 loop). R63 makes the MLIR identical to R15
and uses the compact 2,359,296-byte QKV `.rdna2.hfp`; its cold raw median is
still 3.964 ms. In contrast, the production Rust/C++ wrapper measures current
R63 at 1.0292 ms median, the old spill binary at 1.0288 ms, and the historical
cache at 1.0596 ms. All nine production-wrapper runs pass 327,680 outputs with
`max_abs=2e-7`.

This admits the compact offline HFP plus current production executor for the
next resident-integration step. It does not admit the Python cold-command
number as a kernel ceiling or imply model tok/s.

- `benchmarks/npu_gemm_tuning/r62/`
- `benchmarks/npu_gemm_tuning/r63/`
- `benchmarks/npu_gemm_tuning/results/r63-production-wrapper-ab-20260713.csv`

R64 traces that exact R63/R15 device graph without changing its existing core,
FIFO, or DMA operations. Twelve locked fresh-process rows across shim columns
0-3 pass all 327,680 outputs with `max_abs=3.8147e-6`. Median device
input-to-output span is 241.248 us (240.189-243.356 us), effective aggregate
traffic is 19.559 GB/s, output-DMA starvation is 198.240 us, and padded/useful
compute rates are 2.817/2.086 TOPS. Columns 4-7 lack a terminal S2MM event at
trace stop and are not used for timing.

The traced device span is about 23.4% of the 1.0292-ms warm production-wrapper
time. The approximately 788-us remainder includes preparation, submission,
synchronization, and f32 output deblocking and is not assigned to one component
without further evidence. The next bandwidth-first step is a shared-BO mutable
BF16 attention-layout handoff, while immutable block conversion stays in the
loader/on-disk `.rdna2.hfp` path.

- `benchmarks/npu_gemm_tuning/r64/`
- `benchmarks/npu_gemm_tuning/results/r64-full-qkv-shim-trace-20260713.csv`

R65 replaces the physical f32 return with the exact BF16 raw-attention staging
prefix used by R29. The compact W4 HFP, activation ABI, and R15 compute are
unchanged. Three locked warmed fresh-process runs pass all 327,680 BF16 values
bit-for-bit, preserve every preseeded cos/sin/norm byte, and leave all padding
records zero. Median NPU time is 0.487964 ms (0.485649-0.490328), median host
call is 0.551481 ms, useful projection rate is 1.0315 TOPS, and maximum linked
core text is 9,280 bytes.

This measures projection through mutable BF16 staging, not headnorm/RoPE or
attention. R66 must attach the existing R29 packers and verify the 393,216-byte
Q and 262,144-byte KV layouts before the stage is admitted as a complete
attention handoff.

- `benchmarks/npu_gemm_tuning/r65/`
- `benchmarks/npu_gemm_tuning/results/r65-w4-bf16-raw-attention-20260713.csv`

R66 consumes the exact R65 inline records and emits canonical packed Q/KV.
Correctness matches the R28 gate: Q cosine 0.99999121/max error 0.0078125, K
cosine 0.99999156/max error 0.0078125, and bit-exact V. Three fresh locked
100-command runs measure 0.9511, 0.9915, and 0.9984 ms (median 0.9915 ms).

The schedule is not admitted. Its record broadcast serializes four core-pair
packers; R28's joined input executes them concurrently and is substantially
faster. R67 must recover the joined mutable layout before this handoff enters
the resident layer.

- `benchmarks/npu_gemm_tuning/r66/`
- `benchmarks/npu_gemm_tuning/results/r66-r65-stage-to-qkv-20260713.csv`

R67 changes mutable staging to 36 joined 8-KiB records per role, allowing the
R28 split FIFO to run all four packer pairs concurrently. Projection remains
bit-exact across 327,680 BF16 values and preserves every preseeded tail byte.
Three warmed runs measure 0.725232, 0.751200, and 1.152343 ms (median 0.751200
ms). Pack runs measure 0.3517, 0.3670, and 0.3687 ms (median 0.3670 ms) with the
established Q/K/V oracle.

Sequential medians are about 1.1182 ms before attention. This is faster than
R65+R66 but still dominated by approximately 360 small projection-output DMA
tasks; R68 targets a threefold task-count reduction without changing immutable
HFP order or pack math.

- `benchmarks/npu_gemm_tuning/r67/`
- `benchmarks/npu_gemm_tuning/results/r67-w4-joined-stage-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r67-joined-stage-to-qkv-20260713.csv`

R68 uses overlapping padded 24-token writes to cut the joined-stage producer's
output task count roughly threefold. Projection remains bit-exact and measures
0.465281, 0.494605, and 1.067005 ms (median 0.494605 ms). Pack measures 0.3435,
0.3579, and 0.3627 ms (median 0.3579 ms) with unchanged Q/K/V parity.
Sequential medians total about 0.8525 ms before attention.

R69 must verify a real shared-BO two-context chain; the isolated timing sum is
not yet a resident-layer measurement.

- `benchmarks/npu_gemm_tuning/r68/`
- `benchmarks/npu_gemm_tuning/results/r68-w4-overlap-joined-stage-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r68-overlap-joined-stage-to-qkv-20260713.csv`

R69 rejects the two-context boundary. Independent dma-buf imports are
incoherent, native XDNA SHMEM PRIME export returns `EINVAL`, and the direct
single-GEM-handle path is intermittent. Passing shared-handle runs take
5.45-5.66 ms; context scheduling, not the 0.03-ms BO sync, dominates.

R70 keeps projection and pack in one graph. Its two-channel, role-specialized
build matches the isolated R65 stage and R66 Q/KV byte-for-byte in three fresh
primed 100-command runs. Times are 1.3076, 1.3108, and 1.3006 ms (median 1.3076
ms), with maximum core text of 13,504 bytes. This proves the single-context
native-W4 projection/pack seam only; attention and full-model throughput remain
open.

- `benchmarks/npu_gemm_tuning/r69/`
- `benchmarks/npu_gemm_tuning/r70/`
- `benchmarks/npu_gemm_tuning/results/r69-cross-context-shared-qkv-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r70-single-context-projection-pack-20260713.csv`

R71 proves the complete projection/pack/attention boundary in one context, but
does not pass the speed gate. Moving all Q/V pack ownership to columns 0-3
leaves columns 4-7 for attention and fits at 15,888 bytes maximum core text.
The isolated R70 stage, Q, KV, and R27 attention outputs all match byte-for-byte.
Three primed 100-command runs measure 3.5951, 3.2617, and 3.3118 ms (median
3.3118 ms). Its redistributed pack-only control is 1.5446 ms median and isolated
attention is 0.9141 ms, so the fused attention/feed phase still costs about
1.77 ms. Attempts to split input/output shim columns exceed memory-tile input
DMA channels. Resident integration waits for a graph-local Q/KV stream or an
equivalent existing-channel reuse that preserves the exact result.

- `benchmarks/npu_gemm_tuning/r71/`
- `benchmarks/npu_gemm_tuning/results/r71-single-context-projection-pack-attention-20260713.csv`

R72 proves that Q can remain graph-local, but rejects scalar core streams as
the handoff. The external Q BO is unused while projection stage, external KV,
and final attention remain byte-exact. The fitting graph reuses 24 KiB of
projection-accumulator storage as the query cache and links at 15,248 bytes
maximum core text. Three primed 100-command runs measure 3.9288, 3.7749, and
3.9272 ms (median 3.9272 ms), 18.6% slower than R71. Removing 393,216 bytes of
external Q traffic does not repay per-word stream synchronization and cache
pressure. K/V should not use this scalar topology; the next handoff must retain
burst/vector DMA or reuse an existing graph FIFO.

- `benchmarks/npu_gemm_tuning/r72/`
- `benchmarks/npu_gemm_tuning/results/r72-direct-q-stream-20260713.csv`

R73 replaces scalar Q words with one adjacent-core, depth-one 24-KiB
ObjectFIFO. The Q BO remains unused and projection stage, external KV, and
attention are byte-exact. The producer-local precursor exceeded the 16-KiB
program limit; the adjacent topology fits with a 2-KiB producer stack and
14,912/14,352-byte maximum producer/consumer core text.

Three primed 100-command runs measure 3.6449, 3.7165, and 3.7205 ms (median
3.7165 ms). This recovers 5.4% from scalar R72 but remains 12.2% slower than
R71, so the serial six-group handoff is rejected. Shared tile memory itself is
not implicated as a correctness restriction: the added kernel parameter is the
workaround that stops the platform issue and must remain enabled. Local-memory
use is a separate capacity/performance choice; it does not replace that
workaround.

- `benchmarks/npu_gemm_tuning/r73/`
- `benchmarks/npu_gemm_tuning/results/r73-adjacent-q-objectfifo-20260713.csv`

R74 keeps two query groups and four accumulator/stat sets live per attention
core, reducing full 262-KiB KV replays from six to three while preserving the
observable R71 Q/KV boundary. A 4-KiB stack exceeds the 64-KiB active-tile
allocation by 1,184 bytes; the measured 2-KiB setting fits at 64,672 bytes with
864 bytes spare. Maximum core text is 15,248 bytes.

Projection stage, Q, KV, and attention remain byte-exact. Three primed
100-command runs measure 3.4496, 3.4242, and 3.2867 ms (median 3.4242 ms), 3.4%
slower than R71. Halving KV replay and DMA tasks does not repay the extra live
state, so this topology is rejected. The next rung targets phase scheduling or
core utilization rather than more tile-resident attention state. The added
kernel parameter remains the separate correctness workaround.

- `benchmarks/npu_gemm_tuning/r74/`
- `benchmarks/npu_gemm_tuning/results/r74-qgroup2-kv-replay-20260713.csv`

R75 changes only task scheduling: two groups' ordered Q, KV, and output tasks
are started before await/free, reducing six group barriers to three. A
six-group window exhausts static BD IDs at group 4; a four-group image links but
fails Q parity with 392,405 of 393,216 bytes wrong.

The two-group window is byte-exact. Three primed 100-command runs measure
3.2580, 3.2775, and 3.3314 ms (median 3.2775 ms), 1.0% faster than R71. Because
kernels, tile buffers, traffic, and math are unchanged, R75 is admitted as a
small command-stream scheduling win and the next projection/pack/attention
baseline.

- `benchmarks/npu_gemm_tuning/r75/`
- `benchmarks/npu_gemm_tuning/results/r75-attention-window2-20260713.csv`

R76 increases the correct task window from two groups to three. Projection
stage, Q, KV, and attention remain byte-exact. Three primed 100-command runs
measure 3.4199, 3.2222, and 3.2604 ms (median 3.2604 ms), 0.52% faster than R75
and 1.55% faster than R71. Three is the maximum correct queue window: four
corrupts Q and six exhausts static BD IDs. R76 is the admitted schedule for
resident-weight integration.

- `benchmarks/npu_gemm_tuning/r76/`
- `benchmarks/npu_gemm_tuning/results/r76-attention-window3-20260713.csv`

R77 consumes the real layer-0 QKV `.rdna2.hfp` through the production
`NpuEmbeddingQkvAttentionOpus` executor. It validates the HFP descriptor,
length, and payload SHA-256, allocates the compact weight BO in the destination
R76 context, uploads once, and reuses it across dispatches. No extracted weight
payload is used by the fused path.

Projection stage, Q, KV, and attention remain byte-exact. Three primed
100-command runs measure 3.2753, 3.3165, and 3.2137 ms (median 3.2753 ms),
within 0.46% of raw R76. This admits resident QKV/attention weights only; O
projection, tails, FFN, full-model tokens/s, and package tokens/J remain open.

- `benchmarks/npu_gemm_tuning/r77/`
- `benchmarks/npu_gemm_tuning/results/r77-resident-hfp-r76-20260713.csv`

R78 moves attention to odd columns and external Q/K/V packing to adjacent even
columns as a direct-output topology control. Projection stage, Q, KV, and
attention remain byte-exact; odd cores fit at 15,888 bytes. Three primed
100-command runs measure 3.8331, 3.7729, and 3.7959 ms (median 3.7959 ms), 16.4%
slower than R76, so the remap is rejected as a schedule.

Even cores still carry compact-W4 projection and pack code, leaving
insufficient program space for R32 output projection. The next capacity rung
requires loader-created pair-major compact-W4 HFP weights and paired projection
on odd cores so even cores can specialize in K/V, O projection, and tails.

- `benchmarks/npu_gemm_tuning/r78/`
- `benchmarks/npu_gemm_tuning/results/r78-odd-attention-remap-20260713.csv`

R79 adds the offline `PairedWholeScaledV1` layout needed for compact paired
projection. The loader moves complete schedule blocks from `(column, block)`
to `(pair, block, lane)` order, preserving every encoded byte. Cache identity,
source payload metadata, exact ordering, full block coverage, and reuse are
unit-tested. This is a loader/layout checkpoint only; no NPU timing or kernel
correctness is claimed.

- `benchmarks/npu_gemm_tuning/r79/`

R80 consumes the paired HFP with two accumulators per odd core and no even-core
projection image. After rejecting a six-task output queue that timed out, the
R65 per-slice cadence matches all 327,680 BF16 outputs bit-for-bit and preserves
all tail/padding guards. Three warm process medians are 0.818433, 0.833471, and
0.789289 ms (median 0.818433 ms). Maximum odd-core text is 11,872 bytes; even
cores are free. R80 is a capacity admission, not a speed win.

- `benchmarks/npu_gemm_tuning/r80/`
- `benchmarks/npu_gemm_tuning/results/r80-paired-w4-projection-20260713.csv`

R81 adds external Q/K/V packing to the paired projection and remains exact.
Three 100-command runs have a 1.8370-ms median; maximum odd/even text is
14,032/10,912 bytes. It is capacity-admitted but 40.5% slower than R70.

R82 then adds attention and fails before hardware execution: odd columns 1/3
need 22,416 bytes, 6,032 over the physical program store. The overflow is
program text, not tile SRAM, and carries no correctness or speed claim.

R83 uses the exact R70 single-group projection ABI plus non-LTO attention and
finish trip-count helpers. It packages at 15,888/10,912 bytes maximum odd/even
text and passes stage, Q, KV, and attention byte-for-byte. Three fresh
100-command runs measure 4.1666, 4.0493, and 4.0535 ms (median 4.0535 ms).
This is the first fitting paired projection/pack/attention capacity image, but
it is slower than R78 and R76 and is not the speed baseline. The separate
kernel parameter remains the correctness workaround; LDS avoidance is not.

- `benchmarks/npu_gemm_tuning/r81/`
- `benchmarks/npu_gemm_tuning/r82/`
- `benchmarks/npu_gemm_tuning/r83/`
- `benchmarks/npu_gemm_tuning/results/r81-paired-w4-projection-pack-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r82-program-capacity-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r83-compact-paired-projection-pack-attention-20260713.csv`

R84-R86 add direct O projection and isolate its cost. R85's activation-reuse
kernel is the admitted compute baseline at 5.7945 ms; R86 regresses. R87's
depth-two shim-to-memory-tile O-weight FIFO improves the median to 5.7450 ms,
while R88 depth three is saturated and rejected at 5.7343 ms.

R89 reuses 8 KiB of the final dead 10 KiB activation FIFO object on each even
core and adds a 4 KiB tail to stage three block-aligned 8x256 BF16 tiles. The corrected DMA-only
scatter uses `active_column * 32 + core_row * 8`. Stage and KV are bit-exact;
O reaches 0.99999225 cosine and 0.0625 maximum absolute error. Three fresh
100-command runs have a 5.7202-ms median. This admits local tail storage, not
residual/norm or end-to-end model execution. Maximum even-core program text is
14,544 bytes. The existing
kernel parameter remains the correctness workaround; LDS avoidance is not.

R90 adds post-attention RMSNorm, residual addition, and pre-FFN RMSNorm without
an external tensor round trip. A Q-pack loop rewrite fit but failed the output
oracle and is rejected. Making only the existing 18-record projection-drain
bound runtime-stable preserves all 12 Q-pack calls and fits at 15,952/16,048
bytes maximum even/odd text. Stage and KV remain bit-exact. The 196,608-value
tail has 0.99995399 global cosine, 0.99994058 minimum row cosine, 0.09375
maximum error, and no zeros or non-finite values. Three fresh 100-command runs
measure 6.6044, 6.2915, and 6.3890 ms (median 6.3890 ms). Admit this as the
local residual/norm correctness boundary and feed it directly into FFN next;
it is not yet a full-layer or speed admission.

R91 moves the complete stage ABI forward by one 393,216-byte canonical BF16
tensor and writes normalized H at offset zero. The shift is required because a
literal prefix overwrite destroys 65,536 bytes of immutable pack state that
projection does not regenerate on the next command. Both SHMEM and PRIME/GTT
controls pass sustained tail and KV oracles. The zero-copy resident R35 FFN
reaches 0.99989925 cosine and 0.0118408 maximum error. Producer/FFN/alternating
chain medians are 6.3727/9.7654/22.1772 ms; 6.0391 ms (27.2%) is context
alternation. This is 11,543 M256 rows/s for one layer boundary, not end-to-end
model tokens/s. The data contract is admitted; the cadence is rejected.

R92 loads the same R91/R35 images as peer contexts sharing one DRM file and
device heap. PRIME-exporting producer-owned XDNA SHMEM is rejected with
`EINVAL`, so the physical handoff remains GTT. Correctness is unchanged, while
producer/FFN/chain medians are 6.4109/9.7080/22.1542 ms. The 6.0353-ms context
tax is unchanged from R91. Same-DRM ownership is rejected; optimize the native
FFN bandwidth/compute phase or graph partition instead.

R93 establishes the exact native W4 FFN activation boundary. Canonical
M256xK768 BF16 pre-FFN state is transformed on AIE2P into R25's 108x6,656-byte
input layout with zero int8 mismatches across 589,824 replicated values,
`7e-9` maximum scale error, and clean padding. Core text is 7,856-9,040 bytes.
Three fresh 100-command runs measure 4.0618, 4.1218, and 4.1117 ms (median
4.1117 ms), only 0.263 GiB/s of physical source-plus-output traffic. The byte
contract is admitted, while a separate producer context is rejected; the next
rung must overlap preparation with the first gate/up weight-DMA/compute phase
and avoid materializing replicas externally. The kernel parameter remains the
platform workaround; this result imposes no LDS-avoidance rule.

- `benchmarks/npu_gemm_tuning/r84/`
- `benchmarks/npu_gemm_tuning/r85/`
- `benchmarks/npu_gemm_tuning/r86/`
- `benchmarks/npu_gemm_tuning/r87/`
- `benchmarks/npu_gemm_tuning/r88/`
- `benchmarks/npu_gemm_tuning/r89/`
- `benchmarks/npu_gemm_tuning/r90/`
- `benchmarks/npu_gemm_tuning/r91/`
- `benchmarks/npu_gemm_tuning/r92/`
- `benchmarks/npu_gemm_tuning/r93/`
- `benchmarks/npu_gemm_tuning/results/r87-output-weight-shim-depth2-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r88-output-weight-shim-depth3-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r89-bf16-local-o-stage-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r90-residual-norm-tail-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r91-zero-copy-ffn-handoff-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r92-peer-context-control-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r93-bf16-to-r25-activation-20260713.csv`

R94 vectorizes the admitted activation transform and retains the R25 ABI with
three one-code q differences, `7e-9` maximum scale error, and clean padding.
Its 2.1320-ms fresh-process median is 48.1% below R93 but still only 0.507
GiB/s, so it is a fusion building block rather than a standalone phase.

R95 and R96 recover program capacity without changing the full-FFN oracle.
The dynamic W4 init/accumulate body reduces maximum core text from 16,320 to
13,968 bytes; the compact down-fragment ring reduces it again to 12,944 bytes.

R97 proves canonical-BF16 input DMA, all weight objects, and the complete
256-row activation preparation boundary. Groups 1-2 have zero q mismatches;
group 0 has one one-code difference, with only float-rounding scale differences.
The first fused image contained NaNs because its gate fragment exchange
overwrote R25's still-live partial down spill in the shared `own`/`transit`
buffers. Giving gate preparation two dedicated 784-byte buffers fixes that
kernel state-lifetime alias. The full oracle then reaches gate cosine
`1.00000000`, final cosine `0.99998228`, maximum absolute error `0.2597733`, and
mean absolute error `0.03750710` at 15,456 bytes maximum core text. A fresh
6.4095-ms dispatch corresponds to 39,941 M256 rows/s. R97 already inherits
R15's required `rounding=floor` and `saturation=none` numerical controls. A
20-command run nevertheless reaches the separate four-second command timeout
cadence. Recycling the context every seven commands is a sufficient independent
mitigation: three 100-command runs preserve the full oracle at 6.4974, 6.4844,
and 6.3388 ms (median 6.4844 ms, 39,479 M256 rows/s). This admits sustained
standalone R97, not complete-layer or end-to-end encoder throughput.

The separately added kernel parameter remains the platform-issue workaround;
it is not LDS avoidance and is not the R15 rounding/saturation configuration.
R97's buffer separation is an independent kernel correctness fix. The command timeout,
tile-memory use, capacity refactors, fresh command objects, and context
recycling remain separate issues or choices.

R98-R100 establish the native-W4 output/tail seam. R98 emits compensated BF16
high/low pairs in place at unchanged physical byte volume, fits at 16,032 bytes,
and sustains 38,886 M256 rows/s. R99 scatters the same 884,736 bytes into the
resident tail's existing 1,327,104-byte combined-row BO and sustains 38,788
rows/s. R100 consumes that interleaved prefix with split architectural X and
passes at `0.99999861` cosine and `0.0039062` maximum error. Host-written
split-X verification now explicitly flushes both combined and residual BOs;
the production NPU-to-NPU path does not add that host synchronization.

- `benchmarks/npu_gemm_tuning/r94/`
- `benchmarks/npu_gemm_tuning/r95/`
- `benchmarks/npu_gemm_tuning/r96/`
- `benchmarks/npu_gemm_tuning/r97/`
- `benchmarks/npu_gemm_tuning/r98/`
- `benchmarks/npu_gemm_tuning/r99/`
- `benchmarks/npu_gemm_tuning/r100/`
- `benchmarks/npu_gemm_tuning/results/r94-vector-activation-prep-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r95-dynamic-w4-capacity-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r96-compact-fragment-ring-capacity-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r97-inline-canonical-gate-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r98-interleaved-bf16x2-output-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r99-combined-row-scatter-20260713.csv`
- `benchmarks/npu_gemm_tuning/results/r100-interleaved-tail-20260713.csv`

R99/R100 are now wired into the reusable native-W4 completed-layer path. A
layer-0 hardware comparison reports FFN cosine `0.99997024`, tail cosine
`0.99999873`, and completed-layer cosine `0.99998514`. The integration must
preserve direct architectural X in a separate 442,368-byte shared BO because
the temporary host normalization bridge rewrites the attention hidden buffer
with pre-FFN-normalized H. Reusing that buffer for both roles produced the
rejected `0.90738614` tail cosine.

The resident-only 24-layer OQ4 path completes at M256, measuring 878.003 ms or
291.6 input tok/s in the current bridge implementation. This is a functional
checkpoint, not an admitted performance result: host readback, normalization,
and preparation/output still consume about 10-12 ms per layer. R15's
`rounding=floor` and `saturation=none` remain numerical controls. The separately
added kernel parameter remains the platform-issue workaround; this integration
does not replace it with an LDS-avoidance rule.

R101-R103 reject three attempted host-bridge removals without weakening that
functional checkpoint. The literal R101 row-state relay reaches 16,444 bytes;
a compact shim-DMA scatter fits at 16,380 bytes but produces misaddressed
inverse RMS state and only 0.50248530 layer-0 cosine. Relocating the metadata
object to the even normalized-X FIFO fits at 16,268 bytes but reaches the
separate four-second command timeout and produces invalid state. R102 becomes
allocation-clean at 16,064 bytes after reducing its 15,872-byte weight FIFO to
depth one, but cannot be admitted without a correct producer. The first R104
implementation recomputed RMS inside the FFN but overflowed at 18,352 bytes
(`-Oz`: 20,032 bytes).

These are program-capacity, DMA-layout, and scheduling failures. They do not
implicate LDS or replace the separately added platform-workaround kernel
parameter. R15's `rounding=floor` plus `saturation=none` remain independent
numerical controls. The next bandwidth-first rung keeps R44's
known-good direct-X output and R99/R100's correct W4/tail seam, then performs
only the mutable X-times-inverse activation preparation on device. It must
preserve X for R100, fold immutable pre-FFN norm into loader-side W4 scaling,
and beat the current two 5.1-MiB host synchronizations before integration.

R105/R106 implements that boundary as a separate device context and proves
correctness, but rejects the topology for performance. Standalone R105 reaches
cosine `0.99999122` at a 0.1295-ms median. Integrated layer 0 reaches unit-RMS,
FFN, tail, and completed-layer cosines of `0.99999269`, `0.99990930`,
`0.99999862`, and `0.99996179`. Across all 24 layers, however, explicit
cross-context cache maintenance raises the R105 phase to 2.35-4.14 ms per layer
and regresses execution from 878.003 ms / 291.6 tok/s to 901.432 ms / 284.0
tok/s (about 21 W and 13.5 tok/J). Runtime selection therefore requires
`HIPFIRE_EMBED_UNIT_RMS_BRIDGE=1`; at this checkpoint R99/R100 remains the
default while RMS is compacted into the resident W4 context.

R104 now completes that compaction. Source-level vector mean scaling, inverse
fusion, one runtime-stride FWHT body, and a full `3 x 768` BF16 object per core
bring every core to exactly 16,384 bytes under the normal `-O2` aiecc flow. The
full object is transferred once (442,368 bytes total), scanned once for RMS,
and reused by group. Standalone correctness is `1.00000000` gate cosine and
`0.99996707` final cosine; 100 recycled commands sustain the same oracle at
6.5401 ms. Default layer-0 integration reaches `0.99991494` FFN,
`0.99999844` tail, and `0.99996658` completed-layer cosine.

Paired full-model controls measure R99 at 892.708/909.986 ms and R104 at
859.599/869.015 ms, a 4.1% paired-mean latency reduction. A fresh default R104
run completes all 24 layers in 894.222 ms (286.3 input tok/s, 18.07 W,
15.8 tok/J). R104 is therefore admitted as the default native-W4 resident FFN
when its artifact exists. The next dominant boundary remains the distinct
next-layer/residual preparation contexts at roughly 9-12 ms per layer. The
added kernel parameter remains the platform workaround; no LDS-avoidance rule
is inferred from this result.

R108/R109 remove the next separate residual-copy context. The rejected R107
fusion exceeded memory-tile DMA channels; a six-argument R108 exceeded the DPU
register map; and importing one BO twice in R109 returned `EALREADY`. The
admitted layout keeps completed BF16x2 in an 884,736-byte prefix and R34
activation records in a disjoint suffix of one shared five-argument attention
BO. R109 prepares the suffix in place, while R108 feeds the existing attention
residual FIFO from the rounded high BF16 plane.

Layer-0 FFN/tail/completed cosines are `0.99991644`, `0.99999886`, and
`0.99996836`. Preparation drops from about 9-12 ms to 7-9 ms per layer.
Alternating full-model controls average 815.294 ms for R48 and 804.722 ms for
R108/R109, a 1.30% latency win. Energy samples are inconclusive and are not an
admission claim. The kernel-parameter workaround and LDS placement remain
independent.

R110 refreshes format-generic execution on that path. Native OQ8, calibrated
OQ8+, freshly generated OQ8++, and compact mixed OQ6.5 all pass the locked M256
layer-0 component oracle; completed-layer cosine spans `0.99996270-0.99997103`.
The OQ8++ offline package reports 168/168 successful LDLQ projection packs.
Full 24-layer results are:

| artifact | BF16 embedding cosine | ms | input tok/s | W | tok/J |
|---|---:|---:|---:|---:|---:|
| `.npu.oq8.hfq` | 0.99547863 | 864.929 | 296.0 | 21.05 | 14.1 |
| `.npu.oq8+.hfq` | 0.99584466 | 855.854 | 299.1 | 20.04 | 14.9 |
| `.npu.oq8++.hfq` | 0.99571574 | 838.997 | 305.1 | 21.05 | 14.5 |
| `.npu.oq6.5.hfq` | 0.95821863 | 868.020 | 294.9 | 21.00 | 14.0 |

OQ6.5 is arbitrary mixed-width execution evidence, not OQ8-level quality.
The matrix validates generic dispatch and suffix handling but remains roughly
33x short of 10k tok/s. Durable rows:
`benchmarks/npu_gemm_tuning/results/r110-generic-opus-formats-20260713.csv`.

R111 applies the next bandwidth-first reduction without changing R109's
in-place ABI or immutable `.rdna2.hfp` order. One 3,072-byte completed row is
copied into tile memory, its FIFO is released immediately, and all three K256
activation chunks are produced from the local row. The completed allocation is
884,736 bytes including padding, while each sweep reads 786,432 active bytes.
R111 therefore cuts active completed-state reads from 3,145,728 to 786,432
bytes per layer and completed-input shim tasks from 32 to 8.

The held-FIFO schedule is rejected because it reproduces R54. The first
copy/release build also exposed a distinct alignment error: packer-owned groups
1-2 were only 32-byte aligned but used a 64-byte vector load. Switching that Q
copy to the guaranteed 32-byte alignment passes with five one-code Q
differences and `7e-9` maximum scale error. Core text is 9,072-10,592 bytes.

Four counterbalanced full-model pairs, each averaging three encodes, measure
R109 at 760.323 ms / 336.7 tok/s / 17.83 tok/J and R111 at 749.409 ms / 341.6
tok/s / 18.10 tok/J. R111 wins all four latency pairs with identical BF16
embedding cosine `0.92839295`, and is selected by default when the artifact is
present; R109 remains the fallback. Removing 2.36 MB of active reads yields
only a 1.44% end-to-end latency win, evidence that this boundary is not purely
external-bandwidth limited. The next larger opportunity is context/route
consolidation around the R100 tail; R108 cannot absorb it with only 16 bytes of
program headroom.

The platform workaround remains the separately added kernel parameter. It is
not LDS avoidance, the R15 numerical settings, or R111's alignment correction.
Durable rows:
`benchmarks/npu_gemm_tuning/results/r111-one-pass-next-prep-20260713.csv`.

R112 prepares the R100 tail for in-context R111 fusion without changing
immutable layout or total input traffic. A second split-X memory-tile
broadcast is rejected at compile time because it exceeds the tile output-DMA
channel budget. The admitted graph instead uses the third plane already
reserved in R99's mutable 4,608-byte row for canonical token-major X. Strided
DMA gives each core eight contiguous tokens and returns completed rows in
canonical order; it performs no tensor-block conversion.

R100 and R112 both transfer 1,179,648 active input bytes. R112 removes 24 core
flows and reduces maximum core text from 4,208 to 3,696 bytes. Both hardware
oracles report cosine `0.99999861` and maximum error `0.0039062`. Across four
counterbalanced 100-command pairs, R100 averages `0.324965 ms` and R112
averages `0.218271 ms`; R112 wins every pair and is 32.84% faster. This is
standalone tail admission and route headroom for fusion, not end-to-end encoder
throughput. The separately added kernel parameter remains the platform
workaround; LDS placement is independent. Durable rows:
`benchmarks/npu_gemm_tuning/results/r112-fusion-ready-tail-20260713.csv`.

R113 fuses R111's exact next-layer RMS/AWQ/FWHT/int8 preparation into R112's
tail context. Each core preloads 9,216 bytes of next-layer parameters and packs
each two-row output while it is still local; the rejected literal design kept
an extra 24,576-byte eight-row completed buffer and failed bank allocation.
The admitted phase-local form fits every core at 9,984 text bytes and removes
the separate 786,432-byte completed-state input pass.

Seven simultaneous shim output tasks are not viable: four completed outputs
plus three diagnostic groups leave group 2 all zero. R113 launches all stripes,
retires one completed task per stripe, then queues the three diagnostics. This
passes at tail cosine `1.00000000`, `0.0000310` maximum error, three one-code Q
differences, and `7e-9` scale error. A correct but stripe-serial retirement
schedule took 13.3578 ms and is rejected.

Four live 50-command samples average 5.056325 ms. Current R112 and R111 control
means are 0.236051 and 5.024850 ms, totaling 5.260901 ms, so the fused rung is
3.8886% faster while transferring fewer dynamic bytes. One transient fresh
context returned all-zero output; four immediate fresh contexts and the full
repeat passed, so it remains context-transition evidence rather than an LDS
or platform-workaround diagnosis. R114 next tests in-context R34 suffix
assembly; the now-dominant RMS/FWHT pack body remains an optimization target.

The separately added kernel parameter remains the platform workaround. It is
not LDS avoidance, the queue schedule, R111's alignment fix, or context
recycling. Durable rows:
`benchmarks/npu_gemm_tuning/results/r113-tail-next-pack-fusion-20260713.csv`.

R114 does not admit the next assembly step. Logical-owner stream chains,
physical column-major stream chains, and adjacent neighbor-memory assembly with
a new shim route all fail compilation with `Unable to find a legal routing`.
A zero-destination-stride reuse attempt is invalid because DMA strides must be
positive. Reusing the existing completed-output route for separate Q and scale
planes does build in about nine seconds and fits at 11,200 bytes maximum core
text. Its compact ABI transfers 589,824 bytes, with neither 16 KiB padding nor
five materialized N-macro replicas.

Locked hardware parity rejects that build: the completed tail can pass while
the compact pack has 107,811 mismatches, maximum Q delta 254, and maximum scale
error 0.034057196. Failures span every K256 group and local-memory owner
position, so the assembly/mapping is wrong without yet identifying a single
bad predecessor. Do not select R114.

Proceed by treating R113's correct per-core output as the dynamic compact ABI.
Its 589,824-byte padded diagnostic surface contains 199,680 unique chunk bytes.
The next resident R34 consumer should read those chunks directly and reuse each
across five N-macros rather than building the canonical 2,949,120-byte
replicated activation tensor. Immutable `.rdna2.hfp` tensor blocks stay in
their offline/loader-provided order.

An intervening fresh context produced an all-zero completed tail and the
immediate repeat produced the distributed compact-pack failure. Retain this as
separate context-transition evidence. The added kernel parameter is the
platform workaround that stops the platform issue; LDS avoidance is not the
workaround. Local-memory placement, output routing, R111 alignment, and context
lifetime remain independent. Durable rows:
`benchmarks/npu_gemm_tuning/results/r114-r34-compact-boundary-20260713.csv`.

R115 validates the consumer-side alternative. All 32 cores read their R113
eight-token group-0 chunks directly and execute a scaled K256-by-N16 int8
matrix stage. Outputs scatter to canonical `[256,16]` f32. The graph neither
assembles 24-token records nor materializes N-macro activation replicas, and
immutable weights remain in loader/offline `.rdna2.hfp` order.

The mapping-isolation build still reads each 6,144-byte diagnostic slot:
196,608 physical activation bytes carry 66,560 unique bytes. Maximum core text
is 1,692 bytes. Hardware parity is zero mismatches with `2e-9` maximum absolute
error. Six fresh 1,000-dispatch runs pass at 0.090826-0.094342 ms, mean
0.092506 ms. One preceding fresh process returned mostly zero output; the next
six fresh processes passed. Retain that as separate context-transition
evidence.

R115 admits direct chunk consumption and one compute group only. Add the other
two K256 groups with local f32 accumulation before extending N. The added kernel
parameter is still the workaround that stops the platform issue. It is not LDS
avoidance, and the transient fresh-context result does not change that.
Durable rows:
`benchmarks/npu_gemm_tuning/results/r115-direct-compact-group-n16-20260713.csv`.

R116 extends the direct consumer to all K768 while retaining N16. Each token
owner consumes three R113 chunks and accumulates locally in f32. It reads the
589,824-byte padded compact ABI containing 199,680 unique bytes, not the
2,949,120-byte five-N-macro activation materialization.

The initial group-1 failure affected only columns 0-7. A 128-byte padded weight
record does not change it; per-group DMA tasks and core-loop unrolling produce
zeros. The admitted math stages the prior 8x16 f32 output into a 512-byte local
array before the next MMUL. K512 unit-scale parity is bit-exact and K768 reports
zero mismatches with `4e-9` maximum error. Core text is 2,220 bytes.

Eight passing 1,000-dispatch fresh processes average 0.096384 ms. Two other
fresh processes returned all-zero output before subsequent fresh processes
passed. Therefore the full-K computation and compact ABI are admitted as a
correct rung, but the image is not a context-stable runtime default. The next
compute step is another N slice, preserving zero activation replicas. The added
kernel parameter remains the distinct workaround that stops the platform issue;
the local prior-output stage is not that workaround and is not LDS avoidance.
Durable rows:
`benchmarks/npu_gemm_tuning/results/r116-direct-compact-fullk-n16-20260713.csv`.

R117 produces N32 from the same compact activation transfer. Each activation
load feeds four 8-column MMUL halves; physical/unique activation bytes remain
589,824/199,680 and N-macro replicas remain zero. Both N16 halves pass with zero
mismatches and `3e-9` maximum error. Maximum core text is 3,192 bytes.

Eight passing fresh 1,000-dispatch samples average 0.086916 ms, versus R116's
0.096384 ms for N16. Thus twice the useful N work is 9.82% faster at this
boundary, showing fixed dispatch/traffic costs dominate. Two other fresh
contexts returned all-zero output, so context-stable runtime admission remains
open.

Proceed by staging the three compact chunks once per token-owner core and
streaming multiple N32 weight/output blocks. Do not replay activation DMA as N
grows. The added kernel parameter remains the platform workaround; this wider
consumer and its local prior-output array are independent of LDS avoidance.
Durable rows:
`benchmarks/npu_gemm_tuning/results/r117-direct-compact-fullk-n32-20260713.csv`.

R118 proves activation-once N64. Each core stages three compact chunks, then
streams two N32 weight/output records. A 6,240-byte concatenation is rejected:
group 1 begins at offset 2,080 and violates the 64-byte MMUL load alignment.
Using a 2,112-byte stride and 6,336-byte stage fixes the compute path.

The first repeated S2MM descriptor emits only block 0. Two explicit output
tasks fit queue depth two and pass both blocks with zero mismatches and `5e-9`
maximum error. Core text is 3,736 bytes. Activation DMA remains exactly one
589,824-byte pass with 199,680 unique bytes and no N-macro replicas.

Nine passing fresh 1,000-dispatch samples average 0.106058 ms, 22.0% slower
than N32 for twice the useful N work. One sample returns the known all-zero
context symptom. The next gate combines outer DMA tiling with task
`repeat_count`; this is required for many output blocks without excessive task
queues. The platform-workaround kernel parameter remains distinct from both the
alignment fix and deliberate local-memory staging. Durable rows:
`benchmarks/npu_gemm_tuning/results/r118-staged-fullk-n64-20260713.csv`.

R119 supplies the missing scalable output schedule. An outer DMA tiling
dimension of two plus task `repeat_count=1` consumes both N32 objects with one
output task per stream. R118's outer dimension without task repeat consumed
only block 0.

R119 has zero parity mismatches and `5e-9` maximum error. Eight passing fresh
1,000-dispatch samples average 0.102308 ms, 3.54% faster than R118's explicit
two-task schedule. Two contexts return all zeros. The schedule is admitted for
increasing N32 block count while activation remains one 589,824-byte pass; it is
not yet a context-stable runtime default. The platform workaround remains the
separate kernel parameter. Durable rows:
`benchmarks/npu_gemm_tuning/results/r119-repeat-output-task-20260713.csv`.

R120 increases the admitted schedule to four N32 blocks (N128). Output strides,
outer tiling, and task `repeat_count=3` scale while the kernel and one 589,824
byte activation pass remain unchanged. All four blocks pass with zero
mismatches and `7e-9` maximum error. Four passing fresh 1,000-dispatch contexts
average 0.115102 ms; six other fresh contexts return the known whole-output
zero symptom. This admits topology, not context-stable selection. Durable rows:
`benchmarks/npu_gemm_tuning/results/r120-staged-fullk-n128-20260713.csv`.

R121 completes the 40-block N1280 projection schedule. It reads 7,987,200 W8
diagnostic weight bytes, preserves the one-pass 589,824-byte compact activation
input, and writes 1,310,720 f32 output bytes. The full M256 K768 N1280 oracle
passes with zero mismatches and `6e-9` maximum error in all ten fresh contexts.
The 1,000-dispatch range is 0.319049-0.325542 ms and the mean is 0.320640 ms,
about 798,402 M256 projection rows/s and 30.84 GB/s over the measured payloads.

This is an admitted full-width single-projection schedule, not a full-layer or
encoder tok/s result. Next connect generic runtime Opus packed records to this
schedule and retain the byte oracle through OQ4, arbitrary mixed bitwidths,
OQ8, and +/++ metadata. The kernel parameter remains the platform workaround;
LDS placement and repeated DMA scheduling are independent. Durable rows:
`benchmarks/npu_gemm_tuning/results/r121-staged-fullk-n1280-20260713.csv`.

## R122 — resident component characterization before batched M (2026-07-15)

The active EmbeddingGemma completed-layer route now exposes decision-level
timings for attention prepare/pack/run, unit RMS, FFN, tail, next activation
prep, residual prep, output materialization, final norm/mean, and Dense/L2. A
new trace summarizer assigns repeated encodes to cold/primed samples and derives
per-encode XDNA submit/wait deltas from the cumulative dispatch trace.

One fresh oq8 process ran three M256 encodes; the two primed samples averaged
746.822 ms across the 24 layers plus 20.976 ms finalization. Component totals
were 269.042 ms FFN, 212.178 ms attention (11.540 prepare and 200.629 NPU run),
168.409 ms next-layer prep, 86.612 ms tail, 8.322 ms setup, 2.575 ms final
norm/mean, 18.392 ms Dense/L2, and 1.992 ms final host materialization. The
active route bypasses separate unit-RMS, residual-prep, and interior GPU-pack
work; zeroes in those columns describe route selection, not alternate-kernel
latency.

The same primed samples used 98 XDNA dispatches and averaged 1.894 ms submit vs
744.854 ms wait. A repeated batched FFN hardware check remained bit-exact and
measured M256 10.520 ms vs M512 19.929 ms, only 1.06x row throughput. Therefore
launch batching and naive FFN M growth are rejected. Continue with FFN weight
reuse and next-prep/tail context consolidation; postpone final Dense/L2 work.

Durable rows:
`benchmarks/npu_gemm_tuning/results/embeddinggemma-resident-components-m256-20260715.csv`
and
`benchmarks/npu_gemm_tuning/results/embeddinggemma-resident-samples-m256-20260715.csv`.
