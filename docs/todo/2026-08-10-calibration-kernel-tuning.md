# TODO: tune the three kernels that own streamed calibration

Opened 2026-08-10 from `rocprofv3 --kernel-trace` slices of the live
Qwen3.5-122B-A10B `oq4.25++` conversion. Three kernels are **~84%** of
calibration GPU time and none has had a tuning pass; a fourth item covers the
quantize stage, which is the larger half of a conversion by wall clock (~7 h
against ~1.5 h here).

## The measurement

Two `rocprofv3 --kernel-trace` slices of the real 122B run, taken in place.
88–92% of each profiled span was GPU-busy on kernels, so this is compute-bound —
not I/O-bound and not launch-bound.

**Sample A — layers 24+25 (one FULL-attention layer, one linear).** 258.3 s
kernel / 279.7 s span.

| kernel | share | calls | avg |
|---|---|---|---|
| `attention_f32_routed_batched` | 38.3% | 256 | 386 ms |
| `calib_hessian_outer_f32` | 26.0% | 9,672 | 6.9 ms |
| `gemm_bf16_moe_grouped_wmma_gfx1151` | 23.5% | 1,024 | 59.3 ms |
| `gemm_bf16_x_bf16_wmma_gfx1151_nheavy` | 5.7% | 1,792 | 8.2 ms |
| `__amd_rocclr_copyBuffer` | 1.4% | 2,482,243 | 1.4 µs |

**Sample B — layers 1–3 (ALL linear attention).** 249.6 s kernel / 283.5 s span.

| kernel | share | calls |
|---|---|---|
| `calib_hessian_outer_f32` | 41.4% | 14,508 |
| `gemm_bf16_moe_grouped_wmma_gfx1151` | 37.2% | 1,536 |
| `gemm_bf16_x_bf16_wmma_gfx1151_nheavy` | 9.2% | 3,072 |
| `gated_delta_net_f32_routed_batch_seq` | 4.0% | 768 |
| `attention_f32_routed_batched` | — | absent by construction |

### Weighting matters — sample A alone gives the WRONG ranking

`full_attention_interval: 4`, so the model is **36 linear + 12 full** layers.
Sample A is one of each, which over-weights full-attention layers 2:1 against
their true 1:3 share. Decomposing A with B (subtract B's per-linear-layer cost
from A) gives per-layer kernel time of **83.2 s linear vs 175.1 s full — a 2.10×
ratio**, which independently matches the observed wall clock (98 s vs 196 s).
Reweighting to 36:12:

| kernel | WHOLE MODEL | (sample A alone said) |
|---|---|---|
| `calib_hessian_outer_f32` | **32.1%** | 26.0% |
| `gemm_bf16_moe_grouped_wmma_gfx1151` | **28.9%** | 23.5% |
| `attention_f32_routed_batched` | **23.3%** | 38.3% — *rank 1 → rank 3* |
| `gemm_bf16_x_bf16_wmma_gfx1151_nheavy` | 7.1% | 5.7% |
| `gated_delta_net_f32_routed_batch_seq` | 2.4% | 1.3% |

Attention is still large and still the only scalar-f32 hot kernel, but it runs on
a quarter of the layers and is **not** the top cost. Profile a representative
layer MIX, or reweight, before ranking anything on a hybrid model.

Two traps for whoever reads these DBs next:

- `top_kernels.total_duration` is in **microseconds**. Reading it as ns makes
  the job look 0.09%-GPU-busy and launch-bound, which is wrong by 1000×.
  Cross-check against `SUM(end-start)` on `rocpd_kernel_dispatch_*`, which is ns.
- The millions of `copyBuffer` calls look alarming and are not: ~1.4 µs each,
  1–2% of time. Call count is not cost.

Absolute times carry ~47% profiler overhead. Proportions are sound; don't quote
the seconds.

### Reproducing

`rocprofv3 --attach <pid>` does NOT work here — `ptrace_scope=1`, and even with
it relaxed to 0 the attach fails (`librocprofiler-sdk-rocattach.so ... status 6`)
and the injection pushed an already memory-pressured calibrate into
`hipMalloc: out of memory`. Run rocprofv3 as the PARENT instead, and bound the
work with `--pause-after-layers`, which is an **absolute** layer target, not a
relative count — with 23 layers done you pass `25` to get two more:

```sh
rocprofv3 --kernel-trace --stats -d out -o run -- \
  ./target/release/hipfire-coexistence calibrate --model M.hfa \
  --corpus benchmarks/calib/calib-multi-8m.txt --sequences 192 --context 2048 \
  --expert-coverage-policy preserve-undercovered \
  --pause-after-layers 25 --output M.calib.hfq
```

For a LINEAR-only sample, target layers 1–3 and write to a scratch output.
Calibration resumes from its boundary checkpoint, so a profiled slice costs
nothing but the profiler overhead — the layers it does are real work.

## 1. `calib_hessian_outer_f32` — 32% whole-model, and half the work is redundant

`kernels/src/calib_reduce.hip:66`. Start here: reweighting makes it both the
LARGEST single cost and the cheapest real win in the file.

`H = XᵀX` is **symmetric**, and the kernel computes the whole thing. Every
`(i,j)` is written (`H[i*K+j] += acc`), and the launch grid is the full
`⌈K/16⌉ × ⌈K/16⌉` (`dispatch/mod.rs:3430`). Nothing exploits `H[i,j] == H[j,i]`.

- **Lever A — skip the mirror half.** Launch only blocks with
  `blockIdx.y <= blockIdx.x` and mirror-write, or keep the square grid and early-
  `return` the strictly-lower blocks after writing both triangles. ~2× on 32% of
  runtime ⇒ ~16% end-to-end. Check first whether any consumer reads the lower
  triangle expecting it to be populated independently; the LDLQ/AWQ paths in
  `hipfire-quantize` are the ones to audit.
- **Lever B — WMMA.** The inner loop is *already* a 16×16×16 tiled GEMM with
  `CALIB_TILE 16`, which is exactly `__builtin_amdgcn_wmma_*_16x16x16_*_w32`.
  The obstacle is precision: `x` is f32 activations and the Hessian accumulates
  over the whole corpus, so a bare bf16 cast loses mantissa where it matters.
  Split-precision (bf16 high + bf16 low correction term, "3×bf16") is the
  standard answer and keeps f32 accumulate, which `reference_rdna3_wmma_accumulate`
  says is free on gfx1151. **Validate against the f32 kernel on a real calib
  before trusting it** — a quietly worse Hessian is exactly the silent-quality
  failure this repo keeps getting bitten by.

Do A first and measure; it is a small diff with no numerical risk.

## 2. `gemm_bf16_moe_grouped_wmma_gfx1151` — 29% whole-model, missing the N-heavy treatment

`kernels/src/gfx1151/gemm_bf16_moe_grouped_wmma.gfx1151.hip:23`. Already WMMA
(`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`), so this is tuning, not porting.
Current shape: `M_TILE 128, N_TILE 16, K_BLOCK 256, WARPS 8`.

- **Lever A — port the N-heavy shape.** `N_TILE 16` means one 16-column strip
  per tile, so each A fragment feeds a single WMMA. The dense sibling
  `gemm_bf16_x_bf16_wmma_gfx1151_nheavy` already got exactly this fix
  (`36292ac66` "reuse each A fragment 4×", `14f2ca436` "NSUB 4 → 8",
  `45465217a` "register-block 2×8") and sits at 8.2 ms/call against this one's
  59.3 ms. The grouped MoE kernel never received it. This is the highest-value
  item of the three because the work is already done next door.
- **Lever B — measure the existing m256 variant.** A `_m256` entry point
  (`M_TILE 256, WARPS 16`) already exists behind `HIPFIRE_BF16_MOE_M256=1` and
  is off by default. Nobody has published a number for it on this shape. Cheap
  to A/B.

Consult `reference_gfx1151_iu4_gemm_tuning` before starting: on gfx1151 the
wins were wave64 + double-buffered LDS + N-heavy 2×8 + BK64, and the recorded
DEAD ENDS were register-blocking, `__shfl`, and LDS bank-padding. Do not
re-derive those.

## 3. `attention_f32_routed_batched` — 23% whole-model, and it is scalar f32

`kernels/src/attention_f32_routed_batched.hip` (113 lines). The only hot kernel
with no WMMA and no packed math at all, and by far the most expensive single
call: 386 ms against the 8–59 ms of the WMMA GEMMs beside it. It runs on only
the 12 full-attention layers, which is why it is 23% and not the 38% the blended
sample suggested.

Three separate problems, in likely payoff order:

- **The PV loop's access pattern is the worst case.** Line ~106:
  `for d: for t: val += scores[t] * v_cache[t*n_kv_heads*head_dim + kv_h*head_dim + d]`.
  Each thread walks `t` with a stride of `n_kv_heads*head_dim` floats — 4 KB at
  8×128 — so every iteration is its own cache line. Stage V tiles through LDS,
  or transpose the loop so consecutive threads read consecutive `d`.
- **QKᵀ and PV are both GEMM-shaped and scalar.** `for t: for d: dot += q[d]*k[d]`
  is a rank-1 dot per thread. Both products are 16×16×16-tileable; the repo
  already has the builtins wired for gfx1151.
- **Softmax does two full block reductions with `__syncthreads()` inside the
  loop.** Fine when it is 1% of the kernel; worth revisiting only after the two
  above, since it will not be the limiter until then.

Note this is a **runtime** kernel, not calibration-only, so a win here also
helps serving on every full-attention layer. It ranked first on the blended
sample and only third once reweighted — still worth doing, and the serving
benefit is why it is not last.

## Sequencing

Unchanged by the reweighting, and now better justified: the cheapest item is
also the biggest.

1. Hessian symmetry (small diff, no numerical risk, ~16% end-to-end).
2. MoE GEMM N-heavy port (29%; the shape is already proven next door).
3. Attention PV staging, then WMMA (23%, and it also helps serving).
4. Batched expert factorization — see below; a quantize-side, not calibrate-side,
   cost, but the one that governs whether a 397B is practical.

## 4. LDLQ factorization — the quantize-stage cost, and why it is not Cholesky's fault

Not in the traces above (this is `hipfire-quantize`, not calibrate), but it
dominates the OTHER half of a conversion: ~7 h of the 122B run, against ~1.5 h
of calibration.

The unit of work is not the 1949 tensors — MoE expert weights are STACKED, so
`mlp.experts.gate_up_proj` expands to 256 per-expert factorizations. That is
48 × 256 = **12,288 Cholesky factorizations at K=3072**, ~9.7 GFLOP each.

Three things are already settled, so do not re-derive them:

- **Caching the factor across experts does not work.** Tried in `03f7a2612`,
  reverted in `dc18bdd39` with measurements: `oq4.25++` maps to the AwqLdlq
  recipe whose AWQ half rebases the Hessian per tensor (`H' = diag(1/s) H
  diag(1/s)`, damping from the rebased diagonal), so every expert has a
  different `H'` AND a different `λ`. The cache could never hit and its key
  alone made the build 2× slower.
- **The algebraic shortcut is invalid**: `(DHD + λI)⁻¹ ≠ D(H + λ'I)⁻¹D`.
- **GPU Cholesky is 34× SLOWER** (`HIPFIRE_GPU_CHOLESKY`, off by default): the
  trailing update is faster on device, but ~4500 block iterations each pay two
  device syncs and a ~4 MB panel round-trip.

Krylov methods do not substitute: LDLQ consumes the triangular FACTOR entry by
entry and propagates `(w−ŵ)/L[c,c]` to later columns in order, so it needs `L`,
not a solve, and the error feedback is inherently sequential within a tensor.

The parallelism is BETWEEN factorizations, not within one:

- **Batched factorization.** The 12,288 problems are independent. One K=3072
  Cholesky cannot fill the GPU without paying per-panel syncs; a batched
  `potrf` amortises them into one dispatch. Same flops, right shape.
- **`f64` → `f32`.** `inv_cholesky_dispatch` takes `h: &[f64]` and returns
  `Mat<f64>`. f64 is heavily rate-limited on gfx1151, so this may be several×
  for free — IF conditioning tolerates it against the damping. Measure the
  Hessian error, do not assume.
- Note `down_proj` is already RTN (no Hessian is captured for it), so the entire
  bill is `gate_up`.

Each needs a before/after `rocprofv3` slice using the recipe above, and
correctness checked with
`hipfire-coexistence artifact compare-calibration` against a pre-change
artifact — a calibration that is fast and subtly wrong is worse than a slow one.
