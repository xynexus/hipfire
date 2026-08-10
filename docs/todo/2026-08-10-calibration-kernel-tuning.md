# TODO: tune the three kernels that own streamed calibration

Opened 2026-08-10 from a `rocprofv3 --kernel-trace` of the live
Qwen3.5-122B-A10B `oq4.25++` calibration. These three kernels are **87.8%** of
GPU time, and none of them has had a tuning pass.

## The measurement

Two layers (24–25) of the real 122B run, profiled in place. 92.3% of the
profiled span was GPU-busy on kernels (258.3 s kernel / 279.7 s span), so this
is compute-bound — not I/O-bound and not launch-bound.

| kernel | share | total | calls | avg |
|---|---|---|---|---|
| `attention_f32_routed_batched` | **38.3%** | 98.8 s | 256 | 386 ms |
| `calib_hessian_outer_f32` | **26.0%** | 67.2 s | 9,672 | 6.9 ms |
| `gemm_bf16_moe_grouped_wmma_gfx1151` | **23.5%** | 60.7 s | 1,024 | 59.3 ms |
| `gemm_bf16_x_bf16_wmma_gfx1151_nheavy` | 5.7% | 14.7 s | 1,792 | 8.2 ms |
| `__amd_rocclr_copyBuffer` | 1.4% | 3.5 s | 2,482,243 | 1.4 µs |

Two traps for whoever reads this DB next:

- `top_kernels.total_duration` is in **microseconds**. Reading it as ns makes
  the job look 0.09%-GPU-busy and launch-bound, which is wrong by 1000×.
  Cross-check against `SUM(end-start)` on `rocpd_kernel_dispatch_*`, which is ns.
- The 2.48M `copyBuffer` calls look alarming and are not: 1.4 µs each, 1.4% of
  time. Call count is not cost.

Absolute times carry ~47% profiler overhead (279.7 s profiled vs ~190 s
unprofiled for two layers). Proportions are sound; don't quote the seconds.

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

Calibration resumes from its boundary checkpoint, so a profiled slice costs
nothing but the profiler overhead — the layers it does are real work.

## 1. `calib_hessian_outer_f32` — 26%, and half the work is redundant

`kernels/src/calib_reduce.hip:66`. Start here: it is the cheapest real win in
the file.

`H = XᵀX` is **symmetric**, and the kernel computes the whole thing. Every
`(i,j)` is written (`H[i*K+j] += acc`), and the launch grid is the full
`⌈K/16⌉ × ⌈K/16⌉` (`dispatch/mod.rs:3430`). Nothing exploits `H[i,j] == H[j,i]`.

- **Lever A — skip the mirror half.** Launch only blocks with
  `blockIdx.y <= blockIdx.x` and mirror-write, or keep the square grid and early-
  `return` the strictly-lower blocks after writing both triangles. ~2× on 26% of
  runtime ⇒ ~13% end-to-end. Check first whether any consumer reads the lower
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

## 2. `attention_f32_routed_batched` — 38%, and it is scalar f32

`kernels/src/attention_f32_routed_batched.hip` (113 lines). The single largest
consumer, and the only hot kernel with no WMMA and no packed math at all.
386 ms per call is enormous next to the 8–59 ms of the WMMA GEMMs beside it.

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
helps serving on every full-attention layer — the reason it tops this list is
that calibration hammers it, not that calibration is special.

## 3. `gemm_bf16_moe_grouped_wmma_gfx1151` — 23%, missing the N-heavy treatment

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

## Sequencing

1. Hessian symmetry (small diff, no numerical risk, ~13% end-to-end).
2. MoE GEMM N-heavy port (the shape is already proven next door).
3. Attention PV staging, then WMMA (largest prize, largest change).

Each needs a before/after `rocprofv3` slice using the recipe above, and
correctness checked with
`hipfire-coexistence artifact compare-calibration` against a pre-change
artifact — a calibration that is fast and subtly wrong is worse than a slow one.
