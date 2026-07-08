# NPU offload scoping for embeddinggemma-300m

Status: **scoping only — no NPU execution exists for this model yet.** GPU path is
landed and validated (see `crates/hipfire-arch-embeddinggemma`, example `embed_e2e`).
This doc records what an NPU embedding path would take, why it isn't runnable today,
and the phased plan to get there.

## Why we cannot bench on the NPU today (nix1, gfx1103 Phoenix + XDNA1)

The NPU **hardware** is present and healthy (`/dev/accel/accel0`, amdxdna driver
fw 1.5.5.391), but every software prerequisite for running a kernel is missing:

- **No compiled AIE kernels.** The R6-TS W4A8 GEMM (`benchmarks/npu_gemm_tuning/r6/
  r6_gemm_ts.cc` + `r6_gen_mp.py`) ships as *source*; there is no `final.xclbin` /
  `insts.bin` in the repo, `~/.hipfire`, or `/tmp`. `NpuKernel::load` needs both.
- **No MLIR-AIE toolchain.** `aiecc`, the `aie` Python package, and peano/llvm-aie
  are not installed, so `r6_cache.sh` cannot compile a kernel. See the
  `npu-kernel-build` skill for the toolchain bring-up.
- **No XRT xdna runtime** (`libxrt_driver_xdna.so`) and `HIPFIRE_XDNA1_LIB` is unset.
- **No encoder NPU path.** `embed_forward` runs entirely on HIP/GPU kernels. The only
  NPU offload wired into hipfire inference anywhere is a single SwiGLU FFN op
  (`XDNA_SWIGLU_BACKEND`, qwen35). There is no transformer-on-NPU execution loop.

## What the NPU *can* do (once artifacts exist)

The runtime-callable primitive is `hipfire_xdna::NpuGemmMp` — an **M-parallel,
W-broadcast W4A8 GEMM** (int4 weights, int8 activations), A/C row-major, weights
packed once. Measured ~1.45 TOPS e2e on halo (weight-bandwidth-bound), up to
~20.7 TOPS for the array topology. Shape contract: `K == k()` (single K-chunk),
`N == n()`, `M % rows_per_dispatch() == 0`. **Only GEMM** — there is no AIE kernel for
attention (softmax/online-softmax), RoPE, RMSNorm, GeGLU, or pooling.

## GEMM inventory of one embeddinggemma-300m encode (m tokens)

Per layer × 24, then the two Dense heads once. All are `[out, in]` weight matmuls
(`y = x·Wᵀ`), M = number of tokens:

| GEMM | out×in | notes |
|---|---|---|
| q_proj | 768×768 | attention |
| k_proj / v_proj | 256×768 | GQA (1 kv head × 256) |
| o_proj | 768×768 | |
| gate_proj / up_proj | 1152×768 | GeGLU |
| down_proj | 768×1152 | K=1152 (multi-K-chunk for NpuGemmMp) |
| dense.0 (once) | 3072×768 | ST Matryoshka head |
| dense.1 (once) | 768×3072 | ST Matryoshka head |

These 7 per-layer GEMMs dominate the encode FLOPs and are the natural offload set.
Attention itself (QKᵀ, softmax, ·V), the 6 norms/layer, RoPE, and GeGLU stay on GPU.

## Blockers specific to embeddinggemma

1. **dtype.** Weights are bf16/f16 today; the NPU GEMM is W4A8. Offload requires an
   int4 weight + int8 activation path. hipfire already has int4 weight formats
   (oq4/mq4) and the offline quantizer; the open question is embedding **quality**
   under W4A8 (the cosine-separation gate must hold — the Dense heads especially are
   sensitive and are currently kept near-lossless).
2. **Per-op host round-trips.** With only GEMM on the NPU, each offloaded matmul is a
   GPU→host→NPU→host→GPU hop unless a zero-copy dmabuf path (`npu_gemm_mp_zerocopy`)
   is wired. For a 300M encoder the GEMMs are small (M = a few hundred tokens), so
   dispatch/transfer overhead likely dominates — the win is uncertain and must be
   measured, not assumed.
3. **K-chunking.** `NpuGemmMp` is single-K-chunk; K=1152 (down_proj) and K=3072
   (dense.1) need the tiled `NpuGemm::run` path or K-accumulation.

## Phased plan

- **P0 — toolchain.** Install MLIR-AIE/aiecc + XRT xdna (npu-kernel-build skill);
  compile one W4A8 GEMM via `r6_cache.sh`; smoke `npu_gemm_verify` /
  `npu_gemm_bench` to confirm the device path end-to-end.
- **P1 — op bench.** Bench the model's actual GEMM shapes (768×768, 1152×768,
  3072×768) on NPU via `npu_gemm_bench` vs the GPU `weight_gemm` equivalent at
  M ∈ {32, 128, 512}. Decide per-shape whether NPU wins after transfer overhead.
- **P2 — offload the Dense heads.** They run once per encode and are the largest
  single GEMMs (3072×768, 768×3072); requantize to W4A8, validate the cosine gate,
  route through `NpuGemmMp`, measure NPU vs GPU + the CPU host-matmul baseline.
- **P3 — offload per-layer projections** if P1/P2 show a real win, with a zero-copy
  dmabuf path to kill the round-trips.

## Recommendation

Keep embeddinggemma on the GPU (validated, ~18 ms warm encode). Treat NPU offload as
a separate, measurement-driven effort gated on P0 toolchain bring-up. Do not claim an
NPU number until P1 produces one — the current answer is "not runnable," not "slow."
