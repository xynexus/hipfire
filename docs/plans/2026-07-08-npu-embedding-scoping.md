# NPU offload scoping for embeddinggemma-300m

Status: **toolchain works and is validated on-device; GEMM path needs a version
port before a representative bench.** GPU path is landed and validated (see
`crates/hipfire-arch-embeddinggemma`, example `embed_e2e`). This doc records the NPU
environment state, a real on-device measurement, why embeddinggemma isn't served on
the NPU yet, and the phased plan.

## NPU environment state (nix1, gfx1103 Phoenix + XDNA1 / NPU1 / AIE2)

The NPU is **present, accessible, and the toolchain compiles + runs kernels
end-to-end** (an earlier draft of this doc wrongly reported the toolchain absent —
that was a probe error: `aiecc.py` vs `aiecc`, and the system python vs the
`~/.venv` toolchain):

- Hardware: `/dev/accel/accel0` readable (user in `render`); pyxrt reports
  `RyzenAI-npu1`. amdxdna driver fw 1.5.5.391.
- Toolchain (in `~/.venv`, per the `npu-kernel-build` skill): `aiecc`, Peano
  clang++ (`llvm-aie`), `mlir_aie` **1.3.1**, `xclbinutil` (`/opt/xilinx/xrt/bin`,
  prepend to PATH), pyxrt — all functional.
- **Validated on-device:** built `qwen35-rmsnorm-768.xclbin` via
  `tools/npu/build_qwen35_rmsnorm.py --hidden-size 768` and ran
  `test_rmsnorm_npu.py` → PASS (max_abs_err 0.031, within tol), **NPU mean 184 µs,
  p50 180 µs**. This proves aiecc → xclbin → hw_context → dispatch works.

### The key result: small ops are dispatch-bound

184 µs for a 768-element RMSNorm is almost entirely the **~180 µs NPU dispatch
floor** — the actual compute is negligible. This is the crux of the offload
question: embeddinggemma's per-op work (a 768-wide norm, a rope, one projection over
a few hundred tokens) is small, so **only batched GEMM** (B tokens amortizing the
floor) can plausibly win. Per-op offload of norms/rope/softmax loses to the GPU.

## Blockers to an embeddinggemma NPU bench

1. **GEMM script is version-drifted.** `tools/npu/oq_gemm_design.py` (the int8/OQ8
   NPU GEMM) targets a newer IRON API — `CompileTime`, `In`/`Out`, `kernels.mm`,
   `aie.iron.controlflow.range_`, `aie.helpers.taplib.TensorTiler2D`,
   `aie.utils.benchmark.run_iters` — none of which the installed `mlir_aie 1.3.1`
   exports (1.3.1 uses `Program`/`Runtime`/`Worker`/`ObjectFifo`/`ExternalFunction`).
   The elementwise/reduction scripts (rmsnorm, rope, softmax, swiglu) DO work on
   1.3.1. So the dominant-cost GEMM bench needs either (a) installing the mlir_aie
   version `oq_gemm_design.py` targets (the wheel index
   `github.com/Xilinx/mlir-aie/releases/.../latest-wheels` is reachable — HTTP 200 —
   but a version bump risks desyncing from the pinned Peano), or (b) porting
   `oq_gemm_design.py` to the 1.3.1 `Worker`/`ObjectFifo` matmul API.
2. **No encoder NPU path.** `embed_forward` is HIP-only; the only NPU offload wired
   into hipfire inference is a single SwiGLU op (`XDNA_SWIGLU_BACKEND`, qwen35).
3. **dtype.** Weights are bf16/f16; the NPU GEMM is int8 (OQ8) / W4A8 — offload
   needs an int8-activation + int-weight path, with the cosine-quality gate re-checked.

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

- **P0 — toolchain. DONE.** MLIR-AIE/aiecc + XRT + pyxrt present in `~/.venv`;
  validated by compiling + running `qwen35-rmsnorm-768` on the NPU (PASS, 184 µs).
- **P0.5 — unblock the GEMM.** Port `tools/npu/oq_gemm_design.py` to the installed
  `mlir_aie 1.3.1` `Worker`/`ObjectFifo` matmul API (or install the matching
  mlir_aie version — wheel index reachable, but watch Peano pin). Then
  `bench_oq_gemm_npu.py` compiles.
- **P1 — op bench.** Bench the model's actual GEMM shapes (768×768, 1152×768,
  3072×768, 768×3072) on NPU via `bench_oq_gemm_npu.py` vs the GPU `weight_gemm`
  equivalent at B ∈ {32, 128, 512} tokens. The RMSNorm result (184 µs dispatch
  floor) predicts small-M GEMMs lose; the question is the crossover B.
- **P2 — offload the Dense heads.** They run once per encode and are the largest
  single GEMMs (3072×768, 768×3072); requantize to W4A8, validate the cosine gate,
  route through `NpuGemmMp`, measure NPU vs GPU + the CPU host-matmul baseline.
- **P3 — offload per-layer projections** if P1/P2 show a real win, with a zero-copy
  dmabuf path to kill the round-trips.

## Prior hipfire NPU-GEMM findings (already answer most of this)

hipfire already characterized the W4A8 int8×int4 GEMM on **both** NPU generations
(`benchmarks/npu_gemm_tuning/r6/README.md`):
- **R5 on this Phoenix box (XDNA1/gfx1103):** a working streaming K-cascade W4A8
  GEMM, but the int8×int4 mmul-op throughput **ceilings at ~0.6 TOPS** (~32
  cycles/mmul on AIE-ML gen1). That is the op-rate wall on this NPU.
- **R6 targets NPU2/aie2p (halo/Strix Halo)** — `r6_gen.py` emits `aie.device(npu2)`
  and `r6_cache.sh` compiles `--target=aie2p` — to test whether Strix runs the same
  op faster. It does **not** build for this Phoenix box as-is.

Combine that with the on-device dispatch floor measured here (~180 µs/op) and the
embeddinggemma GEMM inventory: the largest single GEMM (dense.0, 3072×768 ≈ 2.36M
MACs) is ~8 µs of compute at 0.6 TOPS but pays the ~180 µs dispatch floor, and the
encode has dozens of GEMMs. Offloading them individually to the Phoenix NPU is
strictly slower than the GPU's ~18 ms whole-encode. **On this box the answer is
already "no."** The open question R6 chases — does Strix/NPU2 clear the op ceiling —
is a **halo** experiment, not a nix1 one.

## Why a live GEMM bench is blocked on this box (for the record)

Two independent blockers, both real:
1. **iron-API GEMM script (`tools/npu/oq_gemm_design.py`) is version-orphaned.** It
   needs `aie.iron.kernels.linalg.mm` + `CompileTime`/`In`/`Out`/`run_iters` — absent
   from `mlir_aie 1.3.1`, from the newest dev wheel (2026-02, verified by inspection),
   and from the current source-tree path (`python/aie/iron` 404s). It targets a
   specific historical mlir-aie branch/commit.
2. **The R6 GEMM kernel targets NPU2/aie2p (halo), not NPU1/aie2 (this Phoenix box).**
   Running it here would need an aie2 port of `r6_gemm*.cc` + `r6_gen.py` device.

The rmsnorm path works because it uses the `transform_*` helpers that DO exist in
1.3.1 and compiles for `aie2` (npu1).

## Recommendation

Keep embeddinggemma on the **GPU** (validated, ~18 ms warm encode). This is now a
data-backed call, not a punt:
- On-device measurement here: RMSNorm on the NPU = 184 µs, ~180 µs of it the dispatch
  floor. Small per-op offload loses.
- Prior hipfire finding: the Phoenix NPU W4A8 GEMM op ceilings at **~0.6 TOPS** (R5).
- embeddinggemma's GEMMs are small (a few hundred tokens × ≤3072 dims); dozens of them
  each paying the ~180 µs floor is strictly worse than the GPU's 18 ms whole encode.

**Do not invest in an NPU embedding path on this Phoenix box.** If NPU embedding is
ever revisited, do it on **halo (Strix Halo / NPU2 / aie2p)** where the R6 GEMM already
targets and might clear the op ceiling — a separate, halo-hosted experiment. The
`oq_gemm_design.py` iron-API port is not worth doing: no wheel ships its API, and the
answer on Phoenix is already "no."
