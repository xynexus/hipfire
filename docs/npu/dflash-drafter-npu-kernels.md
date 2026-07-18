# DFlash / DSpark drafter on the NPU — kernel design

Design for running the block-diffusion speculator body (DFlash, and DFlash +
DSpark heads) on the AIE2 NPU (npu1 / Phoenix / gfx1103). Grounded in
`tools/npu/aiecost/` (cost model), the measured verify budgets below, and the
existing `tools/npu/build_qwen3_*` kernel inventory.

## Why this is viable (the numbers)

DFlash (z-lab/dflash; `crates/hipfire-runtime/src/dflash.rs`) is a **block/masked
diffusion drafter**: it noise-inits `block_size` positions and denoises the whole
block in ONE bidirectional (non-causal) forward, with per-layer cross-attention
over a few of the target's hidden layers. So the projections are **M = block_size**
(weights reused across the block), not M=1 decode.

The verify budget is the target's OWN decode on this hardware — slow, so the NPU
draft hides under it:

| target on nix1 | decode | verify budget / block | real ~1B NPU draft (16 ms/block) |
|---|---|---|---|
| qwen3.5-9B-mq4 (measured) | 17.6 tok/s, 56.6 ms/tok | ~57 ms | hides, 3.5× margin |
| 27B mq4 (BW-law, 87 GiB/s) | ~6.4 tok/s | ~155 ms | hides, ~10× |
| 31B-oq8 (BW-law) | ~2.9 tok/s | ~345 ms | hides, ~20× |

So NPU-draft ‖ GPU-verify **wins**: the GPU is freed during the ~16 ms draft and
only pays the (much longer) verify, while the draft runs on the NPU's low-power
rails (~107 mJ/block int4). `python -m aiecost.design drafter --block-size 16
--gpu-verify-us 56600` reproduces the pipelining check.

## Build target: the REAL trained drafter (not the tiny greenfield)

The real z-lab drafter (ships with 9B/27B targets) is a 5-layer Qwen3 block:
dim 4096, FFN 12288, GQA 32 q / 8 kv × 128, block_size 16, 8-bit (Q8F16) weights.
Two reasons it beats the tiny (h512/3L) config as the first target:
1. **Trained weights exist** → end-to-end validation against the reference forward,
   not just numeric self-consistency on random weights.
2. **It maps onto the existing Qwen3 NPU kernels directly** — same 8 kv heads /
   128 head_dim the `build_qwen3_segmented_attention` image already requires; the
   tiny config's 4 kv / 64 head_dim does not.

The tiny greenfield config stays a fast warm-up / floor-bound datapoint, built
from the same primitives with smaller dims once the real path works.

## Op → kernel map (per layer, M = block_size = 16)

The drafter body is a Qwen3 transformer block. Most primitives already exist as
validated npu1 kernels under `tools/npu/`:

| op | kernel | status |
|---|---|---|
| input rmsnorm | `build_qwen35_rmsnorm` (`rms_norm_weighted_bf16.cc`) | exists |
| q/k/v/o proj (M=16) | `build_qwen3_oq8_projection` (`generate_mlir(m,k,n)`, int8→f32) | exists, set m=16 |
| q_norm / k_norm | `build_qwen35_headnorm` (`rms_norm_head_bf16.cc`) | exists |
| RoPE | `build_qwen35_rope` (`rope_rotate_bf16.cc`) | exists |
| **block+cross attention** | from `build_qwen3_segmented_attention` | **NEW variant** — see below |
| softmax | `build_qwen35_softmax` (`softmax_bf16.cc`) | exists (used by attn) |
| gate/up/down + SiLU | `build_qwen35_swiglu` (`silu_mul_bf16.cc`) | exists |
| residual adds | fused into rmsnorm / proj epilogue | exists pattern |

### The one genuinely new primitive: non-causal cross-attention

`build_qwen3_segmented_attention` is **causal** and self-only. The drafter needs:
- **Non-causal** (bidirectional) attention within the block — the diffusion block
  attends both directions. Drop the causal mask.
- **Cross-attention context**: K/V = concat(projected target_hidden context,
  current block K/V). Q length = block_size (16), K/V length = ctx_len + 16. The
  context K/V come from the caller's projected `target_hidden` (layers e.g.
  [1,8,15,22,29]) — staged once, read by every layer.
- **Noise-init** of the block hidden at entry (`[block_size, hidden]`), a host-side
  buffer fill, not a kernel change.

This is a mask change + a K/V staging change on the existing attention image, plus
q_len = 16 (vs the segmented image's token chunks). It reuses the same QK^T /
softmax / AV micro-kernels.

## Fusion plan (the imperative)

Per the cost model, the whole game is **minimising dispatches**. The tiny body is
100% floor-bound (92% floor tax unfused); the real body is 27% floor tax at block
16. Either way, fuse the block body into as few dispatches as the L1/memtile
budget allows, keeping the `[16, hidden]` activation resident across the layer:

- Stage 1: fuse **rmsnorm → qkv proj → q/k norm → RoPE** into one dispatch (the
  activation is small, `[16, 4096]` = 128 KiB f32; K/V context is the large read).
- Stage 2: **attention** (its own dispatch — different dataflow shape, memtile K/V).
- Stage 3: fuse **o-proj → residual → rmsnorm → gate/up → SiLU → down → residual**.

Target: **~3 dispatches/layer** (down from ~8), 5 layers → ~15 dispatches/block
vs ~40 unfused. Whole-body single-dispatch is the asymptote but the attention
dataflow likely forces a boundary; ~3/layer already recovers most of the floor
tax. Validate the actual dispatch count against the model's fused-per-layer row.

## DSpark variant (dflash + dspark)

DSpark = the DFlash block body + two small heads run on the final block hidden:
- **markov heads** — small GEMMs producing per-position draft logits/features.
- **confidence head** — a projection + threshold that truncates the block early
  when confidence drops (the `mean_confidence < conf_threshold` gate). This is a
  small GEMM + argmax/reduce/compare; a new tiny kernel or a host-side reduce over
  the block hidden. Early truncation shortens the effective block, which the model
  treats as a smaller block_size.

Build order: DFlash body first (validated end-to-end vs reference), then add the
DSpark heads as an epilogue dispatch.

## Validation

- Per-primitive: the existing `test_*_npu.py` already validate rmsnorm/rope/
  softmax/swiglu/projection/attention numerically on npu1.
- Body: numeric parity of the fused block forward vs `hipfire-runtime::dflash`
  reference (F32 GPU path) on a staged z-lab sidecar, per position, over a fixed
  seed block. Tolerance per the existing kernel tests.
- End-to-end: acceptance rate / τ unchanged vs the GPU drafter in
  `dflash_spec_demo` when the NPU draft is swapped in (a later integration step —
  NPU is a separate execution backend from the HIP hot path).

## Open questions / risks

- **LDS hazard on nix1** (AGENTS.local): LDS-heavy kernels can wedge gfx1103 — but
  these are NPU (AIE) kernels, not GPU LDS kernels, so the hazard does not apply to
  the drafter itself; it only matters for the GPU-side verify/reference.
- **Attention dataflow** may not fuse into the proj dispatches (different tiling) —
  the ~3-dispatch/layer plan assumes a boundary there; confirm against build.
- **int8 vs int4**: the sidecar is Q8F16 (8-bit) → the int8 projection kernel
  matches directly; an int4 (OQ4/MQ4) drafter would double feed headroom but needs
  the `mmul<4,16,8,int8,int4>` path and a re-quantised sidecar (later lever).
