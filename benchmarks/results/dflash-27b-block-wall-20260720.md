# Qwen3.6-27B DFlash — measured NPU draft block wall (task #32)

**Date:** 2026-07-20 · **Branch:** chaingun · **Machine:** nix1 (gfx1103 Phoenix
APU + npu1/aie2, 4 columns) · **Tool:**
`crates/hipfire-xdna/examples/dflash_body_native.rs`

Measures the actual production target (Qwen3.6-27B) DFlash drafter block body on
the NPU, end-to-end, rather than deriving it from the 9B. Companion to the 9B
reality check (`dflash-npu-tokps-reality-check-20260720.md`) and the Phase 0
brief (`docs/plans/2026-07-19-dflash-phase0-brief.md`).

## Config (confirmed from the sidecar)

Qwen3.6-27B DFlash: H=5120, I=17408, NH=32, NKV=8, HD=128, 5 layers, block=16,
ctx=32, tot=48, target_layers=[1,16,31,46,61], θ=1e7. **32q/8kv/128 matches the
9B exactly**, so headnorm/rope/hnrope kernels and the (shape-independent) flash
attention kernel transfer unchanged. Drafter params 1.730 B (×1.65 vs 9B's
1.049 B). Sidecars already on disk:
`~/.hipfire/drafts/Qwen3.6-27B.dflash.f16.hfq` (3300 MiB golden),
`…dflash.npu.oq4.25+.hfq` (877 MiB).

## What was rebuilt for 27B

- **New-shape primitives** (only H- and I-shaped kernels change):
  `qwen35-rmsnorm-5120-b16`, `qwen35-rmsnorm-5120-b32`, `qwen35-swiglu-17408-b16`
  — built via `tools/npu/build_dflash_body_batched.py` (added `rmsnorm16_27b`,
  `rmsnorm32_27b`, `swiglu16_27b` variants). No device wedge.
- **GEMM geometry:** the 9B r14 kernel `r14_1x2x128_nb128` has **K_CHUNK=2048**,
  and every 9B K-dim (4096, 12288, 20480) is a multiple of 2048. **The 27B
  K-dims (5120, 17408, 25600) are NOT** — gcd = 1024. Fix: use the already-built
  **`r14_1x4x64_nb128` (K_CHUNK=1024)**, which divides all three (5120/1024=5,
  17408/1024=17, 25600/1024=25). No new r14 kernel build needed — pass
  `--r14-dir ~/.hipfire/npu/r14_1x4x64_nb128`.
- Golden regenerated at 27B weights: `dflash_ref_dump` on the f16 sidecar, **seed
  3523126622 (0xD1FEA55E — the harness default, same seed the 9B golden used)**,
  then `dflash_ref.py`. numpy-vs-GPU final max_abs 1.25e-2 (< atol 2e-2). OK.

## Parity gates

| path | cos vs f16 golden | cos vs int8/bf16 ref | gate |
|---|---|---|---|
| **int8** (`dflash_body_npu.py`, group proj) | **0.998653** | 0.998771 | **> 0.99 — PASS** |
| W4A8 (`--gemm multicore`, r14 K_CHUNK=1024) | 0.889649 | 0.889401 | acceptance-gated, not cosine |

Int8 body cos 0.998653 clears the >0.99 gate (on par with the 9B's 0.998083).
W4A8 is ~0.89 by construction (4-bit codebook loss); per the brief it is gated on
acceptance rate, not cosine. Note it sits ~0.009 below the 9B's 0.898 — plausibly
27B weight-distribution / the K_CHUNK=1024 int4 grouping, **not** a filed W4
regression (acceptance not measured here).

Context cache: cache-vs-recompute **bit-identical** (max|Δ| K=0, V=0 across all 5
layers); final block_hidden cached-vs-recomputed **cos = 1.000000000**.

## Measured block wall — W4A8 multicore + `--cpu-primitives` + `--ctx-cache` + `--attn flash`

`--blocks 6`, warm = cached steady state (blocks 2–5), reported separately from
cold. `ctx_misses = 0` in every warm block.

| block | wall | npu_busy | note |
|---|---|---|---|
| 0 | 291.6 ms | 241.9 ms | **cold** (weight fault-in, JIT) |
| 1 | 168.0 ms | 118.6 ms | warm, **no-cache** (ctx recompute) |
| 2–5 | **136.1–138.1 ms** | 96–99 ms | warm, **cached steady state** |

- **27B warm block wall (cached) = 136.1 ms** (cold 291.6 ms, reported separately).
- Context cache removes the L-axis: warm no-cache 168.0 → cached 136.1 ms (−31.9 ms,
  1.2×) — and structurally flat in L thereafter (only new query rows recompute).
- CPU primitives (host, single-core, all 36 calls) = **4.34 ms/block**.
- Per-op (warm, r14 W4A8): gate/up `N34816_K5120` 0.76 ms, down `N5120_K17408`
  0.76 ms, o `N5120_K4096` 1.79 ms, qkv `N6144_K5120` 0.76 ms; flash attention
  1.44 ms/dispatch (~7 ms/block). Wall ≈ ~96 ms NPU (r14-dominated) + ~4 ms CPU
  glue + host overhead; dispatches/block 110.

### 9B vs 27B (same harness, same wins)

| | 9B (brief) | 27B (measured) | ratio |
|---|---|---|---|
| warm, no-cache | 111.9 ms | 168.0 ms | 1.50× |
| warm, cached steady state | ~155–160 ms (projected) | **136.1 ms (measured)** | — |
| drafter params | 1.049 B | 1.730 B | 1.65× |

The no-cache scaling (1.50×) is **better than the ×1.65 param ratio** — the GEMM
is weight-bandwidth-bound and the cpu-primitive/ctx-cache wins do not scale with
model size. The brief's *derived* 27B figure was ~273 ms; the measured cached
wall (136 ms) is ~2× under it, because the derivation did not apply the context
cache (task #25) and over-scaled the L-axis GEMMs.

## Verdict — draft vs its real block-verify budget

The corrected budget (task #31): the draft races the GPU **block-verify wall**
(batched-16), not the single-token AR forward. The 9B's single-token 57 ms → real
block-verify ~100 ms (1.75×). For 27B the single-token AR forward is the
older-notes **155 ms**; **no runnable 27B target hfq is on disk** (see below), so
the block-verify wall is **PROJECTED = 155 × 1.75 ≈ 271 ms** (labeled projection,
not measured).

- **27B cached draft 136.1 ms vs 27B block-verify ~271 ms (projected) → draft is
  ~0.50× of the verify wall — UNDER budget with ~2× headroom.** Under overlap
  (draft block N+1 while the GPU verifies block N) the draft is **free**; the step
  is bounded by the verify wall.
- Even against the stricter (wrong) single-token 155 ms budget: 136 < 155 →
  **under by 1.14×.** Contrast the 9B, which is 3.25× over its 57 ms single-token
  budget. **27B is the first DFlash pair whose NPU draft fits under its own
  budget**, on the honest verify-wall comparison and even on the pessimistic
  single-token one.

## Blocker — 27B target conversion (item 1) not completed

The coexistence importer is **GGUF-only** (`hipfire-coexistence import gguf`).
On disk there is only the 27B **safetensors** (52 GB, 15 shards) and a 13 MB
imatrix — **no weight GGUF** (the unsloth GGUF repo is lock-only, not
downloaded). Converting would need a large download or an external
safetensors→GGUF step, then an oq4 quantize whose f16 intermediate risks the
documented 64 GB UMA OOM. Deferred as excessive-effort; the verify wall is
projected accordingly. The **int8 losslessness/parity path did not require the
target** — it is gated by the int8 body cos (0.998653) against the f16 golden.

## Reproduce

```
# 1. golden inputs (GPU) + numpy golden
dflash_ref_dump ~/.hipfire/drafts/Qwen3.6-27B.dflash.f16.hfq --seed 3523126622 --out /tmp/dflash27b_ref
tools/npu/dflash_ref.py --ref-dir /tmp/dflash27b_ref --weights /srv/huggingface/models--z-lab--Qwen3.6-27B-DFlash
( cd /tmp/dflash27b_ref/golden && for f in *.npy; do ln -sf "$f" "rust_$f"; done )
mkdir -p /tmp/dflash27b_ref/rust && cp /tmp/dflash27b_ref/golden/final_block_hidden.npy /tmp/dflash27b_ref/rust/rust_final_block_hidden.npy

# 2. new-shape kernels (fork toolchain, ~/.venv python)
python tools/npu/build_dflash_body_batched.py --which rmsnorm16_27b   # + rmsnorm32_27b, swiglu16_27b

# 3. weights + manifest + int8 ref (fork venv312 python)
tools/npu/dflash_body_npu.py --golden-dir /tmp/dflash27b_ref/golden --weights <27B DFlash safetensors> --dump-weights /tmp/dflash27b_w
tools/npu/dflash_body_npu.py --golden-dir /tmp/dflash27b_ref/golden --weights <…> --batch-ops --dump-manifest /tmp/m.json --dump-ref /tmp/ref_int8.npy
tools/npu/swap_attn_flash_manifest.py --in /tmp/m.json --out /tmp/m_flash.json --tot 48 --block 16 --n-q 32 --q-len 16 --kv-tile 48 --cores 4

# 4. measure (native, GPU/NPU lock held)
target/release/examples/dflash_body_native --manifest /tmp/m_flash.json --weights /tmp/dflash27b_w \
  --golden /tmp/dflash27b_ref --ref /tmp/ref_int8.npy \
  --gemm multicore --r14-dir ~/.hipfire/npu/r14_1x4x64_nb128 \
  --cpu-primitives --ctx-cache --attn flash --blocks 6
```
