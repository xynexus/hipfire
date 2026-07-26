# TODO: explore the STEEL fused-attention math for larger models / longer

Date: 2026-07-15. Follow-up from the EmbeddingGemma M256 NPU work (NPU-RESULTS
R57–R121, the route-A dispatch-fusion scoping, and the STEEL benchmark below).

## What STEEL is

STEEL (arXiv:2607.09385, "Sparsity-Aware Fused Attention for Energy-Efficient
Long-Sequence Inference on AMD's XDNA NPU", Jung et al.) is the first
open-source FlashAttention for XDNA-like NPUs. Open-source in upstream
`github.com/amd/iron` (a **newer** IRON than our vendored `third_party/IRON`,
which only carries the old head_dim=64 `mha` operator). Same toolchain we build
(IRON + MLIR-AIE + LLVM-AIE).

- 3-stage spatial pipeline (Q@Kᵀ → online-softmax → P@V), one AIE core per stage,
  ~10 pipelines over the 32 tiles; K/V broadcast + Q distribute via IRON
  primitives; no off-chip materialization of the score matrix.
- Headline: **22.8× over layer-by-layer** attention (XDNA2), **19.4× less
  off-chip traffic** (9.7 GB → 0.5 GB at N=4096), 1.75× energy vs the RDNA 3.5
  GPU. bf16 only. Tested **2048–32768 tokens**; no data below 1024. Sparsity-aware
  placement (its novel 38%) is **causal-only**.

## Why it does NOT help our current target (and why it's parked, not adopted)

Benchmarked at our shape (M256, head_dim 256, GQA 3:1, bidirectional) on halo:
the attention **core** math is ~5 µs of compute; a full unfused core is ~16 ms of
pure XRT dispatch overhead; the amdxdna-direct resident attention *block*
(QKV+core+O+2 norms) already fuses it into one 9.14 ms dispatch. At M256 the
attention core is **dispatch-bound, not compute- or bandwidth-bound**, and the
256×256 score matrix is on-chip trivially — so STEEL's two wins (op-fusion of the
core, avoiding O(N²) off-chip) buy ~nothing here, and its regime starts at ≥2048
tokens. Conclusion: **do not adopt STEEL for M256 encode.**

## Where STEEL's math IS worth exploring

The value is at **larger head_dim / more heads / longer context / higher batch** —
i.e. the regime where attention stops being dispatch-bound and becomes
compute + KV-bandwidth bound:

1. **Long-context encode / decode on the NPU** — RAG / long-doc / agentic
   contexts (8k–32k), where the O(N²) score matrix genuinely spills off-chip and
   STEEL's 19.4× traffic reduction bites. This is the natural home.
2. **Larger models** — bigger head_dim (256+), more heads, deeper stacks, where
   the per-attention compute is real work rather than launch overhead.
3. **Batched prefill / batched embeddings** — see the batch-size note below;
   batching fills STEEL's 10 pipelines (at M256 batch-1 only ~4 Q-tiles exist,
   underutilizing the array).

## The int8 / W4 / KVarN-4 upgrade (do not keep it bf16)

STEEL is bf16; our stack is Opus-quant int8. The upgrade is a **direct win in the
long-context regime** where the core is bandwidth/compute-bound:

- **P@V → A8(P)×W8(V)** on the existing `aie::mmul<4,16,8,int8,int8>` (r11 `-DW8`).
- **Q@Kᵀ → A8(Q)×W4(K)** on the existing `aie::mmul<4,16,8,int8,int4>` (r11
  default). This is **KVarN-4 on K**: it (a) halves the **K-broadcast bandwidth**,
  which is STEEL's actual bottleneck (42/48 Mem-tile ports), and (b) runs Q@Kᵀ at
  ~2× the int8 MAC rate. V stays Q8 (KVarN's asymmetric K4/V8).
- Softmax stays fp32/bf16 (online-softmax numerics).
- **Encode has no KV cache** (bidirectional, single pass), so KVarN-4's
  "non-recoverable cache floor" (the reason we deploy KVarN-8 for *decode*, see
  the light-QAT-recovery findings) does not apply — 4-bit K is far more
  defensible for encode. For long *decode*, re-check the KV4 quality floor.
- Mixed precision (oq4.5++ etc.) falls out of the Opus swizzle: the int8 mmul
  consumes 8-bit-aligned weights regardless of stored bit-width, so W4/W8/mixed on
  the projections is orthogonal to this attention-core work.

## Concrete exploration steps (when a long-context / batched NPU target lands)

1. Pull the STEEL design from upstream `amd/iron`; get it compiling with our
   from-source `build/bin/aiecc` (the memref.view / full-ELF fix is already in).
2. Adapt shape: head_dim 256 (STEEL tested 64 — tile/local-mem pressure unknown),
   GQA 3:1 (maps favorably onto "broadcast K/V, distribute Q"), **bidirectional**
   (drop the causal mask + sparsity-aware placement; use uniform placement).
3. Benchmark bf16 at 2k / 4k / 8k / 16k tokens vs the resident path and vs the
   RDNA GPU — find the crossover length where fused-NPU attention wins.
4. Swap the two GEMMs to int8 (A8×W8) + KVarN-4 K (A8×W4); measure the traffic
   and latency delta. Validate int8 softmax-input numerics against the fp32 oracle.
5. Decide whether a hipfire↔IRON bridge (author in IRON, compile offline, dispatch
   via amdxdna) is worth productionizing for the long-context path.

## Open questions

- head_dim=256 tiling within 64 KB tile local memory (STEEL assumed 64).
- int8 quantization of Q/K/V/P without hurting softmax — where to place scales.
- Bidirectional dense: does dropping sparsity-aware placement cost pipeline
  balance, or is uniform placement already optimal for dense?
- Batch: block-diagonal (per-sequence) attention vs one big S_total² matrix.

## Batch-size note (batch 1 → 128/256) — the regime shift that makes this pay

Almost all our NPU work is **batch=1**, which is the NPU's *worst* case:
weights (8 MB attn + 11 MB FFN per layer) are read once and applied to only 256
rows, so the pipeline is **weight-bandwidth + dispatch bound**, far below the
~55-TOPS int8 peak. At **batch 128/256** the *same* per-layer weights amortize
over 128–256× more activation rows: arithmetic intensity jumps ~100×, the GEMMs
become **compute-bound**, the fixed per-dispatch overhead spreads ~100× thinner,
and the array actually approaches peak TOPS. That is the operating point where
int8's 2–4× compute, W4 weights, STEEL's pipeline fill, and KVarN-4 finally
deliver — and embeddings are *naturally* batched (you embed corpora, not single
strings). See the separate batched-NPU-encode todo for the kernel work (the
resident kernels are hard-pinned to `rows()==256`; batch needs M=B·S tiling or a
batch loop, block-diagonal attention, and activation staging that scales).
