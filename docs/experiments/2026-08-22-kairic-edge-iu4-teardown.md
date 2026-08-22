# Kairic Edge (ROCmFPX) teardown — what they do that we do not

Source: `github.com/ciru-ai/ROCmFPX`, branch `kairic-edge-qwen38-27b-v1`, HEAD
`e97b324` "release: add Kairic Edge IU4 lane for Qwen3.8 27B". **MIT licensed**
(a llama.cpp / ggml fork), so unlike hipEngine (AGPLv3) its techniques *and* code
are usable with attribution. Same model as ours, same device (`gfx1151`, Radeon
8060S, 40 CU).

## The headline difference: ONE scale over the whole K

`ggml/src/ggml-cuda/promptforge_iu4.cuh` states the contract in its own words:

> One long K segment is the performance-first contract. It keeps scale and
> zero-point correction out of the WMMA inner loop and leaves quality recovery
> to offline weight optimization and bounded keeper corrections.
> `constexpr int kSegments = 1;`

Their packed layout is

```
weights [segment][N/64][K/segments/8][64]     scales [segment][N]     sums [segment][N]
activations [segment][M][K/segments/8]        scales [segment][M]     zeros [segment][M]
```

— **one f32 scale per output column for the entire K**, plus a precomputed int32
column sum. The IU4 WMMA accumulates in i32 across the whole segment and then,
exactly once at the end:

```
corrected = acc - zero[row] * weight_sums[n];
out       = corrected * a_scales[row] * weight_scales[n];
```

**There is no per-group work in the inner loop at all.**

## Why that is the gap

Our compact format is G=256, so the i32 accumulator must be folded into f32 with
a per-group f16 scale every 256 weights — every 4 K-strips at BK=64. That fold
is what forces the second accumulator set, and it is why the wave64 port had to
drop to WNt=4 (WNt=8 spills 113 VGPRs) and therefore landed at 1.26x instead of
the 1.56x the 1-pass twin gets. The per-group contract also *creates* the sparse
overlay: 4.25 bits buys its quality from 3 per-row corrections per group, and
applying them costs a further 23.9% of prefill.

So our two largest prefill costs after the GEMM itself — the overlay pass and the
halved N-tile — are both **downstream of the same decision: per-group scales.**
Kairic moved that entire burden offline.

## How they buy the quality back (all offline)

- **Hadamard rotation at block 1024** (`kHadamardBlock = 1024`) — incoherence
  processing, the same family as our FWHT but a much larger block.
- **Offline weight optimization** and **"bounded keeper corrections"** — their
  analogue of outliers, resolved before serving rather than in the kernel.
- Weights ship as **separate repacked sidecars** (`.pfs`: FFN, GDN,
  GDN-Output) in a WMMA-native layout, not decoded from a general container
  in-kernel. `Qwen3.8-27B-IU4-Kairic-Edge.gguf` carries the base model.

## Scope, honestly stated by them

- "Kairic Edge routes supported **prompt and multi-token verification** shapes
  through native IU4. It is a **guarded hybrid path**: operations outside the
  qualified shape/quality envelope retain their established fallback."
- "**The current native sidecars do not accelerate M1 target decode.**"
- Their own evidence boundary: "do not ... describe this hybrid route as
  whole-model native INT4."

So the >350 figure is a **prefill/verification** number on a hybrid path, which
is exactly the axis we are being compared on. Their decode is not IU4-accelerated
— consistent with our decode (14.2 tok/s) not being the thing under discussion.

## Two independent confirmations of our own findings

- Their runner uses **batch 2048 / ubatch 512**. We measured ubatch 512 as the
  optimum independently (256 -> 199.5, **512 -> 211.7**, 1024 -> 188.3,
  2048 -> 164.3).
- Their GEMM is **IU4 WMMA**, not bf16 and not iu8 — matching our measurement
  that the compact raw-nibble path beats both oq8 (2.4x slower) and dense bf16.

## Also examined: hipEngine (the other engine)

`github.com/shisa-ai/hipEngine`, **AGPLv3** — readable but NOT copyable into
hipfire (Apache-2.0). Its prefill is a port of llama.cpp's MMQ
("Source lineage: llama.cpp HIP 1ebf790cda38, ggml-cuda mmq.cuh/vecdotq.cuh"),
Q5_K/Q6_K x Q8_1, and it notes "no weight sidecar exists". Its `hip_gfx1151/`
directory contains only `__init__.py` — Strix Halo runs the gfx1100 kernels.
Notably llama.cpp's gfx1100 MMQ specialization reserves ~57 KB of LDS for an
I128/J128/K256 tile, against our 20 KB.

So both competitors reach their prefill numbers **without any in-kernel
per-group scale and without a sparse outlier pass** — one via K-quant MMQ, one
via a single-segment IU4 contract.

## What this implies for us

The lever we have not tried is the one they both avoid: **reduce or eliminate the
per-group scale in the prefill GEMM.** Concretely, a prefill-only weight lane
with a single (or very coarse) K segment plus precomputed column sums would:

1. delete the fold-and-rescale from the inner loop, freeing the registers that
   forced WNt=4 -> plausibly recovering the 1-pass twin's 1.56x rather than 1.26x;
2. delete the overlay correction pass (23.9%) along with the per-group contract
   that needs it.

Cost: a second weight layout resident in VRAM (they pay this too — separate
`.pfs` sidecars) and a real quality question, since one scale over K=5120 is very
coarse. They answer it with a 1024-block Hadamard plus offline optimization; we
already have FWHT and a calibration/Hessian pipeline, so the pieces exist.

**Open and unmeasured: whether that coarse-scale contract holds our KLD budget.**
That is the experiment to run before any kernel work — quantize one tensor both
ways offline and compare error, no GPU kernel required.
