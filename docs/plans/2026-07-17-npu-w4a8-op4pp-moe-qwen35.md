# NPU resident W4A8 op4++ FFN for MoE — Qwen3.5-A3B

Status: **active**. Owner: this line of work. Companion:
`docs/npu/npu-kernel-design-guide.md` (part 5 = AIE2P/halo, the measured
authority for every number below), `docs/npu/concurrent-prefill-split-design.md`
(the heterogeneous NPU/GPU split this feeds).

Goal: run the Qwen3.5-A3B routed-expert FFN as a **resident W4A8 op4++** kernel on
the halo NPU (AIE2P / npu2), and stand up a **hardware parity gate** so NPU output
can be trusted interchangeably with the GPU serving path.

---

## 1. What we start from

`NpuResidentFfnW4` (`crates/hipfire-xdna/src/resident_ffn.rs`) is a working,
on-silicon resident **W4A8 op4** FFN — but it is welded to **EmbeddingGemma's
dense shape**: `M=256, K=768, INTER=1152, OUT=768`, **GeGLU**, geometry baked into
a precompiled xclbin (`~/.hipfire/npu/embgemma_r97_canonical_bf16_w4_resident_ffn_*`).
It already fuses gate+up into one packed weight, applies the shared **FWHT-256
rotation (seeds 42/1042) + per-group AWQ scale**, and verifies **bit-exact vs a
CPU op4++ reference** (`npu_resident_ffn_verify`, cosine ≥ 0.999999).

The GPU op4++ path (`weight_gemv` `Oq4G256`) uses the **identical** rotation
convention — the NPU↔GPU numeric bridge is already aligned at the format level.

## 2. Target geometry (Qwen3.5-A3B)

Real `Qwen3-30B-A3B` config: `hidden=2048`, `moe_intermediate=768`,
**128 experts, top-8**, always-on shared expert (`moe_inter=768`), **SwiGLU (SiLU)**.
(The `config.rs` comments cite "256 experts / 512" for a newer Qwen3.5 variant;
the pipeline must read these from the artifact, not hardcode. Rotation target is
**plain-FWHT op4++, non-PARO** — the shipped A3B uses ParoQuant, but the GPU MoE
path has a non-PARO oq4 arm we verify against; PARO-on-NPU is a later option.)

Per routed expert:
- fused **gate_up**: `[K=2048, N=1536]` (2×768), W4
- **down**: `[K=768, N=2048]`, W4, applied post-SiLU
- `K=2048 = 8×256` and `K=768 = 3×256` tile cleanly onto the existing group
  machinery. Activation differs: **SiLU**, not the kernel's current GeLU.

## 3. Roofline (hand-computed — aiecost cannot yet cost npu2 W4A8)

`aiecost.design gemm ... --weight-bits 4` **refuses** on npu2: the int8×int4 arm
is missing `macs_per_native_vmac_int8_int4` (C4) and a C2 feed audit for AIE2P.
Filling those is a tracked side-task (§6). Until then, from design-guide part 5
(measured on halo): 32 cores, 512 MACs/native-VMAC, int4 **sustained
904 G MACs/s/core** (→ array ≈ **28.9 T MACs/s**), feed roof **50 GB/s @4col**,
dispatch floor **c_cmd ≈ 72.6 µs**.

**One expert, M=256 tokens** (805 M + 403 M = 1.21 G MACs; weight bytes @4-bit =
1.5 + 0.75 = **2.25 MiB**):
- compute ≈ 1.21e9 / 28.9e12 ≈ **42 µs**
- weight DMA ≈ 2.36e6 / 50e9 ≈ **47 µs**
- dispatch floor ≈ **72.6 µs** ← **binds**

→ A single fused expert dispatch at M=256 is **dispatch-floor-bound**. Fusing
gate_up + activation + down into **one** dispatch (as the EmbeddingGemma kernel
already does) is mandatory (design rule #1).

**One MoE layer, M=256 prefill batch, top-8/128** ⇒ ~all 128 experts active.
- Weight movement/layer = 128 × 2.25 MiB = **288 MiB** (weights read once, reused
  across a token's expert-mates; **movement is set by distinct active experts, not
  token count**). Cannot be resident (4 MiB memtile) → **stream**.
- Stream @50 GB/s ≈ **5.76 ms/layer**; naive **per-expert dispatch** would add
  128 × 72.6 µs = **9.3 ms/layer of pure floor** → dead on arrival.

**Conclusions (both are the design-guide headline, re-derived for MoE):**
1. NPU MoE FFN prefill is **weight-movement-bound**. The 4-bit **op4++ weight is
   THE lever** — it halves movement (and energy) vs Q8. "Energy is bytes."
2. On **speed**, the GPU wins (halo 8060S LPDDR5X ≫ 50 GB/s NPU feed). On
   **energy**, the NPU wins (5.13 vs 3.73 GB/J ≈ 1.37×). So the NPU role is the
   **energy-optimal / concurrent** arm of a heterogeneous split, not a latency win.
3. Per-dispatch floor forbids per-expert dispatch → **grouped-expert GEMM** (one
   dispatch computes all active experts via gather + per-segment weight indexing).

## 4. Parity contract (the "hardware parity-verify" half)

**The three paths compute different functions by activation precision — so
"NPU == GPU bit-exact" is the WRONG target.** Measured from the code:
- **NPU** resident FFN = **W4A8** (int8 activation × int4 weight).
- **GPU** `weight_gemv` `Oq4G256` = **W4A16 on decode** (B=1: FWHT-rotate the f32
  activation, consume it directly against the 4-bit-resident weight, no act quant —
  `weights.rs:562`) and **W4A4 on prefill** (B>1: `quantize_act_oq4` — `weights.rs:554`).

So the correct contract is three legs, all against the shared `OpusPackedMatrix`
op4++ codec (one source of truth for the int4 weights + scales + FWHT rotation):

- **L1 — weight-decode bit-exactness (the format contract).** NPU and GPU must
  reconstruct the *identical* dequantized op4++ matrix. Encode ONE set of rotated
  int4 weights + per-group scales, feed both device layouts (NPU 130-byte block;
  GPU `[N,K/2]` nibbles ++ f32 scales), and require bit-identical reconstruction.
  This is the only bit-exact leg and it is what actually makes a heterogeneous
  split coherent (each device runs the same weights).
- **L2 — NPU@A8 vs CPU oracle.** `reference_f32` (AWQ→FWHT→int8→int-dot→scale).
  **DONE (M1a): cosine 0.99993 on halo r99 xclbin.**
- **L3 — GPU@A16 (decode) vs CPU oracle.** GPU decode == `reference_dequantized_bf16_f32`
  (f32 activation × dequant-bf16 weight) — the two are the *same math*; expect a
  tight dB. GPU@A4 (prefill) sits at the looser rotation-only int4-act floor
  (~15–20 dB, `oq4_weight_gemv_parity`).

**Measured on silicon (EmbeddingGemma shape, single projection K=768 N=1152):**
`oq4_shared_codec_gpu_parity` (hipfire-runtime) — **L1 bit-exact by construction;
L3 = 55.44 dB on gfx1151** (rotation conventions reconcile); **A8-vs-A16 gap =
43.57 dB** (the int8-activation cost). With M1a's **L2 cosine 0.99993** on the NPU,
all three legs of the contract are proven — the numeric bridge is shape-generic and
retargets to the A3B expert shape once M2's xclbin exists.

Cross-device functional agreement then follows by triangle inequality, and the
**split policy assigns whole experts to one device** (never mixes activation
precision inside one tensor). **Caveat:** the NPU `dequantized_bf16` unrotates with
`cpu_fwht_256(values, signs2, signs1)` (reversed order) then divides by AWQ,
whereas the GPU codec rotates the weight forward with `(signs1, signs2)`; L1/L3
must reconcile this convention explicitly (the first likely bridge bug).

Harness is **shape-parameterized from the xclbin manifest**, so the same binary
retargets from EmbeddingGemma shape (today) to the A3B expert shape (after M2).

### 4a. Parity at the actual Qwen3.5-A3B expert shape

- **GPU leg + weight-decode — PROVEN at expert shape** (`oq4_shared_codec_gpu_parity`,
  gfx1151): gate/up `K=2048 N=768` → **L3 55.73 dB**; down `K=768 N=2048` →
  **L3 55.51 dB**; L1 bit-exact by construction; A8-vs-A16 gap ~43.7 dB. So the
  op4++ **format** and the **GPU serving path** are parity-verified at Qwen3.5
  geometry today.
- **NPU leg at expert shape — PROVEN (projection-level), no IRON needed.** The
  fused resident FFN xclbin is baked to EmbeddingGemma shape, but the op4++ GEMM
  is reachable per-projection through `NpuOpusExecutor::run_f32` (on-device AWQ→
  FWHT→int8→int-dot→scale, deblocking internal) vs `reference_f32`:
  - **down (K=768 → N=2048): cosine 1.0, SQNR 144.23 dB — bit-exact** on halo,
    via the existing `whole8_w4-scaled_m256_kg3_n2304` cache (run N=2304, view
    first 2048). Harness: `npu_opus_whole_scaled_w4_parity`.
  - **gate/up (K=2048 → N=768): bit-exact at EXACT shape** (`mismatches=0,
    max_abs 1e-6`, `npu_opus_verify --fullk` on the freshly-built
    `fullk_submit_w4-scaled_m256_kg8_n768` cache — generated with
    `benchmarks/npu_gemm_tuning/r6/r6_fullk_cache.sh w4-scaled 256 2048 768`).
  So **all three legs (L1/L2/L3) are now closed at BOTH Qwen3.5-A3B expert
  projections at exact geometry.** Projection-level parity for the MoE model is
  established end-to-end.
- **Full fused FFN at expert shape (M2)** — still wants the IRON compile of the
  single-dispatch gate_up+SiLU+down schedule (the per-projection GEMMs above do
  not fuse the FFN); that is the remaining kernel-authoring milestone.

## 5. Milestones

- **M0 — this doc.** ✅
- **M1 — NPU↔GPU op4++ parity harness** at EmbeddingGemma shape on the existing
  r97 xclbin. Locks O2 + the reusable, manifest-driven harness. *No toolchain
  risk; buildable today.* Holds `hipfire lock`.
- **M2 — IRON W4A8 resident FFN at A3B expert shape** (`K=2048→768→2048`) + **SiLU**
  SwiGLU variant. Re-run harness at expert shape → single-expert parity on silicon.
- **M3 — grouped-expert dispatch**: gather tokens per active expert → one batched
  W4A8 GEMM/dispatch; router integration; hot-expert residency/streaming policy;
  whole-MoE-layer parity.
- **M4 — heterogeneous split**: assign experts/layers NPU vs GPU by the tok/J vs
  tok/s objective (`concurrent-prefill-split-design.md`); wire into serving;
  end-to-end tok/J measurement vs 8060S GPU baseline.

## 5a. Perf structure & PP/TG (composed FFN measured + roofline)

The composed expert FFN (`npu_expert_ffn_w4_parity`, committed) is **correct**
(cosine 1.0, SQNR 83.85 dB on silicon) but its wall time (~11 ms/expert @M=256) is
**NOT device time** — it is host round-trips. Structure of `NpuOpusGemmMp::run_f32`:
- **1 device dispatch per GEMM** — fullk accumulates all K-groups in a single array
  dispatch; whole-scaled likewise. Two GEMMs → ~2 dispatches/expert.
- Wall time = host AWQ+FWHT-256+int8-quant prep + a ~12.6 MB int32 **partial
  readback** (`run_resident`) + host scale-reconstruction + host SiLU.
- **Device time ≈ roofline ~145 µs/expert @M=256** (2 × ~73 µs floor; work hidden).

Perf lever is **eliminate host round-trips**, not fewer FLOPs. Two paths toward
device-time wall clock (the clean PP/TG number):
- **`run_resident_scaled`** — f32 reconstructed on-device (no int32 partial readback),
  pre-prepped int8 acts in; prep once, dispatch many.
- **device-resident chaining** — gate_up out → on-array SiLU (`NpuQwen3SwiGlu`) → down
  in via shared dma-buf, no host between GEMMs. The clean PP/TG measurement path.

**PP/TG projection (halo AIE2P, design-guide part 5):**

| | per-expert (device) | per MoE layer | note |
|---|---|---|---|
| **PP** M=256 prefill | ~145 µs (2 disp) / ~73 µs fused | movement-bound ~288 MiB → ~5.76 ms/layer | op4++ 4-bit halves it vs Q8 |
| **TG** M=1 decode | ~145 µs/expert, pure floor | 9 experts × ~145 µs ≈ 1.3 ms × 48 layers ≈ 63 ms/tok → **~16 tok/s** | floor-catastrophic **without M3** |

TG is the problem: at M=1 every expert dispatch is pure 72.6 µs floor, and
top-8 + shared × 48 layers dominates. **M3 grouped-expert dispatch is the TG lever.**

## 5b. M3 grouped-expert dispatch (design)

Amortize the 72.6 µs floor across ALL active experts in **one** dispatch/layer:
- **Gather**: router → per-expert token segments (expected M·top_k/num_experts ≈ 16
  tokens/expert at M=256; ~all 128 experts active in a prefill batch).
- **One batched W4A8 GEMM** over concatenated segments with per-segment weight
  indexing (the GPU MoE-GEMM shape), on the array — 1 dispatch/layer, not ~128.
- **Residency**: 128 × 2.25 MiB = 288 MiB ≫ 4 MiB memtile → stream from GTT; op4++
  4-bit is the movement lever; hot-expert caching for skewed decode routing.
- **Floor math**: 1 grouped dispatch/layer × 48 ≈ 48 × 72.6 µs ≈ 3.5 ms/tok floor
  (vs 63 ms unfused) → moves NPU TG from floor-catastrophic to **movement-bound**,
  i.e. the design-guide energy regime (tok/J win, GPU keeps tok/s).

## 6. Side-tasks / open

- Fill npu2 `macs_per_native_vmac_int8_int4` (C4) + C2 feed audit so
  `aiecost.design` can cost W4A8 on AIE2P (currently refuses; §3 is by-hand).
- No local op4++ A3B artifact exists (only mq4). Need a plain-oq4 A3B (or a
  synthetic single-expert op4++ fixture) to drive O2 at expert shape.
- PARO-on-NPU (apply the learned Givens rotation on-device) — deferred; only if
  plain-FWHT op4++ loses too much quality on A3B experts.
