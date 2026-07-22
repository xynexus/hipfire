<!--
SPDX-License-Identifier: Apache-2.0
hipfire — see LICENSE and NOTICE in the project root.
-->
# hipfire model support — source of truth

This is the **canonical model-support matrix** for hipfire. It tracks what is
*actually implemented and routed* per architecture, with the flagship
**qwen3.5** as the reference for full feature parity.

- Ground truth for arch IDs: `crates/hipfire-model/src/lib.rs` (`ARCH_ID_*`).
- Ground truth for routing/gating: `crates/hipfire-serving-core/src/generate.rs`
  and `load.rs` (where unsupported features are explicitly refused per `arch_id`).
- This table reflects **implemented + served** capability, not the forward-looking
  family roster (that lives in `docs/plans/2026-06-19-arch-roster-feature-matrix.md`).
- **Quant formats** and their weight/activation/calibration tradeoffs:
  `docs/quant-formats/opus-mqplus-eval-plan.md`. Canonical names: **Magnum**=MQ4 /
  **Magnum Plus**=MQ4+ · **Opus Quant**=OQ4 (W4A4) / OQ8 (W8A8) and **Opus
  Plus**=OQ4+ / OQ8+. First plus = clip-search, SmoothQuant/AWQ, or comparable
  activation-aware calibration; second plus = Hessian/LDLQ feedback. Opus =
  symmetric quantized weights.
- **Per-quant × per-GPU-arch kernel coverage** (which formats have tuned
  decode/prefill/WMMA kernels vs generic fallback): see "Kernel coverage" below.

**Last verified:** 2026-06-26 (against `chaingun`).

Legend: ✅ full · 🟡 partial / limited · ❌ not implemented (explicitly refused at load/serve) · — not applicable (e.g. expert sharding on a dense arch)

## Generated capability matrix

The tables below are generated from `docs/model-support.toml` (the single source
of truth, shared with `crates/hipfire-model/src/model_support_generated.rs`). Do
not hand-edit between the markers — run `cargo run -p hipfire-cli -- gen-model-support`.
The richer hand-maintained roster (microbatch / PP / EP columns) follows.

<!-- BEGIN GENERATED model-support (source: docs/model-support.toml — run `cargo run -p hipfire-cli -- gen-model-support`) -->

### Capability matrix (generated)

Machine-readable subset consumed by `arch_features` / admission. Edit `docs/model-support.toml`.

This per-arch chart is the **`family × feature` projection** of the 5-axis capability space (`family × gfx-class × quant × kv × feature`), collapsed at the reference gfx (`rdna3.5`/gfx1151), the family's best stable quant, and its best KV mode. The gfx/quant cross-sections it flattens are rendered as separate derived projections below.

| Arch (arch_id) | Batched prefill | DFlash spec | MTP spec | KV quant | Vision |
|---|---|---|---|---|---|
| qwen3.5 (5, 6) | ✅ | ✅ | ✅ | full | 🟡 |
| deepseek4 (9) | ✅ | ❌ | 🟡 | fp32 | ❌ |
| minimax (10) | 🟡 | ❌ | 🟡 | fp32 | ❌ |
| lfm2-moe (11) | 🟡 | ❌ | ❌ | fp32 | ❌ |
| nemotron_h (14) | ✅ | ❌ | ❌ | fp32 | ❌ |
| mamba2 (15) | ✅ | ❌ | ❌ | no-kv | ❌ |
| zaya (16) | 🟡 | ❌ | ❌ | fp32 | ❌ |
| gemma3 (12) | ✅ | ❌ | ❌ | fp32+q8 | ❌ |
| gemma3-vl (13) | ✅ | ❌ | ❌ | fp32+q8 | ✅ |
| embeddinggemma (19) | ✅ | ❌ | ❌ | none | ❌ |
| gemma4 (24) | 🟡 | ❌ | ❌ | fp32+kvarn | ❌ |
| qwen2 (7) | ✅ | ❌ | ❌ | fp32 | ❌ |
| dots-ocr (8) | ✅ | ❌ | ❌ | fp32 | ✅ |
| llama (0, 1) | 🟡 | ❌ | ❌ | fp32 | ❌ |

### Quant formats (generated)

| Quant | Weight bits | Act bits | Status |
|---|---|---|---|
| bf16 (BF16 (unquantized)) | 16 | 16 | stable |
| q8 (Q8 (W8A16)) | 8 | 16 | stable |
| mq4 (Magnum / MQ4 (W4A16)) | 4 | 16 | stable |
| mq6 (Magnum / MQ6 (W6A16)) | 6 | 16 | stable |
| oq4 (Opus Quant / OQ4 (W4A4, int4 activations)) | 4 | 4 | opt-in |
| oq4+ (Opus Quant Plus / OQ4+ (W4A8, calibrated OQ4 weights)) | 4 | 8 | opt-in |
| oq4++ (Opus Quant Plus / OQ4++ (W4A8, Hessian/LDLQ OQ4 weights)) | 4 | 8 | opt-in |
| oq4.25++ (Opus Quant mixed OQ4.25++ (compact W4 bulk + sparse W8 outliers)) | 4.25 | 8 | opt-in |
| oq8 (Opus Quant / OQ8 (W8A8, int8 activations)) | 8 | 8 | opt-in |
| oq8+ (Opus Quant Plus / OQ8+ (W8A8, calibrated OQ8 weights)) | 8 | 8 | opt-in |
| oq8++ (Opus Quant Plus / OQ8++ (W8A8, Hessian/LDLQ OQ8 weights)) | 8 | 8 | opt-in |
| mq3 (Magnum / MQ3 (W3A16, mixed-precision only)) | 3 | 16 | experimental |
| qtip3 (QTIP-3 (W3A16 trellis decode)) | 3 | 16 | opt-in |
| qtip4 (QTIP-4 (W4A16 trellis decode)) | 4 | 16 | opt-in |

### Intentional gates (generated)

Per-quant overrides of an arch capability (admission consults these before green-lighting).

| Arch | Quant | Feature | Support | Note |
|---|---|---|---|---|
| 5 | oq4 | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side OQ4 W4A4 have finite tiny KLD and fixture golden evidence on gfx1103; batched prefill parity remains gated / opt-in (iu4 WMMA path), and VL keeps the vision tower on hfq4. OQ checklist producer=yes loader=yes decode=smoke prefill=gated tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 5 | oq4+ | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side OQ4+ have calibrated tiny KLD and fixture golden evidence on gfx1103; batched prefill parity and full quality admission remain gated, and VL keeps the vision tower on hfq4. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 5 | oq4++ | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side OQ4++ have Hessian-backed tiny KLD and fixture golden evidence on gfx1103; batched prefill parity and full quality admission remain gated, and VL keeps the vision tower on hfq4. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 5 | oq4.25++ | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side mixed OQ4.25++ have calibrated tiny KLD and fixture golden evidence on gfx1103; batched prefill parity and full quality admission remain gated, and VL keeps the vision tower on hfq4. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending; mixed OQ uses per-tensor Oq4G256/OqPlusCompact dispatch |
| 5 | oq8 | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side OQ8 W8A8 have finite tiny KLD and fixture golden evidence on gfx1103; batched prefill parity remains gated, and VL keeps the vision tower on hfq4. OQ checklist producer=yes loader=yes decode=smoke prefill=gated tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 5 | oq8+ | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side OQ8+ have calibrated tiny KLD and fixture golden evidence on gfx1103; batched prefill parity and full quality admission remain gated, and VL keeps the vision tower on hfq4. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 5 | oq8++ | prefill | 🟡 | Qwen3.5 dense and Qwen3.5-VL text-side OQ8++ have Hessian-backed tiny KLD and fixture golden evidence on gfx1103; batched prefill parity and full quality admission remain gated, and VL keeps the vision tower on hfq4. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 6 | oq4 | prefill | 🟡 | Qwen3.5 MoE OQ4 routed experts load through dense-layout OQ fallback by default while experimental indexed OQ remains opt-in; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=gated tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 6 | oq4+ | prefill | 🟡 | Qwen3.5 MoE OQ4+ routed experts use dense-layout OQ fallback by default with Hessian-backed calibration; tiny KLD and fixture golden are finite on gfx1103, while experimental indexed OQ remains opt-in. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 6 | oq4++ | prefill | 🟡 | Qwen3.5 MoE OQ4++ uses dense-layout OQ fallback by default with Hessian/LDLQ calibration; tiny KLD and fixture golden are finite on gfx1103, while experimental indexed OQ remains opt-in. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 6 | oq4.25++ | prefill | 🟡 | Qwen3.5 MoE mixed OQ4.25++ quantizes, loads, and decodes through the dense-layout OQ fallback by default; calibrated tiny KLD and fixture golden are finite on gfx1103, while experimental indexed OQ remains opt-in. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 6 | oq8 | prefill | 🟡 | Qwen3.5 MoE OQ8 routed experts load through dense-layout OQ fallback by default while experimental indexed OQ remains opt-in; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=gated tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 6 | oq8+ | prefill | 🟡 | Qwen3.5 MoE OQ8+ uses dense-layout OQ fallback by default with Hessian-backed calibration; tiny KLD and fixture golden are finite on gfx1103, while experimental indexed OQ remains opt-in. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 6 | oq8++ | prefill | 🟡 | Qwen3.5 MoE OQ8++ uses dense-layout OQ fallback by default with Hessian/LDLQ calibration; tiny KLD and fixture golden are finite on gfx1103, while experimental indexed OQ remains opt-in. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 9 | oq4 | prefill | 🟡 | DeepSeek4 text-core OQ4 W4A4 repacks through oq4_arch_load in the native loader while routed experts remain MQ2-Lloyd for the DeepSeek4 MoE kernels; tiny KLD and fixture golden are finite on gfx1103. Compressed-KV and MTP OQ4 role variants remain explicitly blocked until compressor/indexer and native MTP dtype policies exist. OQ checklist producer=hybrid-oq-dense-mq2l-experts loader=yes decode=smoke prefill=gated tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 9 | oq4+ | prefill | 🟡 | DeepSeek4 OQ4+ uses native-loader dense OQ routing with routed experts kept MQ2-Lloyd for the DeepSeek4 MoE kernels; calibrated tiny KLD and fixture golden are finite on gfx1103. Compressed-KV/MTP calibrated OQ remains blocked pending auxiliary tensor dtype policies. OQ checklist producer=hybrid-oq-dense-mq2l-experts loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 9 | oq4++ | prefill | 🟡 | DeepSeek4 OQ4++ uses native-loader dense OQ routing with routed experts kept MQ2-Lloyd for the DeepSeek4 MoE kernels; Hessian-backed tiny KLD and fixture golden are finite on gfx1103. Compressed-KV/MTP Hessian OQ remains blocked until auxiliary tensor dtype policies exist. OQ checklist producer=hybrid-oq-dense-mq2l-experts loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 9 | oq4.25++ | prefill | 🟡 | DeepSeek4 mixed OQ4.25++ uses native-loader dense OQ8-family routing with routed experts kept MQ2-Lloyd for the DeepSeek4 MoE kernels; mixed tiny KLD and fixture golden are finite on gfx1103. Compressed-KV/MTP mixed OQ remains blocked pending auxiliary tensor dtype policies. OQ checklist producer=hybrid-oq-dense-mq2l-experts loader=yes decode=smoke prefill=gated tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 9 | oq8 | prefill | 🟡 | DeepSeek4 text-core OQ8 W8A8 repacks through oq8_arch_load in the native loader while routed experts remain MQ2-Lloyd for the DeepSeek4 MoE kernels; tiny KLD and fixture golden are finite on gfx1103. Compressed-KV OQ8 is blocked on compressor F16 upload policy and MTP OQ8 is blocked because generic OQ artifacts omit packaged mtp.0.* tensors. OQ checklist producer=hybrid-oq-dense-mq2l-experts loader=partial-aux-blocked decode=smoke prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 9 | oq8+ | prefill | 🟡 | DeepSeek4 text-core OQ8+ has calibrated tiny KLD and fixture golden evidence on gfx1103 through the native-loader OQ8 route with routed experts kept MQ2-Lloyd; compressed-KV/MTP calibrated OQ8 remains blocked pending auxiliary tensor dtype policies. OQ checklist producer=hybrid-oq-dense-mq2l-experts-hessian loader=partial-aux-blocked decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 9 | oq8++ | prefill | 🟡 | DeepSeek4 text-core OQ8++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the native-loader OQ8 route with routed experts kept MQ2-Lloyd; compressed-KV/MTP Hessian OQ8 remains blocked pending auxiliary tensor dtype policies. OQ checklist producer=hybrid-oq-dense-mq2l-experts-hessian loader=partial-aux-blocked decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 11 | oq4 | prefill | 🟡 | LFM2 OQ4 W4A4 prefill routes through iu4 WMMA; current evidence is 350M smoke/parity plus finite tiny KLD and fixture golden on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 11 | oq4+ | prefill | 🟡 | LFM2 OQ4+ W4A8 has calibrated tiny KLD and fixture golden evidence on gfx1103; real-model eval and routed-expert telemetry remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 11 | oq4++ | prefill | 🟡 | LFM2 OQ4++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103; Hessian/LDLQ routed-expert audit and model eval admission remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 11 | oq4.25++ | prefill | 🟡 | LFM2 mixed OQ4.25++ has calibrated tiny KLD and fixture golden evidence on gfx1103; tiered assignment is still not promoted without model eval. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending; tiered mixed assignment not promoted |
| 11 | oq8 | prefill | 🟡 | LFM2 OQ8 W8A8 prefill routes through iu8 WMMA; current evidence is 350M smoke/parity plus finite tiny KLD and fixture golden on gfx1103. OQ checklist producer=fallback-required loader=yes decode=yes prefill=smoke tiny=gfx1103-ragged-fallback golden=gfx1103 eval=pending artifact=pending |
| 11 | oq8+ | prefill | 🟡 | LFM2 OQ8+ has calibrated tiny KLD and fixture golden evidence on gfx1103; real-model eval and routed-expert telemetry remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 11 | oq8++ | prefill | 🟡 | LFM2 OQ8++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103; Hessian/LDLQ routed-expert audit and model eval admission remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 10 | oq4 | prefill | 🟡 | MiniMax OQ4 W4A4 loads through shared dense OQ4 plus indexed routed-expert OQ4 kernels; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 10 | oq4+ | prefill | 🟡 | MiniMax OQ4+ has calibrated tiny KLD and fixture golden evidence on gfx1103 through the dense and indexed routed-expert OQ runtime kernels; real-model eval and routed-expert telemetry remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 10 | oq4++ | prefill | 🟡 | MiniMax OQ4++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the dense and indexed routed-expert OQ runtime kernels; Hessian/LDLQ routed-expert audit and model eval admission remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 10 | oq4.25++ | prefill | 🟡 | MiniMax mixed OQ4.25++ now routes dense OqPlusCompact tensors, including lm_head, through the shared OQ8-family loader and has calibrated tiny KLD plus fixture golden evidence on gfx1103; real-model eval and routed-expert telemetry remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 10 | oq8 | prefill | 🟡 | MiniMax OQ8 W8A8 loads through shared dense OQ8 plus indexed routed-expert OQ8 kernels; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 10 | oq8+ | prefill | 🟡 | MiniMax OQ8+ has calibrated tiny KLD and fixture golden evidence on gfx1103 through the dense and indexed routed-expert OQ runtime kernels; real-model eval and routed-expert telemetry remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 10 | oq8++ | prefill | 🟡 | MiniMax OQ8++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the dense and indexed routed-expert OQ runtime kernels; Hessian/LDLQ routed-expert audit and model eval admission remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 7 | oq4 | prefill | 🟡 | Qwen2 OQ4 W4A4 loads via oq4_arch_load into the generic iu4 GEMM route; GPU-validated coherent (Qwen2-0.5B), tiny KLD and fixture golden are finite on gfx1103; eval-battery admission pending. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 7 | oq4+ | prefill | 🟡 | Qwen2 OQ4+ W4A8 loads via oq4_to_oq8_combined into the shared int8-activation route and has Hessian-backed calibrated tiny KLD plus fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 7 | oq4++ | prefill | 🟡 | Qwen2 OQ4++ has Hessian-backed calibrated tiny KLD and fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 7 | oq4.25++ | prefill | 🟡 | Qwen2 mixed OQ4.25++ exercises the ragged OQ8 GPU-compatible fallback and has calibrated tiny KLD plus fixture golden evidence on gfx1103. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending; ragged OQ8 must fall back to GPU-compatible Q8 |
| 7 | oq8 | prefill | 🟡 | Qwen2 OQ8 W8A8 loads via oq8_combined into the shared iu8 GEMM route; GPU-validated coherent (Qwen2-0.5B), tiny KLD and fixture golden are finite on gfx1103; eval-battery admission pending. OQ checklist producer=fallback-required loader=yes decode=yes prefill=smoke tiny=gfx1103-ragged-fallback golden=gfx1103 eval=pending artifact=pending |
| 7 | oq8+ | prefill | 🟡 | Qwen2 OQ8+ shares the OQ8 runtime kernels and has Hessian-backed calibrated tiny KLD plus fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=fallback-required-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 7 | oq8++ | prefill | 🟡 | Qwen2 OQ8++ exercises the ragged OQ8 GPU-compatible fallback and has calibrated tiny KLD plus fixture golden evidence on gfx1103. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending; ragged OQ8 must fall back to GPU-compatible Q8 and real-model admission remains pending |
| 12 | oq4 | prefill | 🟡 | Gemma3 text OQ4 W4A4 loads through shared Oq4G256 dispatch; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 12 | oq4+ | prefill | 🟡 | Gemma3 text OQ4+ shares OQ4/OQ8 runtime dispatch and has Hessian-backed activation-aware tiny KLD plus fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 12 | oq4++ | prefill | 🟡 | Gemma3 text OQ4++ has calibrated tiny KLD and fixture golden evidence on gfx1103; promotion still requires Hessian/LDLQ audit and model eval evidence. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 12 | oq4.25++ | prefill | 🟡 | Gemma3 text mixed OQ4.25++ has calibrated tiny KLD and fixture golden evidence on gfx1103; calibration audit and model eval evidence are still pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 12 | oq8 | prefill | 🟡 | Gemma3 text OQ8 W8A8 loads through shared Oq8G256 dispatch; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 12 | oq8+ | prefill | 🟡 | Gemma3 text OQ8+ shares OQ8 runtime dispatch and has Hessian-backed activation-aware tiny KLD plus fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 12 | oq8++ | prefill | 🟡 | Gemma3 text OQ8++ has calibrated tiny KLD and fixture golden evidence on gfx1103; Hessian/LDLQ real-model admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 8 | oq4 | prefill | 🟡 | dots-ocr text-side OQ4 W4A4 loads through the Qwen2 text decoder while the Dots vision tower remains out of the OQ artifact path; synthetic image splice tiny KLD and text-only fixture golden are finite on gfx1103. OQ checklist producer=text-only loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103-text eval=pending artifact=pending |
| 8 | oq4+ | prefill | 🟡 | dots-ocr text-side OQ4+ has calibrated tiny KLD and fixture golden evidence on gfx1103 while the Dots vision tower remains hfq4. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate vision-tower OQ policy remains unadmitted |
| 8 | oq4++ | prefill | 🟡 | dots-ocr text-side OQ4++ has calibrated tiny KLD and fixture golden evidence on gfx1103 while the Dots vision tower remains hfq4. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate vision-tower OQ policy remains unadmitted |
| 8 | oq4.25++ | prefill | 🟡 | dots-ocr mixed OQ4.25++ is limited to text tensors and has calibrated tiny KLD plus fixture golden evidence on gfx1103 while the Dots vision tower remains hfq4. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate vision-tower OQ policy remains unadmitted |
| 8 | oq8 | prefill | 🟡 | dots-ocr text-side OQ8 W8A8 loads through the Qwen2 text decoder while the Dots vision tower remains out of the OQ artifact path; synthetic image splice tiny KLD and text-only fixture golden are finite on gfx1103. OQ checklist producer=text-only loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103-text eval=pending artifact=pending |
| 8 | oq8+ | prefill | 🟡 | dots-ocr text-side OQ8+ has calibrated tiny KLD and fixture golden evidence on gfx1103 while the Dots vision tower remains hfq4. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate vision-tower OQ policy remains unadmitted |
| 8 | oq8++ | prefill | 🟡 | dots-ocr text-side OQ8++ has calibrated tiny KLD and fixture golden evidence on gfx1103 while the Dots vision tower remains hfq4. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate vision-tower OQ policy remains unadmitted |
| 13 | oq4 | prefill | 🟡 | Gemma3-VL text-side OQ4 W4A4 loads through the Gemma3 text decoder while SigLIP/projector tensors remain q8f16; synthetic image splice tiny KLD and text-only fixture golden are finite on gfx1103. OQ checklist producer=text-only loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103-text eval=pending artifact=pending |
| 13 | oq4+ | prefill | 🟡 | Gemma3-VL text-side OQ4+ shares Gemma3 OQ dispatch and has Hessian-backed activation-aware tiny KLD plus fixture golden evidence on gfx1103 while SigLIP/projector tensors remain q8f16. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate SigLIP/projector OQ policy remains unadmitted |
| 13 | oq4++ | prefill | 🟡 | Gemma3-VL text-side OQ4++ has calibrated tiny KLD and fixture golden evidence on gfx1103 while SigLIP/projector tensors remain q8f16. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate SigLIP/projector OQ policy remains unadmitted |
| 13 | oq4.25++ | prefill | 🟡 | Gemma3-VL mixed OQ4.25++ is limited to text tensors and has calibrated tiny KLD plus fixture golden evidence on gfx1103 while SigLIP/projector tensors remain q8f16. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate SigLIP/projector OQ policy remains unadmitted |
| 13 | oq8 | prefill | 🟡 | Gemma3-VL text-side OQ8 W8A8 loads through the Gemma3 text decoder while SigLIP/projector tensors remain q8f16; synthetic image splice tiny KLD and text-only fixture golden are finite on gfx1103. OQ checklist producer=text-only loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103-text eval=pending artifact=pending |
| 13 | oq8+ | prefill | 🟡 | Gemma3-VL text-side OQ8+ shares OQ8 runtime kernels and has Hessian-backed activation-aware tiny KLD plus fixture golden evidence on gfx1103 while SigLIP/projector tensors remain q8f16. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate SigLIP/projector OQ policy remains unadmitted |
| 13 | oq8++ | prefill | 🟡 | Gemma3-VL text-side OQ8++ has calibrated tiny KLD and fixture golden evidence on gfx1103 while SigLIP/projector tensors remain q8f16. OQ checklist producer=text-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-text-calib eval=pending artifact=pending; separate SigLIP/projector OQ policy remains unadmitted |
| 0 | oq4 | prefill | 🟡 | LLaMA OQ4 W4A4 loads via oq4_arch_load; K must be % 256 else the linear stays BF16; tiny KLD and fixture golden are finite on aligned gfx1103 fixture tensors; quality admission pending. OQ checklist producer=aligned-only loader=yes decode=smoke prefill=smoke tiny=gfx1103-aligned golden=gfx1103-aligned eval=pending artifact=pending |
| 0 | oq4+ | prefill | 🟡 | LLaMA OQ4+ W4A8 loads via oq4_to_oq8_combined into the shared int8-activation route and has Hessian-backed calibrated tiny KLD plus fixture golden evidence on aligned gfx1103 fixture tensors; real-model eval admission remains pending. OQ checklist producer=aligned-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 0 | oq4++ | prefill | 🟡 | LLaMA OQ4++ has Hessian-backed calibrated tiny KLD and fixture golden evidence on aligned gfx1103 fixture tensors; real-model eval admission remains pending. OQ checklist producer=aligned-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending; ragged linears stay BF16/F16 |
| 0 | oq4.25++ | prefill | 🟡 | LLaMA mixed OQ4.25++ has Hessian-backed calibrated tiny KLD and fixture golden evidence on aligned gfx1103 fixture tensors; real-model eval admission remains pending. OQ checklist producer=aligned-only-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending; mixed OQ cannot quantize unsupported ragged linears |
| 0 | oq8 | prefill | 🟡 | LLaMA OQ8 W8A8 loads via oq8_combined into the shared iu8 GEMM route; GPU-validated coherent (Llama-3.2-1B), finite tiny KLD and fixture golden on gfx1103; eval-battery admission pending. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 0 | oq8+ | prefill | 🟡 | LLaMA OQ8+ shares the OQ8 runtime kernels and has Hessian-backed calibrated tiny KLD plus fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 0 | oq8++ | prefill | 🟡 | LLaMA OQ8++ has Hessian-backed calibrated tiny KLD and fixture golden evidence on gfx1103; real-model eval admission remains pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 1 | oq4 | prefill | 🟡 | Qwen3/Qwen2 legacy OQ4 follows the LLaMA-family loader for bias-free plain Qwen3; Qwen2 attention-bias artifacts are rejected with an arch-id 7 error. OQ checklist producer=aligned-only loader=conditional-qwen3 decode=gfx1103-qwen3-legacy prefill=smoke tiny=gfx1103-qwen3-legacy golden=gfx1103-qwen3-legacy eval=pending artifact=pending |
| 1 | oq4+ | prefill | 🟡 | Qwen3/Qwen2 legacy OQ4+ follows the LLaMA-family OQ4-to-OQ8 dispatch for bias-free plain Qwen3 with Hessian-backed activation scaling; Qwen2 attention-bias artifacts must be tagged arch 7. OQ checklist producer=aligned-only-hessian loader=conditional-qwen3 decode=gfx1103-qwen3-legacy-calib prefill=smoke tiny=gfx1103-qwen3-legacy-calib golden=gfx1103-qwen3-legacy-calib eval=pending artifact=pending |
| 1 | oq4++ | prefill | 🟡 | Qwen3/Qwen2 legacy OQ4++ exercises Hessian/LDLQ over aligned tensors through the LLaMA-family loader for bias-free plain Qwen3; Qwen2 attention-bias artifacts must be tagged arch 7. OQ checklist producer=aligned-only-hessian loader=conditional-qwen3 decode=gfx1103-qwen3-legacy-calib prefill=smoke tiny=gfx1103-qwen3-legacy-calib golden=gfx1103-qwen3-legacy-calib eval=pending artifact=pending |
| 1 | oq4.25++ | prefill | 🟡 | Qwen3/Qwen2 legacy mixed OQ4.25++ exercises Hessian/LDLQ mixed OQ4/OQ8 tensor dispatch through the LLaMA-family loader for bias-free plain Qwen3; Qwen2 attention-bias artifacts are intentionally routed to arch 7. OQ checklist producer=aligned-only-hessian loader=conditional-qwen3 decode=gfx1103-qwen3-legacy-calib prefill=smoke tiny=gfx1103-qwen3-legacy-calib golden=gfx1103-qwen3-legacy-calib eval=pending artifact=pending |
| 1 | oq8 | prefill | 🟡 | Qwen3/Qwen2 legacy OQ8 follows the LLaMA-family OQ8 dispatch for bias-free plain Qwen3; Qwen2 attention-bias artifacts are rejected with an arch-id 7 error. OQ checklist producer=yes loader=conditional-qwen3 decode=gfx1103-qwen3-legacy prefill=smoke tiny=gfx1103-qwen3-legacy golden=gfx1103-qwen3-legacy eval=pending artifact=pending |
| 1 | oq8+ | prefill | 🟡 | Qwen3/Qwen2 legacy OQ8+ shares OQ8 runtime kernels for bias-free plain Qwen3 with Hessian-backed activation scaling; Qwen2 attention-bias artifacts must be tagged arch 7. OQ checklist producer=yes-hessian loader=conditional-qwen3 decode=gfx1103-qwen3-legacy-calib prefill=smoke tiny=gfx1103-qwen3-legacy-calib golden=gfx1103-qwen3-legacy-calib eval=pending artifact=pending |
| 1 | oq8++ | prefill | 🟡 | Qwen3/Qwen2 legacy OQ8++ exercises Hessian/LDLQ through OQ8 runtime kernels for bias-free plain Qwen3; Qwen2 attention-bias artifacts must be tagged arch 7. OQ checklist producer=yes-hessian loader=conditional-qwen3 decode=gfx1103-qwen3-legacy-calib prefill=smoke tiny=gfx1103-qwen3-legacy-calib golden=gfx1103-qwen3-legacy-calib eval=pending artifact=pending |
| 14 | oq4 | prefill | 🟡 | Nemotron OQ4 W4A4 loads via oq4_arch_load; batched prefill via weight_gemm; hybrid Mamba/MLP/attention tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=smoke tiny=gfx1103-hybrid golden=gfx1103-hybrid eval=pending artifact=pending |
| 14 | oq4+ | prefill | 🟡 | Nemotron OQ4+ W4A8 loads via oq4_to_oq8_combined and has calibrated hybrid Mamba/MLP/attention tiny KLD plus fixture golden evidence on gfx1103; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-hybrid-calib golden=gfx1103-hybrid-calib eval=pending artifact=pending |
| 14 | oq4++ | prefill | 🟡 | Nemotron OQ4++ has Hessian-backed hybrid Mamba/MLP/attention tiny KLD plus fixture golden evidence on gfx1103; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-hybrid-calib golden=gfx1103-hybrid-calib eval=pending artifact=pending |
| 14 | oq4.25++ | prefill | 🟡 | Nemotron mixed OQ4.25++ has Hessian-backed hybrid Mamba/MLP/attention tiny KLD plus fixture golden evidence on gfx1103; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-hybrid-calib golden=gfx1103-hybrid-mixed eval=pending artifact=pending |
| 14 | oq8 | prefill | 🟡 | Nemotron OQ8 W8A8 loads via oq8_combined + weight_gemm Oq8G256 route; GPU-validated coherent (Nemotron-3-Nano-4B), hybrid Mamba/MLP/attention tiny KLD and fixture golden are finite on gfx1103; eval-battery admission pending. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103-hybrid golden=gfx1103-hybrid eval=pending artifact=pending |
| 14 | oq8+ | prefill | 🟡 | Nemotron OQ8+ shares the OQ8 runtime kernels and has calibrated hybrid Mamba/MLP/attention tiny KLD plus fixture golden evidence on gfx1103; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-hybrid-calib golden=gfx1103-hybrid-calib eval=pending artifact=pending |
| 14 | oq8++ | prefill | 🟡 | Nemotron OQ8++ has Hessian-backed hybrid Mamba/MLP/attention tiny KLD plus fixture golden evidence on gfx1103; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-hybrid-calib golden=gfx1103-hybrid-calib eval=pending artifact=pending |
| 15 | oq4 | prefill | 🟡 | Mamba2 OQ4 W4A4 loads through the Nemotron linear OQ4 route while embeddings remain gather-friendly Q8/source precision; pure recurrent tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 15 | oq4+ | prefill | 🟡 | Mamba2 OQ4+ shares the Nemotron linear OQ4/OQ8 route and has calibrated pure recurrent tiny KLD plus fixture golden evidence on gfx1103; model eval admission and recurrent calibration artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 15 | oq4++ | prefill | 🟡 | Mamba2 OQ4++ has Hessian-backed pure recurrent tiny KLD plus fixture golden evidence on gfx1103; model eval admission and recurrent calibration artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 15 | oq4.25++ | prefill | 🟡 | Mamba2 mixed OQ4.25++ has Hessian-backed pure recurrent tiny KLD plus fixture golden evidence on gfx1103; model eval admission and recurrent calibration artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 15 | oq8 | prefill | 🟡 | Mamba2 OQ8 W8A8 loads through the Nemotron linear OQ8 route while embeddings remain gather-friendly Q8/source precision; pure recurrent tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 15 | oq8+ | prefill | 🟡 | Mamba2 OQ8+ shares the OQ8 runtime kernels and has calibrated pure recurrent tiny KLD plus fixture golden evidence on gfx1103; model eval admission and recurrent calibration artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 15 | oq8++ | prefill | 🟡 | Mamba2 OQ8++ has Hessian-backed pure recurrent tiny KLD plus fixture golden evidence on gfx1103; model eval admission and recurrent calibration artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 16 | oq4 | prefill | 🟡 | Zaya OQ4 W4A4 loads through the native zaya GPU loader via oq4_arch_load; CCA attention, EDA/MoD router, and split experts have finite tiny KLD plus fixture golden evidence on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 16 | oq4+ | prefill | 🟡 | Zaya OQ4+ has calibrated tiny KLD and fixture golden evidence on gfx1103 through the native Zaya CCA/EDA calibration forward; model eval admission and artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 16 | oq4++ | prefill | 🟡 | Zaya OQ4++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the native Zaya CCA/EDA calibration forward; model eval admission and artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 16 | oq4.25++ | prefill | 🟡 | Zaya mixed OQ4.25++ has calibrated tiny KLD and fixture golden evidence on gfx1103 through the native Zaya CCA/EDA calibration forward; mixed tier policy is exercised in the tiny artifact, but model eval admission and artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 16 | oq8 | prefill | 🟡 | Zaya OQ8 W8A8 loads through the native zaya GPU loader via oq8_arch_load; CCA attention, EDA/MoD router, and split experts have finite tiny KLD plus fixture golden evidence on gfx1103. OQ checklist producer=yes loader=yes decode=smoke prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 16 | oq8+ | prefill | 🟡 | Zaya OQ8+ has calibrated tiny KLD and fixture golden evidence on gfx1103 through the native Zaya CCA/EDA calibration forward; model eval admission and artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 16 | oq8++ | prefill | 🟡 | Zaya OQ8++ has Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the native Zaya CCA/EDA calibration forward; model eval admission and artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 19 | oq4 | prefill | 🟡 | EmbeddingGemma OQ4 uses the Gemma3 encoder backbone OQ4 route and host Dense-head dequantization, but embedding-quality evidence is pending. OQ checklist producer=yes loader=yes decode=not-ar prefill=encode-smoke tiny=not-ar golden=not-ar eval=embedding_quality-pending artifact=pending |
| 19 | oq4+ | prefill | 🟡 | EmbeddingGemma OQ4+ shares the encoder OQ dispatch route, but activation-calibrated embedding quality evidence is pending. OQ checklist producer=pending-calib loader=yes decode=not-ar prefill=encode-smoke tiny=not-ar golden=not-ar eval=embedding_quality-pending artifact=pending |
| 19 | oq4++ | prefill | 🟡 | EmbeddingGemma OQ4++ requires Hessian/LDLQ calibration audit and embedding quality evidence before promotion. OQ checklist producer=pending-hessian loader=yes decode=not-ar prefill=encode-smoke tiny=not-ar golden=not-ar eval=embedding_quality-pending artifact=pending |
| 19 | oq4.25++ | prefill | 🟡 | EmbeddingGemma mixed OQ4.25++ requires encoder tensor-policy audit, Dense-head coverage, and embedding quality evidence before promotion. OQ checklist producer=pending-hessian loader=yes decode=not-ar prefill=encode-smoke tiny=not-ar golden=not-ar eval=embedding_quality-pending artifact=pending |
| 19 | oq8 | prefill | 🟡 | EmbeddingGemma OQ8 uses the Gemma3 encoder backbone OQ8 route and host Dense-head dequantization; embedding-quality evidence is pending. OQ checklist producer=yes loader=yes decode=not-ar prefill=encode-smoke tiny=not-ar golden=not-ar eval=embedding_quality-pending artifact=pending |
| 19 | oq8+ | prefill | 🟡 | EmbeddingGemma OQ8+ is the target NPU Opus bucket path, but row-padded OQ8 requires resident XDNA or an explicit GPU fallback and embedding quality evidence is pending. OQ checklist producer=pending-calib loader=conditional-xdna decode=not-ar prefill=encode-smoke tiny=not-ar golden=npu-parity-pending eval=embedding_quality-pending artifact=pending |
| 19 | oq8++ | prefill | 🟡 | EmbeddingGemma OQ8++ requires Hessian/LDLQ calibration audit plus resident NPU/GPU fallback parity before promotion. OQ checklist producer=pending-hessian loader=conditional-xdna decode=not-ar prefill=encode-smoke tiny=not-ar golden=npu-parity-pending eval=embedding_quality-pending artifact=pending |
| 24 | oq4 | prefill | 🟡 | Gemma4 dense/PLE/MoE OQ4 W4A4 loads through shared weight_gemv Oq4G256 dispatch; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 24 | oq4+ | prefill | 🟡 | Gemma4 dense, PLE, and dense-MoE OQ4+ have calibrated tiny KLD and fixture golden evidence on gfx1103 through the shared OQ dispatch path; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 24 | oq4++ | prefill | 🟡 | Gemma4 dense, PLE, and dense-MoE OQ4++ have Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the shared OQ dispatch path; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 24 | oq4.25++ | prefill | 🟡 | Gemma4 dense, PLE, and dense-MoE mixed OQ4.25++ have calibrated tiny KLD and fixture golden evidence on gfx1103; dense-MoE strict LDLQ now records finite Hessian attempts. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-mixed eval=pending artifact=pending |
| 24 | oq8 | prefill | 🟡 | Gemma4 dense/PLE/MoE OQ8 W8A8 loads through shared weight_gemv Oq8G256 dispatch; tiny KLD and fixture golden are finite on gfx1103. OQ checklist producer=yes loader=yes decode=yes prefill=smoke tiny=gfx1103 golden=gfx1103 eval=pending artifact=pending |
| 24 | oq8+ | prefill | 🟡 | Gemma4 dense, PLE, and dense-MoE OQ8+ have calibrated tiny KLD and fixture golden evidence on gfx1103 through the shared OQ8 dispatch path; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |
| 24 | oq8++ | prefill | 🟡 | Gemma4 dense, PLE, and dense-MoE OQ8++ have Hessian-backed tiny KLD and fixture golden evidence on gfx1103 through the shared OQ8 dispatch path; model eval admission and product artifact provenance remain pending. OQ checklist producer=yes-hessian loader=yes decode=smoke prefill=smoke tiny=gfx1103-calib golden=gfx1103-calib eval=pending artifact=pending |

### Diffusion capability matrix (generated)

Image/video denoiser families (keyed by their diffusion `arch_id`), graded on the generation-pipeline spine rather than the autoregressive spine above. **ingest** = offline HFQ import + quant precision policy; **text-enc** = prompt conditioning tower; **denoise** = MMDiT/DiT backbone forward; **sampler** = scheduler / denoise-loop; **vae** = latent→RGB decode; **t2i** = end-to-end text-to-image serving. Edit `docs/model-support.toml`.

| Family (arch_id) | Denoiser | Ingest | Text-enc | Denoise | Sampler | VAE | t2i | Quant |
|---|---|---|---|---|---|---|---|---|
| flux2 (23) | flux2-mmdit | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | bf16·q4f16·q8f16·hfq4/6·oq4·oq4+·oq4++·oq8 |
| krea2 (17) | krea2-mmdit | ✅ | 🟡 | ✅ | 🟡 | ✅ | 🟡 | bf16·q4f16·q8f16·hfq4/6·oq4·oq4+·oq4++·oq8 |
| qwen-image (18) | qwen-image-mmdit | ✅ | ❌ | 🟡 | ❌ | 🟡 | ❌ | bf16·q4f16·q8f16·hfq4/6·oq4·oq4+·oq4++·oq8 |

- **flux2**: Only end-to-end-wired family: Qwen3 text tower + SeFi dual-time denoise loop, flow-match Euler + DPM schedulers, shared VAE decode; variants Klein (5/20) and SeFi-2B (4/16); img2img + inpaint via the shared pipeline
- **krea2**: Denoiser topology + Qwen3-VL tower + text-fusion load and a family-specific mixed-precision policy exist, but the generate-path glue is still `#[allow(dead_code)]`; not yet a wired serve loop
- **qwen-image**: Ingest/identity only: MMDiT backbone topology and the Wan-class per-channel VAE are recognized, but NativeDiffusionRuntime builds no Qwen-Image text conditioner, so there is no servable t2i loop

### Batched prefill: quant × gfx-class (derived)

Projection of the prefill axis over **weight-quant × gfx-class**, computed from the runtime predicate `is_batchable_la` (GPU-free). ✅ = batched-prefill GEMM exists; ❌ = falls back to per-token decode; 🔒 = governed by a quality `[[gate]]` (OQ activation-quant formats), see the gates table. This is the kernel-availability truth the per-arch chart collapses to the reference gfx.

| Quant | cdna | rdna12 | rdna3 | rdna3.5 | rdna4 |
|---|---|---|---|---|---|
| bf16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| q8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| mq4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| mq6 | ✅ | ✅ | ✅ | ✅ | ✅ |
| oq4 | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| oq4+ | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| oq4++ | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| oq4.25++ | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| oq8 | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| oq8+ | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| oq8++ | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| mq3 | ❌ | ✅ | ✅ | ✅ | ✅ |
| qtip3 | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |
| qtip4 | 🔒 | 🔒 | 🔒 | 🔒 | 🔒 |

### Batched prefill: kv-mode (derived)

Projection of the prefill axis over **kv-mode**, from `kv_mode_prefill_batchable`. Only Q8 and the rotated asym K modes have a batched flash-masked prefill kernel; fp32 and no-kv (SSM) fall back to per-token decode.

| KV mode | Batched prefill |
|---|---|
| fp32 | ❌ |
| q8 | ✅ |
| asym{2,3,4} | ✅ |
| no-kv (SSM) | ❌ |

### DFlash spec-decode: family × gfx-class (derived)

Projection of the dflash axis over **family × gfx-class**: the per-family `[[arch]]` intent capped by the gfx WMMA gate `dflash_gfx_supported` (GPU-free, shares `arch_caps.has_wmma`). ✅/🟡 = family intent on a WMMA gfx; ❌ = no spec path for the family, or a non-WMMA gfx where dflash falls back to plain decode.

| Family (arch_id) | cdna | rdna12 | rdna3 | rdna3.5 | rdna4 |
|---|---|---|---|---|---|
| qwen3.5 (5, 6) | ❌ | ❌ | ✅ | ✅ | ✅ |
| deepseek4 (9) | ❌ | ❌ | ❌ | ❌ | ❌ |
| minimax (10) | ❌ | ❌ | ❌ | ❌ | ❌ |
| lfm2-moe (11) | ❌ | ❌ | ❌ | ❌ | ❌ |
| nemotron_h (14) | ❌ | ❌ | ❌ | ❌ | ❌ |
| mamba2 (15) | ❌ | ❌ | ❌ | ❌ | ❌ |
| zaya (16) | ❌ | ❌ | ❌ | ❌ | ❌ |
| gemma3 (12) | ❌ | ❌ | ❌ | ❌ | ❌ |
| gemma3-vl (13) | ❌ | ❌ | ❌ | ❌ | ❌ |
| embeddinggemma (19) | ❌ | ❌ | ❌ | ❌ | ❌ |
| gemma4 (24) | ❌ | ❌ | ❌ | ❌ | ❌ |
| qwen2 (7) | ❌ | ❌ | ❌ | ❌ | ❌ |
| dots-ocr (8) | ❌ | ❌ | ❌ | ❌ | ❌ |
| llama (0, 1) | ❌ | ❌ | ❌ | ❌ | ❌ |
<!-- END GENERATED model-support -->

## Feature matrix vs flagship qwen3.5

| Arch (arch_id) | Decode | Batched prefill | Server microbatch | DFlash spec | MTP spec | KV quant modes | Lowered/superop pipeline | Layer shard (PP) | Expert shard (EP/TP) | Vision |
|---|---|---|---|---|---|---|---|---|---|---|
| **qwen3.5 dense / MoE (5 / 6)** — flagship | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ full menu | ✅ | ✅ | ✅ (MoE) | via qwen35-vl |
| qwen3.5-VL (5/6 + splice) | ✅ | ✅ | ✅ (family) | ✅ (family) | ✅ (family) | ✅ full | ✅ | ✅ (family) | ✅ (MoE) | ✅ |
| deepseek4-flash (9) | ✅ | ✅ (own kernels) | ❌ | ❌ | 🟡 native MTP head loads, not wired to spec-serving | 🟡 fp32 only | ✅ | ❌ | ❌ (MoE, unsharded) | ❌ |
| minimax-m2 (10) | ✅ | ❌ per-token | ❌ | ❌ | 🟡 config plumbing only | 🟡 fp32 only | ✅ | ❌ | ❌ (MoE, unsharded) | ❌ |
| lfm2-moe (11) | ✅ | 🟡 in-request batched prefill | ❌ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | ❌ (MoE, unsharded) | ❌ |
| nemotron_h (14) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | 🟡 SimpleAr seam | ❌ | ❌ (MoE, unsharded) | ❌ |
| gemma3 text (12) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | — (dense) | ❌ |
| gemma3-VL / medgemma (13) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 + q8 | ❌ | ❌ | — (dense) | ✅ |
| qwen2 (7) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | ✅ | ❌ | — (dense) | ❌ |
| dots-ocr (8) | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 fp32 only | 🟡 | ❌ | — (dense) | ✅ (OCR) |
| llama / mistral (0), qwen3-legacy (1) | ✅ | 🟡 (llama path) | ❌ | ❌ | ❌ | 🟡 fp32 | 🟡 | ❌ | — (dense) | ❌ |
| toy | test fixture only | — | — | — | — | — | — | — | — | — |

> **Server microbatch** = serving many *concurrent* request streams batched
> together (continuous batching), distinct from in-request *batched prefill* (one
> prompt, many tokens). It's a bespoke qwen35 subsystem (`Qwen35RequestSessionState`
> + `qwen35_decode_batch`), gated to arch 5/6; the grouped-MoE fused batch worker
> requires `arch_id=6`. All other archs run single-session AR only and emit
> `generate_batch_prefill_unsupported`.

## The headline

**qwen3.5 is the only arch that gets the full inference stack.** Three capability
tiers are hard-gated to the qwen3.5 family (`is_qwen35_family_arch_id`, arch 5/6),
and every other arch *explicitly errors* if asked for them:

- **DFlash spec-decode** (the big tok/s lever) — qwen35 only.
- **MTP spec-decode serving** — qwen35 only. (deepseek4 *has* a native MTP head
  that loads, but it isn't routed to the spec path yet; minimax has config
  plumbing only.)
- **CASK eviction + pipeline-parallel (pp>1)** — qwen35 only.
- **Full KV-quant menu** (q8 / asym3 / asym4 / FWHT / KVarN / hierarchical) —
  qwen35 only. gemma3 family adds q8; everyone else is fp32-only.

## Where each arch sits relative to flagship

- **Closest to flagship: deepseek4 (9)** — own batched prefill + decode kernels,
  lowered pipeline, native MTP head present. Missing: spec-decode serving, KV
  quant, CASK/PP.
- **gemma3-VL (13)** — strongest *multimodal* arch (vision grounding + batched
  prefill + q8 KV), but no spec-decode and no lowered pipeline.
- **minimax (10)** — solid validated decode + lowered pipeline, but still
  **per-token prefill** (slow long-context ingest) and fp32-only KV.
- **lfm2-moe (11)** — decode + lowered pipeline plus in-request batched prefill
  are routed. OQ4/OQ4+/OQ8/OQ8+ prefill is still partial because current evidence
  is 350M smoke/parity plus local sidecar experiments, not full admission.
- **nemotron_h (14)** — hybrid Mamba-2 / GQA / ReLU² / MoE AR path with model-level
  batched prefill and fp32 KV. Missing: server microbatch, DFlash/MTP, KV quant,
  CASK/PP/EP.
- **qwen2 (7) / dots-ocr (8)** — basic AR decode + batched prefill, fp32 KV, no
  fast paths.
- **llama / legacy (0 / 1)** — the original baseline path; functional decode, none
  of the modern levers.

## Multi-GPU & sharding (layer / expert / host)

Like the fast paths above, **all sharding is qwen3.5-family-only and single-host.**

| Sharding axis | Status | Scope | Where |
|---|---|---|---|
| **Layer sharding (pipeline-parallel, PP)** | ✅ implemented | qwen35 family only (5/6); `pp>1` explicitly refused for all other archs | `hipfire-runtime/src/multi_gpu.rs` (`Gpus`: layer bands, boundary copy, peer-access). `HIPFIRE_PP_LAYERS=48,16` sets per-device bands. Issue #58 Stage 7. |
| **Expert sharding (expert-parallel, EP)** | ✅ implemented | qwen35-**MoE** (arch 6) — each rank computes only its owned experts + shared expert on rank 0, then all-reduce combine | `hipfire-runtime/src/ep.rs`. `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` for peer-direct combine. |
| **Tensor sharding (TP: Q/KV heads, weight sub-ranges)** | ✅ implemented | qwen35-MoE A3B (Qwen3.5-30B-A3B); `expert_to_rank[e] = e % tp_size` (or contiguous); KV replicated when `tp_size > n_kv_heads` (TP=4 on A3B) | `hipfire-runtime/src/tp_shard.rs`; see `docs/plans/multi-gpu-tp-a3b.md`. |
| **Collectives backend** | ✅ implemented | single-node, multi-GPU | `hip-bridge/src/rccl.rs` — RCCL (AMD NCCL) FFI: `ncclCommInitAll` over local device ids; backs `Gpus::all_reduce_sum`. ~3× faster than a host-driven ring on gfx1201. |
| **Across hosts (multi-node / cross-node)** | ❌ **not implemented** | — | No `ncclCommInitRank` / `ncclUniqueId` / TCP bootstrap / `node_rank`. RCCL init is single-node only. Multi-host inference is unsupported. |

**Summary:** layer + expert + tensor sharding all work **across GPUs on one host**,
and only for the qwen3.5 family (TP/EP tuned specifically for the 30B-A3B MoE).
**Nothing shards across hosts** — there is no cross-node communicator or bootstrap.
HIP work is also single-threaded (one OS thread for all `Gpu::*` calls), so the
multi-GPU orchestrator drives devices from a single host thread.

## Biggest gaps to close (flagship-parity order)

1. **Batched prefill for minimax; broaden/gate LFM2 prefill** — minimax is still
   per-token, while LFM2 needs multi-model quality/perf admission beyond 350M OQ
   smoke/parity.
2. **KV quantization beyond qwen35** — only gemma3 has q8; no asym/FWHT kernels
   for any non-qwen35 arch.
3. **Spec-decode generalization** — DFlash/MTP are architecturally welded to
   qwen35; deepseek4's native MTP head is the cheapest candidate to wire next.
4. **Lowered pipeline for gemma3 / gemma3-vl** — the only "modern" archs still on
   the legacy forward path.
5. **Server microbatch + sharding generalization** — continuous batching, PP, and
   EP/TP are all welded to qwen35 (`Qwen35RequestSessionState`, `qwen35_decode_batch`,
   `*_qwen35` planners). A generic per-arch session/shard abstraction is needed
   before any other arch can microbatch or shard.
6. **Multi-host inference** — no cross-node communicator exists (RCCL is single-node
   `ncclCommInitAll`). Would need rank/uniqueId bootstrap + a transport.

## Kernel coverage (per-quant × per-GPU-arch tuned kernels)

Whether a quant format runs on a *tuned* kernel vs a *generic fallback* is set by
three things: the **decode** GEMV arm (`weight_gemv`), the **prefill** GEMM arm
(`weight_gemm`, batched), and whether it has **`_for_arch` selectors** (per-GPU-arch
WMMA + multi-batch mb2/mb4 variants — the deeply-tuned tier). Missing prefill arms
fall to `W8A8Ref` generic reference (instrumented by `warn_generic_once`, silence
with `HIPFIRE_WARN_GENERIC=0`).

Ground truth: `crates/hipfire-runtime/src/weights.rs` (dispatch arms),
`crates/rdna-compute/src/{dispatch,kernels,generic_warn}.rs`.

| Quant (DType) | Decode GEMV | Fused decode | Prefill GEMM (batched) | Per-arch tuned (`_for_arch`) | Primary model-archs |
|---|---|---|---|---|---|
| **MQ4G256-Lloyd** | ✅ | ✅ resid/swiglu | ✅ WMMA mb2/mb4 + fused QKV/gate-up | ✅ | qwen35 |
| **MQ3G256-Lloyd** | ✅ | ✅ | ✅ WMMA mb4 + fused QKV/gate-up | ✅ | qwen35 |
| **HFQ4G256** | ✅ | ✅ resid | ✅ `gemm_hfq4g256` | ✅ | qwen35, general |
| **HFQ3G256** | ✅ | ✅ resid | ✅ residual WMMA | ✅ | qwen35 |
| HFP4G32 | ✅ | — | 🟡 | ✅ | fp4 path |
| MQ4G256 (plain "magnum") | ✅ | ✅ resid/swiglu | 🟡 via Lloyd/generic | partial | qwen35, lfm2 experts |
| ParoQ4G128 | ✅ | ✅ resid/swiglu | ✅ paro gemm | ❌ | PARO variants |
| MQ8G256 | ✅ | ✅ prerotated | 🟡 | ❌ | q8 weights |
| MQ6G256 / HFQ6G256 | ✅ (HFQ6 indexed-MoE) | 🟡 | 🟡 | ❌ | lfm2 experts (mq6e) |
| MQ2G256(-Lloyd) | ✅ | ✅ (Lloyd) | 🟡 | ❌ | minimax MoE |
| **Oq4G256 (opus/OQ4/OQ4+)** | ✅ | ✅ qkvza/gate-up | ✅ W4A4 iu4 WMMA; OQ4+ W4A8 via int8-activation MMQ/F16 WMMA | partial | qwen35, lfm2 |
| **Oq8G256 (opus/OQ8/OQ8+)** | ✅ | ✅ qkvza/gate-up | ✅ W8A8 iu8 WMMA | partial | qwen35, lfm2 |
| MFP4G32 | ✅ | — | 🟡 | ❌ | microscaling fp4 |
| Q4F16 g32/g64 | ✅ | — | 🟡 | ❌ | gguf-ish |
| Qtip3G256 | 🟡 | — | ❌ | ❌ | trellis |
| W8A8Ref | generic ref (fallback) | — | generic | ❌ | fallback only |

**GPU-arch tuning:** the `_for_arch` tuned variants target **gfx1100 (RDNA3 dGPU)**
and **gfx1151 (RDNA3.5 Strix Halo APU)**. `arch_caps` also recognizes gfx1103,
gfx1152, RDNA4, CDNA3, gfx906, but those JIT the same kernel without a dedicated
tuned variant. wave32 (RDNA) vs wave64 (CDNA/GCN) is auto-selected.

**Takeaways:**
- The deeply-tuned tier (per-arch WMMA, mb2/mb4 batched prefill, fused QKV/gate-up/
  residual) is **MQ4/MQ3-Lloyd + HFQ4/HFQ3 + HFP4**, and lives in **qwen35** — another
  qwen35 concentration.
- **Prefill is still the coverage cliff:** many formats have decode GEMV, fewer
  have tuned batched prefill; the rest hit `W8A8Ref` generic or family-specific
  fallback paths.
- **Opus is no longer decode-only:** OQ4/OQ4+/OQ8/OQ8+ have batched prefill
  routes, but non-qwen35 admission is still smoke/parity gated.
- Non-qwen35 archs mostly run generic/decode paths: lfm2 experts (MQ4/MQ6, HFQ6
  indexed-MoE), minimax (MQ2-Lloyd MoE), gemma3 (bf16/Q8).

## Maintaining this file

This is a **living source of truth** — update it whenever arch support changes:

- Adding/removing an `arch_id` or routing a new feature to an arch → update the
  matrix row + the relevant prose.
- Re-verify against `generate.rs` / `load.rs` gating (search for
  `not supported on arch_id=`) and bump **Last verified**.
- Keep the forward-looking *roster* (planned families, audio/omni/diffusion) in
  `docs/plans/2026-06-19-arch-roster-feature-matrix.md`; this file is
  **shipped capability only**.
