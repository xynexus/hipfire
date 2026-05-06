# 2026-05-06 — MoE Quant Cliff Survey: Recon Synthesis

## Origin

Issue #171 (Qwen3.6-A3B MQ4 incoherence) confirmed that hipfire's MoE Q4 cliff is
sampler-orthogonal — every sampler (greedy+RP=1.05, HF temp=1.0, RP=1.5/2.5,
Qwen-official PP=1.5) attractor-fails on agent prompts. Cure exists *only* via
`thinking=false`. Forward-pass fragility is real and lives below the sampler.

Memory captures the smoking gun: `down_proj` has 37M-magnitude tail outliers in
"a few layers across both 3.5 and 3.6". Per-row absmax MQ4 quant on a row with a
37M outlier crushes the bulk into ~3-4 effective levels.

This doc is the recon synthesis ahead of building a per-tensor NRMSE survey on
hiptrx (4× R9700, gfx1201).

## Recon agents (2026-05-06)

Three parallel research agents:
1. Map hipfire's existing quant + calib infra
2. Spec llama.cpp Q4_K format precisely
3. Survey outlier-isolation methodology literature

## Key findings

### A. Existing hipfire infra

- **HFQ4G256**: 136 B / 256 elements (f32 scale + f32 zero + 128 B nibbles).
  4.250 bpw. Defined in `crates/hipfire-runtime/src/hfq.rs:29-454`.
- **MQ4G256** (`quant_type=13`) is FWHT-rotated HFQ4 — already an outlier-mitigation
  attempt. Routing in `crates/rdna-compute/src/dispatch.rs:208-212`.
- **Q4_K format is already in the quantizer**: `crates/hipfire-quantize/src/main.rs:192-293`,
  GGML-compatible. DType::Q4K exists in dispatch.rs (mapped to GGML kernels).
- **No per-tensor NRMSE / reconstruction-error tooling exists.** Must build.
- **Sidecar pipeline** (Phase 0-5 done) lives on `feat/cdna-calib-mfma` branch
  at commit 05f83b0 — NOT on master/strix-halo. `scripts/calibrate_multigpu.sh`
  uses HIP_VISIBLE_DEVICES per-job for multi-GPU dispatch.
- **Multi-GPU runtime state**: single-device `Gpu` instance; no in-process
  round-robin across 4 GPUs. Fan-out via shell script + multiple processes.
- **HFQ4 dequant entry points**: `kernels/src/hfq4g256_dequantize_to_f16.hip`,
  `kernels/src/gemm_hfq4g256_residual_mw.hip:69-96`,
  `kernels/src/gemv_mq4g256.hip:24-46`.

### B. Q4_K format reality (FLIPS the cheap-cut plan)

- Q4_K is **144 B / 256 elem = 4.500 bpw**. Wider than HFQ4G256 (4.250 bpw)
  by 6%.
- Formula: `weight = d * sub_scale[i] * q4 - dmin * sub_min[i]` (note: `-`, not `+`).
- Quality on LLaMA-7B WikiText (k-quants PR #1684):
  - F16: 5.9066
  - Q4_0: 6.0215
  - **Q4_K_S (pure Q4_K): 6.0215** ← identical to Q4_0
  - Q4_K_M (Q6 attention + Q4 elsewhere): 5.9601
  - Q5_K_S: 5.9419
- **Pure Q4_K_S buys nothing over Q4_0 at iso-bpw.** The headline "Q4_K wins"
  is the *_M mixed-precision policy that lives one level above the format.
- This kills the "cheap sub-block scales" plan as a standalone fix. Q4_K is
  still a useful storage substrate but only with mixed-precision policy on top.
- Reference dequant to port: `ggml/src/ggml-cuda/convert.cu :: dequantize_block_q4_K`
  (32-thread group per super-block, lane decomposition `(il=lane/8, ir=lane%8)`,
  8 outputs/lane — maps cleanly to RDNA wave32).

### C. Outlier-isolation literature winner

**Direct match for our problem**: arXiv 2507.23279 — "Super Experts in MoE":
- Confirms <0.5% of experts in Qwen3-30B-A3B and DeepSeek-R1 produce extreme
  `down_proj` output outliers that drive attention-sink mechanism.
- Pruning or naively quantizing them collapses the model.
- **This is exactly our 37M-tail observation.** Not a quirk of our calibration
  or quant code — it's a structural property of the architecture.

Companion: arXiv 2506.13329 (EAQuant) — MoE-specific. W4A4 PPL improvements via
`smooth_aggregate` per-expert scale + routing-consistency calibration.

Companion: arXiv 2406.08155 (QuantMoE-Bench) — empirically confirms `down_proj`
in late MoE blocks has highest outlier scores; pure uniform Q4 is the worst
configuration.

### D. Method comparison (kernel retrofit cost)

| Method | Outlier mechanism | Retrofit cost | Quality on MoE |
|--------|-------------------|---------------|----------------|
| AWQ (2306.00978) | Per-channel pre-scale (no sparse path) | **Trivial** — fold into RMSNorm | Good |
| HQQ (Mobius) | LP-loss optimizer (calibration-free) | **Trivial** — pack-time only | Good |
| GPTQ (2210.17323) | Hessian-aware error compensation | Trivial — pack-time only | Modest gap to fp16 |
| **Super-Expert (2507.23279)** | **Per-expert mixed precision** | **Low** — per-expert dtype tag + GEMM dispatch branch | **Direct fix for our case** |
| QuaRot/SpinQuant (2404.00456) | Hadamard rotation on residual | Medium — invasive at RMSNorm | Best broad outlier weapon |
| SqueezeLLM (2306.07629) | CSR sparse fp16 sidecar | High — separate sparse-matvec kernel | Good but expensive |
| QuIP# (2402.04396) | E8 lattice vector codebook | Very high — replace dequant inner loop | SOTA at 2-bit |

## Revised plan

The contributor's "sub-block scales close ~half-gap" intuition was over-optimistic
for pure Q4_K_S at iso-bpw. The real fix is **AWQ pre-scaling + Super-Expert
detection at Q8** (combo recommended by Agent 3). Estimated combined cost: ~1%
extra VRAM, hours of calibration, low kernel-touch retrofit.

### Phase 1: Survey runner (1-2 days on hiptrx)

Single calibration pass on Qwen3.5-9B / 3.5-A3B / 3.6-A3B emits structured JSON
per `(model, layer, tensor, expert)` with:
- Per-tensor NRMSE (fp16 ref vs MQ4 quantized)
- Per-row absmax distribution (find the 37M rows)
- **Per-expert `down_proj` output max during calibration** (find the Super Experts)
- **Per-channel activation absmax during calibration** (for AWQ scales)
- FWHT pre-rotation diagnostic: pre/post-rotation absmax distribution
  (does FWHT actually equalize? If not, why does our MQ4 still cliff?)

4-GPU fan-out via per-model jobs (existing `scripts/calibrate_multigpu.sh` pattern).

Output goes to `docs/investigations/2026-05-06-moe-quant-cliff-survey/data/`.

### Phase 2: Verify SE hypothesis (1 day)

Visualize survey JSON. Confirm:
- Top-K (K = 0.5% of total experts ≈ 1-2 per layer) explain most of the 37M tails
- These are stable across the calibration corpus (not random)
- Promoting them to Q8 strictly improves NRMSE on the affected layers

### Phase 3: AWQ + SE retrofit (1-2 weeks)

- Calibration: emit per-channel activation scales (AWQ) + per-expert SE flag
- Quantizer: fold AWQ scales into prior RMSNorm gain at pack time (zero runtime cost)
- HFQ format: per-expert dtype tag (Q4 or Q8) in tensor metadata
- Loader: read tag, allocate per-expert weight buffer at correct precision
- Runtime: per-expert GEMM dispatch chooses Q4 or Q8 path
- Validate on `docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/repro_failing.jsonl`

### Phase 4: Storage substrate decision

If Q4_K substrate (with AWQ+SE policy) matches or beats MQ4 quality at lower bpw
on non-SE experts, switch storage format. Q4_K is already in the quantizer; needs
runtime kernel port (CUDA dequant ref above).

### Phase 5 (only if Phase 3 insufficient)

Add SqueezeLLM-style CSR sparse outlier sidecar for the worst rows. Heaviest retrofit
in the comparison; deferred unless data forces it.

## Open questions for Phase 1 to answer

1. **Where do the 37M tails actually live?** Per-layer? Per-expert? Per-channel?
   Memory says "down_proj" but doesn't pin layer/expert.
2. **Are they FWHT-pre-rotation or FWHT-post-rotation?** If post-rotation, FWHT
   isn't equalizing them and we need a different mitigation. If pre-rotation,
   the rotation is being misapplied at quant time.
3. **How many experts qualify as Super Experts?** Paper says ≤0.5% but our
   models may differ (Qwen3.5-A3B has 128 experts/layer; 0.5% = 0 or 1).
4. **Does the SE pathology correlate with the agent-prompt failure mode?**
   I.e. do the failing prompts route through SE-heavy paths more than working
   prompts?

## References

- arXiv 2306.00978 (AWQ) — https://github.com/mit-han-lab/llm-awq
- arXiv 2507.23279 (Super Experts in MoE)
- arXiv 2506.13329 (EAQuant)
- arXiv 2406.08155 (QuantMoE-Bench)
- llama.cpp Q4_K reference: `ggml/src/ggml-cuda/convert.cu :: dequantize_block_q4_K`
- k-quants PR: https://github.com/ggml-org/llama.cpp/pull/1684
- Issue #171 reproducer: `docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/repro_failing.jsonl`
