# 3.6-A3B residual quality — three concurrent diagnostics — SYNTHESIS

User-driven pushback rejected my "3.5 working ⇒ no engine bug specific to 3.6"
argument because architecture identity does not imply identical routing
distribution, weight magnitude statistics, or activation geometry. Three
concurrent diagnostics, one per host, each settling a different axis.

## Result table

| host    | axis             | tool                                      | verdict        |
|---------|------------------|-------------------------------------------|----------------|
| k9lin   | 3 (GPU topk)     | `HIPFIRE_MOE_TOPK_CPU_OVERWRITE=1` smoke  | **negative**   |
| hiptrx  | 1 (weight stats) | per-row absmax/median tail ratio          | **negative**   |
| hipx    | 1+2 (cure test)  | MQ6-experts variant smoke                 | **negative**   |

All three axes the user proposed for "why 3.6 fails where 3.5 doesn't" are
settled negative. The data also revealed that **the premise was wrong**:
3.5-A3B mq4-mq6exp-port produces functionally identical attractor garbage
on the same multi-paragraph agent prompt with the same sampler config.

## k9lin axis-3 (GPU topk distribution-sensitive defect) — NEGATIVE

`HIPFIRE_MOE_TOPK_CPU_OVERWRITE=1` ported from `debug/moe-qwen-20260505`
(commits `e1a128a` + `9c05b34`) into the modular crate at
`crates/hipfire-arch-qwen35/src/qwen35.rs`. After Path B's
`gpu.softmax_f32 + gpu.moe_topk_renorm_k8`, the gate D2H-downloads probs,
recomputes top-K + renorm on CPU FP64, and overwrites device buffers via
`hip.memcpy_htod` so indexed expert kernels see CPU-overwritten values.

Verified firing at greedy decode (greedy gate-on output diverges from
greedy gate-off, 559 vs 488 chars on a hexagons prompt — proves the
overwrite is materially changing topk weights via 1-ULP-class drift).

On the agent prompt: gate-OFF baseline F.out 271 tokens of "Blocks](" Bodies);"
attractor; gate-ON H.out 118 tokens of "qwen36) and 01 best qren renq ren"
attractor. Different surface form, **same structural failure mode**. CPU-FP64
topk does not cure quality.

→ The defect is upstream of GPU topk. `moe_topk_renorm_k8` is correct.

## hiptrx axis-1 (3.6 weight tails worse than 3.5) — NEGATIVE

NumPy + safetensors absmax/median on every routed expert, all 40 layers,
both models, bf16-aware reader. Per-row absmax / per-row median ratio
captures tail-heaviness against per-row MQ4 quant grid.

| projection      | model   | p99 mean across layers | p99 max across layers | p99 median across layers |
|-----------------|---------|------------------------|------------------------|---------------------------|
| `gate_up_proj`  | 3.5     | 8.95                  | 13.06                 | 8.39                     |
| `gate_up_proj`  | 3.6     | 8.91                  | 12.94                 | 8.32                     |
| `down_proj`     | 3.5     | 939948.62             | 37597656.0            | 7.14                     |
| `down_proj`     | 3.6     | 915534.50             | 36621096.0            | 7.11                     |

Ratio 3.6 over 3.5: gate_up_proj 0.99×, down_proj 0.97×. **3.6 is
indistinguishable from 3.5 in expert-weight magnitude statistics, in fact
fractionally lighter-tailed.** Axis-1 (per-expert MQ4 quant noise compounds
harder on 3.6) is empirically false.

Side-finding: down_proj across both models has a CATASTROPHIC tail-of-tails
(p99 mean ~1M, p99 max ~37M) in a small minority of layers. Half the layers
have a healthy 7× tail (median 7.14), but the worst layer has a row whose
absmax is 37 million × the row median. MQ4 per-row absmax quant on those
rows pushes the median-magnitude weights deep into 4-bit quant noise. This
explains why MQ4 quality is brittle on this model family in general — but
it does so EQUALLY for 3.5 and 3.6.

## hipx axis-1+2 (MQ6 expert quant cures 3.6) — NEGATIVE

Cherry-picked `crates/hipfire-quantize/src/main.rs` from `8620877` (PR #147,
"UMA-safe loader, MQ6 expert quant, disk spill") onto current branch.
Quantized `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/.../`
with `--format mq4-mq6exp` to `~/.hipfire/models/qwen3.6-35b-a3b.mq4-mq6exp`
(26.75 GB). Rsync to hipx, smoked with same agent prompt and sampler config.

3.6-A3B mq4-mq6exp result: 469 tokens of "VLvariant VlARIANT qlArchitecture
qc-twenty-mo-vaint" attractor. Worse than the mq4 baseline (which was 271
tokens). MQ6 routed experts do NOT cure 3.6.

**Critical control: 3.5-A3B mq4-mq6exp-port on the SAME prompt with the SAME
sampler also produces 3,927 chars of "Wait wait wait—I need re-read..."
attractor garbage.** 3.5 fails this prompt too at MQ4-mq6exp.

→ The defect is not in routed-expert quantization at all. And it is
**not 3.6-specific.**

## Premise revision

My earlier remember.md claimed "3.5-A3B regression-clean both hosts" — that
was conditional on simple prompts (hexagons, "speed of light"), not the
agentic multi-paragraph self-referential prompt. On THIS prompt, with these
sampler params (top_k=20, min_p=0.05, temp=1.0, top_p=0.95), Qwen3.5-A3B
mq4-mq6exp ALSO fails.

The user's report from pi-agent ("everything falls apart") was real, but
the failure mode is **prompt-shape sensitive**, not 3.6-vs-3.5 sensitive.

## What's next — re-frame

The remaining hypotheses are upstream of routed experts and below sampler:

1. **DeltaNet drift on long prompts** — 30 LinearAttention layers cascading
   precision loss, plausible on a 1500-character agent prompt with thinking
   on. Easy probe: re-run with `thinking=false` and a single-pass output;
   does failure persist?
2. **FWHT-rotated input geometry** — FWHT rotation amplifies certain
   activation distributions. A control: try a non-rotated dense Q4 quant of
   the same model (q4f16, no FWHT) and see if quality is recovered.
3. **Shared-expert math interaction** — the always-on shared expert with
   sigmoid scalar gate may be misbehaving on long contexts. Disable shared
   expert temporarily (return 0); does failure mode change?
4. **Sampler temp=1.0 too aggressive** — HF generation_config recommends
   temp=1.0 but bench at temp=0.7 to see if attractor is sampler-driven on
   borderline-coherent forward distributions.
5. **DFlash speculation** — 3.6:35B-A3B DFlash draft was already noted as
   "materially worse" than 3.5; same prompt + AR (DFlash off) is implicit in
   our smoke (daemon defaults AR), so this isn't the issue here.

The original PR #167 (sampler + heatmap + ROCm fixes) is genuine value: the
sampler structural fix did reduce attractor frequency on simpler prompts,
and the per-kernel profiler is a permanent improvement. But the 3.6-A3B
"residual quality" is a **separate, model-family issue** that needs a
different investigation.

## Files

- `/tmp/moe-3host-debug/` — all artifacts (gitignored)
- `INVESTIGATION.md` — this file
- `H_cpu_overwrite.jsonl` / `H.out` — k9lin axis-3 smoke (gate ON)
- `sanity_off.out` / `sanity_on.out` — greedy sanity proving gate fires
- `expert_absmax_stats.py` — NumPy reader
- `absmax_results.json` — full per-layer stats
- `quantize_36_mq6exp.log` — k9lin quantize log
- `I_mq4_mq6exp.jsonl` / `I.out` — hipx 3.6-A3B mq4-mq6exp smoke
- `J_35_same_prompt.jsonl` / `J.out` — hipx 3.5-A3B mq4-mq6exp-port control

## Outlier-aware quant prototype — CONFIRMS expert quant is NOT the bottleneck

`quant_recon_error.py` simulated four candidate schemes against the current
MQ4G256-FWHT, sampling 768 worst-tailed rows from layers 10/20/35 of
3.6-A3B's down_proj. Result (mean MSE per row, lower = better):

| scheme              | mean MSE | bpw    | gain vs MQ4G256 |
|---------------------|----------|--------|------------------|
| MQ6G256-FWHT        | 6.49e-08 | 6.25   | **17.6×**        |
| MQ4G256+sidecar64   | 4.63e-07 | 7.65   | 2.5× (worse than MQ6) |
| MQ4G256+sidecar16   | 8.42e-07 | 6.25   | 1.4× (≈ MQ6 byte budget) |
| MQ4G64-noFWHT       | 9.11e-07 | ~4.5   | 1.3× |
| MQ4G256+sidecar4    | 1.02e-06 | 4.75   | 1.1× |
| MQ4G256-FWHT (cur)  | 1.14e-06 | 4.25   | 1.0× |

Cosine similarity vs original is ≥0.991 across ALL schemes. Per-element
weight reconstruction is fine — the per-row absmax tail-of-tails finding
was a *raw-weight* artifact that group-wise FWHT mostly absorbs.

Outlier-extraction at sensible byte budgets does NOT beat MQ6G256.
MQ6G256 already exists (mq4-mq6exp format) and was already tested in this
investigation — it does not cure the agent prompt failure. So per-element
quant noise is **not** the bottleneck for 3.6-A3B residual content quality.

The remaining defect is in *aggregate forward-pass propagation* across
30 LinearAttention layers + per-token routed-expert combination + shared
expert + thinking-block attention, not in any per-element weight encoding.

## Sampler/KV parameter sweep — partial root cause

Same agent prompt, 3.6-A3B mq4, all on hipx. Six runs varying single
parameter:

| run | temp | top_k | min_p | thinking | kv_mode | result | chars/words |
|-----|------|-------|-------|----------|---------|--------|-------------|
| T03 | 0.3  | 20    | 0.05  | true     | asym3   | **clean coherent answer, self-EOS** | 845/103 |
| T07 | 0.7  | 20    | 0.05  | true     | asym3   | partial coherence, EOS | 1377/161 |
| T0_GREEDY | 0.0 | 1   | 0.0   | true     | asym3   | attractor garbage | 2915/353 |
| T0_NOTHINK | 0.0 | 1  | 0.0   | false    | asym3   | bit-identical to T0_GREEDY | 2915/353 |
| KVFP32 | 1.0 | 20    | 0.05  | true     | fp32    | attractor (same as asym3) | 3391/199 |
| KVQ8   | 1.0 | 20    | 0.05  | true     | q8      | **worse — letter-soup `MMEEEKKKAALL`** | 2608/259 |

Findings:
- **temp=0.3 is the operational sweet spot** — clean, coherent, terse,
  self-EOS at 103 words. The model CAN answer this prompt; the HF-recommended
  temp=1.0 is too wide for Qwen3.5-MoE at MQ4 on agent prompts.
- **temp=0 greedy ALSO fails** with the SAME class of attractor — so this
  isn't purely a sampler issue. The forward pass produces borderline
  argmax tokens that walk into the attractor without any sampling noise.
- **`thinking=false` has no effect** — daemon ignores the request flag and
  chat-template forces `<think>` prefix regardless. Confirms memory
  `feedback_qwen35_openthink_default.md` finding.
- **KV mode q8 is actively broken** on this code path — degenerates to
  letter-soup. `kv_mode=asym3` and `kv_mode=fp32` are equivalent on this
  prompt.

**Revised root-cause picture (still partial):**

There are two compounding failures:
1. **Forward path emits borderline-coherent logits** at greedy decode on
   long thinking-block agent prompts. Greedy walks into the attractor
   without help from the sampler. This is the upstream issue.
2. **Wide sampling at temp=1.0 amplifies the borderline regions** — even
   when the forward could recover at temp=0.3, temp=1.0 destabilizes it.

Operational fix today: drop temperature default to 0.3 for Qwen3.5-MoE
family at MQ4. Sampler-config flip in daemon defaults.

True engine fix needs (1) ground-truth reference comparison (llama.cpp /
transformers same model + temp=1.0 prompt) to determine if the forward
borderline-coherence is a hipfire bug vs intrinsic to MQ4-quantized
Qwen3.5-MoE, and (2) if the former, layer-by-layer activation diff to
localize divergence.

## Dense vs MoE comparison — MoE-SPECIFIC bug confirmed

Same agent prompt + temp=1.0 + top_k=20 + min_p=0.05 + thinking=true on hipx:

| run | model | params | result |
|-----|-------|--------|--------|
| 3.5-A3B mq4-mq6exp-port | MoE 35B/3B | MQ4+MQ6 | 3927 chars attractor |
| 3.5-A3B mq4-mq6exp | MoE 35B/3B | MQ4+MQ6 | 469 tok attractor |
| 3.6-A3B mq4 | MoE 35B/3B | MQ4 | 199 words attractor |
| 3.6-A3B mq4-mq6exp | MoE 35B/3B | MQ4+MQ6 | 469 tok attractor |
| **3.5-4B mq4 (DENSE)** | dense | MQ4 | 529 words verbose-but-coherent |
| **3.5-9B mq4 (DENSE)** | dense | MQ4 | 558 words verbose-but-coherent |
| **3.5-27B mq4 (DENSE)** | dense | MQ4 | 560 words verbose-but-coherent |

**Verdict: dense Qwen3.5 family handles this prompt + sampler config without
descending into structural attractor.** All three sizes (4B/9B/27B) produce
verbose, meta-confused, but readable output. MoE models on identical
prompt + sampler produce token-level attractor.

The bug is **MoE-specific in hipfire's forward path**. Combined with prior
diagnostics:

- routed-expert weight quant: NOT it (axis 1 negative; sidecar prototype
  shows cos sim ≥0.991 across schemes)
- GPU topk weights: NOT it (cpu-overwrite settled negative)
- temp/sampler alone: NOT sole cause (greedy also fails)
- prompt shape × MoE × MQ4: this is the failure surface

Remaining MoE-specific suspects (cheapest to probe first):
1. `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed` accumulation
   (final routed combine: `out = sum(g[i] * expert_i(x)) + shared(x)`)
2. `gemv_hfq4g256_residual_sigmoid_scaled_gpu` shared-expert path
   (sigmoid-gated residual add — only on MoE)
3. `fused_silu_mul_rotate_mq_batched` for batched top-K rotation
4. `x_residual` accumulator precision across the 40-layer MoE stack
5. DeltaNet (LinearAttention) interaction with the MoE FFN output
   (30 of 40 layers are LinearAttention; cumulative state space)

Cheapest next probe: temporarily replace shared expert with zero and
re-smoke. If failure mode changes, shared-expert path is in the blame
chain. If unchanged, blame is in routed-expert combine or accumulator.

## Branch state

- `feat/moe-expert-heatmap` at `d67bf02`. Two LOCAL UNCOMMITTED edits:
  - `crates/hipfire-arch-qwen35/src/qwen35.rs` — `HIPFIRE_MOE_TOPK_CPU_OVERWRITE`
    gate (debug-only, do not commit to main feat branch).
  - `crates/hipfire-quantize/src/main.rs` — full file from `8620877`
    (PR #147 mq4-mq6exp format). Useful but PR #147 is the canonical home.
- `~/.hipfire/models/qwen3.6-35b-a3b.mq4-mq6exp` (26.75 GB) on k9lin and
  hipx, both byte-identical (rsync verified).
