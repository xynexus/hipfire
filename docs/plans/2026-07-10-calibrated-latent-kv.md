# Calibrated Latent KV Plan

Date: 2026-07-10. Branch: chaingun.

## Premise

KVarN remains the KV storage codec. The question is whether a calibrated
low-rank latent KV representation should sit in front of KVarN, replacing or
augmenting the current cold-tier CASK/KVarN merge path.

The answer changes under the Opus Quant assumption: calibration, light QAT, and
custom kernels are already acceptable costs. Therefore the design should not
optimize for the smallest possible implementation first. It should optimize for
the best quality/byte/kernel compromise after calibration.

## Decision

Pursue calibrated latent KV as the lead high-ceiling path:

```text
calibrated Q/K/V low-rank contract
  -> latent KV cache
  -> KVarN latent packing
  -> low-rank attention kernel
  -> optional cold residual or exact-token sidecar
```

The earlier simple SVD cold residual remains useful as a probe and fallback, but
it is no longer the preferred end state if model calibration and kernel work are
in scope.

## Why This Is Better Than Plain Cold SVD

Plain per-cache SVD answers "can this cache segment be reconstructed?" That is
not the true attention objective. Attention quality is governed by logits and
outputs:

```text
Q K^T
softmax(Q K^T) V
```

So the calibrated objective should preserve the query-key interaction and the
value-to-output path, not just the raw key/value vectors. This favors:

- KQ-SVD-style objectives for keys, because they target the attention matrix.
- ReCalKV-style value calibration and value/output fusion, because values carry
  output-sensitive information and some reconstruction can be avoided.
- QSVD-style rank allocation, because uniform rank wastes bytes on easy layers
  and under-budgets hard layers.
- KVarN packing of latent vectors, because low-rank reduces feature dimension
  and KVarN reduces scalar storage inside that feature dimension.

## Candidate Roles

### KQ-SVD-Style Key Calibration

Lead role: choose key-side low-rank bases by minimizing attention-logit error,
not key reconstruction error.

Expected benefit:

- Better top-k attention agreement at equal rank.
- Better behavior for GQA, where one KV head services multiple query heads.
- More useful rank sensitivity signal than plain SVD energy.

Cost:

- Calibration must collect Q and K activations per layer/head or per GQA group.
- Basis generation is more complex than plain SVD.
- Runtime needs either reduced-space attention or efficient reconstruction.

Hipfire fit:

- Good. It aligns with the existing quality-lossless KV criteria:
  attention KL, top-1 agreement, top-k overlap, and reconstruction SNR.

### ReCalKV-Style Value Calibration

Lead role: value-side calibration and optional fusion into output projection.

Expected benefit:

- Values are output-sensitive, so value compression should be calibrated
  directly against the downstream output path.
- If part of the value transform folds into `W_o`, runtime avoids full value
  reconstruction.

Cost:

- Requires model-format changes and loader/runtime metadata.
- Fusion must preserve weight quantization contracts and Opus Quant calibration
  assumptions.
- Needs separate treatment for architectures with unusual attention output
  layouts.

Hipfire fit:

- Good as a model conversion path. It is less like a cache-only feature and more
  like a calibrated model format extension.

### QSVD-Style Rank Allocation

Lead role: allocate rank by layer/head sensitivity instead of using one global
rank.

Expected benefit:

- Same or better quality at fewer bytes.
- Natural fit for Opus Quant calibration artifacts.
- Lets easy layers drop to low rank while preserving hard attention layers.

Cost:

- Requires a rank planner and admission artifact.
- Rank choices must be rounded to hardware-friendly kernel shapes.

Hipfire fit:

- Good. Rank policy belongs beside quant policy: calibration decides the format,
  loader metadata records it, kernels consume fixed shapes.

### OjaKV-Style Online Adaptation

Lead role: fallback if static calibrated bases fail under distribution shift.

Expected benefit:

- Prompt-adaptive basis updates.
- Can preserve exact/full-rank anchor tokens while compressing the bulk.

Cost:

- Online basis state, re-orthonormalization, hyperparameters, and periodic decode
  updates.
- Harder to make deterministic and easy to evaluate.
- More moving parts in the runtime hot path.

Hipfire fit:

- Defer. Use only if calibrated static bases miss long-context or domain-shift
  gates.

### Plain SVD Cold Residual

Lead role: simple probe, oracle, and safety fallback.

Expected benefit:

- Training-free.
- Easy to compare with current CASK/KVarN cold merge.
- Can run during `idle_compact`.

Cost:

- Optimizes reconstruction, not attention fidelity.
- Basis overhead only amortizes on wide cold segments.
- Likely inferior to calibrated KQ/value-aware objectives at equal bytes.

Hipfire fit:

- Keep as a benchmark and optional cold-tier residual, not as the main calibrated
  design.

## Target Runtime Shape

### Metadata

Each model package needs a KV policy section with:

- Per layer and KV head/group rank for K and V.
- Basis or projection identifiers.
- Whether values are fused through the output projection.
- Latent scalar codec: KVarN bits for K and V, likely K8V8, K4V8, or K2V4.
- Exact-token or residual sidecar policy for anchors/outliers.
- Kernel shape class, rounded to supported RDNA paths.

### Cache Storage

The cache stores latent vectors instead of full K/V vectors:

```text
K_latent: [token, kv_group, r_k]
V_latent: [token, kv_group, r_v]
```

These latent vectors are then KVarN-packed. KVarN is still the byte codec; low
rank changes the dimensionality being encoded.

### Attention Kernels

There are two implementation modes:

1. Reconstruct-then-attend.
   - Easier to validate.
   - Preserves existing attention structure.
   - Gives memory savings but less compute savings.

2. Reduced-space attention.
   - Computes logits using projected Q and latent K.
   - Computes low-rank value output, then expands or fuses through output.
   - Higher ceiling, but requires custom kernels and careful scaling.

The final path should be reduced-space attention for decode. Reconstruction is a
bring-up and parity path.

### Rank Shape Constraints

Ranks must be hardware-aware. Candidate ranks should be rounded to shapes that
map cleanly to RDNA matrix/vector kernels, likely multiples such as 16, 32, 64,
or 96 depending on the path. Avoid saving bytes with ranks that force awkward
scalar cleanup or low occupancy.

## Calibration Pipeline

1. Collect calibration activations.
   - Q, K, V per layer/head or GQA group.
   - Attention outputs or enough statistics to score output error.
   - Representative long-context and retrieval-heavy prompts.

2. Build candidate bases.
   - Plain K/V SVD baseline.
   - KQ-SVD-style key bases.
   - Calibrated value bases with optional output fusion.

3. Allocate ranks.
   - Sweep target byte budgets.
   - Use rank sensitivity per layer/head/group.
   - Round ranks to kernel-supported shapes.

4. Quantize latent vectors with KVarN.
   - Test K8V8 as the quality reference.
   - Test K4V8 as the main compression candidate.
   - Test K2V4 only where rank reduction preserves enough margin.

5. Optional QAT.
   - Fine-tune only the low-rank/value-fusion pieces or small adapter surfaces.
   - Do not train away true information loss from a bad codec choice.

6. Emit package metadata.
   - Record policy, ranks, basis hashes, codec bits, calibration fingerprint, and
     expected kernel class.

## Evaluation Gates

Do not promote from reconstruction metrics alone. A candidate must pass:

- Reconstruction SNR target, with KVarN-8/K7 as operational full-fidelity floor.
- Attention KL below the quality-lossless threshold.
- 100% top-1 attention agreement on calibration/eval slices.
- Greater than 99.9% top-8 overlap where that is the operating target.
- KLD/PPL against BF16 or accepted high-precision reference.
- Long-context retrieval and multi-turn coherence.
- AR and DFlash decode perf checks.
- RDNA2/RDNA3/RDNA4 kernel compatibility review.

The relevant comparison is quality per byte and tokens/sec per byte versus:

- Current flat KVarN.
- Current hierarchical CASK/KVarN/rotation.
- Plain SVD cold residual.
- Calibrated latent KV.

## Milestones

### M0: Offline Oracle

Build an offline analysis path from captured Q/K/V:

- Plain SVD rank sweep.
- KQ-SVD-style rank sweep.
- Value calibration/fusion simulation.
- Equal-byte comparison against CASK/KVarN cold merge.

Output: rank/byte/quality curves and the first candidate rank policy.

### M1: Reconstruction Runtime Path

Add package metadata and a reconstruction parity path:

- Load calibrated bases.
- Store latent K/V.
- KVarN-pack latent vectors.
- Reconstruct into the existing attention path.

Output: correctness and quality evidence without committing to final kernels.

### M2: Reduced-Space Decode Kernel

Add the real decode path:

- Project Q into key latent space.
- Attend against latent K.
- Consume latent V and expand/fuse output.
- Keep exact-token or residual sidecars where policy requires.

Output: speed/byte evidence on supported RDNA targets.

### M3: Cold-Tier Composition

Compose calibrated latent KV with the existing hierarchical system:

- Hot tier can remain raw or high-fidelity KVarN.
- Cold tier uses calibrated latent KVarN.
- CASK-style merge and low-rank residuals remain as cold archive tools.

Output: long-context Pareto point that beats current CASK/KVarN/rotation at
equal quality or equal bytes.

## Open Questions

- Is KQ-SVD enough, or do value/output interactions dominate the failure cases?
- Does KVarN-4 latent storage preserve the same quality-lossless behavior as
  full-dimensional KVarN-8?
- Which ranks are hardware-native enough to justify as public policy choices?
- Should exact anchor tokens be selected by vnorm, attention error, KQ residual,
  or a hybrid score?
- How much QAT is useful after calibrated basis selection, and where should it
  apply?

## Current Recommendation

Make calibrated latent KV the main research track. Keep simple cold residual SVD
as a measurement baseline and fallback. Do not replace KVarN; use KVarN as the
latent scalar codec. The final goal is a model-packaged KV policy that is
calibrated like Opus Quant, loaded with the model, and consumed by custom
RDNA-aware kernels.
