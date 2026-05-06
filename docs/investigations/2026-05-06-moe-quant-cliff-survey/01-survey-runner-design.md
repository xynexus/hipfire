# 01. Survey Runner Design

**Branch:** `survey/moe-quant-cliff-2026-05-06`
**Hardware:** hiptrx (4× R9700 gfx1201, Threadripper 9970X 32-core single-NUMA, 128 GB DDR5)
**Investigator:** Claude Opus 4.7 (1M context), under Kaden's contract.

## Mission

Run an empirical survey of Qwen 3.5/3.6 family weight statistics, quant
reconstruction errors, and forward-pass activation magnitudes to confirm or
refute the Super-Expert hypothesis (arXiv 2507.23279) on these specific
models at MQ4. The survey is independent of prior evidence; the data
decides.

## Prior evidence (2026-05-05)

Captured in `../2026-05-05-qwen36-a3b-mq4-fragility/`:

- `expert_absmax_stats.py` + `absmax_results.json`: per-row absmax/median
  ratio over every routed expert × all 40 layers of 3.5-A3B and 3.6-A3B.
  Found 3.6 weights are 0.97-0.99× of 3.5 (fractionally lighter-tailed).
  Also surfaced "down_proj p99 max ≈ 37M" tail-of-tails in a minority
  of layers across both models.
- `quant_recon_error.py` + `quant_recon_results.json`: per-row reconstruction
  MSE for MQ4G256 / MQ4G64 / MQ4G256+sidecar{4,16,64} / MQ6G256 on 768
  worst-tailed rows of 3.6-A3B `down_proj`. Cosine similarity ≥ 0.991 across
  all schemes.
- `INVESTIGATION.md`: 12-hypothesis log with verdicts. Bug localized to
  hipfire's MoE forward path (dense Qwen3.5 4B/9B/27B handle the same
  prompt + sampler config without attractor; only MoE models cliff).
- `issue-171-update.md`: 7-prompt × 5-sampler matrix on hipx; no sampler
  config gives 3.6-A3B clean output across the matrix.

What's missing from the prior evidence:

- **Dense controls** (3.5-9B, 3.5-27B) for per-tensor NRMSE comparison.
  Without them we can't distinguish "MoE-specific quant pathology" from
  "all Qwen3.5 weights have these stats but only MoE forward propagates
  them into a cliff."
- **Activation absmax during forward pass.** The 2026-05-05 work was
  weight-only. Per-channel activation absmax is the AWQ signal (and the
  Super-Expert paper's `down_proj_output_max` signal) that distinguishes
  pathological experts from healthy ones.
- **122B-A10B coverage.** Tests whether SE pathology scales linearly with
  expert count (122B has roughly 8× the experts of A3B; expect roughly
  8× the SEs if the pathology is per-expert constant).
- **FWHT pre/post comparison.** The MQ4 path applies Walsh-Hadamard
  rotation per 256-element group before quantization. We have no measurement
  of how much the rotation actually equalizes per-group outliers in
  practice on these specific weight distributions.

## Scope

**Primary scope (four models, available now):**

1. **Qwen 3.5-9B** (dense, control). Local on k9lin; needs rsync to hiptrx.
2. **Qwen 3.5-27B** (dense, mid-size control). Local on k9lin; needs rsync.
3. **Qwen 3.5-35B-A3B** (35B/3B MoE, 128 experts × 40 layers). On hiptrx already.
4. **Qwen 3.6-35B-A3B** (primary issue #171 reproducer target). On hiptrx already.

**Deferred / out of immediate scope:**

5. **Qwen 3.5-122B-A10B** (audited 2026-05-06: not in any local HF cache —
   neither k9lin nor hiptrx has the snapshot). Even if downloaded (~244 GB
   at bf16), D3 (forward pass) would require 244 GB resident, vs hiptrx's
   128 GB VRAM + 122 GB RAM = 250 GB total addressable, with realistic
   Python/transformers/activation overhead pushing total demand over the
   ceiling. Decision deferred to a separate work item: either download +
   D1/D2/D4 only (weight-side, streamable) or skip until SE evidence on
   the 4 primary models warrants the cost.

Reference dtype: bf16 from upstream safetensors. Quant target: MQ4G256 with
FWHT rotation as currently produced by `crates/hipfire-quantize`.

## The four diagnostics

### D1. Per-tensor NRMSE: MQ4 dequant vs bf16 reference

For every weight tensor in the model:

1. Read bf16 reference from safetensors.
2. Apply the same MQ4G256-FWHT pipeline used by hipfire-quantize:
   per-256-element group, FWHT rotate with **production seeds 42 / 1042**
   passed through `gen_fwht_signs` (LCG with `state * 1103515245 + 12345
   & 0x7fffffff`, bit (state >> 16) & 1), per-group min/max → 4-bit
   asymmetric quant, store `(scale, min, nibbles)`. Verified against
   `crates/hipfire-quantize/src/main.rs:1530-1531` (the `gen_fwht_signs(42, 256)`
   and `gen_fwht_signs(1042, 256)` calls used by all MQ format branches).
3. Dequantize back to f32: per-group `weight = scale * nibble + min`,
   inverse-FWHT (`signs1 * fwht256_raw(signs2 * x)`, self-inverse up to
   sign placement).
4. **NRMSE definition (explicit):** `NRMSE = sqrt(MSE) / sqrt(var(reference))`
   where `MSE = mean((reference - dequantized)^2)` and
   `var = mean((reference - mean(reference))^2)`. Equivalent to
   `sqrt(MSE / var)` but written this way to make the units explicit.
   Lower is better; 0 means perfect reconstruction.
5. **Also report:** mean-cosine-similarity per tensor as a separate stat,
   for direct comparison with the 2026-05-05 simulation results
   (which used cos sim + MSE).

Reported per (layer, projection, expert_index_or_dense). Dense projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj` for non-MoE) and MoE projections (`mlp.experts.{i}.gate_up_proj`,
`mlp.experts.{i}.down_proj`, `mlp.gate.weight` router, `mlp.shared_expert.*`)
all measured.

Sample size: full tensor. No subsampling.

**Note on production-matched seeds:** the inherited 2026-05-05
`quant_recon_error.py` simulation used `0xCAFEBABE` with NumPy
`default_rng` (PCG64), NOT production's seeds 42/1042 with the LCG.
Those prior results are valid for inter-scheme comparison (relative
ranking of MQ4G256 vs MQ4G64 vs sidecar etc.) but absolute MSE values
differ from production. The new survey runner MUST use production seeds.

### D2. Per-expert down_proj absmax + ratio statistics

For each MoE layer × each expert × the down_proj weight tensor:

1. Per-row absmax: `row_max[i] = max_j |W[i, j]|` over the K-axis.
2. Per-row median absmax: `row_med[i] = median_j |W[i, j]|`.
3. Per-row tail ratio: `ratio[i] = row_max[i] / max(row_med[i], 1e-9)`.
4. Tensor-level absmax: `tensor_max = max_i row_max[i]` (raw magnitude).
5. Distribution stats on both `row_max[]` and `ratio[]`: mean, p50, p90,
   p99, p99.9, max.

**Important: report BOTH absolute magnitude and ratio.** The 2026-05-05
finding "down_proj p99 max ≈ 37M" was the **absmax/median ratio** per
`per_row_absmax_median()` in `expert_absmax_stats.py:124`, not absolute
weight magnitude. A ratio of 37M means one row has its absmax 37 million
times its median absmax (one extreme outlier). Actual absmax magnitudes
in transformer weights are typically O(0.1-10). The ratio is the
quant-relevant signal because it determines how badly per-row absmax
quant compresses the bulk distribution. Report both so neither is lost.

Reported per (layer, expert). Output also includes:
- Pre-FWHT absmax (= reference per-row absmax).
- Post-FWHT absmax (= what the quant scale actually has to fit).
The pre/post comparison is D4.

Per arXiv 2507.23279 the SE signature is `down_proj output max` which is
weight × activation. The activation half is D3 below; D2 captures the
weight half.

### D3. Per-channel activation absmax during forward pass

Forward pass on a fixed calibration set, with hooks at each transformer
layer recording per-channel absmax of:

1. Input to `gate_proj` / `up_proj` (= `x` after attention residual + RMSNorm).
2. Input to `down_proj` (= `gate_proj(x) * silu(up_proj(x))` for non-MoE,
   or routed-expert intermediate for MoE).
3. Input to attention sub-block (= residual stream pre-RMSNorm).
4. Output of `down_proj` (post-routed-combine for MoE; this is the
   Super-Expert paper's primary diagnostic).

For MoE models, also record per-expert routing frequency (which experts
were selected) per layer, so we can attribute D3#4 outliers to specific
experts.

Calibration set: **derived in 1A** since no `benchmarks/calib/blended-32prompts.jsonl`
exists in tree (only `calib-1m.txt`, `calib-5m.txt`, and `profiles/`).
Source plan:

1. 7 prompts from `docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/`
   prompt matrix (`agent_prompt`, `sheep`, `capital`, `code_simple`,
   `code_complex`, `prose`, `math`, `code_review`).
2. 25 additional prompts sampled from `benchmarks/calib/calib-1m.txt`
   (token-shuffled to ~512 tokens each).
3. Committed as `docs/investigations/2026-05-06-moe-quant-cliff-survey/calibration_corpus.jsonl`
   with md5 recorded in summary. Same byte-identical prompts across all
   models for cross-model comparability.

Output: per (layer, channel, hook_point) → absmax value, plus per-expert
hit count for MoE.

Forward pass dtype: bf16 (matches reference). Implementation: HuggingFace
`transformers` with manual forward hooks; whole-model on one GPU for each
of the four primary models. Loaded weights are bf16 from safetensors
directly (NOT MQ4 dequantized — we want the reference activation
distribution, not the post-quant one).

**Environment requirement:** transformers + torch (ROCm wheel) installed
on hiptrx. Verified 2026-05-06: hiptrx has neither package installed. Phase
1A unit-test step pauses until env setup completes:

```bash
ssh hiptrx 'pip install --user transformers safetensors ml_dtypes accelerate \
  torch --index-url https://download.pytorch.org/whl/rocm6.2'
```

Adjust the rocm wheel URL to match hiptrx's actual ROCm version (7.2.2
per memory). If no matching wheel exists, fall back to building torch
from source against the system ROCm install, OR run D3 on k9lin (single
gfx1100 GPU) with serial per-model execution and accept the GPU-occupancy
loss for D3 only.

**Compatibility note:** stock transformers may not yet have classes for
`Qwen3_5MoeForCausalLM` / `Qwen3_5MoeForConditionalGeneration` (the
Qwen3.5/3.6 MoE config classes). Check at unit-test time. If unsupported,
use `trust_remote_code=True` with the model's bundled `modeling_*.py`,
which Qwen ships in their HF repos.

### D4. FWHT pre/post NRMSE comparison

For every weight tensor:

1. Per-group absmax of bf16 reference (no rotation): `pre_max[g] = max_i
   |W[g*256:(g+1)*256, i]|` for the K-axis.
2. Apply FWHT rotation per-group: `W_rot[g] = FWHT(W[g], signs1, signs2)`
   with the same seeds as MQ4 quantizer.
3. Per-group absmax post-rotation: `post_max[g] = max_i |W_rot[g, i]|`.
4. Reduction ratio per group: `post_max / pre_max`.

Aggregated as: per-tensor mean reduction ratio, p99 ratio, max ratio.
Tells us whether FWHT is actually flattening per-group outliers or just
shifting them around.

If FWHT effectively flattens (mean ratio < 0.5), the SE pathology is
absorbed by rotation and the cliff cannot be in weight quant.
If FWHT only marginally flattens (mean ratio close to 1.0), pre-rotation
outliers persist into the quant grid and the SE signal survives.

## Output schema

Each per-layer JSONL record:

```jsonl
{
  "model": "qwen3.5-a3b",
  "layer_idx": 12,
  "tensor": "mlp.experts.34.down_proj.weight",
  "expert_idx": 34,
  "projection": "down_proj",
  "shape": [2048, 1408],
  "n_elements": 2883584,
  "d1_nrmse_mq4g256_fwht": 0.0341,
  "d2": {
    "row_max_mean": 0.412,
    "row_max_p50": 0.401,
    "row_max_p99": 1.83,
    "row_max_max": 4.71,
    "tensor_max": 4.71,
    "row_med_mean": 1.04e-7,
    "ratio_mean": 8.4e3,
    "ratio_p50": 6.2e3,
    "ratio_p99": 9.2e5,
    "ratio_max": 3.7e7
  },
  "d3": null,
  "d4": {
    "fwht_pre_max_p99": 0.92,
    "fwht_post_max_p99": 0.41,
    "reduction_ratio_mean": 0.44,
    "reduction_ratio_p99": 0.78
  },
  "wall_time_seconds": 1.4
}
```

D3 (activation absmax) lives in a separate JSONL keyed by
`(model, layer_idx, hook_point, channel)` because its dimensionality differs
(per-channel, not per-tensor).

Per-model summary written to `summary.json`:

```json
{
  "model": "qwen3.5-a3b",
  "n_layers": 40,
  "n_experts_per_layer": 128,
  "total_tensors_surveyed": 12867,
  "calibration_corpus_md5": "...",
  "calibration_prompt_count": 32,
  "outlier_experts": [
    {
      "layer": 17,
      "expert": 42,
      "criterion": "d2_ratio_p99 > 3*sigma_global_ratio_p99",
      "ratio_p99": 1.2e7,
      "absmax_p99": 0.84,
      "z_score": 8.4
    }
  ],
  "summary_stats": { ... }
}
```

## Implementation

**Language:** Python 3 with NumPy, safetensors, transformers, ml_dtypes.
Matches the 2026-05-05 scripts and avoids reimplementing bf16 readers.
GPU work uses transformers' built-in CUDA/HIP backend (transformers picks
HIP automatically on ROCm builds of PyTorch).

**Shared FWHT helpers** in `quant_ops.py` codify the production seeds
(42, 1042) and the LCG sign generator from
`crates/hipfire-quantize/src/main.rs:430-436`. Any future quant simulation
imports from this module rather than re-deriving signs. The 2026-05-05
`quant_recon_error.py` is left as-is (history of the prior simulation);
its `0xCAFEBABE` seeds are documented as simulation-only in 01's "Note
on production-matched seeds" above.

**Layout:**

```
scripts/quant-survey/
├── survey_runner.py        # main entry point, CLI
├── diagnostics/
│   ├── __init__.py
│   ├── d1_nrmse.py         # per-tensor NRMSE
│   ├── d2_down_proj_max.py # per-expert magnitude stats
│   ├── d3_activation.py    # forward-pass hooks
│   └── d4_fwht.py          # pre/post rotation comparison
├── quant_ops.py            # mq4g256_fwht_quant, fwht256, signs, dequant
├── safetensors_reader.py   # bf16-aware tensor reader (from 2026-05-05)
├── calibration_corpus.py   # load + tokenize the 32 prompts
└── README.md
```

**CLI:**

```
survey_runner.py \
  --model qwen3.5-9b | qwen3.5-27b | qwen3.5-a3b | qwen3.6-a3b \
  --hf-cache ~/.cache/huggingface/hub \
  --output-dir /tmp/hiptrx-survey/runs/<model>/ \
  --gpu 0 \
  --diagnostics d1,d2,d3,d4 \
  --calibration-corpus benchmarks/calib/blended-32prompts.jsonl
```

The runner is responsible for:

1. Resolving `--model` to the actual HF cache snapshot path.
2. Iterating through layer indices in order, streaming tensors from
   safetensors shards.
3. Computing D1, D2, D4 on CPU (NumPy, parallelizable across cores).
4. For D3: loading the model into transformers with `device_map="auto"`,
   running the calibration corpus with hooks, recording per-channel
   activation absmax.
5. Writing per-tensor JSONL records to `<output-dir>/per_tensor.jsonl`
   and per-channel records to `<output-dir>/per_channel.jsonl`, plus
   a summary at `<output-dir>/summary.json`.
6. Tagging hook output with which expert was active for MoE layers
   (read top-K indices from the routing softmax).

## Parallelization on hiptrx

hiptrx topology (verified 2026-05-06):

- 4× AMD Radeon AI PRO R9700, gfx1201, 32 GB VRAM each = 128 GB aggregate.
- Threadripper 9970X, 32 cores, **single NUMA node** (UMA mode or
  single-CCD config). `numactl --cpunodebind=0 --membind=0` is the only
  valid binding.
- 125 GB DDR5 system RAM.

Round 1 (the four small/mid models, parallel):

```bash
HIP_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  python scripts/quant-survey/survey_runner.py --model qwen3.5-9b --gpu 0 \
    --output-dir /tmp/hiptrx-survey/runs/qwen3.5-9b/ &

HIP_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --membind=0 \
  python scripts/quant-survey/survey_runner.py --model qwen3.5-27b --gpu 1 \
    --output-dir /tmp/hiptrx-survey/runs/qwen3.5-27b/ &

HIP_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 \
  python scripts/quant-survey/survey_runner.py --model qwen3.5-a3b --gpu 2 \
    --output-dir /tmp/hiptrx-survey/runs/qwen3.5-a3b/ &

HIP_VISIBLE_DEVICES=3 numactl --cpunodebind=0 --membind=0 \
  python scripts/quant-survey/survey_runner.py --model qwen3.6-a3b --gpu 3 \
    --output-dir /tmp/hiptrx-survey/runs/qwen3.6-a3b/ &

wait
```

Each process gets exclusive ownership of its GPU. CPU cores are shared
(32 cores ÷ 4 processes = 8 cores each by default; use
`OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8` to keep NumPy from oversubscribing).
System RAM: 4 × ~30 GB peak per worker = ~120 GB, fits in 125 GB. If we
hit pressure, reduce concurrency to 2 at a time.

Round 2 (122B-A10B): **deferred** until model is downloaded to hiptrx
AND a memory plan exists for D3 (current ceiling: 250 GB total addressable
vs 244 GB at bf16 + overhead). Two viable paths if Phase 2 results on
the four primary models warrant the cost:

- **Weight-only:** D1/D2/D4 streamed layer-by-layer from safetensors,
  CPU-only. Works at any model size. Skip D3 for 122B; rely on smaller-
  model D3 for the activation half of the SE signature.
- **Activation-only via int8 load:** transformers `load_in_8bit=True`
  (bitsandbytes-rocm) drops bf16 to int8, ~120 GB. Fits across 4× 32 GB
  VRAM. But activations differ from bf16 reference; D3 results would be
  int8-conditioned, not bf16-conditioned. Document the difference if used.

Decision deferred to a Phase 1B follow-up after the four primary models
complete.

GPU monitoring throughout:

```bash
while true; do
  rocm-smi --showuse --showmemuse --csv >> /tmp/hiptrx-survey/logs/rocm-smi.csv
  sleep 300  # 5 min cadence
done &
```

## Sanity checks

Before declaring Phase 1B done, the following must pass:

1. **Per-layer NRMSE on 3.6-A3B matches recon doc.** The 2026-05-05 recon
   identified layers 10, 20, 35 as having the worst-tailed down_proj rows
   (`max_rows=64` worst-tailed sample per expert). The survey's per-tensor
   NRMSE on those layers' down_proj should be visibly worse than the median
   (top decile). If not, the survey runner has a bug.
2. **Mean cosine similarity ≥ 0.991** on D1 across all four primary models.
   This reproduces the 2026-05-05 sidecar simulation result direction.
   Note: absolute NRMSE/MSE values will differ from 2026-05-05 because
   that script used `0xCAFEBABE` seeds, not production 42/1042. The
   survey uses production. So a re-derivation is expected; the
   ≥ 0.991 cos sim direction should still hold.
3. **Dense models have no MoE-specific tensors.** D2 and D3#4 (down_proj
   output max) should report 0 expert tensors and 1 dense down_proj per
   layer for 3.5-9B / 3.5-27B.
4. **D2 ratio statistics include 37M-class outliers on A3B down_proj.**
   The 2026-05-05 ratio result (`p99_ratio_max_across_layers ≈ 37M` for
   down_proj on both 3.5-A3B and 3.6-A3B) should reproduce. If absent,
   either the runner has a bug or the prior result was an artifact of
   the simulation's seeds (unlikely since `expert_absmax_stats.py` is
   bf16-direct, doesn't apply FWHT).

## Cross-cutting concerns

- **Statistical rigor:** every reported stat carries `n` (sample size).
  Outlier classification uses explicit threshold in the summary
  (default: `> 3σ from per-model median on either D1 NRMSE or D2 ratio_p99`).
  Note: D2 outliers are classified on the **ratio** (absmax/median),
  not raw absmax — the ratio is what determines per-row quant scale
  damage to the bulk distribution.
- **No pre-judging:** the runner emits all data; classification is in
  the synthesis (02). The runner does not say "expert X is a Super Expert."
- **Reproducibility:** FWHT seeds (42, 1042) match `gen_fwht_signs` in
  `crates/hipfire-quantize/src/main.rs:430`. Calibration corpus committed
  as a JSONL with md5 in the summary.
- **Failure recovery:** runner emits per-tensor JSONL incrementally, so
  partial results are preserved if a process dies mid-run. Resume by
  diffing tensor names already present in the output JSONL.

## What's NOT in scope

Deferred to Phase 2 or later:

- **Perplexity ablations** (All-MQ4 / All-Q8 / Outlier-Q8 forward).
  These need a forward pass on a held-out test set, not the calibration
  set. Phase 2 work.
- **Routing-distribution comparison** between fp16 and MQ4 forward.
  Fivetide showed in #171 (2026-05-06) that this is the MoE cliff
  mechanism; their evidence is sufficient. Survey doesn't re-derive it.
- **Outlier-isolation sidecar prototype.** The 2026-05-05 simulation
  showed at sensible byte budgets (sidecar4/16/64) it doesn't beat MQ6G256
  per-element. Phase 3 work if SE confirmation justifies it.
- **AWQ-style activation-aware scaling.** Phase 3 implementation work
  contingent on Phase 2 confirming the SE hypothesis.
- **Kernel changes.** The runner is read-only against existing weights.
  No HIP source edited.

## Pre-registration of Phase 1 success criteria

The survey passes Phase 1 if and only if:

1. All 5 models surveyed end-to-end (`summary.json` written for each).
2. D1, D2, D4 emitted for every tensor in every model. D3 emitted for
   every (layer, hook_point, channel) tuple.
3. Sanity checks 1-4 above pass.
4. Investigation log entry per significant finding or blocker.

The survey does NOT pass/fail on whether SEs are confirmed; that's
Phase 2. Phase 1 is just data.

## Open questions to resolve in 1A code authoring

1. **MoE expert tensor naming.** In some Qwen3.5/3.6 checkpoints
   experts are stacked 3D (`mlp.experts.gate_up_proj` shape
   `[n_experts, M, K]`); in others they're already split per-expert
   (`mlp.experts.{i}.gate_up_proj` 2D). The runner must handle both.
   Reference: `crates/hipfire-quantize/src/main.rs:1876-1938`.
2. **122B activation hooks under tensor-parallel.** transformers'
   `device_map="auto"` shards weights but activations cross GPUs.
   Hook output may be on a different device than expected. Need to
   `.cpu()` or `.to(device_idx)` consistently before recording.
3. **Calibration corpus.** Pin to `benchmarks/calib/<file>` with a
   committed md5. If no suitable corpus exists, derive 32 prompts from
   `docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/repro_failing.jsonl`
   plus the recon doc's 7-prompt validation matrix. Decision lives in
   `calibration_corpus.py`.
