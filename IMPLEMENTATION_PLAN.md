# feat/learnable-fwht — Multi-Phase Implementation Plan

> Branch: `feat/learnable-fwht` off `worktree-awq-raw-sumsq-converter` HEAD `1b58b184`.
> Worktree: `.claude/worktrees/learnable-fwht/`
> Reference plan: `docs/plans/quant-strategy-research-recommendations.md` (master)
> Reference postmortem: `docs/investigations/2026-05-22-mq4-kld-asymptote-hunt-postmortem/README.md`
> Reference paper: ButterflyQuant arXiv:2509.09679 (Xu et al., Feb 2026)
> Comparator: ParoQuant arXiv:2511.10645 (Liang et al., ICLR 2026)

## Locked decisions (from user, 2026-05-22)

| Decision | Value |
|---|---|
| **Quality bar (dense)** | KLD ≤ 0.10 on 0.8B/9B/27B (match ParoQuant range) |
| **A3B-MoE rule** | Push to asymptote = no KLD delta after 5 iterations. Don't apply 0.10 gate. |
| **Model scope** | 0.8B + 9B + 27B + A3B-MoE (full trunk) |
| **Perf ceiling** | ≤5% slower than current MQ4 decode |
| **Calibration recipe** | ButterflyQuant paper hparams (128 samples, SGD+cosine) |
| **Storage (prototype)** | Optional sidecar on existing MQ4G256 |
| **Storage (production)** | NEW quant type (`MQ4G256Butterfly`, fail-closed) |
| **Phase gating** | Auto-fire through phases. Halt ONLY at master push. |
| **Push policy** | Push to own branch freely; **explicit user approval required for master push** |
| **Cross-arch ports** | gfx942 (mi300) + gfx11 (k9lin/hipx) + gfx12 (hiptrx) |
| **Final deliverable** | Default-on after coherence/speed gates pass + benchmarks + investigation writeup |

## Hardware

- **mi300 droplet** (gfx942 / CDNA3): primary compute, this branch's home GPU
- **hiptrx** (gfx1201 / RDNA4 / gfx12): RESERVED for `feat/paro-g256-perfmax` agent — DO NOT ssh during this work unless idle-gated (`ssh hiptrx 'ps -ef | grep hipfire'` returns empty)
- **hipx** (gfx1100 / RDNA3): available for gfx11 port validation, idle-gated
- **k9lin** (gfx1100 / RDNA3): user's local machine, idle-gated

Cross-arch ports happen AFTER mi300 gfx942 implementation is validated.

## Phase structure

The plan has 16 phases. Phases 1-7 are Python smoke + validation (cheap, fast).
Phases 8-12 are Rust + HIP implementation (irreversible commitment).
Phases 13-15 are validation, perf, and rollout. Phase 16 halts for master push approval.

**Auto-fire through phases until stop condition or Phase 16.**

---

## Phase 1 — Python CPU butterfly reference

**Goal:** `butterfly256(theta, x)` implementation with identity-residual verification.

**Files to create:**
- `scripts/verify_butterfly256.py`
- `scripts/butterfly_core.py` (importable module with butterfly + FWHT match)

**Algorithm:**
```python
def butterfly256(x, theta):
    # x: shape [..., 256], theta: shape [8, 128] (8 layers × 128 pairs)
    # 8 stride-doubling layers, each applies 128 independent Givens rotations
    for layer in range(8):
        stride = 1 << layer
        for pair_idx in range(128):
            # pair (i, j) = compute_pair_indices(layer, pair_idx)
            cos_t = cos(theta[layer][pair_idx])
            sin_t = sin(theta[layer][pair_idx])
            a, b = x[..., i], x[..., j]
            x[..., i] = cos_t * a - sin_t * b
            x[..., j] = sin_t * a + cos_t * b
    return x

def fwht_residual_init(signs1, signs2):
    # Return theta values such that butterfly256(x, theta) == cpu_fwht_256(x, signs1, signs2)
    # Then in residual form, residual_theta = theta - theta_fwht.
    # B_residual(0) means residual_theta = 0 means transform = exact FWHT.
    ...
```

**Verification criterion:**
- For 1000 random `x` vectors and the canonical FWHT seeds (42, 1042):
  - `butterfly256(x, fwht_residual_init(signs1, signs2))` must match `cpu_fwht_256(x, signs1, signs2)` to within F32 precision (max abs delta < 1e-5)
- Apply inverse: `butterfly256_inverse(butterfly256(x, theta), theta) == x` for random theta

**Stop condition:** Phase 1 cannot fail without halting — if identity residual doesn't reproduce FWHT, the math has a bug that must be fixed before any optimization.

**Commit:** `feat(butterfly): Python CPU butterfly256 reference + identity residual verification`

---

## Phase 2 — Python offline optimizer

**Goal:** `scripts/learn_butterfly_mq.py` — offline residual butterfly angle optimizer per tensor.

**Algorithm (per ButterflyQuant paper):**
- Optimization: SGD + cosine LR schedule (paper choice; Adam ablation deferred)
- Loss: layer-wise reconstruction MSE — `min_theta E_x ||W·x - dequant(quant_mq4(W·R(theta)^T))·R(theta)·x||²`
- Calibration: 128 samples × 2048 ctx (paper exact hparams)
- Initialize: `theta_residual = 0` → R = current FWHT (KLD floor = baseline)
- 500-700 optimization steps per tensor (paper guidance)
- Regularizer: optional small L2 penalty on `theta_residual` magnitude

**Inputs:**
- HF model dir (BF16 safetensors)
- Calibration corpus (`/workspace/calibration-mix-v1.txt` md5 `68a1d2e62117e692e0e04c2811349aaf` for our pipeline OR use the paper's 128-sample mix — agent picks per ButterflyQuant exact reproduction)
- imatrix or layer-wise activation samples
- AWQ scales sidecar (optional, for `AWQ·butterfly` composition)

**Outputs:**
- Per-tensor `theta_residual.npz` — shape `[8, 128]` of float32 angles per AWQ-target tensor
- Optionally: precomputed `sincos.npz` for runtime efficiency
- HFSC-style sidecar with butterfly metadata

**Phase 2 commit:** `feat(butterfly): Python offline optimizer with SGD+cosine paper hparams`

---

## Phase 3 — Smoke validation (5 tensors × 8 sequences × 2 epochs)

**Goal:** Confirm butterfly residual moves the needle on a SMALL test BEFORE committing to a full-tensor full-slice run.

**Setup:**
- Target tensors: 5 AWQ-target Linears from `model.language_model.layers.0` (q_proj, k_proj, v_proj, gate_proj, up_proj)
- Sequences: 8 (paper-style mini-set)
- Epochs: 2
- Eval slice: `--max-chunks 16` short-slice KLD on Qwen3.5-0.8B

**Pass criterion (HARD):**
- Smoke KLD strictly < MQ4+AWQ baseline on the same 5 tensors × same short-slice
- If smoke KLD ≥ baseline → **STOP, halt, document, do not full-run**
- If smoke KLD < baseline → continue to Phase 4

**Why this gate matters:** This is the gate we DIDN'T have in the grad-scale and PARO K=0 experiments. Without it, the agent runs full 138-tensor × 128-seq × 5-epoch trains and discovers proxy-doesn't-transfer too late. The smoke gate catches the failure mode in ~30 minutes vs ~4 hours.

**Phase 3 deliverable:** `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/smoke-report.md`

---

## Phase 4 — Full Python validation on 0.8B

**Goal:** Establish butterfly residual quality on the smallest dense model.

**Setup:**
- All 138 AWQ-target tensors
- 128 calibration sequences × 2048 ctx (ButterflyQuant paper recipe)
- 500-700 SGD+cosine steps per tensor
- Full 1175-chunk slice KLD eval on canonical kldref

**Pass criterion (HARD):**
- 0.8B butterfly KLD ≤ 0.10 → continue
- 0.8B butterfly KLD > 0.10 → **STOP, halt, document, do not port to Rust**

**Phase 4 deliverable:** `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/0.8b-report.md`

---

## Phase 5 — Python validation on 9B

**Goal:** Confirm butterfly scales to mid-size dense.

**Setup:** Same recipe as Phase 4 but on Qwen3.5-9B.

**Pass criterion (HARD):**
- 9B butterfly KLD ≤ 0.10 → continue
- 9B butterfly KLD > 0.10 → **STOP, halt, document**

**Phase 5 deliverable:** `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/9b-report.md`

---

## Phase 6 — Python validation on 27B

**Goal:** Confirm butterfly scales to large dense.

**Setup:** Same recipe but on Qwen3.6-27B. Note: there's a kldref mismatch concern from the prior session (Qwen3.5-27B vs Qwen3.6-27B kldref). If kldref needs rebuilding, do that as a sub-phase before measurement.

**Pass criterion (HARD):**
- 27B butterfly KLD ≤ 0.10 → continue
- 27B butterfly KLD > 0.10 → **STOP, halt, document**

**Phase 6 deliverable:** `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/27b-report.md`

---

## Phase 7 — Python validation on A3B-MoE (asymptote search)

**Goal:** Find where butterfly residual converges on A3B-MoE. Don't apply 0.10 gate.

**Setup:**
- Same recipe but iterate UNTIL convergence
- Convergence criterion: **no KLD delta > 0.001 across 5 consecutive iterations**
- Record asymptote KLD as the measurement (compare to MQ4+AWQ floor 0.95 and native PARO 0.0933)
- Per-expert butterfly application (A3B has 256 routed experts × layer)

**Pass criterion (SOFT):**
- A3B butterfly converges to asymptote — report number whatever it is
- The A3B asymptote informs the dense/MoE split: if butterfly hits ≤0.15 on A3B, it's competitive with native PARO; if ≥0.5, MoE story is "ship PARO via paro-g256-perfmax for MoE; butterfly for dense"

**Phase 7 deliverable:** `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/a3b-asymptote.md`

---

## Phase 8 — Rust quantizer port (prototype, sidecar)

**Goal:** Wire butterfly residual into `hipfire-quantize` with sidecar (prototype storage format).

**Files to modify:**
- `crates/hipfire-quantize/src/main.rs`:
  - Add `--butterfly-residual <npz_dir>` flag accepting the Phase 4-7 output dir
  - On AWQ-target tensors, after `cpu_fwht_256(W·diag(s))`, additionally apply learned butterfly residual `B_residual(theta)` before quantization
  - Emit `<tensor>.butterfly_residual.weight` sidecar (F16 `[8, 128]` shape)
- `crates/hipfire-quantize/src/lib.rs`: pub mod butterfly
- `crates/hipfire-quantize/src/butterfly.rs` (new): butterfly residual functions matching Phase 1 Python reference

**Verification:**
- Byte-equal output (.hfq file) between Python-reference + hipfire-quantize Rust-impl on small synthetic tensor
- Phase 4-7 Python KLDs reproduced when re-quantizing via Rust

**Phase 8 commit:** `feat(quantize): butterfly residual sidecar emission`

---

## Phase 9 — HIP rotate kernel (gfx942 first)

**Goal:** `kernels/src/rotate_x_mq_bfly.hip` for gfx942 (mi300).

**Algorithm:**
- Inputs: activation `x[K]`, AWQ scales `s[K]` (optional), FWHT seeds (D1, D2), butterfly residual `theta[8][128]` or precomputed `sincos[8][128][2]`
- 1. (Optional) `x[i] /= s[i]` if AWQ sidecar exists
- 2. Apply FWHT-with-signs (matches current `rotate_x_mq.hip`)
- 3. Apply residual butterfly: 8 stride-doubling layers × 128 Givens rotations using sin/cos tables
- 4. Write `x_rot[K]` to scratch for downstream prerotated-GEMV

**Kernel structure:**
- Block dim 256 (1 channel per thread)
- Sin/cos tables loaded to shared memory (~32 KB for 8×128×2×F32)
- Stride-doubling pattern matches FWHT (cache-coherent)

**Verification:**
- Byte-equal between HIP output and Python reference on the same input
- Performance: target ≤5% slower than current `rotate_x_mq.hip`

**Phase 9 commit:** `feat(kernel/gfx942): rotate_x_mq_bfly.hip with sin/cos residual layers`

---

## Phase 10 — HIP kernel ports to gfx11 + gfx12

**Goal:** Same kernel adapted for RDNA3 (gfx1100) and RDNA4 (gfx1201).

**Gate before starting:**
- `ssh hipx 'ps -ef | grep hipfire'` returns empty → OK to use hipx for gfx11 testing
- `ssh hiptrx 'ps -ef | grep hipfire'` returns empty → OK to use hiptrx for gfx12 testing
- If either is busy with paro-g256-perfmax work, defer that arch port

**Files:**
- `kernels/src/rotate_x_mq_bfly.gfx1100.hip` (or unified file with arch detection)
- `kernels/src/rotate_x_mq_bfly.gfx1201.hip`

**Verification:**
- Byte-equal output across all 3 arches on same input
- Performance per arch within 5% of current FWHT rotate kernel on that arch

**Phase 10 commit:** `feat(kernel/gfx11+gfx12): butterfly residual ports`

---

## Phase 11 — Runtime loader + dispatch

**Goal:** Wire butterfly residual sidecar through hipfire-runtime.

**Files:**
- `crates/hipfire-runtime/src/hfq.rs`: detect `.butterfly_residual.weight` sidecar on MQ4G256 tensors, load as F16 `[8, 128]` per tensor, store in `WeightTensor.butterfly_residual: Option<DeviceBuffer<f16>>`
- `crates/hipfire-runtime/src/llama.rs`: add `rotate_x_mq_for_bfly(...)` helper dispatching to butterfly kernel when sidecar exists
- `crates/rdna-compute/src/dispatch.rs`: `rotate_x_mq_bfly(x, x_rot, awq_scale, butterfly_sincos, fwht_signs, n)` dispatch arm

**Critical invariant (from master plan):**
- A `WeightTensor` with `butterfly_residual.is_some()` MUST take the butterfly path
- Bare FWHT rotate path on a butterfly-tagged weight = silent wrong output
- Add coverage tests to verify dispatch invariant

**Phase 11 commit:** `feat(runtime): butterfly sidecar loader + dispatch invariant`

---

## Phase 12 — Production quant type (`MQ4G256Butterfly`, fail-closed)

**Goal:** Migrate from prototype sidecar to a new quant type for safety.

**Changes:**
- Assign new quant type ID (e.g., 25): `MQ4G256Butterfly`
- Storage: same body as MQ4G256 + REQUIRED inline butterfly sin/cos table (not optional sidecar)
- Loader rejects files without butterfly table → explicit error, not silent fallback to MQ4G256
- Quantizer `--format mq4-butterfly` emits new qtype 25

**Why phased (sidecar → quant type):**
- Sidecar lets us iterate fast during Phases 8-11
- Production needs fail-closed semantics — missing butterfly = wrong output, must error not silently degrade

**Phase 12 commit:** `feat(quantize+runtime): MQ4G256Butterfly quant type (qtype 25)`

---

## Phase 13 — Per-arch perf validation

**Goal:** Confirm butterfly meets ≤5% perf ceiling across all 3 arches × all 4 models.

**Bench matrix:**

| Model | gfx942 (mi300) | gfx11 (hipx, idle-gated) | gfx12 (hiptrx, idle-gated) |
|---|---|---|---|
| Qwen3.5-0.8B | bench butterfly vs current MQ4 | bench | bench |
| Qwen3.5-9B | bench | bench | bench |
| Qwen3.6-27B | bench | bench | bench |
| Qwen3.6-35B-A3B | bench | bench | bench |

For each cell: median of 3 fresh `bench_qwen35_mq4` runs, byte-identical prompt with md5 recorded.

**Pass criterion (per cell):**
- decode tok/s with butterfly ≥ 0.95 × decode tok/s with MQ4 (≤5% slower)
- If any cell fails 5% ceiling → **STOP default-flip**, ship as opt-in research format only

**Phase 13 deliverable:** Full perf matrix in `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/perf-matrix.md`

---

## Phase 14 — Coherence + speed gate validation

**Goal:** Standard hipfire pre-merge gates.

**Run:**
- `./scripts/coherence-gate.sh` — must PASS on all 4 models with butterfly enabled
- `./scripts/coherence-gate-dflash.sh` if DFlash drafts use butterfly weights
- `./scripts/speed-gate.sh` — must stay within 5% of baseline

**Phase 14 deliverable:** Gate reports committed.

---

## Phase 15 — Default-on + documentation + benchmarks

**Goal:** Flip default ON, write investigation/paper-quality doc.

**Changes:**
- `crates/hipfire-quantize/src/main.rs`: butterfly emitted by default for MQ4 when calibration data available
- Or `HIPFIRE_MQ4_BUTTERFLY=0` opt-out env var (default-on, opt-out)
- `docs/QUANTIZATION.md`: add MQ4G256Butterfly section
- `docs/PRIOR-ART.md`: add ButterflyQuant + ParoQuant references
- `docs/investigations/2026-05-22-butterfly-tier3-phase1-cpu/FINAL.md`: comprehensive writeup including:
  - Math + algorithm (residual butterfly, ButterflyQuant ancestry)
  - Empirical results (4 models × 3 arches: KLD/PPL + tok/s)
  - Comparison vs native ParoQuant (external baseline)
  - Cost analysis (storage + runtime)
  - Recommendations + caveats

**Phase 15 commit:** `feat(mq4-butterfly): flip default ON + ship docs/benchmarks`

---

## Phase 16 — HALT for master push approval

**Goal:** Final stop point. Branch has all 15 prior phases committed and pushed to `origin/feat/learnable-fwht`.

**Agent must NOT:**
- Push to master
- Open PR to master
- Force-push to any shared branch
- Modify any branch other than `feat/learnable-fwht`

**Agent must:**
- Commit final state on `feat/learnable-fwht`
- Push to `origin/feat/learnable-fwht`
- Write final summary to `docs/investigations/.../FINAL.md`
- Report results to user with:
  - All measured KLD/PPL/perf numbers per model × arch
  - Coherence/speed gate pass status
  - Cost summary (compute hours per phase)
  - Recommendation: merge to master? Or shelf? Or further work?
- **WAIT for explicit user approval** before any master-touching action

## Universal stop conditions

In addition to per-phase gates, the agent halts immediately on:
- **Coherence regression** in any phase that touches model output
- **NaN/inf** in training loss or activations
- **OOM** that can't be resolved by reducing batch
- **API error 500+** mid-phase if it loses state — must reload from last committed checkpoint and not silently retry into stale state
- **Cost ceiling**: estimate at start of each phase, halt if estimate > 2× initial budget
- **Wall-clock**: halt and report if phase > 24 hr unless explicitly time-budgeted

## Failure modes to watch for (from postmortem)

The postmortem identified specific patterns that killed prior session levers. Specifically:

1. **Proxy-doesn't-transfer**: BRECQ/per-Linear MSE going DOWN while production KLD goes UP. Check both at smoke gate (Phase 3) AND at full validation (Phase 4). If divergence appears, halt — don't escalate.

2. **GPTQ noise-floor surprise**: per cell-A finding, GPTQ adds ~0 on MQ4G256 + AWQ + mix-v1. Skip the Hessian build unless empirically needed. Saves 3 min/run.

3. **Channel-scale drift**: PARO K=0 stage-1 drove scales 3-4× from geomean=1.0. Butterfly residual is bounded by `theta` magnitude — no expected drift. But ADD a check: if `||theta_residual||` after optimization exceeds π (i.e., rotations exceed half-turn), warn and reconsider.

4. **Identity-residual check at every phase boundary**: when transitioning Python → Rust → HIP, verify `theta=0` reduces each implementation to current MQ4+AWQ behavior. Bisectability + production fallback.

## Cost budget estimate

- Phase 1-3 (Python smoke): ~6 hr wall-clock, ~$30 mi300
- Phase 4 (0.8B full): ~4 hr, ~$40 mi300
- Phase 5 (9B full): ~8 hr, ~$80 mi300
- Phase 6 (27B full): ~16 hr, ~$160 mi300
- Phase 7 (A3B asymptote): ~12 hr, ~$120 mi300
- Phase 8-12 (Rust + HIP): ~16 hr, mostly CPU (~$30 mi300 for GPU validation)
- Phase 13 (cross-arch perf): ~8 hr × 3 arches = ~24 hr wall-clock, ~$200 across mi300/hipx/hiptrx
- Phase 14-15 (gates + docs): ~6 hr, ~$30 mi300

**Total estimate:** ~80 hr wall-clock + ~$700 GPU compute.

If estimate doubles to ~$1400 mid-execution: halt and report.

## Final deliverable summary (for user review at Phase 16)

```markdown
## Butterfly residual MQ4 implementation — final report

| Model | Arch | KLD (butterfly) | KLD (MQ4+AWQ) | Δ | decode tok/s (butterfly) | decode tok/s (MQ4+AWQ) | Δ |
|---|---|---|---|---|---|---|---|
| 0.8B | gfx942 | X | 0.1327 | -Y% | X | X | -Z% |
| 0.8B | gfx11 | ... | ... | ... | ... | ... | ... |
| ...

Recommendation: MERGE / SHELF / FURTHER WORK
```

