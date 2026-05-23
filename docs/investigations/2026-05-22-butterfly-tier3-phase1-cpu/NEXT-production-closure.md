# Production closure: from "beats PARO-mechanism (Python)" to "beats shipped PARO (kldref)"

The research phase is complete and committed: **MQ4-SLfwht+Lloyd**
(continuous full-butterfly rotation + Lloyd-Max codebook) beats the PARO
*mechanism* (rotation + uniform) on all 4 trunk models in Python in-memory
KLD (0.8B −19%, 9B −8.9%, A3B −5.0% direct-trained; 27B −8.2% baseline). The
Lloyd codebook is the universal edge PARO structurally lacks.

Three things stand between this and the literal goal ("on par with or better
than shipped PARO on the production metric for all 4"). Each is engineering,
not lever-tuning. Ordered by dependency.

## Gap 1 — Rust port of rotation + Lloyd into hipfire-quantize (the big one)

Today the recipe lives only in the Python pseudo-quant trainer
(`scripts/learn_butterfly_mq.py`). To produce a real `.hfq` and a production
kldref number, hipfire-quantize must apply the learned per-group rotation +
Lloyd codebook at quantize time, and the runtime must dequant it.

- **Quantizer** (`crates/hipfire-quantize/src/`): read the per-tensor learned
  butterfly angles + Lloyd codebooks (new sidecar format, analogous to the
  HFBF format the Python trainer already emits), apply
  `B(theta) ∘ FWHT(W·s)` then fit/store the 16-entry codebook per group.
  hipfire already ships MQ3-Lloyd/MQ2-Lloyd, so the codebook-storage + GEMV
  kernel pattern exists — extend to MQ4 + add the rotation stage.
- **Runtime** (`crates/rdna-compute` GEMV): the rotate-x kernel must apply the
  per-group butterfly (8 layers) in addition to FWHT, and the GEMV must do
  codebook-lookup dequant. New quant type (e.g. `MQ4G256LSigLloyd`).
- **Cost / scope**: this is the original IMPLEMENTATION_PLAN.md Phase 8-12,
  ~1-2 weeks. **Runtime is NOT free** — butterfly rotation + codebook lookup is
  comparable to or heavier than PARO; benchmark against the ≤5% MQ4 decode
  ceiling and decide if it ships as default-on or opt-in quality mode.

## Gap 2 — Real shipped-PARO numbers (the comparison bar)

The head-to-head used a faithful PARO *mechanism* proxy (continuous rotation +
uniform), not z-lab's actual quantized weights. To compare against real PARO:

- **Option A**: download `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed`
  (~70 GB; currently only `refs/main` on mi300, no blobs) + build/borrow a
  loader for its packed format, eval with the A3B kldref. Gives ONE real PARO
  anchor (A3B).
- **Option B**: run ParoQuant's own pipeline (github.com/z-lab/paroquant,
  CUDA-only) on 0.8B/9B/27B to produce PARO-quantized models, eval each with
  eval_hipfire + the matching kldref. Gives all 4 real PARO anchors but needs
  the PARO toolchain (CUDA) — may not run on the gfx942 mi300.
- **Risk**: PARO's reported wins are PPL-on-reasoning; KLD-vs-BF16 on our
  calibration slice may rank differently. Match the eval axis carefully.

## Gap 3 — Production kldref eval of MQ4-SLfwht+Lloyd

Once Gap 1 lands: quantize all 4 trunk models with the new quant type, eval
each with `examples/eval_hipfire` against the cached kldrefs
(`/workspace/kldref/*.bin`). Compare to (a) plain MQ4 baseline (validates the
Python→production scale mapping) and (b) the real PARO numbers from Gap 2.

### Cheap pre-step (validates the Python proxy without the full port)
Quantize 9B to plain MQ4 via the EXISTING hipfire-quantize, eval with
eval_hipfire + the 9B kldref, and compare to the Python baseline (0.353). If
they map consistently, the Python head-to-head deltas are a validated proxy
for production — strengthening the research claim while Gap 1 is built. (~1-2 hr,
no new code.)

## Also unblock 27B direct training (orthogonal, smaller)

27B full-butterfly KLD-loss training OOMs in 192 GB even with student gradient
checkpointing. Fix: cache the oracle log-probs once (the calibration seqs are
fixed) and free the oracle model (frees ~54 GB) so only the student + its
backward must fit. ~1 codex cycle. Gets the 27B direct head-to-head number that
is currently baseline-only.

## Recommendation

The research result justifies the Gap-1 port investment. Suggested order:
1. Cheap pre-step (Python→production scale validation on 9B baseline) — 1-2 hr.
2. Oracle-caching → 27B direct head-to-head — 1 codex cycle + 1 run.
3. Gap 1 (Rust port) — the 1-2 week commitment; gate on the user wanting to
   productionize, since it's a kernel/format change with a runtime cost.
4. Gap 2 (real PARO) in parallel with Gap 1.

This is a phase boundary: the lever question is answered (the recipe works and
beats PARO's mechanism); production parity is a build, and should be a fresh
focused effort, not tacked onto the research session.
