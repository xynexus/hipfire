# Opus quant across all model families

Date: 2026-08-05
Status: proposed
Base: `11359ef0c`, with #223 (compact-resident Opus) in flight

## The headline finding: correctness is already universal

Before planning work, the empirical state, read from `tests/tiny-quant-baselines.txt`
(every row is a passing KLD cell on gfx1103, not an aspiration):

| families | Opus formats baselined |
|---|---|
| deepseek4, dots_ocr, gemma3, gemma3_vl, gemma4_dense, gemma4_moe, gemma4_ple, lfm2_moe, llama, mamba2, minimax, nemotron_h, qwen2, qwen3_5, qwen3_5_moe, qwen3_5_vl, qwen3_legacy, zaya | `oq4`, `oq4+`, `oq4++`, `oq4.25++`, `oq8`, `oq8+`, `oq8++` |
| deepseek4_compressed, deepseek4_mtp | **none** — explicitly blocked |

**18 of 20 families already quantize, load and score correctly on the full Opus
set, including the premier `oq4.25++`.** So "implement Opus across all model
families" is not a quantize-or-load project. That part is done.

What is missing is everything *around* correctness: two blocked variants, and —
much larger — Opus on the fast paths and in its compact form. A family can score
perfectly in the KLD gate while still running Opus on the slow path at 2x the
VRAM the format promises.

The rest of this plan is scoped to those gaps.

---

## Phase 1 — Unblock the two deepseek4 variants

The only families with no Opus support at all. Both blockers are declared in
`hipfire-eval/src/executor_tinyquant.rs` with specific causes, so neither is a
mystery:

- **deepseek4_compressed** — "OQ quantizes base tensors, but compressed
  attention tensors still require source-precision/F16 upload policy; keep
  explicit until compressor/indexer OQ dtype routing is implemented."
- **deepseek4_mtp** — "OQ omits packaged `mtp.0.*` tensors in generic OQ
  artifacts; keep explicit until native MTP tensor inclusion and OQ dtype policy
  are implemented."

Two independent pieces: dtype routing for the compressor/indexer tensors, and
MTP tensor inclusion in the OQ artifact. Each ends with its blocked cells
becoming real baselines rather than skips.

**Do this first** — it is the only phase where a family gets Opus *at all*, and
it is bounded by two written-down causes.

---

## Phase 2 — Compact residency beyond qwen35 dense

`oq4.25++` is ~4.25 bits/weight on disk and 8 bits resident everywhere except
the qwen35 dense path behind `HIPFIRE_OQ_COMPACT_RESIDENT=1` (#223). Measured on
the real `qwen3.5-0.8b.oq4.5++` artifact: 481.98 MiB expanded vs 266.94 MiB
compact, **-44.6%**.

The expansion is not in one place. Seven production sites unpack qt=36 today:

    hipfire-runtime/src/oq8_arch.rs        the shared arch-agnostic loader
    hipfire-runtime/src/hfq.rs
    hipfire-runtime/src/oq_moe.rs          MoE expert blocks
    hipfire-runtime/src/weight_pager.rs    paged-expert residency
    hipfire-arch-qwen35/src/qwen35/loading.rs
    hipfire-arch-minimax/src/minimax.rs
    hipfire-arch-lfm2moe/src/lfm2moe.rs    ("mirror of minimax")

Sequence, cheapest-first:

1. **Finish the qwen35 dense path** — `oq4.25++` itself (N_out=3) end to end;
   #223 validated `oq4.5++` (N_out=7) only. Decode parity at length. Then the
   flag can default on for that path.
2. **The shared loader** (`oq8_arch`) covers every family routed through it, so
   it is the highest-leverage single site — but only once each consumer of
   `DType::OqCompactG256` exists for those families' paths.
3. **MoE experts** (`oq_moe`, `weight_pager`) — experts are the bulk of a MoE
   model's weights, so this is where compact residency pays most. Also the
   hardest: the grouped/indexed expert kernels read their own layouts.
4. **minimax and lfm2moe** — retire their private expanders in favour of the
   shared path rather than porting the decode a third and fourth time.

**Hard-won lesson from #223, applies to every step:** making a dtype
compact-resident is not one wiring change, it is *every* path that touches those
weights. That work failed four times in a row on successive unwired consumers
(gemv.prerotated missing, key unregistered, residual arm, lm_head) — each caught
only because the change sat behind a default-off flag. Keep that discipline per
family: opt-in flag, wire until the smoke passes, then flip.

---

## Phase 3 — Opus on the fast paths

Correctness ≠ speed. Continuous batching exists for exactly one family
(qwen35; lfm2moe has serial prefill and no batched decode), and Opus reached
that fused path only in #220.

So this phase is gated on the batching seam growing, not on Opus itself:

- Any family without a `BatchExecutor` impl cannot have Opus-on-fused, because
  it has no fused path. See `docs/plans/2026-08-05-fused-decode-completion.md`
  phase 5a/5b.
- For families that gain one, Opus support is then the same small change #220
  was: admit `DType::Oq8G256` (and `OqCompactG256`) to the weights contract and
  add arms that rotate through `rotate_x_mq_batched_for` then call the batched
  W8A8 GEMM.

**Known open defect to fix here regardless:** `Oq4G256` (`oq4`/`oq4+`/`oq4++`)
is deliberately NOT admitted to the qwen35 fused path — it failed fused-vs-serial
parity at b8 (3 of 8 sessions produced a different first token). Reads numerical,
probably an activation-precision difference against the serial oq4 path, but it
is unexplained. That is tier-2 Opus silently confined to serial, and it should be
understood before this phase spreads the same arms to other families.

---

## Phase 4 — Make the gate prove it

Every phase above should end in the tiny gate rather than a claim.

- Phases 1 and 2 turn skips/absences into baselined cells; record them with
  `HIPFIRE_TINYQUANT_FAMILIES=<family>` scoping, never a bare `--record`, which
  rewrites every cell for the GPU.
- Add compact-residency coverage: the same family+format cell run with
  `HIPFIRE_OQ_COMPACT_RESIDENT=1` must produce the SAME hash as expanded. That
  is the assertion that stops a decode bug reaching a model, and it is cheap
  because the gate already builds those artifacts.
- The affected-gate selector needs `--base origin/master` (or `--files-from`)
  for committed work: with nothing staged it prints "no changed paths" and exits
  0, which reads like a pass and is not one.

---

## Sequencing and honest sizing

    Phase 1  deepseek4 compressed + MTP     bounded, two written-down causes
    Phase 2  compact residency               largest; 7 expander sites, per-family wiring
    Phase 3  Opus on fast paths              blocked on the batching seam, not on Opus
    Phase 4  gate coverage                   folded into each phase, not deferred

Phase 1 is independent and can start now. Phase 2 step 1 is nearly done (#223).
Phase 3 is mostly *someone else's* prerequisite — do not let it become the
reason Opus work stalls, since only one family can use it today.

## What this plan deliberately does not include

- Widening `Oq4G256` on fused paths before its parity failure is explained.
- Keeping compact residency behind a flag forever: each family that completes
  its wiring should flip its default, or the format's VRAM win never reaches
  anyone.
- Any claim that 44.6% is a measured VRAM saving. It is exact arithmetic over
  the artifact's tensor shapes; `rocm-smi` shows no delta on this UMA APU
  because hipMalloc comes from the GTT carveout. Confirm on a discrete card
  before quoting it as VRAM.
