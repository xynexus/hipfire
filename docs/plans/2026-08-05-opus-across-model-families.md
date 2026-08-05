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

### Phase 1 status (2026-08-05)

**deepseek4_compressed — DONE.** `SourcePrecision` is a property of the tensor,
declared by the arch's -spec, but the quantizer gated it on
`use_deepseek4_source_precision`, so generic OQ builds quantized the MLA
compressor/indexer streams anyway. Honouring the class for deepseek4 whatever the
output format unblocks all 7 Opus cells; values track plain deepseek4 closely with
the expected ordering intact.

**Scoping finding, for whoever widens this.** `precision_class_via_ingest`
consults EVERY arch's spec, so honouring it unconditionally reaches far past this
blocker. Measured on gfx1103:

| arch | effect of honouring SourcePrecision for all formats |
|---|---|
| gemma4_ple | **all 9 cells improved** — hfq4 -69%, oq8+/oq8++ -39%, rest -1.5..-9% |
| minimax | **all 7 Opus cells regressed** — oq8 37x worse, oq4 2.7x |

The gemma4 win is real and applies to magnum formats too, so this bug was never
Opus-specific. But a higher-precision artifact scoring *worse* on minimax points
at something there mishandling unquantized tensors rather than precision hurting.
Get that cause before widening.

**deepseek4_mtp — still blocked, and not mechanical.** Three things stack:
`main.rs` skips `mtp.*` for non-deepseek4 formats; `arch.rs:544` has
`mtp_layer: None ("Phase 5 work")` so the loader never wires MTP from the base
artifact; and MTP is designed to ship as a separate `.mtp-addon.hfq` found by
filename convention. Whether an OQ artifact should quantize MTP or keep it at
source is a judgement about acceptance rate — `deepseek4-mtp-precise` exists
because "the V3 paper's 60-80% acceptance benchmark assumes weights at training
precision". Needs a decision plus loader Phase 5, not a quantizer tweak.

### Inherited breakage on master (not from this work)

Both verified against pristine `origin/master`, so neither should be attributed
to Opus work that follows:

- **minimax's 7 Opus cells fail** with identical numbers on master. Any full
  `tiny-quant-gate.sh` run is red until fixed; scope with
  `HIPFIRE_TINYQUANT_FAMILIES` to get a clean signal meanwhile.
- **`tests/no-gpu-ci.sh` is red** — 54 ruff F821 errors in
  `tools/qwen35_full_forward_oracle.py` (from `2f64effaa`), which has a shebang
  and a docstring but no imports at all. Left alone: it looks like in-flight
  work, and the fix is a guess at its author's intent.

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

1. **Finish the qwen35 dense path.** *(N_out=3 shipped; flag still gated — see
   the open logit divergence below.)* #223 validated only `oq4.5++` (N_out=7).
   `Qwen3.5-0.8B--oq4.25` (N_out=3, `block_stride` 136 — the block shape the
   premier `oq4.25++` family uses) now passes `smoke-generate-batch-prefill.sh`
   both expanded and compact-resident, 6 fused cells each. Allocation over its
   186 Opus tensors: 481.98 MiB expanded vs 252.11 MiB compact, **-47.7%**
   (better than N_out=7's -44.6%, as expected — fewer overlay bytes per block).
   Arithmetic over artifact shapes, not a measured VRAM drop.

   **Decode parity at length: tokens match, logits do not.** Via
   `tiny_quant_probe ar-hash --arch qwen3_5 --len 128 --prompt-len 4 --seed 42`:

   |            | expanded             | compact              |
   |------------|----------------------|----------------------|
   | token_hash | `0xe08dc73ee518ae66` | `0xe08dc73ee518ae66` |
   | logit_hash | `0x9fec73050da8e72e` | `0x83d3e5a9628116f8` |

   All 128 free-running greedy tokens are identical, so the difference is small
   — but it is real, not noise: both modes are bit-stable across repeated runs.
   What is known so far:

   - **Not the GEMM/GEMV.** `parity_gemm_oq_compact` is bit-identical 24/24. It
     originally used only power-of-two weight *and* activation scales, which are
     exact under any multiply order and so could not have caught a rounding or
     ordering difference; it now also sweeps arbitrary f16 scales and still
     passes. Both dispatch arms are structurally identical anyway — GEMV is the
     batched GEMM at n=1 in both, with the same `quantize_act_oq8` — and the two
     kernels' epilogues are character-for-character the same expression.
   - **Not the expander's scale.** `oqplus_compact_to_oq8_combined` writes
     `f16_to_f32(block f16)` into the scale plane, exactly what the compact
     kernel reads from the block. f16→f32 is exact.
   - **Present in the very first forward.** At `--prompt-len 1 --len 1` — one
     token, one B=1 GEMV, no KV accumulation, no batching — the logits already
     differ. So it is not drift over steps.

   That leaves dispatch-level routing: some op taking a different (still valid)
   kernel or fallback under `OqCompactG256` than under `Oq8G256`. There is a
   concrete, verified admission gap of exactly that shape — `OqCompactG256` was
   wired into the generic GEMV/GEMM dispatch and into
   `dense_prefill_weight_unsupported_reason`, but NOT into qwen35's fused/
   specialised arms, which still match `Oq8G256` alone:

   - `is_batchable_la`'s `oq8_with_wmma` gate (`qwen35/mod.rs`) — the same
     admission gate whose comment records that missing an `Oq8G256` arm once cost
     every oq8 model 12x on prefill (86 vs 1046 tok/s).
   - `KernelKey::FusedGateUpOq8G256` (`prefill_chunk.rs`), and the `is_oq8` /
     `wo_is_oq8` fused-QKV and fused-wo predicates in the same file.
   - the routed-expert `gemv_oq8g256_moe_*_indexed_batched` arms.

   **This gap is real and worth closing on its own merits, but it is NOT yet
   proven to be the cause of the logit difference.** The control says otherwise
   so far: forcing the expanded path off batched prefill
   (`HIPFIRE_OQ8_BATCHED_PREFILL=0`) changed its `ar-hash` wall time not at all
   (20.15s vs 20.17s at `--len 288 --prompt-len 256`), i.e. the batched path is
   not engaged in this probe, so the gap cannot explain a divergence the probe
   shows. Confirming or killing it needs per-op instrumentation, not yet written.

   Cost of compact residency on that same probe: **26.48s vs 20.17s, ~31%
   slower**, which is the expected price of decoding nibbles and scanning the
   overlay per tile rather than reading a dense int8 plane. That is the tradeoff
   against -47.7% allocation, and it is a reason to keep the flag opt-in for
   latency-sensitive paths even once the divergence is explained.

   (Method note: `tiny_quant_probe ar-hash` requires `--prompt-len <= --len`; an
   invalid pair exits non-zero with empty hashes, which reads as a match if the
   exit code is not checked.) **The flag must not
   default on until this is explained** — token-identity at 128 steps is
   reassuring, not proof, and an unexplained numeric difference in the premier
   quant's residency path is exactly the kind of thing that surfaces later as a
   quality regression nobody can bisect.
   Two controls make the bisection trustworthy rather than suggestive:
   `ONLY_K=999`, matching no tensor, reproduces the expanded logit_hash
   EXACTLY — so compact residency applied to nothing is byte-identical, and the
   filter is doing what it claims; and `ONLY_K=1024,2048,3584` equals plain
   compact-all, confirming those three values cover every compact tensor with
   none unaccounted for.

   Static analysis is now exhausted, with these paths eliminated: prefill_batch
   is symmetric (oq8 and compact both go `*_act_batched` -> quantize ->
   their GEMM); the decode files carry no dtype-specific handling at all, so
   decode is pure generic dispatch; and `gemm_oq8_grouped_prequant` bottoms out
   in the same `gemm_oq8_grouped_wmma` the oracle compares against, so the
   oracle's reference is the right one. The next step is per-op instrumentation
   — dump each op's output under both modes and diff — not more reading.

   **Separately found, and more serious than the divergence:
   `prefill_chunk.rs` has NO compact support at all** — zero `OqCompactG256`
   against 22 `Oq8G256`. Traced through the `wo` chain, a compact tensor there:

   1. makes `wo_is_mq` false, because `OqCompactG256` is missing from the FWHT
      rotation-admission list, so it is handed the **unrotated** activation
      batch — the omission whose comment in that same file records the result
      for oq8 as garbage, PPL 3.5e6; and
   2. matches no `wo_is_*` arm, so it falls to the terminal `else`, which runs
      `KernelKey::GemmHfq4G256Residual` over the compact bytes.

   So it is decoded as HFQ4 with an unrotated input: **silently wrong, not an
   error.** Qwen3.5-0.8B dense does not appear to route through this file, which
   is why the smoke and `ar-hash` still produce sane output — but any model that
   does would be quietly corrupt under the opt-in flag. Nothing is broken for
   users today (the flag defaults off); this is a trap armed for whoever turns
   it on next.

   Two ways to close it, and they are not alternatives — do the guard first:

   - **Guard (one edit, do now).** `run_residual_gemm_key` / `run_gemm_key`
     already take `w_dtype`, and every fallthrough in the file goes through
     them. Rejecting `OqCompactG256` there when the key is not a compact key
     converts silent corruption into a loud error everywhere at once.
   - **Wire it (the real fix).** ~8 LA/FA sites, each needing
     `OqCompactG256` added to the rotation list plus a compact arm calling
     `gemm_oq_compact_act_batched` / `gemm_oq_compact_residual_act_batched`.
     The routed-expert `gemv_oq8g256_moe_*_indexed_batched` arms additionally
     have no compact kernel at all, so those need kernel work, not wiring.

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
