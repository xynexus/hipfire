# Handover: DFlash2 + Qwen3.8-27B oq4.25++ / CASK

Goal: implement DFlash2 into the dspark/ddtree/dflash/ngram machinery and
test/optimize it for Qwen3.8-27B oq4.25+CASK. **Not met.** This records what
is done, what is measured, and the exact state of the open bug.

## Assets on disk

| thing | path | state |
|---|---|---|
| target | `~/.hipfire/models/Qwen3.8-27B--oq4.25++.hfq` | 15.48 GB, serves 7.5 tok/s |
| calib | `~/.hipfire/calib/Qwen3.8-27B.calib.hfq` | 31.6 GB, kld finalized |
| CASK | `~/.hipfire/triattn/Qwen3.8-27B.triattn.hfq` | 598 KB, active |
| DFlash1 drafter (jfan, heretic) | `~/.hipfire/drafts/Qwen3.8-27B--dflash.oq4+.hfq.parked-*` | converts, parked |
| DFlash1 drafter (z-lab, matched) | `~/.hipfire/drafts/Qwen3.6-27B--dflash.oq4+.hfq.parked-*` | converts, parked |
| DFlash2 checkpoint | `/srv/huggingface/models--z-lab--Qwen3.8-27B-DFlash2` | fetched, verified |
| DFlash1 drafter (35B MoE, WORKS) | `~/.hipfire/drafts/Qwen3.5-35B-A3B--dflash.oq4+.hfq` | active |

Vision tower was SKIPPED (`Skipped params: 885429488 (mtp/visual)`); the
artifact is text-only. `--include-vision` re-runs only pass 2 (~2h) — the
calib artifact is reusable.

## Done

* Opus batched lm_head verify arms (`ccf96d961`) — Oq8G256 / OqCompactG256 /
  OqCompactG128 in `dflash_enqueue_verify_lm_head`, on VerifyScratch buffers so
  it stays graph-capturable. **Proven correct in production on the 35B MoE.**
* DFlash2 conversion (`b66832822`) — the z-lab checkpoint converts today: 81
  tensors, 1.924B params, 932 MiB, all 3 `candidate_selector` + 20 conv tensors
  carried. Artifact self-describes (`dflash_version`, `selector_rank`,
  `selector_top_k`, `conv_kernel_size`, `conv_group_size`) and is REFUSED at
  load rather than silently loaded as DFlash1 (which would drop 23 tensors and
  only show up as a mysteriously bad acceptance rate).
* Stale-tape fix (`2ba31acd3`) — `verify_populates_tape` now mirrors the
  forward's KV-tier condition. Real latent bug; did NOT fix the one below.

## NOT done

* DFlash2 conv + candidate-selector RUNTIME kernels. These are validatable in
  isolation with a parity example (see `parity_gemm_oq_compact.rs` for the
  shape) even while the bug below blocks end-to-end use — that is the
  recommended way in, since it does not depend on the verify path.
* Any Qwen3.8-27B DFlash performance number.

## The open bug

Dense + Opus + DFlash miscomputes: `'嘟 plain'` at 0.34-0.40 tok/s vs 7.5 plain.
Reproduce with `HIPFIRE_DFLASH_ALLOW_OPUS=1` and a drafter un-parked.

Reproduced on TWO dense models (Qwen3.8-27B, Qwen3.6-27B) with TWO drafters
(heretic jfan, matched z-lab). Qwen3.5-35B-A3B (MoE, same qt=36 lm_head, same
Opus DeltaNet projections with AWQ sidecars, same CASK) is CORRECT.

ELIMINATED, each by measurement — do not re-test:
1. Opus lm_head arms — `HIPFIRE_DFLASH_NO_BATCHED_LMHEAD=1` gives identical garbage.
2. CASK/TriAttention — identical with the sidecar parked.
3. Verify graph capture — identical with `HIPFIRE_VERIFY_GRAPH=0`.
4. Opus batched GEMM kernels — `parity_oq8_gemm` 45.15 dB, `parity_gemm_oq_compact` bit-identical.
5. Drafter lineage — matched z-lab drafter garbles too, on a second dense model.
6. Stale GDN tape replay — fixed in `2ba31acd3`, output unchanged.
7. `prefill_batch.rs`'s batched DeltaNet/DeltaNetMoe arms — instrumented, NEITHER
   FIRES for either model, draft or not. That file is the wrong place to look.

START HERE: `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1` shows BOTH models take the
per-token fallback (`final=false`), dense via `base=true kv_f32=true`, MoE via
`base=false`. Same fallback path, opposite outcomes. That asymmetry is the
unexplained core. An rocprofv3 draft-vs-plain kernel diff shows the draft run
adds only DRAFTER kernels plus the lm_head arm — the target body reuses the
plain-decode kernels — so the divergence is in how verify DRIVES that shared
body (state or position handling across the block).

## Measured economics — read before optimizing anything

On the one working Opus config (35B MoE), DFlash is a **4.4x REGRESSION**:
8.17 tok/s vs 35.7 baseline. `HIPFIRE_SPEC_PHASES=1` per B=16 cycle:

    verify=540ms  draft=47ms  ngram=0.9ms  replay=0-402ms

Baseline is 28ms/token, so verifying 17 positions costs ~17 serial decodes: on
an A3B each position picks its own top-8 of 128 experts, so batched verify
reads ~17x the expert bytes one decode does and amortizes NOTHING.

Implication: a better drafter (DFlash2 included) cannot fix a verify that costs
as much as decoding outright. Speculation should pay on DENSE targets, where
block weights are shared — which is exactly the broken path. Fix the dense bug
FIRST, confirm verify amortizes there, and only then judge DFlash2 on merit.

`replay` also scales WITH acceptance at ~28ms/accepted token (accept=0 -> 27ms,
accept=14 -> 402ms, accept=15 -> 0ms), i.e. accepted tokens are paid for twice.
Independent of the MoE economics and worth fixing on its own.
