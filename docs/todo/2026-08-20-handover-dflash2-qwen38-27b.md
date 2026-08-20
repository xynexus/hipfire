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

## The open bug — ROOT-CAUSED 2026-08-20

Not a DFlash bug, and not an Opus bug. The hand-written decode arms in
`crates/hipfire-arch-qwen35/src/qwen35/decode_layers.rs` miscompute on DENSE
qwen3.5-family models, and DFlash verify is the only caller left that forces
them: `forward_scratch_layers` routes to the lowered super-op executor unless
`hidden_rb` or `gdn_tape_capture` is `Some`, which verify always passes.

Reproduces with no drafter, no Opus, in ~10s on `qwen3.5-2b--bf16.hfq`:

    HIPFIRE_FORWARD_LOWERED=0 → '...\n0...  ,0...  $ $0...$0...$0...'
    default (lowered)         → coherent

MoE stays correct under the same flag (35B-A3B bf16, `LOWERED=0`, coherent, τ
4.33 with its drafter), so the fault is in the DENSE arms specifically — which
is the whole dense-vs-MoE asymmetry that made this look like a DFlash bug.

Full writeup, evidence table, and the slot-0 probe:
`docs/experiments/2026-08-20-dense-opus-dflash-miscompute.md`.

NEXT: prefer teaching the lowered executor to populate `hidden_rb` + the GDN
tape and deleting the hand arms, over repairing a second forward path that
nothing else exercises. Bisect on the 2B.

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

## DFlash2 semantics — what the tensors mean

Reference implementation: **https://github.com/z-lab/dflash**, file
`dflash/model.py` (659 lines; `GroupedDynamicCausalConv` ~L493,
`CandidateSelector` ~L515, `_grouped_dynamic_convolve` ~L478). There is also an
MLX port in `dflash/model_mlx.py`. No modeling code ships with the HF checkpoint
(config.json + model.safetensors only, no `auto_map`) — the repo is the spec.
The EXACT math is transcribed below, so no guessing is required.

Card: "block-diffusion drafter ... predicts a whole block in a single pass and
keeps the top candidates at every position. A lightweight selector then traces
one coherent path through them. Two-tap dynamic convolutions in the backbone
keep the draft from decaying toward the end of the block. Decoding is lossless."

Mapping to what `dflash_convert` already carries:

* `layers.N.{attention,mlp}_conv.base_kernel [2, 2, 5120]`
  `layers.N.{attention,mlp}_conv.kernel_projection.weight [1280, 5120]`
  -> `GroupedDynamicCausalConv`. NOTE the leading 2 in base_kernel is NOT the
  tap count: it is TWO SEPARATE CONVS, applied around the attention/MLP block
  via `prepare()` (before) and `finish()` (after). base_kernel is
  `[2, kernel_size, hidden]` = [which-conv, tap, channel].

  `kernel_projection: Linear(hidden -> 2 * kernel_size * groups)`, with
  `groups = hidden/group_size = 5120/16 = 320`, so out = 2*2*320 = 1280. The
  projection output is viewed `[..., 2, kernel_size, groups]`; slice `[...,0,:,:]`
  feeds `prepare`, slice `[...,1,:,:]` is carried across the block and feeds
  `finish`. So the dynamic coefficient is PER (position, tap, GROUP) — one
  scalar per 16-channel group, not per channel.

  Per-tap accumulation (`_grouped_dynamic_convolve`), hidden viewed as
  `[batch, len, groups, group_size]`:

      out = 0
      for tap in 0..kernel_size:
          v = hidden shifted causally by `tap` (zero-padded at the front)
          out += base[tap] * v                      # static, per-channel
          out += dynamic[:, :, tap] * v             # dynamic, broadcast per-group

  i.e. a causal depthwise conv whose per-group coefficient is data-dependent,
  ADDED to a static per-channel kernel. Both terms multiply the same shifted v.

* `candidate_selector.hidden_projection.weight [256, 5120]`
  `candidate_selector.predecessor_codebook     [248320, 256]`
  `candidate_selector.successor_codebook       [248320, 256]`
  with `selector_rank=256`, `selector_top_k=16`
  -> `CandidateSelector.select()`. Greedy left-to-right trace (NOT Viterbi — my
  earlier note said Viterbi; the reference takes an argmax per position with no
  backtracking):

      unary, candidates = topk(logits, top_k)        # per position, unsorted
      h = hidden_projection(hidden)                  # [b, len, rank]
      predecessor = anchor_ids                       # seed token
      for position in 0..len:
          scores = unary[:, position] + einsum("br,bkr->bk",
                       predecessor_codebook(predecessor) * h[:, position],
                       successor_codebook(candidates[:, position]))
          index       = argmax(scores)               # or sample at temperature>0
          predecessor = candidates[:, position][index]
          path.append(predecessor)

  So the transition term is a rank-256 triple product: the PREDECESSOR token's
  codebook row is gated ELEMENTWISE by the projected hidden, then dotted against
  each candidate's SUCCESSOR row. Sequential over positions (each step depends
  on the previous pick), top_k=16, block=8 — tiny, and a good fit for one
  wavefront per batch row.

Implementation order that does not depend on the open bug:
1. CPU reference port of conv + selector from the z-lab repo.
2. `parity_dflash2_conv.rs` / `parity_dflash2_selector.rs` against that
   reference (shape: `crates/hipfire-rdna/examples/parity_gemm_oq_compact.rs`).
3. Only then wire into the drafter forward and drop the load-time refusal in
   `DflashConfig::from_source`.

Note the block size differs: DFlash2 uses block_size=8 (7 draft tokens per
verify) where our DFlash1 drafters use 16. Smaller blocks also directly reduce
the verify cost that dominates the economics above.
