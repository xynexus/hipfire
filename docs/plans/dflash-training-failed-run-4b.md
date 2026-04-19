# DFlash 4B Training Post-Mortem

**Date:** 2026-04-19
**Run:** `/root/dflash_4b_agentic/` on MI300X
**Budget:** ~$2 (1× MI300X for ~1 hour, validation scope)
**Outcome:** ❌ **Draft is broken — τ=0.09 vs z-lab baseline τ=4.22 on same target + prompt**
**Priority going forward:** Debug training methodology **BEFORE** any sidecar work.

## TL;DR

Our `scripts/dflash_train_poc.py` runs cleanly (loss 13.1 → 4.2 over 5000
steps), converges on the training loss, produces valid HF-layout safetensors,
converts cleanly to `.hfq`, loads in the engine, and emits plausible output
tokens at inference. All infrastructure works end-to-end.

**But the draft doesn't do speculative decoding.** τ stays at random levels
(0.02 – 0.13) across all 5 intermediate checkpoints. Our loss is optimizing
the wrong objective — or the right one with a fundamental bug in the training
setup.

## What we did

1. Pulled `lambda/hermes-agent-reasoning-traces` via
   `scripts/fetch_calibration_corpus.sh --recipe agentic`. 14.7k conversations,
   1.1 GB text, 365.9M tokens after tokenization.
2. Trained 5-layer DFlash draft against `Qwen/Qwen3.5-4B` for 5000 steps,
   batch=1, seq_len=4096, K=4 anchors per seq, γ=3 per-position loss weighting.
3. Output dir patched manually (see `save_pretrained` bug below) into
   HF-compatible layout: config.json + model.safetensors + training_meta.json.
4. Ran `dflash_convert --mq4` to produce `.hfq`. Converted all intermediate
   checkpoints too.
5. Tested each via `dflash_spec_demo` on a kimi hermes tool-call prompt with
   `--chatml` (proper tokenization).

## Results

| draft | τ | accepted/(cycles×15) |
|---|---|---|
| z-lab HF Qwen3.5-4B-DFlash (baseline) | **4.22** | ~56% |
| ours step 1000 | 0.017 | ~0.1% |
| ours step 2000 | 0.133 | ~0.9% |
| ours step 3000 | 0.072 | ~0.5% |
| ours step 4000 | 0.072 | ~0.5% |
| ours step 5000 | 0.093 | ~0.6% |

Training **did not produce a functional speculative decoder.** This is
confirmed across all checkpoints — not a late-stage LR divergence.

## What went wrong

### Confirmed bugs (during the run)

1. **`save_pretrained` crashed at end of training** — Qwen3Config validator
   rejects `num_hidden_layers=5` when `layer_types` has 32 entries. Our
   `build_draft_config` truncated `num_hidden_layers` but left `layer_types`
   at target length. Worked around by manually writing config.json, but the
   training script still has the bug.
2. **Composite config confusion** — `Qwen/Qwen3.5-4B` returns `Qwen3_5Config`
   (composite with `text_config` + `vision_config`). Our POC cloned the
   composite and set `num_hidden_layers` at the top level. The DRAFT actually
   initialized with head/dim values delegating to `text_config` (16 heads,
   head_dim=256 — correct for current Qwen3.5-4B). But this delegation is
   fragile and may be silently breaking other things.
3. **`dflash_convert` picks up ALL safetensors in input dir** — had to move
   intermediate checkpoints to a subdir before conversion or it would load
   the final checkpoint multiple times.

### Not-yet-diagnosed (what's actually making the draft fail)

**Hypothesis A: Multi-anchor mask is wrong**
The dense `[q_len, L+q_len]` boolean mask I built in
`scripts/dflash_train_poc.py` was derived from Figure 4 of the paper without
running any reference code. Structure:

```
Q axis:  K*B noise positions
K axis:  [target_hidden (L)] ++ [noise (K*B)]

Rules I encoded:
  noise at block k (anchor a_k):
    → target_hidden[j] IFF j < a_k     # causal, strictly before anchor
    → noise[j]        IFF same block    # bidirectional within block
```

Candidate issues:
- Off-by-one on `j < a_k` — should be `j ≤ a_k`? I chose strict based on
  reference inference doing `[:, :accept+1, :]` slicing just before next
  anchor. But maybe training expects different bound.
- Inversion of visibility sense (False where True) — looks right in the
  code but worth re-verifying.
- The bidirectional within-block pattern — is this the right semantics or
  should noise attend causally within a block too?

**Hypothesis B: Training-inference k_cat distribution mismatch**
At inference (ref `spec_generate`):
- Iter 1: K = [k_ctx (full prompt), k_noise (block 1)]
- Iter 2+: `past_kv.update(fresh_k, fresh_v)` appends fresh to cache, then
  `past_kv.crop(start)` after forward. The draft's attention sees:
  `[past_kv (accumulated noise+ctx k/v from prev iters, cropped to accept
  positions)] + [fresh k_ctx (small, just accept+1 rows)] + [fresh k_noise
  (current block)]`.

At our training: always `use_cache=False`. k_cat is always
`[k_ctx (full prompt), k_noise (current block)]`. The draft never sees the
mixed past_kv + fresh pattern. Iter 1 at inference matches training, but
iter 2+ is wildly different. Loss converges on iter-1-like distributions,
but inference τ depends on iter 2+ behavior.

This is plausible as a fundamental source of the gap. Fixing requires
simulating the KV cache accumulation during training OR a paper-level
understanding of how z-lab trains with respect to past_kv.

**Hypothesis C: γ=3 loss weighting makes mid-block positions untrained**
Weights at position k in a block of 15 are `exp(-(k-1)/γ)` normalized:
- pos 1: 0.285
- pos 5: 0.054
- pos 10: 0.010
- pos 14: 0.003

Position 10+ contributes ~1% of gradient. But at inference, τ=5 requires
positions 10-14 to also predict correctly (otherwise we accept 5 and stop).
Draft may be optimized only for positions 1-3 and random at 8+.

Test: train with γ=10 (flatter weighting) and see if τ improves.

**Hypothesis D: Composite config silently breaks the DFlashDraftModel**
Reference `DFlashDraftModel.config_class = Qwen3Config` expects a flat
`Qwen3Config`, but we pass `Qwen3_5Config` (composite). Attribute lookups
like `config.hidden_size` may or may not delegate correctly. If any
attribute returns an unexpected value, the model instantiates with the
wrong shape.

Check: load our saved `model.safetensors` back into a fresh
`DFlashDraftModel(Qwen3Config)` with EXPLICIT values (not via composite
clone), compare state dicts, see if any tensors got silently misaligned.

## Confirmed NOT the cause

- **Architecture dim mismatch with target**: z-lab's draft is 32/8/128 but
  target is 16/4/256; still runs at τ=4.2. Our draft is 16/4/256 matched
  to target; runs at τ=0.09. Matching dims isn't the differentiator.
- **Training data**: Lambda hermes is valid tool-call data; z-lab wikitext
  draft gets τ=4.22 on the SAME agentic prompt. So data isn't the issue.
- **.hfq conversion**: same `dflash_convert --mq4` binary used on z-lab's
  HF safetensors and on ours; both produce valid hfq files.
- **Engine inference**: z-lab's draft works perfectly with the engine, so
  the engine-side code is correct.

## Intermediate artifacts

Kept on MI300X at `/root/dflash_4b_agentic/`:
- `model.safetensors` (final state, step 5000)
- `checkpoints_intermediate/draft_step{1000,2000,3000,4000,5000}.safetensors`
- `config.json` (manually reconstructed — see below for what to fix in the
  script's save_pretrained path)
- `training_meta.json`
- Training log: `/root/dflash_4b_agentic_train.log`

## Fixes to put into the training script regardless of root cause

1. **Truncate `layer_types` in `build_draft_config`**:
   ```python
   if hasattr(cfg, 'layer_types') and cfg.layer_types:
       cfg.layer_types = cfg.layer_types[:draft_layers]
   ```

2. **Handle composite configs**: detect `Qwen3_5Config` and work off
   `text_config` explicitly. Or construct a fresh flat `Qwen3Config` from
   text_config values, then override num_hidden_layers etc. Safer than
   cloning a composite.

3. **Intermediate checkpoints go in a subdir**, not the root output dir,
   so `dflash_convert` doesn't try to load all of them as the final draft.

4. **Write a standalone `dflash_postprocess.py`** that takes any safetensors
   + a target_repo and builds the HF layout. Separates save-pretrained
   logic from training loop so a training crash at save time doesn't lose
   the trained weights.

## What to do next

**Priority 1 (before ANY other DFlash work):** debug the training bug.

### Update 2026-04-19 (Hermes review round)

A Hermes-agent second-opinion round produced concrete diagnostic scripts now
committed to `scripts/dflash_diag_*.py` and a thorough reassessment of the
hypotheses:

**Hypothesis A (mask) — REFUTED by `scripts/dflash_diag_mask.py`.** The mask
is paper-faithful. Reference inference (`.dflash-reference/dflash/model.py:347`)
uses `attention_mask=None` and `is_causal=False` → fully bidirectional within
block. Training's bidirectional-within-block is the correct match. (Hermes
initially suggested causal-within-block as an A/B, but reading the reference
shows this would BREAK alignment.)

**Hypothesis D (composite config) — REFUTED AND IRRELEVANT.** `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")` returns a model with `config` of type `Qwen3_5TextConfig` (flat), NOT composite `Qwen3_5Config`. The
composite variant (which has no top-level hidden_size/num_hidden_layers attrs)
only appears from `AutoConfig.from_pretrained`. So the old `build_draft_config`
did clone a flat config — the post-mortem's "fragile delegation" framing was
wrong.

That said, `build_draft_config` has been rewritten to build a FRESH flat
`Qwen3Config` from text_config-scoped attribute reads. This eliminates the
Qwen3_5TextConfig vs Qwen3Config subclass question, and fixes the original
save_pretrained crash (truncates `layer_types` to draft_layers, asserts the
invariant). Side effects:
  - All draft layers now forced to `"full_attention"` to match z-lab's config.
  - Rope params routed through new-schema `rope_parameters` dict (transformers 5.x).
  - `save_pretrained` verified end-to-end on Qwen3.5-0.8B test.

**Hypothesis B (KV distribution) — mathematically implausible.** Careful trace
through reference spec_generate shows that at cycle 2+, past_kv holds ONLY
cropped target_hidden K/V (noise is dropped each cycle via `pkv_d.crop(start)`).
Effective k_cat at cycle N = [all accepted target ctx] + [current block noise].
Training's K-concat mask with `j < a_k` + within-block bidirectional
EXACTLY replicates this per-block. Distributions match. Not the bug.

**NEW FINDING — rope_parameters partial_rotary_factor=0.25 on Qwen3.5 target.**
Qwen3.5 uses partial rotary (only 25% of head_dim rotated). Our draft inherits
this; z-lab's draft config sets no partial_rotary_factor (full rotary). For
head_dim=256, our draft leaves 192 of 256 dims WITHOUT positional encoding
through RoPE. z-lab's draft (head_dim=128, full rotary) rotates all 128.
Whether this tanks τ is unproven but it's a significant architectural difference.

**NEW DIAGNOSTIC — `scripts/dflash_diag_zlab_loss.py`.** Loads z-lab's
baseline weights, runs through OUR forward (our mask, our loss, our data).
If their known-good weights produce low loss on our task → our
forward/mask/loss are CORRECT, pointing at training dynamics / init /
partial-rotary as the bug. If high loss → our forward has a bug.

**NEW DIAGNOSTIC — `scripts/dflash_diag_iter1_tau.py`.** Caps decode cycles
at 1. Measures τ for just iter 1 vs aggregate. If τ_iter1 much larger than
τ_aggregate, Hypothesis B reactivates.

**NEW TRAINING-TIME τ PROBE — `--tau-probe-every N` flag.** Hermes's
highest-value suggestion. Every N steps, small spec_generate run on a fixed
held-out prompt. Prints τ alongside loss. Would have caught the 4B run's
τ=0.09 in <100 steps instead of 5000. Next training run MUST enable this.

### Revised order to try (replaces original list)

1. **Run the diagnostic scripts on MI300X:**
   ```bash
   # Hypothesis A (already run LOCALLY on CPU — confirmed PASS)
   python3 scripts/dflash_diag_mask.py

   # Hermes cheap #1: z-lab weights through our forward
   python3 scripts/dflash_diag_zlab_loss.py \
       --target-repo Qwen/Qwen3.5-4B \
       --zlab-draft-repo z-lab/Qwen3.5-4B-DFlash \
       --corpus /root/calibration_corpus.txt --num-batches 5

   # Hermes cheap #2: iter-1 τ of our failed draft
   python3 scripts/dflash_diag_iter1_tau.py \
       --target-repo Qwen/Qwen3.5-4B \
       --draft-dir /root/dflash_4b_agentic --max-cycles 1
   python3 scripts/dflash_diag_iter1_tau.py \
       --target-repo Qwen/Qwen3.5-4B \
       --draft-dir /root/dflash_4b_agentic --max-cycles 50  # aggregate
   ```

2. **Retrain 4B from scratch with the patched script + τ probe enabled:**
   ```bash
   python3 scripts/dflash_train_poc.py \
       --target-repo Qwen/Qwen3.5-4B \
       --corpus /root/calibration_corpus.txt \
       --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
       --steps 5000 --ckpt-every 1000 \
       --tau-probe-every 200 \
       --out /root/dflash_4b_agentic_v2
   ```
   If τ probe is still flat at step 500, KILL the run — don't waste compute.

3. **If v2 still fails: force full rotary in the draft** (add a
   `--full-rotary` flag that overrides rope_parameters to drop
   `partial_rotary_factor`). Matches z-lab's architectural convention.

4. **If still failing after 1-3: wait for z-lab to release training code.**
   Their README promises "soon". Their code shortcuts everything.

**Priority 2:** once we have a working draft-training pipeline, THEN
return to sidecar work (which is currently in decent shape — generic
mirror-evict fix shipped, agentic calibration validated end-to-end).

**What NOT to do**: run any more expensive training runs (A3B, 9B, 27B)
until a 4B training produces τ > 3 on agentic. A bad 4B run cost $2; a
bad A3B run at ~20× the compute would be $40 with no more information.

## Open question: is this actually publishable?

If we reverse-engineer successfully, the research claim remains valid:
"domain-trained drafts improve agentic τ over wikitext-trained drafts on
agentic tasks." The methodology is novel and useful regardless of whether
z-lab eventually publishes their recipe — our contribution is the DOMAIN
specialization, not the training algorithm itself.

If z-lab publishes training code and our reverse-engineered approach
matches theirs, still publishable as domain-specialization study. If
theirs differs from ours and ours turns out to be a subtly wrong
implementation, we cite their published code and rerun.

Patience on this is free.
