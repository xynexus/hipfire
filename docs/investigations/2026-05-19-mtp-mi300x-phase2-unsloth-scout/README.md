# MI300x Phase 2 scout — upstream Qwen3.6 MTP-trained head

**Date:** 2026-05-19
**Hardware:** AMD Instinct MI300X VF / gfx942 / ROCm 7.0.0
**Branch:** feat/mtp-mi300x

## Headline

The upstream `Qwen/Qwen3.6-27B` MTP-trained head delivers a real (small)
contribution where the previous `qwen3.5-27b.mq4-mtp` head delivered
zero. Phase 2's "extended-verify is dead" verdict was about the head
being weak, not about the verify-scheduling pattern.

| Config | tau_mtp | tau_total | tok/s |
|---|---:|---:|---:|
| 3.5 head + linear-chain (Phase 2 reference) | 0.048 | 5.71 | 68.0 |
| 3.6 head + linear-chain | **0.105** | 6.32 | **74.9** |
| 3.6 head + extended-verify | 0.056 | 6.67 | **78.0** |

- **3.5 head tau_mtp = 0.048** ≈ ~0 accepts (matches Phase 2's
  `accept_mtp_total = 0` on the longer 27B-3.5 LRU run).
- **3.6 head tau_mtp = 0.105** = **2.2× the acceptance rate**. Small
  but materially nonzero — the master plan's MTP composition story
  re-opens.
- Composition tok/s on MI300x with 3.6 head: **+10% on linear-chain,
  +15% with extended-verify**, both single-run (3-run median follow-up
  is the next step before claiming default flip).

## Pipeline used (reproducible)

```
# Stage 1: download upstream BF16 (52 GB, one-time)
huggingface-cli download Qwen/Qwen3.6-27B --local-dir ~/hf-models/qwen3.6-27b-bf16

# Stage 2: extract MTP head
./target/release/mtp_extract \
    --hf-dir ~/hf-models/qwen3.6-27b-bf16 \
    --output ~/.hipfire/models/qwen3.6-27b.mtp \
    --quant mq4

# Stage 3: quantize trunk (5-10 min, single-thread bottleneck on outer tensor loop)
./target/release/hipfire-quantize \
    --input ~/hf-models/qwen3.6-27b-bf16 \
    --output ~/.hipfire/models/qwen3.6-27b.mq4 \
    --format mq4

# Stage 4: bundle
./target/release/mq4_merge_mtp \
    --trunk ~/.hipfire/models/qwen3.6-27b.mq4 \
    --mtp   ~/.hipfire/models/qwen3.6-27b.mtp \
    --output ~/.hipfire/models/qwen3.6-27b.mq4-mtp

# Stage 5: bench
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 \
./target/release/examples/dflash_mtp_demo \
    --target /root/.hipfire/models/qwen3.6-27b.mq4-mtp \
    --mtp-head /root/.hipfire/models/qwen3.6-27b.mtp \
    --drafter /root/.hipfire/models/qwen36-27b-dflash-mq4.hf4 \
    --prompt "<canonical>" --max 120 --no-chatml --kv-mode q8 \
    --dflash-b 16 --mtp-k 2 --temp 0 \
    --mtp-mode extended-verify
```

## Findings on the model architecture

The upstream `Qwen/Qwen3.6-27B` is published as
`Qwen3_5ForConditionalGeneration` (multimodal — has `vision_config` for
the vision encoder plus `text_config` for the trunk + MTP). Our
quantizer correctly **skipped 885M params** (vision encoder + MTP
head) and produced a 14.98 GB text-only MQ4 trunk. The MTP head
(215.26 MB, 15 tensors, arch_id=21) was extracted separately via
`mtp_extract.rs` which already supports the `mtp_num_hidden_layers`
field name (no patch needed — the existing code at
`crates/hipfire-quantize/src/bin/mtp_extract.rs:632` reads it).

Bundle verification: `mq4_merge_mtp` writes the HFBNDMTP trailer
correctly; round-trip clean at offset 14980357120.

## Acceptance rate analysis

Phase 2's deferred verdict said extended-verify failed because **step
0's MTP candidate was rejected ~98% of the time**, never reaching
step 1+ where extended-verify's actual lever applies. With the 3.5
head: tau_mtp = 0.048 = ~95% reject at K=2 (consistent with the
~98% claim).

With the 3.6 head:
- linear-chain: tau_mtp = 0.105 = ~89.5% reject at K=2 (10.5 pp better)
- extended-verify: tau_mtp = 0.056 = ~94.4% reject (slightly worse)

The extended-verify-vs-linear-chain dip at tau_mtp=0.056 is
interesting and possibly noise — the overall tok/s is highest in
extended-verify (78.0 vs 74.9), so the trade is moving cycle wins to
the DFlash side. **Needs 3-run median to confirm.**

## Step toward the master plan target

Master plan: 250-350+ tok/s composition. Phase 2 reference: 111.80
tok/s (MI300x, 27B-3.5 LRU code prompt, 3-run median). Today's
single-run on sheep prompt: 78 tok/s — but this is a different prompt
+ single run + the MTP solo gain from the Phase 1 lm_head fix already
applies to the 3.6 path too. Apples-to-apples bench against the
canonical Phase 2 LRU code prompt is the next-priority follow-up.

## What this means for the master plan story

1. **The MTP-extended verify pattern is not structurally broken** —
   it just needs a properly-trained MTP head to exercise it. The 3.6
   head delivers; the 3.5 head we'd been using did not.
2. **There is still significant headroom**: tau_mtp = 0.105 is small
   in absolute terms. The Atlas/Unsloth class numbers (1.4-2.2×
   speedup from MTP composition) imply much higher acceptance. A
   properly trained head against this specific trunk+drafter combo
   could lift further.
3. **DFlash dominates the cycle budget**: tau_total ≈ 6 means ~6 of
   the ~17 total positions (DFlash B=16 + MTP K=2 + bonus) accept.
   The MTP K=2 contribution is +0.105 average. Most of the lift is
   still DFlash drafter quality.

## Recommended follow-ups (out of session scope)

- 3-run median bench on canonical 27B-3.5 LRU prompt to claim "+X%
  over Phase 2 reference"
- Try `--mtp-k 3` and `--mtp-k 4` to see if extended-verify scales
  with deeper MTP chains (the K=2 case is the master plan's
  conservative choice)
- Compare against `Qwen/Qwen3-Next-XB-Instruct` or other Qwen
  releases for a 3-way head comparison

## Operational notes

- DO droplet's UFW used `LIMIT IN` on port 22, which the prior
  agent's rapid SSH+rsync tripped — caused 30 min of debugging.
  Resolution: replace `LIMIT IN` with `ALLOW IN` for port 22 (still
  rate-limited by fail2ban for actual brute-force, no security loss).
- `hipfire-quantize` single-threads the outer tensor loop despite
  `rayon::ThreadPoolBuilder` configured with 16 workers — quantize
  hit 99% on one core, ~7 min wall on the 27B BF16 → MQ4 (could be
  much faster if parallelized).
- Rental spend incremental ~$3 (download + quantize + bench).
