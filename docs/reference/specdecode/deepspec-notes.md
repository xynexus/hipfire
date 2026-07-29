# DeepSpec / DSpark — distilled notes for hipfire DFlash NPU drafting

Reference read directly from `/srv/hipfire/references/SpecDecode/DeepSpec/`
(paper `tex/`, code `src/`). Paper: **DSpark: Confidence-Scheduled Speculative
Decoding with Semi-Autoregressive Generation**, DeepSeek-AI, arXiv:2607.05147
(`tex/main.tex:105`, `tex/main.tex:90-98`).

TL;DR for our purposes: **DeepSpec is a training/eval repo, not a serving
system. DSpark is a drafter algorithm = DFlash parallel backbone (KV-injection of
target hidden states) + a lightweight local-autoregressive head + a confidence
head + a hardware-aware verification-length scheduler.** It is *not* MTP-based;
in production it *replaces* DeepSeek's MTP-1 drafter. The pieces most relevant to
hipfire's Goal A (disaggregation) live in the training-infra section and in the
DFlash KV-injection interface, not in a networked serving protocol — DeepSpec
co-locates draft and target on the same GPUs. The pieces most relevant to Goal B
(streaming MoE) are the `SPS(B)` cost-table economics: throughput `Θ = τ·SPS(B)`,
which is exactly "accepted-tokens-per-verify-pass × pass rate".

Direct tie-in: DeepSeek released `DeepSeek-V4-Flash-DSpark` /
`DeepSeek-V4-Pro-DSpark` checkpoints (`tex/main.tex:145`). hipfire already has
`DeepSeek-V4-Flash` and `DeepSeek-V4-Flash-DSpark` on disk — that on-disk DSpark
sidecar **is** this drafter (3 MoE layers + mHC + sliding-window-128, block
`γ=5`, Markov head; `tex/sections/infra.tex:9`).

---

## 1. What DeepSpec / DSpark is

**DeepSpec** (`src/README.md:1-3`) is a full-stack *training + evaluation*
codebase for speculative-decoding drafters. It implements three drafters over a
shared framework: **Eagle3** (autoregressive, TTT), **DFlash** (parallel), and
**DSpark** (the paper's contribution). It is adapted from SpecForge (the Eagle3
training framework) and the original DFlash repo (`src/README.md:79-85`). There
is **no serving engine, no RPC, no continuous-batching scheduler in the repo** —
the online deployment lives inside the proprietary DeepSeek-V4 serving stack; the
paper only describes it.

**DSpark algorithm** (per-cycle, `tex/sections/arch.tex:19-24`, figure caption):
target emits an anchor/bonus token → DSpark drafts a `γ`-token block + per-token
confidence scores → scheduler keeps a confident prefix → target verifies the
prefix in one pass with standard rejection sampling → repeat. Three components:

1. **Semi-autoregressive generation** (`tex/sections/arch.tex:26-94`). A heavy
   **parallel backbone** (DFlash) produces all `γ` hidden states `h_k` and base
   logits `U_k` in one forward pass, so `T_draft` is ~independent of `γ`. Then a
   **lightweight sequential head** adds a prefix-dependent bias `B_k`, giving a
   *locally* autoregressive factorization
   `p_k(v|x0,x_<k) = softmax(U_k(v) + B_k(x0,x_<k,v))`. Crucially the correction
   stays **local softmax** so per-token probabilities remain exact — required for
   lossless rejection sampling (contrast CRF/CTC drafters,
   `tex/sections/related_work.tex:21-23`). Two head variants:
   - **Markov head** (default): bias depends only on previous token,
     `B(x_{k-1},·) = W1[x_{k-1}]·W2`, low-rank `V×r` · `r×V`, `r=256`
     (`tex/sections/arch.tex:70-81`, code `src/deepspec/modeling/dspark/markov_head.py:8-90`).
   - **RNN head**: GRU-like recurrent state over the block; marginal extra gain,
     dropped for deployment (`tex/sections/arch.tex:83-94`,
     `markov_head.py:125-284`). A "gated" Markov variant also exists in code
     (`markov_head.py:93-122`).
2. **Confidence head** (`tex/sections/arch.tex:107-125`): a single linear +
   sigmoid on `[h_k; W1[x_{k-1}]]` predicting the *conditional* survival prob
   `c_k` (that token `k` survives given all earlier accepted). Supervised against
   the analytic per-step acceptance `c*_k = 1 − ½‖p_d − p_t‖₁`. Calibrated
   post-hoc by **Sequential Temperature Scaling (STS)** — per-position 1D grid
   search on temperature minimizing ECE of the cumulative product. Code:
   `AcceptRatePredictor` (`src/deepspec/modeling/dspark/common.py:43-49`),
   `predict_confidence_step` (`modeling/dspark/qwen3/modeling.py:292-307`).
3. **Hardware-aware prefix scheduler** (`tex/sections/arch.tex:127-169`,
   `alg:prefix-scheduler`): picks per-request verify length to maximize
   system-wide throughput. See §3.

**Not MTP, not EAGLE, not block-diffusion.** It's a non-autoregressive/parallel
(NAT-style) block drafter with a *local* AR correction head — explicitly
positioned against MTP (the production baseline it replaces) and against
diffusion drafters (DART) (`tex/sections/related_work.tex:5-8`).

**Numbers**: offline, DSpark beats DFlash by 16–18% and Eagle3 by 27–31% in
accepted length `τ` (`tex/sections/exp.tex:40-66`). Sequential head adds only
0.2–1.3% round latency (`tex/sections/exp.tex:129-131`). Online vs MTP-1:
+60–85% tok/s/user at matched throughput on V4-Flash (`tex/sections/intro.tex:27`).

Config anchors (Qwen3-8B, `src/config/dspark/dspark_qwen3_8b.py`): `block_size=7`,
`num_draft_layers=5`, `target_layer_ids=[1,9,17,25,33]`, `markov_rank=256`,
`num_anchors=512`, loss `0.1·CE + 0.9·TV(L1) + 1.0·conf-BCE`, position weight
`w_k=exp(-(k-1)/γ)` (`tex/sections/arch.tex:178`, `config:26-30`).

---

## 2. Disaggregation / serving architecture (Goal A)

**Important expectation-setter: DeepSpec does *not* split draft and target across
machines or processes.** In the DSpark deployment the drafter is co-deployed on
the same DeepSeek-V4 serving GPUs (`tex/sections/infra.tex:9`, "co-deployed
with"). There is no draft/target wire protocol in the repo. However, three things
DeepSpec *does* specify map almost directly onto what hipfire wants to build, and
they collectively define the small interface Goal A needs.

### 2a. The draft↔target data interface (this is hipfire's cross-device payload)

DFlash-style **KV injection** is the entire coupling surface between target and
draft (`tex/sections/background.tex:39-51`, code
`modeling/dspark/qwen3/modeling.py:87-151`, `_forward_backbone:361-386`):

- The draft consumes **target hidden states from a fixed set of `m` target layers**
  (`target_layer_ids`, e.g. `[1,9,17,25,33]`), concatenated along feature dim:
  `extract_context_feature` = `cat(hidden_states[layer_id+1] for layer_id in ids)`
  → shape `[tokens, m·d]` (`common.py:52-56`). Note `hidden_states[0]` is the
  embedding output (`-1` sentinel) and the **final target layer is forbidden**
  (`base_evaluator.py:100-112`) — must use interior layers, which matters for a
  disaggregated target that would otherwise only expose its last hidden state.
- Draft projects that down: `fc: Linear(m·d → d)` then RMSNorm
  (`modeling.py:240-245, 373`), and injects it as extra **K/V context** that every
  draft layer attends to bidirectionally alongside the `γ` mask/anchor tokens
  (`modeling.py:103-113` concatenates `k_ctx = k_proj(target_hidden)` with
  `k_noise = k_proj(draft_hidden)`; attention is non-causal, `is_causal=False`).

So the **only thing that must cross from target→draft each cycle** is the
`m·d`-wide hidden vector for the freshly committed tokens. Per the evaluator, the
draft context is refreshed to just the accepted span:
`context.target_hidden_states = verified_target_hidden[:, :accepted+1, :]`
(`src/deepspec/eval/dspark/evaluator.py:139-147`). That is exactly hipfire's
"target_hidden ~384 KB/cycle": for `d=4096`, `m=5`, bf16 → `m·d·2 = 40 KB/token`;
at ~5–9 committed tokens/cycle that's ~200–360 KB. **The interface is confirmed:
ship `m` interior target-layer hidden states for the accepted tokens, nothing
else.**

The reverse direction (draft→target) is the proposal to verify. From the verify
path (`base_evaluator.py:186-304`): target needs `verify_input_ids` (the anchor +
`ℓ` scheduled draft tokens) and, for lossless rejection sampling,
`proposal.draft_probs` at those positions (`verify_draft_tokens` computes
`min(1, p_t/p_d)`, cumprod prefix mask, and `sample_residual` on first rejection,
`base_evaluator.py:241-283`). Plus the `γ` confidence scalars if the scheduler
runs target-side.

### 2b. Training-side comms optimization = the disaggregation recipe

The training framework (HAI-LLM) solves the exact bandwidth problem Goal A has,
and states the answer (`tex/sections/infra.tex:14-17`):

- **"Hidden state communication."** Do **not** move full-vocab logits
  (`V≈1e5`) between workers. Cache the target's activations and communicate only
  the **hidden state immediately before the LM head**; run the LM-head projection
  *locally* on the draft worker for just the sampled positions. Per-token comms
  drops from `O(V)` to `O(d)`. The DSpark draft **shares and freezes the target's
  embedding + LM head** (`tex/sections/arch.tex:53`, `modeling.py:270-287`), so
  the draft box can materialize logits itself.
- **"Anchor-bounded sequence packing."** Decouple draft cost from target context
  length by sampling a fixed number of anchors (`num_anchors=512`) and packing
  isolated `γ`-blocks, using **token-level attention indices** (a marker tensor)
  instead of 2D masks (`common.py:78-106` `create_dspark_attention_mask`,
  `sample_anchor_positions:123-169`).

### 2c. Scheduler / pipelining under a real engine (async, ZOS)

The production scheduler (`tex/sections/infra.tex:21-40`) is the one part that is
genuinely a *systems* contribution and directly informs pipelining:

- Real hardware capacity `SPS(B)` is **jagged/step-wise**, not smooth, and
  dynamic per-step batch sizing **clashes with CUDA-graph replay and
  Zero-Overhead Scheduling (ZOS)**, which need next-step batch size known before
  the current step finishes.
- Fix: run the scheduler **asynchronously using confidence from *two steps
  prior*** to set the truncation length `K` (a dynamic top-`K`), while sorting the
  *current* candidates by their up-to-date cumulative confidence. This hides
  scheduling latency and — as a bonus — restores losslessness even with the
  `break` removed, because the admission decision no longer depends on the current
  token's realization (the "causal barrier", `tex/sections/infra.tex:28-30`;
  counterexample proof in `tex/main.tex:171-254`).
- Execution layer: to verify **variable-length** prefixes per request in one
  batch without padding waste, they **flatten all tokens as independent elements**
  and carry intra-sequence structure in a **marker tensor** inside the sparse
  attention kernel; on V4 only the index-attention and compress kernels needed
  modification (`tex/sections/infra.tex:40`).

**What hipfire should take from §2:** the small interface is (target→draft)
`m` interior hidden states for accepted tokens, (draft→target) token ids + draft
probs at scheduled positions + confidence scalars. Keep the LM head on the draft
side. The two-steps-prior async scheduler is the template for hiding LAN RTT.

---

## 3. MoE + spec-decode economics (Goal B)

DSpark's scheduler formalizes exactly hipfire's "accepted-tokens-per-verify-pass
is the throughput driver" hypothesis (`tex/sections/arch.tex:157-169`):

- Batch tokens sent to target: `B = Σ_r (1 + ℓ_r)`.
- Expected accepted tokens: `τ = Σ_r (1 + Σ_{j≤ℓ_r} a_{r,j})`, where
  `a_{r,j} = Π_{i≤j} c_{r,i}` is the calibrated prefix-survival prob.
- **Objective: maximize `Θ = τ · SPS(B)`**, i.e. accepted-tokens × steps-per-sec.
  `SPS(B)` = engine throughput (steps/sec) as a function of verify batch size,
  **profiled once at engine init into a lightweight cost table**
  (`tex/sections/arch.tex:161`). This cost table is the abstraction that captures
  MoE behavior: the flat region (extra verify ~free) then the cliff (where extra
  tokens steal batch capacity).
- Greedy solution: sort all `(r,j)` by `a_{r,j}` descending, admit incrementally,
  stop when `Θ` stops rising (`break` for sync losslessness; removed in async
  ZOS). Monotonic `a_{r,j}` makes greedy optimal (`tex/sections/arch.tex:163`).

The paper's key regime statement (`tex/sections/infra.tex:37-38`) **validates the
streaming-bound assumption directly**: effective batch "persistently remains well
below the GPU's compute-saturating threshold," so maximizing per-GPU throughput
and per-user tok/s become *correlated, not competing*. Under this regime longer
draft blocks are nearly free until the `SPS` cliff — which is why static MTP-3/5
was avoided (fixed verify length hits the cliff under concurrency,
`tex/sections/exp.tex:56-59`) and why the *scheduler*, not just a longer drafter,
is the unlock.

MoE specifics: the V4 draft backbone itself is **3 MoE layers + mHC +
sliding-window-128** (`tex/sections/infra.tex:9`). They also note prefill-decode
disaggregation with a decode load balancer keeping request count and context
length balanced across DP ranks, which is *why* they can assume `SPS` depends
only on `B` (`tex/sections/arch.tex:160` footnote). No expert-streaming or
microbatching detail is given — the cost table absorbs all of it. For hipfire's
streaming Qwen3.5-397B-A17B (weight-bandwidth bound), the actionable analogue is:
**profile `SPS(B)` for the streaming target and let the scheduler pick verify
length against that curve; the flat region confirms streaming-bound.**

---

## 4. MTP relevance (Goal 4)

- DSpark **is not MTP** and does not use MTP heads. Its production baseline and
  the thing it *replaces* is **MTP-1** (DeepSeek-V3-style single MTP head,
  `tex/sections/intro.tex:27`, `tex/sections/exp.tex:56`). MTP appears only as the
  competitor.
- The paper's finding is directly relevant to a Qwen3.5 target that ships its own
  MTP head: a **static multi-token MTP (MTP-3/5) degrades aggregate throughput
  under concurrency** because it verifies a fixed length regardless of load
  (`tex/sections/exp.tex:56-59`). This is the failure mode the confidence
  scheduler exists to fix.
- Composition vs competition for hipfire: a Qwen3.5 MTP head and a DFlash/DSpark
  block drafter are **two alternative drafters for the same target — they
  compete, they don't stack.** DSpark's own "sequential head" is the conceptual
  slot an MTP head would fill, but DSpark keeps it *local and cheap* (rank-256
  Markov) rather than a full transformer MTP layer, precisely to keep
  `T_sequential ≪ T_parallel`. Practical read: if Qwen3.5 ships an MTP head, treat
  it as the cheap MTP-1-style fallback drafter; a DFlash-DSpark block drafter is
  the higher-`τ` replacement, and the confidence scheduler is what makes its
  longer blocks safe under load. Do **not** try to run MTP *and* a block drafter
  as one proposer.

---

## 5. Reusable implementation specifics (Rust/HIP)

All cheap, all local to the draft device — good for the NPU box:

- **Markov head** (`markov_head.py:8-90`): per step `k`: embedding lookup
  `W1[x_{k-1}] ∈ R^r` (r=256), then `r×V` matvec `W2` added to base logits, sample,
  feed token to next step. `γ` serial steps, each dominated by one `r×V` GEMV
  (`V≈151k`, r=256 → ~39M MACs/step). Storage: two `V×r` matrices. Order-preserving
  and exact-softmax → drop-in for lossless verify.
- **Confidence head** (`common.py:43-49`, `modeling.py:292-307`): `Linear(d+r → 1)`
  + sigmoid on `[h_k; W1[x_{k-1}]]`. One dot product per position. Cache the same
  `W1[x_{k-1}]` embedding you already computed for the Markov head.
- **STS calibration**: offline per-position 1D temperature grid search minimizing
  ECE of the cumulative product (`tex/sections/arch.tex:122-125`). Ship `γ` scalar
  temperatures; apply as `sigmoid(logit/T_k)` before cumprod. ECE tooling to mirror
  for validation: `src/deepspec/eval/dspark/confidence_head.py:30-172`
  (per-position ECE/AUROC/Brier, cumprod predictions).
- **Scheduler** (`alg:prefix-scheduler`, `arch.tex:130-153`): global sort by
  `a_{r,j}`, greedy admit, `Θ=τ·SPS(B)` lookup, early-stop. Async production
  variant: use confidence from 2 steps prior for the top-`K` cap, sort current by
  live confidence, drop the `break` (`infra.tex:28-30`). `SPS(B)` = a profiled
  1-D cost table indexed by batch size.
- **Verify / rejection sampling** (`base_evaluator.py:186-304`): `accept_prob =
  clamp(p_t(x)/p_d(x), max=1)`; `accept_prefix_mask = (rand<accept_prob).cumprod`;
  on first reject `sample_residual(p_t, p_d)`; else bonus from `p_t[-1]`. Draft
  probs required only at the `ℓ` scheduled positions.
- **KV-injection attention** (`modeling.py:87-151`): draft attn K/V =
  `cat(proj(target_hidden_ctx), proj(draft_hidden))`; non-causal; RoPE applied to
  both; q/k RMSNorm per head (Qwen3 style). Context projection: `fc: Linear(m·d→d)`
  + RMSNorm once per cycle (`_forward_backbone:373`). This is the seam hipfire's
  `dflash_body_npu.py` is already implementing.
- **Anchor/block masking for training** only (`common.py:78-294`) — not needed at
  inference, but documents the packing scheme if hipfire ever trains its own
  DSpark head. Inference uses a plain block: anchor token + `γ−1` mask tokens
  (`evaluator.py:109-115`, `mask_token_id=151669` for Qwen3).

Loss recipe if retraining (`tex/sections/arch.tex:171-198`): freeze target,
embedding, LM head; train backbone + sequential head + confidence head jointly;
`L = 0.1·CE + 0.9·TV + 1.0·conf-BCE`, position weight `exp(-(k-1)/γ)`.

---

## 6. Recommendations for hipfire's disaggregated draft/target protocol

1. **Adopt DSpark's "hidden-state, not logits" split as the wire contract.** Have
   the GPU/target box ship only the `m` interior target-layer hidden states
   (`target_layer_ids`, avoid the final layer) for the newly *committed* tokens —
   `[accepted+1, m·d]` bf16 ≈ 40 KB/token — and keep the (frozen, shared) LM head
   and the `fc`+RMSNorm context projection on the NPU draft box. This is the exact
   ~384 KB/cycle interface and it is the smallest possible coupling surface
   (`tex/sections/infra.tex:15`, `evaluator.py:139-147`). Draft→target reply is
   just `verify_input_ids[1+ℓ]` + `draft_probs` at the `ℓ` scheduled positions +
   the `γ` confidence scalars. Consider top-k compression of `draft_probs` (only
   the sampled token's prob and enough mass for residual sampling are strictly
   needed).

2. **Run the confidence scheduler on the target/GPU box, asynchronously, using
   two-steps-prior confidence.** The GPU owns the batch and the `SPS(B)` cost
   table; profile `SPS(B)` once at init for the streaming-MoE target. The
   two-steps-prior async design (`infra.tex:28-30`) is purpose-built to hide
   scheduling latency behind the verify pass — reuse it to also hide **NPU draft
   compute + LAN RTT**: pipeline `draft(cycle t+1)` on the NPU during
   `verify(cycle t)` on the GPU, and size the verify length to what will actually
   have arrived. This turns the LAN hop into a pipeline stage rather than a stall.

3. **Make accepted-tokens-per-pass the single optimization target and gate on it,
   not SNR.** Implement `Θ = τ·SPS(B)` with the greedy prefix scheduler; the cost
   table captures the MoE streaming cliff automatically. Because hipfire's
   streaming batch sits below saturation (matching `infra.tex:37`), push `γ`
   higher than the offline default (paper scales to 16) and let the scheduler
   prune — this is where the 60–85% tok/s/user wins came from, and it composes
   with the existing memory note that DFlash acceptance should be judged by
   acceptance rate, not SNR. For lossless behavior over the LAN, keep the async
   causal-barrier property (admission depends only on ≥2-steps-prior info); if you
   run synchronously instead, keep the `break` early-stop.

Secondary: the DSpark checkpoints for `DeepSeek-V4-Flash-DSpark` are public and
match hipfire's on-disk model — the drafter head weights (Markov `W1/W2` r=256,
confidence linear, per-position STS temperatures) can be imported rather than
retrained; only the `fc` context projection dims depend on `target_layer_ids`.
