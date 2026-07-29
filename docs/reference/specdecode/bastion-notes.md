# BASTION — distilled notes (for hipfire DFlash NPU spec-decode)

Source: `/srv/hipfire/references/SpecDecode/BASTION/` — NeurIPS 2026 workshop
paper (`tex/`) + official code (`src/`, git repo, MIT). arXiv 2605.29727.

**Bottom line:** BASTION is a *direct sibling of hipfire's DFlash work*. It takes
the exact same DFlash block-diffusion drafter hipfire runs on the NPU, and adds a
**query-adaptive verification tree** on top of it plus a **hardware-cost-model
budget controller** that decides how big that tree should be each step. It is
highly relevant to Goal B (it is literally an accepted-tokens-per-verify-pass
maximizer) and moderately relevant to Goal A (it changes what crosses the
draft↔target boundary and makes the verify pass a tree, not a chain).

---

## 1. Title + thesis

**Title:** *BASTION: Budget-Aware Speculative Decoding with Tree-structured
Block Diffusion Drafting* (Oh, Cao, Kim, Jung, Ahmad, Bae, Yun — KAIST AI +
Samsung SAIT). Confirmed in `tex/neurips_2026.tex:219-222` and `src/README.md:5`.
The `\algo` macro (`neurips_2026.tex:211`) = "BASTION"; acronym expands to
**b**udget-**a**ware **s**peculative decoding with **t**ree-based
diffus**ion** drafting (`00_abstract.tex:5`).

**Thesis** (`00_abstract.tex`, `01_introduction.tex:24-38`): a one-step block-
diffusion drafter (DFlash/TiDAR) emits **position-wise marginals** `q_k(·)`, not
a jointly-conditioned sequence. Committing to the single greedy `argmax` path
therefore throws away the target-preferred trajectory that usually still lives in
the drafter's top-K. Because block-diffusion drafting is **topology-decoupled**
(one parallel forward gives all marginals at constant cost regardless of
chain/tree shape), you can build a *prefix tree* from those marginals and let the
target verify many candidate paths in one pass — for free on the draft side. The
open question BASTION answers: **how big and what shape** should that tree be,
per query, per GPU. Contribution is (1) a best-first tree construction that is
provably optimal under a path-score surrogate, (2) an online roofline cost model
that predicts verify latency, (3) a controller that grows the tree until marginal
acceptance gain stops paying for its verify cost. Training-free, distribution-
preserving, no per-setting tuning.

**Headline numbers** (`01_introduction.tex:38`, `README.md:15`): 6.61× over AR,
2.45× over EAGLE-3, **1.39× over DFlash** (its own single-path baseline) —
Qwen3-8B, T=0, A100, 8 math/code/chat benchmarks. So the tree+budget machinery
buys ~39% on top of the DFlash drafter hipfire already has.

---

## 2. Core method / algorithm

Reference implementation: `src/bastion/tree_draft.py`,
`src/bastion/cost_model.py`. Math: `tex/tex/04_method.tex`.

**Setup / two exploited properties** (`04_method.tex:13-20`):
- (P1) **Topology decoupling** — `T_draft = t_parallel` is invariant to whether
  candidates form a chain, wide tree, or deep tree. One DFlash forward gives all
  `γ` position marginals.
- (P2) **Marginal independence** — each `q_k` is conditioned only on the
  validated context, *not* on drafted tokens at positions `1..k-1`. So candidate
  tokens across positions can be recombined freely into any prefix tree.

**Path score** (`04_method.tex:24`, eq. path-score): a root→node path gets
`ρ(i) = Π_k q_k(x_{i_k})` (product of per-position marginal probs along the
path). Root `ρ(r)=1`.

**Acceptance surrogate** (`04_method.tex:37-62`, Prop. "Path-Sum Surrogate"):
the expected committed length of a tree `T` under a *self-verification* thought
experiment (drafter verifies itself) is **exactly** the sum of path scores:
`Â(T) = Σ_{i∈T} ρ(i)`. This is the estimator maximized before the target is
ever run. Justified as a proxy for the true target-side `E[A(T)]` by
drafter–target alignment; validated empirically at Pearson ρ ≥ 0.79
(`04_method.tex:65`, appendix `mean_al_correlate`).

**Best-first tree construction** (`04_method.tex:71-94`, Prop. "Optimality of
Best-First"): because `ρ` is *path-monotone* (`ρ(child) = ρ(parent)·q ≤
ρ(parent)`), iteratively popping the highest-`ρ` open node off a max-heap yields
nested trees `T_1 ⊂ T_2 ⊂ …` where each `T_N` maximizes `Â` among all size-N
prefix-closed trees. Marginal gain `ΔÂ(N) = ρ(i_{N+1})` is non-increasing → `Â`
is **concave** in budget N. This beats beam search, which forces a rigid
(width×depth) rectangle and ignores global ordering of path confidences across
depths. Ablation: best-first +7.0%/+6.1% speedup over budget-matched beam search
on Qwen3-4B/8B (`05_experiments.tex:92`).

**Online budget controller** (`04_method.tex:99-136`): maximize estimated speedup
`S(N) = Â(T_N)·L_AR / C(N)` where `C(N) = T_draft + T_aux + T_verify(N)`.
`Â` concave + `C` convex ⇒ `S(N)` **unimodal** ⇒ stop at first N where
`S(N+1) < S(N)` and return `T_N` (Prop. "Unimodality"). Runs as one fused loop
with best-first expansion, updating `Â` incrementally by `+ρ(i_{N+1})` per node.

**Actual stopping test in code** — `tree_draft.py:251-254`:
```
next_cost = prev_cost + latency_estimator.next_delta(prev_tree_size)
lhs = new_path_probability * (draft_latency + non_verify + prev_cost)   # ρ·C_N
rhs = path_prob_sum        * (next_cost - prev_cost)                    # Â_N·ΔC
if lhs <= rhs and prev_tree_size >= min_tree_size: break
```
This is exactly the rearranged `S(N+1) < S(N)` discrete-derivative test:
continue while `ρ_{N+1}·C_N > Â_N·ΔC(N)`. **Note `L_AR` cancels out** — the AR
step latency does not enter the stop decision at all; only the marginal path prob
of the next node vs. the current tree's mass × the incremental verify cost.

**Verify + commit** (`tree_draft.py:447-643`, `bastion_generate`):
1. One DFlash draft forward → `draft_logits [1, block, vocab]`, KV cropped back.
2. `build_adaptive_best_tree_from_draft_logits` → `TreeDraft` (token_ids,
   parent_indices, depths, tree_mask, retrieve_indices, path_lens). Top-K=128
   per position (`_MAX_ADAPTIVE_TOP_K`), min tree 32 / max 8192 nodes.
3. Target forward on the flattened tree with a **custom tree attention mask**
   (`_build_tree_attention_mask`, block-diagonal ancestor mask over the tree +
   full attention to prefix) and **per-node position_ids = start + depth**.
4. `_select_best_tree_path`: greedy-match each path against target argmax,
   `cumprod` to find longest accepted prefix, sample the next token from the
   target logits at the accepted leaf. Exact/greedy acceptance (T=0) — the paper
   claims distribution-preserving.
5. `_reorder_kv_cache_for_accepted_path`: index-select the target KV to keep
   only prefix + accepted nodes, then `crop`. Draft's `target_hidden` for the
   next step is taken from `verify_output.hidden_states` at the accepted node
   indices (`tree_draft.py:635`).

**Cost model** (`cost_model.py`, `tex/tex/appendix/05_cost_models.tex`): analytic
roofline `R(N) = max(FLOPs(N,C)/PeakFLOPs, Bytes(N,C)/BW)`. FLOPs and memory-
traffic are closed-form Transformer formulas (`calculate_qwen_flops`,
`calculate_qwen_inference_memory_footprint`) parameterized by
`{L,h,n_q,n_kv,d,h_ffn,V}` per model and `{peak_bf16, mem_bw}` per GPU (hardcoded
tables `cost_model.py:23-147`). A cheap 3-point quadratic fit gives
`next_delta(N)` in O(1). A **linear per-(GPU,model) calibration**
`α·raw + β` on both branches (fit once by `fit_roofline_calibration`, cached to
`CALIBRATION_JSON_PATH`) corrects for kernel fusion / tile-allocation surges;
cuts latency-prediction RMSE by 87–90% (`06_analysis.tex:52`). Bare roofline
alone systematically *under*-predicts. Offline "Static" calibration is the robust
default; an online EMA-only variant is fragile (`06_analysis.tex:59-60`).

---

## 3. Relevance to Goal A (disaggregation)

**Directly bears on the interface, but complicates the verify side.**

- **The draft↔target interface is exactly `target_hidden`.** Both DFlash and
  BASTION feed the drafter a concatenation of *selected target hidden-state
  layers* at the accepted positions:
  `target_hidden = cat(hidden_states[layer_id+1] for layer_id in target_layer_ids)`
  (`dflash/model.py:39-45`, `tree_draft.py:336-346`). `target_layer_ids` has one
  entry per draft layer, spread across the target stack
  (`build_target_layer_ids`, `model.py:27-36`). The drafter projects this down
  (`fc: len(ids)*h → h`, `model.py:317`) and injects it as the **K/V context**
  of every draft attention layer (`Qwen3DFlashAttention.forward`,
  `model.py:226-231`: `k = cat(k_proj(target_hidden), k_proj(noise))`). So the
  disaggregation payload target→draft is `[accepted_len × (n_draft_layers · h)]`
  in bf16, per cycle. For a 5-layer drafter over hidden-4096: 5·4096·2 =
  **40 KB per accepted token**; a full block (~8 accepted) ≈ 320 KB — matching
  hipfire's stated ~384 KB/cycle budget for Goal A. This confirms the interface
  size is realistic and that **BASTION does not enlarge it** (tree drafting reads
  `target_hidden` only at *accepted* nodes, same as DFlash).
- **Verification protocol is NOT a simple chain any more.** BASTION's verify pass
  needs the target to accept (a) a flattened tree of token_ids, (b) arbitrary
  per-node `position_ids = start + depth`, (c) a dense `[tree_len × (prefix +
  tree_len)]` tree attention mask, and (d) a scatter/index KV reorder after
  commit. For a disaggregated big-memory GPU target this means the target box
  must expose a *tree-mask attention verify* entrypoint, not just "run N tokens
  causally." FlashAttention can't do the custom mask — BASTION falls back to SDPA
  for verify (`README.md:62`). On hipfire this is a real kernel ask: the RDNA
  attention verify kernel would need a boolean tree mask + gather.
- **The controller lives on the draft box and is cheap.** Tree construction is
  pure CPU heap ops on the draft logits (`tree_draft.py` uses `heapq`), and the
  cost model is a few flops. So in a disaggregated layout the *drafter* can decide
  the tree/budget locally from its own logits + a static per-target-GPU
  calibration JSON — no round-trip needed to size the tree. That fits Goal A's
  "small interface": you ship a calibration blob for the target GPU once, then the
  Phoenix box sizes trees autonomously.
- **Caveat for LAN latency:** BASTION's cost model only accounts for on-box
  `T_verify`; it has **no term for network RTT** between draft and target. Under
  disaggregation the real per-cycle cost gains a fixed `T_net` (hidden-state
  ship + tokens back). That constant inflates the denominator `C(N)`, which
  *raises* the optimal tree size (amortize the fixed network cost over more
  accepted tokens). Adding a `T_net` constant to `non_verify_latency_s`
  (`tree_draft.py:252` already sums a `non_verify` term into `C_N`) is the
  natural hook — the code already has the slot.

---

## 4. Relevance to Goal B (streaming MoE / accepted-tokens-per-pass)

**This is the paper's home turf and the most transferable part.**

- BASTION *is* an accepted-tokens-per-verify-pass optimizer. Its entire objective
  `S(N) = Â·L_AR / C(N)` is "maximize expected accepted length per unit verify
  cost." Goal B's throughput driver (accepted tokens per verify pass) is exactly
  BASTION's numerator `Â(T_N)`. The framework *is* the tool for "drafter
  quality/width multiplies throughput" — tree width directly raises `Â`.
- **Streaming-bound regime is where wide trees pay most.** BASTION's own cost
  model shows `T_verify` is `max(compute, memory)` roofline. When the target is
  streaming-bound (Qwen3.5-397B-A17B: weights dominate memory traffic, verify is
  memory-bound and *nearly flat in N* until compute crosses over), the marginal
  verify cost `ΔC(N)` of adding tree nodes is tiny — so the unimodal controller
  will choose **large trees**, because acceptance gain keeps paying until compute
  finally dominates. This is precisely the "verify is cheap per extra token → make
  the tree wide" logic Goal B wants. Their `next_delta`-based stop rule
  operationalizes it: continue while `ρ_{N+1}·C_N > Â_N·ΔC(N)`; when `ΔC≈0`
  (flat memory-bound branch) the tree grows until top-K/depth exhausts.
- **BUT the cost model is dense-Transformer only and MoE-blind.** `calculate_
  qwen_flops` / `..._memory_footprint` (`cost_model.py:248-322`) assume a dense
  FFN (`3·h·h_ffn` weights read every token). For a 512-expert/10-active MoE the
  memory traffic per verify token is *not* the full FFN — it's the union of
  experts touched by the tree's tokens. Two consequences hipfire must model:
  (a) more tree tokens → more distinct experts activated → verify memory grows
  *super*-linearly early then saturates at "all experts resident"; (b) the
  crossover point and thus optimal N are entirely different from the dense
  formula. **The roofline *shape* (max(compute,memory), α·raw+β calibration) is
  reusable; the FLOP/byte formulas must be rewritten for MoE.** This is the
  single biggest adaptation for Goal B.
- **Acceptance-rate lever is orthogonal and free on the draft side (P1).** Since
  block-diffusion draft cost is topology-invariant, widening the tree costs the
  drafter nothing — all the gain (`+39%` over single-path DFlash) is verify-side
  utilization. For a giant MoE target where each verify pass streams 397B params
  regardless, accepting 3–4× more tokens per pass for near-zero extra verify cost
  is the dominant win. Goal B's thesis ("accepted-tokens-per-pass is the driver")
  is exactly what BASTION monetizes.

---

## 5. Relation to DFlash / DDTree / DSpark / block-diffusion drafting

- **DFlash is BASTION's substrate, verbatim.** `src/dflash/model.py` is a
  vendored DFlash Qwen3 drafter (`chen2026dflash`); BASTION uses the *same HF
  checkpoints* hipfire targets (`z-lab/Qwen3-{4,8}B-DFlash-b16`,
  `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`; `README.md:67-71`). The drafter
  architecture — mask-token block, `target_hidden` injected as attention K/V
  context, `is_causal=False`, non-causal cross-attention denoise — is identical
  to hipfire's DFlash NPU bring-up (which is at Phase D block-body assembly). So
  BASTION's tree layer drops onto hipfire's existing drafter *without changing
  the NPU kernel work*: the NPU still produces `draft_logits [1, block, vocab]`;
  BASTION only changes what happens to those logits (CPU tree build) and how the
  *GPU target* verifies (tree mask).
- **DFlash baseline = the single-path `argmax` decode** (`dflash/model.py:107-
  143`, `dflash_generate`): draft block → `argmax` each position → target verifies
  the one chain → longest accepted prefix by `cumprod`. This is hipfire's current
  "lossless serial spec-decode." BASTION replaces the `argmax` chain with the
  top-K tree and swaps chain-verify for tree-verify. Everything else (KV crop,
  `target_hidden` re-extraction at accepted positions) is unchanged.
- **DDTree / DSpark:** BASTION does not mention DSpark or a "DDTree" by those
  names (grep of `tex/` finds neither; DSpark is hipfire-internal per MEMORY).
  BASTION *is* the "tree over a diffusion drafter" idea that DDTree would name.
  hipfire's own `npu-spec-decode-drafter-cost` note found tiny drafters are
  floor-bound and 5-layer DSpark bodies are feed-starved — BASTION is orthogonal:
  it doesn't make the drafter cheaper, it extracts more accepted tokens from
  whatever the drafter already produced. The two compose.
- **Block-diffusion drafting:** BASTION cements *why* block-diffusion (vs AR
  drafting) is the right base for trees — P1 (topology-free draft cost) is only
  true for parallel/diffusion drafters. AR drafters (EAGLE-3) pay per tree branch.
  This is a strong argument for hipfire's block-diffusion NPU direction over an
  AR NPU drafter.

---

## 6. Reusable implementation specifics + recommendations

**Reusable, low-risk to port:**
- `build_adaptive_best_tree_from_draft_logits` (`tree_draft.py:155-322`) — a
  self-contained ~150-line CPU/heap algorithm: top-K per position → max-heap on
  path log-prob → best-first expand with sibling+child pushes → unimodal stop.
  Input is just `draft_logits [1, depth, vocab]` + 4 scalars (draft latency,
  non-verify latency, prefix len, calibration). Portable to Rust on the NPU/host
  side almost 1:1.
- The **stop test** (`tree_draft.py:251-254`) is the whole controller in 4 lines;
  `L_AR` cancels, so you only need `ρ_next`, running `Â` (`path_prob_sum`),
  current cycle cost `C_N`, and `ΔC = next_delta(N)`.
- Roofline `VerifiedLatencyEstimator` (`cost_model.py:196-238`): `max(α_c·compute
  + β_c, α_m·mem + β_m)`, O(1) `next_delta`, 3-point quadratic fit. The
  **structure** (calibrated max-of-two-branches, per-(GPU,model) `α,β` JSON) is
  the reusable part; hipfire would swap RDNA/NPU peak-bf16 + HBM/UMA bandwidth
  and rewrite the FLOP/byte closed forms.
- Tree plumbing: `_build_tree_mask`, `_build_retrieve_indices`,
  `_build_tree_attention_mask`, `_reorder_kv_cache_for_accepted_path`,
  `_select_best_tree_path` — reference semantics for a tree-verify target kernel.

**Concrete recommendations for hipfire:**
1. **Add tree drafting on top of the existing DFlash NPU path — it's additive and
   free on the drafter.** The NPU already emits `draft_logits`; build the
   best-first tree on the host (port `tree_draft.py:155-322` to Rust), and add a
   *tree-mask attention verify* entrypoint to the RDNA target (boolean tree mask +
   per-node position_ids + accepted-path KV gather). Expect ~1.4× on top of
   current lossless serial DFlash, per BASTION's own DFlash→BASTION delta. This is
   the highest-ROI item and does not touch NPU kernels.
2. **For Goal B, keep the roofline+calibration skeleton but rewrite the cost
   formulas for MoE.** Replace `calculate_qwen_flops/..._footprint` with an
   MoE-aware model: verify memory ≈ (active-expert weight bytes actually touched
   by the tree's tokens, saturating at all-experts-resident) + KV + activations.
   Keep the `α·raw+β` per-(GPU,model) calibration — it's what turns a rough
   analytic model into a usable one (87–90% RMSE cut). This lets the controller
   correctly choose the *wide* trees that a streaming-bound 397B-A17B target
   rewards. Without it, the dense formula will badly mis-size trees.
3. **For Goal A, thread a network-cost constant into `C(N)` and size trees on the
   draft box.** The `non_verify_latency_s` slot (`tree_draft.py:252`) already adds
   into the cycle cost — put the LAN hidden-ship+token-return RTT there. The
   controller then automatically grows trees to amortize the fixed network cost,
   and the whole tree/budget decision stays local to the Phoenix drafter (needs
   only a static calibration JSON for the remote target GPU). Confirm the
   interface payload is `n_draft_layers · h · 2` bytes per accepted token
   (~40 KB/token for a 5-layer/h4096 drafter), read only at accepted positions —
   BASTION does not grow it.

**Honesty flags:** batch-size-1 only, Transformers/CUDA/NVIDIA-only reference
impl, dense-target cost model (no MoE, no quant, no NPU, no network). The *ideas*
(marginal→tree, path-sum surrogate, best-first optimality, unimodal budget stop,
calibrated roofline) transfer cleanly; the *code's cost tables and verify kernel*
do not and must be rebuilt for RDNA/MoE/disaggregated hipfire.
