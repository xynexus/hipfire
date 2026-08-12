# Weaver / DFlash-TfM — technical notes

Source: `arXiv 2607.06763v2`, "Trees from Marginals: Autoregressive drafting
with factorized priors" (Oda, Mathieu, Knyazhitskiy, Chakhvadze — getmirai.co).
35 pp; body ends p.23, rest is bibliography. Citations below are to sections /
figures / tables of that PDF.

> One-line: **Weaver is a tiny (56.7M) autoregressive adapter bolted on top of a
> factorized DFlash drafter. It reuses DFlash's top-K marginals as a candidate
> vocabulary and re-introduces the conditional dependencies DFlash's
> independence assumption throws away — turning DFlash's flat marginals into a
> proposal *tree* — while never projecting to the full vocabulary.** The full
> method is called **DFlash-TfM** (Trees from Marginals); "Weaver" is the adapter
> network itself.

This is the *most directly on-point* reference of the batch for hipfire's DFlash
NPU drafter: it is literally "what to build on top of DFlash next," and it names
DFlash as its baseline throughout.

---

## 1. Title + thesis (§1.2, Abstract)

- **Problem it attacks (§2.1.2, §3.5):** factorized/diffusion drafters (DFlash,
  Medusa, I-DLM, PARD) predict `T` future positions in **one** forward pass as
  *conditionally-independent marginals* given the prefix. That parallelism is
  why they win at small budgets, but position `n+k` is drawn ignoring the
  realized values at `n…n+k-1`, so the draft distribution diverges from the
  target's chain-rule conditionals. Per-position acceptance decays with depth →
  a hard **acceptance ceiling** that no amount of drafter capacity fixes (§6,
  Fig 4). This caps useful block size and expected accepted length.
- **Thesis:** the ceiling comes from the *independence assumption*, not drafter
  capacity, and a small **autoregressive** adapter is enough to lift it (§6).
- **Contribution (§1.2):** (a) *Weaver*, an AR adapter that predicts the
  **residual** between DFlash's marginals and the true conditional, restricted
  to DFlash's top-K candidates so it never pays the full-vocab projection; (b) a
  **rollback-free tree-verification kernel for Gated Delta Net (GDN) layers** —
  previously an open problem for non-diagonal linear-attention targets; (c)
  SGLang CUDA kernels.
- **Headline (§5.1):** on Qwen3.6-27B, B200, bf16, batch 1 — **4.37× over AR
  decoding, 24.7% over a tuned DFlash chain baseline**, 392.8 tok/s avg. Mean
  accepted length (MAL) **+77% vs chain DFlash, +32% vs DDTree at equal tree
  size** (Table 2).

So: this is a **drafter-quality + verification-systems** paper, not a
disaggregation or ensembling paper. It matters to hipfire because it improves
*accepted-tokens-per-verify-pass* — the exact quantity Goal B is bottlenecked on
— and because its drafter interface is a leaner version of hipfire's
`target_hidden` protocol (Goal A).

---

## 2. Core method / algorithm

### 2.1 Data flow (§3.1–3.2, Fig 2)
1. Start of a spec round: DFlash takes the last committed token's verifier
   hidden state `h_verifier` and emits `L` future-state lookaheads
   `h_dflash^{1..L}` (DFlash block is 16 positions, §6).
2. `ℓ_dflash^i = W_vocab · h_dflash^i` gives the baseline **marginal logits**.
   In vanilla DFlash you'd sample draft tokens from these. Here they are only a
   **prior**.
3. **Top-K selection:** per position, keep the top-`K` marginal tokens
   (K = 512). This is the *candidate vocabulary* — a tiny subset of full vocab.
   Empirically the verifier puts **97.8% of its mass inside the top-512 pool**
   (§4.1.3 footnote), so restricting to it is nearly lossless.
4. **Weaver** (an EAGLE-3-style AR transformer, §3.2) compresses inputs into
   continuous "conditioning tokens":
   ```
   u_0 = W_c · RMSNorm(h_verifier)                    (eq 5)
   u_i = W_c · RMSNorm(h_dflash^i) + p_i               (p_i = positional enc.)
   ```
   These `u_{0..L}` form a **static KV cache** (the "prompt"). Expanding a tree
   path with tokens `t_{1..d}`, Weaver's input is `concat(u_{0..L}, t_{1..d})`.
5. `ℓ_draft = WeaverStep(u_{0..L}, t_{1..d})` (eq 6). **Truncated projection:**
   only the **K rows** of `W_vocab` for the top-K DFlash tokens are read;
   Weaver's residual logits are *added* to DFlash's logits and normalized over
   the K-candidate set (§3.2). This is where the memory-bandwidth win lives —
   full-vocab projection is the dominant cost of a normal AR drafter and it is
   entirely avoided.

### 2.2 Tree construction (§3.3)
- Based on DySpec best-first (expand max-draft-probability node), but **batched**:
  pop the **top-`w` nodes** from the candidate heap and expand all `w`
  concurrently. Because a batched Weaver step of width `w` costs ~the same
  wall-time as a single unbatched step, a token budget `B` builds the whole tree
  in `⌈B/w⌉` sequential ops. **Optimal `w` = 2–8.** Trades some node optimality
  for bandwidth/compute balance.

### 2.3 Rollback-free GDN tree verification (§3.4) — the systems contribution
- Attention targets verify trees with a simple ancestor mask; **recurrent/SSM
  targets don't**. Diagonal (Mamba) recurrences were solved by STree via
  cumulative-product gates, but **GDN's non-commuting transition
  `I − β_t k_t k_tᵀ` has no cumulative-product form** — open problem until here.
- Trick: use the **dual chunk form** of the linear recurrence, but define
  `X`/`Y` interaction matrices over the tree's **partial order** (`j ≺ i` =
  ancestor) instead of a linear order (eq 9, Fig 3). Cumulative decay
  `a_t = ∏_{i⪯t} α_i` accumulated along branches (in log space).
- Solve `(I+X)U = βV`, `(I+X)W = βaK` (eq 10); output `O = (aQS_0 + Y(U−WS_0))/√d_k`
  (eq 11). **Never speculatively writes state**: the read-only verify reduces to
  a **masked triangular solve**; state commit is *delayed* until the accepted
  branch is known, then the path is **replayed** with the plain recurrence (eq 7)
  — the only state write in the whole decode step. No rollback needed.
- Kernel (§3.4.2): tree metadata computed **once per decode step, shared by all
  GDN layers**; ancestor sets as **64-bit bitmasks**; dense shapes independent of
  tree content → whole pass captured in a **CUDA graph**. Forward substitution
  tiled into `B_c = 32`-node blocks, diagonal blocks inverted in-register by
  repeated squaring (eq 13). tf32 dot / fp32 accum, tf32x3 for log-decay; stays
  within 1e-4 of fp64.
- Cost (Table 1, Fig 5): GDN verify = **12% of decode step (2.5 of 21 ms)**;
  target forward ≈13 ms; draft+Weaver prep ≈5.1 ms. Fused kernel is **7.1×**
  faster than per-branch recurrent at `T=128`. Solve stage is the one that grows
  super-linearly in tree size `T` (2.1× from T=64→128).

### 2.4 Training (§4.1)
- **LK loss** (eq 17): `λ·KL(p‖q) + (1−λ)·TV(p,q)`, with
  `λ = exp(−η·sg(1−TV))`, η=2. Acts as forward-KL while far, shifts to TV (=
  `1 − p_accept`) as it converges. Restricted to the K=512 candidates +
  renormalize; plus a small argmax-match term `−γ·log q(ĉ)`, γ=0.1 (eq 18) so
  greedy verification sees the right argmax.
- **Curriculum-ish masking:** apply loss only at positions the drafter would
  reach under speculative sampling; mask everything past the first reject.
- 56.7M params: **single transformer layer, dim 2048, 16 heads, MLP width 2048**,
  K=512. Muon for big matrices + AdamW, WSD schedule, LR 2e-4. Trained on top of
  the frozen public `Qwen3.6-27B-DFlash` checkpoint; 300k completions, 1 epoch.
- **Rollout trick (§4.1.1):** store only **token IDs** offline, then recompute
  verifier logits + DFlash lookaheads at train time (storing hidden states is too
  large on-disk). Directly relevant to hipfire's recompute-vs-transfer tradeoff.

---

## 3. Relevance to Goal A (disaggregation / draft↔target interface)

Weaver is **not** a disaggregation paper — it assumes a single B200 and even
notes the serving path handles one request at a time (§6). But its **drafter
interface is a strictly leaner version of hipfire's `target_hidden` protocol**,
and that is the load-bearing part for a LAN split.

- **What crosses the drafter interface here (§3.2, eq 5):** the verifier's
  `h_verifier` (last committed token) plus DFlash's `L` lookahead hidden states
  `h_dflash^{1..L}`. That is the *same shape* as hipfire's committed-token
  `target_hidden` block. Weaver immediately compresses each with
  `W_c·RMSNorm(·)` into a `u_i` — i.e. the **first thing the drafter does is
  down-project the hidden state**. This is direct support for hipfire's known
  optimization ("move the `fc` projection to the target so only projected
  `[rows, hidden]` crosses"): Weaver's `W_c` **is** that projection, and it is
  applied before anything else. On a disaggregated split, run `W_c` on the
  target side and ship `u_i` (compressed) rather than raw `h`.
- **The K=512 top-K candidate set is a second, even smaller interface (§3.2).**
  DFlash's contribution downstream is *only* the top-K token IDs and their
  logits per position — 512 ids + 512 logits/position, not a full-vocab
  distribution and not a full hidden state. If the DFlash box and the AR/verify
  box are split, this top-K bundle is the natural wire format: `~512×(id+logit)`
  is ~2–3 KB/position, dwarfed by hipfire's current ~384 KB/cycle. The
  **truncated vocab projection** (read only K rows of `W_vocab`) means whichever
  box owns `W_vocab` only touches 512 rows.
- **Verification protocol is standard, target-side (§4.3):** DFlash-TfM uses
  **Traversal verification** (tree BlockVerify) on the sampling path, because
  each draft token is sampled conditioned on its predecessors. Only token IDs +
  accept/reject come back. This is compatible with hipfire's "token IDs back"
  return path — the tree just makes the forward payload a tree instead of a chain.
- **Caveat for Goal A:** Weaver is autoregressive *over the tree path*, so on a
  disaggregated split the Weaver steps (`⌈B/w⌉` of them, §3.3) each need the
  static KV `u` but then run purely on the draft box — good, they don't re-touch
  the target. But **tree construction depends on DFlash marginals which depend on
  `h_verifier`**, so the target→draft hidden-state hop still gates the round, same
  as hipfire today. The win is that only *one* hidden-state hop per round is
  needed (to seed DFlash+Weaver), after which the whole tree is built draft-side.

**Net for Goal A:** Weaver validates and sharpens the "project on the target,
ship the small thing" idea. The concrete recommendation is to make the wire
format the **compressed conditioning tokens `u` + the top-K candidate bundle**,
not raw `target_hidden`. See §6.

---

## 4. Relevance to Goal B (streaming MoE / accepted-tokens-per-pass)

This is where Weaver hits hipfire hardest. Goal B's throughput ≈
accepted-tokens-per-verify-pass ÷ pass-time, and **Weaver's entire point is to
raise accepted-tokens-per-pass without a heavier drafter.**

- **Trees multiply accepted tokens per pass (§5.1, Table 2):** at tree budget
  64, DFlash-TfM commits **τ ≈ 8.07 tokens/step (296.9 tok/s)** vs the block-16
  chain's **2.67 tokens/step (121.5 tok/s)** — same drafter, the tree is the
  difference. Macro-avg τ up to **9.22–9.69**. The paper explicitly notes a
  longer *chain* would **not** close this gap: chain acceptance length *saturates*
  with draft length (Fig 6), the marginal ceiling. For a streaming/MoE target
  where the verify pass is the expensive streamed thing, more accepted tokens per
  streamed pass is pure throughput.
- **Beating the factorized ceiling (§3.5, Fig 4):** they derive the acceptance
  upper bound of *any* marginal drafter via TV distance (eq 14–16). DFlash-TfM's
  acceptance at later positions is **above** what any marginal-only or
  argmax-marginal drafter can reach. Directly relevant to hipfire's memoized
  observation that SNR is the wrong gate and *acceptance rate* is the right one:
  this paper gives the analytic acceptance ceiling and shows AR conditioning
  beats it.
- **Tree width `w` and budget `B` are the tuning knobs (§3.3):** budgets tested
  {32,64,128,256,512}, expansion width `w`=2–8. For a streaming-bound verify,
  push `B` up — verify cost is amortized over more accepted tokens, and the GDN
  verify is only 12% of the step even at `T=128` (§5.2).
- **Verify economics for linear-attention/GDN targets (§3.4, §5.2):** the
  rollback-free masked-solve keeps verify cheap and, crucially, **shares tree
  metadata across all recurrent layers** and captures the pass in a CUDA graph.
  If hipfire's target uses any linear-attention / delta-rule layers, this is the
  algorithm to reduce verify to a triangular solve rather than per-branch scans.
  (Qwen3.5-397B-A17B is standard softmax-attention MoE, so the *GDN kernel*
  itself may not apply — but the "delay state commit, replay accepted path"
  discipline maps onto any per-request cache state hipfire mutates during verify.)
- **Naive-verify vs spec-sampling crossover (§5.3, Table 3, Fig 6):** for a
  *marginal* drafter (DFlash argmax), **naive verification beats speculative
  sampling** beyond draft length ~2–4; for the *AR* drafter (DFlash-TfM) spec
  sampling wins and shows no crossover. Practical rule: pick the coupling per
  drafter type. Also flagged as future work: **per-position temperature
  annealing** of the marginal to raise long-chain acceptance.

**Net for Goal B:** adopt tree drafting (not longer chains) and tune budget high
when verify is streaming-bound; the AR-residual conditioner is what lets the tree
stay accurate at depth.

---

## 5. Relation to DFlash / block-diffusion / draft trees / MTP

- **DFlash (block-diffusion factorized drafter):** Weaver *is built on DFlash* —
  same `Qwen3.6-27B-DFlash` checkpoint, DFlash frozen, Weaver is "the only
  artifact" trained (§4.1). DFlash provides the top-K marginals + lookahead
  hidden states; Weaver adds the conditional residual. hipfire already has the
  DFlash NPU drafter running (block body ~82/136 ms cached) — **Weaver is the
  natural next layer**, and the paper says the method is drafter-agnostic (works
  on I-DLM too, §3.1 footnote), so it isn't tied to DFlash specifics.
- **Draft trees from block-diffusion marginals (DDTree/BASTION):** DDTree also
  builds trees from the *same* DFlash marginals but *without* the conditional
  residual. The **+32% MAL** gap (§5.1) is exactly the value of Weaver's
  conditioning over a pure-marginal tree. So DDTree = "tree, no conditioning,"
  Weaver = "tree + AR conditioning."
- **MTP:** Gemma4-MTP (§2.1.1) uses the same *top-K-cluster output restriction*
  trick to dodge the full-vocab projection — Weaver's K=512 truncated projection
  is the same family of idea, but conditioned autoregressively rather than a
  parallel MTP head. Weaver's static-KV `u` prompt + AR path is closer to
  EAGLE-3 than to parallel MTP heads.
- **Verification lineage:** naive verification, speculative sampling (eq 3–4),
  BlockVerify, Traversal verification, SpecInfer/Sequoia trees, STree (diagonal
  SSM trees) — Weaver's GDN kernel extends the STree line to **non-diagonal**
  (delta-rule) recurrences (§2.2.5, §3.4).

---

## 6. Reusable implementation specifics + recommendations for hipfire

### Reusable specifics
- **Top-K candidate vocabulary (K=512) + truncated projection.** Read only K
  rows of `W_vocab`; add drafter residual to DFlash logits; renorm over K. 97.8%
  verifier mass in top-512 (§4.1.3). Kills the drafter's dominant bandwidth cost
  — attractive on the Phoenix NPU where MM2S channel bandwidth is the pinned
  limiter (per hipfire's own NPU GEMM findings).
- **Conditioning compression `u_i = W_c·RMSNorm(h) + p_i` (eq 5)** — a single
  small matmul that both feeds Weaver and defines the natural *reduced* wire
  format for a disaggregated split.
- **Batched best-first tree build, width w=2–8, `⌈B/w⌉` steps (§3.3).**
- **Rollback-free verify discipline (§3.4):** never speculatively write cache
  state; delay commit; replay accepted path once. Applies to any mutable
  per-request state, not just GDN.
- **LK loss + argmax-match term + reached-position masking (§4.1.3).** Train
  against a *frozen* drafter; store only token IDs offline and recompute
  logits/lookaheads (§4.1.1).
- **56.7M / 1 layer / dim 2048 / 16 heads** — a Weaver-sized adapter is cheap
  enough to consider running on the NPU alongside DFlash.

### Concrete recommendations

**A. Disaggregated protocol — make the wire format `u` + top-K, not raw hidden.**
Run DFlash's lookahead + the `W_c` conditioning projection **and** the top-K
selection on the *target* box (both are `W_vocab`/`W_c` matmuls that want the big
memory). Ship to the NPU draft box: (1) the compressed conditioning tokens
`u_{0..L}` (hidden→W_c reduced), (2) the top-K candidate ids+logits per position
(~2–3 KB/position at K=512). This is *far* below hipfire's current
~384 KB/cycle `target_hidden`, and it generalizes hipfire's "move `fc` to target"
optimization (Weaver's `W_c` == that `fc`). Weaver's AR tree steps then run
entirely draft-side off the static KV, so only **one** target→draft hop gates a
round. Consider even shipping only DFlash's top-K (the `u` for lookaheads can be
recomputed NPU-side if the NPU already runs DFlash).

**B. Streaming-MoE spec decode — switch chain → tree and push budget.** For
Qwen3.5-397B-A17B where verify is expert/layer-streaming-bound, adopt the
budget-64 (or higher) batched tree instead of a longer serial chain: chain MAL
*saturates* (Fig 6) but the tree lifts committed τ from ~2.7 to ~8 tokens/verify
(§5.1). Throughput ≈ accepted-per-pass ÷ streamed-pass-time, and the streamed
pass-time is ~fixed per verify, so more accepted tokens per streamed pass is a
near-linear throughput win. Tune budget upward until acceptance flattens; verify
overhead stays a small fraction of the streamed forward.

**C. Add the AR residual conditioner (Weaver) on top of the existing DFlash NPU
drafter, and gate on acceptance not SNR.** hipfire already found int4 SNR loss is
the wrong metric and acceptance is the right one; Weaver is the mechanism that
raises acceptance *at depth* past the marginal ceiling (§3.5, Fig 4) for +77% MAL
vs the chain (§5.1) at only 56.7M params and no full-vocab projection. Train it
against the frozen DFlash NPU checkpoint with the LK loss and token-ID-only
rollouts (§4.1). Choose verification coupling per drafter: **naive/argmax for the
bare DFlash marginal path, speculative sampling once Weaver conditioning is in**
(§5.3). If any future target uses linear-attention/delta-rule layers, port the
rollback-free masked-solve verify (§3.4); for the current softmax-attention MoE
target, reuse only the "delay commit / replay accepted path" state discipline.

### Caveats / limits
- Single-request interactivity regime; batched/throughput serving is future work
  (§6) — but the verify kernel already runs on batched state, so the scheduler is
  the gap, not the math.
- Draft depth capped by DFlash's 16-position block (§6); re-anchoring mid-draft
  is unexplored.
- GDN kernel specifics are Triton/CUDA-on-B200 (tf32x3, `B_c=32`, CUDA graphs);
  the *algorithm* ports, the code does not — and only if the target has
  delta-rule layers, which Qwen3.5-397B-A17B does not.
