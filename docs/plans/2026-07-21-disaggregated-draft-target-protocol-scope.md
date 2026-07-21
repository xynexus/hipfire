# Disaggregated DFlash draft/target protocol — scope

Scopes running the DFlash spec-decode drafter on one machine (the Phoenix NPU box,
nix1) and the target model on another (a big-memory GPU box), talking over the
fleet LAN. Motivated by Qwen3.5-397B-A17B: the target (~200 GB int4) fits no
NPU-bearing box, so if we want the NPU drafter at all, draft and target must be
disaggregated.

Grounded in seven SOTA references read into `docs/references/specdecode/`
(dflash, ddtree, weaver, bastion, dspark, deepspec, dflare). Citations below are
to those notes; the notes cite the papers/code.

## 0. Why this is feasible and worth doing

Two facts, both now corroborated by the literature:

1. **The draft↔target interface is tiny and one-directional.** The drafter
   consumes `target_hidden` (interior target-layer hidden states), NOT logits.
   DeepSpec states the contract explicitly — *ship hidden states (O(d)), not
   logits (O(V)); keep the frozen `lm_head`/`embed` on the draft side*
   ([[deepspec-notes]]). Volume is `n_committed × T × H × 2` bytes/cycle (bf16),
   **feed back only the accepted path** ([[ddtree-notes]]) — so a tree does not
   inflate it.

2. **The verify is streaming-bound and dominates by 10–100×**, so the draft (and
   the LAN round-trip) is free by ratio. This is not a hipfire hypothesis
   anymore: DeepSpec formalizes throughput as `Θ = τ · SPS(B)` with `SPS`
   profiled once (flat region then cliff) and notes the effective batch "stays
   below the compute-saturating threshold"; Weaver shows trees lift committed τ
   from ~2.67 to ~8+ on the same drafter — near-linear throughput when verify is
   the bottleneck ([[weaver-notes]], [[deepspec-notes]]).

Corollary that reframes the effort: DFlare found draft-depth gains "cancel" on a
shared device — but that is a *same-device serialization artifact*. Disaggregated,
draft latency hides behind verify, so **we can push drafter depth/capacity/heads
harder than any of these papers' co-located numbers suggest** ([[dflare-notes]]).
Disaggregation doesn't just make the 397B runnable; it unlocks drafter quality the
baselines leave on the table.

## 1. Topology & placement

```
  DRAFT BOX (nix1, Phoenix APU: XDNA NPU + iGPU, 64 GB UMA)
    - DFlash/DSpark NPU body (1.29B drafter for the 397B target: h4096, 6L, T=8)
    - frozen embed_tokens + lm_head (shared with target; ~2 GB bf16 each)
    - tree builder (CPU best-first heap; DDTree/BASTION algo)
    - trimmable draft attention KV cache over committed tokens
    - [optional] Weaver AR adapter (56.7M) or DSpark Markov+confidence heads
            │  draft→target: candidate token ids + tree (parents) + draft top-K   (~5-10 KB)
            │  target→draft: accepted token ids + bonus + accepted-path hidden     (~40-430 KB, see §3-4)
            ▼
  TARGET BOX (medusa 384 GB MI50 cluster — holds 397B int4 resident;
              or halo 128 GB — needs the expert-streaming path)
    - Qwen3.5-397B-A17B target (streamed or resident experts)
    - hidden-extraction hook: dump the T target layers for the accepted path
    - [optional] fc/W_c compression of those hidden states (§4)
    - tree-verify: masked SDPA (FlashAttention cannot do the tree mask) + MoE
    - streaming / microbatch engine (the DFlash block IS a microbatch)
```

Placement rules from the references:
- **Frozen `embed_tokens` + `lm_head` live on the DRAFT box** — the drafter needs
  `lm_head` to turn its hidden into draft logits/marginals for the tree, and
  `embed` to embed the block. Both frozen, identical to the target's
  ([[dflash-notes]], [[deepspec-notes]]). The target keeps its own copy (part of
  the 397B).
- **The tree is built entirely on the draft box** from DFlash's per-position
  marginals — cheap CPU heap work, no target involvement ([[ddtree-notes]],
  [[bastion-notes]]).
- **The target only ever runs one streamed forward per cycle** and returns the
  accepted-path hidden.

## 2. Per-cycle data flow (the dependency that shapes everything)

The hard dependency (established in the overlap-seam scope,
`2026-07-20-dflash-overlap-seam-scope.md`, and re-confirmed here): **draft N+1
needs `target_hidden` for block N's committed tokens, which only the verify of
block N produces.** It is a numerical dependency, so the loop is serial:

1. **Draft box** builds a tree from the drafter's marginals (optionally corrected,
   §8) and sends `{candidate_token_ids, parents, [draft_topK/probs]}`.
2. **Target box** rebuilds the ancestor-only mask + position ids from `parents`,
   runs **one** streamed verify forward over the whole tree (all `B+1` nodes),
   greedily walks the tree to find the accepted path + bonus token, extracts the
   T target-layer hidden states **for the accepted path only**, (optionally
   compresses them, §4), and sends back `{accepted_token_ids, bonus_token,
   accepted_hidden}`.
3. **Draft box** appends `accepted_hidden` to the trimmable draft KV context,
   trims rejected/speculative positions, drafts N+1.

Because verify dominates, this serial loop is near-optimal — **the DFlash overlap
seam (draft N+1 during verify N) is unnecessary here** and was measured a NO-GO
for the small-target case anyway (seed-oracle 2%). For a streaming 397B, draft
(~98 ms) ÷ verify (seconds) ≈ a few % overhead; no overlap trick needed.

## 3. Wire protocol & volume

| direction | payload | volume/cycle |
|---|---|---|
| draft→target | candidate token ids + `parents` (tree topology) + optional draft top-K | **~5–10 KB** (do NOT ship the dense B×B mask — 256 KB at B=512; rebuild it target-side, [[ddtree-notes]]) |
| target→draft | accepted token ids + bonus + **accepted-path** hidden (`n_commit × T × H × 2`) | **~40 KB/token uncompressed** → ~240 KB (T=5) / ~384 KB (T=8) / ~432 KB (T=9) per cycle |

**The target→draft figure is T-dependent, not fixed.** DFlash reads T≈5–8 layers,
DFlare T=9 ([[dflare-notes]]). "~384 KB/cycle" is the DFlash-T=8 case for the 397B
drafter (num_extract=8, `target_layer_ids [1,9,…,57]`).

Throughput impact: at a streaming cadence of ~0.2–2 s/cycle, even the uncompressed
432 KB is **< 2.2 MB/s** — ~1.5% of 1 GbE, negligible on the fleet LAN. Latency:
LAN RTT (~0.1–0.5 ms) is < 1% of a verify measured in hundreds of ms to seconds.
**The interface is nowhere near a bottleneck; the design is verify-bound, as it
should be.**

## 4. The compress-on-target decision (fc / W_c / layer-fusion)

Three papers independently point at the same optimization: the projection that
turns `T×H` raw hidden into the drafter's `H`-wide conditioning is a *static
linear map*, so run it on the **target** and ship the compressed result.

- DFlash's `fc: Linear(T·H → H)` lives inside the draft; moved to the target it
  ships `[rows, H]` = **~48 KB/cycle instead of 384 KB — an 8× cut**
  ([[dflash-notes]]).
- Weaver compresses each hidden with `W_c·RMSNorm(·)` (its eq 5) — literally the
  same move, and its downstream tree needs only DFlash's **top-K=512 ids+logits**
  (~2–3 KB/position), not the full hidden ([[weaver-notes]]).
- DFlare pre-fuses T→D on the GPU box (D=7 < T=9 → 56 KB, 22% cut)
  ([[dflare-notes]]).

**Tradeoff:** compress-on-target *version-couples* the draft and target (the target
must hold the drafter's projection weights and re-ship on any drafter retrain).
Given bandwidth is a non-issue, **the default is NOT to compress** — keep the draft
self-contained and ship raw hidden. Revisit compression only if (a) many drafters
fan out from one target, or (b) a slower link than the LAN is used. Record it as an
available 8× lever, not a day-1 requirement.

## 5. Tree verification on the target (the real target-side work)

- **Masked SDPA, not FlashAttention.** The ancestor-only tree mask is not
  expressible in FlashAttention-2 ([[ddtree-notes]], [[bastion-notes]]); the target
  needs a masked-SDPA verify path (an RDNA tree-mask attention kernel on medusa's
  gfx906 / gfx1201, or halo gfx1151). This is the main new target-side kernel.
- **MoE + tree interaction is the #1 unknown.** The tree's `B` tokens touch a set
  of experts that grows with `B` (`distinct_experts(B)`), then saturates. Under
  streaming, per-cycle verify cost ≈ streaming that expert set once. So the optimal
  tree width is bounded **not by compute (as in the H200 papers, whose budget
  optimum of 256–512 is a resident-weight artifact) but by MoE expert-set growth**
  ([[ddtree-notes]], [[bastion-notes]]). BASTION's roofline cost model is explicitly
  dense-only and MoE-blind — the calibration skeleton (`α·raw+β`) is reusable but
  the FLOP/byte formulas must be rewritten so verify-memory reflects only the
  experts the tree actually activates.
- The DFlash block is already a microbatch; DDTree/Weaver widen it into a tree that
  one streamed pass verifies. Delay-commit / replay-accepted-path discipline
  (Weaver §3.4) is the losslessness-preserving pattern; the GDN-kernel part of
  Weaver's rollback-free verify does not apply (397B is softmax-attention MoE).

## 6. Latency & pipelining

Verify-dominated, so the serial loop is fine. Two ready-made mechanisms if we ever
want to hide the (small) draft+LAN cost:
- DeepSpec's **async "two-steps-prior" scheduler** — pipeline draft(t+1) under
  verify(t) — is the template ([[deepspec-notes]]).
- BASTION's budget controller has an empty `non_verify_latency_s` slot to drop LAN
  RTT into, so it grows trees to amortize the round-trip ([[bastion-notes]]).

Neither is needed for a first cut; both are cheap CPU-side additions later.

## 7. Losslessness

Unchanged by disaggregation: the **target verifies every tree path**, so the split
is a transport detail. Greedy tree-match (DDTree/Weaver) commits only the target's
own argmax path; the bonus token is the target's correction at the first mismatch.
Gate exactly as the spec-decode path is gated (drafter-independence: AR == every
drafter, ≥3 repeats — the digest is now target-specific, `a099a2729d04…` for the
rebuilt 9B mq4, per the brief). The network moves tokens + hidden; it cannot change
what the target accepts.

## 8. Drafter composition — what to actually run (the "DDTree × DFlash × DSpark" question)

The references show these attack **orthogonal axes** and largely compose:

| axis | mechanism | source | on-NPU today? |
|---|---|---|---|
| base marginals | DFlash block-diffusion body | dflash | yes (bring-up done) |
| drafter capacity | deeper draft, adaptive layer fusion, more training data | DFlare | partial |
| **conditional dependency** (the ceiling) | DSpark Markov head **or** Weaver AR adapter | dspark / weaver | DSpark: Phase E infra exists; Weaver: not built |
| verified width | best-first tree from marginals | DDTree / BASTION / Weaver | `spec_step_ddtree_*` exists |
| budget / pruning | BASTION controller + DSpark confidence head | bastion / dspark | no |

**The one real decision: which conditional-dependency mechanism.** Weaver and
DSpark's Markov head both attack the same independence-assumption ceiling and are
likely *redundant*, not additive:
- **Weaver AR adapter** — 56.7M params over top-K=512, tree-native, **+32% MAL vs
  DDTree** ([[weaver-notes]]). Stronger, but needs training and is a new component.
- **DSpark Markov head** — a serial per-slot loop; hipfire has the Phase E infra
  (`dspark_body.rs`, `dspark_heads_npu.py`) and the on-disk
  `DeepSeek-V4-Flash-DSpark` checkpoint *is* this drafter ([[dspark-notes]]). The
  serial chain is a cost on-device, but **free-by-ratio in the disaggregated
  streaming setup**.

Recommended stack for the streaming 397B, in build order:
1. **DFlash body + DDTree tree** (both largely exist) — widen the tree aggressively
   since expert-set growth, not compute, is the bound.
2. **Add ONE conditional corrector** — start with the **DSpark Markov head**
   (infra + checkpoint exist), measure the tree-lift; evaluate the **Weaver
   adapter** as the higher-ceiling alternative if the Markov head underdelivers.
3. **DSpark confidence head** for prune/verify-depth scheduling — the primary knob
   for a verify-bound target ([[dspark-notes]]).
4. **BASTION budget controller** with an MoE-aware cost model — sizes the tree per
   cycle and absorbs LAN RTT.
5. **DFlare capacity/training upgrades** last — biggest τ lever is training data
   (270K→2.4M ≈ +2.4 τ) + early-position loss, which directly attacks our measured
   30%-diverge-≤1 bimodality ([[dflare-notes]]).

Do **not** run the Qwen3.5 target's own MTP head *and* a block drafter as one
proposer — DSpark replaces MTP, doesn't stack on it ([[deepspec-notes]],
[[dspark-notes]]).

## 9. Streaming-MoE economics & the #1 measurement — MEASURED (task #40)

Throughput ≈ `accepted_tokens_per_pass ÷ verify_pass_time`, verify_pass_time set by
streaming the activated expert set once. The single measurement that governs the
whole design: **`distinct_experts(B)` — how the activated expert set grows with
tree width B.**

**Measured (task #40, `86d4703cd`, `benchmarks/results/distinct-experts-vs-tree-width-20260721.md`)** — exact router-argtopk capture on a Wikipedia corpus, decomposed into DEPTH (B consecutive positions) vs WIDTH (B sibling next-token candidates at one prefix), on LFM2.5-8B-A1B (E=32,k=4, nix1) and **Qwen3.6-35B-A3B (E=256,k=8 — same `Qwen3_5Moe` family as the 397B)**:

| B | LFM depth/width | 35B depth/width (%E) |
|---|---|---|
| 8 | 41% / 34% | 13% / 11% |
| 32 | 67% / 53% | 31% / 22% |
| 64 | 75% / 61% | 43% / 29% |
| 256 | 86% / 73% | 65% / 43% |

Ordering at every B: **width < depth < random-empirical < analytic-uniform.**

**Verdict: WIDE-OVER-DEEP, and width is materially cheaper — but not free.** Same-prefix branches route alike, so width adds experts at only **~0.6–0.7× the rate of an extra depth position** (width/depth ratio 0.87 at E=32 → 0.66 at E=256 — the advantage *grows* with the expert pool). The strong "width ≈ depth-bounded / free" hypothesis is **refuted**: width still grows with B, just sub-linearly and slower than depth.

**397B implication (extrapolated via the 35B discounts — conservative):** at B=64, width touches ~24% of the 512 experts, depth ~35%; at B=128, ~34% / ~51%. **Free tree-width budget ≈ up to ~64 nodes** touches only ¼–⅓ of the pool per verify — so the §11 "wide trees stream most of the 397B" risk is **NOT realized**. Prefer bushy (wide, shallow) trees.

**Consequences:** the BASTION cost model must price width at a real **~0.6–0.7× depth marginal rate (not zero)**, plus the always-on shared expert and the diffuse-layer tail (per-layer max ≫ mean). Residuals: the 35B run was **first-20-of-40 layers** (halo APU memory; aggregate is first-half, flagged), medusa was unreachable so the **real 397B run is unclosed**, and the full-stack DFlash→DDTree→verify tree number awaits the masked-SDPA verify path (§5).

## 9.5 The drafter ↔ tree ↔ step-interval co-design (distinct_experts is ONE input, not the answer)

`distinct_experts(B)` bounds the verify side, but the drafter and the tree are not
independent knobs — they close a loop, and the drafter should be **sized to the
step interval**, not fixed. The coupled quantities:

- **Step interval `T`** = verify pass time = `f(B)` via `distinct_experts(B)` +
  tree attention. This is the clock.
- **Draft budget = `T`.** Disaggregation hides the draft under the verify, so the
  drafter + tree-build get the *whole* interval to run in.
- **τ (accepted/cycle)** = `h(drafter quality, tree shape)`, and the **tree shape
  is a function of the drafter's per-position marginals** — sharper marginals →
  narrower tree for the same τ.

The loop: `drafter → marginals → tree shape → B → distinct_experts(B) → T → draft
budget → drafter size → …`. Solve for the fixed point that maximizes `τ / T`.

**Two consequences the fixed-drafter framing (and every co-located paper) misses:**

1. **The 1.29 B drafter is under-sized for the streaming regime.** If the
   streaming-397B verify is ~1–2 s and the drafter runs in ~98 ms, the NPU is
   ~95% idle per step. That budget should buy a bigger drafter, the Weaver/DSpark
   conditional adapter, a **larger `block_size`** (commit >16 tokens/pass), and a
   deeper tree search — sized to *fill* `T`. **Drafter capacity is an output pinned
   to the sustainable interval, not a fixed input** (DFlare's "push harder",
   generalized).
2. **Drafter investment pays twice.** A better drafter → sharper marginals →
   narrower tree for the same τ → fewer distinct experts → faster verify → shorter
   `T`. Quality raises τ AND cheapens verify. `distinct_experts(B)` is the cost
   curve; its operating point moves when the drafter improves.

Structurally a **two-level optimization**: OUTER loop sizes the drafter (capacity,
block_size, adapter, tree-search depth) to `T`; INNER per-cycle loop is BASTION's
adaptive controller re-solving the tree shape from the current marginals under the
`distinct_experts` cost model. BASTION does the inner loop but assumes a fixed
drafter — we own the outer loop.

**`T` is WEIGHT-BANDWIDTH-DERIVED, not a live measurement.** Same principle as the
NPU drafter (time linear in weight bytes). For a streamed target the compute is
irrelevant; the wall is set by the weight bytes that stream per pass ÷ the target's
weight bandwidth (halo ≈ **10 GB/s from NVMe**):

```
T(B) = [ dense_bytes + streamed_expert_bytes(B) ] / BW
```

For the 397B at int4: expert = 3·[4096×1024] = 12.6 M params → 6.3 MB; ×512×60
layers = **193 GB of experts**; dense ≈ 5.5 GB. **Cold cache** (every activated
expert streams): `bytes(B) = 5.5 + φ(B)·193 GB` with φ = the measured
distinct-experts fraction → T ≈ 3.4 s (B≈16, φ~15%) … 8.3 s (B=128, φ~40%).

**⚠ Cold, wider trees LOSE.** Streamed bytes grow with `φ(B)·193 GB` while τ
saturates, so `τ(B)/T(B)` gets *worse* with width — the opposite of §9's
"wide-over-deep". On a pure NVMe stream, narrow beats wide.

**What flips it is the resident expert cache.** halo's 128 GB holds ~57% of the
193 GB pool, so per pass you stream only the **cache MISSES**, not all
`distinct_experts(B)`. Here §9's sub-linearity pays off: a wider tree adds few
*new* distinct experts, and if they are mostly resident the marginal traffic is
tiny. Therefore:

- **The objective is `τ / cache-MISS-bytes(B)`, not `τ / distinct_experts(B)`.**
  `distinct_experts` is the demand; the cache converts demand → traffic.
- **The decisive quantity is expert working-set residency / miss rate** — the
  real follow-up measurement (supersedes "measure T on medusa"; T is now derived).
- **Weight bytes are the only lever** (same verdict as the NPU path): lower-bit
  experts (oq4/oq3) → more of the pool fits in 128 GB → fewer misses → wider trees
  affordable. Quant sets how wide the tree can be, because it sets cache residency.

Deployment still picks the regime: **medusa resident** (384 GB > 200 GB) → ~0
expert streaming, near-fully-cached, so tree width is bounded by attention/compute
(the H200 regime); **halo** → NVMe-streaming, and the cache/miss model above governs.

## 9.6 Microbatching per expert + batch size — the utilization axis the bandwidth model hides

Efficient streamed MoE executes **per expert module** (gather-scatter): load an
expert's weights once, run every token — across the tree AND across concurrent
requests — that routes to it. That is what makes the weight cost
`distinct_experts(B)·expert_bytes` (each distinct expert streamed once), compute
amortized. But it adds a second axis the pure `T = bytes/BW` model omits:
**tokens-per-expert**, i.e. is each streamed expert actually doing work.

**Crossover (halo, int4, ~50 TFLOP/s):** stream one expert = 6.3 MB / 10 GB/s =
**0.63 ms**; compute `n` tokens through it = `n·2·12.6 M / 50 TFLOP/s` = `n·0.5 µs`.
Crossover `n* ≈ 1260 tokens/expert`. Below it → memory-bound (wait to load an
expert, compute it in `<<` the load time); above → compute-bound.

**A single tree is ~300× below crossover.** B=64, k=10, ~154 distinct experts →
`640/154 ≈ 4` tokens/expert → `2 µs` work vs a `0.63 ms` load → **~0.3% expert
utilization.** A single-user streaming tree streams a 6.3 MB expert to do 4 tokens
of work — this is *why* cold-cache wide trees looked bad in §9.5: you pay to move
weights you barely use.

**Increasing batch (tree width + concurrent request streams, grouped per expert):**
1. `distinct_experts` **saturates toward E** — enough tokens hit every expert, so
   you stream the whole 193 GB pool once per pass and it stops growing.
2. **tokens-per-expert rises** — each additional token rides an already-streamed
   expert nearly free until `n* ≈ 1260`.
3. So **aggregate throughput rises with batch** (`tok/s → batch·BW/193 GB` once
   saturated, linear in batch) while **per-request latency `T` rises**. The classic
   throughput↔latency serving curve.

**Consequences (architectural):**
- **A single-user streaming MoE is bandwidth-wasteful regardless of spec-decode**
  (~0.3% expert utilization). The streaming target MUST be a **multi-tenant
  batching server** (continuous batching + expert-grouped execution) driving
  tokens-per-expert up — not a single-stream engine.
- **Tree width is the WRONG utilization knob** (buys τ, barely moves
  tokens-per-expert). **Concurrency is the utilization knob** — and concurrent
  trees *overlap* in expert usage (union sub-linear across users too), so N users
  cost `≪ N×` the experts; each streamed expert just serves ~N× more tokens. That
  overlap is what makes streaming viable.
- **Software-pipeline the stream** (load expert e+1 while computing expert e — the
  `--pipeline-glue` trick one level up) hides compute under the stream, but only
  *up to* crossover; below it, compute is too short to hide anything.

**Deployment split sharpens:** medusa-resident → no streaming, compute-bound,
single-user fine, crossover never bites. halo-streaming → efficient ONLY at high
concurrency; the target is a batching server or the NVMe is wasted.

## 10. Phased implementation (smallest-first; each verifiable)

1. **Target-side hidden-extraction hook + wire format** — instrument the target
   forward to dump the T accepted-path hidden layers; define the two message
   structs; validate against a *co-located* run (draft and target same box, socket
   loopback) so the protocol is proven before crossing machines.
2. **Cross-machine bring-up (linear block, no tree)** — draft on nix1, target on
   medusa, over LAN; prove losslessness (AR == drafter, ≥3 repeats) and measure the
   real per-cycle wire volume + RTT. This is the disaggregation proof.
3. **`distinct_experts(B)`** (§9, DONE on proxy) **+ the step interval `T`** (§9.5)
   — measure the actual streaming-397B verify wall per deployment (halo-streaming
   vs medusa-resident). `T` is the go/no-go for tree width AND the budget that sizes
   the drafter; both feed the co-design fixed point.
4. **Tree verify on the target** — masked-SDPA kernel + DDTree tree build on the
   draft box; widen per §3.
5. **Conditional corrector (DSpark Markov head)** + confidence pruning.
6. **BASTION budget controller** (MoE-aware) + optional async pipelining.

## 11. Risks & open questions

- ~~**`distinct_experts(B)` is unmeasured and load-bearing.**~~ **MEASURED (task
  #40) — risk NOT realized.** Wide-over-deep: ~64-node trees touch only ¼–⅓ of the
  512-expert pool (§9). Residual: extrapolated from the 35B proxy (first-20-of-40
  layers) — the real 397B run on medusa is still unclosed (medusa was unreachable).
- **Masked-SDPA tree verify on the target GPU** is a new kernel on gfx906
  (medusa) / gfx1151 (halo); FlashAttention can't do the mask.
- **BASTION cost model is dense-only** — must be rewritten for MoE streaming, or the
  budget controller will mis-size trees.
- **Weaver vs DSpark redundancy** — running both conditional correctors likely
  wastes effort; pick one (§8).
- **Target-side GDN rewind is NOT a draft concern** but IS a 397B concern: the
  hybrid target has linear-attention/GatedDeltaNet layers whose recurrent state
  needs correct prefix-replay on partial accept ([[dflash-notes]], [[dspark-notes]]).
  Independent of the protocol, but on the critical correctness path for this target.
- **Version coupling** if compress-on-target is adopted (§4) — default off.
- **medusa is gfx906 (old)** — the target verify runs there; confirm the streamed
  MoE + masked-SDPA path is viable on GCN5, or use halo/W7800 (gfx1201) with
  streaming.

## Reference notes
`docs/references/specdecode/{dflash,ddtree,weaver,bastion,dspark,deepspec,dflare}-notes.md`
