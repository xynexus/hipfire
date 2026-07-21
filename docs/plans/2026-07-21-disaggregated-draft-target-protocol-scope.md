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

## 9. Streaming-MoE economics & the #1 measurement

Throughput ≈ `accepted_tokens_per_pass ÷ verify_pass_time`, verify_pass_time set by
streaming the activated expert set once. The single measurement that governs the
whole design: **`distinct_experts(B)` — how the activated expert set grows with
tree width B on the 397B (512 experts, 10 active/token).** It bounds the optimal
tree size and is what every H200 paper leaves untested. Measure it before building
the BASTION cost model or committing to a tree width.

## 10. Phased implementation (smallest-first; each verifiable)

1. **Target-side hidden-extraction hook + wire format** — instrument the target
   forward to dump the T accepted-path hidden layers; define the two message
   structs; validate against a *co-located* run (draft and target same box, socket
   loopback) so the protocol is proven before crossing machines.
2. **Cross-machine bring-up (linear block, no tree)** — draft on nix1, target on
   medusa, over LAN; prove losslessness (AR == drafter, ≥3 repeats) and measure the
   real per-cycle wire volume + RTT. This is the disaggregation proof.
3. **`distinct_experts(B)` measurement** on the 397B (§9) — the go/no-go for tree
   width.
4. **Tree verify on the target** — masked-SDPA kernel + DDTree tree build on the
   draft box; widen per §3.
5. **Conditional corrector (DSpark Markov head)** + confidence pruning.
6. **BASTION budget controller** (MoE-aware) + optional async pipelining.

## 11. Risks & open questions

- **`distinct_experts(B)` is unmeasured and load-bearing** — if the expert set
  grows fast, wide trees stream most of the 397B per cycle and the tree's advantage
  shrinks. §9 measurement gates everything downstream.
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
