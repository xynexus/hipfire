# DDTree — Diffusion Draft Tree: distilled notes for hipfire

**Paper:** "Accelerating Speculative Decoding with Block Diffusion Draft Trees",
Liran Ringel & Yaniv Romano (Technion), arXiv:2604.12989.
**Code:** `github.com/liranringel/ddtree`, read locally at
`/srv/hipfire/references/SpecDecode/ddtree/{tex,src}`.
Reference commit is a HF-Transformers PyTorch/CUDA implementation on top of the
public DFlash checkpoints (Qwen3-4B / 8B / Coder-30B-A3B).

**One-line thesis:** vanilla DFlash throws away almost everything a block
diffusion forward pass produces — it collapses `L` per-position *marginal*
distributions into a single trajectory. DDTree keeps them: it builds the
provably-optimal draft *tree* (under the drafter's own factorized surrogate)
from that **single** drafter pass, verifies the whole tree in **one** target
forward with an ancestor-only mask, and lifts mean accept length `τ` by ~1.35–1.5×
with zero extra drafter passes and no change to output distribution.

This is the exact method hipfire wants to layer on DFlash. The important thing
for us: DDTree's structure maps cleanly onto both the disaggregation interface
(Goal A) and the streaming-MoE economics (Goal B), and in the streaming-bound
regime DDTree is *more* favorable than the paper's H200 numbers suggest. Details
and corrections below.

---

## 1. Tree-construction algorithm

### 1.1 Why a tree is even possible from one pass (the key structural fact)

A block diffusion drafter takes `[b, m, …, m]` (bonus token `b` + `L` masked
positions) and emits, in **one** forward pass, `L` logit vectors
`ℓ_i ∈ ℝ^|V|` → per-position distributions `q_i(v) = softmax(ℓ_i)_v`
(main.tex:153–160). Crucially each `q_i` is a **marginal**, *not*
path-conditioned: position `i` conditions on context `(c,b)` and DFlash's target
features, but **not** on the token choices at positions `1..i-1` inside the same
block (main.tex:161, 170–177). So the natural distribution of the drafter is the
factorized product

```
Q(y_1:L | c,b) = ∏_{i=1..L} q_i(y_i | c,b)        (main.tex:171–176, eq. factorized-approx)
```

This is *why* one drafter pass suffices to score an entire tree: the prefix mass
of any node is just a product of already-computed marginals. Autoregressive tree
drafters (EAGLE-2, OPT-Tree) need one drafter pass *per tree depth*; DFlash needs
one pass total (main.tex:140, 272). That is the whole advantage DDTree inherits.

### 1.2 Surrogate scoring function

The ideal objective — expected accept length under the *target* — needs the
target's path-conditioned factors `p(y_i | c,b,y_{1:i-1})`, which don't exist at
draft time (main.tex:224–228). DDTree optimizes the surrogate: expected accept
length under `Q`. Prop. 1 (main.tex:246–252) shows this decomposes additively:

```
E_{Y~Q}[ α_T(Y) ] = Σ_{u ∈ T} q(u | c,b),   where   q(u|c,b) = ∏_{i=1..|u|} q_i(u_i)   (eq. prefix-mass, main.tex:238–241)
```

`α_T(y)` = length of the longest prefix of `y` present in tree `T`
(main.tex:220). So **the score of a node is the product of the marginal
probabilities along its root-to-node path** — i.e. its prefix mass under `Q`.
Maximizing the surrogate under budget `B` = "pick the `B` highest-prefix-mass
nodes", and Prop. 2 (main.tex:259–270) proves the top-`B` prefixes are
automatically prefix-closed (every ancestor strictly dominates its descendants
because factors are `<1`), hence a *valid* tree. This is exactly OPT-Tree's
objective (Wang 2025) transplanted to the one-pass block-diffusion setting
(main.tex:272).

Work in **log space**: `σ(ρ) = Σ_i log q_i^{(ρ_i)}` (main.tex:302–306). Ranking by
`σ` preserves order and is numerically stable.

### 1.3 Best-first heap construction (the actual algorithm)

Lemma 1 (main.tex:316–319) restricts the search to the **top-K tokens per
depth**, `K = min(B, |V|)` — no node in some optimum ever uses a token ranked
worse than `B` at its depth. So you `topk(logits, K)` once per position and then
run a max-heap enumeration over rank-tuples. Paper Algorithm 1
(main.tex:324–343); code `build_ddtree_tree` in `src/ddtree.py:84–166`:

```
# indices ρ = (ρ_1,…,ρ_d) are per-depth token RANKS, not vocab ids
H = max-heap seeded with rank-tuple (1) at depth 1, score = log q_1^(1)
T = ∅
while |T| < B and H nonempty:
    pop ρ = (ρ_1,…,ρ_d) with largest σ(ρ)          # highest prefix mass remaining
    add prefix (v_1^{ρ_1},…,v_d^{ρ_d}) to T
    if ρ_d + 1 ≤ K:                                # next SIBLING: swap last token for next-best at same depth
        push (ρ_1,…,ρ_{d-1}, ρ_d+1),  score = σ(ρ) − log q_d^{(ρ_d)} + log q_d^{(ρ_d+1)}
    if d < L:                                      # first CHILD: extend one depth with that depth's best token
        push (ρ_1,…,ρ_d, 1),          score = σ(ρ) + log q_{d+1}^{(1)}
return T
```

Each pop spawns ≤2 pushes (sibling + child); heap stays `O(B)`; total cost
`O(B log B)` (main.tex:351–353). Prop. 3 (main.tex:345–348) proves the pop order
is non-increasing in prefix mass, so after `B` pops you have exactly the optimal
top-`B` tree. `L` here is `block_size − 1` = `draft_horizon` (ddtree.py:307): the
root `b` is fixed and free, budget counts only the drafted positions.

**Code specifics worth stealing (ddtree.py:116–166):**
- Heap tuple is `(-logw, ranks, parent_index, depth, rank, logw)` — negated log
  weight for a min-heap-as-max-heap; carries `parent_index` so the tree is wired
  up on pop, not reconstructed later.
- Outputs, all indexed with node 0 = root `b`:
  `node_token_ids[node_count]`, `node_depths[node_count]`,
  `parents[node_count+1]` (`parents[0]=-1`), and `child_maps` = per-node
  `dict{token_id → child_node_index}` used at verify-walk time.
- **Visibility (ancestor mask) is built incrementally from `parents` in
  `O(B·B)` bool ops** (ddtree.py:151–159): `vis[i,:i] = vis[parent(i),:i]`,
  `vis[i,i]=True`. Row `i` inherits its parent's row and adds itself → each node
  "sees" exactly root + ancestors + self. This is the whole ancestor-only mask,
  computed on CPU with numpy.
- `topk = min(budget, vocab)`; logits promoted to fp32; `log_z = logsumexp`
  gives exact log-softmax; top-`K` log-probs and ids are pulled **to CPU** and
  the heap loop is pure-Python/numpy scalars (ddtree.py:102–130). On the H200
  path this CPU heap is measured as a distinct `tree_build_heap` stage — a real,
  non-trivial serial cost (see §3 and §6).

---

## 2. Single-pass tree verification

### 2.1 Compile (ddtree.py:169–209, `compile_ddtree_tree`)

The tree is flattened to a token sequence rooted at `b`:
- `verify_input_ids = [b, node_token_ids…]`, length `1 + node_count`
  (ddtree.py:190–193).
- **Position ids by depth**: `pos[0] = start`, `pos[node] = start + node_depth`
  (ddtree.py:195–199). All siblings at a depth share a position id, so the target
  applies the correct RoPE per tree level (main.tex:360).
- **Attention mask** = big additive mask over `[past_context ‖ tree]`
  (ddtree.py:204–208): the `current×current` tree block is filled with
  `finfo.min` then `masked_fill_(visibility, 0)` — i.e. within the tree a node
  attends only to root+ancestors+self; every node attends fully to the past KV
  cache. This is SpecInfer-style tree attention (Miao 2023; main.tex:360).
- Buffers (`attention_mask_buffer`, etc.) are **pre-allocated once** at
  `max_tree_nodes = 1 + budget` and sliced per round; the previous round's tree
  block is zeroed out before reuse (ddtree.py:187–188). Portable, allocation-free
  steady state.

### 2.2 One target forward + accepted-path walk

One target call over `verify_input_ids` with the tree mask and `past_key_values`
(ddtree.py:409–416) produces logits and hidden for **every** tree node
simultaneously. Then `follow_verified_tree` (ddtree.py:212–223) walks it:

```
posterior = target's own decode(logits)      # argmax if temp 0, else sample()  (ddtree.py:420, utils.sample)
idx = 0 (root); next = posterior[root]
while next ∈ child_maps[idx]:                 # does target's chosen token match a tree child?
    idx = child_maps[idx][next]; accept idx; next = posterior[idx]
# stop at first mismatch; `next` becomes the bonus token for the next round
```

Because the single forward already scored all nodes, following the accepted path
needs **no extra target call** (main.tex:362). Accepted tokens are gathered by
index, the first unmatched target token is carried as next `b`
(ddtree.py:421–426).

### 2.3 Losslessness

Verification follows the **target's own decoding rule** (main.tex:361–362,
ddtree.py:420). At every followed position the emitted token is a genuine target
draw (argmax at temp 0; `torch.multinomial` at temp>0), and the divergence token
is likewise the target's own draw. So the output stream is distributionally
identical to plain target decoding — same guarantee as vanilla DFlash.
**Correction/nuance for us:** this is *greedy-match* tree verification (accept iff
token equals a tree child), **not** SpecInfer/Medusa residual-multinomial
rejection sampling over the tree. It is exactly lossless at temp 0. At temp>0 it
is lossless in the "emit the target's sample" sense (one trajectory is followed),
but it does **not** recover the multi-branch acceptance probability that true
rejection sampling gives — a hotter tree could in principle accept more with a
rejection-style rule. hipfire's existing lossless serial DFlash verify uses the
same greedy-match logic, so this is a drop-in generalization, not a new
correctness regime.

---

## 3. Accepted-tokens-per-pass economics (Goal B)

This is the section that matters most for streaming MoE. Restating hipfire's
model: verify is streaming-bound, so
`throughput ≈ accepted_tokens_per_verify_pass / pass_time`, and `pass_time` is
dominated by weight/expert streaming, nearly independent of how many tree tokens
ride along. DDTree's entire job is to raise the numerator.

### 3.1 Measured lift (main.tex Table 1, temp 0.0, best budget per cell)

`τ` = mean accept length **including the bonus token**. DFlash → DFlash+DDTree:

| Dataset (Qwen3-8B) | DFlash τ | DDTree τ | Δ | speedup |
|---|---|---|---|---|
| MATH-500 | 7.79 | **10.73** | +38% | 5.56× → 7.52× |
| AIME-2024 | 7.46 | **10.42** | +40% | 5.38× → 7.35× |
| GSM8K | 6.57 | **9.54** | +45% | 4.78× → 6.75× |
| HumanEval | 6.61 | **9.67** | +46% | 4.84× → 6.90× |
| Alpaca | 3.12 | **5.09** | +63% | 2.07× → 3.36× |
| MT-Bench | 4.28 | **6.58** | +54% | 2.56× → 4.10× |
| SWE-bench Lite | 3.60 | **5.91** | +64% | 2.65× → 4.23× |

DDTree improves **all 60** dataset×model×temp cells (main.tex:418). Gains are
*largest, in relative terms, where vanilla `τ` is lowest* (Alpaca, SWE-bench,
MT-Bench: +54–64%) and where temperature is higher (temp-1.0 table, main.tex:400–410,
relative lifts are bigger than temp-0.0). Coder-30B-A3B (an **MoE** target, the
closest analogue to our Qwen3.5-397B-A17B) shows the same pattern:
MATH-500 5.58→8.10 τ, 4.29×→6.21×.

### 3.2 Budget↔accept-length tradeoff (main.tex:420–429, Fig. budget-tradeoff)

Case study MATH-500 / Qwen3-8B / temp 0: as budget `B` grows, `τ` rises
**monotonically**, but end-to-end speedup peaks at **B≈256–512** and *regresses*
by B=1024 — on H200 the verifier's cost of processing more tree tokens
eventually outweighs the longer accepted prefix (main.tex:422). Vanilla DFlash is
the `B = block_size − 1 = 15` point (one flat block of 16); DDTree beats it "under
the same conceptual budget" because the tree is **front-heavy** — it spends nodes
on high-prefix-mass early continuations instead of a flat wide block. The paper
explicitly notes the optimal budget "can shift across hardware platforms and
implementations" (main.tex:422).

### 3.3 Does it match hipfire's "early + variable divergence" (bimodal) regime? — **Yes, directly.**

Fig. acceptance-histogram (main.tex:431–440, B=512): vs vanilla DFlash, DDTree
"shifts substantial probability mass toward longer accepted prefixes… it becomes
much rarer to observe acceptance lengths below 4, while full-block acceptance at
length 16 becomes substantially more common." That is precisely the collapse of
the low-accept mode. hipfire measured ~30% of cycles diverging at ≤1 token; the
tree attacks exactly this mode, because a *single* early divergence no longer
kills the whole block — the target can match a **sibling** at depth 1 (the tree
carries the top-`K` alternatives at each early position, and §1.2's front-heavy
budgeting puts most nodes there). DDTree is, structurally, an early-divergence
mitigator. The regime hipfire is in is the regime DDTree helps most.

### 3.4 The streaming-MoE correction (this is the high-value part)

The paper's B=256–512 optimum and the B=1024 regression are an **H200 artifact**:
there the target weights are resident, so verify cost scales with tree-token count
(SDPA tree-attention is quadratic in tree length, main.tex:545 — they can't even
use FlashAttention-2). In hipfire's streaming-MoE regime the cost model is
inverted:

- Verify `pass_time` is floored by streaming 60 layers × active experts of a
  397B model over the LAN/PCIe. Whether the pass carries 16 or 512 tokens, the
  **weights stream once**. So the marginal cost of extra tree nodes is ~compute
  only, which is cheap relative to the streaming floor.
- Therefore the speedup-vs-budget curve does **not** turn over at 512 the way it
  does on H200; the favorable region extends to **much larger `B`**, and hipfire
  should push budget until either (a) tree-attention compute or (b) NPU-side
  heap-build time (§1.3, a serial CPU cost) becomes comparable to the streaming
  floor — not stop at the paper's 512.
- **Caveat / open risk (MoE-specific, not in the paper):** a wider tree contains
  more distinct tokens → potentially routes to **more distinct experts** per
  verify pass (each token picks 10 of 512). If the streaming engine fetches only
  activated experts, a wide tree can *inflate the expert working set* and thus
  the streaming floor itself. The paper's Coder-30B-A3B has only ~a handful of
  active experts and never stress-tests this. hipfire must measure
  `distinct_experts(B)` — the tree's economic advantage assumes the expert set is
  saturated early and doesn't keep growing with `B`. If it does grow, the optimal
  `B` is set by expert-set saturation, not by compute. This is the single most
  important thing to measure before committing to a large budget.

Bottom line for Goal B: DDTree's `τ` lift (~1.4× typical, up to ~1.6× in the
low-accept regime hipfire lives in) multiplies straight through the
streaming-bound throughput formula, and the paper's budget ceiling is *softer*
for us than for them — provided expert-set growth is bounded.

---

## 4. Draft↔target interface for a TREE (Goal A)

hipfire's stated interface: target→draft = `target_hidden` for committed tokens
(~384 KB/cycle bf16); draft→target = token ids. Here is exactly how a tree changes
each direction, from the reference code.

### 4.1 target→draft (the hidden feedback) — **unchanged in magnitude per token.**

**This corrects the natural worry that a tree multiplies hidden traffic.** After
the verify forward, the target produces hidden for *all* `B+1` nodes, but DDTree
immediately **indexes down to the accepted path only** before feeding it back:

```python
target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
                    .index_select(1, accepted_index_tensor)     # ddtree.py:429
```

So what crosses the boundary target→draft is hidden for the **accepted path**
(`accept_len` tokens), *not* for all tree nodes. Per *committed* token the
feedback is identical to vanilla DFlash. The tree makes `accept_len` bigger, so
per cycle you ship more hidden — but per *generated token* the byte cost is
unchanged. Your ~384 KB/cycle estimate should be read as **per-token**, and the
tree does not inflate it. Concretely: `extract_context_feature`
(`model/utils.py:17–26`) concatenates the hidden states of the drafter's
`target_layer_ids` layers → feature width = `len(target_layer_ids) × hidden_size`,
**not** `hidden_size`. For a 397B target with `hidden=4096` and, say, 3 selected
layers that's `3×4096×2B = 24 KB` per token — check that against your 384 KB/cycle
figure: if 384 KB is a full block it implies ~16 tokens × 24 KB, i.e. your figure
is already the multi-layer-concatenated feature for a block, and DDTree keeps it
proportional to `accept_len`, not to `B`.

### 4.2 draft→target (the tree structure) — **this is the new traffic, and it is tiny if you send `parents`, not the dense mask.**

The target forward needs three things (ddtree.py:190–208): flattened
`verify_input_ids` (`B+1` ids), `verify_position_ids` (derivable from depths),
and the ancestor mask. Naively the mask is `B×B` bools = **256 KB at B=512** — do
**not** put that on the wire. The dense visibility is reconstructable on the
target side from the compact `parents` array in `O(B·B)` bool ops
(ddtree.py:151–159). So the minimal draft→target payload is:

- `node_token_ids` : `B × int32/`vocab-width ≈ `B×4B` = **2 KB** @ B=512
- `node_depths`    : `B × int8` (depth ≤ L ≤ 15) = **0.5 KB** (or recompute from parents)
- `parents`        : `B × int32` = **2 KB** — the tree topology; mask rebuilt on target

**~4–5 KB per cycle** for B=512, versus 256 KB if you shipped the dense mask.
This is the concrete wire format hipfire should adopt: send `(token_ids, parents)`,
rebuild `position_ids` (`= depth`) and the additive attention mask on the GPU
(target) box. Everything needed is in ddtree.py:151–208; none of it needs the
drafter's logits to cross the boundary.

### 4.3 Net for disaggregation

- target→draft: `accept_len × feature_width` (unchanged per token; tree only
  raises `accept_len`).
- draft→target: `~B × (token + parent)` ≈ **4–5 KB @ B=512** (topology only).
- The drafter (NPU) also does the heap build (§1.3) locally before sending — the
  target never sees `q_i`, only the chosen tree. The **round-trip count drops**
  by the `τ` lift (~1.4×), which is the real disaggregation win: fewer LAN
  round-trips per generated token, each carrying a few-KB tree up and an
  accept-path hidden blob down.

---

## 5. Reusable implementation specifics (portable to Rust/HIP)

- **Tree data structure** (ddtree.py:120–166): parallel arrays, not pointers —
  `node_token_ids[]`, `node_depths[]`, `parents[]` (`parents[0]=-1`,
  root=index 0), plus `child_maps[]` = per-node `HashMap<token_id → child_index>`
  for the O(1) verify walk. Trivially a `Vec<i32>`/`Vec<u8>` + `Vec<HashMap>` in
  Rust. Budget `B` bounds every allocation.
- **Heap build** (ddtree.py:116–149): binary max-heap keyed on cumulative
  log-prob `σ`; entries `(σ, ranks, parent_index, depth, rank)`. Pop → emit node,
  push sibling (`rank+1`, if `<K`) and child (append rank 1, if `depth<L`). Pure
  scalar arithmetic on top-`K` log-probs — **no GPU needed for the heap**; on the
  NPU box this is host CPU work. `K = min(B, vocab)`, `topk` per depth is the only
  GPU op (a `topk` over `L` rows).
- **Mask construction** (ddtree.py:151–159): row-inherit from parent +
  self-bit. In HIP this is a `B×B` bool fill you can do on host and upload, or a
  trivial kernel: `vis[i] = vis[parent[i]]; vis[i][i]=1`. Then the additive mask
  is `where(vis, 0, -inf)` over the tree block, with the past-context columns all
  0 (ddtree.py:204–208).
- **Position ids** = `start + depth` (ddtree.py:195–199); siblings share a
  position. One `add` over the depth array.
- **KV-cache compaction after accept** (ddtree.py:226–277, `compact_dynamic_cache`
  / `_compact_appended_window`): after verify, the target KV cache holds `B+1`
  appended entries; keep only the accepted-path indices, then crop. Done by
  `index_select` on the appended window `[past_length : past_length+current]` then
  `crop(past_length + accept_len)`. There's even an inline-C++ fast path
  (ddtree.py:30–74, `compact_tail_inplace`) that does the gather in-place —
  the moral for HIP: this is a per-layer `index_select` over the K/V tail, a
  cheap gather kernel; do it in-place to avoid reallocating the cache. **This is
  the one target-side bookkeeping step a tree adds over serial verify** — the
  draft KV cache is just `crop(start)` (ddtree.py:373) as before.
- **Buffer reuse** (ddtree.py:320–327): all verify buffers allocated once at
  `1+budget`; per round you slice `[:current_length]` and clear the previous
  tree block (ddtree.py:187–188). No steady-state allocation — matches hipfire's
  arena style.
- **Attention-impl constraint** (main.tex:545): FlashAttention-2 does **not**
  support the tree mask; the reference falls back to PyTorch SDPA for the target
  verify (the drafter itself still uses FA2). For hipfire this means the target
  verify attention kernel must accept a **custom additive mask** (dense `B×B`
  block + full past columns). Your existing no-LDS / register-tiled attention
  kernels on gfx1103 would need a masked variant; the tree block is small
  (`B ≤ ~512`) so a non-flash masked SDPA over the appended window is acceptable
  and avoids the FA-tree incompatibility entirely.
- **`block_size ≤ 1` shortcut** (ddtree.py:293–303): DDTree falls back to plain
  `dflash_generate`. Tree only exists for `block_size ≥ 2`; `L = block_size − 1`.

---

## 6. Recommendations for hipfire (DDTree-on-NPU + disaggregation + streaming MoE)

1. **Build the tree on the NPU/host, ship topology not mask.** Port
   `build_ddtree_tree` as host-CPU code on the Phoenix box: one GPU/NPU `topk`
   over the `L` draft-logit rows, then the `O(B log B)` scalar heap. Emit
   `(node_token_ids, parents)` and rebuild `position_ids`/mask on the target GPU
   box (§4.2). Wire cost ~4–5 KB/cycle @ B=512 vs 256 KB for a dense mask — this
   makes DDTree essentially free on the LAN for Goal A, and the hidden feedback
   stays at your existing per-token ~384 KB (accept-path only, §4.1). **The heap
   is a serial CPU cost** the paper measures as a real stage; on the NPU host make
   sure it overlaps the target's previous verify (it depends only on the drafter
   logits, which are ready before verify returns).

2. **Re-tune the node budget for the streaming floor, expect it far above 512,
   but gate it on `distinct_experts(B)`.** Do not inherit the paper's B≈256–512
   optimum — that's H200-resident-weight physics (§3.4). In hipfire's
   streaming-bound verify, extra tree tokens are nearly free until compute or the
   MoE expert working-set catches the streaming floor. Measure two curves on the
   397B-A17B target: `τ(B)` (monotone up, from the tree) and `distinct_experts(B)`
   (the risk). Set `B` at the knee of `distinct_experts(B)` — the point past which
   a wider tree starts streaming more experts per pass. If experts saturate early
   (likely, since a front-heavy tree concentrates on a few high-prob early
   continuations), you can run `B` into the thousands and bank most of the `τ`
   lift; if not, expert-set growth, not compute, is your budget ceiling. This
   measurement is the go/no-go for aggressive budgets.

3. **Target the early-divergence (≤1 token) mode explicitly, and consider a
   depth-shaped budget.** hipfire's 30%-diverge-at-≤1 bimodality is exactly what
   DDTree's front-heavy tree fixes (§3.3): the fix is *siblings at shallow
   depth*. Since our low-accept mode is heavier than the paper's benchmarks,
   consider biasing construction toward breadth at depths 1–3 (the standard
   surrogate already does this via prefix mass, but you can cap depth `L` lower
   and spend the budget wider). Validate through the existing
   `coherence-gate-dflash.sh` path: DDTree is lossless at temp 0 by the same
   greedy-match rule hipfire already uses (§2.3), so the correctness gate is the
   accept-path walk + KV compaction (ddtree.py:212–277), not a new sampling
   theory. Watch the temp>0 case: DDTree here follows the target's sample (lossless
   as one trajectory) but is **not** SpecInfer rejection sampling — matches
   hipfire's current serial semantics, so no regression, but don't expect extra
   multi-branch acceptance at temperature.

### Corrections to hipfire's stated assumptions
- **"Does the target return hidden for all tree nodes?"** — It computes hidden for
  all `B+1` nodes but returns/feeds back only the **accepted path**
  (ddtree.py:429). Per-token hidden traffic is unchanged by the tree; only
  `accept_len` per cycle grows.
- **The ~384 KB/cycle feedback does not scale with tree size `B`.** It scales with
  `accept_len × (len(target_layer_ids)×hidden)`. The tree's cost on the wire is
  the *upward* topology payload (~KB), not the downward hidden.
- **The paper's budget ceiling (512) is not hipfire's.** It is H200-resident-weight
  physics; hipfire's streaming-bound verify pushes the optimum higher, bounded by
  MoE expert-set growth rather than token-count compute.
- **Verification is greedy-match tree attention, not rejection sampling** — same
  lossless guarantee hipfire already relies on, generalized from one trajectory to
  a tree.
