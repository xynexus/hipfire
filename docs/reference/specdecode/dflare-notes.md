# DFlare — distilled notes for hipfire

**Paper:** *DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative
Decoding* (Zhang et al., PKU + Tencent). Code: Tencent/AngelSlim.
**Local copy:** `/srv/hipfire/references/SpecDecode/DFlare/` (`tex/acl_latex.tex`,
`src/angelslim/compressor/speculative/train/models/draft/`).

DFlare is an incremental architecture + training upgrade to **DFlash**. Same
block-diffusion draft-then-verify loop, same interface shape, +8–11% wall-clock
over DFlash. The two draft models differ by ~40 lines. Read this before assuming
DFlare gives hipfire a new *per-pass* lever — **it largely does not** (see §1, §6).

---

## TL;DR corrections to hipfire's stated assumptions

1. **"Scaling draft capacity" in DFlare = scaling the drafter's own model
   capacity (depth + conditioning breadth), NOT tokens-per-verify-pass.** Block
   size is fixed at **16** in both DFlash and DFlare (8 for GPT-OSS-20B);
   `configs/qwen3_dflash.json:7` and `configs/qwen3_dflare.json:7` both set
   `block_size: 16`. The only structural change is draft **layers 5 → 7**
   (`qwen3_dflash.json:37` = 5 vs `qwen3_dflare.json:43` = 7) plus per-layer
   target fusion. Candidate tokens verified per pass are unchanged. This is *not*
   the Goal-B "more draft capacity per verify pass" lever in the wide-tree /
   bigger-block sense — it raises the **acceptance rate inside a fixed 16-wide
   linear block**.

2. **There is no tree.** `spec_generate` (qwen_dflare.py:337, qwen_dflash.py:290)
   drafts one linear block and does a longest-common-prefix match
   (`cumprod(dim=1).sum` at qwen_dflare.py:409). No DDTree, no tree attention, no
   branching. τ is hard-capped at `block_size − 1 = 15` and in practice lands
   ~4.5–9. DFlare and a DDTree are **orthogonal and composable**, not competitors
   (§3).

3. **DFlare *increases* the draft↔target interface volume**, it does not hold it
   fixed: it captures **T = 9** target layers vs DFlash's 5, and all T layers'
   hidden states must reach the drafter (§4). For a disaggregated hipfire this is
   1.8× the target_hidden bytes/token unless you pre-fuse (§4, §5).

4. **The architecture change is a small part of the win.** The layer-wise fusion
   contributes ≈ +0.05–0.1 τ; **scaling training data 270K → 800K → 2.4M
   contributes ≈ +2.4 τ** (Table `data_scaling_ablation`, acl_latex.tex:569–572:
   τ 4.90 → 6.59 → 7.33). For Goal B the dominant lever is *drafter training*,
   not the fusion trick.

---

## 1. What "scaling draft capacity" means, and the algorithm

DFlash's block-diffusion drafter is a small bidirectional cross-attention
transformer. Each iteration (Preliminaries §2.2, acl_latex.tex:161–176):

- Input: the last committed target token `x_t` + `(B−1)` mask tokens for
  positions `t+1..t+B−1` (`noise_embedding`, embedded via the *target's* frozen
  `embed_tokens`; qwen_dflare.py:383).
- Every draft layer cross-attends to **target context** `C` (fused target hidden
  states for all prior positions, injected as KV) plus the block's own positions.
  Attention is **bidirectional within the block** (`is_causal=False`,
  qwen_dflare.py:119, 393).
- After the last draft layer, the **target's** frozen `lm_head`
  (qwen_dflare.py:384) emits logits at the `B−1` masked positions → all candidate
  tokens sampled in **one** forward pass. Drafting cost is ~decoupled from block
  size (that is the block-diffusion premise, Intro acl_latex.tex:92–94).

DFlash's bottleneck (paper's thesis, acl_latex.tex:99–103, 150–157): it fuses the
T target layers with a **single** `Linear(T·H → H)` FC (qwen_dflash.py:254,
275) and feeds the **identical** fused vector `c_t` to **every** draft layer.
Shared input ⇒ layers can't specialize ⇒ depth scaling saturates (DFlash τ at
5/7/9 layers = 4.13 / 4.18 / 4.19, Table `target_results` acl_latex.tex:507–509).

**DFlare scales capacity along three axes** (Method §4, acl_latex.tex:196):

1. **Adaptive Layer Fusion** (§4.1, the core mechanism). A learnable
   `layer_fusion_weights[D, T]` matrix (qwen_dflare.py:283). Per draft layer `i`,
   `α^(i) = softmax(W_fuse[i,:])` over the T target layers, and the layer's
   conditioning is `f^(i)_t = RMSNorm(Σ_j α^(i)_j · h^(j)_t)`
   (qwen_dflare.py:318–321, einsum `bsth,dt->bsdh`). Every draft layer gets its
   **own** weighted combination of target layers → distinct per-layer input →
   layers specialize → depth scaling unlocks. Cost: **D·T scalars** (63 for
   D=7,T=9); α is static after training and can be precomputed
   (acl_latex.tex:212). Negligible FLOPs. This is what lets them add both more
   target layers (T 5→9) *and* more draft layers (5→7) with real gains.

2. **Heterogeneous KV projections** (§4.2). Target context and draft (noise)
   tokens get **separate** K/V projections: `k_proj_target/v_proj_target` for
   context vs `k_proj/v_proj` for noise (qwen_dflare.py:130–144, 171–176). DFlash
   shares one projection for both (qwen_dflash.py:149–152). Decouples the two
   representational subspaces. Ablation: removing it costs ~0.04 τ
   (Table `structure_ablation` `−KVProj`, acl_latex.tex:385).

3. **Progressive position-weighted loss** (§4.3). CE weight per block position
   `w_k = exp(−(k−1)/γ)` with γ **warming up** linearly across epochs
   (`γ_0 = 4.5`, +1/epoch; GPT-OSS fixed γ=4). Code:
   online_dflash_trainer.py:898–901 (`decay = exp(-(k-1)/gamma)`) and the
   per-epoch bump at :698–707. Early = focus early positions (fast convergence);
   late = flatten to fix hard tail positions. Ablation `−Loss`: ~0.05 τ
   (acl_latex.tex:386).

**Net:** capacity is scaled by making each draft layer a stronger unit (distinct
target conditioning) so you can afford more draft layers and more target layers —
all at a fixed block width of 16.

---

## 2. Compute / acceptance tradeoff (Goal B)

Throughput ≈ accepted-tokens-per-pass (τ) ÷ verify-pass-time. DFlare raises τ but
the conversion to wall-clock is where the cost model bites:

**Draft-depth scaling** (DFlare, Qwen3-4B, Table `draft_layer_ablation`
acl_latex.tex:543–545), layers 5 → 6 → 7:
- τ: 6.55 → 6.72 → **6.90** (+5.3% for +2 layers)
- speedup: 5.12 → 5.13 → **5.16** (+0.8%)

The τ gain is real but the **wall-clock speedup nearly flatlines**: added draft
depth adds per-pass latency that almost cancels the acceptance gain **on a shared
device** where drafting and verify serialize on the same GPU. DFlash saturates
harder (τ 6.30 → 6.57 → 6.67; speedup 4.98 → 5.07 → 5.01 — *speedup drops* at 7
layers).

**Target-layer breadth** T ∈ {5,7,9} (Analysis §5.2, Fig. 4 right): DFlare's τ
rises monotonically 5→9; DFlash gains 5→7 then **flatlines** 7→9 (acl_latex.tex:410).
This is the fusion mechanism's headline: it can actually absorb more target
layers, where DFlash can't.

**Data scaling dominates** (Table `data_scaling_ablation`, Qwen3-8B):
270K → 800K → 2.4M ⇒ τ 4.90 → 6.59 → **7.33**, speedup 3.74 → 4.97 → **5.46**.
≈ +2.4 τ from data vs ≈ +0.1 τ from the architecture.

**Where it saturates / cost model:** on same-device H20, the marginal accepted
token from a deeper drafter is worth ~one extra draft-layer latency; beyond ~7
draft layers and T≈9 the speedup curve is flat. The acceptance ceiling is
structural (τ < block_size = 16, single linear path; best observed τ ≈ 7–9 at
temp 0, dropping to ~5–6 at temp 1, acl_latex.tex:279–306).

**Serving (verify-bound regime, closest to hipfire's Goal B):** SGLang H20,
Qwen3-8B, GSM8K throughput tok/s (Table `sglang_h20`, acl_latex.tex:471–473):
baseline 162 → DFlash 596 → **DFlare 642** at concurrency 1; at concurrency ≥4,
DFlare ≈ **1.7–1.8k** vs DFlash ≈ 1.05k. "As concurrency increases and the system
becomes more compute-bound, the advantage of DFlare over DFlash grows"
(acl_latex.tex:484) — i.e. the acceptance gain matters *most* when verify is the
scarce resource, which is exactly hipfire's streaming-MoE case.

---

## 3. Relation to DFlash and DDTree

- **vs DFlash:** DFlare **extends** DFlash — same loop, same trainer
  (`OnlineDFlashTrainer` consumes both models; qwen_dflare.py docstring:28–30),
  same block size, same interface *shape*. Three swaps: FC→layer-fusion,
  shared→heterogeneous KV, fixed→warmup γ. It does **not** subsume DFlash's block
  mechanism; it re-parameterizes the target-conditioning path.

- **vs DDTree / tree drafting:** DFlare has **no tree**. It scales the *depth and
  conditioning* of a single linear drafter; a DDTree scales the *width/branching*
  verified per pass. They attack different terms:
  - DDTree ↑ candidate tokens per verify pass (more τ headroom, more verify FLOPs
    per pass).
  - DFlare ↑ probability each drafted token is accepted (higher realized τ within
    whatever width you allow), at ~zero extra verify cost.
  They **compose**: a DFlare-style layer-fused block drafter could emit a *tree*
  of candidates instead of one linear block, and the fusion/heterogeneous-KV
  tricks are agnostic to that. If both "scale capacity," the distinction is
  **quality-per-slot (DFlare) vs number-of-slots (DDTree)**. For a fixed verify
  budget you want both: widen with a tree, then raise per-slot acceptance with
  DFlare-style conditioning.

---

## 4. Draft↔target interface (Goal A)

What crosses the boundary each cycle (`spec_generate`, qwen_dflare.py:399–419):

- **target → draft:** hidden states of the newly committed tokens, at the **T
  captured target layers**, concatenated along feature dim → `[B, n_commit, T·H]`
  (`extract_context_feature`, qwen_dflare.py:88–98, 417). This is the dominant
  volume.
- **draft → target:** the `block_size` candidate token IDs (tiny, integers).
- target also returns the sampled posterior token IDs (tiny).

**Scaling capacity DOES change interface volume.** DFlare raises T from 5 → 9
(acl_latex.tex:352: "9 layers uniformly selected between the 2nd and 3rd-to-last
target layer; 8/8/7 for GPT-OSS"). Interface bytes scale **linearly in T**:

- Qwen3.5-397B hidden H = 4096, bf16 = 2 B/elem ⇒ **8 KB per token per captured
  layer**.
- DFlash T=5 → 40 KB/token; **DFlare T=9 → 72 KB/token** (1.8×).
- Per verify cycle you ship `n_commit = accept_len+1` tokens (~7 typical) ⇒
  ~504 KB/cycle at T=9 vs ~280 KB at T=5.

So hipfire's "~384 KB/cycle" figure is **T-dependent** — it corresponds to
roughly 5–6 captured layers × ~6–7 committed tokens. Adopting DFlare's T=9 raises
it accordingly. **Correction:** the interface is not a fixed per-cycle constant;
it is `n_commit × T × H × 2` bytes and DFlare's capacity scaling is *directly* a
scaling of this LAN payload. This is the one place where DFlare's "more capacity"
does cost you on the Goal-A wire — plan for it, and mitigate per §5.

---

## 5. Reusable implementation specifics for a Rust/HIP/NPU port

- **Layer fusion is trivial to port and cache-friendly.** After training, α =
  `softmax(layer_fusion_weights, dim=1)` is a static `[D, T]` matrix
  (qwen_dflare.py:318). Precompute it once. Per committed token the fusion is
  `D` weighted sums over `T` H-vectors + an RMSNorm (qwen_dflare.py:320–321) —
  `D·T·H` madds/token, negligible on NPU. No LDS pressure (matches nix1's no-LDS
  preference).
- **Minimize the LAN payload by pre-fusing on the GPU/target box.** The drafter
  consumes `D` layer-specific fused vectors `F^(i)`, derived from the `T` raw
  layers by the static α. When **D < T** (paper: D=7 < T=9), compute the fused
  `[n_commit, D, H]` on the target side and ship **D·H per token (56 KB)** instead
  of **T·H (72 KB)** — a 22% wire saving and it moves the (tiny) fusion off the
  NPU. The RMSNorm is per-vector so it pre-composes cleanly. Alternatively ship
  raw T and fuse on the NPU; either way the einsum `bsth,dt->bsdh` is the whole
  op.
- **Heterogeneous KV → context KV is a per-commit constant, not per-iteration.**
  The context K/V (`F^(i) W_K^t`, `F^(i) W_V^t`, qwen_dflare.py:171–176) depend
  only on committed tokens, so compute them **once per commit** and hold them in
  the draft KV cache; only the block/noise K/V recompute each draft iteration.
  Heterogeneous projections thus add **2× the K/V projection weights** (small,
  `d×d_kv` each) and a one-time per-commit cost — **no** per-draft-iteration
  penalty. This fits hipfire's existing DFlash KV-injection cache directly:
  duplicate the K/V-proj weight set, route context through the `_target` set.
- **RoPE detail:** context (target) KV and block positions share the RoPE tables;
  `apply_rotary_pos_emb` slices `cos[..., -q_len:, :]` for the query
  (qwen_dflare.py:66–72) so context keys get full-length positions, queries get
  the block tail. Mirror this offset exactly or numerics drift.
- **q/k RMSNorm** (Qwen3 head-dim norm) is applied to q and to the *concatenated*
  context+noise k before RoPE (qwen_dflare.py:170, 177). Order matters:
  norm → transpose → RoPE → cache.
- **Masks/batching (training only, but informative):** blocks are packed and
  isolated with a FlexAttention BlockMask — each block sees target context
  strictly before its anchor (`kv_idx < anchor_pos`) and its own block positions,
  nothing across blocks (`create_dflash_block_mask`,
  online_dflash_trainer.py:47–86; anchors sampled at :733–770, 512/seq). At
  inference the mask is just "block attends to all committed context + itself,"
  bidirectional. For a batched NPU drafter you'd replicate the block-diagonal +
  context-prefix mask.
- **Frozen shared weights:** drafter reuses the target's `embed_tokens` and
  `lm_head`, frozen (Preliminaries §2.3, acl_latex.tex:183; qwen_dflare.py:383–384).
  In disaggregation the NPU box needs a copy of `embed_tokens` (to build
  `noise_embedding`) and `lm_head` (to sample candidates) — these do **not** cross
  the wire per cycle, but must be resident on the draft box. For a 397B target
  that is the vocab×H embedding + LM head only, not the MoE stack.
- **GPT-OSS uses block_size 8** and fixed γ=4 (acl_latex.tex:353, 447) — smaller
  block ⇒ lower τ ceiling; the loss schedule matters less on a short block.

---

## 6. Recommendations for hipfire (streaming MoE target, verify is the scarce resource)

**Context:** throughput ≈ τ ÷ verify-pass-time, verify is streaming-bound, and
hipfire's DFlash τ is bimodal (~30% diverge ≤1 token). The expensive resource is
the verify pass, and drafting on a separate NPU box is overlapped/hidden.

1. **Adopt the two "free" DFlare pieces; be skeptical of raw depth.** Layer-wise
   fusion + heterogeneous KV cost almost nothing at inference (63 scalars; 2×
   small K/V-proj weights) and are the parts that raise acceptance without
   touching the verify pass. They are worth porting. The **draft-depth** increase
   (5→7 layers) buys only ~+0.05 τ and, on a *shared* device, nearly zero
   speedup (§2) — but hipfire's **disaggregated** setup changes this calculus:
   with draft latency hidden behind verify, the depth-scaling speedup saturation
   that DFlare reports on H20 is a same-device serialization artifact. **hipfire
   can push draft depth/capacity harder than DFlare's on-GPU numbers suggest,**
   because the marginal draft-layer latency is off the critical path. Add layers
   until either (a) draft-box latency exceeds the verify-pass time (the overlap
   budget), or (b) τ stops rising — not until GPU speedup flatlines.

2. **Spend the biggest budget on drafter training data, targeting early
   positions.** Data 800K→2.4M gave +0.7 τ where architecture gave +0.1
   (§2). And the progressive position-weighted loss (γ warmup,
   online_dflash_trainer.py:898–901) directly attacks hipfire's "30% diverge ≤1
   token" mode — early-position CE weight `exp(−(k−1)/γ)` makes token 1–2 accuracy
   the training priority. This is the highest-leverage, verify-free change for the
   bimodal-acceptance problem. Regenerate training responses with the *actual*
   397B target at temp 0.6 (acl_latex.tex:447) so the draft distribution matches.

3. **For more accepted-tokens-per-verify-pass, widen with a tree — DFlare alone
   won't get you there.** DFlare's τ ceiling is block_size−1 (=15) on a single
   linear path and it realistically hits ~7–9. If the MoE verify pass is so
   expensive that you want 15–30 accepted tokens per pass, that requires a
   **bigger block and/or a DDTree** (more verified slots), which DFlare does not
   provide. The right stack for Goal B is **DDTree/wider-block for slot count ×
   DFlare-style per-layer fusion for per-slot acceptance** (§3) — they multiply.
   Concretely: keep (or grow) the DDTree width to fill the verify pass, and use
   layer-wise fusion + heterogeneous KV + early-position loss to raise the
   acceptance rate of every slot in that tree. Widening a DDTree raises verify
   FLOPs/pass (bad when verify is scarce); DFlare's conditioning raises acceptance
   at ~zero verify cost — so **prefer DFlare-style quality gains first, widen the
   tree only until the verify pass is full**, then stop.

4. **Budget the LAN interface for T, and pre-fuse to D.** If you raise captured
   target layers to T=9 for the fusion gain, ship the **D pre-fused** vectors from
   the GPU box, not the T raw ones (§5) — 56 KB/token vs 72 KB/token for D=7<T=9,
   and it keeps the drafter's per-layer conditioning intact. Revisit the
   "~384 KB/cycle" budget: it is `n_commit × T × H × 2` and scales with both the
   capacity knob (T) and realized acceptance (n_commit).
