# KV-compression adoption plan — 5 levers from the literature review

Status: **active**. Branch: `chaingun`. Date: 2026-07-13.

Executes the literature map in
`docs/plans/2026-07-12-hot-cold-hierarchical-kv-implementation.md` (§"KV-compression
literature map"). Scopes the five adoptable papers into one sequenced plan. All five
are **training-free** (offline calibration at most; no hot-path training). TriAttention
is already shipped; QSVD is deprioritized (mostly weight/VLM compression, and KVarN's
Sinkhorn already does the incoherence job it targets).

## The governing facts (established this session — do not re-litigate)

- **The merge is the only real quality cost**, and the loss is **content, not phase**
  (ceiling analysis): our cold merge averages tokens that *differ in content* because it
  groups by **position-adjacency**. This is the root cause the papers converge on.
- **The importance *scorer* is a spent lever for ranking** (T1: vnorm ≥ TriAttn at 0.8B
  and 9B; CASK's own Phase 1 agrees). But the TriAttention **κ-kernel is useful for the
  merge *grouping* distance** (near-duplicate detection), not for ranking.
- **KVarN wins quant**; asym is deprecated. **head_dim = 256** — low-rank is materially
  harder here than in the papers (best results at hd 64–128), so low-rank is a
  *supplement* at the 128–256 B/tok band, not the main path.
- hipfire has **two disconnected KV-compression subsystems**: the standalone
  TriAttention/CASK eviction (`triattn.rs`/`cask.rs`) and the hierarchical hot/cold cache
  (`kv_hier.rs`/`kv_compact.rs`, vnorm + position merge). Lever 1 unifies them.

## Frozen eval (all levers)

Reuse the rig from the parent plan: `perplexity --kv-mode kvarn` + `HIPFIRE_KV_*`
knobs, PPL+KLD vs bf16 on `wikitext2-1024s-2048ctx.txt` at ctx ∈ {2048, 16384},
0.8B (`qwen3.5-0.8b-mq4+`) then 9B (`qwen3.5-9b-mq4`); `parity_kv_hier` oracle for read
correctness; `coherence-gate-dflash.sh` after codec/kernel changes; nix2 gfx1103 (CWSR
live), coordinate with `hipfire lock`. Baselines to beat, from the head-to-head
(0.8B, ctx=16384): asym4 18.15/0.096, **single-tier KVarN 17.71/0.085**, hier-vnorm
18.41/0.149. **A lever "wins" only if it moves hier toward the KVarN floor.**

---

## Lever 1 — CASK similarity-based merge grouping (the gating experiment)

**Goal.** Replace the cold-merge's position-local grouping with **content-similarity
clustering** (merge near-duplicate keys → averaging is ~lossless) + a role-based
protected core. Directly attacks the content-loss root cause. This is the single
highest-value lever and the one that could make hierarchical merge *quality*-competitive
rather than memory-only.

**Change (files).** `compact_cold_kv` (`crates/hipfire-kvquant/src/kv_compact.rs`): the
grouping of non-core tokens. Today it sorts non-core by position and folds `fold_m`
adjacent (`position_local`). Instead, cluster non-core tokens by similarity and fold
each cluster. `cask.rs` **already implements exactly this grouping** ("L2-grouped greedy
matching, softmax-weighted weighted-average K/V") for its Q8 path — port/reuse that
routine rather than writing new clustering.

**Algorithm.** For the non-core tokens: greedy similarity grouping — repeatedly take an
unassigned token, pull its `fold_m−1` nearest neighbors by distance `d(k_i,k_j)`, form a
group, merge via the existing mass-weighted m-fold. Distance options, cheapest first:
(a) plain cosine on K; (b) **κ-weighted** `d_κ = Σ_f |κ(ω_f)|·‖k_{i,f}−k_{j,f}‖` using
the future-relevance kernel from the loaded `triattn` centers (reuses the sidecar we
already load for `ImportanceMode::TriAttn`). Core (protected, never merged) = top
`core_frac` by importance (keep vnorm — ranking is fine for the *core*, spent only for
choosing merge partners). Gate behind `HIPFIRE_KV_MERGE=similarity` (default
`position`, byte-identical).

**Training cost.** None for cosine; κ-variant reuses the existing TRIA sidecar.

**Gate.** hier(similarity) beats hier(position) on KLD at 2k **and** 16k, and closes
toward single-tier KVarN (0.085); `parity_kv_hier` green; coherence gate green. If it
does not beat position-merge, the whole CASK→hier thesis is closed (position-adjacency
was already near-optimal), and hierarchical stays a memory-only play.

**Difficulty.** Medium. Grouping swap in one function + reuse of `cask.rs` logic.
**Sequencing: FIRST.** Everything else composes on top; this decides whether the
hierarchical-quality direction lives.

---

## Lever 2 — PyramidKV per-layer budgets

**Goal.** Layer-varying budgets (lower layers bigger / less merge, upper layers smaller
/ more merge) instead of uniform, per the depth-wise attention-sparsity observation.

**Change (files).** `HierKvState::from_env` (`kv_hier.rs`): `hot_budget`/`fold_m`/
`core_frac` scalar → per-layer (compute from layer index; no need to store a Vec).
`migrate_n` already carries `layer` — pass the layer-specific `fold_m`/`core_frac` into
`compact_cold_kv`.

**Algorithm.** PyramidKV arithmetic sequence: instruction window α (last α tokens) kept
uniformly; remaining budget distributed linearly, top layer `k^{m-1}=k_total/(β·m)`,
bottom `k^0=2k_total/m − k^{m-1}`, linear in between (α=8, β=20 defaults). Apply the same
shape to hot_budget and inversely to fold_m. Gate `HIPFIRE_KV_PYRAMID=1`.

**Training cost.** None (heuristic + arithmetic).

**Gate.** Same-VRAM KLD uplift OR same-KLD VRAM reduction vs uniform, 0.8B + 9B; parity
green. Low bar; low risk.

**Difficulty.** Low. **Sequencing: parallel to Lever 1** (independent; composes).

---

## Lever 3 — KQ-SVD V·Wᵒ objective (principled V compression)

**Goal.** Compress V by *how it's used through the output projection* (`V·Wᵒ`), not V in
isolation — the "correct objective" for V low-rank. This is the principled version of
T4 ("compress V further").

**Change (files).** Offline quant path (`hipfire-quantize` / `kvquant` V codec), NOT the
runtime hot path. Store/quantize V in the reduced `V·Wᵒ`-aware basis; drop the V
directions that `Wᵒ` annihilates.

**Algorithm.** Offline, per (layer, kv-head): SVD of the value–output interaction
(`V·Wᵒ` over a calibration set) → keep the top directions; represent cold V in that
basis (fewer effective dims / lower bits at equal output error). Fuse the reconstruction
into `Wᵒ` where possible so runtime read cost is unchanged.

**Training cost.** Offline calibration (~256 samples), closed-form/SVD. No training loop.

**Gate.** V compressed below the current `cold_v_bits` floor at equal attention-output
error / KLD; coherence gate green (it's a codec change).

**Difficulty.** Medium–high (offline transform + V codec change). **Sequencing: the V
track (Levers 3+4 compose)**, after Lever 1 (so V work runs against the better merge).

---

## Lever 4 — ReCalKV OVC + HSR (offline low-rank V calibration)

**Goal.** If doing low-rank V (Lever 3), make the factors optimal: closed-form **Offline
Value Calibration** (a strict win over vanilla SVD) + **Head-wise Similarity Reordering**
(group similar heads before grouped SVD; K needs full reconstruction under RoPE).

**Change (files).** Same offline quant/low-rank path as Lever 3 — this is the
*calibration method* for it. `L_v`, `R_v` closed-form (ReCalKV Eq. 7/8), fuse `R_v·Wᵒ`.
HSR: CKA head-similarity clustering before per-group SVD.

**Algorithm.** OVC: given calib activations X, `L_v = WXXᵀR_vᵀ(R_vXXᵀR_vᵀ)⁻¹`,
`R_v = ((L_v)ᵀL_v)⁻¹(L_v)ᵀW`; deterministic, no SGD. HSR: greedy CKA clustering of heads,
grouped SVD per cluster. Note ReCalKV's Fisher finding: **V is *more* low-rank-sensitive
than K** — so protect V rank; be more aggressive on K low-rank if pursued.

**Training cost.** Offline, 256 samples, closed-form. Negligible.

**Gate.** OVC beats vanilla SVD-V at equal rank on KLD; combined with Lever 3, V track
hits its compression target with no coherence regression.

**Difficulty.** Low–medium (closed-form matrix ops in the offline path). **Sequencing:
composes with Lever 3** — 3 sets the objective/basis, 4 sets the optimal factors.

---

## Lever 5 — OjaKV online-PCA low-rank KV track (conditional)

**Goal.** A runtime low-rank KV mode whose basis is tracked **online** (Oja's incremental
PCA) instead of a per-cache SVD — cheaper (O(r²d) vs O(d³)) and adapts to distribution
shift; high-residual tokens kept full-rank (hybrid). The long-context 128–256 B/tok
supplement.

**Change (files).** A new low-rank KV path (`kv.rs` + a module), an Oja-update + QR-reorth
kernel, hybrid full/compressed storage keyed on reconstruction residual.

**Algorithm.** Init basis from a small corpus; prefill Oja on salient (importance-scored,
pooled) tokens; during decode accumulate a buffer, Oja-update every T≈8 steps at a
conservative rate, QR-reorthonormalize. Store residual-heavy tokens full-rank, rest as
r-dim; `OjaKV-PF` variant keeps prefill full-rank, compresses decode only.

**Training cost.** None (online, training-free).

**Gate — feasibility first.** Before building the runtime path, an **offline probe**:
does static rank-r low-rank even hold at **head_dim=256** on qwen3.5 (cos/KLD at r ∈
{64,128,192} vs the 256B/tok band)? Our prior static-SVD note (rank-64 cos 0.991@256B)
suggests yes, but re-confirm on the current models. Only if the static probe clears do we
build the online (Oja) machinery. Then: Oja-basis ≥ static-basis under a domain shift
(prefill→decode / cross-domain), at lower compute.

**Difficulty.** High (new runtime path + kernel). **Sequencing: LAST / conditional** —
gated on the head_dim=256 static-low-rank feasibility probe and on the 128–256 B/tok
long-ctx band actually being a target.

---

## Overall sequencing

1. **Lever 1 (CASK similarity-merge)** — *start here*. Decides the whole
   hierarchical-quality thesis; reuses `cask.rs` + `triattn` centers.
2. **Lever 2 (PyramidKV budgets)** — parallel, cheap, composes.
3. **Levers 3 + 4 (V low-rank track: KQ-SVD objective + ReCalKV OVC)** — offline
   quant-tooling track; run after Lever 1 so V work targets the improved merge.
4. **Lever 5 (OjaKV low-rank runtime)** — only after its static-low-rank feasibility
   probe at head_dim=256 passes and if the long-ctx band is a priority.

## Risk / rollback

Every lever is flag-gated and default-off (`HIPFIRE_KV_MERGE=position`,
`HIPFIRE_KV_PYRAMID=0`, V-low-rank behind a quant flag, OjaKV behind a KV-mode). Baseline
stays byte-identical. Levers 3–5 touch codecs/kernels → gate on
`coherence-gate-dflash.sh` and cross-arch (RDNA2/3/4) before default-on. QSVD excluded
(weight/VLM-oriented; KVarN already covers its incoherence goal); TriAttention already
shipped.
