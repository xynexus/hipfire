# DFlash — distilled technical notes (base drafter hipfire implements)

Source: `/srv/hipfire/references/SpecDecode/dflash/`
Paper: Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", arXiv:2602.06036 (ICML 2026 submission).
Code read: `src/dflash/model.py` (PyTorch reference), `src/dflash/model_mlx.py` (MLX reference — cleaner cache semantics), `src/dflash/benchmark.py`, `tex/sections/{preliminaries,method,appendix}.tex`, `src/README.md`.

All file:line citations below are into that reference tree, not the hipfire repo.

---

## 0. One-paragraph summary

DFlash is a **small block-diffusion draft model** for lossless speculative decoding. Instead of an autoregressive drafter that runs γ sequential forward passes, DFlash predicts an entire B-token block in **one forward pass**: the block is `[anchor_token, MASK, MASK, …]` and the draft denoises all masks in parallel with bidirectional (non-causal) attention. Draft quality comes almost entirely from **conditioning on the target model's own hidden states** — a fixed set of target layers is extracted, concatenated, projected once, and **injected as extra Keys/Values into every draft layer** (KV injection). Without this conditioning a 5-layer diffusion drafter only gets ~2–3× (appendix Table `naive_diffusion`); with it DFlash reaches >6× lossless and up to 2.5× over EAGLE-3 (abstract).

---

## 1. Core method — block-diffusion drafting

### One forward pass → a B-token block
Per decode cycle the draft consumes a block `block_output_ids = [anchor, MASK, …, MASK]` of length `block_size` (`model.py:108`, `model_mlx.py:497`):
- position 0 = **anchor** = a *clean* token the target already committed (the bonus token from the previous verify).
- positions 1..B-1 = `mask_token_id` (`model.py:79-80`; `mask_id` MLX `:447`).

The draft embeds this block with the **shared, frozen target embedding** (`noise_embedding = target.model.embed_tokens(block_output_ids)`, `model.py:111`), runs `num_hidden_layers` DFlash decoder layers, and emits logits for the B-1 masked positions in a single pass. It keeps only the predicted tail (`[:, 1-block_size:]` PyTorch `model.py:119`; `logits_start=1` MLX `:503`), samples them (greedy `argmax` at temp 0, `model.py:48-54`), and writes them back into positions 1..B-1 (`model.py:121`). That is the drafted block. **No inner loop** — the "denoising" is a single step, not iterative diffusion.

### Attention structure (the important bit)
Each draft layer is a Qwen3-style attention block (`Qwen3DFlashAttention`, `model.py:185-255`; `DFlashAttention`, `model_mlx.py:66-116`) with `is_causal = False` (`model.py:194`). Within a block, queries come only from draft tokens; keys/values are the **concatenation of target-context K/V and draft-token K/V**:

```
Q_i = W_i^Q · H_d                             (draft tokens only)
K_i = [ W_i^K · H_t ;  W_i^K · H_d ]_seq      (target ctx  ++  draft tokens)
V_i = [ W_i^V · H_t ;  W_i^V · H_d ]_seq
```
(appendix `kv-injection`, tex lines 57-64; code `model.py:226-231`, `model_mlx.py:93-108`). So draft tokens attend **bidirectionally to each other** and to the injected target features. `W_i^Q/K/V/O` and the FFN are the only trained weights per layer.

### Training objective
- Frozen AR target + frozen shared embedding & LM head; only the draft transformer layers train (`method.tex:55`, "diffusion adapter").
- Take a prompt+response, run it once through the target, extract+fuse hidden features for *all* tokens, inject as K/V (`method.tex:35`).
- **Random anchor sampling**: sample anchor tokens anywhere in the response, make each the first position of a block, mask the next `block_size-1`, train to predict them in parallel (`method.tex:39`). Blocks are packed into one sequence with a sparse mask (bidirectional within-block, injected-ctx visible, no cross-block leakage) via FlexAttention (`method.tex:41`). This matches inference exactly (block always starts from a clean target token) and augments data.
- **Position-weighted CE loss** — early positions dominate acceptance, so weight token k in a block by
  `w_k = exp(-(k-1)/γ)` (Eq. `loss-decay`, `method.tex:48-53`). γ=7 for B=16, 5 for B=10, 4 for B=8 (`appendix.tex:7`).
- 6 epochs, AdamW lr 6e-4, cosine, seq len 3072 (4096 for coder), 512 anchors/seq (`appendix.tex:7`). Training feature extraction can be online or offline-cached (`appendix.tex:9`).

### Why it wins (preliminaries.tex)
Per-token latency `L = (T_draft + T_verify)/τ`, τ = accepted tokens/cycle incl. bonus (Eq., `preliminaries.tex:18-21`). AR drafter `T_draft = γ·t_step` grows with budget; diffusion `T_draft = t_parallel` is ~flat in γ (`preliminaries.tex:28-39`). So the diffusion drafter can be **deeper** (5 layers) at fixed latency, buying higher τ — a strictly better Pareto point than EAGLE-3's 1 layer (`preliminaries.tex:49`).

---

## 2. The draft↔target INTERFACE (critical for Goal A — disaggregation)

### What the drafter consumes from the target
The **hidden states of a fixed set of target layers**, extracted every time the target runs a forward that commits tokens.

- Which layers: `target_layer_ids`, either from `dflash_config` or `build_target_layer_ids(num_target_layers, num_draft_layers)` (`model.py:27-36, 312-314`). Default = **uniformly spaced from layer 1 to `num_target_layers-3`, one id per draft layer** (`build_target_layer_ids`). Number extracted = `len(target_layer_ids)` = number of draft layers **by default**, but it is config-driven and can be set independently.
- Extraction: `extract_context_feature` concatenates the selected layers along the feature axis, `dim=-1`, with `offset=1` because `hidden_states[0]` is the embedding output (`model.py:39-45`). Result shape `[B, rows, len(layer_ids)·hidden]`. Precision **bf16** (`benchmark.py:220-221`).
- **The `fc` projection lives INSIDE the draft model**, not the target: `self.fc = Linear(len(target_layer_ids)·hidden → hidden, bias=False)` then `hidden_norm` RMSNorm (`model.py:317-318, 334`; MLX `:139, 189`). So over a wire, the *raw concatenated* per-layer hidden states are what cross the interface unless you deliberately move `fc` to the target side.

**Rows sent per cycle = accepted+1** (the committed window). After each verify the target hidden is sliced to exactly the committed tokens: `[:, :acceptance_length+1, :]` (`model.py:143`; MLX `hidden = hidden[:, :accepted+1, :]`, `:567`). At prefill it's the full prompt hidden once (`model.py:99`).

### Per-cycle data both directions
- **target → draft**: committed-window target hidden, `[accepted+1, len(layer_ids)·hidden]` bf16, **plus** the committed token IDs / bonus token (needed to form the next block anchor, `model.py:136-137`). The hidden dominates.
- **draft → target**: the drafted block token IDs, `block_size-1` ints (greedy). Tiny (~60 B for B=16). If sampling, top-K logits instead. The draft does **not** send hidden back.
- The **target runs verify** (Section 3) and produces the next `posterior` + next target hidden. So the round trip is: draft ships block IDs → target verifies + emits committed hidden → draft.

### Confirming hipfire's numbers (8 layers × hidden, ~384 KB/cycle)
Formula: `bytes = (accepted+1) · len(target_layer_ids) · hidden · 2`.
With `len=8`, `hidden=4096`, `accepted+1 ≈ 6`:
`6 · 8 · 4096 · 2 = 393,216 B = 384 KiB`. **hipfire's ~384 KB/cycle is correct** and implies it assumes ~6 committed rows/cycle. The `num_extract=8 × hidden` shape is consistent with an 8-entry `target_layer_ids` (verify against the specific `Qwen3.5-122B-A10B-DFlash` / 397B drafter's `config.json → dflash_config.target_layer_ids` and `fc.weight` shape `[hidden, 8·hidden]`; the paper's worked example uses 5 layers for the 35B-A3B drafter, `appendix.tex:48-56`, so 8 is model-specific, not universal).

### Correction / optimization for Goal A
Because `fc + hidden_norm` is a *fixed* `(len·hidden → hidden)` linear+RMSNorm producing `[rows, hidden]`, you can **move it onto the target box** and ship the projected feature instead of the raw stack:
`6 · 4096 · 2 = 48 KiB/cycle` — an **8× LAN reduction** (with 8 extract layers). Cost: the target must then know the draft's `fc`/`hidden_norm` weights and `target_layer_ids`, coupling the two boxes' versions. Keeping `fc` on the draft (as the reference does) keeps the draft self-contained but pays the 8× wire tax. This is the single highest-leverage interface decision for disaggregation.

Also note the draft box must host the **shared frozen `embed_tokens` and `lm_head`** (`model.py:111-112`; MLX `bind()` `:153-168`). For a 397B-class vocab these tables are large; either co-locate them on the NPU box or have the target ship the 16 anchor/mask embedding rows per cycle (`16·4096·2 = 128 KiB`, same order as the hidden). Co-locating is cleaner — they're static.

---

## 3. Verify + acceptance mechanism (what makes it lossless)

Reference loop `dflash_generate`, `model.py:107-143`; MLX `:491-567`.

1. Draft produces `block_output_ids = [anchor, d_1..d_{B-1}]`.
2. Target runs **one parallel forward** over the whole block (`model.py:126-132`) and samples `posterior = sample(target.logits)` — its own next-token at every block position.
3. **Acceptance = longest matching greedy prefix**:
   ```python
   acceptance_length = (block_output_ids[:,1:] == posterior[:,:-1]).cumprod(1).sum(1)   # model.py:135
   ```
   Accept draft tokens while they equal the target's argmax; stop at first mismatch (MLX: `accepted = first i where d[i] != t[i]`, `:519`).
4. **Bonus token**: at the first rejected position (or one past a fully accepted block) commit the target's own token `posterior[acceptance_length]` (`model.py:137`; MLX `t_list[accepted]`, `:520`). So each cycle commits `acceptance_length + 1` tokens, always ≥1 and the extra is target-sampled.
5. Advance `start += acceptance_length + 1`, crop target KV cache to `start` (`model.py:138-139`), slice target hidden to the committed window, loop.

**Losslessness:** the committed sequence is exactly what greedy AR decoding of the target would produce, because every committed token either equals the target's argmax (accepted) or *is* the target's argmax (bonus). Output distribution = target's. 

**Caveat / correction on "lossless" at temperature > 0.** The reference `dflash_generate` uses the **same exact-match rule** even when `sample()` draws multinomially (`model.py:48-54, 134-135`). Exact-match acceptance of two independent multinomial draws is **distribution-preserving only in the greedy (temp≈0) limit**; it is not the classic Leviathan/Chen rejection-sampling rule (accept w.p. `min(1, p_target/p_draft)`, resample from the residual). So the *reference PyTorch/MLX loop is strictly lossless only for greedy decoding*; its temp>0 path is a simplified heuristic. Production integrations (vLLM `--speculative-config method=dflash`, SGLang `DFLASH`, `README.md:96-123`) are where proper sampled verification would live — do not assume the reference loop's temp>0 behavior is the distribution-preserving one when porting. For hipfire's greedy/argmax spec-decode path this is a non-issue; flag it only if hipfire ever runs DFlash draft at temperature.

---

## 4. Recurrent / state story (hipfire believes "stateless cross-attention" — partial correction)

**Correction: the drafter is not fully stateless.** It is an attention transformer that carries a **persistent KV cache over the committed sequence across blocks** — but it has *no* recurrent/SSM state.

- The draft keeps `past_key_values_draft` (`model.py:84, 115-120`) / `draft_cache` (`model_mlx.py:451, 498-505`) alive across cycles. Each block feeds only `[last_committed_token, MASK…]` (`model_mlx.py:497`), and the draft's self-attention sees all **previously committed tokens** through this cache.
- After each block the speculative positions are **cropped/trimmed away**: `past_key_values_draft.crop(start)` (`model.py:120`) / trim back to `prompt.size + n - 1` (`model_mlx.py:504-505`). So only committed-token KV persists; rejected drafts are discarded. This is the drafter's cross-block memory.
- **The target context features are NOT persistent draft state.** They are recomputed and re-fed every cycle for the current committed window only (`hidden` sliced to `accepted+1`, `model.py:143`, `model_mlx.py:567`), and re-injected as fresh K/V each forward (`model.py:226-238`). In that sense the *conditioning is per-block / stateless w.r.t. target features* — which is the part of hipfire's belief that is right.

**Net for the port:** the draft needs (a) its own attention KV cache over committed tokens (trimmable, drop rejected block tail each cycle), and (b) fresh injected target-context K/V per block. There is no hidden recurrent state to rewind on the *draft* side.

**Where recurrent-state rewind DOES appear:** on the **target** side when the target is a hybrid model with `GatedDeltaNet` layers (Qwen3.5 hybrid). Those caches are non-trimmable, so on partial acceptance the MLX code replays GDN over the accepted prefix to reconstruct the correct state (`_GDNStateCapture.rollback`, `model_mlx.py:293-397`, esp. `:374-397`). This is exactly hipfire's open "rewind of hybrid recurrent state" item (MEMORY: dflash-spec-decode-correctness) — it's a **target-model** concern, independent of the drafter, and only bites when the *target* has SSM/GDN layers. Sliding-window (SWA) target/draft layers are handled analogously (`RotatingKVCache`, `model_mlx.py:85-91, 110-114, 176`).

---

## 5. Reusable implementation specifics for a Rust/HIP/NPU port

- **Draft layer = standard Qwen3 decoder block**, differences: `is_causal=False`, and K/V are `concat(ctx_kv, token_kv)` before attention (`model.py:194, 226-231`). q/k RMSNorm on head_dim (`model.py:207-208, 225, 232`). GQA groups `num_attention_heads/num_key_value_heads` (`model.py:191`). o_proj + Qwen3MLP + two RMSNorms, pre-norm residual (`model.py:280-299`).
- **Fusion front-end (once per cycle, shared by all layers):** `H_t = RMSNorm(fc(concat(selected_hidden)))`, `fc ∈ R^{hidden × len·hidden}` no bias (`model.py:334`, appendix Eq `:48-56`). Then every layer applies its own `k_proj/v_proj` to `H_t`.
- **RoPE offsets** (subtle — get these right): draft-token queries/keys use offset `cache.offset + S` (S = ctx length), injected ctx keys use offset `cache.offset` (`model_mlx.py:102-104`). Positions for the draft forward span `[past_len : start+block_size]` (`model.py:115`). apply_rotary slices cos/sin to the query length with `[..., -q_len:, :]` (`model.py:176-182`).
- **Block construction:** `[anchor] + [mask_id]*(B-1)`; embed via shared table (× `embed_scale` for Gemma-style, `model_mlx.py:151,165,188`); take logits from positions `1:` only.
- **Cache lifecycle per cycle:** draft — forward whole block, read tail logits, sample, **then trim cache to committed length** (drop the B-1 masked positions), keep committed anchor. Target — forward block, argmax, compute acceptance prefix, commit `accepted+1`, crop KV to new `start`, slice hidden to `accepted+1`.
- **Config fields to read** (`DFlashConfig`, `model_mlx.py:29-49, 217-236`): `hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, rms_norm_eps, rope_theta, block_size, dflash_config.target_layer_ids, dflash_config.mask_token_id, num_target_layers, layer_types, sliding_window, final_logit_softcapping`.
- **Softcap** (Gemma targets): `logits = tanh(logits/cap)·cap` (`model_mlx.py:195-197`).
- **Serving knobs from README:** vLLM `num_speculative_tokens: 15`, SGLang `--speculative-num-draft-tokens 16` — i.e. block_size ≈ 16 in production (`README.md:89-123`).
- **NPU relevance to nix1 LDS hazard:** the draft body is register/GEMM-friendly (attention + MLP over ≤16 tokens + a short ctx window); no large LDS-resident reduction is intrinsic. Keep the injected-ctx concat as a straight K/V append, not an LDS-staged gather.

---

## 6. Recommendations for hipfire's Goals A (disaggregation) & B (streaming MoE)

**A1 — Move `fc`+`hidden_norm` to the target box to cut the LAN payload 8×.** The interface currently ships raw `[rows, 8·hidden]` bf16 (~384 KB/cycle, confirmed). `fc` is a fixed linear→RMSNorm that collapses that to `[rows, hidden]` (~48 KB/cycle). Since the target box already has the hidden states in hand, projecting there before the wire is nearly free and gives an 8× reduction. Accept the version-coupling (draft's `fc`/`target_layer_ids` must match target's projector) — encode both in the artifact metadata. If you want the boxes independently versioned, keep `fc` on the draft and eat 384 KB. This is the dominant Goal-A lever and directly validated by the reference's projector placement (`model.py:317, 334`).

**A2 — Co-locate the shared `embed_tokens`+`lm_head` on the NPU draft box; ship only token IDs.** These are static, so hosting them draft-side keeps draft→target = block token IDs (tens of bytes) and target→draft = hidden only. Otherwise the target must also stream 16 embedding rows/cycle (~128 KB), doubling the return payload. Pin the draft KV cache (committed-token memory, Section 4) on the NPU box too — it's small (≤ committed_len × layers × kv_heads × head_dim) and must persist across cycles with a trim-on-reject step.

**B1 — Maximize τ per verify pass; block_size is free under streaming.** In a streaming-MoE-bound regime, verify cost ≈ one expert/layer stream regardless of how many tokens are in the block (one parallel forward), so throughput ≈ (τ = accepted+1) / pass_time. DFlash's whole advantage is producing a high-τ block in **one** draft forward: reported τ ≈ 7.2–9.1 on Math500/HumanEval for Qwen3.5-27B/35B-A3B (`appendix.tex:102-110`). Push `block_size` toward 16 (production default) and use a deeper draft (5 layers) — draft latency is ~flat in block size (`preliminaries.tex:35-39`), and each accepted token amortizes the fixed stream. Throughput scales ~linearly with τ until acceptance saturates. Extracting the 8 conditioning layers is free during a streaming verify since every layer's hidden is already materialized as it streams.

**B2 — The target-side recurrent-state rewind (GDN) is the real correctness risk for the 397B MoE, not the drafter.** Qwen3.5-397B-A17B is a hybrid (GatedDeltaNet + attention). On partial acceptance you must roll GDN state back to the accepted prefix (replay-over-accepted, `model_mlx.py:374-397`), exactly hipfire's open item. The drafter itself only needs a trimmable attention KV cache (no rewind). Prioritize a correct, cheap GDN prefix-replay on the target box; the drafter disaggregation can proceed independently.

---

## Quick-reference numbers

| Quantity | Value | Cite |
|---|---|---|
| Headline speedup | >6× lossless; up to 2.5× over EAGLE-3 | abstract |
| Draft depth (paper) | 5 layers, block 16 | preliminaries.tex:49 |
| Production block size | 15–16 draft tokens | README.md:89-123 |
| τ (accept len) Qwen3.5-27B / 35B-A3B | 7.7 / 7.2 (Math500), 9.1 / 7.9 (HumanEval) | appendix.tex:102-110 |
| Naive diffusion (no target feat) | only 2–3× | appendix Table naive_diffusion |
| Extract layers → interface width | `len(target_layer_ids)·hidden` bf16, rows = accepted+1 | model.py:39-45,143 |
| hipfire 384 KB/cycle | = 6·8·4096·2 = 384 KiB — CONFIRMED | — |
| `fc` location | inside draft (movable to target for 8× wire cut) | model.py:317,334 |
| Loss weight | `w_k = exp(-(k-1)/γ)`, γ=7/5/4 for B=16/10/8 | method.tex:48-53; appendix.tex:7 |
