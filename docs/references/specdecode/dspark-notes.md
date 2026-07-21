# DSpark — distilled notes

Reference read directly from `/srv/hipfire/references/SpecDecode/dspark/src/`
(DeepSeek-AI **DeepSpec** repo; paper `DSpark_paper.pdf`). Not the hipfire tree.
Citations are `path:line` within that `src/` root, or `paper §` from the PDF
abstract (only the abstract extracted cleanly; body is font-subsetted).

TL;DR: **DSpark is not a new drafter — it is DFlash's block backbone plus two
small heads.** DFlash and DSpark are literally the same PyTorch class
(`Qwen3DSparkModel`); the DFlash config just zeroes the two heads. So for
hipfire, DSpark = the DFlash body you already brought up, + a sequential Markov
head (intra-block dependency) + a confidence head (verify-length scheduling).

---

## 1. What DSpark is

**Paper framing** (`DSpark_paper.pdf` §Abstract): *"DSpark: Confidence-Scheduled
Speculative Decoding with Semi-Autoregressive Generation"*, DeepSeek-AI + Peking
Univ. Two problems it targets in parallel drafters:

1. **Suffix / acceptance decay** — parallel drafters emit a whole block in one
   forward with *no inter-token dependency*, so later block positions reject
   fast. Fix: a **semi-autoregressive** architecture = "coupling a parallel
   backbone with a lightweight sequential module" to add intra-block dependency.
2. **Verification waste** — blindly verifying long blocks burns batch capacity on
   tokens likely to reject (bad in high-concurrency serving). Fix:
   **confidence-scheduled verification** — a head estimates per-position
   *prefix-survival probability* and tailors verify length per request against
   "engine-specific throughput profiles".

Production claim: deployed in the **DeepSeek-V4** serving system; vs the MTP-1
production baseline, **+60–85 % per-user generation speed at matched
throughput**, and shifts the throughput/interactivity Pareto frontier.

**Structure (the three DeepSpec algorithms share one framework).** DFlash,
DSpark, Eagle3 all live in `deepspec/` (`README.md:67-69`). Critically:

- `config/dspark/dspark_qwen3_8b.py` and `config/dflash/dflash_qwen3_8b.py` are
  **identical** except three fields. DSpark: `markov_rank=256`,
  `markov_head_type='vanilla'`, `confidence_head_alpha=1.0`,
  `l1_loss_alpha=0.9 / ce_loss_alpha=0.1`. DFlash: `markov_rank=0`,
  `confidence_head_alpha=0.0`, CE-only (`ce_loss_alpha=1.0, l1_loss_alpha=0.0`).
- Both build the **same** `Qwen3DSparkModel` backbone
  (`config/dspark/qwen3/config.py:38` `architectures=["Qwen3DSparkModel"]`; the
  DFlash config path reuses it). Same 5 draft layers, same
  `block_size=7`, same `target_layer_ids=[1,9,17,25,33]`, same `num_anchors=512`.

So **DSpark neither replaces DFlash nor adds rounds**: it adds two heads on top of
the identical body and swaps CE-only training for TVD-distillation + confidence
BCE.

**Backbone** (`deepspec/modeling/dspark/qwen3/modeling.py`):
- Input `noise_embedding`: `block_size` positions set to `mask_token_id`
  (`151669`) except slot 0 = the real anchor token; `embed_tokens(noise_ids)`
  (`common.py:264` `create_noise_embed`). This is the masked-diffusion / parallel
  block init.
- `target_hidden_states` = concat of target hidden at `target_layer_ids`
  (5 layers, `common.py:52` `extract_context_feature`) → `fc` [5·h→h] →
  `hidden_norm` (RMSNorm) (`modeling.py:373`). This is "main_x", the context.
- Per-layer attention `Qwen3DSparkAttention` (`modeling.py:87-151`): Q from block
  hidden; **K/V = concat([k_proj(main_x) ctx, k_proj(block) noise])**, so the
  block attends over `[target-context KV ++ block KV]`. `q_norm`/`k_norm`
  (k_norm applied to the concatenated K, `:113`), RoPE, **non-causal /
  bidirectional** GQA (`is_causal=False`, `:58`). Single `position_embeddings`
  call shared across all 5 layers (`:374`).
- `draft_logits = lm_head(norm(hidden))` → `[block_size, vocab]`.

**Markov head — the "sequential module"** (`markov_head.py`): three variants,
config uses `vanilla`.
- `VanillaMarkov`: `markov_w1` = `Embedding[vocab, rank=256]` on the *previous*
  token, `markov_w2` = `Linear[rank→vocab]`; adds `markov_w2(w1[prev])` as a bias
  to the base logit (`:26-32`). Memoryless (only prev token).
- At sample time `sample_block_tokens` (`:55-90`) runs a **serial loop over the
  block**: step k adds the markov bias for the just-sampled `prev_token`, samples
  token k, feeds it as prev for k+1. **This serial feedback is the whole
  "semi-AR" mechanism** — it injects x_<k dependency a dependency-free parallel
  drafter lacks. Cost per step: one `[vocab,256]` embed gather + `[256,vocab]`
  GEMV + argmax. No attention.
- `GatedMarkovHead` (`:93`) gates the prev-embedding by `sigmoid(gate_proj([h;
  w1[prev]]))`. `RNNHead` (`:125`) carries a GRU-style recurrent state across
  block positions (`_rnn_step`, `:149`: `new_state = gate*state +
  (1-gate)*cand`) → strongest intra-block dependency. hipfire wires **only
  vanilla** (see §4).

**Confidence head** (`common.py:43` `AcceptRatePredictor` = `Linear[in→1]`; with
`confidence_head_with_markov=True` the input is `[hidden ++ w1[prev]]`, dim
`h+rank`). Trained by BCE against `accept_rate = 1 − ½·Σ|p_draft − p_target|`
(TVD) (`loss.py:60-70`). At eval, `sigmoid` → per-step acceptance; `cumprod` →
**prefix-survival**; the proposal is truncated at the first position whose
survival < threshold (`draft_ops.py:82-93` `_confident_prefix_length`, default
threshold 0.5). A full calibration harness (ECE/AUROC/Brier reliability) is in
`eval/dspark/confidence_head.py`.

---

## 2. Draft ↔ target interface (Goal A: disaggregation)

The speculative loop is `eval/base_evaluator.py:307` `generate_decoding_sample`;
the DSpark hooks are in `eval/dspark/evaluator.py`.

**Per round, target → draft:** the target model runs with
`output_hidden_states=True`; the draft reads `extract_context_feature(hidden,
target_layer_ids)` = the hidden states at the **5 intermediate layers**
`[1,9,17,25,33]` for the newly accepted tokens. After each verify,
`_update` (`evaluator.py:134-147`) sets
`context.target_hidden_states = verified_target_hidden[:, :accepted+1, :]` — i.e.
the target ships back **`(accepted+1) × 5 × hidden`** raw hidden values (bf16).
This is the *only* heavy cross-machine payload and it is **identical to plain
DFlash** — both consume the same 5-layer `target_hidden`.

**Per round, draft → target:** `build_dspark_proposal` (`draft_ops.py:96`) returns
`verify_input_ids` = `[anchor_token ++ draft_tokens[:proposal_len]]` and
`draft_probs` (for rejection sampling, `base_evaluator.py:240-258`). With the
confidence head, `proposal_len` is already **trimmed** to the confident prefix
(`draft_ops.py:127`) — so the block the target must verify can be *shorter than
block_size*, decided on the draft side.

**DSpark vs DFlash for disaggregation:** the wire format for the expensive path
(target_hidden read, tokens+probs back) is unchanged. The *additions* are both
draft-side and cheap:
- the **Markov head** = a serial `block_size`-step loop of tiny GEMVs on the
  draft box (no target interaction);
- the **confidence head** = a `[1, h+rank]` GEMV per position, letting the draft
  box *pre-trim* the block so the target verifies fewer tokens.

Net: DSpark makes the disaggregation interface **strictly better** than DFlash —
same target read, but the target receives a shorter, higher-survival block and
does less verify work. All new logic (sequential markov, survival scheduling)
lives on the draft/NPU side.

> Note on hipfire's "~384 KB/cycle" target_hidden figure: the actual payload is
> `(accepted+1) × len(target_layer_ids) × hidden × 2 B` (bf16), i.e. it scales
> with *accepted length*, not a fixed block. For 5 layers × 4096 × 2 B that is
> ~40 KB **per accepted token**; a full `block_size=7` window ≈ 287 KB, ~8 accepted
> ≈ 328 KB. Reconcile the constant against `accepted+1`, not `block_size`.

---

## 3. Accepted-tokens-per-pass (Goal B) and tree interaction

**Does DSpark raise accepted-tokens-per-verify?** Yes, two independent levers:
1. **Markov head → higher deep-position acceptance.** By adding x_<k dependency
   it directly fights suffix decay, so `accept_rate@k` for larger k rises →
   longer accepted prefix per verify. Metrics tracked per position:
   `accept_rate@{pos}` (`loss.py:192`), `verify_rate = accepted/(proposal+1)`,
   `acceptance_length` (`base_evaluator.py:482-489`).
2. **Confidence scheduling → higher throughput per verify capacity.** It does not
   lengthen the accepted prefix; it *removes* low-survival tail positions so the
   verify pass isn't spent on tokens that will reject. In a **verify-bound /
   batched** regime this is the throughput driver — exactly the paper's "+60–85 %
   at matched throughput" and its "verification waste" thesis (§Abstract).

**Interaction with a DDTree — they STACK, and DSpark's heads are the ideal tree
signals** (they do not compete for the same role):
- As shipped, DSpark proposes a **single linear chain** of `block_size` tokens
  (`sample_block_tokens` emits one token per position). No tree.
- But the Markov head produces a proper **per-step conditional**
  `p(x_k | x_<k, block-hidden)` = `softmax(base_logit_k + markov_bias(x_{k-1}))`.
  That is precisely the distribution a tree wants to branch on at each node —
  branching *these* corrected logits gives a token tree with real inter-token
  dependency at every node, far better than branching a dependency-free parallel
  drafter's marginal logits.
- The **confidence head's** per-step / cumprod survival is a natural
  **tree-pruning and verify-budget** score (which paths to keep, how deep to
  verify).
- The only competition is compute: the markov loop is **serial**; a wide tree
  multiplies markov-head evals per position. Favor narrow-deep trees so the
  serial head cost stays bounded (esp. on the NPU). RNN head > vanilla for deep
  tree nodes because it carries state along a path.

So the clean composition is: **DDTree provides the branch/verify structure,
DSpark's Markov head provides each node's conditional, DSpark's confidence head
provides per-node survival for pruning + scheduling.**

---

## 4. Mapping to hipfire's existing DSpark code

hipfire has already built most of this (DFlash Phase A–E). Mapping:

| Reference | hipfire |
|---|---|
| `Qwen3DSparkModel._forward_backbone` (`modeling.py:361`) | `dspark_qwen3_block_forward` in `crates/hipfire-arch-llama/src/dspark_body.rs` (docstring cites `modeling.py:99-116,373,386`) |
| 5-layer body, dim 4096 / FFN 12288 / GQA 32×8 / head_dim 128 / qk_norm / rope 1e6 | `Qwen3DrafterAssets.config` (`dspark_body.rs:76-77`) — **exact match** |
| `fc` [5·h→h] + `hidden_norm` | `main_proj` [dim, 5·dim] + `main_norm`; `main_proj_ingest[_batched]` in `crates/hipfire-specdecode-dspark/src/dspark_core.rs:497,553` |
| `markov_w1`/`markov_w2` | `DsparkWeights.markov_w1/markov_w2` (`dspark_core.rs:192-193`); `run_heads` does embed-gather + `[vocab,rank]` GEMV + argmax |
| `AcceptRatePredictor` (in = `h+rank` when `with_markov`) | `confidence_proj` `[1, dim+rank]` + `confidence_bias` (`dspark_core.rs:194`; `dspark_body.rs:50` notes qwen3 HAS bias, deepseek4 does not) |
| `_confident_prefix_length` threshold 0.5 | `DsparkDrafter::mtp_step` `conf_threshold` default 0.5 (`tools/npu/dspark_ref.py:34`) |
| head epilogue golden | `tools/npu/dspark_ref.py` (`run_heads`), NPU kernels `tools/npu/dspark_heads_npu.py` (Phase E) |

**Corrections / gaps to hipfire's assumptions:**

1. **aiecost frames DSpark as pure block-diffusion and undercounts the serial
   heads.** `tools/npu/aiecost/design.py:224-226` describes the drafter as a
   "block/masked-DIFFUSION drafter … denoises the whole block in ONE
   bidirectional (non-causal) forward, so M = block_size." That is correct for
   the **backbone** but **omits DSpark's semi-AR Markov head**, which is a
   *serial* `block_size`-step loop (embed-gather + rank-GEMV + argmax per slot,
   each feeding the next) sitting on the critical path. The single-forward
   `M=block_size` cost model therefore *understates* DSpark decode latency by that
   serial head chain. `dspark_heads_npu.py` does model the heads separately and
   flags large padding waste (confidence `[1,1280]→[32,1280]`, 31/32 wasted; N
   padded to 16, 15/16 wasted) — fold that serial + padded cost into the
   `drafter` estimate for DSpark (vs DFlash which genuinely has no heads).

2. **hipfire wires only the `vanilla` Markov head.** The reference also ships
   `gated` and `rnn` (`markov_head.py:93,125`). `rnn` carries recurrent state
   across block positions and is the strongest intra-block model — a drop-in
   upgrade (`joint_proj[2r+d → 3r]` + serial recurrence) if draft quality needs to
   rise, especially for tree nodes.

3. **block_size:** reference default is **7** (`config/dspark/dspark_qwen3_8b.py`,
   and `dspark_body.rs`); `aiecost/design.py` defaults `block_size=4`. Use 7 for
   DSpark-parity cost runs.

4. **`target_layer_ids` must exclude the final target layer**
   (`base_evaluator.py:100-112`): HF `output_hidden_states` stores the *normalized*
   final hidden there, which mismatches the raw decoder-layer outputs used to
   build the target cache. For disaggregation the target box must export layers
   `[1,9,17,25,33]` (Qwen3-8B/36L), not the last.

---

## 5. Reusable implementation specifics (Rust/HIP/NPU)

- **Body forward:** bidirectional (non-causal) GQA over `[ctx KV ++ block KV]`;
  ctx K/V = `k_proj/v_proj(main_x)`, block K/V = `k_proj/v_proj(noise_hidden)`;
  `k_norm` on the concatenated K (`modeling.py:113`); RoPE positions: ctx =
  `arange(seq)`, draft = `anchor_pos + [0..block_size)` (`common.py:251`); one
  shared `position_embeddings` for all layers. Anchor slot 0 = real last-accepted
  token, rest = `mask_token_id` (`common.py:264`).
- **Markov (vanilla) numerics:** `bias = markov_w2 @ (markov_w1[prev_token])`;
  add to base logit before argmax; serial feed of the sampled token. GPU/NPU-
  resident chain is viable (token id stays on device) — hipfire's `dspark_core.rs`
  already has a device-token embed path (`markov_w1_device_embeddable`, `:297`).
- **Confidence numerics:** predict `logit(accept_rate)`; `accept_rate = 1 − TVD(
  p_draft, p_target)` (`loss.py:69`); eval survival = `cumprod(sigmoid(·))`;
  truncate at first `< conf_threshold`.
- **Training recipe** (only if retraining/fine-tuning a sidecar):
  `ce_loss_alpha=0.1`, `l1_loss_alpha=0.9` (TVD distillation dominates),
  `confidence_head_alpha=1.0`, `loss_decay_gamma=4.0` (exp position weighting,
  earlier positions weighted more, `loss.py:33-36`), `num_anchors=512` sampled
  training blocks per sequence (`common.py:123`). Data pipeline caches target
  hidden — huge (~38 TB for Qwen3-4B, `README.md:29`); hipfire imports the
  released `deepseek-ai/dspark_qwen3_*_block7` checkpoints instead.
- **NPU head kernel shape traps** (already discovered, `dspark_heads_npu.py`):
  int8 GEMM needs `m % (4·r)==0`, `n % (2·t)==0` → confidence `[1,·]` padded to
  `[32,·]`, activation N padded to 16; free-running argmax chain cascades one flip
  into all later slots, so validate teacher-forced to isolate per-slot error.

---

## 6. Recommendations — DDTree + DFlash + DSpark on streaming MoE + disaggregation

1. **Stack, don't choose: DDTree = structure, DSpark heads = signals.** Feed the
   Markov head's corrected per-step logits as the tree **expansion** distribution
   (real conditionals, not marginals) and the confidence head's cumprod survival
   as the tree **pruning + verify-depth** score. Keep the tree **narrow-deep** to
   bound the *serial* Markov loop; use the **RNN** Markov head (not vanilla) for
   tree nodes since it carries per-path state. This turns DFlash's dependency-free
   block into a dependency-aware tree with almost no extra target interaction.

2. **Streaming-MoE (Qwen3.5-397B-A17B) is verify-bound → schedule with the
   confidence head.** Trim each proposal to survival ≥ threshold **before** it
   crosses to the target box, so every streaming verify pass carries only
   high-survival tokens = more accepted-tokens per streamed-expert-load. Tune
   `conf_threshold` to the engine profile (the paper's "engine-specific
   throughput profiles"): raise it when a verify pass is expensive (MoE weight
   streaming) — trade a little draft length for much less wasted verify. This is
   the single highest-leverage knob for Goal B.

3. **Disaggregation: put both heads on the Phoenix NPU, keep the target read
   DFlash-identical.** The wire stays `5×hidden` of accepted tokens out, trimmed
   block + `draft_probs` back — unchanged from DFlash. Run the (already
   Phase-E-validated) Markov + confidence heads on the draft/NPU box so the big
   GPU box only emits intermediate hidden and verifies a short, pre-scheduled
   block. Before trusting split economics, **fix the aiecost `drafter` model to
   add the serial `block_size`-step head chain (and its padding waste) to the
   critical path** — currently only the single bidirectional body forward is
   costed, which flatters DSpark vs the headless DFlash body.
