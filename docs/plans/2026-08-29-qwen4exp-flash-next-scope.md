# Qwen3.8-Flash-Next (`qwen4_exp`) — support scope

Artifact: `/home/sadara/.hipfire/models/models--Qwen--Qwen3.8-Flash-Next.hfa`
(238 GB HFAR0002, 131 shards, 1658 tensors, **360 GB bf16 logical**).
`model_type: qwen4_exp`, `Qwen4ExpForConditionalGeneration`. Proposed `ARCH_ID_QWEN4EXP = 26`
(25 is the current high-water mark in `hipfire-arch-api/src/lib.rs:104`).

Everything below was read out of the archive's own index and shard headers, verified against
files in this tree, or read from the vendored reference implementation at
`third_party/transformers-qwen4_exp/` (transformers `5f8ab9bb`, 2026-08-28). Where a claim is
an assumption it says so.

> **Revised 2026-08-29 after vendoring the reference.** Four questions this document
> originally listed as open are now settled, and two inferences it made from the artifact
> alone were **wrong**: the n-gram hash is an **XOR**, not a sum (§4), and the PLE conv is
> **dilated**, so our existing conv1d kernel does not cover it (§2.5). Both are corrected
> below. `ple_layer_ids` being 1-based also explains, rather than contradicts, the `layers.1`
> tensor naming.
>
> **Revised again after reading the merged llama.cpp port** (ggml-org/llama.cpp
> [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), merged 2026-08-28). It
> independently confirms the XOR hash and most of this plan's structural calls, hands us a
> **free exact test oracle** for QSA (§8), and supplies field evidence on a 128 GB
> unified-memory box that moves decision 3. See §8. Our `third_party/llama.cpp` checkout is at
> `c15c5c77a` and predates the port — `git fetch` it before reading the source.

---

## 1. What this model is

48 layers at hidden 2560, vocab 248320, untied head. The stack is
`12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`, and every layer
reads and writes a **4-wide residual stream** (10240 = 4 × 2560) instead of a plain
pre-norm residual. Five things are new relative to anything we serve today, in descending
order of how much they cost us:

1. **MoE at 512 experts / top-10 / `moe_intermediate_size` 640** — 241.6 GB of the 360.
2. **N-gram embedding**, 102.4 GB bf16, injected at one layer, 16 hash heads over a 320 M-row table.
3. **Qwen Sparse Attention (QSA)** — a lightning indexer selecting 512 micro-blocks of 4 tokens (budget 2048).
4. **Gated residual** — 4 streams mixed through a rank-320 bottleneck, per block, twice.
5. **Embedded MTP head** — a full QSA+indexer+MoE+HC block, 5 GB, in-artifact.

Plus a Qwen3-VL-shaped vision tower (depth 27, hidden 1152, no deepstack) and interleaved
mrope with section `[11,11,10]`.

Budget on halo (128 GiB UMA ≈ 137.4 GB, minus GTT tax):

| Group | bf16 | at oq4-ish | note |
|---|---:|---:|---|
| MoE experts (48 layers × 512, + MTP) | 241.6 GB | ~64 GB | the whole quant question |
| **N-gram table** | **102.4 GB** | 69.6 GB | **stays on drive, losslessly packed; see §4** |
| GDN + QSA + norms + HC mixers | ~13 GB | ~4 GB | HC mixers are 634 M dense params |
| embed + lm_head (untied) | 2.5 GB | 2.5 GB | keep bf16 |
| vision tower | 0.9 GB | 0.9 GB | |

Text-only, experts at 4-bit, n-gram on drive ≈ **70 GB resident**. KV is cheap: only 12
layers have full attention, 2 KV heads × 256, so 24 KB/token → 6.4 GB at the native
262144 context, plus 0.8 GB of indexer K. GDN recurrent state is 113 MB at f32 (56 MB at
the f16 state that measured +6.7% on the 27B). It fits, *if* §3.1 is solved.

---

## 2. What already exists

This is the important section. The model is much less alien to this tree than its config
suggests — it is roughly **deepseek4's residual + indexer + MTP skeleton** × **qwen35's
GDN** × **qwen35-vl's tower**, plus one genuinely new table.

### 2.1 Gated DeltaNet — `reuse-with-extension`, closest to free

**Qwen3.8-27B already serves with this model's exact GDN geometry.** `linear_num_key_heads`
16, `linear_num_value_heads` 48, key/value head dim 128, `linear_conv_kernel_dim` 4,
`full_attention_interval` 4 — measured 63.9 tok/s at DFlash B=10
(`docs/plans/2026-08-25-qwen38-27b-next-phases.md:1353`, geometry at `:1371`).

- The split projections `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` are
  **hipfire's native form**, not a remap. `grep -rn 'qkvz|in_proj_ba' --include=*.rs crates/`
  returns zero weight-name hits. HF's fused `in_proj_qkvz`/`in_proj_ba` is the outlier;
  this checkpoint matches us. Loader at `hipfire-arch-qwen35/src/qwen35/loading.rs:4851-4924`
  matches all nine `linear_attn.*` tensors 1:1 by shape.
- K:V ratio 3 (48/16) needs no code: `repeat_interleave_qk_f32` materialises Q/K up to V-head
  count at eleven call sites (`decode_layers.rs:683`, `lowered.rs:486`, `ep.rs:661`, …);
  ratio 2 and 3 are both in production. Kernel is generic (`kernels/src/repeat_interleave_qk.hip:20`).
- Nine GDN recurrence kernel variants, all `#define HD 128` — the target's head dim.
  `conv1d_silu_split.hip` hardcodes kernel size 4 and takes `k_dim`/`v_dim` separately, so
  the asymmetric 2048/2048/6144 split of the 10240 is already expressible.
- GDN tape / rollback (`speculative.rs:2286`) derives everything from config; 36 GDN layers
  is *smaller* than the 27B's 48.

Two real gaps, both small:

- `output_gate_type: sigmoid`. `grep -rn output_gate_type` → **zero hits repo-wide**.
  `kernels/src/gated_norm.hip:35-36` hardcodes swish. One config field, one kernel branch.
  Confirmed by the reference: it feeds `config.output_gate_type or config.hidden_act` straight
  into the gated-RMSNorm's activation, so it names exactly that gate and nothing else.
  (Distinct from `attn_output_gate`, which we do parse and which controls the *full-attention*
  gate — the two keys don't collide.)
- **This checkpoint has no `input_layernorm` and no `post_attention_layernorm`, anywhere.**
  Grepping all 1658 tensor names for "layernorm" returns only `self_attn.indexer.{q,k}_layernorm`.
  There is also no final `model.language_model.norm`. `hc_norm [10240]` replaces both per-block
  norms; the model-level `hyper_connection_mixer.hc_norm` is the only pre-lm_head norm. Our
  `DeltaNetLayerWeights` loads both missing names at twelve sites in `loading.rs`.

### 2.2 Gated residual / hyper-connections — `reuse-with-extension`

**deepseek4 runs a 4-wide residual stream today, default-on, with `hc_mult = 4` — the
target's `hc_count`.** Fourteen `hc_*` kernels with batched twins, prefill scratch, MTP
integration, expert-parallel integration, and three tiny fixtures in the affected-gate.

| target tensor | deepseek4 equivalent |
|---|---|
| 4-wide `residual_streams` | `forward.rs:5568-5601` (alloc + broadcast init) |
| read side (collapse 4 → 1) | `hc_input_map_4stream`, `kernels/src/hc_input_map.hip:22` |
| write side (mix + inject) | `hc_mix_4stream`, `kernels/src/hc_mix_4stream.hip:34` |
| rank-N control generator | `hc_compute_control`, runtime `n_ctrl`/`x_dim` params |
| pre-lm_head collapse | `hc_head_compute_pre` + `hc_input_map_4stream` |

What differs is the *mixing function*, not the plumbing. DS4 generates 4 per-stream
**scalars** through a Sinkhorn-normalised 4×4; the target generates a per-channel
`[10240]` mix through `down[320,10240] → up[10240,320]`. So:

- read side needs `w[s*hidden + d]` instead of `a_vec[s]` — two ~40-line elementwise kernels.
- the rank-320 expand is a plain `[10240,320]` GEMV via `weight_gemv`. **Do not** reuse
  `gemv_lowrank_moe_expand`: hard rank cap of 32.
- force the 4×4 cross-stream term to identity, or fork a no-mix variant. Do not port Sinkhorn.
- the HC mixer weights are **634 M dense params read every token** (~1.27 GB/token at f16).
  They must go through `WeightTensor` + `weight_gemv` and get a `PrecisionClass` in the spec
  crate — not DS4's `upload_global_raw`.

**`block_inject_weight` is settled** — it is `nn.Linear(hc_count·hidden, hc_count)`, so 4
per-stream **scalars**, and `hc_mix_4stream`'s `scale[4]` shape is reuse-as-is. The exact
reference math, all on the RMSNorm'd streams:

```
normed  = hc_norm(streams)                                  # grouped, see below
mix     = sigmoid( up( silu( down(normed) / hc_count ) ) )  # [hc_count, hidden], per-channel
mixed_in = mean_s( mix[s] * normed[s] )                     # MEAN over streams, not sum
inject  = 2 * sigmoid( block_inject_weight(normed) / hc_count )   # 4 scalars in (0,2)
```

There is **no 4×4 cross-stream matrix at all** — the identity-forcing recommended above is
correct in effect, and the read-side gate is a sigmoid, not a Sinkhorn row-stochastic mix.

One fact the artifact could not show, which touches every `[10240]` norm in the model:
**`Qwen4ExpTextRMSNorm` is a *grouped* RMSNorm** (`group_size = hidden_size`). `hc_norm[10240]`
normalizes each 2560-wide stream **independently**, not over the full 10240. Same for
`ple.norm_query` / `norm_key` / `norm_conv`. A flat `rmsnorm_f32` over 10240 is wrong.

Note: do **not** widen `Qwen35Scratch` — that inflates every existing Qwen3.5 model. Add a
separate `residual_streams` tensor in the new arch's state, as DS4 does.

### 2.3 Sparse attention — `substantial-new-work`, but on top of a lot

- **deepseek4 has a complete, live lightning-indexer pipeline**: score → top-k → gather →
  fused joint-softmax sparse attention (`forward.rs:1374` decode, `:6649` batched).
- **PFlash has block-granular selection over the generic `KvCache` at five KV dtypes**
  (`kernels/src/pflash_score_q8_kv.hip:27`, `hipfire-arch-qwen35/src/pflash.rs:701`) — block-mean
  score, descending rank, block-id → span expansion, budget + anchors. This, not deepseek4,
  is the closest prior art for QSA's micro-block selection.
- Partial rotary at **exactly** the target's geometry — rotary dim 64 of head_dim 256 —
  `config.rs:85`, `kernels/src/rope_partial_interleaved.hip`.
- Doubled Q projection carrying Q interleaved with an output gate (`attn_output_gate`,
  auto-detected from the HFQ) — matches `q_proj [12288,2560]` = 6144 Q + 6144 gate.
- GQA at head_dim 256 with slot gather: `kernels/src/attention_cold_slots.hip` (CHD 256).
- `attention_swa_gqa_batched.hip` is a completed GQA port of deepseek4's SWA kernel — the
  sibling that proves the GQA-ification is mechanical.

**QSA's exact semantics, from the reference** (`Qwen4ExpTextQSAIndexer`):

```
block_topk = indexer_budget / indexer_compress_ratio = 2048 / 4 = 512 blocks
q          = rope( q_layernorm(q) )                     # 4 heads × 128, current positions
block_key  = rope( k_layernorm( mean(K[4 tokens]) ) )   # rope at the block's FIRST position
score[b]   = sum_h relu( q[h] · block_key[b] ) / sqrt(128)
select     = topk(score, 512) → expand each block to its 4 tokens
           + the ragged tail (the < 4 tokens past the last complete block) ALWAYS visible
```

So the budget is 512 blocks *and* 2048 tokens — consistently, one implies the other. ReLU
then a **plain sum over the 4 query heads**, no learned per-head weight, confirming the
`weights_proj` asymmetry with deepseek4. Two details worth pinning: keys are **mean-pooled**
over the block and RoPE'd at the block's **first** token position, and the always-visible
ragged tail is unconditional. The reference emits a *mask* ANDed into the causal mask; we will
gather instead, but the mask defines the semantics.

Gaps: selection must become **per-query, on-GPU, non-destructive** (PFlash selects once at
prefill and permanently drops losers; its rank/expand half runs on the host). `MAX_TOTAL 1024`
caps the live sparse kernels. The indexer scorer asserts `n_idx_heads == 64` (target: 4) —
a 4-head indexer wants a GEMV shape, not one-thread-per-head. Top-k is O(N·K) selection sort
on decode and O(N²) batched, both written for N ≤ 2048. And deepseek4's selection runs over a
*learned-compressed* cache (gated-softmax-pooled), where this one is a plain **mean** over
raw KV — reuse the kernel skeletons, not the pipeline. Note the reference indexer is a
double Python loop over batch × query: it is a semantic oracle, not a perf model.

**mrope is parsed and logged but never consumed.** `mrope_interleaved` / `mrope_section`
land in `Qwen35Config` (`config.rs:93-94`) and are only printed (`loading.rs:4393`) or copied
into MTP sidecar metadata. A whole-tree grep finds five files, none of which compute anything.
This is silent-wrong-output class: text-only prompts have t==h==w, so it looks fine until
the first image.

### 2.4 Vision — closest to `reuse-as-is` of any axis

`hipfire-arch-qwen35-vl`'s config parser defaults match the target **exactly**: depth 27,
heads 16, intermediate 4304, patch 16, temporal_patch 2, spatial_merge 2,
`num_position_embeddings` 2304, `out_hidden` falling back to `text_config.hidden_size`
(`qwen35_vl.rs:40`). LayerNorm-with-bias block layout, merger `[4608,4608]`/`[2560,4608]`
derived not hardcoded, patch_embed conv3d consumed as a flat `[1152,1536]` linear,
`fast_pos_embed_interpolate` with HF semantics, 2D vision RoPE, full-tower GPU forward.
`extract_patches` is numerically validated against HF (rel-L1 0.35 → 0.002).

deepstack is absent from the whole repo — and the target's `deepstack_visual_indexes` is
empty, so that is a match, not a gap.

Real gaps here are the *request plumbing*, not the tower: multi-image/video is complete only
on the gemma3-vl path (`generate_vl.rs:990`); VL prefill is per-token because
`forward_scratch_embed` isn't batched; the tower is validated on gfx1100/1101/1102 only and
warns on gfx1151 — establish a baseline on halo before trusting any grounding result.

### 2.5 PLE — implemented, for a different model

Gemma 4's PLE is real (`hipfire-arch-gemma4/src/weights.rs:94`, forward at `forward.rs:1175`):
per-layer embedding table, model projection, per-layer merge. Different semantics from the
target's cross-gate, so write it fresh. The reference math:

```
key   = grouped_norm( key_proj(ngram_emb) ).view(hc_count, hidden)
query = grouped_norm( streams ).view(hc_count, hidden)
gate  = (key * query).sum(-1) / sqrt(hidden)         # one scalar per stream
gate  = sign(gate) * sqrt(|gate| clamped to ≥1e-6)   # signed sqrt
out   = sigmoid(gate) * value_proj(ngram_emb)        # broadcast over hidden
out   = out + silu( dilated_depthwise_conv1d( grouped_norm(out) ) )   # width 10240
```

**Correction to this document's first draft:** `ple.conv1d` is **not** a plain kernel-4
depthwise conv. It has `dilation = ngram_size = 3`, giving a state length of
`(4-1)*3 = 9`, and the reference explicitly notes it cannot use the standard conv path for
that reason. `hipfire-rdna/src/dispatch/conv1d.rs` does not do dilation — this is a new
kernel (or a strided-gather variant of the existing one), not a reuse. Small, but it is work,
and the GDN `conv1d` in the same model *is* undilated, so the two must not share a path.

**Do not inherit gemma4's `needs_reference` flag** (`forward.rs:1929`): its PLE forces
`forward_step_reference` and skips the lowered superop path entirely. That is direct evidence
the lowered-executor cost of a new mid-stack op is real, and it was already paid once.

### 2.6 MTP and onboarding

deepseek4 has an in-base-HFQ `mtp.0.*` path (`arch.rs:1195-1210`) — the working precedent for
an *embedded* MTP head, which is what this artifact ships. `mtp_extract` round-tripping is
pure overhead here. The quantizer currently **skips `mtp.` tensors**; that's a one-line fix
that exposes the real problem, which is that the MTP block needs the trunk's per-tensor
policy, not deepseek4's flat Q8/F16.

Onboarding is mechanical but wide: a `-spec` crate from `hipfire-arch-template-spec`, a
serving crate, five registry edits (`hipfire-arch-specs`, `hipfire-archs` migration ledger,
`hipfire-serving-core` dep + force-link), a `[[arch]]` row plus seven `[[gate]]` rows in
`docs/model-support.toml`, and the tiny-fixture chain across seven files. Two traps:

- the force-link (`use hipfire_arch_<fam> as _;`) fails **silently at runtime**, not at compile time.
- `fixture_golden.rs:203-205` ends in a catch-all `_ => run_qwen35_golden(...)`, so a missing
  arm silently scores the new model through Qwen3.5's golden path instead of failing.
- `crates/hipfire-arch-template/README.md:52-58` and `docs/architecture-ids.md:70-72` both
  tell the porter to edit `crates/hipfire-quantize/src/main.rs`, which does not exist. The
  README contradicts itself two lines earlier. Fix while you're there.

---

## 3. The genuinely new work, by risk × size

### 3.1 MoE at k=10, `moe_intermediate_size` 640 — **M–L, downgraded from XL by Stage 1**

**Stage 1 ran on 2026-08-29 and this section is now measurement, not inference.** The probe
fixture is `Qwen35Tiny::moe_mi640_k10_preset` (`hipfire-arch-qwen35-spec/src/lib.rs`), emitted
as `qwen3_5_moe_mi640_k10`: 12 experts, top-10, `moe_inter = 640`, `shared_inter = 640`,
14.43 M params, ~2 s end to end, no GPU.

**Confirmed: the quantizer mis-formats this geometry, silently.** Same command, same
`--format oq4`, exit 0, no warning, on two fixtures differing only in `moe_inter`:

| fixture | `gate_up` K | format | `down` K | format |
|---|---:|---|---:|---|
| `qwen3_5_moe_indexed` (mi 512) | 256 | `Oq4G256` (34) | 512 | `Oq4G256` (34) |
| `qwen3_5_moe_mi640_k10` | 256 | `Oq4G256` (34) | **640** | **`HFQ4G128` (7)** |

At a `K` that is not a multiple of 256 the routed `down_proj` leaves the Opus family entirely
for `HFQ4G128` — a different kernel family — while its own `gate_up_proj` stays Opus. One
expert, two halves wanting different GEMV families. Filed in `BUGS.md`. (The summary's
`Mean quant error: 0.00000000` is **not** a symptom: it reads 0 on the known-good `indexed`
preset too. I flagged it as suspicious before running the control; the control refuted it.)

**The `k = 10` half is not XL, and the survey's estimate was wrong about why.** `K_TOP` is a
**runtime kernel argument** in every indexed routed-expert GEMV — a grid dimension and a
stride, nothing more:
`gemv_oq4g256_moe_down_k8_indexed_batched_expanded.hip:28`,
`gemv_oq8g256_..._expanded.hip:21`, `gemv_oq4g256_moe_gate_up_indexed_batched.hip:25`,
`moe_down_combine_k8_batched.hip:26`. The `k8` in those names is inherited from the kernel
they were ported from. **The expert compute path is already k-generic.** The only genuine
compile-time `k` is in the two top-k *selection* kernels — `moe_softmax_topk_k8.hip:27` and
`moe_topk_renorm_k8.hip:21` — which touch no expert weights and no layout. So this half is
those two kernels plus threading `k` consistently through `INDEXED_MOE_K_TOP`,
`oq_indexed_admissible`, `use_gpu_topk`, and the loader repack. That consistency requirement
is still real and still the dangerous part — four sites must agree or a layout mismatch serves
silent garbage, which has already produced measured NaN on a top-2 fixture
(`families/moe.rs:112-124`).

**The `mi = 640` half has far more prior art than the survey found.** Every piece of a G128
Opus path exists somewhere:

- `QuantType::OqPlusCompactG128 = 52` — a G128 Opus format already in the enum.
- `kernels/src/gemv_mq4g128.hip` — **FWHT-128 already exists**; the 128-wide rotate is not new.
- `kernels/src/gemv_oq_compact_grouped.hip:76` — already group-parameterized:
  `per_lane = group >> 5; // positions per lane: 8 (G256), 4 (G128)`.
- `kernels/src/gemm_oq_compact_grouped_wmma.hip:26` — serves "G256 and OqCompactG128 (G=128,
  header 66) from one body".
- `kernels/src/gemv_paro_q4g128_moe_{down,gate_up}_k8_indexed_batched.hip` — the **indexed MoE
  kernels already ported to G128** for the ParoQuant layout, and the header calls it a
  "mechanical G128 port of" its G256 sibling.

So the gap is an Opus-G128 indexed MoE pair, with a proven-mechanical precedent for exactly
that port on a sibling format, against a rotate and a grouped-GEMV that already handle G128.

**Correction to the exit options.** The survey offered "a group size that divides 640 — 128
(5 groups) or 160". **160 is not viable**: the Opus rotate is an FWHT, which is defined on
powers of two only (`fwht_groups`, `dispatch/mod.rs:144`, hardcodes 256). `640 = 2^7 x 5`, so
the largest power of two dividing it is **128**, and 128 is therefore the only choice, not one
of two.

**Stage 1b, GPU half (2026-08-29) — ran, then was adversarially verified, and the headline
claim did NOT survive. Read this before trusting any number in it.**

*Established, and only this:* **the admission guard refuses correctly, and the refusal is a
property of shape+k rather than of the env flag.** Proved three independent ways —
`HIPFIRE_QWEN35_MOE_OQ_INDEXED={1,0}` is a **bit-identical no-op** on the probe
(`logit_hash 0x7b91c68f2d8b56be` both ways) while it moves the control
(`0x00aae836…` → `0x9d9c1f30…`); the probe's kernel trace contains **zero** `*_k8_indexed*`
dispatches where the control's has `gemv_oq4g256_moe_gate_up_k8_indexed_batched` x16; and the
loader skips the repack and says so 24x by name (`loading.rs:6558`). Separately, the CPU-top-K
fallback executor is sound **at k=8** (forcing the control onto it reproduces argmax 30/30,
`mean_kld` moves 1.4e-7). And decode on the probe does not crash or NaN **with experts
resident**. That is the entire list.

*Retracted:* **the "73.3% vs 43.3% top-1 agreement" figure this document previously cited is
withdrawn — the metric cannot fail.** Both arms of that comparison run the *same* non-indexed
routing code, so a top-10 selection or renorm bug is common-mode and cancels exactly. Three
null tests: a full kernel-family swap on the control moved **0/30 argmax tokens in 6/6
seed x len cells** while merely changing `--seed` moves the same number 26.7% -> 50.0%
(signal:noise = 0:23 points); the metric is **inverted in bit count** — on the control, 8-bit
`Q8F16` scores 36.7% against its own bf16 while 4-bit `HFQ4` scores 60.0%; and 73.3/43.3 was
the widest cell of a 48-cell sweep whose ordering reverses in 4/24 `tf` cells. In `ar` mode
both fixtures sit at the 0-3.3% chance floor. **Whether the fallback computes top-10 correctly
at all is still unmeasured**, and this is structurally the llama.cpp 8.44e-08 failure mode
(§8) — a plumbing check wearing a correctness check's clothes.

*Retracted:* **"performance, not correctness" is wrong on three counts,** each verified:

1. **Calibration coverage is silently lost** (see `BUGS.md`). `supports_g256 = inner_k % 256
   == 0` (`cli.rs:10841`) gates the Opus arm at `cli.rs:11081`, the only route into
   `quantize_hfq_source_tensor` — which "applies AWQ/LDLQ and contributes to strict Hessian
   validation". At mi=640 every routed `down_proj` skips it for plain RTN
   (`quantize_hfq4g128`, `cli.rs:11183`), with no `awq_pre_scale_weights` where the MQ4 branch
   above it has one. **`oq4++ --hessian` would ship uncalibrated routed down-weights, exit 0.**
2. **Batched prefill is excluded permanently**, and as a correctness hazard rather than a
   slowdown: `profile=Invalid` (`qwen35/mod.rs:1655`), `HFQ4G128` absent from the
   `moe_prefill_quant_family` ladder (`:2597-2628`), and under daemon-default KVarN the runtime
   prints its own *"Output from this point is not trustworthy"*. `fixture_golden` dodged this
   only by hardcoding Q8 KV (`fixture_golden.rs:328`).
3. **Paged experts have no decode route at k=10** — confirmed empirically, not just read:
   `HIPFIRE_QWEN35_PAGED_EXPERTS=1` on the same artifact panics
   `moe.decode-routed-dtype-unsupported-no-fallback` (`pipeline/mod.rs:216-223`), with a second
   refusal behind it (`:976`). The control keeps decoding under the identical flag.
   "Falls back cleanly" is a property of the fixture's resident weights, not of the geometry.

*Fixture defect, since fixed:* the probe was `hidden = 256` — exactly **one** G256 group on
the `gate_up` reduction dim, which is the thing `moe_indexed_preset`'s own docstring rejects
because it "would stop covering multi-group accumulation". Raised to `hidden = 512`
(29.4 M params); the three-family split reproduces unchanged. The target is hidden 2560, ten
groups, so treat even the fixed fixture as a lower bound on coverage. Weights are still
random-init, and the target has `tie_word_embeddings: false` where every fixture is tied.

*What stands:* the `Mean quant error: 0.00000000` note — verifiers confirmed the control reads
0 too, so it is correctly excluded as a symptom.

**Consequently the M-L estimate is CONDITIONAL on residency.** The `K_TOP`-is-runtime finding
holds and no verifier found a hole in it, but at k=10 `use_gpu_topk` is false unconditionally,
so `pipeline/mod.rs:216` and `:976` between them leave a **paged** model with no decode route
at all. If the experts do not fit resident, k=10 indexed decode is a **blocker**, not a perf
item. See the GTT arithmetic below — and note it is arithmetic, not a measurement.

**The cheapest experiment that would settle the real question**, and it is not a kernel port:
a ~40-line from-scratch CPU top-k MoE reference in `tiny_harness`, differenced against the GPU
fallback on the same weights, in the style of the existing `prefill-band-parity` probe (which
already demonstrates it *can* fail, via `--corrupt-kv-prefix`). Validate the instrument first
by fault injection — patch the CPU top-K loop to drop the 10th expert and confirm `mean_kld`
moves. If it does not, every number above is void. Report `mean_kld` against a seed sweep with
its noise band; **drop argmax agreement entirely.**

### 3.1a The GTT 2 MiB tax, and grouped allocation — the lever that may dissolve §3.1

`gtt_alloc_cost` (`weight_pager.rs:842-880`) is measured, not assumed: above 2 MiB the granule
is a flat **2 MiB, not the next power of two**. Its own doc names the consequence — "which is
what distinguishes this from a buddy allocator and what makes grouping small allocations
worthwhile."

At the target's geometry (`gate_up` Oq4G256 + `down` Oq4G128), one expert module is
2,560,000 B = **2.441 MiB**, which rounds to **4.000 MiB — a 1.638x tax**. Across
512 x 49 = 25,088 modules that is **64.2 GB of weights occupying 105.2 GB of GTT, 41 GB
burned**. Grouping amortizes the fixed 2 MiB over more bytes per allocation:

| experts per allocation | 1 | 2 | **4** | 8+ |
|---|---:|---:|---:|---:|
| tax | 1.638x | 1.229x | **1.024x** | 1.024x |

**N=4 is the knee: 105.2 GB -> 65.8 GB — and this is now MEASURED, not arithmetic.**
`examples/gtt_granularity` on gfx1151, 64 allocations per size:

| requested | actual per-alloc | ratio |
|---:|---:|---:|
| 1,689,600 (`gate_up`) | 2,064,384 | 1.222x |
| 870,400 (`down`) | 1,015,808 | 1.167x |
| **2,560,000 (one module)** | **4,194,304** | **1.638x** |
| 5,120,000 (N=2) | 6,291,456 | 1.229x |
| **10,240,000 (N=4)** | **10,485,760** | **1.024x** |

The per-module 1.638x and the N=4 collapse to 1.024x are exactly as predicted, so the
grouping win is real: **105.2 GB -> 65.8 GB on measured numbers.**

**Consequence: the experts FIT RESIDENT, so k=10 is a perf item and not a blocker.**
~65.8 GB of experts plus ~14 GB of everything else is ~80 GB against ~137 GB usable. That
satisfies `check_moe_decode_supported` branch (b), which is the condition that was refusing a
paged model at `k != 8`. M17/M18 (the Opus-G128 pair and the top-k generalisation) come off
the critical path for a first serving model — they buy throughput, not correctness.

**Also measured: `gtt_alloc_cost`'s middle regime is wrong.** It models 512 KiB..2 MiB as
"next power of two", which would give 2,097,152 and 1,048,576 for the two tensors above; the
driver actually returns 2,064,384 and 1,015,808. It OVER-estimates by ~1.6% there. That does
not change the module-level numbers (the >2 MiB regime is exact) and it does not rescue the
estimator-vs-pager gap in `BUGS.md` — recomputing with measured values gives 77.3 GB against
the pager's 105.2 GB, so the shortfall is ~28 GB rather than ~26 GB.

**The 1.638x vs 1.24x discrepancy flagged earlier is now reconciled, and it was a bug, not a
modelling difference.** `estimated_module_resident_bytes` (`weight_pager.rs:929`) rounds
**per tensor**; `ensure_expert_module_resident` (`:1573`) allocates **once per module**, as
its own comment intends. For this geometry the per-tensor split lands in the
512 KiB..2 MiB power-of-two regime (1,689,600 -> 2 MiB, 870,400 -> 1 MiB = 3 MiB) while the
module lands in the >2 MiB multiple-of-2-MiB regime (2,560,000 -> 4 MiB). So admission sees
**78.9 GB** and the pager actually takes **105.2 GB** — a **26.3 GB shortfall** that would
fail mid-load as a `page allocation failure`. Filed in `BUGS.md`. No existing fixture can
catch it: on the tiny MoE fixtures every expert tensor is under 512 KiB, where the two
formulas agree exactly (measured 1.006x both). Both figures remain **arithmetic**; the GTT
measurement on a real artifact is still owed.

Most of the mechanism exists and is battle-tested:

- **`RawExpertStorage`** (`qwen35/layout.rs:198`) — expert buffers as "interior slices of one
  stacked allocation". Exactly the grouped block.
- **`BufferOrigin::NonOwning`** + `DeviceBuffer::alias` + `GpuTensor::sub_offset`
  (`hip-bridge/src/lib.rs:89-99`) — correct disposal for views into memory owned elsewhere.
- **`free_moe_ffn_maybe_slab`** already handles both aliasing sources, and the hazard is
  understood the hard way: #262 was "a non-owning slab alias returned to the pool was
  re-handed out as scratch and written over another layer's live weights".

What is missing is only the wiring: the serving loader passes `raw_expert_storage: None` at
all three sites (`loading.rs:1887`, `:6903`, `:7042`); only `calibration_stream.rs:1747` builds
it. And the pager allocates one pooled buffer per module — `GpuPool` is a power-of-2 **reuse**
free-list with the real `hipMalloc` at the requested size (`rdna/src/pool.rs:35-60`), so it
fixes allocation churn and does nothing for footprint.

**Design note.** Grouping N experts per allocation is free when experts are **resident**. When
**paged** it makes allocation granularity equal residency granularity — fetching 4 experts to
use 1, and at top-10-of-512 co-selection is ~2%, so ~4x the page-in bandwidth on the axis that
is already the unknown. Decouple them: allocate a few large slabs (~64 MiB), sub-allocate
**uniform expert slots** inside them, run the LRU over slots. Experts are the same shape in all
48 layers, so this is a fixed-slot arena (`slab = slot / SLOTS_PER_SLAB`), not a general
allocator, and `NonOwning` aliases make it representable with no new primitive.

**Why this matters more than 41 GB.** At ~66 GB of experts plus ~14 GB of everything else
against ~137 GB usable, the experts fit **resident** — and residency is exactly what
`check_moe_decode_supported` branch (b) requires at k != 8. Fix the tax and the k=10 blocker
dissolves into a perf item. That is why the residency measurement, not the kernel port, is the
next step.

Two things this does not change. Page-in throughput — 480 expert page-ins per token against
three synchronous transports with no prefetch — is untouched by any of the above and remains
the axis's real unknown. And the stacked-expert Opus arm still gates on a hard-coded 256
(`cli.rs:10841`), which is the specific line that produced the fallback measured here.

### 3.2 N-gram table: hash + on-disk residency — **L**

Covered in §4. The hash is decoded and the container already preads; the work is real but
bounded and independent of §3.1.

### 3.3 QSA per-query block selection — **M–L**

Composed from PFlash's block scorer and deepseek4's gather + joint-softmax kernels, but
neither half is directly droppable (host-side rank, destructive selection, `MAX_TOTAL 1024`,
64-head assertion, O(N²) top-k past 2048). The two unknowns are now **resolved** against the
reference — see §2.3 for the exact score reduction and the block/token budget identity — which
takes this from "design against a guess" to "port a specified algorithm", and is the main
reason it is M–L rather than L.

Worth resolving first: `deepseek4_attn_swa_topk_{direct_,batched_}wmma.hip` are written,
documented, and never wired (`#[allow(dead_code)]`), with dynamic score LDS and **no
`MAX_TOTAL` cap**. If they validate they are a better base for a 2048-budget kernel than
raising a constant on the f32 ones.

### 3.4 mrope — **L**

3-position-stream generation for text/image/video spans, a section-split rope kernel, and a
per-token 3-tuple position buffer. Rated L not because it's deep but because it's
silent-wrong and the config parser makes it look done.

### 3.5 Gated-residual per-channel mix — **M**

Plumbing exists (§2.2). Cost is the read-side per-channel kernels, the rank-320 GEMV, routing
634 M params through the quantizable weight path, and accepting the loss of the fused residual
epilogue (~96 extra launches/token — DS4 already made that call).

### 3.6 Arch crate + registries + fixtures — **XL aggregate, low unit risk**

Mechanical, wide, and the place where silent failures hide (force-link, golden catch-all).

---

## 4. The n-gram table on drive

### What it is, exactly

128 tensors `…ple_embedding.ngram_embedding.shard_{0..127}.weight`, each `[2500012, 160]`
BF16 → 800,003,840 B logical, stored `bf16h` at ~529.7 MB (1.511×). Total **102,400,491,520 B
logical**, 67.8 GB stored.

**They are named `layers.1.ple.…` in the checkpoint, not `layers.2`, even though
`ple_layer_ids = [2]`** — because `ple_layer_ids` is **1-based**: the reference matches on
`ple_layer_ids.index(layer_idx + 1)`. So `[2]` means `layer_idx == 1`. Convert on ingest;
don't hardcode either number.

### The addressing scheme — settled against the reference

`ngram_heads_vocab_sizes` (I64[16]) is sixteen **distinct primes just above 20,000,000**:
20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 — the reference
generates these as `_find_nth_prime_after(ngram_vocab_size_base - 1, k+1)`.
`ngram_heads_offsets` is their running prefix sum. `layer_multipliers` (I64[3]) =
`[23703573157769, 20109073645365, 8052911324071]`, derived from a splitmix64 of
`config.seed` — but read them from the checkpoint buffers, don't regenerate.

It is the multi-hash trick: **8 heads per order × 2 orders**, each head hashing the *same*
key under a *different* prime so collisions decorrelate, each emitting
`ple_embed_dim / 16` = 160 dims → concat 2560.

```
mixed_n = (t[i]·m0) XOR (t[i-1]·m1) XOR … XOR (t[i-n+1]·m[n-1])    (u64, n ∈ {2,3})
row     = offsets[h] + (mixed_n mod vocab_sizes[h])
shard   = row / 2_500_012 ;  local = row % 2_500_012
byte    = shard_data_offset + local * 320
```

**It is XOR, not the polynomial sum this document originally inferred.** Both orders start at
`m0` on the *current* token: the bigram (heads **0–7**) uses `m0, m1`; the trigram (heads
**8–15**) uses `m0, m1, m2`. The four combinations previously listed as unresolved are now
zero.

Semantics the artifact could not have told us, all load-bearing:

- **The shift is EOS-segment-aware.** `_shift_right_ignore_eos` fills any position that would
  reach back across an `eos_token_id` with `eos_token_id` itself, so an n-gram never spans a
  document boundary. A naive shift silently produces different rows near every EOS.
  Two edges, both explicit in the llama.cpp port: **a token's own EOS does not cut its own
  context** (`ctx[0]` is the raw token, the cut only applies from `s ≥ 1`), and **a missing
  predecessor — before sequence start, or no cached cell — reads as EOS**.
- **Decode carries a 2-token history** (`ngram_size - 1`), which the reference stores in the
  KV cache as a third conv state (`conv_states[2]`) and **initialises to `eos_token_id`, not
  zero**. llama.cpp instead reads predecessors out of the attention KV cells; same semantics,
  different bookkeeping. Our own 2-token per-sequence history is simpler and matches the
  reference — prefer it.
- **Embedding batches (images) have no token id.** llama.cpp substitutes
  `ple_image_token_id`, falling back to EOS. Deferrable on a text-first path, but it must be
  handled before vision, and a wrong stand-in is silent.
- llama.cpp asserts **PLE does not support tokens shared by multiple sequences**. Worth
  inheriting as an explicit constraint rather than discovering it in the batch executor.

**Store the concatenated table as ONE tensor.** llama.cpp's converter folds the 128 shards
into a single `per_layer_token_embd.weight` (streaming one shard at a time so the table is
never resident). Doing the same at quantize time removes the `row / 2_500_012` division and
reduces the gather to a single `data_offset + row * 320`. Recommended.

Σ vocab_sizes = 320,001,446, padded to **320,001,536** = 128 × 2,500,012 exactly — so the
shards *are* uniform slices of one flat padded table, and `2500012 × 320 = 800,003,840`
matches the stored logical length. (One survey pass claimed the slices are non-uniform; the
arithmetic and the reference's `ceil(total/128)*128` both say otherwise.)

### Why disk is the right answer

Per token the model reads exactly **16 rows = 5120 bytes**. Measured on this box
(`/home/sadara` = btrfs on nvme0n1p3, Samsung 990 PRO), 320-byte random preads scattered over
the 238 GB archive:

| queue depth | µs/row | 16 rows (one token) |
|---|---:|---:|
| 1 | 138.8 | **2.22 ms** |
| 8 | 19.7 | 0.32 ms |
| **16** | **12.3** | **0.20 ms** |
| 32 | 13.4 | 0.21 ms |

512-token prefill (8192 sorted rows, qd32): **88 ms**.

And the decisive property: **n-gram keys are a pure function of already-committed token IDs.**
They are known before the forward pass starts — the read can be issued the instant the
previous token is sampled and fully overlaps 48 layers of compute. Even the serial 2.2 ms
hides. (`hipfire-runtime/src/cpu_router.rs` is the in-tree prefetch-scheduling prior art, and
it was built for the *harder* MoE case where indices aren't known ahead.)

### The on-disk format: 4 KiB blocks

**The block is the I/O unit, so make it the format unit.** A row read costs one 4 KiB
filesystem sector however few bytes we want, so any compression fitting inside that sector is
free. Measured on halo at qd16:

| read | µs | 16 reads (one token) |
|---|---:|---:|
| 320 B raw row | 10.5 | 0.169 ms |
| **4096 B, block-aligned** | **11.0** | **0.176 ms** |
| 16 KiB aligned | 11.7 | 0.187 ms |

btrfs `sectorsize` is 4096 here (the device logical sector is 512, but the filesystem operates
in 4 KiB), so 4096 is the unit. 16 KiB is equally free and wastes proportionally less
end-of-block slack (~0.6% vs ~2.6%), but at 4× the worst-case decode — not worth it, though
worth remembering if the ratio ever matters.

```text
tensor header   [32 B] Huffman symbols + code lengths, one table for the whole tensor
index           u32 x n_blocks - row index of each block's FIRST row (monotonic)
blocks          4096 B each, 4096-aligned, variable row count
  [    2 B] u16 n_rows
  [2*n B] u16 code bit-offset x n_rows   - so a row decodes without touching its neighbours
  [160*n B] mant  - sign<<7 | mantissa[6:0], directly indexed
  [  rest ] codes - exponent codes, packed until the next row would not fit
```

**Rows per block is variable, and an index maps row → block.** Pack rows into a block until
the next one does not fit, then start a new block; a row therefore never straddles, so a
lookup is still exactly one aligned 4096 B pread. `block = upper_bound(index, row) - 1`.

Sizing, from the ratio the archive already gives us — its own `bf16h` payloads are
1.5108× on these exact shards, so this is measured, not assumed:

| | |
|---|---:|
| rows, total | 320,001,536 |
| effective ratio after ~2.6% end-of-block slack | ~1.47× |
| **on disk** | **69.6 GB** (from 102.4) |
| blocks | 17.0 M |
| average rows/block | 18.8 |
| **index, `u32` per block** | **68 MB** |

The index is small enough to simply load at model open and hold. The search is 24 steps over
a monotonic array, but rows-per-block barely varies, so seeding with
`row * n_blocks / n_rows` lands within a block or two and it becomes ~2 accesses.

### One Huffman table for the whole tensor — measured, not assumed

The obvious alternative is a per-block table (or a pool of tables with an id in the index).
Neither is worth it, and the archive settles it without decoding anything: every one of the
128 shards carries its own `stored_len`, so the per-region ratio is already recorded.

| region | mean ratio | min | max |
|---|---:|---:|---:|
| bigram heads (0–7), 56 shards | 1.5102 | 1.5102 | 1.5103 |
| trigram heads (8–15), 57 shards | 1.5111 | 1.5110 | 1.5111 |
| **all 128 shards** | **1.5107** | **1.5102** | **1.5111** |

**Total spread 0.0009 — 0.06%.** There *is* real structure: every bigram head lands at
1.5102–1.5103 and every trigram head at 1.5110–1.5111, with zero overlap. That is the
predictable effect — bigrams recur more, so those rows are better trained and use a wider
exponent range. It is a clean signal worth 0.06%.

The homogeneity is structural rather than a quirk of this checkpoint: **the hash-scattering
that forces random access in the first place is what makes the data uniform.** Every shard is
a sample of 2.5 M arbitrary n-grams drawn from one distribution, so they converge.

That also kills per-block tables *analytically*, which matters because the shard measurement
cannot see 4 KiB-block variance directly (a block is 130,000× smaller). A per-block table
costs 32 B of 4096 = **0.78%**. A block is ~3040 elements over ~15 symbols, so fitting it
exactly recovers a bias of about `K / (2N ln2)` ≈ 0.0036 bits against H ≈ 2.59 bits —
**~0.14% available**. Paying 0.78% to chase 0.14% makes the file bigger. And whatever variance
does exist at 3040 elements is sampling noise around the global distribution; a code fitted to
noise is worse than the global code, not better.

So: **one 32 B table in the tensor header**, as `bf16_huff` already does with its
symbols/lengths pair. If the 0.06% split ever mattered it is free to exploit *without* a
per-block id — the head is derivable from the row (`row → h` via `offsets[]`, which the gather
already computes), so store 2 or 16 tables in the header at 32 B each and index by `h` at zero
per-block cost. Upgrade path, not v1.

Why variable rather than a fixed 18 rows per block, which was the first draft of this format:
a fixed count has to be sized for the worst block, which wastes slack in the average one and
still needs an **escape path** for blocks whose rows code badly. Variable packing dissolves
that — a block that codes badly simply holds fewer rows. Three consequences worth naming:

- **No overflow path exists.** The format cannot fail on unexpected data; it only gets bigger.
- **It removes a measurement from the critical path.** The fixed design needed an exponent
  histogram of the real shards to decide whether fixed-width 3-bit LUT coding would fit inside
  the budget. With variable packing, take Huffman unconditionally and let the data set the
  size.
- **It degrades gracefully.** If some region of the table compresses worse than 1.51×, blocks
  there hold fewer rows and nothing breaks.

What the whole format buys, in increasing order of importance:

1. **~33 GB of disk** — the least interesting one.
2. **A block cache, if ever wanted, holds ~1.47× more rows per byte** than raw, and is a plain
   `HashMap<u32, [u8; 4096]>` keyed by block index.
3. **O_DIRECT becomes trivial, so the page cache never sees the table.** This is the real
   prize and the reason to align to 4096 at all. Reads that are 4096-aligned and 4096-sized
   are exactly what `O_DIRECT` requires; `Transport::alignment()` already returns 4096 for
   that path (`weight_pager.rs:190`), and `AlignedHostBuffer` + `read_direct_allow_eof`
   already exist (`:1067-1133`). On this UMA box page cache and GTT are one pool, so 102 GB of
   scattered reads filling the cache was the open risk in the "cut from v1" list below —
   **this design removes that risk rather than bounding it.** No `drop_pages_range` hedge, no
   unbounded growth, no competition with expert residency.

CPU cost is not a constraint. A row's codes are ~160 Huffman symbols given the per-row bit
offset — 1–2 µs; even decoding whole blocks is ~0.3 ms/token single-threaded, on a prefetch
thread that has the entire forward pass to hide in. The index search adds ~2 memory accesses
per row on top of that.

**Why neither existing codec works as-is.** `bf16_lut3` is *planar* — six regions (`esc_tab`,
`lut`, `mant`, `codes`, `escapes`) at computed bases — which is what lets a GPU kernel decode a
whole tensor in place, and what makes a single random row cost **5 preads at scattered
offsets**. Worse, `gcd(160, 256) = 32`, so a 160-element row straddles two 256-element blocks
50% of the time, and the `lut` + `esc_tab` planes total ~2.4 GB across the table, too big to
pin. The repo already reached this conclusion at `hfq.rs:3001`: *"The GATHER cannot use it: a
lookup reads one arbitrary row, and BF16L3's escape plane is only addressable by walking a
block, so there is no gather kernel for it."* And stock `bf16_huff` chunks at 8192 elements
(51.2 rows), so reaching one row means decoding up to 8192 symbols; re-blocking it is a format
change, because the decoder validates the stored chunk count against the global constant
(`bf16_huff.rs:548`). Both codecs are built for whole-tensor sequential decode. This one is
built for the access pattern we actually have.

Then, in order of how much they can silently ruin things:

1. **Keep the table out of the residency estimator.** `check_load_headroom`
   (`hipfire-serving-core/src/load.rs:727`) will refuse the artifact against ~120 GB before
   anything runs. It needs a never-resident tensor class. `HIPFIRE_LOAD_MEM_CHECK=0` is the
   bring-up escape hatch, not the answer.
2. **Teach three name predicates about the n-gram tensors**, which all miss today because the
   names contain neither `lm_head` nor `embed_tokens`: writer-side `is_gather_shaped`
   (`hfq_out.rs:113`), loader-side `is_head_tensor` (`hfq.rs:1009`), and
   `is_embedding_table_name` (`cli.rs:7275`). Getting this wrong writes the single largest
   tensor group in the wrong format, silently. The n-gram table takes the new 4 KiB-block
   codec, not the general path: `--bf16-codec` defaults to `huff` at the CLI (`cli.rs:7787`)
   and `compress_bf16_tensor` silently falls back to Huff when LUT3 declines
   (`hfq_out.rs:176-180`), so the block codec has to be asserted for these tensors, not
   assumed — and a silent fall-through to stock `huff` produces a table that reads but whose
   rows are not addressable.
3. **A block read on `HfqFile`** — `tensor_block(name, block_idx, dst: &mut [u8; 4096])`
   routed through `physical_range` (`hfq.rs:1804`), refusing loudly when the tensor is not in
   the block codec, plus a companion that hands back the 68 MB block index at open so the
   caller can resolve row → block itself. `physical_range` returns the *compressed* extent while
   `info.data_size` advertises the *expanded* length; mixing them returns garbage that passes
   every shape check. Also: `tensor_data_pread` reuses one shared `RefCell<Vec<u8>>`, so two
   live views clobber each other (`hfq.rs:1637`) — the row reader must not build on it.
4. **Read 16 blocks O_DIRECT, decode 16 rows, one 5120 B H2D.** `DirectH2DTransport` already
   opens with `O_DIRECT` and reads through `AlignedHostBuffer` (`weight_pager.rs:373`, `:1095`),
   and `Transport::alignment()` already advertises 4096 (`:190`) — the 4 KiB-block format is
   exactly what that path wants. Issue the 16 block reads on a small thread pool at qd16 the
   moment the token is sampled, decode into pinned staging, one `memcpy_htod`. (The buffered
   `PreadH2DTransport` sets `POSIX_FADV_SEQUENTIAL`, the wrong hint for random rows — another
   reason this path takes the direct transport rather than that one.)
5. **Prefill**: sort the 16·N row indices and sweep once at qd32.

### Cut from v1, and when to add it back

- **Block LRU cache.** N-gram frequency is Zipfian, so a few GB would catch most lookups.
  Cut from v1 anyway — the reads are already hidden by prefetch-on-sample, so a cache
  accelerates something that costs nothing. Add it only when a trace says otherwise; the
  4 KiB-block format makes it a plain `HashMap<u32, [u8; 4096]>` keyed by block index when
  that day comes, holding ~1.47× more rows per byte than raw would.
  **The page-cache hazard that used to live here is designed out.** It read: on this UMA box
  page cache and GTT are one pool — which is why mmap was ripped out (`7c3ca08b2`) — so an
  unbounded page cache over a 102 GB table is the same failure in a new costume, and must be
  bounded with `drop_pages_range` or bypassed with `O_DIRECT`. The 4 KiB-aligned format makes
  `O_DIRECT` the default rather than the escape hatch, so the table never enters the page
  cache at all and there is nothing left to bound.
- **Async / io_uring.** `Transport::wait` is a no-op in all three impls and there is no
  io_uring in the workspace. Prefetch-on-sample makes it unnecessary at decode; revisit for
  prefill if the 88 ms/chunk shows up in a trace.
- **GPU-side gather from a dma-buf.** Worth noting it is *proven*, not speculative:
  `alloc_shared_gtt` + `import_dmabuf` + `ImportedTensor::view()` is dispatched in production
  by `hipfire-arch-embeddinggemma/src/npu_opus.rs:1888`. (The stale note at
  `gtt_share.rs:7-9` still claims a compute kernel would need `hipHostRegister`; delete it.)
  `kernels/src/gemv_bf16_gather_f32.hip` — the two-stage lm_head fine pass — is the closest
  existing kernel to the required pattern. Still not v1: 5120 B/token is a latency problem,
  not a bandwidth one.
- **Quantizing the table.** See §6.
- **Extending `WeightPager`.** Build a separate structure. `ExpertResidency`
  (`families/moe.rs:313-334`) documents that returning `Ok` with incomplete residency is an
  unrecoverable MES hang and full device reset on gfx1103; do not share that identity space.

Existing harness to extend before designing anything further:
`crates/hipfire-runtime/examples/load_transport_probe.rs` already measures
pread/pinned/O_DIRECT tensor-vs-slab. Add a random-row arm.

---

## 5. Milestones

Each milestone is independently shippable, has ONE gate that proves it, and is
sized so it fits in a sitting. Nothing depends on a milestone that is not listed
above it. Status as of 2026-08-29.

### Done

| # | Milestone | Gate | Evidence |
|---|---|---|---|
| M0 | Reference vendored, semantics settled | reading | `third_party/transformers-qwen4_exp/` @ `5f8ab9bb`; caught XOR-vs-sum and the dilated PLE conv |
| M1 | MoE blocker characterised | tiny fixture, no GPU | `qwen3_5_moe_mi640_k10`; 3-family split filed in `BUGS.md`; XL → M–L |
| M2 | CPU top-k routing extracted + covered | `cargo test -p hipfire-dispatch` | `cpu_topk_select` + 6 tests incl. a mutant; byte-identical logit hash |
| M3 | Offline spec crate | `cargo test -p hipfire-arch-qwen4exp-spec` | arch 26, 20 tests, quant policy validated over all 1658 real tensor names |
| M4 | Fixtures + quantize path | `--emit-fixture qwen4_exp` → `--format oq4` | 2 fixtures; source-precision opt-in (else the n-gram table shipped 8-bit) |
| M5 | Config | `cargo test -p hipfire-arch-qwen4exp` | 18 tests vs the SHIPPED config; caught `seed=1234` and `ple_index` ≠ `layer_idx` |
| M6 | Weight planning | same | `plan()` matches all 1294 trunk tensors exactly; 2 fault injections |
| M7 | N-gram addressing | same | derived multipliers reproduce the stored buffer exactly |
| M8 | Disk row-gather | same | 4 KiB block store, 8 tests, one aligned read per row |
| M9 | Gated-residual CPU oracle | same | 7 tests; caught the `(1.0 + weight)` norm form |
| M10 | First kernel: per-channel HC collapse | `parity_qwen4exp_kernels` | max\|Δ\| 1.19e-7 on gfx1151 + a mean-not-sum control |
| M11 | Grouped RMSNorm kernel | `parity_qwen4exp_kernels` | max\|Δ\| 1.19e-6 + a grouped-not-flat control |
| M12 | Low-rank mix wiring (`down`→silu→`up`→sigmoid) | `parity_qwen4exp_kernels` | whole read path max\|Δ\| 4.76e-5 at the REAL geometry (2560/4/320) |
| M13 | Residual write-back | same | max\|Δ\| 2.38e-7; inject gates 4.41e-6 |
| M19 | PLE dilated depthwise conv | `parity_qwen4exp_kernels` | 6 steps at ch=10240: max\|Δ\| 2.38e-7, **state Δ = 0** + a dilated-not-adjacent control |
| M16 | CPU MoE oracle | `cargo test -p hipfire-arch-qwen4exp` | 6 tests incl. fault injection; pins gate-first split, softmax-then-cut, always-on shared expert |
| M15a | **Slab win measured with real allocations** | `gtt_slab_vs_permodule` | 1.638x → 1.024x per module; **39.5 GB saved** across 25,088 modules (37.5%) |
| M14 | **GTT residency measured** | `examples/gtt_granularity` | 1.638x per module, 1.024x at N=4, both exact; experts FIT RESIDENT ⇒ k=10 is perf, not a blocker |

### Next, in dependency order

| # | Milestone | Gate | Size | Blocked on |
|---|---|---|---|---|

| M15b | Slab wiring into the serving loader | a paged qwen35 MoE load, before/after | M | M15a |
| M17 | Opus-G128 indexed MoE pair | parity vs M16 | L | M15 (M16 done) |
| M18 | Generalise the two top-k selection kernels off `#define K_TOP 8` | `tiny-affected-gate` | M | M17 |
| M20a | GDN sigmoid output gate | `parity_qwen4exp_kernels` | max\|Δ\| 2.98e-7 at nh=48/hd=128 + a sigmoid-not-silu control |
| M20b | **GDN layer forward** (`gdn::gdn_decode_step`) | `parity_qwen4exp_kernels` | runs at hidden 2560 / qkv 10240 / 48 V heads; finite output and a **stateful** check (replaying step 0 after 4 steps must differ) |

**M20b sizing note (measured, not estimated).** The GDN kernels qwen4_exp needs are all
present and already exercised at its exact geometry — `fused_qk_l2_norm_scale_f32`,
`repeat_interleave_qk_f32` at ratio 3 (48/16), `conv1d_silu_split` with asymmetric k/v spans,
and the recurrence. The only compute delta is the output gate, which M20a supplied.
The cost is in the ASSEMBLY, and there are two routes, neither free:

* **Thread a flag through qwen35's driver.** `gated_norm_f32` has **8 call sites across 6
  files** (`decode_layers.rs` x2, `lowered.rs`, `ep.rs` x2, `prefill_lowered.rs`,
  `prefill_batch.rs` x2), and the batched variant `gated_norm_f32_batched` has no sigmoid
  sibling yet. Every one of those is on a shipping Qwen3.5/3.8 hot path. This is exactly the
  "widening qwen35 taxes every shipping model" hazard the new-crate decision (§6.5) exists to
  avoid, so it is the wrong route even though it is the smaller diff.
* **Assemble in `hipfire-arch-qwen4exp`,** calling the `hipfire-rdna` kernels directly. That
  duplicates qwen35's scratch/state management (`Qwen35Scratch`'s `dn_*` buffers, the conv
  ring, the tape hooks), which is the real work — and it is TIER-INDEPENDENT, so it does not
  wait on §6.4.

Recommended: the second. Budget it as the scratch/state layer, not as kernel work.
| M21a | QSA selection oracle (CPU) | `cargo test -p hipfire-arch-qwen4exp` | 8 tests; pins mean-pooling, per-head ReLU-before-sum, unconditional ragged tail, and the dense-below-budget property |
| M28 | **Everything above put behind a gate** | `./tests/qwen4exp-gate.sh` | CPU stage (oracles + checkpoint-pinned tests + quantize round trip) and GPU stage (8 kernels, 6 controls); `--cpu-only` for CI |
| M21b | QSA block scoring kernel | `parity_qwen4exp_kernels` | max\|Δ\| 3.58e-7 over 1024 blocks + a relu-before-sum control that **caught a wave32 bug** |
| M21c | QSA top-k by threshold (CPU, O(N) exact) | `cargo test -p hipfire-arch-qwen4exp` | matches a full sort at n=65536/k=512, and under heavy ties; retires the O(N·K)/O(N²) limit |
| M21d | **QSA top-k mask kernel** | `parity_qwen4exp_kernels` | exact vs the CPU select at n=65536/k=512 (native context) + a tie-band control that a `== t` implementation fails |
| M21e | **QSA selection pipeline on GPU** (score → top-k → token mask) | `parity_qwen4exp_kernels` | exact vs `qsa::select`; the dense boundary lands at n=67 (all kept) / n=68 (excludes), as predicted |
| M21f | **QSA sparse attention, whole decode path** — slot list → gather → `attention_cold_slots` | `parity_qwen4exp_kernels` | **no new attention kernel**; index list exact vs `qsa::select`; output BIT-IDENTICAL to dense at n≤67, differs at n≥68 |
| M22a | **Reference oracle** — pinned transformers @5f8ab9bb forward, weights + activations | `tests/reference_oracle.rs`, `scripts/qwen4exp_oracle.py` | plan validated against the REFERENCE state dict, not just the checkpoint; found 2 defects |
| M22b | Trunk ends: embed + wide seed, mixer collapse, lm_head | `reference_oracle.rs` | exact vs reference |
| M22c | n-gram addressing + row ids (EOS-segment window) | `reference_oracle.rs` | EXACT integers vs reference |
| M22d | PLE block (streaming) | `reference_oracle.rs` | max\|Δ\| < 5e-6 |
| M22e | MoE block, all layers | `reference_oracle.rs` | max\|Δ\| < 5e-6 |
| M22f | Layer composition (HC read/write, PLE add) | `reference_oracle.rs` | reference's own block outputs substituted ⇒ isolates wiring |
| M22g | Gated DeltaNet (CPU recurrent) — `gdn_cpu.rs` | `reference_oracle.rs` | recurrent vs reference's CHUNKED kernel, < 2e-4 |
| M22h | QSA attention + indexer + RoPE — `attn.rs`, `rope.rs` | `reference_oracle.rs` | indexer mask EXACT (bool); attention < 5e-5 |
| **M22** | **Full text trunk — `trunk.rs`** | `reference_oracle.rs` | **tokens→logits vs reference, max\|Δ\| ~1e-5, argmax identical at all 16 positions, nothing substituted** |
| M23a | `Qwen4ExpConfig::from_metadata_json` — HFQ envelope | `real_config.rs` + gate | config survives quantization into the served `.hfq` |
| M24 | model-support row + `KNOWN_RUNTIME_ARCH_IDS` | `gen-model-support --check` in no-gpu-ci | arch 26 graded; all marks `none` (no GPU path yet) |
| M23b-1 | **GPU hyper-connection** — `hc_gpu.rs` (+ `hc_inject_gate` kernel) | `parity_hc_gpu_vs_cpu` | read 2.6e-6 / write 1.4e-6 / mixer 2.6e-6 vs CPU |
| M23b-2 | **GPU Gated DeltaNet composition** | `parity_gdn_gpu_vs_cpu` | 24 streamed steps, 1.6e-5, recurrent state asserted live |
| M23b-3 | **GPU PLE block** — `ple_gpu.rs` (+ `ple_stream_gate` kernel) | `parity_ple_gpu_vs_cpu` | 20 streamed steps, 1.1e-5, conv state asserted live |
| M23b-4 | **GPU QSA indexer** — `qsa_pool_norm_blocks` + `qsa_score_prepared` + `qsa_attn_prep` kernels | `parity_indexer_gpu_vs_cpu` | selected token set EXACT vs CPU at 5 lengths; found & fixed a kernel that could not express the reference pipeline (BUGS.md) |
| M23b-5 | **GPU QSA attention block** — `attn_gpu.rs`, decode with a KV cache | `parity_qsa_attn_gpu_vs_cpu` | 24 streamed tokens, 6.5e-5; no new attention kernel |
| M18 | **Top-k generalised to a RUNTIME k** (`moe_softmax_topk_renorm`) | `parity_moe_gpu_vs_cpu` | was `#define K_TOP 8`; k=10 routing now EXACT |
| M23b-6 | **GPU MoE block** — `moe_gpu.rs` | `parity_moe_gpu_vs_cpu` | routing exact at k=8 AND k=10; output 6.9e-5 on mag 152.8 |
| **M23b** | **GPU TEXT TRUNK — `trunk_gpu.rs`, decode with full per-layer state** | `parity_trunk_gpu_vs_cpu` | **16 tokens vs the CPU trunk: 1.6e-4 on mag 11.13, argmax IDENTICAL. Chain complete: upstream → CPU → GPU for the whole model** |
| **M23c** | **Daemon wiring — `arch.rs`: `Architecture` impl + `HfqTensorReader`** | `load_hfq_decode` in the gate | real `.hfq` → config → weights → state → finite, varying logits |
| M27a | **MTP head in the weight plan** (31 tensors) | `real_weights.rs` | exact vs the shipped checkpoint, both directions. ⚠️ **NO reference forward exists** — upstream drops `^mtp.*` on load — so the COMPOSITION is unpinned and deliberately not implemented |
| **M26** | **Vision tower (CPU) — `vision.rs`** + 333 tensors in the plan | `reference_oracle.rs`, `real_weights.rs` | patch-embed / every block / merger / position grid / rotary vs the reference, NOTHING substituted; plan exact vs the checkpoint |
| M26b | **Vision tower on GPU — `vision_gpu.rs`** (reuses `vit_attention_f32` unchanged) | `parity_vision_gpu_vs_cpu` | blocks 2.4e-6 / merger 1.1e-6 RELATIVE vs the CPU tower |
| M26c | **Text/vision fusion** — `splice_image_embeds` + the width invariant | `reference_oracle.rs` | fused embeds exact vs the FULL `Qwen4ExpModel`; count mismatch is an error, not a truncation |
| M27b | **MTP forward — `mtp.rs`** | `real_weights.rs` (the ARGUMENT is tested) | ⚠️ **SHAPE-INFERRED, NOT reference-verified** — upstream drops `^mtp.*`. Four checkpoint facts leave one consistent composition; the premises are asserted against the real file so the inference breaks loudly if a revision changes them. Not wired into serving. |
| M15b | **Expert slab — RESOLVED, not needed** | `parity_trunk_gpu_vs_cpu` asserts 1 slab/projection/layer | qwen4exp has its OWN loader and never touches qwen35's; `stack_experts` groups by construction. The #262 hazard that made this risky is CLOSED by 5184255fe (provenance-routed `free_tensor`). |
| M22 | Text trunk end-to-end, n-gram stubbed to zero | fixture golden + `hipfire eval` KLD | M | M20, M21 |
| M23 | N-gram wired (M7 + M8 + M19) | KLD delta vs M22 must IMPROVE | M | M22 |
| M24 | `model-support.toml` row + serving-core force-link | `gen-model-support --check` | S | M22 |
| M25 | mrope (3 position streams, section split) | image prompt vs reference | L | M22 |
| M26 | Vision tower (reuse qwen35-vl) | grounding check on halo | M | M25 |
| M27 | Embedded MTP head | spec-decode acceptance | M | M22 |

**M14 ran and answered its question:** the experts fit resident at N=4 grouping, so
M17/M18 buy throughput rather than correctness and are off the critical path for a
first serving model. **M15a has since measured the slab win with real allocations
— 39.5 GB, 37.5% of expert GTT — so M15b (wiring it into the loader) is justified by
measurement rather than arithmetic.** M15b is deliberately separate: it touches
qwen35's shared loader at three sites and that path carries the #262 double-free
hazard, so it wants its own change and its own before/after on a real paged load.

**M21 has the only free exact oracle in the project** — below the budget QSA is
dense by construction, so sparse and dense must agree bit-for-bit. Use it.

## 6. Decisions I need from you

0. ~~**Get the reference implementation.**~~ Done — vendored at
   `third_party/transformers-qwen4_exp/`.
1. **Text-only first?** `config.language_model_only` exists. Recommend **yes** — it removes
   the vision tower, mrope, and the video path from the critical path, and mrope alone is an
   L-sized silent-wrong risk. *Default if you don't answer: text-only.*
2. **§3.1 exit: new k=10 / mi=640 kernel family, or accept non-indexed decode?**
   Recommend **measure both on the Stage 1 fixture before choosing** — the non-indexed path
   costs compact residency, which may or may not matter once the n-gram table is off-GPU.
   This is the one decision I would not make on theory.
3. **N-gram table: lossless bf16 on disk, or lossy 4-bit?** **Lossless, in the 4 KiB-block
   format of §4 (69.6 GB). Settled by measurement, not by weighing quality.**

   Note this is *lossless compression* (~1.47×), not quantization — every bf16 value is
   reproduced bit-for-bit. The question below is only whether to throw away bits.

   On disk, quantizing this table buys **exactly zero speed**. Measured on halo at qd16 over
   the 238 GB archive, varying only the row size:

   | row bytes | what it models | µs/row | 16 rows (one token) |
   |---:|---|---:|---:|
   | 80 | 4-bit, raw | 10.5 | 0.167 ms |
   | 88 | 4-bit + scale | 7.5 | 0.120 ms |
   | 160 | 8-bit | 6.4 | 0.102 ms |
   | **320** | **bf16** | **7.4** | **0.119 ms** |
   | 1024 | 4× bf16 | 7.5 | 0.120 ms |

   No trend across a 12.8× range — the 80-byte read is the slowest, which is noise. Every row
   size here is one 4 KiB block fetch, so the cost is block latency, not bytes. Two
   second-order effects that could have rescued the 4-bit case both collapse: **locality**
   (smaller rows → more rows per block) is dead because the multi-hash scatters rows uniformly
   by construction, and **page-cache hit rate** (26 GB caches better than 102 GB) is dead
   because the keys are known before the forward pass, so a miss costs nothing.

   **In memory the trade-off does flip — against residency, not toward 4-bit.** bf16 resident
   is 102.4 GB (69.6 GB compressed, but it decodes on read), which on a 128 GB box alongside
   ~64 GB of experts is not a trade-off but an
   impossibility; so residency *forces* 4-bit, and quantization there is an entry ticket
   rather than a choice. But residency buys ~nothing for this tensor — 5120 B/token, perfectly
   predictable, fully prefetchable — while costing ~26 GB of the most contended resource on
   the box. The experts want that memory far more, and expert paging is the one access pattern
   that **cannot** be prefetched (480 page-ins per token, routing-dependent). Paying 26 GB and
   a quality hit to accelerate the access that was already free is the wrong direction.

   **On the Unsloth counter-evidence (§8), which this document briefly took too seriously.**
   UD-Q4_K_XL ships the table at ~4 bits (~24 GiB vs 97.7 GiB) at whole-model wikitext ppl
   4.0068 vs 4.0126 — evidence that 4-bit is *survivable*, not that it is free. And it exists
   because llama.cpp's path needs the table small enough to mmap and wire; that comment is
   someone fighting 114 GiB of wired memory. **We do not have that constraint, and choosing
   pread over mmap is exactly what buys the precision back for nothing.**

   So the quality question never has to be answered. It is a real question — the table is
   51.2 G params, 41% of the model, applied at every token at layer 1, feeding all four
   residual streams, and it is a hash-bucketed representation that already carries collision
   noise, so whether quantization noise compounds or hides under that floor is genuinely open.
   It also cannot be *calibrated*: with 320 M rows any corpus touches well under 1% of them,
   and a per-row codec fit on that is the train-on-test shape that has burned this repo twice.
   Keep it bf16, keep it calibration-exempt, and don't spend an experiment resolving a
   trade-off where one arm is free.

   **The one condition for revisiting: large-batch serving.** At B=32 a step needs 512 rows,
   the prefetch window shrinks, and residency — hence 4-bit — starts to earn its 26 GB. That
   is a serving-shape decision, not a model-quality one. Revisit when batch serving is
   actually on the table.
4. **Serving tier: `ServingFactory` or the arch_id ladder?** ✅ **RESOLVED 2026-08-29 (user):
   the arch_id ladder.** Factory is much cheaper but **rejects DFlash outright**
   (`load.rs:1462`) and its options carry no MTP/drafter/vision inputs. Since the MTP head is
   embedded in the artifact, the factory's limits are exactly the features this model ships
   with, so the cheap path would need unwinding almost immediately. Cost accepted: editing
   4900-line files. This unblocks M22-M24.
5. **New arch crate, or extend qwen35?** Recommend **new crate** (`hipfire-arch-qwen4exp`,
   arch_id 26). The GDN loader is the only large piece worth sharing and it is already 4×
   duplicated inside qwen35; widening qwen35's scratch or layer struct taxes every shipping
   Qwen3.5 model.

---

## 7. What not to do

- **Don't port deepseek4's Sinkhorn mixer.** The target has no 4×4 weight and no Sinkhorn
  config. Force identity, or fork a no-mix variant.
- **Don't reuse `gemv_lowrank_moe_expand`** for the rank-320 expand — hard rank cap of 32.
- **Don't route the HC mixer weights through `upload_global_raw`** like DS4 does. 634 M dense
  params per token is not a "global".
- **Don't widen `Qwen35Scratch`** or any shared simple-AR scratch to 10240.
- **Don't extend `WeightPager`** for n-gram rows (MES hang class), and don't build the row
  reader on `tensor_data_pread` (shared buffer).
- **Don't mmap the n-gram table**, in any form, however tempting a 102 GB random-access table
  makes it. Same pool, same failure as `7c3ca08b2`.
- **Don't convert the 238 GB artifact before Stage 1.**
- **Don't trust `mrope_*` being present in `Qwen35Config`** as evidence mrope works. It is
  parsed and printed and nothing else.
- **Don't skip the `fixture_golden.rs` arch arm** — the catch-all silently scores through
  Qwen3.5's golden path.

---

## 8. Cross-check: the merged llama.cpp port

ggml-org/llama.cpp [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) — "model: add
Qwen3.8-Flash-Next (qwen4exp)", merged 2026-08-28, 28 files, +2881/−38. A second independent
implementation. It is not a design we can copy — they lean on `ggml_get_rows`,
`ggml_rope_multi`, and a hybrid memory abstraction we don't have, and `git diff -- ggml/` is
empty, meaning they added **no new op at all**. But it is a strong check on our reading.

**Confirms, independently:**

- The **XOR hash**, verbatim: `mixed_n = (t[p]*m[0]) ^ … ^ (t[p-n+1]*m[n-1])`, with a comment
  noting the hash must run host-side "because ggml has no int64 and no xor". Our correction
  above stands on two sources now.
- **Host-side row indices then a gather** is the right shape for the PLE — exactly the design
  in §4. They emit `I32[ple_n_heads * n_tokens]` and `ggml_get_rows`.
- **The dilated PLE conv** (`dil = ple_ngram_size`).
- **GDN's only delta from Qwen3.5 is the sigmoid output gate** — precisely the one-field,
  one-branch gap in §2.1.
- **Not sharing a base with deepseek4's hyper-connections**, and for our reason: "the two
  formulations agree on the residual layout and nothing else (full-rank projection with
  Sinkhorn normalisation there, low-rank silu gate and a mean collapse here)."
- Three quantizer bugs they had to fix map onto ours by class: a block-size fallback that
  aborted instead of falling back (their 32, our 256-vs-640 in §3.1); `--tensor-type` unable
  to name `per_layer_token_embd` because the embedding path returned first (**exactly** our
  three shadowing predicates in §4); and a work buffer sized `nelements * 4` — 150 GB on a
  51.2 G-element tensor — which is our `check_load_headroom` problem wearing a different hat.

**New and directly useful — a free exact oracle for QSA:**

> QSA is **dense by construction** below `indexer_top_k + compress_ratio - 1` cached tokens,
> so under the budget it is bit-identical to dense attention. They measured max logit delta
> **0.0** over all 2051 rows on BF16 and F32.

That validates our entire selection + masking + gather path exactly, with no reference model
and no tolerance argument — just run both and diff. Above the budget it diverges (they see 3%
of positions at 8192 tokens), so it is an oracle for correctness of the machinery, not of the
selection policy; for that they report 0.975 mean Jaccard against a 0.991 precision floor.
**Fold the under-budget dense equivalence into the Stage 2 gate.** Their own warning is worth
inheriting too: `test-llama-archs` passed at 8.44e-08 on three different GDN fused-QKV
segmentations, two of them deliberately wrong — a plumbing check that looks like a
correctness check, which is the same trap as our `fixture_golden.rs` catch-all.

**Field evidence on comparable hardware (the linked comment).** An M5 Max, 128 GB unified
memory — the closest public analogue to halo — running Unsloth's UD-Q4_K_XL: 34–36 tok/s
short-context decode, 29 tok/s at 20K depth, 614 tok/s prefill on 20K.

| PLE storage layout | system wired @ 262144 ctx | result |
|---|---:|---|
| interleaved with GPU tensors (as shipped) | 114.4 GiB | `Compute error` above ~196K ctx |
| `--no-mmap` | 90.1 GiB | full window OK |
| PLE alone in its own shard | 90.8 GiB | 35 tok/s, wired flat under generation |

The ~24 GiB delta is the n-gram table — it is quantized to ~4 bits in that quant. Read that
as evidence of a *constraint they are under*, not of a choice we should copy: llama.cpp needs
the table small enough to mmap and wire, and that comment is someone fighting 114 GiB of wired
memory. Our pread path has no such pressure, which is why decision 3 keeps bf16.

Two things follow for us:

1. **This is empirical proof the plan works** — an SSD-resident n-gram table serving at
   35 tok/s on a 128 GB unified-memory machine, with "pages fault in as clean file-backed
   pages and stay evictable".
2. **The failure mode is our mmap ban playing out on Metal.** The PLE shared an mmap region
   with GPU-assigned tensors, so the whole mapping got wired and could not be evicted. Our
   pread design sidesteps the mechanism entirely, but the *lesson* transfers: **the n-gram
   region must not share a residency decision with anything we make resident.** Concretely,
   lay the n-gram table out contiguously at the end of the `.hfq`, and note
   `supports_slab_load()` is already an all-or-nothing whole-file predicate (§4 trap 5).

---

## Appendix: how the facts here were obtained

Config, tensor shapes, and stored codecs were parsed directly out of the `.hfa` index
(trailer at offset 238,441,133,355, length 4,109,570) and the 131 verbatim safetensors
headers, not from documentation. `ngram_heads_offsets`, `ngram_heads_vocab_sizes`, and
`layer_multipliers` were decoded from their raw I64 payloads. The latency table is measured
on this machine against this file. The repo claims carry `file:line` anchors and were
cross-checked by a second pass; where the two passes disagreed, the disagreement is stated
rather than resolved by preference.

Model semantics — the n-gram hash, the gated-residual math, the QSA score reduction, the
grouped RMSNorm, the dilated PLE conv — come from the vendored reference at
`third_party/transformers-qwen4_exp/` (transformers `5f8ab9bb`, fetched 2026-08-29), not from
inference over the artifact. Two claims in this document's first draft were derived from the
artifact alone and turned out wrong; both are corrected in place and flagged at the top. That
is the argument for reading the reference before writing a kernel, not after.
