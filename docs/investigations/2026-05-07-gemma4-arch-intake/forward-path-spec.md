# Gemma 4 Forward-Path Implementation Spec

**Date:** 2026-05-07
**Source commit:** `origin/gemma4` HEAD `8aa13bd` (pre-modular `crates/engine/`)
**Companion doc:** `arch-report.md` (architecture characterization, do not duplicate)
**Branch:** `survey/moe-quant-cliff-2026-05-06`

This spec documents the *implementation-level* details of the working
pre-modular Gemma 4 forward path — the exact dispatch order, kernel
signatures, scratch shapes, dtype gates, and env-var fall-throughs that
the modular re-port on master must match. All file:line refs are into
`origin/gemma4:crates/engine/src/gemma4.rs` unless otherwise noted.

The arch-report covers the *what*. This spec covers the *how*.

## 1. Forward path block trace (per-layer)

`pub fn forward_scratch(...)` lives at gemma4.rs L2047–L2164. Body:

```
[1] Embedding lookup + scale  (L2064–L2071)
[2] pos_buf htod              (L2074–L2075)
[2b] Per-layer-embed pre-loop (L2089–L2123, only if n_embd_per_layer > 0)
[3] Layer loop                (L2134–L2150)
       sliding_layer_decode  | full_layer_decode
[4] Final RMSNorm             (L2153)
[5] LM head GEMV              (L2156)
[6] Logit softcap             (L2159–L2161)
```

Steps [1]/[4]/[5]/[6] are common to all layer types; the per-layer body is
where Gemma 4 diverges from Qwen 3.5.

### 1a. Sliding-window layer (LayerType::Sliding) — `sliding_layer_decode`

Source: gemma4.rs L2193–L2415. Reference layer = layer 0 of gemma-4-31B.

Shape conventions for the table below: `[D]` = `[5376]` = `[dim]`;
`[Hq]` = `[n_heads]` = `[32]`; `[Hkv_s]` = `[sliding_n_kv_heads]` = `[16]`;
`[d_s]` = `[sliding_head_dim]` = `[256]`. Decode batch is implicit `B=S=1`.

| Step | Op (file:line) | Kernel | Inputs → outputs (shape) |
|------|----------------|--------|--------------------------|
| 1.1  | `memcpy_dtod` (L2211)            | hipMemcpy d2d                          | `scratch.x[D]` → `scratch.residual[D]` |
| 1.2  | `prep_norm_for_proj` (L2223)     | `fused_rmsnorm_rotate_mq` OR `rmsnorm_f32` | `x[D], input_layernorm[D]` → `tmp[D]` (or `tmp_rot[D]`) |
| 1.3a | `fused_qkv_or_fallback` MQ4 (L2231) | `fused_qkv_hfq4g256` (1 launch) | `proj_x[D]` → `q[Hq*d_s], k[Hkv_s*d_s], v[Hkv_s*d_s]` |
| 1.3b | else 3× `weight_gemv` (L2234–L2240) | `gemv_<dtype>_*`                        | same as 1.3a |
| 1.4  | `rmsnorm_batched` q_norm (L2246) | `rmsnorm_f32`                            | `q, q_norm[d_s]` → `q` (in-place) |
| 1.5  | `rmsnorm_batched` k_norm (L2248) | `rmsnorm_f32`                            | `k, k_norm[d_s]` → `k` (in-place; `owns_kv` only) |
| 1.6  | `rmsnorm_batched` v with ones (L2249) | `rmsnorm_f32`                       | `v, v_norm_ones_full[d_s]` → `v` (in-place; `owns_kv` only) |
| 1.7  | `scale_f32 sqrt(d_s)` (L2255)    | `scale_f32`                              | `q *= sqrt(256)` (cancels FA's `1/sqrt(d_s)`) |
| 1.8  | `rope_f32` (L2262)               | `rope_interleaved_f32` (full rotation)   | `q[Hq*d_s], k[Hkv_s*d_s], pos_buf, theta=10000` → in-place |
| 1.9a | `kv_cache_write_asym3_fused` (L2274) | `kv_cache_write_asym3_fused`         | `k, v, pos_buf, givens_cos, givens_sin` → KV slot |
| 1.9b | `attention_flash_asym3` (L2278)  | `attention_flash_asym3_tile` **+ `window_size: u32 = 1024`** | `q, K[slot], V[slot]` → `attn_out[Hq*d_s]` |
| 1.9c | (else asym4 / asym2 / q8 branches L2285–L2326, same pattern, all pass `window_size=1024`) | | |
| 1.10 | `weight_gemv` o_proj (L2341)     | `gemv_<dtype>_*`                         | `attn_out[Hq*d_s], o_proj[D, Hq*d_s]` → `tmp[D]` |
| 1.11 | `rmsnorm_f32` post_attn (L2344)  | `rmsnorm_f32` (in-place on `tmp`)        | `tmp, post_attention_layernorm[D]` → `tmp[D]` |
| 1.12 | `memcpy_dtod` x ← residual (L2347) | hipMemcpy d2d                          | reset x |
| 1.13 | `add_inplace_f32` (L2348)        | `add_inplace_f32`                        | `x += tmp` |
| 1.14 | `memcpy_dtod` residual ← x (L2351) | save residual                          | residual = x (for FFN side) |
| 1.15 | `prep_norm_for_proj` pre-FFN (L2366) | same as 1.2                          | `x, pre_feedforward_layernorm[D]` → `tmp[D]` (or `tmp_rot`) |
| 1.16 | `fused_gate_up_or_fallback` (L2380) | `fused_gate_up_hfq4g256` OR 2× `weight_gemv` | `proj_x` → `gate_ffn[H_ff], up_ffn[H_ff]` |
| 1.17 | `gelu_tanh_f32` (L2382)          | `gelu_tanh` (gelu_pytorch_tanh)          | `gate_ffn` → `ffn_hidden[H_ff]` |
| 1.18 | `mul_f32` (L2383)                | `mul_f32`                                | `ffn_hidden *= up_ffn` |
| 1.19 | `weight_gemv` down_proj (L2384)  | `gemv_<dtype>_*`                         | `ffn_hidden[H_ff]` → `ffn_out[D]` |
| 1.20a | (dense path) `rmsnorm_f32` post-FFN (L2394) | `rmsnorm_f32`                | `ffn_out, post_feedforward_layernorm[D]` → `tmp[D]` |
| 1.20b | (MoE path 26B-A4B only) `apply_moe_branch` (L2392)  | see §1c        | uses `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1/_2`, `router.proj`, experts |
| 1.21 | `memcpy_dtod` x ← residual (L2399) | reset x                                | |
| 1.22 | `add_inplace_f32` (L2400)        | `add_inplace_f32`                        | `x += tmp` |
| 1.23 | (E-series only) `apply_per_layer_inject` (L2403–L2409) | see §1d            | injects per-layer side-channel signal |
| 1.24 | `scale_f32 layer_scalar` (L2412) | `scale_f32`                              | `x *= lw.layer_scalar_host` (HOST-side f32 const, not D2H) |

### 1b. Full-attention layer (LayerType::Full) — `full_layer_decode`

Source: gemma4.rs L2438–L2609. Differs from sliding in 6 places:

* **head_dim = 512** (full_head_dim), `Hkv_f = 4` (full_n_kv_heads), Q is `[32*512]=16384`, KV `[4*512]=2048`.
* **K=V on dense families (31B / 26B-A4B):** when `attention_k_eq_v=true`, no `v_proj` weight. After `weight_gemv(k_proj, x, k)` (L2484), V is captured by `memcpy_dtod(v, k, kv_bytes)` (L2487) **before** k_norm runs. Then k_norm overwrites k in place, leaving v with the pre-k_norm bytes. v_norm (no-scale) is then applied to v at L2497 with the ones-buffer.
* **E-series (E2B/E4B):** `attention_k_eq_v=false`, so `lw.v_proj` is `Some` and a third `weight_gemv` runs at L2486.
* **Proportional partial RoPE** (L2506–L2509). Different kernel: `rope_partial_halved_f32`. `n_rot_pairs = head_dim * partial_rotary_factor / 2 = 512 * 0.25 / 2 = 64` pairs rotate; pairs `[64, 256)` are NoPE. theta = `1e6`.
* **No sliding window in attention dispatch** — pass `window_size = 0` (L2533) which means full causal.
* **Quantized KV is asym3-only:** the `*_hd512` flash kernel variants only exist for asym3. Other quant modes (asym4 / asym2 / q8) at hd=512 hard-error at L2536–L2541; FP32 fallback runs at L2542–L2552.

The MLP block (post-attn norm, swiglu, MoE branch, residual, per-layer-inject, layer_scalar) is structurally identical to sliding (L2558–L2606).

### 1c. MoE branch (26B-A4B only) — `apply_moe_branch`

Source: gemma4.rs L1543–L1989 (~450 LOC). Mirrors llama.cpp gemma4-iswa.cpp L125–L179. Runs *inside* the post-FFN-norm step (replaces the simple `rmsnorm_f32` at L2394 / L2587). Sequence:

```
1) cur_mlp = rmsnorm(ffn_out, post_feedforward_layernorm_1)            [L1558]
2) pre2    = rmsnorm(attn_out, pre_feedforward_layernorm_2)            [L1600]
3) router_in = rmsnorm(attn_out, router_scale) * (1/sqrt(dim))         [L1607–L1609]
4) router_logits = router_proj @ router_in                              [L1612]   [n_exp=128]
5) (topk_idx, topk_w) = moe_softmax_topk_renorm_k8(router_logits)       [L1621]   k_top hardcoded to 8
6) memset cur_moe = 0                                                   [L1637]
7) per-expert dispatch (two paths, see §3 fused MoE):
     fused: fused_rmsnorm_rotate_mq + gemv_hfq4g256_moe_gate_up_k8_indexed
            (1 launch instead of 8 gate_up GEMVs; weights MUST be MQ4G256)
     legacy: 8 × { weight_gemv gate_up; gelu_tanh; mul; weight_gemv down } per token
8) cur_moe = rmsnorm(cur_moe, post_feedforward_layernorm_2)             [L1951]
9) tmp = cur_mlp + cur_moe                                              [L1955]
10) tmp = rmsnorm(tmp, post_feedforward_layernorm)                      [L1958]
```

`HIPFIRE_MOE_BYPASS=1` skips the branch entirely (degenerates to dense forward, useful for isolating MoE-induced bugs). `HIPFIRE_MOE_ZERO=1` keeps the branch but zeros the experts (cur_moe=0; only cur_mlp survives). `HIPFIRE_GEMMA4_MOE_FUSED=0` forces the legacy serialized path. Default is fused-on when all expert `gate_up_proj.gpu_dtype == DType::MQ4G256`.

### 1d. Per-layer-embedding inject (E2B/E4B only) — `apply_per_layer_inject`

Source: gemma4.rs L1991–L2040. Runs after `x += residual + post_ffn_norm(...)` and BEFORE `scale_f32(x, layer_scalar)`. Sequence:

```
pe_in = x                                                       [L2005]
tmp_pl = per_layer_input_gate @ x                               [L2008]   [pl_w] = [n_embd_per_layer]
tmp_pl = gelu_erf(tmp_pl)  # NOT gelu_tanh — see L2017 comment  [L2023]
tmp_pl *= per_layer_inp[il*pl_w .. (il+1)*pl_w]                 [L2028]   slice of pre-loop staged signal
x_pl   = per_layer_projection @ tmp_pl                          [L2031]
x_pl   = rmsnorm(x_pl, post_per_layer_input_norm)               [L2033]
x      = pe_in + x_pl                                           [L2036–L2037]
```

`HIPFIRE_GEMMA4_PLE_GELU_TANH=1` swaps to tanh-approx for diagnosis (default is `gelu_erf` — empirically required for E-series coherence per the L2009–L2018 comment). The per-token side-channel `scratch.per_layer_inp` is built once per forward step in the pre-loop at L2089–L2123:

```
a = embed_tokens_per_layer[token]                               [L2101–L2102]
a *= sqrt(n_embd_per_layer)                                     [L2107]
b = per_layer_model_proj @ x_post_embed                         [L2113]
b *= 1 / sqrt(dim)                                              [L2114]
b = rmsnorm_batched(b, per_layer_proj_norm,
                     batch=n_layers, n=n_embd_per_layer)        [L2117]
per_layer_inp = (a + b) * (1/sqrt(2))                           [L2121–L2122]
```

### 1e. Sandwich RMSNorm sites

Per the L1–L15 module docstring, every layer applies **four** RMSNorm
weights to the residual stream (vs. two on Qwen 3.5):

| Site                            | gemma4.rs site                     | Qwen 3.5 equivalent |
|---------------------------------|------------------------------------|---------------------|
| pre-attention `input_layernorm` | L2223 (sliding) / L2467 (full)     | `attn_norm` (yes)   |
| **post-attention `post_attention_layernorm`** | L2344 / L2559         | **none**            |
| pre-FFN `pre_feedforward_layernorm`           | L2366 / L2571         | `ffn_norm` (yes)    |
| **post-FFN `post_feedforward_layernorm`**     | L2394 / L2587         | **none**            |

Plus `q_norm` / `k_norm` (head-dim RMSNorm absorbing the `1/sqrt(d)`
scale) and the no-scale `v_norm` (ones-vector trick). MoE layers add
three more (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`,
`post_feedforward_layernorm_2`) — totaling 7 norm sites per MoE layer.

### 1f. Final softcap

`gpu.logit_softcap_f32(&scratch.logits, vocab_size, 30.0)` (L2160).
`tanh(x[i] / 30) * 30` over the entire vocab in one launch. Emitted only when `final_logit_softcapping > 0.0`. NOT a per-attention-layer softcap (that was Gemma 2; Gemma 4 dropped it).

## 2. Kernel inventory

Format: kernel-name | source-file | qwen35-uses? | gemma-specific? | dtype variants exercised by gemma4.

Generated by walking every `gpu.<method>` call in `/tmp/gemma4_src.rs` and cross-checking the same method in `/tmp/qwen35_src.rs`.

| Kernel (Rust method)                        | `kernels/src/<file>`               | qwen35? | Gemma-specific? | DType variants used |
|---------------------------------------------|------------------------------------|--------|-----------------|---------------------|
| `embedding_lookup`                          | `embedding.hip` (F32)              | Y      | N               | F32                 |
| `embedding_lookup_q8`                       | `embedding_q8.hip`                 | Y      | N               | Q8_0                |
| `embedding_lookup_hfq4g256`                 | `embedding_hfq4g256.hip`           | Y      | N               | HFQ4G256            |
| `embedding_lookup_hfq4g128`                 | `embedding_hfq4g128.hip`           | Y      | N               | HFQ4G128            |
| `scale_f32`                                 | (small util)                       | Y      | N               | F32                 |
| `rmsnorm_f32`                               | `rmsnorm.hip`                      | Y      | N               | F32                 |
| `rmsnorm_batched`                           | `rmsnorm_batched.hip`              | Y      | N               | F32                 |
| `fused_rmsnorm_rotate_mq`                   | `fused_rmsnorm_mq_rotate.hip`      | Y      | N               | F32 in / MQ4G256-precondition |
| `fused_qkv_hfq4g256`                        | `fused_qkv_hfq4g256.hip`           | Y      | N               | MQ4G256             |
| `fused_gate_up_hfq4g256`                    | `fused_gate_up_hfq4g256.hip`       | Y      | N               | MQ4G256             |
| `fused_qkvza_hfq4g256`                      | `fused_qkvza_hfq4g256.hip`         | Y (MoE)| N               | MQ4G256             |
| `weight_gemv` (dispatcher)                  | many `gemm_*g256*.hip`             | Y      | N               | F32/Q8_0/Q4K/HFQ{2,3,4,6}G{128,256}/MQ{2,3,4,6,8}G256/MG4G256 |
| `rope_f32` (full-rotation interleaved)      | `rope.hip`                         | N (uses partial_interleaved) | Mixed | F32 |
| `rope_partial_halved_f32`                   | `rope_partial_halved.hip`          | N      | **Y**           | F32                 |
| `rope_partial_interleaved_f32`              | `rope_partial_interleaved.hip`     | Y      | N               | (used by qwen35; gemma4 does NOT call this) |
| `kv_cache_write`                            | `kv_cache_write.hip` (F32)         | Y      | N               | F32                 |
| `kv_cache_write_q8_0`                       | `kv_cache_write_q8_0.hip`          | Y      | N               | Q8_0                |
| `kv_cache_write_asym2_fused`                | `kv_cache_write_asym2_fused.hip`   | Y      | N               | asym2               |
| `kv_cache_write_asym3_fused`                | `kv_cache_write_asym3_fused.hip`   | Y      | N               | asym3 (incl. hd=512) |
| `kv_cache_write_asym4_fused`                | `kv_cache_write_asym4_fused.hip`   | Y      | N               | asym4               |
| `attention_f32`                             | `attention.hip`                    | Y      | N               | F32 KV (full-attn FP32 fallback path) |
| `attention_flash_q8_0`                      | `attention_flash_q8_0_tile.hip`    | Y      | N (signature conflict — see §7) | Q8_0 KV |
| `attention_flash_asym2`                     | `attention_flash_asym2_tile.hip`   | Y      | N (signature conflict) | asym2 KV |
| `attention_flash_asym3`                     | `attention_flash_asym3_tile.hip` + `_hd512.hip` | Y | N (signature conflict; `_hd512` variant is gemma-only) | asym3 KV (hd=256 sliding, hd=512 full) |
| `attention_flash_asym4`                     | `attention_flash_asym4_tile.hip`   | Y      | N (signature conflict) | asym4 KV |
| `gelu_tanh_f32`                             | `gelu_tanh.hip`                    | Y (Qwen3.5-VL only) | N | F32 |
| `gelu_erf_f32`                              | `gelu_erf.hip`                     | Y (MoE expert opt-in) | N | F32 |
| `mul_f32`                                   | (small util)                       | Y      | N               | F32 |
| `add_f32`                                   | `add.hip`                          | Y      | N               | F32 |
| `add_inplace_f32`                           | `add_inplace.hip`                  | Y      | N               | F32 |
| `moe_softmax_topk_renorm_k8`                | (kernel name TBD, MoE-shared)      | Y (3.5 MoE) | N (k_top=8 hardcoded — same for both) | F32 |
| `gemv_hfq4g256_moe_gate_up_k8_indexed`      | (indexed MoE GEMV, MoE-shared)     | Y      | N               | MQ4G256 (gate-side via FWHT precondition) |
| `gemv_q8_0_moe_down_residual_scaled_k8_indexed` | (indexed Q8 down)              | Y      | N               | Q8_0 |
| **`logit_softcap_f32`**                     | **`logit_softcap.hip`**            | **N**  | **Y (only Gemma 2/3/4 have softcap)** | F32 |

### 2a. Net-new kernels for the modular re-port

Two HIP files exist in `kernels/src/` only on `origin/gemma4` (and on the
already-rebased `gemma4-rebased-2026-05-07` branch); they are absent on
`master`:

* `kernels/src/logit_softcap.hip` — final-logit softcap, `tanh(x/cap)*cap`. 21 LOC. No MMA. Trivial.
* `kernels/src/rope_partial_halved.hip` — proportional partial RoPE with HF rotate_half pairing `(i, i+head_dim/2)`. 64 LOC. No MMA. Differs from `rope_partial_interleaved.hip` (Qwen 3.5's path) in pair geometry and `n_rot_pairs` semantics (count-of-pairs vs Qwen's count-of-rotated-dims).

Plus their batched twin `rope_partial_halved_batched.hip` (used by the prefill batched path at gemma4.rs L2624).

## 3. Shape reference (gemma-4-31B dense, validated against `Gemma4Config` parser at L260–L373 plus arch-report.md §2)

```
dim                = 5376    (config.hidden_size)
n_layers           = 60
vocab_size         = 262144
n_heads            = 32      (NOT 24 or 12 as suggested in the task prompt — config.json: num_attention_heads=32)
sliding_head_dim   = 256     (config.head_dim)
sliding_n_kv_heads = 16      (config.num_key_value_heads)
sliding_window     = 1024    (config.sliding_window)
sliding_rope_theta = 10000
full_head_dim      = 512     (config.global_head_dim — 2× sliding)
full_n_kv_heads    = 4       (config.num_global_key_value_heads — GQA ratio = 32:4 = 8)
full_rope_theta    = 1e6
full_partial_rotary_factor = 0.25  → n_rot_pairs = 64 of 256 pairs
attention_k_eq_v   = true (31B / 26B-A4B); false (E2B / E4B)
hidden_dim (FFN)   = 21504
layer_types        = [S, S, S, S, S, F] × 10  → 5:1 sliding:full
embed_scale        = sqrt(5376) ≈ 73.32
final_logit_softcapping = 30.0
n_embd_per_layer   = 0   (31B / 26B-A4B); 256 (E2B / E4B)
num_kv_shared_layers = 0 (31B / 26B-A4B); 18 (E4B); 20 (E2B)
```

The task prompt's "n_heads=24 (sliding) / 12 (full)" is incorrect — gemma-4-31B uses `n_heads=32` for both, and varies `n_kv_heads` (16 sliding, 4 full) instead. Gemma 4's GQA collapse is on the KV side, not the Q side. Verified via the explicit `let n_heads = tc.get("num_attention_heads")?` parse at gemma4.rs L273, used unchanged at L2206 and L2451.

### 3a. Per-layer tensor shapes

| Tensor                          | Shape (sliding)         | Shape (full)            | Notes |
|---------------------------------|-------------------------|-------------------------|-------|
| `input_layernorm.weight`        | `[5376]`                | `[5376]`                | x*weight (no +1 shift) |
| `post_attention_layernorm.weight` | `[5376]`              | `[5376]`                | sandwich post-attn |
| `pre_feedforward_layernorm.weight` | `[5376]`             | `[5376]`                | sandwich pre-FFN |
| `post_feedforward_layernorm.weight` | `[5376]`            | `[5376]`                | sandwich post-FFN |
| `layer_scalar`                  | `[1]`                   | `[1]`                   | mirrored to host f32 |
| `q_proj.weight`                 | `[8192, 5376]` (32×256) | `[16384, 5376]` (32×512)| |
| `k_proj.weight`                 | `[4096, 5376]` (16×256) | `[2048, 5376]` (4×512)  | |
| `v_proj.weight`                 | `[4096, 5376]`          | `None` (k_eq_v=true) / `[2048, 5376]` (E-series) | |
| `o_proj.weight`                 | `[5376, 8192]`          | `[5376, 16384]`         | |
| `q_norm.weight`                 | `[256]`                 | `[512]`                 | per-head-dim |
| `k_norm.weight`                 | `[256]`                 | `[512]`                 | per-head-dim |
| (`v_norm.weight`)               | (none — no-scale)       | (none — no-scale)       | uses `v_norm_ones_full` ones-buffer |
| `gate_proj.weight`              | `[21504, 5376]`         | `[21504, 5376]`         | (E2B last-20 layers: `[43008, 5376]`) |
| `up_proj.weight`                | `[21504, 5376]`         | `[21504, 5376]`         | |
| `down_proj.weight`              | `[5376, 21504]`         | `[5376, 21504]`         | |

### 3b. Model-level tensors

| Tensor                          | Shape                              | Notes |
|---------------------------------|------------------------------------|-------|
| `embed_tokens.weight`           | `[262144, 5376]` Q8F16 forced      | aliased as `lm_head` (tie_word_embeddings=true) |
| `final_norm.weight` (= `model.language_model.norm.weight`) | `[5376]` | x*weight (no +1) |
| `embed_tokens_per_layer` (E-only) | `[262144, n_embd_per_layer * n_layers]` Q8F16 | size 262144 × 256 × n_layers on E |
| `per_layer_model_projection`    | `[n_embd_per_layer * n_layers, 5376]` | |
| `per_layer_projection_norm`     | `[n_embd_per_layer]`               | applied per-slot via rmsnorm_batched |

### 3c. Scratch buffer sizes (Gemma4Scratch::new at L1369)

```
x, residual, tmp        = [dim]                  = [5376]
q                       = [max(Hq*d_s, Hq*d_f)]   = [16384]   (full layer dominates)
k, v                    = [max(Hkv_s*d_s, Hkv_f*d_f)] = [4096] (sliding dominates: 16*256 vs 4*512)
attn_out                = same as q              = [16384]
gate_ffn, up_ffn, ffn_hidden = [max_ffn_hd]      = [21504]    (E2B doubles last 20 layers)
ffn_out                 = [dim]                  = [5376]
logits                  = [vocab_size]           = [262144]
v_norm_ones_full        = [full_head_dim]        = [512]   (only full layers use; ones-filled)
flash_partials          = [n_heads * max_tiles_full * (2 + full_head_dim)]
                          where max_tiles_full = ceil(max_seq / 128)
tmp_rot                 = [dim]                  = [5376]   (FWHT-rotated-x staging for fused-MQ4 path)
pos_buf                 = 4 bytes (single i32)
```

Per-layer-embedding scratch (zero-sized when `n_embd_per_layer == 0`):
`per_layer_inp`/`per_layer_inp_proj` = `[n_embd_per_layer * n_layers]`,
`per_layer_tmp` = `[n_embd_per_layer]`,
`per_layer_out`/`per_layer_pe_in` = `[dim]`.

MoE scratch (zero-sized when `enable_moe_block == false`): see L1437–L1460
for the full set; `moe_pre2_rot` + `moe_expert_gate_batch` + `moe_expert_up_batch` + `moe_expert_hidden_batch` + `moe_topk_weights_fused` are the indexed-fast-path stagers.

## 4. KV cache layout

Two separate `llama::KvCache` instances per loaded model — `kv_sliding`
and `kv_full` — sized by per-type own-KV layer count from `kv_share_plan()`
(L1117–L1143 in daemon.rs).

* **kv_sliding:** `n_sliding_own` slots, each shaped for `(sliding_n_kv_heads, sliding_head_dim) = (16, 256)`. Quant mode = whatever the user asked for (`asym3` default, or `q8`/`asym4`/`asym2` per CLI).
* **kv_full:** `n_full_own` slots, each shaped for `(full_n_kv_heads, full_head_dim) = (4, 512)`. Quant mode is asym3 ONLY when sliding is also asym3 AND `HIPFIRE_GEMMA4_FULL_KV != fp32`. All other configurations fall back to FP32 because only the asym3 flash kernel ships an `_hd512` variant; asym4/asym2/q8 hardcode the hd=256 layout and would silently truncate. Hard-error at gemma4.rs L2536–L2541 if a full-attn layer hits a non-asym3 quant mode.

`kv_slot[layer_idx]` indirection from `kv_share_plan()`:

* On 31B/26B-A4B (`num_kv_shared_layers == 0`): every layer owns its KV. `kv_slot` is "running count of same-type layers seen so far". Sliding layer indices `0,1,2,3,4` map to slots `0,1,2,3,4`; full layer index `5` maps to slot `0` of `kv_full`; sliding layers `6,7,8,9,10` map to slots `5,6,7,8,9`; etc.
* On E2B/E4B (sharing enabled): the LAST `num_kv_shared_layers` layers do NOT compute K/V. They read the anchor slot — sliding shared layers all read sliding-anchor at index `n_kv_start - 2`'s slot; full shared layers all read full-anchor at index `n_kv_start - 1`'s slot. `Gemma4Config::kv_share_plan()` (L193–L257) builds this and asserts the anchor types match (sliding anchor MUST be a Sliding-typed layer; full anchor MUST be Full).

The forward path passes `owns_kv: bool` to layer-decode and skips the `kv_cache_write_*` step when false; it also passes `n_heads_k = 0` to the RoPE kernel to skip the K loop on shared layers (L2261, L2507). The KV cache slot's existing data — written by the anchor layer earlier in the same forward pass — is what shared layers attend against.

## 5. Daemon-side quirks (`crates/engine/examples/daemon.rs` on `origin/gemma4`)

Gemma 4 takes a dedicated `generate_gemma4` codepath (L2508–L2944, ~440 LOC) — does NOT share the Qwen 3.5 generate loop because the chat template, KV-cache split (sliding+full), and forward API (`forward_scratch(token, pos, kv_sliding, kv_full, scratch)`) all differ structurally.

### 5a. Chat template

Hardcoded at L2528–L2553 / L2615–L2630:

```
<bos><start_of_turn>user
{system}\n\n{user}<end_of_turn>
<start_of_turn>model
```

System prompt is injected **only on seq_pos == 0** (first turn or post-reset turn) — re-injecting on every turn is documented as confusing the model and bloating KV (L2555–L2559).

Activation gate (L2546–L2553):

* `HIPFIRE_GEMMA4_CHAT=on/1/true` forces template on
* `HIPFIRE_GEMMA4_CHAT=off/0/false` forces off
* default: ON when `model_path` contains `-it` or `_it`, OR when `system_prompt` is provided

`<start_of_turn>` and `<end_of_turn>` are NOT single tokens. They each tokenize as a 7-token sequence (`<`, `start`, `_`, `of`, `_`, `turn`, `>`) by the SPM-BPE tokenizer (L2539–L2541).

### 5b. End-of-turn detection

Two parallel detectors (L2742–L2790):

1. **Byte-string scan:** keep a rolling window of the decoded byte stream and search for the literal `b"<end_of_turn>"`. Stops at first match.
2. **Compact-EOT token:** `const GEMMA4_END_OF_TURN_TOKEN: u32 = 106;` (L2749) — token 106 in the Gemma 4 vocabulary IS a single end-of-turn marker that the model sometimes emits directly (Gemma's special-tokens table puts EOT at vocab[106] alongside the multi-byte `<end_of_turn>` literal). When it's seen, stop immediately.

Streaming hold-back: the last `len("<end_of_turn>") - 1 = 12` bytes are buffered (not yet emitted to stdout) when the running tail looks like a prefix of the marker (L2843–L2862), so partial-marker bytes don't leak to the user.

### 5c. KV / scratch sizing

`max_seq` defaults to 131072 for Gemma 4 (`max_position_embeddings` from config). The daemon forces `physical_cap = max_seq` (L1168–L1172) — Gemma 4 does NOT use CASK m-folding (Qwen3.5-only) nor DFlash (Qwen3.5-only) in this branch.

`init_scratch_constants(gpu, &scratch, full_head_dim)` (L1148–L1150) is a one-time loader-side init that fills `v_norm_ones_full` with ones. Forgetting this call would make the no-scale v_norm path produce zeros (RMS divides by `sqrt(eps)`).

### 5d. Multi-turn KV consistency knobs

* `HIPFIRE_GEMMA4_FULL_KV=fp32` — force full-attn layers to FP32 KV (escape hatch when asym3 quant noise flips argmax on small-context decode; specifically observed on 26B-A4B at single-token decode, see L1115–L1116).
* `HIPFIRE_GEMMA4_BATCHED_PREFILL=1` — opt into the L2634 `forward_prefill_batch` path. **Default OFF** — there is a known correctness divergence on 31B with `_BATCHED_KEQV=1` (L2674–L2677). Per-token prefill loop is the safe path.
* `HIPFIRE_GEMMA4_FUSED_PROJ=0/off/false` — force fall-back to non-fused QKV / gate_up GEMVs (default ON). Diagnostic only.
* `HIPFIRE_GEMMA4_MOE_FUSED=0` — force legacy serialized 8× per-expert MoE path. Default ON when MQ4G256.
* `HIPFIRE_MOE_BYPASS=1` / `HIPFIRE_MOE_ZERO=1` — debug isolation. Bypass disables the entire MoE branch; Zero keeps the branch but mutes experts.
* `HIPFIRE_GEMMA4_PLE_GELU_TANH=1` — diagnostic swap for E-series PLE branch (default `gelu_erf` per L2009–L2018).
* `HIPFIRE_GEMMA4_NORM_PLUS_ONE=1` — diagnostic +1 shift on RMSNorm weights (Gemma 2/3 convention). Default OFF — Gemma 4 is plain `x*weight`.
* `HIPFIRE_DUMP_LAYER0=1` / `HIPFIRE_DUMP_MOE=1` / `HIPFIRE_DUMP_MOE_VALS=1` — magnitude dumps for debugging.

## 6. MG4G256 quantization spec

Source: `/tmp/quantize_src.rs` L439–L523 (`fn quantize_mg4g256`).

### 6a. Algorithm (per 256-element group)

```
INPUT:  group: [f32; 256]   (slice of weight tensor — pre-rotation)
        signs1, signs2: production FWHT seeds (same as MQ4G256)
OUTPUT: 136 bytes:  [scale: f32; 4][min: f32; 4][q: u4; 256]

1. cpu_fwht_256(group, signs1, signs2)
        # Identical FWHT preconditioning to MQ4G256.
        # signs1/signs2 are the production seeds shared with MQ4 — kernel-side
        # x rotation cancels at GEMV time.

2. percentile_clip:
        sorted = group.copy()
        select_nth_unstable(sorted, 5)    → p02 = sorted[5]
        select_nth_unstable(sorted, 250)  → p98 = sorted[250]
        # P02 = 5th-smallest of 256 values  (~2% from bottom)
        # P98 = 251st (=index 250) of 256   (~2% from top)

3. degenerate fallback:
        if |p98 - p02| < 1e-12:
            (lo, hi) = (true_min, true_max)    # degenerate / all-zero block
        else:
            (lo, hi) = (p02, p98)

4. quantize:
        scale = (hi - lo) / 15.0
        inv_scale = 1.0 / scale  (or 0 if range == 0)
        for i in 0..128:
            lo_v = group[2i];   hi_v = group[2i+1]
            lo_q = clamp((lo_v - lo) * inv_scale + 0.5, 0..15)  as u8
            hi_q = clamp((hi_v - lo) * inv_scale + 0.5, 0..15)  as u8
            packed[i] = lo_q | (hi_q << 4)        # 2 nibbles per byte
        write [scale, lo, packed[0..128]] to 136-byte block
```

### 6b. Differences from MQ4G256 (`fn quantize_mq4g256`, L528 onward)

|                       | MQ4G256                            | MG4G256                            |
|-----------------------|------------------------------------|------------------------------------|
| FWHT seeds            | production signs1/signs2           | identical                          |
| Calibration `lo`/`hi` | true `min`, `max` of rotated block | percentile-clip P02/P98            |
| Saturation            | none — every value fits exactly    | top + bottom ~2% saturate to edges |
| Block size            | 136 B                              | 136 B (identical layout)           |
| Engine dtype          | `DType::MQ4G256`                   | `DType::MQ4G256` (loader collapses qt=19 → MQ4G256 at L716, L1044) |

### 6c. Why two formats

MQ4 spends codebook range on the FWHT-rotated distribution's tails (~3σ on each side post-FWHT). After rotation, the bulk of values lives within ~0.4σ of zero; if the codebook is calibrated to the full range, the bulk gets only 4-5 of the 16 codes, badly under-quantized. Gemma 4's per-layer learned `layer_scalar` amplifies per-element error layer-over-layer (60 layers); MQ4 hits a single-token attractor on Gemma 4 at this compounded error level. MG4 trades ~4% saturated values for ~30% better resolution per bin in the bulk; saturating ~10/256 values is cheaper in MSE than the bin-width loss avoided.

### 6d. Engine impact

**Zero engine code change.** The decode path is `q * scale + min` for both formats. The `quant_type=19` enum value is mapped to `DType::MQ4G256` everywhere (gemma4.rs L716, L1044; quantize main.rs L1116). Existing kernels (`gemv_mq4g256_*`, `fused_qkv_hfq4g256` when MQ4G256 — note the kernel-name mismatch: `_hfq4g256` is the historical name, `MQ4G256` is the enum) handle MG4 transparently.

### 6e. Quantizer activation

CLI: `--format mg4` or `--format mg4g256` sets `use_mg4g256 = true` (quantize.rs L1808). Triggers the percentile-clip code path at L2270–L2284 for 2D weights, including `embed_tokens.weight` and `embed_tokens_per_layer.weight` (forced to Q8F16 instead per L2262 — `is_embed` carve-out). MoE expert tensors get MG4 too when `--format mg4` is set (L2030).

## 7. Open questions for the modular re-port

These are details that cannot be derived from reading gemma4.rs alone — the modular re-port will need empirical validation or external reference checks:

1. **Sliding-window kernel signature collision (HARD BLOCKER, see arch-report §3 Gap 1).**
   On `origin/gemma4`, FOUR `attention_flash_*_tile{,_batched}.hip` kernels carry a trailing `uint32_t window_size` argument. On `master` HEAD `25df27f`, the same kernels do NOT have this arg. **Both qwen35.rs and gemma4.rs on the gemma4 branch already pass `window_size` (with qwen35 passing `0` for full causal, gemma4 passing `sliding_window=1024` or `0`).** The port-time conflict: the gemma4 branch's `window_size` extension was applied to kernels that have since been substantially rewritten on master (gfx906 wave64 prefetch / dp4a / MMQ tile redesign). The diff cannot be applied verbatim. Three-way merge on each kernel + matching trailing `u32` arg on the Rust dispatch helpers. Affects: `attention_flash_asym2_tile`, `attention_flash_asym3_tile`, `attention_flash_asym3_tile_hd512` (gemma-only), `attention_flash_asym4_tile`, `attention_flash_q8_0_tile` — and their `_batched` twins for prefill.

2. **gelu_pytorch_tanh in SwiGLU.** gemma4 calls `gpu.gelu_tanh_f32` at L2382/L2578 standalone, then `gpu.mul_f32` separately. Master's qwen35 SwiGLU goes through `weight_gemv_swiglu_residual` which **hardcodes silu**. Two fixes possible — either (a) call the unfused `gelu_tanh_f32` + `mul_f32` pair like gemma4 does, costing one extra launch, or (b) add a `weight_gemv_swiglu_gelutanh_residual` kernel variant. The branch's gemma4.rs already chose option (a). Decision deferrable to perf measurement.

3. **PLE branch activation mismatch (gelu_erf vs HF spec).** The L2009–L2018 comment explicitly notes that HF Gemma4TextDecoderLayer wires `act_fn = ACT2FN[config.hidden_activation]` (= `gelu_pytorch_tanh`), but empirically `gelu_erf` produces materially better E4B/E2B output. Tested 2026-05-02. Reason unresolved. The port should keep `gelu_erf` as default and `HIPFIRE_GEMMA4_PLE_GELU_TANH=1` for diagnosis until a Python reference forward of layer 0 with both activations is run side-by-side.

4. **`attention_k_eq_v` ordering subtlety.** On full layers with `attention_k_eq_v=true`, V is captured by `memcpy_dtod(v, k, kv_bytes)` AFTER `weight_gemv(k_proj, x, k)` and BEFORE `rmsnorm_batched(k, k_norm, k)` overwrites k in place. The order matters — V holds the pre-k_norm bytes. See L2484–L2487. The modular re-port must preserve this exact memcpy-then-norm ordering; reordering would silently corrupt V.

5. **`physical_cap` plumbing collision.** Master's qwen35 `attention_flash_asym3` takes `kv_cache.physical_cap` as the cache-stride argument (CASK-derived). Gemma 4's `attention_flash_asym3` takes `kv_cache.max_seq` (L2281). The two values diverge under CASK. Gemma 4 has no CASK; the modular re-port must keep `max_seq` for Gemma 4 even though qwen35-shared kernel signatures use `physical_cap`. Consider whether to (a) add a separate gemma4 kernel signature, or (b) ensure `physical_cap == max_seq` for Gemma 4 at the daemon level (L1168–L1172 already does this).

6. **Default values for env knobs.** All env knobs default to "production-tuned" values per the gemma4 branch:
   * `HIPFIRE_GEMMA4_FUSED_PROJ` — default ON
   * `HIPFIRE_GEMMA4_MOE_FUSED` — default ON when MQ4G256 weights
   * `HIPFIRE_GEMMA4_BATCHED_PREFILL` — default OFF (correctness gap)
   * `HIPFIRE_GEMMA4_PLE_GELU_TANH` — default OFF (default activation = gelu_erf)
   * `HIPFIRE_GEMMA4_NORM_PLUS_ONE` — default OFF
   * `HIPFIRE_GEMMA4_FULL_KV` — empty (auto-asym3 when sliding is asym3)
   * `HIPFIRE_GEMMA4_CHAT` — empty (auto-detect on `-it` path)

7. **Token 106 as compact `<end_of_turn>`.** The daemon at L2749 hardcodes `GEMMA4_END_OF_TURN_TOKEN: u32 = 106`. This is from inspection of the SPM-BPE vocab; cross-check against tokenizer.json `added_tokens` on the modular re-port to confirm 106 is correct on every Gemma 4 variant (31B / 26B-A4B / E4B / E2B). Variant-specific vocab would silently break EOT detection.

8. **Per-expert pool device pointers as `[2 * n_exp]` F32.** The MoE indexed-GEMV path stores `n_exp × u64` device addresses in a tensor declared as `DType::F32` of length `2 * n_exp` (L1115, L1123). This relies on aligned u64 → 2× f32 pun. The modular re-port must use the same dispatch-side reinterpretation; declaring as `DType::U64` would surprise the existing kernel.

9. **Tokenizer SPM-BPE flag.** Branch commits `f867423` + `2c4f9fd` added an `is_spm_bpe: bool` field to `Tokenizer` plus a Unigram-vs-BPE detection guard. Master added `eot_id: Option<u32>` and `from_gguf_meta_json` to the same file. Re-apply additively — the two diffs touch the constructor but not overlapping line ranges.

10. **Quantizer arch_id=7 dispatch.** Branch commit `3db5759` added 27 LOC to `crates/hipfire-quantize/src/main.rs` for the gemma4 case (HFQ tensor naming + which weights need Q8 floor). PR #180 on master changed the same area (router-Q8 fix). Re-apply additively. Specifically: `is_q8_tensor` (L1238 in branch) carve-outs need to merge with master's PR-180 additions (`mlp.gate.weight`, `mlp.shared_expert_gate.weight`).

## 8. Summary: port-time conflicts and net-new files

### 8a. Net-new kernels (no master equivalent — drop in directly)

* `kernels/src/logit_softcap.hip`
* `kernels/src/rope_partial_halved.hip`
* `kernels/src/rope_partial_halved_batched.hip`

### 8b. Shared-name kernels with **signature conflict** (port-time merge required)

These appear in both gemma4.rs and qwen35.rs on the gemma4 branch with a trailing `window_size: u32` arg. On master, the kernels lack this arg AND have been independently rewritten (gfx906 wave64 / dp4a / MMQ). Three-way merge per file; signature update on Rust dispatch helpers.

* `kernels/src/attention_flash_asym2_tile.hip` + `_batched.hip`
* `kernels/src/attention_flash_asym3_tile.hip` + `_batched.hip` + `_hd512.hip` (the `_hd512` variant is gemma-only — straight add, not a merge)
* `kernels/src/attention_flash_asym4_tile.hip` + `_batched.hip`
* `kernels/src/attention_flash_q8_0_tile.hip`

The Rust-side dispatch helpers (`gpu.attention_flash_asym3`, etc.) need a trailing `window_size: u32` parameter on master.

### 8c. Shared kernels with **identical signatures** (no conflict, just register)

Every kernel in §2's table whose row says "qwen35? Y" and "Gemma-specific? N" is already on master with the right shape. The modular gemma4 crate just needs to call them with Gemma-specific arg values (heads count, head_dim, theta).

### 8d. Files in the gemma4 branch that need adapting, not literal porting

* `crates/engine/src/gemma4.rs` (1077 LOC after import patches in arch-report §1.4) — port to `crates/hipfire-arch-gemma4/src/gemma4.rs`.
* `crates/engine/src/gemma4_vision.rs` (169 LOC) — port to `crates/hipfire-arch-gemma4/src/gemma4_vision.rs` (vision is Phase 7, scaffolded only).
* `crates/engine/examples/daemon.rs` 126-LOC arch_id=7 dispatch + `generate_gemma4` (440 LOC) — port to `crates/hipfire-runtime/examples/daemon.rs`.
* `crates/hipfire-quantize/src/main.rs` 27-LOC arch=7 case + `quantize_mg4g256` (~85 LOC) — re-apply additively over PR #180.
* `crates/hipfire-runtime/src/tokenizer.rs` SPM-BPE field + guard — re-apply additively.

End of spec.
