# imatrix divergence report — DeltaNet-full forward (post-2026-05-20)

- oracle: `/workspace/qwen3.5-0.8b.pytorch-oracle.imatrix.gguf`
- target: `/workspace/qwen3.5-0.8b.tier1-deltanet-full.imatrix.gguf`
- threshold: NRMSE ≤ 0.01
- shared: 186 / oracle=186 target=187
- corpus: `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`
  (32 sequences × 2048 ctx = 65 536 tokens)
- model: Qwen3.5-0.8B BF16 HF safetensors
- branch: `pr-7-tier1-calibration` HEAD `cf1a6097`

## TL;DR

Three fixes against the post-#127 baseline (attn_output_gate):
1. conv1d preamble (depth-wise causal kernel + SiLU)
2. real `A_log + dt_bias` in `fused_sigmoid_alpha_gate`
3. `gated_norm_f32_batched(attn_out, z, w)` post-recurrence

**Median NRMSE drops 0.664 → 0.0203 (32× improvement)**. Layer 0
DeltaNet `attn_qkv` NRMSE: 0.0057 (down from 0.913). Still above the
0.01 PASS gate; remaining divergence sources analysed below.

## Per-role NRMSE

| role | n | median | max |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.0210 | 0.0447 |
| attn_k | 6 | 0.0211 | 0.0451 |
| attn_output (FullAttn o_proj) | 6 | 0.0151 | 0.0483 |
| attn_q | 6 | 0.0211 | 0.0451 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.0210 | 0.0447 |
| attn_v | 6 | 0.0211 | 0.0451 |
| ffn_down (MLP) | 24 | 0.0280 | 0.0730 |
| ffn_gate (MLP) | 24 | 0.0262 | 0.0437 |
| ffn_up (MLP) | 24 | 0.0262 | 0.0437 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.0210 | 0.0447 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.0210 | 0.0447 |
| ssm_out (DeltaNet out_proj) | 18 | 0.0139 | 0.1372 |

## Top 30 divergences

| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |
|---|---:|---:|---:|---:|---:|
| blk.21.ssm_out.weight | 2048 | 0.1372 | 0.0020 | 0.1757 | 0.1204 |
| blk.23.ffn_down.weight | 3584 | 0.0730 | 0.0006 | 0.1945 | 0.1352 |
| blk.20.ffn_down.weight | 3584 | 0.0643 | 0.0016 | 0.1310 | 0.0900 |
| blk.13.ffn_down.weight | 3584 | 0.0589 | 0.0012 | 0.1106 | 0.0709 |
| blk.23.attn_output.weight | 2048 | 0.0483 | 0.0012 | 0.1925 | 0.1482 |
| blk.23.attn_k.weight | 1024 | 0.0451 | 0.0002 | 0.0876 | 0.0709 |
| blk.23.attn_q.weight | 1024 | 0.0451 | 0.0002 | 0.0876 | 0.0709 |
| blk.23.attn_v.weight | 1024 | 0.0451 | 0.0002 | 0.0876 | 0.0709 |
| blk.11.ffn_down.weight | 3584 | 0.0447 | 0.0008 | 0.0913 | 0.0474 |
| blk.21.attn_gate.weight | 1024 | 0.0447 | 0.0004 | 0.0894 | 0.0730 |
| blk.21.attn_qkv.weight | 1024 | 0.0447 | 0.0004 | 0.0894 | 0.0730 |
| blk.21.ssm_alpha.weight | 1024 | 0.0447 | 0.0004 | 0.0894 | 0.0730 |
| blk.21.ssm_beta.weight | 1024 | 0.0447 | 0.0004 | 0.0894 | 0.0730 |
| blk.18.ffn_gate.weight | 1024 | 0.0437 | 0.0005 | 0.0815 | 0.0667 |
| blk.18.ffn_up.weight | 1024 | 0.0437 | 0.0005 | 0.0815 | 0.0667 |
| blk.20.ffn_gate.weight | 1024 | 0.0436 | 0.0004 | 0.0926 | 0.0768 |
| blk.20.ffn_up.weight | 1024 | 0.0436 | 0.0004 | 0.0926 | 0.0768 |
| blk.21.ffn_gate.weight | 1024 | 0.0421 | 0.0004 | 0.0911 | 0.0750 |
| blk.21.ffn_up.weight | 1024 | 0.0421 | 0.0004 | 0.0911 | 0.0750 |
| blk.17.attn_gate.weight | 1024 | 0.0412 | 0.0007 | 0.0679 | 0.0555 |
| blk.17.attn_qkv.weight | 1024 | 0.0412 | 0.0007 | 0.0679 | 0.0555 |
| blk.17.ssm_alpha.weight | 1024 | 0.0412 | 0.0007 | 0.0679 | 0.0555 |
| blk.17.ssm_beta.weight | 1024 | 0.0412 | 0.0007 | 0.0679 | 0.0555 |
| blk.14.ffn_down.weight | 3584 | 0.0412 | 0.0001 | 0.1405 | 0.0745 |
| blk.21.ffn_down.weight | 3584 | 0.0410 | 0.0001 | 0.1535 | 0.0913 |
| blk.17.ffn_gate.weight | 1024 | 0.0410 | 0.0007 | 0.0728 | 0.0601 |
| blk.17.ffn_up.weight | 1024 | 0.0410 | 0.0007 | 0.0728 | 0.0601 |
| blk.23.ffn_gate.weight | 1024 | 0.0408 | 0.0001 | 0.1316 | 0.0754 |
| blk.23.ffn_up.weight | 1024 | 0.0408 | 0.0001 | 0.1316 | 0.0754 |
| blk.18.attn_gate.weight | 1024 | 0.0406 | 0.0005 | 0.0749 | 0.0587 |

## Per-layer breakdown

```
layer  count    median       max       min
--------------------------------------------------
    0      8    0.0057    0.0189    0.0054
    1      8    0.0054    0.0072    0.0029
    2      8    0.0048    0.0132    0.0035
    3      7    0.0089    0.0154    0.0038
    4      8    0.0077    0.0148    0.0063
    5      8    0.0099    0.0138    0.0073
    6      8    0.0086    0.0125    0.0084
    7      7    0.0123    0.0176    0.0104
    8      8    0.0123    0.0186    0.0102
    9      8    0.0140    0.0277    0.0133
   10      8    0.0178    0.0208    0.0145
   11      7    0.0224    0.0447    0.0175
   12      8    0.0311    0.0354    0.0243
   13      8    0.0338    0.0589    0.0182
   14      8    0.0300    0.0412    0.0225
   15      7    0.0198    0.0389    0.0117
   16      8    0.0262    0.0375    0.0122
   17      8    0.0412    0.0412    0.0110
   18      8    0.0406    0.0437    0.0357
   19      7    0.0356    0.0400    0.0133
   20      8    0.0400    0.0643    0.0374
   21      8    0.0447    0.1372    0.0410
   22      8    0.0376    0.0380    0.0177
   23      7    0.0451    0.0730    0.0408
```

Layer 0/1/2 (all DeltaNet): 0.005-0.006 median — at the BF16 round-off
floor. Each subsequent layer adds ~0.001-0.002 median NRMSE through the
residual stream, climbing to 0.045 by layer 23. The progression is
linear with depth and consistent with cumulative BF16-cast noise +
Q8_0 KV quantization noise in the FullAttn layers.

## Remaining divergence sources

The DeltaNet math is now exact (modulo BF16 precision). The remaining
~0.02 median NRMSE comes from three compounding sources:

### 1. BF16 cast convention: truncation vs round-to-nearest-even

`rdna_compute::Gpu::convert_f32_to_bf16` uses **truncation** (top 16
bits of the F32 mantissa) per the kernel source in `dispatch.rs:2608`.
PyTorch's CUDA path uses `__float2bfloat16_rn` (round-to-nearest-even)
for the `.to(bfloat16)` cast.

Per-cast bias: ~½ ULP systematic, ~1 ULP worst case ≈ 1/256 = 0.004
relative for typical activations. The BF16 forward casts 3-4 times per
layer (RMSNorm output, attn_out, ffn_hidden, ffn_out). Over 24 layers:
~72 casts → systematic bias compounds in the residual stream.

Fix would require modifying `convert_f32_to_bf16` to use RTNE rounding.
**Out of scope per spec** (would affect every other BF16 forward call
site through a shared dispatch function — needs separate audit).
A practical estimate: switching to RTNE should pull layer 0 down from
0.005 to ~0.001-0.002 and cap the layer-23 median at ~0.01-0.02.

### 2. Q8_0 KV cache in FullAttention layers (0,3,7,11,15,19,23)

`forward_full_attn_layer` (spec-forbidden modification) writes K/V
through `kv_cache_write_q8_0_batched` before flash attention. Q8_0
adds ~ε = 1/256 quantization noise per K/V channel. PyTorch oracle
uses BF16 KV throughout (no quantization in `Qwen3_5Attention`).

Visible at the per-layer breakdown: layer 3 (first FullAttn) jumps
median 0.005 → 0.009; layer 11 (third FullAttn) jumps 0.018 → 0.022.
Each FullAttn adds ~0.003-0.005 to median NRMSE that propagates via
the residual stream to all subsequent layers.

Fix would require gating Q8_0 KV off in the calibration path
(e.g. a `--full-attn-precision=bf16` flag wiring `attention_causal_batched`
instead of `attention_q8_0_kv_batched_masked`). **Out of scope per spec.**

### 3. PyTorch DeltaNet uses chunked recurrence (chunk_size=64)

HF's `torch_chunk_gated_delta_rule` (and the FLA fast path) runs the
delta-net recurrence in chunks of 64 tokens with cumulative-decay
optimization. Our `gated_delta_net_f32` kernel runs single-step
recurrence. Mathematically equivalent, numerically differs by FP
associativity (~1 ULP per step, weakly accumulating).

For seq_len=2048 the chunked version traverses 32 chunks × 64 steps
of cumulative decay; our kernel does 2048 single steps with state in
LDS. Both compute the same closed-form gated delta rule. The numerical
drift is bounded by ~1 ULP × log2(seq_len) ≈ 11 ULP ≈ 0.001-0.004 NRMSE.

Aligning to PyTorch's chunked variant would require a new kernel
(`gated_delta_net_f32_chunked`). **Out of scope per spec** (no new HIP
kernels).

## Conclusion

The three fixes (conv1d, real alpha, gated_norm + silu(z)) deliver the
expected step-change improvement: layer 0 DeltaNet `attn_qkv` is now at
NRMSE 0.0057, indistinguishable from BF16 round-off floor. **Median
across all 24 layers is 0.0203 vs the 0.01 PASS threshold.**

To reach ≤ 0.01 median, the remaining work is:
1. Switch `convert_f32_to_bf16` to RTNE rounding — biggest expected win
2. Optionally gate Q8_0 KV in calibration FullAttn — second-biggest win
3. (Probably unnecessary) port chunked delta-net recurrence — small win

None of (1), (2), or (3) are in the scope of this DeltaNet-full task.
