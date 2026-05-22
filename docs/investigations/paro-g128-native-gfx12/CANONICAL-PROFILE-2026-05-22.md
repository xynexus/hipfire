# z-lab A3B-PARO canonical-state rocprof snapshot (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability @ 99d2e271`
> Bench: shisa-z-lab `Qwen3.6-35B-A3B-PARO`, --prefill 256, prefill-runs 2
> Env: production-canonical (no broad-WMMA-on-router opt-ins)

```
HIPFIRE_F32_SHARED_EXPERT_WMMA_GFX12=1
HIPFIRE_HFQ4G128_WMMA_GFX12=1
HIPFIRE_MOE_PARO_FP16_GFX12=1
HIPFIRE_PARO_BATCHED=1
HIPFIRE_GRAPH=0
HIPFIRE_KV_MODE=q8
```

## Headline

**Prefill: 2997.7 tok/s (wall) / 85.40 ms / 163.4 ms rocprof GPU time**

(2-run bench measurement; the 4-run-warmup median is 3130 tok/s — the
extra warmup eliminates a residual cold-start cost. Both reflect the
same production-canonical state.)

## Top 20 kernels by GPU time

| % | Kernel | Calls | µs/call | Total ms | Optimization status |
|---:|---|---:|---:|---:|---|
| 19.0 | `gemm_paro_q4g128_moe_grouped_wmma_gfx12` | 160 | 193.9 | 31.0 | gfx12 FP16 WMMA (shipped a457fa34) |
| 18.0 | `gemm_hfq4g128_wmma_gfx12` | 260 | 113.3 | 29.5 | gfx12 FP16 WMMA (shipped 6c097ad5, the 70% hidden lever) |
| 17.6 | `gemm_f32_batched` | 280 | 102.9 | 28.8 | scalar F32; mostly router (precision-locked) + alphas (small M) |
| 12.1 | `gated_delta_net_q8` | 60 | 329.3 | 19.8 | LDS-tiled baseline; register variant falsified (afc88620) |
| 5.2 | `gemm_f32_wmma_gfx12` | 240 | 35.7 | 8.6 | gfx12 FP16 WMMA for shared_expert (shipped 5a556225) |
| 4.5 | `gemv_f32` | 2 | 3643.9 | 7.3 | lm_head GEMV, already at 523 GiB/s |
| 3.5 | `givens_rotate_to_f32` | 340 | 16.9 | 5.7 | PARO Givens pre-pass |
| 2.9 | `attention_q8_0_kv_batched` | 20 | 240.5 | 4.8 | FA attention with Q8 KV |
| 2.7 | `__amd_rocclr_copyBuffer` | 1762 | 2.5 | 4.4 | runtime overhead, can't optimize |
| 2.2 | `conv1d_silu_split_f32` | 60 | 60.2 | 3.6 | DeltaNet conv1d preamble |
| 1.6 | `rmsnorm_f32` | 202 | 13.2 | 2.7 | per-layer norm |
| 1.4 | `convert_f32_to_f16` | 320 | 6.9 | 2.2 | ensure_fp16_x amortized cache |
| 1.2 | `moe_down_combine_grouped_k8` | 80 | 25.1 | 2.0 | routed-MoE combine |
| 0.9 | `moe_gate_up_unscatter_k8` | 80 | 18.9 | 1.5 | routed-MoE unscatter |
| 0.8 | `moe_scatter_fused_k8` | 80 | 16.9 | 1.3 | routed-MoE scatter |
| 0.7 | `fused_silu_mul_mq_rotate` | 160 | 7.3 | 1.2 | (legacy MQ path, small) |
| 0.6 | `mq_rotate_x` | 80 | 13.0 | 1.0 | (legacy MQ path) |
| 0.6 | `fused_qk_l2_norm_scale_f32` | 60 | 17.0 | 1.0 | DeltaNet Q/K normalization |
| 0.6 | `softmax_f32` | 80 | 12.1 | 1.0 | MoE router softmax |
| 0.6 | `moe_topk_renorm_k8_batched` | 80 | 12.1 | 1.0 | MoE top-K renorm |

Top 5 = 71.9% of GPU time. Top 10 = 88.8%.

## Where time is now

| Bucket | % | Notes |
|---|---:|---|
| FP16 WMMA matmul (the WMMA family I optimized) | **42.2** | gemm_paro_q4g128_moe_grouped_wmma_gfx12 + gemm_hfq4g128_wmma_gfx12 + gemm_f32_wmma_gfx12 |
| Scalar F32 GEMM (precision-locked) | **17.6** | gemm_f32_batched — router, alphas, w_beta |
| DeltaNet | **14.3** | gated_delta_net_q8 + conv1d_silu_split + fused_qk_l2_norm_scale |
| lm_head | **4.5** | gemv_f32, already at 523 GiB/s |
| Full-attention KV + qkv | **2.9** | attention_q8_0_kv_batched |
| PARO Givens / fused-silu paths | **3.5** | givens_rotate_to_f32 |
| MoE scatter/combine/topk | **3.2** | moe_scatter_fused + moe_gate_up_unscatter + moe_down_combine + moe_topk_renorm + softmax |
| RMSNorm + F16 convert + memcpy | **~6.7** | rmsnorm_f32, convert_f32_to_f16, copyBuffer |

## Where the optimization ceiling lives now

If hipGraph capture for prefill collapses the ~3000+ dispatches per token
into one graph replay (commit-overhead pattern, not data-dependency
rewrite), each call's launch overhead (~5-10 µs) compounds across
hundreds of calls. Conservative estimate: 1-2 ms saved per prefill on
top of 85.4 ms = +1-2 % wall.

If DeltaNet attention is redesigned with S-split-across-WGs (recovering
the 8× parallelism the register-array variant lost), the 14.3 % DeltaNet
slice could drop 30-50 %. That's +4-7 % wall.

PARO4G256 pivot (separate plan doc) unlocks m2 on G256 layout. On the
42.2 % WMMA-matmul slice that could be 10-15 % within-slot — so +4-7 %
wall total.

None of these are bounded by the session's remaining budget. All are
multi-day projects requiring kernel-by-kernel coherence validation.

## Reproduction

```bash
SNAP=$(ls -d ~/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-PARO/snapshots/*/ | head -1)
rm -rf /tmp/zlab-canonical && mkdir -p /tmp/zlab-canonical
scripts/rocprof-wrap.sh /tmp/zlab-canonical -- \
    env HIPFIRE_F32_SHARED_EXPERT_WMMA_GFX12=1 HIPFIRE_HFQ4G128_WMMA_GFX12=1 \
        HIPFIRE_MOE_PARO_FP16_GFX12=1 HIPFIRE_PARO_BATCHED=1 \
        HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8 \
        HIPFIRE_PROFILE_NO_ROCPROF=1 \
    ./target/release/examples/bench_qwen35_mq4 "$SNAP" \
        --prefill 256 --prefill-runs 2 --warmup 0 --gen 0

scripts/rocprof_top.py /tmp/zlab-canonical/trace_kernel_stats.csv 20
```

Run on hiptrx (R9700 / gfx1201 / RDNA4 / ROCm 7.2).
