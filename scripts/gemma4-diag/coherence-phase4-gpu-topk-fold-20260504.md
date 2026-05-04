# Gemma 4 Phase 4 — GPU-side topk fold (eliminate D2H syncs)

- Branch: `gemma4`
- Date: 2026-05-04
- Hardware: 7900 XTX (gfx1100)

## What changed

New 8-thread kernel `fold_topk_with_per_expert_scale_k8`:

```c
fused_weights[ki] = topk_weights[ki] * per_expert_scale[topk_indices[ki]]
```

Replaces the prior CPU fold in `apply_moe_branch`, which required:
- 1× D2H download of `moe_topk_indices` (32 B)
- 1× D2H download of `moe_topk_weights` (32 B)
- CPU loop folding `topk_w[ki] * per_expert_scale_host[indices[ki]]`
- 1× H2D upload of the fused weights (32 B)

Per layer per token: 2 D2H + 1 H2D + a CPU loop. Across 30 MoE layers
that's **60 D2H syncs per decode step**. Each sync drains the GPU pipeline
(forces all in-flight kernels to complete so the CPU can read), so the
real cost was much larger than the bytes-transferred suggests — the GPU
sits idle waiting for the CPU to finish the loop before the next layer
can dispatch.

The fused-down branch now keeps every MoE op on device end-to-end. CPU
download is conditional: only when the legacy / half-fused path runs
(needs `moe.experts[indices[ki]]` lookup) or `HIPFIRE_DUMP_MOE=1` is set.

## Coherence

`coherence-gemma4.sh` — 6/6 strict pass, **bit-identical** to Phase 3
across all models. The GPU fold computes the same `f32 × f32` product
the CPU did, just in device registers.

## Bench (creative prompt, max_tokens=300)

| Phase | Decode | Prefill | TTFT |
|---|---|---|---|
| Phase 3 (CPU fold) | 77.6 | 86.2 | 569 ms |
| Phase 4 (GPU fold) | 86.8 | 98.1 | 499 ms |

Phase 3 → Phase 4: **+12% decode, +14% prefill, –12% TTFT**.

## Cumulative Phase 0 → Phase 4 on 26B-A4B

| Metric | Phase 0 | Phase 4 | Delta |
|---|---|---|---|
| Decode tok/s | 60.8 | 86.8 | **+43%** |
| Prefill tok/s | 65.2 | 98.1 | **+50%** |
| TTFT (ms) | 752 | 499 | **–34%** |

## Why D2H elimination beat its 3% estimate

The naive cost of the eliminated D2H is 60 × ~7 µs = ~420 µs/step
(~3% of a 15 ms decode). The actual 12% delta tells us **the syncs were
serializing kernel dispatch across MoE layer boundaries** — each
download forced the previous layer's gate_up + SwiGLU + down to drain
before the CPU could fold weights. With Phase 4, the GPU sits in a
back-to-back kernel pipe across all 30 layers, hiding per-launch
overhead under the previous launch's runtime.

Same lesson as the qwen35 GPU-only top-K refactor: removing CPU
intervention doesn't just save the bytes-transferred cost; it lets the
runtime overlap launches.

## Next levers

1. **Batched prefill** (Gemma 4 stub) — biggest remaining win.
2. **fused_qk-only kernel** — for full layers with `attention_k_eq_v=true`
   (31B / 26B-A4B).
3. **rmsnorm_batched merging** — 3 separate q_norm/k_norm/v_norm calls
   per attn block; merge to 1 batched-batched call.
4. **hipGraph capture** of the now-fully-on-device decode loop.
