# Gemma 4 26B-A4B fused indexed-GEMV plan

## Status

Locate + spec + integration plan committed; phased rollout.

## Problem

`apply_moe_branch` in `crates/engine/src/gemma4.rs` runs an 8-expert
serialized loop, dispatching 5 kernels per expert per layer:

```
for ki in 0..k_top {
    let e = topk_indices[ki];
    weight_gemv(gate_up_proj_e, moe_pre2, expert_gate_up_buf);  // 1
    let (gate, up) = split_2(expert_gate_up_buf);
    gpu.gelu_tanh_f32(gate, hidden, mi);                         // 2
    gpu.mul_f32(hidden, up, hidden);                             // 3
    weight_gemv(down_proj_e, hidden, expert_out);                // 4
    gpu.scaled_add_inplace_cpu_scalar_f32(cur_moe, expert_out, w); // 5
}
```

For 26B-A4B that's **40 launches per layer × 30 layers = 1,200
launches per token** just for the expert FFN, plus ~10 others
(attention, norms, sample). Decode is launch-bound at 5-7 tok/s on
gfx1100; the dense Qwen3.5-MoE A3B path which uses indexed kernels
hits 80-100 tok/s on the same hardware.

## Existing infrastructure

The Qwen3.5-MoE A3B path has the kernels we need. Located via:

```
kernels/src/gemv_hfq4g256_moe_gate_up_indexed.hip
kernels/src/gemv_hfq4g256_moe_down_indexed.hip
kernels/src/gemv_hfq4g256_moe_gate_up_indexed_wave64.hip   (CDNA3)
kernels/src/gemv_hfq4g256_moe_down_indexed_wave64.hip
kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched.hip  (prefill)
kernels/src/gemv_hfq4g256_moe_down_indexed_batched.hip
```

Dispatch wrappers at:
- `gemv_hfq4g256_moe_gate_up_k8_indexed`
- `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`

API (gate_up):
```
expert_ptrs:  &[u64; n_exp]   device pointers, one per expert weight
topk_indices: &[i32; 8]       which experts this token activates
x:            &[f32; k]       FWHT-rotated hidden state (shared input)
y_gate:       &[f32; 8 * m]   gate output, contiguous per-expert
y_up:         &[f32; 8 * m]
m, k:         expert weight shape
```

API (down with residual + scale):
```
expert_ptrs:  &[u64; n_exp]
topk_indices: &[i32; 8]
topk_weights: &[f32; 8]       per-expert scale (already includes per_expert_scale)
rot_batch:    &[f32; 8 * k]   rotated activated hidden, per-expert
x_residual:   &[f32; m]       atomicAdd target (the cur_moe accumulator)
m, k:         shape
```

Single launch grid `[m / row_factor, 8, 1]` covers all 8 experts.

## Format constraints

| Tensor              | Required dtype       | Gemma 4 26B-A4B actual | Compatible? |
|---------------------|----------------------|------------------------|-------------|
| gate_up_proj        | HFQ4G256 / MQ4G256 / MG4G256 (byte-compatible) | MG4G256 (k=2816 divides 256) | YES |
| down_proj           | HFQ4G256 / MQ4G256 / MG4G256 | Q8F16 (k=704 forces Q8 fallback) | NO |

The Q8 down fallback comes from the explicit fallback chain we
landed in commit 488f8f1: when `k % 256 != 0` the only safe quant
for our existing kernels is Q8F16. There is no indexed-Q8 down GEMV
in the codebase today.

Three options for the down side:
- **A. Hybrid** — indexed gate_up + per-expert Q8 down. 8 → 1 launch
  on gate_up (saves 7 launches/layer), down loop unchanged. ~210
  launches saved per token total. Lowest-risk integration.
- **B. New kernel** — author `gemv_q8_0_moe_down_indexed_k8`. Saves
  another 7 launches/layer. Requires kernel + CPU reference + bench.
- **C. Re-quant** — pad down_proj from k=704 to k=768 (zero-pad,
  mathematically lossless) so MG4 indexed-down works. Requires
  quantizer change + re-quant of 26B-A4B. Wastes ~9% in down weights.

This session: ship A. B and C noted as follow-ups.

## Phase 1 plan (this session)

1. **Locate** — done. `apply_moe_branch` line 1525-1571.
2. **Build per-layer expert_ptrs tables** — at `load_moe_layer_extras`,
   populate a `[2 * n_exp]` F32 tensor per layer containing the u64
   device addresses of each expert's `gate_up_proj.buf`. Stash on
   `MoeLayerExtras`.
3. **Build a per-call rotation buffer for the moe_pre2 input** — the
   indexed GEMV expects FWHT-rotated input. Existing scratch already
   has `moe_pre2`; we can rotate it in-place via `gpu.rotate_x_mq`
   (mirrors qwen35's path).
4. **Replace the serialized gate_up_proj weight_gemv** — single call
   to `gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(...)` writes
   `y_gate[8, mi]` and `y_up[8, mi]` in one launch.
5. **Per-expert SwiGLU + down stays as-is** — each iteration reads
   from `y_gate[ki * mi..(ki+1) * mi]` and `y_up[ki * mi..(ki+1) * mi]`.
6. **Gate behind `HIPFIRE_GEMMA4_MOE_FUSED=1`** — default off. Shadow-
   compare both paths on a fixed prompt; outputs must match within
   bf16 ULP.
7. **Bench** — kv_seq=256 single-token decode, 50-token sample, σ
   over 3 trials. Compare against the 5-7 tok/s baseline. Quality:
   coherence-gemma4 must remain 6/6 strict pass.
8. **Promote** — flip default if quality clean and perf improved.

## Phase 2 follow-up (out of session)

Author `gemv_q8_0_moe_down_indexed_k8` to fuse the down side and the
final scaled-residual add. Saves another 8 launches/layer × 30 = 240
launches per token. Combined with Phase 1, expected 5-7 tok/s →
30-40 tok/s on gfx1100 (still launch-limited but ~6× fewer launches).

## CPU reference (Phase 1)

For shadow-mode verification, the CPU path is:

```python
def apply_moe_phase1_ref(hidden, top_idx, top_w, expert_gate_up, expert_down,
                         per_expert_scale, post_norm_2_w, post_norm_w,
                         pre_norm_2_w, dense_mlp_out):
    # Hidden state already pre_feedforward_layernorm_2'd (moe_pre2).
    pre2 = rmsnorm(hidden, pre_norm_2_w)
    cur_moe = zeros_like(hidden)
    for ki, e in enumerate(top_idx):
        w = top_w[ki] * per_expert_scale[e]
        gate_up = expert_gate_up[e] @ pre2     # [2 * mi]
        gate, up = chunk(gate_up, 2)
        h = gelu_tanh(gate) * up               # [mi]
        out = expert_down[e] @ h               # [dim]
        cur_moe += w * out
    cur_moe = rmsnorm(cur_moe, post_norm_2_w)
    cur_mlp = rmsnorm(dense_mlp_out, post_norm_1_w)
    combined = cur_mlp + cur_moe
    return rmsnorm(combined, post_norm_w)
```

The Phase 1 fused gate_up changes the GEMV call shape but is
mathematically equivalent (same `gate_up @ pre2` per expert, just
batched into one launch). No reference rewrite needed for Phase 1
verification — same outputs, fewer launches.

## Risks / escalation

- **Shadow-mode divergence**: if the indexed kernel produces values
  that don't match the per-expert path within bf16 ULP, halt
  promotion and capture a per-expert diff log.
- **expert_ptrs alignment**: kernel reads u64 from an F32-typed
  tensor. The qwen35 path uses `2 * n_exp` F32 slots per ptr table.
  Mirror this.
- **MG4 vs HFQ4G256 byte format**: the kernel reads HFQ4G256 layout
  (4-bit packed groups of 256 with f16 scale + zero). MG4G256 was
  designed to be byte-compatible with the same kernel; verify by
  smoke-testing the indexed kernel reads MG4 bytes correctly first
  (smoke a single MoE layer with HIPFIRE_GEMMA4_MOE_FUSED=1 and
  diff the gate_up output against the per-expert loop).
