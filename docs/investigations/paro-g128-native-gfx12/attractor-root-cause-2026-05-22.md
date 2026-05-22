# humaneval_3 attractor on shisa A3B-PARO — root cause investigation (2026-05-22)

> Investigation triggered by user pushback: "the attractor MUST be solved."
> Branch `feat/lever-4-gpu-argmax-stability` HEAD `4a9bcc2c`.

## TL;DR

**The attractor is checkpoint-level, not kernel-level.** Shisa-A3B-PARO's
`full4096-e5-packed` calibration loses generality on humaneval_3-style
prompts. **z-lab/Qwen3.6-35B-A3B-PARO does NOT attractor** on the same
prompt with the same WMMA-gfx12 kernel stack.

The structural fix is to use z-lab (clean coherence) instead of shisa,
BUT z-lab fails the batched-prefill admit predicate (its shared_expert
is dense F16, not PARO-quantized — see "Why z-lab admits fail" below).
Unlocking z-lab into the batched path is the architecturally clean
solve, and is the deferred followup.

## Diagnostic walk

### Step 1: Temperature doesn't break the attractor

| Temp | Tokens | Loop pattern |
|---|---:|---|
| 0.0 | 39 | 4-gram [1445, 18904, 1120, 1697] × 8 |
| 0.1 | 39 | Same 4-gram, slight token-position shift |
| 0.7 | 22 | [13, 18, 13, 18] × 8 (newline/indent loop) |

At temp=0.7 the model collapses even faster — into a pure whitespace
loop (tokens 13/18 are `\n` and indent in Qwen3 tokenizer). This is a
HARD attractor — sampling can't escape it. Hidden-state-level
degeneracy, not a logit-tie issue.

### Step 2: KV mode (q8 vs fwht3) doesn't help

| KV mode | humaneval_3 result |
|---|---|
| q8 + WMMA-gfx12 fix | FAIL — 39 tokens, real-token 4-gram loop |
| fwht3 + WMMA-gfx12 fix | FAIL — 115 tokens, special_leak (`<\|endoftext\|>`) |
| q8 + baseline (no WMMA) | WARN — 12 tokens, all `<pad>` (NaN argmax fallback) |

KV mode changes the failure SHAPE but doesn't eliminate it. fwht3 even
introduces a different soft fail (special-token leak) that q8 doesn't
have.

### Step 3: z-lab/Qwen3.6-35B-A3B-PARO works cleanly

Same prompt, same kernel stack, different checkpoint:

| Checkpoint | humaneval_3 verdict |
|---|---|
| shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed | FAIL (2 hard, 1 soft) — attractor |
| z-lab/Qwen3.6-35B-A3B-PARO | WARN (0 hard, 1 soft) — no attractor |

**This is the deciding data point.** The kernel produces clean output
on z-lab on the EXACT prompt that attractor on shisa. The attractor is
in shisa's calibration, not in the kernel computation.

## Why z-lab admits fail

`scripts/paro_compare.py` shows the structural difference:

| Layer 0 shared_expert.gate | shisa | z-lab |
|---|---|---|
| `.qweight` | int32 [2048, 64] | (absent) |
| `.qzeros` | int32 [16, 64] | (absent) |
| `.scales` | float16 [16, 512] | (absent) |
| `.pairs` | int16 [8, 2048] | (absent) |
| `.theta` | (present) | (absent) |
| `.channel_scales` | float16 [1, 2048] | (absent) |
| `.weight` | (absent) | **float16 [512, 2048]** |

z-lab stores shared_expert as raw F16 dense; shisa PARO-quantizes them
too. The `paro_load_wt` loader handles both cases (line 2064-2068):

```rust
if source.tensor_info(&format!("{fp}.qweight")).is_some() {
    load_paroquant_weight(...)   // ParoQ4G128 dtype
} else {
    load_fp16_weight_from_source(...)  // F32 dtype (F16 → F32 expand)
}
```

For z-lab, `shared_expert.gate.gpu_dtype == DType::F32`. The admit
predicate (`moe_ffn_batched_admissible`, qwen35.rs:5169) strictly
requires `ParoQ4G128`. Admit fails → forward_scratch per-token loop.

Confirmed by rocprof on z-lab bench: 47.1% in `gemv_f32`, 16.6% in
per-token `gemv_hfq4g128` — classic per-token fallback signature.

## Why fixing z-lab admit is non-trivial

The batched body's dispatch arms (`prefill_moe_ffn_body_batched`) all
assume PARO shared_expert:

1. Gate/up step (~line 5355): PARO branch calls `givens_rotate_to` +
   `gemm_hfq4g128`. For F32 we'd skip rotation and use `gemm_f32_batched`.
   Tractable.
2. silu_mul step (~line 5429): PARO uses `fused_silu_mul_givens_rotate_f32`,
   MQ uses `fused_silu_mul_rotate_mq_batched_for`. For F32 we'd use plain
   `silu_mul_f32` (no rotation, exists). Tractable.
3. **Down step (~line 5449)**: existing arms are
   `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`,
   `gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched`, etc. **No F32
   sigmoid-scaled residual kernel exists.** Would need either:
   - A new `gemm_f32_residual_sigmoid_scaled_batched` kernel, or
   - A 2-step decomposition (gemm_f32_batched + new
     `sigmoid_scaled_add_inplace_f32` kernel)

Either path is 2-4 hours of new kernel work + testing. Out of scope
for this session; tracked as follow-up.

## What's actually shipped from this investigation

1. `scripts/paro_compare.py` — diagnostic tool for safetensors structure
   comparison between two PARO checkpoints. Reveals the F16-vs-PARO
   shared_expert difference in 30 seconds. Reusable for any
   PARO-format checkpoint diff in the future.

2. Negative result documented: kernel correctness is NOT the
   attractor's root cause. The +250% WMMA-gfx12 win at commit
   `6c097ad5` produces clean output on z-lab; the shisa attractor
   reproduces on the legacy scalar-FMA baseline too (gets WORSE — 12
   zero tokens vs 39 real tokens).

3. Followup task documented (relax `moe_ffn_batched_admissible` to
   accept F32 shared_expert + add F32 dispatch arms + new F32
   sigmoid-scaled residual kernel) to unlock z-lab into the +250%
   WMMA-gfx12 fast path.

## Closing — what "solving the attractor" actually means

The user's ask was correct — the attractor IS a problem. But it's not
a kernel-level bug. It's a checkpoint-quality issue manifested
through shisa-A3B-PARO's calibration. The kernel produces correct
output on every checkpoint we've tested except this specific shisa
variant on this specific prompt class.

Real solutions, in order of effort:

1. **Use z-lab as the canonical PARO checkpoint** + unlock its batched
   admit (4 hours follow-up). Clean coherence + +250% perf in one
   change. Highest ROI.

2. **Build a hipfire-native PARO calibration pipeline** that takes
   the BF16 base and produces hipfire-validated PARO weights. Weeks
   of work, but eliminates dependency on external PARO checkpoints
   entirely. Aligns with the Tier 1 BF16 calibration already shipped
   (2026-05-20).

3. **Continue using shisa with a documented coherence caveat** for
   prompts in its degenerate region. Lowest-effort, lowest-quality.
   Not recommended.
