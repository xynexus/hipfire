# MQ4 RDNA3.5 No-Unpack Prototype

## Summary

Build an experimental gfx1150/gfx1151 MQ4 path that keeps MQ4 weight nibbles packed and feeds them directly to RDNA3.5 packed-4-bit dot instructions. The tradeoff is intentional: activations become packed signed i4, so this is a speed prototype with a new activation-quantization error budget, not byte-equivalent MQ4.

Reference basis: AMD RDNA3.5 exposes packed 4-bit integer dot machinery such as `V_DOT8_I32_IU4` and `V_WMMA_I32_16X16X16_IU4`.

## Key Changes

- Add an experimental `MQ4 x I4` activation path gated behind a new env flag such as `HIPFIRE_MQ4_I4_DOT8=1`, arch-gated to `gfx1150/gfx1151`.
- Quantize rotated activations per MQ group:
  - For each 256-element group, compute `dx = max(abs(x_rot)) / 7`.
  - Pack `a_i = clamp(round(x_rot_i / dx), -8, 7)` as signed i4, eight values per `u32`.
  - Store `dx` and `sum_a = sum(a_i)` per group.
- Replace per-nibble weight extraction in the experimental kernel with direct packed dot:
  - MQ4 group weight is `w_i = sc * q_i + zp`, with packed unsigned `q_i`.
  - Activation approximation is `x_i ~= dx * a_i`, with packed signed `a_i`.
  - Per group: `dot ~= sc * dx * dot8_u4_i4(q, a) + zp * dx * sum_a`.
  - Use inline asm for `v_dot8_i32_iu4` first, since the repo does not currently use a dot8 builtin.
- Start with GEMV decode/residual kernels only. If accuracy and speed are useful, add a second-stage prefill kernel using `V_WMMA_I32_16X16X16_IU4`.

## Test Plan

- Unit kernel test: synthetic MQ4 blocks plus packed i4 activations against a CPU reference using the same quantized activation math.
- Runtime comparison on `qwen3.5-0.8b.mq4` and `qwen3.5-9b.mq4`: current MQ4 vs `HIPFIRE_MQ4_I4_DOT8=1`.
- Quality gates: KLD/perplexity slice first, then `./scripts/coherence-gate-dflash.sh`.
- Perf gates: report decode tok/s, kernel time, and activation-pack overhead separately; only keep the path if end-to-end speed wins after quantization cost.

## Prototype Knobs

- `HIPFIRE_MQ4_I4_DOT8=1`: broad opt-in for the experimental gfx1150/gfx1151 path.
- `HIPFIRE_MQ4_I4_DOT8_GEMV=1`: enable only non-residual MQ4 GEMV sites that route through `gemv_mq4g256_prerotated`.
- `HIPFIRE_MQ4_I4_DOT8_RESIDUAL=1`: enable residual MQ4 sites as a group.
- `HIPFIRE_MQ4_I4_DOT8_WO=1`: enable only attention-output residual (`wo`) sites.
- `HIPFIRE_MQ4_I4_DOT8_DOWN=1`: enable only FFN down-projection residual (`w_down`) sites.
- `HIPFIRE_MQ4_I4_QMAX=<float>`: activation i4 scale divisor, default 8 after first 0.8B PPL sweep.
- `HIPFIRE_MQ4_I4_CLIP_RMS=<float>`: optional per-group RMS clip multiplier before packing; unset or 0 disables clipping.

## First Quality Findings

- Raw `qmax=7` produced a clear PPL regression on `qwen3.5-0.8b.mq4`.
- `qmax=8` was the best first-pass setting on the 2048-token Wikitext slice.
- Projection isolation showed both `wo` and `w_down` can hurt individually at 2048 tokens, while combined residual routing with `qmax=8` measured better than baseline on the first slice; this needs a broader KLD/PPL matrix before treating it as a real quality win.
- RMS clipping at 3-4x RMS hurt the 512-token slice and is not a promising default.
- The prototype is coherent under the short DFlash/DDTree gate, but decode throughput is still slower because activation packing is a separate launch.

## Assumptions

- "No unpack" means no explicit software expansion of MQ4 weight nibbles into byte or FP lanes; the dot instruction's internal interpretation of packed u4 is acceptable.
- Activation i4 quantization error is acceptable for a prototype, but the path must remain opt-in until coherence and quality gates pass.
- Existing MQ4 model files stay unchanged; only runtime activation packing and experimental kernels are added.
