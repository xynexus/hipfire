# OqPlusCompact group size: is G ∈ {64, 128, 256, 512, 1024} viable?

**Date:** 2026-08-06
**Verdict:** G=256 is the sweet spot. Every other group size costs MORE bits per
weight to reach the same quality. G=1024 additionally fails as a universal
format. **No change recommended.**

Reproduce: `python3 tools/oq_compact_group_sweep.py` (experiment tooling — not
production; see AGENTS.md).

## What was measured

A faithful port of `quantize_oqplus_compact` + `mixed_clipsearch`
(`crates/hipfire-quantize/src/codecs.rs`), generalised from the hardcoded G=256
to arbitrary G: `symmetric_clipsearch(±7)` → overlay indices ranked by
`err4² − err8²` → `refit_mixed_scale` → indices → refit, with the bulk clamped
to ±7, the overlay to ±127, and ONE shared f16 scale.

Real bf16 weights from `Qwen--Qwen3.5-0.8B` (q_proj, k_proj, gate_proj,
down_proj), randomised Hadamard as in `cpu_fwht_256`.

Block cost: `2 (f16 scale) + G/2 (nibbles) + N_out × entry`, where `entry` is
2 bytes for G ≤ 256 (u8 index + i8 value) and **3 bytes for G > 256**, because a
u8 index cannot address a position ≥ 256. That widening is part of the
economics, not an implementation detail that can be optimised away.

## Result — cost to match G=256 quality

Apples-to-apples on the three K=1024 tensors (0.8M weights), so every G sees
identical data. Reference: G=256, N_out=3 → **20.71 dB @ 4.250 bits/weight**.

| G | N_out to match | bits/weight | SNR dB | vs G=256 |
|---|---|---|---|---|
| 64 | 1 | 4.500 | 20.78 | **+0.250 bits** |
| 128 | 2 | 4.375 | 20.91 | **+0.125 bits** |
| **256** | **3** | **4.250** | **20.74** | **baseline** |
| 512 | 8 | 4.406 | 21.03 | **+0.156 bits** |
| 1024 | 12 | 4.297 | 20.72 | **+0.047 bits** |

G=256 reaches its quality level more cheaply than any alternative. The curve is
shallow and slightly non-monotonic (G=512 lands worse than both 256 and 1024),
which is consistent with a real optimum near 256 plus sampling noise.

The tension it reflects: smaller G gives finer scale granularity but amortises
the 2-byte scale over fewer weights; larger G amortises better but coarsens the
scale AND pays a 3-byte overlay entry past 256.

## Result — hard constraints

**K % G == 0 is required.** On this model alone, `down_proj` has K=3584, which
is divisible by 64/128/256/512 but **not 1024**. FFN widths that are not
multiples of 1024 are common, so G=1024 cannot be the universal group size — it
would need a per-tensor fallback, which is exactly the complexity the single
group size exists to avoid.

**u8 overlay index caps G at 256.** `mixed_overlay_indices` stores the position
as `u8`, and `quantize_oqplus_compact` already clamps `n_out` to `1..=255`.
G > 256 requires a u16 index — a format change, and the 3-byte entry above.

**G is not stored anywhere.** `N_out` is *inferred* from the block stride
(`n_out = (block_bytes − 130) / 2`, where 130 = 2 + 256/2) in both
`oqplus_compact_to_oq8_combined` and `gemm_oq_compact_grouped_wmma`. A second
group size therefore needs its own quant-type code and dtype, not just a
parameter — G lives only in the name `OqCompactG256`.

**The FWHT is coupled to the group.** Weights are rotated offline by
`cpu_fwht_256` and activations at runtime via `RotationPlan::FwhtG256`. Any
other G needs a matching FWHT-G on both sides, or the rotate-offline /
rotate-x-at-runtime identity breaks.

## Caveats

- **SNR is a screening metric, not the gate.** Weight-reconstruction SNR
  predicts nothing about end-task quality on its own (cf. the DFlash finding
  that SNR was the wrong gate and acceptance rate was right). It is used here
  only to answer "is any G worth pursuing", and the answer is no — so no KLD or
  eval confirmation was run. If a G had looked promising, that confirmation
  would be required before believing it.
- One model's shapes (K ∈ {1024, 3584}), 0.8M weights in the fair comparison,
  one seed for the Hadamard signs.
- The fair comparison excludes `down_proj` (K=3584), since G=1024 cannot divide
  it; the full-tensor run in the script includes it for the G values that can.

## Conclusion

The existing G=256 is well chosen. Going smaller costs bits for no quality gain;
going larger costs bits, a u16 index, a new quant code, an FWHT-G, and — at
G=1024 — universality. Not worth doing.
