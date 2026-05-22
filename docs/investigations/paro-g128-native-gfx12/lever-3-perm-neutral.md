# G128-native gfx12 PARO MoE — lever 3: vectorized perm_b32 nibble unpack (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` (post-commit).
> Hardware: hiptrx R9700 / gfx1201 / RDNA4 / ROCm 7.2.
> Continuation of the asymptote-search after m2 and m4 falsified.

## TL;DR

**Neutral on prefill (705 ↔ 705 tok/s on shisa A3B-PARO).** The lever
correctly fires — 16 v_perm_b32 emitted, lshrrev count drops 54 → 14,
VGPR count unchanged at 97 — but the kernel is **WMMA-throughput-bound**,
not VALU-bound. The scalar nibble unpack was already hidden behind WMMA
latency. Kept as opt-in research artifact; a future kernel that removes
the WMMA throughput bottleneck would then collect the perm win.

## Lever (theory)

Replace 8 scalar `(pk >> X) & 0xFu` ops + 8 byte stores per pk32 with
4 vectorized ops via `__builtin_amdgcn_perm` (v_perm_b32):

```c
unsigned int lo = pk & 0x0F0F0F0Fu;       // bytes {n0,n2,n4,n6}
unsigned int hi = (pk >> 4) & 0x0F0F0F0Fu; // bytes {n1,n3,n5,n7}
out[0] = perm(hi, lo, 0x05010400u);  // {n0,n1,n2,n3}
out[1] = perm(hi, lo, 0x07030602u);  // {n4,n5,n6,n7}
```

Baseline disasm shows the compiler did NOT auto-vectorize:
```
v_perm_b32:     0
v_lshrrev_b32:  54
```

Per WMMA k-substep: 16 scalar VALU ops → 4 vectorized. Per WG (8
WMMA × 16 groups): savings ≈ 1500 VALU ops.

## Measured (4 runs, shisa A3B-PARO, --prefill 256)

```
k2 baseline:    705.5 / 706.1 / 703.9 / 703.6   median ≈ 705 tok/s
perm variant:   704.3 / 705.1 / 704.6 / 704.6   median ≈ 705 tok/s
```

**Δ ≈ 0.0 % (neutral).** Run-to-run variance smaller than k2's own
spread.

## Disasm confirmation

```
                       VGPR   SGPR  v_perm_b32  v_lshrrev_b32  private
k2 baseline            97     18    0           54             0
perm variant           97     18    16          14             0
```

VGPR count is byte-identical — no occupancy regression and no occupancy
win. The 4× VALU reduction did happen, but the cycles were already
free (issued under WMMA-pipeline stalls), so wall-clock is unchanged.

## Implication (the bottleneck)

This is a useful negative result: it **rules out VALU-throughput** as
the gfx12 PARO MMQ prefill bottleneck. Combined with m2's finding
(rules out B-gather BW) and m4's finding (rules out X-load BW), the
remaining candidates are:

1. **WMMA throughput** — wmma_i32_16x16x16_iu8 issues at fixed cadence
   on gfx12 wave32 (~16-cycle latency per call). Theoretical ceiling is
   that throughput times the per-WG WMMA count; current k2 may be at
   that ceiling.
2. **Scale-FMA chain serialization** — the 8 (sc·d_x·cacc + zp·sum_x)
   FMA chain per sub-block is serial across `j` and creates a
   dependency chain that may stall WMMA issue.
3. **Per-group sc/zp shuffle overhead** — `__shfl(sc_self, src_lane)`
   × 8 outputs × 2 (sc, zp) per group is 16 cross-lane shuffles per
   group, each on the critical path before the FMAs.

The cheapest remaining lever to test would be **(3) — pre-shuffle
sc/zp ONCE at WG entry across all groups, paying VGPR for the static
table**. groups_per_row = K/128, so for A3B-PARO K=2048 → 16 groups →
16×8×2 = 256 floats of static sc/zp per row per lane = ~256 VGPRs.
That's likely a spill — won't work straightforwardly. A partial
pre-shuffle (e.g., 4 groups ahead) might fit.

Levers (1) and (2) require a structural redesign (e.g., 2 cacc
accumulators interleaved within a sub-block so FMA + WMMA overlap —
the k4 G256 pattern, but at K-tile granularity, not sub-block).

## Files shipped (opt-in research artifact)

```
kernels/src/gemm_paro_q4g128_moe_grouped_mmq_perm.gfx12.hip
crates/rdna-compute/src/kernels.rs                              ← SRC registration
crates/rdna-compute/src/dispatch.rs                             ← dispatch method
crates/hipfire-arch-qwen35/src/qwen35.rs                        ← env-gated routing
```

Env gate: `HIPFIRE_MOE_PARO_I8_PERM_GFX12=1` (default-off).

**Correctness caveat:** the perm encoding has NOT been coherence-gated.
Numerical sanity check is "4 bench runs at expected tok/s with no panic"
— circumstantial. Before flipping default-on, run
`./scripts/coherence-gate.sh` with the env var enabled on shisa A3B-PARO
+ z-lab A3B-PARO.
