# Tuning levers catalog

The actual optimization patterns hipfire uses, with pointers to the
commits where they landed (or were tried-and-reverted). Pick the one
that matches the bottleneck you root-caused in `playbook.md` step 2.

Rule of thumb: pick ONE lever per commit. Bundling makes bisect
useless when the win turns out to be one of three changes and the
other two are wash-or-regression.

## 1. Wave-size port (CDNA3 ⇄ RDNA)

**When**: target arch is wave64 (CDNA1/2/3) and you're running a
wave32 kernel — half the lanes are silently masked out, kernel
correctness still works but throughput is halved or worse.

**Reference commit**: `4105035` — "perf(cdna3): full wave64 port of
all hot HFQ4 kernels — MI300X decode 48.6 → 96 tok/s". Ten HFQ4
kernels ported to 2-rows-per-block wave64 layout. A3B decode jumped
from 48.6 to 96 tok/s, matching 7900 XTX on the same model.

**Pattern**: separate `<name>.wave64.hip` file (or `.gfx942.hip`
chip-specific) using wave-64 lane decomposition (32 M-rows × 2 K
groups). Dispatch via the chip detection in `Gpu::init`.

**Don't try**: porting wave64 → wave32 for RDNA. RDNA is wave32
native; the wave64 mode (when supported) costs occupancy and rarely
helps inference.

## 2. Multi-row GEMV

**When**: decode is launch-overhead-bound (kernel-launch count
dominates per-token cost), not BW-bound. Profile shows low GB/s on
GEMV kernels but high invocation count.

**Pattern**: process R output rows per warp instead of 1, sharing
the `x` register state across rows. R=2 / R=4 / R=8 variants exist
in the tree:

```
kernels/src/gemv_hfq4g256_multirow.hip
kernels/src/gemv_hfq4g256_multirow.gfx1100.hip      # chip-tuned
kernels/src/gemv_hfq4g256_residual_multirow.gfx1100.hip
```

**Trade-off**: more VGPR pressure. Past R=8 you spill on RDNA3 and
the win disappears. Tuning is per-arch — gfx1100 likes R=4–8 for
hot decode kernels; gfx1010 prefers R=2 because of the smaller VGPR
budget.

**Reference**: existing `*_multirow*` kernels. Don't reinvent —
fork the closest existing variant and adjust R.

## 3. K-tile depth (K2 / K4 / K-split)

**When**: WMMA-bound prefill kernel where the inner K-tile loop
dominates wall-clock. Deeper unrolls amortize per-tile overhead but
push register pressure.

**Pattern**: K2 (process 2 K-tiles per loop body, soft-pipelined
via early load + lagged WMMA) is the canonical baseline. K4 has
more software pipelining headroom. K-split is for when K isn't a
multiple of the tile depth.

**Reference (positive)**: `gemm_hfq4g256_residual_wmma_k2.hip` is
the deployed baseline across all dominant prefill GEMMs
(`gemm_gate_up_hfq4g256_wmma`, `gemm_qkv_hfq4g256_wmma`,
`gemm_qkvza_hfq4g256_wmma`, residual). The K2 step is fully shipped.

**K4 step status**: `gemm_hfq4g256_residual_wmma_k4.hip` had a
swapped-output-mapping investigation (case-studies §8). After fix, K4
ties K2 byte-for-byte at m=4096 on 9B residual but loses to ksplit by
~33% per-call at small batch (CU-starved grid: 3.3 vs 13 blocks/CU under
K4's `__launch_bounds__(32, 1)`). Verify the current dispatch policy
before using K4, and do not cite an exact historical hash unless it exists
in the checkout you are reporting from.

**Reference (NEGATIVE — null result, important to know)**: commit
`f670e16` — "experiment(gemm): k2x32 wider-row lm_head — null
result". 32-row block × K2 unroll measured 46% **slower** at
M=248320. Hypothesis: doubled accumulator + 4× dequant live ranges
push past comfortable register budget, forcing spills. Lesson:
register pressure is the gating constraint past a certain point;
more parallel work doesn't help when you can't pipeline it.

## 4. WMMA / MFMA matrix engine

**When**: prefill GEMM (batch_size > 1) on an arch with a matrix
engine. Hipfire dispatches WMMA on gfx11/gfx12 and MFMA on gfx94x;
non-WMMA archs (gfx1010, gfx1030) fall through to the dot2 / scalar
path.

**Pattern**: 16×16×16 fp16→fp32 tile is the workhorse. The actual
builtin name + operand layout differs per arch family — see
`.agents/skills/hipfire-arch-port/wmma-matrix.md` if you're porting to a
new arch.

**Reference (positive)**: PR #56 — gfx12 WMMA port, channel-tested
on R9700, all 6 hot fused-projection kernels covered.

**Reference (silent-corruption cautionary tale)**: commit `b7ac66a`
— gfx11 C-mapping (`acc[j] = C[2*j + (tid>>4)][tid & 15]`) was
silently wrong for ~6 weeks. WMMA passed every functional test that
didn't compare element-by-element against a CPU reference. Lesson:
**channel-test is non-negotiable** for any WMMA / MFMA work.

## 5. Software prefetch (`s_prefetch_data`)

**When**: hot decode kernel that re-reads weights every token; the
profiler shows L2 misses dominating decode latency. Available on
gfx12 (RDNA4) only.

**Reference**: `kernels/src/gemv_hfq4g256.gfx1201.hip` — uses
`s_prefetch_data` for 2-group lookahead (~272 bytes). Comment in
the kernel header explains: "RDNA4 allows 96 VGPRs at max 16-wave
occupancy" — bigger budget there enables more aggressive prefetch.

**Pattern**: chip-specific override. `<name>.gfx1201.hip` covers
9070 XT only; if you want 9070 XT + R9700 (also gfx1201), the family
tag `<name>.gfx12.hip` covers both. See `cross-arch.md`.

**Don't try**: porting `s_prefetch_data` to gfx11. The RDNA3
intrinsic doesn't expose the same prefetch primitive.

## 6. Fused projections (multi-output kernels)

**When**: multiple GEMM/GEMV calls share an input vector (Q, K, V
all read the same x; gate + up share the same x). Each separate
launch eats kernel-launch overhead; fusing into one kernel lets you
load x once and write multiple Y outputs.

**Pattern**: kernels named `gemm_qkv_*` (3-way fused QKV),
`gemm_qkvza_*` (4-way for DeltaNet), `gemm_gate_up_*` (2-way for
SwiGLU FFN). Each takes multiple A weight buffers + multiple Y
output buffers in one launch.

**Reference (positive)**: `9d05c9f` — "perf(fused-projection):
consolidate gfx1100 + baseline kernels into one cross-arch family".
The fused QKV path is one of the main reasons hipfire's decode
beats llama.cpp's by 1.7–2.1× on small models — llama.cpp does 3
separate GEMV launches per layer; hipfire does 1.

**Caveat**: fused kernels are bigger (more VGPRs). On VRAM-tight
arches (gfx1010, gfx1013, gfx1032) they sometimes lose to the
unfused path because of occupancy. Fall back via the dispatch tree
when this matters.

## 7. Per-kernel hipcc flags (magic comments)

**When**: a single kernel benefits from non-default compiler flags
(`-mcumode`, `-fno-unroll-loops`, custom `-ffast-math`-style, etc.)
that you don't want to apply globally.

**Reference**: `5f65005` — "feat(compile): per-kernel hipcc flag
magic-comment plumbing + bisect helpers". The kernel JIT picks up
`// HIPCC_FLAGS: -foo -bar` magic comments at the top of the .hip
file and adds them to that kernel's compile invocation only.

**Pattern**: add the magic comment, rebuild kernel hashes
(`./scripts/write-kernel-hashes.sh`), validate via the three gates.
Per-kernel flags don't propagate to other kernels in the tree, so
this is the safe way to experiment without globally affecting
compilation.

## 8. rocBLAS / cuBLAS-class GEMM fallback

**When**: prefill GEMM on a CDNA3 arch (gfx94x). The MFMA path
through rocBLAS can outpace hipfire's hand-rolled MFMA at very
large M.

**Reference**: `07a2b1c` and friends — "feat(rocblas): wire CDNA3
MFMA path into gemm_hfq4g256 + gate_up". `HIPFIRE_ROCBLAS_OFF=1`
kill-switch (`1316f8e`) for A/B benching;
`HIPFIRE_ROCBLAS_ALL_ARCHS=1` opens the path to RDNA3 (`05d104d`).

**Caveat**: rocBLAS dispatch is per-arch. Don't enable on RDNA
unless you've measured a win on YOUR arch — the default off for
RDNA is an empirical decision.

## 9. Quantized-operand memory layout (gfx1151, W4A8)

**When**: a WMMA kernel whose operands are packed sub-byte and must be
unpacked before they can be issued. On gfx1151 this is the dominant
VALU cost in a compact-Opus GEMM — an instruction census of the
simplest correct W4A8 kernel put nibble unpacking at **20.4 of 71
instructions per WMMA, 29% of the total**.

**The ISA reason it hurts more than it looks**: of the shifts, only
`V_DUAL_LSHLREV_B32` is dual-issuable. **Right shifts — which is what
nibble unpacking needs — cannot co-issue on gfx1151**
(`kb/dtypes-and-conversion.md`, VOPD tables `r35:9416-9445`). So the
unpack is both numerous and serialising.

**The lever**: make the DRAM layout match the operand layout, so there
is no unpack. But the obvious version of this LOSES — see below.

Measured on the ladder kernel, gate/up 17408x5120 B=256, parity PASS
on every arm:

| activation layout | ms | TOPS | |
|---|---|---|---|
| int8, split in-kernel | 2.484 | 36.7 | baseline |
| two separate nibble planes (hi at `[0,B*K/2)`, lo after) | 3.585 | 25.5 | **-30%** |
| **hi/lo interleaved at FRAGMENT granularity** | **2.191** | **41.7** | **+13%** |

Separate planes removes every unpack instruction and still loses badly:
it doubles the transaction count, halves their size, and reads two
streams `B*K/2` apart. Interleaving instead — per column, each 16-K step
stores 8 bytes of hi nibbles immediately followed by 8 of lo — makes one
16-byte load yield both WMMA operands with no VALU, at exactly the same
column stride and locality as int8, and the same byte volume
(`K/2 + K/2 == K`), so it is free in DRAM.

**Generalisation**: this is the same failure mode as §7 of
`case-studies.md` (LDS-staged X share) and the `nontemporal` revert —
**on RDNA3/3.5, a change that removes instructions but worsens the
access pattern loses.** Count transactions and streams before counting
ops.

**Not yet landed**: the shipping compact GEMMs all take int8 and split
in-kernel. `quantize_act_oq8_batched` already visits each value and
could emit the interleaved form at no cost, handing the same ~13% to
the wave32, wave64 and tiled kernels without touching them. Needs an
audit first — the iu8 GEMM and the k-major overlay transpose also read
`oq8_xq_batch` as plain int8.

## 10. Matrix-instruction cycle costs (gfx1151) — governs FORMAT choice

From the AMD matrix calculator, per 16x16x16 block:

| instruction | cycles |
|---|---|
| `v_wmma_i32_16x16x16_iu4` | **16** |
| `v_wmma_i32_16x16x16_iu8` | 32 |
| `v_wmma_f32_16x16x16_bf16` | 32 |
| `v_wmma_f32_16x16x16_f16` | 32 |

Two consequences that decide format questions before any kernel is
written:

- **Exact W4A8 via two iu4 passes costs the SAME WMMA cycles as one iu8
  pass**, while feeding the compact nibbles raw with no unpack. This is
  why compact beats oq8 by 2.4x and dense bf16 outright on prefill
  (`docs/experiments/2026-08-22-prefill-gap-vs-mainstream.md`).
- **Promoting a K-tile to int8 WEIGHTS is cycle-neutral** in a W4A8
  two-pass kernel: the normal tile already spends 2 iu4 (32 cyc) and the
  promoted one spends 1 iu8 (32 cyc). Mixed weight precision along K
  therefore costs only bytes, never cycles.

Wave64's real property, same source: `iu4` needs **4 GPRs for C/D in
wave64 against 8 in wave32** (the 16x16 tile spreads over 64 lanes, not
32). Physical cost per tile is identical; the per-lane VGPR COUNT halves,
and that is what hits the 256-VGPR wall. Wave64 buys 2x the tile before
spilling.

## 11. Things to NOT try (negative results documented)

These failed in past experiments. Don't burn cycles re-running
unless you have a NEW idea about why this time would be different.

### `nontemporal` weight loads on gfx1100

Commit `34eb024` (revert of `0532579`). Original commit claimed
+2% based on within-session A/B. Bisect against the committed
baseline showed actual −13% on 9B MQ4 decode. Hypothesis: bypassing
cache-line allocation also defeats wave-level coalescing on RDNA3.
Don't re-try without a different mechanism for cache control.

### k2x32 wider-row variant

Commit `f670e16` — null result, kept the kernel for future revisit.
46% slower on 27B M=248320 lm_head due to register-pressure spills.
If you want to retry, lead with an LDS-staged B-share + manual
register budget plan.

### Always-on hipGraph capture

Commits `33b8861` / `5705a59` / `0180b68` / `688b4fd` — series of
fixes to make hipGraph capture safe. Default-on hipGraph caused
silent garbage output (dangling stack-pointer kernargs from raw
`launch_kernel` calls in `forward_scratch_layers`). Now opt-in via
`HIPFIRE_GRAPH=1` only, and even then perf-neutral or slightly
worse on most archs. Don't make it default-on without a thorough
correctness pass.

### Two separate nibble planes for activations (gfx1151)

See §9. Removes all ~20 unpack instructions per WMMA and measures **30%
SLOWER** than splitting in-kernel. Interleave at fragment granularity
instead.

### LDS weight tile to kill cross-wave redundancy (gfx1151)

Ladder rung 2: all 8 waves of a workgroup share `blockIdx.x`, so each
re-loads the same 32 weight rows — 8x redundant global traffic, ~170 GB/s
of nominal against a 248 GB/s ceiling. Staging that tile in LDS once per
group and reading fragments from there bought **+1%** (2.105 -> 2.086 ms).
The redundancy was free because 4 KB per group is L2-resident; the
nominal bandwidth figure was almost entirely cache hits. Same lesson as
the overlay gather (`docs/perf/gfx1151-gemm-levers.md`): a big traffic
number is not a DRAM number until you check the working set against L2.

### M-blocking past 2 in a two-accumulator W4A8 kernel (gfx1151)

Ladder LWMt sweep, gate/up, parity PASS each: 1 -> 35.8 TOPS (86 VGPR,
16 waves), **2 -> 43.4 (144 VGPR, 10 waves)**, 4 -> 36.9 (256 VGPR, 5
waves), 8 -> 16.8 (256 VGPR, **221 spills**). Peaks at 2 and falls off
hard — lower than the R=4-8 that §2's multi-row GEMV likes, because
exact W4A8 carries TWO i32 accumulator sets (hi and lo) so the same tile
costs double the registers. Same shape as §3's k2x32 null result.

### LDS-staged X share on gate_up (gfx1100)

Historical branch note: variant kernel `gemm_gate_up_hfq4g256_wmma_ldsx.hip`
opt-in via `HIPFIRE_GATE_UP_VARIANT=ldsx`, default off. ISA-clean (75 VGPRs vs
80 baseline, single-wave block makes `__syncthreads()` a no-op so
the compiler kept weight prefetch above the LDS-write phase) but
**per-call wall regressed +20% / +29% / +37% at pp32 / pp128 /
pp512 on Qwen 3.5 9B**. Replaced the baseline's 2 VMEM stalls per
inner-iteration with 3 VMEM + 2 LGKM stalls — the new stalls
weren't hidden by wave-level ILP the way the original `vmcnt(0)`
was. See case-studies §7 for the full diagnosis. Don't re-try
without (a) a fundamentally different LDS layout that doesn't
serialize through register, or (b) on RDNA4 (gfx12), where
`s_prefetch_data` may change the calculus.

## When you're done

Commit your win (or your null-result revert) with the commit-message
template from `playbook.md` step 6. The next contributor reading the
git log saves the same hours you spent — that's the durable value
of this kind of writeup.
