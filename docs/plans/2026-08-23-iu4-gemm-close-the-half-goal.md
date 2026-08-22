# Goal: close the exact-W4A8 GEMM's missing half (Qwen3.8-27B, gfx1151)

## Objective

`gemm_oq_compact_iu4x2_w64` runs at **54% of the achievable iu4 issue rate**.
Close as much of that as the hardware allows, and raise Qwen3.8-27B prefill
accordingly. Secondary: establish what fraction of prefill the GEMM even is, so
the effort lands where the time is.

**Hard floor: never below 4.25 bits per weight.** Kernels, dispatch, runtime,
quantization and requantization are all in scope. Activation precision is NOT
covered by the floor -- activations are transient.

## Baseline (measured 2026-08-23, do not re-derive)

Halo, gfx1151, 40 CU / 20 WGP, ~2.9 GHz, 248.5 GB/s DRAM, 32 MB MALL, ROCm 7.14.

- **iu4 ceiling = 110.9 TOPS** measured on a register-only WMMA probe (93% of
  the 118.8 paper figure). Target 110.9, not 118.8.
- **One wave per SIMD saturates the matrix pipe** given >=4 independent
  accumulators (109.7 TOPS at 80 blocks). Occupancy is not a lever; a VGPR diet
  buys nothing unless it stops an actual scratch spill.
- Shipping kernel: **60.3 hwTOPS on gate/up = 54%** of achievable.
- Ablation split: staging **21%**, group fold **10%**, LDS reads **~20%**.
- End to end: **~234 tok/s** prefill (2059-token prompt, kvarn, MAX_BATCH=512).

Full detail: `docs/experiments/2026-08-23-iu4-gemm-ceiling-attribution.md`.
Levers and negatives: `.agents/skills/hipfire-kernel-tuning/levers.md` §9-§15.

## Already closed -- do not re-attempt

- **Hand-written double-K / wider WMMA operand reads.** LLVM already coalesces
  them; hand-writing emits byte-identical ISA. Every LDS load is already
  `ds_load_b128`. No quadruple-K exists (16 B is RDNA's widest single load).
- **Growing the workgroup.** 8-wave workgroups lose ~18% regardless of tile
  shape -- barrier coupling, see levers §12. Stay at <=4 waves.
- **Growing the per-wave tile.** Flat until it spills, then a cliff (WMt/WNt 4x4
  spills 85 VGPRs -> 37.5 TOPS).
- **Widening the A staging load.** 4 B -> 8 B measured +0.1%.
- **XPAD.** 16 is optimal; 32 is worse, 48 is identical, non-multiples of 16
  break the b128 reads.
- **`s_singleuse_vdst`.** Absent from ROCm 7.14 three ways. Re-check only on a
  toolchain bump.
- **Concurrent streams for independent projections.** Occupancy is unchanged
  whether the second resident workgroup is from the same kernel or another, so
  there are no extra bubbles to fill.
- **Overlay fusion into the GEMM epilogue.** Tried twice, lost 6-14% both times.

## Progress (updated 2026-08-23)

**216.3 -> 256.3 tok/s prefill, +18.5%**, cross-process verified with
`scripts/probe_commits.sh e741131d5 29c01daf8` (scratch worktree, separate build
dir). Prefill window 8.457 s -> 7.381 s at 98.7% GPU busy.

Step 0 changed the plan's priorities: the GEMM was 46% of prefill and the sparse
overlay -- correcting 1.17% of the weight positions -- was 30.2%. Four of the
five wins came from outside the GEMM.

| # | change | e2e |
|---|--------|-----|
| 1 | overlay: compile-time loop bound (killed `v_movrels`) | +3.8% |
| 2 | overlay: LDS transpose so Y stores coalesce | +9.3% |
| 3 | GEMM: grid swizzle, N as the fast axis (step 1) | +2.7% |
| 4 | overlay: hoist the k-major transpose out of the route | +1.8% |
| 5 | GEMM: seed group accumulators from a shared zero | +1.7% |

Detail: `docs/experiments/2026-08-23-prefill-overlay-and-swizzle.md`.
Levers: `.agents/skills/hipfire-kernel-tuning/levers.md` §16-§19.

Current split: GEMM 50.3% (3667.5 ms), overlay 20.3% (1476.6 ms), attention
14.1% (1030.0 ms), gated delta net 5.1%.

### Closed since this plan was written -- do not re-attempt

- **BK=128** (was step 2). Loses 13% on gate/up. LDS 45056 B drops the WGP to one
  resident workgroup: fewer barriers does not pay if it costs the second
  workgroup that was hiding them. levers §12 from the other direction.
- **Manual fragment software-pipelining** (was step 2, B1). LLVM already hoists
  14 `ds_load`s to the top of the body and consumes them under progressively
  relaxed `lgkmcnt(12)...(5)`. Nothing left to hand-write. levers §19.
- **LDS staging of the overlay's activation tile.** Break-even is R = 85 rows per
  workgroup (32 kB staged against R*384 B gathered) and the accumulator registers
  cap R well below that. Not worth building.

### Still open, in order

1. **GEMM, 50.3% of prefill, ~57% of the 110.9 TOPS ceiling.** The 21% staging
   and 20% LDS-read costs remain latency-shaped and have now resisted width,
   tile geometry, wave grid, BK, and pipelining. What is left is step 3 (4-bit
   activations, halving the matrix work) and `s_prefetch_data`.
2. **Overlay, 20.3%.** The store is fixed; the gather now dominates, reading
   ~535 MB per gate/up call to consume a 2.6 MB activation. Needs cross-row
   sharing, which needs either big row blocks (register-bound) or a CSC-style
   reorder (write-bound). No cheap version identified.
3. **Attention, 14.1%** (`attention_flash_kvarn_tile_batched`, 1030 ms / 1168
   calls). First scan done, no fix yet. What it says:
   - 42-53 VGPRs, 16 waves/SIMD, no spills, no LDS. Not occupancy-limited.
   - Its three largest loops are **SALU-dominated**: 47 SALU / 39 VALU / 1 global
     load out of 92 instructions. That is scalar address and index math, not
     memory and not float work.
   - 11 `v_movrel` per kernel, but only 3 land in the biggest loop, so levers §16
     is NOT the lever here. An attempt to apply it anyway (compile-time MAXDPT
     bound + break, as in the overlay) took movrel 22 -> 64: MAXDPT is 8/16, far
     larger than MAXQ=4, so the unroll multiplies copies instead of collapsing
     indices. Reverted, do not retry that shape.
   - **Attribution done 2026-08-23 by ablation** (kernel total 1012.5 ms):
     Phase D (V accumulate) 360 ms = 36%, Phase A (Q.K + KVarN dequant) 264 ms
     = 26%, Phase C `expf()` FREE, remainder ~38%. **No single dominant phase,
     so no single big lever.** The kernel's own comment claiming "Phase D is
     99.9%" was stale for batched prefill and has been corrected in place.
   - PC sampling is NOT available: rocprofv3 rejects every configuration on
     gfx1151 ("not supported on any of the agents"). Ablation is the only
     per-phase attribution here.
   - Static ISA reading MISLEADS on this kernel: the biggest static loop (92
     instrs, 1 FMA, 47 SALU) is unchanged by edits to Phase D, so it is not
     Phase D at all. Three static-driven attempts were made and all reverted.
   - Phase D's inner dim loop is `for (i = 0; i < dpt; i++) out_vec[i] += wvs *
     vq[i]` with `dpt = head_dim/32` RUNTIME. The compiler neither unrolls it nor
     gives `out_vec` static registers, so the ISA shows **1 FMA per 39-92 loop
     instructions** -- the rest is address math and movrel.
   - **The cheap fix does not work here.** `#pragma unroll` to MAXDPT + `break`
     (which took movrel 24 -> 0 in the overlay at MAXQ=4) makes it WORSE at
     MAXDPT=8/16: 22 -> 36 for one loop, -> 42 for two, -> 64 for six, with the
     biggest loop byte-identical. Tried three ways, all reverted.
   - Templating `dpt` was probed (`kvarn_tile_body<4, NW>` with `dpt = MAXDPT`):
     it roughly doubles ISA arithmetic density (1 FMA per 92 instructions -> 4
     per 193) but does NOT remove the movrels, and given Phase D is only 36% the
     e2e ceiling on it is small. Worth folding into a larger Phase D rewrite,
     not worth a dispatch change on its own.
   - Pointer strength-reduction in Phase D (`vbh` advances by a constant
     `NW * v_row_stride`) is ALREADY done by LLVM -- hand-writing the induction
     variable emits identical ISA. Negative.

## Work, in order

### 0. Frame it: what fraction of prefill is the GEMM?

One `rocprofv3 --kernel-trace` driven through the daemon's **stdin JSON
protocol** (see AGENTS.md -- `--attach` and profiling `hipfire eval` both fail).
Split prefill wall time across GEMM / sparse overlay / attention / norms / KV.

This decides everything downstream: if the GEMM is 60% of prefill, a 1.8x GEMM
is only ~1.3x end to end and the 25%-priced overlay pass is the better target.
**Do this before any requantization.**

### 1. Workgroup swizzle (cheap, and it gates step 3)

The launch is `[M/BM, B/BN, 1]` = `[272, 4, 1]` with `blockIdx.x` fastest, so
the full 47 MB weight set is swept once per N-block -- 4 sweeps, 190 MB of DRAM,
against a 32 MB MALL that cannot hold one sweep.

Remap so the N-blocks of one M-block run together. Resident working set becomes
~174 kB of weights plus the whole 2.6 MB X tensor -- both cached. Note the
current order is not arbitrary (it keeps X resident); verify the swizzle keeps
BOTH, and measure DRAM bytes, not just wall time.

### 2. Attack the two latency costs

Neither responds to width or tile size, so both need structural change:

- **Fragment-level software pipelining** (levers §3, never attempted): prefetch
  step s+1's LDS fragments into registers during step s's WMMA. ~12-24 VGPRs,
  and we have headroom at 214. Targets the 20% LDS-read cost directly.
- **Shared-zero C operand**: 64 `v_mov_b32` per group fold exist only to zero
  the accumulators. Pass one shared zero quad as the WMMA C operand on the first
  K-step of each group instead. Targets part of the 10% fold cost.
- **BK=128**: halves the barrier count. Drops to one resident workgroup, but
  that is 4 barrier-coupled waves, not 8, and 1 wave/SIMD is enough per the
  baseline. Genuinely uncertain -- it also tests whether the §12 mechanism is
  right.
- **`s_prefetch_data`** on the weight stream, against the staging latency.
- **Dual-issue the fold**: 32 `v_fmac_f32` are VOPD-eligible, `v_fma_mix_f32` is
  not. Restructure toward dual-issuable ops.

### 3. Halve the matrix work: 4-bit activations

The two WMMA passes are a **radix-16 split of the activations**
(`x = 16*x_hi + x_lo`), not an outlier correction -- every int8 activation has
both nibbles, so the cost is dense and unavoidable at A8. The sparse overlay is
a separate mechanism (weight outliers, separate kernel, priced at 25%). Do not
conflate them.

Two routes, and **3a bounds 3b**:

**3a. CLOSED 2026-08-23 -- the hi-pass skip is dead.** Measured on real
Qwen3.8-27B prefill activations with `HIPFIRE_HIPASS_STATS=1`
(`act_hipass_tilemax`, max|x| per 16x16 fragment):

    fragments      0 / 17,301,504  = 0.0000% dead
    BN=128 blocks  0 /  2,162,688  = 0.0000% dead
    mean fragment max|x| = 122.1  (of 127)

Not marginal -- zero, at both granularities, over 17.3 M fragments.

**The cause is structural and worth stating, because it also rules out variants
of the idea.** Activations carry a per-group (256-K) SYMMETRIC scale, so every
group is normalised to put its max at ~127 by construction. A 16x16 fragment
draws 256 values from that group, so it essentially always contains a value
>= 16, and `x_hi` is essentially never all-zero. Making this work would need a
COARSER activation scale so some groups are genuinely small against a global
max -- and coarse scale is already a measured dead end on quality.

So there is no "skip the second pass where it is free". Halving the matrix work
requires making ALL activations fit in 4 bits, i.e. 3b, which is now the only
route.

**3b. W4A4 via learned rotations.** `gemm_oq_compact_iu4_w64` already exists,
and SpinQuant Cayley-SGD rotations were found to make 4-bit activations serve
coherently -- with the specific finding that **learned rotations are
prefill-only** (decode prefers plain FWHT), which is exactly this case. A true
2x on matrix work.

Caveat: at 2x compute the weight stream approaches 232 GB/s against a 248 GB/s
wall, so expect ~1.8x and only if step 1 has landed. Quality must be held to the
existing KLD bar -- report KLD alongside tok/s or the number means nothing.

### 4. Long shots, only if 0-3 stall

- **NPU co-execution.** Measured NPU ~55 TOPS vs GPU ~56 int8; W4A8 is already
  bit-exact in-tree. Prefill's large batches are the one workload that can
  amortize the ~4 ms/dispatch encode floor.
- **Fragment-order weight layout in DRAM.** The activation twin landed at +2.4%
  e2e / 1.184x on the kernel. For weights it would let each lane
  `global_load_b64` its own fragment and skip the LDS round-trip entirely --
  attacking staging AND LDS reads together. Costs a quantizer format change; it
  is a pure permutation, so bit-exact at the same 4.25 bits.

## Verification protocol -- non-negotiable

- **Alternate A/B in one session, >=3 rounds.** Session drift is ~5% here: a
  standalone run read 245.7 tok/s for a binary that reads 234.0 when paired. An
  unpaired before/after WILL book drift as a win.
- **Cross-process verify anything that survives**: `scripts/probe_commits.sh
  <baseline> <candidate>` builds each commit in a scratch worktree and is safe
  on a dirty tree.
- **`include_str!` stale-kernel trap.** Kernel sources are compiled INTO the
  Rust binary. Deleting the `.hsaco` re-JITs the stale string. `touch
  crates/hipfire-rdna/src/kernels.rs` and rebuild, every time. This trap has
  already invalidated an entire ablation sweep twice.
- **`hipfire eval` caches on model+prompt+binary_hash, NOT env.** Env-only A/Bs
  replay arm 1. Use `--force`. Identical numbers mean a cache hit.
- **Parity + coherence on every kernel change**: run
  `parity_gemm_oq_compact_iu4x2_w64` and check the 64-token greedy coherence
  sha is unchanged (`c79a9bfe8711` at time of writing).
- **`./tests/no-gpu-ci.sh`** before handing off. Regenerate `docs/env-vars.md`
  with `hipfire gen-env-docs` if the freshness gate trips.
- **GPU lock**: wrap benches in `hipfire lock acquire/release`. NEVER wrap
  `hipfire-eval`, the `tiny-*` gates, or `coexistence calibrate` -- they
  self-lock and deadlock naming your own label.

## Definition of done

Either the GEMM is above ~80% of 110.9 TOPS, or every remaining gap has a
measured cause and a documented reason it cannot be closed. Negative results
ship: commit them with the hypothesis and the number that killed it, and add
them to levers.md so the next attempt does not re-run the same experiment.

Land work incrementally on `perf/dram-read-bandwidth` with real measured numbers
in the commit messages.
