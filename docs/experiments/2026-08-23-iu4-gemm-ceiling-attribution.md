# Where the exact-W4A8 GEMM's missing half goes (gfx1151 / halo)

State box: halo, Strix Halo gfx1151, 40 CU / 20 WGP, ~2.9 GHz, 128 GB UMA,
248.5 GB/s measured DRAM, ROCm 7.14. Kernel is
`gemm_oq_compact_iu4x2_w64` (exact W4A8 as two iu4 WMMA passes,
`x = 16*x_hi + x_lo`), Qwen3.8-27B oq4.25++ shapes.

## 1. The ceiling is real and it is 110.9 TOPS

Theoretical iu4 issue rate is 80 SIMD * (8192 ops / 16 cyc) * 2.9 GHz = 118.8
TOPS. A probe that issues nothing but `v_wmma_i32_16x16x16_iu4` on registers --
no LDS, no global, ACC independent accumulators so the 16-cycle result latency
never stalls -- measures:

| ACC | 80 waves | 160 | 240 | 320 |
|-----|---------|-----|-----|-----|
| 4   | 106.7   | 108.5 | 109.1 | 109.7 |
| 8   |  74.1   | 110.4 | 110.7 | 110.9 |
| 12  | 108.3   | 110.5 | 110.8 | 110.9 |

**110.9 TOPS = 93% of theoretical.** Use 110.9, not 118.8, as the target.

**ACC=4 at 80 blocks is ONE wave per SIMD and already reaches 109.7.** A single
wave saturates the matrix pipe provided it has >=4 independent accumulators.
This kills a whole class of tuning: for a matrix-issue-bound kernel on gfx1151,
**occupancy is not a lever.** Do not spend register budget buying waves. (ACC=8
at 80 waves dips to 74.1 -- one wave cannot cover 8 accumulators' worth of
register pressure and scheduling; the fix is more waves OR fewer accumulators,
and 4 is enough.)

## 2. The shipping kernel is at 54% of that

`bench_iu4x2_w64_peak`, B=512. hwTOPS counts iu4 ops the matrix unit retires;
useful is half that by construction (two passes per MAC).

| proj | M | K | ms | hwTOPS | % of 118.8 |
|------|---|---|----|--------|-----------|
| gate/up | 17408 | 5120 | 3.03 | 60.3 | 50.7% |
| down | 5120 | 17408 | 3.45 | 53.0 | 44.6% |
| qkv | 6144 | 5120 | 1.05 | 61.1 | 51.5% |
| wo | 5120 | 4096 | 0.84 | 51.3 | 43.1% |

The kernel has 16 independent accumulators, so per §1 it is not
dependency-stalled and not occupancy-starved.

## 3. Ablation attribution

Timing-only builds (results deliberately wrong), gate/up, % of 118.8:

| variant | %peak | delta |
|---------|-------|-------|
| shipped | 50.7% | -- |
| fold once at the end instead of per group | 56.1% | **fold = 5.4 pts** |
| no staging after the first strip | 64.5% | **staging = 13.8 pts** |
| neither | 74.6% | 23.9 pts |
| pure WMMA, no memory at all (§1) | 93.3% | **LDS reads = 18.7 pts** |

So: ~21% of runtime is staging, ~10% is the group fold, and ~20% is the LDS
reads that remain even with staging removed.

## 4. Double-K on the A operand: LLVM already does it

One iu4 fragment is 16 K = 8 B, so a per-step weight read looks like a 64-bit
`ds_load` -- half the 128-bit LDS interface -- and the obvious fix is to read
32 K at once and feed two K-steps from the halves.

**It is already happening.** The full unroll makes the s and s+1 reads provably
adjacent and LLVM coalesces them. Hand-writing the double-K read emits
**byte-identical ISA** (the only diff is the `__hip_cuid_*` symbol and the
source filename). Every LDS load in the kernel is already `ds_load_b128`:
20 of them per 64 `v_wmma`, zero narrower.

There is no quadruple-K: `ds_load_b128` / `global_load_dwordx4` at 16 B are the
widest single loads that exist, so 128 bits is the ceiling per instruction.

The X side reaches 128-bit by construction rather than by luck -- the
fragment-interleaved layout packs hi+lo of one 16-K step into one 16 B read.

## 5. Eight-wave workgroups lose ~18%, and it is the BARRIER not the tile

Staging is 12288 B per K-block for 256 WMMA = 48 B/WMMA, while LDS use is only
24 kB of 64 kB. Growing the tile should cut that ratio. It does, and it loses:

| WARPS_M x WARPS_N | BM x BN | waves | LDS | gate/up hwTOPS |
|---|---|---|---|---|
| 2 x 2 | 64 x 128 | 4 | 24576 | **60.7** |
| 4 x 1 | 128 x 64 | 4 | 18432 | 59.4 |
| 1 x 2 | 32 x 128 | 2 | 22528 | 58.2 |
| 4 x 2 | 128 x 128 | 8 | 28672 | 49.3 |
| 2 x 4 | 64 x 256 | 8 | 45056 | 50.0 |

Both 8-wave shapes lose ~18%; every shape at 4 waves or fewer is fine. Shape is
not the variable, **wave count per workgroup is**.

Mechanism: at 4 waves and 24 kB, TWO independent workgroups fit per WGP, so one
can compute through the other's `s_barrier`. At 8 waves only one fits, and all
eight stall on the same barrier. Waves resident per WGP is 8 either way -- what
changes is whether they are barrier-coupled.

**Rule: on gfx1151 prefer two small workgroups over one big one.** LDS headroom
is not free real estate if spending it drops you to one resident workgroup.

## 6. Growing the PER-WAVE tile does not help either

At 4 waves (2x2), sweeping the per-wave tile:

| WMt x WNt | BM x BN | VGPR | spill | gate/up |
|---|---|---|---|---|
| 2 x 4 | 64 x 128 | 218 | 0 | 59.3 |
| 3 x 4 | 96 x 128 | 256 | 0 | 59.9 |
| 4 x 2 | 128 x 64 | 236 | 0 | 59.8 |
| 4 x 4 | 128 x 128 | 256 | **85** | 37.5 |

Flat within noise until it spills, then it falls off a cliff. Combined with §5:
**no tile geometry reachable in the register/LDS envelope recovers the 21%
staging cost**, which means that cost is latency and issue, not bytes.

## 7. Widening the A staging load 4 B -> 8 B: neutral

The A stage was one `global_load_b32` per thread. Widening to `i32x2` halves the
A load count and shows up in the ISA (4x b32 + 2x b64, was 8x b32).

End to end it is worth nothing: alternating A/B, 3 rounds, Qwen3.8-27B,
2059-token prompt, kvarn, MAX_BATCH=512:

    head  232.5   awide 234.1
    head  233.9   awide 234.0
    head  233.8   awide 234.0

+0.1%. Kept anyway (two fewer instructions, parity PASS), but it is not a win.

**Measurement note.** A standalone run minutes earlier read 245.7 tok/s for the
same binary that reads 234.0 here. That 5% is session drift. Only the
alternating form is trustworthy; an isolated before/after across sessions would
have reported this neutral change as a +5% win.

## 8. `s_singleuse_vdst` is not reachable in this toolchain

Checked three ways against ROCm 7.14: no `single-use-vdst` subtarget feature in
`llc -mattr=help`, no `AMDGPUInsertSingleUseVDST` symbol in `libLLVM`, and
`llvm-mc -mcpu=gfx1151` rejects the mnemonic. Not available as an instruction
and not available as an LLVM pass.

Separately, if the intent is a cache hint ("these lines will not be reused"),
that is a different mechanism on RDNA -- the nontemporal / TH cache-policy bits
-- and hipfire already measured that as a fake +2% that was really -13%
(`.agents/skills/hipfire-kernel-tuning/case-studies.md` §2).

## What is left

Against 110.9 TOPS achievable the kernel is at 54%. The 21% staging and 20%
LDS-read costs are both latency-shaped, and neither responds to width or tile
size. The remaining structural idea is to stop paying them per K-strip at all --
i.e. fewer, larger K-strips per barrier, which is bounded by LDS, or a
persistent-workgroup formulation that keeps weights resident across N-blocks.
Neither is attempted here.
