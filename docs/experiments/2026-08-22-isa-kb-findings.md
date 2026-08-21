# What the ISA knowledge base (~/kb) settles about the compact WMMA kernels

Read `~/kb/{_BRIEF,wmma-register-layout,occupancy-and-limits}.md` and
`~/kb/gfx1151/`. Several things this session established empirically are now
confirmed from the ISA, one design assumption is corrected, and one of the KB's
own open questions can be closed from our parity results.

## Confirmed — our kernels are ISA-correct, not merely parity-correct

* **wave64 C/D uses 4 VGPRs, wave32 uses 8.** `C[m][n]` lives in `V(m>>2)`, lane
  `16*(m&3) + n` for wave64. Our writeback is `r = base + 4*g + (wtid>>4)`,
  `oc = base + lane` — algebraically the same mapping. This was ported from a
  working kernel and validated only by parity; it is now known correct by
  construction.
* **Only lanes 0-15 carry A/B; the caller must replicate into 16-31 (and 32-63
  in wave64). The hardware does NOT do it.** Our fragment reads index with
  `lane = wtid & 15`, so all four lane groups read the same address and the
  replication happens implicitly. Correct, but by accident of addressing rather
  than by intent — worth knowing, because a future rewrite that indexes by full
  `wtid` would silently produce garbage, not a fault.
* **`V_WMMA_*_IU4`/`IU8` default to UNSIGNED**; `NEG[0]`/`NEG[1]` select
  signedness per operand. Every call site passes explicit flags — and the 2-pass
  kernel's `(true, a, false, b_lo, ...)` mixed pairing is exactly the documented
  mechanism, not a trick.

## Confirmed — why one accumulator chain saturates

The no-op probe measured ~105 TOPS flat from a single chain, which looked
surprising. The ISA says why: of the four WMMA scheduling rules, only
"second WMMA's A or B overlaps the first's D" is a correctness requirement
needing a `V_NOP`. **`C == D` accumulation is explicitly listed as needing
none.** Our inner loops are pure `C == D`, so there is no dependency stall to
hide, and extra accumulator tiles buy reuse rather than latency tolerance.

Note also: **no throughput or cycle counts appear in ANY of the four ISA
documents** — both RDNA3 and RDNA3.5 defer to the matrix calculator. So the
no-op probe is the only source we have for the issue rate, which raises its
value rather than lowering it.

## CORRECTED — occupancy, and why 240 VGPRs is acceptable

gfx1151 is a **1024-VGPR/SIMD** part (gfx1100 is the 1536 one), wave32 granule
16, wave64 granule 8, and **a wave64 costs 2x the register file for the same
logical VGPR count**. So for wave64, `waves/SIMD = 512 / VGPRs`.

Measured from the compiled `.hsaco`:

| kernel | VGPRs | LDS | wave | waves/SIMD |
|---|---|---|---|---|
| compact iu4 **w64** | **240** | 20 kB | 64 | **2** |
| reference `gemm_iu4_i32_wmma_lds` | **136** | 20 kB | 64 | **3** |
| compact iu4 wave32 | 214 | 33 kB | 32 | 4 |
| no-op probe | 9 | 0 | 64 | 16 (capped) |

No spills anywhere. The compact kernel carries **104 more VGPRs than the
reference**, entirely the f32 accumulator set the per-group rescale forces, and
that costs a third of the occupancy.

**Tested rather than assumed:** WNt 8 -> 4 drops VGPRs 240 -> 166 and raises
occupancy 2 -> 3 waves/SIMD, and it is SLOWER (1.4x vs 1.6x over wave32). Data
reuse beats occupancy here, which is consistent with the no-op result — WMMA
needs no latency hiding, and the register-staged double buffer already covers
memory latency. **240 VGPRs at 2 waves/SIMD is the right operating point**, and
"reduce registers to raise occupancy" would have been a wrong turn.

## NEW — a wave32 advantage not previously considered

**VOPD (2x VALU dual-issue) is wave32-ONLY on every RDNA arch.** Since this
session established that the kernels are ISSUE-bound and that WMMA and ordinary
VALU share a port, a wave32 kernel can in principle dual-issue its non-WMMA VALU
and halve the epilogue's issue cost, where wave64 cannot.

That is a real tension with wave64's 2x accumulator density, and it may explain
why both measured kernels beat their VALU/WMMA-derived predictions (wave32:
predicted 22%, measured 32%; wave64: predicted 40%, measured 51%). Not chased
here; recorded because it is the only lever found so far that argues FOR wave32.

## Closing one of the KB's open questions

`wmma-register-layout.md` lists as unresolved: "the exact packing for `IU8` (4
elements per VGPR?) and `IU4` (8 per VGPR?) is not given by these figures and is
not in the text."

Our kernels answer it for IU4 empirically: the A/B operand is passed as
`int32x2` (2 VGPRs) carrying **16 int4 values**, i.e. **8 per VGPR**, packed
`byte = k_even | (k_odd << 4)`. Parity against a CPU oracle passes on every
shape, so the packing is confirmed. IU8 by the same construction is `int32x4`
(4 VGPRs) for 16 int8 values = **4 per VGPR**, likewise parity-confirmed.
