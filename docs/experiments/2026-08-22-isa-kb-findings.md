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

---

# Part 2: what the disassembly says, once the KB tells you what to look for

Disassembling `gemm_oq_compact_iu4_w64.hsaco` and counting the **steady-state
loop body only** (one 256-element compact group) gives the exact issue budget:

| class | count per group |
|---|---|
| WMMA | **64** |
| non-WMMA VALU | **291** |
| SALU | 92 |
| VMEM | 16 |
| LDS | 14 |

and the 291 breaks down as:

| op | count | what it is |
|---|---|---|
| `v_cvt_f32_i32` | 64 | rescale: i32 acc -> f32 |
| `v_fma_mix_f32` | 64 | rescale: apply the f16 weight scale |
| `v_fmac_f32` | 64 | rescale: apply the activation scale, accumulate |
| **`v_mov_b32`** | **64** | **accumulator reset — pure shuffling, no arithmetic** |
| address adds | 32 | |

This **calibrates the issue-bound model** that the no-op probe established.
Solving `64X / (64X + 291) = 0.51` (the measured fraction of peak) gives
**X = 4.7 cycles per WMMA**, with VALU at 1. The model is self-consistent: WMMA
and VALU serialize on one issue port, a WMMA costs ~4.7 slots, and every VALU op
in the loop is one slot of lost matrix throughput.

## The rescale is arithmetically irreducible at G=256

`facc += (float)acc * sw * sx` compiles to exactly 3 ops (cvt, fma_mix, fmac),
and that is minimal:

* `sw` depends on the row `m`, `sx` on the column `n`. In the wave64 C layout
  (`C[m][n]` in `V(m>>2)`, lane `16*(m&3)+n`) `m` varies across lane groups and
  `n` within them, so **neither scale is uniform over a VGPR** and neither
  factors out of the elementwise loop.
* Pre-combining `sw*sx` does not help — the product depends on `i`, `j` and `g`,
  so there are exactly as many products as elements.
* The cvt cannot be folded: `v_fma_mix_f32` reads f16/f32, never i32, and the
  bit-hack int->float alternative (`or` + `sub`) is 2 ops, worse than 1 cvt.

The only way to cut rescale cost per WMMA is **more WMMAs per rescale**, i.e. a
larger quant group. G=256 gives 64 WMMA per 192 rescale ops; G=512 would halve
it. That is a format and quality change, not a kernel change, and it is not on
the table here.

## FAILED: inline-constant C to delete the 64 `v_mov_b32`

The KB notes "inline constants may only be used for the C matrix", which looked
like a free 22% cut of the non-WMMA VALU: the 64 movs are the per-group
accumulator reset, and if the first WMMA of each group took a literal zero as C
instead, they would vanish. Predicted +12% (53.4 -> ~60 TOPS) under the model
above.

Implemented by restructuring the strip loop into an explicit group/strip nest so
that "this strip starts a group" is compile-time known, then passing
`((i32x4){0,0,0,0})` as C on that first WMMA. **Parity passes, performance does
not improve, and the disassembly shows why it cannot:**

* **No WMMA received an inline constant.** Every one still names a VGPR tuple as
  its C operand. The ISA permits the encoding; the **LLVM builtin cannot express
  it** — `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64`'s C operand is constrained
  to a v4i32 register class, so the backend materialises the zero into registers
  and the movs survive.
* The forced full unroll of the group made it strictly worse: VGPRs 240 -> 256,
  and **`vgpr_spill_count` 0 -> 2** — the kernel began spilling. `v_mov_b32` per
  group went **up**, 64 -> 132.

Reverted. Baseline restored and re-verified: 240 VGPRs, 0 spills, parity PASS.

**The lesson generalises:** an ISA capability is only reachable if the compiler
intrinsic exposes it. Reading the ISA tells you what the silicon can do, not what
you can ask for through HIP. Checking the disassembly for the encoding you
expected is the only way to tell the difference, and it is cheap.

## Also closed: the wave32 VOPD question, negatively

Part 1 flagged wave32's VOPD dual-issue as the one lever arguing for wave32 over
wave64. The opcode list settles it — **it is not worth pursuing**:

* The `V_DUAL_*` set is **f32-only** (FMAC/MUL/ADD/SUB/MOV/CNDMASK/MIN/MAX/
  DOT2ACC) plus three integer Y-column ops (`ADD_NC_U32`, `LSHLREV_B32`,
  `AND_B32`).
* **No `V_CVT_*` is dual-issuable anywhere**, so the rescale's cvt — a third of
  its cost — can never pair.
* Only a **left** shift is in the set. Nibble unpacking needs a **right** shift,
  which cannot co-issue on gfx1151.

So of the loop's non-WMMA VALU, only the `v_fmac_f32` could ever pair. Even
assuming *perfect* dual-issue of everything eligible, wave32's ratio improves
from 4.56 to ~2.78, predicting ~38 TOPS — still well below wave64's measured
53.4. **wave64 wins decisively and the wave32 path is closed.**

## Unused: the RDNA3.5 scalar FP unit

gfx1151 has a scalar float unit gfx1100 lacks (`S_CVT_F32_F16`, scalar F32/F16
arithmetic, 4-cycle SALU latency), and the KB suggests offloading uniform
scale/zero-point math to it — attractive when VALU issue is the binding
constraint, since SALU is a different port. **The compiled kernel contains zero
scalar float ops.**

It does not obviously help *this* kernel: neither scale is wave-uniform, which is
the same property that makes the rescale irreducible. It is worth revisiting for
the **decode GEMV**, where B=1 makes the activation scale a single wave-uniform
scalar per group — though that kernel is already at ~90% of the DRAM ceiling, so
freeing VALU slots there would buy nothing today.
