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

> **This section was wrong when first written and is corrected here.** It read
> gfx1151 as a 1024-VGPR/SIMD part, from `occupancy-and-limits.md`. That is a
> **KB error**: `rocm-toolchain.md` states gfx1151 carries LLVM's
> `Feature1536VGPRs` (gfx1150/1152/1153 do not), and the installed LLVM confirms
> it — `libLLVM.so` contains `1536-physical-vgprs` / `has1536VGPRs`. Every wave
> count below is now taken from the compiler
> (`-Rpass-analysis=kernel-resource-usage`) rather than derived from a constant.

gfx1151 has **1536 VGPRs/SIMD**, the same file as gfx1100 — not 1024. A wave64
still costs 2x the register file for the same logical VGPR count, so for wave64
`waves/SIMD = 768 / VGPRs`. The compiler agrees, and reports the same occupancy
for this kernel on gfx1151 and gfx1100.

Compiler-reported, not derived:

| kernel | VGPRs | LDS | wave | waves/SIMD | (I had claimed) |
|---|---|---|---|---|---|
| compact iu4 **w64** | **240** | 20 kB | 64 | **3** | ~~2~~ |
| reference `gemm_iu4_i32_wmma_lds` | **136** | 20 kB | 64 | **5** | ~~3~~ |
| compact iu4 wave32 | 214 | 33 kB | 32 | **7** | ~~4~~ |
| no-op probe | 9 | 0 | 64 | 16 (capped) | 16 |

No spills anywhere. The compact kernel still carries **104 more VGPRs than the
reference**, entirely the f32 accumulator set the per-group rescale forces, and
it still costs occupancy — but the gap is 3 vs 5 waves, not 2 vs 3.

**The tested conclusion is unaffected.** WNt 8 -> 4 drops VGPRs 240 -> 166,
which raises occupancy from 3 to 4 waves/SIMD, and it is SLOWER (1.4x vs 1.6x
over wave32). Data reuse beats occupancy here, consistent with the no-op result —
WMMA needs no latency hiding, and the register-staged double buffer already
covers memory latency. **240 VGPRs is the right operating point**, and "reduce
registers to raise occupancy" would have been a wrong turn. Only the absolute
wave counts were wrong; the ordering, the experiment, and the conclusion stand.

**Lesson for using this KB:** where a doc derived from the ISA PDF disagrees with
one derived from LLVM about a *target-specific* constant, the LLVM-derived one
wins — the ISA PDF describes RDNA3.5 generically and does not enumerate per-part
register files. Better still, ask the compiler: `-Rpass-analysis=kernel-resource-usage`
prints VGPRs, spills, LDS and occupancy directly, and costs one compile.

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

---

# Part 3: the rest of the KB

Read the remaining documents: `matrix-and-dot-ops`, `dtypes-and-conversion`,
`lds-and-crosslane`, `memory-and-sync`, `hazards-and-waitstates`,
`rocm-toolchain`, and all of `gfx1151/`. Findings, in descending order of what
they change.

## 1. TOP CANDIDATE: `V_DOT8_I32_IU4` eats packed nibbles with no unpacking

`dtypes §10.4` — "`V_DOT8_*_*4` consumes a packed dword of eight nibbles
**directly, with no unpacking at all** — one instruction replaces 16 ops of
§10.1 plus 8 FMAs... For W4A4/W4A8 MoE GEMV this is by far the best path on
gfx1151."

**`gemv_oq_compact_multicol` does exactly the unpacking this avoids.** It spends
a `v_perm_b32` interleave plus OR-based 4->8-bit sign extension to widen nibbles,
then feeds two `sudot4` (dp4a). Per 8 weights that is roughly perm + or + 2x dot4
= 4 ops. Using `sudot8` on the raw nibbles instead: split the int8 activation into
two int4 planes **once per activation** (amortised across every row), then 2x dot8
= 2 ops in the inner loop. **~2x less inner-loop ALU.**

Verified reachable — this compiles and selects correctly on gfx1151:

```
__builtin_amdgcn_sudot8(true, a, true, b, acc, false)  ->  v_dot8_i32_iu4 ... neg_lo:[1,1,0]
__builtin_amdgcn_udot8(a, b, acc, false)               ->  v_dot8_u32_u4
```

Note the guard names: on gfx11/11.5 the builtins are `udot4`/`udot8`
(`dot7-insts`) and `sudot4`/`sudot8` (`dot8-insts`). **`sdot4`/`sdot8` are
`dot1-insts` and do not exist here** — reaching for them by CDNA muscle memory
fails to compile, and using `udot*` on signed data is silently wrong.

Worth doing because multicol sits at ~59 GB/s against a ~248 GB/s ceiling, i.e.
~24% — it is not bandwidth-bound, so inner-loop ALU is the right thing to attack.

## 2. Portability of the 240-VGPR w64 kernel — measured, not assumed

The KB warns a kernel tuned to the 1536-VGPR budget "will spill on
gfx1150/1152/1153 despite them all being RDNA 3.5". Compiled the actual kernel
for each target:

| target | VGPRs | waves/SIMD | spill |
|---|---|---|---|
| gfx1151 | 240 | 3 | 0 |
| gfx1100 | 240 | 3 | 0 |
| **gfx1150** | **252** | **2** | **0** |
| **gfx1201** (RDNA4) | — | — | **hard compile error** |

Milder than warned on gfx1150 — the allocator absorbs it by spending 12 more
VGPRs and dropping a wave, with no spill. RDNA4 fails loudly rather than
silently: `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64` does not exist on gfx12,
which needs the `_gfx12` builtins and a different fragment layout (RDNA4 packs
distinct K-elements into the upper lane half instead of replicating). A loud
failure is the good outcome; the dangerous port is the reverse direction.

## 3. Confirmations that close open threads

* **Cache bits are clean.** Zero `glc`/`slc`/`dlc` on any load in the kernel, so
  everything is default LRU/CU-scope. This matters because `GLC=1` "forces a miss
  *and invalidates the matching L0 line*" — it would have silently destroyed the
  L1 reuse the MW row-tiling exists to get.
* **The accumulate loop is NOP-free by the rule, not by luck.** "Matrix A/B can
  overlap C as long as C is distinct from D. The typical case is that C and D are
  the same." Same C=D set across many WMMAs with distinct A/B is explicitly the
  safe case. Note `s_nop` would NOT satisfy the hazard where it does apply — the
  filler must occupy the VALU pipe.
* **The scalar FP unit still cannot help.** `S_CVT_F32_I32` and scalar F32 math do
  exist on RDNA3.5 (not on gfx1100). But the rescale's weight scale indexes row
  `m = 4g + (wtid>>4)` and the activation scale indexes a lane-varying column, so
  neither is wave-uniform. Same property that makes the rescale irreducible.
* **`S_CLAUSE` is not a lever.** A VALU clause locks the arbiter onto one wave,
  which does not reduce instruction count in an issue-bound loop.

## 4. LDS bank conflicts: real, and correctly ignored

Bank index is `(byte_addr >> 2) & 31` over a 32-bank set. The fragment reads use
a `BK/2 = 32`-byte row pitch = 8 DWORDs, so lanes 0-15 touch only banks
{0,1},{8,9},{16,17},{24,25} — **a 4-way conflict**, on both the A and X reads.

It is genuinely there, and it is genuinely not worth fixing: the loop has 14 LDS
ops against ~590 issue slots. This is the ISA-level explanation for why
"LDS bank-pad" is already recorded in our tuning notes as a tried dead end — the
kernel is issue-bound, not LDS-bound, so removing the conflict buys nothing.
(A wave64 DS op also costs a 2-cycle minimum vs wave32's 1.)

## 5. Filed for later

* **`V_CVT_OFF_F32_I4`** is a signed-INT4 -> F32 LUT converter present on all four
  archs. It reads only `S0[3:0]`, so **no AND mask**, and the two's-complement sign
  is handled by the LUT: 2 VALU ops per weight versus the conventional
  shift+and+cvt+sign-fixup chain of 4-5. Output is `s4/16`, so fold the x16 into
  the block scale once. We have no f32 INT4 dequant path today, but any future one
  should start here — the KB notes it "is almost never used".
* **Cross-lane reductions**: `row_xmask` (XOR butterfly in a row of 16) and
  `row_share` are the RDNA reduction backbone; GCN's `row_bcast:15/31` and
  `wave_shl/shr/rol/ror` were **removed**. `ds_bpermute` in wave64 only crosses 32
  lanes on RDNA3/3.5, executing as two independent wave32s — any wave64 reduction
  must handle the split by hand. And **DPP cannot be used on `V_DOT4`/`V_DOT8`/WMMA
  or in VOPD**, so a cross-lane reduction cannot be fused into a dot.

## 6. A caveat on our own bandwidth number

`gfx1151/memory-system.md`: "Memory bandwidth: neither source gives a number...
**measure it, and measure it *with a CPU load running*, because the CPU shares the
same controller.**"

Our 248.5 GB/s pure-read ceiling was measured with an idle CPU. During real
serving the CPU is busy (tokenizer, sampler, daemon), so the bandwidth actually
available to a decode GEMV is likely lower. That does not change any ranking, but
it means "GEMV is at ~90% of the ceiling" is if anything an **under**-statement of
how close to done that kernel is, and a re-measure under load would tighten it.
