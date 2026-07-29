# RDNA 3.5 ISA — Performance Optimizations for LLM Kernels

Hardware-level optimization opportunities mined from the AMD "RDNA3.5" Instruction Set
Architecture Reference Guide, aimed at engineers writing GEMM, attention/softmax,
quantized-inference, normalization, and memory-bound elementwise kernels for
**RDNA 3.5 (gfx1150/gfx1151)** in HIP or assembly.

RDNA 3.5 provides `V_WMMA_*` matrix-multiply-accumulate for **F16, BF16, F32 and INT8/INT4**,
packed 16-bit VALU math, VOPD dual-issue, and a rich cross-lane permute set. It has **no FP8
support** — that arrives with RDNA 4 — so the densest sub-FP32 formats here are F16/BF16 and
the IU8/IU4 integer dots.

Every claim below cites a line number in `rdna35_instruction_set_architecture.md`, formatted
as `INSTR_NAME` (line N). Where a capability exists only in AMD's machine-readable XML and the
manual's prose omits it, the citation says so explicitly. Where the manual is silent on
throughput or cycle counts, the text says that rather than guessing — **this document contains
no invented performance numbers.**

> **How this was produced.** An automated multi-agent sweep read all 29,430 lines of the manual
> (one agent per ~120-line window) plus twelve thematic deep-dives, then adversarially
> re-verified every finding's verbatim quote against its cited line. 889 findings survived
> verification; an independent audit re-confirmed **99.8%** of them at their cited location.
> Corrections applied during assembly are noted inline. Treat cycle-level claims as a starting
> point for measurement, not as vendor-published performance guarantees.

## Top optimizations at a glance

| Optimization | Key instructions | LLM kernel it helps | Impact | Hidden? |
|---|---|---|---|---|
| 16×16×16 matrix MAC engine | `V_WMMA_F16_16X16X16_F16`, `V_WMMA_BF16…`, `V_WMMA_F32…` | GEMM, attention QK^T/PV | High | No |
| Keep `C==D` and one WMMA type per accumulation chain | `V_WMMA_*` (line 3059) | GEMM inner loop | High | **Yes** |
| INT8/INT4 matrix + dot products | `V_WMMA_I32_16X16X16_IU8/IU4`, `V_DOT4_I32_IU8`, `V_DOT8_I32_IU4` | Quantized GEMM/GEMV | High | No |
| Per-operand sign via NEG → free mixed signed×unsigned | `V_DOT4_I32_IU8` (line 16818) | Asymmetric quantization | High | **Yes** |
| Dual-issued F16/BF16 dot-accumulate | `V_DUAL_DOT2ACC_F32_F16/_BF16` (line 17187) | GEMV, skinny GEMM | High | **Yes** |
| VOPD packs two independent VALU ops (wave32 only) | `VOPD`, `V_DUAL_*` (line 2751) | Epilogues, activations, norms | High | No |
| VOPD bank + even/odd parity rules that gate pairing | `VOPD` (lines 2767, 2773) | Any dual-issued loop | High | **Yes** |
| Packed 16-bit math: two ops, one VGPR | `V_PK_ADD_F16`, `V_PK_MUL_F16`, `V_PK_FMA_F16` | Elementwise, bias, activation | High | No |
| Free `CLAMP`: [0,1] float saturate + integer saturation | VOP3/VOP3P CLMP bit (lines 2427, 2708) | Activations, quant requant | High | **Yes** |
| Mixed F16×F16+F32 FMA with per-source half-select and free ABS | `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, `V_FMA_MIXHI_F16` (lines 2675, 16945) | FP32-accurate epilogues on FP16 data | High | **Yes** |
| Free per-half NEG; but OMOD is ignored on packed FP16 | VOP3P NEG/NEG_HI (lines 2655, 2419) | Layernorm, residual | Medium | **Yes** |
| Direct global→LDS DMA (XML-only; manual omits it) | `GLOBAL_LOAD_LDS_B32`, `BUFFER_LOAD_LDS_*` (XML) | GEMM tile staging under VGPR pressure | Medium | **Yes** |
| `SADDR` mode: SGPR base + per-lane VGPR offset | `GLOBAL_LOAD_*` (line 4457) | Any global load — saves VGPRs | High | **Yes** |
| Prefer GLOBAL over FLAT (FLAT double-counts waitcnt) | `GLOBAL_*` vs `FLAT_*` (line 4528) | All memory-bound kernels | High | **Yes** |
| Wide loads need matched alignment (B128=16B, B64=8B) | `GLOBAL_LOAD_B128/B96/B64` (line 850) | Tile staging | High | **Yes** |
| Two addresses / 64-bit per LDS instruction | `DS_LOAD_2ADDR_*`, `DS_STORE_2ADDR_*` (line 4699) | LDS-bound GEMM staging | High | **Yes** |
| Cross-lane shuffle that uses **no** LDS banks or storage | `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, `DS_SWIZZLE_B32` (line 5125) | Softmax/layernorm reductions | High | **Yes** |
| XOR-butterfly lane shuffles for log₂ tree reductions | `DPP_ROW_XMASK` (line 6986) | Softmax max/sum, layernorm | High | **Yes** |
| Transcendentals are a separate `TRANS32` scoreboard class | `V_EXP_F32`, `V_RCP_F32`, `S_DELAY_ALU` (line 1787) | Softmax, GELU, RMSNorm | High | **Yes** |
| `S_BARRIER` does **not** wait on memory counters | `S_BARRIER`, `S_WAITCNT` (line 11143) | Any LDS-staged tile loop | High | **Yes** |
| Cache-scope hints for streaming vs reused data | `GLC`/`SLC`/`DLC` bits (lines 1447–1502) | KV-cache streaming, weight reuse | Medium | No |


Note: claude-sonnet-5[1m] (the safety classifier) was unavailable when reviewing this subagent's work. Please carefully verify the subagent's actions and output before acting on them.

All citations verified. Writing the section.

## Matrix engine (WMMA) for GEMM

RDNA 3.5 (gfx1150/gfx1151) accelerates matrix multiply-accumulate with **WMMA** (Wave Matrix Multiply-Accumulate), VOP3P-encoded ops that compute `D = A·B + C` over a fixed **16×16×16** tile with the row-column dot products distributed across the vector ALU (`V_WMMA_F32_16X16X16_F16` line 16996). This is the highest-throughput path for the QKV/output projections, MLP GEMMs, and attention score/context matmuls in an LLM.

### The instruction menu is fixed and small

There are exactly **six** WMMA opcodes (64–69), one tile geometry (16×16×16), and one accumulate form. Pick the datatype pairing your kernel needs:

| Opcode | Instruction | A / B | C / D (accumulator) | Use |
|---|---|---|---|---|
| 64 | `V_WMMA_F32_16X16X16_F16` | F16 | F32 | F16 GEMM, F32-accurate accumulate |
| 65 | `V_WMMA_F32_16X16X16_BF16` | BF16 | F32 | BF16 weights, F32 accumulate |
| 66 | `V_WMMA_F16_16X16X16_F16` | F16 | F16 | F16 GEMM, half the accumulator VGPRs |
| 67 | `V_WMMA_BF16_16X16X16_BF16` | BF16 | BF16 | BF16 GEMM, reduced accumulator footprint |
| 68 | `V_WMMA_I32_16X16X16_IU8` | IU8 | I32 | INT8 quantized GEMM |
| 69 | `V_WMMA_I32_16X16X16_IU4` | IU4 | I32 | INT4 quantized GEMM |

(Table 33, line 2945; opcode bodies at lines 16996–17101.) **There is no FP8 matrix type, and no SWMMAC / 2:4-structured-sparsity instruction.** The token "SWMMAC" appears only as a generic placeholder in the scheduling table ("In the table below 'WMMA' is either WMMA or SWMMAC", line 3052) — no such opcode exists on this architecture. Do not port CDNA/other-GPU FP8 or sparse-WMMA paths here; for precision below IU4 or for FP8, fall back to `DOT`/packed math or upcast.

**Choose F32 accumulation (opcodes 64/65) for anything with a long K reduction** — F16/BF16 accumulation (66/67) loses precision as the partial sum grows but halves accumulator VGPR count (see below), so reserve it for short-K or well-scaled tiles. WMMA float results are **round-to-nearest-even only** and raise **no ALU exceptions** (lines 2937, 2958) — deterministic across precisions, but you cannot lean on WMMA for overflow/NaN detection; do numerical-stability guards in a separate VALU/epilogue pass.

### Operand placement: A/B are VGPR-only and lane-replicated

- **A and B (SRC0/SRC1) must be VGPRs.** Unlike normal VOP3P sources they cannot be SGPR, VCC, M0, EXEC, or a constant (line 2700). Only **C (SRC2) may be an inline constant** — and for F16/BF16 that inline value is auto-replicated into both 16-bit halves of the DWORD (line 2960). Start each K-accumulation chain with an inline-0 (or inline-bias) C on the first WMMA to zero-init the accumulator without spending a VGPR tile or a clear pass.
- **Lane replication is mandatory.** WMMA reads A/B from mirrored lane groups: lanes 0–15 must be byte-identical in lanes 16–31 for wave32, and additionally in 32–47 and 48–63 for wave64 (line 2954). Skipping replication silently corrupts every matmul. The replication is **free in register terms** — total A/B VGPRs are identical for wave32 and wave64 — so pick wave size for occupancy/VMEM reasons, not WMMA register cost. Emit the broadcast (permute or LDS layout) as part of the tile loader before the first WMMA of the K-loop, and keep the resident A/B tile reused across the accumulation chain.

### VGPR layout: A is column-major, B/C/D are row-major

The register-to-element mapping is fixed by the hardware and asymmetric (diagram at line 3009): **A is stored column-major (transposed from the natural view), while B, C, and D are row-major.** For 16-bit types one VGPR holds two rows/columns (low and high halves). Design the LDS→VGPR staging so A lands transposed and B/C/D in natural row order; a wrong assumption yields a transposed or garbage product. Use the AMD Matrix Instruction Calculator to verify element→register mappings.

**Accumulator (C/D) footprint** (line 3040): the 16×16 accumulator is unpacked, one element per lane. Wave32 packs 2 rows per VGPR → **8 VGPRs** (V0–V7); wave64 packs 4 rows per VGPR → **4 VGPRs**. For register-bound WMMA GEMM, wave64 halves accumulator pressure and can buy an extra resident wave.

**A/B footprint scales with input precision** (packed A/B format): each lane holds 2×16-bit, 4×8-bit, or 8×4-bit elements, so IU8 A/B is half the bits and IU4 a quarter of F16/BF16. The win is entirely on the A/B side — the I32 accumulator stays full unpacked width regardless of input precision. Exploit the denser IU8/IU4 operands to grow the M/N tile or raise occupancy in quantized inference.

For 16-bit-**output** WMMA (opcodes 66/67), **OPSEL[2]** selects whether C is read from and D written to the upper or lower 16 bits of each accumulator VGPR (line 2706). OPSEL[0]/[1] are unused for WMMA. Alias two paired accumulators and distinguish them with OPSEL[2] to pack two F16/BF16 output tiles into one VGPR set when they need not be simultaneously live.

### Free modifiers via the NEG field

The VOP3P NEG bits are overloaded on WMMA (line 2939):

- **Float forms (64–67):** `NEG[0]`/`NEG_HI[0]` negate matrix A's low/high-16 operand, `NEG[1]`/`NEG_HI[1]` do the same for B, and `{NEG_HI[2], NEG[2]}` apply `{ABS, NEG}` to the C/SRC2 accumulator. Fold a subtraction (`D = C − A·B` by negating A or B) or an accumulator sign flip / absolute value into the matmul instead of a separate VALU op. Note the CLAMP bit is **ignored** on WMMA — saturate on a following conversion op, not on the matmul.
- **Integer forms (68/69):** `NEG[1:0]` is repurposed as a **per-operand signedness flag** — `NEG[0]=1` marks A signed, `NEG[1]=1` marks B signed (0 = unsigned), independently. This lets a single instruction multiply signed-int8 weights by unsigned/asymmetric int8 activations with no sign-extension pass. `NEG[2]` and `NEG_HI[2:0]` must be zero; the accumulator is always signed I32.

### EXEC is ignored — pad edge tiles, don't mask

Each WMMA internally forces `EXEC = all-ones` for the matrix evaluation and restores it afterward (pseudocode, line 17001):

```
saved_exec = EXEC;
EXEC = 64'B(-1);
eval "D0.f32(16x16) = S0.f16(16x16) * S1.f16(16x16) + S2.f32(16x16)";
EXEC = saved_exec
```

All 32/64 lanes participate regardless of your mask, and WMMA is **never EXEC-skipped** the way ordinary VALU is — masking off lanes does not reduce its cost. For sequence lengths not a multiple of 16, causal masking, or ragged tail tiles, **zero-pad the staged A/B/C operands** rather than trying to predicate with EXEC; apply causal/padding masks to the scores in a separate VALU pass. Ensure inactive/garbage lanes hold benign (zero) values so they cannot pollute active output elements.

### Accumulation chain scheduling

Structure the K-loop as in-place accumulation: the **same VGPR set is both matrix C (input) and matrix D (output)** every step (`D += A·B`). This is the fast path and it sidesteps the correctness hazard below. Two rules from the WMMA scheduling table (line 3050):

- **Correctness (D→A/B hazard):** if the first WMMA's matrix-D **is or overlaps** the next WMMA's matrix A or B, you **must** insert one `V_NOP` or one independent VALU instruction between them — the multi-cycle D writeback would otherwise be read stale. This hits chained matmuls like attention's `P = Q·Kᵀ` feeding `O = P·V`, or fused layer stacks. Fill the mandatory bubble with useful work (e.g. the next tile's dequant or replication), or interleave two independent accumulator chains so each supplies the other's filler and hides WMMA latency. Pure C=D accumulation does **not** trigger this — A/B may overlap C as long as C is distinct from D.
- **Stall avoidance (D→C reuse):** when the next WMMA reuses the prior D as its C accumulator (the normal case), keep **every WMMA in the chain the same type** and avoid an IMOD on SRC2 of the follower — mixing WMMA types or modifying SRC2 breaks the accumulator forwarding path and serializes. Partial (non-identical) VGPR overlap of D-as-C may also stall, so keep C exactly identical to the prior D or fully disjoint. Do type conversions (e.g. F16→F32) only at the epilogue, never mid-chain. Also, a plain VALU op that reads a WMMA's D immediately after may be stalled — defer bias/activation epilogue consumers by a few instructions or interleave the next tile's work.

### When to skip WMMA entirely

WMMA is built on top of the same `DOT` datapath but imposes the rigid 16×16×16 tile and the lane-replication contract (line 2954). For **decode-phase GEMV, batch-1 attention, tall-skinny, or non-16-aligned shapes**, the replication and tile-quantization overhead is wasted. Drop to the standalone dot instructions — `V_DOT2_F32_F16`/`_BF16` (F16/BF16, F32 accumulate), `V_DOT4_I32_IU8`, `V_DOT8_I32_IU4` (optionally dual-issued as `V_DUAL_DOT2ACC_F32_{F16,BF16}` in wave32 VOPD) — which reach the same MAC units at single-lane, arbitrary-shape granularity without the layout constraint. Reserve WMMA for large, 16-aligned tiles where the replication cost amortizes over a long K reduction.

**Throughput note:** the manual documents the shape, layout, and scheduling constraints but does not give exact per-instruction WMMA cycle counts or peak MAC rates; the prose only states WMMA "work[s] over multiple cycles" and directs users to the AMD Matrix Instruction Calculator for computational-throughput and register-usage figures (line 2954). Do not assume a specific FLOP/cycle number beyond that.

## Quantized & mixed-precision dot products

The `V_DOT` family is RDNA 3.5's **vector-ALU** path for low-precision inner products: one instruction reads two 32-bit VGPRs holding 2, 4, or 8 packed sub-word elements each, computes all the products, horizontally sums them, and adds a wider running accumulator supplied as the third operand — all in a single VALU issue slot. It is the right tool for exactly the shapes WMMA is bad at: decode-phase GEMV, batch-1 attention, ragged/non-16-aligned tiles, K-reduction tails, and cross-lane reductions.

RDNA 3.5 (gfx1150/gfx1151) has **no FP8 anywhere** — not in WMMA and not in the DOT family. F16/BF16 (2-wide), IU8 (4-wide), and IU4 (8-wide) are the complete menu of sub-FP32 dot formats.

### The variant table

| Instruction | Encoding (opcode) | Input format | # products | Accumulator (source) | Result |
|---|---|---|---|---|---|
| `V_DOT2_F32_F16` | VOP3P (19) | 2 × F16 | 2 | F32, from `S2` | F32 |
| `V_DOT2_F32_BF16` | VOP3P (26) | 2 × BF16 | 2 | F32, from `S2` | F32 |
| `V_DOT4_I32_IU8` | VOP3P (22) | 4 × IU8 (per-operand signed/unsigned) | 4 | I32 (signed domain), from `S2` | I32 |
| `V_DOT4_U32_U8` | VOP3P (23) | 4 × U8 | 4 | U32, from `S2` | U32 |
| `V_DOT8_I32_IU4` | VOP3P (24) | 8 × IU4 (per-operand signed/unsigned) | 8 | I32 (signed domain), from `S2` | I32 |
| `V_DOT8_U32_U4` | VOP3P (25) | 8 × U4 | 8 | U32, from `S2` | U32 |
| `V_DOT2_F16_F16` | VOP3 (614) | 2 × F16 | 2 | **F16**, from `S2` | F16 |
| `V_DOT2_BF16_BF16` | VOP3 (615) | 2 × BF16 | 2 | **BF16**, from `S2` | BF16 |
| `V_DOT2ACC_F32_F16` | VOP2 (2) / VOP3SD | 2 × F16 | 2 | F32, **in the destination** | F32 |
| `V_DUAL_DOT2ACC_F32_F16` | VOPD X-op 12 / Y-op 12 | 2 × F16 | 2 | F32, **in the destination** | F32 |
| `V_DUAL_DOT2ACC_F32_BF16` | VOPD X-op 13 / Y-op 13 | 2 × BF16 | 2 | F32, **in the destination** | F32 |

Opcode numbers: VOP3P Table 89 (lines 6832–6837), VOP3 Table (lines 6651–6652), VOP2 Table 78 (line 6066), VOPD X/Y Tables 91–92 (lines 6923–6924, 6932–6933). `V_DOT2ACC_F32_F16` is also one of the few opcodes that uses the VOP3SD encoding (line 2239).

**There is no standalone `V_DOT2ACC_F32_BF16`.** It appears in the VOPD X and Y opcode tables (`V_DUAL_DOT2ACC_F32_BF16`, lines 6924/6933), in the VOPD SRC2 rule (line 2774), and in the inline-constant table (line 2747) — but not in the VOP2 or VOP3 opcode maps. If you want a destination-accumulating BF16 dot outside a VOPD packet, you must use `V_DOT2_F32_BF16` with `S2` = the accumulator VGPR and `D0` = the same register.

---

### 1. One issue slot, 2 / 4 / 8 MACs, and the accumulate is free

Every DOT reads the running sum from `S2` and writes the updated sum to `D0` inside the same instruction — there is no separate add. The exact semantics (`V_DOT2_F32_F16`, line 16804; header at 16801):

```
tmp = S2.f32;
tmp += f16_to_f32(S0[15 : 0].f16)  * f16_to_f32(S1[15 : 0].f16);
tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
D0.f32 = tmp
```

`V_DOT2_F32_BF16` is identical with `bf16_to_f32` (line 16914; header 16911). Note the up-conversion: the F16/BF16 inputs are widened to F32 **before** the multiply, and both the multiply and the accumulate happen in the F32 domain.

**How to use.** Pack the K dimension 2/4/8 elements per 32-bit VGPR, hold the partial sum in one accumulator VGPR, and issue one DOT per group of K instead of a chain of `V_FMAC`. Element order within the DWORD only has to be *consistent between the two operands* — the instruction is a sum, so bit-field *i* of `S0` always pairs with bit-field *i* of `S1`. That means you can match an unusual weight packing order (e.g. GPTQ's interleaved nibble layout) by permuting the activation packing instead of repacking the weights.

**LLM kernels this accelerates:** the K-loop of every GEMV in decode (`o_proj`, `down_proj`, `lm_head`), attention QKᵀ and PV when the tile is too small or too ragged for WMMA, and any dot-shaped reduction (cosine similarity in routing, per-head norms).

---

### 2. INT8 GEMM/GEMV: `V_DOT4_I32_IU8` and `V_DOT4_U32_U8` — 4 MACs per issue, no feature gate

`V_DOT4_I32_IU8` (line 16816) computes four INT8 products into a signed 32-bit accumulator. The full pseudocode makes the signedness handling explicit:

```
declare A : 32'I[4];
declare B : 32'I[4];
for i in 0 : 3 do
    A8 = S0[i * 8 + 7 : i * 8];
    B8 = S1[i * 8 + 7 : i * 8];
    A[i] = NEG[0].u1 ? 32'I(signext(A8.i8)) : 32'I(32'U(A8.u8));
    B[i] = NEG[1].u1 ? 32'I(signext(B8.i8)) : 32'I(32'U(B8.u8))
endfor;
C = S2.i32;
// Signed multiplier/adder. Extend unsigned inputs with leading 0.
tmp = C.i32;
tmp += A[0] * B[0]; tmp += A[1] * B[1];
tmp += A[2] * B[2]; tmp += A[3] * B[3];
D0.i32 = tmp
```

Each operand is widened to a full 32-bit signed value, so **there is no intermediate 8-bit or 16-bit overflow** — a DOT4 of four `±127 × ±127` products cannot wrap. Accumulation is plain 32-bit wrapping arithmetic (see optimization 11 for saturation). The prose description says "unsigned 8-bit inputs ... in the signed 32-bit integer domain" — read the pseudocode, not the prose: the inputs are signed or unsigned per the NEG bits.

`V_DOT4_U32_U8` (line 16847) is the pure-unsigned form, `u8_to_u32` on every element, U32 accumulator, no NEG interpretation.

**Both integer DOT4 opcodes carry an explicit availability note:** "This opcode does not depend on the inference or deep learning features being enabled." (lines 16842 and 16860). Emit them unconditionally in the INT8 path — no runtime feature probe, no mode-set instruction, no fallback branch needed.

**How to use.** Pack four consecutive K-elements per DWORD: `S0[7:0]=k`, `S0[15:8]=k+1`, `S0[23:16]=k+2`, `S0[31:24]=k+3`. A single `global_load_dword` of an int8 tensor is then exactly one DOT4 operand — no unpack, no `V_CVT`, no `V_LSHRREV`. Chain `S2 = D0` across the K-loop.

---

### 3. INT4 weight-only inference: `V_DOT8_I32_IU4` — 8 MACs per issue, the densest non-WMMA MAC on the chip

`V_DOT8_I32_IU4` (line 16865; header 16862) is structurally the same as DOT4 but over eight 4-bit fields:

```
for i in 0 : 7 do
    A4 = S0[i * 4 + 3 : i * 4];
    B4 = S1[i * 4 + 3 : i * 4];
    A[i] = NEG[0].u1 ? 32'I(signext(A4.i4)) : 32'I(32'U(A4.u4));
    B[i] = NEG[1].u1 ? 32'I(signext(B4.i4)) : 32'I(32'U(B4.u4))
endfor;
tmp = S2.i32;
tmp += A[0] * B[0]; ... ; tmp += A[7] * B[7];
D0.i32 = tmp
```

`V_DOT8_U32_U4` (line 16896; header 16893) is the unsigned twin (`u4_to_u32`, U32 accumulator). Note that unlike DOT4, **the DOT8 opcodes carry no "does not depend on inference features" note** — the manual states that only for `V_DOT4_I32_IU8` and `V_DOT4_U32_U8`.

**How to use.** Nibble *i* of the DWORD sits at bits `[4i+3 : 4i]`, so a standard 8-values-per-uint32 4-bit weight block is directly a DOT8 operand. This is the W4A4 path; for W4A16/W4A8 (4-bit weights, wider activations) you still have to dequantize, so DOT8 only applies when both sides are 4-bit. Eight products per issue slot is 4× the MAC density of `V_DOT2_F32_F16` and 8× a scalar FMA — for a memory-bound decode GEMV this is enough headroom that the kernel stays firmly VMEM-limited.

---

### 4. Free per-operand signedness: NEG is repurposed on the IU dots (asymmetric quantization for zero instructions)

On the `...IU...` opcodes the VOP3P `NEG` field stops meaning "negate" (line 2724):

> These instructions use the NEG[1:0] bits to indicate signed (0=unsigned, 1=signed) per input source instead of meaning "negate". NEG[2] should be set to zero (behavior is undefined). NEG_HI must be zero.

The encoding field description agrees (line 2704): "For `DOT...IU...` and `WMMA...IU...` NEG[1:0] = signed(1)/unsigned(0) for src0 and src1, and Neg[2] behavior is undefined." `NEG_HI` is likewise undefined for these ops (line 2705).

**Why this matters for LLM inference.** Asymmetric quantization is the norm: activations are uint8 after a zero-point shift while weights are int8 (or the reverse for weight-only schemes). All four sign combinations — S×S, S×U, U×S, U×U — cost the same single instruction. Without this you would need explicit sign-extension or zero-point-correction MACs in the inner loop.

**How to use.** In assembly, set `neg_lo:[1,0]` (or the equivalent bit pattern for your assembler) to mark src0 signed and src1 unsigned; choose *which* tensor you load into src0 vs src1 to match your scheme. Keep `NEG[2] = 0` and `NEG_HI = 0`. Applies identically to `V_DOT4_I32_IU8` (line 16818) and `V_DOT8_I32_IU4` (line 16867).

---

### 5. Accumulate in F32, not F16: pick `_F32_F16` over `_F16_F16`

There are two FP16 dots and two BF16 dots, and they differ *only* in accumulator width:

| | Accumulator | Result | Line |
|---|---|---|---|
| `V_DOT2_F32_F16` | F32 | F32 | 16804 |
| `V_DOT2_F16_F16` | F16 | F16 | 20509 |
| `V_DOT2_F32_BF16` | F32 | F32 | 16914 |
| `V_DOT2_BF16_BF16` | BF16 | BF16 | 20525 |

`V_DOT2_F16_F16` keeps the whole chain in the 16-bit domain (line 20509):

```
tmp = S2.f16;
tmp += S0[15 : 0].f16  * S1[15 : 0].f16;
tmp += S0[31 : 16].f16 * S1[31 : 16].f16;
D0.f16 = tmp
```

**Default the K-reduction to `V_DOT2_F32_F16` / `V_DOT2_F32_BF16`.** Both do the same two products per issue; the F32 form gives you a full-precision partial sum for free, which is what long-K GEMM and attention-logit accumulation require. Reserve `V_DOT2_F16_F16` / `V_DOT2_BF16_BF16` for cases where the output must land back in packed 16-bit storage and K is short — their real advantage is that they halve accumulator VGPR footprint and can pack two accumulators per register (optimization 9).

**Rounding and denormals differ by type** (line 2467):

> DOT2_F16_F16 and DOT2_BF16_BF16 support round-to-nearest-even rounding. DOT2_F16_F16 supports denorms, and DOT2_BF16_BF16 disables all denorms.

BF16 dot flushes denormals unconditionally, so very small attention weights or normalization residuals round to zero on the BF16 path but survive on the F16 path. If you are chasing numerical parity between an F16 and a BF16 kernel variant, this is the first place to look.

---

### 6. `V_DOT2ACC_F32_F16`: destination-accumulate in a compact VOP2 encoding

`V_DOT2ACC_F32_F16` (VOP2 opcode 2, line 11405) folds the accumulator into the destination:

```
tmp = D0.f32;
tmp += f16_to_f32(S0[15 : 0].f16)  * f16_to_f32(S1[15 : 0].f16);
tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
D0.f32 = tmp
```

Semantically identical to `V_DOT2_F32_F16` with `S2 == D0`, but it encodes as a **32-bit VOP2 instruction** instead of a 64-bit VOP3P one. In a tight, unrolled K-loop that is a real instruction-cache and fetch-bandwidth saving; it is also the form that maps onto the dual-issue opcode (optimization 7). Use `V_DOT2ACC_F32_F16` whenever the accumulator and destination are the same register — which, in a reduction loop, is always.

---

### 7. Dual-issue: `V_DUAL_DOT2ACC_F32_F16` / `_BF16` occupy **both** VOPD slots

`V_DUAL_DOT2ACC_F32_F16` (opcode 12) and `V_DUAL_DOT2ACC_F32_BF16` (opcode 13) appear in the VOPD **X-opcode** table (lines 6923–6924) *and* the VOPD **Y-opcode** table (lines 6932–6933). A single VOPD packet can therefore carry two independent dot-accumulates: **four F16/BF16 MACs per lane per issued instruction** on the vector path, with no WMMA setup and no lane replication.

Semantics (line 17190 for the X-slot definition, 17262 for the Y-slot): "Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and accumulate the resulting single-precision float value into the destination vector register. **The initial value in D is used as S2.**" So pre-load your accumulators into the destination registers.

These are the **only** DOT opcodes that dual-issue. `V_DOT4_*`, `V_DOT8_*`, `V_DOT2_F32_*`, `V_DOT2_F16_F16`, and `V_DOT2_BF16_BF16` have no `V_DUAL_` form — **there is no dual-issue path for the INT8/INT4 dots at all.**

**Hard rules you must satisfy** (§7.6, lines 2751–2781; "These are hard rules - the instruction does not function if these rules are broken", line 2775):

- **Wave32 only** (line 2781). A wave64 kernel forfeits VOPD entirely.
- **No DPP in a VOPD packet** (line 2780); the DPP table also lists `VOPD | ALL | NO DPP` (line 2853).
- The two operations must be **independent** (line 2779). Reads and writes in the same packet do not race — a read gets the old value (line 2758).
- **Bank-disjoint sources.** There are 4 VGPR banks indexed by `SRC[1:0]`, each with 3 read ports (one each for SRC0/SRC1/SRC2), and a bank cannot serve two SRC0 reads at once (lines 2768–2770). So `SRCX0` and `SRCY0` must be in different banks, and `VSRCX1` and `VSRCY1` must be in different banks (lines 2771–2772).
- **SRC2 parity.** This one bites DOT2ACC specifically (line 2774): "If both operations use the SRC2 input, then one SRC2 input must be even and the other SRC2 input must be odd. The following operations use SRC2: FMAMK_F32 (second input operand); DOT2ACC_F32_F16, DOT2ACC_F32_BF16, FMAC_F32 (destination operand)." DOT2ACC reads its accumulator through the SRC2 port, so **two dual DOT2ACCs require one even and one odd accumulator VGPR.**
- **Destination parity** (line 2778): one dest VGPR even, the other odd. The encoding enforces this — `VDSTY` is stored without its LSB, and "LSB is the opposite of VDSTX[0]" (line 6912).
- At most 1 SGPR per op, at most one literal (or a shared one), ≤2 VGPRs per op (lines 2762–2764, 2777).

**How to use.** In an FP16/BF16 GEMV or small-N GEMM, advance two independent output accumulators per VOPD packet — e.g. two output columns, or two rows of a 2×N micro-tile. Allocate the accumulator pair at `v[2n]` / `v[2n+1]` to satisfy both the SRC2 and destination parity rules in one stroke, and stagger the A/B operand registers across banks (`v & 3`). A register allocator that is unaware of the parity rule will silently produce a non-functioning kernel, not a slow one.

---

### 8. DPP works on the float DOT2s and nowhere else in the family

The DPP capability table (Table 30, lines 2830–2855) splits the family cleanly:

| Opcodes | DPP | Line |
|---|---|---|
| `V_DOT2_F32_F16`, `V_DOT2_F32_BF16` (with `V_FMA_MIX_*`, "ALL Others") | **Allow DPP** | 2850 |
| `V_DOT4_I32_IU8`, `V_DOT4_U32_U8`, `V_DOT8_I32_IU4`, `V_DOT8_U32_U4` | **NO DPP** | 2844–2847 |
| `V_PK_*`, WMMA | NO DPP | 2848–2849 |
| VOPD (all `V_DUAL_*`) | NO DPP | 2853 |

DPP costs "an extra cycle of delay to execute" (line 2828).

**Why this matters.** Softmax row sums, layernorm/RMSNorm partial sums, and attention score reductions all need cross-lane data movement. On the F16/BF16 path you can fold a DPP16 row-shift or DPP8 swizzle directly into the SRC0 of `V_DOT2_F32_F16` / `V_DOT2_F32_BF16`, fusing the lane fetch into an FP32-accumulating MAC and saving a standalone `V_PERMLANE`/`V_MOV` per reduction step. This is also the reason to prefer `V_DOT2_F32_*` over `V_PK_ADD_F16` in reduction trees: packed math cannot take DPP at all, and the F32 accumulation is the numerically better choice anyway.

**On the quantized path you get none of this.** INT8/INT4 reductions must route cross-lane data through an explicit `V_PERMLANE*`, `DS_BPERMUTE`, or LDS round-trip *before* the DOT. Budget for that when you plan an INT8 GEMV that reduces across lanes rather than within a lane.

(The VOP2 rows of Table 30 list only 64-bit opcodes and FMAMK/FMAAK as NO DPP, with "All Others: Allow DPP" at line 2843; `V_DOT2ACC_F32_F16` is VOP2 opcode 2 and is not in the exclusion list, so by that table it allows DPP — but the manual never names it explicitly, so verify on hardware before relying on it.)

---

### 9. OPSEL packs two 16-bit accumulators into one VGPR (`V_DOT2_F16_F16` / `V_DOT2_BF16_BF16`)

Both VOP3 packed dots carry the same note (lines 20520 and 20536):

> OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are ignored.

The general OPSEL field description adds the constraint (line 2259): "DOT2_F16 and _BF16: src0 and src1 must have OPSEL[1:0] = 0" — the two multiply inputs always consume both halves of their DWORD, which is the whole point of a packed dot. Both opcodes are on the OPSEL-eligible list (Table 27, line 2403).

**How to use.** Run two independent 16-bit accumulator chains inside a single VGPR: chain A uses `OPSEL[2]=0, OPSEL[3]=0`, chain B uses `OPSEL[2]=1, OPSEL[3]=1`. This halves accumulator register pressure in an FP16 GEMM micro-tile — directly buying occupancy — and it eliminates the `V_PACK_B32_F16` / shift-and-or you would otherwise need to reassemble packed 16-bit output. The VOP3P `_F32_*` dots do not offer this (their accumulator and result are full 32-bit), so this is the one concrete reason to reach for the F16-domain dots.

---

### 10. DOT ops are the *only* sub-16-bit packed math that accepts inline constants

Inline constants normally only populate the low 16 bits of a source, which makes them useless for packed math below 16 bits — with a documented exception (line 2736): "Any packed math instructions that use data sizes less than 16 bits do not work with inline constants, other than the DOT instructions below:" (table, lines 2738–2747):

| Opcode | Inline behavior |
|---|---|
| `DOT4_I32_IU8`, `DOT8_I32_IU4`, `DOT4_U32_U8`, `DOT8_U32_U4` | use 32-bit inline src0/1 (ignore OPSEL) |
| `DOT2_F32_F16` | use FP32 inline, supports OPSEL |
| `DOT2_F32_BF16` | upper16(FP32) / same as replicate (src0/1), ignore OPSEL |
| `DOT2ACC_F32_F16`, `DOT2ACC_F32_BF16` | duplicate lo to hi, ignore OPSEL |

Two related rules: BF16 inline constants are taken as the **upper 16 bits of a 32-bit float constant**, which matches BF16's definition as truncated FP32 (line 2732); and for `WMMA_F16_F16_16x16x16` or **VOPD `DOT2_F32_F16`** the hardware automatically selects the low 16 bits of the constant (line 2734). For the VOP3 packed dots there is a separate rule (line 2329): "For these 2 instructions, the inline constant for sources 0 and 1 replicate the inline constant value into bits[31:16]. For source2, the OPSEL bit is used to control replication or not (gets zero if not replicating low bits)."

**How to use.** When one DOT operand is a compile-time constant — a broadcast weight, an all-ones vector used to turn a DOT into a horizontal sum of the other operand, a fixed quantization pattern — encode it as an inline constant instead of materializing it in a VGPR. That is one fewer live register in the inner loop and no literal dword in the instruction stream. The all-ones trick is particularly useful in quantized kernels: `V_DOT4_U32_U8` with an inline `0x01010101` sums the four bytes of the other operand, which is exactly the activation-sum term you need for zero-point correction in asymmetric INT8 GEMM.

---

### 11. Saturation and output modifiers: float DOTs silently ignore both

- **CLAMP is ignored on float DOTs.** The clamp-exclusion table (lines 2436–2444) lists "Float DOT instructions" alongside `V_PERMLANE*`, `V_PERM_B32`, WMMA ops, and `V_ADD3`. You cannot get a free `[0.0, 1.0]` saturate out of `V_DOT2_F32_F16` or its siblings.
- **OMOD is unsupported on the VOP3 packed dots.** Lines 2421–2425: "Output Modifiers are not supported for: V_PERMLANE, DOT2_F16_F16, DOT2_BF16_BF16." The free ×0.5 / ×2 / ×4 output scale (line 2255) does not apply. Note the manual names only these two opcodes here; the VOP3P dots have no OMOD field at all in their encoding (lines 2696–2708).
- **Integer DOT accumulation as written is non-saturating.** The pseudocode for all four integer dots is plain `tmp += A[i] * B[i]` with no clamp step, so a long K-chain that overflows I32/U32 wraps. The VOP3P encoding does carry a `CLMP` bit ("Signed integer arithmetic: clamp result to [min_int, +max_int]; Unsigned integer arithmetic: clamp result to [0, +max_uint]", line 2708), and the clamp-exclusion table names only *float* DOT instructions — so by exclusion the integer dots should honor it. **The manual does not state this affirmatively;** verify on hardware before depending on saturating INT8/INT4 accumulation.

**Practical consequence.** Do not plan to fold the attention `1/√d` scale into a dot via OMOD, or to saturate a quantized accumulator via CLAMP on a float dot. Apply the scale to one operand before the dot (fold `1/√d` into the Q projection weights, which is free at load time), and emit an explicit `V_MED3_*` / min-max for any saturation you need.

---

### 12. Free per-half negation on the float DOT2s (fused subtract inside the reduction)

On float DOT opcodes the VOP3P `NEG` field negates the **lower**-16-bit operand and `NEG_HI` negates the **upper**-16-bit operand, per source — bit 0 = src0, bit 1 = src1, bit 2 = src2 (lines 2704–2705). Because the two packed lanes have independent negate bits, either or both products in the dot can be sign-flipped at no instruction cost, and the F32 accumulator in `S2` can be negated too.

**How to use.** `a0*b0 - a1*b1` is one `V_DOT2_F32_F16` with `NEG_HI[0]` set, not a dot plus a negate. Useful for residual/difference reductions (e.g. computing a sum of squared differences, or a `x - mean` variance term staged as a dot) and for sign-flipping one factor in an attention or normalization expression. Remember this is exactly the field that means signed/unsigned on the IU dots — the same bits, opposite meaning; never carry a NEG pattern across from a float dot to an integer one.

---

### 13. When to use DOT instead of WMMA

WMMA is built on the same arithmetic: "These instructions work over multiple cycles to compute the result matrix and **internally use the DOT instructions**. In order to achieve this performance, the user must arrange the data such that: A and B matrices: lanes 0-15 data are replicated into lanes 16-31 (for wave64: also into lanes 32-47 and 48-63)." (lines 2954–2956; the row-column dot products are "distributed across the vector ALU", line 16996).

So WMMA is the DOT hardware plus a fixed 16×16×16 tile plus a mandatory lane-replication step. That trade is excellent when you can amortize it and terrible when you cannot:

| Use WMMA | Use the DOT family |
|---|---|
| Large, 16-aligned GEMM tiles (prefill, batched MLP) | GEMV / batch-1 decode, M=1 shapes |
| K ≥ 16 with a resident A/B tile reused across the chain | K-loop tails, non-multiple-of-16 sequence lengths |
| You control the LDS→VGPR staging and can replicate lanes | Ragged / irregular / gathered access (MoE expert routing) |
| Throughput-bound | Memory-bound decode where MAC density is already sufficient |
| — | Cross-lane reductions (DOT2 accepts DPP, WMMA does not) |
| — | Anything that must respect EXEC (WMMA forces EXEC = all-ones internally, line 17001) |

The DOT path also has no lane-replication cost, no tile-shape quantization waste, and — for `V_DUAL_DOT2ACC_F32_{F16,BF16}` — dual issue. A common structure is WMMA for the prefill/prompt GEMMs and a DOT2ACC or DOT4/DOT8 kernel for the decode step of the same model.

---

### Throughput: the manual gives none

**The RDNA 3.5 ISA manual publishes no cycle counts, instruction rates, or issue-rate table for the DOT family** — or for VALU instructions generally. The only rate field anywhere in the document is `DP_RATE` for double-precision units (line 1031). The "N MACs per instruction" figures above are *architectural* — they count products retired per issued instruction as defined by the pseudocode — and are not a claim about clocks per instruction or sustained FLOPs. Measure `V_DOT4_I32_IU8` / `V_DOT8_I32_IU4` / `V_DUAL_DOT2ACC` throughput on gfx1150/gfx1151 before building a performance model on them; do not assume a DOT8 retires in the same cycle count as a DOT2.


All citations check out. Writing the section.

## Packed low-precision math & free modifiers

RDNA 3.5's VALU natively packs **two 16-bit values (F16/BF16/I16) or four 8-bit values into a single 32-bit VGPR** and operates on both halves as if they were separate threads (line 2124). For the memory-bound and elementwise half of an LLM kernel — bias/residual add, activation, dequant scale, layernorm/softmax pre- and post-scaling — this is a straight 2× throughput lever over scalar 16-bit ops, and it pairs with a set of *free* input/output modifiers (sign flip, absolute value, output scale, saturation, half-select) that fold what would be separate instructions into the operand read/write path. The catch that trips up most kernels: the packed (VOP3P) encoding does **not** carry the same modifier menu as the scalar (VOP3) encoding, and several modifiers are silently ignored on exactly the ops you most want to attach them to.

### Packed 16-bit ALU (V_PK_*)

Packed math uses the VOP3P microcode format and computes the low-half and high-half results independently in one issue slot (line 2653). A packed add `V0 = V1 + V2` is two adds — low halves into `V0[15:0]`, high halves into `V0[31:16]`. Keep both operands of a packed op co-resident in one VGPR: the register file services 32 lanes of a `V0.L`+`V0.H` pair in a single cycle (line 2641), so there is no register-file penalty for the 2× — only the win.

| Pattern | Packed ops | Line | LLM use |
|---|---|---|---|
| F16 fused multiply-add | `V_PK_FMA_F16` | 16745 | GEMM epilogue / non-WMMA F16 MAC, activation polynomials (single rounding) |
| F16 accumulate-in-place | `V_PK_FMAC_F16` | 12107 | packed reduction / SwiGLU-style gate loops (dest is the addend) |
| F16 add / mul | `V_PK_ADD_F16`, `V_PK_MUL_F16` | 16757, 16768 | residual add, scale, elementwise gating |
| F16 min / max | `V_PK_MIN_F16`, `V_PK_MAX_F16` | 16779, 16790 | packed ReLU (max vs packed 0), clamp, online-softmax running max |
| I16 multiply-add (saturating) | `V_PK_MAD_I16`, `V_PK_MAD_U16` | 16596, 16690 | INT16 dequant/requant, zero-point + scale accumulation |
| I16 add/sub, min/max | `V_PK_ADD_U16`, `V_PK_SUB_U16`, `V_PK_MAX_I16`, `V_PK_MIN_I16` | 16701, 16707, 16670, 16637 | zero-point correction, saturating quant clamp |
| I16 packed shifts | `V_PK_LSHRREV_B16`, `V_PK_ASHRREV_I16` | 16648, 16659 | two-lane-at-a-time INT4/INT8 bit-unpack (packed shift **count** in the first operand) |

To *build* packed F16 operands from two separate values, `V_PACK_B32_F16` writes `S0` into the low half and `S1` into the high half in one op (`V_PACK_B32_F16` line 20841) — no shift/or sequence.

Two hard limits on the packed path:

- **No DPP.** Packed-math ops cannot carry a DPP cross-lane control DWORD (line 2828). Any lane shuffle for a reduction must be a separate op on non-packed data.
- **Inline constants only populate the low 16 bits** (line 2730). A packed op that uses an inline constant *requires* `OPSEL` to route the low-half constant into both lanes; without it the high lane silently gets a wrong (zero) operand. This is the single most common packed-math correctness trap when broadcasting a scalar (scale, epsilon, bias) across both halves. For BF16, supply a 32-bit FP32 inline constant and the BF16 operand takes its upper 16 bits automatically (line 2732).

### The two modifier menus: VOP3 vs VOP3P

The free modifiers live in the instruction encoding and are applied by hardware at the operand read (input) or result write (output) stage — zero extra instructions, zero extra cycles. **Which modifiers exist depends on the encoding**, and this asymmetry is deliberate:

| Modifier | Scalar VOP3 | Packed VOP3P | Effect |
|---|---|---|---|
| `NEG` (per-source sign flip, float only) | yes, `NEG[2:0]` (line 2256) | yes, `NEG[2:0]` = low-half (line 2704) | fold `-x` into the consuming op |
| `NEG_HI` (per-source, high half) | — | yes (line 2705) | high-lane sign flip |
| `ABS` (per-source `|x|`, float only) | yes, `ABS[2:0]`, applied **before** NEG (line 2257) | **no** (line 2655) | fold `|x|` / `-|x|` into the op |
| `OMOD` (×0.5, ×2, ×4 output scale, float only) | yes (line 2255) | **no** (line 2655) | free power-of-two output scale |
| `CLMP` (saturate) | yes (line 2258) | yes (`CLMP`, line 2708) | float→[0,1]; int→type min/max |
| `OPSEL` / `OPSEL_HI` (16-bit half select) | `OPSEL` (line 2259) | `OPSEL` + `OPSEL_HI` (line 2706) | pick hi/lo 16-bit half of each src and dest |

The takeaway for kernel authors: **packed VOP3P has only NEG/NEG_HI, OPSEL/OPSEL_HI, and CLMP — no ABS and no OMOD** (line 2655). If your packed-F16 activation kernel needs an absolute value or a ×2 / ×0.5 scale, that must be a separate op (or folded into NEG or into a constant beforehand). The full ABS + OMOD menu is only available on the scalar VOP3 form.

### Input modifiers: free NEG / ABS on FP sources

Any VOP3-encoded float instruction negates or absolute-values its sources for free via `ABS[i]`/`NEG[i]` (bit 0=src0, 1=src1, 2=src2), and ABS is applied *before* NEG so `-|x|` is one source with both bits set (lines 2256–2257). These modifiers also work on `V_MOV_B32`/`V_MOV_B16`, `V_MOVREL*_B32`, and `V_CNDMASK` — so a move that stages a value can simultaneously negate or magnitude it (`V_MOV_B32` line 12185; `V_MOV_B16` line 12535). They are **undefined for integer/bitwise ops, readlane/writelane, and permlane** — do not rely on them there.

For LLM kernels the negate is pervasive: softmax's `exp(x − max)` becomes a NEG on the addend rather than a separate subtract; layernorm mean subtraction, residual sign handling, and GELU/SiLU sign math all fold in. ABS drives abs-max quantization-scale search and L1/norm reductions.

On the packed side, `V_FMA_MIX_*` is the notable exception that recovers a free absolute value: for the MIX opcodes the `NEG_HI` field is **repurposed as an ABS modifier on all three inputs** (line 2705), and `NEG` negates them — useful for stabilization/normalization math that consumes F16 halves directly.

### Output modifiers: OMOD scale and CLAMP saturate — and where they vanish

`OMOD` multiplies a float result by 0.5, 2.0, or 4.0 for free (line 2255), letting a fixed power-of-two scale (attention averaging, `/2` pooling, some activation constants) ride the producing op. But it is fragile and **silently ignored** in several cases you are likely to hit (line 2419):

- integer and **packed-F16** results ignore OMOD entirely;
- OMOD is ignored when the **IEEE mode bit is set**, and when **output denormals are enabled**;
- explicitly unsupported on `DOT2_F16_F16` / `DOT2_BF16_BF16` (line 2421).

Because most GEMM epilogues use non-power-of-two scales and the packed/DOT/WMMA accumulators ignore it, OMOD's practical reach is narrow — verify the ISA actually folded it before relying on it, and note it is not IEEE-compatible (`−0` flushes to `+0`).

The `CLMP` bit gives a free saturate in the same op: float results clamp to `[0.0, 1.0]` (a free `[0,1]`/ReLU-clamp / probability clamp), signed integers clamp to `[min_int, max_int]`, unsigned to `[0, max_uint]` — exactly the saturating narrow a quantized INT8/INT4 output cast needs (line 2258). The critical gotcha: CLAMP is **ignored on WMMA ops and on float DOT instructions** (plus permlane, bitwise ops, `V_ADD3`, integer `V_CMP`, and more — line 2436). So a quantized matmul cannot saturate on the WMMA/DOT itself; apply CLMP on the subsequent conversion op (`V_CVT_PK_*`, packed/scalar requantize) instead.

### OPSEL / OPSEL_HI: free half-register selection

For 16-bit math, `OPSEL` selects whether each source and the destination reads/writes the **high or low 16 bits** of its 32-bit VGPR (`[0]`=src0, `[1]`=src1, `[2]`=src2, `[3]`=dest; 1=high, 0=low), addressing packed data in place without a shift/unpack (line 2259). In VOP3P, `OPSEL` steers the low-result lane and `OPSEL_HI` steers the high-result lane independently, so packed ops can cross low/high halves (swap, broadcast, rotate-half) for free (line 2706). Constraints: `OPSEL` must be zero for any non-16-bit operand, and must be zero for inline-constant sources (the value only exists in the low 16 bits — line 2259), which is why the packed inline-constant broadcast above needs the low-half-into-both-halves trick.

OPSEL also unlocks the larger register namespace: VOP1/2/C-encoded 16-bit ops can only reach 256 sixteen-bit VGPRs, while **VOP3/VOP3P/VINTERP with OPSEL address the full 512** (line 2631). Register-heavy F16/BF16 inner loops that spill under the 256 limit should be steered toward the VOP3-class encoding.

### Practical guidance

- Keep FP16/BF16 activation tiles packed two-per-VGPR (`V0.L`/`V0.H`) and emit `V_PK_*` for every elementwise pass; reach for `V_PK_FMA_F16` for fused MAC and `V_PK_MAX_F16` (vs a packed-zero) for ReLU.
- Fold sign/negation into the consuming op's `NEG`/`NEG_HI`, and — only on scalar VOP3 — abs into `ABS` and power-of-two scales into `OMOD`. Do not expect ABS or OMOD on packed ops.
- Use `CLMP` for `[0,1]` activation clamps and saturating integer requant on the **conversion/scalar** op, never on WMMA or float DOT (ignored there).
- When broadcasting an inline constant into a packed op, always set `OPSEL` to replicate the low-16 constant into both lanes; use a 32-bit FP32 inline for BF16.
- The manual does not give packed-op throughput figures beyond "two 16-bit operations per instruction / single-cycle read of the pair"; treat the win as ~2× issue-rate over scalar 16-bit, not a stated cycle count.

All citations check out. Writing the section.

## Precision, conversions & dequant

RDNA 3.5 lets you move between F32, F16, BF16, INT16 and packed INT8/INT4 almost entirely with fused, mode-controlled conversion ops — most of them packing two elements per instruction and folding rounding, saturation, sign-extension or a scale into the same op. Getting the conversion path and the MODE register right is the difference between a dequant loop that is one instruction per element and one that is three.

### Format facts that make conversions free

- **BF16 is literally the top 16 bits of FP32.** Sign + exp8 + mant7, so F32→BF16 is a truncate of the low 16 bits and BF16→F32 is a zero-pad. For `V_DOT2_BF16_BF16` / `V_WMMA_*_BF16` inline constants you supply a full 32-bit FP32 immediate and the BF16 operand takes its upper 16 bits automatically — no lookup, no separate convert (`rdna35_instruction_set_architecture.md` line 2732).
- **F16 is fully IEEE** (denormals, INF, NaN all supported), so masked/attention math can rely on F16 special values; BF16's wider exponent is why the BF16 dot path (below) can flush denormals safely.

### F32 → F16 narrowing: pick the rounding you want

| Op | Rounding | Packs 2? | Notes |
|---|---|---|---|
| `V_CVT_PK_RTZ_F16_F32` (line 11896) | round-toward-zero, **ignores MODE** | yes → one VGPR | Saves/forces/restores ROUND_MODE internally; matches D16-store truncation |
| `V_CVT_F16_F32` (line 12323) | honors FP_ROUND MODE bits | no | Free abs/neg input modifier; creates F16 denormals per denorm mode |
| D16 buffer store (line 3521) | **truncation for F32→F16** (RNE for other input formats) | writes packed 2×16 | Conversion done in the texture unit, no VALU cost |

`V_CVT_PK_RTZ_F16_F32` is the workhorse for writing F16 activations/KV-cache from F32 accumulators: two conversions and the packed 2×F16 layout WMMA consumes, in one op, without perturbing the global round mode of surrounding F32 math. Use `V_CVT_F16_F32` when you need RNE (round-nearest) and want to fold a sign flip/abs into the down-cast; note it biases differently from the truncating D16 store, so do not mix the two when bit-matching a reference.

Widening the other way, `V_CVT_F32_F16` (line 12336) is **exact (0 ULP)** and accepts F16 denormals — F16-store/F32-compute kernels (attention softmax, layernorm) upcast losslessly and need no compensation.

### INT8 / INT4 dequant: fused extract-and-convert

A packed uint8 tensor loads four values per 32-bit VGPR; `V_CVT_F32_UBYTE{0,1,2,3}` (lines 12424–12451; VOP3 forms at 17586) convert **one selected byte lane straight to F32** — the byte select is built into the opcode, so four instructions per DWORD dequant a uint8 quad with no `V_AND`/`V_LSHRREV` masking:

```
D0.f32 = u32_to_f32(S0[31:24].u32)   // V_CVT_F32_UBYTE3
```

These are unsigned-only; signed int8 needs a zero-point/sign correction after. For sign-extension at the memory boundary instead, the **D16 sub-word loads** (`BUFFER/GLOBAL/DS_LOAD_D16_I8` and `_HI` variants, e.g. lines 26642, 25960, 28763) load an 8-bit value, sign- or zero-extend it to 16 bits, and write only the low or high half of a VGPR while preserving the other — so two extended int8 values land packed in one register with no shift/OR.

For INT4, `V_CVT_OFF_F32_I4` (line 12367) maps a signed 4-bit nibble to F32 through a fixed hardware offset table (−0.5 … +0.4375 in 0.0625 steps) with zero memory traffic — usable directly when your codebook matches that grid or can be affine-rescaled from it, replacing an LDS lookup. For arbitrary sub-byte unpacking, `V_BFE_U32`/`V_BFE_I32` (line 19500) extract (and, for the signed form, sign-extend) an arbitrary offset/width bitfield in one op, and `V_PERM_B32` (line 19981) gathers any byte from a `{S0,S1}` pair with per-byte selects that can also sign-extend (selectors 8–11) or emit 0x00/0xFF (12/≥13) — ideal for reordering packed INT8 weights into WMMA byte order or widening signed int8 to int16 without a shift/mask/sign-extend chain.

### F → INT quantization output: saturation and rounding are free

Float→int conversions **saturate out-of-range values (including INF) and map NaN→0 in hardware**, so a naive quantizing cast needs no surrounding min/max clamp or NaN guard (`V_CVT_I32_F32`/`V_CVT_U32_F32`, lines 12293, 17458). The **CLAMP bit doubles as the INEXACT-exception enable** on these converts — leave `CLAMP=0` in bulk quantize loops to suppress exception generation while keeping the free saturation (lines 17458, 12468, 13175, 18343).

To choose rounding without an `S_SETREG` mode toggle, use the dedicated variants `V_CVT_NEAREST_I32_F32` (round-to-nearest-even, line 12341) and `V_CVT_FLOOR_I32_F32` (floor, line 12354); both ignore the current round mode.

Packing and saturation for the output stage:

| Op | Function | Line |
|---|---|---|
| `V_SAT_PK_U8_I16` | two i16 → saturate to [0,255], pack into a 16-bit word (two of these build a uint8×4 DWORD) | 13448 |
| `V_CVT_NORM_I16_F16` / `_U16_F16` | F16 → normalized int16, **fused round + saturate** (0.5 ULP) | 13469 |
| `V_CVT_PK_I16_F32` / `_U16_F32` | two F32 → packed int16 in one VGPR | 20752 / 20764 |
| `V_CVT_PK_NORM_I16_F32` / `_U16_F32` | two F32 → packed normalized int16, clamp folded in | 20966 |
| `V_CVT_PK_NORM_I16_F16` / `_U16_F16` | two F16 → packed normalized int16 | 20848 |
| `V_CVT_PK_U16_U32` / `V_CVT_PK_I16_I32` | narrow two int32 accumulators → packed int16 | 20990 |
| `V_PACK_B32_F16` | assemble two loose F16 into one packed 32-bit register | 20841 |

The `_PK_NORM_*` and `V_SAT_PK_U8_I16` forms fold the clamp that quantization otherwise needs into the same instruction and emit an already-packed word ready for a coalesced store — cutting a scale→round→clamp→convert→pack sequence to one or two ops.

Note: the CLAMP modifier is **ignored on WMMA and float DOT outputs** (line 2436), so saturate on the following `V_CVT_PK_*` / scalar op, never on the matmul itself.

### Power-of-two scaling and scale extraction

Dequant scales that are powers of two are exponent adjusts, not multiplies: `V_LDEXP_F32`/`V_LDEXP_F16` (lines 20875, 12097) compute `x * 2^n` exactly in one op. To derive a power-of-two scale from data, `V_FREXP_EXP_I32_F32`/`V_FREXP_MANT_F32` (line 13016; F16 forms at 13306) split a float into integer exponent and a `[0.5,1.0)` significand in single instructions, replacing bit-manipulation when computing per-block quantization scales.

### Denormal & rounding MODE — the speed/precision lever

Change FP behavior with the dedicated `S_ROUND_MODE` / `S_DENORM_MODE` (immediate) ops, **not** `S_SETREG(MODE)` — they "avoid the wait state penalty that would be imposed by S_SETREG" (lines 2454, 10851, 10858).

- **FP_DENORM** (MODE[7:4]) is two 2-bit fields, `{allow_output_denorms, allow_input_denorms}`: `0` flushes both (fast path), `3` allows both. Bits [5:4] cover F32; **bits [7:6] cover FP16 *and* F64 together** (line 932) — setting F16 flush also moves F64, so isolate F32 work that must keep denormals. Denorm mode affects VALU, LDS, and VMEM atomics. For typical F16/BF16 inference, mode 0 (flush) avoids the denormal slow path and matches expected numerics.
- **FP_ROUND** (MODE[3:0]): [1:0] is F32, [3:2] is the shared FP16/F64 round mode (line 931). Round mode affects VALU only, not LDS or memory conversions.
- **FP16_OVFL** (MODE bit 23): overflowed FP16 VALU results clamp to ±MAX_FP16 instead of becoming INF, while genuine INF/÷0 still yield INF (line 938) — free saturation for overflow-prone F16 softmax/exp without per-op min/max.
- **DX10_CLAMP** (bit 8): clamps NaN→0 (when the op's CLAMP bit is set) and suppresses all VALU exceptions (line 933) — folds NaN sanitization into an existing CLAMP.
- **IEEE** (bit 9): selects MIN/MAX semantics (`>=` vs `>` compare, sNaN handling); only matters for ±0 and flushed denormals, single-instruction either way (line 934).

Note also that `V_CVT_PK_RTZ_F16_F32` and the scalar `S_CVT_PK_RTZ_F16_F32` force round-toward-zero regardless of MODE and restore it, so they never need a mode change bracketing them.

### Precision-vs-speed notes

- **The two F16 dot paths differ in denormals:** `V_DOT2_F16_F16` supports denormals; `V_DOT2_BF16_BF16` **disables all denormals** (both RNE) (line 2467). BF16's wide exponent makes the flush safe; the F16-domain path preserves tiny attention weights that would otherwise round to zero.
- **WMMA float ops are round-to-nearest-even only and raise no ALU exceptions** (lines 2937, 2958). You cannot select another rounding mode for the matrix path, and cannot lean on WMMA exception status for overflow/NaN detection — do numerical-stability guards in a separate VALU/epilogue pass.
- **Accuracy ladder** for the primitives you convert *through*: int→F32 is correctly rounded (0.5 ULP) and int→F64 is exact (0 ULP), so dequant needs no defensive rounding. F16/F32 arithmetic and transcendentals are 1 ULP (F16 tighter at 0.51 ULP); F64 reciprocal/rsqrt/sqrt are only ~2²⁹ ULP *seeds* — never route conversion-adjacent reciprocals through raw F64.

Relevant source files: manual at `/home/sadara/.hipfire/src/docs/reference/rdna35-isa-markdown/rdna35_instruction_set_architecture.md`; per-instruction chunks under `/home/sadara/.hipfire/src/docs/reference/rdna35-isa-markdown/derived/instructions/`.

## Direct-to-LDS & coalesced memory movement

Every LLM kernel that touches the matrix engine is really a memory-movement kernel with a WMMA epilogue. A GEMM inner loop stages A/B tiles global→LDS; attention streams K and V blocks out of the KV cache; a memory-bound layernorm or dequant pass is nothing *but* movement. This section covers the path **into** LDS and the addressing rules that decide how many memory transactions a wave costs: the direct-to-LDS load family, wide vector loads, VGPR-free address generation, alignment, hardware bounds checking, and the buffer resource descriptor. (Bank-conflict layout of the LDS tile itself and the GLC/SLC/DLC cache-policy bits are covered in their own sections.)

---

### 1. RDNA 3.5 *does* have direct-to-LDS loads — but they are documented only in the XML

The manual's LDS overview states the path exists: *"When loading from memory, the data may be loaded into VGPRs first or for some types of loads it may be loaded directly into LDS from memory"* (line 4711). It then never names those instructions — the FLAT/GLOBAL/SCRATCH instruction table (line 4474 ff.), Table 104 MUBUF Opcodes (line 7341), Table 109 GLOBAL Opcodes (line 7590) and Table 110 SCRATCH Opcodes (line 7625) list only load-to-VGPR forms.

The gaps in those opcode tables are exactly where the XML puts the missing family. Table 104 jumps 44 → 51; Table 109 jumps 41 → 51; Table 110 stops at 37. The XML fills them:

| Instruction | Encoding | Opcode | Bytes/lane |
|---|---|---|---|
| `BUFFER_LOAD_LDS_U8` / `_I8` (XML) | MUBUF | 45 / 46 | 1 (zero/sign-extended to a DWORD in LDS) |
| `BUFFER_LOAD_LDS_U16` / `_I16` (XML) | MUBUF | 47 / 48 | 2 (zero/sign-extended) |
| `BUFFER_LOAD_LDS_B32` (XML) | MUBUF | 49 | 4 |
| `BUFFER_LOAD_LDS_FORMAT_X` (XML) | MUBUF | 50 | 4, with format conversion from the V# |
| `GLOBAL_LOAD_LDS_ADDTID_B32` (XML) | FLAT_GLOBAL | 42 | 4, **no address VGPR at all** |
| `GLOBAL_LOAD_LDS_U8` / `_I8` / `_U16` / `_I16` / `_B32` (XML) | FLAT_GLOBAL | 45–49 | 1 / 1 / 2 / 2 / 4 |
| `SCRATCH_LOAD_LDS_U8` / `_I8` / `_U16` / `_I16` / `_B32` (XML) | FLAT_SCRATCH | 45–49 | 1 / 1 / 2 / 2 / 4 |

**What the hardware does.** The XML operand lists for these opcodes carry **no `VDST`/`VDATA` operand** — unlike every other load, nothing is written to the register file. Instead each one takes an *implicit* 32-bit `M0` source (`OPR_SDST_M0`). The XML description is uniform: *"Untyped buffer load data, zero/sign extend and store in LDS destination."* `GLOBAL_LOAD_LDS_ADDTID_B32` (XML) is the extreme case — its only explicit operand is the 64-bit `SADDR` base; its description reads *"No VGPR address is supplied in this instruction. TID is added to the address"*, i.e. the global-side address is the `GT` mode `Saddr₆₄ + Ioff + TID*4` (line 4646) and the LDS side comes from `M0`.

**Why it matters for LLM kernels.** The conventional staging pipeline burns a VGPR *tile* as a bounce buffer: `GLOBAL_LOAD_B128` → (wait `vmcnt`) → `DS_STORE_B128`. On a WMMA GEMM those staging VGPRs compete directly with the accumulator tile, and accumulators are what set your tile size. Direct-to-LDS removes the bounce buffer and one whole instruction per element from the prologue:

- **GEMM tile staging** — the A/B prefetch for iteration *k+1* costs zero VGPRs, so the freed registers go to a larger C accumulator tile (better compute intensity) or higher occupancy.
- **KV-cache streaming** — decode-phase attention re-reads a growing K/V region every step and uses it once. Landing it straight in LDS avoids allocating a register staging tile whose only job is to be immediately spilled to LDS.
- **Quantized inference** — the `_U8`/`_I8` forms zero/sign-extend at the memory boundary, so INT8 weights arrive in LDS already widened, with no `V_BFE`/`V_CVT` unpack in the register file.

**How to use it, and the caveats.**

1. Set `M0` to the LDS destination before issuing. **The manual's M0 field table (Table 7, line 946) does not list an entry for buffer/global load-to-LDS** — it documents `M0` only for `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, LDS ADDTID, GDS, MOVREL, SMEM and SENDMSG. The XML proves `M0` is read; it does not define the field layout. Treat the exact `M0` encoding as unverified against this manual and confirm against your assembler / LLVM `gfx1150` definitions before hand-coding.
2. Likewise, **the manual gives no waitcnt accounting for these opcodes** — no statement of whether completion is tracked on `VMcnt`, `LGKMcnt`, or both. Do not guess; measure, or wait conservatively.
3. The manual's *"no LDS bandwidth is used by global instructions"* (line 4551) is written about the ordinary VGPR-destination GLOBAL forms. A `GLOBAL_LOAD_LDS_*` obviously must write the LDS array; the manual offers no bandwidth or port-conflict accounting for it. Budget it as LDS write traffic contending with your `DS_*` compute-side reads.
4. Because the manual omits the family entirely, compiler support is the practical gate. Check whether your HIP toolchain emits them (it will not from plain `__shared__` copies); expect to reach them via inline asm or a builtin, and verify the assembler accepts the mnemonic for `gfx1150`/`gfx1151` before designing a pipeline around them.

> **Correction note for readers of older analyses:** it is easy to conclude from the manual alone that RDNA 3.5 dropped the gfx9/CDNA direct-to-LDS path, because every prose table omits it. The opcode-number gaps plus the XML show the opcodes are present in the ISA.

---

### 2. Direct-to-LDS maxes out at 4 bytes per lane — the VGPR path does 16

This is the single most important trade-off in this section. **Every direct-to-LDS opcode is at most `B32`.** There is no `..._LOAD_LDS_B64/B96/B128` in the XML — the widest is one DWORD per lane. The VGPR-destination path goes 4× wider: buffer ops *"can operate on data as small as one byte, and up to four DWORDS per work-item"* (line 3247), giving `BUFFER_LOAD_B64` / `B96` / `B128` (line 3355) and `GLOBAL_LOAD_B64` / `B96` / `B128` (line 4474 ff.).

So the two staging strategies trade different resources:

| | Direct-to-LDS | `GLOBAL_LOAD_B128` → `DS_STORE_B128` |
|---|---|---|
| Bytes moved per instruction (wave32) | 128 B | 512 B |
| Instructions per 512 B staged | 4 | 2 (one load + one store) |
| VGPRs consumed as staging | 0 | 4 per outstanding B128 per lane |
| Register-file traffic | none | write + read of the whole tile |

**Decision rule.** Pick direct-to-LDS when you are **VGPR-bound** — a large WMMA accumulator tile, or an attention kernel juggling Q fragments, running max and running sum, where the prefetch buffer is the thing pushing you down an occupancy step. Pick wide `B128` + `DS_STORE` when you are **issue- or latency-bound** and have registers to spare — the 4× bytes-per-instruction advantage is real and the two hops software-pipeline cleanly across loop iterations (issue iteration *k+1*'s loads, do iteration *k*'s WMMA, then `DS_STORE`). A hybrid works well in practice: wide VGPR-path loads for the big dense operand (B/weights), direct-to-LDS for the narrow streaming operand (K/V blocks, INT8 scales) that would otherwise need its own register tile.

---

### 3. Wide loads: the fewest instructions per coalesced byte

Independent of LDS, always emit the widest load that matches the per-lane contiguous run. `BUFFER_LOAD_B128` (line 3357) / `GLOBAL_LOAD_B128` (line 4477) move 16 bytes per lane with one address computation and one issue slot instead of four `B32` loads. In HIP this means loading through `float4` / `int4` / `__int128`-shaped types (or `HIP_vector_type`) rather than element-at-a-time, and checking the generated ISA that you actually got `global_load_b128` and not four `global_load_b32`.

Two hard requirements come with the width:

- **16-byte alignment.** LDS/GDS native alignment is `B8`: 1 B, `B16`/`D16`: 2 B, `B32`: 4 B, `B64`: 8 B, `B128` and `B96`: 16 B (lines 852–860). Buffer-side formatted ops need *"4-byte and larger formats require 4-byte alignment"* (line 3736). Pad tensor base pointers and row strides to 16 B.
- **Bounds are checked per DWORD, not per lane** — see §6.

For F16/BF16/INT8 tensors, the `D16` load variants land two 16-bit elements per 32-bit VGPR — *"For loads, data returned from the texture unit is converted to 16 bits and a pair of data are stored in each 32bit VGPR (LSBs first, then MSBs)"* (line 3521). `BUFFER_LOAD_D16_B16` fills `VGPR[15:0]` and `BUFFER_LOAD_D16_HI_B16` fills `VGPR[31:16]` (lines 3352–3353), so a pair of loads builds the packed 2×16 operand WMMA and `V_PK_*` want with no repack. That halves the VGPR footprint of an F16 staging buffer, which partially closes the register-pressure gap that motivates direct-to-LDS in the first place.

---

### 4. Address generation that costs zero VGPRs and zero VALU

Address math is a silent tax on memory-bound kernels: a per-lane 64-bit address is two VGPRs plus the `V_ADD`/`V_LSHL_ADD` chain to build it. RDNA 3.5 offers three ways to make it disappear.

**SADDR base + 32-bit VGPR offset (`GVS` mode).** For `GLOBAL_*`, *"use the SGPR to provide a base address and the VGPR provides a 32-bit byte offset"* (line 4457); the address is `Saddr₆₄ + Voff₃₂ + Ioff` (line 4645). One 32-bit offset VGPR per lane instead of a 64-bit pair, and the 64-bit add happens in the memory pipe. Structure kernels so the tile base is provably wave-uniform (scalar) and only the intra-tile index is per-lane — the compiler will then emit `global_load_* ... , s[base]`.

**13-bit signed INST_OFFSET.** `OFFSET` is a *"13-bit **signed** byte offset"* (line 4456), free in the address unit. Negative offsets are allowed for Global and Scratch-SS/-SV modes (line 4623 ff.), but must be positive for FLAT. This covers a ±4 KB window around a shared base — enough to address several unrolled K-slices or a whole small tile row from one base register, with no VALU per access.

**ADDTID: no address VGPR at all.** *"Global includes two instructions which do not use any VGPRs for addressing, just SGPRs and INST_OFFSET"* (line 4555): `GLOBAL_LOAD_ADDTID_B32` (line 4557) and `GLOBAL_STORE_ADDTID_B32` (line 4558), computing `Saddr₆₄ + Ioff + TID*4` (line 4646). This is the canonical wave-contiguous copy — lane *i* reads DWORD *i* — and the `TID*4` stride is unit-stride by construction, so the wave's accesses are perfectly contiguous with no chance of a stride bug. The LDS-destination sibling `GLOBAL_LOAD_LDS_ADDTID_B32` (XML) is the pure-DMA form: no address VGPR on either side, only `SADDR` and `M0`. It is the ideal opcode for the "cooperatively copy a contiguous KV-cache row into LDS" pattern.

Buffers have the same trick in the descriptor: `const_add_tid_enable` (line 3560) makes the hardware fold the thread ID into the index — `Index = (inst_idxen ? vgpr_index : 0) + (const_add_tid_enable ? workitem_id[5:0] : 0)` (line 3589). *"TID is a constant value (0..63) unique to each thread in the wave. It is ignored when resource bit ADD_TID_ENABLE == 0"* (line 3241).

---

### 5. Keep the staging path on GLOBAL, never FLAT

*"**GLOBAL** is used when all of the address fall into global memory, not LDS or Scratch. This should be used when possible (instead of "Flat") as Global does not tie up LDS resources"* (line 4419). A FLAT op is *"effectively a simultaneous issue of an LDS and GLOBAL instruction at the same time with the same address"* (line 4409), and *"no LDS bandwidth is used by global instructions"* (line 4551) — nor by scratch (line 4562).

For a global→LDS staging loop this is decisive twice over: FLAT would consume LDS crossbar bandwidth that your `DS_*` compute reads need, *and* it destroys the counter separation the pipeline depends on (FLAT charges both `VMcnt`/`VScnt` and `LGKMcnt`, forcing a full `s_waitcnt 0` — detailed in the scheduling section). Annotate device pointers so LLVM can prove `addrspace(1)`, and audit the generated ISA for `global_load_*` rather than `flat_load_*` in any inner loop.

---

### 6. Out-of-range rules: delete the ragged-tile branches

*"Buffer addresses are checked against the size of the memory buffer. Loads that are out of range return zero, and stores and atomics are dropped"* (line 3595). Notably, *"Range checking is per-component for non-formatted loads and stores that are larger than one DWORD. Note that load/store_B64, B96 and B128 are considered "2-DWORD/3-DWORD/4-DWORD load/store", and each DWORD is bounds checked separately"* (line 3595). *"For MTBUF, if any component of the thread is out of bounds, the whole thread is considered out of bounds and returns zero. For MUBUF, only the components that are out of bounds return zero"* (line 3614), and format ops and atomics are checked *"all or nothing"* (line 3613).

Clamping is selected by the 2-bit `OOB_SELECT` field of the V# (line 3597, Table 46 at line 3599, descriptor bits 125:124 at line 3774):

| `OOB_SELECT` | Check | Use |
|---|---|---|
| 0 | `(index >= NumRecords) \|\| (offset+payload > stride)` | structured buffers |
| 1 | `(index >= NumRecords)` | raw buffers |
| 2 | `(NumRecords == 0)` | do not check bounds except empty buffer |
| 3 | raw / swizzled hybrid; `num_records` reduced by `sgpr_offset` | raw |

For a raw buffer, *"NumRecords for raw buffer is in units of bytes. This is an exact range check, meaning it includes the payload and handles multi-DWORD and unaligned correctly"* (line 3634) — the payload size is accounted for, so a `B128` straddling the tensor end is handled exactly.

**How to use it.** Bind each tensor as a raw buffer V# whose `Num_records` (line 3765) is the exact byte length. Then delete the tail-tile guards from your inner loop: on the last M/N/K tile the out-of-range DWORDs return **zero**, which is the additive identity for a dot-product accumulation and the correct pad for a GEMM edge tile. For attention, zeros are safe for the K/V pad *after* you apply the `-inf` logit mask — order matters, because a zero *logit* is not neutral in a softmax while a zero *K/V element* is. Out-of-range stores are dropped, so the output-tile writeback needs no guard either. This removes divergent branches from the hottest loop and lets a single tile shape handle interior and edge tiles identically.

**LDS is not symmetric here.** On the LDS side, *"Writes out-of-range are discarded. Reads return the value zero. For multi-DWORD reads, if any part of the LDS-address is out of range, the entire instruction returns zero"* (line 845). A partially out-of-range `DS_LOAD_B128` gives you **all zeros**, not partial data — the opposite of the MUBUF per-DWORD behavior, and a silent-corruption trap if you rely on buffer semantics by analogy. (`STATUS.BUFFER_OOB` at line 1107 is a sticky status bit you can check while debugging.)

---

### 7. Alignment: three rule sets, one config register

Alignment failures on this architecture are mostly *silent*, not loud, so they are worth spelling out.

**Buffer/VMEM formatted ops** (line 3732 ff.): 1-byte formats need 1-byte alignment, 2-byte formats 2-byte, *"4-byte and larger formats require 4-byte alignment"* (line 3736). **Atomics** are stricter: *"Atomics must be aligned to the data size, or triggers a `MEMVIOL`"* (line 3738) — this applies to your `GLOBAL_ATOMIC_ADD_F32` reduction accumulators.

**Non-formatted ops** are governed by `SH_MEM_CONFIG.alignment_mode` (line 3740):

| Mode | Behavior |
|---|---|
| 0 `DWORD` | hardware auto-aligns to min(element-size, DWORD); *"the two LSBs of the byte-address are ignored, thus forcing DWORD alignment"* (line 3745) — **silently drops address bits** |
| 1 `DWORD_STRICT` | must be aligned to min(element-size, DWORD) |
| 2 `STRICT` | must be aligned to the data size |
| 3 `UNALIGNED` | any alignment allowed |

*"Options 1 and 2 report MEMVIOL if a request is made with incorrect address alignment. In options 1 and 2, loads that are misaligned return zero, and stores that are misaligned are discarded"* (line 3750). And generally, *"Memory instructions return MEMVIOL for any misaligned access when the alignment mode does not allow it"* (line 3243).

**LDS** has its own auto-alignment: *"Any DS_LOAD or DS_STORE of any size can be byte aligned if the alignment mode is set to "unaligned". For all other alignment modes, LDS forces alignment by zeroing out address least significant bits"* (line 836), with the mask table `B32 → 0xffffC`, `B64 → 0xffff8`, `B96/B128 → 0xffff0` (line 866 ff.). LDS atomics always require natural alignment regardless of mode (line 838).

**Scratch** *"instructions support multi-DWORD access and mis-aligned access (although mis-aligned is slower)"* (line 4562); in Scratch-SS mode *"the inst_offset must be aligned to the payload size: 4 byte aligned for 1-DWORD, 16-byte aligned for 4-DWORD"* and `(SADDR + INST_OFFSET)` must be at least DWORD-aligned (lines 4626–4627). Allocate register-spill slots on 16-byte boundaries and spill with `SCRATCH_STORE_B128`.

**Practical guidance for LLM kernels.** Pad every tensor base pointer and every row stride to 16 bytes so `B128` is always legal. Know which `alignment_mode` your runtime programs — packed INT4/INT8 layouts at non-native offsets need `UNALIGNED` (mode 3), or a strict mode will hand you zeros with no fault you notice in a numerics-only test. Never rely on mode 0's bit-dropping to "round down" an address; it will quietly read the wrong element.

---

### 8. Buffer resource descriptors and swizzled addressing

The V# is four consecutive 4-SGPR-aligned SGPRs (line 3752 ff.), and it is where you encode a tensor's shape once so every load in the kernel gets bounds checking and address folding for free. The relevant fields (Table 47, line 3758): `Base address` [47:0], `Stride` [61:48] (0–16383 bytes), `swizzle Enable` [63:62], `Num_records` [95:64] (*"In units of stride if (stride >= 1), else in bytes"*, line 3765), `Index stride` [118:117] (8/16/32/64), `Add tid enable` [119], `OOB_SELECT` [125:124].

Two addressing modes matter:

- **Raw buffer** (line 3627 ff.): `ADDR = Base + baseOff + Ioff + (OffEn ? Voff : 0)`, stride ignored, `Num_records` in bytes, exact range check. This is the right binding for a flat weight matrix or a contiguous KV-cache slab.
- **Structured buffer** (line 3616 ff.): `ADDR = Base + baseOff + Ioff + Stride*Vidx + (OffEn ? Voff : 0)`, `Num_records` in units of stride. Natural for a KV cache addressed as `[token][head_dim]` or a per-head record: put the token index in the index VGPR and let the hardware multiply by the record stride, saving a `V_MAD` per access.

**Swizzled addressing** targets exactly the array-of-structures layout that KV caches and interleaved-head tensors tend to have: *"Swizzled addressing rearranges the data in the buffer that may improve cache locality for arrays of structures. Swizzled addressing also requires DWORD-aligned accesses. A single fetch instruction must not fetch a unit larger than `const_element_size`. The buffer's STRIDE must be a multiple of `const_element_size`"* (line 3666). `const_element_size` is 4 or 16 bytes depending on `V#.swizzle_enable` (`1` → 4 B, `3` → 16 B; line 3560, line 3764), and `const_index_stride` (line 3771) selects how many consecutive indices are grouped: 8, 16, 32 or 64. The address formula (line 3671 ff.) is:

```
index_msb  = index / const_index_stride      offset_msb = offset / const_element_size
index_lsb  = index % const_index_stride      offset_lsb = offset % const_element_size

buffer_offset = (index_msb*const_stride + offset_msb*const_element_size)*const_index_stride
              + index_lsb*const_element_size + offset_lsb
Final Address = const_base + sgpr_offset + buffer_offset
```

The effect: the same field from `const_index_stride` consecutive records is packed contiguously, so a wave reading "component *j* of records *i..i+31*" walks a dense run instead of striding by the full record size. For a KV cache stored as AoS per token, setting `const_index_stride = 32` (or 64 for wave64) with `const_element_size = 16` makes a 32-lane read of one head-dim chunk hit contiguous memory. Set `swizzle_enable` and `index_stride` in the descriptor and the hardware computes it — no VALU swizzle math. Constraints to respect: STRIDE must be a multiple of `const_element_size`, accesses DWORD-aligned, and no single fetch wider than `const_element_size` (so with a 4-byte element size you cannot use `B128`).

Scratch is *already* swizzled by TID — `Addr = FLAT_SCRATCH + swizzle(Voff + Ioff, TID)` (line 4638 ff.) with *"No range checking (using OOB mode 2)"* (line 3645) — which is why per-thread spill/reload is naturally coalesced across a wave.

Setting the entire V# to zero forces loads to return zero and stores to be ignored (line 3779) — a cheap way to null out an optional input (e.g. an absent bias or attention mask) without branching.

---

### 9. What the manual does *not* say about coalescing

Be careful about claims in this area; the ISA document is thin here and it is easy to over-specify.

The manual **does** describe two coalescing mechanisms explicitly. First, within a load clause, *"Load requests that overlap within the clause are cached with respect to each other"* — so redundant or overlapping fetches issued back-to-back under `S_CLAUSE` hit the clause's cached data instead of re-reading memory. Second, on the write path, the first level of the output cache is a *"write-combining cache (collect scatter and store operations and combine them to provide good access patterns to memory)"* (line 475). Both are worth exploiting: group the tile-load burst and the tile-store burst into homogeneous clauses (see the scheduling section for `S_CLAUSE` rules), and arrange output-tile store addresses so lanes share destination lines.

The manual **does not** state: a data cache-line size (the only "64 bytes" figure given is for *instruction* prefetch, line 595), a rule for how many memory transactions a wave's 32 lane addresses collapse into, or any GB/s bandwidth number for L0/L1/L2 or LDS. The closest hint is that scalar loads *"can return partial results at different times when the load crosses two cache lines"* (line 3162), confirming line-granular transfer without naming the size. **Do not cite a coalescing width or a bandwidth figure from this manual — there isn't one.** Design for the invariants that *are* documented (unit-stride lane mapping, widest legal load, 16-byte alignment, per-DWORD bounds) and measure the rest with `S_MEMTIME`/`REALTIME` or a profiler.

---

### Putting it together: a staging loop skeleton

```
// Prologue: bind tensors as raw buffer V#s with exact Num_records (line 3634)
//           -> all tail-tile guards deleted; OOB DWORDs read as 0 (line 3595)
// Set M0 = LDS destination before any *_LOAD_LDS_* (M0 layout not in Table 7, line 946 - verify)

loop over K tiles:
    s_clause N-1                      // homogeneous load burst (line 475, 1554)
    // VGPR-bound path, narrow streaming operand -> straight into LDS, 0 VGPRs:
    global_load_lds_b32   v_off, s[base]        // (XML) opcode 49, GVS-style addressing
    // or, for a fully contiguous cooperative copy, no address VGPR at all:
    global_load_lds_addtid_b32 s[base] offset:K // (XML) opcode 42, Saddr + Ioff + TID*4

    // Issue-bound path, dense operand -> wide load through VGPRs, 16 B/lane:
    global_load_b128      v[t:t+3], v_off, s[base] offset:0     // (line 4477)
    ...
    // compute on the PREVIOUS tile here (WMMA / DS_LOAD from the other LDS buffer)
    s_waitcnt vmcnt(k)
    ds_store_b128         v_lds, v[t:t+3]                       // 16 B aligned
    s_barrier
```

Rules of thumb, in priority order: **(1)** widest legal load, 16-byte aligned; **(2)** unit-stride lane→element mapping, ideally ADDTID; **(3)** scalar base in `SADDR`, constant displacement in the 13-bit `INST_OFFSET`, only the intra-tile index in a VGPR; **(4)** exact `Num_records` so the hardware does your bounds checks; **(5)** `GLOBAL`, never `FLAT`, in the inner loop; **(6)** reach for the direct-to-LDS family when — and only when — staging VGPRs are the constraint, after verifying your toolchain emits them.


## Cache control (GLC/SLC/DLC) & atomics

RDNA 3.5 exposes three per-instruction cache bits and a per-resource no-allocate field. The single most common mistake in HIP kernels is treating them as one "how coherent do you want it" dial. They are not. Per `Cache Controls: SLC, GLC and DLC` (line 1441): **GLC** controls the graphics first-level cache (line 1445), **SLC** controls the graphics L2 (line 1447), and **DLC** controls the Memory-Attached Last-Level cache / MALL, "if it is present (ignored otherwise)" (line 1449). The normative decode is at lines 1494–1504:

- `ISA.GLC` is a **scope** bit on loads: `0 = CU (work-group) scope`, `1 = DEVICE scope` (lines 1494–1496).
- `ISA.SLC` is a **temporal hint** for the graphics client caches: `0 = Regular`, `1 = Stream (non-temporal)` (lines 1499–1501).
- `ISA.DLC` is a **temporal hint** for the Infinity Cache: `0 = Regular`, `1 = Non-temporal` (lines 1502–1504).
- On atomics, GLC is repurposed entirely: `0: return nothing`, `1: return pre-operation value from memory to VGPR` (lines 1506–1509).

Note the manual does not state whether any specific gfx1150/gfx1151 SKU ships a MALL, so DLC=1 is always safe but may be a no-op on your part.

### The cache-policy tables (condensed)

**Loads** — from the 16-row table at lines 1455–1472. Rows collapse cleanly because the three bits are independent:

| `SRD.llc_noalloc` | DLC | SLC | GLC | MALL no-alloc | GL2 | GL1 | Tex (L0) | Scope | Line |
|---|---|---|---|---|---|---|---|---|---|
| 0 or 1 | 0 | 0 | 0 | 0 | LRU | HIT_LRU | HIT_LRU | CU | 1457 |
| 0 or 1 | 0 | 0 | 1 | 0 | LRU | MISS_EVICT | MISS_EVICT | DEVICE | 1458 |
| 0 or 1 | 0 | 1 | 0 | 0 | **STREAM** | HIT_EVICT | HIT_LRU | CU | 1459 |
| 0 or 1 | 0 | 1 | 1 | 0 | **STREAM** | MISS_EVICT | MISS_EVICT | DEVICE | 1460 |
| 0 or 1 | 1 | 0 | 0 | **1** | LRU | HIT_LRU | HIT_LRU | CU | 1461 |
| 0 or 1 | 1 | 1 | 0 | **1** | **STREAM** | HIT_EVICT | HIT_LRU | CU | 1463 |
| 2 or 3 | 0 | 0 | 0 | **1** | LRU | HIT_LRU | HIT_LRU | CU | 1465 |
| 2 or 3 | 1 | 1 | 1 | **1** | **STREAM** | MISS_EVICT | MISS_EVICT | DEVICE | 1472 |

Reading the whole table: **GLC picks the L0/GL1 column** (HIT_LRU when 0, MISS_EVICT when 1), **SLC picks the GL2 column** (LRU vs STREAM, and GL1 HIT_LRU vs HIT_EVICT), **DLC-or-`llc_noalloc` picks the MALL column**. Nothing else moves.

**Stores and atomics** — lines 1479–1488. There is no GLC/scope column at all: "All stores/atomic ops are device scope (GLC has non-perf related functionality)" (line 1498).

| `SRD.llc_noalloc` | DLC | SLC | MALL no-alloc | GL2 | Line |
|---|---|---|---|---|---|
| 0 or 2 | 0 | 0 | 0 | LRU | 1481 |
| 0 or 2 | 0 | 1 | 0 | **STREAM** | 1482 |
| 0 or 2 | 1 | 0 | **1** | LRU | 1483 |
| 1 or 3 | 0 | 0 | **1** | LRU | 1485 |
| 1 or 3 | 1 | 0 | **1** | LRU | 1487 |

Caveat: the tables also carry "Hint: MALL / GL2 / GL1 / Tex(L0)" columns, and the manual glosses "Temporal Hint" as "expect data to have temporal reuse" (line 1490) — yet the `yes` entries appear on the *non-temporal* rows. That labelling is self-contradictory in the manual. **Reason from the Policy columns, which are unambiguous, and ignore the Hint columns.**

---

### 1. Keep GLC=0 on every hot tile load; GLC=1 is an L0 kill switch, not free coherence

**Hardware.** "Typically loads use GLC=0 (except for load-acquire). GLC=1 forces a miss in the first level cache and reads data from the L2 cache. If there was a line in the GPU L0 that matched, it is invalidated; L2 is reread." (line 1451). The table confirms it: every GLC=1 load row has GL1 and Tex(L0) policy `MISS_EVICT` (lines 1458, 1460, 1462, 1464).

**Why it matters for LLM kernels.** GEMM and attention inner loops re-read the same A/B tile across lanes and across waves in a work-group; the L0 hit is the entire point of the tiling. A stray `__ldg`-style "coherent" load, an `atomic_load` with device ordering, or a `volatile` pointer in the K-loop turns every tile fetch into an L2 round trip and evicts the line for the neighbouring wave too.

**How to use.** Default all `GLOBAL_LOAD_*` / `BUFFER_LOAD_*` in the K-loop to GLC=0. Reserve GLC=1 for exactly one thing: the load-acquire that must observe another work-group's store — the split-K "partials ready" flag, the flash-decoding combine-phase partial read, the work-queue head. Disassemble with `llvm-objdump` and grep for `glc` inside your inner loop; if it appears on a tile load, something upstream (a `volatile`, a `__threadfence`-adjacent access, or an atomic-typed pointer) leaked device scope into the hot path.

### 2. SLC=1 (non-temporal) on stream-once tensors to stop them evicting the reuse set

**Hardware.** SLC=1 flips GL2 policy from `LRU` to `STREAM` and GL1 from `HIT_LRU` to `HIT_EVICT` (line 1459 vs 1457). The line passes through rather than being retained.

**Why it matters.** Decode-phase LLM work is bandwidth-bound and highly asymmetric in reuse: KV-cache blocks, the weight matrix in a memory-bound GEMV/thin GEMM, and elementwise activation streams are read exactly once per launch, while per-channel scales/zero-points, the RoPE table, layernorm gains, and the accumulator tile are read many times. Without SLC, the one-shot gigabytes evict the kilobytes that actually have reuse.

**How to use.** Tag the large single-pass operand SLC=1 and leave the small reused operands SLC=0. Concretely in an INT4/INT8 dequant-GEMM: weights and the packed quantized stream get SLC=1; the scale/zero-point vectors and the activation tile staged in LDS stay SLC=0. In flash-attention prefill, the K/V blocks streamed once per query tile get SLC=1 while Q stays SLC=0. The manual documents only the ISA bit; it does not document a HIP intrinsic — verify whatever `__builtin_nontemporal_load` emits on your toolchain against the disassembly.

### 3. SLC=1 on the epilogue store so the output does not evict the input

**Hardware.** Stores have no scope choice, but SLC still selects GL2 `LRU` vs `STREAM` (lines 1481–1482).

**Why it matters.** A GEMM epilogue, a logits write, or an elementwise output is written once and never re-read by that kernel. At LLM output sizes (vocab-sized logits, full activation tensors) that write stream will displace the weight/KV working set in L2 for the waves still running.

**How to use.** Non-temporal store on the final `GLOBAL_STORE_*` / `BUFFER_STORE_*` of a fused epilogue. Keep SLC=0 on stores you intend to re-read within the same kernel (e.g. split-K partials that the same launch's combine phase will consume).

### 4. DLC=1 as an independent MALL knob

**Hardware.** DLC=1 raises the MALL no-allocate column to 1 without touching GL2 or L0 policy (line 1461 vs 1457). It is "ignored otherwise" if no MALL is present (line 1449).

**Why it matters.** When the weight matrix exceeds MALL capacity — the normal case for a multi-billion-parameter model — allocating it in MALL is pure thrash. The reused KV working set and accumulator traffic are what you want resident at last level.

**How to use.** DLC=1 on the weight stream, DLC=0 on KV/activation. Because DLC is a separate bit from SLC you can stream at L2 but allocate at MALL (SLC=1, DLC=0) for a tensor that is read once per work-group but re-read across work-groups — a shared K/V block in grouped-query attention is the canonical case.

### 5. Bake MALL no-allocate into the buffer descriptor instead of every instruction

**Hardware.** The `SRD (llc_noalloc)` column drives MALL no-allocate independently of DLC. For loads, `llc_noalloc` = 2 or 3 forces MALL NOA=1 on every row (lines 1465–1472). For stores, values 1 or 3 do it (lines 1485–1488). The split is a strong hint that bit1 governs loads and bit0 governs stores — the manual states the row values but does not name the bits, so treat that reading as inference.

**Why it matters.** A whole-tensor policy costs zero instruction bits and cannot be lost by a compiler pass that rewrites your loads.

**How to use.** Build the weight-tensor V# with `llc_noalloc` set for loads (2 or 3) and the partials/output V# set for stores (1 or 3) if you want writes to skip MALL. This is also the *only* MALL control available to `S_BUFFER_LOAD`: "For S_BUFFER_LOAD instructions, LLC_NOALLOC comes from V#.LLC_noalloc" (line 1474).

### 6. The scalar path cannot be made non-temporal — do not route bulk tensors through SMEM

**Hardware.** "SMEM operations have SLC set to zero" and "For S_LOAD, LLC_NOALLOC is zero" (lines 1474–1475).

**Why it matters.** Scalar loads are otherwise excellent for LLM kernels — `S_BUFFER_LOAD_B512` pulls 16 uniform DWORDs into SGPRs in one instruction with no VGPR or vector-issue cost. But a "uniform" pointer that actually addresses a large streamed tensor will be allocated LRU in L2 with no way to say otherwise, and will evict the reused tiles.

**How to use.** Use SMEM for genuinely small uniform data — descriptors, dimensions, per-tensor scales, kernel arguments. If a broadcast-shaped read is actually bulk (a whole bias row, a large lookup table), issue it through the vector path where SLC/DLC are honoured, or at minimum through `S_BUFFER_LOAD` with `V#.llc_noalloc` set so you retain the MALL control (line 1474).

### 7. Trust §4.1.1, not the encoding-field tables — the manual contradicts itself

**Hardware.** The per-encoding field tables describe the bits in pre-RDNA coherence language: MUBUF/MTBUF call SLC "System Level Coherent" and DLC "Device Level Coherent" (lines 3313–3314, 7273–7274); FLAT calls DLC "Device Level Coherent. Controls behavior of L1 cache (GL1)" and GLC "Group Level Coherent - controls behavior of L0 cache" (lines 4453–4454); SMEM calls GLC "Globally memory Coherent. Force bypass of L1 cache" (line 5985). Section 4.1.1 instead defines SLC and DLC as pure temporal hints for GL2 and MALL (lines 1447, 1449, 1499–1504) and GLC as a CU-vs-DEVICE scope selector (lines 1494–1496).

**Why it matters.** If you reason from the FLAT field table you will conclude DLC controls GL1 and set it for the wrong reason. The 4.1.1 policy tables are the ones with actual per-level policy outcomes, and they show DLC only ever moving the MALL column.

**How to use.** Use lines 1455–1504 for all performance reasoning. The field tables are reliable only for **bit positions**, which do differ per encoding and matter if you are hand-assembling: FLAT/GLOBAL/SCRATCH is DLC[13], GLC[14], SLC[15] (lines 7545–7547, confirmed 27992–27994); MUBUF/MTBUF is SLC[12], DLC[13], GLC[14] (lines 7273–7275); SMEM is DLC[14], GLC[16] (lines 5984–5985).

### 8. Pick the narrowest cache maintenance op — there are four, at three different scopes

**Hardware.**

| Op | Effect | Line |
|---|---|---|
| `BUFFER_GL0_INV` | "Write back **and** invalidate the shader L0. Returns ACK to shader." | 26704 |
| `BUFFER_GL1_INV` | "Invalidate the GL1 cache only. Returns ACK to shader." | 26709 |
| `S_GL1_INV` | "Invalidate the GL1 cache only." | 11369 |
| `S_DCACHE_INV` | "Invalidate the scalar data L0 cache." | 11374 |
| `S_ICACHE_INV` | Invalidate first-level instruction cache for this WGP | 1550 |

Note the summary table at line 3415 describes `BUFFER_GL0_INV` as only "invalidate the shader L0 cache (texture cache)", omitting the writeback that the opcode description at line 26704 states. The detailed opcode page is the stronger statement; if you depend on the writeback, validate it. `BUFFER_GL1_INV` is scoped "for this wave's VMID" (line 3416).

**Why it matters.** Persistent / megakernel designs — fused attention that keeps a work-group resident across phases, cooperative split-K where the combine phase runs in the same launch, speculative-decode verify kernels — need a producer→consumer handoff without a full device flush between kernel launches. That is exactly what these narrow ops buy.

**How to use.** After the producer stores, `BUFFER_GL0_INV` to push the WGP's L0 out and drop stale lines; invalidate GL1 only if consumers live on a different WGP within the shader array. Do not reach for a device-wide flush when the sharing is WGP-local. `S_DCACHE_INV` is separate — the scalar data cache is not covered by the vector invalidates, so if a consumer re-reads a *scalar*-loaded descriptor or scale that the producer rewrote, you need it explicitly.

### 9. Invalidates are LGKMcnt events — you must `S_WAITCNT` before the dependent load

**Hardware.** "LGKMcnt is incremented by 1 for every fetch of a single DWORD, **or cache invalidates**" (line 3164), and "Cache invalidate instructions are not known to have completed until the shader waits for LGKMcnt==0" (line 3170). The invalidates take no address or data arguments (line 3158), and SMEM offset is "Ignored for cache invalidations" (line 5988).

**Why it matters.** This is a silent-corruption trap in persistent LLM kernels: issue `S_GL1_INV`, immediately load, and you can still read the pre-invalidate line. The bug looks like a rare numerical mismatch in the combine phase, not a hang.

**How to use.** `S_GL1_INV` / `S_DCACHE_INV` / `BUFFER_GL0_INV` → `S_WAITCNT lgkmcnt(0)` → dependent loads. Because SMEM returns out of order anyway, "the only sensible way to use this counter is to implement `S_WAITCNT LGKMcnt 0`" (line 3168), so batch all your invalidates together and pay one full drain rather than one per invalidate.

### 10. Cache-less loads inside a clause: forced-DRAM reads that still de-duplicate

**Hardware.** "Specific cache-less load instructions can force data to be retrieved from device memory during an execution of a load clause. Load requests that overlap within the clause are cached with respect to each other." (line 475).

**Why it matters.** The combine phase of split-K or flash-decoding reads partials another work-group just wrote. You want freshness, but the lanes' addresses overlap heavily, and paying DRAM for each overlapping request would be ruinous. Within an `S_CLAUSE` the hardware coalesces them.

**How to use.** Wrap the combine-phase partial loads in `S_CLAUSE` (2–63 instructions, single type, "Clauses lock the instruction arbiter onto this wave until the clause completes", line 3174). **Caveat:** the manual describes the cache-less load capability at line 475 but never names which opcodes are the "cache-less load instructions." Do not assume a specific mnemonic; the safe portable construction is a GLC=1 clause, which at least guarantees the L0 miss and L2 reread (line 1451).

---

### 11. GLC=0 on accumulation atomics: no return value, no VGPR, no VMCNT wait

**Hardware.** On atomics GLC is purely a return selector — "0: return nothing / 1: return pre-operation value from memory to VGPR" (lines 1506–1509), restated per-encoding at lines 3312, 3860, 4454, 7275, 7546. Every opcode page says the same: "Store the original value ... into a vector register **iff the GLC bit is set**" (`GLOBAL_ATOMIC_ADD_F32`, line 29420; `BUFFER_ATOMIC_ADD_F32`, line 27082).

The second-order effect is the important one. Atomics split across two different wait counters: `VMcnt` covers "Texture/Buffer/Global/Scratch/Flat Loads **and atomic-with-return**", `VScnt` covers "Stores **and atomic-without-return**" (lines 1704–1705). Confirmed at the opcode level: `S_WAITCNT_VSCNT` "tracks the number of outstanding vector memory stores and atomics that *do not* return data" (line 8817); `S_WAITCNT_VMCNT` tracks those that *do* (line 8838); and plain `S_WAITCNT`'s VMCNT field "only counts vector memory loads, image sample instructions, and vector memory atomics that return data" (line 10808).

**Why it matters.** This is the single highest-leverage atomics choice for LLM kernels. A split-K GEMM epilogue issuing `GLOBAL_ATOMIC_ADD_F32` with GLC=0 (a) allocates no VDST, which is real VGPR relief in a register-starved WMMA kernel, (b) does not perturb `vmcnt`, so your software-pipelined `S_WAITCNT vmcnt(N)` on the *next* tile's loads stays exact, and (c) can be drained once at the end with a single `S_WAITCNT_VSCNT` instead of serializing on each accumulate. The same applies to attention denominator accumulation and any global sum/histogram.

**How to use.** Ensure the discarded-return form is emitted — in HIP, `atomicAdd(...)` whose result is unused should compile to GLC=0, but this is a compiler decision, not an ISA guarantee; check the disassembly. Then drain with `S_WAITCNT_VSCNT` before the flag store, not with a blanket `S_WAITCNT 0`. One further subtlety: out-of-range destination-VGPR nullification applies to atomics ("issued, but with EXEC mask of zero", line 802) but "`VDST` is only checked for `lds/gds/mem-atomic` that actually return a value" (line 804) — a GLC=0 atomic has no VDST to check.

### 12. Hardware FP32 atomics: ADD / MIN / MAX / CMPSWAP across LDS, buffer, and flat/global/scratch

**Hardware.** "Floating point atomics can be issued as LDS, Buffer, and Flat/Global/Scratch instructions." (line 5248). The full FP32 set:

| Op | Buffer | Flat / Global | LDS |
|---|---|---|---|
| add | `BUFFER_ATOMIC_ADD_F32` (27079) | `FLAT_ATOMIC_ADD_F32` / `GLOBAL_ATOMIC_ADD_F32` (4491, 29417) | `DS_ADD_F32` (24714), `DS_ADD_RTN_F32` (25796) |
| min | `BUFFER_ATOMIC_MIN_F32` (3396) | `FLAT_ATOMIC_MIN_F32` / `GLOBAL_ATOMIC_MIN_F32` (4503) | `DS_MIN_F32` (24677), `DS_MIN_RTN_F32` (24975) |
| max | `BUFFER_ATOMIC_MAX_F32` (3395) | `FLAT_ATOMIC_MAX_F32` / `GLOBAL_ATOMIC_MAX_F32` (4504) | `DS_MAX_F32` (24693), `DS_MAX_RTN_F32` (24991) |
| cmpswap | `BUFFER_ATOMIC_CMPSWAP_F32` (3394) | `FLAT_ATOMIC_CMPSWAP_F32` (28527) / `GLOBAL_ATOMIC_CMPSWAP_F32` (29368) | `DS_CMPSTORE_F32` (7186) |

Buffer atomics are "Automatically globally coherent. Operates on 32bit or 64bit values." (line 3258) — no manual invalidate between accumulating waves.

**Why it matters.** Three canonical LLM patterns map straight onto these: split-K / stream-K GEMM accumulates FP32 partials into the C tile with ADD_F32; online-softmax and attention partial merging need a global running max, which is MAX_F32; and layernorm/RMSNorm cross-block statistics are ADD_F32 sums. Each replaces a CAS retry loop.

**How to use.** Prefer the dedicated ADD/MIN/MAX opcodes over CMPSWAP wherever the op is expressible — CAS loops retry under contention and each retry is a full memory round trip with a mandatory GLC=1 return. Use `GLOBAL_*` rather than `BUFFER_*` when you have a raw 64-bit pointer and no descriptor; use `BUFFER_*` when you want the V# to carry `llc_noalloc` (see #5) or you want hardware bounds checking on the accumulator (see #17).

### 13. There is no FP16, BF16, packed, or FP64 atomic add — plan split-K accordingly

**Hardware.** Searching the complete atomic opcode tables (`BUFFER` at 3384–3414 and 26711+, `FLAT`/`GLOBAL` at 4489–4518 and 28250+/29052+, `DS` at 7186–7218 and 24463+), the only floating-point atomic types present are **F32** (add/min/max/cmpswap) and, in LDS only, **F64 min/max/cmpstore** (`DS_MIN_F64` line 25461, `DS_MAX_F64` line 25477, `DS_CMPSTORE_F64` line 7186). There is no `*_ATOMIC_ADD_F64` anywhere, and no `PK_ADD_F16` / `PK_ADD_BF16` in any aperture. Chapter 13 is titled "Float Memory Atomics" (line 5246) and describes only F32 add and F32/F64 min/max/cmpstore.

**Why it matters.** On other vendors' parts, packed-FP16 atomic add is the standard split-K trick for halving reduction bandwidth. It does not exist here. A kernel ported with `atomicAdd(__half2*)` will either fail to compile or be expanded by the compiler into a CAS loop — which is dramatically slower under the contention a split-K epilogue generates.

**How to use.** Accumulate split-K partials in **FP32** and atomically add in FP32, converting to FP16/BF16 only in a separate final pass. If you truly need 16-bit accumulation in memory, build it on `GLOBAL_ATOMIC_CMPSWAP_B32` over a packed `half2` and accept the retry cost — but measure against the FP32 path first. For FP64 accumulation there is no atomic add at all; use CAS on `_B64` or restructure to a tree reduction.

### 14. Float atomic add has fixed rounding and hardwired denormal flush — and differs between LDS and global

**Hardware.** "LDS and Memory atomics have the rounding mode for float-atomic-add fixed at 'round to nearest even'. The MODE.round bits are ignored." (line 5252). "Float atomic add is hardwired to flush input denormals - it does not use the MODE.fp_denorm bits." (line 5286). But the denormal table at lines 5265–5274 splits the behaviour by aperture:

| Op | Cache atomics (Buffer/Flat/Global/Scratch) | LDS atomics |
|---|---|---|
| `Add_F32` | **Flush** (fixed) | **Mode** (follows `MODE.fp_denorm`) |
| `Min/Max_F32` | Mode | Mode |
| `CmpStore_F32`, `_F64` | Mode | Mode |
| `Min/Max_F64` | — | Mode |

**Why it matters.** The same source-level `atomicAdd(float*)` has *different* denormal semantics depending on whether the pointer lands in LDS or global memory. In a two-level reduction (LDS partial → global total) that is an invisible precision seam. It also means `S_DENORM_MODE` tuning around your reduction affects the LDS stage but not the global stage.

**How to use.** Do not attempt to configure `MODE.fp_denorm` or `MODE.round` to influence global float atomic-add; it will not respond. Do account for the fact that a global FP32 atomic-add flushes denormal inputs — for attention denominators and softmax sums this is harmless (values are far from denormal after max-subtraction), but for gradient-style or heavily-scaled accumulations it is a real precision difference from the LDS path. If bit-exactness between an LDS-reduction build and a global-atomic build matters, set `MODE.fp_denorm` to flush for the LDS stage so the two agree.

### 15. FP min/max atomics resolve NaN in your favour — free NaN-robust running max

**Hardware.** Every float atomic opcode page carries "Floating-point compare handles NAN/INF/denorm" (e.g. `GLOBAL_ATOMIC_MAX_F32` line 29415, `GLOBAL_ATOMIC_ADD_F32` line 29430, `DS_ADD_F32` line 24727). The selection rules are explicit (lines 5303–5319):

- **Max**: `"Larger" order from smallest to largest: QNaN, -inf, -float, -denorm, -0, +0, +denorm, +float, +inf` (line 5309) — QNaN sorts **smallest**, so max returns the real value.
- **Min**: `"Smaller" order from smallest to largest: -inf, ... , +inf, QNaN` (line 5318) — QNaN sorts **largest**, so min returns the real value.
- SNaN in either input is converted to QNaN and returned (lines 5306–5307, 5315–5316).

**Why it matters.** Online/flash softmax computes a running row max across blocks. Masked positions are frequently `-inf`, and fused kernels sometimes produce QNaN from `-inf - -inf` or from `exp` of a fully-masked row. Because QNaN sorts below `-inf` for max, a `GLOBAL_ATOMIC_MAX_F32`-based running max cannot be poisoned by a stray quiet NaN from one block — the finite blocks win.

**How to use.** Use `GLOBAL_ATOMIC_MAX_F32` / `DS_MAX_F32` directly for cross-block row-max without a guard. Note the asymmetry: `-inf` *does* participate normally, so a fully-masked row still yields `-inf` and your downstream `exp` path must handle that. Also note MIN/MAX "when flushing denormals only do it for the comparison, but the result is an unmodified copy of one of the sources" (line 5280) — the stored value is bit-exact from the source, never a flushed rewrite.

### 16. Two-level reduction: LDS atomics first, one global atomic per work-group

**Hardware.** Each WGP's LDS "contains 64 integer atomic units to enable fast, unordered atomic operations" (line 467), and "LDS atomics are performed in the LDS hardware. Although ALUs are not directly used for these operations, latency is incurred by the LDS executing this function." (line 4713). Latency is quantified: LDS indexed and atomic operations "can complete in as little as one cycle (for wave32, or 2 cycles for wave64), or take as many 64 cycles, depending upon the number of bank conflicts" (line 4967). LDS atomics track on `LGKMcnt` and "stay in-order with other LDS instructions from the same wave" (line 4973). "Atomic operations have the option of returning the LDS 'pre-op' value to VGPRs." (line 4971).

**Why it matters.** A naive layernorm or softmax reduction that has every lane issue a global atomic to one address serializes 32–64 lanes × N work-groups on a single cache line. Funnelling through LDS first collapses that to one global atomic per work-group. The LDS atomic units are separate hardware, so the reduction does not consume VALU issue slots that your `V_EXP_F32` / WMMA stream needs.

**How to use.** Wave-level reduction (DPP/permute) → `DS_ADD_F32` to one LDS slot per work-group → barrier → lane 0 issues one `GLOBAL_ATOMIC_ADD_F32` with GLC=0. Spread the LDS accumulator slots across banks: "conflict-free" is 1 cycle, full conflict is 64 (line 4967), a 64× swing. The manual quantifies LDS atomic latency but gives **no throughput or latency figures for global/buffer atomics** — do not assume; measure.

### 17. LDS return-vs-no-return is a separate *opcode*, not a bit

**Hardware.** Unlike vector memory where GLC selects the return, the DS encoding has distinct opcodes: `DS_ADD_F32` is opcode 21 (line 24714) while `DS_ADD_RTN_F32` is opcode 121 (line 25796); `DS_MIN_F32` is 18 (line 24677) while `DS_MIN_RTN_F32` is 50 (line 24975); `DS_MAX_F32` is 19 (line 24693) vs `DS_MAX_RTN_F32` at 51 (lines 7217–7218).

**Why it matters.** Hand-written LDS reduction assembly that reaches for the RTN form out of habit pays for a VGPR writeback and an LGKMcnt dependency it does not need.

**How to use.** Use the non-RTN opcode for pure accumulation. Use RTN when you genuinely need the pre-op value — an in-LDS ticket/index allocator, or a per-work-group compaction counter.

### 18. Align atomic accumulators exactly, or take a MEMVIOL

**Hardware.** "Atomics must be aligned to the data size, or triggers a `MEMVIOL`." (line 3738) — a fault, not a slow path. Buffer atomics are "range-checked 'all or nothing' - either entirely in or out" (line 3612), and out-of-range stores/atomics "do not store anything" (line 3611). LDS atomics also require alignment while LDS indexed loads may be misaligned in `UNALIGNED` mode.

**Why it matters.** Split-K partial buffers are frequently laid out as `[K_splits][M][N]` with a padded inner stride chosen for bank/coalescing reasons. If that padding is not a multiple of the atomic width, the first out-of-phase accumulator faults the wave. The 64-bit case is the usual offender: an 8-byte atomic on a 4-byte-aligned address faults.

**How to use.** Pad split-K accumulator strides to the atomic's natural size — 4 bytes for `_F32`/`_U32`, 8 bytes for `_U64`/`_B64`. If you route atomics through a buffer descriptor, the all-or-nothing bounds check at line 3612 gives you free branchless edge handling on ragged M/N tiles: an out-of-range atomic is silently dropped rather than corrupting a neighbour.

### 19. Prefer `GLOBAL_ATOMIC_*` over `FLAT_ATOMIC_*` — FLAT doubles the counter cost

**Hardware.** "Since Flat instruction are executed as both an LDS and a Global instruction, Flat instructions increment both VMcnt (or VScnt) and LGKMcnt and are not considered done until both have been decremented. There is no way a priori to determine whether a Flat instruction uses only LDS or Global memory space." (line 4528). This is echoed in the counter list: "FLAT instructions (uses both LGKMcnt and either VMcnt or VScnt)" (line 1710). And in scratch: "Flat atomics which map into scratch: 4-byte atomics are supported, and 8-byte atomics return MEMVIOL." (line 4532).

**Why it matters.** Because a FLAT op is pending on two counters, precise `S_WAITCNT vmcnt(N)` software pipelining stops working around it — you effectively need a full drain, which destroys the memory-latency hiding that a split-K epilogue interleaved with the next tile's loads depends on.

**How to use.** Emit `GLOBAL_ATOMIC_*` whenever the pointer is statically global (the normal case for a partials buffer). In HIP that means avoiding generic pointers into the accumulator — use `__global__`-qualified or address-space-cast pointers so the compiler picks the GLOBAL form. Never emit 8-byte atomics on a pointer that could resolve to scratch.

### 20. `GLOBAL_ATOMIC_CSUB_U32` — one-instruction work-queue claim for persistent/stream-K kernels

**Hardware.** "Subtract an unsigned 32-bit integer location in the global aperture from a value in the data register and clamp the result to zero" (line 29098), with the full clamp semantics at lines 29101–29109. It is the one atomic where the return is mandatory: the opcode table annotates it "(GLC must be set to 1)" (line 4518), matching `BUFFER_ATOMIC_CSUB_U32` "returns previous . **GLC must be set to 1**" (line 3391).

**Why it matters.** Stream-K and persistent-kernel LLM GEMMs claim tile ranges from a global work queue. `CSUB` gives you claim-and-clamp in one atomic: each work-group subtracts its claim size, gets the pre-op counter as its tile base, and the counter saturates at zero instead of underflowing — so no wraparound guard and no CAS loop at the queue tail.

**How to use.** Initialise the counter to the total tile count, have each persistent work-group `CSUB` its chunk size, and derive tile indices from the returned pre-op value; a return of 0 means the queue is drained. Do not try to save the return by clearing GLC — the hardware requires GLC=1 here.

### 21. 64-bit atomics for wide counters and packed state

**Hardware.** The full 64-bit integer set exists in all three apertures: `BUFFER_ATOMIC_ADD_U64` (line 3386), `AND/OR/XOR_B64` (3388, 3400, 3414), `MIN/MAX_I64/U64` (3402, 3404, 3410, 3412), `INC/DEC_U64` (3393, 3398), `SWAP/CMPSWAP_B64` (3408, 3390), mirrored as `FLAT_/GLOBAL_ATOMIC_*_B64/U64` (lines 4505–4517). Buffer atomics "Operate on 32bit or 64bit values" (line 3258).

**Why it matters.** `GLOBAL_ATOMIC_CMPSWAP_B64` is the primitive for atomically publishing a *pair* of values — the classic case in attention is committing `(running_max, running_sum)` together so a consumer never observes a torn combination during partial-result merging. `_OR_B64` gives you a 64-slot arrival bitmask for a split-K combine barrier in one instruction.

**How to use.** Align to 8 bytes (see #18) and keep the pointer provably global (see #19 — 8-byte flat atomics that reach scratch fault). For the max/sum pair, pack both FP32 values into one 64-bit word and CAS the pair.

### 22. `CMPSWAP_F32` when the reduction is not add/min/max

**Hardware.** `GLOBAL_ATOMIC_CMPSWAP_F32` (line 29368) and `FLAT_ATOMIC_CMPSWAP_F32` (line 28527) take both operands from one aligned VGPR pair: `src = DATA[31:0].f32; cmp = DATA[63:32].f32` (lines 29375–29376, 28534–28535). The buffer form matches: "Src is from vdata, cmp from vdata+1" (line 3394). The comparison is a float compare, and denormal flushing for cmpstore follows `MODE` (line 5268), with "CompareStore ('compare swap') flushes the result when input denormal flushing occurs" (line 5280).

**Why it matters.** Flash-attention partial merging is not a plain add: combining `(m_i, l_i, O_i)` with `(m_j, l_j, O_j)` requires rescaling by `exp(m_i - m_max)`, which no hardware atomic implements. A CAS loop on the merged state is the general fallback.

**How to use.** Load current, compute merged value, `CMPSWAP_F32` with GLC=1, retry on mismatch. Because this uses a float compare, be aware that a NaN in memory never compares equal and the loop will spin forever — bound the retry count or use `CMPSWAP_B32` (bitwise) instead, which is the safer primitive for CAS loops precisely because it has no float compare semantics. Prefer restructuring to a deterministic two-pass reduction (all blocks write partials, one kernel merges) over a contended CAS loop; the CAS path forces GLC=1, which forces a VMCNT dependency per retry.

### 23. Stores from one wave to different addresses are **not** ordered — fence before publishing a flag

**Hardware.** "It is possible for data to be written to VGPRs out-of-order, but the counter-decrement still reflects in-order completion. **Stores from a wave are not kept in order with stores from that same wave when they write to different addresses.**" (line 1717). The device-memory description adds that "Each scatter write from a given PE to a given memory channel maintains order" and that the write-acknowledgment "enables one processing element to implement a fence to maintain serial consistency by ensuring all writes have been posted to memory prior to completing a subsequent write. In this manner, the system can maintain a relaxed consistency model" (lines 480–482).

**Why it matters.** The split-K / flash-decoding combine handshake is exactly this pattern: write N partial values, then set a "ready" flag. Without an explicit drain the flag can land first and a consumer reads garbage partials. The manual describes a relaxed consistency model and gives you the counters to build a fence, but it does **not** define a formal memory model or a documented acquire/release instruction mapping — the recipe below is assembled from the documented primitives, not quoted from a spec section.

**How to use.** Producer: partial stores → `S_WAITCNT_VSCNT 0` (line 8817, drains stores and no-return atomics) → flag publish via `GLOBAL_ATOMIC_ADD_U32`. Consumer: poll the flag with a **GLC=1** load (device scope, forces the L0 miss and L2 reread, line 1451) → on success, `BUFFER_GL0_INV` → `S_WAITCNT lgkmcnt(0)` (line 3170) → read partials. Do not substitute a plain GLC=0 poll; it can spin on a stale L0 line indefinitely.

### 24. Buffer atomics are automatically globally coherent — do not add redundant invalidates

**Hardware.** "`BUFFER_ATOMIC_{<op>}`: Buffer object atomic operation. **Automatically globally coherent.**" (line 3258). Structurally, the write path's second level "is a read/write cache with atomic units that lets each processing element complete unordered atomic accesses that return the initial value" (line 475), with the ack path enabling a PE "to recover the pre-op value from an atomic operation by performing a cache-less load from its return address after receipt of the write confirmation acknowledgment" (lines 479).

**Why it matters.** A common over-defensive pattern is to bracket every atomic accumulation with `__threadfence()` and cache invalidates. Since the atomic units live in the coherent read/write cache and all stores/atomics are device scope by construction (line 1498), the accumulation itself needs no manual coherence management. The invalidates are needed only for *ordinary loads* that must observe another work-group's *ordinary stores* (#23).

**How to use.** Accumulate freely with GLC=0 atomics and no fences between them; place the single fence/invalidate only at the phase boundary where non-atomic reads begin. Note that the atomics are explicitly "unordered" (lines 467, 471, 475), so FP32 atomic accumulation is **not run-to-run deterministic** — split-K FP32 atomic add will produce bit-different results across launches. If reproducibility is a requirement (regression tests, numerics debugging), use a deterministic two-pass reduction instead of atomics.

---

**Quick reference — bit settings by kernel role**

| Access | GLC | SLC | DLC | Rationale |
|---|---|---|---|---|
| GEMM/attention tile load (reused) | 0 | 0 | 0 | CU scope, HIT_LRU everywhere (line 1457) |
| Weight stream, read once | 0 | 1 | 1 | GL2 STREAM + MALL no-alloc (line 1463) |
| KV block, reused across work-groups | 0 | 1 | 0 | GL2 STREAM, keep MALL residency (line 1459) |
| Scale/zero-point, RoPE table | 0 | 0 | 0 | small, high reuse (line 1457) |
| Cross-work-group flag / partial (acquire) | **1** | 0 | 0 | DEVICE scope, forced L0 miss + L2 reread (lines 1451, 1458) |
| Epilogue output store | n/a | 1 | 0 | GL2 STREAM, don't evict inputs (line 1482) |
| Split-K partial store (re-read this launch) | n/a | 0 | 0 | keep in L2 for the combine phase (line 1481) |
| Accumulation atomic (return unused) | **0** | 0 | 0 | no return, counts on VScnt (lines 1508, 1705) |
| Fetch-and-op / CAS / CSUB | **1** | 0 | 0 | pre-op value required (lines 1509, 4518) |

**What the manual does not tell you.** It gives no cycle counts, latency, or throughput figures for global, buffer, or flat atomics; no contention model; no L0/GL1/GL2/MALL capacities or line sizes; and no statement of which gfx1150/gfx1151 SKUs have a MALL. LDS is the only level with published timing (1 cycle wave32 / 2 wave64 conflict-free, up to 64 under full bank conflict, line 4967). Every atomics-vs-tree-reduction and streaming-hint decision above must be validated by measurement on the target part.


## Cross-lane reductions, permutes & transposes

Softmax, layernorm/RMSNorm and GEMM epilogues are dominated by *reduction tails* — the log-step sum/max across a wave, plus the broadcast of the result back to every lane. RDNA 3.5 gives you three physically distinct paths for moving a DWORD between lanes, and picking the wrong one is the difference between a reduction that hides inside the VALU pipe and one that stalls on `s_waitcnt`:

| Path | Instructions | Where it executes | Reach | Index source |
|---|---|---|---|---|
| **DPP operand modifier** | any eligible VOP1/VOP2/VOPC/VOP3/VOP3P op + DPP8/DPP16 DWORD | VALU operand mux (fused into the arithmetic) | 8 lanes (DPP8) / 16 lanes (DPP16) | encoded in the instruction |
| **VALU permute opcodes** | `V_PERMLANE16_B32`, `V_PERMLANEX16_B32`, `V_PERMLANE64_B32`, `V_READLANE_B32`, `V_WRITELANE_B32`, `V_READFIRSTLANE_B32` | VALU, standalone instruction | 16 / 32 / 64 lanes | SGPR (wave-uniform) |
| **LDS-crossbar ops** | `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, `DS_SWIZZLE_B32` | LDS block, but **no LDS RAM access** | 32 lanes | per-lane VGPR (permute) or immediate (swizzle) |

`DS_PERMUTE_B32` (line 26082): *"This does not access LDS memory and may be called even if no LDS memory is allocated to the wave. It uses LDS to implement an arbitrary swizzle across threads in a wavefront."* `DS_SWIZZLE_B32` (line 25029): *"does not read or write the DS memory banks."* So all three DS forms consume **zero LDS capacity** (no occupancy cost) and are **immune to bank conflicts** — unlike a real LDS round-trip, which per `LDS` (line 4967) *"can complete in as little as one cycle (for wave32, or 2 cycles for wave64), or take as many 64 cycles, depending upon the number of bank conflicts."*

> **Throughput caveat.** The only timing number the manual gives for any of these is for DPP: *"DPP instructions incur an extra cycle of delay to execute"* — `DPP` (line 2828). The manual states **no** cycle counts, issue rates, or latencies for `V_PERMLANE*`, `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, or `DS_SWIZZLE_B32`. Every relative-cost statement below is an *architectural* argument (which pipe the op issues on, how many instructions it takes, what it does not consume), not a measured throughput claim. Measure on gfx1150/gfx1151 before committing to a schedule.

---

### 1. Fuse the shuffle into the reduction op: DPP costs one extra cycle, not an extra instruction

**Hardware.** DPP is not a move instruction — it is an operand modifier. Setting SRC0 to the inline constant `DPP8` or `DPP16` causes the real SRC0 VGPR address *and* a cross-lane selection pattern to come from a trailing DPP DWORD: *"DPP operations allow VALU instruction to select operands from different lanes (threads) rather than just using a thread's own data... since SRC0 is set to the DPP value, the actual VGPR address for SRC0 comes from the DPP DWORD"* — `DPP` (line 2818). The lane crossing and the arithmetic happen in one instruction; the entire cost is *"an extra cycle of delay"* — `DPP` (line 2828).

**Why it matters for LLM kernels.** A reduction step written as `shuffle; add` is two instructions; written with DPP it is one. A 4-step 16-lane butterfly is 4 VALU ops total, with no LDS traffic, no barrier, and no `s_waitcnt`. This is the entire intra-wave portion of a softmax row-max, a softmax denominator, and a layernorm mean/variance pass.

**How to use.** Attach the DPP DWORD to the reduction op itself (`V_MAX_F32`, `V_ADD_F32`, `V_DOT2_F32_F16`), never to a separate `V_MOV_B32` followed by an op. In HIP this is `__builtin_amdgcn_update_dpp` / `__builtin_amdgcn_mov_dpp` (compiler builtins, not ISA); the compiler will fold the move into the consumer when the DPP result has a single arithmetic use — check the disassembly, because a spilled or multiply-used DPP temp turns back into a real `v_mov_b32_dpp`.

The full DPP16 pattern set — this is the complete `dpp_ctrl` enumeration on RDNA 3.5, `DPP16` (line 2879) and Table 94 (lines 6978–6986):

| `dpp_ctrl` | Hex | Semantics | Reduction use |
|---|---|---|---|
| `DPP_QUAD_PERM{00:FF}` | `000–0FF` | `pix[n] = pix[(n&0x3c) + ctrl[n%4*2+1 : n%4*2]]` | arbitrary 4-lane all-to-all; 2×2 in-register transpose |
| `DPP_ROW_SL{1:15}` | `101–10F` | shift left 1–15 lanes within the 16-lane row | exclusive scan |
| `DPP_ROW_SR{1:15}` | `111–11F` | shift right 1–15 lanes within the row | inclusive scan (Kogge-Stone) |
| `DPP_ROW_RR{1:15}` | `121–12F` | rotate right 1–15 lanes within the row | wrap-around scans, cyclic staging |
| `DPP_ROW_MIRROR` | `140` | `pix[n] = pix[15-(n&0xf)]` | reverse a 16-lane row |
| `DPP_ROW_HALF_MIRROR` | `141` | `pix[n] = pix[7-(n&7)]` | reverse within 8 lanes |
| `DPP_ROW_SHARE{0:15}` | `150–15F` | `lane[n] = lane[(n&0x30) + lanesel]` | **broadcast one lane to all 16 in the row** |
| `DPP_ROW_XMASK{0:15}` | `160–16F` | `lane[n] = lane[(n&0x30) + ((n&0xf) ^ mask)]` | **XOR butterfly all-reduce** |

A "row" is 16 lanes; *"out of range means the lane offset goes outside a group of 16 lanes (e.g. 0..15, or 16..31)"* — `DPP16` (line 2891). **DPP16 never crosses a 16-lane boundary.**

---

### 2. The softmax/layernorm workhorse: `DPP_ROW_XMASK` butterfly all-reduce

**Hardware.** `DPP_ROW_XMASK` sets the fetch lane to `(current lane) XOR mask`, clamped to the row: *"Fetch lane ID is the current lane ID XOR'd with a mask specified by DPP_CTRL[3:0]"* — `DPP_ROW_XMASK` (line 6986). Running masks 1, 2, 4, 8 gives a Hillis–Steele butterfly: after 4 fused ops **every lane in the row holds the full 16-lane reduction**, not just the last one. That "all-reduce for free" property is what you want, because softmax's `exp(x - m) / d` and layernorm's `(x - µ) * rstd` need `m`, `d`, `µ`, `rstd` in *every* lane.

**Why it matters.** This replaces the `__shfl_xor`-style LDS or generic-shuffle reduction tree. Four VALU ops, no LDS allocation, no barrier, no waitcnt, and no separate broadcast phase.

**How to use** (wave32, 32-lane row max then sum):

```
; ---- 16-lane butterfly max, result replicated to all 16 lanes of each row
v_max_f32_dpp v_m, v_m, v_m  row_xmask:1  row_mask:0xf bank_mask:0xf
v_max_f32_dpp v_m, v_m, v_m  row_xmask:2  row_mask:0xf bank_mask:0xf
v_max_f32_dpp v_m, v_m, v_m  row_xmask:4  row_mask:0xf bank_mask:0xf
v_max_f32_dpp v_m, v_m, v_m  row_xmask:8  row_mask:0xf bank_mask:0xf
; ---- cross the 16-lane row boundary (see #3)
v_permlanex16_b32 v_t, v_m, s_idlo, s_idhi     ; identity lanesel = row swap
v_max_f32         v_m, v_m, v_t                ; now all 32 lanes hold the wave max
; ---- plain (non-DPP) transcendental, then the same 5-step tree with v_add_f32
v_sub_f32 v_x, v_x, v_m
v_exp_f32 v_x, v_x                             ; NO DPP here - see #7
```

Assembler spellings of the DPP bound-control bit differ between toolchains and have historically been inverted relative to the ISA `BC` bit; verify the encoded bit with `llvm-objdump` rather than trusting the mnemonic. The ISA-level rule is #6 below.

---

### 3. Crossing the 16-lane row: `V_PERMLANEX16_B32` (wave32) and `V_PERMLANE64_B32` (wave64) — and the `ROW_BCAST` that no longer exists

**Hardware.** DPP16 is hard-limited to 16 lanes, so the last step(s) of a wave-wide reduction must use a permute opcode:

| Instruction | Op | Reach | Cite |
|---|---|---|---|
| `V_PERMLANE16_B32` | 603 | *"arbitrary gather-style operation within a row (16 contiguous lanes)"* | line 20284 |
| `V_PERMLANEX16_B32` | 604 | *"arbitrary gather-style operation across two rows (each row is 16 contiguous lanes)"*; pseudocode uses `altrow = {row[1], ~row[0]}` — i.e. `1<->0, 3<->2` | lines 20338, 20356 |
| `V_PERMLANE64_B32` | 103 | *"the high half and low half of a wave64 are swapped. Performs no operation in wave32 mode"* (`altlane = {~lane[5], lane[4:0]}`, 0↔32 … 31↔63) | lines 13528, 13542 |

Both PERMLANE16 forms take their pattern from **two SGPRs concatenated into a 64-bit set of 16×4-bit lane selects**: *"the second and third source are combined into a single 64-bit value representing lane selects used to swizzle within each row"* — `V_PERMLANE16_B32` (line 20286). The pattern is therefore wave-uniform and normally a compile-time constant pair.

**⚠ Porting trap — there is no `DPP_ROW_BCAST` on RDNA 3.5.** The classic GCN/CDNA reduction epilogue (`row_shr:8` … `row_bcast:15`, `row_bcast:31`, `wave_shl`, `wave_ror`) does not exist here: the complete `dpp_ctrl` enumeration (lines 6978–6986, and the field list at line 2879) contains only `QUAD_PERM`, `ROW_SL`, `ROW_SR`, `ROW_RR`, `ROW_MIRROR`, `ROW_HALF_MIRROR`, `ROW_SHARE`, `ROW_XMASK`. Ported CDNA reduction code that reaches lanes 16–31 or 32–63 with a DPP control **must** be rewritten onto `V_PERMLANEX16_B32` / `V_PERMLANE64_B32` / `DS_SWIZZLE_B32`.

**How to use.**
- **wave32 full-wave reduce:** 4× `row_xmask` (masks 1,2,4,8) + one `V_PERMLANEX16_B32` with an identity lane-select pair (`s0=0x76543210`, `s1=0xfedcba98`) + one final `V_MAX/V_ADD`. Total 6 VALU ops, all-lanes result, zero LDS.
- **wave64 full-wave reduce:** the same 6 ops reduce each 32-lane half, then one `V_PERMLANE64_B32` + one reduction op merges the halves. In wave32 that instruction *"is translated to V_NOP and performs no writes"* — `V_PERMLANE64_B32` (line 13553) — so a wave-size-generic kernel can emit it unconditionally at the cost of one NOP slot.
- `V_PERMLANE64_B32` reads through EXEC: *"the EXEC mask of the destination lane is used as the read mask for the alternate lane; as a result this opcode may read values from disabled lanes"* — (line 13555). Its source must be a VGPR, SVGPRs are not allowed (line 13557), and ABS/NEG/OMOD must be zero (line 13559).
- `V_PERMLANEX16_B32` requires **distinct source and destination VGPRs** for the manual's own wave32-rotation recipe: *"Note for this to work, source and destination VGPRs must be different"* — (line 20373).

---

### 4. Broadcasting the reduced value back to every lane

If you used the XMASK butterfly you already have an all-reduce and need nothing here. If instead you reduced into a single lane (e.g. a strided tree, or a value that arrived from another wave), pick by scope:

| Need | Instruction | Cost model | Cite |
|---|---|---|---|
| One lane → all 16 lanes of its row | `DPP_ROW_SHARE{0:15}` fused into the consuming op | operand modifier, +1 cycle | line 6985 |
| One lane → all 32 lanes | `DS_SWIZZLE_B32` `BCASTX32` (`xor=0x00, or=thread, and=0x00`) | one DS op, no LDS RAM | line 25066 |
| Group broadcasts (16/8/4/2) | `DS_SWIZZLE_B32` `BCASTX16/8/4/2` (`and_mask = 0x10/0x18/0x1c/0x1e`) | one DS op | lines 25067–25070 |
| Vector → scalar (value is wave-uniform) | `V_READFIRSTLANE_B32` | one VALU op, frees a VGPR | lines 17363, 17391 |
| Arbitrary lane → scalar, scalar → arbitrary lane | `V_READLANE_B32` / `V_WRITELANE_B32` | one VALU op each | lines 21317, 21338 |

`V_READFIRSTLANE_B32` *"Read[s] the scalar value in the lowest active lane of the input vector register and store[s] it into a scalar register"* and *"Overrides EXEC mask for the VGPR read"* (lines 17363, 17391) — it forces lane 0 if EXEC is empty, so it is safe under full divergence. `V_READLANE_B32`/`V_WRITELANE_B32` likewise *"Override[] EXEC mask"* (lines 21336, 21357) with lane-select width `[4:0]` in wave32 and `[5:0]` in wave64 (lines 21325–21329).

**LLM use.** In attention, the per-row running max `m_i` and running sum `l_i` of the online-softmax recurrence are wave-uniform once reduced; `V_READFIRSTLANE_B32` moves them to SGPRs so the rescale factor becomes a free scalar operand on the FMA that rescales the accumulator, removing a VGPR from the innermost loop and improving the occupancy step. Same for a dequantization scale shared by a whole wave in INT8/INT4 GEMM.

---

### 5. Prefix scans without LDS: `ROW_SR` / `ROW_SL` / `ROW_RR`

**Hardware.** `DPP_ROW_SR{1:15}`: *"if ((n&0xf) >= cntl[3:0]) pix[n].srca = pix[n - cntl[3:0]].srca else use bound_cntl"* — (line 6981); `ROW_SL` is the mirror image (line 6980) and `ROW_RR` wraps within the row instead of falling off (line 6982). The manual explicitly frames DPP around this: *"A scan operation is one that computes a value per thread that is based on the values of the previous threads and possibly itself... A reduction operation is essentially a scan that returns a single value from the highest numbered active thread. A scan operation requires that the EXEC mask to be set to all 1's for proper operation. Unused threads (lanes) should be set to a value that does not change the result prior to the scan."* — `DPP` (line 2820).

**Why it matters.** Prefix sums are the backbone of MoE expert routing (computing per-expert write offsets from a one-hot count), token compaction / padding removal in variable-length attention batching, and top-k / sparsity selection. Doing them in-register avoids the LDS store+barrier+load and the associated bank tuning.

**How to use.** Kogge-Stone with offsets 1, 2, 4, 8 and `BC=1` so lanes shifted in from outside the row contribute the additive identity 0:

```
v_add_u32_dpp  v_s, v_s, v_s  row_shr:1  bound_ctrl (BC=1)
v_add_u32_dpp  v_s, v_s, v_s  row_shr:2  bound_ctrl (BC=1)
v_add_u32_dpp  v_s, v_s, v_s  row_shr:4  bound_ctrl (BC=1)
v_add_u32_dpp  v_s, v_s, v_s  row_shr:8  bound_ctrl (BC=1)   ; inclusive scan over 16 lanes
```

Note the two ISA requirements the manual states for scans: **EXEC must be all 1s**, and inactive lanes must be pre-seeded with the operation's identity (line 2820). This is a correctness precondition, not a performance hint.

---

### 6. `BC` and `FI`: the two bits that decide whether your masked reduction is correct

**Hardware.** Two independent DPP16 bits govern edge and inactive lanes:

- `BC` (bit 19) — *"Bound_ctrl is used to determine what a thread should do if its source operand is from a disabled thread or invalid input: use the value zero, or disable the write... 19==0: Do not write when source is invalid or out-of-range (DPP_BOUND_OFF); 19==1: Use zero as input if source is invalid or out-of-range (DPP_BOUND_ZERO)"* — `DPP16` (line 2876).
- `FI` (bit 18) — *"18 == 1: If the source lane is disabled, fetch the source value anyway (ignoring the bound_ctrl bit). If the source lane is out-of-range, behavior is decided by the bound_ctrl bit."* — `DPP16` (line 2877).

Table 31 (`DPP16`, lines 2882–2889) is the whole truth table:

| BC | FI | source out-of-range | source in-range but disabled | source active |
|---|---|---|---|---|
| 0 | 0 | disable write | disable write | normal |
| 1 | 0 | `Src0 = 0` | `Src0 = 0` | normal |
| 0 | 1 | `Src0 = 0` | normal | normal |
| 1 | 1 | normal | normal | normal |

DPP8 has no choice: *"DPP8 follows DPP16's 'BC = 1' behavior and assumes all source lanes are in-range"* — `DPP8` (line 2897), and the FI decision is made by choosing the `DPP8` vs `DPP8FI` inline constant: *"normal, which reads zero from lanes whose EXEC mask bit is zero, and DPP8FI, which fetches data from inactive lanes"* — (line 2895).

**Why it matters.** Every real attention kernel reduces under a non-full EXEC mask: causal masks, sliding-window masks, ragged tail tokens in the last block. `BC=1` injects **zero**, which is the identity for a sum but is catastrophically wrong for a **max** reduction over negative logits — a zero silently becomes the row max and the softmax collapses.

**How to use.**
- **Sum reductions:** `BC=1` is safe and branchless.
- **Max reductions:** do **not** rely on `BC=1`. Either pre-seed masked lanes with `-inf` and use `FI=1` so their real (identity) values are fetched, or set `BC=0` so out-of-range lanes simply skip the destination write.
- `row_mask` (31:28) and `bank_mask` (27:24) *"Appl[y] to the VGPR destination write only, [do] not impact the thread mask when fetching source VGPR data"* — `DPP16` (lines 2872, 2873). Every lane still supplies data; you can restrict *commit* to the surviving lanes of a tree step without losing sources. Leave both at `0xf` unless you specifically want partial commit.
- With `V_CMP`/`V_CMPX`: *"V_CMP and V_CMPX write the full mask, not a partial mask... 'FI' with DPP16 causes a lane to act as if it is active when supplying data, but the compare result for that lane is still zero for V_CMPX (V_CMPX with FI=1 does not turn on a lane that was off)"* — (line 2857).
- On `V_PERMLANE16_B32`/`V_PERMLANEX16_B32` the same two controls arrive through OPSEL: *"OPSEL[0] is overloaded to represent the DPP 'FI' (Fetch Inactive) bit and OPSEL[1] is overloaded to represent the DPP 'BOUND_CTRL' bit"* — (lines 20286, 20340).

---

### 7. What you may **not** fuse a shuffle into (Table 30) — plan the reduction around it

**Hardware.** *"DPP may be used only with: VOP1, VOP2, VOPC, VOP3 and VOP3P (but not 'packed math' ops)"* — (line 2828). Table 30 (lines 2830–2853) is the authoritative eligibility list. The parts that bite LLM kernels:

| Class | DPP? | Consequence for GEMM/attention | Cite |
|---|---|---|---|
| `WMMA` ops | **NO DPP** | you cannot reduce *inside* the matrix op; reduce on the F32 accumulators afterwards | line 2849 |
| `V_PK_*` packed 16-bit math | **NO DPP** | packed FP16 elementwise stages cannot carry a shuffle | line 2848 |
| `V_DOT4_I32_IU8`, `V_DOT4_U32_U8`, `V_DOT8_I32_IU4`, `V_DOT8_U32_U4` | **NO DPP** | INT8/INT4 quantized reduction paths need a *separate* permute step | lines 2844–2847 |
| `V_FMA_MIX_*`, `V_DOT2_F32_{BF16,F16}` | **Allow DPP** | the FP16/BF16→FP32 accumulate ops *can* absorb a lane fetch — use these for mixed-precision reduction trees | line 2850 |
| All 64-bit opcodes (VOP1/VOP2/VOP3/VOPC) | **NO DPP** | FP64 reductions need explicit permutes | lines 2834, 2841, 2854 |
| `V_MUL_LO_U32`, `V_MUL_HI_U32/I32`, QSAD/MQSAD | **NO DPP** | — | lines 2835–2840 |
| `V_READFIRSTLANE_B32`, `V_READLANE_B32`, `V_WRITELANE_B32`, `PERMLANE16/X16`, `PERMLANE64`, `V_SWAP_B32` | **NO DPP** | you cannot chain a DPP fetch into a permute opcode | lines 2835, 2836, 2839, 2842–2844, 2850 |
| `VOPD` (all) | **NO DPP** | **dual-issue and cross-lane are mutually exclusive** | line 2853 |
| `VINTERP` (all) | **NO DPP** | — | line 2851 |

Four more traps in the same family:

1. **No literal + DPP.** *"Literals may not be used with DPP"* — (line 2375). A reduction step that wants `× 1/N` or `+ bias` must take the constant from an inline constant (which are free and do not count against the two-scalar-value limit, line 2374) or from a preloaded SGPR/VGPR. Encoding it as a 32-bit literal is illegal.
2. **VOPD forfeits the whole reduction tail.** VOPD *"Must not use DPP"* and *"Must be wave32"* (lines 2780–2781). The dual-issue pipe that doubles your FP32 elementwise throughput is unavailable for every DPP shuffle+reduce step. Schedule accordingly: maximize VOPD in the per-lane phases (the `exp`, the rescale, the accumulate) and accept single-issue across the 5–6 op reduction tail.
3. **Transcendentals + DPP is doubly penalized.** Transcendentals fall under VOP1 "All Others → Allow DPP" (line 2840), but they already sit on a slower pipe and DPP adds *"an extra cycle of delay"* (line 2828). Structure softmax as: DPP reductions on `V_MAX_F32`/`V_ADD_F32`, then a **plain** `V_EXP_F32` / `V_RCP_F32` with no DPP DWORD.
4. **On `*REV` shifts, DPP moves the shift count, not the data.** For `V_LSHLREV_B32`, `V_LSHRREV_B32`, `V_ASHRREV_I32` the manual states flatly: *"DPP operates on the shift count, not the data being shifted"* — `V_LSHLREV_B32` (line 18988), `V_LSHRREV_B32` (line 19001), `V_ASHRREV_I32` (line 19018). INT4/INT8 pack-unpack code that tries to fuse a lane shuffle into an unpack shift silently swizzles the wrong operand. Either put the value you want shuffled in the *count* slot deliberately, or do the DPP move separately.
5. **CLAMP is silently ignored** on `V_PERMLANE*`, `V_READLANE`, `V_READFIRSTLANE`, `V_WRITELANE`, WMMA ops, and float DOT instructions (lines 2436–2444). Do not expect a free saturate on a permuted or matrix-produced value — emit an explicit `V_MED3`/min-max.

---

### 8. Data-dependent cross-lane movement: `DS_BPERMUTE_B32` (gather) and `DS_PERMUTE_B32` (scatter)

**Hardware.** When the permutation is a *runtime value* — MoE token→expert routing inside a wave, top-k reordering, a gather driven by an index tensor — no DPP pattern and no SGPR lane-select can express it. `DS_PERMUTE`/`DS_BPERMUTE` take a **per-lane VGPR index**:

- `ds_bpermute_b32`: `Dst[0..31] = src[index[0..31]]` (gather) — line 5128
- `ds_permute_b32`: `Dst[index[0..31]] = src[0..31]` (scatter) — line 5127

*"These instructions use the LDS hardware but do not use any memory storage, and may be used by waves that have not allocated any LDS space"* — `DS_PERMUTE_B32`/`DS_BPERMUTE_B32` (line 5125). **This is the key occupancy property**: an attention kernel whose LDS budget is fully spent on K/V tiles can still do arbitrary 32-lane shuffles, because these cost zero LDS bytes.

**How to use — the four rules that cause the most bugs:**

1. **The index is a byte address.** *"index values are in bytes (so multiply by 4), and have the 'offset0' field added to them before use"* — (line 5132). `ds_bpermute_b32 v_dst, v_addr, v_src` with `v_addr = src_lane << 2`.
2. **32-lane reach only, even in wave64.** *"in wave64 mode the permute operates only across 32 lanes at a time on each half of a wave64... it executes as if were two independent wave32's. Each half-wave can use indices in the range 0-31 to reference lanes in that same half-wave"* — (line 5123). And the scatter pseudocode is explicit: *"NOTE: destination lane is MOD 32 regardless of wave size"* — `DS_PERMUTE_B32` (line 26105). A wave64 kernel that indexes lanes 32–63 silently aliases onto 0–31. Cross the halves with `V_PERMLANE64_B32` instead.
3. **EXEC is honored on both ends, and disabled lanes read as zero.** *"The EXEC mask is honored for both reading the source and writing the destination... Reading from disabled lanes returns zero"* — (line 5130); restated for gather at `DS_BPERMUTE_B32` (line 26146): *"If src_lane selects a disabled thread then zero is returned."* Under a causal/window mask, a gather from a masked lane yields `0.0`, not `-inf`.
4. **Out-of-range indices wrap, they do not fault.** *"Index values out of range wrap around (only index bits [6:2] are used, the other bits of the index are ignored)"* — (line 5130).

**Prefer gather (`BPERMUTE`) over scatter (`PERMUTE`).** Scatter has a collision problem the manual flags twice, once as a warning and once as a defined-but-lossy rule: *"If multiple sources map to the same destination lane, it is not deterministic which source lane writes to the destination lane"* — `DS_PERMUTE_B32` (line 26086), while the pseudocode comment says *"If multiple sources select the same destination thread, the highest-numbered source thread wins"* — (lines 26110–26112). Either way overlapping maps drop data. A gather is always one-to-one by construction, so it cannot lose a value.

---

### 9. Index-free swizzles: `DS_SWIZZLE_B32` when the pattern is a compile-time constant

**Hardware.** `DS_SWIZZLE_B32` (op 53) encodes the entire permutation in the 16-bit `offset` immediate — **no index VGPR at all**, saving a register and the VALU ops that would compute it. *"Dword swizzle, no data is written to LDS memory... Swizzles input thread data based on offset mask and returns; note does not read or write the DS memory banks"* — (lines 25027, 25029). Reading an invalid thread returns `0x0` (line 25031), which doubles as a free additive identity.

Four modes (`DS_SWIZZLE_B32`, lines 25035–25054):

| `offset` | Mode | Mapping | Use |
|---|---|---|---|
| `≥ 0xE000` | FFT decomposition on `offset[4:0]` | bit-reversal based | butterfly stages of an FFT (line 25035) |
| `0xC000–0xDFFF` | rotate by `offset[9:5]`, left if `offset[10]==0` | `j = (i & mask) \| ((i + rotate) & ~mask)` | cyclic staging, ring shifts (lines 25042, 25089) |
| `offset[15] == 1` | full all-to-all within **groups of 4** | lane *k* of each quad reads `offset[2k+1:2k]` | quad transpose / fragment reshuffle (line 25052) |
| `offset[15] == 0` | 32-lane and/or/xor map | `j = (((i & 0x1f) & and_mask) \| or_mask) ^ xor_mask` | butterfly, reverse, broadcast (lines 25054, 25111) |

The 32-lane mode's `offset` packs `xor_mask = offset[14:10]`, `or_mask = offset[9:5]`, `and_mask = offset[4:0]` (lines 25109–25111). The manual names the useful encodings directly (lines 25056–25070):

```
SWAPX16/8/4/2/1 : xor_mask = 0x10/0x08/0x04/0x02/0x01, or_mask = 0x00, and_mask = 0x1f
REVERSEX32/16/8/4 : xor_mask = 0x1f/0x0f/0x07/0x03,    or_mask = 0x00, and_mask = 0x1f
BCASTX32/16/8/4/2 : xor_mask = 0x00, or_mask = thread, and_mask = 0x00/0x10/0x18/0x1c/0x1e
```

**Why it matters.** `SWAPX1 → SWAPX2 → SWAPX4 → SWAPX8 → SWAPX16` is a complete 32-lane XOR butterfly all-reduce that **crosses the 16-lane row boundary that DPP16 cannot**, in 5 DS ops with no index register and no LDS. `BCASTX32` is a one-instruction whole-wave broadcast. Unlike DPP, `DS_SWIZZLE_B32` issues on the LDS block, not the VALU — which is exactly what you want when the VALU is the bottleneck (a GEMM epilogue, or a softmax whose `exp` chain is saturating the transcendental pipe), and exactly what you do **not** want when the LDS pipe is already saturated by tile loads.

**How to use.** `__builtin_amdgcn_ds_swizzle(x, pattern)` in HIP takes the 16-bit pattern as a compile-time constant. Note the `and_mask`/`or_mask`/`xor_mask` fields are 5 bits and *"the offset bits apply to each group of 32 within a wavefront"* (line 25054) — in wave64 the pattern repeats independently on lanes 0–31 and 32–63, so like the permutes it does not cross the 32-lane boundary.

---

### 10. In-register transposes: `DPP_QUAD_PERM`, DPP8, and the quad swizzle mode

**Hardware.** Three primitives give arbitrary (not just butterfly) local permutations, which is what a transpose needs:

- `DPP_QUAD_PERM{00:FF}` (`000–0FF`): *"pix[n].srca = pix[(n&0x3c) + dpp_cntl[n%4*2+1 : n%4*2]]"*, described as *"Permute of four threads"* — (line 6978). Each of the 4 lanes in a quad independently names its source via a 2-bit field; the same pattern applies to every quad in the wave. This is a **free 4×4 lane gather fused into a VALU op**.
- **DPP8**: *"DPP8 allows arbitrary cross-lane swizzling within groups of 8 lanes"* — (line 2895), encoded as eight 3-bit `SEL0..SEL7` fields (lines 6994–7001, described at lines 7011–7018), where *"SEL0 selects which lane to read from to supply data into lane 0"* (line 2905). An **arbitrary 8-lane gather in one instruction**, still fused into the arithmetic.
- `DS_SWIZZLE_B32` `offset[15]==1`: the same 4-lane all-to-all, but on the LDS pipe instead of the VALU (line 25052) — useful to offload when the VALU is the critical resource.

**Why it matters for GEMM.** The `A` fragment for WMMA is stored **column-major** while `B`, `C`, `D` are row-major: *"the A matrix is column-major while the others are in row-major order"* — `V_WMMA_*` (line 3009), and the VGPR view annotates it as *"(A matrix is transposed from normal view)"* (line 3018). When a kernel loads A and B from global memory in the same layout (the common case for a row-major×row-major GEMM), one of the two operands must be transposed before the matrix op. Doing that transpose in-register with QUAD_PERM/DPP8 stages avoids an LDS store+barrier+load round-trip per K-step — which is otherwise the dominant cost in a small-K GEMM or a skinny decode-phase matvec.

**How to use.** Decompose an N×N in-register transpose into log₂N butterfly stages of "swap element *j* between lane pairs at distance *d*". For `d ∈ {1,2}` use `DPP_QUAD_PERM`; for `d ∈ {4}` use DPP8; for `d ∈ {8}` use `DPP_ROW_XMASK:8` or `DPP_ROW_HALF_MIRROR`; for `d = 16` use `V_PERMLANEX16_B32`; for `d = 32` (wave64) use `V_PERMLANE64_B32`. Each stage is a `V_MOV_B32`/`V_CNDMASK_B32` pair plus the DPP DWORD, with no LDS and no barrier. `DPP_ROW_MIRROR` (line 6983) and `DPP_ROW_HALF_MIRROR` (line 6984) give free 16-wide and 8-wide reversals for the anti-diagonal cases.

---

### 11. WMMA operand staging: replicating lanes 0–15 into 16–31 in one instruction

**Hardware.** WMMA has a hard data-layout precondition: *"These instructions work over multiple cycles to compute the result matrix and internally use the DOT instructions. In order to achieve this performance, the user must arrange the data such that: A and B matrices: lanes 0-15 data are replicated into lanes 16-31 (for wave64: also into lanes 32-47 and 48-63)"* — `V_WMMA_*` (line 2956), confirmed in the VGPR-view tables (lines 3018, 3029).

**Why it matters.** Every WMMA-based GEMM/attention kernel pays this replication on every A and B fragment. Done naively (LDS store + broadcast reload, or a `V_READLANE`/`V_WRITELANE` chain) it is 16 instructions or an LDS round-trip per fragment register. Done as a lane permute it is **one instruction per 32-bit register**.

**How to use** — two equivalent single-instruction forms:

```
; Form A: LDS-crossbar gather, no LDS storage, no EXEC manipulation
v_and_b32      v_idx, 15, v0          ; v0 = lane id
v_lshlrev_b32  v_idx, 2, v_idx        ; byte index = (lane & 15) * 4
ds_bpermute_b32 v_frag, v_idx, v_frag ; dst[i] = src[i & 15]  -> lanes 16-31 mirror 0-15
s_waitcnt lgkmcnt(0)

; Form B: VALU permute, follows the manual's own cross-row recipe
s_mov_b32 exec_lo, 0xffff0000         ; write only lanes 16-31
v_permlanex16_b32 v_frag, v_src, s_idlo, s_idhi fi   ; FI=1 to read the disabled low row
s_mov_b32 exec_lo, -1
```

Form A's semantics follow directly from `DS_BPERMUTE_B32` (lines 5128, 5132, 26144). Form B follows the structure of the manual's documented wave32-rotation example, which sets EXEC to select which lanes fetch from the other row and sets FI *"needed for lanes 15 and 31"* to read across the boundary — `V_PERMLANEX16_B32` (lines 20373–20398); note its requirement that source and destination VGPRs differ (line 20373).

**Pick by pipe pressure**: Form A costs one DS slot and an `s_waitcnt lgkmcnt`, Form B costs VALU slots plus two EXEC writes and cannot be dual-issued. In a WMMA inner loop the VALU is usually the scarce resource, so Form A is generally the better trade — but the manual gives no cycle counts for either, so measure.

---

### 12. Scheduling rules around lane-crossing ops

Four hazards/scheduling facts that are cheap to obey and expensive to discover:

1. **`V_PERMLANE` cannot immediately follow `V_CMPX`.** *"V_PERMLANE may not occur immediately after a V_CMPX. To prevent this, any other VALU opcode may be inserted (e.g. V_NOP)"* — `V_PERMLANE*` (line 2506). This pattern is *exactly* what a masked attention reduction looks like (`v_cmpx` to build the causal/window mask, then a permute to reduce). Budget one filler slot, or reorder so useful VALU work lands in the gap.
2. **DS shuffles cost a `waitcnt`, DPP does not.** `DS_PERMUTE_B32`/`DS_BPERMUTE_B32`/`DS_SWIZZLE_B32` are DS-encoded instructions issued to the LDS block; LGKMcnt counts *"LDS indexed operations"* — `S_WAITCNT` (lines 1706–1707) — so their results must be awaited before use. DPP has no counter at all; it retires in the VALU with *"an extra cycle of delay"* (line 2828). For a short reduction tail on the critical path, DPP wins on latency; for a long shuffle sequence that can overlap with independent VALU work, the DS path wins by moving traffic off the VALU. Do not interleave a `s_waitcnt lgkmcnt(0)` for a swizzle with an unrelated LDS tile load — you will serialize the tile load too.
3. **Back-to-back dependent WMMA needs a filler, and a permute is a legal one.** *"Back-to-back dependent WMMA instructions require one V_NOP (or independent VALU op) between them if the first instruction's matrix D is the same or overlaps with the second instruction's matrices A or B"* — `V_WMMA_*` (line 3050). A `V_PERMLANE16_B32` or DPP move staging the *next* fragment is a productive substitute for that `V_NOP`.
4. **Two SGPRs is the whole scalar budget on a permlane.** `V_PERMLANE16_B32`/`V_PERMLANEX16_B32` consume two SGPRs for the lane-select pair, and *"Instructions may use at most two Scalar Values"* — (line 2372). You cannot additionally reference a third SGPR or a literal on the same instruction. Hoist the lane-select constants into a fixed SGPR pair outside the loop.

---

### Summary: choosing a cross-lane primitive

| Situation | Use | Reason |
|---|---|---|
| Softmax/layernorm all-reduce, ≤16 lanes | `DPP_ROW_XMASK` masks 1,2,4,8 fused into `V_MAX_F32`/`V_ADD_F32` | 4 fused ops, all-lanes result, no LDS, no waitcnt (lines 6986, 2828) |
| Crossing the 16-lane row in wave32 | `V_PERMLANEX16_B32` | DPP16 cannot cross a row (lines 20338, 2891) |
| Crossing the 32-lane half in wave64 | `V_PERMLANE64_B32` | only op that reaches lane `n^32`; NOP in wave32 (lines 13528, 13553) |
| 32-lane butterfly with the VALU already saturated | `DS_SWIZZLE_B32` `SWAPX1..16` | runs on the LDS block, needs no index VGPR (lines 25056–25060) |
| Broadcast a reduced scalar to the wave | `DS_SWIZZLE_B32` `BCASTX32`, or `V_READFIRSTLANE_B32` to SGPR | one op; SGPR form also frees a VGPR (lines 25066, 17363) |
| Runtime/data-dependent index (MoE routing, top-k, gather) | `DS_BPERMUTE_B32` | only primitive taking a per-lane VGPR index; zero LDS allocation (lines 5125, 5128) |
| Small in-register transpose / fragment reshuffle | `DPP_QUAD_PERM`, DPP8, `DS_SWIZZLE` quad mode | arbitrary gather within 4 or 8 lanes, fused into a VALU op (lines 6978, 2895, 25052) |
| WMMA A/B lane 0–15 → 16–31 replication | `ds_bpermute_b32` with `idx=(lane&15)*4` | one instruction per register, no LDS storage (lines 2956, 5128) |
| Prefix sum (token compaction, expert offsets) | `DPP_ROW_SR` 1,2,4,8 with `BC=1` | Kogge-Stone in 4 fused adds (lines 6981, 2820) |
| Reduction over INT8/INT4 dot products | separate permute, **then** `V_DOT4/V_DOT8` | integer DOTs are `NO DPP` (lines 2844–2847) |
| Reduction over FP16/BF16 accumulating in FP32 | DPP fused into `V_DOT2_F32_F16` / `V_FMA_MIX_F32` | these *do* allow DPP (line 2850) |


## LDS layout & bank conflicts

The local data share is the software-managed staging buffer every GEMM, attention, layernorm, and softmax kernel funnels its A/B/K/V tiles through. On RDNA 3.5 its throughput is set almost entirely by one thing — whether a wave's lane addresses land in distinct banks. Get the layout right and a tile read is effectively free (1 cycle); get it wrong and the same read costs up to 64 cycles.

### Bank geometry and the conflict cost model

LDS is **128 kB per WGP, organized as 64 DWORD-wide banks**, each a 512×32 two-port RAM (1 read + 1 write per clock). The 64 banks are split into **two 32-bank halves, each tied to one SIMD32 pair**; DWORDs are striped serially across banks, so `bank = (byte_addr / 4) mod 32` within a half (`4697`). All 32 banks of a half can service one access per cycle, so a wave32 op whose 32 lanes hit 32 distinct banks completes in **1 cycle (2 cycles for wave64)**. Any set of lane addresses that lands two or more accesses in the same bank in one instruction is serialized one-per-cycle — a full 32-way collision stretches that op to **as many as 64 cycles** (`4967`). This is a hard cycle cost, not an abstract bandwidth figure: a pathological transpose read can be 64× slower than the conflict-free case.

Because consecutive DWORDs map to consecutive banks and the pattern repeats every 32 DWORDs *per half*, the classic failure mode is a tile whose leading dimension is a multiple of 32 DWORDs: a column read then maps every lane to the same bank.

**How to avoid it — pad the leading dimension.** Size each LDS tile row so its stride, modulo the 32-bank period, spreads the dominant access across all banks:

```
// 32-wide FP32 tile, column reads by 32 lanes:
//   stride = 32 DWORDs -> all lanes hit one bank (64-cycle read)
//   stride = 33 DWORDs -> lane i hits bank (i*33 mod 32) = distinct banks (1-cycle read)
__shared__ float tileA[K][32 + 1];   // +1 DWORD pad; +4 if you read B128 vectors
```

An XOR-swizzle of the LDS index (`addr ^= (row & mask)`) achieves the same distribution without the padding overhead when the extra column would waste too much of the 64 kB budget. Validate that the per-lane address stride modulo 32 is coprime-ish with 32 for whichever access — row store vs. column/transpose read — dominates. Note only 32 banks serve each access, so even a work-group spanning the whole WGP gets 32-bank (not 64-bank) parallelism per op; prefer **wave32** for LDS-heavy inner loops to hit the 1-cycle floor rather than wave64's 2-cycle minimum.

### Move more per instruction: wide, dual-address, and strided DS ops

Feeding WMMA fragments from LDS is issue-bound as much as bandwidth-bound. The LDS can run 32 concurrent 32-bit accesses, and the *extended* forms move 64 bits per lane per instruction (`4699`). Use the widest DS op that matches your fragment to cut instruction count 2–4×:

| Instruction | Moves | Use for |
|---|---|---|
| `DS_LOAD_B64 / B96 / B128` | 2 / 3 / 4 DWORDs per lane | vectorized fragment loads; keep B128 16-byte aligned so it issues as one transaction |
| `DS_LOAD_2ADDR_B32 / B64` (`5013`) | two independent addresses, one issue slot | paired A/B fetches, double-buffered reads, two K-slices |
| `DS_LOAD_2ADDR_STRIDE64_B32 / B64` (`5014`) | two addresses, each `offset × 64` | a value and its full-wave-stride (32 lanes × 2) neighbor without extra address math |

The 2ADDR address formula is `LDS_BASE + VGPR[ADDR] + InstOffset*ADJ`, with `ADJ = 4` for ≤32-bit data and `ADJ = 8` for 64-bit; the two 8-bit offsets share one base VGPR (`5039`). One caveat specific to bank conflicts: pick the two 2ADDR offsets so the two accesses land in **different banks** to avoid an in-instruction self-conflict. The STRIDE64 forms space accesses exactly 256 bytes apart — one full bank period — so the two offsets alias the same bank column; rely on the per-lane *base* addresses (not the two offsets) to spread across banks there.

For F16/BF16/INT8 tiles feeding WMMA, stage packed data directly with the **D16 forms** — `DS_LOAD_U16_D16` then `DS_LOAD_U16_D16_HI` write the low and high halves of one VGPR, building the packed 2×16 operand WMMA expects with no pack/unpack VALU and half the LDS footprint (`5023`).

**Address-VGPR-free staging.** For the common contiguous per-lane copy where lane *i* touches `base + i·4`, use `DS_LOAD_ADDTID_B32 / DS_STORE_ADDTID_B32`: the address is `LDS_BASE + {InstOffset} + TID·4 + M0` computed in hardware, so no VGPR holds the address (`5047`). This frees a register for accumulators *and* the `TID·4` stride is inherently one-per-bank — conflict-free by construction. Keep M0 DWORD-aligned.

### Cross-lane shuffles that never touch a bank

Many "LDS" data movements — softmax max/sum reductions, layernorm partials, small register-level transposes — are pure lane permutations that do not need storage. `DS_PERMUTE_B32` (scatter), `DS_BPERMUTE_B32` (gather), and `DS_SWIZZLE_B32` route data through the LDS crossbar **without reading or writing the banks**, so they carry zero bank-conflict risk and can run even in waves that allocated no LDS (`5123`). Prefer `DS_BPERMUTE_B32` (deterministic gather; index is a per-lane byte address, `lane×4`, bits `[6:2]` used) over an `DS_STORE`+`DS_STORE` round-trip for intra-wave reductions. In wave64 these operate as two independent 32-lane halves, so confine indices to 0–31 per half. `DS_SWIZZLE_B32` additionally gives index-free quad/butterfly/broadcast patterns (BCASTX32 splats lane 0 to all 32; xor-masks build tree-reduction steps) driven purely by the immediate.

The **64 integer atomic units** — one per bank — let work-group reductions (softmax denominators, histogram/quant counters) run in the LDS hardware without occupying the VALU; spread accumulator addresses across banks so the 64 units run in parallel, and use the non-return `DS_ADD_*` forms when the pre-op value is unused (`4713`, `5085`). LDS `DS_ADD_F32` rounds round-to-nearest-even and follows MODE for denorm flushing (unlike the cache Add_F32 which hardwires flush).

### LDS allocation vs. occupancy

LDS is the **second occupancy knob after VGPRs**. It is allocated per work-group in **1 kB blocks, 0–64 kB max** (a group using 1025 bytes consumes two blocks) (`824`). LDS-limited occupancy is `floor(available_LDS / rounded_LDS_per_group)`; compare it against the VGPR-limited wave count and take the minimum as your real occupancy. Size double-buffered A/B tiles and softmax scratch to whole-kB multiples just under a block boundary.

Two structural facts constrain layout:

- **Physical split.** LDS is two 64 kB blocks, CU0 = bytes 0–65535, CU1 = 65536–131071 (`826`). In **CU mode** each wave's allocation stays on its own side and the two 32-bank halves run in parallel — higher aggregate bandwidth, but upper waves cannot read the lower half. In **WGP mode** LDS is one contiguous pool all four SIMD32s can address (needed for a single >64 kB-reach or cross-half-shared tile), at the cost of that parallelism; note `LDS_PARAM_LOAD`/`LDS_DIRECT_LOAD` are unsupported in WGP mode (`581`, `585`). Choose CU mode for per-wave/pair-local staging (GEMM double-buffers), WGP mode only when the whole work-group must share one large tile. For latency-sensitive buffers, offset the allocation so a hot structure stays within one 64 kB half rather than straddling byte 65536.

- **Direct global→LDS DMA exists, but the manual omits it.** The overview prose mentions a load-into-LDS path (`4711`), and the manual's MUBUF/GLOBAL/SCRATCH opcode tables appear to expose only load-to-VGPR forms — but that is a gap in the prose conversion, not the hardware. The machine-readable XML defines 17 direct-to-LDS opcodes that land exactly in the tables' numeric holes: `BUFFER_LOAD_LDS_{U8,I8,U16,I16,B32,FORMAT_X}` (MUBUF 45–50), `GLOBAL_LOAD_LDS_{ADDTID_B32,U8,I8,U16,I16,B32}` (42, 45–49) and `SCRATCH_LOAD_LDS_{U8,I8,U16,I16,B32}` (45–49). They carry no VDST/VDATA operand and take an implicit `M0` source. **Caveat:** every variant is at most `B32` — there is no `_LDS_B64/B96/B128` — so the DMA path moves 4 B/lane against 16 B/lane for `GLOBAL_LOAD_B128`. It saves VGPRs and the second hop, not raw bytes-per-instruction. The register round-trip (`GLOBAL_LOAD_Bxx` → `DS_STORE_Bxx`, double-buffered) remains the higher-bandwidth choice for wide tiles; prefer the LDS-DMA form when VGPR pressure, not bandwidth, is the binding constraint. See the memory-movement section for details. (Do not confuse these with `LDS_DIRECT_LOAD`, which reads LDS→VGPR for pixel-shader parameters — the opposite direction, and not a DMA.)

For byte-granular quantized (INT8/INT4) staging at non-native offsets, set `SH_MEM_CONFIG.alignment_mode = UNALIGNED`; otherwise LDS silently zeroes the low address bits, and a partially out-of-range multi-DWORD read returns **all zeros**, not partial data — a silent-corruption trap for boundary tiles. Atomics always require natural alignment regardless of mode.

The manual does not give a peak GB/s figure for LDS; reason in the cycle terms above (≈1–2 cycles conflict-free per DS op vs. up to 64 under full conflict) when scheduling LDS traffic around the WMMA/VALU pipeline.

## Scheduling, clauses, waitcnt & dual-issue

RDNA 3.5 resolves most data hazards in hardware, so `S_NOP` padding is never required for correctness (`S_NOP` line 1686). What the compiler *does* control is latency hiding: four wait counters, a zero-cost ALU-dependency hint, arbiter clauses, wave priority, and — the single biggest wave32 throughput lever — dual-issue VALU. These are the tools that keep the WMMA/DOT pipe and the memory pipe from stalling in GEMM, attention, and elementwise kernels.

### s_waitcnt: four independent counters for software pipelining

The wave tracks outstanding memory in four separate counters; instructions of one class complete **in issue order** but **out of order** across classes (`S_WAITCNT` line 1697). Because same-class loads retire in order, `s_waitcnt vmcnt(N)` reliably means "all but the newest N loads have landed" — the foundation of double/triple-buffered pipelines: issue B tile loads, compute on the oldest while B−1 stay in flight, step the count down as each is consumed. Never blanket-wait `vmcnt(0)` unless all B are needed at once (`S_WAITCNT` lines 1722–1725).

| Counter | Tracks | Wait when |
|---|---|---|
| `VMcnt` | VMEM/global/scratch/flat **loads**, image samples, **atomics-with-return** (`S_WAITCNT_VMCNT` line 1704, 10808) | before consuming loaded data |
| `VScnt` | VMEM **stores** and atomics-**without**-return (`S_WAITCNT_VSCNT` lines 1704, 8817) | before reusing a store's source VGPRs / at a fence |
| `LGKMcnt` | LDS indexed ops, **SMEM** loads, GDS, messages (line 1706) | before consuming an LDS-staged tile or a scalar constant |
| `EXPcnt` | LDS param/direct-load, exports (line 1713) | before consuming LDS-direct data |

Key consequences:

- **Loads and stores are decoupled.** An inner loop that reads the next tile (`VMcnt`) while writing back the previous result (`VScnt`) never falsely serializes the write-back against the next compute step. Consume loads via `VMcnt`; drain stores via `S_WAITCNT_VSCNT` only where ordering demands it (line 1704).
- **A non-returning atomic bumps `VScnt`, not `VMcnt`.** Fire-and-forget accumulation (`GLOBAL_ATOMIC_ADD_F32` with `GLC=0`) never forces a `VMcnt` wait on the compute path — it is not tracked by the load counter at all (line 10808).
- **SMEM completes fully out of order.** The threshold trick is invalid for scalar loads; the only safe wait is `s_waitcnt lgkmcnt(0)`, which drains *every* pending SMEM. Batch all prologue `S_LOAD`s up front, then one `lgkmcnt(0)`, and overlap that latency with VGPR-address setup (`S_WAITCNT_LGKMCNT` line 1708; SMEM lines 3162, 3168).
- **FLAT is a waitcnt trap.** A `FLAT_*` op issues to *both* the LDS and global paths, so it increments **both** `VMcnt`/`VScnt` **and** `LGKMcnt`, and the two return out of order — "the only sensible S_WAITCNT value to use after Flat instructions is zero" (`FLAT` lines 4528, 4547). Use `GLOBAL_*` whenever the pointer is provably global so the counters stay independent and fine-grained overlap survives (`GLOBAL_LOAD_B128` line 4419).
- **VMEM address/data VGPRs are read at issue.** A later VALU write to a load's address or store-data register is hardware-stalled until the read drains (`GLOBAL_LOAD_B128` line 4285). Give the memory pipe a few instructions before clobbering those registers, or allocate the next iteration's address into distinct VGPRs.

`S_ENDPGM` implicitly executes `s_waitcnt 0`/`vscnt 0`, so a trailing full-drain wait before program end is redundant (line 11066).

### S_DELAY_ALU: zero-cost dependency-stall hints

When a VALU op reads a result produced ≤4 VALU instructions earlier and no independent work fills the gap, `S_DELAY_ALU` tells the arbiter exactly how far back the producer was, so it inserts idle cycles up front instead of discovering the hazard at issue and stalling the ALU pipe (which would waste cycles other waves could use). It **executes in zero cycles** — it co-issues with the instruction before it — and it is correctness-neutral, so add it only where a genuine short producer→consumer gap exists (`S_DELAY_ALU` lines 1740, 10757). Prefer interleaving real independent work first; delays do useful nothing (line 1736).

One `S_DELAY_ALU` packs **two** dependencies via a skip field (`InstID0`, `Skip`, `InstID1`), and a later one **replaces** rather than accumulates the previous unconsumed hint (lines 1738, 1775). The dependency *class* matters — the `DEP` codes distinguish producer types, and using the wrong one under-delays:

| Producer | `DEP` code | Note |
|---|---|---|
| ordinary VALU, 1–4 back | 1–4 (`VALU_DEP_*`) | FMA/add/mul chains |
| transcendental, 1–4 back | 5–7 (`INSTID_TRANS32_DEP_*`) | `V_EXP/RCP/RSQ/LOG` — longer latency, separate scoreboard (lines 1787, 10701) |
| SALU feeding VALU | 9–11 (`INSTID_SALU_CYCLE_*`) | 1–3 cycle wait; scalar FP is 4-cycle (lines 1789, 2077) |

`INSTID` counts *issued* VALU ops: EXEC==0 instructions still count (scoreboard marks ready immediately), branched-over ones do not (line 1771). Under wave64 the pass count (1 vs 2) is EXEC-dependent, so the hint encodes the dependency *type* and lets hardware pick the right delay (line 1736) — a reason wave32's deterministic single-pass timing is easier to schedule.

`S_DELAY_ALU` must appear **before** an `S_CLAUSE` (the instruction after `S_CLAUSE` defines the clause type) and is **illegal inside a VALU clause** — structure the clause's ops far enough apart instead (lines 10658, 10765, 1578).

### Instruction clauses (S_CLAUSE)

`S_CLAUSE` locks the instruction arbiter onto this wave for an uninterrupted run of same-type instructions, even if that leaves execution units idle, overriding the default cross-wave interleaving (line 1538). For a memory clause this batches issue so the memory pipe stays saturated with one wave's addresses, and **overlapping loads within the clause are cached against each other** — redundant/overlapping fetches hit the clause's cached data (lines 475, 1554). Wrap a tile's `GLOBAL_LOAD_B128` burst (or the LDS-store phase) in one clause; keep it homogeneous — no `S_WAITCNT` and no mixed types inside.

- **Length** is `SIMM16[5:0]+1`, valid range **2–63** instructions (`SIMM16[5:0]` must be 1–62) (line 10640).
- **`BREAK_SPAN`** (`SIMM16[11:8]`, ≤15): nonzero lets the arbiter break the clause every N instructions so other waves interleave (line 10644). Set it (e.g. 8) when many waves are resident to recover cross-wave latency hiding; set 0 for a maximal single-wave burst at low occupancy.
- A VALU clause with EXEC==0 at entry is ignored (no wasted lock); once started it runs to its last instruction even if EXEC goes zero mid-clause (line 1589).

### VOPD / V_DUAL dual-issue — the wave32 throughput lever

`VOPD` encodes two independent VALU ops (an X-op and a Y-op) that execute in parallel, roughly doubling VALU issue rate. It is **wave32-only** — "hardware does not function correctly" otherwise, and it is skipped for wave64 (`VOPD` line 2751). This is the decisive argument for compiling ALU-bound GEMM epilogues, dequant, activation, softmax, and layernorm kernels as wave32: wave64 forfeits the second pipe entirely, and separately issues every VALU/VMEM op twice (line 511).

The two ops must be independent, and satisfying the **VGPR-bank / port rules is mandatory — the pack silently malfunctions if broken** (line 2775):

- `SRCX0` and `SRCY0` must be in **different VGPR banks** (bank = `SRC[1:0]`); `VSRCX1` and `VSRCY1` likewise (lines 2771–2772).
- The two destination VGPRs must be **one even + one odd** (line 2778).
- If both ops use the `SRC2` port — which is `FMAC_F32`/`DOT2ACC_F32_{F16,BF16}` destination-accumulate, or `FMAMK_F32`'s second input — the two `SRC2` registers must be **opposite parity (one even, one odd)** (line 2774).
- At most one literal (or a shared literal); **no DPP** in a VOPD pair (lines 2777, 2780).

A register allocator that bank-collides accumulators leaves dual-issue on the table, so pair-hot accumulators must be laid out deliberately. Cross-lane reductions and dual-issue therefore cannot combine in one instruction (DPP forbidden) and must be scheduled around each other.

**Eligible opcodes** (both slots unless noted): `V_DUAL_FMAC_F32`, `FMAAK_F32`, `FMAMK_F32`, `MUL_F32`, `ADD_F32`, `SUB/SUBREV_F32`, `MUL_DX9_ZERO_F32`, `MOV_B32`, `CNDMASK_B32`, `MAX_F32`, `MIN_F32`, `DOT2ACC_F32_F16`, `DOT2ACC_F32_BF16`; Y-slot adds `ADD_NC_U32`, `LSHLREV_B32`, `AND_B32` (X- and Y-opcode tables, lines 6923–6937). **Transcendentals cannot dual-issue** — there is no VOPD encoding for `V_EXP/RCP/RSQ/LOG` — so each exp/reciprocal occupies a full un-pairable slot; only the scale/reciprocal-multiply math around it packs (line 17197). Two consequences for LLM math:

- `V_DUAL_DOT2ACC_F32_{F16,BF16}` appears in **both** slots (lines 6923–6933), so one VOPD advances two independent FP16/BF16 dot-accumulate chains — effectively 4 FP16 MACs per issued instruction into F32 accumulators. This is the packed-reduction path for GEMV/batch-1/ragged shapes where WMMA's 16×16 tile granularity and lane-replication overhead don't amortize. It uses the shared `SRC2` port, so the two accumulator VGPRs (and the VDSTs) **must be opposite parity** (line 2778).
- Pairing address arithmetic (`ADD_NC_U32`/`LSHLREV_B32`/`AND_B32`) into the Y-slot lets integer offset math ride alongside an FP `FMAC` for free.

```
; wave32 dual-accumulator FP16 K-reduction, one VOPD per two MACs
;   a0/a1 packed 2xF16 activations, w0/w1 packed 2xF16 weights
;   acc_even (V.even) and acc_odd (V.odd)  <- opposite parity, distinct banks
v_dual_dot2acc_f32_f16  acc_even, a0, w0  ::  v_dual_dot2acc_f32_f16  acc_odd, a1, w1
```

### S_SETPRIO and sleep/wake for arbiter control

`S_SETPRIO SIMM16[1:0]` sets user priority 0 (low) – 3 (high); effective priority is `{MIN(3, SysPrio+UserPrio), WaveAge}`, and `SysPrio` cannot be changed from the wave (`S_SETPRIO` line 11098). Bracket a latency-critical WMMA/reduction region with `S_SETPRIO 3` before and `S_SETPRIO 0/1` after, so the wave grabs the ALU during its hot phase and yields issue slots to compute-ready siblings while it stalls on memory. Use sparingly — it can starve peers.

For producer/consumer handoffs, prefer `S_SLEEP` (deschedules the wave for ~64·N clocks, freeing issue slots) over a busy spin, and have the producer issue `S_WAKEUP` after the store to wake same-workgroup waiters early; a missed ping is race-safe because the sleeper still resumes on its timer (`S_SLEEP`/`S_WAKEUP` lines 1535, 11089).

At the store-heavy kernel tail, `S_SENDMSG` with the dealloc-VGPRs message (0x03) frees this wave's VGPRs before `S_ENDPGM` so a successor wave can allocate them during the store-drain wait — a one-way action (no reallocation, only termination follows) that overlaps drain latency with the next wave's startup (line 1634).

All citations verified. Writing the section.

## Occupancy & wave width

Occupancy — the number of waves resident per SIMD — is the primary latency-hiding lever for LLM kernels: the GPU covers HBM and WMMA latency by keeping many waves in flight and overlapping their compute with other waves' memory access (`overview` line 457). On RDNA 3.5 (gfx1150/gfx1151) only two resources actually gate occupancy — **VGPRs** and **LDS** — plus three structural WGP ceilings. SGPRs do not. Everything below is about spending those two resources, and about the wave32-vs-wave64 choice that reshapes both the register math and the issue-rate math.

### VGPR allocation is a step function, not linear

VGPRs are handed out in coarse blocks, so register pressure crosses occupancy tiers in discrete jumps, not smoothly (`VGPR allocation` line 754):

| Wave size | Block on 1024-VGPR SIMD | Block on 1536-VGPR SIMD | Max VGPRs |
|---|---|---|---|
| wave32 | 16 | 24 | 256 |
| wave64 | 8 | 12 | 256 |

Occupancy is `floor(VGPRs_per_SIMD / round_up_to_block(kernel_VGPRs))`. A GEMM tile that spills from 32 to 33 VGPRs in wave32 allocates a full extra block (48), silently losing an occupancy step for one register. **Size register-resident accumulator tiles to land just under a block boundary** (e.g. 32, 48, 64, 96, 128 on 1024-VGPR parts; multiples of 24 on 1536-VGPR parts). Per-lane the granularity is identical between wave sizes (wave64 blocks are half the register count but cover twice the lanes); the practical difference is that wave32's finer 16-register step exposes more occupancy tiers to hit.

Two levers reduce the pressure that sets that tier:
- **Pack 16-bit data two-per-VGPR.** A 32-bit VGPR holds two independently addressable halves (`V0.L`/`V0.H`); 16-bit VALU can target either, and D16 loads (`BUFFER_LOAD_D16_B16` / `DS_LOAD_U16_D16` + `_HI`) land FP16/BF16 pairs directly into both halves — halving the footprint of activation/weight tiles (line 2613). Note the encoding cap: VOP1/2/C-form 16-bit ops address only 256 of the 16-bit VGPRs; use VOP3/VOP3P/VINTERP (OPSEL selects the half) to reach all 512 when 16-bit register pressure is the limiter (line 2631).
- **Free VGPRs early at the store tail.** A store-heavy epilogue (writing C/O tiles to HBM) can issue `S_SENDMSG` with the Dealloc-VGPRs message (0x03) immediately before `S_ENDPGM`, returning its whole VGPR block to the pool while it waits on write-confirms so the next wave allocates and launches during the drain (line 756, line 1634). One-way: after dealloc the wave may only terminate, so no VGPR may be read past that point.

### SGPRs are free — offload uniforms off the VGPR file

Every wave gets a fixed 106 SGPRs + VCC + 16 TTMP regardless of use (`SGPR` line 611); there is no variable SGPR allocation that trades against wave count. **Do not spend effort minimizing SGPR usage for occupancy** — only VGPRs and LDS gate residency. Instead, aggressively hoist every wave-uniform value (tile base pointers, leading dimensions, K-loop bounds, dequant scales/zero-points, softmax scale) into SGPRs via `S_LOAD`/`S_BUFFER_LOAD` through the constant cache — this relieves the VGPR pressure that actually limits occupancy, and uniform FP math can even run on the scalar ALU (`S_ADD_F32`, `S_MUL_F32`, `S_FMAC_F32`) concurrently with the VALU. Two SGPR subtleties matter in wave64: VCC physically occupies SGPR 106/107 and counts against an instruction's scalar-source limit (line 999), and a per-lane mask (CNDMASK mask, add carry-in) reads two consecutive SGPRs versus one for a broadcast scalar (line 2494) — wave32 halves that mask bookkeeping (line 987).

Address VGPRs are a hidden VGPR cost you can eliminate: GLOBAL `SADDR` mode puts the base in an SGPR and needs only a 32-bit offset VGPR per lane (line 4457), and `GLOBAL_LOAD_ADDTID_B32` / `DS_LOAD_ADDTID_B32` derive the address entirely from SGPR base + `laneID*4` with **no address VGPR at all** (line 4557, line 5048) — reclaim those registers for accumulators.

### Wave32 vs wave64

The SIMD is physically 32-wide, so a wave64 serializes every **VALU and VMEM** instruction into two 32-lane passes — two issue slots per instruction — while scalar ALU, scalar memory, branch, and message ops issue once regardless of wave size (line 511). Because LLM inner loops (FMA/WMMA/VMEM) are dominated by exactly the instructions that double, **wave32 is the right default for compute- and issue-bound kernels.** The asymmetry compounds:

| Factor | wave32 | wave64 | Line |
|---|---|---|---|
| VALU/VMEM issue slots per instruction | 1 | 2 | 511 |
| Dual-issue VOPD (2 VALU ops/instr) | available | illegal/skipped | 2751 |
| SGPR/VCC-writing VALU (e.g. `V_CMP`) | 1 pass | 2 passes, never EXEC-skipped | 662 |
| Conflict-free LDS indexed/atomic op | 1 cycle | 2 cycles | 4967 |
| Max work-items per 1024-item group | 32 waves | 16 waves | 573 |

The decisive throughput argument is **VOPD**: it packs two independent VALU ops into one wave32 instruction and is simply unavailable to wave64, so dequant, activation/elementwise, GEMM epilogue, and softmax pre/post-scale math can approach 2 ops/issue only in wave32 (line 2751; realizing it needs bank-disjoint operand allocation — see the scheduling section). Wave64 also issues `V_CMP`/SGPR-writing VALU twice and cannot EXEC-skip them, penalizing reduction-heavy softmax/layernorm code (line 662).

Reach for wave64 only when you can exploit its width: fewer wave-management overheads, 64-wide reductions/permutes, or divergence you can align to 32-lane halves. When an entire 32-lane half's EXEC is all-zeros, hardware skips issuing that half — but not both halves of a VMEM op unless no VMEM is outstanding, and never either half of a VALU that writes an SGPR (line 513). So masked/padded/causal-tile code recovers wave64 issue bandwidth only when divergence is 32-lane-aligned.

**Register footprint is the one place wave64 helps register-bound WMMA kernels.** A 16×16 F32 accumulator occupies 8 VGPRs in wave32 (2 rows/VGPR, V0–V7) but only 4 in wave64 (4 rows/VGPR) (line 3040), so a purely accumulator-pressure-limited GEMM can gain occupancy in wave64. Weigh that against forfeiting VOPD and paying 2× on every VALU. Note the WMMA A/B replication requirement is occupancy-neutral: lanes 0-15 must be replicated into 16-31 (and into 32-47, 48-63 for wave64), but total A/B VGPRs are identical either way — pick wave size for issue rate and accumulator footprint, never for WMMA operand cost (line 2954).

### LDS is the second occupancy knob

LDS is allocated per work-group in 1 KB blocks, 0–64 KB (line 824); LDS-limited occupancy is `floor(available_LDS / round_up_1KB(LDS_per_group))`. Round staging/double-buffer buffers to whole-KB multiples — 1025 bytes costs two blocks. The WGP's 128 KB LDS is physically two CU-affiliated 64 KB halves (bytes 0–65535 → CU0, 65536–131071 → CU1) (line 826), and **CU mode** splits it so each SIMD-pair sees only its half — "both halves run in parallel" for higher aggregate bandwidth, at the cost that a work-group cannot share across the split and you budget against a 64 KB half, not the full pool (line 581). Choose CU mode for per-wave/pair-local tiles; choose WGP mode only when one large tile must be visible to all 4 SIMD32s (and note `LDS_PARAM_LOAD`/`LDS_DIRECT_LOAD` are unsupported in WGP mode).

Compute all three limits and take the **minimum** as your real occupancy — structural caps bind before you expect:

```
occ = min( floor(VGPRs_per_SIMD / vgpr_block_rounded),   # VGPR-limited
           floor(avail_LDS      / lds_1KB_rounded),       # LDS-limited (per CU-half in CU mode)
           32 work-groups/WGP, 1024 work-items/group )    # structural (16 wave64 / 32 wave32)
```

A WGP holds at most 32 work-groups and 1024 work-items/group = 32 wave32s or 16 wave64s (line 573); many small groups sharing little LDS can hit the 32-group cap first, not the LDS cap. Two structural bonuses for one-wave-per-tile designs (common in decode/persistent attention): single-wave work-groups **do not count against the 32-group limit** and allocate no barrier resource — `S_BARRIER` becomes a free `S_NOP` (line 575).

## Transcendentals & softmax/activation math

Every transcendental on RDNA 3.5 is a **single-issue op on a dedicated TRANS32 pipe** — one instruction per element, base-2 for exp/log, IEEE-rules for rcp/rsq/sqrt. There is no polynomial expansion to write and no library call; the entire cost model for softmax, GELU, and normalization is "one TRANS32 op per element, plus the cheap arithmetic you fuse around it."

### The core primitives

| Op | Result | F32 acc. | F16 acc. | Notes |
|---|---|---|---|---|
| `V_EXP_F32` / `V_EXP_F16` | `2^x` (base-2) | 1 ULP | 0.51 ULP | denorms flushed (F32) / supported (F16) |
| `V_LOG_F32` / `V_LOG_F16` | `log2(x)` | 1 ULP | 0.51 ULP | same denorm split |
| `V_RCP_F32` / `V_RCP_F16` | `1/x` | 1 ULP | 0.51 ULP | F16 note omits denorm support — flushes near-zero |
| `V_RSQ_F32` / `V_RSQ_F16` | `1/sqrt(x)` | 1 ULP | 0.51 ULP | one op, not sqrt+rcp |
| `V_SQRT_F32` / `V_SQRT_F16` | `sqrt(x)` | 1 ULP | 0.51 ULP | prefer RSQ when you want the reciprocal |

Citations: `V_EXP_F32` (line 12613), `V_LOG_F32` (line 12634), `V_RCP_F32` (line 12658), `V_RSQ_F32` (line 12713), `V_EXP_F16` (line 13288), `V_RSQ_F16` (line 13238), `V_RCP_F16` (line 13192). A single hardware op is already faithfully-rounded for inference — **no Newton–Raphson polish is needed** for softmax/activation accuracy. F64 rcp/rsq/sqrt are the trap here: they are only ~2^29-ULP *seeds* (`V_RSQ_F64`, line 12744), never usable results, so keep all LLM math in F32/F16 and never let an F64 reciprocal into a kernel raw.

### exp/log are base-2 — fold the change-of-base into an existing multiply

`V_EXP_F32` computes `2^x`, not `e^x`. This is the single most common softmax pitfall: people insert an extra `V_MUL` for the base conversion when it belongs inside the scale you already apply.

```
// natural-exp softmax, fused
scale' = softmax_scale * 1.44269504f     // 1.44269504 = log2(e), precomputed
p      = V_EXP_F32( fma(logit, scale', -max*scale') )   // (logit-max) prescale folds in
```

The `log2(e)` multiply merges into the `QK^T`-scaling / max-subtraction FMA (which itself dual-issues under VOPD, see below), so `V_EXP_F32` stays exactly one instruction per element. For log-sum-exp / cross-entropy, `V_LOG_F32` returns `log2(x)`; recover `ln(x)` with a `× ln(2)` (0.69314718) that likewise folds into a neighboring op.

### exp saturates cleanly at the infinities — mask attention with −inf, NaN-free

`V_EXP_F32(-INF) = 0x00000000` (exactly zero) and `V_EXP_F32(+INF) = +INF` (`V_EXP_F32`, line 12626). Implement causal/padding masks by **adding −INFINITY to scaled logits before exp**: masked lanes contribute exactly 0 to both the running max and the running sum with no NaN pollution, and no separate select/branch. (By contrast `V_LOG_F32` of a negative or ±0 returns NaN / −INF, so guard log-domain code.)

### RMSNorm / LayerNorm: one `V_RSQ_F32`, not sqrt+rcp

The normalization factor `1/sqrt(mean(x²)+eps)` is exactly what `V_RSQ_F32` targets — a single transcendental at 1 ULP. Do **not** emit `V_SQRT_F32` followed by `V_RCP_F32` (two TRANS32 issues on an un-pairable pipe). Pattern:

```
ss   = sum(x*x)                     // DPP/permute reduction on cheap V_ADD_F32
inv  = V_RSQ_F32(ss * (1/N) + eps)  // one op, broadcast to the row
y    = x * inv                      // packed multiply-through
```

For the softmax denominator, take the reciprocal **once** with `V_RCP_F32` and multiply every numerator by it rather than issuing a divide per element (`V_RCP_F32`, line 12666). Skip the documented 2-FMA Newton–Raphson refinement (it converges to <0.5 ULP) for inference-grade accuracy; add it only if you need near-correctly-rounded division.

### Scheduling: the TRANS32 pipe is separate, higher-latency, and un-pairable

Two hardware facts govern how you schedule the exp/rcp/rsq that feed softmax and normalization:

1. **Distinct latency class.** `S_DELAY_ALU` carries separate dependency codes for transcendental producers (`INSTID_TRANS32_DEP_1..3`, codes 5–7) vs. ordinary VALU (codes 1–4) — proving TRANS32 has longer latency than the FMA pipe (`S_DELAY_ALU`, line 1787). A consumer of a transcendental result (e.g. the running-sum FMA reading `exp(x)`) must use the TRANS32 delay code, and back-to-back producer→consumer stalls unless you interleave independent work. **Unroll the softmax inner loop** so several `V_EXP_F32` are in flight before their sums accumulate.

2. **No VOPD dual-issue.** The VOPD opcode tables contain only FMAC/FMAAK/FMAMK/MUL/ADD/SUB/MAX/MIN/MOV/CNDMASK/DOT2ACC — **no exp/log/rcp/rsq/sqrt** (VOPD is wave32-only, line 2751). So each transcendental costs one un-pairable VALU slot; only the *surrounding* scale FMAs, reciprocal-multiplies, and max-subtractions can be packed 2-per-issue via VOPD in wave32. Budget one un-hideable TRANS32 slot per exp/rcp element and keep enough independent transcendentals in flight to fill the pipe.

3. **Keep DPP off the transcendental.** Transcendentals allow DPP but a DPP-modified op costs an extra execute cycle (line 2828), compounding the already-high TRANS32 latency. Do the softmax max/sum **cross-lane reductions on the cheap `V_MAX_F32`/`V_ADD_F32` with DPP16**, and keep the `V_EXP_F32`/`V_RCP_F32` as plain per-lane ops in between.

### Free VOP3 modifiers on the transcendental itself

The VOP3 (64-bit) encoding of any transcendental carries `abs`/`neg` input modifiers and `omod`/`clamp` output modifiers (line 2415/2419), applied in the operand/write path at no extra instruction:

- **`neg` for `exp(-x)`** in GELU/SiLU approximations — no separate negate.
- **`omod`** post-scales the result by 0.5/2/4 (a free average or doubling).
- **`clamp`** saturates a float result to [0,1].

Caveats: `omod` is silently ignored when IEEE mode is set or output denormals are enabled; and the VOP3 form is 64-bit, so only use it when the folded modifier actually replaces a standalone op.

### Precision & numerical behavior

- **F32 transcendentals flush denormals unconditionally** (independent of `S_DENORM_MODE`), while the F16 variants preserve them (`V_EXP_F32`, line 12621; `V_EXP_F16`, line 13296). In `exp(logit − max)` the argument is ≤ 0 so results are ≤ 1 and can underflow to denormals — in F32 those tiny probabilities flush to 0 (usually fine, slightly changes the sum). If you need the extra dynamic range near underflow, the **F16 exp path keeps subnormals**; but note `V_RCP_F16` alone omits the denorm-support note, so tiny reciprocal inputs may still flush — safe for softmax denominators (sum ≥ 1) but not for arbitrary values.
- **No packed (2-wide) F16 transcendental exists.** Unlike `V_PK_ADD_F16`/`V_PK_FMA_F16`, the F16 transcendentals process **one element per lane** and use OPSEL only to place the 16-bit result in the high/low half of a VGPR (line 13288, 2259). Switching softmax exp/rcp to F16 buys **register pressure and bandwidth savings, not 2× transcendental throughput** — model the per-element TRANS32 cost the same as F32, and keep accumulation in F32.
- **FP16 overflow guard.** For FP16-heavy softmax/exp, set `MODE.FP16_OVFL` (bit 23, line 938) so an overflowing FP16 VALU result clamps to ±MAX_FP16 instead of becoming INF, while genuine INF/÷0 still yield INF — hardware saturation that keeps large magnitudes finite without per-op min/max, reducing NaN propagation.

### Integer division inside quant/index math

When a kernel does integer division (per-group dequant indices, tile/stride math), the compiler's macro uses `V_RCP_IFLAG_F32` (line 12678), a reciprocal that **can raise integer DIV_BY_ZERO but no FP exceptions** — the correct primitive so spurious FP flags aren't raised on the reciprocal. It shares the TRANS32 pipe, so the same latency/no-dual-issue rules apply; prefer power-of-two group sizes (shifts) to avoid it on hot index paths.
## Hidden / non-obvious wins

Consolidated list of the buried items — things stated once in passing, implied by an encoding
table, or absent from the prose entirely. These are the highest-value-per-line findings.

**Free modifiers and encoding tricks**
- **`CLAMP` is free saturation.** One bit gives float `[0,1]` saturation *and* signed/unsigned
  integer saturation on the same arithmetic op — no follow-up min/max (lines 2427, 2708).
- **`NEG`/`NEG_HI` are free per-half sign flips** on packed ops, but **`OMOD` is silently
  ignored** for packed FP16 and integer results — do not use it to fold a ×2/×0.5 scale
  (lines 2655, 2419).
- **Inline constants only populate the low 16 bits.** A packed op needs `OPSEL`/`OPSEL_HI` set
  to broadcast one constant to *both* halves; getting it wrong silently corrupts the high
  element (lines 2728, 2730).
- **`NEG_HI` means ABS on the MIX opcodes** — free `|x|` per source (line 2705).
- **Per-operand sign on integer dots.** `V_DOT4_I32_IU8`'s NEG bits select signed vs unsigned
  *per operand*, giving free mixed signed×unsigned — exactly what asymmetric (zero-point)
  quantization needs (line 16818).

**Dual-issue (VOPD) rules that silently block pairing**
- **Wave32 only.** VOPD is skipped entirely in wave64 — a wave64 kernel forgoes dual-issue
  with no diagnostic (line 2751).
- **VGPR bank rule:** paired same-position operands must sit in different banks (`addr mod 4`);
  violation doesn't slow down, it *does not function* (lines 2767–2772).
- **SRC2 even/odd parity** governs accumulator pairing for `V_DUAL_FMAC_F32` and
  `V_DUAL_DOT2ACC_*` — allocate accumulators in even/odd pairs or the inner loop won't
  co-issue (line 2773).
- **Destination parity is hardwired:** `vdstY`'s LSB is `!vdstX[0]`, so two same-parity
  destinations are unencodable (line 2778).
- **VOPD cannot use DPP** — cross-lane reduction tails never dual-issue (line 2853).
- **Integer ops pair only in the Y slot** (`V_DUAL_ADD_NC_U32`, `LSHLREV_B32`, `AND_B32`);
  two integer ops can never co-issue (lines 6934–6936).

**Memory and LDS**
- **Direct global→LDS DMA exists but the manual never documents it.** 17 opcodes
  (`BUFFER_LOAD_LDS_*`, `GLOBAL_LOAD_LDS_*`, `SCRATCH_LOAD_LDS_*`) appear only in the
  machine-readable XML, landing exactly in the numeric gaps of the manual's opcode tables.
  All are ≤`B32`, so they save VGPRs and a hop, not bandwidth.
- **`SADDR` mode** (SGPR base + per-lane VGPR offset) cuts address-VGPR pressure on every
  global access (line 4457).
- **Prefer `GLOBAL_*` over `FLAT_*`:** FLAT consumes both LDS and global bandwidth and
  double-counts on `VMcnt` *and* `LGKMcnt`, forcing conservative waits (line 4528).
- **`DS_*_2ADDR`** moves two addresses (or 64 bits) per instruction — double the LDS bandwidth
  per issue slot (line 4699).
- **`DS_PERMUTE`/`DS_BPERMUTE`/`DS_SWIZZLE` use LDS *hardware* but touch no LDS banks and
  allocate no LDS** — full cross-lane shuffle with zero LDS footprint and no bank conflicts
  (line 5125). This is the cheapest reduction primitive on the chip.
- **Partially out-of-range multi-DWORD LDS reads return all zeros**, not partial data — a
  silent-corruption trap on boundary tiles.
- **`S_BARRIER` does not wait on memory counters.** An `S_WAITCNT` must precede it when the
  barrier is protecting LDS or global traffic (line 11143) — a correctness bug that presents
  as flaky numerics.

**Scheduling and math**
- **Transcendentals occupy a separate `TRANS32` scoreboard class** with its own `S_DELAY_ALU`
  dependency encoding — softmax `exp` chains need different scheduling than regular VALU
  (line 1787).
- **`DPP_ROW_XMASK` gives XOR-butterfly lane exchange**, the natural shape for a 4-step
  log₂(16) tree reduction inside each 16-lane row (line 6986).
- **`V_FMA_MIXLO_F16`/`MIXHI_F16` compute in FP32 and write one FP16 half** — assemble a
  packed FP16 pair at full FP32 intermediate precision with no separate convert-and-pack
  (line 16945).
- **`DPP` costs one extra cycle** — the only cycle figure the manual actually states for the
  cross-lane path (line 2828).

## Caveats & unknowns

**No FP8.** RDNA 3.5 has no FP8/E4M3/E5M2 arithmetic anywhere in the ISA. The lowest-precision
paths are F16/BF16 and the IU8/IU4 integer dot and WMMA ops.

**The manual is silent on most throughput.** It gives essentially no cycles-per-instruction or
bandwidth figures — the notable exceptions being the +1 cycle for DPP (line 2828) and the LDS
bank-conflict serialization behaviour. Every relative-speed statement in this document derives
from *structural* facts (how many operations an instruction encodes, how many VGPRs or issue
slots it consumes) rather than measured rates. **Benchmark before committing to a design.**

**The manual contradicts itself on cache-control semantics.** §4.1.1 describes `SLC`/`DLC` as
pure temporal/locality hints, while the per-encoding field tables gloss the same bits as
"System/Device Level Coherent." The policy tables' "Hint" columns also appear labelled
inversely to their own gloss. Treat cache-bit guidance as provisional and verify on hardware.

**Some instructions exist only in the XML.** The prose conversion omits 31 instructions that
AMD's machine-readable XML defines — including the entire direct-to-LDS family and the GWS
(global wave sync) ops. Conversely two manual entries (`DS_CMPSWAP_RTN_B64/F64`) appear under
different names in the XML (`DS_CMPSTORE_RTN_*`). Cross-check both sources for anything
load-bearing.

**Corrections applied during assembly.** Two automated findings were overturned by direct
verification and are corrected in the text above: (1) the claim that RDNA 3.5 has *no*
load-to-LDS opcode — it has 17, XML-only; (2) the assumption that `DPP_ROW_BCAST` is available
— it is **not** in the RDNA 3.5 `dpp_ctrl` enumeration, so GCN/CDNA reduction epilogues using
`row_bcast:15`/`row_bcast:31`/`wave_shl`/`wave_ror` must be rewritten onto `V_PERMLANEX16_B32`,
`V_PERMLANE64_B32`, or `DS_SWIZZLE_B32`.

**Source-conversion risk.** The manual used here is a PDF→Markdown conversion, proofread but
imperfect; figures were dropped. Anything diagram-dependent should be checked against the
original PDF.

## Sources

- **Prose and line numbers:** `docs/reference/rdna35-isa-markdown/rdna35_instruction_set_architecture.md`
  (29,430 lines). All `(line N)` citations refer to this file. Original PDF:
  <https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture>
- **Formal encodings, bit fields, operand lists:** `docs/reference/rdna35-isa-markdown/amdgpu_isa_rdna3_5.xml`
  (AMD machine-readable ISA, <https://gpuopen.com/machine-readable-isa/>). Cited as "(XML)".
- **Navigation helpers:** `derived/manifest.jsonl` (instruction → opcode/encoding/line),
  `derived/xref.tsv` (manual↔XML gap report), `derived/instructions/<NAME>.md` (per-instruction
  chunks), and `build/search.py` for semantic lookup.
