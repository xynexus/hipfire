# Kernel functions → gfx1151 (RDNA3.5) capabilities: the actual map

Answering "have we mapped what our kernels need onto what the ISA provides?" —
the honest previous answer was no, only point investigations. This is the map.

Scope: the four kernels that are ~97% of both phases on Qwen3.8-27B oq4.25++.

| phase | kernel | share |
|---|---|---|
| prefill | `gemm_oq_compact_grouped_wmma` | 84.6% |
| prefill | `attention_flash_kvarn_tile_batched` / `gated_delta_net_f32` | 3.2% / 3.1% |
| decode | `gemv_oq_compact_grouped_v3` | 86.6% |
| decode | `attention_flash_kvarn_tile_batched` | 6.6% |

Method: disassemble each kernel, inventory emitted ISA ops, cross-reference
`/srv/hipfire/docs/kb`. "Status" is what the mapping actually is, not what it
could be.

## The map

| # | Required function | gfx1151 capability | What we emit | Status |
|---|---|---|---|---|
| 1 | W4×A4 matmul | `V_WMMA_I32_16X16X16_IU4` (VOP3P 69) | same | **optimal** |
| 2 | W4×A8 matmul | none — K=16 IU4 is the only INT4 shape; no mixed-width WMMA | int8 WMMA, or 2× iu4 passes | **best available** |
| 3 | Accumulator zero per group | ISA: inline constant legal **for C only** | 64 `v_mov_b32`/group, 22% of loop VALU | **GAP — compiler** |
| 4 | i32→f32 + f16 scale fold | `V_CVT_F32_I32` + `V_FMA_MIX_F32` | same | **optimal** |
| 5 | INT4 unpack (decode GEMV) | `V_BFE_I32`+cvt = 2 ops; `V_CVT_OFF_F32_I4` = 2 ops | 29 `v_bfe_i32` + 34 `v_cvt_f32_i32` | **tie — no win** |
| 6 | 8×INT4 dot (W4A4 GEMV) | `V_DOT8_I32_IU4` (VOP3P 24) | **unused** — unpack to int8 + 2× `sudot4` | **GAP — unused ISA** |
| 7 | Cross-lane sum reduce | DPP `row_xmask` (VALU) + `v_permlanex16_b32` | 40 `ds_bpermute_b32` (LDS pipe, LGKMcnt) | **GAP — unused ISA** |
| 8 | Rotation / FWHT butterfly | `DS_SWIZZLE_B32` — fixed pattern, no LDS storage | 40 `ds_swizzle_b32` | **optimal** |
| 9 | f32 dual-issue | VOPD — wave32 only, f32-only opcode set | n/a (wave64 GEMM) | **closed, measured** |
| 10 | bf16 ↔ f32 | **no `V_CVT_BF16_F32` on any RDNA** | software shift/round | **GAP — hardware** |
| 11 | FP8/BF8 | none on gfx1151 (RDNA4 only) | n/a | **N/A** |
| 12 | Scalar float (uniform math) | RDNA3.5-only SALU FP unit, `S_CVT_F32_I32` etc. | **zero scalar-float ops emitted** | **checked, not applicable** |

## The three real gaps, with what they are worth

**#3 — accumulator zero (measured).** The ISA permits an inline constant for the
WMMA C operand; LLVM's `__builtin_amdgcn_wmma_*` constrains C to a v4i32 register
class, so it cannot be expressed and the compiler materialises zeros into
registers: 64 `v_mov_b32` per 256-group, **22% of the loop's non-WMMA VALU**.
Attempted and reverted (`2026-08-22-isa-kb-findings.md`) — forcing the unroll to
express it pushed VGPRs 240→256 and began spilling. Worth ~12% of the GEMM if a
future LLVM exposes it. Nothing we can do in HIP today.

**#6 — `V_DOT8_I32_IU4` is used nowhere in the tree.** Grep hits in
`fused_qkvza_oq4_dp4a.hip` / `gemv_mq8g256.hip` are comments naming the
`dot8-insts` *target feature* (which gates `sudot4`), not the instruction. The
compact multicol GEMV instead unpacks nibbles to int8 (`v_perm_b32` + OR
sign-extend) and issues two `sudot4`. `V_DOT8_I32_IU4` consumes a packed dword of
eight nibbles directly. **Caveat that makes this less attractive than it looks:**
it needs INT4 activations on both operands, and our compact path is W4A8, so it
would require splitting the int8 activation into two int4 planes — which is
exactly the pre-pass whose cost killed this idea once already
(`2026-08-22-...multicol`). Real, but not free.

**#7 — cross-lane reduction goes through the LDS pipe.** All our reductions are
`__shfl_xor`, which lowers to `ds_bpermute_b32` (DS op, tracked on LGKMcnt). The
ISA offers `row_xmask` DPP — an XOR butterfly within a row of 16, executed on the
VALU with no LDS traffic — which covers the first four steps of a 5-step
reduction; only the final cross-row step needs `v_permlanex16_b32`. The KB calls
`row_share`/`row_xmask` "the backbone of every good RDNA reduction". We already
use `DS_SWIZZLE_B32` correctly for the rotation butterfly (#8), so the pattern is
understood — it just was not applied to the reductions.

## What the map also settles

- **#5 is a genuine tie.** The KB flags `V_CVT_OFF_F32_I4` as "the cheapest INT4
  dequant primitive available and almost never used", which reads like free money.
  Our decode GEMV already achieves the same 2 ops/weight with `V_BFE_I32` + cvt.
  Checked so nobody re-derives it.
- **#12 is checked and negative.** gfx1151 has a scalar FP unit gfx1100 lacks, and
  the KB suggests offloading uniform scale math to it. The kernels emit **zero**
  scalar-float ops — and cannot benefit, because in the wave64 C layout neither
  the weight nor the activation scale is uniform over a VGPR. Same property that
  bounds the rescale.
- **#9 is closed by measurement, not opinion.** VOPD is wave32-only and its
  `V_DUAL_*` set is f32-only plus three integer Y-column ops; no `V_CVT_*` is
  dual-issuable and only a LEFT shift is in the set. Even perfect dual-issue
  predicts ~38 TOPS against wave64's measured 53.4.

## Method note

Every "what we emit" column came from `llvm-objdump -d --mcpu=gfx1151` on the
cached `.hsaco`, not from reading the HIP source. That matters: the source says
`__shfl_xor`, and only the disassembly says `ds_bpermute_b32`.
