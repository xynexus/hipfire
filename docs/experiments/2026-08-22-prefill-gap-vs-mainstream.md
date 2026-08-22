# The prefill gap vs mainstream engines is the overlay, and 350 tok/s is
# arithmetically impossible while it exists

State box: halo, gfx1151, 40 CU / 20 WGP, ~2.9 GHz, 248.5 GB/s measured DRAM.
Qwen3.8-27B `OqPlusCompact` (oq4.25++), KVarN KV, batched prefill, 2059-token
prompt. **Reported gap: vLLM, llama.cpp and Kairic all exceed 350 tok/s prefill
on this class of model, one of them at bf16 KV with 6-bit weights. We are at
217.**

## The instruction budget rules out kernel tuning as the answer

From the AMD matrix calculator on gfx1151, all for a 16x16x16 block:

| instruction | cycles | note |
|---|---|---|
| `v_wmma_i32_16x16x16_iu4` | **16** | |
| `v_wmma_i32_16x16x16_iu8` | 32 | |
| `v_wmma_f32_16x16x16_bf16` | 32 | |
| `v_wmma_f32_16x16x16_f16` | 32 | |

Two consequences worth stating because they kill two plausible theories:

1. **iu4 is exactly 2x iu8**, so our exact-W4A8 scheme (two iu4 passes) costs the
   *same WMMA cycles* as one iu8 pass. The 2-pass structure is not a tax.
2. **bf16 is 32 cycles, same as iu8** — so "dequantize to bf16 and run a dense
   bf16 GEMM", which is what llama.cpp does for large batches, has **no
   instruction-rate advantage over what we already do.** They are not winning on
   the GEMM instruction.

iu4 peak: 2048 ops/WGP/cycle x 20 WGP x 2.9 GHz = **119 TOPS**. Our W4A8 needs
two iu4 issues per useful MAC, so its useful-MAC ceiling is **59 TOPS**.

## Where the 9.49 s actually goes

| component | share | seconds |
|---|---|---|
| `gemm_oq_compact_iu4x2_wmma` | 52.3% | 4.96 |
| **`oq_compact_overlay_correct_t`** | **23.9%** | **2.27** |
| `attention_flash_kvarn_tile_batched` | 11.1% | 1.05 |
| `gated_delta_net_f32` | 3.7% | 0.35 |
| **`oq_compact_x8_transpose`** | **2.0%** | **0.19** |
| everything else (32 kernels) | ~7% | 0.66 |

GEMM achieves 19.8 TOPS useful = **33% of the 59 TOPS ceiling**, and 74% of the
best compact GEMM we have ever measured (53.4 TOPS of iu4 issue).

## The result: 350 tok/s cannot be reached with the overlay. Not "is hard" — cannot.

350 tok/s over 2059 tokens is a 5.88 s budget.

    fixed non-GEMM (attn + gdn + other)   2.06 s
    overlay correction + transpose        2.46 s
    ------------------------------------------
    left for the GEMM                     1.36 s
    GEMM needs 196 TOP of iu4 issue in 1.36 s = 144 TOPS
    hardware peak is 119 TOPS                 => IMPOSSIBLE

Even a *perfect* GEMM running at 100% of silicon peak misses 350 while the
overlay pass exists. Drop it:

    fixed non-GEMM                        2.06 s
    left for the GEMM                     3.82 s
    GEMM needs 51 TOPS of iu4 issue
      = 96% of our own best-measured 53.4, 43% of peak   => feasible

## So this is a FORMAT decision, not a kernel decision

The sparse overlay is a structural tax mainstream engines simply do not pay.
Ours costs **25.9% of prefill** (correction + transpose) to apply **1.2% of the
arithmetic** — 3 scattered per-row corrections per 256 weights. Two fusion
attempts (see `2026-08-22-overlay-gemm-epilogue-fusion.md` and the megakernel
scope) both measured 6-12% WORSE than the separate pass, and the reason is now
understood: the gather is L2-resident, not DRAM-bound, so there is no traffic to
save by fusing — the cost is intrinsic to applying per-row scattered corrections.

The competitor data point is the decisive one: **one engine uses 6-bit weights
and still beats us.** It spends ~1.4x our weight bytes and skips 26% of the work,
and prefill is compute-bound, so the extra bytes cost it almost nothing.

**The 4.25-bit floor is a MINIMUM, not a target.** Spending 6 or 8 bits is fully
compliant with the goal's constraint. We have been optimizing the wrong side of
it for prefill.

## The tension this creates, and the experiment that resolves it

Decode is bandwidth-bound (~90% of the 248.5 GB/s ceiling), so more bits costs
decode directly: 4.25 -> 8 bits is 1.88x the weight bytes. Naively that is
14.2 -> ~7.6 tok/s decode.

BUT the overlay is also measured at ~25% of the *decode* GEMV
(`gemv_oq_compact_multicol`). So it is taxing both paths, and the trade is not
obviously one-sided: fewer bytes but more compute, in a kernel where the compute
is apparently not hidden.

`oq8` is available **today** — quantizer CLI support and native int8 kernels
(including the 7.3x M-slab GEMM), no overlay, no in-kernel 4-bit decode, and at
32 cycles/block it does the same WMMA work our 2-pass W4A8 does. `Oq6G256`
exists in the format enum (qt=40) but has no CLI and no kernels.

**Next: quantize a small model (qwen3.5-2b--bf16) to oq8 and to oq4.25++ and
measure prefill AND decode on both.** That is a few minutes of GPU time and it
settles the format question directly instead of by arithmetic. If oq8 prefill is
~1.35x and decode does not collapse, rebuild the 27B and re-baseline.

Independently and regardless of format: the GEMM is at 74% of our own best
measured rate, and closing that is worth 217 -> ~250 tok/s on its own.
