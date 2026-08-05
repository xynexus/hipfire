# Confirmed in the daemon: prefill gets no batching benefit

`decoder-layer-npu-scope.md` measured a ~540-token prompt through
`examples/infer_hfq` and found it ran the DECODE path one position at a time —
85008 `gemv_oq4_grouped` dispatches, `gemv_f32` 759 times, and **no `gemm_*`
kernel at all**. It flagged the obvious caveat: that profile is not the daemon,
and "confirming it requires profiling the daemon ... Do that before changing
serving code."

Confirmed, by throughput signature rather than by profiler. `hipfire bench` on
the daemon, 2026-08-06, gfx1151:

| model | pp512 | tg128 | ratio |
|---|---|---|---|
| `Llama-3.2-1B-Instruct--bf16` | 37.40 | 36.28 | **1.03x** |
| `Llama-3.2-1B-Instruct--oq4++` | 96.90 | 90.18 | **1.07x** |

**Prefill throughput equals decode throughput.** A batched prefill amortises
weight reads across the whole prompt, so it should run many times decode's
per-token rate — FLM's own ratio is ~45x (2750 t/s prefill against 60.1 decode).
hipfire gets 1.03-1.07x, which is what per-position evaluation predicts and
nothing else does.

## Stronger: cost per token is FLAT in prompt length

The ratio above says prefill costs what decode costs. Sweeping prompt length
says there is no amortisation at all:

| prompt | prefill | ms/token |
|---|---|---|
| 64 | 634.2 ms | 9.91 |
| 128 | 1278.7 ms | 9.99 |
| 512 | 5289.6 ms | **10.33** |

Flat across 8x. A batched prefill reads each weight once for the whole prompt,
so ms/token must fall roughly as 1/N until the GEMM saturates; instead it is
constant, and mildly WORSE at 512. The weights are being re-read per token.

## What this rules out

**Not quant-specific.** bf16 and oq4++ show the same ratio, so it is not the
OQ4 GEMV path or a missing quantised GEMM — it is structural to how prefill is
driven.

**Not the lm_head.** That was 43% of the `infer_hfq` profile at 5217.9 us per
`gemv_f32` dispatch — the `[128256, 2048]` F32 tied head, since replaced by a
1.92 ms path (`tied-lmhead-f32-expansion.md` in hipfire). Real, and already
banked, but it cannot explain a 1.03x ratio: the head is one dispatch per
position either way.

**Not a missing batched kernel.** `weight_gemm`'s generic arm loops per-token
GEMV and `warn_generic_once`s when it does. A pp512 run emits no such warning,
so the batched kernels are selected.

**Not upstream chunking.** `SimpleAr::prefill` is handed the whole
`prompt_tokens` slice (`generate_arch.rs:1403`), and `prefill_forward` sets
`batch = tokens.len()`, so `weight_gemm` really is called with batch_size=512.

**Not `prefill_forward` being unbatched by construction.** It is documented as
the `attention_causal_batched` path and a same-build A/B preferred it over the
chunked path (pp512 602 vs 581 t/s on gfx1103 / MiniCPM5-1B.bf16). Whatever is
happening, it is not that llama picked a knowingly serial routine — and note
602 t/s there is 16x the 37.40 measured here on the same family, so the
regression is not universal across builds or hosts.

## Why this matters more than the NPU decode path

The NPU work has been aimed at decode. The repo's own numbers say not to:
`decoder-layer-npu-scope.md:92` — "Decode does NOT need the NPU" — and the
measured NPU ceiling for the 1B with optimal qkv/gate_up fusion is **~38 tok/s**
(`wire-in-r6-prefill-offload.md:404`) against GPU decode now at **102.41**. A
perfect NPU decode path would be a 2.7x regression.

Prefill is the only losing axis, by 34x, and this is its cause. Fixing it is
also the prerequisite for the NPU being useful at all: the prefill-only MoE
offload that `decoder-layer-npu-scope.md` calls "the smallest change that flips
the last axis" assumes prefill is batched before it is offloaded.

## Next

The gap is now narrow and strange: the call site is batched, the batch size is
the full prompt, the batched kernels are selected, and yet per-token cost is
flat. That leaves the batched GEMM kernels themselves performing like N GEMVs at
these shapes, or `hipfire bench` not reaching `SimpleAr::prefill` at all.

Two checks, in order:

1. Confirm the bench actually routes through `SimpleAr::prefill` for this model
   — a `dbg!`/log at that call site during a pp512 run settles it in one run and
   costs nothing. If it does not, everything above describes a path the
   measurement never took.
2. If it does, time `gemm_oq4_grouped_act_batched` directly at the prefill shape
   against 512 `gemv_oq4_grouped` calls. `examples/bench_lmhead_dtype.rs` is the
   template — it already does exactly this shape of A/B with a correctness
   column.

The 602-vs-37 gap between the recorded gfx1103 A/B and this measurement stays
the sharpest external lead.
