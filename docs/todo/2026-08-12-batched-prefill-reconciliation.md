# Reconciling `feat/qwen35-fp32-state-batched-prefill` onto master

Written 2026-08-12 after attempting the merge and **abandoning it deliberately**.
Read this before trying again — the merge shape is wrong, and one correctness
question gates the whole thing.

## Why this branch matters

Three commits, local only (never pushed):

- `7fd0a43a3 feat(qwen35): let the MoE family take batched prefill (FP32 GDN state)`
- `3d42a658c fix(qwen35): per-expert AWQ scales in the batched routed-MoE path`
- `8fd63132e feat(server): admit new requests into a running batch at a decode-step boundary`

The prize is in the first: **pp512 38.50 → 495.44 ± 11.10 t/s** (12.9× baseline,
1.7× FastFlowLM's ~290) on qwen3.6-35B-A3B oq4++.

## Do NOT merge it. The merge shape is wrong.

`3d42a658c` (2026-07-30) and master's `0714f618e` (2026-08-12) are **independent
fixes of the same bug** — per-expert AWQ scales in the routed MoE path. Same root
cause, same design, near-identical evidence (branch: layer-0 cosine 0.239752 →
1.000000; master: 0.244 → 0.999999). They differ in every data structure:

| | branch | master |
|---|---|---|
| gate_up rotate | `rotate_x_mq_awq_indexed` in `rotate_x_mq_awq.hip` | `rotate_x_mq_awq_indexed_batched` in its own `.hip` |
| down rotate | `fused_silu_mul_mq_rotate_awq_indexed` **added inside** `fused_silu_mul_mq_rotate_awq.hip` | same entry-point name, **own file** |
| params fields | `gate_up_awq_ptrs`, `down_awq_ptrs` | `expert_gate_up_awq_ptrs`, `expert_down_awq_ptrs` |
| null table | required `&GpuTensor` | `Option<&GpuTensor>` |
| env parse | a 4th copy, `oq_moe::moe_expert_blocks_repacked()` (`"1"` only) | centralized `oq_indexed_decode_enabled()` |

Merging produces **duplicate definitions**, not conflicts git can help with:
`rotate_x_mq_awq_indexed_batched` defined twice in `dispatch/`, the same HIP
entry point in two `.hip` files, `expert_*_awq_ptrs` declared twice in
`MoeFfnWeights`. Pinning each file to one side then breaks its neighbours,
because the two implementations are structurally different across ~8 files. That
cascade is the signal to stop merging.

**Master's implementation wins** on every axis: it is shipped, it has bit-exact
parity harnesses (`parity_rotate_x_mq_awq_indexed`,
`parity_silu_mul_rotate_awq_indexed`), it supports a null pointer table, and it
covers all 18 call sites including `arch-minimax` and `arch-lfm2moe`.

## The right shape: re-apply the unique work on top of master

Three independent pieces. Do them as separate commits, not a merge.

### 1. The eligibility unlock — small, and the actual perf win

In `qwen35/mod.rs`, the batched-prefill predicate carries

```rust
&& (weights.layers.iter()
        .all(|lw| matches!(lw, LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_))))
```

which excludes every MoE stack. That clause dates from 2026-04, before
`gated_delta_net_f32_batch_seq` existed (landed 2026-06-22). The per-layer dtype
check below it already admits `DeltaNetMoe` on its own terms
(`moe_topk_ok && moe_router_logits_present && ...`), so the clause is redundant
and is what forced qwen3.6-35B-A3B down the per-token fallback, re-reading all 40
layers' attention projections once per token.

Note the quant list is now `FP32 | FP16`, **not** the `FP32 | Q8` the branch has
— PR #247 deleted Q8 recurrent state and `StateQuant::Q8` no longer exists.

**CORRECTION (measured 2026-08-12): deleting that clause is NOT sufficient, and
on an OQ model it changes nothing at all.** An earlier revision of this document
called it "the unlock". It is not. Tried on
`Qwen3.5-35B-A3B--oq4.25++.hfq`, same binary, same 543-token prompt,
`max_tokens=1`, batched vs `HIPFIRE_PREFILL_BATCHED=0`:

```
batched (clause removed)   wall 64.48 s   (load 35.69 s)
forced per-token fallback  wall 64.52 s   (load 35.81 s)
```

Identical. The reason is a **second, independent gate**: the per-layer
`is_batchable_la` admits only
`MQ4G256 | HFQ4G256 | MQ6G256 | HFQ6G256 | Q8_0 | ParoQ4G128 | F32 | F16 | BF16`
— **no OQ dtype**. So for any OQ-quantized MoE model `pbs_eligible` returns
false whether or not the layer-kind clause is there, and removing it is inert.
`HIPFIRE_KERNEL_TRACE=1` confirms the predicate runs
(`n=543 dn_quant=FP32 all_layers_dense_la=false moe_topk_ok=true K=8 E=256
router_logits=true`) but it prints its INPUTS, never its verdict — which is why
the timing A/B, not the trace, is what settled this.

So the real work is widening `is_batchable_la` to the OQ family and proving the
batched LA/MoE dispatch is correct for those dtypes. That is a bigger job than
deleting a clause, and it is the actual prerequisite for the pp512 number. Note
the branch measured qwen3.**6**-35B-A3B oq4++ while this test used
qwen3.**5**-35B-A3B oq4.25++; if the branch also widened `is_batchable_la`, that
widening — not the clause — is where its speedup came from. Check that first.

### 2. ⚠️ BLOCKER — settle this before landing (1)

`gemm_oq4g256_moe_grouped_wmma` takes a **10th** parameter on master and an
**11th** on the branch: `weight_byte_offset`, threaded from a
`routed_oq_arch_combined` params field, computed as

```rust
if oq_arch_combined { m * (k / 2) + m * (k / 256) * 4 } else { 0 }
```

The branch's rationale: resident OQ4 experts sit in the oq4_arch combined layout,
so the interleaved 132-byte block stream starts *after* the split nibbles and
split f32 scales; only the `oq_moe` repack emits it at offset 0.

This is not optional, because `moe_grouped_gemm_supported_for_dtype` admits
`DType::Oq4G256 => arch.starts_with("gfx11")` — **gfx1151 included**. So enabling
(1) on an OQ4 MoE model routes resident experts into that kernel. If master's
offset-0 read is wrong for the combined layout, (1) turns a currently-unreachable
bug into a live one.

**Answer this first:** does master's grouped-WMMA path already handle resident
qt=34/37 experts correctly at offset 0, or did the branch find a latent bug?
Landing (1) without knowing is the same mistake as flipping
`HIPFIRE_QWEN35_MOE_OQ_INDEXED` on one model's KLD — see
`docs/todo/2026-08-12-handover-indexed-oq-moe.md`.

### 3. Batch admission at a decode-step boundary

`8fd63132e`, server-side, independent of the MoE work. Should port cleanly on its
own.

## Verification note

The tiny gate has **no coverage** for batched MoE prefill — the arch-6 toy preset
is `experts_per_tok: 2` and `use_gpu_topk` requires `k_top == 8`, so the fixture
never exercises this path. Evidence has to come from a real 35B-A3B run: the
pp512 number plus a per-layer cosine against the per-token reference (the branch
recorded layer-0 1.000000 @16 tok, layer-39 0.999378 @16 tok, 0.998326 @128 tok).
